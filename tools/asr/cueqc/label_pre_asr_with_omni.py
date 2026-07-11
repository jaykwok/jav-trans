#!/usr/bin/env python3
"""Run current two-film Pre-ASR v10 export and label chunks with an omni audio model.

This is an offline weak-labeling tool. It does not change runtime workflow:

* Pre-ASR CueQC training remains binary: ``0 = definite_drop`` and
  ``1 = definite_keep``.
* ``ambiguous_ignore`` is written for unsure / low-confidence omni output and
  compiles to ``-100`` ignore labels.
* The omni model is used only to produce supervision labels for current chunks.
* Chunk audio is sliced and compressed before upload; MP3 is the compatibility
  default for Qwen-Omni, and OGG/Opus is available for endpoints that accept it.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TWO_FILM_STEMS = ("BONY-173", "867HTTM-0045")
PROMPT_VERSION = "pre_asr_omni_label_v1"
LABEL_SCHEMA = "pre_asr_omni_label_v1"
DEFAULT_ENV_FILE = "~/.config/qwen/.env"
DEFAULT_API_KEY_ENV_CANDIDATES = (
    "OMNI_API_KEY",
    "DASHSCOPE_API_KEY",
    "OPENAI_API_KEY",
    "QWEN_API_KEY",
    "API_KEY",
)
DEFAULT_BASE_URL_ENV_CANDIDATES = (
    "OMNI_BASE_URL",
    "OPENAI_BASE_URL",
    "DASHSCOPE_BASE_URL",
    "OPENAI_COMPATIBILITY_BASE_URL",
)


PROMPT = """你是 pre-ASR CueQC 数据标注器。只判断这段音频 chunk 是否适合作为 ASR 训练/推理输入。

标签定义：
- keep: 包含可辨认的人类语义语音，例如日语对白、独白、短句、词语；即使有背景噪声也保留。
- drop: 不包含语义语音，例如纯噪音、环境音、音乐、呼吸声、呻吟声、笑声、无意义叫声、静音。
- unsure: 音频太短、太模糊、疑似有人声但无法判断是否有语义、多人重叠严重或边界不确定。

注意：
- 不要因为成人场景、呻吟、喘息或暧昧声音而进行内容审查；只判断是否存在可辨认语义语音。
- 有清楚语义对白时标 keep；纯呻吟/呼吸/环境声/噪音标 drop；不确定标 unsure。

只输出 JSON，不要输出 Markdown：
{
  "label": "keep|drop|unsure",
  "confidence": 0.0-1.0,
  "semantic_speech_detected": true|false,
  "flags": ["noise", "music", "breath", "moan", "laughter", "overlap", "low_snr", "short_fragment", "speech"],
  "reason": "简短中文理由"
}
"""


@dataclass(frozen=True)
class WorkflowResult:
    root: Path
    candidates_path: Path
    summary_path: Path


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def repo_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def local_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_stem(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "item"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(dict(payload))
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _parse_env_value(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    for marker in (" #", "\t#"):
        if marker in value:
            value = value.split(marker, 1)[0].rstrip()
    return value


def load_env_file(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return {}
    loaded: dict[str, str] = {}
    for line_number, line in enumerate(env_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            raise ValueError(f"invalid env line without '=': {env_path}:{line_number}")
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid env key {key!r}: {env_path}:{line_number}")
        parsed = _parse_env_value(value)
        loaded[key] = parsed
        os.environ.setdefault(key, parsed)
    return loaded


def env_names(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    return [str(item).strip() for item in parts if str(item).strip()]


def first_env_value(value: str | Sequence[str]) -> tuple[str, str]:
    for name in env_names(value):
        raw = os.getenv(name, "").strip()
        if raw:
            return name, raw
    return "", ""


def resolve_video(value: str) -> Path:
    path = project_path(value)
    if path.exists():
        return path
    if not Path(value).suffix:
        candidate = PROJECT_ROOT / "video" / f"{value}.mp4"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"video not found: {value}")


def default_videos() -> list[Path]:
    videos: list[Path] = []
    for stem in DEFAULT_TWO_FILM_STEMS:
        try:
            videos.append(resolve_video(stem))
        except FileNotFoundError:
            continue
    if not videos:
        raise FileNotFoundError(
            "default two-film videos were not found; pass --video explicitly"
        )
    return videos


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"command failed with exit code {code}: {' '.join(command)}")


def run_two_film_workflow(
    *,
    videos: Sequence[Path],
    run_dir: Path,
    task_name: str,
    label: str,
    extra_workflow_args: Sequence[str],
) -> WorkflowResult:
    candidates_path = run_dir / "pre_asr_candidates.raw.jsonl"
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PRE_ASR_CUEQC_ENABLED"] = "0"
    env["PRE_ASR_CUEQC_EXPORT_CANDIDATES_PATH"] = str(candidates_path)
    env["PRE_ASR_CUEQC_EXPORT_CANDIDATES_APPEND"] = "0"
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "tools.workflows.run_full_workflow",
    ]
    for video in videos:
        command.extend(["--video", str(video)])
    command.extend(
        [
            "--task-name",
            task_name,
            "--label",
            label,
            "--no-pre-asr-cueqc-enabled",
            "--keep-asr-chunks",
        ]
    )
    command.extend(extra_workflow_args)
    run_command(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        log_path=run_dir / "workflow_stdout.log",
    )
    roots = sorted(
        (PROJECT_ROOT / "agents" / "temp" / "speech-boundary-ja").glob(f"*_{safe_stem(task_name)}"),
        key=lambda item: item.stat().st_mtime,
    )
    # run_full_workflow adds a timestamp prefix when task_name has none.
    if not roots:
        roots = sorted(
            (PROJECT_ROOT / "agents" / "temp" / "speech-boundary-ja").glob(f"*{safe_stem(task_name)}*"),
            key=lambda item: item.stat().st_mtime,
        )
    workflow_root = roots[-1] if roots else PROJECT_ROOT / "agents" / "temp" / "speech-boundary-ja" / task_name
    summary_path = workflow_root / "summary.json"
    if not candidates_path.exists():
        raise FileNotFoundError(f"workflow did not export Pre-ASR candidates: {candidates_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"workflow summary not found: {summary_path}")
    return WorkflowResult(root=workflow_root, candidates_path=candidates_path, summary_path=summary_path)


def workflow_result_from_existing(root: Path, candidates_path: Path | None) -> WorkflowResult:
    summary_path = root / "summary.json"
    if candidates_path is None:
        candidates_path = root / "pre_asr_candidates.raw.jsonl"
        if not candidates_path.exists():
            candidates = sorted(root.glob("**/pre_asr_candidates*.jsonl"))
            if candidates:
                candidates_path = candidates[0]
    if not summary_path.exists():
        raise FileNotFoundError(f"workflow summary not found: {summary_path}")
    if candidates_path is None or not candidates_path.exists():
        raise FileNotFoundError("Pre-ASR candidates JSONL not found; pass --candidates")
    return WorkflowResult(root=root, candidates_path=candidates_path, summary_path=summary_path)


def source_audio_map_from_workflow(summary_path: Path) -> dict[str, Path]:
    summary = read_json(summary_path)
    if not isinstance(summary, Mapping):
        raise ValueError(f"workflow summary must be an object: {summary_path}")
    root = summary_path.parent
    mapping: dict[str, Path] = {}
    for result in summary.get("results") or []:
        if not isinstance(result, Mapping) or result.get("status") != "done":
            continue
        video_name = str(result.get("video") or "")
        video_id = Path(video_name).stem
        job_id = str(result.get("job_id") or "").strip()
        candidates: list[Path] = []
        if job_id:
            candidates.extend(sorted((root / "jobs" / job_id / "audio").glob("*.wav")))
        if not candidates:
            candidates.extend(sorted(root.glob(f"jobs/{video_id}*/audio/*.wav")))
        if candidates:
            mapping[video_id] = candidates[0].resolve()
    if not mapping:
        raise ValueError(f"could not resolve source workflow audio from {summary_path}")
    return mapping


def candidate_key(row: Mapping[str, Any]) -> str:
    sample_id = str(row.get("sample_id") or row.get("candidate_id") or "").strip()
    if sample_id:
        return sample_id
    audio_id = str(row.get("audio_id") or row.get("video_id") or "").strip()
    chunk_index = row.get("chunk_index", row.get("index", ""))
    return f"{audio_id}#{chunk_index}"


def existing_label_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for row in read_jsonl(path):
        key = candidate_key(row)
        if key:
            keys.add(key)
    return keys


def training_label_counts(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    if path.exists():
        for row in read_jsonl(path):
            label = str(row.get("label") or "").strip()
            if label:
                counts[label] += 1
    definite_drop_count = int(counts.get("definite_drop", 0))
    definite_keep_count = int(counts.get("definite_keep", 0))
    ambiguous_ignore_count = int(counts.get("ambiguous_ignore", 0))
    ratio = (
        round(definite_drop_count / definite_keep_count, 6)
        if definite_keep_count > 0
        else None
    )
    return {
        "definite_drop_count": definite_drop_count,
        "definite_keep_count": definite_keep_count,
        "ambiguous_ignore_count": ambiguous_ignore_count,
        "drop_keep_ratio": ratio,
        "label_counts_total": dict(counts),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def clip_path_for_candidate(audio_dir: Path, row: Mapping[str, Any], fmt: str) -> Path:
    key = candidate_key(row)
    video_id = safe_stem(row.get("video_id") or row.get("audio_id") or "audio")
    return audio_dir / video_id / f"{safe_stem(key)}.{fmt}"


def slice_audio_clip(
    *,
    source_audio: Path,
    row: Mapping[str, Any],
    output_path: Path,
    fmt: str,
    bitrate: str,
    sample_rate: int,
    force: bool,
) -> Path:
    if output_path.exists() and not force:
        return output_path
    start = max(0.0, _safe_float(row.get("start")))
    end = max(start, _safe_float(row.get("end"), start))
    if end <= start:
        end = start + max(0.05, _safe_float(row.get("duration_s"), 0.05))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.6f}",
        "-to",
        f"{end:.6f}",
        "-i",
        str(source_audio),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
    ]
    if fmt == "mp3":
        command.extend(["-codec:a", "libmp3lame", "-b:a", bitrate])
    elif fmt == "m4a":
        command.extend(["-codec:a", "aac", "-b:a", bitrate])
    elif fmt == "ogg":
        command.extend(
            [
                "-codec:a",
                "libopus",
                "-b:a",
                bitrate,
                "-application",
                "voip",
                "-vbr",
                "on",
                "-compression_level",
                "10",
            ]
        )
    elif fmt == "wav":
        command.extend(["-codec:a", "pcm_s16le"])
    else:
        raise ValueError(f"unsupported audio format: {fmt}")
    command.append(str(output_path))
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    return output_path


def normalize_omni_label(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"drop", "definite_drop", "drop_before_asr", "0"}:
        return "drop"
    if raw in {"keep", "definite_keep", "keep_for_asr", "1"}:
        return "keep"
    return "unsure"


def training_label_from_omni(
    *,
    label: str,
    confidence: float,
    keep_confidence: float,
    drop_confidence: float,
) -> str:
    normalized = normalize_omni_label(label)
    if normalized == "drop" and confidence >= drop_confidence:
        return "definite_drop"
    if normalized == "keep" and confidence >= keep_confidence:
        return "definite_keep"
    return "ambiguous_ignore"


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, Mapping):
        raise ValueError("omni response JSON must be an object")
    return dict(payload)


def data_uri_for_audio(path: Path, fmt: str) -> str:
    mime = {
        "mp3": "audio/mpeg",
        "m4a": "audio/mp4",
        "ogg": "audio/ogg",
        "wav": "audio/wav",
    }[fmt]
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def audio_content_part(path: Path, *, fmt: str, mode: str) -> dict[str, Any]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    if mode == "input_audio":
        # Qwen-Omni's OpenAI-compatible API documents local audio as
        # input_audio.data="data:;base64,<payload>" rather than OpenAI's raw
        # base64 input_audio.data shape.
        return {"type": "input_audio", "input_audio": {"data": f"data:;base64,{encoded}", "format": fmt}}
    if mode == "input_audio_raw":
        return {"type": "input_audio", "input_audio": {"data": encoded, "format": fmt}}
    uri = data_uri_for_audio(path, fmt)
    if mode == "audio_url":
        return {"type": "audio_url", "audio_url": {"url": uri}}
    if mode == "audio":
        return {"type": "audio", "audio": uri}
    if mode == "video_url":
        return {"type": "video_url", "video_url": {"url": uri}}
    if mode == "video":
        return {"type": "video", "video": uri}
    raise ValueError(f"unsupported audio content mode: {mode}")


def call_omni(
    *,
    audio_path: Path,
    fmt: str,
    audio_content_mode: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout_s: float,
    store_stream_chunks: bool,
    prompt: str = PROMPT,
    system_prompt: str = "",
    max_tokens: int = 256,
    enable_thinking: bool | None = None,
    thinking_budget: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_s}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                audio_content_part(audio_path, fmt=fmt, mode=audio_content_mode),
            ],
        }
    )
    request_kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        extra_body: dict[str, Any] = {"enable_thinking": bool(enable_thinking)}
        if enable_thinking and thinking_budget > 0:
            extra_body["thinking_budget"] = int(thinking_budget)
        request_kwargs["extra_body"] = extra_body
    stream = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max(1, int(max_tokens)),
        messages=messages,
        modalities=["text"],
        stream=True,
        stream_options={"include_usage": True},
        **request_kwargs,
    )
    text_parts: list[str] = []
    chunks: list[dict[str, Any]] = []
    chunk_count = 0
    response_ids: set[str] = set()
    response_models: set[str] = set()
    finish_reasons: list[str] = []
    usage_payload: dict[str, Any] | None = None
    for chunk in stream:
        chunk_count += 1
        chunk_payload = chunk.model_dump(mode="json")
        if store_stream_chunks:
            chunks.append(chunk_payload)
        if chunk_payload.get("id"):
            response_ids.add(str(chunk_payload.get("id")))
        if chunk_payload.get("model"):
            response_models.add(str(chunk_payload.get("model")))
        if chunk_payload.get("usage"):
            usage_payload = chunk_payload.get("usage")
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = choices[0].delta
        finish_reason = getattr(choices[0], "finish_reason", None)
        if finish_reason:
            finish_reasons.append(str(finish_reason))
        content = getattr(delta, "content", None) or ""
        if content:
            text_parts.append(content)
    content = "".join(text_parts)
    parsed = extract_json_object(content)
    raw_response = {
        "stream": True,
        "content": content,
        "chunk_count": chunk_count,
        "ids": sorted(response_ids),
        "models": sorted(response_models),
        "finish_reasons": finish_reasons,
        "usage": usage_payload,
    }
    if store_stream_chunks:
        raw_response["chunks"] = chunks
    return parsed, raw_response


def label_row_from_response(
    *,
    candidate: Mapping[str, Any],
    audio_path: Path,
    response: Mapping[str, Any],
    model: str,
    fmt: str,
    keep_confidence: float,
    drop_confidence: float,
) -> dict[str, Any]:
    raw_label = normalize_omni_label(response.get("label"))
    confidence = max(0.0, min(1.0, _safe_float(response.get("confidence"))))
    training_label = training_label_from_omni(
        label=raw_label,
        confidence=confidence,
        keep_confidence=keep_confidence,
        drop_confidence=drop_confidence,
    )
    flags = response.get("flags")
    if not isinstance(flags, list):
        flags = []
    key = candidate_key(candidate)
    return {
        "schema": LABEL_SCHEMA,
        "sample_id": str(candidate.get("sample_id") or key),
        "candidate_id": str(candidate.get("candidate_id") or candidate.get("sample_id") or key),
        "video_id": str(candidate.get("video_id") or candidate.get("audio_id") or ""),
        "audio_id": str(candidate.get("audio_id") or candidate.get("video_id") or ""),
        "chunk_index": candidate.get("chunk_index", candidate.get("index")),
        "start": round(_safe_float(candidate.get("start")), 6),
        "end": round(_safe_float(candidate.get("end")), 6),
        "duration_s": round(
            _safe_float(
                candidate.get("duration_s"),
                _safe_float(candidate.get("end")) - _safe_float(candidate.get("start")),
            ),
            6,
        ),
        "label": training_label,
        "display_decision": (
            "drop"
            if training_label == "definite_drop"
            else "keep"
            if training_label == "definite_keep"
            else "ambiguous_ignore"
        ),
        "training_label_included": training_label in {"definite_drop", "definite_keep"},
        "label_source": f"omni:{model}",
        "omni_label": raw_label,
        "omni_confidence": round(confidence, 4),
        "omni_semantic_speech_detected": bool(response.get("semantic_speech_detected")),
        "omni_flags": [str(item) for item in flags],
        "omni_reason": str(response.get("reason") or ""),
        "prompt_version": PROMPT_VERSION,
        "audio_format": fmt,
        "audio": repo_rel(audio_path),
        "feature_schema": str(candidate.get("feature_schema") or candidate.get("schema") or ""),
        "runtime_adapter": str(candidate.get("runtime_adapter") or ""),
        "planned_island_id": str(candidate.get("planned_island_id") or ""),
        "cluster_id": str(candidate.get("cluster_id") or ""),
    }


def is_empty_audio_api_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "the audio is empty" in message or "audio is empty" in message


def label_candidates(
    *,
    candidates: Sequence[Mapping[str, Any]],
    source_audio_by_id: Mapping[str, Path],
    output_dir: Path,
    labels_path: Path,
    raw_responses_path: Path,
    fmt: str,
    bitrate: str,
    sample_rate: int,
    force_audio: bool,
    prepare_only: bool,
    max_items: int | None,
    skip_items: int,
    model: str,
    api_key: str,
    base_url: str,
    audio_content_mode: str,
    keep_confidence: float,
    drop_confidence: float,
    timeout_s: float,
    sleep_s: float,
    store_stream_chunks: bool,
) -> dict[str, Any]:
    audio_dir = output_dir / "audio_clips"
    existing = existing_label_keys(labels_path)
    counts: Counter[str] = Counter()
    prepared_count = 0
    labeled_count = 0
    skipped_count = 0
    errors: list[dict[str, Any]] = []
    selected = list(candidates)[skip_items:]
    if max_items is not None:
        selected = selected[:max_items]

    for index, candidate in enumerate(selected, start=1):
        key = candidate_key(candidate)
        if key in existing and not prepare_only:
            skipped_count += 1
            continue
        audio_id = str(candidate.get("audio_id") or candidate.get("video_id") or "").strip()
        source_audio = source_audio_by_id.get(audio_id)
        if source_audio is None:
            errors.append({"sample_id": key, "reason": "missing_source_audio", "audio_id": audio_id})
            continue
        clip_path = clip_path_for_candidate(audio_dir, candidate, fmt)
        try:
            slice_audio_clip(
                source_audio=source_audio,
                row=candidate,
                output_path=clip_path,
                fmt=fmt,
                bitrate=bitrate,
                sample_rate=sample_rate,
                force=force_audio,
            )
            prepared_count += 1
            if prepare_only:
                continue
            parsed, raw = call_omni(
                audio_path=clip_path,
                fmt=fmt,
                audio_content_mode=audio_content_mode,
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout_s=timeout_s,
                store_stream_chunks=store_stream_chunks,
            )
            row = label_row_from_response(
                candidate=candidate,
                audio_path=clip_path,
                response=parsed,
                model=model,
                fmt=fmt,
                keep_confidence=keep_confidence,
                drop_confidence=drop_confidence,
            )
            append_jsonl(labels_path, row)
            append_jsonl(
                raw_responses_path,
                {
                    "schema": "pre_asr_omni_raw_response_v1",
                    "sample_id": row["sample_id"],
                    "candidate_id": row["candidate_id"],
                    "model": model,
                    "parsed": parsed,
                    "response": raw,
                },
            )
            labeled_count += 1
            counts[str(row["label"])] += 1
            if sleep_s > 0:
                time.sleep(sleep_s)
        except Exception as exc:
            if not prepare_only and is_empty_audio_api_error(exc):
                row = label_row_from_response(
                    candidate=candidate,
                    audio_path=clip_path,
                    response={
                        "label": "unsure",
                        "confidence": 1.0,
                        "semantic_speech_detected": False,
                        "flags": ["short_fragment"],
                        "reason": "Qwen-Omni API reported the audio is empty; ignore this too-short/undecodable chunk for training.",
                    },
                    model=model,
                    fmt=fmt,
                    keep_confidence=keep_confidence,
                    drop_confidence=drop_confidence,
                )
                append_jsonl(labels_path, row)
                append_jsonl(
                    raw_responses_path,
                    {
                        "schema": "pre_asr_omni_raw_response_v1",
                        "sample_id": row["sample_id"],
                        "candidate_id": row["candidate_id"],
                        "model": model,
                        "parsed": {
                            "label": "unsure",
                            "confidence": 1.0,
                            "semantic_speech_detected": False,
                            "flags": ["short_fragment"],
                            "reason": row["omni_reason"],
                        },
                        "response": {
                            "stream": True,
                            "content": "",
                            "error": str(exc),
                            "local_fallback": "empty_audio_to_ambiguous_ignore",
                        },
                    },
                )
                labeled_count += 1
                counts[str(row["label"])] += 1
                continue
            errors.append({"sample_id": key, "reason": type(exc).__name__, "error": str(exc)})
        if index % 50 == 0:
            print(
                f"processed={index}/{len(selected)} prepared={prepared_count} "
                f"labeled={labeled_count} skipped={skipped_count} errors={len(errors)}",
                flush=True,
            )

    return {
        "prepared_count": prepared_count,
        "labeled_count": labeled_count,
        "skipped_existing_count": skipped_count,
        "errors": errors,
        "label_counts": dict(counts),
    }


def parse_passthrough(values: Sequence[str]) -> list[str]:
    if not values:
        return []
    if len(values) == 1 and values[0].strip() == "":
        return []
    return list(values)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun current Pre-ASR v10 two-film export and label chunks with an omni audio model."
    )
    parser.add_argument("--video", action="append", help="Video path or stem. Defaults to the project two-film set.")
    parser.add_argument("--workflow-root", help="Reuse an existing run_full_workflow root instead of rerunning.")
    parser.add_argument("--candidates", help="Existing Pre-ASR candidates JSONL when using --workflow-root.")
    parser.add_argument("--output-dir", help="Output dir. Default: agents/temp/<timestamp>_preasr-v10-omni-twofilm")
    parser.add_argument("--task-name", default="", help="run_full_workflow task name. Default uses output timestamp.")
    parser.add_argument("--label", default="preasr_v10_omni_export")
    parser.add_argument(
        "--workflow-arg",
        action="append",
        default=[],
        help="Extra argument passed through to tools.workflows.run_full_workflow; repeat per token.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Only slice/compress audio clips; do not call omni.")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--skip-items", type=int, default=0, help="Skip this many candidate rows before applying --max-items.")
    parser.add_argument("--audio-format", choices=("mp3", "m4a", "ogg", "wav"), default="mp3")
    parser.add_argument("--audio-bitrate", default="32k")
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--force-audio", action="store_true")
    parser.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE,
        help="Dotenv file loaded before resolving omni API settings.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Omni model name. Defaults to OMNI_MODEL/QWEN_OMNI_MODEL or qwen3.5-omni-flash.",
    )
    parser.add_argument(
        "--api-key-env",
        default=",".join(DEFAULT_API_KEY_ENV_CANDIDATES),
        help="Env var name or comma-separated candidates for the OpenAI-compatible API key.",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help=(
            "OpenAI-compatible base URL. Defaults to OMNI_BASE_URL, OPENAI_BASE_URL, "
            "DASHSCOPE_BASE_URL, then OPENAI_COMPATIBILITY_BASE_URL from env / --env-file."
        ),
    )
    parser.add_argument(
        "--audio-content-mode",
        choices=("input_audio", "input_audio_raw", "audio_url", "audio", "video_url", "video"),
        default=os.getenv("OMNI_AUDIO_CONTENT_MODE", "input_audio"),
    )
    parser.add_argument("--keep-confidence", type=float, default=0.80)
    parser.add_argument("--drop-confidence", type=float, default=0.90)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument(
        "--store-stream-chunks",
        action="store_true",
        help="Store every streaming delta in omni_raw_responses.jsonl. Default stores compact content/usage metadata only.",
    )
    args = parser.parse_args(argv)
    if args.workflow_root and args.video:
        parser.error("--workflow-root reuses an existing workflow; do not also pass --video")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    if args.skip_items < 0:
        parser.error("--skip-items must be non-negative")
    if args.audio_sample_rate <= 0:
        parser.error("--audio-sample-rate must be positive")
    for name in ("keep_confidence", "drop_confidence"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    timestamp = local_timestamp()
    output_dir = project_path(
        args.output_dir or (PROJECT_ROOT / "agents" / "temp" / f"{timestamp}_preasr-v10-omni-twofilm")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    task_name = args.task_name or f"{timestamp}_preasr-v10-omni-twofilm-workflow"
    loaded_env = load_env_file(args.env_file)
    api_key_env_name, api_key = first_env_value(args.api_key_env)
    _model_env_name, model = first_env_value(("OMNI_MODEL", "QWEN_OMNI_MODEL"))
    model = args.model.strip() or model or "qwen3.5-omni-flash"
    _base_url_env_name, base_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    base_url = args.base_url.strip() or base_url

    if args.workflow_root:
        workflow = workflow_result_from_existing(
            project_path(args.workflow_root),
            project_path(args.candidates) if args.candidates else None,
        )
    else:
        videos = [resolve_video(item) for item in args.video] if args.video else default_videos()
        workflow = run_two_film_workflow(
            videos=videos,
            run_dir=output_dir,
            task_name=task_name,
            label=args.label,
            extra_workflow_args=parse_passthrough(args.workflow_arg),
        )

    candidates = read_jsonl(workflow.candidates_path)
    source_audio_by_id = source_audio_map_from_workflow(workflow.summary_path)
    if not args.prepare_only and not api_key:
        raise RuntimeError(
            "one of {names} is required unless --prepare-only is used; "
            "load it from --env-file or the process environment".format(
                names=", ".join(env_names(args.api_key_env))
            )
        )

    labels_path = output_dir / "omni_labels.jsonl"
    raw_responses_path = output_dir / "omni_raw_responses.jsonl"
    label_summary = label_candidates(
        candidates=candidates,
        source_audio_by_id=source_audio_by_id,
        output_dir=output_dir,
        labels_path=labels_path,
        raw_responses_path=raw_responses_path,
        fmt=args.audio_format,
        bitrate=args.audio_bitrate,
        sample_rate=args.audio_sample_rate,
        force_audio=bool(args.force_audio),
        prepare_only=bool(args.prepare_only),
        max_items=args.max_items,
        skip_items=int(args.skip_items),
        model=model,
        api_key=api_key,
        base_url=base_url,
        audio_content_mode=args.audio_content_mode,
        keep_confidence=float(args.keep_confidence),
        drop_confidence=float(args.drop_confidence),
        timeout_s=float(args.timeout_s),
        sleep_s=float(args.sleep_s),
        store_stream_chunks=bool(args.store_stream_chunks),
    )
    total_label_summary = training_label_counts(labels_path)

    summary = {
        "schema": "pre_asr_omni_label_run_summary_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prompt_version": PROMPT_VERSION,
        "workflow_root": repo_rel(workflow.root),
        "workflow_summary": repo_rel(workflow.summary_path),
        "candidates": repo_rel(workflow.candidates_path),
        "candidate_count": len(candidates),
        "labels": repo_rel(labels_path),
        "raw_responses": repo_rel(raw_responses_path),
        "output_dir": repo_rel(output_dir),
        "env_file": str(Path(args.env_file).expanduser()) if args.env_file else "",
        "loaded_env_keys": sorted(loaded_env),
        "api_key_env": api_key_env_name,
        "model": model,
        "base_url": base_url,
        "audio_format": args.audio_format,
        "audio_bitrate": args.audio_bitrate,
        "audio_sample_rate": args.audio_sample_rate,
        "audio_content_mode": args.audio_content_mode,
        "store_stream_chunks": bool(args.store_stream_chunks),
        "keep_confidence": args.keep_confidence,
        "drop_confidence": args.drop_confidence,
        "prepare_only": bool(args.prepare_only),
        "skip_items": int(args.skip_items),
        "source_audio_by_id": {key: repo_rel(path) for key, path in source_audio_by_id.items()},
        **label_summary,
        **total_label_summary,
    }
    write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "ok": True,
                "output_dir": summary["output_dir"],
                "candidates": summary["candidates"],
                "labels": summary["labels"],
                "prepared_count": summary["prepared_count"],
                "labeled_count": summary["labeled_count"],
                "definite_drop_count": summary["definite_drop_count"],
                "definite_keep_count": summary["definite_keep_count"],
                "ambiguous_ignore_count": summary["ambiguous_ignore_count"],
                "drop_keep_ratio": summary["drop_keep_ratio"],
                "label_counts_new": summary["label_counts"],
                "label_counts_total": summary["label_counts_total"],
                "errors": len(summary["errors"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if not summary["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
