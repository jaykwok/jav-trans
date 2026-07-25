#!/usr/bin/env python3
"""Create Scorer v12 dual-evidence vocal-envelope preaudit labels.

Protect and Non-vocal are independent Gemini calls.  The result is reviewable
evidence only; a separate canonical compiler is required before training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.ja.vocal_envelope_v12 import (  # noqa: E402
    VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX,
    VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
)
from tools.asr.cueqc.label_pre_asr_with_omni import (  # noqa: E402
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    GEMINI_THINKING_LEVELS,
    audio_content_mode_for_profile,
    call_omni,
    first_env_value,
    load_env_file,
    normalize_openai_compat_base_url,
)
from tools.omni.timestamp_contract import (  # noqa: E402
    TIMESTAMP_CONTRACT_ID,
    TIMESTAMP_PROMPT_CONTRACT_ZH,
    parse_mmss_span,
    timestamp_request_contract,
)


FRAME_HOP_S = 0.02
CONTRACT_ID = "boundary_acoustic_binary_v12"
SUMMARY_SCHEMA = "vocal_envelope_scorer_v12_dual_evidence_summary_v1"
PROGRESS_SCHEMA = "vocal_envelope_scorer_v12_dual_evidence_progress_v1"
PROMPT_PROFILE = "vocal-envelope-protect-nonvocal-v1"
PROTECT_PROMPT_VERSION = "vocal-envelope-protect-v1-gemini36-medium-mmss"
NONVOCAL_PROMPT_VERSION = "vocal-envelope-nonvocal-v1-gemini36-medium-mmss"
PROMPT_VERSION = f"{PROTECT_PROMPT_VERSION}__{NONVOCAL_PROMPT_VERSION}"
EXPECTED_MODEL = "google/gemini-3.6-flash"
EXPECTED_PROFILE = "gemini"
EXPECTED_REASONING = "medium"
EXPECTED_MAX_TOKENS = 8192
EXPECTED_EXECUTION_CONTRACT = "gemini_openrouter_reasoning_require_parameters_v1"


PROTECT_SYSTEM_PROMPT = """你是 1.7B Scorer v12 的 Human Vocal Event Envelope 保护通道预审 teacher。
音频主要来自日语 JAV、Galgame 或类似场景，但场景和声音类别不能直接决定标签。

你的唯一任务是找出所有“由人类声道、口腔或呼吸系统产生”的连续发声事件包络，输出 vocal_spans。
这不是语义判断，也不是 ASR 价值判断。对白、耳语、呻吟、喘息、呼吸、哭笑、亲吻/口腔声、歌唱、远处人声和含混的人声都属于 vocal，优先完整保留。
同一发声事件中的短停顿、吸气、释气和非语义过渡应随包络保留；不要按音节、字、极短停顿或每个脉冲切碎。若相邻人声属于同一连续事件，合并为一个较完整的包络。

不要标记纯机械、撞击、衣物/床体、水声、纯音乐、静音、环境噪声；这些留给 Non-vocal 通道。
边界覆盖完整起始和衰减，不要为了贴声学起点而截断人声。无法判断时可省略，不要猜测。
只输出当前完整 source 的 0-based vocal_spans JSON 数组，不转写、不判断语义、不标注 non_vocal_spans。

输出对象：{"source_id":"...","vocal_spans":[{"start_ts":"00:00.000","end_ts":"00:01.000","reason":"简短声学理由"}],"overall_reason":"..."}
""" + "\n" + TIMESTAMP_PROMPT_CONTRACT_ZH

NONVOCAL_SYSTEM_PROMPT = """你是 1.7B Scorer v12 的 Human Vocal Event Envelope 非发声通道预审 teacher。
音频主要来自日语 JAV、Galgame 或类似场景，但场景和声音类别不能直接决定标签。

你的唯一任务是找出“明确不含任何人类声道、口腔或呼吸发声”的连续 non-vocal 区间，输出 non_vocal_spans。
允许：纯机械、碰撞、衣物/床体、动作、水声、纯音乐、静音、底噪、风扇空调、电流和其他环境噪声。
只要存在对白、耳语、呻吟、喘息、呼吸、哭笑、亲吻/口腔声、歌唱或远处/背景人声的合理声学证据，就不要标记 non-vocal；宁可省略。
不要把短时长、低响度或 ASR 无结果当成证据。不要切分人类发声，不判断语义。
边界必须与任何可能人声清晰分离；不确定就输出空数组。

输出对象：{"source_id":"...","non_vocal_spans":[{"start_ts":"00:00.000","end_ts":"00:01.000","category":"mechanical|impact|cloth|bed|water|music|silence|ambience|other","reason":"简短声学理由"}],"overall_reason":"..."}
""" + "\n" + TIMESTAMP_PROMPT_CONTRACT_ZH

NONVOCAL_CATEGORIES = frozenset({"mechanical", "impact", "cloth", "bed", "water", "music", "silence", "ambience", "other"})


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    _write_json(path, payload)


def _resolve_audio(row: Mapping[str, Any], *, manifest: Path) -> Path:
    value = str(row.get("audio") or row.get("audio_path") or "")
    if not value:
        raise ValueError(f"source {row.get('source_id')} has no audio path")
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [manifest.parent / raw, PROJECT_ROOT / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(candidates[0])


def _frame_count(row: Mapping[str, Any], duration_s: float) -> int:
    frame_count = int(row.get("frame_count") or round(duration_s / FRAME_HOP_S))
    if frame_count <= 0:
        raise ValueError(f"invalid frame_count for {row.get('source_id')}")
    if abs(frame_count * FRAME_HOP_S - duration_s) > FRAME_HOP_S:
        raise ValueError(f"duration/frame geometry mismatch for {row.get('source_id')}")
    return frame_count


def _normalize_spans(parsed: Mapping[str, Any], *, field: str, duration_s: float, frame_count: int) -> list[dict[str, Any]]:
    raw = parsed.get(field)
    if raw is None:
        raise ValueError(f"teacher response must contain {field}")
    if not isinstance(raw, list):
        raise ValueError(f"teacher response {field} must be an array")
    result: list[dict[str, Any]] = []
    previous_end = 0
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field}[{index}] must be an object")
        start, end = parse_mmss_span(item, field=f"{field}[{index}]", duration_s=duration_s)
        assert start is not None and end is not None
        if field == "vocal_spans":
            # Protect evidence expands to the containing frame so a vocal edge
            # is never lost merely because the wire timestamp is off-grid.
            start_frame = max(0, int(math.floor(start / FRAME_HOP_S + 1e-9)))
            end_frame = min(frame_count, int(math.ceil(end / FRAME_HOP_S - 1e-9)))
        else:
            # Non-vocal evidence contracts to fully-contained frames.  This is
            # label quantization, not runtime smoothing or boundary repair.
            start_frame = max(0, int(math.ceil(start / FRAME_HOP_S - 1e-9)))
            end_frame = min(frame_count, int(math.floor(end / FRAME_HOP_S + 1e-9)))
        if start_frame < previous_end or start_frame < 0 or end_frame <= start_frame or end_frame > frame_count:
            raise ValueError(f"{field}[{index}] has invalid/overlapping frame coordinates")
        previous_end = end_frame
        normalized = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_s": round(start_frame * FRAME_HOP_S, 6),
            "end_s": round(end_frame * FRAME_HOP_S, 6),
            "reason": str(item.get("reason") or ""),
        }
        if field == "non_vocal_spans":
            category = str(item.get("category") or "other")
            if category not in NONVOCAL_CATEGORIES:
                raise ValueError(f"unsupported non-vocal category: {category}")
            normalized["category"] = category
        result.append(normalized)
    return result


def _state_runs(protect: list[bool], nonvocal: list[bool], frame_count: int) -> dict[str, list[dict[str, Any]]]:
    states: list[str] = []
    for p, n in zip(protect, nonvocal, strict=True):
        states.append(
            "vocal"
            if p and not n
            else "non_vocal"
            if n and not p
            else "conflict"
            if p and n
            else "unsure"
        )
    output: dict[str, list[dict[str, Any]]] = {"vocal_spans": [], "non_vocal_spans": [], "unsure_spans": [], "conflict_spans": []}
    start = 0
    for index in range(1, frame_count + 1):
        if index < frame_count and states[index] == states[start]:
            continue
        state = states[start]
        span = {
            "start_frame": start,
            "end_frame": index,
            "start_s": round(start * FRAME_HOP_S, 6),
            "end_s": round(index * FRAME_HOP_S, 6),
            "label": "vocal_candidate" if state == "vocal" else "non_vocal_candidate" if state == "non_vocal" else "unsure",
        }
        if state == "vocal":
            output["vocal_spans"].append(span)
        elif state == "non_vocal":
            output["non_vocal_spans"].append(span)
        else:
            output["unsure_spans"].append(span)
            if state == "conflict":
                output["conflict_spans"].append(span)
        start = index
    return output


def merge_dual_evidence(*, vocal_spans: list[Mapping[str, Any]], non_vocal_spans: list[Mapping[str, Any]], frame_count: int) -> dict[str, list[dict[str, Any]]]:
    if frame_count <= 0:
        raise ValueError("v12 dual evidence requires positive frame_count")
    protect = [False] * frame_count
    nonvocal = [False] * frame_count
    for span in vocal_spans:
        start, end = int(span["start_frame"]), int(span["end_frame"])
        if not (0 <= start < end <= frame_count):
            raise ValueError("vocal evidence is out of bounds")
        for index in range(start, end):
            protect[index] = True
    for span in non_vocal_spans:
        start, end = int(span["start_frame"]), int(span["end_frame"])
        if not (0 <= start < end <= frame_count):
            raise ValueError("non-vocal evidence is out of bounds")
        for index in range(start, end):
            nonvocal[index] = True
    return _state_runs(protect, nonvocal, frame_count)


def _request_prompt(row: Mapping[str, Any], *, pass_name: str, feedback: str = "") -> str:
    payload = {
        "source_id": str(row["source_id"]),
        **timestamp_request_contract(float(row["duration_s"])),
        "task": "vocal_event_envelope" if pass_name == "protect" else "non_vocal_only",
    }
    if feedback:
        payload["previous_validation_error"] = feedback
    return json.dumps(payload, ensure_ascii=False)


def _validate_manifest(rows: list[dict[str, Any]], *, manifest: Path) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("v12 source manifest is empty")
    source_ids: set[str] = set()
    core_ids: set[str] = set()
    partitions: dict[str, str] = {}
    video_partitions: dict[str, str] = {}
    result: list[dict[str, Any]] = []
    for row in rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in source_ids:
            raise ValueError(f"duplicate/empty v12 source_id: {source_id!r}")
        partition = str(row.get("partition") or "")
        if partition not in {"train", "val", "test"}:
            raise ValueError(f"v12 source has invalid partition: {source_id}")
        video_id = str(row.get("video_id") or "")
        if not video_id:
            raise ValueError(f"v12 source has no frozen video_id: {source_id}")
        previous_video_partition = video_partitions.setdefault(video_id, partition)
        if previous_video_partition != partition:
            raise ValueError(f"v12 video crosses partitions: {video_id}")
        core_values = row.get("core_ids") or row.get("core_id") or []
        if isinstance(core_values, str):
            core_values = [core_values]
        cores = [str(value) for value in core_values if str(value)]
        if len(cores) != 1 or cores[0] in core_ids:
            raise ValueError(f"v12 core must be present once and unique: {source_id}")
        audio = _resolve_audio(row, manifest=manifest)
        duration = float(row.get("duration_s") or 0.0)
        if duration <= 0:
            raise ValueError(f"v12 source duration is invalid: {source_id}")
        if row.get("audio_sha256") and _sha256(audio) != str(row["audio_sha256"]):
            raise ValueError(f"v12 source audio SHA mismatch: {source_id}")
        copied = dict(row)
        copied.update({"source_id": source_id, "video_id": video_id, "partition": partition, "core_ids": cores, "audio": str(audio), "audio_sha256": _sha256(audio), "duration_s": duration, "frame_count": _frame_count(row, duration)})
        source_ids.add(source_id)
        core_ids.add(cores[0])
        partitions[source_id] = partition
        result.append(copied)
    if not {"train", "val", "test"}.issubset(set(partitions.values())):
        raise ValueError("v12 frozen manifest must contain train/val/test partitions")
    return result


def _load_env(profile: str) -> tuple[str, str, str]:
    env_file = (Path.home() / ".config" / "omni" / profile).resolve()
    load_env_file(env_file)
    _, model = first_env_value(("OMNI_MODEL", "QWEN_OMNI_MODEL"))
    _, key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    _, base = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    if not model or not key:
        raise RuntimeError("Gemini model and API key are required")
    return model, key, normalize_openai_compat_base_url(base)


def _call_pass(*, row: Mapping[str, Any], pass_name: str, model: str, api_key: str, base_url: str, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    system_prompt = PROTECT_SYSTEM_PROMPT if pass_name == "protect" else NONVOCAL_SYSTEM_PROMPT
    expected_field = "vocal_spans" if pass_name == "protect" else "non_vocal_spans"
    audio = Path(str(row["audio"]))
    last_error: Exception | None = None
    for attempt in range(1, int(args.max_attempts) + 1):
        try:
            parsed, raw = call_omni(
                audio_path=audio,
                fmt=audio.suffix.lstrip(".") or "wav",
                audio_content_mode=audio_content_mode_for_profile("gemini"),
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout_s=float(args.timeout_s),
                store_stream_chunks=False,
                prompt=_request_prompt(
                    row,
                    pass_name=pass_name,
                    feedback="" if last_error is None else str(last_error),
                ),
                system_prompt=system_prompt,
                max_tokens=EXPECTED_MAX_TOKENS,
                enable_thinking=True,
                thinking_budget=0,
                provider_profile="gemini",
                reasoning_effort="medium",
                exclude_reasoning=False,
                require_provider_parameters=True,
                response_format={"type": "json_object"},
            )
            if str(parsed.get("source_id") or "") != str(row["source_id"]):
                raise ValueError("teacher source_id mismatch")
            normalized = _normalize_spans(parsed, field=expected_field, duration_s=float(row["duration_s"]), frame_count=int(row["frame_count"]))
            parsed = dict(parsed)
            parsed[expected_field] = normalized
            return parsed, raw
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt < int(args.max_attempts):
                time.sleep(min(8.0, float(attempt)))
    raise RuntimeError(f"v12 {pass_name} teacher failed for {row['source_id']}: {last_error}") from last_error


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.env_file != "gemini":
        raise ValueError("Scorer v12 teacher is Gemini-only; Qwen must not be used")
    manifest = Path(args.manifest).expanduser().resolve()
    manifest_sha = _sha256(manifest)
    rows = _validate_manifest(_rows(manifest), manifest=manifest)
    if args.source_id:
        wanted = list(dict.fromkeys(args.source_id))
        index = {row["source_id"]: row for row in rows}
        missing = [value for value in wanted if value not in index]
        if missing:
            raise ValueError(f"manifest missing source ids: {missing}")
        rows = [index[value] for value in wanted]
    if args.limit > 0:
        rows = rows[: int(args.limit)]
    model, api_key, base_url = _load_env("gemini")
    if args.model:
        model = args.model
    if model != EXPECTED_MODEL:
        raise ValueError(f"Scorer v12 requires {EXPECTED_MODEL}, got {model}")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels_path = output / "preaudit.jsonl"
    raw_path = output / "raw_responses.jsonl"
    progress_path = output / "progress.json"
    existing: dict[str, dict[str, Any]] = {}
    if labels_path.is_file():
        selected_ids = {str(row["source_id"]) for row in rows}
        selected_index = {str(row["source_id"]): row for row in rows}
        for saved in _rows(labels_path):
            source_id = str(saved.get("source_id") or "")
            if not source_id or source_id in existing or source_id not in selected_ids:
                raise ValueError(f"invalid v12 resume preaudit source: {source_id!r}")
            current = selected_index[source_id]
            checks = {
                "schema": VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
                "boundary_serialization_contract_id": CONTRACT_ID,
                "model": model,
                "provider_profile": EXPECTED_PROFILE,
                "reasoning_effort": EXPECTED_REASONING,
                "max_tokens": EXPECTED_MAX_TOKENS,
                "prompt_version": PROMPT_VERSION,
                "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
                "teacher_execution_contract_id": EXPECTED_EXECUTION_CONTRACT,
                "source_manifest_sha256": manifest_sha,
                "partition": current["partition"],
                "audio_sha256": current["audio_sha256"],
                "frame_count": current["frame_count"],
            }
            for field, expected in checks.items():
                if saved.get(field) != expected:
                    raise ValueError(f"v12 resume preaudit {field} mismatch: {source_id}")
            if list(saved.get("core_ids") or ()) != list(current["core_ids"]):
                raise ValueError(f"v12 resume preaudit core mismatch: {source_id}")
            existing[source_id] = saved
    pending = [row for row in rows if row["source_id"] not in existing]
    started = time.perf_counter()
    _write_progress(progress_path, {"schema": PROGRESS_SCHEMA, "status": "running", "model": model, "reasoning_effort": "medium", "max_tokens": EXPECTED_MAX_TOKENS, "completed": len(existing), "total": len(rows), "pending": len(pending)})
    for index, row in enumerate(pending, start=1):
        print(f"v12_teacher={len(existing)+1}/{len(rows)} source={row['source_id']} pass=protect", flush=True)
        protect, protect_raw = _call_pass(row=row, pass_name="protect", model=model, api_key=api_key, base_url=base_url, args=args)
        print(f"v12_teacher={len(existing)+1}/{len(rows)} source={row['source_id']} pass=non_vocal", flush=True)
        nonvocal, nonvocal_raw = _call_pass(row=row, pass_name="nonvocal", model=model, api_key=api_key, base_url=base_url, args=args)
        merged = merge_dual_evidence(vocal_spans=protect["vocal_spans"], non_vocal_spans=nonvocal["non_vocal_spans"], frame_count=int(row["frame_count"]))
        label = {
            "schema": VOCAL_ENVELOPE_SCORER_V12_PREAUDIT_SCHEMA,
            "boundary_serialization_contract_id": CONTRACT_ID,
            "source_id": row["source_id"], "video_id": str(row.get("video_id") or ""),
            "partition": row["partition"], "core_ids": row["core_ids"],
            "audio": row["audio"], "audio_sha256": row.get("audio_sha256") or _sha256(Path(row["audio"])),
            "duration_s": row["duration_s"], "frame_count": row["frame_count"], "frame_hop_s": FRAME_HOP_S,
            "model": model, "provider_profile": "gemini", "reasoning_effort": "medium", "max_tokens": EXPECTED_MAX_TOKENS,
            "temperature": None, "top_p": None, "top_k": None,
            "prompt_profile": PROMPT_PROFILE, "prompt_version": PROMPT_VERSION,
            "protect_prompt_version": PROTECT_PROMPT_VERSION, "nonvocal_prompt_version": NONVOCAL_PROMPT_VERSION,
            "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
            "teacher_execution_contract_id": EXPECTED_EXECUTION_CONTRACT,
            "source_manifest": str(manifest), "source_manifest_sha256": manifest_sha,
            "protect_response": protect, "non_vocal_response": nonvocal,
            "vocal_spans": merged["vocal_spans"], "non_vocal_spans": merged["non_vocal_spans"],
            "unsure_spans": merged["unsure_spans"], "conflict_spans": merged["conflict_spans"],
            "teacher_failed_closed": False, "training_manifest_allowed": False,
            "unsure_training_label": VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX,
            "preaudit_provenance": f"omni:{model}:independent_vocal_nonvocal_evidence",
        }
        with labels_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")
        with raw_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"source_id": row["source_id"], "protect": protect_raw, "non_vocal": nonvocal_raw}, ensure_ascii=False, sort_keys=True) + "\n")
        existing[row["source_id"]] = label
        elapsed = time.perf_counter() - started
        rate = len(existing) / max(elapsed, 1e-9)
        eta = (len(rows) - len(existing)) / max(rate, 1e-9)
        _write_progress(progress_path, {"schema": PROGRESS_SCHEMA, "status": "running", "model": model, "reasoning_effort": "medium", "max_tokens": EXPECTED_MAX_TOKENS, "completed": len(existing), "total": len(rows), "pending": len(rows) - len(existing), "last_source_id": row["source_id"], "elapsed_s": round(elapsed, 3), "eta_s": round(eta, 3)})
        if index < len(pending) and args.request_interval_s > 0:
            time.sleep(float(args.request_interval_s))
    summary = {"schema": SUMMARY_SCHEMA, "boundary_serialization_contract_id": CONTRACT_ID, "model": model, "provider_profile": "gemini", "reasoning_effort": "medium", "max_tokens": EXPECTED_MAX_TOKENS, "omitted_sampling_parameters": ["temperature", "top_p", "top_k"], "prompt_profile": PROMPT_PROFILE, "prompt_version": PROMPT_VERSION, "teacher_timestamp_contract_id": TIMESTAMP_CONTRACT_ID, "source_manifest": str(manifest), "source_manifest_sha256": manifest_sha, "source_count": len(rows), "result_count": len(existing), "results": str(labels_path), "raw_responses": str(raw_path), "training_manifest_allowed": False}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_progress(progress_path, {"schema": PROGRESS_SCHEMA, "status": "completed", "model": model, "reasoning_effort": "medium", "max_tokens": EXPECTED_MAX_TOKENS, "completed": len(existing), "total": len(rows), "pending": 0, "elapsed_s": round(time.perf_counter() - started, 3), "summary": str(output / "summary.json")})
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--env-file", choices=("gemini",), default="gemini")
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-interval-s", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.limit < 0 or args.max_attempts <= 0 or args.request_interval_s < 0:
        parser.error("limit/attempt/interval values are invalid")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
