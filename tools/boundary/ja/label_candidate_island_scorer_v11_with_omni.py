#!/usr/bin/env python3
"""Create model-independent Scorer v11 candidate-island teacher preaudits.

This is deliberately a preaudit, never a canonical-truth compiler.  The
teacher labels the continuous dialogue/candidate envelope that Scorer should
preserve.  It must not split sentences, dialogue turns, or ASR units; Proposal
and Split own that later decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from tools.asr.cueqc.label_pre_asr_with_omni import (
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    DEFAULT_ENV_FILE,
    call_omni,
    first_env_value,
    load_env_file,
    normalize_openai_compat_base_url,
)


FRAME_HOP_S = 0.02
SCHEMA = "candidate_island_scorer_v11_omni_preaudit_v2"
SUMMARY_SCHEMA = "candidate_island_scorer_v11_omni_preaudit_summary_v2"
PROMPT_VERSION = "candidate_island_scorer_v11_omni_preaudit_dialogue_islands_v4"

SYSTEM_PROMPT = """你是 1.7B Scorer v11 的候选岛预审 teacher。你的唯一职责是判断哪些连续波形必须先保留给后续 Proposal / Split / CueQC；你不是 Split，不负责句子切分，也不是 CueQC，不负责最终 drop。\n\n关键合同：\n- Scorer 只保证候选对话岛完整性和高召回。中间几乎没有可安全删除间隔的连续对话、同一轮对话、相邻短语、句内停顿、尾音、短呼吸、短非语义声，都合并在同一个 inside_candidate 岛中。不要按单句、语义单元、说话人变化或标点拆岛。\n- 句子/事件切分属于 Proposal + Split。\n- 孤立呻吟、喘息、含混人声、短叫声若无法明确证明可以独立删除，保守放在 inside_candidate，交给 CueQC；不要让 Scorer 冒险删除。\n- 只有清楚、独立、纯背景且删除不会截断或破坏任何对话连续性的区间，才标 outside_candidate。\n- 不使用固定时长、静音阈值、hysteresis、ASR 文本作为唯一依据，也不要为了好看把岛切碎。\n- 输出完整 source 坐标，区间用秒；不添加前后文。\n\n只输出一个 JSON 对象，不要 Markdown：\n{\n  "source_id":"...",\n  "islands":[{"start_s":0.0,"end_s":1.0,"confidence":0.0,"reason":"连续对话岛的简短理由"}],\n  "unsure_spans":[{"start_s":0.0,"end_s":1.0,"reason":"无法判断的局部"}],\n  "overall_confidence":0.0,\n  "overall_reason":"简短整体理由"\n}\n"""
SYSTEM_PROMPT += """\n补充纠偏：目标是潜在语义对话的连续波形，不是所有人声活动。能够明确听出只是独立呻吟、喘息、呼吸、动作声或音乐且不含词语时，应留在 outside_candidate；只有无法判断是否含词语的含混人声才保守标 inside_candidate 或 unsure。非语义人声不能桥接相距较远的对白。连续对白内部夹着的短停顿、尾音、呼吸或动作声仍保留在同一个岛。属于同一场景本身绝不是合并理由。\n"""


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw, path)
    finally:
        if os.path.exists(raw):
            os.unlink(raw)


def _resolve_audio(value: str, *, manifest: Path) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [manifest.parent / raw, PROJECT_ROOT / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(value)


def _prompt(row: Mapping[str, Any], *, feedback: str = "") -> str:
    payload = {
        "source_id": str(row["source_id"]),
        "duration_s": float(row["duration_s"]),
        "task": "mark continuous candidate dialogue islands for Scorer v11",
        "do_not_split": [
            "same dialogue round",
            "adjacent phrases with almost no gap",
            "intra-utterance pauses and tails",
            "isolated ambiguous vocal sounds unless clearly independent pure background",
        ],
        "outside_candidate": "only clear independent pure background safely removable without harming dialogue continuity",
        "output_units": "continuous islands, not single semantic sentences",
        "must_split": "same scene is not enough to merge: a sustained independent non-dialogue region must end the prior island; do not preserve an entire long scene as one island",
        "nonsemantic_vocal_policy": "clear isolated moan, pant, breath, action sound or music without words is outside; ambiguous possible words are inside or unsure; nonsemantic vocals must not bridge distant dialogue",
        "anti_overmerge": "do not return 0..duration as one island merely because people interact in one scene; mark each contiguous dialogue group and leave sustained pure nonsemantic intervals outside",
    }
    if feedback:
        payload["previous_validation_error"] = feedback
    return json.dumps(payload, ensure_ascii=False)


def _number(value: Any, *, name: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return result


def _spans(parsed: Mapping[str, Any], *, duration_s: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    islands: list[dict[str, Any]] = []
    previous_end = 0.0
    for index, raw in enumerate(parsed.get("islands") or ()):
        if not isinstance(raw, Mapping):
            raise ValueError(f"island {index} is not an object")
        start = float(raw.get("start_s")); end = float(raw.get("end_s"))
        if not 0.0 <= start < end <= duration_s or start < previous_end:
            raise ValueError(
                f"island {index} has invalid local-source coordinates: "
                f"start_s={start}, end_s={end}, previous_end_s={previous_end}, "
                f"required_range=0..{duration_s}; use this 0-based audio clip timeline, "
                "never timestamps from the original video"
            )
        previous_end = end
        islands.append({"label": "inside_candidate", "start_s": start, "end_s": end, "start_frame": round(start / FRAME_HOP_S), "end_frame": round(end / FRAME_HOP_S), "confidence": _number(raw.get("confidence", 0.0), name="island confidence"), "reason": str(raw.get("reason") or "")})
    unsure: list[dict[str, Any]] = []
    previous_end = 0.0
    for index, raw in enumerate(parsed.get("unsure_spans") or ()):
        if not isinstance(raw, Mapping):
            raise ValueError(f"unsure span {index} is not an object")
        start = float(raw.get("start_s")); end = float(raw.get("end_s"))
        if not 0.0 <= start < end <= duration_s or start < previous_end:
            raise ValueError(
                f"unsure span {index} has invalid local-source coordinates: "
                f"start_s={start}, end_s={end}, previous_end_s={previous_end}, "
                f"required_range=0..{duration_s}; use this 0-based audio clip timeline, "
                "never timestamps from the original video"
            )
        previous_end = end
        unsure.append({"label": "unsure", "start_s": start, "end_s": end, "start_frame": round(start / FRAME_HOP_S), "end_frame": round(end / FRAME_HOP_S), "reason": str(raw.get("reason") or "")})
    return islands, unsure


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = Path(args.manifest).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile = str(args.env_file)
    env_file = Path.home() / ".config" / "omni" / profile
    load_env_file(env_file)
    _model_env, model = first_env_value(tuple(args.model_env.split(",")))
    model = args.model or model
    _key_env, api_key = first_env_value(tuple(args.api_key_env.split(",")))
    _url_env, raw_base_url = first_env_value(tuple(args.base_url_env.split(",")))
    base_url = normalize_openai_compat_base_url(raw_base_url)
    if not model or not api_key:
        raise RuntimeError("Omni model and API key are required")
    rows = _rows(manifest)
    labels_path = output / "preaudit.jsonl"
    raw_path = output / "raw_responses.jsonl"
    existing = {str(row["source_id"]): row for row in _rows(labels_path) if row.get("schema") == SCHEMA and row.get("model") == model}
    pending = [row for row in rows if str(row["source_id"]) not in existing]
    if args.limit > 0:
        pending = pending[: args.limit]
    progress_path = output / "progress.json"
    started = time.perf_counter()
    profile_name = env_file.name
    audio_content_mode = {"qwen": "input_audio", "gemini": "input_audio_raw"}[profile_name.lower()]
    _write_progress(progress_path, {"schema": "candidate_island_scorer_v11_omni_progress_v1", "status": "running", "provider_profile": profile_name, "model": model, "completed": len(existing), "total": len(rows), "pending": len(pending), "elapsed_s": 0.0})
    for index, row in enumerate(pending, start=1):
        source_id = str(row["source_id"])
        audio = _resolve_audio(str(row["audio"]), manifest=manifest)
        feedback = ""
        last_error: Exception | None = None
        for attempt in range(1, args.max_attempts + 1):
            parsed: dict[str, Any] | None = None
            raw: dict[str, Any] | None = None
            try:
                request_started = time.perf_counter()
                print(f"omni_request={len(existing)+1}/{len(rows)} provider={profile_name} source_id={source_id} attempt={attempt}/{args.max_attempts}", flush=True)
                parsed, raw = call_omni(audio_path=audio, fmt=audio.suffix.lstrip(".") or "wav", audio_content_mode=audio_content_mode, model=model, api_key=api_key, base_url=base_url, timeout_s=args.timeout_s, store_stream_chunks=False, prompt=_prompt(row, feedback=feedback), system_prompt=SYSTEM_PROMPT, max_tokens=args.max_tokens, enable_thinking=args.enable_thinking, thinking_budget=args.thinking_budget)
                islands, unsure = _spans(parsed, duration_s=float(row["duration_s"]))
                label = {"schema": SCHEMA, "prompt_version": PROMPT_VERSION, "source_id": source_id, "partition": str(row.get("partition") or ""), "frame_count": int(row.get("frame_count") or 0), "frame_hop_s": FRAME_HOP_S, "audio": str(row["audio"]), "audio_sha256": str(row.get("audio_sha256") or _sha256(audio)), "model": model, "base_url_host": base_url.split("/", 3)[2] if "://" in base_url else base_url, "env_file_name": profile_name, "overall_confidence": _number(parsed.get("overall_confidence", 0.0), name="overall confidence"), "overall_reason": str(parsed.get("overall_reason") or ""), "islands": islands, "unsure_spans": unsure, "reviewed_full_source": False, "preaudit_provenance": f"omni:{model}", "human_review_required": True, "training_manifest_allowed": False, "attempts": attempt}
                with labels_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")
                with raw_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"source_id": source_id, "attempt": attempt, "parsed": parsed, "response": raw}, ensure_ascii=False, sort_keys=True) + "\n")
                existing[source_id] = label
                elapsed = time.perf_counter() - request_started
                total_elapsed = time.perf_counter() - started
                rate = (len(existing)) / max(total_elapsed, 1e-9)
                eta = max(0.0, (len(rows) - len(existing)) / max(rate, 1e-9))
                print(f"omni_candidate_island={len(existing)}/{len(rows)} provider={profile_name} source_id={source_id} islands={len(islands)} unsure={len(unsure)} request_s={elapsed:.1f} eta_s={eta:.0f}", flush=True)
                _write_progress(progress_path, {"schema": "candidate_island_scorer_v11_omni_progress_v1", "status": "running", "provider_profile": profile_name, "model": model, "completed": len(existing), "total": len(rows), "pending": len(rows) - len(existing), "last_source_id": source_id, "last_request_s": round(elapsed, 3), "elapsed_s": round(total_elapsed, 3), "eta_s": round(eta, 3), "islands": len(islands), "unsure": len(unsure)})
                last_error = None
                break
            except Exception as error:  # noqa: BLE001
                last_error = error
                feedback = str(error)
                print(f"omni_error provider={profile_name} source_id={source_id} attempt={attempt}/{args.max_attempts} error={type(error).__name__}: {error}", flush=True)
                _write_progress(progress_path, {"schema": "candidate_island_scorer_v11_omni_progress_v1", "status": "running", "provider_profile": profile_name, "model": model, "completed": len(existing), "total": len(rows), "pending": len(rows) - len(existing), "last_source_id": source_id, "last_error": f"{type(error).__name__}: {error}", "elapsed_s": round(time.perf_counter() - started, 3)})
                with raw_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"source_id": source_id, "attempt": attempt, "error": repr(error), "parsed": parsed, "response": raw}, ensure_ascii=False, sort_keys=True) + "\n")
                if attempt < args.max_attempts:
                    time.sleep(min(8.0, float(attempt)))
        if last_error is not None:
            duration_s = float(row["duration_s"])
            frame_count = int(row.get("frame_count") or round(duration_s / FRAME_HOP_S))
            label = {
                "schema": SCHEMA,
                "prompt_version": PROMPT_VERSION,
                "source_id": source_id,
                "partition": str(row.get("partition") or ""),
                "frame_count": frame_count,
                "frame_hop_s": FRAME_HOP_S,
                "audio": str(row["audio"]),
                "audio_sha256": str(row.get("audio_sha256") or _sha256(audio)),
                "model": model,
                "base_url_host": base_url.split("/", 3)[2] if "://" in base_url else base_url,
                "env_file_name": profile_name,
                "overall_confidence": 0.0,
                "overall_reason": f"teacher validation failed closed: {type(last_error).__name__}: {last_error}",
                "islands": [],
                "unsure_spans": [{
                    "label": "unsure",
                    "start_s": 0.0,
                    "end_s": duration_s,
                    "start_frame": 0,
                    "end_frame": frame_count,
                    "reason": "teacher request failed validation; exclude the whole source from outside truth",
                }],
                "reviewed_full_source": False,
                "preaudit_provenance": f"omni:{model}:validation_failure",
                "human_review_required": True,
                "training_manifest_allowed": False,
                "teacher_failed_closed": True,
                "attempts": args.max_attempts,
            }
            with labels_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")
            existing[source_id] = label
            print(
                f"omni_candidate_island={len(existing)}/{len(rows)} provider={profile_name} "
                f"source_id={source_id} failed_closed_unsure=1",
                flush=True,
            )
        if index < len(pending) and args.request_interval_s > 0:
            time.sleep(args.request_interval_s)
    result_rows = _rows(labels_path)
    summary = {"schema": SUMMARY_SCHEMA, "prompt_version": PROMPT_VERSION, "model": model, "env_file_name": profile_name, "audio_content_mode": audio_content_mode, "base_url_host": base_url.split("/", 3)[2] if "://" in base_url else base_url, "manifest": str(manifest), "manifest_sha256": _sha256(manifest), "source_count": len(result_rows), "labeled_count": len(result_rows), "manual_review_required": True, "training_manifest_allowed": False, "labels": str(labels_path), "raw_responses": str(raw_path)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_progress(progress_path, {"schema": "candidate_island_scorer_v11_omni_progress_v1", "status": "completed", "provider_profile": profile_name, "model": model, "completed": len(result_rows), "total": len(rows), "pending": 0, "elapsed_s": round(time.perf_counter() - started, 3), "summary": str(output / "summary.json")})
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--env-file",
        default="gemini",
        choices=("qwen", "gemini"),
        help="Named ~/.config/omni profile. Gemini is the default; use qwen explicitly.",
    )
    parser.add_argument("--api-key-env", default=",".join(DEFAULT_API_KEY_ENV_CANDIDATES))
    parser.add_argument("--model-env", default="OMNI_MODEL,QWEN_OMNI_MODEL")
    parser.add_argument("--base-url-env", default=",".join(DEFAULT_BASE_URL_ENV_CANDIDATES))
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-interval-s", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
