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
PROMPT_VERSION = "candidate_island_scorer_v11_omni_preaudit_dialogue_islands_v6"

SYSTEM_PROMPT = """你是 1.7B Scorer v11 的候选岛预审 teacher。你的唯一职责是标出必须先保留给后续 Proposal / Split / CueQC 的连续候选对话岛。你不是 Split，不按句子、说话人、标点或语义单元切分；你也不是 CueQC，不做最终 keep/drop。

输入来自以日语为主的 JAV / Galgame 相近成人音频域。目标语言是日语；应特别留意短日语词、助词、应答词、耳语、口吃、含混或残缺发音，以及夹在呻吟或动作声中的带词义成人场景发言。这些目标不能按普通干净语音标准漏掉。但场景与语言信息只用于提高日语词语/对白锚点的召回，绝不能把纯呻吟、喘息、呼吸、亲吻声或动作声自动升级为 inside_candidate。

必须按以下优先顺序判断：
1. 先寻找明确或很可能含词语、音节、耳语、口吃、残缺发音、句尾或对白的波形，把它们作为 inside_candidate 锚点。不要以 ASR 能否转录作为判据。
2. 再围绕这些锚点形成连续候选岛。同一轮连续对话、几乎无安全间隔的相邻发言，以及对白内部或边缘的停顿、尾音、短呼吸、呻吟或动作声，应保持在同一个岛中，保证完整波形交给下游。句子和事件切分属于 Proposal + Split。
3. 明确不含词语且能够独立于对白删除的纯呻吟、喘息、呼吸、哭声、亲吻声、动作声、impact、音乐、静音或环境声属于 outside_candidate。即使它持续很久、强度很高、有人互动或与对白处于同一场景，也不能仅因此成为 inside_candidate。若整条 source 都是明确的纯非语义声音，必须允许 islands=[]。
4. 非语义声音只有在夹在同一轮对话内部、紧贴对白边缘，且移除会截断尾音或破坏连续对话波形时，才随该对话岛保留。纯非语义声音本身不能桥接相距较远的两轮对白。
5. 若局部听起来可能是词语、也可能只是呻吟或噪声，优先标为 unsure；不要为了高召回直接扩大成 inside_candidate。unsure 是标注不确定性，之后会映射为 ignore=-100。

边界合同：
- 不使用固定时长、静音阈值、hysteresis、ASR 文本、duration-only 规则或其他启发式。
- 同一场景、同一说话人、持续互动或声音连续，本身都不是合并理由。
- islands 与 unsure_spans 必须各自按时间排序、互不重叠，并且两组之间也不得重叠；它们之外的完整差集就是 outside_candidate。
- 输出当前 0-based 完整 source 坐标，单位为秒，不添加前后文，不使用原视频时间轴。

判例：
- 对白1 + 对白内部短呼吸/呻吟 + 对白2，且属于同一轮连续对话：输出一个完整 island。
- 全段只有明确纯呻吟/喘息，没有词语：islands=[]，不要因为声音连续而整段保留。
- 某段可能是词语，也可能只是呻吟：该局部输出 unsure_span。
- 对白 + 独立纯非语义活动 + 后续另一轮对白：输出两个 islands，中间差集为 outside_candidate。

只输出一个 JSON 对象，不要 Markdown：
{
  "source_id":"...",
  "islands":[{"start_s":0.0,"end_s":1.0,"confidence":0.0,"reason":"连续候选对话岛的简短理由"}],
  "unsure_spans":[{"start_s":0.0,"end_s":1.0,"reason":"无法确认是否含词语的局部"}],
  "overall_confidence":0.0,
  "overall_reason":"简短整体理由"
}
"""


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


def _resume_index(path: Path, *, model: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["source_id"]): row
        for row in _rows(path)
        if row.get("schema") == SCHEMA
        and row.get("model") == model
        and row.get("prompt_version") == PROMPT_VERSION
    }


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
        "target_language": "Japanese",
        "target_domain": "Japanese JAV / adult-scene audio with Galgame-like speech characteristics",
        "domain_rule": "increase recall for short, whispered, stuttered, ambiguous, or partially masked Japanese lexical speech, but never keep pure moans, pants, breaths, kisses, or action sounds merely because the source is JAV",
        "decision_order": [
            "find definite or probable lexical/dialogue anchors",
            "preserve the continuous waveform envelope of the same dialogue round",
            "leave independently removable definite nonlexical sound in the outside complement",
            "mark locally ambiguous possible words as unsure rather than defaulting to inside",
        ],
        "do_not_split": [
            "the same continuous dialogue round",
            "adjacent dialogue turns with almost no safely removable interval",
            "intra-dialogue pauses, pronunciation tails, breaths, or action sounds",
        ],
        "outside_candidate": "the complement containing definite independent nonlexical sound; a continuous pure moan/pant/breath scene may be entirely outside",
        "output_units": "continuous dialogue-candidate envelopes, never individual sentences or semantic units",
        "must_split": "independent definite nonlexical activity between separate dialogue rounds ends the prior island even when the scene and interaction continue",
        "nonsemantic_vocal_policy": "keep nonlexical sound only when attached to or enclosed by the same dialogue envelope; otherwise outside; possible lexical ambiguity goes to unsure",
        "anti_overmerge": "never return 0..duration merely because vocal activity, intimacy, emotion, or interaction is continuous; if there is no definite or probable dialogue and no ambiguity, return islands=[]",
        "range_contract": "islands and unsure_spans are sorted and mutually exclusive; their omitted complement is outside_candidate",
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
    classified = sorted(
        (
            (float(span["start_s"]), float(span["end_s"]), str(span["label"]))
            for span in (*islands, *unsure)
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    for previous, current in zip(classified, classified[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                "Scorer v11 teacher islands and unsure_spans must be mutually "
                f"exclusive: {previous[2]} {previous[0]}..{previous[1]} overlaps "
                f"{current[2]} {current[0]}..{current[1]}"
            )
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
    existing = _resume_index(labels_path, model=model)
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
