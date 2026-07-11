#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.asr.cueqc.label_pre_asr_with_omni import (  # noqa: E402
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    DEFAULT_ENV_FILE,
    append_jsonl,
    call_omni,
    first_env_value,
    is_empty_audio_api_error,
    load_env_file,
    slice_audio_clip,
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _candidate_key(row: dict) -> str:
    return str(row.get("candidate_id") or row.get("sample_id"))


def _select(labels: list[dict], *, per_label: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    selected = [
        row for row in labels if row["label"] == "ambiguous_ignore"
    ]
    for label in ("definite_keep", "definite_drop"):
        pool = [row for row in labels if row["label"] == label]
        rng.shuffle(pool)
        selected.extend(pool[:per_label])
    result: list[dict] = []
    seen: set[str] = set()
    for row in selected:
        key = _candidate_key(row)
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def _clip_window(
    row: dict,
    *,
    old_labels: list[dict],
    max_clip_s: float,
) -> tuple[float, float]:
    start = float(row["start"])
    end = float(row["end"])
    if end - start <= max_clip_s:
        return start, end
    center = (start + end) / 2.0
    if row["label"] == "definite_keep":
        overlaps = [
            old
            for old in old_labels
            if old.get("label") == "definite_keep"
            and float(old["end"]) > start
            and float(old["start"]) < end
        ]
        if overlaps:
            best = max(
                overlaps,
                key=lambda old: min(end, float(old["end"]))
                - max(start, float(old["start"])),
            )
            center = (
                max(start, float(best["start"]))
                + min(end, float(best["end"]))
            ) / 2.0
    clip_start = min(max(start, center - max_clip_s / 2.0), end - max_clip_s)
    return clip_start, clip_start + max_clip_s


def _call_omni_with_rate_limit_retry(**kwargs):
    for attempt in range(6):
        try:
            return call_omni(**kwargs)
        except Exception as exc:
            message = str(exc).lower()
            if "429" not in message and "limit_requests" not in message:
                raise
            if attempt == 5:
                raise
            delay_s = min(30, 5 * (attempt + 1))
            print(
                f"omni_rate_limited retry={attempt + 1}/5 delay_s={delay_s}",
                flush=True,
            )
            time.sleep(delay_s)
    raise AssertionError("unreachable")


def run(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    candidates: dict[str, dict] = {}
    for path in args.candidates:
        for row in _read_jsonl(Path(path)):
            candidates[_candidate_key(row)] = row
    inherited = _read_jsonl(Path(args.inherited_labels))
    if args.all_candidates:
        selected = [
            row
            for row in inherited
            if not (
                row["label"] == "definite_drop"
                and float(row["duration_s"]) > args.max_clip_s
            )
        ]
    else:
        selected = _select(
            inherited,
            per_label=args.audit_per_label,
            seed=args.seed,
        )
    selected = selected[args.skip_items :]
    if args.max_items is not None:
        selected = selected[: args.max_items]
    old_by_audio: dict[str, list[dict]] = {}
    for row in _read_jsonl(Path(args.old_omni_labels)):
        old_by_audio.setdefault(str(row["audio_id"]), []).append(row)
    audio_by_id = {
        item.split("=", 1)[0]: Path(item.split("=", 1)[1])
        for item in args.audio
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels_path = output / "omni_labels.jsonl"
    raw_path = output / "omni_raw_responses.jsonl"
    existing = {
        _candidate_key(row) for row in _read_jsonl(labels_path)
    } if labels_path.exists() else set()
    _model_name, model = first_env_value(("OMNI_MODEL", "QWEN_OMNI_MODEL"))
    model = args.model or model or "qwen3.5-omni-flash"
    _key_name, api_key = first_env_value(DEFAULT_API_KEY_ENV_CANDIDATES)
    _url_name, base_url = first_env_value(DEFAULT_BASE_URL_ENV_CANDIDATES)
    counts: Counter[str] = Counter()
    agreements: Counter[str] = Counter()
    for position, inherited_row in enumerate(selected, start=1):
        key = _candidate_key(inherited_row)
        if key in existing:
            continue
        candidate = candidates[key]
        audio_id = str(inherited_row["audio_id"])
        clip_start, clip_end = _clip_window(
            inherited_row,
            old_labels=old_by_audio[audio_id],
            max_clip_s=args.max_clip_s,
        )
        clip_path = (
            output
            / "audio_clips"
            / audio_id
            / f"{key}-{clip_start:.3f}-{clip_end:.3f}.mp3"
        )
        slice_audio_clip(
            source_audio=audio_by_id[audio_id],
            row={
                "start": clip_start,
                "end": clip_end,
                "duration_s": clip_end - clip_start,
            },
            output_path=clip_path,
            fmt="mp3",
            bitrate=args.audio_bitrate,
            sample_rate=16000,
            force=False,
        )
        try:
            parsed, raw = _call_omni_with_rate_limit_retry(
                audio_path=clip_path,
                fmt="mp3",
                audio_content_mode=args.audio_content_mode,
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout_s=args.timeout_s,
                store_stream_chunks=False,
            )
        except Exception as exc:
            if not is_empty_audio_api_error(exc):
                raise
            parsed = {
                "label": "drop",
                "confidence": 1.0,
                "flags": ["short_fragment"],
                "reason": "压缩后的极短音频为空，不含可送 ASR 的语义语音。",
            }
            raw = {"error": str(exc), "local_fallback": "empty_audio_to_drop"}
        omni_label = str(parsed.get("label") or "unsure").lower()
        confidence = min(1.0, max(0.0, float(parsed.get("confidence") or 0.0)))
        if confidence < args.confidence or omni_label not in {"keep", "drop"}:
            label = "ambiguous_ignore"
        else:
            label = f"definite_{omni_label}"
        result = {
            "schema": "pre_asr_semantic_chunk_omni_audit_v1",
            "sample_id": candidate["sample_id"],
            "candidate_id": candidate["candidate_id"],
            "audio_id": audio_id,
            "video_id": audio_id,
            "chunk_index": int(candidate["chunk_index"]),
            "start": float(candidate["start"]),
            "end": float(candidate["end"]),
            "duration_s": float(candidate["duration_s"]),
            "label": label,
            "display_decision": (
                omni_label if label != "ambiguous_ignore" else "ambiguous_ignore"
            ),
            "training_label_included": label != "ambiguous_ignore",
            "label_source": f"omni:{model}",
            "omni_label": omni_label,
            "omni_confidence": confidence,
            "omni_flags": list(parsed.get("flags") or []),
            "omni_reason": str(parsed.get("reason") or ""),
            "inherited_label": inherited_row["label"],
            "clip_start": clip_start,
            "clip_end": clip_end,
            "audio_clip": str(clip_path),
        }
        append_jsonl(labels_path, result)
        append_jsonl(
            raw_path,
            {
                "schema": "pre_asr_semantic_chunk_omni_raw_v1",
                "candidate_id": key,
                "parsed": parsed,
                "response": raw,
            },
        )
        counts[label] += 1
        agreements[
            "agree" if label == inherited_row["label"] else "disagree"
        ] += 1
        if position % 25 == 0:
            print(
                f"processed={position}/{len(selected)} labels={dict(counts)} "
                f"agreement={dict(agreements)}",
                flush=True,
            )
    summary = {
        "schema": "pre_asr_semantic_chunk_omni_audit_summary_v1",
        "selected": len(selected),
        "labels": dict(counts),
        "agreement": dict(agreements),
        "max_clip_s": args.max_clip_s,
        "audio_format": "mp3",
        "audio_bitrate": args.audio_bitrate,
        "sample_rate": 16000,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", action="append", required=True)
    parser.add_argument("--inherited-labels", required=True)
    parser.add_argument("--old-omni-labels", required=True)
    parser.add_argument("--audio", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-per-label", type=int, default=100)
    parser.add_argument("--all-candidates", action="store_true")
    parser.add_argument("--skip-items", type=int, default=0)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--max-clip-s", type=float, default=6.0)
    parser.add_argument("--audio-bitrate", default="32k")
    parser.add_argument("--confidence", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--model", default="")
    parser.add_argument("--audio-content-mode", default="input_audio")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
