#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


PUNCT_ONLY_CHARS = set(" \t\r\n。、，,.!?！？…・･ー-~～'\"“”‘’()（）[]【】{}<>《》:：;；/\\|")


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    rows.append(dict(payload))
    return rows


def informative_text(value: str) -> str:
    text = str(value or "").strip()
    return "".join(char for char in text if char not in PUNCT_ONLY_CHARS)


def text_char_count(value: str) -> int:
    return len(informative_text(value))


def resolve_audio_path(value: str) -> str:
    path = Path(str(value or ""))
    if not str(path):
        return ""
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def row_reason(row: Mapping[str, Any], *, no_overlap_s: float, low_overlap_ratio: float) -> str | None:
    text = str(row.get("text") or "")
    if text_char_count(text) <= 0:
        return None
    label_quality = str(row.get("label_quality") or "")
    manual_overlap_s = float(row.get("manual_overlap_s") or 0.0)
    manual_overlap_ratio = float(row.get("manual_overlap_ratio") or 0.0)
    if label_quality == "negative":
        return "manual_negative_asr_text"
    if manual_overlap_s <= no_overlap_s:
        return "no_overlap_asr_text"
    if manual_overlap_ratio < low_overlap_ratio:
        return "low_overlap_asr_text"
    return None


def candidate_audio_id(row: Mapping[str, Any]) -> str:
    audio_id = str(row.get("audio_id") or "audio")
    vad = str(row.get("vad") or "vad").replace("-", "_")
    chunk_index = int(row.get("chunk_index") or 0)
    return f"{audio_id}__{vad}__chunk{chunk_index:03d}"


def to_candidate(row: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    text = str(row.get("text") or "")
    raw_text = str(row.get("raw_text") or "")
    audio = resolve_audio_path(str(row.get("chunk_path") or ""))
    duration_s = float(row.get("duration_s") or 0.0)
    manual_overlap_s = float(row.get("manual_overlap_s") or 0.0)
    manual_overlap_ratio = float(row.get("manual_overlap_ratio") or 0.0)
    vad = str(row.get("vad") or "")
    original_audio_id = str(row.get("audio_id") or "")
    chunk_index = int(row.get("chunk_index") or 0)
    display_text = (
        f"ASR: {text}\n"
        f"raw: {raw_text}\n"
        f"vad: {vad} chunk={chunk_index} "
        f"source_label={row.get('label_quality') or ''} "
        f"overlap={manual_overlap_s:.2f}s/{manual_overlap_ratio:.3f}"
    )
    return {
        "audio_id": candidate_audio_id(row),
        "original_audio_id": original_audio_id,
        "audio": audio,
        "duration_s": duration_s,
        "source": f"downstream-asr::{vad}",
        "text": display_text,
        "asr_text": text,
        "raw_text": raw_text,
        "reason": reason,
        "label_quality": str(row.get("label_quality") or ""),
        "vad": vad,
        "chunk_index": chunk_index,
        "original_start": float(row.get("start") or 0.0),
        "original_end": float(row.get("end") or 0.0),
        "manual_overlap_s": manual_overlap_s,
        "manual_overlap_ratio": manual_overlap_ratio,
        "avg_logprob": row.get("avg_logprob"),
        "no_speech_prob": row.get("no_speech_prob"),
        "compression_ratio": row.get("compression_ratio"),
        "teacher_segments": {},
    }


def select_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    no_overlap_s: float,
    low_overlap_ratio: float,
    max_candidates: int | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        reason = row_reason(row, no_overlap_s=no_overlap_s, low_overlap_ratio=low_overlap_ratio)
        if reason is None:
            continue
        candidate = to_candidate(row, reason=reason)
        key = str(candidate["audio_id"])
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)

    candidates.sort(
        key=lambda row: (
            reason_priority(str(row.get("reason") or "")),
            -text_char_count(str(row.get("asr_text") or "")),
            -float(row.get("duration_s") or 0.0),
            float(row.get("manual_overlap_ratio") or 0.0),
            str(row.get("audio_id") or ""),
        )
    )
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return candidates


def reason_priority(reason: str) -> int:
    order = {
        "manual_negative_asr_text": 0,
        "no_overlap_asr_text": 1,
        "low_overlap_asr_text": 2,
    }
    return order.get(reason, 99)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "reason",
        "audio_id",
        "original_audio_id",
        "audio",
        "duration_s",
        "vad",
        "chunk_index",
        "label_quality",
        "manual_overlap_s",
        "manual_overlap_ratio",
        "avg_logprob",
        "no_speech_prob",
        "compression_ratio",
        "asr_text",
        "raw_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl_rows(Path(args.asr_outputs))
    candidates = select_candidates(
        rows,
        no_overlap_s=args.no_overlap_s,
        low_overlap_ratio=args.low_overlap_ratio,
        max_candidates=args.max_candidates,
    )

    jsonl_path = output_dir / "audit_candidates.jsonl"
    csv_path = output_dir / "audit_candidates.csv"
    summary_path = output_dir / "audit_candidate_summary.json"
    write_jsonl(jsonl_path, candidates)
    write_csv(csv_path, candidates)
    summary = {
        "asr_outputs": args.asr_outputs,
        "rows": len(rows),
        "candidates": len(candidates),
        "max_candidates": args.max_candidates,
        "no_overlap_s": args.no_overlap_s,
        "low_overlap_ratio": args.low_overlap_ratio,
        "reason_counts": dict(sorted(Counter(str(row.get("reason") or "") for row in candidates).items())),
        "label_quality_counts": dict(sorted(Counter(str(row.get("label_quality") or "") for row in candidates).items())),
        "output_jsonl": str(jsonl_path),
        "output_csv": str(csv_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"candidates={jsonl_path}")
    print(f"csv={csv_path}")
    print(f"summary={summary_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select downstream ASR hard-negative or target-nonverbal chunks for manual review."
    )
    parser.add_argument("--asr-outputs", required=True, help="ASR outputs JSONL from a downstream diagnostics run.")
    parser.add_argument("--no-overlap-s", type=float, default=0.0)
    parser.add_argument("--low-overlap-ratio", type=float, default=0.1)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "speech-boundary-ja" / "asr-hard-negative-audit"),
    )
    args = parser.parse_args(argv)
    if args.no_overlap_s < 0.0:
        parser.error("--no-overlap-s must be non-negative")
    if args.low_overlap_ratio < 0.0:
        parser.error("--low-overlap-ratio must be non-negative")
    if args.max_candidates is not None and args.max_candidates <= 0:
        parser.error("--max-candidates must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
