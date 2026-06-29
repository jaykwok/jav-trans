#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.asr.cueqc.offline_candidates import (
    aligned_payload_to_candidates,
    compact_payload_requires_transcript,
    infer_transcript_path,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def run(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]]
    aligned_path = Path(args.aligned)
    payload = read_json(aligned_path)
    if not isinstance(payload, Mapping):
        raise SystemExit("--aligned must point to a JSON object payload")
    transcript_path = Path(args.transcript) if args.transcript else infer_transcript_path(aligned_path)
    transcript_payload = None
    if transcript_path.exists():
        loaded_transcript = read_json(transcript_path)
        if not isinstance(loaded_transcript, Mapping):
            raise SystemExit("--transcript must point to a JSON object payload")
        transcript_payload = loaded_transcript
    elif args.transcript:
        raise SystemExit(f"--transcript not found: {transcript_path}")
    rows = aligned_payload_to_candidates(
        payload,
        video_id=args.video_id,
        transcript_payload=transcript_payload,
    )
    if not rows and compact_payload_requires_transcript(payload) and transcript_payload is None:
        raise SystemExit(
            "compact aligned_segments payload has transcript_chunk_count but no embedded chunks; "
            "pass --transcript or place the sibling .transcript.json next to --aligned"
        )
    if args.max_items is not None:
        rows = rows[: args.max_items]
    count = write_jsonl(Path(args.output), rows)
    summary = {
        "schema": "cueqc_candidate_export_summary_v1",
        "output": args.output,
        "candidate_count": count,
        "source_aligned": args.aligned or "",
        "source_transcript": str(transcript_path) if transcript_payload is not None else "",
        "video_id": args.video_id,
    }
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(f"candidates={args.output}")
    print(f"count={count}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CueQC cluster-first candidate JSONL.")
    parser.add_argument("--aligned", required=True, help="aligned_segments.json payload from a full run")
    parser.add_argument(
        "--transcript",
        default="",
        help="transcript.json from the same run. Defaults to a sibling .transcript.json next to --aligned.",
    )
    parser.add_argument("--output", required=True, help="cueqc_candidates.jsonl")
    parser.add_argument("--summary", default="", help="optional summary JSON")
    parser.add_argument("--video-id", default="")
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args(argv)
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
