#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vad.fusionvad_ja import build_weak_positive_record, write_jsonl


def load_manifest_rows(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("materialized HF manifest must be a JSON list")
    return [row for row in payload if isinstance(row, Mapping)]


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest_rows(Path(args.manifest))
    records = []
    skipped = []
    for index, row in enumerate(rows):
        if row.get("error"):
            skipped.append({"index": index, "reason": "materialize_error", "error": row.get("error")})
            continue
        audio_id = str(row.get("audio_id") or Path(str(row.get("audio") or f"galgame-{index:06d}")).stem)
        duration_s = float(row.get("duration_s") or 0.0)
        if duration_s <= 0.0:
            skipped.append({"index": index, "audio_id": audio_id, "reason": "invalid_duration"})
            continue
        text = str(row.get("text") or "")
        source = str(row.get("source") or args.source)
        records.append(
            build_weak_positive_record(
                audio_id=audio_id,
                source=source,
                duration_s=duration_s,
                text=text,
                frame_hop_s=args.frame_hop_s,
                teacher_name=args.teacher_name,
                trim_head_s=args.trim_head_s,
                trim_tail_s=args.trim_tail_s,
            )
        )

    labels_path = output_dir / args.output_jsonl
    summary_path = output_dir / "weak_label_summary.json"
    skipped_path = output_dir / "weak_label_skipped.json"
    write_jsonl(labels_path, records)
    skipped_path.write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "manifest": str(Path(args.manifest)),
        "records": len(records),
        "skipped": len(skipped),
        "label_quality_counts": dict(sorted(Counter(record.label_quality for record in records).items())),
        "frame_hop_s": args.frame_hop_s,
        "trim_head_s": args.trim_head_s,
        "trim_tail_s": args.trim_tail_s,
        "labels": str(labels_path),
        "skipped_report": str(skipped_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"weak_labels={labels_path}")
    print(f"skipped_report={skipped_path}")
    print(f"summary={summary_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Galgame weak-positive FusionVAD-JA labels from materialized HF audio.")
    parser.add_argument("--manifest", required=True, help="hf_audio_manifest.json from materialize_hf_audio.py.")
    parser.add_argument("--source", default="litagin/Galgame_Speech_ASR_16kHz")
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--teacher-name", default="galgame_weak_positive")
    parser.add_argument("--trim-head-s", type=float, default=0.0)
    parser.add_argument("--trim-tail-s", type=float, default=0.0)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "galgame-weak-labels"))
    parser.add_argument("--output-jsonl", default="galgame_weak_labels.jsonl")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
