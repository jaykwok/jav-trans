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

from vad.fusionvad_ja import LabelRecord, read_jsonl, write_jsonl


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"manifest must be a JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[LabelRecord] = []
    manifest_rows: list[dict[str, Any]] = []
    for labels_path in args.labels:
        records.extend(read_jsonl(Path(labels_path)))
    for manifest_path in args.manifest:
        manifest_rows.extend(load_manifest(Path(manifest_path)))

    labels_path = output_dir / args.output_jsonl
    manifest_path = output_dir / args.output_manifest
    summary_path = output_dir / "combined_summary.json"
    write_jsonl(labels_path, records)
    manifest_path.write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "labels": [str(Path(path)) for path in args.labels],
        "manifests": [str(Path(path)) for path in args.manifest],
        "output_labels": str(labels_path),
        "output_manifest": str(manifest_path),
        "records": len(records),
        "manifest_rows": len(manifest_rows),
        "duration_s_total": sum(record.duration_s for record in records),
        "label_quality_counts": dict(sorted(Counter(record.label_quality for record in records).items())),
        "source_counts": dict(sorted(Counter(record.source for record in records).items())),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"labels={labels_path}")
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine FusionVAD-JA label JSONL files and audio manifests.")
    parser.add_argument("--labels", action="append", required=True, help="Label JSONL. Repeatable.")
    parser.add_argument("--manifest", action="append", required=True, help="Audio manifest JSON list. Repeatable.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "combined"))
    parser.add_argument("--output-jsonl", default="labels.jsonl")
    parser.add_argument("--output-manifest", default="manifest.json")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
