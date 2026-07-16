#!/usr/bin/env python3
"""Project exact unique-core spans onto new Runtime v10 provisional chunks."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "cueqc_v13_runtime_chunk_label_v1"
SUMMARY_SCHEMA = "cueqc_v13_runtime_chunk_label_summary_v1"
SAMPLE_RATE = 16000


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _overlap_samples(
    start_sample: int, end_sample: int, core: dict[str, Any]
) -> int:
    return max(
        0,
        min(end_sample, int(core["end_sample"]))
        - max(start_sample, int(core["start_sample"])),
    )


def project(
    *, source_manifest: Path, runtime_chunks: Path, output: Path
) -> dict[str, Any]:
    sources = {str(row["sample_id"]): row for row in _rows(source_manifest)}
    result: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for chunk in _rows(runtime_chunks):
        sample_id = str(chunk["sample_id"])
        source = sources.get(sample_id)
        if source is None:
            raise ValueError(f"runtime chunk has no source truth: {sample_id}")
        start_sample = int(round(float(chunk["start_s"]) * SAMPLE_RATE))
        end_sample = int(round(float(chunk["end_s"]) * SAMPLE_RATE))
        overlaps = [
            {
                "core_id": str(core["core_id"]),
                "overlap_samples": _overlap_samples(start_sample, end_sample, core),
            }
            for core in source.get("core_spans") or []
        ]
        overlaps = [row for row in overlaps if row["overlap_samples"] > 0]
        label = "keep" if overlaps else "drop"
        counts[label] += 1
        result.append(
            {
                "schema": SCHEMA,
                "sample_id": sample_id,
                "subisland_id": str(chunk["subisland_id"]),
                "audio": str(chunk["audio"]),
                "start_s": float(chunk["start_s"]),
                "end_s": float(chunk["end_s"]),
                "duration_s": float(chunk["duration_s"]),
                "source_partition": str(source["source_partition"]),
                "label": label,
                "label_source": "exact_unique_semantic_core_sample_intersection_v1",
                "semantic_core_overlaps": overlaps,
                "pre_asr_candidate": chunk.get("pre_asr_candidate"),
                "semantic_event_ids": list(chunk.get("semantic_event_ids") or []),
                "inner_edge_prediction": dict(chunk.get("inner_edge_prediction") or {}),
            }
        )
    _write(output, result)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "source_count": len({row["sample_id"] for row in result}),
        "chunk_count": len(result),
        "label_counts": dict(sorted(counts.items())),
        "new_chunk_independent_labeling": True,
        "parent_label_inheritance": False,
        "sample_coordinate_contract": "float_seconds_rounded_to_exact_16khz_samples_v1",
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--runtime-chunks", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            project(
                source_manifest=Path(args.source_manifest),
                runtime_chunks=Path(args.runtime_chunks),
                output=Path(args.output),
            ),
            ensure_ascii=False,
        )
    )
