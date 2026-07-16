#!/usr/bin/env python3
"""Compile teacher-approved full clips into a unique semantic-core inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.boundary.ja.build_galgame_synthetic_timeline import (
    load_excluded_source_audio_ids,
)
from tools.boundary.ja.label_galgame_semantic_core_text_with_omni import (
    load_excluded_candidate_audio_ids,
)


TEACHER_SCHEMA = "galgame_semantic_core_text_teacher_v1"
INVENTORY_SCHEMA = "galgame_approved_semantic_core_v1"
SUMMARY_SCHEMA = "galgame_approved_semantic_core_summary_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compile_inventory(
    *,
    labels: Path | list[Path],
    output: Path,
    summary_path: Path,
    count: int,
    seed: int,
    excluded_candidate_manifests: list[Path] | None = None,
    excluded_source_manifests: list[Path] | None = None,
) -> dict[str, Any]:
    label_paths = [labels] if isinstance(labels, Path) else list(labels)
    label_rows = [row for path in label_paths for row in _rows(path)]
    if not label_rows or any(row.get("schema") != TEACHER_SCHEMA for row in label_rows):
        raise ValueError("semantic-core teacher labels are empty or incompatible")
    label_ids = [str(row["audio_id"]) for row in label_rows]
    if len(label_ids) != len(set(label_ids)):
        raise ValueError("semantic-core teacher labels contain duplicate audio ids")

    excluded: set[str] = set()
    excluded_source_ids = load_excluded_source_audio_ids(
        [str(path) for path in excluded_source_manifests or []]
    )
    excluded.update(excluded_source_ids)
    excluded_candidate_ids = load_excluded_candidate_audio_ids(
        [str(path) for path in excluded_candidate_manifests or []]
    )
    excluded.update(excluded_candidate_ids)
    overlap = sorted(set(label_ids) & excluded)
    if overlap:
        raise ValueError(f"teacher labels overlap excluded cores: {len(overlap)}")

    approved = [row for row in label_rows if row.get("label") == "all_semantic"]
    if len(approved) < count:
        raise ValueError(f"approved semantic cores {len(approved)} < requested {count}")
    rng = np.random.default_rng(seed)
    indexes = rng.choice(len(approved), size=count, replace=False)
    selected = [approved[int(index)] for index in indexes]
    output_rows = [
        {
            "schema": INVENTORY_SCHEMA,
            "approval_schema": TEACHER_SCHEMA,
            "approval_prompt_version": str(row["prompt_version"]),
            "approval_model": str(row["model"]),
            "approval_label": "all_semantic",
            "audio_id": str(row["audio_id"]),
            "audio": str(row["audio"]),
            "duration_s": float(row["duration_s"]),
            "text": str(row["reference_text"]),
            "source": str(row.get("source") or ""),
        }
        for row in selected
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    label_counts = Counter(str(row.get("label") or "") for row in label_rows)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "teacher_schema": TEACHER_SCHEMA,
        "inventory_schema": INVENTORY_SCHEMA,
        "teacher_label_count": len(label_rows),
        "teacher_label_manifests": [str(path) for path in label_paths],
        "teacher_label_counts": dict(sorted(label_counts.items())),
        "approved_available_count": len(approved),
        "selected_count": len(output_rows),
        "unique_core_count": len({row["audio_id"] for row in output_rows}),
        "max_core_use_count": 1,
        "excluded_total_audio_id_count": len(excluded),
        "excluded_candidate_audio_id_count": len(excluded_candidate_ids),
        "excluded_source_audio_id_count": len(excluded_source_ids),
        "excluded_overlap_count": 0,
        "selection": "seeded_without_replacement_v1",
        "seed": seed,
        "inventory": str(output),
        "inventory_sha256": digest,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile unique all-semantic Galgame full clips."
    )
    parser.add_argument("--labels", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--exclude-candidate-manifest", action="append")
    parser.add_argument("--exclude-source-manifest", action="append")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            compile_inventory(
                labels=[Path(path) for path in args.labels],
                output=Path(args.output),
                summary_path=Path(args.summary),
                count=args.count,
                seed=args.seed,
                excluded_candidate_manifests=[
                    Path(path) for path in args.exclude_candidate_manifest or []
                ],
                excluded_source_manifests=[
                    Path(path) for path in args.exclude_source_manifest or []
                ],
            ),
            ensure_ascii=False,
        )
    )
