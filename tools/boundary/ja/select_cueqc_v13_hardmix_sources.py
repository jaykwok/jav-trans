#!/usr/bin/env python3
"""Select structurally discriminative hardmix sources for CueQC v13 replay."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _quotas(count: int) -> dict[str, int]:
    val = int(round(count * 0.10))
    test = int(round(count * 0.05))
    return {"train": count - val - test, "val": val, "test": test}


def select(
    *, details: Path, audio_root: Path, output: Path, count: int, seed: int
) -> dict[str, Any]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_partitions: dict[str, set[str]] = defaultdict(set)
    for row in _rows(details):
        boundaries = list(row.get("utterance_boundaries") or [])
        speech_segments = list(row.get("actual_speech_segments") or [])
        overlap = dict((row.get("augmentation") or {}).get("overlap_speech") or {})
        audio = audio_root / f"{row['audio_id']}.wav"
        if (
            len(speech_segments) < 3
            or len(boundaries) < 2
            or bool(overlap.get("enabled"))
            or not audio.exists()
        ):
            continue
        source_id = str(row.get("source_id") or "").strip()
        partition = str(row.get("source_partition") or "").strip()
        if not source_id or partition not in {"train", "val", "test"}:
            raise ValueError(
                "hardmix details require frozen source_id and "
                "source_partition=train|val|test"
            )
        source_partitions[source_id].add(partition)
        pools[partition].append(
            {
                "sample_id": str(row["audio_id"]),
                "source_id": source_id,
                "audio": str(audio),
                "source_partition": partition,
                "speech_unit_count": len(speech_segments),
                "inter_unit_boundary_count": len(boundaries),
                "duration_s": float(row.get("duration_s") or 0.0),
            }
        )
    if any(len(values) != 1 for values in source_partitions.values()):
        raise ValueError("hardmix source identity crosses partitions")
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    quotas = _quotas(count)
    if any(quotas[partition] <= 0 for partition in ("train", "val", "test")):
        raise ValueError("hardmix selection requires non-empty train/val/test quotas")
    for partition, quota in quotas.items():
        pool = pools[partition]
        if len(pool) < quota:
            raise ValueError(f"not enough {partition} sources: {len(pool)} < {quota}")
        durations = np.asarray([row["duration_s"] for row in pool], dtype=np.float64)
        order = np.argsort(durations)
        bins = np.array_split(order, max(1, quota))
        choices = [int(bucket[int(rng.integers(0, len(bucket)))]) for bucket in bins[:quota]]
        selected.extend(pool[index] for index in choices)
    rng.shuffle(selected)
    _write(output, selected)
    summary = {
        "schema": "cueqc_v13_hardmix_source_selection_v1",
        "seed": seed,
        "source_count": len(selected),
        "partition_counts": dict(Counter(row["source_partition"] for row in selected)),
        "selection": "partition_and_duration_stratified_without_overlap_v1",
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            select(
                details=Path(args.details),
                audio_root=Path(args.audio_root),
                output=Path(args.output),
                count=args.count,
                seed=args.seed,
            ),
            ensure_ascii=False,
        )
    )
