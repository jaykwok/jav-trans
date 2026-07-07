#!/usr/bin/env python3
"""Re-export joint-window Split features under the CURRENT decoder params and
remap existing Omni split labels onto the new candidate set by time.

Why: joint v1/v2 window features were exported with the old candidate decoder
(quantile floors 0.50 / NMS 0.20s). The runtime now proposes a denser candidate
set (0.10 / 0.12s), so training on the old islands would re-create the
train/runtime candidate mismatch documented on 2026-07-04. This tool re-runs
the boundary chain per window into ``features_requant/`` and rewrites each
label's ``feature_index`` to the nearest new candidate within ``--time-match-s``
(default 0.05s); unmatched labels are dropped and counted, never guessed.

Only the Semantic Split side is touched: ``pre_asr_candidates`` paths keep
pointing at the original export so Pre-ASR labels stay aligned with the chunk
set they were produced from.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for value in (PROJECT_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def remap_label_rows(
    labels: list[dict[str, Any]],
    times_by_window: dict[str, np.ndarray],
    *,
    time_match_s: float,
    cut_time_match_s: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Move each label to the nearest new candidate; drop labels with no match.

    A new candidate is consumed by at most one label (greedy by distance) so two
    old labels cannot collapse onto the same new candidate. ``cut`` / ``unsure``
    labels may use a wider tolerance (``cut_time_match_s``): the runtime cut
    match window is 0.30s, so a proposer peak sitting 50-150ms from the old
    energy-valley position is still the same semantic boundary. They are also
    matched BEFORE ``continue`` labels so scarce real cut anchors cannot lose
    their nearest candidate to an abundant continue label.
    """

    wide_tolerance = time_match_s if cut_time_match_s is None else cut_time_match_s

    def label_tolerance(row: dict[str, Any]) -> float:
        return (
            wide_tolerance
            if str(row.get("label")) in {"cut", "unsure"}
            else time_match_s
        )

    def label_phase(row: dict[str, Any]) -> int:
        return 0 if str(row.get("label")) in {"cut", "unsure"} else 1

    by_window: dict[str, list[int]] = {}
    for position, row in enumerate(labels):
        by_window.setdefault(str(row["window_id"]), []).append(position)
    remapped: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for window_id, positions in sorted(by_window.items()):
        times = times_by_window.get(window_id)
        if times is None or times.size == 0:
            for position in positions:
                dropped.append({**labels[position], "drop_reason": "window_missing"})
            continue
        candidates = np.asarray(times, dtype=np.float64)
        pairs = sorted(
            (
                (
                    label_phase(labels[position]),
                    abs(float(labels[position]["time_s"]) - float(candidates[index])),
                    position,
                    index,
                )
                for position in positions
                for index in range(candidates.size)
            ),
            key=lambda item: (item[0], item[1]),
        )
        assigned_labels: set[int] = set()
        assigned_candidates: set[int] = set()
        chosen: dict[int, int] = {}
        for _phase, distance, position, index in pairs:
            if distance > label_tolerance(labels[position]):
                continue
            if position in assigned_labels or index in assigned_candidates:
                continue
            assigned_labels.add(position)
            assigned_candidates.add(index)
            chosen[position] = index
        for position in positions:
            row = labels[position]
            index = chosen.get(position)
            if index is None:
                dropped.append({**row, "drop_reason": "no_candidate_within_tolerance"})
                continue
            remapped.append(
                {
                    **row,
                    "feature_index": int(index),
                    "time_s": float(candidates[index]),
                    "remap": {
                        "source_feature_index": int(row["feature_index"]),
                        "source_time_s": float(row["time_s"]),
                        "distance_s": abs(float(row["time_s"]) - float(candidates[index])),
                    },
                }
            )
    return remapped, dropped


def run(args: argparse.Namespace) -> None:
    from tools.datasets.prepare_joint_boundary_omni_dataset import (
        _export_boundary_features,
    )

    dataset = Path(args.dataset_dir)
    windows = _read_jsonl(dataset / "source_windows.jsonl")
    if not windows:
        raise ValueError(f"source_windows.jsonl is empty: {dataset}")
    labels_path = dataset / "annotations" / "omni_joint" / "split_labels.jsonl"
    if not labels_path.exists():
        labels_path = dataset / "semantic_split" / "labels.jsonl"
    labels = _read_jsonl(labels_path)
    if not labels:
        raise ValueError(f"no split labels found under {dataset}")

    times_by_window: dict[str, np.ndarray] = {}
    new_paths: dict[str, dict[str, str]] = {}
    for position, row in enumerate(windows, start=1):
        window_id = str(row["window_id"])
        feature_dir = dataset / "features_requant" / window_id
        payload = _export_boundary_features(
            wav_path=Path(row["audio_wav"]),
            feature_dir=feature_dir,
        )
        bundle = np.load(payload["semantic_split_features"])
        times_by_window[window_id] = np.asarray(
            bundle["proposal_times_s"], dtype=np.float64
        )
        new_paths[window_id] = {
            "semantic_split_features": str(payload["semantic_split_features"]),
            "semantic_split_metadata": str(payload["semantic_split_metadata"]),
        }
        print(
            f"reexported window={position}/{len(windows)} id={window_id} "
            f"candidates={times_by_window[window_id].size} "
            f"resumed={bool(payload.get('resumed'))}",
            flush=True,
        )

    remapped, dropped = remap_label_rows(
        labels,
        times_by_window,
        time_match_s=args.time_match_s,
        cut_time_match_s=args.cut_time_match_s,
    )
    remapped_path = dataset / "semantic_split" / "labels.requant.jsonl"
    dropped_path = dataset / "semantic_split" / "labels.requant.dropped.jsonl"
    _write_jsonl(remapped_path, remapped)
    _write_jsonl(dropped_path, dropped)
    summary = {
        "schema": "joint_split_feature_requant_summary_v1",
        "dataset_dir": str(dataset.resolve()),
        "window_count": len(windows),
        "label_count": len(labels),
        "remapped_count": len(remapped),
        "dropped_count": len(dropped),
        "drop_reasons": dict(
            Counter(str(row.get("drop_reason")) for row in dropped)
        ),
        "label_counts_remapped": dict(
            Counter(str(row.get("label")) for row in remapped)
        ),
        "time_match_s": args.time_match_s,
        "cut_time_match_s": args.cut_time_match_s,
        "applied": bool(args.apply),
        "remapped_labels": str(remapped_path.resolve()),
        "dropped_labels": str(dropped_path.resolve()),
    }
    if args.apply:
        backup_windows = dataset / "source_windows.pre_requant.jsonl"
        if not backup_windows.exists():
            backup_windows.write_text(
                (dataset / "source_windows.jsonl").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        updated_rows = []
        for row in windows:
            window_id = str(row["window_id"])
            updated_rows.append({**row, **new_paths.get(window_id, {})})
        _write_jsonl(dataset / "source_windows.jsonl", updated_rows)
        split_labels = dataset / "semantic_split" / "labels.jsonl"
        backup_labels = dataset / "semantic_split" / "labels.pre_requant.jsonl"
        if split_labels.exists() and not backup_labels.exists():
            backup_labels.write_text(
                split_labels.read_text(encoding="utf-8"), encoding="utf-8"
            )
        _write_jsonl(split_labels, remapped)
        summary["windows_backup"] = str(backup_windows.resolve())
        summary["labels_backup"] = str(backup_labels.resolve())
    summary_path = dataset / "semantic_split" / "requant_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-export joint-window Split candidate features under current decoder "
            "params and remap Omni labels by time."
        )
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--time-match-s", type=float, default=0.05)
    parser.add_argument(
        "--cut-time-match-s",
        type=float,
        default=0.15,
        help=(
            "Wider match tolerance for cut/unsure labels (runtime cut match is "
            "0.30s; must stay below split NMS 0.12s x2 so boundaries cannot swap)."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Swap source_windows.jsonl split-feature paths and semantic_split/"
            "labels.jsonl to the requantized versions (originals backed up)."
        ),
    )
    args = parser.parse_args()
    if args.time_match_s <= 0:
        parser.error("--time-match-s must be positive")
    if args.cut_time_match_s < args.time_match_s:
        parser.error("--cut-time-match-s must be >= --time-match-s")
    return args


if __name__ == "__main__":
    run(parse_args())
