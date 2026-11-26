#!/usr/bin/env python3
"""The P1b gate: how the trained frame head confuses the three classes on val.

Held-out rows only. The labels and the head were built from the same corpora, so
a train-split number would report how well the head memorised the L1 alignment
rather than whether the classes are separable.

The cell that decides the gate is `vocalisation -> speech`. That is the direction
that puts moaning back on screen, and it is the one the whole three-class change
exists to control; `speech -> vocalisation` is the expensive direction for the
viewer and is reported beside it, because a head that fixes one by breaking the
other has not fixed anything.

Frames labelled `ignore` are excluded, not counted as correct. They are the
boundary bands and the regions no source could label, and scoring them either
way would report the label builder's caution as the head's accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.alignment import FRAME_CLASSES, AlignmentHead  # noqa: E402
from tools.align.frame_teacher_supervision import IGNORE_LABEL  # noqa: E402

SCHEMA = "frame_class_head_val_confusion_v1"


def read_index(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", required=True)
    parser.add_argument("--labels", required=True, help="frame_class_labels.npz")
    parser.add_argument("--cache-root", default="datasets/train/align-features-v2")
    parser.add_argument(
        "--cache",
        action="append",
        default=None,
        help="repeatable cache directory name; defaults to every cache present",
    )
    parser.add_argument("--limit-per-cache", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    head = AlignmentHead.load(str(PROJECT_ROOT / args.head))
    if not head.frame_head_available:
        raise SystemExit(
            f"{args.head} carries no frame classifier; this gate is about the "
            "v2 head's three-class output and a v1 checkpoint cannot answer it"
        )
    cache_root = PROJECT_ROOT / args.cache_root
    names = args.cache or sorted(
        path.name for path in cache_root.iterdir() if (path / "index.jsonl").exists()
    )

    with np.load(PROJECT_ROOT / args.labels) as payload:
        labels = {key: np.asarray(payload[key], dtype=np.int8) for key in payload.files}

    size = len(FRAME_CLASSES)
    confusion = np.zeros((size, size), dtype=np.int64)
    per_cache: dict[str, dict] = {}
    for name in names:
        cache_dir = cache_root / name
        rows = [
            row
            for row in read_index(cache_dir / "index.jsonl")
            if row.get("partition") == "val" and f"{name}/{row['audio_id']}" in labels
        ]
        if args.limit_per_cache:
            rows = rows[: args.limit_per_cache]
        if not rows:
            continue
        shards: dict[str, np.ndarray] = {}
        local = np.zeros((size, size), dtype=np.int64)
        for row in rows:
            shard = shards.get(row["shard"])
            if shard is None:
                shard = np.load(cache_dir / row["shard"], mmap_mode="r")
                shards[row["shard"]] = shard
            start = int(row["offset"])
            features = np.asarray(
                shard[start : start + int(row["frames"])], dtype=np.float32
            )
            posteriors = head.frame_posteriors(features)
            truth = labels[f"{name}/{row['audio_id']}"]
            width = min(len(truth), len(posteriors))
            truth = truth[:width]
            predicted = posteriors[:width].argmax(axis=-1)
            keep = truth != IGNORE_LABEL
            if not keep.any():
                continue
            np.add.at(local, (truth[keep].astype(int), predicted[keep].astype(int)), 1)
        confusion += local
        per_cache[name] = {
            "val_rows_scored": len(rows),
            "labelled_frames": int(local.sum()),
            "confusion": local.tolist(),
        }
        print(f"  {name}: {len(rows)} val rows, {int(local.sum())} labelled frames",
              flush=True)

    totals = confusion.sum(axis=1)
    rates = np.divide(
        confusion, np.maximum(totals[:, None], 1), dtype=np.float64
    )
    report = {
        "schema": SCHEMA,
        "head": args.head,
        "labels": args.labels,
        "classes": list(FRAME_CLASSES),
        "confusion_counts": confusion.tolist(),
        "confusion_row_normalised": [[round(v, 6) for v in row] for row in rates],
        "labelled_frames": int(confusion.sum()),
        "per_class_recall": {
            name: round(float(rates[index, index]), 6)
            for index, name in enumerate(FRAME_CLASSES)
        },
        "vocalisation_to_speech": round(float(rates[1, 2]), 6),
        "speech_to_vocalisation": round(float(rates[2, 1]), 6),
        "by_cache": per_cache,
    }
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nlabelled val frames: {confusion.sum():,}")
    header = "  ".join(f"{name[:5]:>8}" for name in FRAME_CLASSES)
    print(f"{'truth \\ pred':>14}  {header}      recall")
    for index, name in enumerate(FRAME_CLASSES):
        cells = "  ".join(f"{rates[index, other]:>8.4f}" for other in range(size))
        print(f"{name:>14}  {cells}    {rates[index, index]:>8.4f}")
    print(f"\nvocalisation -> speech: {rates[1, 2]:.2%}   (gate <= 10%)")
    print(f"speech -> vocalisation: {rates[2, 1]:.2%}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
