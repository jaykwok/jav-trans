#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LABEL_IDS = {"cut": 0, "continue": 1, "unsure": 2}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(args: argparse.Namespace) -> None:
    base = np.load(args.base_dataset)
    runtime = np.load(args.runtime_features)
    labels = [
        row
        for row in _read_jsonl(Path(args.omni_labels))
        if row["label"] in LABEL_IDS
    ]
    train_indexes = np.asarray(
        [int(row["index"]) for position, row in enumerate(labels) if position % 5],
        dtype=np.int64,
    )
    val_indexes = np.asarray(
        [int(row["index"]) for position, row in enumerate(labels) if position % 5 == 0],
        dtype=np.int64,
    )
    label_by_index = {
        int(row["index"]): LABEL_IDS[str(row["label"])]
        for row in labels
    }
    repeated_train = np.tile(train_indexes, args.real_repeat)
    real_indexes = np.concatenate((repeated_train, val_indexes))
    real_partitions = np.concatenate(
        (
            np.full(repeated_train.size, "train"),
            np.full(val_indexes.size, "val"),
        )
    )
    real_labels = np.asarray(
        [label_by_index[int(index)] for index in real_indexes],
        dtype=np.int64,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frame_features=np.concatenate(
            (
                base["frame_features"].astype(np.float32),
                runtime["frame_features"][real_indexes].astype(np.float32),
            )
        ),
        scalar_features=np.concatenate(
            (
                base["scalar_features"].astype(np.float32),
                runtime["scalar_features"][real_indexes].astype(np.float32),
            )
        ),
        labels=np.concatenate((base["labels"].astype(np.int64), real_labels)),
        partitions=np.concatenate(
            (base["partitions"].astype(str), real_partitions)
        ),
    )
    summary = {
        "schema": "semantic_split_omni_augmented_dataset_v1",
        "base_count": int(base["labels"].size),
        "real_unique_count": len(labels),
        "real_train_unique_count": int(train_indexes.size),
        "real_val_count": int(val_indexes.size),
        "real_repeat": args.real_repeat,
        "output_count": int(base["labels"].size + real_indexes.size),
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", required=True)
    parser.add_argument("--runtime-features", required=True)
    parser.add_argument("--omni-labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--real-repeat", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
