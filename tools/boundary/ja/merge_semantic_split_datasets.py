#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def run(args: argparse.Namespace) -> None:
    bundles = [np.load(path) for path in args.dataset]
    frame_shape = tuple(bundles[0]["frame_features"].shape[1:])
    scalar_shape = tuple(bundles[0]["scalar_features"].shape[1:])
    for path, bundle in zip(args.dataset, bundles):
        if tuple(bundle["frame_features"].shape[1:]) != frame_shape:
            raise ValueError(f"frame feature shape mismatch: {path}")
        if tuple(bundle["scalar_features"].shape[1:]) != scalar_shape:
            raise ValueError(f"scalar feature shape mismatch: {path}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    repeats = [max(1, int(value)) for value in args.repeat]
    roles = np.concatenate(
        [
            np.full(int(bundle["labels"].shape[0]) * repeat, role)
            for role, bundle, repeat in zip(args.role, bundles, repeats)
        ]
    )
    def repeated(bundle: np.lib.npyio.NpzFile, key: str, repeat: int) -> np.ndarray:
        values = bundle[key]
        return (
            values
            if repeat == 1
            else np.concatenate([values] * repeat, axis=0)
        )

    np.savez_compressed(
        output,
        frame_features=np.concatenate(
            [
                repeated(bundle, "frame_features", repeat).astype(np.float32)
                for bundle, repeat in zip(bundles, repeats)
            ]
        ),
        scalar_features=np.concatenate(
            [
                repeated(bundle, "scalar_features", repeat).astype(np.float32)
                for bundle, repeat in zip(bundles, repeats)
            ]
        ),
        labels=np.concatenate(
            [
                repeated(bundle, "labels", repeat).astype(np.int64)
                for bundle, repeat in zip(bundles, repeats)
            ]
        ),
        partitions=np.concatenate(
            [
                repeated(bundle, "partitions", repeat).astype(str)
                for bundle, repeat in zip(bundles, repeats)
            ]
        ),
        dataset_roles=roles,
    )
    summary = {
        "schema": "semantic_split_merged_dataset_v1",
        "output": str(output),
        "count": int(
            sum(
                bundle["labels"].shape[0] * repeat
                for bundle, repeat in zip(bundles, repeats)
            )
        ),
        "sources": [
            {
                "path": str(path),
                "role": role,
                "count": int(bundle["labels"].shape[0]),
                "repeat": repeat,
                "effective_count": int(bundle["labels"].shape[0] * repeat),
            }
            for path, role, bundle, repeat in zip(
                args.dataset,
                args.role,
                bundles,
                repeats,
            )
        ],
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--role", action="append", required=True)
    parser.add_argument("--repeat", action="append", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.dataset) != len(args.role):
        parser.error("--dataset and --role counts must match")
    if args.repeat is None:
        args.repeat = [1] * len(args.dataset)
    if len(args.repeat) != len(args.dataset):
        parser.error("--repeat and --dataset counts must match")
    if any(value <= 0 for value in args.repeat):
        parser.error("--repeat values must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
