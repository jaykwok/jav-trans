#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def run(args: argparse.Namespace) -> None:
    import torch

    bundles = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.features
    ]
    max_chunks = max(int(bundle["chunk_mask"].shape[1]) for bundle in bundles)

    def padded(tensor, value: float):
        if int(tensor.shape[1]) == max_chunks:
            return tensor
        shape = list(tensor.shape)
        shape[1] = max_chunks - int(tensor.shape[1])
        tail = torch.full(shape, value, dtype=tensor.dtype)
        return torch.cat((tensor, tail), dim=1)

    groups = []
    group_offset = 0
    for role, bundle in zip(args.role, bundles):
        for group in bundle["groups"]:
            groups.append(
                {
                    **group,
                    "group_index": group_offset + int(group["group_index"]),
                    "dataset_role": role,
                }
            )
        group_offset += len(bundle["groups"])
    merged = {
        **{
            key: bundles[0][key]
            for key in (
                "schema",
                "feature_schema",
                "runtime_adapter",
                "feature_names",
                "all_feature_names",
                "ptm_bin_count",
                "ptm_dim",
                "asr_repo_id",
            )
        },
        "rows": [row for bundle in bundles for row in bundle["rows"]],
        "groups": groups,
        "source_files": [
            source for bundle in bundles for source in bundle["source_files"]
        ],
        "label_files": [
            source for bundle in bundles for source in bundle["label_files"]
        ],
        "scalar_features": torch.cat(
            [padded(bundle["scalar_features"], 0.0) for bundle in bundles]
        ),
        "ptm_bins": torch.cat(
            [padded(bundle["ptm_bins"], 0.0) for bundle in bundles]
        ),
        "bin_mask": torch.cat(
            [padded(bundle["bin_mask"], 0.0) for bundle in bundles]
        ),
        "chunk_mask": torch.cat(
            [padded(bundle["chunk_mask"], 0.0) for bundle in bundles]
        ),
        "labels": torch.cat(
            [padded(bundle["labels"], -100) for bundle in bundles]
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    print(
        f"features={output} groups={len(groups)} "
        f"chunks={int((merged['labels'] != -100).sum())}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", action="append", required=True)
    parser.add_argument("--role", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.features) != len(args.role):
        parser.error("--features and --role counts must match")
    return args


if __name__ == "__main__":
    run(parse_args())
