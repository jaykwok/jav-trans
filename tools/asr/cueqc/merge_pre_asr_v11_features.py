#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def run(args: argparse.Namespace) -> None:
    import torch

    source_bundles = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.features
    ]
    bundles: list[dict] = []
    roles: list[str] = []
    for role, bundle in zip(args.role, source_bundles):
        group_width = int(bundle["chunk_mask"].shape[1])
        window_size = int(args.max_chunks)
        if window_size <= 0 or group_width <= window_size:
            bundles.append(bundle)
            roles.append(role)
            continue
        for group_index, group in enumerate(bundle["groups"]):
            valid_count = int(bundle["chunk_mask"][group_index].sum().item())
            row_ids = list(group.get("row_ids") or [])
            for start in range(0, valid_count, window_size):
                end = min(valid_count, start + window_size)
                window_group = {
                    **group,
                    "group_index": 0,
                    "row_ids": row_ids[start:end],
                    "planned_island_id": (
                        f"{group.get('planned_island_id', 'sequence')}:"
                        f"window{start // window_size:04d}"
                    ),
                    "source_group_index": int(group.get("group_index", group_index)),
                    "source_chunk_start": start,
                    "source_chunk_end": end,
                }
                bundles.append(
                    {
                        **bundle,
                        "groups": [window_group],
                        "scalar_features": bundle["scalar_features"][
                            group_index : group_index + 1, start:end
                        ].clone(),
                        "ptm_bins": bundle["ptm_bins"][
                            group_index : group_index + 1, start:end
                        ].clone(),
                        "bin_mask": bundle["bin_mask"][
                            group_index : group_index + 1, start:end
                        ].clone(),
                        "chunk_mask": bundle["chunk_mask"][
                            group_index : group_index + 1, start:end
                        ].clone(),
                        "labels": bundle["labels"][
                            group_index : group_index + 1, start:end
                        ].clone(),
                    }
                )
                roles.append(role)
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
    for role, bundle in zip(roles, bundles):
        for group in bundle["groups"]:
            groups.append(
                {
                    **group,
                    "group_index": group_offset + int(group["group_index"]),
                    "dataset_role": (
                        str(group.get("dataset_role") or "train")
                        if role == "preserve"
                        else role
                    ),
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
        "rows": [row for bundle in source_bundles for row in bundle["rows"]],
        "groups": groups,
        "source_files": [
            source for bundle in source_bundles for source in bundle["source_files"]
        ],
        "label_files": [
            source for bundle in source_bundles for source in bundle["label_files"]
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
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help=(
            "Split source groups into bounded temporal windows before padding. "
            "Use the training sequence window size to avoid padding every short "
            "group to the longest full-video sequence."
        ),
    )
    args = parser.parse_args()
    if len(args.features) != len(args.role):
        parser.error("--features and --role counts must match")
    if args.max_chunks < 0:
        parser.error("--max-chunks must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
