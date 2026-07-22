#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def run(args: argparse.Namespace) -> None:
    import torch

    bundles = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.features
    ]
    if not bundles:
        raise ValueError("no CueQC feature bundles supplied")
    identity_keys = (
        "schema",
        "feature_schema",
        "runtime_adapter",
        "feature_names",
        "all_feature_names",
        "ptm_bin_count",
        "ptm_dim",
        "asr_repo_id",
        "boundary_serialization_contract_id",
        "training_manifest_allowed",
        "semantic_split_weights_sha256",
        "inner_edge_refiner_weights_sha256",
    )
    for key in identity_keys:
        if any(key not in bundle for bundle in bundles):
            raise ValueError(f"CueQC merge input is missing required {key}")
        values = {json.dumps(bundle.get(key), sort_keys=True) for bundle in bundles}
        if len(values) != 1:
            raise ValueError(f"CueQC merge input mismatch for {key}")
    if bundles[0].get("training_manifest_allowed") is not True:
        raise ValueError("CueQC merge inputs are not approved training manifests")
    source_partitions: dict[str, set[str]] = defaultdict(set)
    seen_row_ids: set[str] = set()
    seen_core_ids: set[str] = set()
    for bundle in bundles:
        for group in bundle.get("groups") or []:
            source_id = str(group.get("source_id") or "").strip()
            partition = str(group.get("dataset_role") or "").strip()
            if not source_id or partition not in {"train", "val", "test"}:
                raise ValueError(
                    "CueQC merge requires frozen source_id and dataset_role"
                )
            source_partitions[source_id].add(partition)
            for row_id in group.get("row_ids") or []:
                value = str(row_id)
                if value in seen_row_ids:
                    raise ValueError(
                        f"CueQC provisional subisland is duplicated across inputs: {value}"
                    )
                seen_row_ids.add(value)
            for core_id in group.get("source_core_ids") or []:
                value = str(core_id).strip()
                if not value:
                    raise ValueError("CueQC merge found an empty source core identity")
                if value in seen_core_ids:
                    raise ValueError(
                        f"CueQC semantic core is duplicated across inputs: {value}"
                    )
                seen_core_ids.add(value)
    if any(len(values) != 1 for values in source_partitions.values()):
        raise ValueError("CueQC source identity crosses merged partitions")
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
    for bundle in bundles:
        for group in bundle["groups"]:
            groups.append(
                {
                    **group,
                    "group_index": group_offset + int(group["group_index"]),
                }
            )
        group_offset += len(bundle["groups"])
    merged = {
        **{
            key: bundles[0][key]
            for key in identity_keys
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
    output.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "schema": "cueqc_v13_merged_feature_summary_v2",
                "feature_bundle": str(output),
                "source_bundle_count": len(bundles),
                "group_count": len(groups),
                "source_count": len(source_partitions),
                "partition_counts": {
                    partition: sum(
                        str(group["dataset_role"]) == partition for group in groups
                    )
                    for partition in ("train", "val", "test")
                },
                "context_preserved": True,
                "partition_reassignment": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"features={output} groups={len(groups)} "
        f"chunks={int((merged['labels'] != -100).sum())}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
