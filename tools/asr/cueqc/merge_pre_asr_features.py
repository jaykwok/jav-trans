#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_save_atomic(payload: object, path: Path) -> None:
    import torch

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_bundle(bundle: dict[str, Any], *, path: Path, torch) -> None:
    tensor_keys = (
        "scalar_features",
        "ptm_bins",
        "bin_mask",
        "chunk_mask",
        "labels",
    )
    missing = [key for key in tensor_keys if key not in bundle]
    if missing:
        raise ValueError(f"CueQC merge input {path} is missing tensors {missing}")
    scalar = bundle["scalar_features"]
    ptm = bundle["ptm_bins"]
    bin_mask = bundle["bin_mask"]
    chunk_mask = bundle["chunk_mask"]
    labels = bundle["labels"]
    if not all(torch.is_tensor(value) for value in (scalar, ptm, bin_mask, chunk_mask, labels)):
        raise ValueError(f"CueQC merge input {path} contains non-tensor features")
    if scalar.ndim != 3 or ptm.ndim != 4 or bin_mask.ndim != 3:
        raise ValueError(f"CueQC merge input {path} has invalid feature dimensions")
    if chunk_mask.ndim != 2 or labels.ndim != 2:
        raise ValueError(f"CueQC merge input {path} has invalid mask/label dimensions")
    group_count, chunk_count = labels.shape
    if group_count <= 0 or chunk_count <= 0:
        raise ValueError(f"CueQC merge input {path} is empty")
    if scalar.shape[:2] != labels.shape or ptm.shape[:2] != labels.shape:
        raise ValueError(f"CueQC merge input {path} feature rows do not match labels")
    if bin_mask.shape != ptm.shape[:3] or chunk_mask.shape != labels.shape:
        raise ValueError(f"CueQC merge input {path} masks do not match features")
    if int(ptm.shape[2]) != int(bundle["ptm_bin_count"]):
        raise ValueError(f"CueQC merge input {path} PTM bin count mismatch")
    if int(ptm.shape[3]) != int(bundle["ptm_dim"]):
        raise ValueError(f"CueQC merge input {path} PTM width mismatch")
    for key, value in (
        ("scalar_features", scalar),
        ("ptm_bins", ptm),
        ("bin_mask", bin_mask),
        ("chunk_mask", chunk_mask),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"CueQC merge input {path} has non-finite {key}")
    if not torch.all((bin_mask == 0) | (bin_mask == 1)):
        raise ValueError(f"CueQC merge input {path} bin_mask must be binary")
    if not torch.all((chunk_mask == 0) | (chunk_mask == 1)):
        raise ValueError(f"CueQC merge input {path} chunk_mask must be binary")
    if not torch.all((labels == -100) | (labels == 0) | (labels == 1)):
        raise ValueError(f"CueQC merge input {path} has a retired training label")
    if not torch.all(labels[chunk_mask == 0] == -100):
        raise ValueError(f"CueQC merge input {path} labels padded chunks as training data")
    groups = list(bundle.get("groups") or [])
    if len(groups) != int(group_count):
        raise ValueError(f"CueQC merge input {path} group metadata count mismatch")
    rows = bundle.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"CueQC merge input {path} row metadata is missing")
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"CueQC merge input {path} has invalid row metadata")
        row_id = str(row.get("id") or "").strip()
        if not row_id or row_id in rows_by_id:
            raise ValueError(
                f"CueQC merge input {path} has duplicate or missing row ids"
            )
        raw_core_ids = row.get("source_core_ids")
        if not isinstance(raw_core_ids, list):
            raise ValueError(
                f"CueQC merge input {path} row {row_id!r} lacks source_core_ids"
            )
        core_ids = [str(value).strip() for value in raw_core_ids]
        if any(not value for value in core_ids) or len(core_ids) != len(set(core_ids)):
            raise ValueError(
                f"CueQC merge input {path} row {row_id!r} has invalid source_core_ids"
            )
        rows_by_id[row_id] = {**dict(row), "source_core_ids": core_ids}
    grouped_row_ids: set[str] = set()
    for index, group in enumerate(groups):
        if int(group.get("group_index", -1)) != index:
            raise ValueError(f"CueQC merge input {path} has non-canonical group indexes")
        valid_count = int((chunk_mask[index] != 0).sum().item())
        row_ids = [str(value).strip() for value in (group.get("row_ids") or [])]
        if len(row_ids) != valid_count or len(row_ids) != len(set(row_ids)):
            raise ValueError(f"CueQC merge input {path} row_ids do not match chunk_mask")
        missing_rows = [row_id for row_id in row_ids if row_id not in rows_by_id]
        if missing_rows:
            raise ValueError(
                f"CueQC merge input {path} group references unknown rows: "
                f"{missing_rows[:3]}"
            )
        reused_rows = [row_id for row_id in row_ids if row_id in grouped_row_ids]
        if reused_rows:
            raise ValueError(
                f"CueQC merge input {path} reuses rows across groups: "
                f"{reused_rows[:3]}"
            )
        grouped_row_ids.update(row_ids)
        expected_core_ids = sorted(
            {
                core_id
                for row_id in row_ids
                for core_id in rows_by_id[row_id]["source_core_ids"]
            }
        )
        actual_core_ids = [
            str(value).strip() for value in (group.get("source_core_ids") or [])
        ]
        if (
            any(not value for value in actual_core_ids)
            or len(actual_core_ids) != len(set(actual_core_ids))
            or sorted(actual_core_ids) != expected_core_ids
        ):
            raise ValueError(
                f"CueQC merge input {path} group core_ids do not match row metadata"
            )
    if grouped_row_ids != set(rows_by_id):
        raise ValueError(f"CueQC merge input {path} has ungrouped row metadata")


def run(args: argparse.Namespace) -> None:
    import torch

    feature_paths = [Path(path) for path in args.features]
    bundles = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in feature_paths
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
    for path, bundle in zip(feature_paths, bundles, strict=True):
        _validate_bundle(bundle, path=path, torch=torch)
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
    _torch_save_atomic(merged, output)
    _write_json_atomic(
        output.with_suffix(".summary.json"),
        {
            "schema": "cueqc_v13_merged_feature_summary_v3",
            "training_manifest_allowed": True,
            "feature_bundle": str(output.resolve()),
            "feature_bundle_sha256": _sha256(output),
            "source_bundle_count": len(bundles),
            "source_bundles": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in feature_paths
            ],
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
            "semantic_split_weights_sha256": bundles[0][
                "semantic_split_weights_sha256"
            ],
            "inner_edge_refiner_weights_sha256": bundles[0][
                "inner_edge_refiner_weights_sha256"
            ],
        },
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
