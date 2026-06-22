#!/usr/bin/env python3
"""Merge sharded CueQC v4 binary feature bundles."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _compatible_config(base: dict[str, Any], other: dict[str, Any]) -> bool:
    keys = (
        "asr_dim",
        "token_dim",
        "decoder_dim",
        "structured_dim",
        "feature_source",
        "uses_bge",
        "text_embedding",
        "token_feature_names",
        "decoder_feature_names",
        "structured_feature_names",
    )
    return all(base.get(key) == other.get(key) for key in keys)


def merge(inputs: list[Path], output: Path) -> dict[str, Any]:
    if not inputs:
        raise ValueError("no input feature bundles")
    merged_samples: list[dict[str, Any]] = []
    merged_labels: list[int] = []
    merged_meta: list[dict[str, Any]] = []
    base_config: dict[str, Any] | None = None
    source_shards: list[dict[str, Any]] = []

    for path in inputs:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        if bundle.get("schema") != "cueqc_mamba_v4_binary_features":
            raise ValueError(f"unexpected schema in {path}: {bundle.get('schema')!r}")
        config = dict(bundle.get("feature_config") or {})
        if base_config is None:
            base_config = dict(config)
        elif not _compatible_config(base_config, config):
            raise ValueError(f"incompatible feature_config in {path}")
        samples = list(bundle.get("samples") or [])
        labels = bundle.get("labels")
        label_values = labels.tolist() if labels is not None else []
        meta = list(bundle.get("meta") or [])
        if len(samples) != len(label_values) or len(samples) != len(meta):
            raise ValueError(f"sample/label/meta length mismatch in {path}")
        merged_samples.extend(samples)
        merged_labels.extend(int(value) for value in label_values)
        merged_meta.extend(meta)
        source_shards.append({
            "path": str(path),
            "samples": len(samples),
            "start_index": config.get("start_index"),
            "processed_rows": config.get("processed_rows"),
        })

    assert base_config is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_config = dict(base_config)
    feature_config["source_mode"] = "merged"
    feature_config["source_shards"] = source_shards
    payload = {
        "schema": "cueqc_mamba_v4_binary_features",
        "version": 3,
        "samples": merged_samples,
        "labels": torch.tensor(merged_labels, dtype=torch.long),
        "meta": merged_meta,
        "feature_config": feature_config,
        "label_config": {"drop": 0, "keep": 1, "unlabeled": -1},
    }
    torch.save(payload, output)
    return {
        "output": str(output),
        "shards": len(inputs),
        "samples": len(merged_samples),
        "labels_keep": merged_labels.count(1),
        "labels_drop": merged_labels.count(0),
        "labels_unlabeled": merged_labels.count(-1),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge CueQC v4 binary feature shards.")
    parser.add_argument("--input", action="append", required=True, help="Input .pt shard. Repeatable.")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = merge([Path(path) for path in args.input], Path(args.output))
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
