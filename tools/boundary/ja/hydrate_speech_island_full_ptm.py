#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _source_map(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        audio_id, separator, path = value.partition("=")
        if not separator or not audio_id or not path:
            raise ValueError("--source-feature must use AUDIO_ID=PATH")
        result[audio_id] = Path(path)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    labels = _read_jsonl(Path(args.labels))
    rows = _read_jsonl(Path(args.manifest))
    sources = _source_map(args.source_feature)
    output_dir = Path(args.output_dir)
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    hydrated_paths: dict[str, str] = {}
    prefix_delta_max = 0.0
    hydrated_unique = 0

    for audio_id, source_path in sources.items():
        source_rows = [
            row
            for row in rows
            if int(row.get("ptm_dim") or 0) < args.raw_ptm_dim
            and str(
                (labels[int(row["label_index"])].get("boundary_metadata") or {}).get(
                    "source_audio_id"
                )
                or ""
            )
            == audio_id
        ]
        unique_rows = {
            str(row["feature_path"]): row for row in source_rows
        }
        with np.load(source_path) as source:
            source_ptm = np.asarray(source["ptm"], dtype=np.float32)
            frame_hop_s = float(np.asarray(source["frame_hop_s"]).reshape(-1)[0])
            if int(source_ptm.shape[1]) != args.raw_ptm_dim:
                raise ValueError(
                    f"source {audio_id} PTM dim {source_ptm.shape[1]} "
                    f"!= {args.raw_ptm_dim}"
                )
            for old_path_text, row in unique_rows.items():
                record = labels[int(row["label_index"])]
                metadata = dict(record.get("boundary_metadata") or {})
                start = int(round(float(metadata["source_start_s"]) / frame_hop_s))
                count = int(row["frame_count"])
                projected = np.ascontiguousarray(
                    source_ptm[start : start + count], dtype=np.float32
                )
                old_ptm, old_mfcc = _load_feature(Path(old_path_text))
                if projected.shape != (count, args.raw_ptm_dim):
                    raise ValueError(f"source window is incomplete: {old_path_text}")
                prefix_delta = float(
                    np.max(np.abs(projected[:, : old_ptm.shape[1]] - old_ptm), initial=0.0)
                )
                prefix_delta_max = max(prefix_delta_max, prefix_delta)
                new_path = feature_dir / Path(old_path_text).name
                np.savez(
                    new_path,
                    ptm=projected,
                    mfcc=np.ascontiguousarray(old_mfcc, dtype=np.float32),
                )
                hydrated_paths[old_path_text] = str(new_path)
                hydrated_unique += 1
        del source_ptm

    updated: list[dict[str, Any]] = []
    for row in rows:
        old_path = str(row["feature_path"])
        if old_path in hydrated_paths:
            updated.append(
                {
                    **row,
                    "feature_path": hydrated_paths[old_path],
                    "ptm_dim": args.raw_ptm_dim,
                }
            )
        else:
            if int(row.get("ptm_dim") or 0) != args.raw_ptm_dim:
                raise ValueError(f"compact PTM row was not hydrated: {old_path}")
            updated.append(row)

    manifest_path = output_dir / "feature_manifest.jsonl"
    _write_jsonl(manifest_path, updated)
    summary = {
        "schema": "speech_island_full_ptm_hydration_v1",
        "source_manifest": args.manifest,
        "source_labels": args.labels,
        "raw_ptm_dim": args.raw_ptm_dim,
        "manifest_rows": len(updated),
        "hydrated_unique_features": hydrated_unique,
        "hydrated_manifest_rows": sum(
            1 for row in rows if str(row["feature_path"]) in hydrated_paths
        ),
        "prefix_delta_max": prefix_delta_max,
        "feature_manifest": str(manifest_path),
        "source_features": {key: str(value) for key, value in sources.items()},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def _load_feature(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as source:
        return (
            np.asarray(source["ptm"], dtype=np.float32),
            np.asarray(source["mfcc"], dtype=np.float32),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hydrate compact real SpeechIsland windows from full PTM sequences."
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-feature", action="append", default=[], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-ptm-dim", type=int, default=2048)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
