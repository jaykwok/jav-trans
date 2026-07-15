#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from boundary.inner_refiner_v1 import (  # noqa: E402
    INNER_EDGE_REFINER_V1_FEATURE_SCHEMA,
)
from boundary.ja.dataset import effective_frame_weights, read_jsonl  # noqa: E402
from boundary.ja.features import load_cached_feature  # noqa: E402
from boundary.ja.model import SPEECH_ISLAND_SCORER_LABELS  # noqa: E402


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(args: argparse.Namespace) -> None:
    records = read_jsonl(Path(args.labels))
    rows = _rows(Path(args.feature_manifest))
    if not rows or str(rows[0].get("ptm") or "") != QWEN_ASR_17B_REPO_ID:
        raise ValueError("Inner Edge Refiner v1 dataset is 1.7B-only")
    output_dir = Path(args.output_dir)
    feature_dir = output_dir / "subislands"
    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    target_index = SPEECH_ISLAND_SCORER_LABELS.index("semantic_target")
    discardable_index = SPEECH_ISLAND_SCORER_LABELS.index("discardable")
    for row_index, row in enumerate(rows):
        record = records[int(row["label_index"])]
        ptm, mfcc = load_cached_feature(Path(str(row["feature_path"])))
        total = min(int(ptm.shape[0]), int(mfcc.shape[0]), len(record.speech_frames))
        if total <= 0:
            continue
        if int(ptm.shape[1]) != 2048:
            raise ValueError("Inner Edge Refiner v1 requires full PTM2048 features")
        truth = np.where(
            np.asarray(record.speech_frames[:total], dtype=np.int64) > 0,
            target_index,
            discardable_index,
        ).astype(np.int64)
        weights = np.asarray(effective_frame_weights(record)[:total], dtype=np.float32)
        feature_path = feature_dir / f"row{row_index:06d}_subisland000.npz"
        np.savez(feature_path, labels=truth, weights=weights)
        metadata = dict(record.boundary_metadata or {})
        manifest_rows.append(
            {
                "schema": INNER_EDGE_REFINER_V1_FEATURE_SCHEMA,
                "audio_id": record.audio_id,
                "feature_path": str(feature_path),
                "source_feature_path": str(row["feature_path"]),
                "start_frame": 0,
                "end_frame": total,
                "frame_hop_s": float(record.frame_hop_s),
                "partition": str(metadata.get("source_partition") or "train"),
                "contains_target": bool(np.any(truth == target_index)),
                "pipeline_entry_stage": "inner_edge_refiner_v1",
                "entry_contract": "cueqc_keep_known_core_noisy_subisland_v1",
            }
        )
        if args.log_every and (row_index + 1) % args.log_every == 0:
            print(
                f"inner_v1_dataset={row_index + 1}/{len(rows)} "
                f"subislands={len(manifest_rows)}",
                flush=True,
            )
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    summary = {
        "schema": INNER_EDGE_REFINER_V1_FEATURE_SCHEMA,
        "source_count": len(rows),
        "subisland_count": len(manifest_rows),
        "target_subisland_count": sum(bool(row["contains_target"]) for row in manifest_rows),
        "labels": list(SPEECH_ISLAND_SCORER_LABELS),
        "decision_mode": "argmax",
        "ptm_projection": "checkpoint_learned_linear_2048_to_128",
        "pipeline_entry_stage": "inner_edge_refiner_v1",
        "entry_contract": "cueqc_keep_known_core_noisy_subisland_v1",
        "manifest": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 1.7B Inner Edge Refiner v1 data.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
