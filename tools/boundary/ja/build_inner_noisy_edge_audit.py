#!/usr/bin/env python3
"""Build a discriminative Inner-edge audit from known semantic cores with edge noise."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.outer_refiner_v2 import load_outer_edge_refiner_v2  # noqa: E402
from tools.boundary.ja.build_inner_subisland_edge_audit import (  # noqa: E402
    ITEM_SCHEMA,
    build_page,
    edge_features,
)


SUMMARY_SCHEMA = "inner_noisy_edge_audit_summary_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    timeline = _rows(Path(args.timeline_labels))
    if len(timeline) != 5:
        raise ValueError(f"Inner noisy-edge audit requires exactly 5 rows; got {len(timeline)}")
    features = {
        str(row.get("sample_id") or row["audio_id"]): row
        for row in _rows(Path(args.feature_manifest))
    }
    model = load_outer_edge_refiner_v2(
        args.checkpoint,
        device=args.device,
        expected_ptm_repo_id=args.ptm_repo_id,
    )
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for row in timeline:
        sample_id = str(row["sample_id"])
        feature_row = features[sample_id]
        with np.load(Path(str(feature_row["feature_path"])), allow_pickle=False) as payload:
            ptm_key = "ptm2048" if "ptm2048" in payload.files else "ptm"
            ptm = np.asarray(payload[ptm_key], dtype=np.float32)
            mfcc = np.asarray(payload["mfcc"], dtype=np.float32)
        frame_hop_s = float(feature_row.get("frame_hop_s") or 0.02)
        total = min(int(feature_row["frame_count"]), ptm.shape[0], mfcc.shape[0])
        prediction = model.predict_islands(
            frame_feature_groups=[edge_features(ptm[:total], mfcc[:total])],
            raw_spans=[(0.0, float(row["duration_s"]))],
            frame_hop_s=frame_hop_s,
        )[0]
        source_audio = Path(str(row["audio"]))
        target_audio = audio_dir / f"{sample_id}{source_audio.suffix.lower() or '.wav'}"
        shutil.copyfile(source_audio, target_audio)
        items.append(
            {
                "schema": ITEM_SCHEMA,
                "audit_mode": "known_core_empirical_noisy_edges_v1",
                "sample_id": sample_id,
                "subisland_id": f"{sample_id}__inner",
                "audio": target_audio.relative_to(output_dir).as_posix(),
                "source_duration_s": float(row["duration_s"]),
                "raw_start_s": 0.0,
                "raw_end_s": float(row["duration_s"]),
                "refined_start_s": float(prediction.start_s),
                "refined_end_s": float(prediction.end_s),
                "start_requires_inner": True,
                "end_requires_inner": True,
                "reference_text": str(row.get("reference_text") or ""),
                "known_semantic_core_span": row.get("semantic_core_span"),
                "edge_noise": row.get("edge_noise"),
                "bootstrap_prediction": {
                    **asdict(prediction),
                    "class_probabilities": None,
                },
                "bootstrap_checkpoint_sha256": model.sha256,
                "teacher_usage": "bootstrap_preview_only_not_training_truth",
            }
        )
    items_path = output_dir / "inner_items.jsonl"
    _write(items_path, items)
    page = build_page(
        rows=items,
        output_dir=output_dir,
        update_latest=not args.no_update_latest,
        noisy_edge_mode=True,
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "sample_count": len(items),
        "owned_start_count": len(items),
        "owned_end_count": len(items),
        "known_edge_pollution_count": len(items) * 2,
        "bootstrap_checkpoint_sha256": model.sha256,
        "bootstrap_usage": "preview_only_not_training_truth",
        "items": str(items_path),
        "page": str(page),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Inner noisy-edge fixed-5 audit.")
    parser.add_argument("--timeline-labels", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ptm-repo-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-update-latest", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False))
