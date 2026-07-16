#!/usr/bin/env python3
"""Build Inner v1 targets from the promoted learned Outer v2 edge teacher."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.inner_refiner_v1 import (  # noqa: E402
    INNER_EDGE_REFINER_V1_FEATURE_SCHEMA,
)
from boundary.ja.model import SPEECH_ISLAND_SCORER_LABELS  # noqa: E402
from boundary.outer_refiner_v2 import load_outer_edge_refiner_v2  # noqa: E402
from tools.boundary.ja.build_inner_subisland_edge_audit import (  # noqa: E402
    edge_features,
)


SUMMARY_SCHEMA = "inner_edge_refiner_v1_outer_distilled_summary_v1"
TEACHER_CONTRACT = "promoted_outer_v2_argmax_paired_edges_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def labels_from_prediction(
    prediction: SimpleNamespace | Any,
    *,
    total_frames: int,
    frame_hop_s: float,
) -> tuple[np.ndarray, str]:
    discardable = SPEECH_ISLAND_SCORER_LABELS.index("discardable")
    target = SPEECH_ISLAND_SCORER_LABELS.index("semantic_target")
    unsure = SPEECH_ISLAND_SCORER_LABELS.index("unsure")
    if (
        str(prediction.start_action) == "abstain"
        or str(prediction.end_action) == "abstain"
    ):
        return np.full(total_frames, unsure, dtype=np.int64), "teacher_abstain"
    start = int(round(float(prediction.start_s) / frame_hop_s))
    end = int(round(float(prediction.end_s) / frame_hop_s))
    start = min(total_frames, max(0, start))
    end = min(total_frames, max(0, end))
    if end <= start:
        return np.full(total_frames, unsure, dtype=np.int64), "teacher_invalid_span"
    labels = np.full(total_frames, discardable, dtype=np.int64)
    labels[start:end] = target
    return labels, "teacher_refined"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.ptm_repo_id != QWEN_ASR_17B_REPO_ID:
        raise ValueError("Outer-distilled Inner v1 dataset is 1.7B-only")
    apply_vram_safety_cap(0.95)
    rows = _rows(Path(args.input_manifest))
    if not rows or any(
        row.get("schema") != INNER_EDGE_REFINER_V1_FEATURE_SCHEMA for row in rows
    ):
        raise ValueError("input Inner feature manifest is empty or incompatible")
    teacher = load_outer_edge_refiner_v2(
        args.outer_teacher_checkpoint,
        device=args.device,
        expected_ptm_repo_id=args.ptm_repo_id,
    )
    output_dir = Path(args.output_dir)
    labels_dir = output_dir / "subislands"
    labels_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        source_feature_path = Path(str(row["source_feature_path"]))
        with np.load(source_feature_path, allow_pickle=False) as payload:
            ptm = np.asarray(payload["ptm"], dtype=np.float32)
            mfcc = np.asarray(payload["mfcc"], dtype=np.float32)
        total = min(int(ptm.shape[0]), int(mfcc.shape[0]))
        if total <= 0 or int(ptm.shape[1]) != 2048:
            raise ValueError(f"invalid raw PTM2048 feature row: {row['audio_id']}")
        frame_hop_s = float(row.get("frame_hop_s") or args.frame_hop_s)
        prediction = teacher.predict_islands(
            frame_feature_groups=[edge_features(ptm[:total], mfcc[:total])],
            raw_spans=[(0.0, total * frame_hop_s)],
            frame_hop_s=frame_hop_s,
        )[0]
        labels, decision = labels_from_prediction(
            prediction,
            total_frames=total,
            frame_hop_s=frame_hop_s,
        )
        decision_counts[decision] += 1
        label_path = labels_dir / f"row{index:06d}_subisland000.npz"
        np.savez(
            label_path,
            labels=labels,
            weights=np.ones(total, dtype=np.float32),
        )
        output_rows.append(
            {
                **row,
                "schema": INNER_EDGE_REFINER_V1_FEATURE_SCHEMA,
                "feature_path": str(label_path),
                "contains_target": bool(
                    np.any(
                        labels
                        == SPEECH_ISLAND_SCORER_LABELS.index("semantic_target")
                    )
                ),
                "entry_contract": "outer_v2_distilled_retained_subisland_v1",
                "teacher_contract": TEACHER_CONTRACT,
                "teacher_checkpoint_sha256": teacher.sha256,
                "teacher_decision": decision,
                "teacher_prediction": {
                    "start_s": float(prediction.start_s),
                    "end_s": float(prediction.end_s),
                    "start_action": str(prediction.start_action),
                    "end_action": str(prediction.end_action),
                    "abstain_reason": str(prediction.abstain_reason),
                },
            }
        )
        if args.log_every and (index + 1) % args.log_every == 0:
            print(f"outer_distilled_inner={index + 1}/{len(rows)}", flush=True)
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "source_count": len(rows),
        "output_count": len(output_rows),
        "target_count": sum(bool(row["contains_target"]) for row in output_rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "feature_schema": INNER_EDGE_REFINER_V1_FEATURE_SCHEMA,
        "entry_contract": "outer_v2_distilled_retained_subisland_v1",
        "teacher_contract": TEACHER_CONTRACT,
        "teacher_checkpoint": str(Path(args.outer_teacher_checkpoint)),
        "teacher_checkpoint_sha256": teacher.sha256,
        "ptm_repo_id": args.ptm_repo_id,
        "ptm_projection": "student_checkpoint_learned_linear_2048_to_128",
        "decision_mode": "teacher_argmax_no_threshold",
        "rule_fallback": False,
        "manifest": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build learned Outer-v2-distilled Inner v1 targets."
    )
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--outer-teacher-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ptm-repo-id", default=QWEN_ASR_17B_REPO_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
