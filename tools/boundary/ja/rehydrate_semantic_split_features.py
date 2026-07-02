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

from boundary.ja.model import (  # noqa: E402
    load_speech_island_scorer_checkpoint,
    score_speech_island_probabilities,
)
from boundary.sequence_features import (  # noqa: E402
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
)


def _score_windowed(
    scorer,
    *,
    ptm: np.ndarray,
    mfcc: np.ndarray,
    window_frames: int = 1500,
    overlap_frames: int = 250,
) -> np.ndarray:
    total = min(ptm.shape[0], mfcc.shape[0])
    stride = window_frames - overlap_frames
    score_sum = np.zeros(total, dtype=np.float64)
    score_count = np.zeros(total, dtype=np.float64)
    for start in range(0, total, stride):
        end = min(total, start + window_frames)
        scores = score_speech_island_probabilities(
            scorer,
            ptm=ptm[start:end],
            mfcc=mfcc[start:end],
        )
        score_sum[start:end] += scores
        score_count[start:end] += 1.0
    return np.asarray(score_sum / np.maximum(score_count, 1.0), dtype=np.float32)


def run(args: argparse.Namespace) -> None:
    source = np.load(args.sequence_features)
    ptm = np.asarray(source["ptm"], dtype=np.float32)
    mfcc = np.asarray(source["mfcc"], dtype=np.float32)
    frame_hop_s = float(source["frame_hop_s"][0])
    scorer = load_speech_island_scorer_checkpoint(
        args.scorer_checkpoint,
        device=args.device,
    )
    actual_repo_id = str(scorer.metadata.get("ptm_repo_id") or "")
    if actual_repo_id != args.ptm_repo_id:
        raise ValueError(
            f"scorer ptm_repo_id={actual_repo_id!r} does not match {args.ptm_repo_id!r}"
        )
    speech = _score_windowed(scorer, ptm=ptm, mfcc=mfcc)
    provider = FrameSequenceFeatureProvider(
        duration_s=min(ptm.shape[0], mfcc.shape[0]) * frame_hop_s,
        frame_hop_s=frame_hop_s,
        ptm=ptm,
        mfcc=mfcc,
        config=FrameSequenceFeatureConfig(max_ptm_dims=args.ptm_dim),
    )
    rows = [
        json.loads(line)
        for line in Path(args.candidate_metadata).open("r", encoding="utf-8")
        if line.strip()
    ]
    frame_rows: list[np.ndarray] = []
    scalar_rows: list[np.ndarray] = []
    for row in rows:
        frames, scalars = provider.features_for_split_candidate(
            core_start_s=float(row["core_start"]),
            core_end_s=float(row["core_end"]),
            candidate=row,
            speech_probabilities=speech,
            left_context_s=1.6,
            right_context_s=1.6,
            gap_context_s=0.3,
            left_bins=8,
            gap_bins=4,
            right_bins=8,
            ptm_dim=args.ptm_dim,
        )
        frame_rows.append(frames)
        scalar_rows.append(scalars)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frame_features=np.stack(frame_rows),
        scalar_features=np.stack(scalar_rows),
        proposal_times_s=np.asarray([row["time_s"] for row in rows], dtype=np.float32),
        core_starts_s=np.asarray([row["core_start"] for row in rows], dtype=np.float32),
        core_ends_s=np.asarray([row["core_end"] for row in rows], dtype=np.float32),
    )
    summary = {
        "schema": "rehydrated_semantic_split_features_v1",
        "ptm_repo_id": args.ptm_repo_id,
        "candidate_count": len(rows),
        "frame_shape": list(np.stack(frame_rows).shape),
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild labeled semantic-split candidate features for another PTM."
    )
    parser.add_argument("--sequence-features", required=True)
    parser.add_argument("--candidate-metadata", required=True)
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--ptm-repo-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ptm-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
