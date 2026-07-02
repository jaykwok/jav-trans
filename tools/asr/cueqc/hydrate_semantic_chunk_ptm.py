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

from asr.pre_asr_cueqc import (  # noqa: E402
    PRE_ASR_CUEQC_FEATURE_NAMES,
    PRE_ASR_CUEQC_FEATURE_SCHEMA,
    PRE_ASR_CUEQC_PTM_BINS,
    PRE_ASR_CUEQC_PTM_DIM,
    PRE_ASR_CUEQC_PTM_BIN_FEATURE_NAMES,
)
from boundary.sequence_features import (  # noqa: E402
    CHUNK_POOLED_PTM_SCHEMA,
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
)


def run(args: argparse.Namespace) -> None:
    bundle = np.load(args.sequence_features)
    provider = FrameSequenceFeatureProvider(
        duration_s=bundle["ptm"].shape[0] * float(bundle["frame_hop_s"][0]),
        frame_hop_s=float(bundle["frame_hop_s"][0]),
        ptm=bundle["ptm"],
        mfcc=bundle["mfcc"],
        config=FrameSequenceFeatureConfig(max_ptm_dims=PRE_ASR_CUEQC_PTM_DIM),
    )
    rows = [
        json.loads(line)
        for line in Path(args.candidates).open("r", encoding="utf-8")
        if line.strip()
    ]
    if args.audio_id:
        rows = [
            row
            for row in rows
            if str(row.get("audio_id") or row.get("video_id") or "")
            == args.audio_id
        ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            pooled = provider.chunk_pooled_ptm_features(
                start_s=float(row["start"]),
                end_s=float(row["end"]),
                bins=PRE_ASR_CUEQC_PTM_BINS,
            )
            row.update(
                {
                    "schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
                    "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
                    "ptm_pooling_schema": CHUNK_POOLED_PTM_SCHEMA,
                    "ptm_pooling_available": True,
                    "ptm_pooling_bins": PRE_ASR_CUEQC_PTM_BINS,
                    "ptm_pooling_dim": len(pooled),
                    "pre_asr_ptm_pooled_features": pooled,
                    "ptm_bin_feature_names": list(
                        PRE_ASR_CUEQC_PTM_BIN_FEATURE_NAMES
                    ),
                    "feature_names": list(PRE_ASR_CUEQC_FEATURE_NAMES),
                }
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "schema": "pre_asr_semantic_chunk_ptm_hydration_v1",
                "count": len(rows),
                "ptm_dim": PRE_ASR_CUEQC_PTM_DIM,
                "ptm_bins": PRE_ASR_CUEQC_PTM_BINS,
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--sequence-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio-id", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
