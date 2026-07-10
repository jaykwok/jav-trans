from __future__ import annotations

import argparse
import json

import numpy as np

from tools.asr.cueqc.rehydrate_pre_asr_projected_ptm_dataset import run


def test_rehydrate_pre_asr_uses_projected_ptm_frames(tmp_path):
    sequence = tmp_path / "sequence.npz"
    np.savez_compressed(
        sequence,
        ptm=np.ones((4, 128), dtype=np.float32),
        mfcc=np.zeros((4, 40), dtype=np.float32),
        frame_hop_s=np.asarray([0.1], dtype=np.float32),
        ptm_projected=np.full((4, 128), 2.0, dtype=np.float32),
        ptm_projection_digest=np.asarray(["digest-a"]),
    )
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-a",
                "start": 0.0,
                "end": 0.4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_windows = tmp_path / "source_windows.jsonl"
    source_windows.write_text(
        json.dumps(
            {
                "window_id": "window-a",
                "speech_sequence_features": str(sequence),
                "pre_asr_candidates": str(candidates),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"

    summary = run(
        argparse.Namespace(
            source_windows=str(source_windows),
            output_dir=str(output),
        )
    )
    hydrated_path = output / "features" / "window-a" / "pre_asr_candidates.jsonl"
    hydrated = json.loads(hydrated_path.read_text(encoding="utf-8"))

    assert summary["ptm_projection_digest"] == "digest-a"
    assert hydrated["ptm_pooling_schema"] == "pre_asr_chunk_projected_ptm_v2"
    assert hydrated["ptm_projection_digest"] == "digest-a"
    assert hydrated["pre_asr_ptm_pooled_features"][:128] == [2.0] * 128
    assert hydrated["pre_asr_ptm_pooled_features"][128:256] == [0.0] * 128
