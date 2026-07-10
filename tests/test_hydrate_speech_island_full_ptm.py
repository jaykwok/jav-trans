from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.boundary.ja.hydrate_speech_island_full_ptm import run


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_hydrate_compact_real_windows_preserves_prefix_and_repeated_rows(
    tmp_path: Path,
) -> None:
    full_ptm = np.arange(12 * 4, dtype=np.float32).reshape(12, 4)
    source_path = tmp_path / "source.npz"
    np.savez(
        source_path,
        ptm=full_ptm,
        mfcc=np.zeros((12, 2), dtype=np.float32),
        frame_hop_s=np.asarray([0.5], dtype=np.float32),
    )
    compact_path = tmp_path / "compact.npz"
    np.savez(
        compact_path,
        ptm=full_ptm[2:5, :2],
        mfcc=np.ones((3, 2), dtype=np.float32),
    )
    labels_path = tmp_path / "labels.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_jsonl(
        labels_path,
        [
            {
                "boundary_metadata": {
                    "source_audio_id": "movie",
                    "source_start_s": 1.0,
                    "source_end_s": 2.5,
                }
            }
        ],
    )
    compact_row = {
        "feature_path": str(compact_path),
        "frame_count": 3,
        "label_index": 0,
        "ptm_dim": 2,
        "mfcc_dim": 2,
    }
    _write_jsonl(manifest_path, [compact_row, compact_row])
    output_dir = tmp_path / "hydrated"
    summary = run(
        argparse.Namespace(
            labels=str(labels_path),
            manifest=str(manifest_path),
            source_feature=[f"movie={source_path}"],
            output_dir=str(output_dir),
            raw_ptm_dim=4,
        )
    )
    rows = [
        json.loads(line)
        for line in (output_dir / "feature_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    with np.load(rows[0]["feature_path"]) as hydrated:
        np.testing.assert_array_equal(hydrated["ptm"], full_ptm[2:5])
        np.testing.assert_array_equal(hydrated["mfcc"], np.ones((3, 2)))
    assert rows[0]["feature_path"] == rows[1]["feature_path"]
    assert all(row["ptm_dim"] == 4 for row in rows)
    assert summary["hydrated_unique_features"] == 1
    assert summary["hydrated_manifest_rows"] == 2
    assert summary["prefix_delta_max"] == 0.0
