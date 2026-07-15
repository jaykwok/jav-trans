from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.boundary.ja.build_outer_v2_noisy_edge_fixed5 import (
    build,
    select_empirical_edge_assets,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _audio(path: Path, duration_s: float, value: float) -> None:
    sf.write(
        path,
        np.full(int(round(duration_s * 16000)), value, dtype=np.float32),
        16000,
    )


def test_select_empirical_edge_assets_is_unique_and_kind_disjoint(tmp_path: Path) -> None:
    rows = []
    for flag in (
        "moaning",
        "kissing",
        "breathing",
        "crying",
        "non_verbal_vocalization",
    ):
        for index in range(3):
            path = tmp_path / f"vocal-{flag}-{index}.wav"
            _audio(path, 0.1 + index * 0.01, 0.01)
            rows.append(
                {
                    "audio_id": f"vocal-{flag}-{index}",
                    "audio": str(path),
                    "source": "omni_definite_drop",
                    "source_partition": "train",
                    "duration_s": 0.1 + index * 0.01,
                    "background_type": flag,
                    "omni_flags": [flag],
                }
            )
    for kind, flag in (("noise", "noise"),):
        for index in range(12):
            path = tmp_path / f"{kind}-{index}.wav"
            _audio(path, 0.1 + index * 0.01, 0.01)
            rows.append(
                {
                    "audio_id": f"{kind}-{index}",
                    "audio": str(path),
                    "source": "omni_definite_drop",
                    "source_partition": "train",
                    "duration_s": 0.1 + index * 0.01,
                    "background_type": flag,
                    "omni_flags": [flag],
                }
            )

    selected = select_empirical_edge_assets(rows)

    assert len(selected["vocal"]) == 5
    assert len(selected["noise"]) == 5
    selected_ids = [row["audio_id"] for values in selected.values() for row in values]
    assert len(set(selected_ids)) == 10

    excluded = {selected["vocal"][0]["audio_id"], selected["noise"][0]["audio_id"]}
    reselection = select_empirical_edge_assets(rows, excluded_audio_ids=excluded)
    reselected_ids = {
        row["audio_id"] for values in reselection.values() for row in values
    }
    assert reselected_ids.isdisjoint(excluded)


def test_build_noisy_edge_fixed5_uses_each_core_and_negative_once(tmp_path: Path) -> None:
    semantic_rows = []
    for index in range(5):
        path = tmp_path / f"core-{index}.wav"
        _audio(path, 0.4 + index * 0.02, 0.1)
        semantic_rows.append(
            {
                "sample_id": f"core-{index}",
                "audio": str(path),
                "reference_text": f"台詞 {index}",
            }
        )
    semantic_labels = tmp_path / "semantic.jsonl"
    _write_jsonl(semantic_labels, semantic_rows)

    negative_rows = []
    for flag in (
        "moaning",
        "kissing",
        "breathing",
        "crying",
        "non_verbal_vocalization",
    ):
        for index in range(3):
            path = tmp_path / f"vocal-{flag}-{index}.wav"
            duration_s = 0.12 + index * 0.01
            _audio(path, duration_s, 0.02)
            negative_rows.append(
                {
                    "audio_id": f"vocal-{flag}-{index}",
                    "audio": str(path),
                    "source": "omni_definite_drop",
                    "source_partition": "train",
                    "duration_s": duration_s,
                    "background_type": flag,
                    "omni_flags": [flag],
                    "label_source": "manual",
                }
            )
    for kind, flag, value in (("noise", "noise", 0.03),):
        for index in range(12):
            path = tmp_path / f"{kind}-{index}.wav"
            duration_s = 0.12 + index * 0.01
            _audio(path, duration_s, value)
            negative_rows.append(
                {
                    "audio_id": f"{kind}-{index}",
                    "audio": str(path),
                    "source": "omni_definite_drop",
                    "source_partition": "train",
                    "duration_s": duration_s,
                    "background_type": flag,
                    "omni_flags": [flag],
                    "label_source": "manual",
                }
            )
    negative_manifest = tmp_path / "negatives.jsonl"
    _write_jsonl(negative_manifest, negative_rows)

    output = tmp_path / "output"
    summary = build(
        Namespace(
            semantic_timeline_labels=str(semantic_labels),
            negative_manifest=str(negative_manifest),
            output_dir=str(output),
            seed=15,
            crossfade_ms_min=5.0,
            crossfade_ms_max=30.0,
        )
    )

    assert summary["sample_count"] == 5
    assert summary["max_semantic_source_use_count"] == 1
    assert summary["edge_asset_count"] == 10
    assert summary["max_edge_asset_use_count"] == 1
    timeline = [json.loads(line) for line in (output / "timeline_labels.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(timeline) == 5
    assert timeline[0]["edge_noise"]["leading"]["kind"] == "vocal"
    assert timeline[0]["edge_noise"]["trailing"]["kind"] == "noise"
    assert timeline[1]["edge_noise"]["leading"]["kind"] == "noise"
    assert timeline[1]["edge_noise"]["trailing"]["kind"] == "vocal"
    assert all(row["semantic_core_span"]["start_s"] > 0.0 for row in timeline)
    assert all(row["semantic_core_span"]["end_s"] < row["duration_s"] for row in timeline)
