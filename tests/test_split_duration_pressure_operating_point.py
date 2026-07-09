from __future__ import annotations

import json
from pathlib import Path

from tools.datasets.analyze_split_duration_pressure_operating_point import (
    analyze_duration_pressure,
    export_duration_pressure,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _make_reexport(tmp_path: Path) -> Path:
    reexport = tmp_path / "reexport"
    feature_dir = reexport / "features" / "vid-w00"
    _write_jsonl(
        feature_dir / "pre_asr_candidates.jsonl",
        [
            {
                "candidate_id": "preasr-vid-w00-chunk00000",
                "audio_id": "vid-w00",
                "chunk_index": 0,
                "start": 0.0,
                "end": 20.0,
                "duration_s": 20.0,
            },
            {
                "candidate_id": "preasr-vid-w00-chunk00001",
                "audio_id": "vid-w00",
                "chunk_index": 1,
                "start": 20.0,
                "end": 24.0,
                "duration_s": 4.0,
            },
            {
                "candidate_id": "preasr-vid-w00-chunk00002",
                "audio_id": "vid-w00",
                "chunk_index": 2,
                "start": 24.0,
                "end": 42.0,
                "duration_s": 18.0,
            },
        ],
    )
    _write_jsonl(
        feature_dir / "semantic_split_features.jsonl",
        [
            {"time_s": 5.0, "core_start": 0.0, "core_end": 20.0, "label": "continue", "p_cut": 0.05, "accepted": False},
            {"time_s": 10.0, "core_start": 0.0, "core_end": 20.0, "label": "cut", "p_cut": 0.62, "accepted": False},
            {"time_s": 22.0, "core_start": 20.0, "core_end": 24.0, "label": "cut", "p_cut": 0.80, "accepted": False},
            {"time_s": 32.0, "core_start": 24.0, "core_end": 42.0, "label": "cut", "p_cut": 0.49, "accepted": False},
        ],
    )
    return reexport


def test_duration_pressure_only_splits_overlong_chunks(tmp_path: Path) -> None:
    summary = analyze_duration_pressure(
        reexport_dir=_make_reexport(tmp_path),
        floors=[0.50],
        long_chunk_min_s=15.0,
        min_chunk_after_split_s=1.2,
        pure_adaptive_policy={
            "abs_floor": 0.50,
            "percentile_floor": 0.80,
            "z_floor": 99.0,
        },
    )

    assert summary["long_chunk_count"] == 2
    assert summary["pure_adaptive_new_cut_placement"]["inside_short_chunks"] == 1
    assert summary["duration_pressure_variant"]["0.50"]["new_cuts"] == 1
    assert summary["duration_pressure_variant"]["0.50"]["chunks_hit"] == 1
    assert summary["ceiling"]["max_p_below_0.50"]["count"] == 1
    assert summary["unfixable_at_floor_0.50"][0]["candidate_id"] == "preasr-vid-w00-chunk00002"


def test_export_duration_pressure_writes_summary(tmp_path: Path) -> None:
    summary = export_duration_pressure(
        reexport_dir=_make_reexport(tmp_path),
        output_dir=tmp_path / "out",
        floors=[0.50],
        long_chunk_min_s=15.0,
        min_chunk_after_split_s=1.2,
        pure_adaptive_policy={
            "abs_floor": 0.50,
            "percentile_floor": 0.80,
            "z_floor": 99.0,
        },
    )

    stored = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert stored["schema"] == "split_duration_pressure_operating_point_summary_v1"
    assert stored["duration_pressure_variant"]["0.50"]["new_cuts"] == 1
    assert summary["output_dir"].endswith("out")
