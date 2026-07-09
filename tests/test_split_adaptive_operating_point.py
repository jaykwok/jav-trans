from __future__ import annotations

import json
from pathlib import Path

from tools.datasets.analyze_split_adaptive_operating_point import (
    analyze_adaptive_policies,
    export_adaptive_analysis,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_adaptive_policy_uses_local_percentile_with_absolute_floor(tmp_path: Path) -> None:
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
            }
        ],
    )
    _write_jsonl(
        feature_dir / "semantic_split_features.jsonl",
        [
            {"time_s": 3.0, "core_start": 0.0, "core_end": 20.0, "label": "continue", "p_cut": 0.05, "accepted": False},
            {"time_s": 6.0, "core_start": 0.0, "core_end": 20.0, "label": "continue", "p_cut": 0.10, "accepted": False},
            {"time_s": 10.0, "core_start": 0.0, "core_end": 20.0, "label": "cut", "p_cut": 0.62, "accepted": False},
            {"time_s": 14.0, "core_start": 0.0, "core_end": 20.0, "label": "cut", "p_cut": 0.40, "accepted": False},
            {"time_s": 18.0, "core_start": 0.0, "core_end": 20.0, "label": "continue", "p_cut": 0.20, "accepted": False},
        ],
    )

    rows, summary = analyze_adaptive_policies(
        reexport_dir=reexport,
        policies=[{"abs_floor": 0.50, "percentile_floor": 0.80, "z_floor": 99.0}],
        long_chunk_min_s=15.0,
        min_chunk_after_split_s=1.2,
    )

    assert summary["candidate_count"] == 1
    assert rows[0]["new_cut_count_vs_current"] == 1
    assert rows[0]["long_chunk_with_new_internal_cut"] == 1
    assert rows[0]["examples"][0]["best_internal_p_cut"] == 0.62
    assert rows[0]["examples"][0]["best_internal_percentile"] == 1.0


def test_adaptive_policy_does_not_promote_low_absolute_peak(tmp_path: Path) -> None:
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
            }
        ],
    )
    _write_jsonl(
        feature_dir / "semantic_split_features.jsonl",
        [
            {"time_s": 3.0, "core_start": 0.0, "core_end": 20.0, "label": "continue", "p_cut": 0.01, "accepted": False},
            {"time_s": 10.0, "core_start": 0.0, "core_end": 20.0, "label": "cut", "p_cut": 0.49, "accepted": False},
            {"time_s": 17.0, "core_start": 0.0, "core_end": 20.0, "label": "continue", "p_cut": 0.02, "accepted": False},
        ],
    )

    rows, _summary = analyze_adaptive_policies(
        reexport_dir=reexport,
        policies=[{"abs_floor": 0.50, "percentile_floor": 0.80, "z_floor": 1.0}],
        long_chunk_min_s=15.0,
        min_chunk_after_split_s=1.2,
    )

    assert rows[0]["new_cut_count_vs_current"] == 0
    assert rows[0]["long_chunk_with_new_internal_cut"] == 0


def test_export_adaptive_analysis_writes_outputs(tmp_path: Path) -> None:
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
                "end": 16.0,
                "duration_s": 16.0,
            }
        ],
    )
    _write_jsonl(
        feature_dir / "semantic_split_features.jsonl",
        [
            {"time_s": 2.0, "core_start": 0.0, "core_end": 16.0, "label": "continue", "p_cut": 0.05, "accepted": False},
            {"time_s": 8.0, "core_start": 0.0, "core_end": 16.0, "label": "cut", "p_cut": 0.66, "accepted": False},
        ],
    )

    summary = export_adaptive_analysis(
        reexport_dir=reexport,
        output_dir=tmp_path / "out",
        policies=[{"abs_floor": 0.50, "percentile_floor": 0.80, "z_floor": 99.0}],
        long_chunk_min_s=15.0,
        min_chunk_after_split_s=1.2,
    )

    manifest = tmp_path / "out" / "adaptive_operating_point.jsonl"
    stored = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 1
    assert summary["rows"][0]["long_chunk_with_new_internal_cut"] == 1
    assert stored["manifest"].endswith("adaptive_operating_point.jsonl")
