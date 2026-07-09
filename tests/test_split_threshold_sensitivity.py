from __future__ import annotations

import json
from pathlib import Path

from tools.datasets.analyze_split_threshold_sensitivity import (
    analyze_thresholds,
    export_threshold_sensitivity,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_analyze_thresholds_counts_long_chunks_with_new_internal_cuts(tmp_path: Path) -> None:
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
                "end": 23.0,
                "duration_s": 3.0,
            },
        ],
    )
    _write_jsonl(
        feature_dir / "semantic_split_features.jsonl",
        [
            {
                "time_s": 10.0,
                "core_start": 0.0,
                "core_end": 20.0,
                "label": "cut",
                "p_cut": 0.72,
                "accepted": False,
            },
            {
                "time_s": 18.95,
                "core_start": 0.0,
                "core_end": 20.0,
                "label": "cut",
                "p_cut": 0.95,
                "accepted": False,
            },
            {
                "time_s": 19.0,
                "core_start": 0.0,
                "core_end": 20.0,
                "label": "cut",
                "p_cut": 0.98,
                "accepted": True,
            },
        ],
    )

    rows, summary = analyze_thresholds(
        reexport_dir=reexport,
        thresholds=[0.75, 0.70],
        long_chunk_min_s=15.0,
        min_chunk_after_split_s=1.2,
    )

    assert summary["candidate_count"] == 2
    assert summary["split_candidate_count"] == 3
    assert rows[0]["threshold"] == 0.75
    assert rows[0]["new_cut_count_vs_current"] == 0
    assert rows[0]["long_chunk_with_new_internal_cut"] == 0
    assert rows[1]["threshold"] == 0.70
    assert rows[1]["new_cut_count_vs_current"] == 1
    assert rows[1]["long_chunk_with_new_internal_cut"] == 1
    assert rows[1]["examples"][0]["candidate_id"] == "preasr-vid-w00-chunk00000"
    assert rows[1]["examples"][0]["best_internal_p_cut"] == 0.72


def test_export_threshold_sensitivity_writes_outputs(tmp_path: Path) -> None:
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
            {
                "time_s": 8.0,
                "core_start": 0.0,
                "core_end": 16.0,
                "label": "cut",
                "p_cut": 0.66,
                "accepted": False,
            }
        ],
    )

    summary = export_threshold_sensitivity(
        reexport_dir=reexport,
        output_dir=tmp_path / "out",
        thresholds=[0.65],
        long_chunk_min_s=15.0,
        min_chunk_after_split_s=1.2,
    )

    manifest = tmp_path / "out" / "threshold_sensitivity.jsonl"
    stored = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 1
    assert summary["rows"][0]["long_chunk_with_new_internal_cut"] == 1
    assert stored["manifest"].endswith("threshold_sensitivity.jsonl")
