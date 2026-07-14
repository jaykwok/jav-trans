from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.boundary.ja.build_semantic_anchor_proposer_audit import (
    adaptive_event_regions,
    build_audit_html,
    select_stratified_proposer_frames,
)


def _timeline_row() -> dict:
    return {
        "sample_id": "s1",
        "duration_s": 10.0,
        "semantic_events": [
            {
                "event_id": "e00",
                "left_unit_id": "u00",
                "right_unit_id": "u01",
                "status": "matched",
                "interval_start_s": 2.0,
                "interval_end_s": 2.4,
            },
            {
                "event_id": "e01",
                "left_unit_id": "u01",
                "right_unit_id": "u02",
                "status": "matched",
                "interval_start_s": 6.8,
                "interval_end_s": 7.2,
            },
        ],
    }


def test_adaptive_regions_use_neighbor_anchor_midpoints_and_source_edges() -> None:
    regions = adaptive_event_regions(_timeline_row())

    assert regions[0]["coarse_anchor_s"] == 2.2
    assert regions[0]["region_start_s"] == 0.0
    assert regions[0]["region_end_s"] == 4.6
    assert regions[1]["region_start_s"] == 4.6
    assert regions[1]["region_end_s"] == 10.0


def test_stratified_candidates_use_proposer_argmax_without_threshold() -> None:
    probabilities = np.zeros(50, dtype=np.float32)
    expected = [1, 12, 24, 35, 48]
    probabilities[expected] = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])

    selected = select_stratified_proposer_frames(
        probabilities,
        region_start_s=0.0,
        region_end_s=1.0,
        frame_hop_s=0.02,
        candidate_count=5,
    )

    assert selected == expected


def test_audit_identifies_learned_projection_and_candidate_labels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.ogg"
    region = tmp_path / "region.wav"
    marked = tmp_path / "marked.wav"
    for path in (source, region, marked):
        path.write_bytes(b"audio")
    row = {
        "event_key": "s1__e00",
        "sample_id": "s1",
        "event_id": "e00",
        "audio": str(source),
        "region_audio": str(region),
        "left_text": "左",
        "right_text": "右",
        "coarse_anchor_s": 1.0,
        "region_start_s": 0.0,
        "region_end_s": 2.0,
        "projection_file_sha256": "projection-sha",
        "proposer_sha256": "proposer-sha",
        "candidates": [
            {
                "candidate_id": "c00",
                "time_s": 0.9,
                "proposer_probability": 0.8,
                "marked_audio": str(marked),
            }
        ],
    }

    page = build_audit_html(
        rows=[row], audit_dir=tmp_path / "audit", update_latest=False
    ).read_text(encoding="utf-8")

    assert "full PTM2048" in page
    assert "Linear(2048→128)" in page
    assert "不是 PCA、不是前 128 截断" in page
    assert "切早" in page
    assert "安全" in page
    assert "切晚" in page
    assert "semantic_anchor_learned_proposer_manual_verdict_v1" in page
