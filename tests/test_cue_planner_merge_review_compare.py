from __future__ import annotations

import json
from pathlib import Path

from tools.subtitles.compare_cue_planner_merge_reviews import compare_reviews


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_compare_cue_planner_merge_reviews_exports_extra_and_dropped(tmp_path: Path):
    baseline = _write_jsonl(
        tmp_path / "baseline.jsonl",
        [
            {"left_cue_id": 1, "right_cue_id": 2, "risk_tags": "near_speaker_threshold"},
            {"left_cue_id": 3, "right_cue_id": 4, "risk_tags": "fallback_risk"},
        ],
    )
    candidate = _write_jsonl(
        tmp_path / "candidate.jsonl",
        [
            {"left_cue_id": 1, "right_cue_id": 2, "risk_tags": "near_speaker_threshold"},
            {"left_cue_id": 5, "right_cue_id": 6, "risk_tags": "loose_gap,high_speaker_score"},
        ],
    )

    summary = compare_reviews(
        baseline_path=baseline,
        candidate_path=candidate,
        output_dir=tmp_path / "out",
        candidate_label="th95-constrained",
    )

    assert summary["extra_count"] == 1
    assert summary["dropped_count"] == 1
    assert summary["extra_risk_tag_counts"]["loose_gap"] == 1
    assert summary["dropped_risk_tag_counts"]["fallback_risk"] == 1
    extra = [
        json.loads(line)
        for line in (tmp_path / "out" / "candidate_extra_merge_review_items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert extra[0]["comparison"] == "th95-constrained_extra_vs_baseline"
    assert (tmp_path / "out" / "summary.md").exists()
