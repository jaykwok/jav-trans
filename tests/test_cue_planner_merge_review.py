from __future__ import annotations

import json
from pathlib import Path

from tools.subtitles.export_cue_planner_merge_review import build_review


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_export_cue_planner_merge_review_prioritizes_risky_merges(tmp_path: Path):
    before_blocks = _write_json(
        tmp_path / "before_blocks.json",
        [
            {"start": 0.0, "end": 0.7, "ja_text": "あ", "zh_text": "啊", "cue_id": 0},
            {"start": 0.8, "end": 1.4, "ja_text": "ん", "zh_text": "嗯", "cue_id": 1},
            {"start": 2.0, "end": 2.4, "ja_text": "だめ", "zh_text": "不行", "cue_id": 2},
            {"start": 2.5, "end": 3.1, "ja_text": "やめて", "zh_text": "停下", "cue_id": 3},
        ],
    )
    actions = _write_json(
        tmp_path / "planner_actions.json",
        [
            {
                "left_index": 0,
                "right_index": 1,
                "score": 0.6,
                "gap_s": 0.1,
                "combined_duration_s": 1.4,
                "combined_text_units": 4.0,
                "reasons": ["tight_gap"],
                "annotations": {},
            },
            {
                "left_index": 2,
                "right_index": 3,
                "score": 0.5,
                "gap_s": 0.1,
                "combined_duration_s": 1.1,
                "combined_text_units": 8.0,
                "reasons": ["tight_gap", "fallback_risk_boundary"],
                "annotations": {
                    "diagnostics": {
                        "crosses_chunk": True,
                        "risky_chunks": [9],
                        "warn_chunks": [9],
                    },
                },
            },
        ],
    )

    summary = build_review(
        before_blocks_path=before_blocks,
        planner_actions_path=actions,
        output_dir=tmp_path / "review",
        reading_warn_units_per_s=6.0,
        top_n=5,
    )

    assert summary["review_item_count"] == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "review" / "merge_review_items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["left_index"] == 2
    assert "fallback_risk" in rows[0]["risk_tags"]
    assert "crosses_chunk" in rows[0]["risk_tags"]
    assert rows[0]["priority_rank"] == 1
    assert rows[1]["left_index"] == 0
    assert (tmp_path / "review" / "merge_review_items.csv").exists()
    assert (tmp_path / "review" / "summary.md").read_text(encoding="utf-8").startswith(
        "# Cue Planner Merge Review"
    )
