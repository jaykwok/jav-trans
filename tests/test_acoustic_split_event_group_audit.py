from __future__ import annotations

import json
from pathlib import Path

from tools.boundary.ja.build_acoustic_split_event_group_audit import (
    build_items,
    build_page,
    split_runs,
)
from tools.boundary.ja.evaluate_acoustic_split_event_group_audit import evaluate


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_split_runs_are_broken_only_by_continue() -> None:
    candidates = [{"candidate_id": f"c{index}"} for index in range(5)]
    verdicts = [
        {"candidate_id": "c0", "label": "split"},
        {"candidate_id": "c1", "label": "split"},
        {"candidate_id": "c2", "label": "continue"},
        {"candidate_id": "c3", "label": "split"},
        {"candidate_id": "c4", "label": "split"},
    ]

    runs = split_runs(candidates, verdicts)

    assert [[row["candidate_id"] for row in run] for run in runs] == [
        ["c0", "c1"],
        ["c3", "c4"],
    ]


def test_builder_embeds_candidates_and_creates_only_ambiguous_links(tmp_path: Path) -> None:
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"RIFF")
    items = tmp_path / "items.jsonl"
    verdicts = tmp_path / "verdicts.jsonl"
    _write(
        items,
        [
            {
                "sample_id": "s1",
                "audio": str(audio),
                "duration_s": 3.0,
                "candidates": [
                    {"candidate_id": "c0", "time_s": 0.5, "proposer_probability": 0.3, "context_start_s": 0.0, "context_end_s": 1.0},
                    {"candidate_id": "c1", "time_s": 1.0, "proposer_probability": 0.8, "context_start_s": 0.75, "context_end_s": 1.25},
                    {"candidate_id": "c2", "time_s": 2.0, "proposer_probability": 0.6, "context_start_s": 1.5, "context_end_s": 2.5},
                ],
            }
        ],
    )
    _write(
        verdicts,
        [
            {
                "sample_id": "s1",
                "complete": True,
                "candidates": [
                    {"candidate_id": "c0", "label": "split"},
                    {"candidate_id": "c1", "label": "split"},
                    {"candidate_id": "c2", "label": "continue"},
                ],
            }
        ],
    )

    rows = build_items(audit_items=items, verdicts=verdicts, output_dir=tmp_path / "out")

    assert len(rows[0]["links"]) == 1
    assert rows[0]["links"][0]["link_id"] == "c0__c1"
    assert [row["candidate_id"] for row in rows[0]["runs"][0]["candidates"]] == [
        "c0",
        "c1",
    ]


def test_page_explains_same_gap_vs_new_gap(tmp_path: Path) -> None:
    page = build_page(
        rows=[
            {
                "sample_id": "s1",
                "audio": "audio.wav",
                "split_candidate_count": 2,
                "automatic_singleton_event_count": 0,
                "links": [
                    {
                        "link_id": "c0__c1",
                        "left_candidate_id": "c0",
                        "right_candidate_id": "c1",
                        "left_time_s": 1.0,
                        "right_time_s": 1.2,
                        "left_probability": 0.7,
                        "right_probability": 0.8,
                        "play_start_s": 0.5,
                        "play_end_s": 1.7,
                    }
                ],
            }
        ],
        output_dir=tmp_path / "audit",
        update_latest=False,
    ).read_text(encoding="utf-8")

    assert "两个时间点之间有没有恢复真实 speech" in page
    assert "中间无 speech＝同一 gap" in page
    assert "中间有 speech＝两个 gap" in page
    assert "只播放两点之间" in page
    assert "这不是重复审核样本" in page
    assert ".join('\\n')+'\\n'" in page


def test_evaluator_compiles_event_groups_and_subislands(tmp_path: Path) -> None:
    items = tmp_path / "items.jsonl"
    verdicts = tmp_path / "verdicts.jsonl"
    _write(
        items,
        [
            {
                "schema": "acoustic_split_event_group_audit_v1",
                "sample_id": "s1",
                "audio": "audio.wav",
                "duration_s": 5.0,
                "links": [
                    {"link_id": "c0__c1"},
                    {"link_id": "c1__c2"},
                ],
                "runs": [
                    {
                        "run_id": "r00",
                        "candidate_ids": ["c0", "c1", "c2"],
                        "candidates": [
                            {"candidate_id": "c0", "time_s": 1.0, "proposer_probability": 0.5},
                            {"candidate_id": "c1", "time_s": 1.2, "proposer_probability": 0.9},
                            {"candidate_id": "c2", "time_s": 3.0, "proposer_probability": 0.7},
                        ],
                    }
                ],
            }
        ],
    )
    _write(
        verdicts,
        [
            {
                "schema": "acoustic_split_event_group_manual_verdict_v1",
                "sample_id": "s1",
                "links": [
                    {"link_id": "c0__c1", "decision": "same_event"},
                    {"link_id": "c1__c2", "decision": "new_event"},
                ],
            }
        ],
    )

    summary = evaluate(items=items, verdicts=verdicts, output_dir=tmp_path / "gate")

    assert summary["training_ready"] is True
    assert summary["event_count"] == 2
    assert summary["provisional_subisland_count"] == 3
    events = [
        json.loads(line)
        for line in (tmp_path / "gate" / "events.jsonl").read_text().splitlines()
    ]
    assert events[0]["representative_candidate_id"] == "c1"
