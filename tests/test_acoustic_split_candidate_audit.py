from __future__ import annotations

import json
from pathlib import Path

from tools.boundary.ja.build_acoustic_split_candidate_audit import (
    adaptive_candidate_count,
    adaptive_internal_frame_region,
    build_audit_html,
    candidate_contexts,
    select_fixed_sources,
)
from tools.boundary.ja.evaluate_acoustic_split_candidate_audit import evaluate


def test_adaptive_candidate_count_stays_in_fixed_audit_budget() -> None:
    assert adaptive_candidate_count(25) == 5
    assert 5 < adaptive_candidate_count(400) < 9
    assert adaptive_candidate_count(5000) == 9


def test_adaptive_internal_region_excludes_outer_edge_peaks() -> None:
    start, end = adaptive_internal_frame_region(150, candidate_count=5)

    assert start == 21
    assert end == 129
    assert end - start >= 5


def test_candidate_contexts_partition_without_overlap() -> None:
    contexts = candidate_contexts([1.0, 2.0, 4.0], duration_s=6.0)

    assert contexts[0] == {
        "context_start_s": 0.0,
        "candidate_time_s": 1.0,
        "context_end_s": 1.5,
    }
    assert contexts[0]["context_end_s"] == contexts[1]["context_start_s"]
    assert contexts[1]["context_end_s"] == contexts[2]["context_start_s"]
    assert contexts[2]["context_end_s"] == 6.0


def test_fixed_selection_covers_duration_buckets() -> None:
    rows = [
        {
            "sample_id": f"s{index}",
            "duration_s": float(index + 1),
            "max_proposer_probability": float(index % 3) / 2.0,
            "proposer_probability_std": 0.1,
        }
        for index in range(10)
    ]

    selected = select_fixed_sources(rows, count=5)

    assert len(selected) == 5
    assert len({row["sample_id"] for row in selected}) == 5
    assert [row["duration_s"] for row in selected] == sorted(
        row["duration_s"] for row in selected
    )


def test_page_is_audio_only_and_explains_inner_responsibility(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    rows = [
        {
            "sample_id": "s1",
            "audio": str(audio),
            "duration_s": 2.0,
            "outer_checkpoint_sha256": "outer-sha",
            "proposer_sha256": "proposer-sha",
            "projection_digest": "projection-digest",
            "candidates": [
                {
                    "candidate_id": "c00",
                    "time_s": 1.0,
                    "context_start_s": 0.0,
                    "context_end_s": 2.0,
                    "proposer_probability": 0.8,
                }
            ],
        }
    ]

    page = build_audit_html(
        rows=rows, output_dir=tmp_path / "audit", update_latest=False
    ).read_text(encoding="utf-8")

    assert "不提供文本、Omni 时间轴或旧切点" in page
    assert "provisional speech sub-island" in page
    assert "Inner Refiner" in page
    assert "同一静音块里的多个候选全部标" in page
    assert "one-sided speech" in page
    assert "不重叠试听区间" in page
    assert "split" in page and "continue" in page and "unsure" in page
    assert "acoustic_split_candidate_manual_verdict_v1" in page
    assert "reference_text" not in page
    assert ".join('\\n')+'\\n'" in page


def test_page_refreshes_audit_navigation_with_index_path(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    monkeypatch.setattr(
        "tools.boundary.ja.build_acoustic_split_candidate_audit.update_audit_entrypoints",
        lambda **kwargs: calls.append(kwargs),
    )
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    rows = [
        {
            "sample_id": "s1",
            "audio": str(audio),
            "duration_s": 2.0,
            "outer_checkpoint_sha256": "outer-sha",
            "proposer_sha256": "proposer-sha",
            "projection_digest": "projection-digest",
            "candidates": [
                {
                    "candidate_id": "c00",
                    "time_s": 1.0,
                    "context_start_s": 0.0,
                    "context_end_s": 2.0,
                    "proposer_probability": 0.8,
                }
            ],
        }
    ]

    page = build_audit_html(rows=rows, output_dir=tmp_path / "audit")

    assert calls == [
        {"latest_html": page, "title": "Acoustic Split v3 fixed audit"}
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_gate_requires_complete_coverage_and_both_training_classes(
    tmp_path: Path,
) -> None:
    items = tmp_path / "items.jsonl"
    verdicts = tmp_path / "verdicts.jsonl"
    _write_jsonl(
        items,
        [
            {
                "schema": "acoustic_split_candidate_audit_v1",
                "sample_id": "s1",
                "candidates": [
                    {"candidate_id": "c00"},
                    {"candidate_id": "c01"},
                ],
            }
        ],
    )
    _write_jsonl(
        verdicts,
        [
            {
                "schema": "acoustic_split_candidate_manual_verdict_v1",
                "sample_id": "s1",
                "coverage": "complete",
                "candidates": [
                    {"candidate_id": "c00", "label": "continue"},
                    {"candidate_id": "c01", "label": "split"},
                ],
            }
        ],
    )

    summary = evaluate(
        items=items, verdicts=verdicts, output=tmp_path / "summary.json"
    )

    assert summary["manual_fixed_gate_pass"] is True
    assert summary["proposal_coverage"] == 1.0
    assert summary["training_ready"] is True


def test_gate_does_not_turn_missing_or_unsure_into_continue(tmp_path: Path) -> None:
    items = tmp_path / "items.jsonl"
    verdicts = tmp_path / "verdicts.jsonl"
    _write_jsonl(
        items,
        [
            {
                "schema": "acoustic_split_candidate_audit_v1",
                "sample_id": "s1",
                "candidates": [{"candidate_id": "c00"}],
            }
        ],
    )
    _write_jsonl(
        verdicts,
        [
            {
                "schema": "acoustic_split_candidate_manual_verdict_v1",
                "sample_id": "s1",
                "coverage": "missed",
                "candidates": [{"candidate_id": "c00", "label": "unsure"}],
            }
        ],
    )

    summary = evaluate(
        items=items, verdicts=verdicts, output=tmp_path / "summary.json"
    )

    assert summary["label_counts"] == {"unsure": 1}
    assert summary["manual_fixed_gate_pass"] is False
    assert summary["training_ready"] is False
