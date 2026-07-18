from __future__ import annotations

from tools.boundary.ja.evaluate_display_span_exact_core_gate import (
    evaluate_exact_display_gate,
    paired_display_chunk_rows,
    reconstruct_kept_display_chunks,
)


def _runtime_row(
    row_id: str,
    start: float,
    end: float,
    position: int,
    *,
    inner_start: float | None = None,
    inner_end: float | None = None,
) -> dict:
    return {
        "row_id": row_id,
        "source_id": "source-a",
        "start_s": start,
        "end_s": end,
        "planned_island_id": "island-a",
        "position": position,
        "inner_prediction": {
            "start_s": start if inner_start is None else inner_start,
            "end_s": end if inner_end is None else inner_end,
            "start_action": "refined",
            "end_action": "refined",
        },
    }


def test_reconstruct_kept_display_chunks_applies_only_adjacent_kept_pair() -> None:
    runtime = [
        _runtime_row("a", 0.8, 3.0, 0, inner_end=2.1),
        _runtime_row("b", 3.0, 5.2, 1, inner_start=3.9),
        _runtime_row("c", 5.2, 6.0, 2),
        _runtime_row("d", 6.0, 7.0, 3),
    ]
    predictions = {
        "a": {"prediction": "keep", "prob_keep": 0.9, "truth_label": "keep"},
        "b": {"prediction": "keep", "prob_keep": 0.8, "truth_label": "keep"},
        "c": {"prediction": "drop", "prob_keep": 0.1},
        "d": {"prediction": "keep", "prob_keep": 0.7},
    }

    chunks = reconstruct_kept_display_chunks(runtime, predictions)

    assert len(chunks) == 3
    assert chunks[0]["display_end_s"] == 2.1
    assert chunks[1]["display_start_s"] == 3.9
    assert chunks[2]["display_start_s"] == 6.0
    assert chunks[2]["inner_action"] == "not_adjacent"


def test_reconstruct_kept_display_chunks_merges_abstained_adjacent_pair() -> None:
    runtime = [
        _runtime_row("a", 1.0, 2.0, 0),
        _runtime_row("b", 2.0, 3.0, 1),
    ]
    runtime[0]["inner_prediction"]["end_action"] = "abstain"
    predictions = {
        "a": {"prediction": "keep", "prob_keep": 0.9, "truth_label": "keep"},
        "b": {"prediction": "keep", "prob_keep": 0.8, "truth_label": "keep"},
    }

    chunks = reconstruct_kept_display_chunks(runtime, predictions)

    assert len(chunks) == 1
    assert chunks[0]["display_start_s"] == 1.0
    assert chunks[0]["display_end_s"] == 3.0
    assert chunks[0]["member_row_ids"] == ["a", "b"]
    assert chunks[0]["inner_action"] == "abstain_merge"


def test_reconstruct_inner_all_edges_changes_display_only() -> None:
    runtime = [_runtime_row("a", 1.0, 4.0, 0, inner_start=1.4, inner_end=3.6)]
    predictions = {"a": {"prediction": "keep", "prob_keep": 0.9}}

    chunks = reconstruct_kept_display_chunks(
        runtime,
        predictions,
        display_mode="inner_all_edges",
    )

    assert chunks[0]["start_s"] == 1.0
    assert chunks[0]["end_s"] == 4.0
    assert chunks[0]["display_start_s"] == 1.4
    assert chunks[0]["display_end_s"] == 3.6


def test_reconstruct_model_mode_merges_overlapping_predicted_spans() -> None:
    runtime = [
        _runtime_row("a", 1.0, 2.0, 0),
        _runtime_row("b", 2.0, 3.0, 1),
    ]
    predictions = {
        "a": {"prediction": "keep", "prob_keep": 0.9},
        "b": {"prediction": "keep", "prob_keep": 0.8},
    }
    chunks = reconstruct_kept_display_chunks(
        runtime,
        predictions,
        display_mode="model",
        display_overrides={
            "a": {"display_start_s": 0.9, "display_end_s": 2.6},
            "b": {"display_start_s": 1.4, "display_end_s": 3.1},
        },
    )
    assert len(chunks) == 1
    assert chunks[0]["display_start_s"] == 0.9
    assert chunks[0]["display_end_s"] == 3.1
    assert chunks[0]["member_row_ids"] == ["a", "b"]
    assert chunks[0]["inner_action"] == "model_overlap_merge"


def test_reconstruct_model_mode_keeps_non_overlapping_predicted_spans_separate() -> None:
    runtime = [
        _runtime_row("a", 1.0, 2.0, 0),
        _runtime_row("b", 2.0, 3.0, 1),
    ]
    predictions = {
        "a": {"prediction": "keep", "prob_keep": 0.9},
        "b": {"prediction": "keep", "prob_keep": 0.8},
    }
    chunks = reconstruct_kept_display_chunks(
        runtime,
        predictions,
        display_mode="model",
        display_overrides={
            "a": {"display_start_s": 0.9, "display_end_s": 1.8},
            "b": {"display_start_s": 2.2, "display_end_s": 3.1},
        },
    )
    assert len(chunks) == 2


def test_exact_display_gate_passes_two_precise_core_spans() -> None:
    sources = {
        "source-a": {
            "source_id": "source-a",
            "source_partition": "test",
            "audio": "source.wav",
            "cores": [
                {"core_id": "core-a", "text": "a", "start_s": 1.0, "end_s": 2.0},
                {"core_id": "core-b", "text": "b", "start_s": 4.0, "end_s": 5.0},
            ],
        }
    }
    runtime = {
        "source-a": [
            _runtime_row("a", 0.8, 3.0, 0, inner_end=2.1),
            _runtime_row("b", 3.0, 5.2, 1, inner_start=3.9),
        ]
    }
    predictions = {
        "a": {"prediction": "keep", "prob_keep": 0.9, "truth_label": "keep"},
        "b": {"prediction": "keep", "prob_keep": 0.8, "truth_label": "keep"},
    }

    rows, summary = evaluate_exact_display_gate(
        sources=sources,
        runtime_groups=runtime,
        predictions=predictions,
        tolerance_s=0.3,
        required_coverage=0.95,
    )

    assert len(rows) == 2
    assert summary["start_within_tolerance_coverage"] == 1.0
    assert summary["end_within_tolerance_coverage"] == 1.0
    assert summary["missing_core_count"] == 0
    assert summary["fragmented_core_count"] == 0
    assert summary["gate_pass"] is True


def test_exact_display_gate_rejects_missing_and_fragmented_cores() -> None:
    sources = {
        "source-a": {
            "source_id": "source-a",
            "source_partition": "test",
            "audio": "source.wav",
            "cores": [
                {"core_id": "core-a", "text": "a", "start_s": 1.0, "end_s": 3.0},
                {"core_id": "core-b", "text": "b", "start_s": 5.0, "end_s": 6.0},
            ],
        }
    }
    runtime = {
        "source-a": [
            _runtime_row("a", 0.9, 2.0, 0),
            _runtime_row("b", 2.0, 3.1, 2),
            _runtime_row("c", 5.0, 6.0, 4),
        ]
    }
    predictions = {
        "a": {"prediction": "keep", "prob_keep": 0.9, "truth_label": "keep"},
        "b": {"prediction": "keep", "prob_keep": 0.9, "truth_label": "keep"},
        "c": {"prediction": "drop", "prob_keep": 0.1, "truth_label": "keep"},
    }

    _rows, summary = evaluate_exact_display_gate(
        sources=sources,
        runtime_groups=runtime,
        predictions=predictions,
        tolerance_s=0.3,
        required_coverage=0.95,
    )

    assert summary["missing_core_ids"] == ["core-b"]
    assert summary["fragmented_core_count"] == 1
    assert summary["missing_reason_counts"] == {"cueqc_removed_all_overlaps": 1}
    assert summary["gate_pass"] is False


def test_paired_display_chunk_gate_uses_union_of_constituent_cores() -> None:
    runtime = [_runtime_row("a", 0.8, 5.2, 0)]
    predictions = {"a": {"prediction": "keep", "truth_label": "keep"}}
    rows, missing = paired_display_chunk_rows(
        source_id="source-a",
        source_partition="test",
        cores=[
            {"core_id": "one", "start_s": 1.0, "end_s": 2.0},
            {"core_id": "two", "start_s": 4.0, "end_s": 5.0},
        ],
        runtime_rows=runtime,
        predictions=predictions,
        final_chunks=[{
            **runtime[0], "display_start_s": 0.9, "display_end_s": 5.1,
            "member_row_ids": ["a"],
        }],
        tolerance_s=0.3,
    )
    assert missing == set()
    assert rows[0]["target_core_ids"] == ["one", "two"]
    assert rows[0]["target_start_s"] == 1.0
    assert rows[0]["target_end_s"] == 5.0
    assert rows[0]["start_within_tolerance"] is True
    assert rows[0]["end_within_tolerance"] is True


def test_paired_display_chunk_gate_locks_internal_split_edges() -> None:
    runtime = [_runtime_row("a", 2.0, 4.0, 0)]
    runtime[0]["left_edge_is_split_cut"] = True
    runtime[0]["right_edge_is_split_cut"] = True
    predictions = {"a": {"prediction": "keep", "truth_label": "keep"}}
    rows, missing = paired_display_chunk_rows(
        source_id="source-a",
        source_partition="test",
        cores=[{"core_id": "one", "start_s": 1.0, "end_s": 5.0}],
        runtime_rows=runtime,
        predictions=predictions,
        final_chunks=[{
            **runtime[0], "display_start_s": 2.0, "display_end_s": 4.0,
            "member_row_ids": ["a"],
        }],
        tolerance_s=0.3,
    )
    assert missing == set()
    assert rows[0]["target_start_s"] == 2.0
    assert rows[0]["target_end_s"] == 4.0
    assert rows[0]["start_within_tolerance"] is True
    assert rows[0]["end_within_tolerance"] is True
