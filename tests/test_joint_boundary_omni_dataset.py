from __future__ import annotations

from tools.datasets.label_joint_boundary_preasr_with_omni import (
    _build_prompt,
    _normalize_chunk_decision,
    _normalize_split_decision,
    _select_chunk_rows,
    _select_split_rows,
)
from tools.datasets.prepare_joint_boundary_omni_dataset import _window_starts


def test_joint_prompt_contains_both_label_tasks() -> None:
    prompt = _build_prompt(
        [
            {
                "index": 4,
                "time_s": 1.25,
                "label": "continue",
                "p_cut": 0.2,
            }
        ],
        [
            {
                "chunk_index": 2,
                "start": 0.0,
                "end": 2.0,
            }
        ],
        duration_s=3.0,
    )
    assert "split_candidates=" in prompt
    assert "runtime_chunks=" in prompt
    assert '"id":"s000"' in prompt
    assert '"id":"p000"' in prompt
    assert "重复但有明确词义" in prompt


def test_invalid_cut_semantics_become_unsure() -> None:
    decision = _normalize_split_decision(
        {
            "label": "cut",
            "confidence": 0.99,
            "left_complete": True,
            "right_complete": False,
            "merged_better": False,
        },
        confidence_threshold=0.8,
    )
    assert decision["label"] == "unsure"


def test_chunk_confidence_thresholds_are_asymmetric() -> None:
    keep = _normalize_chunk_decision(
        {"label": "keep", "confidence": 0.85},
        keep_confidence=0.8,
        drop_confidence=0.9,
    )
    drop = _normalize_chunk_decision(
        {"label": "drop", "confidence": 0.85},
        keep_confidence=0.8,
        drop_confidence=0.9,
    )
    assert keep["label"] == "definite_keep"
    assert drop["label"] == "ambiguous_ignore"


def test_joint_selection_is_bounded_and_deterministic() -> None:
    split_rows = [
        {
            "index": index,
            "time_s": float(index),
            "p_cut": index / 100.0,
            "accepted": index % 9 == 0,
        }
        for index in range(100)
    ]
    chunk_rows = [
        {
            "chunk_index": index,
            "duration_s": float(index + 1),
        }
        for index in range(100)
    ]
    assert _select_split_rows(split_rows, limit=16, seed="x") == _select_split_rows(
        split_rows, limit=16, seed="x"
    )
    assert len(_select_split_rows(split_rows, limit=16, seed="x")) == 16
    assert _select_chunk_rows(chunk_rows, limit=20, seed="x") == _select_chunk_rows(
        chunk_rows, limit=20, seed="x"
    )
    assert len(_select_chunk_rows(chunk_rows, limit=20, seed="x")) == 20


def test_window_starts_are_reproducible_and_in_bounds() -> None:
    import random

    first = _window_starts(
        duration_s=1000.0,
        window_s=75.0,
        count=2,
        rng=random.Random(17),
    )
    second = _window_starts(
        duration_s=1000.0,
        window_s=75.0,
        count=2,
        rng=random.Random(17),
    )
    assert first == second
    assert len(first) == 2
    assert all(0.0 <= value <= 925.0 for value in first)
