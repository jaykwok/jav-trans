from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr.pre_asr_cueqc import PRE_ASR_CUEQC_IGNORE_LABEL
from tools.asr.cueqc.compile_pre_asr_v12_features import read_labels
from tools.datasets.build_pre_asr_label_overrides import build_overrides
from tools.datasets.compile_joint_boundary_preasr_dataset import (
    _pre_asr_override_summary,
)


def _replay_row(
    window_id: str,
    chunk_index: int,
    *,
    truth: str,
    prediction: str,
) -> dict:
    return {
        "window_id": window_id,
        "video_id": window_id.rsplit("-w", 1)[0],
        "chunk_index": chunk_index,
        "start": float(chunk_index),
        "end": float(chunk_index) + 1.0,
        "duration_s": 1.0,
        "truth": truth,
        "prediction": prediction,
        "prob_drop": 0.99,
        "partition": "train",
    }


def _verdict_row(window_id: str, chunk_index: int, *, verdict: str, direction: str) -> dict:
    return {
        "candidate_id": f"preasr-{window_id}-chunk{chunk_index:05d}",
        "verdict": verdict,
        "direction": direction,
        "window_id": window_id,
        "chunk_index": chunk_index,
        "omni_label": "definite_keep" if direction == "A" else "definite_drop",
        "omni_confidence": 0.9,
    }


def test_build_overrides_policy() -> None:
    false_decisions = [
        _replay_row("vid-a-w00", 1, truth="keep", prediction="drop"),  # audited A
        _replay_row("vid-a-w00", 2, truth="keep", prediction="drop"),  # audited A unsure
        _replay_row("vid-b-w00", 3, truth="drop", prediction="keep"),  # audited B keep
        _replay_row("vid-b-w00", 4, truth="keep", prediction="drop"),  # unaudited A
        _replay_row("vid-c-w00", 5, truth="drop", prediction="keep"),  # unaudited B
    ]
    verdicts = [
        _verdict_row("vid-a-w00", 1, verdict="drop", direction="A"),
        _verdict_row("vid-a-w00", 2, verdict="unsure", direction="A"),
        _verdict_row("vid-b-w00", 3, verdict="keep", direction="B"),
    ]
    overrides, summary = build_overrides(verdicts, false_decisions, audit_id="t")
    by_id = {row["candidate_id"]: row for row in overrides}
    assert len(overrides) == 4
    assert by_id["preasr-vid-a-w00-chunk00001"]["label"] == "definite_drop"
    assert by_id["preasr-vid-a-w00-chunk00002"]["label"] == "ambiguous_ignore"
    assert by_id["preasr-vid-b-w00-chunk00003"]["label"] == "definite_keep"
    unaudited = by_id["preasr-vid-b-w00-chunk00004"]
    assert unaudited["label"] == "ambiguous_ignore"
    assert unaudited["override_source"] == "audit_policy_unaudited_a"
    # unaudited B keeps its Omni drop label: no override row at all
    assert "preasr-vid-c-w00-chunk00005" not in by_id
    assert summary["direction_counts"] == {"A": 3, "B": 2}
    assert summary["unaudited_a_ambiguous_ignore"] == 1
    assert summary["unaudited_b_noop"] == 1
    # every row carries the span key for stage-D projection onto new chunks
    assert all({"video_id", "start", "end"} <= set(row) for row in overrides)


def test_build_overrides_rejects_unknown_candidate() -> None:
    false_decisions = [_replay_row("vid-a-w00", 1, truth="keep", prediction="drop")]
    verdicts = [_verdict_row("vid-a-w00", 9, verdict="drop", direction="A")]
    with pytest.raises(ValueError, match="not in replay"):
        build_overrides(verdicts, false_decisions, audit_id="t")


def test_build_overrides_rejects_direction_mismatch() -> None:
    false_decisions = [_replay_row("vid-a-w00", 1, truth="keep", prediction="drop")]
    verdicts = [_verdict_row("vid-a-w00", 1, verdict="drop", direction="B")]
    with pytest.raises(ValueError, match="direction"):
        build_overrides(verdicts, false_decisions, audit_id="t")


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_override_summary_validates_and_counts(tmp_path: Path) -> None:
    base = [
        {"candidate_id": "preasr-v-w00-chunk00000", "label": "definite_keep"},
        {"candidate_id": "preasr-v-w00-chunk00001", "label": "definite_drop"},
    ]
    overrides = _write_jsonl(
        tmp_path / "overrides.jsonl",
        [
            {"candidate_id": "preasr-v-w00-chunk00000", "label": "definite_drop"},
            {"candidate_id": "preasr-v-w00-chunk00001", "label": "definite_drop"},
        ],
    )
    summary = _pre_asr_override_summary(base, [overrides])
    assert summary["count"] == 2
    assert summary["changed_from_base"] == 1
    assert summary["by_label"] == {"definite_drop": 2}

    bad = _write_jsonl(
        tmp_path / "bad.jsonl",
        [{"candidate_id": "preasr-v-w00-chunk00099", "label": "definite_drop"}],
    )
    with pytest.raises(ValueError, match="not present in base"):
        _pre_asr_override_summary(base, [bad])


def test_read_labels_applies_override_last(tmp_path: Path) -> None:
    base = _write_jsonl(
        tmp_path / "labels.jsonl",
        [
            {"candidate_id": "preasr-v-w00-chunk00000", "label": "definite_keep"},
            {"candidate_id": "preasr-v-w00-chunk00001", "label": "definite_keep"},
        ],
    )
    override = _write_jsonl(
        tmp_path / "overrides.jsonl",
        [
            {"candidate_id": "preasr-v-w00-chunk00000", "label": "definite_drop"},
            {"candidate_id": "preasr-v-w00-chunk00001", "label": "ambiguous_ignore"},
        ],
    )
    labels = read_labels([str(base), str(override)])
    assert labels["preasr-v-w00-chunk00000"]["label_index"] == 0
    assert labels["preasr-v-w00-chunk00001"]["label_index"] == PRE_ASR_CUEQC_IGNORE_LABEL
    # the expanded "<window>#<index>" key must resolve to the override too
    assert labels["v-w00#0"]["label_index"] == 0
