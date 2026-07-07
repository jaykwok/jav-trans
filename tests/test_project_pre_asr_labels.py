from __future__ import annotations

import json
from pathlib import Path

from tools.datasets.project_pre_asr_labels import (
    effective_label_rows,
    project_label_rows,
)


def _label(
    window_id: str,
    index: int,
    start: float,
    end: float,
    label: str,
    *,
    manual: bool = False,
) -> dict:
    row = {
        "candidate_id": f"preasr-{window_id}-chunk{index:05d}",
        "sample_id": f"preasr-{window_id}-chunk{index:05d}",
        "window_id": window_id,
        "audio_id": window_id,
        "chunk_index": index,
        "start": start,
        "end": end,
        "duration_s": end - start,
        "label": label,
        "label_source": "omni:qwen3.5-omni-flash",
    }
    if manual:
        row["override_source"] = "manual_audit"
        row["label_source"] = "manual_audit:test"
    return row


def _candidate(window_id: str, index: int, start: float, end: float) -> dict:
    return {
        "candidate_id": f"preasr-{window_id}-newchunk{index:05d}",
        "sample_id": f"preasr-{window_id}-newchunk{index:05d}",
        "window_id": window_id,
        "audio_id": window_id,
        "video_id": window_id,
        "chunk_index": index,
        "start": start,
        "end": end,
        "duration_s": end - start,
        "feature_schema": "pre_asr_cueqc_features_v9",
        "runtime_adapter": "pre_asr_semantic_chunk_sequence_v4",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_projects_near_boundary_match() -> None:
    labels, unmatched, summary = project_label_rows(
        [_label("w0", 0, 1.00, 2.00, "definite_keep")],
        [_candidate("w0", 0, 1.08, 1.95)],
    )

    assert not unmatched
    assert labels[0]["label"] == "definite_keep"
    assert labels[0]["training_label_included"] is True
    assert labels[0]["projection"]["method"] == "boundary_match"
    assert summary["projection_methods"] == {"boundary_match": 1}


def test_projects_dominant_overlap_with_flag() -> None:
    labels, unmatched, summary = project_label_rows(
        [_label("w0", 0, 0.0, 10.0, "definite_drop")],
        [_candidate("w0", 0, 0.5, 9.5)],
    )

    assert not unmatched
    assert labels[0]["label"] == "definite_drop"
    assert labels[0]["projection"]["method"] == "dominant_overlap"
    assert labels[0]["projection_flag"] == "dominant_overlap"
    assert summary["dominant_overlap_flagged_count"] == 1


def test_conflicting_cross_boundary_overlap_stays_unmatched() -> None:
    labels, unmatched, summary = project_label_rows(
        [
            _label("w0", 0, 0.0, 1.0, "definite_keep"),
            _label("w0", 1, 1.0, 2.0, "definite_drop"),
        ],
        [_candidate("w0", 0, 0.2, 1.2)],
    )

    assert not labels
    assert unmatched[0]["unmatched_reason"] == "cross_boundary_label_conflict"
    assert summary["unmatched_candidate_count"] == 1


def test_failed_manual_projection_forces_overlapping_new_chunk_to_ignore() -> None:
    labels, unmatched, summary = project_label_rows(
        [_label("w0", 0, 0.0, 1.0, "definite_drop", manual=True)],
        [_candidate("w0", 0, 0.5, 1.5)],
    )

    assert not unmatched
    assert labels[0]["label"] == "ambiguous_ignore"
    assert labels[0]["training_label_included"] is False
    assert labels[0]["projection"]["method"] == "manual_projection_failed_ignore"
    assert labels[0]["projection"]["source_manual_audit"] is True
    assert summary["manual_projection_failed_old_count"] == 1
    assert summary["manual_projection_failed_ignore_count"] == 1


def test_manual_labels_outside_projected_windows_are_not_failures() -> None:
    labels, unmatched, summary = project_label_rows(
        [_label("other-w00", 0, 0.0, 1.0, "definite_drop", manual=True)],
        [_candidate("w0", 0, 0.0, 1.0)],
    )

    assert not labels
    assert unmatched[0]["unmatched_reason"] == "no_overlap"
    assert summary["manual_projection_failed_old_count"] == 0
    assert summary["manual_projection_failed_ignore_count"] == 0


def test_manual_projection_failure_overrides_machine_projection() -> None:
    labels, unmatched, summary = project_label_rows(
        [
            _label("w0", 0, 0.0, 10.0, "definite_keep"),
            _label("w0", 1, 4.0, 5.0, "definite_drop", manual=True),
        ],
        [_candidate("w0", 0, 0.0, 10.0)],
    )

    assert not unmatched
    assert labels[0]["label"] == "ambiguous_ignore"
    assert labels[0]["projection"]["method"] == "manual_projection_failed_ignore"
    assert summary["projection_methods"] == {"manual_projection_failed_ignore": 1}


def test_effective_label_rows_apply_override_last(tmp_path: Path) -> None:
    base = _write_jsonl(
        tmp_path / "labels.jsonl",
        [_label("w0", 0, 0.0, 1.0, "definite_keep")],
    )
    override = _write_jsonl(
        tmp_path / "overrides.jsonl",
        [_label("w0", 0, 0.0, 1.0, "definite_drop", manual=True)],
    )

    rows = effective_label_rows([base, override])
    labels, unmatched, _summary = project_label_rows(
        rows,
        [_candidate("w0", 0, 0.0, 1.0)],
    )

    assert not unmatched
    assert labels[0]["label"] == "definite_drop"
    assert labels[0]["projection"]["source_manual_audit"] is True
