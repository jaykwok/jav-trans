from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from tools.audits.generate_boundary_preference_audit_html import generate_audit
from tools.boundary.boundary_preference import (
    AXES,
    CANDIDATE_SCHEMA,
    aligned_offset_frames,
    build_blind_items,
    category_quotas,
    perturbation_category,
    select_balanced_candidates,
    summarize_preferences,
    write_jsonl,
)
from tools.boundary.compile_boundary_preferences import compile_rows


VIDEO_IDS = ["short", "medium", "long"]


def _result(text: str, *, qc: str = "ok", fallback: str = "none") -> dict:
    return {
        "text": text,
        "raw_text": text,
        "asr_qc_severity": qc,
        "asr_qc_reasons": [],
        "alignment_quality": "forced" if fallback == "none" else "proportional",
        "fallback_type": "none" if fallback == "none" else "coarse",
        "fallback_subtype": fallback,
        "sentinel": fallback == "proportional_after_sentinel",
        "nonlexical_text": False,
        "repeat_profile": {},
        "cue_density_cps": 3.0,
    }


def _candidate(
    *,
    video_id: str,
    index: int,
    axis: str,
    offset_frames: int,
    risk_bucket: str = "stable_control",
) -> dict:
    offset_s = offset_frames * 0.02
    left_end = 10.0 + index
    right_start = left_end + 0.2
    baseline_interval = (
        {"start": right_start, "end": right_start + 1.2}
        if axis == "right.start"
        else {"start": left_end - 1.2, "end": left_end}
    )
    challenger_interval = dict(baseline_interval)
    if axis == "right.start":
        challenger_interval["start"] += offset_s
    else:
        challenger_interval["end"] += offset_s
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_id": f"{video_id}-{axis}-{offset_frames}-{index}",
        "video_id": video_id,
        "video_label": video_id.title(),
        "media_path": f"video/{video_id}.mp4",
        "boundary_index": index,
        "boundary_time_s": left_end + 0.1,
        "context_start": left_end - 2.0,
        "context_end": right_start + 2.0,
        "axis": axis,
        "offset_frames": offset_frames,
        "offset_ms": int(offset_s * 1000),
        "perturbation_category": perturbation_category(axis, offset_frames),
        "risk_bucket": risk_bucket,
        "selection_score": float(index % 11),
        "baseline_interval": baseline_interval,
        "challenger_interval": challenger_interval,
        "baseline_boundary": {
            "left_end_s": left_end,
            "right_start_s": right_start,
        },
        "challenger_boundary": {
            "left_end_s": left_end + (offset_s if axis == "left.end" else 0.0),
            "right_start_s": right_start + (offset_s if axis == "right.start" else 0.0),
        },
        "raw_left_end_s": left_end - 0.03,
        "raw_right_start_s": right_start + 0.02,
        "feature_schema": "frame_sequence_features_v1",
        "feature_schema_hash": "abc123",
        "feature_signature": {"feature_schema": "frame_sequence_features_v1"},
        "feature_names": ["gap_s", "left_duration_s"],
        "sequence_feature": [0.2, 1.2],
        "baseline_result": _result("baseline"),
        "challenger_result": _result("challenger"),
    }


def _candidate_pool() -> list[dict]:
    rows = []
    risk_buckets = [
        "text_changed",
        "qc_alignment_change",
        "repeat_nonlexical",
        "cue_density",
        "gap_crossing",
        "stable_control",
    ]
    for video_id in VIDEO_IDS:
        index = 0
        for axis in AXES:
            for offset_frames in (-8, -4, 4, 8):
                for _ in range(8):
                    rows.append(
                        _candidate(
                            video_id=video_id,
                            index=index,
                            axis=axis,
                            offset_frames=offset_frames,
                            risk_bucket=risk_buckets[index % len(risk_buckets)],
                        )
                    )
                    index += 1
    return rows


def _label_for_identity(answer: dict, identity: str) -> str:
    if answer["a_identity"] == identity:
        return "a_better"
    if answer["b_identity"] == identity:
        return "b_better"
    raise AssertionError(identity)


def _passing_labels(answers: list[dict]) -> list[dict]:
    answers_by_id = {row["item_id"]: row for row in answers}
    unique_answers = [row for row in answers if not row["is_hidden_duplicate"]]
    challenger_ids: set[str] = set()
    first_by_category: dict[str, str] = {}
    for answer in unique_answers:
        category = answer["perturbation_category"]
        first_by_category.setdefault(category, answer["item_id"])
    challenger_ids.update(first_by_category.values())
    for answer in unique_answers:
        if len(challenger_ids) >= 30:
            break
        challenger_ids.add(answer["item_id"])

    labels_by_id: dict[str, dict] = {}
    for answer in unique_answers:
        identity = "challenger" if answer["item_id"] in challenger_ids else "baseline"
        labels_by_id[answer["item_id"]] = {
            "item_id": answer["item_id"],
            "primary_label": _label_for_identity(answer, identity),
        }
    for answer in answers:
        if not answer["is_hidden_duplicate"]:
            continue
        canonical_answer = answers_by_id[answer["canonical_item_id"]]
        canonical_label = labels_by_id[answer["canonical_item_id"]]
        canonical_identity = (
            canonical_answer["a_identity"]
            if canonical_label["primary_label"] == "a_better"
            else canonical_answer["b_identity"]
        )
        labels_by_id[answer["item_id"]] = {
            "item_id": answer["item_id"],
            "primary_label": _label_for_identity(answer, canonical_identity),
        }
    return list(labels_by_id.values())


def test_offsets_are_exact_feature_frames():
    assert aligned_offset_frames(-160, 0.02) == -8
    assert aligned_offset_frames(-80, 0.02) == -4
    assert aligned_offset_frames(80, 0.02) == 4
    assert aligned_offset_frames(160, 0.02) == 8
    with pytest.raises(ValueError, match="not aligned"):
        aligned_offset_frames(80, 0.03)


def test_balanced_selection_produces_108_unique_rows():
    selected = select_balanced_candidates(
        _candidate_pool(),
        video_ids=VIDEO_IDS,
        per_video=36,
    )

    assert len(selected) == 108
    assert len({row["candidate_id"] for row in selected}) == 108
    assert Counter(row["video_id"] for row in selected) == {
        "short": 36,
        "medium": 36,
        "long": 36,
    }
    for video_id in VIDEO_IDS:
        video_rows = [row for row in selected if row["video_id"] == video_id]
        assert Counter(row["axis"] for row in video_rows) == {
            "right.start": 18,
            "left.end": 18,
        }
        assert sum(category_quotas(video_ids=VIDEO_IDS)[video_id].values()) == 36


def test_blinding_builds_12_independently_randomized_hidden_duplicates():
    selected = select_balanced_candidates(
        _candidate_pool(),
        video_ids=VIDEO_IDS,
        per_video=36,
    )
    blind, answers = build_blind_items(selected, hidden_duplicate_count=12, seed=7)

    assert len(blind) == 120
    assert len(answers) == 120
    assert sum(row["is_hidden_duplicate"] for row in answers) == 12
    assert all("baseline" not in row and "challenger" not in row for row in blind)
    assert all(set((row["a_identity"], row["b_identity"])) == {"baseline", "challenger"} for row in answers)
    duplicate_video_counts = Counter(
        row["video_id"] for row in answers if row["is_hidden_duplicate"]
    )
    assert duplicate_video_counts == {"short": 4, "medium": 4, "long": 4}
    for video_id in VIDEO_IDS:
        categories = {
            row["perturbation_category"]
            for row in answers
            if row["is_hidden_duplicate"] and row["video_id"] == video_id
        }
        assert len(categories) == 4


def test_summary_gate_uses_hidden_consistency_and_unique_challenger_wins():
    selected = select_balanced_candidates(
        _candidate_pool(),
        video_ids=VIDEO_IDS,
        per_video=36,
    )
    _, answers = build_blind_items(selected, hidden_duplicate_count=12, seed=11)
    labels = _passing_labels(answers)

    summary = summarize_preferences(labels, answers)

    assert summary["gate_passed"] is True
    assert summary["hidden_duplicate_consistency"] == 1.0
    assert summary["usable_label_count"] == 120
    assert summary["decisive_ratio"] == 1.0
    assert summary["challenger_wins"] == 30
    assert summary["challenger_win_category_coverage"] > 3


def test_summary_gate_stays_closed_without_human_labels():
    selected = select_balanced_candidates(
        _candidate_pool(),
        video_ids=VIDEO_IDS,
        per_video=36,
    )
    _, answers = build_blind_items(selected, hidden_duplicate_count=12, seed=15)

    summary = summarize_preferences([], answers)

    assert summary["usable_label_count"] == 0
    assert summary["hidden_duplicate_consistency"] == 0.0
    assert summary["challenger_wins"] == 0
    assert summary["gate_passed"] is False


def test_summary_gate_uses_strict_challenger_win_threshold():
    selected = select_balanced_candidates(
        _candidate_pool(),
        video_ids=VIDEO_IDS,
        per_video=36,
    )
    _, answers = build_blind_items(selected, hidden_duplicate_count=12, seed=17)
    labels = _passing_labels(answers)
    answers_by_id = {row["item_id"]: row for row in answers}
    changed = 0
    for label in labels:
        answer = answers_by_id[label["item_id"]]
        if answer["is_hidden_duplicate"]:
            continue
        preferred = (
            answer["a_identity"]
            if label["primary_label"] == "a_better"
            else answer["b_identity"]
        )
        if preferred != "challenger" or changed >= 5:
            continue
        label["primary_label"] = _label_for_identity(answer, "baseline")
        for duplicate in answers:
            if duplicate["canonical_item_id"] != answer["item_id"]:
                continue
            duplicate_label = next(
                row for row in labels if row["item_id"] == duplicate["item_id"]
            )
            duplicate_label["primary_label"] = _label_for_identity(duplicate, "baseline")
        changed += 1

    summary = summarize_preferences(labels, answers)

    assert summary["challenger_wins"] == 25
    assert summary["gate_checks"]["challenger_wins_gt_25"] is False
    assert summary["gate_passed"] is False


def test_compiler_supervises_only_compared_axis():
    selected = select_balanced_candidates(
        _candidate_pool(),
        video_ids=VIDEO_IDS,
        per_video=36,
    )
    _, answers = build_blind_items(selected, hidden_duplicate_count=12, seed=13)
    labels = _passing_labels(answers)

    rows, summary = compile_rows(
        labels=labels,
        answers=answers,
        candidates=selected,
    )

    assert summary["gate"]["gate_passed"] is True
    assert len(rows) == 108
    assert all(row["schema"] == "boundary_refiner_frame_sequence_dataset_v5" for row in rows)
    for row in rows:
        weights = row["sequence_boundary_delta_weights"][0]
        reason = row["sequence_reasons"][0]
        if "right.start" in reason:
            assert weights == [1.0, 0.0]
            assert row["sequence_boundary_delta_targets"][0][1] == 0.0
        else:
            assert weights == [0.0, 0.6]
            assert row["sequence_boundary_delta_targets"][0][0] == 0.0


def test_audit_page_uses_local_storage_and_jsonl_export(tmp_path: Path):
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"video")
    candidate = _candidate(
        video_id="short",
        index=1,
        axis="right.start",
        offset_frames=4,
    )
    candidate["media_path"] = str(media)
    blind, _ = build_blind_items([candidate], hidden_duplicate_count=0, seed=3)
    manifest = tmp_path / "blind.jsonl"
    write_jsonl(manifest, blind)
    output_dir = tmp_path / "audit"

    summary = generate_audit(
        blind_manifest=manifest,
        output_dir=output_dir,
        title="Preference Audit",
        dataset_id="test-preference",
        update_entrypoints=False,
    )
    page = (output_dir / "index.html").read_text(encoding="utf-8")

    assert summary["review_item_count"] == 1
    assert "localStorage" in page
    assert "manual_boundary_preference_labels.jsonl" in page
    assert "A better" in page
    assert "both bad" in page
    assert "A start 更准" in page
    assert "const PLAYBACK_PREROLL_S = 1.5" in page
    assert "const PLAYBACK_POSTROLL_S = 1.5" in page
    assert "side.start - PLAYBACK_PREROLL_S" in page
    assert "side.end + PLAYBACK_POSTROLL_S" in page
    assert "media.currentTime >= side.start && media.currentTime < side.end" in page
    assert '"a_identity"' not in page
    audit_summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert audit_summary["label_schema"] == "boundary_preference_label_v1"
    assert audit_summary["playback_preroll_s"] == 1.5
    assert audit_summary["playback_postroll_s"] == 1.5
