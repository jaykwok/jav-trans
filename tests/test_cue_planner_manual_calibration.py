from __future__ import annotations

import json
from pathlib import Path

from tools.subtitles.calibrate_cue_planner_from_manual_audit import build_calibration


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_cue_planner_manual_calibration_summarizes_risk_tags(tmp_path: Path):
    manifest = _write_jsonl(
        tmp_path / "manifest.jsonl",
        [
            {
                "sample_id": "a",
                "risk_tags": "near_speaker_threshold,reading_density_high",
                "speaker_change_score": 0.92,
                "score": 0.66,
                "source_start_s": 0.0,
                "source_end_s": 1.0,
                "merged_ja": "あ あ",
            },
            {
                "sample_id": "b",
                "risk_tags": "high_speaker_score,loose_gap",
                "speaker_change_score": 0.91,
                "score": 0.52,
                "source_start_s": 2.0,
                "source_end_s": 3.0,
                "merged_ja": "だめ",
            },
            {
                "sample_id": "c",
                "risk_tags": "high_speaker_score,loose_gap",
                "speaker_change_score": 0.86,
                "score": 0.51,
                "source_start_s": 4.0,
                "source_end_s": 4.7,
                "merged_ja": "ん",
            },
        ],
    )
    manual = _write_jsonl(
        tmp_path / "manual.jsonl",
        [
            {"sample_id": "a", "manual_label": "keep_text", "reviewed": True},
            {"sample_id": "b", "manual_label": "needs_realign", "reviewed": True},
            {"sample_id": "c", "manual_label": "bad_asr", "reviewed": True},
        ],
    )

    summary = build_calibration(
        manual_labels_path=manual,
        manifest_path=manifest,
        output_dir=tmp_path / "out",
    )

    assert summary["reviewed_rows"] == 3
    assert summary["overall"]["label_counts"]["keep_text"] == 1
    assert summary["risk_tag_stats"]["high_speaker_score"]["problem_rate"] == 1.0
    assert summary["risk_tag_stats"]["loose_gap"]["problem_rate"] == 1.0
    assert summary["risk_tag_stats"]["near_speaker_threshold"]["keep"] == 1
    args = summary["recommendation"]["suggested_planner_args"]
    assert args["speaker_threshold"] == 0.95
    assert args["max_gap_s"] == 0.5
    assert args["speaker_score_penalty_threshold"] == 0.85
    assert (tmp_path / "out" / "summary.md").exists()


def test_cue_planner_manual_calibration_accepts_multi_labels(tmp_path: Path):
    manifest = _write_jsonl(
        tmp_path / "manifest.jsonl",
        [
            {
                "sample_id": "a",
                "risk_tags": "near_speaker_threshold",
                "speaker_change_score": 0.7,
                "score": 0.6,
                "source_start_s": 0.0,
                "source_end_s": 1.0,
                "merged_ja": "あ",
            },
            {
                "sample_id": "b",
                "risk_tags": "near_speaker_threshold",
                "speaker_change_score": 0.7,
                "score": 0.6,
                "source_start_s": 2.0,
                "source_end_s": 3.0,
                "merged_ja": "ん",
            },
        ],
    )
    manual = _write_jsonl(
        tmp_path / "manual.jsonl",
        [
            {
                "sample_id": "a",
                "manual_label": "keep_text",
                "manual_labels": ["keep_text", "timing_accurate"],
                "reviewed": True,
            },
            {
                "sample_id": "b",
                "manual_label": "needs_realign",
                "manual_labels": ["keep_text", "needs_realign"],
                "reviewed": True,
            },
        ],
    )

    summary = build_calibration(
        manual_labels_path=manual,
        manifest_path=manifest,
        output_dir=tmp_path / "out-multi",
    )

    assert summary["overall"]["label_counts"]["timing_accurate"] == 1
    assert summary["overall"]["label_counts"]["needs_realign"] == 1
    assert summary["overall"]["problem_bucket_counts"]["keep"] == 1
    assert summary["overall"]["problem_bucket_counts"]["merge_timing"] == 1


def test_cue_planner_manual_calibration_separates_low_info_vocal_from_hard_drop(tmp_path: Path):
    manifest = _write_jsonl(
        tmp_path / "manifest.jsonl",
        [
            {
                "sample_id": "a",
                "risk_tags": "reading_density_high",
                "speaker_change_score": 0.5,
                "score": 0.7,
                "source_start_s": 0.0,
                "source_end_s": 1.0,
                "merged_ja": "はぁ はぁ",
            },
            {
                "sample_id": "b",
                "risk_tags": "reading_density_high",
                "speaker_change_score": 0.5,
                "score": 0.7,
                "source_start_s": 2.0,
                "source_end_s": 3.0,
                "merged_ja": "",
            },
            {
                "sample_id": "c",
                "risk_tags": "reading_density_high",
                "speaker_change_score": 0.5,
                "score": 0.7,
                "source_start_s": 4.0,
                "source_end_s": 5.0,
                "merged_ja": "ん",
            },
        ],
    )
    manual = _write_jsonl(
        tmp_path / "manual.jsonl",
        [
            {
                "sample_id": "a",
                "manual_labels": ["keep_text", "timing_accurate", "low_info_vocal"],
                "reviewed": True,
            },
            {
                "sample_id": "b",
                "manual_labels": ["drop_non_speech"],
                "reviewed": True,
            },
            {
                "sample_id": "c",
                "manual_labels": ["low_info_vocal"],
                "reviewed": True,
            },
        ],
    )

    summary = build_calibration(
        manual_labels_path=manual,
        manifest_path=manifest,
        output_dir=tmp_path / "out-low-info",
    )

    assert summary["overall"]["label_counts"]["low_info_vocal"] == 2
    assert summary["overall"]["problem_bucket_counts"]["low_info_keep"] == 1
    assert summary["overall"]["problem_bucket_counts"]["low_info_review"] == 1
    assert summary["overall"]["problem_bucket_counts"]["asr_qc"] == 1
    assert summary["overall"]["keep"] == 1
    assert summary["overall"]["review_only"] == 1
    assert summary["overall"]["problem"] == 1
    assert summary["overall"]["problem_rate"] == 0.333333


def test_cue_planner_manual_calibration_tracks_side_specific_mixed_labels(tmp_path: Path):
    manifest = _write_jsonl(
        tmp_path / "manifest.jsonl",
        [
            {
                "sample_id": "a",
                "risk_tags": "near_speaker_threshold",
                "speaker_change_score": 0.88,
                "score": 0.62,
                "source_start_s": 0.0,
                "source_end_s": 2.0,
                "left_ja": "誤認識",
                "right_ja": "いいよ",
                "merged_ja": "誤認識 いいよ",
            },
            {
                "sample_id": "b",
                "risk_tags": "reading_density_high",
                "speaker_change_score": 0.5,
                "score": 0.7,
                "source_start_s": 3.0,
                "source_end_s": 4.0,
                "left_ja": "はぁ",
                "right_ja": "",
                "merged_ja": "はぁ",
            },
        ],
    )
    manual = _write_jsonl(
        tmp_path / "manual.jsonl",
        [
            {
                "sample_id": "a",
                "manual_labels": ["needs_split", "left_bad_asr", "right_keep_text", "timing_accurate"],
                "reviewed": True,
            },
            {
                "sample_id": "b",
                "manual_labels": ["left_keep_text", "right_drop_non_speech"],
                "reviewed": True,
            },
        ],
    )

    summary = build_calibration(
        manual_labels_path=manual,
        manifest_path=manifest,
        output_dir=tmp_path / "out-side",
    )

    assert summary["overall"]["problem_bucket_counts"]["side_mixed"] == 2
    assert "merge_timing" not in summary["overall"]["problem_bucket_counts"]
    assert summary["overall"]["side_label_counts"]["left"]["bad_asr"] == 1
    assert summary["overall"]["side_label_counts"]["left"]["keep_text"] == 1
    assert summary["overall"]["side_label_counts"]["right"]["keep_text"] == 1
    assert summary["overall"]["side_label_counts"]["right"]["drop_non_speech"] == 1
