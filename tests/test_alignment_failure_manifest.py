from __future__ import annotations

import json
from pathlib import Path

from tools.fusionvad_ja.export_alignment_failure_manifest import main, manifest_row
from tools.fusionvad_ja.select_alignment_failure_audit_subset import main as subset_main


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_manifest_row_preserves_source_and_review_type():
    row = manifest_row(
        {
            "case_label": "full15000",
            "video": "sample",
            "source_audio_path": "temp/audio.wav",
            "aligned_path": "sample.aligned_segments.json",
            "chunk_index": 7,
            "position": 3,
            "start": 1.25,
            "end": 2.75,
            "duration_s": 1.5,
            "failure_bucket": "vad_coarse_alignment",
            "failure_reasons": ["alignment_fallback"],
            "alignment_quality": "vad_coarse",
            "fallback_type": "vad_coarse",
            "display_text": "  ああ\n  ",
        }
    )

    assert row["sample_id"] == "full15000__sample__chunk0007__vad_coarse_alignment"
    assert row["source_audio_path"] == "temp/audio.wav"
    assert row["review_type"] == "review_coarse_timing"
    assert row["display_text"] == "ああ"
    assert row["manual_label"] == ""


def test_manifest_row_exports_repetition_and_low_information_fields():
    repeat_row = manifest_row(
        {
            "case_label": "full",
            "video": "sample",
            "source_audio_path": "audio.wav",
            "chunk_index": 2,
            "start": 1.0,
            "end": 3.0,
            "duration_s": 2.0,
            "failure_bucket": "repeat_repair_suggested",
            "alignment_quality": "drop_or_review",
            "fallback_type": "none",
            "display_text": "あっ、あっ、あっ、あっ",
            "repetition_suggested_text": "あっあっあっ",
            "repetition_repair": {"action": "truncate_repetition", "changed": True},
        }
    )
    low_info_row = manifest_row(
        {
            "case_label": "full",
            "video": "sample",
            "source_audio_path": "audio.wav",
            "chunk_index": 3,
            "start": 3.0,
            "end": 5.0,
            "duration_s": 2.0,
            "failure_bucket": "low_information_text",
            "alignment_quality": "forced",
            "fallback_type": "none",
            "display_text": "んんんん",
            "low_information_level": "repeated_nonlexical",
            "low_information": {"level": "repeated_nonlexical"},
        }
    )

    assert repeat_row["review_type"] == "review_repetition_repair"
    assert repeat_row["repetition_suggested_text"] == "あっあっあっ"
    assert repeat_row["repetition_repair"] == {
        "action": "truncate_repetition",
        "changed": True,
    }
    assert low_info_row["review_type"] == "review_low_information_text"
    assert low_info_row["low_information_level"] == "repeated_nonlexical"


def test_export_alignment_failure_manifest_cli(tmp_path):
    candidates = tmp_path / "failure_candidates.jsonl"
    _write_jsonl(
        candidates,
        [
            {
                "case_label": "base",
                "video": "sample",
                "source_audio_path": "audio.wav",
                "chunk_index": 0,
                "start": 0.0,
                "end": 1.0,
                "duration_s": 1.0,
                "failure_bucket": "align_text_empty",
                "alignment_quality": "drop_or_review",
                "fallback_type": "proportional",
                "display_text": "~~~",
            },
            {
                "case_label": "full",
                "video": "sample",
                "source_audio_path": "audio.wav",
                "chunk_index": 1,
                "start": 1.0,
                "end": 2.0,
                "duration_s": 1.0,
                "failure_bucket": "asr_dropped_uncertain",
                "alignment_quality": "drop_or_review",
                "fallback_type": "none",
                "display_text": "んー",
            },
            {
                "case_label": "full",
                "video": "sample",
                "source_audio_path": "audio.wav",
                "chunk_index": 2,
                "start": 2.0,
                "end": 3.0,
                "duration_s": 1.0,
                "failure_bucket": "repeat_repair_suggested",
                "alignment_quality": "drop_or_review",
                "fallback_type": "none",
                "display_text": "あっ、あっ、あっ、あっ",
                "repetition_suggested_text": "あっあっあっ",
            },
        ],
    )
    output_dir = tmp_path / "out"

    assert main(["--failure-candidates", str(candidates), "--output-dir", str(output_dir)]) == 0

    rows = [
        json.loads(line)
        for line in (output_dir / "alignment_failure_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((output_dir / "alignment_failure_manifest_summary.json").read_text(encoding="utf-8"))

    assert len(rows) == 3
    assert {row["review_type"] for row in rows} == {
        "review_alignment_text",
        "review_asr_text",
        "review_repetition_repair",
    }
    assert summary["failure_bucket_counts"] == {
        "align_text_empty": 1,
        "asr_dropped_uncertain": 1,
        "repeat_repair_suggested": 1,
    }


def test_select_alignment_failure_audit_subset_cli(tmp_path):
    manifest = tmp_path / "alignment_failure_manifest.jsonl"
    rows = [
        {
            "sample_id": "repeat-0",
            "case_label": "full",
            "video": "sample",
            "chunk_index": 0,
            "start": 0.0,
            "review_type": "review_repetition_repair",
            "failure_bucket": "repeat_repair_suggested",
        },
        {
            "sample_id": "low-1",
            "case_label": "full",
            "video": "sample",
            "chunk_index": 1,
            "start": 1.0,
            "review_type": "review_low_information_text",
            "failure_bucket": "long_low_information_text",
        },
        {
            "sample_id": "coarse-2",
            "case_label": "full",
            "video": "sample",
            "chunk_index": 2,
            "start": 2.0,
            "review_type": "review_coarse_timing",
            "failure_bucket": "vad_coarse_alignment",
        },
        {
            "sample_id": "coarse-3",
            "case_label": "full",
            "video": "sample",
            "chunk_index": 3,
            "start": 3.0,
            "review_type": "review_coarse_timing",
            "failure_bucket": "vad_coarse_alignment",
        },
        {
            "sample_id": "coarse-4",
            "case_label": "full",
            "video": "sample",
            "chunk_index": 4,
            "start": 4.0,
            "review_type": "review_coarse_timing",
            "failure_bucket": "vad_coarse_alignment",
        },
    ]
    _write_jsonl(manifest, rows)
    output_dir = tmp_path / "subset"

    assert (
        subset_main(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output_dir),
                "--sample-review-type",
                "review_coarse_timing:2",
            ]
        )
        == 0
    )

    selected = [
        json.loads(line)
        for line in (output_dir / "alignment_failure_audit_subset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((output_dir / "alignment_failure_audit_subset_summary.json").read_text(encoding="utf-8"))

    assert [row["sample_id"] for row in selected] == ["repeat-0", "low-1", "coarse-2", "coarse-4"]
    assert summary["review_type_counts"] == {
        "review_coarse_timing": 2,
        "review_low_information_text": 1,
        "review_repetition_repair": 1,
    }
