from __future__ import annotations

import json
from pathlib import Path

from tools.fusionvad_ja.export_alignment_failure_manifest import main, manifest_row


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

    assert len(rows) == 2
    assert {row["review_type"] for row in rows} == {"review_alignment_text", "review_asr_text"}
    assert summary["failure_bucket_counts"] == {
        "align_text_empty": 1,
        "asr_dropped_uncertain": 1,
    }
