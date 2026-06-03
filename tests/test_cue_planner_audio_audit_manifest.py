from __future__ import annotations

import json
from pathlib import Path

from tools.subtitles.export_cue_planner_audio_audit_manifest import export_manifest


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_export_cue_planner_audio_audit_manifest(tmp_path: Path):
    review = _write_jsonl(
        tmp_path / "review.jsonl",
        [
            {
                "timeline_start_s": 1.25,
                "timeline_end_s": 2.5,
                "timeline_start": "00:00:01.250",
                "timeline_end": "00:00:02.500",
                "left_index": 3,
                "left_ja": "あ",
                "right_ja": "ん",
                "merged_ja": "あ ん",
                "risk_tags": "reading_density_high",
                "review_priority": 20,
                "speaker_change_score": 0.7,
                "score": 0.6,
            }
        ],
    )
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"fake")

    summary = export_manifest(
        review_items_path=review,
        source_audio_path=source_audio,
        output_dir=tmp_path / "out",
        dataset_id="cue-test",
        max_rows=None,
    )

    assert summary["rows"] == 1
    row = json.loads((tmp_path / "out" / "cue_planner_audio_audit_manifest.jsonl").read_text(encoding="utf-8"))
    assert row["sample_id"].startswith("cue-test-01")
    assert row["start"] == 1.25
    assert row["end"] == 2.5
    assert "merged: あ ん" in row["display_text"]
