from __future__ import annotations

import json
from pathlib import Path

from tools.audits.generate_asr_attribution_audit_html import build_review_items
from tools.audits.generate_asr_attribution_audit_html import main


def _row(chunk_index: int, **overrides):
    row = {
        "chunk_index": chunk_index,
        "start": float(chunk_index * 3),
        "end": float(chunk_index * 3 + 2),
        "duration_s": 2.0,
        "fallback_window_start": float(chunk_index * 3),
        "fallback_window_end": float(chunk_index * 3 + 2),
        "fallback_duration_s": 2.0,
        "alignment_quality": "forced",
        "alignment_mode": "forced_aligner",
        "fallback_type": "none",
        "fallback_subtype": "none",
        "asr_qc_severity": "ok",
        "asr_qc_reasons": [],
        "failure_bucket": "",
        "failure_reasons": [],
        "text_density_level": "normal_dialogue",
        "text_density": {"level": "normal_dialogue"},
        "repetition_repair": {"changed": False},
        "display_text": f"テスト{chunk_index}",
        "text": f"テスト{chunk_index}",
        "video": "sample",
    }
    row.update(overrides)
    return row


def test_build_review_items_balances_attribution_buckets():
    diagnostics = [
        _row(
            0,
            asr_qc_severity="reject",
            asr_qc_reasons=["repeat_ngram_loop"],
            failure_bucket="repeat_repair_suggested",
            repetition_repair={"changed": True, "run": 4, "suggested_text": "だめだめだめ"},
        ),
        _row(
            1,
            alignment_quality="nonlexical",
            fallback_type="proportional",
            fallback_subtype="nonlexical_text",
            align_text_empty=True,
            text_density_level="empty_or_punctuation",
        ),
        _row(
            2,
            alignment_quality="proportional",
            fallback_type="proportional",
            fallback_subtype="proportional_after_sentinel",
        ),
        _row(3, text_density_level="short_vocalization_candidate"),
        _row(4, asr_qc_severity="warn", asr_qc_reasons=["empty_text_for_speech_chunk"]),
        _row(5),
    ]
    cues = [
        {"index": i, "start": float(i * 3), "end": float(i * 3 + 2), "text": f"字幕{i}"}
        for i in range(6)
    ]

    items = build_review_items(
        diagnostics_rows=diagnostics,
        aligned_segments_by_chunk={2: [{"start": 6.0, "end": 8.0, "text": "aligned"}]},
        subtitle_cues=cues,
        context_pad_s=0.5,
        max_per_bucket=1,
        max_items=None,
        video_label="sample",
    )

    assert [item["bucket"] for item in items] == [
        "repeat_or_qc_reject",
        "nonlexical_empty",
        "sentinel_fallback",
        "low_info_vocal",
        "asr_qc_warn",
        "forced_control",
    ]
    by_bucket = {item["bucket"]: item for item in items}
    assert "repeat_repair_suggested" in by_bucket["repeat_or_qc_reject"]["reason_hints"]
    assert by_bucket["sentinel_fallback"]["aligned_segments"][0]["text"] == "aligned"
    assert by_bucket["low_info_vocal"]["subtitle_text"] == "字幕3"


def test_asr_attribution_audit_generator_writes_audio_page(tmp_path: Path):
    diagnostics = tmp_path / "diagnostics.jsonl"
    aligned = tmp_path / "aligned.json"
    subtitle = tmp_path / "sample.ja.srt"
    source_audio = tmp_path / "sample.wav"
    output_dir = tmp_path / "audit"

    rows = [
        _row(
            0,
            asr_qc_severity="reject",
            asr_qc_reasons=["repeat_ngram_loop"],
            failure_bucket="repeat_repair_suggested",
            repetition_repair={"changed": True, "run": 4, "suggested_text": "だめだめだめ"},
        ),
        _row(1, fallback_subtype="proportional_after_sentinel", fallback_type="proportional"),
    ]
    diagnostics.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    aligned.write_text(
        json.dumps(
            {
                "segments": [
                    {"source_chunk_index": 0, "start": 0.0, "end": 2.0, "text": "テスト0"},
                    {"source_chunk_index": 1, "start": 3.0, "end": 5.0, "text": "テスト1"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nテスト0\n\n"
        "2\n00:00:03,000 --> 00:00:05,000\nテスト1\n\n",
        encoding="utf-8",
    )
    source_audio.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")

    rc = main(
        [
            "--diagnostics",
            str(diagnostics),
            "--aligned",
            str(aligned),
            "--subtitle-srt",
            str(subtitle),
            "--media",
            str(source_audio),
            "--media-mode",
            "audio",
            "--output-dir",
            str(output_dir),
            "--max-per-bucket",
            "2",
            "--no-update-entrypoints",
        ]
    )

    assert rc == 0
    html = (output_dir / "index.html").read_text(encoding="utf-8")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = (output_dir / "asr_attribution_review_items.jsonl").read_text(encoding="utf-8")
    assert '<audio id="media"' in html
    assert "真实低信息人声" in html
    assert "ASR 幻觉/错听" in html
    assert "manual_asr_attribution_labels.jsonl" in html
    assert summary["label_schema_version"] == "asr_attribution_v1"
    assert summary["media_mime"] == "audio/wav"
    assert summary["bucket_counts"]["repeat_or_qc_reject"] == 1
    assert "repeat_or_qc_reject" in manifest
