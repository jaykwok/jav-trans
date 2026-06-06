from __future__ import annotations

import json
from pathlib import Path

from tools.audits.generate_long_fallback_chunk_audit_html import PROJECT_ROOT
from tools.audits.generate_long_fallback_chunk_audit_html import main


def test_long_fallback_audit_audio_mode_uses_audio_element(tmp_path: Path):
    long_chunks = tmp_path / "unsafe_fallback_chunks.jsonl"
    subtitle = tmp_path / "sample.ja.srt"
    source_audio = tmp_path / "sample.wav"
    output_dir = tmp_path / "audit"

    long_chunks.write_text(
        json.dumps(
            {
                "chunk_index": 7,
                "start": 1.0,
                "end": 5.0,
                "core_start": 1.2,
                "core_end": 4.8,
                "fallback_window_start": 1.5,
                "fallback_window_end": 4.5,
                "fallback_duration_s": 3.0,
                "duration_s": 4.0,
                "fallback_subtype": "proportional",
                "split_reason": "dp",
                "risk_reasons": ["review"],
                "display_text": "ありがとう",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    subtitle.write_text(
        "1\n00:00:01,500 --> 00:00:04,500\nありがとう\n\n",
        encoding="utf-8",
    )
    source_audio.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")

    rc = main(
        [
            "--media-mode",
            "audio",
            "--long-chunks",
            str(long_chunks),
            "--subtitle-srt",
            str(subtitle),
            "--video",
            str(source_audio),
            "--output-dir",
            str(output_dir),
            "--no-update-entrypoints",
        ]
    )

    assert rc == 0
    html = (output_dir / "index.html").read_text(encoding="utf-8")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert '<audio id="media"' in html
    assert '<video id="media"' not in html
    assert "直接打开音频" in html
    assert '"timing_accurate"' in html
    assert '"timing_ok"' not in html
    assert "时间轴准确" in html
    assert "时间轴需重对齐" in html
    assert "开头偏早" in html
    assert "开头偏晚" in html
    assert "结尾偏早" in html
    assert "结尾偏晚" in html
    assert "时间窗过长" in html
    assert "时间窗过短" in html
    assert "跨无声/噪声" in html
    assert summary["media_mode"] == "audio"
    assert summary["media_mime"] == "audio/wav"
    assert summary["media_path"] == source_audio.relative_to(PROJECT_ROOT).as_posix()
