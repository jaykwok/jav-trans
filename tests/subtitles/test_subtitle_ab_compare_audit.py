from __future__ import annotations

import json
from pathlib import Path

from tools.audits.generate_subtitle_ab_compare_audit_html import main


def _write_srt(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _run_generator(tmp_path: Path, *, media_mode: str) -> str:
    media = tmp_path / f"sample.{ 'wav' if media_mode == 'audio' else 'mp4' }"
    old_srt = _write_srt(
        tmp_path / "old.srt",
        "1\n00:00:01,000 --> 00:00:02,000\nbase\n\n",
    )
    new_srt = _write_srt(
        tmp_path / "new.srt",
        "1\n00:00:01,000 --> 00:00:02,500\ncandidate\n\n",
    )
    media.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    old_quality = _write_json(tmp_path / "old_quality.json", {})
    new_quality = _write_json(tmp_path / "new_quality.json", {})
    old_summary = _write_json(tmp_path / "old_summary.json", {"results": []})
    new_summary = _write_json(tmp_path / "new_summary.json", {"results": []})
    output_dir = tmp_path / "audit"

    rc = main(
        [
            "--output-dir",
            str(output_dir),
            "--media-mode",
            media_mode,
            "--case",
            "sample",
            "Sample",
            str(media),
            str(old_srt),
            str(new_srt),
            str(old_quality),
            str(new_quality),
            f"{old_summary}:{new_summary}",
        ]
    )

    assert rc == 0
    return (output_dir / "index.html").read_text(encoding="utf-8")


def test_audio_subtitle_ab_page_uses_independent_players_and_animation_frame_sync(tmp_path: Path):
    html = _run_generator(tmp_path, media_mode="audio")

    assert '<audio id="oldPlayer"' in html
    assert '<audio id="newPlayer"' in html
    assert "<video" not in html
    assert "window.requestAnimationFrame" in html
    assert "pauseOtherPlayers(player)" in html
    assert "const dualMode = Boolean(oldPlayer && newPlayer)" in html
    assert 'updateCurrentText(startTime, "both")' in html


def test_video_subtitle_ab_page_uses_independent_players(tmp_path: Path):
    html = _run_generator(tmp_path, media_mode="video")

    assert '<video id="oldPlayer"' in html
    assert '<video id="newPlayer"' in html
    assert 'id="video"' not in html
    assert "pauseOtherPlayers(player)" in html
    assert "定位当前窗口" in html
