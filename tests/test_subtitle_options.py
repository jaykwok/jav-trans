import pytest

from subtitles.options import SubtitleOptions
from subtitles.options import FALLBACK_VIDEO_FPS
from subtitles import writer as subtitle


def test_subtitle_options_from_env(monkeypatch):
    monkeypatch.setenv("MAX_SUBTITLE_DURATION", "9.5")
    monkeypatch.setenv("SUBTITLE_SOFT_MAX_S", "7.5")
    monkeypatch.setenv("SUBTITLE_SOFT_SPLIT_ENABLED", "0")
    monkeypatch.setenv("SRT_LINE_MAX_CHARS", "18")
    monkeypatch.setenv("SUBTITLE_TIMELINE_MODE", "reading")
    monkeypatch.setenv("SUBTITLE_READING_CPS", "10")
    monkeypatch.setenv("SUBTITLE_MERGE_ADJACENT", "0")
    monkeypatch.setenv("SUBTITLE_SHOW_SPEAKER", "1")

    options = SubtitleOptions.from_env()

    assert options.max_duration == 9.5
    assert options.soft_max == 7.5
    assert options.soft_split_enabled is False
    assert options.timeline_mode == "reading"
    assert options.reading_cps == 10
    assert options.merge_adjacent is False
    assert options.line_max_chars == 18
    assert options.show_speaker is True


def test_subtitle_options_defaults_are_conservative(monkeypatch):
    monkeypatch.delenv("MAX_SUBTITLE_DURATION", raising=False)
    monkeypatch.delenv("SUBTITLE_SOFT_MAX_S", raising=False)

    options = SubtitleOptions.from_env()

    assert options.max_duration == 6.5
    assert options.soft_max == 5.5
    assert options.effective_video_fps == FALLBACK_VIDEO_FPS
    assert options.frame_gap_s == 2 / FALLBACK_VIDEO_FPS


def test_subtitle_options_video_fps_falls_back_to_ntsc():
    options = SubtitleOptions(video_fps=0)

    assert options.effective_video_fps == FALLBACK_VIDEO_FPS
    assert options.with_video_fps(None).effective_video_fps == FALLBACK_VIDEO_FPS
    assert options.with_video_fps(24).frame_gap_s == pytest.approx(2 / 24)


def test_write_srt_explicit_options_override_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBTITLE_SHOW_SPEAKER", "0")
    path = tmp_path / "speaker.srt"

    subtitle.write_srt(
        [{"start": 0.0, "end": 1.0, "zh_text": "过来", "speaker": "S0", "gender": "M"}],
        str(path),
        options=SubtitleOptions(show_speaker=True),
    )

    content = path.read_text(encoding="utf-8")
    assert "[S0] 过来" in content
    assert "[SS0]" not in content
    assert "[M]" not in content


def test_prepare_srt_blocks_timeline_mode_is_per_options(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBTITLE_TIMELINE_MODE", "alignment")
    path = tmp_path / "reading.srt"
    options = SubtitleOptions(
        timeline_mode="reading",
        reading_cps=20.0,
        reading_base=0.0,
        min_duration=0.6,
        duration_ratio_cap=1.0,
        duration_grace=0.0,
        max_duration=8.0,
    )
    blocks = subtitle.prepare_srt_blocks(
        [{"start": 0.0, "end": 10.0, "zh_text": "短句"}],
        options=options,
    )

    subtitle.write_srt(
        blocks,
        str(path),
        options=options,
    )

    content = path.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:00,600" in content


def test_prepare_bilingual_merge_adjacent_is_per_options(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBTITLE_MERGE_ADJACENT", "1")
    path = tmp_path / "no-merge.srt"
    options = SubtitleOptions(merge_adjacent=False)
    blocks = subtitle.prepare_srt_blocks(
        [
            {"start": 0.0, "end": 1.0, "ja_text": "あ", "zh_text": "甲"},
            {"start": 1.05, "end": 2.0, "ja_text": "い", "zh_text": "乙"},
        ],
        options=options,
        mode="bilingual",
    )

    subtitle.write_bilingual_srt(
        blocks,
        str(path),
        options=options,
    )

    content = path.read_text(encoding="utf-8")
    assert "\n1\n" not in content
    assert content.count("-->") == 2
