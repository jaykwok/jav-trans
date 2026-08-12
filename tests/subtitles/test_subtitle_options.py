import pytest

from subtitles.options import SubtitleOptions
from subtitles.options import BASE_FPS
from subtitles import writer as subtitle


def test_subtitle_options_from_env(monkeypatch):
    monkeypatch.setenv("SRT_LINE_MAX_CHARS", "18")
    monkeypatch.setenv("SUBTITLE_TIMELINE_MODE", "reading")
    monkeypatch.setenv("SUBTITLE_READING_CPS", "10")
    monkeypatch.setenv("SUBTITLE_TIMING_POLISH_ENABLED", "0")
    monkeypatch.setenv("SUBTITLE_SHORT_GAP_COLLAPSE_S", "0.4")
    monkeypatch.setenv("SUBTITLE_LINGER_S", "0.3")
    monkeypatch.setenv("SUBTITLE_MAX_SOURCE_CHARS", "19")
    monkeypatch.setenv("SUBTITLE_MAX_DISPLAY_DURATION_S", "6.5")

    options = SubtitleOptions.from_env()

    assert options.timeline_mode == "reading"
    assert options.reading_cps == 10
    assert options.line_max_chars == 18
    assert options.timing_polish_enabled is False
    assert options.short_gap_collapse_s == 0.4
    assert options.linger_s == 0.3
    assert options.max_source_chars == 19
    assert options.max_display_duration_s == 6.5


def test_subtitle_options_defaults_are_conservative():
    options = SubtitleOptions.from_env()

    assert options.max_display_duration_s == 7.0
    assert options.max_source_chars == 20
    assert options.layout_engine == "measured_safe_boundary_dp_v3_1"
    assert options.timing_model == "measured_lexical_extent_v3"
    assert options.frame_duration_s == pytest.approx(1 / BASE_FPS)
    assert options.frame_gap_s == pytest.approx(2 / BASE_FPS)
    assert options.frame_min_duration_s == pytest.approx(20 / BASE_FPS)
    assert options.timing_polish_enabled is True
    assert options.short_gap_collapse_s == 0.5
    assert options.linger_s == 0.5
    assert options.line_max_chars == 16
    assert options.max_display_shift_from_acoustic_end_s == 0.5


def test_an_unknown_layout_engine_is_refused_instead_of_relabelled(monkeypatch):
    """The name stamps the output, it does not select an implementation.

    While two layouts coexisted, the retired name silently produced current
    output labelled with the old engine. A rollback that cannot happen has to
    say so, not hand back cues that lie about which layout made them.
    """
    monkeypatch.setenv("SUBTITLE_LAYOUT_ENGINE", "anchor_aware_dp_v2")

    with pytest.raises(ValueError, match="SUBTITLE_LAYOUT_ENGINE"):
        SubtitleOptions.from_env()


def test_the_previous_layout_stamp_is_refused_like_any_other(monkeypatch):
    """`v3` and `v3_1` differ in where ~1.4% of cuts land.

    Accepting the older name would stamp v3_1 cues as v3 output, which is the
    exact failure the refusal exists for - and this is the name most likely to
    be tried, because it was the default until this build.
    """
    monkeypatch.setenv("SUBTITLE_LAYOUT_ENGINE", "measured_safe_boundary_dp_v3")

    with pytest.raises(ValueError, match="SUBTITLE_LAYOUT_ENGINE"):
        SubtitleOptions.from_env()


def test_an_unknown_timing_model_is_refused(monkeypatch):
    monkeypatch.setenv("SUBTITLE_TIMING_MODEL", "acoustic_display_dual_timeline_v1")

    with pytest.raises(ValueError, match="SUBTITLE_TIMING_MODEL"):
        SubtitleOptions.from_env()


def test_write_srt_ignores_speaker_metadata(tmp_path):
    path = tmp_path / "speaker.srt"

    subtitle.write_srt(
        [{"start": 0.0, "end": 1.0, "zh_text": "过来", "speaker": "S0"}],
        str(path),
        options=SubtitleOptions(),
    )

    content = path.read_text(encoding="utf-8")
    assert "过来" in content
    assert "[S0]" not in content


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
    assert "00:00:00,000 --> 00:00:01,334" in content


def test_prepare_bilingual_keeps_adjacent_cues_separate(tmp_path):
    path = tmp_path / "no-merge.srt"
    options = SubtitleOptions()
    blocks = subtitle.prepare_srt_blocks(
        [
            {"start": 0.0, "end": 1.0, "ja_text": "私", "zh_text": "甲"},
            {"start": 1.05, "end": 2.0, "ja_text": "君", "zh_text": "乙"},
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
