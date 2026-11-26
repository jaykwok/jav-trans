"""Overlaying two subtitle versions in one ASS file.

The parsing is the part worth pinning: these SRTs come out of the pipeline, but
also out of release archives and third-party tools, and a cue silently dropped
at parse time would show up as a gap in the comparison - which reads as "this
version said nothing here", the exact conclusion the file exists to support.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imported through the `tools` package on purpose: `src` is also on the path, so
# a bare `subtitles.` would resolve to the production package instead.
from tools.subtitles.build_dual_track_ass import (  # noqa: E402
    ass_text,
    ass_time,
    build,
    on_screen_seconds,
    parse_srt,
)

SRT = """\
1
00:00:00,346 --> 00:00:04,301
お二人とも再出演、ありがとう
ございます

2
00:00:04,384 --> 00:00:06,147
来てしまったね
"""


def write(tmp_path: Path, text: str, name: str = "a.srt") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestParsing:
    def test_a_two_line_cue_keeps_both_lines(self, tmp_path: Path) -> None:
        cues = parse_srt(write(tmp_path, SRT))
        assert len(cues) == 2
        assert cues[0][0] == 0.346
        assert cues[0][1] == 4.301
        assert cues[0][2] == "お二人とも再出演、ありがとう\nございます"

    def test_a_bom_does_not_swallow_the_first_cue(self, tmp_path: Path) -> None:
        """Every SRT this pipeline writes starts with one."""
        assert len(parse_srt(write(tmp_path, "﻿" + SRT))) == 2

    def test_dots_and_short_milliseconds_are_accepted(self, tmp_path: Path) -> None:
        cues = parse_srt(write(tmp_path, "1\n00:00:01.5 --> 00:00:02.25\nはい\n"))
        assert cues == [(1.5, 2.25, "はい")]

    def test_a_cue_with_no_index_line_still_parses(self, tmp_path: Path) -> None:
        cues = parse_srt(write(tmp_path, "00:00:01,000 --> 00:00:02,000\nはい\n"))
        assert cues == [(1.0, 2.0, "はい")]

    def test_an_empty_cue_is_dropped_rather_than_shown_blank(self, tmp_path: Path) -> None:
        cues = parse_srt(write(tmp_path, SRT + "\n3\n00:00:07,000 --> 00:00:08,000\n\n"))
        assert len(cues) == 2


class TestRendering:
    def test_braces_cannot_smuggle_in_an_override_tag(self) -> None:
        """ASR text is not trusted markup: `{\\pos(0,0)}` would move the line."""
        rendered = ass_text("{\\pos(0,0)}はい")
        assert "{" not in rendered and "}" not in rendered

    def test_newlines_become_ass_line_breaks(self) -> None:
        assert ass_text("あ\nい") == "あ\\Nい"

    def test_time_is_centiseconds_not_milliseconds(self) -> None:
        assert ass_time(3661.239) == "1:01:01.24"
        assert ass_time(-5.0) == "0:00:00.00"

    def test_each_track_gets_its_own_style_and_anchor(self) -> None:
        out = build([(0.0, 1.0, "old")], [(0.0, 1.0, "new")], old_label="A", new_label="B")
        # Alignment 8 is top-centre, 2 is bottom-centre - the two rows.
        assert ",1,3,1,8," in out and ",1,3,1,2," in out
        assert "Old,A,0,0,0,,old" in out
        assert "New,B,0,0,0,,new" in out

    def test_a_zero_length_cue_is_given_a_visible_duration(self) -> None:
        out = build([(1.0, 1.0, "x")], [(0.0, 1.0, "y")], old_label="A", new_label="B")
        assert "0:00:01.00,0:00:01.05" in out


class TestOnScreenSeconds:
    def test_overlapping_cues_are_not_double_counted(self) -> None:
        assert on_screen_seconds([(0.0, 2.0, "a"), (1.0, 3.0, "b")]) == 3.0

    def test_a_gap_is_not_counted(self) -> None:
        assert on_screen_seconds([(0.0, 1.0, "a"), (5.0, 6.0, "b")]) == 2.0
