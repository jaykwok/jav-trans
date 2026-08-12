"""A cue that ends on its last spoken character is often too short to read.

v3 locks the measured timeline, and that lock also switched off the out-point
extension every subtitle spec asks for: measured on eight films, 487 of 7,016
cues fall under the 20-frame floor. The silence after them is already empty, so
holding the line there costs nothing and invents nothing.

Everything here is about the boundary of "nothing invented": starts never move,
the acoustic edge stays where the speech stopped, the extension never reaches
the next cue, and it never turns a compliant cue into an over-long one.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from subtitles import writer as subtitle  # noqa: E402
from subtitles.options import BASE_FPS, SubtitleOptions  # noqa: E402


FRAME_GAP_S = 2.0 / BASE_FPS
OPTIONS = SubtitleOptions(drop_vocalisation_only_cues=False)


def _cue(text: str, start: float, char_s: float = 0.2) -> dict:
    words = []
    cursor = start
    for char in text:
        words.append(
            {
                "word": char,
                "start": cursor,
                "end": cursor + char_s,
                "timestamp_kind": "ctc_forced_alignment",
            }
        )
        cursor += char_s
    return {
        "start": start,
        "end": cursor,
        "ja_text": text,
        "zh_text": text,
        "words": words,
    }


def _prepare(blocks: list[dict], *, options: SubtitleOptions = OPTIONS) -> list[dict]:
    return subtitle.prepare_srt_blocks(blocks, options=options)


class TestWhatItBuys:
    def test_a_short_cue_gains_reading_time_from_the_silence_after_it(self) -> None:
        prepared = _prepare([_cue("はい", 0.0), _cue("そうですね", 8.0)])

        assert prepared[0]["display_duration"] == pytest.approx(0.4 + 0.5)
        assert prepared[0]["end"] == pytest.approx(0.4 + 0.5)

    def test_the_last_cue_lingers_too(self) -> None:
        """Nothing follows it, so the only thing that could stop it is the cap."""
        prepared = _prepare([_cue("おわり", 0.0)])

        assert prepared[0]["end"] == pytest.approx(0.6 + 0.5)


class TestWhatItMustNotDo:
    def test_the_start_never_moves(self) -> None:
        blocks = [_cue("はい", 0.0), _cue("そうですね", 8.0)]
        prepared = _prepare(blocks)

        assert prepared[0]["start"] == pytest.approx(0.0)
        assert prepared[1]["start"] == pytest.approx(8.0)

    def test_the_acoustic_edge_stays_where_the_speech_stopped(self) -> None:
        """The record of what was actually said must survive the display change,
        or nothing downstream can tell speech from linger."""
        prepared = _prepare([_cue("はい", 0.0), _cue("そうですね", 8.0)])

        assert prepared[0]["acoustic_end"] == pytest.approx(0.4)
        assert prepared[0]["display_shift_end_s"] == pytest.approx(0.5)

    def test_it_stops_two_frames_before_the_next_cue(self) -> None:
        prepared = _prepare([_cue("はい", 0.0), _cue("そうですね", 0.7)])

        assert prepared[0]["end"] == pytest.approx(0.7 - FRAME_GAP_S)
        assert prepared[0]["end"] < prepared[1]["start"]

    def test_a_cue_whose_neighbour_is_already_close_is_left_alone(self) -> None:
        """Not shortened. Pulling this end in would cut measured speech, which
        is the one thing the locked timeline exists to prevent - and 553 of the
        7,016 shipped cues sit inside the two-frame gap for exactly this reason.
        """
        prepared = _prepare([_cue("はい", 0.0), _cue("そうですね", 0.42)])

        assert prepared[0]["end"] == pytest.approx(0.4)
        assert prepared[0]["acoustic_end"] == pytest.approx(0.4)

    def test_a_cue_near_the_soft_cap_is_not_pushed_over_it(self) -> None:
        """`spec_duration_over_7s_count` is a zero-tolerance QC counter, so
        buying reading time must not manufacture entries for it."""
        prepared = _prepare([_cue("あ" * 17, 0.0, char_s=0.4), _cue("次", 20.0)])

        assert prepared[0]["display_duration"] == pytest.approx(7.0)
        assert prepared[0]["duration_soft_cap_violation"] is False

    def test_an_already_over_long_cue_is_not_extended_further(self) -> None:
        prepared = _prepare([_cue("あ" * 20, 0.0, char_s=0.4), _cue("次", 20.0)])

        assert prepared[0]["end"] == pytest.approx(8.0)
        assert prepared[0]["acoustic_end"] == pytest.approx(8.0)


class TestStability:
    def test_running_the_preparation_twice_changes_nothing(self) -> None:
        """The cap is measured from `acoustic_end`, not from the current end, so
        a second pass cannot compound the first."""
        once = _prepare([_cue("はい", 0.0), _cue("そうですね", 8.0)])
        twice = _prepare(once)

        assert [cue["end"] for cue in twice] == [
            pytest.approx(cue["end"]) for cue in once
        ]
        assert [cue["start"] for cue in twice] == [
            pytest.approx(cue["start"]) for cue in once
        ]

    def test_disabling_timing_polish_disables_the_linger(self) -> None:
        prepared = _prepare(
            [_cue("はい", 0.0), _cue("そうですね", 8.0)],
            options=SubtitleOptions(
                drop_vocalisation_only_cues=False,
                timing_polish_enabled=False,
            ),
        )

        assert prepared[0]["end"] == pytest.approx(0.4)

    def test_the_diagnostics_report_what_was_added(self) -> None:
        diagnostics: dict = {}
        subtitle.prepare_srt_blocks(
            [_cue("はい", 0.0), _cue("そうですね", 8.0)],
            options=OPTIONS,
            diagnostics=diagnostics,
        )

        assert diagnostics["display_linger_applied_count"] == 2
        assert diagnostics["display_linger_total_s"] == pytest.approx(1.0)


class TestSilenceLeftByADroppedRun:
    def test_the_gap_a_dropped_vocalisation_run_leaves_is_usable(self) -> None:
        """The filter runs first on purpose. The moaning cues are gone from the
        file, so the silence they occupied is silence like any other."""
        blocks = [
            _cue("そのときね", 0.0),
            _cue("んっんっ", 2.0),
            _cue("あぁはぁ", 4.0),
            _cue("それで帰った", 8.0),
        ]

        prepared = subtitle.prepare_srt_blocks(
            blocks,
            options=SubtitleOptions(drop_vocalisation_only_cues=True),
        )

        assert [cue["ja_text"] for cue in prepared] == ["そのときね", "それで帰った"]
        assert prepared[0]["end"] == pytest.approx(1.0 + 0.5)
