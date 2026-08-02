"""The post-gate marks and never deletes, and that has to be structural.

The retired pre-ASR chain deleted audio on weak evidence and its mistakes were
unrecoverable; the whole point of moving quality control after the decoder is
that a wrong call here costs a label. So the first thing these tests pin is that
no input can make `kept` False, and the second is that an uncalibrated threshold
stays switched off rather than quietly marking everything.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr import postgate  # noqa: E402
from asr.cueqc import build_candidate  # noqa: E402
from asr.postgate import PostGateConfig, review, review_all  # noqa: E402


def _candidate(text: str, *, start: float = 0.0, end: float = 4.0, neighbours=()):
    chunk = {"index": 0, "start": start, "end": end}
    chunks = [chunk]
    results = [{"text": text, "raw_text": text}]
    position = 0
    for offset, neighbour in enumerate(neighbours, start=1):
        chunks.append({"index": offset, "start": end, "end": end + 2.0})
        results.append({"text": neighbour, "raw_text": neighbour})
    return build_candidate(
        chunk=chunk,
        text_result=results[0],
        position=position,
        chunks=chunks,
        text_results=results,
        audio_id="probe",
    )


class TestMarkingOnly:
    def test_nothing_is_ever_dropped(self) -> None:
        for text in ("", "ああああああああああああ", "本当にありがとうございます"):
            assert review(_candidate(text))["kept"] is True

    def test_a_flagged_cue_is_still_kept(self) -> None:
        verdict = review(_candidate("ああああああああああああああああ"))
        assert verdict["flagged"] is True
        assert verdict["kept"] is True


class TestFlags:
    def test_empty_text_is_flagged(self) -> None:
        assert postgate.FLAG_EMPTY in review(_candidate(""))["flags"]

    def test_a_runaway_loop_is_flagged(self) -> None:
        """The measured case: pilot runaway regions scored `unique_ratio`
        0.107 against 0.475 for real speech."""
        verdict = review(_candidate("あああああああああああああああああああ"))
        assert postgate.FLAG_RUNAWAY in verdict["flags"]
        assert verdict["observed"]["unique_ratio"] < 0.25

    def test_ordinary_speech_is_not_flagged(self) -> None:
        verdict = review(_candidate("今日はとてもいい天気ですね", end=4.0))
        assert verdict["flags"] == []

    def test_an_impossible_rate_is_flagged(self) -> None:
        verdict = review(
            _candidate("本当にありがとうございました、また明日会いましょう", end=0.5)
        )
        assert postgate.FLAG_IMPOSSIBLE_RATE in verdict["flags"]

    def test_a_short_interjection_is_not_treated_as_a_loop(self) -> None:
        """This domain is full of two-character interjections. On three
        characters `unique_ratio` is quantised to thirds and would flag them."""
        verdict = review(_candidate("ああ", end=1.0))
        assert postgate.FLAG_RUNAWAY not in verdict["flags"]

    def test_a_cue_repeating_its_neighbour_is_flagged(self) -> None:
        verdict = review(_candidate("同じ台詞です", neighbours=("同じ台詞です",)))
        assert postgate.FLAG_DUPLICATE_NEIGHBOUR in verdict["flags"]

    def test_distinct_neighbours_are_not_flagged(self) -> None:
        verdict = review(_candidate("違う台詞です", neighbours=("別の台詞です",)))
        assert postgate.FLAG_DUPLICATE_NEIGHBOUR not in verdict["flags"]


class TestAlignmentScore:
    def test_the_check_is_off_until_it_is_calibrated(self) -> None:
        """An uncalibrated threshold that marks a third of the output is worse
        than no check; the default must therefore do nothing."""
        assert PostGateConfig().min_alignment_score is None
        verdict = review(_candidate("今日はいい天気ですね"), alignment_score=-99.0)
        assert postgate.FLAG_UNSUPPORTED_BY_AUDIO not in verdict["flags"]
        assert verdict["alignment_score_checked"] is False

    def test_a_calibrated_threshold_flags_unsupported_text(self) -> None:
        config = PostGateConfig(min_alignment_score=-3.0)
        verdict = review(
            _candidate("今日はいい天気ですね"), alignment_score=-8.0, config=config
        )
        assert postgate.FLAG_UNSUPPORTED_BY_AUDIO in verdict["flags"]
        assert verdict["alignment_score_checked"] is True

    def test_a_missing_score_is_not_read_as_a_pass(self) -> None:
        """No measurement is not evidence of support. The check must record
        that it did not run rather than let the cue through as verified."""
        config = PostGateConfig(min_alignment_score=-3.0)
        verdict = review(_candidate("今日はいい天気ですね"), config=config)
        assert postgate.FLAG_UNSUPPORTED_BY_AUDIO not in verdict["flags"]
        assert verdict["alignment_score_checked"] is False

    def test_a_supported_cue_passes(self) -> None:
        config = PostGateConfig(min_alignment_score=-3.0)
        verdict = review(
            _candidate("今日はいい天気ですね"), alignment_score=-1.2, config=config
        )
        assert verdict["flags"] == []


class TestConfigContract:
    def test_out_of_range_values_are_refused(self) -> None:
        with pytest.raises(ValueError, match="min_unique_ratio"):
            PostGateConfig(min_unique_ratio=1.5)
        with pytest.raises(ValueError, match="max_chars_per_sec"):
            PostGateConfig(max_chars_per_sec=0.0)
        with pytest.raises(ValueError, match="max_repeat_run"):
            PostGateConfig(max_repeat_run=0)

    def test_the_unique_ratio_threshold_is_the_measured_one(self) -> None:
        assert PostGateConfig().min_unique_ratio == 0.25


class TestBatch:
    def test_scores_must_line_up_with_candidates(self) -> None:
        with pytest.raises(ValueError, match="line up"):
            review_all([_candidate("あ"), _candidate("い")], alignment_scores=[-1.0])

    def test_omitting_scores_reviews_every_candidate(self) -> None:
        verdicts = review_all([_candidate("あ"), _candidate("い")])
        assert len(verdicts) == 2
        assert all(verdict["kept"] for verdict in verdicts)
