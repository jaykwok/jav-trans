"""The loop guard cuts decode time by stopping a sequence early, so its whole
risk is stopping the wrong one.

Measured cost of not having it: on an RTX 4060 Ti, batch 8, 20 s chunks, one
chunk decoded `んじゅるるる…` and never emitted EOS. `generate` returns only when
every sequence is done, so that one chunk pushed the batch to the 128-token cap
while the median sequence finished at 59 - 128 steps for work that needed 64,
6.59 s against 3.47 s. Per-step cost is flat in batch size, so the wasted steps
were the entire loss.

Measured effect of having it: the same batch came back with 7 of 8 sequences
byte-identical, and the eighth truncated at the point its repetition began.
These tests pin the thresholds that make that true - this domain says
`あっ…あっ…あっ` and `すー…すー…すー` in ordinary speech, and the guard must sit
far above them.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from asr import decode_guard  # noqa: E402
from asr.decode_guard import detect_repetition_loop  # noqa: E402


def _suffix(*rows: list[int]) -> "torch.Tensor":
    width = max(len(row) for row in rows)
    padded = [[0] * (width - len(row)) + row for row in rows]
    return torch.tensor(padded, dtype=torch.long)


class TestItFiresOnRealLoops:
    def test_one_token_repeating_to_the_threshold(self) -> None:
        """The measured failure: a single kana token repeated ~90 times."""
        assert detect_repetition_loop(_suffix([7, 8] + [9] * 10))[0]

    def test_a_two_token_unit_repeating(self) -> None:
        assert detect_repetition_loop(_suffix([1] + [4, 5] * 6))[0]

    def test_a_four_token_unit_repeating(self) -> None:
        assert detect_repetition_loop(_suffix([1] + [4, 5, 6, 7] * 6))[0]

    def test_the_loop_must_reach_the_end_of_the_sequence(self) -> None:
        """A sequence that looped and then recovered is still generating real
        text; stopping it there would truncate what came after."""
        assert not detect_repetition_loop(_suffix([9] * 12 + [1, 2, 3, 4, 5]))[0]


class TestItLeavesOrdinarySpeechAlone:
    def test_three_repeats_are_not_a_loop(self) -> None:
        """`すー…すー…すー` and `あっ…あっ…あっ` are ordinary in this domain."""
        assert not detect_repetition_loop(_suffix([1, 2, 3] * 3))[0]

    def test_five_repeats_of_a_pair_are_not_a_loop(self) -> None:
        assert not detect_repetition_loop(_suffix([4, 5] * 5))[0]

    def test_a_single_token_needs_ten_not_six(self) -> None:
        """Six copies clears `min_repeats` but not `min_tokens`; the extra bar
        is there because one repeating token is the cheapest way to trip."""
        assert not detect_repetition_loop(_suffix([9] * 9))[0]
        assert detect_repetition_loop(_suffix([9] * 10))[0]

    def test_a_short_sequence_is_never_stopped(self) -> None:
        assert not detect_repetition_loop(_suffix([1, 2, 3]))[0]

    def test_varied_text_is_never_stopped(self) -> None:
        assert not detect_repetition_loop(_suffix(list(range(40))))[0]


class TestPerSequence:
    def test_only_the_looping_row_is_marked(self) -> None:
        """The whole point. A batch-wide verdict would truncate the seven
        healthy chunks that were being held hostage by the eighth."""
        done = detect_repetition_loop(
            _suffix(
                list(range(20)),
                [3, 4] + [9] * 12,
                list(range(20, 40)),
            )
        )
        assert done.tolist() == [False, True, False]

    def test_the_verdict_is_one_bool_per_row(self) -> None:
        done = detect_repetition_loop(_suffix([1, 2], [3, 4]))
        assert done.shape == (2,)
        assert done.dtype == torch.bool


class TestThresholdsAreConfigurable:
    def test_a_stricter_repeat_count_can_be_set(self, monkeypatch) -> None:
        monkeypatch.setenv("ASR_DECODE_LOOP_MIN_REPEATS", "20")
        monkeypatch.setenv("ASR_DECODE_LOOP_MIN_TOKENS", "40")
        max_ngram, min_repeats, min_tokens = decode_guard.loop_guard_config()
        assert (min_repeats, min_tokens) == (20, 40)
        assert not detect_repetition_loop(
            _suffix([9] * 12),
            max_ngram=max_ngram,
            min_repeats=min_repeats,
            min_tokens=min_tokens,
        )[0]

    def test_a_junk_threshold_falls_back_to_the_measured_default(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("ASR_DECODE_LOOP_MIN_REPEATS", "not-a-number")
        assert decode_guard.loop_guard_config()[1] == decode_guard.MIN_REPEATS

    def test_the_defaults_are_the_ones_that_were_measured(self) -> None:
        assert decode_guard.loop_guard_config() == (4, 6, 10)


class TestSwitch:
    def test_the_guard_is_on_by_default(self, monkeypatch) -> None:
        monkeypatch.delenv("ASR_DECODE_LOOP_GUARD", raising=False)
        assert decode_guard.loop_guard_enabled()

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
    def test_it_can_be_switched_off(self, monkeypatch, value: str) -> None:
        monkeypatch.setenv("ASR_DECODE_LOOP_GUARD", value)
        assert not decode_guard.loop_guard_enabled()

    def test_switching_it_off_passes_no_criteria_at_all(self, monkeypatch) -> None:
        """None rather than an empty list, so the disabled path calls `generate`
        exactly as it did before this module existed."""
        monkeypatch.setenv("ASR_DECODE_LOOP_GUARD", "0")
        assert decode_guard.build_stopping_criteria(10) is None


class TestCriteriaObject:
    def test_it_reports_done_only_after_enough_generated_tokens(
        self, monkeypatch
    ) -> None:
        """`input_ids` handed to a criterion includes the prompt; measuring the
        loop against that would let prompt tokens count as generated text."""
        monkeypatch.delenv("ASR_DECODE_LOOP_GUARD", raising=False)
        criteria = decode_guard.build_stopping_criteria(prompt_length=5)
        prompt = [9] * 5

        early = torch.tensor([prompt + [9, 9]], dtype=torch.long)
        assert not criteria(early, None)[0]

        looping = torch.tensor([prompt + [9] * 10], dtype=torch.long)
        assert criteria(looping, None)[0]
