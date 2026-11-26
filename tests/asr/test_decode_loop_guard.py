"""Two bounds stop an ASR decode, and only one of them is safe.

The safe one is arithmetic: audio of `duration_s` seconds cannot contain more
than `duration_s * TOKENS_PER_SECOND_CEILING` tokens of speech, so stopping there
cannot cut a transcription short. The unsafe one is the repetition guard, which
buys decode steps by guessing that a repeating tail will keep repeating.

Measured cost of having no guard at all: on an RTX 4060 Ti, batch 8, 20 s chunks,
one chunk decoded `んじゅるるる…` and never emitted EOS. `generate` returns only
when every sequence is done, so that one chunk pushed the batch to its cap while
the median sequence finished at 59 - 6.59 s against 3.47 s, and per-step cost is
flat in batch size, so the wasted steps were the entire loss.

Measured cost of guessing wrong: on 2026-08-03 a 72-token bar cut nine chunks of
a village-ritual film that were genuinely chanting, and one of them lost the
sentence that followed the chant, `綾様、お召し上がりください`. A chunk is decoded
once. These tests pin the bar far above every genuine repetition measured on that
corpus, and pin the arithmetic bound as the thing that actually catches runaways.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from asr import decode_guard  # noqa: E402
from asr.decode_guard import (  # noqa: E402
    detect_repetition_loop,
    plausible_token_budget,
)


@pytest.fixture(autouse=True)
def _unset_decode_env(monkeypatch) -> None:
    """Every threshold here is derived from the environment, so a developer with
    `ASR_MAX_NEW_TOKENS` exported would be testing a different rule."""
    for name in (
        "ASR_MAX_NEW_TOKENS",
        "ASR_DECODE_TOKENS_PER_SECOND",
        "ASR_DECODE_LOOP_GUARD",
        "ASR_DECODE_LOOP_BUDGET_FRACTION",
        "ASR_DECODE_LOOP_MIN_REPEATS",
        "ASR_DECODE_LOOP_MIN_TOKENS",
        "ASR_DECODE_LOOP_MAX_NGRAM",
    ):
        monkeypatch.delenv(name, raising=False)


def _bar() -> int:
    """The repetition bar for a full-length chunk, which is what almost every
    chunk is: cuts take the latest pause under the 30 s ceiling, so chunk length
    tracks the ceiling."""
    return decode_guard.loop_guard_config(plausible_token_budget(30.0))[2]


def _suffix(*rows: list[int]) -> "torch.Tensor":
    width = max(len(row) for row in rows)
    padded = [[0] * (width - len(row)) + row for row in rows]
    return torch.tensor(padded, dtype=torch.long)


class TestTheBudgetFollowsTheAudio:
    def test_a_longer_chunk_gets_a_larger_budget(self) -> None:
        """The whole point. A flat budget is what silently truncated dialogue:
        128 tokens spread over a 30 s chunk is 4.27 tok/s, and this domain was
        measured at 4.45."""
        assert plausible_token_budget(10.0) < plausible_token_budget(30.0)

    def test_it_clears_the_fastest_speech_measured(self) -> None:
        """4.45 tok/s over 26.7 s was the fastest chunk in a 283-chunk corpus
        decoded with a deliberately generous budget. The bound has to sit above
        it with room for a dialogue-denser film."""
        assert plausible_token_budget(26.73) > 119

    def test_a_short_chunk_still_gets_enough_to_speak(self) -> None:
        """A budget of `duration x rate` alone starves the tail chunk: the output
        format costs tokens before any speech is transcribed."""
        assert plausible_token_budget(0.4) >= decode_guard.MIN_TOKEN_BUDGET

    def test_a_missing_duration_does_not_produce_a_zero_budget(self) -> None:
        assert plausible_token_budget(0.0) >= decode_guard.MIN_TOKEN_BUDGET
        assert plausible_token_budget(None) >= decode_guard.MIN_TOKEN_BUDGET  # type: ignore[arg-type]

    def test_the_rate_ceiling_is_configurable(self, monkeypatch) -> None:
        monkeypatch.setenv("ASR_DECODE_TOKENS_PER_SECOND", "20")
        assert plausible_token_budget(30.0) > 600

    def test_a_junk_rate_falls_back_to_the_measured_default(self, monkeypatch) -> None:
        monkeypatch.setenv("ASR_DECODE_TOKENS_PER_SECOND", "not-a-number")
        assert (
            decode_guard.tokens_per_second_ceiling()
            == decode_guard.TOKENS_PER_SECOND_CEILING
        )

    def test_an_explicit_cap_can_only_lower_the_budget(self, monkeypatch) -> None:
        """`ASR_MAX_NEW_TOKENS` stays available for bounding decode cost, but it
        is an override, not the default - as the default it truncates."""
        monkeypatch.setenv("ASR_MAX_NEW_TOKENS", "128")
        assert plausible_token_budget(30.0) == 128
        assert plausible_token_budget(2.0) < 128

    @pytest.mark.parametrize("value", ["", "auto", "0", "none", "off", "not-a-number"])
    def test_no_explicit_cap_means_follow_the_audio(self, monkeypatch, value) -> None:
        monkeypatch.setenv("ASR_MAX_NEW_TOKENS", value)
        assert decode_guard.explicit_token_cap() is None
        assert plausible_token_budget(30.0) > 128


class TestTheArithmeticBoundIsPerRow:
    def test_a_short_chunk_does_not_inherit_a_long_chunks_budget(self) -> None:
        """`max_new_tokens` is one number for the batch and has to be the largest
        budget in it, so without a per-row stop the shortest chunk would be free
        to generate the longest chunk's worth of tokens."""
        criteria = decode_guard.build_stopping_criteria(
            prompt_length=3, token_budgets=[80, 300]
        )
        prompt = [1, 2, 3]
        done = criteria(torch.tensor([prompt + [9] * 80] * 2, dtype=torch.long), None)
        assert done.tolist() == [True, False]

    def test_it_survives_the_guard_being_switched_off(self, monkeypatch) -> None:
        """The arithmetic bound is not a heuristic, so it is not part of what the
        guard switch turns off."""
        monkeypatch.setenv("ASR_DECODE_LOOP_GUARD", "0")
        criteria = decode_guard.build_stopping_criteria(
            prompt_length=1, token_budgets=[10]
        )
        assert criteria is not None
        assert criteria(torch.tensor([[5] + [9] * 10], dtype=torch.long), None)[0]

    def test_a_row_count_mismatch_fails_open(self) -> None:
        """Beam search or `num_return_sequences > 1` would break the row-to-chunk
        mapping. Stopping the wrong sequence is worse than not stopping."""
        criteria = decode_guard.build_stopping_criteria(
            prompt_length=1, token_budgets=[10]
        )
        rows = [[5] + list(range(20))] * 3
        done = criteria(torch.tensor(rows, dtype=torch.long), None)
        assert done.tolist() == [False, False, False]


class TestTheGuardFiresOnRealLoops:
    def test_the_bar_is_one_token_count_whatever_the_unit(self) -> None:
        """One budget for every unit length. The measured runaways repeated units
        of 4, 5, 6, 7, 9 and 13 tokens; a rule expressed in *copies* fires at
        wildly different token counts across that range, which is how the old
        shape cut real audio and missed real loops simultaneously."""
        for ngram in (1, 2, 3, 4, 5, 6, 7, 9, 13, 16, 32):
            copies = -(-_bar() // ngram)
            unit = list(range(100, 100 + ngram))
            assert detect_repetition_loop(_suffix([7, 8] + unit * copies))[0], ngram

    def test_a_single_token_repeating_is_the_original_failure(self) -> None:
        """`んじゅるるるるる…` - one kana token repeated to the budget."""
        assert detect_repetition_loop(_suffix([7, 8] + [9] * _bar()))[0]

    def test_the_loop_must_reach_the_end_of_the_sequence(self) -> None:
        """A sequence that looped and then recovered is still generating real
        text; stopping it there would truncate what came after."""
        assert not detect_repetition_loop(_suffix([9] * (_bar() + 8) + [1, 2, 3, 4, 5]))[0]

    def test_a_unit_longer_than_the_window_is_out_of_reach(self) -> None:
        """Honest about the remaining hole rather than pretending there is none:
        past `MAX_NGRAM` a loop runs to the arithmetic bound and is reported as
        `decode_cap_truncations`, not as a guard stop."""
        max_ngram = decode_guard.loop_guard_config(plausible_token_budget(30.0))[0]
        unit = list(range(200, 200 + max_ngram + 4))
        assert not detect_repetition_loop(_suffix(unit * 6))[0]


class TestTheGuardLeavesRealChantingAlone:
    def test_three_repeats_are_not_a_loop(self) -> None:
        """`すー…すー…すー` and `あっ…あっ…あっ` are ordinary in this domain."""
        assert not detect_repetition_loop(_suffix([1, 2, 3] * 3))[0]

    def test_the_longest_genuine_repetition_measured_survives(self) -> None:
        """65 tokens: `んぐっ、んぐっ、んぐっ…!` five times, filling a whole 30 s
        chunk the model terminated on its own. It was the longest real repeating
        tail in a 283-chunk corpus, at 2.18 tok/s."""
        assert not detect_repetition_loop(_suffix(list(range(13)) * 5))[0]

    def test_the_ritual_chant_survives(self) -> None:
        """`ありがとやぁ、ありがとやぁ…` four times over 30 s - a crowd, not a
        loop. This is the film that started the investigation."""
        assert not detect_repetition_loop(_suffix(list(range(10)) * 4))[0]

    def test_the_chant_that_lost_a_sentence_survives(self) -> None:
        """`ありがとよ` six times, after which the audio said
        `綾様、お召し上がりください`. The 72-token bar stopped the decode at the
        sixth copy and that sentence was never transcribed."""
        assert not detect_repetition_loop(_suffix(list(range(5)) * 6))[0]

    def test_the_bar_sits_well_above_the_fastest_real_repetition(self) -> None:
        """The bar is a share of a duration-derived budget, so it is a *rate*:
        this pins it above the 2.18 tok/s of the fastest genuine chanting
        measured, with margin for a faster chant."""
        assert _bar() / 30.0 > 2 * 2.18

    def test_a_long_phrase_appearing_twice_is_not_a_loop(self) -> None:
        """What the copies floor is for: two copies of a long phrase clears the
        token bar on its own."""
        assert not detect_repetition_loop(_suffix(list(range(40, 40 + _bar())) * 2))[0]

    def test_a_short_sequence_is_never_stopped(self) -> None:
        assert not detect_repetition_loop(_suffix([1, 2, 3]))[0]

    def test_varied_text_is_never_stopped(self) -> None:
        assert not detect_repetition_loop(_suffix(list(range(_bar() + 40))))[0]


class TestPerSequence:
    def test_only_the_looping_row_is_marked(self) -> None:
        """The whole point. A batch-wide verdict would truncate the seven
        healthy chunks that were being held hostage by the eighth."""
        done = detect_repetition_loop(
            _suffix(
                list(range(20)),
                [3, 4] + [9] * _bar(),
                list(range(20, 40)),
            )
        )
        assert done.tolist() == [False, True, False]

    def test_the_verdict_is_one_bool_per_row(self) -> None:
        done = detect_repetition_loop(_suffix([1, 2], [3, 4]))
        assert done.shape == (2,)
        assert done.dtype == torch.bool


class TestThresholdsAreDerivedThenOverridable:
    def test_the_bar_moves_with_the_chunk_length(self) -> None:
        """A token count calibrated at one chunk length is a different amount of
        speech at another, which is why the tunable is a fraction."""
        short = decode_guard.loop_guard_config(plausible_token_budget(10.0))[2]
        long = decode_guard.loop_guard_config(plausible_token_budget(30.0))[2]
        assert short < long

    def test_the_bar_moves_with_the_rate_ceiling(self, monkeypatch) -> None:
        base = decode_guard.loop_guard_config(plausible_token_budget(30.0))[2]
        monkeypatch.setenv("ASR_DECODE_TOKENS_PER_SECOND", "20")
        assert decode_guard.loop_guard_config(plausible_token_budget(30.0))[2] > base

    def test_the_window_is_derived_from_the_bar(self) -> None:
        """`MAX_NGRAM` stops being a guess: a unit too long to repeat
        `MIN_REPEATS` times inside the bar cannot be judged a loop at all."""
        max_ngram, min_repeats, min_tokens = decode_guard.loop_guard_config(
            plausible_token_budget(30.0)
        )
        assert max_ngram == min_tokens // min_repeats

    def test_a_stricter_repeat_count_can_be_set(self, monkeypatch) -> None:
        monkeypatch.setenv("ASR_DECODE_LOOP_MIN_REPEATS", "20")
        monkeypatch.setenv("ASR_DECODE_LOOP_MIN_TOKENS", "40")
        max_ngram, min_repeats, min_tokens = decode_guard.loop_guard_config()
        assert (min_repeats, min_tokens) == (20, 40)
        assert not detect_repetition_loop(
            _suffix([9] * 39),
            max_ngram=max_ngram,
            min_repeats=min_repeats,
            min_tokens=min_tokens,
        )[0]

    def test_a_junk_threshold_falls_back_to_the_measured_default(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("ASR_DECODE_LOOP_MIN_REPEATS", "not-a-number")
        assert decode_guard.loop_guard_config()[1] == decode_guard.MIN_REPEATS

    @pytest.mark.parametrize("value", ["0", "-0.5", "1.5", "not-a-number"])
    def test_a_fraction_outside_the_unit_interval_is_refused(
        self, monkeypatch, value
    ) -> None:
        """Above 1 the bar exceeds the budget and the guard never fires; at or
        below 0 it fires on everything."""
        monkeypatch.setenv("ASR_DECODE_LOOP_BUDGET_FRACTION", value)
        assert decode_guard.loop_budget_fraction() == decode_guard.LOOP_BUDGET_FRACTION

    def test_the_defaults_are_the_ones_that_were_measured(self) -> None:
        """A 30 s chunk at 10 tok/s plus structure tokens is a 316-token budget;
        half of it is 158 repeated tokens, which is 5.3 tok/s of pure repetition
        against the 2.18 tok/s of the fastest real chanting measured."""
        assert plausible_token_budget(30.0) == 316
        assert decode_guard.loop_guard_config(316) == (52, 3, 158)


class TestSwitch:
    def test_the_guard_is_on_by_default(self, monkeypatch) -> None:
        monkeypatch.delenv("ASR_DECODE_LOOP_GUARD", raising=False)
        assert decode_guard.loop_guard_enabled()

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
    def test_it_can_be_switched_off(self, monkeypatch, value: str) -> None:
        monkeypatch.setenv("ASR_DECODE_LOOP_GUARD", value)
        assert not decode_guard.loop_guard_enabled()

    def test_switching_it_off_with_no_budgets_passes_no_criteria_at_all(
        self, monkeypatch
    ) -> None:
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

        early = torch.tensor([prompt + [9] * 40], dtype=torch.long)
        assert not criteria(early, None)[0]

        looping = torch.tensor([prompt + [9] * _bar()], dtype=torch.long)
        assert criteria(looping, None)[0]
