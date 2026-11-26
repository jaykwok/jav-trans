"""A chunk that runs out of token budget is the one decode failure with no
other symptom.

`postgate` catches repetition and the loop guard stops it; both leave a mark. A
chunk that used its whole budget comes back well-formed and just stops - the tail
never reaches a subtitle and nothing downstream can tell it from a chunk that had
less to say.

The budget is per row, derived from that chunk's own duration, so the count means
something specific: the model emitted more tokens than its audio can physically
contain. Under the old flat 128 it was ambiguous - at 30s chunks, 128 tokens is
4.27 tok/s against a measured 4.45, so hitting it could equally mean real
dialogue was being amputated.

The discrimination that matters here is out-of-budget vs stopped-by-guard: the
guard's victims are already reported as `runaway_repetition`, and counting them
twice would turn a real number into a scary one.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from asr import pipeline as pipeline_module  # noqa: E402
from asr.local_backend import _rows_truncated_at_cap  # noqa: E402


class _GenerationConfig:
    def __init__(self, eos_token_id, pad_token_id) -> None:
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id


class _Model:
    def __init__(self, eos_token_id=9, pad_token_id=0) -> None:
        self.generation_config = _GenerationConfig(eos_token_id, pad_token_id)
        self.config = _GenerationConfig(eos_token_id, pad_token_id)


def _suffix(*rows: list[int]) -> "torch.Tensor":
    return torch.tensor(rows, dtype=torch.long)


class TestRowClassification:
    def test_a_row_without_a_stop_token_at_full_length_is_truncated(self) -> None:
        assert _rows_truncated_at_cap(_suffix([1, 2, 3, 4]), _Model(), 4) == [True]

    def test_a_row_that_emitted_its_stop_token_is_not(self) -> None:
        assert _rows_truncated_at_cap(_suffix([1, 2, 3, 9]), _Model(), 4) == [False]

    def test_a_row_the_guard_cut_short_is_not(self) -> None:
        """Padded below its budget, so it finished on a stopping criterion. That
        is `runaway_repetition` in the postgate report, not a budget problem."""
        assert _rows_truncated_at_cap(_suffix([1, 2, 0, 0]), _Model(), 4) == [False]

    def test_each_row_is_judged_against_its_own_budget(self) -> None:
        """The batch runs to the longest chunk's budget, so a short chunk padded
        out to that width has not used *its* budget - and a per-row budget is the
        only reason the flag can be read as "loop" rather than "cap too low"."""
        suffix = _suffix([1, 2, 3, 4], [1, 2, 3, 4])
        assert _rows_truncated_at_cap(suffix, _Model(), [2, 8]) == [True, False]

    def test_a_budget_list_of_the_wrong_length_reports_nothing(self) -> None:
        """Instrumentation must not invent a verdict when the mapping it needs is
        broken."""
        assert _rows_truncated_at_cap(_suffix([1, 2, 3, 4]), _Model(), [4, 4]) == [
            False
        ]

    def test_rows_are_judged_independently(self) -> None:
        """The failure mode this has to survive: one row holds the batch open to
        the cap, so every finished row is padded out to the same width."""
        suffix = _suffix([1, 2, 3, 4], [1, 9, 0, 0], [5, 6, 7, 8])
        assert _rows_truncated_at_cap(suffix, _Model(), 4) == [True, False, True]

    def test_a_list_of_stop_ids_is_accepted(self) -> None:
        model = _Model(eos_token_id=[8, 9], pad_token_id=0)
        suffix = _suffix([1, 2, 3, 8], [1, 2, 3, 4])
        assert _rows_truncated_at_cap(suffix, model, 4) == [False, True]

    def test_padding_indistinguishable_from_a_stop_token_falls_back(self) -> None:
        """With `pad_token_id` inside the stop set there is no way to measure how
        many tokens a row emitted, so lean on the stop token alone rather than
        report every padded row as truncated."""
        model = _Model(eos_token_id=[0, 9], pad_token_id=0)
        suffix = _suffix([1, 2, 0, 0], [1, 2, 3, 4])
        assert _rows_truncated_at_cap(suffix, model, 4) == [False, True]

    def test_no_stop_token_at_all_reports_nothing(self) -> None:
        """Without a terminator, "finished" and "out of budget" are the same
        observation, and guessing would flag every chunk of every film."""
        model = _Model(eos_token_id=None, pad_token_id=None)
        assert _rows_truncated_at_cap(_suffix([1, 2, 3, 4]), model, 4) == [False]

    def test_a_row_under_the_cap_without_a_stop_token_is_not_flagged(self) -> None:
        """`max_new_tokens` was not reached, so whatever ended this row, it was
        not the budget."""
        assert _rows_truncated_at_cap(_suffix([1, 2, 3]), _Model(), 8) == [False]


class TestPipelineAccounting:
    def test_flagged_chunks_are_counted_and_logged(self) -> None:
        log: list[str] = []
        count = pipeline_module._count_decode_cap_truncations(
            [
                {"asr_generation": {"truncated_at_cap": True}},
                {"asr_generation": {}},
                {"asr_generation": {"truncated_at_cap": True}},
            ],
            log=log,
        )
        assert count == 2
        assert len(log) == 1
        assert "2/3" in log[0]
        # Reads as a runaway counter, not as "raise the cap" - the budget already
        # tracks the audio, so there is no cap left to raise.
        assert "失控" in log[0]

    def test_a_clean_run_says_nothing(self) -> None:
        """A zero line every job trains the reader to skip the section."""
        log: list[str] = []
        assert pipeline_module._count_decode_cap_truncations(
            [{"asr_generation": {}}], log=log
        ) == 0
        assert log == []

    def test_a_result_without_generation_metadata_is_not_an_error(self) -> None:
        """Cache hits and timeout placeholders both reach here."""
        log: list[str] = []
        assert pipeline_module._count_decode_cap_truncations(
            [{}, {"asr_generation": None}], log=log
        ) == 0
