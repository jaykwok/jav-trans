"""Translation batches use one cues-per-worker rule; ordering handles skew.

The remaining scheduling optimization was measured by replaying seven real cue
lists through a list-scheduling model:

  Order. Batches are id-addressed and independent, so which one starts first is
  free - but submitted in index order, the largest batch can be picked up last
  and the whole stage then ends one full large batch after the pool went idle.
  Longest-first removed 0.5-13.4% of the makespan. These tests pin that ordering;
  cues-per-worker sizing is covered with the translator batch tests.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from llm.engine import _batch_cost, _submission_order  # noqa: E402


def _batch(*texts: str) -> list[dict]:
    return [{"text": text} for text in texts]


class TestBatchCost:
    def test_cost_is_the_source_length(self) -> None:
        assert _batch_cost(_batch("あい", "うえお")) == 5

    def test_an_empty_batch_costs_nothing(self) -> None:
        assert _batch_cost([]) == 0

    def test_a_missing_text_field_is_not_an_error(self) -> None:
        """Cue dicts come from several producers; a KeyError here would fail a
        whole film over a scheduling hint."""
        assert _batch_cost([{"start": 0.0}]) == 0

    def test_batches_of_equal_count_can_differ_in_cost(self) -> None:
        """The reason count-based batching leaves an uneven queue: twelve long
        lines and twelve grunts are both twelve."""
        long_batch = _batch(*["これは長い台詞です" * 3] * 12)
        short_batch = _batch(*["あっ"] * 12)
        assert _batch_cost(long_batch) > 5 * _batch_cost(short_batch)


class TestSubmissionOrder:
    def test_the_most_expensive_batch_is_submitted_first(self) -> None:
        pending = [
            (0, _batch("短い")),
            (1, _batch("とても長い台詞がここにあります")),
            (2, _batch("ふつう")),
        ]
        assert [index for index, _ in _submission_order(pending)] == [1, 2, 0]

    def test_every_batch_is_still_submitted_exactly_once(self) -> None:
        """Reordering must not drop or duplicate work - the ids addressed by a
        lost batch would come back untranslated."""
        pending = [(index, _batch("あ" * (index % 5 + 1))) for index in range(20)]
        ordered = _submission_order(pending)
        assert sorted(index for index, _ in ordered) == list(range(20))
        assert len(ordered) == len(pending)

    def test_the_batches_themselves_are_untouched(self) -> None:
        pending = [(0, _batch("あ")), (1, _batch("い"))]
        by_index = {index: batch for index, batch in _submission_order(pending)}
        assert by_index[0] == _batch("あ")
        assert by_index[1] == _batch("い")

    def test_ties_keep_index_order(self) -> None:
        """Deterministic submission keeps a rerun's provider-side prefix cache
        warm in the same order it was warmed."""
        pending = [(index, _batch("あ")) for index in range(5)]
        assert [index for index, _ in _submission_order(pending)] == list(range(5))

    def test_an_empty_queue_is_not_an_error(self) -> None:
        assert _submission_order([]) == []


class TestMakespan:
    """The property the ordering exists for, on the shape that shows it."""

    @staticmethod
    def _makespan(order: list[tuple[int, list[dict]]], workers: int) -> int:
        finish = [0] * workers
        for _index, batch in order:
            slot = min(range(workers), key=lambda item: finish[item])
            finish[slot] += _batch_cost(batch)
        return max(finish)

    def test_longest_first_is_never_worse_on_a_realistic_queue(self) -> None:
        pending = [(index, _batch("あ" * (1 + (index * 7) % 30))) for index in range(40)]
        workers = 8
        assert self._makespan(_submission_order(pending), workers) <= self._makespan(
            pending, workers
        )

    def test_the_worst_case_for_index_order_is_fixed(self) -> None:
        """One big batch last: index order runs it after the pool has drained,
        longest-first runs it alongside everything else."""
        pending = [(index, _batch("あ")) for index in range(7)]
        pending.append((7, _batch("あ" * 20)))
        assert self._makespan(pending, 4) == 21
        assert self._makespan(_submission_order(pending), 4) == 20

    @pytest.mark.parametrize("workers", [2, 4, 8, 16])
    def test_it_holds_across_pool_sizes(self, workers: int) -> None:
        pending = [(index, _batch("あ" * (1 + (index * 13) % 17))) for index in range(50)]
        assert self._makespan(_submission_order(pending), workers) <= self._makespan(
            pending, workers
        )
