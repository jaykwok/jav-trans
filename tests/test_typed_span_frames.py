"""Frame materialization must never invent supervision it was not given.

The typed-span sources supervise to three different depths, and each one has a
distinct way of turning into a silent lie:

  * a real window is only ~55% covered, so uncovered time must stay IGNORE - as
    non-speech it would manufacture ~7 h of fake negatives across the corpus
  * a real `ambiguous_ignore` chunk knows neither track and must stay IGNORE on
    both, rather than defaulting to non-speech
  * a synthetic gap knows `speech=False` but no subtype, so it must supervise
    the binary track while leaving the 3-way track IGNORE

Event scoring has the matching hazard: a truth run that straddles an uncovered
gap must not be bridged into one event, or the metric credits a span the labels
never asserted.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.boundary.ja.train_typed_span_falsification import (  # noqa: E402
    _runs,
    event_metrics,
)
from tools.datasets.materialize_typed_span_frames import (  # noqa: E402
    IGNORE_INDEX,
    TYPE_TO_INDEX,
    frame_labels,
)


def _example(spans: list[dict], *, hop: float = 0.02) -> dict:
    return {"frame_hop_s": hop, "spans": spans}


def test_uncovered_time_stays_ignore_on_both_tracks() -> None:
    example = _example(
        [{"start_s": 0.2, "end_s": 0.4, "type": "speech", "speech": True}]
    )
    speech, types = frame_labels(example, 50, stats=Counter())

    assert np.all(speech[10:20] == 1)
    assert np.all(types[10:20] == TYPE_TO_INDEX["speech"])
    # Everything outside the single labelled span was never asserted.
    assert np.all(speech[:10] == IGNORE_INDEX)
    assert np.all(speech[20:] == IGNORE_INDEX)
    assert np.all(types[:10] == IGNORE_INDEX)
    assert np.all(types[20:] == IGNORE_INDEX)


def test_real_ambiguous_span_supervises_neither_track() -> None:
    example = _example(
        [{"start_s": 0.0, "end_s": 0.2, "type": "unsure", "speech": None}]
    )
    stats: Counter = Counter()
    speech, types = frame_labels(example, 20, stats=stats)

    assert np.all(speech == IGNORE_INDEX)
    assert np.all(types == IGNORE_INDEX)
    assert stats["span_speech_unknown"] == 1


def test_synthetic_gap_supervises_binary_but_not_subtype() -> None:
    """`speech=False` with `type=unsure` is the shape of all 116 h of synthetic."""
    example = _example(
        [
            {"start_s": 0.0, "end_s": 0.1, "type": "unsure", "speech": False},
            {"start_s": 0.1, "end_s": 0.2, "type": "speech", "speech": True},
        ]
    )
    speech, types = frame_labels(example, 10, stats=Counter())

    assert np.all(speech[:5] == 0)
    assert np.all(speech[5:] == 1)
    assert np.all(types[:5] == IGNORE_INDEX)
    assert np.all(types[5:] == TYPE_TO_INDEX["speech"])


def test_drop_subtypes_land_on_distinct_indices() -> None:
    example = _example(
        [
            {"start_s": 0.0, "end_s": 0.1, "type": "non_semantic_vocal", "speech": False},
            {"start_s": 0.1, "end_s": 0.2, "type": "non_vocal", "speech": False},
        ]
    )
    speech, types = frame_labels(example, 10, stats=Counter())

    assert np.all(speech == 0)
    assert np.all(types[:5] == TYPE_TO_INDEX["non_semantic_vocal"])
    assert np.all(types[5:] == TYPE_TO_INDEX["non_vocal"])


def test_spans_are_clipped_to_the_feature_grid() -> None:
    """Labels may run a rounding hair past the decoded audio; frames may not."""
    example = _example(
        [{"start_s": 0.0, "end_s": 1.0, "type": "speech", "speech": True}]
    )
    speech, types = frame_labels(example, 10, stats=Counter())

    assert speech.shape == (10,)
    assert np.all(speech == 1)


def test_sub_frame_span_is_counted_not_silently_dropped() -> None:
    example = _example(
        [{"start_s": 0.001, "end_s": 0.002, "type": "speech", "speech": True}]
    )
    stats: Counter = Counter()
    speech, _ = frame_labels(example, 10, stats=stats)

    assert stats["span_collapsed_on_grid"] == 1
    assert np.all(speech == IGNORE_INDEX)


def test_runs_finds_half_open_intervals() -> None:
    mask = np.array([0, 1, 1, 0, 1, 0], dtype=bool)
    assert _runs(mask) == [(1, 3), (4, 5)]
    assert _runs(np.zeros(4, dtype=bool)) == []
    assert _runs(np.ones(3, dtype=bool)) == [(0, 3)]


def test_event_scoring_does_not_bridge_an_uncovered_gap() -> None:
    """Two truth runs split by unknown time stay two events, not one."""
    truth = np.array([1, 1, IGNORE_INDEX, IGNORE_INDEX, 1, 1])
    predicted = np.array([1, 1, 1, 1, 1, 1])
    known = truth != IGNORE_INDEX

    counts = event_metrics(truth, predicted, known, iou=0.5)

    assert counts["tp"] == 2
    assert counts["fp"] == 0
    assert counts["fn"] == 0


def test_event_scoring_penalizes_a_missed_run() -> None:
    truth = np.array([1, 1, 1, 0, 0, 1, 1, 1])
    predicted = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    known = np.ones(8, dtype=bool)

    counts = event_metrics(truth, predicted, known, iou=0.5)

    assert counts["tp"] == 1
    assert counts["fn"] == 1
    assert counts["fp"] == 0


def test_event_scoring_rejects_a_low_overlap_match() -> None:
    truth = np.array([1] * 10)
    predicted = np.array([1] + [0] * 9)
    known = np.ones(10, dtype=bool)

    counts = event_metrics(truth, predicted, known, iou=0.5)

    assert counts["tp"] == 0
    assert counts["fn"] == 1
    assert counts["fp"] == 1


@pytest.mark.parametrize("hop", [0.01, 0.02, 0.04])
def test_rasterization_follows_the_declared_hop(hop: float) -> None:
    example = _example(
        [{"start_s": 0.0, "end_s": hop * 3, "type": "speech", "speech": True}], hop=hop
    )
    speech, _ = frame_labels(example, 10, stats=Counter())

    assert int(np.count_nonzero(speech == 1)) == 3
