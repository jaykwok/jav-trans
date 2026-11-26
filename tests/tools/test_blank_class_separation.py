from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.align.measure_blank_class_separation import (  # noqa: E402
    SILENT,
    VOICED_WORDLESS,
    WORD,
    _summarize,
    classify_frames,
    energy_threshold,
    frame_energy,
)


FRAME_S = 0.05


def test_frame_energy_reports_the_peak_not_the_mean_inside_a_frame():
    # One 10 ms burst inside an otherwise quiet 50 ms frame. Averaging would
    # dilute it to a fifth and report the frame as a pause.
    sample_rate = 16000
    audio = np.zeros(int(sample_rate * FRAME_S * 2), dtype=np.float32)
    burst = int(sample_rate * 0.010)
    audio[:burst] = 0.5

    energy = frame_energy(
        audio, frame_count=2, frame_s=FRAME_S, sample_rate=sample_rate
    )

    assert energy is not None
    assert energy[0] == pytest.approx(0.5)
    assert energy[1] == 0.0


def test_energy_threshold_refuses_a_clip_that_is_only_noise():
    # A clip whose loudest window is under the absolute floor has nothing to
    # measure; calling its own noise "voiced" would invent the class.
    quiet = np.full(8, 10.0 ** (-70.0 / 20.0), dtype=np.float64)

    assert energy_threshold(quiet, relative_db=-35.0, floor_dbfs=-55.0) is None


def test_energy_threshold_is_relative_to_the_clip_peak_but_floored():
    energy = np.array([0.0, 1.0], dtype=np.float64)

    relative = energy_threshold(energy, relative_db=-20.0, floor_dbfs=-100.0)
    floored = energy_threshold(energy, relative_db=-80.0, floor_dbfs=-40.0)

    assert relative == pytest.approx(0.1)
    assert floored == pytest.approx(10.0 ** (-40.0 / 20.0))


def _masks_for_one_word(*, boundary_ignore_s: float, long_gap_min_s: float = 0.5):
    # 20 frames of 50 ms = 1.0 s. One word over [0.4, 0.6).
    frame_count = 20
    energy = np.zeros(frame_count, dtype=np.float64)
    # Voiced-but-wordless region at [0.7, 0.9): a moan the teacher ignored.
    energy[14:18] = 1.0
    energy[8:12] = 1.0  # the word itself is loud too
    return classify_frames(
        frame_count=frame_count,
        frame_s=FRAME_S,
        duration_s=1.0,
        islands=[(0.4, 0.6)],
        energy=energy,
        threshold=0.5,
        boundary_ignore_s=boundary_ignore_s,
        long_gap_min_s=long_gap_min_s,
    )


def test_word_edges_belong_to_no_class():
    masks = _masks_for_one_word(boundary_ignore_s=0.10)

    # Frames whose centres sit in [0.3, 0.4) and [0.6, 0.7) are the teacher's
    # least trustworthy region and must not move either rate.
    assigned = masks[WORD] | masks[VOICED_WORDLESS] | masks[SILENT]
    assert not assigned[6:8].any()
    assert not assigned[12:14].any()


def test_the_three_classes_partition_everything_they_cover():
    masks = _masks_for_one_word(boundary_ignore_s=0.10)

    assert not (masks[WORD] & masks[VOICED_WORDLESS]).any()
    assert not (masks[WORD] & masks[SILENT]).any()
    assert not (masks[VOICED_WORDLESS] & masks[SILENT]).any()
    assert masks[WORD].sum() == 4
    assert masks[VOICED_WORDLESS].sum() == 4
    # Everything else inside the clip and outside the ignore margins.
    assert masks[SILENT].sum() == 20 - 4 - 4 - 4


def test_long_gap_subset_excludes_short_gaps():
    masks = _masks_for_one_word(boundary_ignore_s=0.10, long_gap_min_s=0.5)

    # The leading gap is [0, 0.3) = 0.3 s and the trailing one [0.7, 1.0) =
    # 0.3 s, so neither reaches the 0.5 s regime the blank labels were trained
    # on and the subset must be empty rather than silently equal to all gaps.
    assert not masks["long_gap"].any()

    relaxed = _masks_for_one_word(boundary_ignore_s=0.10, long_gap_min_s=0.25)
    assert relaxed["long_gap"].any()


def test_a_pure_silence_detector_shows_a_zero_non_semantic_margin():
    # The head answers "is it quiet", not "is it a word": blank on silence,
    # non-blank on anything voiced including the moan. This is the reading that
    # says the gate cannot separate the two, and it must be visible as a near
    # zero margin_vs_non_semantic_pp with a large silence_over_voiced_pp.
    pooled = {
        WORD: {"frames": 100.0, "blank": 2.0, "probability": 2.0},
        VOICED_WORDLESS: {"frames": 100.0, "blank": 3.0, "probability": 3.0},
        SILENT: {"frames": 100.0, "blank": 98.0, "probability": 98.0},
    }

    summary = _summarize(pooled, {}, FRAME_S)

    assert summary["margins_pp"]["margin_vs_non_semantic_pp"] == 1.0
    assert summary["margins_pp"]["silence_over_voiced_pp"] == 95.0
    assert summary["margins_pp"]["margin_vs_silence_pp"] == 96.0


def test_missing_class_reports_none_rather_than_zero():
    # A clip set with no voiced-wordless frames has not measured a zero margin;
    # it has not measured the margin at all.
    pooled = {
        WORD: {"frames": 10.0, "blank": 1.0, "probability": 1.0},
        VOICED_WORDLESS: {"frames": 0.0, "blank": 0.0, "probability": 0.0},
        SILENT: {"frames": 10.0, "blank": 9.0, "probability": 9.0},
    }

    summary = _summarize(pooled, {}, FRAME_S)

    assert summary["margins_pp"]["margin_vs_non_semantic_pp"] is None
    assert summary[VOICED_WORDLESS]["argmax_blank_rate"] is None
