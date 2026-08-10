from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.align.measure_core_leading_silence import (  # noqa: E402
    SAMPLE_RATE,
    edge_silence_s,
    leading_silence_s,
)
from tools.align.sweep_edge_caps import (  # noqa: E402
    AUDIBLE_EARLY_S,
    quantized_cap_s,
    summarize,
)


def _clip(leading_s: float, speech_s: float, trailing_s: float) -> np.ndarray:
    rng = np.random.default_rng(7)

    def silence(seconds: float) -> np.ndarray:
        return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)

    speech = rng.normal(0.0, 0.3, int(speech_s * SAMPLE_RATE)).astype(np.float32)
    return np.concatenate([silence(leading_s), speech, silence(trailing_s)])


class TestEdgeSilence:
    def test_both_edges_are_measured_from_one_pass(self):
        leading, trailing = edge_silence_s(_clip(0.30, 1.00, 0.50))

        assert leading == pytest.approx(0.30, abs=0.02)
        assert trailing == pytest.approx(0.50, abs=0.02)

    def test_the_crossings_round_outward_at_both_edges(self):
        # Rounding inward would shrink the measured speech interval and hand the
        # head free credit at the head and free blame at the tail. Both edges
        # round toward "the speech lasted longer", so neither reported silence
        # may exceed the silence actually present.
        leading, trailing = edge_silence_s(_clip(0.255, 1.0, 0.255))

        assert leading <= 0.255 + 1e-9
        assert trailing <= 0.255 + 1e-9

    def test_silence_below_the_absolute_floor_measures_nothing(self):
        # A clip of pure silence has nothing to measure; returning 0.0 would say
        # "speech fills it", which is the opposite of the truth.
        assert edge_silence_s(np.zeros(SAMPLE_RATE, dtype=np.float32)) is None

    def test_the_trailing_edge_uses_the_full_clip_not_the_window_multiple(self):
        # A clip whose length is not a whole number of windows must not lose its
        # remainder, or every such clip reports less trailing silence than it has.
        width = int(0.010 * SAMPLE_RATE)
        clip = np.concatenate(
            [_clip(0.0, 0.5, 0.30), np.zeros(width // 2, dtype=np.float32)]
        )
        _, trailing = edge_silence_s(clip)

        assert trailing == pytest.approx(0.305, abs=0.011)

    def test_the_leading_only_wrapper_still_answers(self):
        assert leading_silence_s(_clip(0.20, 1.0, 0.4)) == pytest.approx(0.20, abs=0.02)
        assert leading_silence_s(np.zeros(SAMPLE_RATE, dtype=np.float32)) is None


class TestSweepSummaries:
    def test_reachable_cap_is_the_truncated_frame_count_not_the_request(self):
        # speech_extent turns the cap into whole frames by truncation, so a sweep
        # that reported the requested value would invent distinctions the walk
        # cannot make: at 38.46 ms/frame, 0.30 and 0.27 are the same seven frames.
        assert quantized_cap_s(0.30, upsample=2) == pytest.approx(0.26923, abs=1e-4)
        assert quantized_cap_s(0.27, upsample=2) == pytest.approx(0.26923, abs=1e-4)
        assert quantized_cap_s(0.10, upsample=2) == pytest.approx(0.07692, abs=1e-4)
        assert quantized_cap_s(0.0, upsample=2) == 0.0

    def test_a_negative_cap_reaches_nothing_rather_than_walking_backwards(self):
        assert quantized_cap_s(-1.0, upsample=2) == 0.0

    def test_the_two_directions_are_reported_separately(self):
        # Which direction is expensive flips between the edges, so a summary that
        # folded them into an absolute error could not size either cap.
        before = summarize([-0.1, -0.1, -0.1, -0.1])
        after = summarize([0.1, 0.1, 0.1, 0.1])

        assert before["share_before"] == 1.0 and before["share_after"] == 0.0
        assert after["share_after"] == 1.0 and after["share_before"] == 0.0
        assert before["median_ms"] == pytest.approx(-100.0)
        assert after["median_ms"] == pytest.approx(100.0)

    def test_the_audible_tail_uses_the_blind_audit_threshold(self):
        assert AUDIBLE_EARLY_S == pytest.approx(0.200)

        # Exactly 200 ms is not counted; the threshold is where it became
        # audible, so the tail must be what exceeds it.
        borderline = summarize([-0.200, -0.201])
        assert borderline["share_before_over_200ms"] == pytest.approx(0.5)

    def test_an_empty_arm_reports_no_count_rather_than_a_zero_median(self):
        assert summarize([]) == {"count": 0}

    def test_percentiles_keep_the_sign_convention(self):
        result = summarize([-0.3, -0.2, -0.1, 0.0, 0.1])

        assert result["p05_ms"] < result["p95_ms"]
        assert result["p05_ms"] == pytest.approx(-280.0, abs=1.0)
        assert result["count"] == 5
