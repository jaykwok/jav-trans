"""The pre-gate decides what never gets transcribed, so its bias must be fixed.

Every test here is about the asymmetry rather than about accuracy: audio the
gate skips is unrecoverable, audio it keeps in error is filtered later. So the
properties worth pinning are the ones that make the gate err toward keeping, and
the ones that stop it emitting geometry the decoder cannot use.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.chunking import cut_at_pauses  # noqa: E402
from tools.align.pregate_reference import (  # noqa: E402
    PreGateConfig,
    covered_seconds,
    duration,
    speech_regions,
)

PLAIN = PreGateConfig(
    min_blank_s=0.0, min_speech_s=0.0, merge_gap_s=0.0, pad_s=0.0, max_region_s=1e6
)


class TestComplement:
    def test_regions_are_the_gaps_between_pauses(self) -> None:
        assert speech_regions([(2.0, 4.0), (7.0, 9.0)], 12.0, PLAIN) == [
            (0.0, 2.0),
            (4.0, 7.0),
            (9.0, 12.0),
        ]

    def test_all_blank_audio_yields_nothing_to_decode(self) -> None:
        assert speech_regions([(0.0, 30.0)], 30.0, PLAIN) == []

    def test_no_pause_means_decode_everything(self) -> None:
        assert speech_regions([], 30.0, PLAIN) == [(0.0, 30.0)]

    def test_pauses_are_clamped_into_the_window(self) -> None:
        """A run reported past the end must not create a negative region."""
        assert speech_regions([(-5.0, 2.0), (8.0, 99.0)], 10.0, PLAIN) == [(2.0, 8.0)]

    def test_an_empty_window_decodes_nothing(self) -> None:
        assert speech_regions([], 0.0, PLAIN) == []


class TestKeepBias:
    def test_a_short_pause_does_not_split_an_utterance(self) -> None:
        """`min_blank_s` is applied upstream in `blank_runs`; `merge_gap_s` is
        the second line of defence for a pause that only just cleared it."""
        config = PreGateConfig(
            min_speech_s=0.0, merge_gap_s=0.4, pad_s=0.0, max_region_s=1e6
        )
        assert speech_regions([(3.0, 3.3)], 6.0, config) == [(0.0, 6.0)]

    def test_bridging_happens_before_the_length_rule(self) -> None:
        """The ordering that a naive implementation gets wrong.

        Two 0.7 s halves either side of a 0.2 s pause are one 1.6 s utterance.
        Dropping fragments first would discard both and lose the whole line.
        """
        config = PreGateConfig(
            min_speech_s=1.0, merge_gap_s=0.4, pad_s=0.0, max_region_s=1e6
        )
        assert speech_regions([(0.7, 0.9)], 1.6, config) == [(0.0, 1.6)]

    def test_padding_does_not_rescue_a_rejected_fragment(self) -> None:
        """The other ordering, and it has to go the other way.

        Padding is there to widen lines that survived, not to promote 0.2 s of
        noise into a decode call by inflating it to 0.5 s.
        """
        config = PreGateConfig(
            min_speech_s=1.0, merge_gap_s=0.0, pad_s=0.15, max_region_s=1e6
        )
        assert speech_regions([(0.0, 5.0), (5.2, 10.0)], 10.0, config) == []

    def test_padding_widens_a_surviving_region_both_ways(self) -> None:
        config = PreGateConfig(
            min_speech_s=1.0, merge_gap_s=0.0, pad_s=0.15, max_region_s=1e6
        )
        assert speech_regions([(0.0, 2.0), (5.0, 8.0)], 8.0, config) == [
            (pytest.approx(1.85), pytest.approx(5.15))
        ]

    def test_padding_never_leaves_the_window(self) -> None:
        config = PreGateConfig(
            min_speech_s=0.0, merge_gap_s=0.0, pad_s=1.0, max_region_s=1e6
        )
        assert speech_regions([], 4.0, config) == [(0.0, 4.0)]

    def test_padding_that_collides_merges_instead_of_overlapping(self) -> None:
        """Overlapping regions would decode the same audio twice and produce two
        cues for one utterance; dropping one would lose speech. Merge is the
        only option that does neither."""
        config = PreGateConfig(
            min_speech_s=0.0, merge_gap_s=0.0, pad_s=0.2, max_region_s=1e6
        )
        regions = speech_regions([(2.0, 2.3)], 5.0, config)
        assert regions == [(0.0, 5.0)]


class TestDecoderGeometry:
    def test_a_long_region_is_split_into_equal_pieces(self) -> None:
        config = PreGateConfig(
            min_speech_s=0.0, merge_gap_s=0.0, pad_s=0.0, max_region_s=20.0
        )
        regions = speech_regions([], 50.0, config)
        assert len(regions) == 3
        assert regions[0][0] == 0.0 and regions[-1][1] == pytest.approx(50.0)
        widths = [end - begin for begin, end in regions]
        assert widths == [pytest.approx(50.0 / 3)] * 3

    def test_splitting_leaves_no_gaps(self) -> None:
        config = PreGateConfig(
            min_speech_s=0.0, merge_gap_s=0.0, pad_s=0.0, max_region_s=7.0
        )
        regions = speech_regions([], 30.0, config)
        for earlier, later in zip(regions, regions[1:]):
            assert earlier[1] == pytest.approx(later[0])

    def test_regions_are_sorted_and_disjoint(self) -> None:
        config = PreGateConfig(min_speech_s=0.5, pad_s=0.3)
        regions = speech_regions(
            [(1.0, 2.0), (9.0, 9.9), (14.0, 15.0)], 40.0, config
        )
        for earlier, later in zip(regions, regions[1:]):
            assert earlier[1] <= later[0] + 1e-9
        assert regions == sorted(regions)


class TestConfigContract:
    def test_negative_geometry_is_refused(self) -> None:
        for field in ("min_blank_s", "min_speech_s", "merge_gap_s", "pad_s"):
            with pytest.raises(ValueError, match=field):
                PreGateConfig(**{field: -0.1})

    def test_a_split_shorter_than_the_length_rule_is_refused(self) -> None:
        """Otherwise splitting emits pieces the gate has already called too
        short to be lines, and the two rules silently contradict."""
        with pytest.raises(ValueError, match="max_region_s"):
            PreGateConfig(min_speech_s=5.0, max_region_s=2.0)

    def test_the_defaults_are_the_ones_the_pilot_arrived_at(self) -> None:
        """`min_blank_s=0.35` cut inside utterances and left 1 s fragments that
        mostly failed to align. Pinned so it cannot drift back."""
        config = PreGateConfig()
        assert config.min_blank_s == 0.6
        assert config.min_speech_s == 1.0
        assert config.merge_gap_s == 0.4
        assert config.pad_s == 0.15


class TestOverlapAccounting:
    def test_a_span_straddling_a_region_edge_is_counted_in_part(self) -> None:
        """The whole point of measuring in seconds rather than in spans."""
        assert covered_seconds([(1.0, 5.0)], [(3.0, 9.0)]) == pytest.approx(2.0)

    def test_a_span_split_across_two_regions_counts_both_pieces(self) -> None:
        assert covered_seconds([(0.0, 10.0)], [(1.0, 2.0), (6.0, 8.0)]) == (
            pytest.approx(3.0)
        )

    def test_overlapping_regions_are_not_double_counted(self) -> None:
        assert covered_seconds([(0.0, 10.0)], [(1.0, 5.0), (3.0, 7.0)]) == (
            pytest.approx(6.0)
        )

    def test_no_overlap_is_zero_not_an_error(self) -> None:
        assert covered_seconds([(0.0, 1.0)], [(5.0, 6.0)]) == 0.0
        assert covered_seconds([], [(0.0, 1.0)]) == 0.0
        assert covered_seconds([(0.0, 1.0)], []) == 0.0

    def test_duration_sums_spans(self) -> None:
        assert duration([(0.0, 1.5), (2.0, 2.25)]) == pytest.approx(1.75)


class TestLosslessCutting:
    """The reading of the blank runs that survived falsification.

    `speech_regions` decides what not to decode and was measured to lose real
    lines on this domain. `cut_at_pauses` uses the same signal only to choose
    boundaries, and the invariant that makes it safe is that it tiles the audio
    exactly - so every test here is about coverage, not about placement.
    """

    def test_the_chunks_tile_the_audio_exactly(self) -> None:
        chunks = cut_at_pauses([(21.0, 22.0), (44.0, 45.0)], 70.0)
        assert chunks[0][0] == 0.0
        assert chunks[-1][1] == pytest.approx(70.0)
        for earlier, later in zip(chunks, chunks[1:]):
            assert earlier[1] == pytest.approx(later[0])

    def test_no_audio_is_lost_even_with_no_pauses(self) -> None:
        chunks = cut_at_pauses([], 95.0, max_s=30.0)
        assert sum(end - begin for begin, end in chunks) == pytest.approx(95.0)
        assert all(end - begin <= 30.0 + 1e-9 for begin, end in chunks)

    def test_no_audio_is_lost_when_pauses_are_everywhere(self) -> None:
        blanks = [(float(t), float(t) + 0.5) for t in range(1, 90)]
        chunks = cut_at_pauses(blanks, 90.0)
        assert sum(end - begin for begin, end in chunks) == pytest.approx(90.0)

    def test_short_audio_is_a_single_chunk(self) -> None:
        assert cut_at_pauses([(3.0, 4.0)], 12.0, max_s=30.0) == [(0.0, 12.0)]

    def test_a_cut_lands_in_the_middle_of_a_pause(self) -> None:
        """Not at its edge - that is where it does least damage to either side."""
        chunks = cut_at_pauses([(20.0, 21.0)], 45.0, max_s=30.0)
        assert chunks[0][1] == pytest.approx(20.5)

    def test_the_latest_pause_under_the_ceiling_is_chosen(self) -> None:
        """The decode window is the thing being maximised, not evenness.

        A 20s target picked the middle pause here, and that cost transcription
        quality on 2026-08-02: `max_s` is the encoder's audio window and shorter
        chunks are padded up to it anyway, so the context given away buys nothing.
        """
        chunks = cut_at_pauses(
            [(3.0, 3.2), (19.8, 20.2), (28.0, 28.4)], 60.0, max_s=30.0
        )
        assert chunks[0][1] == pytest.approx(28.2)

    def test_a_pause_past_the_ceiling_is_not_reachable(self) -> None:
        """Greedy must stay inside `max_s`, or a chunk outgrows the encoder."""
        chunks = cut_at_pauses([(9.0, 9.4), (31.0, 31.4)], 60.0, max_s=30.0)
        assert chunks[0][1] == pytest.approx(9.2)

    def test_a_hard_cut_is_taken_when_no_pause_is_available(self) -> None:
        chunks = cut_at_pauses([(0.5, 0.7)], 60.0, max_s=25.0)
        assert chunks[0][1] == pytest.approx(25.0)

    def test_a_sliver_tail_is_merged_backwards(self) -> None:
        chunks = cut_at_pauses([], 61.0, max_s=30.0, min_s=2.0)
        assert chunks[-1][1] == pytest.approx(61.0)
        assert all(end - begin >= 2.0 for begin, end in chunks)

    def test_empty_audio_yields_nothing(self) -> None:
        assert cut_at_pauses([(0.0, 1.0)], 0.0) == []

    def test_incoherent_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="min_s"):
            cut_at_pauses([], 100.0, min_s=40.0, max_s=30.0)

    def test_the_separate_target_length_is_gone(self) -> None:
        """It was a knob that never ran until the head was configured, and then
        it silently shortened every decode window. Pinned so it cannot return."""
        with pytest.raises(TypeError):
            cut_at_pauses([], 100.0, target_s=20.0, max_s=30.0)
