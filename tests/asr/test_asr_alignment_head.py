"""Contract tests for the CTC alignment head.

The failure mode this guards against is not a crash. An aligner that is subtly
wrong - off by a frame, or silently reordering - emits timestamps that look
entirely plausible and are simply not true, and there is no downstream check
that would catch it: the subtitle layer will happily render whatever times it is
given. So the alignment tests build posteriors whose correct answer is known by
construction and demand the exact spans back, rather than asserting that
something ran.

The frame-rate and blank-index tests exist because both numbers are shared
contracts. 13 fps comes from the encoder, not from this module, and blank being
index 0 is what lets the gate read "no words here" off the argmax.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

torch = pytest.importorskip("torch")

from asr import alignment  # noqa: E402

VOCAB = alignment.AlignmentVocab(chars=tuple("あいうえおかきくけこ"))


def _posteriors(frame_tokens: list[int], *, vocab_size: int, confidence: float = 0.9):
    """Per-frame log-probs that put `confidence` on the token named per frame."""
    probs = torch.full((len(frame_tokens), vocab_size), 0.0)
    spread = (1.0 - confidence) / (vocab_size - 1)
    probs.fill_(spread)
    for frame, token in enumerate(frame_tokens):
        probs[frame, token] = confidence
    return probs.log()


class TestFrameGeometry:
    def test_the_frame_rate_matches_the_encoder(self) -> None:
        """13 fps is the encoder's number; this module must not invent its own."""
        from asr.encoder_features import qwen3_asr_audio_output_lengths

        # 100 mel frames is one second of audio at the extractor's hop.
        one_second = qwen3_asr_audio_output_lengths(torch.tensor([100]))
        assert int(one_second[0]) == alignment.ENCODER_FPS

        thirty_seconds = qwen3_asr_audio_output_lengths(torch.tensor([3000]))
        assert int(thirty_seconds[0]) == int(alignment.ENCODER_FPS * 30)

    def test_upsampling_divides_the_frame_duration(self) -> None:
        assert alignment.frame_to_seconds(13, upsample=1) == pytest.approx(1.0)
        assert alignment.frame_to_seconds(26, upsample=2) == pytest.approx(1.0)
        assert alignment.frame_to_seconds(52, upsample=4) == pytest.approx(1.0)
        # The resolution claim in the module docstring, stated as a test.
        assert alignment.frame_to_seconds(1, upsample=1) == pytest.approx(0.0769, abs=1e-4)
        assert alignment.frame_to_seconds(1, upsample=2) == pytest.approx(0.0385, abs=1e-4)

    def test_the_head_emits_one_frame_per_upsampled_input_frame(self) -> None:
        for upsample in (1, 2, 4):
            head = alignment.build_head(vocab_size=VOCAB.size, upsample=upsample, blocks=1)
            output = head(torch.zeros(2, 17, 2048))
            assert output.shape[0] == 2
            assert output.shape[1] == alignment.output_frame_count(17, upsample=upsample)
            assert output.shape[1] == 17 * upsample
            assert output.shape[2] == VOCAB.size

    def test_the_head_returns_normalised_log_probabilities(self) -> None:
        head = alignment.build_head(vocab_size=VOCAB.size, upsample=2, blocks=1)
        output = head(torch.randn(1, 9, 2048))
        assert torch.allclose(output.exp().sum(dim=-1), torch.ones(1, 18), atol=1e-5)


class TestVocab:
    def test_blank_is_zero_and_unknown_is_one(self) -> None:
        """The gate reads `argmax == 0` directly; moving blank breaks it."""
        assert alignment.BLANK_INDEX == 0
        assert alignment.UNK_INDEX == 1
        assert VOCAB.index_of("あ") >= alignment.RESERVED_INDICES
        assert VOCAB.char_at(alignment.BLANK_INDEX) == ""

    def test_out_of_vocabulary_characters_keep_their_frame(self) -> None:
        """Dropping an unknown character would shift every later timestamp."""
        encoded = VOCAB.encode("あXい")
        assert len(encoded) == 3
        assert encoded[1] == alignment.UNK_INDEX

    def test_encoding_folds_width_and_drops_whitespace(self) -> None:
        assert alignment.normalize_text("　あ い\tう\n") == "あいう"
        assert alignment.normalize_text("ＡＢ１") == "AB1"
        assert len(VOCAB.encode("あ い")) == 2

    def test_frequency_order_makes_truncation_a_coverage_decision(self) -> None:
        counts = {"あ": 5, "い": 100, "う": 1}
        assert alignment.AlignmentVocab.from_counts(counts).chars == ("い", "あ", "う")
        assert alignment.AlignmentVocab.from_counts(counts, max_size=2).chars == ("い", "あ")
        assert alignment.AlignmentVocab.from_counts(counts, min_count=5).chars == ("い", "あ")

    def test_a_checkpoint_that_moved_blank_is_refused(self) -> None:
        payload = VOCAB.to_payload()
        assert alignment.AlignmentVocab.from_payload(payload).chars == VOCAB.chars
        for field in ("blank_index", "unk_index"):
            broken = dict(payload)
            broken[field] = 7
            with pytest.raises(ValueError, match=field):
                alignment.AlignmentVocab.from_payload(broken)
        wrong_schema = dict(payload, schema="something_else")
        with pytest.raises(ValueError, match="schema"):
            alignment.AlignmentVocab.from_payload(wrong_schema)

    def test_the_classifier_width_covers_the_reserved_indices(self) -> None:
        assert VOCAB.size == len(VOCAB.chars) + 2
        with pytest.raises(ValueError, match="vocab_size"):
            alignment.build_head(vocab_size=2, upsample=1, blocks=1)


class TestForcedAlignment:
    def test_it_recovers_spans_that_were_built_into_the_posteriors(self) -> None:
        """The decisive test: known audio-to-character layout, exact spans back."""
        a, i = VOCAB.index_of("あ"), VOCAB.index_of("い")
        blank = alignment.BLANK_INDEX
        # frames:  0-1 blank | 2-4 あ | 5 blank | 6-7 い | 8-9 blank
        frames = [blank, blank, a, a, a, blank, i, i, blank, blank]
        spans = alignment.align_text(
            _posteriors(frames, vocab_size=VOCAB.size), "あい", VOCAB, upsample=1
        )
        assert [(s.char, s.start_frame, s.end_frame) for s in spans] == [
            ("あ", 2, 5),
            ("い", 6, 8),
        ]
        assert spans[0].start_s == pytest.approx(2 / 13.0)
        assert spans[1].end_s == pytest.approx(8 / 13.0)

    def test_repeated_characters_stay_separated_by_a_blank(self) -> None:
        """Without the mandatory blank, 'ああ' would collapse to one span."""
        a, blank = VOCAB.index_of("あ"), alignment.BLANK_INDEX
        frames = [a, a, blank, a, a]
        spans = alignment.align_text(
            _posteriors(frames, vocab_size=VOCAB.size), "ああ", VOCAB, upsample=1
        )
        assert len(spans) == 2
        assert spans[0].end_frame <= spans[1].start_frame
        assert (spans[0].start_frame, spans[1].start_frame) == (0, 3)

    def test_spans_are_monotonic_and_never_overlap(self) -> None:
        text = "あいうえおかきくけこ"
        torch.manual_seed(0)
        log_probs = torch.randn(120, VOCAB.size).log_softmax(dim=-1)
        spans = alignment.align_text(log_probs, text, VOCAB, upsample=2)
        assert [s.char for s in spans] == list(text)
        for earlier, later in zip(spans, spans[1:]):
            assert earlier.end_frame <= later.start_frame
            assert earlier.end_s <= later.start_s
        assert spans[0].start_frame >= 0
        assert spans[-1].end_frame <= 240

    def test_every_character_receives_at_least_one_frame(self) -> None:
        torch.manual_seed(1)
        text = "あいうえお"
        log_probs = torch.randn(11, VOCAB.size).log_softmax(dim=-1)
        spans = alignment.align_text(log_probs, text, VOCAB, upsample=1)
        assert len(spans) == len(text)
        assert all(s.end_frame > s.start_frame for s in spans)

    def test_text_too_long_for_the_audio_is_refused_not_guessed(self) -> None:
        """Silently truncating here would emit timestamps for unspoken text."""
        log_probs = torch.randn(3, VOCAB.size).log_softmax(dim=-1)
        with pytest.raises(ValueError, match="cannot align"):
            alignment.align_text(log_probs, "あいうえおかきくけこ", VOCAB, upsample=1)

    def test_alignment_score_separates_supported_text_from_unsupported(self) -> None:
        """S5's hallucination signal: text the acoustics contradict scores low."""
        a, i = VOCAB.index_of("あ"), VOCAB.index_of("い")
        supported = _posteriors([a, a, a, i, i, i], vocab_size=VOCAB.size)
        spans_good = alignment.align_text(supported, "あい", VOCAB, upsample=1)
        spans_bad = alignment.align_text(supported, "かき", VOCAB, upsample=1)
        good = sum(s.score for s in spans_good) / len(spans_good)
        bad = sum(s.score for s in spans_bad) / len(spans_bad)
        assert good > bad
        assert good > -1.0 and bad < -3.0

    def test_blanks_may_not_be_aligned_as_targets(self) -> None:
        log_probs = torch.randn(10, VOCAB.size).log_softmax(dim=-1)
        with pytest.raises(ValueError, match="blank"):
            alignment.forced_align(log_probs, [alignment.BLANK_INDEX], upsample=1)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_viterbi_matches_exhaustive_enumeration(self, seed: int) -> None:
        """The Viterbi must find the true optimum, checked by brute force.

        This is the test torchaudio's `forced_align` would otherwise have
        provided as a reference implementation - it cannot be installed here
        (Python 3.14 against a cu132 index that stops at cp312), so the
        reference is exhaustive enumeration instead. On inputs small enough to
        enumerate, that is strictly better than agreeing with another library:
        it compares against the definition rather than against someone else's
        implementation of it.
        """
        torch.manual_seed(seed)
        targets = [VOCAB.index_of(ch) for ch in "あいあ"]
        frames = 9
        log_probs = torch.randn(frames, VOCAB.size).log_softmax(dim=-1)

        extended = [alignment.BLANK_INDEX]
        for token in targets:
            extended.extend((token, alignment.BLANK_INDEX))
        states = len(extended)

        best_score = float("-inf")
        best_path: list[int] = []

        def walk(frame: int, state: int, score: float, path: list[int]) -> None:
            nonlocal best_score, best_path
            score += float(log_probs[frame, extended[state]])
            path = path + [state]
            if frame == frames - 1:
                if state >= states - 2 and score > best_score:
                    best_score, best_path = score, path
                return
            for step in (0, 1, 2):
                nxt = state + step
                if nxt >= states:
                    continue
                if step == 2 and (
                    extended[nxt] == alignment.BLANK_INDEX
                    or extended[nxt] == extended[nxt - 2]
                ):
                    continue
                walk(frame + 1, nxt, score, path)

        walk(0, 0, 0.0, [])
        walk(0, 1, 0.0, [])
        assert best_path, "enumeration found no valid path"

        expected = []
        for label_index in range(len(targets)):
            state_index = 2 * label_index + 1
            occupied = [t for t, s in enumerate(best_path) if s == state_index]
            expected.append((occupied[0], occupied[-1] + 1))

        spans = alignment.forced_align(log_probs, targets, upsample=1)
        assert [(s.start_frame, s.end_frame) for s in spans] == expected


class TestMinimumFrames:
    """The feasibility bound, which was wrong in both directions.

    The old guard was `frames < states - len(targets)`, i.e. `T < L + 1`. That
    number is not the CTC minimum: leading and trailing blanks are optional, so
    a sequence of distinct labels needs only `L` frames, while every pair of
    adjacent identical labels needs a mandatory blank between them and so costs
    one extra. Both errors are silent in different ways - the first refuses a
    clip that aligns fine, the second lets an impossible one through the gate and
    fails several hundred lines later in the backtrace.
    """

    def test_distinct_labels_need_one_frame_each(self) -> None:
        assert alignment.minimum_ctc_frames([2, 3]) == 2
        assert alignment.minimum_ctc_frames("あいうえお") == 5

    def test_each_adjacent_repeat_costs_a_mandatory_blank(self) -> None:
        assert alignment.minimum_ctc_frames([2, 2]) == 3
        assert alignment.minimum_ctc_frames([2, 2, 2]) == 5
        # Only *adjacent* pairs: あいあ repeats a character but never twice in a
        # row, so no blank is forced.
        assert alignment.minimum_ctc_frames("あいあ") == 3
        assert alignment.minimum_ctc_frames("ああい") == 4

    def test_an_empty_target_needs_no_frames(self) -> None:
        assert alignment.minimum_ctc_frames([]) == 0
        assert alignment.minimum_ctc_frames("") == 0

    def test_two_distinct_characters_in_two_frames_are_aligned(self) -> None:
        """Regression: `T = L` is legal and the old `T < L+1` guard rejected it."""
        a, i = VOCAB.index_of("あ"), VOCAB.index_of("い")
        log_probs = _posteriors([a, i], vocab_size=VOCAB.size)
        spans = alignment.forced_align(log_probs, [a, i], upsample=1)
        assert [(s.start_frame, s.end_frame) for s in spans] == [(0, 1), (1, 2)]

    def test_a_repeated_character_without_room_for_its_blank_is_refused(self) -> None:
        """Regression: あああ needs 5 frames, not 4. At 4 the old guard passed
        and the failure surfaced as `alignment path skipped character 0` from
        deep inside the backtrace, which reads like a corrupt lattice rather
        than like text that does not fit."""
        a = VOCAB.index_of("あ")
        log_probs = torch.randn(4, VOCAB.size).log_softmax(dim=-1)
        with pytest.raises(ValueError, match="cannot align"):
            alignment.forced_align(log_probs, [a, a, a], upsample=1)

    def test_a_repeated_character_with_room_aligns(self) -> None:
        a = VOCAB.index_of("あ")
        blank = alignment.BLANK_INDEX
        log_probs = _posteriors([a, blank, a, blank, a], vocab_size=VOCAB.size)
        spans = alignment.forced_align(log_probs, [a, a, a], upsample=1)
        assert [(s.start_frame, s.end_frame) for s in spans] == [(0, 1), (2, 3), (4, 5)]

    def test_the_error_names_the_frame_count_the_caller_needs(self) -> None:
        """`raise the upsample factor` is only actionable with a target."""
        a = VOCAB.index_of("あ")
        log_probs = torch.randn(4, VOCAB.size).log_softmax(dim=-1)
        with pytest.raises(ValueError, match="at least 5"):
            alignment.forced_align(log_probs, [a, a, a], upsample=1)

    def test_every_feasible_length_actually_aligns(self) -> None:
        """The bound is exact, not merely safe: at exactly `minimum_ctc_frames`
        an alignment must exist for every one of these shapes. A conservative
        over-estimate would pass the two regressions above while still throwing
        away short clips."""
        a, i, u = (VOCAB.index_of(ch) for ch in "あいう")
        for targets in ([a], [a, i], [a, a], [a, i, u], [a, a, i], [a, i, i], [a, a, a]):
            frames = alignment.minimum_ctc_frames(targets)
            log_probs = torch.randn(frames, VOCAB.size).log_softmax(dim=-1)
            spans = alignment.forced_align(log_probs, targets, upsample=1)
            assert len(spans) == len(targets), targets


class TestBlankRuns:
    def test_a_silent_stretch_reads_as_one_run(self) -> None:
        a = VOCAB.index_of("あ")
        blank = alignment.BLANK_INDEX
        frames = [a, a, blank, blank, blank, blank, a, blank]
        runs = alignment.blank_runs(
            _posteriors(frames, vocab_size=VOCAB.size), upsample=1
        )
        assert runs == [
            (pytest.approx(2 / 13.0), pytest.approx(6 / 13.0)),
            (pytest.approx(7 / 13.0), pytest.approx(8 / 13.0)),
        ]

    def test_short_runs_can_be_filtered_out(self) -> None:
        """Inter-syllable blanks are not pauses; the gate must not cut on them."""
        a = VOCAB.index_of("あ")
        blank = alignment.BLANK_INDEX
        frames = [a, blank, a] + [blank] * 20 + [a]
        runs = alignment.blank_runs(
            _posteriors(frames, vocab_size=VOCAB.size), upsample=1, min_seconds=0.5
        )
        assert len(runs) == 1
        assert runs[0][1] - runs[0][0] == pytest.approx(20 / 13.0)

    def test_all_blank_audio_is_one_run_covering_everything(self) -> None:
        frames = [alignment.BLANK_INDEX] * 13
        runs = alignment.blank_runs(
            _posteriors(frames, vocab_size=VOCAB.size), upsample=1
        )
        assert runs == [(pytest.approx(0.0), pytest.approx(1.0))]


class TestSpeechExtent:
    """The onset correction that the 2026-07-31 listening pass forced.

    Forced alignment marks where the head is confident about a character, which
    for the first and last one sits inside the sound rather than at its edge.
    These tests pin the walk that recovers the edge - and, more importantly, pin
    what it must refuse to do.
    """

    def _spans(self, frames: list[int], text: str, *, upsample: int = 1):
        log_probs = _posteriors(frames, vocab_size=VOCAB.size)
        spans = alignment.align_text(log_probs, text, VOCAB, upsample=upsample)
        return log_probs, spans

    def test_the_edge_walks_out_through_blank(self) -> None:
        a, blank = VOCAB.index_of("あ"), alignment.BLANK_INDEX
        frames = [blank] * 3 + [a] * 2 + [blank] * 3
        log_probs, spans = self._spans(frames, "あ")
        extent = alignment.speech_extent(log_probs, spans, upsample=1)
        assert extent == (pytest.approx(0.0), pytest.approx(8 / 13.0))
        # And it really moved: the character's own span is the inner one.
        assert spans[0].start_s > extent[0]

    def test_it_stops_at_a_neighbouring_character(self) -> None:
        """The invariant that makes this safe to apply without a tuned constant.

        A backoff that ran on a fixed number of milliseconds would swallow the
        end of the previous word whenever the pause was shorter than the
        constant. Stopping at the first non-blank frame cannot.
        """
        a, ka, blank = VOCAB.index_of("あ"), VOCAB.index_of("か"), alignment.BLANK_INDEX
        frames = [ka] * 2 + [blank] + [a] * 2 + [blank] + [ka] * 2
        log_probs = _posteriors(frames, vocab_size=VOCAB.size)
        spans = alignment.align_text(log_probs, "あ", VOCAB, upsample=1)
        extent = alignment.speech_extent(log_probs, spans, upsample=1)
        # One blank frame on each side is available, and not one frame more.
        assert extent == (pytest.approx(2 / 13.0), pytest.approx(6 / 13.0))

    def test_a_long_pause_is_capped_rather_than_swallowed(self) -> None:
        a, blank = VOCAB.index_of("あ"), alignment.BLANK_INDEX
        frames = [blank] * 40 + [a] * 2 + [blank] * 40
        log_probs, spans = self._spans(frames, "あ")
        extent = alignment.speech_extent(
            log_probs, spans, upsample=1, backoff_max_s=0.3, extend_max_s=0.4
        )
        assert spans[0].start_s - extent[0] <= 0.3 + 1e-9
        assert extent[1] - spans[-1].end_s <= 0.4 + 1e-9
        # A 40-frame pause is ~3 s, so the cap is what bound here, not the audio.
        assert extent[0] > 0.0

    def test_a_tight_boundary_is_left_alone(self) -> None:
        """No pause, no correction. The walk is not a constant subtraction."""
        a, blank = VOCAB.index_of("あ"), alignment.BLANK_INDEX
        frames = [a] * 3
        log_probs, spans = self._spans(frames, "あ")
        extent = alignment.speech_extent(log_probs, spans, upsample=1)
        assert extent == (pytest.approx(spans[0].start_s), pytest.approx(spans[-1].end_s))
        assert blank == 0

    def test_no_spans_measures_nothing(self) -> None:
        log_probs = _posteriors([alignment.BLANK_INDEX] * 4, vocab_size=VOCAB.size)
        assert alignment.speech_extent(log_probs, [], upsample=1) is None

    def test_the_caps_are_read_from_the_measured_inset(self) -> None:
        """Both caps are one-sided and positive, and the tail one is larger.

        The composite geometry that motivated them is asymmetric - median inset
        230.8 ms at the head against 371.7 ms at the tail - so equal caps would
        be a claim the data does not make.
        """
        assert 0.0 < alignment.ONSET_BACKOFF_MAX_S < alignment.CODA_EXTEND_MAX_S
