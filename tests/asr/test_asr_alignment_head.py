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
        # Caps passed explicitly and sized to clear the pause: this pins the
        # walk, not the production constants. Reading them from the module made
        # the assertion silently depend on how many frames 0.30 happened to
        # quantize to, so retuning the cap failed a test about the mechanism.
        extent = alignment.speech_extent(
            log_probs, spans, upsample=1, backoff_max_s=0.3, extend_max_s=0.3
        )
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

    def test_the_caps_are_sized_by_cost_not_by_inset(self) -> None:
        """Both caps are one-sided and positive, and the ONSET one is larger.

        This assertion used to read the other way, justified by insets of 230.8
        and 371.7 ms. Both figures were measured against the core's placement
        window, so both counted the clip's own edge silence as alignment error -
        90.0 ms of it at the head and 274.8 ms at the tail. Decontaminated, the
        insets are near enough equal (110.8 / 99.2 ms) that they cannot order the
        caps at all.

        What orders them is which direction is cheap. At the onset, starting
        early only shows the line sooner, so the cap is pushed past the
        median-zero point to buy back late starts. At the coda both directions
        reach the visible out-point 1:1 - `linger_s` and
        `max_display_shift_from_acoustic_end_s` are both 0.5, so the ceiling
        binds - leaving no free direction and no reason to go past median-zero.
        """
        assert 0.0 < alignment.CODA_EXTEND_MAX_S < alignment.ONSET_BACKOFF_MAX_S


class TestBlankBias:
    """The peak-widening knob, and the two things it must not touch.

    CTC posteriors are peaky: the path stays in blank until evidence has piled
    up, so every character's span starts inside its own sound. Subtracting a
    constant from the blank column before the search makes staying in blank
    slightly more expensive and widens the characters back out. It is free at
    runtime and needs no retraining, which is why it is worth a sweep - but it
    is also a knob that can manufacture accuracy on any metric that only
    penalises being late, so what it must NOT do is pinned here.
    """

    def test_zero_reproduces_the_untouched_search_exactly(self) -> None:
        """The default has to be bit-identical or every stored timestamp moves."""
        a, i = VOCAB.index_of("あ"), VOCAB.index_of("い")
        blank = alignment.BLANK_INDEX
        log_probs = _posteriors([a, blank, i], vocab_size=VOCAB.size)
        plain = alignment.forced_align(log_probs, [a, i], upsample=1)
        biased = alignment.forced_align(log_probs, [a, i], upsample=1, blank_bias=0.0)
        assert [(s.start_frame, s.end_frame) for s in plain] == [
            (s.start_frame, s.end_frame) for s in biased
        ]

    def test_a_bias_widens_a_character_into_the_blank_beside_it(self) -> None:
        a = VOCAB.index_of("あ")
        blank = alignment.BLANK_INDEX
        # Blank is only mildly preferred either side of the sound, so a small
        # penalty is enough to hand those frames to the character.
        probs = torch.full((5, VOCAB.size), 0.02)
        probs[:, blank] = 0.6
        probs[2, blank] = 0.1
        probs[2, a] = 0.7
        probs[1, a] = 0.35
        probs[3, a] = 0.35
        log_probs = probs.log()

        narrow = alignment.forced_align(log_probs, [a], upsample=1)
        wide = alignment.forced_align(log_probs, [a], upsample=1, blank_bias=1.5)
        assert wide[0].start_frame <= narrow[0].start_frame
        assert wide[0].end_frame >= narrow[0].end_frame
        assert (wide[0].end_frame - wide[0].start_frame) > (
            narrow[0].end_frame - narrow[0].start_frame
        )

    def test_the_score_is_read_off_the_unbiased_posteriors(self) -> None:
        """The score is the hallucination signal. If the bias moved it, the same
        audio and the same text would score differently because a TIMING knob was
        turned, and the post-gate's threshold would drift with it."""
        a = VOCAB.index_of("あ")
        log_probs = _posteriors([a, a, a], vocab_size=VOCAB.size)
        plain = alignment.forced_align(log_probs, [a], upsample=1)
        biased = alignment.forced_align(log_probs, [a], upsample=1, blank_bias=2.0)
        assert plain[0].score == pytest.approx(biased[0].score, abs=1e-6)
        # And the score stays a log-probability, not a penalised one.
        assert biased[0].score == pytest.approx(float(log_probs[0, a]), abs=1e-6)

    def test_a_negative_bias_is_refused(self) -> None:
        """It would shrink characters further into their middles - the defect
        this exists to correct, with a knob to produce it on demand."""
        a = VOCAB.index_of("あ")
        log_probs = _posteriors([a], vocab_size=VOCAB.size)
        with pytest.raises(ValueError, match="blank_bias"):
            alignment.forced_align(log_probs, [a], upsample=1, blank_bias=-0.5)

    def test_the_input_tensor_is_not_modified(self) -> None:
        """The caller reuses this tensor for `speech_extent` and `blank_runs`."""
        a = VOCAB.index_of("あ")
        log_probs = _posteriors([a, a], vocab_size=VOCAB.size)
        before = log_probs.clone()
        alignment.forced_align(log_probs, [a], upsample=1, blank_bias=1.0)
        assert torch.equal(log_probs, before)

    def test_the_default_is_off_until_a_sweep_says_otherwise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(alignment.ALIGNMENT_BLANK_BIAS_ENV, raising=False)
        assert alignment.blank_bias_from_env() == 0.0

    def test_the_env_knob_reads_a_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(alignment.ALIGNMENT_BLANK_BIAS_ENV, "1.25")
        assert alignment.blank_bias_from_env() == pytest.approx(1.25)

    def test_a_malformed_or_negative_knob_reads_as_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This stage's contract is to degrade, not to take transcription down,
        and 0.0 is the arm that was actually measured."""
        for raw in ("", "  ", "abc", "-1.0"):
            monkeypatch.setenv(alignment.ALIGNMENT_BLANK_BIAS_ENV, raw)
            assert alignment.blank_bias_from_env() == 0.0


class TestAcousticTargets:
    """Punctuation is not a sound, so it is not a class.

    16.9% of the training targets were Unicode punctuation or symbols, 92.3% of
    clips carried some, and 527 clips had `...` as their entire target. Every
    one of those asks the head to emit a character where the audio is a pause -
    inside the blank run the chunker reads to choose a cut. The v2 vocab aligns
    only what can be pronounced and puts the rest back afterwards, because the
    subtitle layer indexes spans by character position and drops to synthetic
    timing on any count mismatch.
    """

    ACOUSTIC = alignment.AlignmentVocab(chars=tuple("あいうえお"), acoustic_only=True)

    def test_letters_and_digits_are_acoustic_and_marks_are_not(self) -> None:
        for char in "あアか漢A7":
            assert alignment.is_acoustic_char(char), char
        for char in "、。…！？「」・♪♡〜":
            assert not alignment.is_acoustic_char(char), char

    def test_prolongation_and_iteration_marks_count_as_sound(self) -> None:
        """`ー` and `々` are pronounced and occupy audio; dropping them would
        leave that audio to be explained by the character before them."""
        assert alignment.is_acoustic_char("ー")
        assert alignment.is_acoustic_char("々")

    def test_the_origin_of_every_kept_character_is_recorded(self) -> None:
        spoken, origins = alignment.acoustic_text("あ、い。")
        assert spoken == "あい"
        assert origins == [0, 2]

    def test_a_v2_vocab_refuses_to_hold_punctuation(self) -> None:
        with pytest.raises(ValueError, match="acoustic-only"):
            alignment.AlignmentVocab(chars=("あ", "、"), acoustic_only=True)

    def test_the_inventory_filters_counts_rather_than_trusting_the_caller(self) -> None:
        vocab = alignment.AlignmentVocab.from_counts(
            {"あ": 10, "、": 99, "…": 50, "い": 5}, acoustic_only=True
        )
        assert vocab.chars == ("あ", "い")

    def test_encoding_drops_what_the_head_cannot_emit(self) -> None:
        assert len(self.ACOUSTIC.encode("あ、い。")) == 2
        punctuated = alignment.AlignmentVocab(chars=tuple("あい、。"))
        assert len(punctuated.encode("あ、い。")) == 4

    def test_the_schema_says_which_kind_of_head_this_is(self) -> None:
        """Not a flag on v1: a reader that ignored it would align punctuation
        against a model whose only answer there is blank."""
        payload = self.ACOUSTIC.to_payload()
        assert payload["schema"] == alignment.ACOUSTIC_VOCAB_SCHEMA
        assert alignment.AlignmentVocab.from_payload(payload) == self.ACOUSTIC
        old = alignment.AlignmentVocab(chars=tuple("あい"))
        assert old.to_payload()["schema"] == alignment.ALIGNMENT_VOCAB_SCHEMA
        assert alignment.AlignmentVocab.from_payload(old.to_payload()) == old

    def test_an_unknown_schema_is_still_refused(self) -> None:
        with pytest.raises(ValueError, match="schema"):
            alignment.AlignmentVocab.from_payload(
                {"schema": "something_else", "blank_index": 0, "unk_index": 1}
            )

    def test_one_span_per_character_of_the_original_text(self) -> None:
        """The contract `build_aligned_word_timestamps` enforces: a count
        mismatch there silently falls back to proportional timing."""
        a, i = self.ACOUSTIC.index_of("あ"), self.ACOUSTIC.index_of("い")
        log_probs = _posteriors(
            [a, alignment.BLANK_INDEX, i], vocab_size=self.ACOUSTIC.size
        )
        spans = alignment.align_text(log_probs, "あ、い。", self.ACOUSTIC, upsample=1)
        assert [s.char for s in spans] == ["あ", "、", "い", "。"]
        assert [s.index for s in spans] == [0, 1, 2, 3]

    def test_punctuation_takes_no_audio_from_the_character_before_it(self) -> None:
        a, i = self.ACOUSTIC.index_of("あ"), self.ACOUSTIC.index_of("い")
        log_probs = _posteriors(
            [a, alignment.BLANK_INDEX, i], vocab_size=self.ACOUSTIC.size
        )
        spans = alignment.align_text(log_probs, "あ、い。", self.ACOUSTIC, upsample=1)
        assert [(s.start_frame, s.end_frame) for s in spans] == [
            (0, 1),
            (1, 1),
            (2, 3),
            (3, 3),
        ]

    def test_leading_punctuation_anchors_to_the_first_sound(self) -> None:
        a = self.ACOUSTIC.index_of("あ")
        log_probs = _posteriors(
            [alignment.BLANK_INDEX, a], vocab_size=self.ACOUSTIC.size
        )
        spans = alignment.align_text(log_probs, "…あ", self.ACOUSTIC, upsample=1)
        assert spans[0].start_frame == spans[0].end_frame == spans[1].start_frame

    def test_the_spans_stay_monotonic(self) -> None:
        a, i, u = (self.ACOUSTIC.index_of(ch) for ch in "あいう")
        blank = alignment.BLANK_INDEX
        log_probs = _posteriors([a, blank, i, blank, u], vocab_size=self.ACOUSTIC.size)
        spans = alignment.align_text(log_probs, "…あ、い。う!", self.ACOUSTIC, upsample=1)
        starts = [s.start_frame for s in spans]
        assert starts == sorted(starts)
        assert all(s.end_frame >= s.start_frame for s in spans)

    def test_text_with_nothing_to_pronounce_is_refused(self) -> None:
        """`...` alone was 527 clips of the corpus. There is no honest span for
        it, and the caller's fallback to synthetic timing is the right answer."""
        log_probs = _posteriors([alignment.BLANK_INDEX] * 4, vocab_size=self.ACOUSTIC.size)
        with pytest.raises(ValueError, match="acoustic"):
            alignment.align_text(log_probs, "……", self.ACOUSTIC, upsample=1)

    def test_a_v1_head_still_aligns_punctuation_as_before(self) -> None:
        """The old checkpoint is the one in production; its behaviour is fixed."""
        vocab = alignment.AlignmentVocab(chars=tuple("あい、"))
        a, comma = vocab.index_of("あ"), vocab.index_of("、")
        log_probs = _posteriors([a, comma], vocab_size=vocab.size)
        spans = alignment.align_text(log_probs, "あ、", vocab, upsample=1)
        assert [(s.start_frame, s.end_frame) for s in spans] == [(0, 1), (1, 2)]


class TestPaddingMask:
    """A clip's output must not depend on what it was batched with.

    Padded positions are zeros, but the head's input `LayerNorm` maps an
    all-zero vector to its own bias, and every conv in the stack reads across
    time - so without masking, the padding's bias is convolved into the real
    frames near a clip's tail. Training batches clips; inference never does. The
    head therefore learned on a tail context that does not exist in production,
    and length bucketing only made the padding shorter, not absent.
    """

    @staticmethod
    def _head(blocks: int = 2, upsample: int = 2):
        head = alignment.build_head(
            vocab_size=VOCAB.size,
            input_dim=8,
            hidden_dim=8,
            upsample=upsample,
            blocks=blocks,
            dropout=0.0,
        )
        return head.eval()

    def test_a_padded_clip_matches_the_same_clip_alone(self) -> None:
        head = self._head()
        torch.manual_seed(0)
        short, long_ = 11, 40
        clip = torch.randn(1, short, 8)
        batch = torch.zeros(2, long_, 8)
        batch[0] = torch.randn(long_, 8)
        batch[1, :short] = clip[0]
        lengths = torch.tensor([long_, short])

        with torch.inference_mode():
            batched = head(batch, lengths)
            alone = head(clip)

        assert torch.allclose(batched[1, : short * head.upsample], alone[0], atol=1e-6)

    def test_the_padding_content_cannot_reach_the_real_frames(self) -> None:
        """The sharper form: fill the pad with noise instead of zeros.

        Zeros are a special case the head might survive by accident. If the
        masking is real, the pad can hold anything and the answer is the same.
        """
        head = self._head()
        torch.manual_seed(1)
        short, long_ = 9, 40
        content = torch.randn(short, 8)
        quiet = torch.zeros(1, long_, 8)
        quiet[0, :short] = content
        loud = torch.randn(1, long_, 8) * 10.0
        loud[0, :short] = content
        lengths = torch.tensor([short])

        with torch.inference_mode():
            first = head(quiet, lengths)
            second = head(loud, lengths)

        assert torch.allclose(
            first[0, : short * head.upsample],
            second[0, : short * head.upsample],
            atol=1e-6,
        )

    def test_without_lengths_the_padding_does_reach_them(self) -> None:
        """The bug being fixed, pinned so the test above cannot pass vacuously."""
        head = self._head()
        torch.manual_seed(2)
        short, long_ = 9, 40
        content = torch.randn(short, 8)
        quiet = torch.zeros(1, long_, 8)
        quiet[0, :short] = content
        loud = torch.randn(1, long_, 8) * 10.0
        loud[0, :short] = content

        with torch.inference_mode():
            first = head(quiet)
            second = head(loud)

        assert not torch.allclose(
            first[0, : short * head.upsample],
            second[0, : short * head.upsample],
            atol=1e-4,
        )

    def test_a_single_full_length_clip_is_unaffected_by_the_argument(self) -> None:
        """Inference passes one unpadded clip, so it must be exactly as before."""
        head = self._head()
        torch.manual_seed(3)
        clip = torch.randn(1, 20, 8)
        with torch.inference_mode():
            assert torch.equal(head(clip), head(clip, torch.tensor([20])))


class TestReceptiveField:
    """The head reports its own context so callers do not restate it.

    The pipeline has to overlap its windows by this much; a dilation schedule
    changed here and not there would silently reintroduce the seam.
    """

    def test_the_default_stack_needs_fifteen_encoder_frames(self) -> None:
        head = alignment.build_head(vocab_size=VOCAB.size, input_dim=8, hidden_dim=8)
        # kernel 5 at dilations 1/2/4/8 spans 1 + 4*(1+2+4+8) = 61 output frames,
        # so 30 per side at 26 fps, i.e. 15 encoder frames of 77 ms.
        assert head.context_frames == 15
        assert alignment.ENCODER_FRAME_S * 15 == pytest.approx(1.1538, abs=1e-4)

    def test_more_blocks_means_more_context(self) -> None:
        wider = alignment.build_head(
            vocab_size=VOCAB.size, input_dim=8, hidden_dim=8, blocks=6
        )
        assert wider.context_frames > 15

    def test_the_loaded_head_exposes_the_same_number(self) -> None:
        module = alignment.build_head(vocab_size=VOCAB.size, input_dim=8, hidden_dim=8)
        head = alignment.AlignmentHead(module, VOCAB, 2, torch.device("cpu"))
        assert head.context_frames == module.context_frames
        assert head.context_seconds == pytest.approx(
            module.context_frames * alignment.ENCODER_FRAME_S
        )


class TestOverlapSaveWindows:
    """Long audio is windowed for the encoder; the head must not pay for it.

    Before this, `pipeline.py` ran the head on butt-jointed 30 s windows and
    concatenated the results, so every frame within ~1.15 s of a seam was
    convolved against zeros standing in for audio that exists - one seam every
    30 s, i.e. ~7.6% of a 40-minute timeline computed on absent context.
    Concatenating encoder features instead of head outputs would not fix it: the
    encoder runs per window too, so the audio itself has to overlap.
    """

    WIDTH = 30 * 16000
    CONTEXT = 15

    def _plan(self, seconds: float):
        return alignment.plan_head_windows(
            int(seconds * 16000), window_samples=self.WIDTH, context_frames=self.CONTEXT
        )

    @staticmethod
    def _frames(start: int, end: int) -> int:
        return int(round((end - start) * alignment.ENCODER_FPS / 16000))

    def test_consecutive_windows_overlap_by_twice_the_context(self) -> None:
        plan = self._plan(120.0)
        assert len(plan) > 1
        bases = [base for _, _, base in plan]
        window_frames = int(round(self.WIDTH * alignment.ENCODER_FPS / 16000))
        for earlier, later in zip(bases, bases[1:]):
            assert later - earlier == window_frames - 2 * self.CONTEXT

    def test_the_kept_slices_tile_the_timeline_exactly_once(self) -> None:
        plan = self._plan(300.0)
        lengths = [self._frames(start, end) for start, end, _ in plan]
        bases = [base for _, _, base in plan]
        slices = alignment.overlap_save_slices(
            list(zip(bases, lengths)), context_frames=self.CONTEXT
        )
        covered: list[int] = []
        for base, (begin, finish) in zip(bases, slices):
            covered.extend(range(base + begin, base + finish))
        assert covered == list(range(len(covered))), "holes or repeats on the frame axis"
        assert abs(len(covered) - int(300 * alignment.ENCODER_FPS)) <= 1

    def test_every_kept_frame_has_real_context_on_both_sides(self) -> None:
        """The property the overlap exists for, checked frame by frame."""
        total_frames = int(round(300.0 * alignment.ENCODER_FPS))
        plan = self._plan(300.0)
        lengths = [self._frames(start, end) for start, end, _ in plan]
        bases = [base for _, _, base in plan]
        slices = alignment.overlap_save_slices(
            list(zip(bases, lengths)), context_frames=self.CONTEXT
        )
        for base, length, (begin, finish) in zip(bases, lengths, slices):
            for frame in range(base + begin, base + finish):
                # What this window can offer the frame, against what the clip
                # has to offer at all - at the very edges of the file there is
                # nothing to be had and zeros are the truth.
                inside_left = frame - base
                inside_right = base + length - 1 - frame
                assert inside_left >= min(self.CONTEXT, frame)
                assert inside_right >= min(self.CONTEXT, total_frames - 1 - frame)

    def test_audio_shorter_than_one_window_is_a_single_untrimmed_pass(self) -> None:
        plan = self._plan(4.0)
        assert len(plan) == 1
        start, end, base = plan[0]
        assert (start, base) == (0, 0)
        length = self._frames(start, end)
        assert alignment.overlap_save_slices(
            [(base, length)], context_frames=self.CONTEXT
        ) == [(0, length)]

    def test_audio_too_short_to_chunk_gets_no_window_at_all(self) -> None:
        """Under half a second decides no pause and the processor pads it to a
        full window anyway, so it is pure encoder time - the same filter the
        butt-jointed version applied, now reachable only here. Every later
        window carries at least the 2 * context of new audio that stopped its
        predecessor from reaching the file end."""
        assert self._plan(0.3) == []
        tail = self._plan(30.2)
        assert len(tail) == 2
        start, end, _ = tail[1]
        assert end - start >= 2 * self.CONTEXT * 16000 / alignment.ENCODER_FPS

    def test_a_window_returning_fewer_frames_than_planned_cannot_duplicate_output(
        self,
    ) -> None:
        """Defensive: the trim continues from where the previous window stopped
        rather than from an assumed length, so a short middle window degrades to
        a gap instead of to repeated frames - repeats would move every later
        timestamp."""
        bases = [0, 360, 720]
        lengths = [390, 200, 390]
        slices = alignment.overlap_save_slices(
            list(zip(bases, lengths)), context_frames=self.CONTEXT
        )
        covered: list[int] = []
        for base, (begin, finish) in zip(bases, slices):
            covered.extend(range(base + begin, base + finish))
        assert len(covered) == len(set(covered))
        assert covered == sorted(covered)

    def test_a_context_wider_than_the_window_is_refused(self) -> None:
        with pytest.raises(ValueError, match="context"):
            alignment.plan_head_windows(
                16000 * 60, window_samples=16000, context_frames=200
            )
