"""A CTC alignment head over the SFT'd ASR encoder.

The pipeline has never had a real time axis. `src/asr/subtitle_timing.py` spreads
each segment's text across the segment window in proportion to character count
and reports `word_timestamps_real: False`; every branch does, so a subtitle's
in-segment timing has never been measured, only assumed.

Forced alignment was tried before and abandoned, for a specific reason: the ASR
is full-SFT'd on this domain and a general-purpose aligner is not, so the aligner
became the bottleneck rather than the fix. This head avoids that by not bringing
its own acoustic model. It sits on `get_audio_features` output from the same
SFT'd encoder, so the domain adaptation is already paid for, and only a thin
head is learned. Its training pairs are free: audio with known text.

Two design choices that differ from the obvious ones, both measured:

  * **targets are characters, not kana.** Kana would need `pyopenjtalk` g2p,
    which adds a dependency and, more importantly, a reading-error source on
    kanji. Characters need neither. The density works out better too - the
    galgame corpus runs 4.67 chars/s against a 13 fps encoder, so 2.8 frames per
    character, where kana would be nearer 2. Characters are also the unit the
    subtitle layer already measures in (`SUBTITLE_READING_CPS`,
    `src/subtitles/writer.py::_count_text_units`), so timestamps land on the
    unit that line-splitting consumes.
  * **the encoder is upsampled before the classifier.** It adds no information,
    but CTC cannot emit more tokens than it has frames, and timestamp resolution
    is bounded by frame duration - 76.9 ms at the native rate. Upsampling is
    cheap and buys back both.

The blank channel is not just a CTC formality here: a stretch of audio the head
covers entirely with blank is a stretch with no words in it, which is the
pre-decode gate the pipeline needs. So blank is index 0 and stays interpretable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import json
import math
import unicodedata

# Frames the audio encoder emits per second of audio. Derived from
# `qwen3_asr_audio_output_lengths` in `src/asr/encoder_features.py`, which
# returns 13 frames per 100 mel frames, and confirmed by measurement
# (390 frames for a 30 s clip).
ENCODER_FPS = 13.0
ENCODER_FRAME_S = 1.0 / ENCODER_FPS

# CTC blank. Index 0 by torch convention, and kept at 0 deliberately so that
# "argmax == 0" reads as "no word here" without a lookup.
BLANK_INDEX = 0
# Characters outside the vocabulary still consume a frame, so they get an index
# rather than being dropped - dropping them would shift every later timestamp.
UNK_INDEX = 1
RESERVED_INDICES = 2

ALIGNMENT_VOCAB_SCHEMA = "asr_ctc_alignment_char_vocab_v1"
# v2 differs in one way that cannot be expressed as a flag on v1: its classes
# are pronounceable characters only, so its targets are not the same sequence
# for the same text.
ACOUSTIC_VOCAB_SCHEMA = "asr_ctc_alignment_char_vocab_v2"
ALIGNMENT_MODEL_SCHEMA = "asr_ctc_alignment_head_v1"


def minimum_ctc_frames(targets: Sequence) -> int:
    """Shortest frame count that admits a monotonic CTC path for `targets`.

    Every label needs one frame of its own, and each pair of *adjacent identical*
    labels needs a mandatory blank between them - otherwise the two collapse into
    one emission and the path spells something shorter than the target. Leading
    and trailing blanks are optional, so they cost nothing.

    The obvious `len(targets) + 1` is wrong in both directions, and both were
    reproduced: `[a, b]` at T=2 has the legal path `a b` yet gets rejected, while
    `[a, a, a]` at T=4 needs 5 frames, passes the guard, and dies much later in
    the backtrace with `alignment path skipped character 0`. Japanese hits the
    second case routinely - `ああ`, `っっ`, `ーー`, and any doubled punctuation.

    Works on encoded indices or on the characters themselves. Characters are a
    lower bound when several out-of-vocabulary characters sit next to each other,
    since those all encode to `UNK_INDEX` and become an adjacent-identical pair
    the character view cannot see; the encoded view is the authoritative one.
    """
    if len(targets) == 0:
        return 0
    repeats = sum(1 for a, b in zip(targets, targets[1:]) if a == b)
    return len(targets) + repeats


def normalize_text(text: str) -> str:
    """Fold width and strip whitespace, so the same sound gets one target.

    NFKC collapses full-width latin/digits onto ASCII. Whitespace is removed
    entirely rather than mapped to a symbol: it is not pronounced, so giving it
    a target would force the head to explain silence it cannot hear.
    """
    folded = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in folded if not ch.isspace())


def is_acoustic_char(char: str) -> bool:
    """True when a character stands for a sound the head could emit.

    Letters and digits, by Unicode category - which keeps kana, kanji, latin,
    numbers, and the modifier letters Japanese pronounces (`ー` prolongation,
    `々` iteration), and drops punctuation and symbols: `。、…！？「」・♪♡`.

    The reason for the split is that punctuation was 16.9% of the training
    targets, 92.3% of clips carried some, and 527 clips had `...` as their
    ENTIRE target - so the head was being asked to emit a class at a pause, in
    the middle of the blank run the chunker reads. There is no sound to align it
    to; the best it can learn is "punctuation follows silence", which is exactly
    the confusion that makes a blank run stop being a clean pause.

    `〜` is dropped with the rest of the punctuation even though it is often
    voiced: it marks prolongation OF the previous vowel, which already has a
    target, so it has no acoustic extent of its own to occupy.
    """
    if not char:
        return False
    return unicodedata.category(char)[0] in {"L", "N"}


def acoustic_text(text: str) -> tuple[str, list[int]]:
    """Split normalised text into what is pronounced and where it came from.

    Returns the pronounceable characters and, for each, its index in `text`.
    The indices are what lets punctuation be put back afterwards: alignment runs
    on sound, and the subtitle layer still needs one span per character of the
    original.
    """
    kept: list[str] = []
    origins: list[int] = []
    for index, char in enumerate(text):
        if is_acoustic_char(char):
            kept.append(char)
            origins.append(index)
    return "".join(kept), origins


@dataclass(frozen=True)
class AlignmentVocab:
    """Character inventory shared by the head, its targets and its decoder.

    `acoustic_only` says whether this inventory covers only pronounceable
    characters. It travels in the checkpoint because it is not a preference at
    inference time: a head trained on acoustic-only targets has no punctuation
    classes to emit, and asking it to align punctuation would put a character
    where the model can only answer with blank.
    """

    chars: tuple[str, ...]
    acoustic_only: bool = False

    def __post_init__(self) -> None:
        if len(set(self.chars)) != len(self.chars):
            raise ValueError("alignment vocab contains duplicate characters")
        if any(len(ch) != 1 for ch in self.chars):
            raise ValueError("alignment vocab entries must be single characters")
        if self.acoustic_only and not all(is_acoustic_char(ch) for ch in self.chars):
            raise ValueError(
                "an acoustic-only vocab cannot contain punctuation or symbols"
            )

    @property
    def size(self) -> int:
        """Classifier width: the characters plus blank and unknown."""
        return len(self.chars) + RESERVED_INDICES

    def index_of(self, char: str) -> int:
        try:
            return self._lookup[char]
        except KeyError:
            return UNK_INDEX

    @property
    def _lookup(self) -> dict[str, int]:
        cached = getattr(self, "_lookup_cache", None)
        if cached is None:
            cached = {
                ch: index + RESERVED_INDICES for index, ch in enumerate(self.chars)
            }
            object.__setattr__(self, "_lookup_cache", cached)
        return cached

    def char_at(self, index: int) -> str:
        if index == BLANK_INDEX:
            return ""
        if index == UNK_INDEX:
            return "�"
        position = index - RESERVED_INDICES
        if not 0 <= position < len(self.chars):
            raise IndexError(f"alignment vocab index out of range: {index}")
        return self.chars[position]

    def encode(self, text: str) -> list[int]:
        """Target sequence for CTC. Normalisation is applied here, once.

        An acoustic-only vocab drops what it cannot pronounce, so the targets
        and the classifier agree about what a class is.
        """
        normalized = normalize_text(text)
        if self.acoustic_only:
            normalized = acoustic_text(normalized)[0]
        return [self.index_of(ch) for ch in normalized]

    @classmethod
    def from_counts(
        cls,
        counts: dict[str, int],
        *,
        max_size: int = 0,
        min_count: int = 1,
        acoustic_only: bool = False,
    ) -> "AlignmentVocab":
        """Build from character frequencies, most frequent first.

        Ordering by frequency makes `max_size` a coverage decision rather than an
        arbitrary cut: on the galgame corpus the top 2,000 of 3,080 characters
        cover 99.81% of occurrences.

        Filtering here rather than at the call site is deliberate: the inventory
        is what defines a class, so an acoustic-only head cannot acquire a
        punctuation class through a caller that forgot to strip its counts.
        """
        eligible = [
            (char, count)
            for char, count in counts.items()
            if count >= min_count
            and len(char) == 1
            and not char.isspace()
            and (not acoustic_only or is_acoustic_char(char))
        ]
        eligible.sort(key=lambda item: (-item[1], item[0]))
        if max_size > 0:
            eligible = eligible[:max_size]
        return cls(
            chars=tuple(char for char, _ in eligible), acoustic_only=acoustic_only
        )

    def to_payload(self) -> dict:
        return {
            "schema": (
                ACOUSTIC_VOCAB_SCHEMA if self.acoustic_only else ALIGNMENT_VOCAB_SCHEMA
            ),
            "blank_index": BLANK_INDEX,
            "unk_index": UNK_INDEX,
            "size": self.size,
            "acoustic_only": self.acoustic_only,
            "chars": list(self.chars),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "AlignmentVocab":
        schema = str(payload.get("schema") or "")
        # The schema, not just a flag, because the two are not interchangeable:
        # a v2 head has no punctuation classes, and a reader that ignored the
        # flag would align punctuation against a model that can only answer
        # blank there. An old build must fail on a v2 checkpoint, loudly.
        if schema not in {ALIGNMENT_VOCAB_SCHEMA, ACOUSTIC_VOCAB_SCHEMA}:
            raise ValueError(
                f"alignment vocab schema must be one of "
                f"{ALIGNMENT_VOCAB_SCHEMA}/{ACOUSTIC_VOCAB_SCHEMA}, got {schema!r}"
            )
        # A checkpoint whose blank moved would silently reinterpret every frame.
        if int(payload.get("blank_index", -1)) != BLANK_INDEX:
            raise ValueError("alignment vocab blank_index must be 0")
        if int(payload.get("unk_index", -1)) != UNK_INDEX:
            raise ValueError("alignment vocab unk_index must be 1")
        return cls(
            chars=tuple(str(ch) for ch in payload.get("chars") or ()),
            acoustic_only=schema == ACOUSTIC_VOCAB_SCHEMA,
        )

    def dumps(self) -> str:
        return json.dumps(self.to_payload(), ensure_ascii=False)


def frame_to_seconds(frame: int, *, upsample: int) -> float:
    """Left edge of an output frame, in seconds of source audio."""
    if upsample < 1:
        raise ValueError("upsample must be >= 1")
    return float(frame) * ENCODER_FRAME_S / float(upsample)


def output_frame_count(encoder_frames: int, *, upsample: int) -> int:
    if upsample < 1:
        raise ValueError("upsample must be >= 1")
    return int(encoder_frames) * int(upsample)


def plan_head_windows(
    total_samples: int,
    *,
    window_samples: int,
    context_frames: int,
    sample_rate: int = 16000,
    min_samples: int = 8000,
) -> list[tuple[int, int, int]]:
    """Overlap-save windows for running the head over audio longer than one pass.

    Returns `(start_sample, end_sample, base_frame)` per window, where
    `base_frame` is the window's first frame on the whole clip's frame axis.

    The head has to see `context_frames` on each side of a frame to compute it
    the way it was trained. Butt-jointed windows give the frames at a seam zeros
    instead, so consecutive windows overlap by `2 * context_frames` and the
    overlap is dropped afterwards - the classic overlap-save arrangement.
    Overlapping the AUDIO is what is required, not concatenating features from
    butt-jointed windows: the encoder runs per window too, so its features are
    already missing that context by the time anything is concatenated.

    The frame axis is authoritative and sample offsets are derived from it,
    rather than the other way round. One encoder frame is 16000/13 = 1230.77
    samples, so a hop chosen in samples would land between frames and the error
    would accumulate across a long file; a hop chosen in frames keeps every
    window's `base_frame` exact and pushes the sub-sample rounding somewhere
    harmless.
    """
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    if context_frames < 0:
        raise ValueError("context_frames must be >= 0")
    window_frames = int(round(window_samples * ENCODER_FPS / sample_rate))
    hop_frames = window_frames - 2 * context_frames
    if hop_frames < 1:
        raise ValueError(
            f"window of {window_frames} frames cannot carry {context_frames} "
            "frames of context on both sides"
        )
    windows: list[tuple[int, int, int]] = []
    index = 0
    while True:
        base_frame = index * hop_frames
        start = int(round(base_frame * sample_rate / ENCODER_FPS))
        if start >= int(total_samples):
            break
        end = min(start + window_samples, int(total_samples))
        # A sliver carries no pause decision and the processor pads it to a full
        # window anyway, so it is only encoder time. With overlap this can only
        # ever fire on the first window - i.e. on a clip that is itself a sliver.
        # A later window always has at least the 2 * context_frames of new audio
        # that made its predecessor stop short of the file end.
        if end - start < min_samples:
            break
        windows.append((start, end, base_frame))
        if end >= int(total_samples):
            break
        index += 1
    return windows


def overlap_save_slices(
    windows: list[tuple[int, int]], *, context_frames: int
) -> list[tuple[int, int]]:
    """Which frames to keep from each window, as local `(start, end)` slices.

    `windows` is `(base_frame, frames_returned)` per window, in order. The
    returned slices tile the frame axis exactly once: each window gives up its
    trailing `context_frames` to its successor, and every frame is taken from
    the window that had real audio on both sides of it.

    The leading trim is not applied explicitly - it falls out of continuing from
    wherever the previous window stopped. That is also what makes this safe when
    a window comes back shorter than planned: the result can never contain a
    frame twice, and the seam does not silently shift.
    """
    slices: list[tuple[int, int]] = []
    emitted = 0
    for index, (base, frames) in enumerate(windows):
        last = index == len(windows) - 1
        end = base + int(frames) - (0 if last else context_frames)
        start = max(int(base), emitted)
        if end <= start:
            slices.append((0, 0))
            continue
        slices.append((start - int(base), end - int(base)))
        emitted = end
    return slices


@dataclass(frozen=True)
class CharSpan:
    """One character's measured extent on the audio time axis."""

    char: str
    index: int
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    score: float


def forced_align(
    log_probs,
    targets: list[int],
    *,
    upsample: int = 2,
    chars: str = "",
    blank_bias: float = 0.0,
):
    """Viterbi-align a known character sequence to per-frame posteriors.

    Written here rather than taken from `torchaudio.functional.forced_align`
    because torchaudio is not installed and pulling a CUDA wheel for one operator
    risks disturbing the working torch build. It also returns per-character
    scores, which S5 needs as its hallucination signal: text the acoustics do not
    support aligns badly, and the score is where that shows up.

    `log_probs` is (T, V) for a single utterance. Returns the character spans.

    `blank_bias` is subtracted from the blank column before the search, and only
    for the search. CTC posteriors are peaky - the path sits in blank until
    evidence has accumulated - so making blank fractionally more expensive widens
    every character onto the frames it actually occupies. Zero reproduces the
    untouched behaviour exactly.

    The scores are deliberately read off the ORIGINAL tensor. They feed the
    hallucination signal downstream, and a score computed against a biased blank
    would move that threshold silently: the same audio and the same text would
    score differently because a timing knob was turned.
    """
    import torch

    if log_probs.dim() != 2:
        raise ValueError(f"expected (T, V) log-probs, got shape {tuple(log_probs.shape)}")
    frames, vocab_size = int(log_probs.shape[0]), int(log_probs.shape[1])
    if not targets:
        return []
    if any(not 0 <= t < vocab_size for t in targets):
        raise ValueError("target index outside the classifier width")
    if any(t == BLANK_INDEX for t in targets):
        raise ValueError("targets must not contain the blank index")
    if blank_bias < 0.0:
        # A negative bias makes blank cheaper, i.e. shrinks characters further
        # into their own middles. Nothing wants that, and allowing it would make
        # the knob able to manufacture the defect it exists to correct.
        raise ValueError("blank_bias must be >= 0")

    searched = log_probs
    if blank_bias > 0.0:
        searched = log_probs.clone()
        searched[:, BLANK_INDEX] -= float(blank_bias)

    # The CTC lattice: blanks interleaved around every label, so that repeated
    # characters are forced apart by a blank and cannot collapse into one.
    extended = [BLANK_INDEX]
    for token in targets:
        extended.extend((token, BLANK_INDEX))
    states = len(extended)
    required = minimum_ctc_frames(targets)
    if frames < required:
        # Too few frames for any monotonic path: one per character, plus the
        # blank each pair of identical neighbours must be held apart by.
        raise ValueError(
            f"cannot align {len(targets)} characters to {frames} frames "
            f"(needs at least {required}); "
            "raise the upsample factor or shorten the segment"
        )

    device = log_probs.device
    state_tokens = torch.tensor(extended, dtype=torch.long, device=device)
    # A skip from s-2 to s is legal only when it hops a blank between two
    # DIFFERENT labels; between identical labels the blank is mandatory.
    can_skip = torch.zeros(states, dtype=torch.bool, device=device)
    for s in range(2, states):
        if extended[s] != BLANK_INDEX and extended[s] != extended[s - 2]:
            can_skip[s] = True

    neg_inf = -1e30
    scores = torch.full((states,), neg_inf, dtype=torch.float32, device=device)
    scores[0] = searched[0, BLANK_INDEX]
    if states > 1:
        scores[1] = searched[0, extended[1]]
    backpointers = torch.zeros((frames, states), dtype=torch.uint8, device=device)

    for t in range(1, frames):
        stay = scores
        advance = torch.cat((torch.full((1,), neg_inf, device=device), scores[:-1]))
        skip = torch.cat((torch.full((2,), neg_inf, device=device), scores[:-2]))
        skip = torch.where(can_skip, skip, torch.full_like(skip, neg_inf))
        stacked = torch.stack((stay, advance, skip), dim=0)
        best, choice = stacked.max(dim=0)
        scores = best + searched[t].index_select(0, state_tokens)
        backpointers[t] = choice.to(torch.uint8)

    # A valid path ends on the last label or on the blank after it.
    tail = states - 1
    state = tail if scores[tail] >= scores[tail - 1] else tail - 1
    path = [0] * frames
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state -= int(backpointers[t][state].item())

    frame_scores = log_probs.detach().float().cpu()
    spans: list[CharSpan] = []
    for label_index in range(len(targets)):
        state_index = 2 * label_index + 1
        occupied = [t for t in range(frames) if path[t] == state_index]
        if not occupied:
            # Unreachable for a valid Viterbi path, but a corrupt path would
            # otherwise produce a silently wrong timestamp rather than an error.
            raise RuntimeError(f"alignment path skipped character {label_index}")
        start_frame, end_frame = occupied[0], occupied[-1] + 1
        token = targets[label_index]
        score = float(
            sum(float(frame_scores[t, token]) for t in occupied) / len(occupied)
        )
        spans.append(
            CharSpan(
                char=chars[label_index] if label_index < len(chars) else "",
                index=label_index,
                start_frame=start_frame,
                end_frame=end_frame,
                start_s=frame_to_seconds(start_frame, upsample=upsample),
                end_s=frame_to_seconds(end_frame, upsample=upsample),
                score=score if math.isfinite(score) else float("-inf"),
            )
        )
    return spans


def align_text(
    log_probs,
    text: str,
    vocab: AlignmentVocab,
    *,
    upsample: int = 2,
    blank_bias: float = 0.0,
):
    """Align raw text by normalising and encoding it the same way training did.

    Returns one span per character of `normalize_text(text)` - including the
    characters an acoustic-only head never aligned, because the subtitle layer
    indexes spans by character position and falls back to synthetic timing on
    any count mismatch.
    """
    normalized = normalize_text(text)
    if not vocab.acoustic_only:
        targets = [vocab.index_of(ch) for ch in normalized]
        return forced_align(
            log_probs,
            targets,
            upsample=upsample,
            chars=normalized,
            blank_bias=blank_bias,
        )

    spoken, origins = acoustic_text(normalized)
    if not spoken:
        # Nothing pronounceable to align against: `...` on its own was 527 clips
        # of the training corpus, and there is no honest span for it.
        raise ValueError("text has no acoustic characters to align")
    targets = [vocab.index_of(ch) for ch in spoken]
    aligned = forced_align(
        log_probs, targets, upsample=upsample, chars=spoken, blank_bias=blank_bias
    )
    return _spans_for_full_text(aligned, normalized, origins, upsample=upsample)


def _spans_for_full_text(
    aligned: list[CharSpan], normalized: str, origins: list[int], *, upsample: int
) -> list[CharSpan]:
    """Put the unpronounced characters back, as zero-width marks.

    A comma occupies no audio, so it gets no width: it is anchored to the end of
    the character before it, or to the start of the one after it when it leads.
    Giving it width would take that width from a character that was actually
    spoken, and every downstream consumer reads these as measurements.

    The score is inherited from the anchor rather than invented. It feeds the
    hallucination signal as a mean over characters, and both alternatives are
    worse: a fabricated score would move that mean, and a NaN would poison it.
    """
    by_origin = {origin: span for origin, span in zip(origins, aligned)}
    spans: list[CharSpan] = []
    previous: CharSpan | None = None
    for index, char in enumerate(normalized):
        span = by_origin.get(index)
        if span is not None:
            previous = span
            spans.append(
                CharSpan(
                    char=char,
                    index=index,
                    start_frame=span.start_frame,
                    end_frame=span.end_frame,
                    start_s=span.start_s,
                    end_s=span.end_s,
                    score=span.score,
                )
            )
            continue
        anchor = previous if previous is not None else aligned[0]
        frame = anchor.end_frame if previous is not None else anchor.start_frame
        moment = frame_to_seconds(frame, upsample=upsample)
        spans.append(
            CharSpan(
                char=char,
                index=index,
                start_frame=frame,
                end_frame=frame,
                start_s=moment,
                end_s=moment,
                score=anchor.score,
            )
        )
    return spans


def blank_runs(log_probs, *, upsample: int = 2, min_seconds: float = 0.0):
    """Stretches the head covers entirely with blank, as (start_s, end_s).

    This is the gate reading of the same tensor the alignment uses: a run of
    blank is a run with no character evidence in it. It is deliberately computed
    from the argmax rather than a tuned threshold, so the pre-gate has no free
    parameter beyond the minimum run length.
    """
    predicted = log_probs.argmax(dim=-1).detach().cpu().tolist()
    runs: list[tuple[float, float]] = []
    start: int | None = None
    for frame, token in enumerate(list(predicted) + [-1]):
        if token == BLANK_INDEX:
            if start is None:
                start = frame
            continue
        if start is not None:
            begin = frame_to_seconds(start, upsample=upsample)
            end = frame_to_seconds(frame, upsample=upsample)
            if end - begin >= min_seconds:
                runs.append((begin, end))
            start = None
    return runs


# How far an utterance edge may be walked outward, away from the first/last
# character, to reach the acoustic boundary. Sized from the composite geometry
# that made the correction necessary: on cores whose bodies align cleanly the
# predicted extent sits INSIDE the true one by a median 230.8 ms at the head and
# 371.7 ms at the tail, with p75 307.7 ms / 536.2 ms. The caps cover the bulk of
# that inset without being free parameters in the usual sense - the walk stops at
# the first non-blank frame, so on a tight boundary the cap never binds and only
# a real pause lets the edge move at all.
ONSET_BACKOFF_MAX_S = 0.30
CODA_EXTEND_MAX_S = 0.40


def speech_extent(
    log_probs,
    spans,
    *,
    upsample: int = 2,
    backoff_max_s: float = ONSET_BACKOFF_MAX_S,
    extend_max_s: float = CODA_EXTEND_MAX_S,
):
    """Acoustic extent of an aligned utterance, as (start_s, end_s).

    Forced alignment answers "which frames is the model most confident carry
    this character", and for the first and last character that is not the same
    question as "where does the sound begin and end". CTC posteriors are peaky:
    the path stays in blank until evidence has accumulated, so the opening
    character's span starts partway into its own acoustic realisation and the
    closing one ends before its decay. The measured signature is an inset at
    BOTH edges - a lag would shift both the same way.

    The evidence is geometric, not perceptual, and this docstring used to claim
    otherwise: it cited a blind pass hearing the inset at "48.1% chopped against
    a 3.3% floor", which were the mid-pass numbers from a `result.json` written
    before its own verdicts finished. The completed 110/110 pass says 11.5%
    against a 0.0% floor, CI95 of the difference [-0.7, +23.8] - i.e. the ear
    could NOT separate the uncorrected onset from the floor. What still supports
    the correction is the composite geometry above; what no longer supports it
    is anyone's hearing.

    The fix does not guess a constant to subtract. It walks outward from the
    edge characters through frames the head itself labels blank, which is the
    gate reading of this same tensor. That makes the correction per-line and
    self-limiting: it stops at the first frame carrying character evidence, so
    it can never cross into a neighbouring word, and where there is no pause to
    move into it does nothing at all.

    Returns None when there is nothing to measure.
    """
    if not spans:
        return None
    if upsample < 1:
        raise ValueError("upsample must be >= 1")
    predicted = log_probs.argmax(dim=-1).detach().cpu().tolist()
    total = len(predicted)
    if total <= 0:
        return None

    frame_s = ENCODER_FRAME_S / float(upsample)
    back_limit = max(0, int(max(0.0, backoff_max_s) / frame_s))
    forward_limit = max(0, int(max(0.0, extend_max_s) / frame_s))

    head = max(0, min(int(spans[0].start_frame), total))
    tail = max(0, min(int(spans[-1].end_frame), total))

    steps = 0
    while head > 0 and steps < back_limit and predicted[head - 1] == BLANK_INDEX:
        head -= 1
        steps += 1

    steps = 0
    while tail < total and steps < forward_limit and predicted[tail] == BLANK_INDEX:
        tail += 1
        steps += 1

    start_s = frame_to_seconds(head, upsample=upsample)
    end_s = frame_to_seconds(max(head, tail), upsample=upsample)
    return start_s, end_s


ALIGNMENT_HEAD_PATH_ENV = "ASR_ALIGNMENT_HEAD_PATH"
# Log-probability penalty applied to blank during the Viterbi search only.
# Default 0.0: the sweep that would set it has to be run against composite
# geometry first, and shipping an unmeasured non-zero here would move every
# subtitle boundary on a hunch.
ALIGNMENT_BLANK_BIAS_ENV = "ASR_ALIGNMENT_BLANK_BIAS"
ALIGNMENT_HEAD_HF_SCHEME = "hf:"
ALIGNMENT_HEAD_DEFAULT_FILENAME = "ctc_aligner.pt"


def _parse_hf_reference(reference: str) -> tuple[str, str, str]:
    """`hf:<repo>@<revision>#<filename>` -> (repo, revision, filename).

    `#` and `@` are the separators because a repo id contains `/` and a commit
    sha contains neither, so the split stays unambiguous without quoting.
    """
    body = reference.strip()
    if body.lower().startswith(ALIGNMENT_HEAD_HF_SCHEME):
        body = body[len(ALIGNMENT_HEAD_HF_SCHEME) :].strip()
    if "#" in body:
        body, filename = body.rsplit("#", 1)
    else:
        filename = ALIGNMENT_HEAD_DEFAULT_FILENAME
    if "@" in body:
        repo, revision = body.rsplit("@", 1)
    else:
        repo, revision = body, ""
    repo, filename, revision = repo.strip(" /"), filename.strip(), revision.strip()
    if not repo or not filename:
        raise ValueError(f"malformed alignment head reference: {reference!r}")
    return repo, revision, filename


def _bundled_head_path(filename: str) -> str:
    """A copy shipped inside the packaged app, or "" when there is none.

    The Windows build carries the head next to the ASR weights, because a first
    run on a machine with no network must still produce measured timings rather
    than silently falling back to proportional ones. Consulted only when frozen,
    so a stray file in a source checkout's models/ cannot displace the pinned
    revision without anyone noticing.
    """
    from utils.model_paths import BUNDLED_MODELS_ROOT
    from utils.runtime_paths import is_frozen

    if not is_frozen():
        return ""
    candidate = BUNDLED_MODELS_ROOT / filename
    return str(candidate) if candidate.is_file() else ""


def _revision_marker(head_path: Path) -> Path:
    return head_path.with_name(head_path.name + ".revision")


def _downloaded_head_path(filename: str, revision: str) -> str:
    """The head already sitting in models/, if it is the revision we asked for.

    It lives next to the ASR weights rather than in the Hub cache: the weights
    are in models/, the packaged build ships the head at models/, and the Hub
    cache root is tmp/, a directory whose name invites deletion. The sidecar
    records which revision the file is, so re-pinning the default sha fetches
    the new head instead of silently loading the old one under the same name.
    """
    from utils.model_paths import MODELS_ROOT

    candidate = MODELS_ROOT / filename
    if not candidate.is_file():
        return ""
    if not revision:
        return str(candidate)
    try:
        recorded = _revision_marker(candidate).read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return str(candidate) if recorded == revision else ""


def resolve_alignment_head_path(reference: str, *, download: bool = True) -> str:
    """Turn the configured reference into a local checkpoint path.

    A plain path is returned untouched. `hf:...` resolves to `models/<file>`,
    downloading it there once (honouring the project proxy settings) and staying
    offline on every later run.

    The default reference pins a commit sha rather than a branch. The head is
    encoder-specific, and a moving `main` would let a retrained head change
    every subtitle's timing with nothing in the run saying so.
    """
    reference = (reference or "").strip()
    if not reference.lower().startswith(ALIGNMENT_HEAD_HF_SCHEME):
        return reference

    repo, revision, filename = _parse_hf_reference(reference)
    for local in (_bundled_head_path(filename), _downloaded_head_path(filename, revision)):
        if local:
            return local
    if not download:
        return ""

    from huggingface_hub import hf_hub_download

    from utils.model_paths import MODELS_ROOT

    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo,
        filename=filename,
        revision=revision or None,
        local_dir=str(MODELS_ROOT),
    )
    if revision:
        # Written after the file lands, so an interrupted download leaves no
        # marker and the next run re-fetches rather than trusting a partial.
        _revision_marker(Path(downloaded)).write_text(revision, encoding="utf-8")
    return downloaded


def blank_bias_from_env() -> float:
    """`ASR_ALIGNMENT_BLANK_BIAS`, or 0.0 when unset or unreadable.

    A malformed value reads as "off" rather than raising: this is a timing knob
    on a stage whose whole design is to degrade instead of taking transcription
    down, and 0.0 is the behaviour that was actually measured.
    """
    import os

    raw = (os.environ.get(ALIGNMENT_BLANK_BIAS_ENV) or "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return value if value > 0.0 else 0.0


def alignment_head_configured() -> bool:
    """True when `ASR_ALIGNMENT_HEAD_PATH` points at something.

    Cheap env probe for callers that must make lifecycle decisions (keep the
    ASR model loaded through the alignment pass, consult the finalize cache)
    before anyone pays for actually loading the head.
    """
    import os

    return bool((os.environ.get(ALIGNMENT_HEAD_PATH_ENV) or "").strip())


class AlignmentHead:
    """A loaded head, ready to turn encoder features into character times.

    Deliberately not auto-discovered. The head is enabled only by pointing
    `ASR_ALIGNMENT_HEAD_PATH` at a checkpoint, and with nothing set the caller
    keeps its previous proportional timing. The reason is the same one that
    produced `require_boundary_pipeline_ready`: a stage that quietly starts
    running before it has been measured on the production domain reports its
    own unvalidated output as fact. This head's alignment accuracy has so far
    been established on clean speech only.
    """

    def __init__(
        self,
        module,
        vocab: AlignmentVocab,
        upsample: int,
        device,
        blank_bias: float = 0.0,
    ) -> None:
        self.module = module
        self.vocab = vocab
        self.upsample = int(upsample)
        self.device = device
        self.blank_bias = float(blank_bias)

    @property
    def context_frames(self) -> int:
        """Encoder frames of context the head needs on each side of a frame.

        Whoever slices audio into windows has to overlap them by at least this
        much and drop the overlap afterwards; a frame computed with less than
        this on either side was convolved against zeros standing in for audio
        that exists.
        """
        return int(self.module.context_frames)

    @property
    def context_seconds(self) -> float:
        return self.context_frames * ENCODER_FRAME_S

    @classmethod
    def load(cls, checkpoint_path: str, *, device=None, blank_bias=None) -> "AlignmentHead":
        import torch

        resolved_path = resolve_alignment_head_path(checkpoint_path)
        payload = torch.load(resolved_path, map_location="cpu", weights_only=False)
        schema = str(payload.get("schema") or "")
        if schema != ALIGNMENT_MODEL_SCHEMA:
            raise ValueError(
                f"expected {ALIGNMENT_MODEL_SCHEMA} checkpoint, got {schema!r}"
            )
        vocab = AlignmentVocab.from_payload(payload["vocab"])
        module = build_head(
            vocab_size=vocab.size,
            input_dim=int(payload.get("input_dim", 2048)),
            hidden_dim=int(payload["hidden_dim"]),
            upsample=int(payload["upsample"]),
            blocks=int(payload["blocks"]),
            dropout=0.0,
        )
        module.load_state_dict(payload["state_dict"])
        resolved = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        module.to(resolved).eval()
        return cls(
            module,
            vocab,
            int(payload["upsample"]),
            resolved,
            blank_bias=(
                blank_bias_from_env() if blank_bias is None else float(blank_bias)
            ),
        )

    @classmethod
    def from_env(cls, *, device=None) -> "AlignmentHead | None":
        import os

        path = (os.environ.get(ALIGNMENT_HEAD_PATH_ENV) or "").strip()
        if not path:
            return None
        return cls.load(path, device=device)

    def log_probs(self, features):
        """(T, input_dim) encoder features -> (T*upsample, vocab) log-probs."""
        import numpy as np
        import torch

        tensor = torch.as_tensor(np.asarray(features, dtype=np.float32))
        with torch.inference_mode():
            return self.module(tensor.unsqueeze(0).to(self.device))[0].float().cpu()

    def align(self, features, text: str):
        """Character spans for `text`, or None when it cannot be aligned.

        Returning None rather than raising is deliberate: the caller's job is to
        fall back to synthetic timing, and an unalignable segment - text longer
        than the audio can hold, usually a runaway decode - is an expected
        condition on this domain rather than an error.
        """
        aligned = self.align_extent(features, text)
        return None if aligned is None else aligned[0]

    def align_extent(self, features, text: str):
        """`(spans, start_s, end_s)`, or None when it cannot be aligned.

        The extent is returned alongside rather than folded into the spans on
        purpose: a character's span is a measurement of that character, and
        widening the first one to the acoustic boundary would make it report a
        stretch of audio the character does not occupy. Line edges want the
        extent; per-character timing wants the spans; conflating them would
        quietly corrupt the alignment scores the post-gate reads.
        """
        normalized = normalize_text(text)
        if not normalized:
            return None
        try:
            log_probs = self.log_probs(features)
            spans = align_text(
                log_probs,
                text,
                self.vocab,
                upsample=self.upsample,
                blank_bias=self.blank_bias,
            )
        except (ValueError, RuntimeError):
            return None
        extent = speech_extent(log_probs, spans, upsample=self.upsample)
        if extent is None:
            return None
        return spans, extent[0], extent[1]


def build_head(
    *,
    vocab_size: int,
    input_dim: int = 2048,
    hidden_dim: int = 512,
    upsample: int = 2,
    blocks: int = 4,
    dropout: float = 0.1,
):
    """Construct the head. Imported lazily so this module stays importable."""
    import torch
    from torch import nn

    if vocab_size <= RESERVED_INDICES:
        raise ValueError("vocab_size must exceed the reserved blank/unk indices")
    if upsample < 1:
        raise ValueError("upsample must be >= 1")

    class ResidualConvBlock(nn.Module):
        """Dilated depthwise-separable conv, pre-norm, residual.

        Convolutional rather than attentional on purpose: alignment is a
        monotonic, local problem, and a conv stack cannot learn to reorder time
        the way self-attention can.
        """

        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(channels)
            self.depthwise = nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=2 * dilation,
                dilation=dilation,
                groups=channels,
            )
            self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: "torch.Tensor", mask=None) -> "torch.Tensor":
            residual = x
            y = self.norm(x)
            # Re-zero INSIDE the block, not only around it. `LayerNorm` maps an
            # all-zero padded position to its own bias, and the convolution one
            # line below reads across time - so an unmasked pad would feed that
            # bias into every real frame within the receptive field. Zeroing here
            # makes the batched result identical to running the clip alone, where
            # `conv1d` pads with actual zeros.
            if mask is not None:
                y = y * mask
            y = y.transpose(1, 2)
            y = self.pointwise(self.activation(self.depthwise(y))).transpose(1, 2)
            if mask is not None:
                y = y * mask
            return residual + self.dropout(y)

    class CtcAlignmentHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.upsample = int(upsample)
            self.input_norm = nn.LayerNorm(input_dim)
            self.project = nn.Linear(input_dim, hidden_dim)
            self.expand = (
                nn.ConvTranspose1d(
                    hidden_dim, hidden_dim, kernel_size=upsample, stride=upsample
                )
                if upsample > 1
                else None
            )
            self.blocks = nn.ModuleList(
                [
                    ResidualConvBlock(hidden_dim, dilation=2**index)
                    for index in range(blocks)
                ]
            )
            self.output_norm = nn.LayerNorm(hidden_dim)
            self.classifier = nn.Linear(hidden_dim, vocab_size)

        @property
        def context_frames(self) -> int:
            """One-sided receptive field, in ENCODER frames, rounded up.

            Read off the modules rather than restated as a constant, so that a
            change to the dilation schedule or the block count reaches every
            caller that has to supply context - notably the pipeline, which
            slices long audio into windows and must overlap them by at least
            this much or the head loses that context at every seam.

            With the defaults: kernel 5 at dilations 1/2/4/8 gives a 61-frame
            receptive field at 26 fps (2.35 s), so 30 output frames per side,
            i.e. 15 encoder frames (1.15 s).
            """
            span = 1
            for block in self.blocks:
                kernel = int(block.depthwise.kernel_size[0])
                dilation = int(block.depthwise.dilation[0])
                span += (kernel - 1) * dilation
            one_sided = (span - 1) // 2
            return -(-one_sided // self.upsample)

        def forward(self, features: "torch.Tensor", lengths=None) -> "torch.Tensor":
            """(B, T, input_dim) -> (B, T*upsample, vocab) log-probabilities.

            `lengths` is the real frame count of each row. Give it whenever the
            batch is padded: without it a short clip's tail is convolved against
            the padding's LayerNorm bias instead of against silence, so its
            output depends on which clips it was batched with. Length bucketing
            shrinks that contamination but cannot remove it. A single unpadded
            clip - which is every inference call - is unaffected either way, so
            omitting it stays numerically exact there.
            """
            mask = None
            if lengths is not None:
                positions = torch.arange(features.shape[1], device=features.device)
                mask = (positions[None, :] < lengths.to(features.device)[:, None]).to(
                    features.dtype
                )[..., None]
                features = features * mask
            x = self.project(self.input_norm(features))
            if mask is not None:
                x = x * mask
            if self.expand is not None:
                x = self.expand(x.transpose(1, 2)).transpose(1, 2)
                if mask is not None:
                    # Stride equals kernel, so a padded input frame maps to
                    # exactly `upsample` padded output frames and nothing mixes.
                    mask = mask.repeat_interleave(self.upsample, dim=1)
            if mask is not None:
                x = x * mask
            for block in self.blocks:
                x = block(x, mask)
            logits = self.classifier(self.output_norm(x))
            return nn.functional.log_softmax(logits, dim=-1)

    return CtcAlignmentHead()
