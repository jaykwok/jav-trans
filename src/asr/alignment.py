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

from dataclasses import dataclass
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
ALIGNMENT_MODEL_SCHEMA = "asr_ctc_alignment_head_v1"


def normalize_text(text: str) -> str:
    """Fold width and strip whitespace, so the same sound gets one target.

    NFKC collapses full-width latin/digits onto ASCII. Whitespace is removed
    entirely rather than mapped to a symbol: it is not pronounced, so giving it
    a target would force the head to explain silence it cannot hear.
    """
    folded = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in folded if not ch.isspace())


@dataclass(frozen=True)
class AlignmentVocab:
    """Character inventory shared by the head, its targets and its decoder."""

    chars: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.chars)) != len(self.chars):
            raise ValueError("alignment vocab contains duplicate characters")
        if any(len(ch) != 1 for ch in self.chars):
            raise ValueError("alignment vocab entries must be single characters")

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
        """Target sequence for CTC. Normalisation is applied here, once."""
        return [self.index_of(ch) for ch in normalize_text(text)]

    @classmethod
    def from_counts(
        cls, counts: dict[str, int], *, max_size: int = 0, min_count: int = 1
    ) -> "AlignmentVocab":
        """Build from character frequencies, most frequent first.

        Ordering by frequency makes `max_size` a coverage decision rather than an
        arbitrary cut: on the galgame corpus the top 2,000 of 3,080 characters
        cover 99.81% of occurrences.
        """
        eligible = [
            (char, count)
            for char, count in counts.items()
            if count >= min_count and len(char) == 1 and not char.isspace()
        ]
        eligible.sort(key=lambda item: (-item[1], item[0]))
        if max_size > 0:
            eligible = eligible[:max_size]
        return cls(chars=tuple(char for char, _ in eligible))

    def to_payload(self) -> dict:
        return {
            "schema": ALIGNMENT_VOCAB_SCHEMA,
            "blank_index": BLANK_INDEX,
            "unk_index": UNK_INDEX,
            "size": self.size,
            "chars": list(self.chars),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "AlignmentVocab":
        schema = str(payload.get("schema") or "")
        if schema != ALIGNMENT_VOCAB_SCHEMA:
            raise ValueError(
                f"alignment vocab schema must be {ALIGNMENT_VOCAB_SCHEMA}, got {schema!r}"
            )
        # A checkpoint whose blank moved would silently reinterpret every frame.
        if int(payload.get("blank_index", -1)) != BLANK_INDEX:
            raise ValueError("alignment vocab blank_index must be 0")
        if int(payload.get("unk_index", -1)) != UNK_INDEX:
            raise ValueError("alignment vocab unk_index must be 1")
        return cls(chars=tuple(str(ch) for ch in payload.get("chars") or ()))

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
):
    """Viterbi-align a known character sequence to per-frame posteriors.

    Written here rather than taken from `torchaudio.functional.forced_align`
    because torchaudio is not installed and pulling a CUDA wheel for one operator
    risks disturbing the working torch build. It also returns per-character
    scores, which S5 needs as its hallucination signal: text the acoustics do not
    support aligns badly, and the score is where that shows up.

    `log_probs` is (T, V) for a single utterance. Returns the character spans.
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

    # The CTC lattice: blanks interleaved around every label, so that repeated
    # characters are forced apart by a blank and cannot collapse into one.
    extended = [BLANK_INDEX]
    for token in targets:
        extended.extend((token, BLANK_INDEX))
    states = len(extended)
    if frames < states - len(targets):
        # Fewer frames than characters: no monotonic path exists at all.
        raise ValueError(
            f"cannot align {len(targets)} characters to {frames} frames; "
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
    scores[0] = log_probs[0, BLANK_INDEX]
    if states > 1:
        scores[1] = log_probs[0, extended[1]]
    backpointers = torch.zeros((frames, states), dtype=torch.uint8, device=device)

    for t in range(1, frames):
        stay = scores
        advance = torch.cat((torch.full((1,), neg_inf, device=device), scores[:-1]))
        skip = torch.cat((torch.full((2,), neg_inf, device=device), scores[:-2]))
        skip = torch.where(can_skip, skip, torch.full_like(skip, neg_inf))
        stacked = torch.stack((stay, advance, skip), dim=0)
        best, choice = stacked.max(dim=0)
        scores = best + log_probs[t].index_select(0, state_tokens)
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


def align_text(log_probs, text: str, vocab: AlignmentVocab, *, upsample: int = 2):
    """Align raw text by normalising and encoding it the same way training did."""
    normalized = normalize_text(text)
    targets = [vocab.index_of(ch) for ch in normalized]
    return forced_align(log_probs, targets, upsample=upsample, chars=normalized)


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
    BOTH edges - a lag would shift both the same way - and a blind listening
    pass on 2026-07-31 heard the head-side inset directly: cutting at the
    predicted start was called chopped 48.1% of the time against a 3.3% floor.

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

    def __init__(self, module, vocab: AlignmentVocab, upsample: int, device) -> None:
        self.module = module
        self.vocab = vocab
        self.upsample = int(upsample)
        self.device = device

    @classmethod
    def load(cls, checkpoint_path: str, *, device=None) -> "AlignmentHead":
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
        return cls(module, vocab, int(payload["upsample"]), resolved)

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
            spans = align_text(log_probs, text, self.vocab, upsample=self.upsample)
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

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            residual = x
            y = self.norm(x).transpose(1, 2)
            y = self.pointwise(self.activation(self.depthwise(y)))
            return residual + self.dropout(y.transpose(1, 2))

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

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            """(B, T, input_dim) -> (B, T*upsample, vocab) log-probabilities."""
            x = self.project(self.input_norm(features))
            if self.expand is not None:
                x = self.expand(x.transpose(1, 2)).transpose(1, 2)
            for block in self.blocks:
                x = block(x)
            logits = self.classifier(self.output_norm(x))
            return nn.functional.log_softmax(logits, dim=-1)

    return CtcAlignmentHead()
