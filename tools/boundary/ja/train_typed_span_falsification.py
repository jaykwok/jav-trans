#!/usr/bin/env python3
"""Falsification experiment for the typed-span target: does real-domain hold up?

This is deliberately NOT a bid for a production model.  It answers one question
before any of `src/boundary` gets rewritten around the typed-span target:

    when real and synthetic are mixed under a leakage-safe split, does the
    real-domain held-out metric survive?

The v11 post-mortem (docs/HISTORY.md:51) is the failure being tested for: train
inside-truth was entirely synthetic and held-out entirely real, so the model
learned provenance instead of speech and real continuity collapsed to 21.05%.
A single mixed run cannot detect that, because there is nothing to compare it
against.  Three arms on ONE held-out set can:

    real        train on the 13.7 h of real windows only
    synthetic   train on the 116.6 h of synthetic composites only
    mixed       train on both

All three are evaluated on the same real val/test partitions.  The readings:

  * `synthetic` scoring near `real` on real held-out means the synthetic domain
    transfers, and the 116 h are usable supervision.
  * `synthetic` collapsing on real held-out while scoring well on synthetic
    held-out is the v11 shortcut reproducing itself, and no amount of mixing
    ratio tuning fixes a target that only works in-domain.
  * `mixed` beating `real` is the result that justifies the typed-span rewrite;
    `mixed` losing to `real` means the 116 h are a liability at this ratio.

Frames whose label is unknown are excluded from both loss and metrics.  Real
windows are only ~55% covered, so this matters more than usual: scoring the
uncovered half as non-speech would let a degenerate model look competent.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
import threading
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boundary.gpu_safety import (  # noqa: E402
    apply_host_memory_cap,
    apply_vram_safety_cap,
)

IGNORE_INDEX = -100
SCHEMA = "typed_span_falsification_v1"
# speech / non_semantic_vocal / non_vocal, matching the packed `type_to_index`.
# `unsure` is data-level only and never occupies a class slot - it arrives as
# IGNORE_INDEX and is excluded from loss and metrics alike.
TYPE_CLASSES = 3
TYPE_NAMES = ("speech", "non_semantic_vocal", "non_vocal")
# The two tracks are exactly redundant on the speech axis: across 5,993,159
# jointly-labelled frames, `speech==1` iff `type==0`, with zero exceptions. So
# once the binary segmentation has been decided, the only open question on a
# non-speech span is non_semantic_vocal vs non_vocal - asking a span head to
# re-derive speech as well just hands it a 77% majority class to collapse onto.
NONSPEECH_TYPE_CLASSES = 2


def _rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class FrameStore:
    """Read-only view over the packed features plus the two label tracks.

    Two access modes, because the widest basis does not fit this box under the
    obvious one:

      mmap   whole file mapped; slicing is free. Touched pages join the process
             working set, so an epoch over a 12 GB file eventually holds all of
             it resident and machine-wide available RAM goes to ~0. On a 15.9
             GiB host that trips the system floor even though the process itself
             is small - the pages are clean, but they are not *available*.
      pread  seek and read each window. The bytes land in standby cache, which
             the OS counts as available and evicts under pressure, so RSS-like
             growth never happens. Costs a syscall and a copy per window.

    `auto` picks pread once the file is a large fraction of RAM. Both modes
    return identical arrays; only residency differs.
    """

    def __init__(self, run_dir: Path, *, loader: str = "auto") -> None:
        import psutil

        summary = json.loads((run_dir / "features_summary.json").read_text("utf-8"))
        self.width = int(summary["width"])
        self.total_frames = int(summary["total_frames"])
        self.ptm_dim = int(summary.get("ptm_dim") or 0)
        self.mfcc_dim = int(summary.get("mfcc_dim") or 0)
        self._path = run_dir / "features.f16"
        self._itemsize = 2  # float16

        size = self.total_frames * self.width * self._itemsize
        if loader == "auto":
            loader = (
                "pread"
                if size > 0.25 * int(psutil.virtual_memory().total)
                else "mmap"
            )
        self.loader = loader
        self._memmap = (
            np.memmap(
                self._path,
                dtype=np.float16,
                mode="r",
                shape=(self.total_frames, self.width),
            )
            if loader == "mmap"
            else None
        )
        self._handle = None if loader == "mmap" else self._path.open("rb")
        self._lock = threading.Lock()

        labels = np.load(run_dir / "frame_labels.npz")
        self.speech = labels["speech"]
        self.type = labels["type"]
        self.index = list(_rows(run_dir / "index.jsonl"))
        if self.speech.shape[0] != self.total_frames:
            raise SystemExit("label/feature frame count mismatch")

    def read(self, start: int, count: int) -> np.ndarray:
        """Frames [start, start+count) as float32 [count, width]."""
        if count <= 0:
            return np.zeros((0, self.width), dtype=np.float32)
        if self._memmap is not None:
            return np.asarray(self._memmap[start : start + count], dtype=np.float32)
        row_bytes = self.width * self._itemsize
        with self._lock:
            self._handle.seek(start * row_bytes)
            raw = self._handle.read(count * row_bytes)
        return (
            np.frombuffer(raw, dtype=np.float16)
            .reshape(-1, self.width)
            .astype(np.float32)
        )

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def subset(self, *, partition: str, provenance: str | None = None) -> list[dict]:
        return [
            row
            for row in self.index
            if row["partition"] == partition
            and (provenance is None or row["provenance"] == provenance)
        ]


def compute_normalization(
    store: FrameStore, examples: list[dict], *, sample_frames: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Feature mean/std from a frame sample.

    Computed once on the union of all training data and shared by every arm, so
    the arms differ only in which examples they see - not in their input scaling.

    Accumulated streaming rather than stacked: at 2088 dims the sampled frames
    are ~2.8 GB, and `np.concatenate` briefly doubles that. That transient was
    enough to push a 512-wide projector run to 9.12 GiB private and trip the
    host cap. Running sums cost O(width) instead.
    """
    rng = np.random.default_rng(seed)
    total = 0
    accum = np.zeros(store.width, dtype=np.float64)
    accum_sq = np.zeros(store.width, dtype=np.float64)
    budget = sample_frames
    order = rng.permutation(len(examples))
    for position in order:
        row = examples[int(position)]
        offset, count = int(row["frame_offset"]), int(row["frame_count"])
        take = min(count, 512)
        start = offset + int(rng.integers(0, max(1, count - take + 1)))
        block = store.read(start, take).astype(np.float64)
        accum += block.sum(axis=0)
        # einsum avoids materialising a second full-size squared array.
        accum_sq += np.einsum("ij,ij->j", block, block)
        total += block.shape[0]
        budget -= take
        if budget <= 0:
            break
    if total <= 0:
        raise SystemExit("normalization sampled no frames")
    mean = (accum / total).astype(np.float32)
    variance = accum_sq / total - (accum / total) ** 2
    std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
    std[std < 1e-5] = 1.0
    return mean, std


def build_model(
    width: int,
    hidden: int,
    dilations: tuple[int, ...],
    *,
    ptm_in: int = 0,
    projector_dim: int = 0,
    type_head: str = "none",
    aux_frame_type: bool = False,
):
    """Frame classifier, optionally behind a learnable PTM projection.

    With `projector_dim`, a bias-free Linear compresses the PTM block from
    `ptm_in` to `projector_dim` before the stem; MFCC columns pass through
    untouched. It is IDENTITY-SLICE initialised, so at step 0 the network sees
    exactly the leading-`projector_dim` truncation - the same protocol the
    07-06 island ablation used, which makes "learned projection vs truncation"
    a clean single-variable comparison that can only start level and diverge.

    Per-dimension input normalisation makes that equality exact rather than
    approximate: normalising 2048 dims then slicing N is identical to
    normalising a pre-sliced N.
    """
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, dilation: int) -> None:
            super().__init__()
            padding = 2 * dilation
            self.conv = nn.Conv1d(hidden, hidden, 5, padding=padding, dilation=dilation)
            self.norm = nn.GroupNorm(8, hidden)
            self.act = nn.GELU()

        def forward(self, x):
            y = self.conv(x)
            y = y[:, :, : x.shape[-1]]
            return x + self.act(self.norm(y))

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ptm_in = int(ptm_in)
            self.projector_dim = int(projector_dim)
            if self.projector_dim:
                self.ptm_projector = nn.Linear(
                    self.ptm_in, self.projector_dim, bias=False
                )
                with torch.no_grad():
                    weight = torch.zeros(self.projector_dim, self.ptm_in)
                    keep = min(self.projector_dim, self.ptm_in)
                    weight[:keep, :keep] = torch.eye(keep)
                    self.ptm_projector.weight.copy_(weight)
                stem_width = self.projector_dim + (width - self.ptm_in)
            else:
                self.ptm_projector = None
                stem_width = width
            self.stem = nn.Conv1d(stem_width, hidden, 1)
            self.blocks = nn.ModuleList(Block(d) for d in dilations)
            self.head = nn.Conv1d(hidden, 2, 1)
            # Typing is the other half of the typed-span target. `frame` types
            # every frame independently; `span` types a whole decoded span from
            # one pooled representation, which is the form plan B actually
            # calls for (one type token per span, not per frame).
            self.type_head = str(type_head)
            if self.type_head == "frame":
                self.type_frame = nn.Conv1d(hidden, TYPE_CLASSES, 1)
            elif self.type_head == "span":
                # mean over the span plus both endpoints: a span's identity is
                # carried as much by how it starts and ends as by its average.
                self.type_span = nn.Linear(3 * hidden, TYPE_CLASSES)
            elif self.type_head == "span_cond":
                # Conditioned on the segmentation: speech spans are typed by
                # construction, so the head only separates the two non-speech
                # kinds and never sees the majority class at all.
                self.type_span = nn.Linear(3 * hidden, NONSPEECH_TYPE_CLASSES)
            elif self.type_head != "none":
                raise ValueError(f"unknown type head: {type_head!r}")
            # Training-only auxiliary head. The span head sees one gradient per
            # span, which for a class holding 1.9% of supervised frames is thin
            # enough that two seeds in three collapse onto the other class.
            # This one supervises every non-speech frame, so the encoder is
            # pushed toward local separability even when the span token is not
            # learning. It never participates in decoding.
            self.aux_frame_type = bool(aux_frame_type)
            if self.aux_frame_type:
                if self.type_head not in ("span", "span_cond"):
                    raise ValueError(
                        "auxiliary frame typing only supplements a span type head"
                    )
                self.type_frame_aux = nn.Conv1d(hidden, NONSPEECH_TYPE_CLASSES, 1)

        def forward_features(self, x):
            """x: [B,T,width] -> (encoded [B,T,hidden], logits [B,T,2]).

            The Dense Span decoder scores spans from the pre-head representation
            as well as the frame logits, so the encoder output has to be
            reachable without recomputing the stack.
            """
            if self.ptm_projector is not None:
                x = torch.cat(
                    (
                        self.ptm_projector(x[..., : self.ptm_in]),
                        x[..., self.ptm_in :],
                    ),
                    dim=-1,
                )
            h = self.stem(x.transpose(1, 2))
            for block in self.blocks:
                h = block(h)
            return h.transpose(1, 2), self.head(h).transpose(1, 2)

        def forward(self, x):
            # x: [B, T, width] -> logits [B, T, 2]
            return self.forward_features(x)[1]

    return Net()


INPUT_MODALITIES = ("both", "ptm", "mfcc")
# Index of `non_vocal` in the packed type track.
RARE_TYPE_INDEX = 2


def windows_containing_type(
    store: FrameStore, examples: list[dict], type_index: int
) -> list[dict]:
    """Examples holding at least one frame of the given type."""
    selected = []
    for row in examples:
        offset, count = int(row["frame_offset"]), int(row["frame_count"])
        if bool((store.type[offset : offset + count] == type_index).any()):
            selected.append(row)
    return selected


def apply_modality_mask(
    features: np.ndarray, ptm_dim: int, modality: str
) -> np.ndarray:
    """Zero one input block so the encoder cannot read it.

    Zeroing rather than slicing keeps the architecture, the parameter count and
    the normalisation statistics identical across arms, so the only thing that
    differs between them is which information reaches the stem. Applied after
    normalisation, since a zeroed column is only uninformative once the mean
    has already been removed.
    """
    if modality == "both":
        return features
    if modality not in INPUT_MODALITIES:
        raise ValueError(f"unknown input modality: {modality!r}")
    if ptm_dim <= 0 or ptm_dim >= features.shape[-1]:
        raise ValueError("modality ablation needs a real PTM/MFCC split")
    masked = features.copy()
    if modality == "ptm":
        masked[..., ptm_dim:] = 0.0
    else:
        masked[..., :ptm_dim] = 0.0
    return masked


def sample_batch(
    store: FrameStore,
    pools: list[tuple[list[dict], float]],
    *,
    batch_size: int,
    window: int,
    rng: np.random.Generator,
    mean: np.ndarray,
    std: np.ndarray,
    modality: str = "both",
    rare_pools: list[list[dict]] | None = None,
    rare_fraction: float = 0.0,
):
    """Draw a batch, choosing the source pool by weight then the example within.

    Weights matter because the natural mixture is 89% synthetic: sampling
    uniformly over examples answers "does synthetic help at its own scale",
    while equal pool weights answer "does synthetic help at all". Those are
    different questions and the first one cannot stand in for the second.
    """
    weights = np.array([w for _, w in pools], dtype=np.float64)
    weights = weights / weights.sum()
    features = np.zeros((batch_size, window, store.width), dtype=np.float32)
    labels = np.full((batch_size, window), IGNORE_INDEX, dtype=np.int64)
    types = np.full((batch_size, window), IGNORE_INDEX, dtype=np.int64)
    for slot in range(batch_size):
        pool_index = int(rng.choice(len(pools), p=weights))
        examples = pools[pool_index][0]
        # Window-level oversampling. This is a different lever from class
        # weighting, which was tried and hurt: weighting scales the gradient of
        # whatever was drawn, while `non_vocal` appears in only 20.5% of real
        # windows with a per-window median of 0.00%, so most steps have no
        # gradient for it at all. Reweighting nothing is still nothing.
        if rare_fraction > 0.0 and rare_pools is not None:
            rare = rare_pools[pool_index]
            if rare and float(rng.random()) < rare_fraction:
                examples = rare
        row = examples[int(rng.integers(0, len(examples)))]
        offset, count = int(row["frame_offset"]), int(row["frame_count"])
        take = min(window, count)
        start = offset + int(rng.integers(0, max(1, count - take + 1)))
        block = store.read(start, take)
        features[slot, :take] = (block - mean) / std
        labels[slot, :take] = store.speech[start : start + take]
        types[slot, :take] = store.type[start : start + take]
    return apply_modality_mask(features, store.ptm_dim, modality), labels, types


def constant_runs(values: np.ndarray) -> list[tuple[int, int]]:
    """Maximal half-open runs of a constant value: [(start, end), ...]."""
    if values.size == 0:
        return []
    edges = np.flatnonzero(values[1:] != values[:-1]) + 1
    bounds = [0, *edges.tolist(), int(values.size)]
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


def gold_type_spans(
    speech: np.ndarray, types: np.ndarray
) -> list[tuple[int, int, int]]:
    """Spans of the gold segmentation that carry a known type.

    A span is a maximal run of constant speech/non-speech, cut at the edge of
    the supervised region so an unlabelled gap never merges two spans into one.
    Its type is the majority known type inside it; spans whose type is entirely
    unknown are dropped rather than guessed, because `unsure` is exactly the
    label the schema refuses to force.
    """
    spans: list[tuple[int, int, int]] = []
    for region_start, region_end in _runs(speech != IGNORE_INDEX):
        local = speech[region_start:region_end]
        for start, end in constant_runs(local):
            window = types[region_start + start : region_start + end]
            known = window[window != IGNORE_INDEX]
            if known.size == 0:
                continue
            counts = np.bincount(known, minlength=TYPE_CLASSES)
            spans.append(
                (region_start + start, region_start + end, int(counts.argmax()))
            )
    return spans


def type_class_counts(
    store: FrameStore,
    pools: list[tuple[list[dict], float]],
    *,
    type_head: str,
) -> np.ndarray:
    """Training support per type class, counted at the head's own granularity.

    A frame head sees frames and a span head sees spans, and the two
    distributions are not proportional: non_vocal is a small fraction of frames
    but its spans are short, so it is a larger fraction of spans. Weighting a
    span head by frame counts would therefore over-correct.
    """
    conditional = type_head == "span_cond"
    size = NONSPEECH_TYPE_CLASSES if conditional else TYPE_CLASSES
    counts = np.zeros(size, dtype=np.int64)
    seen: set[int] = set()
    for examples, _ in pools:
        for row in examples:
            offset, count = int(row["frame_offset"]), int(row["frame_count"])
            if offset in seen:  # `mixed` passes the same pool twice
                continue
            seen.add(offset)
            types = store.type[offset : offset + count]
            if type_head == "frame":
                known = types[types != IGNORE_INDEX]
                if known.size:
                    counts += np.bincount(known, minlength=TYPE_CLASSES)
                continue
            speech = store.speech[offset : offset + count]
            for _, _, value in gold_type_spans(speech, types):
                if conditional:
                    if value != 0:
                        counts[value - 1] += 1
                else:
                    counts[value] += 1
    return counts


def inverse_frequency_weights(counts: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights normalised to mean 1.

    Mean 1 keeps the type term's overall magnitude comparable to the unweighted
    run, so `--type-weight` still means the same thing across arms and the
    weighting is a single isolated change.
    """
    positive = np.maximum(counts, 1).astype(np.float64)
    weights = positive.sum() / (positive.size * positive)
    return weights / weights.mean()


def nonspeech_frame_targets(types: np.ndarray) -> np.ndarray:
    """Per-frame nsv/nv targets, with speech and unknown frames masked out.

    The auxiliary head answers the same question as `span_cond` - which of the
    two non-speech kinds is this - so it must be blind to speech for the same
    reason: speech is already decided by the segmentation, and letting the
    majority class into the target is what made the unconditional span head
    collapse.
    """
    targets = np.full(types.shape, IGNORE_INDEX, dtype=np.int64)
    non_speech = (types == 1) | (types == 2)
    targets[non_speech] = types[non_speech] - 1
    return targets


def pool_spans(encoded, spans: list[tuple[int, int]]):
    """Mean-over-span concatenated with both endpoints, vectorised by cumsum.

    `encoded` is [T, hidden] for one sequence; returns [len(spans), 3*hidden].
    """
    import torch

    if not spans:
        return encoded.new_zeros((0, 3 * encoded.shape[-1]))
    starts = torch.tensor([s for s, _ in spans], device=encoded.device)
    ends = torch.tensor([e for _, e in spans], device=encoded.device)
    prefix = torch.cat(
        (encoded.new_zeros((1, encoded.shape[-1])), encoded.cumsum(dim=0)), dim=0
    )
    means = (prefix[ends] - prefix[starts]) / (ends - starts).unsqueeze(-1)
    return torch.cat((means, encoded[starts], encoded[ends - 1]), dim=-1)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of True."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def event_metrics(
    truth: np.ndarray, predicted: np.ndarray, known: np.ndarray, *, iou: float
) -> Counter:
    """Event-level speech detection, evaluated only inside known regions.

    Runs are cut at the edge of each known region rather than bridged across it,
    so an uncovered gap never fabricates or merges an event.
    """
    counts: Counter = Counter()
    for region_start, region_end in _runs(known):
        truth_runs = _runs(truth[region_start:region_end] == 1)
        pred_runs = _runs(predicted[region_start:region_end] == 1)
        used = set()
        for t_start, t_end in truth_runs:
            best, best_score = -1, 0.0
            for position, (p_start, p_end) in enumerate(pred_runs):
                if position in used:
                    continue
                overlap = max(0, min(t_end, p_end) - max(t_start, p_start))
                if overlap <= 0:
                    continue
                union = max(t_end, p_end) - min(t_start, p_start)
                score = overlap / union if union else 0.0
                if score > best_score:
                    best, best_score = position, score
            if best >= 0 and best_score >= iou:
                used.add(best)
                counts["tp"] += 1
            else:
                counts["fn"] += 1
        counts["fp"] += len(pred_runs) - len(used)
    return counts


def evaluate(
    model,
    store: FrameStore,
    examples: list[dict],
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device,
    window: int,
    iou: float,
    decode_mode: str = "argmax",
    modality: str = "both",
) -> dict:
    import torch

    model.eval()
    frame = Counter()
    events = Counter()
    type_confusion = np.zeros((TYPE_CLASSES, TYPE_CLASSES), dtype=np.int64)
    aux_confusion = np.zeros(
        (NONSPEECH_TYPE_CLASSES, NONSPEECH_TYPE_CLASSES), dtype=np.int64
    )
    type_head = getattr(model, "type_head", "none")
    per_window: list[dict] = []
    with torch.no_grad():
        for row in examples:
            offset, count = int(row["frame_offset"]), int(row["frame_count"])
            block = apply_modality_mask(
                (store.read(offset, count) - mean) / std, store.ptm_dim, modality
            )
            truth = store.speech[offset : offset + count].astype(np.int64)

            predicted = np.zeros(count, dtype=np.int64)
            predicted_type = np.full(count, IGNORE_INDEX, dtype=np.int64)
            aux_predicted_type = np.zeros(count, dtype=np.int64)
            encoded_window = (
                np.zeros((count, model.head.in_channels), dtype=np.float32)
                if type_head in ("span", "span_cond")
                else None
            )
            for start in range(0, count, window):
                end = min(count, start + window)
                chunk = torch.from_numpy(block[start:end]).to(device).unsqueeze(0)
                encoded, logits = model.forward_features(chunk)
                if decode_mode == "viterbi":
                    prefix = torch.ones(
                        chunk.shape[:2], dtype=torch.bool, device=device
                    )
                    lattice = model.dense_span.span_scores(encoded, logits, prefix)
                    predicted[start:end] = (
                        model.dense_span.decode(lattice, prefix)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )
                else:
                    predicted[start:end] = (
                        logits.argmax(dim=-1).squeeze(0).cpu().numpy()
                    )
                if type_head == "frame":
                    predicted_type[start:end] = (
                        model.type_frame(encoded.transpose(1, 2))
                        .argmax(dim=1)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )
                elif type_head in ("span", "span_cond"):
                    encoded_window[start:end] = encoded.squeeze(0).cpu().numpy()
                if getattr(model, "aux_frame_type", False):
                    aux_predicted_type[start:end] = (
                        model.type_frame_aux(encoded.transpose(1, 2))
                        .argmax(dim=1)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )

            if type_head in ("span", "span_cond"):
                # Type the spans the model actually produced, then project the
                # per-span token back onto frames so the span head and the
                # frame head are scored on one identical axis.
                spans = constant_runs(predicted)
                tensor = torch.from_numpy(encoded_window).to(device)
                if type_head == "span":
                    pooled = pool_spans(tensor, spans)
                    assigned = model.type_span(pooled).argmax(dim=-1).cpu().numpy()
                    for (span_start, span_end), value in zip(
                        spans, assigned.tolist(), strict=True
                    ):
                        predicted_type[span_start:span_end] = int(value)
                else:
                    # Predicted-speech spans are typed by the segmentation
                    # itself; only the rest reach the head.
                    non_speech = [(s, e) for s, e in spans if predicted[s] == 0]
                    for span_start, span_end in spans:
                        if predicted[span_start] == 1:
                            predicted_type[span_start:span_end] = 0
                    if non_speech:
                        assigned = (
                            model.type_span(pool_spans(tensor, non_speech))
                            .argmax(dim=-1)
                            .cpu()
                            .numpy()
                        )
                        for (span_start, span_end), value in zip(
                            non_speech, assigned.tolist(), strict=True
                        ):
                            predicted_type[span_start:span_end] = 1 + int(value)

            if getattr(model, "aux_frame_type", False):
                # Scored only where the truth is non-speech, which is the only
                # place this head is defined. Comparing it against the span
                # head on the same frames is the diagnostic: frame separable
                # but span not points at pooling, neither separable points at
                # the encoder or the labels.
                truth_type = store.type[offset : offset + count].astype(np.int64)
                aux_scored = (truth_type == 1) | (truth_type == 2)
                if bool(aux_scored.any()):
                    np.add.at(
                        aux_confusion,
                        (
                            truth_type[aux_scored] - 1,
                            aux_predicted_type[aux_scored],
                        ),
                        1,
                    )

            if type_head != "none":
                truth_type = store.type[offset : offset + count].astype(np.int64)
                scored = (truth_type != IGNORE_INDEX) & (
                    predicted_type != IGNORE_INDEX
                )
                if bool(scored.any()):
                    np.add.at(
                        type_confusion,
                        (truth_type[scored], predicted_type[scored]),
                        1,
                    )

            known = truth != IGNORE_INDEX
            if not bool(known.any()):
                continue
            t = truth[known]
            p = predicted[known]
            frame["n"] += int(t.size)
            frame["correct"] += int(np.count_nonzero(t == p))
            frame["tp"] += int(np.count_nonzero((t == 1) & (p == 1)))
            frame["fp"] += int(np.count_nonzero((t == 0) & (p == 1)))
            frame["fn"] += int(np.count_nonzero((t == 1) & (p == 0)))
            frame["tn"] += int(np.count_nonzero((t == 0) & (p == 0)))
            local_events = event_metrics(truth, predicted, known, iou=iou)
            events.update(local_events)
            # Per-window counts are what make an eval-set-size bootstrap
            # possible: the window is the resampling unit, because frames
            # within one window are anything but independent.
            per_window.append({
                "example_id": row["example_id"],
                "n": int(t.size),
                "tp": int(np.count_nonzero((t == 1) & (p == 1))),
                "fp": int(np.count_nonzero((t == 0) & (p == 1))),
                "fn": int(np.count_nonzero((t == 1) & (p == 0))),
                "tn": int(np.count_nonzero((t == 0) & (p == 0))),
                "e_tp": int(local_events["tp"]),
                "e_fp": int(local_events["fp"]),
                "e_fn": int(local_events["fn"]),
            })
    model.train()

    def prf(tp: int, fp: int, fn: int) -> dict:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    type_report: dict = {}
    if type_confusion.sum():
        per_class = {}
        for index, name in enumerate(TYPE_NAMES):
            tp = int(type_confusion[index, index])
            fp = int(type_confusion[:, index].sum()) - tp
            fn = int(type_confusion[index, :].sum()) - tp
            per_class[name] = {**prf(tp, fp, fn), "support": tp + fn}
        total = int(type_confusion.sum())
        type_report = {
            "frames_scored": total,
            "accuracy": round(int(np.trace(type_confusion)) / total, 4),
            # Macro F1 is the number to read: `non_vocal` is only ~4% of the
            # scored frames, so a micro average hides it entirely.
            "macro_f1": round(
                sum(v["f1"] for v in per_class.values()) / TYPE_CLASSES, 4
            ),
            "per_class": per_class,
            "confusion": type_confusion.tolist(),
        }

    aux_report: dict = {}
    if aux_confusion.sum():
        aux_per_class = {}
        for index, name in enumerate(TYPE_NAMES[1:]):
            tp = int(aux_confusion[index, index])
            fp = int(aux_confusion[:, index].sum()) - tp
            fn = int(aux_confusion[index, :].sum()) - tp
            aux_per_class[name] = {**prf(tp, fp, fn), "support": tp + fn}
        aux_total = int(aux_confusion.sum())
        aux_report = {
            "frames_scored": aux_total,
            "accuracy": round(int(np.trace(aux_confusion)) / aux_total, 4),
            "macro_f1": round(
                sum(v["f1"] for v in aux_per_class.values()) / NONSPEECH_TYPE_CLASSES,
                4,
            ),
            "per_class": aux_per_class,
            "confusion": aux_confusion.tolist(),
        }

    n = frame["n"] or 1
    return {
        **({"type": type_report} if type_report else {}),
        **({"type_aux_frame": aux_report} if aux_report else {}),
        "frames_scored": int(frame["n"]),
        "frame_accuracy": round(frame["correct"] / n, 4),
        "speech_positive_rate_truth": round((frame["tp"] + frame["fn"]) / n, 4),
        "speech_positive_rate_pred": round((frame["tp"] + frame["fp"]) / n, 4),
        "speech_frame": prf(frame["tp"], frame["fp"], frame["fn"]),
        # Non-speech is the class the v12 target could not supervise; it is the
        # one worth reading first.
        "non_speech_frame": prf(frame["tn"], frame["fn"], frame["fp"]),
        "speech_event": {
            **prf(events["tp"], events["fp"], events["fn"]),
            "tp": int(events["tp"]),
            "fp": int(events["fp"]),
            "fn": int(events["fn"]),
        },
        "per_window": per_window,
    }


def run_arm(
    *,
    arm: str,
    store: FrameStore,
    train_pools: list[tuple[list[dict], float]],
    eval_sets: dict[str, list[dict]],
    mean: np.ndarray,
    std: np.ndarray,
    steps: int,
    eval_every: int,
    batch_size: int,
    window: int,
    hidden: int,
    learning_rate: float,
    seed: int,
    iou: float,
    device,
    memory_guard,
    projector_dim: int = 0,
    decoder: str = "frame",
    dense_span_rank: int = 32,
    dense_span_duration_hidden: int = 32,
    dense_span_hinge_weight: float = 1.0,
    type_head: str = "none",
    type_weight: float = 1.0,
    type_class_weighting: str = "none",
    aux_frame_type_weight: float = 0.0,
    modality: str = "both",
    rare_window_fraction: float = 0.0,
) -> dict:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(
        store.width,
        hidden,
        (1, 2, 4, 8, 16, 32),
        ptm_in=store.ptm_dim,
        projector_dim=projector_dim,
        type_head=type_head,
        aux_frame_type=aux_frame_type_weight > 0.0,
    ).to(device)
    if decoder == "dense_span":
        from boundary.dense_span import BinaryDenseSpanDecoder

        # Attached before the optimizer exists so its parameters are trained.
        model.dense_span = BinaryDenseSpanDecoder(
            hidden_size=hidden,
            rank=dense_span_rank,
            duration_hidden_size=dense_span_duration_hidden,
            context_window_frames=window,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    rare_pools = None
    if rare_window_fraction > 0.0:
        rare_pools = [
            windows_containing_type(store, examples, RARE_TYPE_INDEX)
            for examples, _ in train_pools
        ]
        print(
            f"  [{arm}] rare-window oversampling {rare_window_fraction:.2f}: "
            + ", ".join(
                f"{len(rare)}/{len(examples)}"
                for rare, (examples, _) in zip(rare_pools, train_pools, strict=True)
            ),
            flush=True,
        )

    type_class_weights = None
    if type_head != "none" and type_class_weighting == "inverse":
        counts = type_class_counts(store, train_pools, type_head=type_head)
        type_class_weights = inverse_frequency_weights(counts)
        print(
            f"  [{arm}] type support {counts.tolist()} -> class weights "
            f"{[round(float(w), 3) for w in type_class_weights]}",
            flush=True,
        )
    type_loss_fn = nn.CrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
        weight=(
            None
            if type_class_weights is None
            else torch.tensor(type_class_weights, dtype=torch.float32, device=device)
        ),
    )
    # Unweighted on purpose: the auxiliary head exists to densify the gradient,
    # and stacking class weighting on top would confound "more supervision"
    # with "reweighted supervision", which was already measured separately.
    aux_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    history: list[dict] = []
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if memory_guard is not None:
        # Rebase so the reported peak belongs to this arm, not to whatever the
        # previous arm left resident.
        memory_guard.peak_bytes = memory_guard.process_bytes()
    for step in range(1, steps + 1):
        # Both budgets are checked every step so an over-large batch fails on
        # the step that caused it, not after an hour of swapping.
        if memory_guard is not None:
            if memory_guard.tripped is not None:
                raise memory_guard.tripped
            memory_guard.check()
        features, labels, types = sample_batch(
            store,
            train_pools,
            batch_size=batch_size,
            window=window,
            rng=rng,
            mean=mean,
            std=std,
            modality=modality,
            rare_pools=rare_pools,
            rare_fraction=rare_window_fraction,
        )
        x = torch.from_numpy(features).to(device)
        y = torch.from_numpy(labels).to(device)
        encoded, logits = model.forward_features(x)
        loss = loss_fn(logits.reshape(-1, 2), y.reshape(-1))
        if type_head == "frame":
            type_logits = model.type_frame(encoded.transpose(1, 2)).transpose(1, 2)
            loss = loss + type_weight * type_loss_fn(
                type_logits.reshape(-1, TYPE_CLASSES),
                torch.from_numpy(types).to(device).reshape(-1),
            )
        elif type_head in ("span", "span_cond"):
            # Typed from the GOLD segmentation during training: the point of
            # the comparison is the pooling, so the span head must not also be
            # handed the decoder's segmentation errors while the frame head is
            # handed none.
            pooled, targets = [], []
            for slot in range(batch_size):
                spans = gold_type_spans(labels[slot], types[slot])
                if type_head == "span_cond":
                    # Speech spans carry no signal for this head; their type is
                    # already fixed by the segmentation.
                    spans = [(s, e, t) for s, e, t in spans if t != 0]
                if not spans:
                    continue
                pooled.append(
                    pool_spans(encoded[slot], [(s, e) for s, e, _ in spans])
                )
                offset = 1 if type_head == "span_cond" else 0
                targets.extend(t - offset for _, _, t in spans)
            if targets:
                loss = loss + type_weight * type_loss_fn(
                    model.type_span(torch.cat(pooled, dim=0)),
                    torch.tensor(targets, device=device, dtype=torch.long),
                )
        if getattr(model, "aux_frame_type", False):
            aux_logits = model.type_frame_aux(encoded.transpose(1, 2)).transpose(1, 2)
            aux_targets = torch.from_numpy(nonspeech_frame_targets(types)).to(device)
            if bool((aux_targets != IGNORE_INDEX).any()):
                loss = loss + aux_frame_type_weight * aux_loss_fn(
                    aux_logits.reshape(-1, NONSPEECH_TYPE_CLASSES),
                    aux_targets.reshape(-1),
                )
        if decoder == "dense_span":
            supervised = y != IGNORE_INDEX
            # A batch drawn entirely from uncovered regions has no definite
            # owner run; the decoder raises on that rather than silently
            # scoring nothing, so skip the hinge for that step instead.
            if bool(supervised.any()):
                prefix = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
                lattice = model.dense_span.span_scores(encoded, logits, prefix)
                loss = loss + dense_span_hinge_weight * model.dense_span.structured_hinge(
                    lattice, y.clamp_min(0), supervised
                )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()

        if step % eval_every == 0 or step == steps:
            point = {"step": step, "train_loss": round(float(loss.item()), 4)}
            primary = "viterbi" if decoder == "dense_span" else "argmax"
            for name, examples in eval_sets.items():
                if not examples:
                    continue
                point[name] = evaluate(
                    model, store, examples, mean=mean, std=std, device=device,
                    window=window, iou=iou, decode_mode=primary,
                )
                if primary == "viterbi":
                    # Same weights, plain argmax. Reading the two side by side
                    # separates what the structured loss did to the frame head
                    # from what the Viterbi decode did on top of it - otherwise
                    # "dense span helped" conflates two changes.
                    point[f"{name}__argmax"] = evaluate(
                        model, store, examples, mean=mean, std=std, device=device,
                        window=window, iou=iou, decode_mode="argmax",
                    )
            history.append(point)
            real_val = point.get("val_real", {})
            print(
                f"  [{arm}] step {step:>5}  loss {point['train_loss']:.4f}  "
                f"val_real acc {real_val.get('frame_accuracy', 0.0):.4f}  "
                f"speechF1 {real_val.get('speech_frame', {}).get('f1', 0.0):.4f}  "
                f"nonspeechF1 {real_val.get('non_speech_frame', {}).get('f1', 0.0):.4f}",
                flush=True,
            )

    # Selection on real val non-speech F1: the majority class is speech, so
    # accuracy and speech F1 both reward the degenerate all-speech predictor.
    best = max(
        history,
        key=lambda point: point.get("val_real", {})
        .get("non_speech_frame", {})
        .get("f1", 0.0),
    )
    peak_vram = (
        torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    )
    return {
        "arm": arm,
        "train_examples": sum(len(pool) for pool, _ in train_pools),
        "train_frames": sum(
            int(r["frame_count"]) for pool, _ in train_pools for r in pool
        ),
        "pool_weights": [round(float(w), 4) for _, w in train_pools],
        "projector_dim": int(projector_dim),
        "ptm_in": int(store.ptm_dim),
        "steps": steps,
        "seconds": round(time.time() - started, 1),
        "peak_vram_gib": round(peak_vram / 2**30, 3),
        "peak_host_gib": (
            round(memory_guard.peak_bytes / 2**30, 3)
            if memory_guard is not None
            else None
        ),
        "history": history,
        "selected": best,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, help="materializer output dir")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--arms",
        default="real,synthetic,mixed",
        help="Comma-separated subset of real,synthetic,mixed.",
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--norm-sample-frames", type=int, default=2_000_000)
    parser.add_argument(
        "--vram-ratio",
        type=float,
        default=None,
        help="Cap CUDA allocations at this fraction of physical VRAM (default 0.95).",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction of real training VIDEOS to keep, for a learning curve.",
    )
    parser.add_argument(
        "--projector-dim",
        type=int,
        default=0,
        help=(
            "Learn a PTM projection to this width, identity-slice initialised. "
            "0 uses the packed features as-is."
        ),
    )
    parser.add_argument(
        "--loader",
        choices=["auto", "mmap", "pread"],
        default="auto",
        help="Feature access mode; pread avoids pinning a large file in RAM.",
    )
    parser.add_argument(
        "--decoder",
        choices=["frame", "dense_span"],
        default="frame",
        help=(
            "frame: per-frame CE + argmax (the falsification baseline). "
            "dense_span: adds src/boundary/dense_span.py structured hinge and "
            "decodes with exact Viterbi over the full span lattice, which is "
            "threshold-free by construction."
        ),
    )
    parser.add_argument(
        "--type-head",
        choices=["none", "frame", "span", "span_cond"],
        default="none",
        help=(
            "none: binary speech only. frame: a 3-way type per frame. "
            "span: ONE 3-way type token per span. span_cond: one token per "
            "span, but conditioned on the segmentation so the head only "
            "decides non_semantic_vocal vs non_vocal. Synthetic data carries "
            "no non-speech type at all, so all of these are supervised by the "
            "real domain only."
        ),
    )
    parser.add_argument("--type-weight", type=float, default=1.0)
    parser.add_argument(
        "--type-class-weighting",
        choices=["none", "inverse"],
        default="none",
        help=(
            "inverse: weight the type loss by inverse class frequency, counted "
            "at the head's own granularity (frames for --type-head frame, "
            "spans for the span heads) and normalised to mean 1."
        ),
    )
    parser.add_argument(
        "--aux-frame-type-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of a training-only per-frame nsv/nv head that supplements "
            "a span type head. The span head gets one gradient per span, which "
            "is thin for a class holding under 2%% of supervised frames; this "
            "one supervises every non-speech frame. It never takes part in "
            "decoding, and its own frame-level score is reported separately as "
            "`type_aux_frame` so span-vs-frame separability can be compared."
        ),
    )
    parser.add_argument(
        "--rare-window-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of batch slots drawn from windows that contain at least "
            "one non_vocal frame. Only 20.5%% of real training windows do, and "
            "the median window contains none, so most steps carry no gradient "
            "for the class. This changes WHICH windows are drawn, unlike "
            "--type-class-weighting, which changes how a drawn one is scored."
        ),
    )
    parser.add_argument(
        "--input-modality",
        choices=list(INPUT_MODALITIES),
        default="both",
        help=(
            "Zero one input block to measure what it contributes. The stem "
            "sees PTM and MFCC concatenated with no alignment layer between "
            "them, so a branch the encoder learned to ignore would look "
            "exactly like a branch that carries nothing."
        ),
    )
    parser.add_argument("--dense-span-rank", type=int, default=32)
    parser.add_argument("--dense-span-duration-hidden", type=int, default=32)
    parser.add_argument(
        "--dense-span-hinge-weight",
        type=float,
        default=1.0,
        help="Weight on the structured hinge; frame CE always stays at 1.0.",
    )
    parser.add_argument(
        "--host-memory-ratio",
        type=float,
        default=None,
        help="Cap process RSS at this fraction of available host RAM (default 0.95).",
    )
    args = parser.parse_args(argv)

    import torch

    vram_ratio = apply_vram_safety_cap(args.vram_ratio)
    memory_guard = apply_host_memory_cap(args.host_memory_ratio)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"vram cap      : {vram_ratio if vram_ratio is None else f'{vram_ratio:.0%}'}"
        " of physical VRAM"
    )
    print(
        f"host ram cap  : {memory_guard.ratio:.0%} -> "
        f"{memory_guard.budget_bytes / 2**30:.2f} GiB process budget, "
        f"{memory_guard.system_floor_bytes / 2**30:.2f} GiB system floor"
    )

    frames_dir = Path(args.frames).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    store = FrameStore(frames_dir, loader=args.loader)

    train_real = store.subset(partition="train", provenance="real_omni_joint")
    if args.train_fraction < 1.0:
        # Subsample by GROUP (video), not by window. Dropping individual windows
        # would leave sibling windows of the same video in the set, so a smaller
        # "train set" would still have seen every video - which measures nothing
        # about how much new footage is worth.
        groups = sorted({str(r["group_id"]) for r in train_real})
        keep_n = max(1, int(round(len(groups) * float(args.train_fraction))))
        rng = np.random.default_rng(int(args.seed))
        keep = set(rng.permutation(groups)[:keep_n].tolist())
        before = len(train_real)
        train_real = [r for r in train_real if str(r["group_id"]) in keep]
        print(
            f"train subsample: {len(keep)}/{len(groups)} videos, "
            f"{len(train_real)}/{before} windows, "
            f"{sum(int(r['frame_count']) for r in train_real) * 0.02 / 3600:.2f} h"
        )
    train_synth = store.subset(partition="train", provenance="synthetic_hardmix")
    eval_sets = {
        "val_real": store.subset(partition="val", provenance="real_omni_joint"),
        "test_real": store.subset(partition="test", provenance="real_omni_joint"),
        "val_synthetic": store.subset(partition="val", provenance="synthetic_hardmix"),
    }

    print(f"device        : {device}")
    print(f"train real    : {len(train_real)} examples")
    print(f"train synth   : {len(train_synth)} examples")
    for name, examples in eval_sets.items():
        print(f"eval {name:<14}: {len(examples)} examples")

    mean, std = compute_normalization(
        store,
        train_real + train_synth,
        sample_frames=int(args.norm_sample_frames),
        seed=int(args.seed),
    )
    memory_guard.check()
    np.savez(output / "normalization.npz", mean=mean, std=std)

    # `mixed` weights each pool by its example count, reproducing the corpus as
    # it actually is (89% synthetic). `mixed_balanced` weights the two pools
    # equally, which is the only way to tell a bad domain from a bad ratio.
    pools = {
        "real": [(train_real, 1.0)],
        "synthetic": [(train_synth, 1.0)],
        "mixed": [
            (train_real, float(len(train_real))),
            (train_synth, float(len(train_synth))),
        ],
        "mixed_balanced": [(train_real, 1.0), (train_synth, 1.0)],
    }
    results = []
    for arm in (a.strip() for a in args.arms.split(",") if a.strip()):
        if arm not in pools:
            raise SystemExit(f"unknown arm: {arm}")
        print(f"\n=== arm: {arm} ===", flush=True)
        results.append(
            run_arm(
                arm=arm,
                store=store,
                train_pools=pools[arm],
                eval_sets=eval_sets,
                mean=mean,
                std=std,
                steps=int(args.steps),
                eval_every=int(args.eval_every),
                batch_size=int(args.batch_size),
                window=int(args.window),
                hidden=int(args.hidden),
                learning_rate=float(args.learning_rate),
                seed=int(args.seed),
                iou=float(args.iou),
                device=device,
                memory_guard=memory_guard,
                projector_dim=int(args.projector_dim),
                decoder=str(args.decoder),
                dense_span_rank=int(args.dense_span_rank),
                dense_span_duration_hidden=int(args.dense_span_duration_hidden),
                dense_span_hinge_weight=float(args.dense_span_hinge_weight),
                type_head=str(args.type_head),
                type_weight=float(args.type_weight),
                type_class_weighting=str(args.type_class_weighting),
                aux_frame_type_weight=float(args.aux_frame_type_weight),
                modality=str(args.input_modality),
                rare_window_fraction=float(args.rare_window_fraction),
            )
        )

    memory_guard.stop()
    summary = {
        "schema": SCHEMA,
        "frames_dir": str(frames_dir),
        "config": vars(args),
        "memory_caps": {
            "vram_ratio": vram_ratio,
            "host_memory_ratio": memory_guard.ratio,
            "host_metric": memory_guard.metric,
            "host_budget_gib": round(memory_guard.budget_bytes / 2**30, 3),
            "host_peak_gib": round(memory_guard.peak_bytes / 2**30, 3),
        },
        "results": results,
    }
    with (output / "falsification_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)

    print("\n=== selected checkpoints, real held-out ===")
    header = f"{'arm':<16}{'set':<14}{'acc':>8}{'speechF1':>10}{'nonspF1':>10}{'eventF1':>10}"
    print(header)
    for result in results:
        for name in ("val_real", "test_real"):
            block = result["selected"].get(name)
            if not block:
                continue
            print(
                f"{result['arm']:<16}{name:<14}"
                f"{block['frame_accuracy']:>8.4f}"
                f"{block['speech_frame']['f1']:>10.4f}"
                f"{block['non_speech_frame']['f1']:>10.4f}"
                f"{block['speech_event']['f1']:>10.4f}"
            )
    print(f"\nwrote {output / 'falsification_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
