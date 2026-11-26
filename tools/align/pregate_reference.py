"""The pre-gate that was rejected, kept so it can go on being measured.

This is the *first* reading of the alignment head's blank runs: treat a run of
blank as audio with no words in it and skip the decoder for it. The appeal was
real - decode is 178x the cost of the encoder forward, so gating would have cut
most of the ASR bill.

A 2026-07-31 measurement on real audio falsified it. The head separates
lexically dense speech from vocalisation-dense audio, not speech from silence,
so the stretches it calls blank routinely contain real lines embedded in
moaning. Dropping them is unrecoverable: audio the gate skips is never
transcribed, and a word lost here is lost for good. Production therefore uses
the same signal only to choose cut points (`asr.chunking.cut_at_pauses`), where
a mistake moves a boundary instead of deleting a line.

It lives under tools/ rather than src/ because nothing in the runtime may import
it, and it is **deliberately self-contained**: `_clamp_spans` and `_merge` are
duplicated from `asr.chunking` rather than imported. A baseline that follows
production refactors is not a baseline - the numbers
`tools/align/evaluate_pregate_loss.py` reports would silently change meaning.

Consumers: `evaluate_pregate_loss.py` (loss against human-audited spans) and
`measure_pregate_dropped_audio.py` (how much real audio this would have thrown
away).
"""
from __future__ import annotations

from dataclasses import dataclass

PREGATE_SCHEMA = "blank_run_pregate_v1"


@dataclass(frozen=True)
class PreGateConfig:
    """Geometry of the keep/skip decision, in seconds.

    Defaults come from the 2026-07-31 real-audio pilot, where the first attempt
    used `min_blank_s=0.35` and cut *inside* utterances at ordinary inter-word
    pauses - it produced ~1 s fragments and only 33 of 102 regions could be
    aligned at all. The lesson is in the name: `min_blank_s` is not "how much
    silence is there", it is "how long must silence last before it is a
    boundary", and Japanese speech is full of shorter ones.
    """

    # Blank must last this long to count as a pause at all.
    min_blank_s: float = 0.6
    # A kept region shorter than this is not a line, it is a fragment.
    min_speech_s: float = 1.0
    # Two kept regions closer than this are bridged rather than left split, so
    # an utterance is not chopped by a pause that only just cleared the bar.
    merge_gap_s: float = 0.4
    # Every kept region grows by this much on each side. Directly buys back the
    # onset and coda that CTC peakiness places outside the character spans (see
    # `alignment.speech_extent`), and costs only decode time.
    pad_s: float = 0.15
    # A region longer than this is split, because a single decode call over a
    # very long span is where runaway repetition appears.
    max_region_s: float = 20.0

    def __post_init__(self) -> None:
        for name in ("min_blank_s", "min_speech_s", "merge_gap_s", "pad_s"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if self.max_region_s <= 0.0:
            raise ValueError("max_region_s must be > 0")
        if self.max_region_s < self.min_speech_s:
            # Otherwise splitting would immediately produce pieces the
            # min-length rule has already declared too short to be lines.
            raise ValueError("max_region_s must be >= min_speech_s")


def _clamp_spans(
    spans: list[tuple[float, float]], total_s: float
) -> list[tuple[float, float]]:
    clamped: list[tuple[float, float]] = []
    for begin, end in spans:
        begin = max(0.0, min(float(total_s), float(begin)))
        end = max(begin, min(float(total_s), float(end)))
        if end > begin:
            clamped.append((begin, end))
    return sorted(clamped)


def _merge(spans: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for begin, end in spans:
        if merged and begin - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    return [(begin, end) for begin, end in merged]


def speech_regions(
    blank_spans: list[tuple[float, float]],
    total_s: float,
    config: PreGateConfig | None = None,
) -> list[tuple[float, float]]:
    """Complement the pauses, then bridge, pad, drop fragments and split.

    Order matters and is not arbitrary:

      * bridge before dropping fragments, or a real utterance split by one
        marginal pause would have both halves discarded as too short;
      * drop fragments before padding, or padding would rescue exactly the
        fragments the length rule just rejected;
      * split last, so `max_region_s` is enforced on what will actually be sent
        to the decoder rather than on an intermediate shape.
    """
    config = config or PreGateConfig()
    total_s = max(0.0, float(total_s))
    if total_s <= 0.0:
        return []

    regions: list[tuple[float, float]] = []
    cursor = 0.0
    for begin, end in _clamp_spans(blank_spans, total_s):
        if begin > cursor:
            regions.append((cursor, begin))
        cursor = max(cursor, end)
    if cursor < total_s:
        regions.append((cursor, total_s))

    regions = _merge(regions, config.merge_gap_s)
    regions = [(b, e) for b, e in regions if e - b >= config.min_speech_s]
    if config.pad_s > 0.0:
        regions = [
            (max(0.0, b - config.pad_s), min(total_s, e + config.pad_s))
            for b, e in regions
        ]
        # Padding can make two regions touch or cross. Merging them is the only
        # answer that keeps the output non-overlapping without discarding audio.
        regions = _merge(regions, 0.0)

    split: list[tuple[float, float]] = []
    for begin, end in regions:
        span = end - begin
        if span <= config.max_region_s:
            split.append((begin, end))
            continue
        pieces = int(span // config.max_region_s) + (
            1 if span % config.max_region_s > 1e-9 else 0
        )
        width = span / pieces
        split.extend(
            (begin + index * width, begin + (index + 1) * width)
            for index in range(pieces)
        )
    return split


def covered_seconds(
    spans: list[tuple[float, float]], regions: list[tuple[float, float]]
) -> float:
    """Seconds of `spans` that fall inside `regions`.

    Both sides are treated as sets of intervals rather than as aligned lists,
    because a labelled span routinely straddles a region edge and counting it
    as wholly kept or wholly lost would misreport the very quantity the gate is
    being judged on.
    """
    if not spans or not regions:
        return 0.0
    ordered = _merge(sorted((float(b), float(e)) for b, e in regions), 0.0)
    total = 0.0
    for begin, end in spans:
        begin, end = float(begin), float(end)
        for region_begin, region_end in ordered:
            if region_end <= begin:
                continue
            if region_begin >= end:
                break
            total += min(end, region_end) - max(begin, region_begin)
    return total


def duration(spans: list[tuple[float, float]]) -> float:
    return sum(max(0.0, float(end) - float(begin)) for begin, end in spans)
