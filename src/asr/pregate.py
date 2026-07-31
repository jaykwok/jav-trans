"""Which stretches of audio are worth decoding, read off the alignment head.

This is the gate reading of the same tensor `asr.alignment` uses for timing. A
run of frames the head labels blank is a run with no character evidence in it,
so it does not need the decoder - and the decoder is the entire cost of ASR
(encoder RTF 0.00069 against 0.12273 for decode, a 178x asymmetry measured in
Phase 0). Running the encoder over everything and the decoder over a third of it
is what makes a pre-gate affordable without a separate acoustic model.

**The decisive property of this gate is that its errors are not symmetric.**
Audio the gate drops is never transcribed, so a word lost here is lost for good;
audio it keeps that turns out to be wordless costs some decode time and is
filtered downstream by the post-gate, which is reversible. Every default below
is therefore biased toward keeping: blank has to be sustained before it counts
as a pause, short pauses inside an utterance are bridged rather than cut on, and
each surviving region is padded outward. The project has already paid for the
opposite mistake once - the retired pre-ASR chain dropped audio on acoustic
evidence alone, and 55 real sources went with it.

Nothing here is learned and nothing here is tuned against its own output. The
parameters are geometric, and the number that decides them is measured against
human-audited spans by `tools/align/evaluate_pregate_loss.py`.
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


def cut_at_pauses(
    blank_spans: list[tuple[float, float]],
    total_s: float,
    *,
    target_s: float = 20.0,
    max_s: float = 30.0,
    min_s: float = 2.0,
) -> list[tuple[float, float]]:
    """Contiguous chunks covering ALL of the audio, cut inside pauses.

    This is the second reading of the blank runs and the one that survived. The
    gate above uses them to decide what *not* to decode, and a 2026-07-31
    measurement falsified that on this domain: the head separates lexically
    dense speech from vocalisation-dense audio rather than speech from silence,
    so skipping the blank stretches lost real lines embedded in moaning.

    Choosing *where to cut* has no such failure mode, because nothing is
    dropped. The output tiles `[0, total_s]` exactly - adjacent chunks share an
    edge and the last one ends at `total_s` - so a mistake here can only put a
    boundary in a slightly worse place, never lose a word. That makes it safe to
    use the very signal that was too weak to gate with.

    Cuts land at the middle of a pause rather than its edge, which is where a
    boundary does least damage to the words on either side.
    """
    total_s = max(0.0, float(total_s))
    if total_s <= 0.0:
        return []
    if max_s <= 0.0 or target_s <= 0.0:
        raise ValueError("target_s and max_s must be > 0")
    if target_s > max_s:
        raise ValueError("target_s must be <= max_s")
    if total_s <= max_s:
        return [(0.0, total_s)]

    candidates = [
        (begin + end) / 2.0
        for begin, end in _clamp_spans(blank_spans, total_s)
        if 0.0 < (begin + end) / 2.0 < total_s
    ]

    chunks: list[tuple[float, float]] = []
    cursor = 0.0
    while total_s - cursor > max_s:
        window = [
            point
            for point in candidates
            if cursor + min_s <= point <= cursor + max_s
        ]
        # Nearest to the target length, so chunks stay evenly sized instead of
        # collapsing to the first pause after `min_s`.
        cut = (
            min(window, key=lambda point: abs(point - (cursor + target_s)))
            if window
            else cursor + max_s
        )
        chunks.append((cursor, cut))
        cursor = cut
    chunks.append((cursor, total_s))

    if len(chunks) > 1 and chunks[-1][1] - chunks[-1][0] < min_s:
        # A sliver at the end is not worth its own decode call, and merging it
        # backwards keeps the tiling exact.
        tail = chunks.pop()
        chunks[-1] = (chunks[-1][0], tail[1])
    return chunks


def gate_audio(
    log_probs,
    total_s: float,
    *,
    upsample: int = 2,
    config: PreGateConfig | None = None,
) -> list[tuple[float, float]]:
    """Regions worth decoding, straight from the head's per-frame posteriors."""
    from asr.alignment import blank_runs

    config = config or PreGateConfig()
    runs = blank_runs(log_probs, upsample=upsample, min_seconds=config.min_blank_s)
    return speech_regions(runs, total_s, config)


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
