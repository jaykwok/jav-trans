#!/usr/bin/env python3
"""Do punctuation classes fragment the pauses the chunker reads?

`blank_runs` (`src/asr/alignment.py`) takes the free-running argmax and calls a
run of frames a pause only while the argmax stays on blank. A head whose
vocabulary contains `。`, `、` and `…` can spend a frame on one of those in the
middle of a silence, and that single frame ends the run. `cut_at_pauses` then
sees two short pauses where there was one long one, and below
`ASR_CHUNK_MIN_PAUSE_S` (0.6s by default) it sees no usable cut at all.

That is the concrete mechanism behind `is_acoustic_char`'s warning that
punctuation "has no sound to align it to; the best it can learn is 'punctuation
follows silence', which is exactly the confusion that makes a blank run stop
being a clean pause". It has been argued from first principles; this measures it.

Two readings of the same argmax, per head:

  strict   runs where the argmax is blank - what `blank_runs` actually returns
  lenient  runs where the argmax is blank OR a punctuation class - what the runs
           would be if punctuation frames were treated as the silence they sit in

For an acoustic-only head the two are identical by construction, and that is the
control: it says the difference measured on a punctuated head is the punctuation
and not the measurement.

The number that decides anything operational is `usable_pauses` - runs at least
`--min-pause-s` long, because those are the only ones the chunker can cut at.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.align.compare_heads_on_film import encode_film  # noqa: E402

# v2 replaced the reported quantity, not just its name: v1 compared the two sides'
# *counts* of usable pauses, which conflates merging (two already-cuttable runs
# joining, costing nothing) with fragmentation (one silence split below the floor).
# v1 reports `usable_pauses_lost_to_punctuation`, v2 `pauses_invisible_to_chunker`.
# The bump exists so a v1 file cannot be read as a v2 one - they were briefly
# indistinguishable, and a stale report was quoted alongside three fresh ones.
SCHEMA = "alignment_head_pause_structure_v2"


def runs_of(mask) -> list[tuple[int, int]]:
    """Half-open frame intervals of every maximal True run."""
    spans: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(list(mask) + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append((start, index))
            start = None
    return spans


def describe(spans: list[tuple[int, int]], frame_s: float, minimum: float) -> dict:
    lengths = [(end - start) * frame_s for start, end in spans]
    usable = [value for value in lengths if value >= minimum]
    return {
        "runs": len(lengths),
        "total_s": round(sum(lengths), 2),
        "median_s": round(statistics.median(lengths), 4) if lengths else 0.0,
        "usable_pauses": len(usable),
        "usable_total_s": round(sum(usable), 2),
    }


def pauses_lost_to_fragmentation(
    strict: list[tuple[int, int]],
    lenient: list[tuple[int, int]],
    frame_s: float,
    minimum: float,
) -> tuple[int, float]:
    """Silences the chunker cannot see at all because punctuation split them.

    Comparing the two *counts* would say nothing: merging two already-usable
    runs into one lowers the count while losing nothing, and that effect can
    outweigh the one being looked for. The question is per-silence - take each
    lenient run long enough to cut at, and ask whether any of the strict pieces
    inside it is still long enough. If none is, that silence is invisible.
    """
    lost = 0
    lost_seconds = 0.0
    index = 0
    for begin, finish in lenient:
        if (finish - begin) * frame_s < minimum:
            continue
        while index < len(strict) and strict[index][0] < begin:
            index += 1
        longest = 0
        probe = index
        while probe < len(strict) and strict[probe][1] <= finish:
            longest = max(longest, strict[probe][1] - strict[probe][0])
            probe += 1
        if longest * frame_s < minimum:
            lost += 1
            lost_seconds += (finish - begin) * frame_s
    return lost, round(lost_seconds, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--head", action="append", required=True, help="label=path")
    parser.add_argument(
        "--min-pause-s",
        type=float,
        default=0.6,
        help="the chunker's floor; runs below it are not cut points",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import numpy as np
    import torch

    from core.config import load_config

    load_config()

    from asr.alignment import (
        BLANK_INDEX,
        UNK_INDEX,
        AlignmentHead,
        frame_to_seconds,
        is_acoustic_char,
        overlap_save_slices,
    )

    heads = []
    for spec in args.head:
        label, _, path = spec.partition("=")
        heads.append((label, AlignmentHead.load(str(PROJECT_ROOT / path))))
        print(f"loaded {label}: {path}")

    context = {head.context_frames for _, head in heads}
    if len(context) != 1:
        raise SystemExit("heads disagree on context frames")
    hidden_states, returned_frames, bases, duration_s = encode_film(
        args.audio, context_frames=next(iter(context))
    )

    report = {
        "schema": SCHEMA,
        "audio": args.audio,
        "audio_duration_s": round(duration_s, 3),
        "min_pause_s": args.min_pause_s,
        "heads": {},
    }
    for label, head in heads:
        pieces = [
            head.log_probs(np.asarray(state, dtype=np.float32))
            for state in hidden_states
        ]
        keep = overlap_save_slices(
            list(zip(bases[: len(pieces)], returned_frames)),
            context_frames=head.context_frames,
        )
        kept = [
            piece[begin * head.upsample : finish * head.upsample]
            for piece, (begin, finish) in zip(pieces, keep)
            if finish > begin
        ]
        log_probs = torch.cat(kept, dim=0) if len(kept) > 1 else kept[0]
        predicted = log_probs.argmax(dim=-1).detach().cpu().numpy()
        frame_s = frame_to_seconds(1, upsample=head.upsample)

        # Which classes are punctuation. An acoustic-only vocab has none, which
        # is what makes it the control rather than just another arm.
        punctuation_indices = {
            index
            for index in range(2, head.vocab.size)
            if not is_acoustic_char(head.vocab.char_at(index))
        }
        is_blank = predicted == BLANK_INDEX
        is_punctuation = np.isin(predicted, list(punctuation_indices)) if punctuation_indices else np.zeros_like(is_blank)
        is_unknown = predicted == UNK_INDEX

        strict = runs_of(is_blank)
        lenient = runs_of(is_blank | is_punctuation)
        lost, lost_seconds = pauses_lost_to_fragmentation(
            strict, lenient, frame_s, args.min_pause_s
        )
        entry = {
            "vocab_size": head.vocab.size,
            "punctuation_classes": len(punctuation_indices),
            "frames": int(len(predicted)),
            "blank_frame_share": round(float(is_blank.mean()), 6),
            "punctuation_frame_share": round(float(is_punctuation.mean()), 6),
            "unknown_frame_share": round(float(is_unknown.mean()), 6),
            # Of the frames that are NOT blank, how many are punctuation? This is
            # the share of the head's positive evidence that stands for no sound.
            "punctuation_share_of_nonblank": round(
                float(is_punctuation.sum() / max(1, (~is_blank).sum())), 6
            ),
            "strict": describe(strict, frame_s, args.min_pause_s),
            "lenient": describe(lenient, frame_s, args.min_pause_s),
            "pauses_invisible_to_chunker": lost,
            "pause_seconds_invisible": lost_seconds,
        }
        report["heads"][label] = entry
        print(
            f"  {label}: punctuation {entry['punctuation_frame_share']:.4%} of frames, "
            f"{entry['punctuation_share_of_nonblank']:.2%} of non-blank; "
            f"{lost} silences fragmented below the {args.min_pause_s}s floor "
            f"({lost_seconds}s)"
        )

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"\n{'head':>16} {'punct/frame':>12} {'punct/nonblank':>15} "
        f"{'cuttable':>9} {'invisible':>10} {'seconds':>9}"
    )
    for label, entry in report["heads"].items():
        print(
            f"{label:>16} {entry['punctuation_frame_share']:>12.4%} "
            f"{entry['punctuation_share_of_nonblank']:>15.2%} "
            f"{entry['strict']['usable_pauses']:>9} "
            f"{entry['pauses_invisible_to_chunker']:>10} "
            f"{entry['pause_seconds_invisible']:>9}"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
