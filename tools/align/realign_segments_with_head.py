#!/usr/bin/env python3
"""Re-align a finished film's segments with a different head, text held fixed.

Swapping the alignment head changes subtitle timing, which is the user-visible
half of what it does - a head that reads blank better but times words worse is a
bad trade, and the blank measurement cannot see that at all. This produces an
`aligned_segments.json` per head so `compare_head_to_teacher.py` can put each one
against Grok's word times on the same film.

**The transcript is frozen.** Segments, their acoustic windows and their text all
come from the finished production run, so the only thing that differs between the
outputs is the head. Re-running the pipeline would also re-run the ASR and mix
transcription differences into a boundary measurement.

**This is not the production alignment path**, and the difference is worth
stating: the pipeline aligns per chunk and then walks segment edges outward under
`ONSET_BACKOFF_MAX_S` / `CODA_EXTEND_MAX_S`, which this does not reproduce. The
island comparison is near-blind to that anyway - 2026-08-10 measured 286 segments
producing 9,249 head islands, so almost every island boundary is a word inside a
segment rather than a walked edge - and both heads are treated identically here,
so the comparison between them stands even where the absolute number would shift.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.align.compare_heads_on_film import encode_film  # noqa: E402

ALIGNED_KIND = "ctc_forced_alignment"


def film_log_probs(head, hidden_states, returned_frames, bases):
    import torch

    import numpy as np

    from asr.alignment import overlap_save_slices

    pieces = [
        head.log_probs(np.asarray(state, dtype=np.float32)) for state in hidden_states
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
    return torch.cat(kept, dim=0) if len(kept) > 1 else kept[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-segments", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--head", action="append", required=True, help="label=path")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from core.config import load_config

    load_config()

    from asr.alignment import AlignmentHead, align_text, frame_to_seconds

    heads = []
    for spec in args.head:
        label, _, path = spec.partition("=")
        heads.append((label, AlignmentHead.load(str(PROJECT_ROOT / path))))
        print(f"loaded {label}: {path}")

    payload = json.loads(Path(args.aligned_segments).read_text(encoding="utf-8"))
    segments = payload.get("segments") or []
    print(f"segments: {len(segments)}")

    context = {head.context_frames for _, head in heads}
    if len(context) != 1:
        raise SystemExit("heads disagree on context frames; one encoder pass cannot serve both")
    hidden_states, returned_frames, bases, duration_s = encode_film(
        args.audio, context_frames=next(iter(context))
    )

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, head in heads:
        started = time.perf_counter()
        log_probs = film_log_probs(head, hidden_states, returned_frames, bases)
        frame_s = frame_to_seconds(1, upsample=head.upsample)
        total_frames = int(log_probs.shape[0])
        rebuilt = []
        failed = 0
        for segment in segments:
            start = float(segment.get("acoustic_start") or segment.get("start") or 0.0)
            end = float(segment.get("acoustic_end") or segment.get("end") or 0.0)
            text = str(segment.get("text") or "")
            first = max(0, int(start / frame_s))
            last = min(total_frames, max(first + 1, int(round(end / frame_s))))
            window = log_probs[first:last]
            spans = None
            if text and window.shape[0] > 0:
                try:
                    spans = align_text(
                        window,
                        text,
                        head.vocab,
                        upsample=head.upsample,
                        blank_bias=head.blank_bias,
                    )
                except (ValueError, RuntimeError):
                    spans = None
            if spans is None:
                failed += 1
                rebuilt.append({**segment, "words": []})
                continue
            words = [
                {
                    "start": start + float(span.start_s),
                    "end": start + float(span.end_s),
                    "word": span.char,
                    "timestamp_kind": ALIGNED_KIND,
                    "alignment_mode": ALIGNED_KIND,
                    "alignment_quality": "aligned",
                }
                for span in spans
            ]
            rebuilt.append({**segment, "words": words})
        out = out_dir / f"{label}.aligned_segments.json"
        out.write_text(
            json.dumps(
                {**payload, "segments": rebuilt, "realigned_head": label},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        aligned_words = sum(len(segment["words"]) for segment in rebuilt)
        print(
            f"  {label}: {aligned_words} words, {failed} segments unalignable, "
            f"{time.perf_counter() - started:.1f}s -> {out}"
        )


if __name__ == "__main__":
    main()
