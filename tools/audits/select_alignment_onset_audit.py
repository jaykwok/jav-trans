#!/usr/bin/env python3
"""Sample an onset-accuracy audit for the CTC alignment head, on real audio.

Alignment error is measured in milliseconds, which an ear cannot report. So the
question is turned into a detection experiment: every clip is cut at some offset
from the predicted line start and runs for a fixed duration, and the auditor is
asked only whether the first sound is intact or already chopped. Deliberate
offsets of known size then calibrate what the ear can detect, and the predicted
onset is scored against that scale.

  * `aligned`            - cut exactly at the predicted start. The measurement.
  * `control_early`      - cut 0.50 s earlier. Nothing can be clipped, so this
                           is the ceiling: it says what "intact" scores when it
                           is true, and validates the auditor and the question.
  * `probe_late_150ms`   - cut 0.15 s late. The plan's median gate.
  * `probe_late_400ms`   - cut 0.40 s late. The plan's p90 gate.

The probes are what make the result readable. If a 400 ms clip is reliably heard
as chopped while `aligned` scores like `control_early`, then the predicted onsets
are not late by 400 ms - and that is an inference from measured detectability
rather than from an assumed threshold.

**Every clip is exactly `--clip-seconds` long.** Cutting to the predicted end
instead would make duration vary with the stratum, and an auditor who notices
that no longer answers blind - the span-position audit was already weakened once
by a design where stratum leaked through clip length.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

SELECTION_SCHEMA = "alignment_onset_audit_selection_v1"
MANIFEST_SCHEMA = "alignment_onset_audit_manifest_v1"

# Offset applied to the predicted line start, per stratum, in seconds.
STRATUM_OFFSETS: dict[str, float] = {
    "aligned": 0.0,
    "control_early": -0.50,
    "probe_late_150ms": 0.15,
    "probe_late_400ms": 0.40,
}
# Run-up carried at the head of every clip. Owned here because this file decides
# the cut geometry; the page imports it so the "play from the cut" button enters
# the clip at exactly the offset the manifest was built with.
CONTEXT_SECONDS = 2.0

DEFAULT_STRATUM_SIZES: dict[str, int] = {
    "aligned": 30,
    "control_early": 30,
    "probe_late_150ms": 25,
    "probe_late_400ms": 25,
}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    # Run-up carried at the head of every clip so the page can offer a "play
    # with context" button. It is identical for every stratum, so it adds no
    # signal about which arm a clip belongs to - it only lets the ear hear what
    # led into the cut, which is what makes "is this mid-sound" answerable
    # rather than inferred.
    parser.add_argument("--context-seconds", type=float, default=CONTEXT_SECONDS)
    parser.add_argument("--min-line-duration-s", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260731)
    for name, size in DEFAULT_STRATUM_SIZES.items():
        parser.add_argument(f"--{name.replace('_', '-')}-n", type=int, default=size)
    args = parser.parse_args()

    lines = _read_jsonl(Path(args.lines))
    rejected: Counter[str] = Counter()
    pool: list[dict] = []
    for line in lines:
        start = float(line["line_start_s"])
        window = float(line["window_duration_s"])
        # Room for the earliest cut MINUS its run-up, and for the full clip, or
        # the strata would not be comparable on this line. Without the run-up
        # term a line near the window start would silently get less context than
        # the rest, and less context reads as a harder judgement.
        earliest = start + min(STRATUM_OFFSETS.values()) - args.context_seconds
        latest = start + max(STRATUM_OFFSETS.values()) + args.clip_seconds
        if earliest < 0.0 or latest > window:
            rejected["no_room_for_every_offset"] += 1
            continue
        if float(line["line_duration_s"]) < args.min_line_duration_s:
            # A line shorter than the largest probe offset would be entirely
            # skipped by that probe, making "chopped" trivially true.
            rejected["line_too_short"] += 1
            continue
        pool.append(line)

    if not pool:
        raise SystemExit("no eligible lines")

    rng = np.random.default_rng(args.seed)
    # One stratum per line, so the same onset is never heard twice by the same
    # auditor at two different offsets - that would make the deliberate ones
    # obvious by comparison.
    order = rng.permutation(len(pool))
    assignments: list[tuple[str, dict]] = []
    cursor = 0
    for stratum in STRATUM_OFFSETS:
        size = getattr(args, f"{stratum}_n")
        take = order[cursor : cursor + size]
        cursor += size
        for index in take:
            assignments.append((stratum, pool[int(index)]))
    if cursor > len(pool):
        raise SystemExit(
            f"requested {cursor} clips but only {len(pool)} eligible lines exist"
        )

    rng.shuffle(assignments)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    answers: list[dict] = []
    for index, (stratum, line) in enumerate(assignments):
        row_id = f"onset-{index:04d}"
        cut = float(line["line_start_s"]) + STRATUM_OFFSETS[stratum]
        # The clip begins BEFORE the cut. The page enters it at
        # `context_seconds` for the judgement and at 0 for the context replay,
        # so the cut point stays at a fixed, stratum-independent offset.
        manifest.append(
            {
                "schema": MANIFEST_SCHEMA,
                "row_id": row_id,
                "audio": line["audio"],
                "start_s": round(cut - args.context_seconds, 4),
                "end_s": round(cut + args.clip_seconds, 4),
            }
        )
        answers.append(
            {
                "row_id": row_id,
                "stratum": stratum,
                "offset_s": STRATUM_OFFSETS[stratum],
                "line_id": line["line_id"],
                "source_id": line["source_id"],
                "source_partition": line["source_partition"],
                "line_start_s": line["line_start_s"],
                "line_duration_s": line["line_duration_s"],
                "alignment_score": line.get("alignment_score"),
                "characters": line.get("characters"),
                "text": line.get("text"),
            }
        )

    # The manifest carries only what a clip needs. The stratum lives in a
    # separate file the page never reads, which is what keeps the audit blind.
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for row in answers:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema": SELECTION_SCHEMA,
        "lines_available": len(lines),
        "lines_eligible": len(pool),
        "rejected": dict(rejected),
        "clip_seconds": args.clip_seconds,
        "context_seconds": args.context_seconds,
        "stratum_offsets_s": STRATUM_OFFSETS,
        "stratum_counts": dict(Counter(stratum for stratum, _ in assignments)),
        "videos": len({line["source_id"] for _, line in assignments}),
        "total_clips": len(assignments),
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
