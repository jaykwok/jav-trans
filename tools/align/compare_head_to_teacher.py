#!/usr/bin/env python3
"""How far apart are the production head and Grok on the SAME real JAV seconds.

Every real-domain verdict so far has been by ear (blind audits) or inconclusive
(the 08-09 A/B, p=0.189). Grok's per-word times on an archived film give a
numeric answer instead - not against truth, since neither side is truth, but the
agreement of two independent teachers bounds both.

Words cannot be matched one to one: the two sides transcribe different text. So
each side is merged into speech islands and only MUTUALLY UNIQUE overlaps are
compared - one head island overlapping one Grok island and vice versa. A Grok
island swallowed by a longer head island is reported separately rather than
contributing a boundary offset, because its start would then be measuring
segmentation, not timing.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unicodedata

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALIGNED_KIND = "ctc_forced_alignment"
MERGE_GAP_S = 0.15


def acoustic(text: str) -> str:
    return "".join(c for c in str(text or "") if unicodedata.category(c)[0] in {"L", "N"})


def merge(intervals, gap_s=MERGE_GAP_S):
    out: list[list[float]] = []
    for a, b in sorted((float(x), float(y)) for x, y in intervals):
        if b <= a:
            continue
        if out and a - out[-1][1] <= gap_s:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def teacher_islands(path: Path, film_id: str):
    spans = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("film_id")) != film_id or not acoustic(row.get("text")):
                continue
            spans.append((float(row["start_s"]), float(row["end_s"])))
    return merge(spans), len(spans)


def head_islands(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload if isinstance(payload, list) else payload.get("segments") or []
    spans = []
    for segment in segments:
        for word in segment.get("words") or []:
            if str(word.get("timestamp_kind") or "") != ALIGNED_KIND:
                continue
            if str(word.get("alignment_quality") or "") != "aligned":
                continue
            if not acoustic(word.get("word")):
                continue
            spans.append((float(word["start"]), float(word["end"])))
    return merge(spans), len(spans)


def describe(values, label):
    if not values:
        return {"label": label, "count": 0}
    array = np.asarray(values, dtype=np.float64) * 1000.0
    absolute = np.abs(array)
    return {
        "label": label,
        "count": len(values),
        "median_ms": round(float(np.median(array)), 1),
        "p05_ms": round(float(np.percentile(array, 5)), 1),
        "p25_ms": round(float(np.percentile(array, 25)), 1),
        "p75_ms": round(float(np.percentile(array, 75)), 1),
        "p95_ms": round(float(np.percentile(array, 95)), 1),
        "abs_median_ms": round(float(np.median(absolute)), 1),
        "abs_p90_ms": round(float(np.percentile(absolute, 90)), 1),
        "share_within_100ms": round(float((absolute <= 100).mean()), 4),
        "share_within_200ms": round(float((absolute <= 200).mean()), 4),
    }


def main() -> None:
    film_id = sys.argv[1]
    aligned_path = Path(sys.argv[2])
    # The archive covers the two films timed on 2026-08-10; films timed later
    # live in their own runner output, so the path is an argument rather than a
    # constant. Same format either way - the runner assembles both.
    words_path = (
        Path(sys.argv[3])
        if len(sys.argv) > 3
        else PROJECT_ROOT
        / "datasets/train/jav-grok-stt-frame-teacher-v1/teacher/grok.words.jsonl"
    )

    grok, grok_words = teacher_islands(words_path, film_id)
    head, head_words = head_islands(aligned_path)

    # Overlap graph, then keep only mutually unique pairs.
    grok_hits: dict[int, list[int]] = {i: [] for i in range(len(grok))}
    head_hits: dict[int, list[int]] = {i: [] for i in range(len(head))}
    j = 0
    for i, (a, b) in enumerate(grok):
        while j < len(head) and head[j][1] <= a:
            j += 1
        k = j
        while k < len(head) and head[k][0] < b:
            grok_hits[i].append(k)
            head_hits[k].append(i)
            k += 1

    starts: list[float] = []
    ends: list[float] = []
    unique_pairs = 0
    grok_unmatched = 0
    grok_ambiguous = 0
    for i, hits in grok_hits.items():
        if not hits:
            grok_unmatched += 1
            continue
        if len(hits) != 1 or len(head_hits[hits[0]]) != 1:
            grok_ambiguous += 1
            continue
        unique_pairs += 1
        starts.append(head[hits[0]][0] - grok[i][0])
        ends.append(head[hits[0]][1] - grok[i][1])

    grok_total = sum(b - a for a, b in grok)
    covered = 0.0
    for i, hits in grok_hits.items():
        a, b = grok[i]
        for k in hits:
            covered += max(0.0, min(b, head[k][1]) - max(a, head[k][0]))

    print(json.dumps({
        "film_id": film_id,
        "aligned_segments": str(aligned_path),
        "grok": {"words": grok_words, "islands": len(grok), "speech_s": round(grok_total, 1)},
        "head": {"words": head_words, "islands": len(head)},
        "grok_speech_also_called_speech_by_head": round(covered / max(grok_total, 1e-9), 4),
        "islands": {
            "mutually_unique_pairs": unique_pairs,
            "grok_islands_head_heard_nothing": grok_unmatched,
            "grok_islands_with_ambiguous_pairing": grok_ambiguous,
        },
        "start_offset": describe(starts, "head_start - grok_start (negative = head earlier)"),
        "end_offset": describe(ends, "head_end - grok_end (positive = head later)"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
