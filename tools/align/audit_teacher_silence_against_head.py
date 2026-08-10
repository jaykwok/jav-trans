#!/usr/bin/env python3
"""Decide whether a film may be used as a source of real-domain blank frames.

**The rule this gates.** Real-domain negatives are proposed as "a long enough
stretch with no Grok word is blank". That rule is only safe when Grok's silence
means nothing was said. Grok's diarization only attaches a speaker to words it
transcribed, and its VAD can skip quiet or breathy speech, so on some films its
silence means "Grok did not hear it" instead.

**The check costs nothing.** Any film considered for supervision has to be run
through the production head anyway to get encoder features, and that run already
emits `<film>.aligned_segments.json` with per-word forced-alignment times. This
compares the two readings of the same seconds.

**Which number decides.** Not "what share of the proposed blank is disputed" -
any long-run rule drives that toward zero by making the denominator the whole
film. The decisive quantity is the reverse: **what share of the speech the
production head itself found would be relabelled blank**. On a film where Grok
mostly failed, that number is close to 1 while the disputed share still looks
like a rounding error.

**Local ASR is not ground truth.** It hallucinates in this domain, which is why
the post-gate exists. Overlap therefore means the two teachers disagree, not that
speech is definitely present. The gate is deliberately one-directional: it can
reject a film, never certify one.

**Timeline correctness is a precondition.** Both readings must be on the source
PTS timeline. An alignment produced before the 2026-08-10 audio-PTS fix runs
progressively early against the source and will manufacture disagreement; pass
only a run whose audio was extracted with the timeline filter in place.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

SCHEMA = "asr_teacher_silence_admission_v1"
ALIGNED_KIND = "ctc_forced_alignment"


def resolve_repo_path(path_text: str) -> Path:
    path = Path(str(path_text).replace("\\", "/"))
    return path if path.is_absolute() else PROJECT_ROOT / path


def acoustic(text: str) -> str:
    """Letters and numbers only, matching the head's own vocabulary rule.

    Punctuation is dropped on both sides: CTC gives `。` a frame, but a frame of
    punctuation is not evidence that anything was spoken there.
    """
    return "".join(
        ch for ch in str(text or "") if unicodedata.category(ch)[0] in {"L", "N"}
    )


def merge(intervals, *, gap_s: float) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted((float(a), float(b)) for a, b in intervals):
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= gap_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def silent_runs(
    speech: list[tuple[float, float]],
    *,
    total_s: float,
    margin_s: float,
    minimum_s: float,
) -> list[tuple[float, float]]:
    padded = merge(
        [(max(0.0, a - margin_s), min(total_s, b + margin_s)) for a, b in speech],
        gap_s=0.0,
    )
    runs: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in padded:
        if start - cursor >= minimum_s:
            runs.append((cursor, start))
        cursor = max(cursor, end)
    if total_s - cursor >= minimum_s:
        runs.append((cursor, total_s))
    return runs


def overlap_s(spans, windows) -> float:
    total = 0.0
    index = 0
    for start, end in spans:
        while index < len(windows) and windows[index][1] <= start:
            index += 1
        cursor = index
        while cursor < len(windows) and windows[cursor][0] < end:
            total += max(
                0.0,
                min(end, windows[cursor][1]) - max(start, windows[cursor][0]),
            )
            cursor += 1
    return total


def head_speech_spans(
    aligned_segments: dict | list,
    *,
    window_s: float | None = None,
    confident_only: bool = True,
) -> list[tuple[float, float]]:
    segments = (
        aligned_segments
        if isinstance(aligned_segments, list)
        else aligned_segments.get("segments") or []
    )
    spans: list[tuple[float, float]] = []
    for segment in segments:
        for word in segment.get("words") or []:
            if str(word.get("timestamp_kind") or "") != ALIGNED_KIND:
                continue
            if confident_only and str(word.get("alignment_quality") or "") != "aligned":
                continue
            if not acoustic(word.get("word") or ""):
                continue
            start = float(word["start"])
            end = float(word["end"])
            if end <= start:
                continue
            if window_s is not None:
                if start >= window_s:
                    continue
                end = min(end, window_s)
            spans.append((start, end))
    return merge(spans, gap_s=0.0)


def evaluate_film(
    *,
    teacher_words: list[tuple[float, float]],
    head_spans: list[tuple[float, float]],
    duration_s: float,
    window_s: float | None,
    merge_gap_s: float,
    boundary_ignore_s: float,
    minimum_blank_s: float,
    max_swallowed_share: float,
) -> dict:
    compare_s = window_s or duration_s
    teacher = merge(teacher_words, gap_s=merge_gap_s)
    teacher_speech_s = sum(b - a for a, b in teacher)
    head_speech_s = sum(b - a for a, b in head_spans)

    runs = silent_runs(
        teacher,
        total_s=duration_s,
        margin_s=boundary_ignore_s,
        minimum_s=minimum_blank_s,
    )
    if window_s is not None:
        runs = [(a, min(b, window_s)) for a, b in runs if a < window_s]
        runs = [(a, b) for a, b in runs if b > a]
    proposed_s = sum(b - a for a, b in runs)
    swallowed_s = overlap_s(head_spans, runs)
    swallowed_share = swallowed_s / head_speech_s if head_speech_s > 0 else None

    if swallowed_share is None:
        admitted = False
        reason = "head_found_no_speech_to_check_against"
    elif swallowed_share > max_swallowed_share:
        admitted = False
        reason = "teacher_silence_swallows_head_speech"
    else:
        admitted = True
        reason = ""

    return {
        "duration_s": round(duration_s, 3),
        "comparison_window_s": round(compare_s, 3),
        "teacher_speech_s": round(teacher_speech_s, 3),
        "teacher_speech_share": round(teacher_speech_s / max(duration_s, 1e-9), 6),
        "head_speech_s": round(head_speech_s, 3),
        "head_speech_share": round(head_speech_s / max(compare_s, 1e-9), 6),
        "minimum_blank_s": minimum_blank_s,
        "proposed_blank_runs": len(runs),
        "proposed_blank_s": round(proposed_s, 3),
        "proposed_blank_share_of_window": round(
            proposed_s / max(compare_s, 1e-9), 6
        ),
        "disputed_s": round(swallowed_s, 3),
        # Reported because it is the number people reach for first, and it is
        # the misleading one: it shrinks as the rule gets more conservative
        # precisely because the denominator grows.
        "disputed_share_of_blank": round(swallowed_s / max(proposed_s, 1e-9), 6),
        "head_speech_swallowed_share": (
            None if swallowed_share is None else round(swallowed_share, 6)
        ),
        "max_swallowed_share": max_swallowed_share,
        "admitted_as_blank_source": admitted,
        "rejection_reason": reason,
    }


def _load_teacher_words(path: Path, film_id: str) -> list[tuple[float, float]]:
    words: list[tuple[float, float]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if film_id and str(row.get("film_id") or "") != film_id:
                continue
            if not acoustic(row.get("text") or ""):
                continue
            words.append((float(row["start_s"]), float(row["end_s"])))
    return words


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-words", required=True)
    parser.add_argument("--film-id", default="")
    parser.add_argument(
        "--aligned-segments",
        required=True,
        help="<film>.aligned_segments.json from a PTS-correct production run",
    )
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument(
        "--window-s",
        type=float,
        default=0.0,
        help="clip the comparison when the production run covered only a prefix",
    )
    parser.add_argument("--merge-gap-s", type=float, default=0.15)
    parser.add_argument("--boundary-ignore-s", type=float, default=0.10)
    parser.add_argument("--minimum-blank-s", type=float, default=0.8)
    parser.add_argument(
        "--max-swallowed-share",
        type=float,
        default=0.10,
        help="reject the film when the blank rule would relabel more than this "
        "share of the head's own high-confidence speech",
    )
    parser.add_argument("--include-unaligned", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    aligned = json.loads(
        resolve_repo_path(args.aligned_segments).read_text(encoding="utf-8")
    )
    window_s = args.window_s if args.window_s > 0 else None
    result = evaluate_film(
        teacher_words=_load_teacher_words(
            resolve_repo_path(args.teacher_words), args.film_id
        ),
        head_spans=head_speech_spans(
            aligned,
            window_s=window_s,
            confident_only=not args.include_unaligned,
        ),
        duration_s=float(args.duration_s),
        window_s=window_s,
        merge_gap_s=float(args.merge_gap_s),
        boundary_ignore_s=float(args.boundary_ignore_s),
        minimum_blank_s=float(args.minimum_blank_s),
        max_swallowed_share=float(args.max_swallowed_share),
    )
    payload = {"schema": SCHEMA, "film_id": args.film_id, **result}
    if args.output:
        out = resolve_repo_path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["admitted_as_blank_source"] else 2)


if __name__ == "__main__":
    main()
