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

**"Speech" here means something was said, not that a character was emitted.**
The head's words come from forced alignment, so they include the kana the local
ASR writes for moaning, which Grok drops on purpose. Counting that as head
speech reads agreement as conflict, and it does so in proportion to how much
vocalisation a film has - so the gate was hardest on exactly the films whose
vocalisation is the thing worth learning. Islands of pure non-semantic
vocalisation are therefore excluded from the head side before the comparison;
`--count-vocalisation-as-speech` restores the v2 reading for audit.

**The two teachers' text is never compared.** Two transcribers word the same
seconds differently, so a cross-teacher text match would manufacture
disagreement out of vocabulary. Only time is compared. Text is read on the head
side alone, and only to answer "is the head claiming something was said here".

**Local ASR is not ground truth.** It hallucinates in this domain, which is why
the post-gate exists. Overlap therefore means the two teachers disagree, not that
speech is definitely present.

**The verdict is one-directional and scoped.** There is no "admitted" state: a
clean result is `no_conflict_observed`, which is the absence of evidence against
the film over the seconds that were actually compared. `scope` says whether that
was the whole film or a prefix, and every reported quantity is computed over the
same window - a prefix run reporting whole-film teacher coverage next to
prefix-window disagreement invites reading one as context for the other when they
do not share a denominator. `inconclusive` is separate from `reject`: a reference
run with no speech in it, or a window that proposes no blank at all, has measured
nothing rather than measured agreement.

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

from subtitles.vocalisation import is_non_semantic_vocalisation  # noqa: E402

SCHEMA = "asr_teacher_silence_admission_v3"
ALIGNED_KIND = "ctc_forced_alignment"

REJECT = "reject"
NO_CONFLICT_OBSERVED = "no_conflict_observed"
INCONCLUSIVE = "inconclusive"

# Exit codes keep the one-directional meaning usable from a shell: only a
# rejection is a hard stop, and "nothing was measured" must not look like a pass.
EXIT_CODES = {NO_CONFLICT_OBSERVED: 0, REJECT: 2, INCONCLUSIVE: 3}


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
    semantic_only: bool = True,
    island_gap_s: float = 0.15,
) -> list[tuple[float, float]]:
    """The head's own speech, by default only where something was *said*.

    **Why `semantic_only` is the default.** The head's words come from forced
    alignment, so they contain every character the local ASR emitted - including
    the kana it writes for moaning. Grok discards non-semantic vocalisation on
    purpose. Counting it as head speech makes the two teachers look like they
    disagree in exactly the places where they agree about what was said, and it
    does so hardest on the films with the most vocalisation - which are the
    films this supervision is wanted from. Measured on one real film, the
    swallowed population is 51.7% core kana against 30.3% in the kept
    population, and 7.0% kanji against 17.5%.

    **The unit is an island, not a character.** Words here are single
    characters, so `あたし` would lose its `あ` to a per-character test while
    keeping the rest - eroding real speech to no purpose. Characters are grouped
    into islands at `island_gap_s`, and an island is excluded only when the
    whole of its text is non-semantic vocalisation, which is the same question
    `subtitles.vocalisation` answers for cues and the same answer.

    **Text is never compared between the two teachers.** Two transcribers will
    word the same seconds differently, so any cross-teacher text match would
    manufacture disagreement out of vocabulary. Only time is compared; the text
    is read on one side only, to decide whether the head is claiming that
    something was said there at all.
    """
    segments = (
        aligned_segments
        if isinstance(aligned_segments, list)
        else aligned_segments.get("segments") or []
    )
    words: list[tuple[float, float, str]] = []
    for segment in segments:
        for word in segment.get("words") or []:
            if str(word.get("timestamp_kind") or "") != ALIGNED_KIND:
                continue
            if confident_only and str(word.get("alignment_quality") or "") != "aligned":
                continue
            text = acoustic(word.get("word") or "")
            if not text:
                continue
            start = float(word["start"])
            end = float(word["end"])
            if end <= start:
                continue
            if window_s is not None:
                if start >= window_s:
                    continue
                end = min(end, window_s)
            words.append((start, end, text))

    words.sort()
    if not semantic_only:
        return merge([(start, end) for start, end, _ in words], gap_s=0.0)

    islands: list[list] = []
    for start, end, text in words:
        if islands and start - islands[-1][1] <= island_gap_s:
            islands[-1][1] = max(islands[-1][1], end)
            islands[-1][2] += text
            islands[-1][3].append((start, end))
        else:
            islands.append([start, end, text, [(start, end)]])

    spans: list[tuple[float, float]] = []
    for _, _, text, members in islands:
        if is_non_semantic_vocalisation(text):
            continue
        spans.extend(members)
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
    compare_s = min(window_s, duration_s) if window_s is not None else duration_s
    scope = "prefix" if compare_s < duration_s - 1e-9 else "full_film"

    teacher = merge(teacher_words, gap_s=merge_gap_s)
    teacher_full_s = sum(b - a for a, b in teacher)
    # Every reported quantity is computed over the compared seconds. Reporting
    # whole-film teacher coverage beside prefix-window disagreement puts two
    # different denominators in one table and invites reading them together.
    window = [(0.0, compare_s)]
    teacher_speech_s = overlap_s(teacher, window)
    head_speech_s = overlap_s(head_spans, window)

    runs = silent_runs(
        teacher,
        total_s=duration_s,
        margin_s=boundary_ignore_s,
        minimum_s=minimum_blank_s,
    )
    runs_in_window = [
        (a, b)
        for a, b in ((a, min(b, compare_s)) for a, b in runs if a < compare_s)
        if b > a
    ]
    proposed_s = sum(b - a for a, b in runs_in_window)
    swallowed_s = overlap_s(head_spans, runs_in_window)
    swallowed_share = swallowed_s / head_speech_s if head_speech_s > 0 else None

    if head_speech_s <= 0.0:
        verdict = INCONCLUSIVE
        reason = "head_found_no_speech_to_check_against"
    elif proposed_s <= 0.0:
        verdict = INCONCLUSIVE
        reason = "no_blank_proposed_in_window"
    elif swallowed_share is not None and swallowed_share > max_swallowed_share:
        verdict = REJECT
        reason = "teacher_silence_swallows_head_speech"
    else:
        verdict = NO_CONFLICT_OBSERVED
        reason = ""

    return {
        "verdict": verdict,
        # Combined so a stored result cannot be quoted without its scope. A
        # prefix result is never a statement about the whole film.
        "verdict_id": f"{scope}_{verdict}",
        "verdict_reason": reason,
        "scope": scope,
        "duration_s": round(duration_s, 3),
        "comparison_window_s": round(compare_s, 3),
        "teacher_speech_s_in_window": round(teacher_speech_s, 3),
        "teacher_speech_share_in_window": round(
            teacher_speech_s / max(compare_s, 1e-9), 6
        ),
        "teacher_speech_s_full_film": round(teacher_full_s, 3),
        "teacher_speech_share_full_film": round(
            teacher_full_s / max(duration_s, 1e-9), 6
        ),
        "head_speech_s_in_window": round(head_speech_s, 3),
        "head_speech_share_in_window": round(
            head_speech_s / max(compare_s, 1e-9), 6
        ),
        "minimum_blank_s": minimum_blank_s,
        "proposed_blank_runs": len(runs_in_window),
        "proposed_blank_s_in_window": round(proposed_s, 3),
        "proposed_blank_share_of_window": round(
            proposed_s / max(compare_s, 1e-9), 6
        ),
        "disputed_s_in_window": round(swallowed_s, 3),
        # Reported because it is the number people reach for first, and it is
        # the misleading one: it shrinks as the rule gets more conservative
        # precisely because the denominator grows.
        "disputed_share_of_blank": round(swallowed_s / max(proposed_s, 1e-9), 6),
        "head_speech_swallowed_share": (
            None if swallowed_share is None else round(swallowed_share, 6)
        ),
        "max_swallowed_share": max_swallowed_share,
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
    parser.add_argument(
        "--count-vocalisation-as-speech",
        action="store_true",
        help="the v2 reading: count every aligned character as head speech, "
        "including the kana written for moaning that Grok drops on purpose",
    )
    parser.add_argument("--island-gap-s", type=float, default=0.15)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    aligned = json.loads(
        resolve_repo_path(args.aligned_segments).read_text(encoding="utf-8")
    )
    window_s = args.window_s if args.window_s > 0 else None
    teacher_words = _load_teacher_words(
        resolve_repo_path(args.teacher_words), args.film_id
    )
    semantic_only = not args.count_vocalisation_as_speech

    def evaluate(*, semantic: bool) -> dict:
        return evaluate_film(
            teacher_words=teacher_words,
            head_spans=head_speech_spans(
                aligned,
                window_s=window_s,
                confident_only=not args.include_unaligned,
                semantic_only=semantic,
                island_gap_s=float(args.island_gap_s),
            ),
            duration_s=float(args.duration_s),
            window_s=window_s,
            merge_gap_s=float(args.merge_gap_s),
            boundary_ignore_s=float(args.boundary_ignore_s),
            minimum_blank_s=float(args.minimum_blank_s),
            max_swallowed_share=float(args.max_swallowed_share),
        )

    result = evaluate(semantic=semantic_only)
    # The v2 reading is reported alongside whenever it is not the one deciding,
    # because a stored verdict that changed when the definition of head speech
    # changed must carry both numbers or nobody can tell which rule produced it.
    counted_vocalisation = evaluate(semantic=False) if semantic_only else None
    payload = {
        "schema": SCHEMA,
        "film_id": args.film_id,
        "head_speech_counts_vocalisation": not semantic_only,
        **result,
        "if_vocalisation_counted_as_speech": (
            None
            if counted_vocalisation is None
            else {
                "verdict_id": counted_vocalisation["verdict_id"],
                "head_speech_s_in_window": counted_vocalisation[
                    "head_speech_s_in_window"
                ],
                "head_speech_swallowed_share": counted_vocalisation[
                    "head_speech_swallowed_share"
                ],
            }
        ),
    }
    if args.output:
        out = resolve_repo_path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(EXIT_CODES[result["verdict"]])


if __name__ == "__main__":
    main()
