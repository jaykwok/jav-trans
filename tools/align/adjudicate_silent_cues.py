#!/usr/bin/env python3
"""Was that cue real speech, or the ASR writing words over a moan?

The acceptance criterion "no lexical dialogue cue may read 100% blank" assumes
the cue is speech because the ASR wrote kanji there. On this domain that
assumption is not safe: `気持ちいい` is close to the most likely thing a
Japanese ASR emits over any JAV audio, and a head that calls it blank would then
be right while the criterion called it a regression.

Grok settles it, and it is genuinely independent - a different model, a different
run, and it never saw the pipeline's transcript. If Grok also placed lexical
words inside the span, the cue is speech and a 100% blank reading is a real false
silence. If Grok placed nothing there, the cue is at best unconfirmed.

**Grok's silence is not proof of absence** - 2026-08-11 measured it returning
nothing on 55.51% of clips whose script carried clear words, concentrated exactly
on moaning-heavy audio. So this splits the failing cues into "confirmed speech"
and "unconfirmed", and only the confirmed half is charged against a head.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

TEACHER = PROJECT_ROOT / "datasets/train/jav-grok-stt-frame-teacher-v1"
SCHEMA = "silent_cue_adjudication_v1"


def lexical(text: str) -> bool:
    """Kanji, latin or digits: a witness that cannot be spelled by a moan."""
    for char in str(text or ""):
        if char.isdigit() or (char.isascii() and char.isalpha()):
            return True
        if "CJK UNIFIED" in unicodedata.name(char, ""):
            return True
    return False


DEFAULT_WORD_FILES = (
    "datasets/train/jav-grok-stt-frame-teacher-v1/teacher/grok.words.jsonl",
    "agents/temp/20260811_154632_grok-fullfilm-3films/grok.words.jsonl",
)


def grok_words(film_id: str, word_files: list[str]) -> list[tuple[float, float, str]]:
    """Word spans on the video timeline, from the runner's assembled output.

    Reading the assembled file rather than re-deriving offsets from the raw
    responses is deliberate: the runner extracts chunks preserving source PTS, so
    it already owns the one piece of arithmetic that can silently shift every
    word by the chunk padding. `compare_head_to_teacher.py` consumes the same
    file, which keeps the two measurements on one clock.
    """
    words: list[tuple[float, float, str]] = []
    for relative in word_files:
        path = PROJECT_ROOT / relative
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("film_id")) != film_id:
                    continue
                try:
                    start, end = float(row["start_s"]), float(row["end_s"])
                except (KeyError, TypeError, ValueError):
                    continue
                if end > start:
                    words.append((start, end, str(row.get("text") or "")))
    words.sort()
    return words


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance", required=True, help="compare_heads_on_film output")
    # Required rather than defaulted: this filters `grok.words.jsonl` rows by
    # `film_id`, so a default that names one film is either wrong for every
    # other film or a real id sitting in the repository. It used to be the
    # latter.
    parser.add_argument(
        "--film-id", required=True, help="film_id to select in the Grok word files"
    )
    parser.add_argument("--head", required=True, help="which head's failures to judge")
    parser.add_argument("--threshold", type=float, default=0.99999)
    parser.add_argument(
        "--words",
        action="append",
        default=None,
        help="repeatable grok.words.jsonl; defaults to both archived runs",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    report = json.loads((PROJECT_ROOT / args.acceptance).read_text(encoding="utf-8"))
    cues = report["cues"]
    words = grok_words(args.film_id, list(args.words or DEFAULT_WORD_FILES))
    print(f"grok words for {args.film_id}: {len(words)}")
    lexical_words = [word for word in words if lexical(word[2])]
    print(f"  lexical: {len(lexical_words)}")
    if not lexical_words:
        raise SystemExit(
            f"no Grok lexical words for {args.film_id}; nothing can be adjudicated "
            "and reporting zero confirmed failures would be a false pass"
        )

    blocks = json.loads(
        (PROJECT_ROOT / report["bilingual"]).read_text(encoding="utf-8")
    )["blocks"]

    def span(index: int) -> tuple[float, float] | None:
        block = blocks[index]
        start = block.get("acoustic_start")
        end = block.get("acoustic_end")
        if start is None or end is None:
            start, end = block.get("start"), block.get("end")
        if start is None or end is None:
            return None
        return float(start), float(end)

    failing = [
        cue
        for cue in cues
        if cue["group"] == "dialogue_lexical"
        and float(cue.get(args.head, 0.0)) >= args.threshold
    ]
    print(f"\n{args.head}: {len(failing)} lexical dialogue cues at blank >= {args.threshold}")

    confirmed: list[dict] = []
    unconfirmed: list[dict] = []
    for cue in failing:
        bounds = span(cue["index"])
        if bounds is None:
            continue
        start, end = bounds
        overlapping = [
            word
            for word in lexical_words
            if word[1] > start and word[0] < end
        ]
        record = {
            **cue,
            "span_s": [round(start, 3), round(end, 3)],
            "grok_lexical_words": "".join(word[2] for word in overlapping),
            "grok_lexical_word_count": len(overlapping),
        }
        (confirmed if overlapping else unconfirmed).append(record)

    print(f"\n  confirmed by Grok (real false silence): {len(confirmed)}")
    for record in confirmed:
        print(f"    cue={record['text'][:34]}")
        print(f"    grok={record['grok_lexical_words'][:34]}")
    print(f"\n  unconfirmed (Grok heard no lexical word there): {len(unconfirmed)}")
    for record in unconfirmed[:12]:
        print(f"    {record['text'][:44]}")

    # The same question asked of every head, so the comparison is like for like.
    heads = [key for key in report["heads"]]
    summary_by_head = {}
    for head in heads:
        head_failing = [
            cue
            for cue in cues
            if cue["group"] == "dialogue_lexical"
            and float(cue.get(head, 0.0)) >= args.threshold
        ]
        head_confirmed = 0
        for cue in head_failing:
            bounds = span(cue["index"])
            if bounds is None:
                continue
            start, end = bounds
            if any(word[1] > start and word[0] < end for word in lexical_words):
                head_confirmed += 1
        summary_by_head[head] = {
            "at_threshold": len(head_failing),
            "confirmed_by_grok": head_confirmed,
            "unconfirmed": len(head_failing) - head_confirmed,
        }

    print("\n=== lexical dialogue cues read as fully blank, by head ===")
    print(f"{'head':>22}  {'at 1.000':>9}  {'grok-confirmed':>15}  {'unconfirmed':>12}")
    for head, counts in summary_by_head.items():
        print(
            f"{head:>22}  {counts['at_threshold']:>9}  "
            f"{counts['confirmed_by_grok']:>15}  {counts['unconfirmed']:>12}"
        )

    if args.out:
        out = PROJECT_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "film_id": args.film_id,
                    "threshold": args.threshold,
                    "grok_words": len(words),
                    "grok_lexical_words": len(lexical_words),
                    "by_head": summary_by_head,
                    "confirmed": confirmed,
                    "unconfirmed": unconfirmed,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
