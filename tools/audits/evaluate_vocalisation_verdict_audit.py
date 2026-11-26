#!/usr/bin/env python3
"""Read the listening verdicts against the answer key, and check W6.

The two reference strata are read FIRST and can void the run. The `has words`
control certainly contains words - kanji-bearing cues for the joint verdict, the
surviving half of the same cue for a split; if the ear did not hear words there,
then "no words" on the strata under test means nothing, and the honest outcome is
"this audit did not measure anything" rather than a negative result.

`unsure` is never folded into either answer. It was offered as the exit for an
unjudgeable clip precisely so that a forced guess would not become evidence, and
averaging it in afterwards would undo that.

The stratum names are arguments, not constants: the same arithmetic reads the
joint-verdict page and the split-fragment page, and a second copy of it would be
a second place for the `unsure` rule to be got wrong.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA = "vocalisation_verdict_audit_result_v1"

# The strata the joint verdict newly deletes. A word heard in one of these is a
# cue the change would remove from the screen. Defaults; see `--under-test`.
UNDER_TEST = ("new_isolated", "new_kana")
# Read first; they say whether the page measured anything at all.
CONTROL_WORDS = "dialogue_control"
CONTROL_NO_WORDS = "already_dropped"


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts", required=True)
    parser.add_argument("--answer-key", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--control-floor",
        type=float,
        default=0.80,
        help="share of the `has words` control that must come back as "
        "`has_words` for the run to count as valid",
    )
    parser.add_argument(
        "--under-test",
        default=",".join(UNDER_TEST),
        help="comma-separated strata whose `has_words` count is the result",
    )
    parser.add_argument("--control-words", default=CONTROL_WORDS)
    parser.add_argument("--control-no-words", default=CONTROL_NO_WORDS)
    args = parser.parse_args()

    under_test = tuple(
        name.strip() for name in args.under_test.split(",") if name.strip()
    )
    control_words = args.control_words
    control_no_words = args.control_no_words

    key = {row["row_id"]: row for row in read(PROJECT_ROOT / args.answer_key)}
    verdicts = read(PROJECT_ROOT / args.verdicts)
    unknown = [v["row_id"] for v in verdicts if v["row_id"] not in key]
    if unknown:
        raise SystemExit(f"{len(unknown)} verdicts have no key row; first={unknown[0]}")

    tally: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    words_heard: dict[str, list[dict]] = collections.defaultdict(list)
    for verdict in verdicts:
        row = key[verdict["row_id"]]
        stratum = row["stratum"]
        answer = str(verdict.get("verdict") or "unreviewed")
        tally[stratum][answer] += 1
        if answer == "has_words":
            words_heard[stratum].append(
                {
                    "row_id": row["row_id"],
                    "text": row.get("text"),
                    # Absent on a split page, where the key carries the fragment
                    # rather than a per-cue posterior. Reported as None instead
                    # of raising: which clips a word was heard in is the point,
                    # and it does not depend on these.
                    "speech": row.get("speech"),
                    "vocalisation": row.get("vocalisation"),
                    "speech_run_s": row.get("speech_run_s"),
                    "start_s": row.get("start_s"),
                    "end_s": row.get("end_s"),
                    "note": verdict.get("note", ""),
                }
            )

    def share(stratum: str, answer: str) -> float:
        counts = tally[stratum]
        judged = counts["has_words"] + counts["no_words"]
        return counts[answer] / judged if judged else 0.0

    print(f"{'stratum':>18} {'A has words':>12} {'B no words':>11} {'unsure':>7} "
          f"{'A share of judged':>18}")
    for stratum in (control_words, control_no_words, *under_test):
        counts = tally[stratum]
        print(
            f"{stratum:>18} {counts['has_words']:>12} {counts['no_words']:>11} "
            f"{counts['unsure']:>7} {share(stratum, 'has_words'):>17.1%}"
        )

    control_share = share(control_words, "has_words")
    valid = control_share >= args.control_floor
    print(
        f"\ncontrol: {control_share:.1%} of {control_words!r} heard as words "
        f"(floor {args.control_floor:.0%}) -> "
        f"{'the page measured something' if valid else 'VOID'}"
    )

    tested = collections.Counter()
    for stratum in under_test:
        tested.update(tally[stratum])
    judged = tested["has_words"] + tested["no_words"]
    rate = f"{tested['has_words'] / judged:.1%}" if judged else "n/a"
    print(
        f"\nW6: words heard in audio the filter would remove: "
        f"{tested['has_words']}/{judged} judged ({rate}), "
        f"{tested['unsure']} unsure"
    )
    print("     target = 0")
    for stratum in under_test:
        for hit in words_heard[stratum]:
            print(
                f"       {stratum}  {hit['row_id']}  "
                f"text={str(hit['text'])[:28]!r}  p_sp={hit['speech']}  "
                f"run={hit['speech_run_s']}  note={hit['note'][:40]!r}"
            )
    # A truncated word shows up in what SURVIVED a split, not in what was
    # removed, so a `no_words` here is a finding rather than a control failure.
    control_misses = tally[control_words]["no_words"]
    if control_misses:
        print(
            f"\n     {control_misses} clip(s) in {control_words!r} came back as "
            f"`no words`. On a split page that is where a cut placed too early "
            f"would show, so these are worth reading individually."
        )

    report = {
        "schema": SCHEMA,
        "verdicts": args.verdicts,
        "answer_key": args.answer_key,
        "control_floor": args.control_floor,
        "under_test": list(under_test),
        "control_words": control_words,
        "control_no_words": control_no_words,
        "control_has_words_share": round(control_share, 4),
        "control_words_heard_as_no_words": tally[control_words]["no_words"],
        "valid": bool(valid),
        "by_stratum": {
            stratum: dict(counts) for stratum, counts in sorted(tally.items())
        },
        "w6_words_in_newly_dropped": tested["has_words"],
        "w6_judged": judged,
        "w6_unsure": tested["unsure"],
        "words_heard": {k: v for k, v in words_heard.items()},
    }
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
