#!/usr/bin/env python3
"""Pool the per-film acceptance reports into one verdict.

One film cannot separate "the head got better" from "this film happened to suit
it". The per-film reports already hold everything needed; this only pools them,
and it pools the two headline numbers differently on purpose:

  * **AUC and the matched-false-drop recall are recomputed on the pooled cues**,
    not averaged. Averaging AUCs weights a 400-cue film the same as a 2000-cue
    one, and the quantity being estimated is per-cue.
  * **Per-film numbers stay in the table** anyway, because a pooled win that
    comes entirely from one film is a different claim from one that holds on
    each - and only the table can tell those apart.

A v2 head appears twice, as `<head>` and `<head>~frame`. The first is the blank
share every head has always been scored on; the second is `1 - speech_share`
from its frame classifier - the same question asked of the class that exists to
answer it. They are rows rather than columns because everything here is computed
from one score per cue, and the two readings are exactly that with a different
score. The `auc_blank` and `score=1` slots are therefore about whichever score
the row names, not about blank specifically.

The operating point is re-derived per head at a fixed false-drop budget rather
than fixed at one threshold. Heads put their probability mass at different
absolute levels, so a shared threshold compares calibration; what the pipeline
would actually do is pick a threshold for the head it has.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from tools.align.compare_heads_on_film import auc, percentiles  # noqa: E402

SCHEMA = "alignment_head_acceptance_pooled_v1"


def operating_point(positive: list[float], negative: list[float], budget: float):
    """Highest recall whose false-drop stays inside `budget`.

    Walks the threshold down, because lowering it trades false drops for recall;
    the last threshold still inside the budget is the one to report.
    """
    if not positive or not negative:
        return None
    best = None
    for step in range(1000, 599, -1):
        cut = step / 1000.0
        false_drop = sum(1 for value in negative if value >= cut) / len(negative)
        if false_drop > budget:
            break
        recall = sum(1 for value in positive if value >= cut) / len(positive)
        best = (cut, recall, false_drop)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True, help="label=path")
    parser.add_argument(
        "--false-drop-budget",
        type=float,
        default=0.053,
        help="the shipped head's measured operating point on sample-v",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    films: dict[str, dict] = {}
    for spec in args.report:
        label, _, path = spec.partition("=")
        films[label] = json.loads(
            (PROJECT_ROOT / path).read_text(encoding="utf-8")
        )

    # A v2 head is pooled twice: once on `blank_share`, as every head always
    # was, and once on `1 - speech_share` from its frame classifier. They are
    # entered as separate rows rather than separate columns because the whole
    # comparison - AUC, the matched-false-drop operating point, the per-film
    # table - is about one score per cue, and the two readings are exactly that
    # with a different score. `~frame` is the suffix; a v1 head has no such row.
    FRAME_SUFFIX = "~frame"

    def cue_score(cue: dict, head: str):
        if head.endswith(FRAME_SUFFIX):
            base = head[: -len(FRAME_SUFFIX)]
            speech = cue.get(f"{base}:speech")
            return None if speech is None else 1.0 - float(speech)
        return cue.get(head)

    heads = sorted(
        {head for report in films.values() for head in report["heads"]},
        key=lambda name: (name != "shipped", name),
    )
    framed = sorted(
        {
            f"{head}{FRAME_SUFFIX}"
            for report in films.values()
            for head, entry in report["heads"].items()
            if entry.get("frame_head_available")
        }
    )
    heads = heads + framed
    print(f"films: {list(films)}")
    print(f"heads: {heads}\n")

    pooled: dict[str, dict[str, list[float]]] = {
        head: {"positive": [], "negative": [], "kana": [], "isolated": []}
        for head in heads
    }
    per_film: dict[str, dict[str, dict]] = {}
    for film, report in films.items():
        per_film[film] = {}
        for head in heads:
            base = head.split(FRAME_SUFFIX)[0]
            if base not in report["heads"]:
                continue

            def scores(group: str, head=head) -> list[float]:
                values = (
                    cue_score(cue, head)
                    for cue in report["cues"]
                    if cue["group"] == group
                )
                return [value for value in values if value is not None]

            positive = scores("vocalisation_dropped")
            negative = scores("dialogue_lexical")
            if not positive or not negative:
                continue
            pooled[head]["positive"].extend(positive)
            pooled[head]["negative"].extend(negative)
            pooled[head]["kana"].extend(scores("kana_only_kept"))
            point = operating_point(positive, negative, args.false_drop_budget)
            per_film[film][head] = {
                "cues_vocalisation": len(positive),
                "cues_dialogue": len(negative),
                "auc_blank": round(auc(positive, negative), 4),
                "dialogue_median": percentiles(negative).get("median"),
                "vocalisation_median": percentiles(positive).get("median"),
                "dialogue_at_1000": sum(1 for value in negative if value >= 0.99999),
                "operating_point": (
                    {
                        "threshold": round(point[0], 3),
                        "recall": round(point[1], 4),
                        "false_drop": round(point[2], 4),
                    }
                    if point
                    else None
                ),
            }

    print(f"{'film':>14} {'head':>22} {'cues':>10} {'AUC':>7} {'recall@budget':>14} {'fd':>7} {'score=1':>8}")
    for film in films:
        for head in heads:
            entry = per_film[film].get(head)
            if not entry:
                continue
            point = entry["operating_point"] or {}
            print(
                f"{film:>14} {head:>22} "
                f"{entry['cues_vocalisation']:>4}/{entry['cues_dialogue']:<5} "
                f"{entry['auc_blank']:>7.4f} "
                f"{point.get('recall', float('nan')):>14.1%} "
                f"{point.get('false_drop', float('nan')):>7.1%} "
                f"{entry['dialogue_at_1000']:>8}"
            )
        print()

    # A reading that produced no cues anywhere is dropped rather than printed as
    # a row of nulls - which is what a v1 head's frame row would be.
    heads = [head for head in heads if pooled[head]["positive"]]

    summary = {}
    for head in heads:
        positive = pooled[head]["positive"]
        negative = pooled[head]["negative"]
        point = operating_point(positive, negative, args.false_drop_budget)
        summary[head] = {
            "cues_vocalisation": len(positive),
            "cues_dialogue": len(negative),
            "auc_blank": round(auc(positive, negative), 4),
            "dialogue_median": percentiles(negative).get("median"),
            "vocalisation_median": percentiles(positive).get("median"),
            "kana_only_median": percentiles(pooled[head]["kana"]).get("median"),
            "dialogue_at_1000": sum(1 for value in negative if value >= 0.99999),
            "operating_point": (
                {
                    "threshold": round(point[0], 3),
                    "recall": round(point[1], 4),
                    "false_drop": round(point[2], 4),
                }
                if point
                else None
            ),
            "auc_by_film": {
                film: per_film[film][head]["auc_blank"]
                for film in films
                if head in per_film[film]
            },
        }

    print("=== pooled over every film ===")
    print(
        f"{'head':>22} {'AUC':>8} {'recall@budget':>14} {'fd':>7} "
        f"{'dialogue med':>13} {'voc med':>9} {'score=1':>8}  per-film AUC"
    )
    for head in heads:
        entry = summary[head]
        point = entry["operating_point"] or {}
        spread = " ".join(f"{value:.3f}" for value in entry["auc_by_film"].values())
        print(
            f"{head:>22} {entry['auc_blank']:>8.4f} "
            f"{point.get('recall', float('nan')):>14.1%} "
            f"{point.get('false_drop', float('nan')):>7.1%} "
            f"{entry['dialogue_median']:>13.4f} {entry['vocalisation_median']:>9.4f} "
            f"{entry['dialogue_at_1000']:>8}  {spread}"
        )

    # Does the win hold on every film, or is it one film carrying the pool?
    if "shipped" in heads:
        print("\n=== per-film AUC delta against the shipped head ===")
        for head in heads:
            if head == "shipped":
                continue
            deltas = [
                per_film[film][head]["auc_blank"]
                - per_film[film]["shipped"]["auc_blank"]
                for film in films
                if head in per_film[film] and "shipped" in per_film[film]
            ]
            if not deltas:
                continue
            wins = sum(1 for value in deltas if value > 0)
            print(
                f"{head:>22}  wins {wins}/{len(deltas)} films  "
                f"median {statistics.median(deltas):+.4f}  "
                f"range {min(deltas):+.4f}..{max(deltas):+.4f}"
            )
            summary[head]["auc_delta_wins"] = f"{wins}/{len(deltas)}"

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "false_drop_budget": args.false_drop_budget,
                "pooled": summary,
                "by_film": per_film,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
