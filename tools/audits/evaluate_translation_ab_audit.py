#!/usr/bin/env python3
"""Reveal a completed blinded translation A/B and summarize the preferences.

The page never knew which arm was which; this joins the saved verdicts back to
`answers.jsonl` and reports who won. Two guards matter more than the counts:

* the sign test is over **decisive** cards only (one side preferred). "Both
  usable" is not half a win for each arm - it is the auditor saying the
  difference does not matter, and folding it into a win rate is how a tie gets
  reported as a result;
* unreviewed cards are counted and named. A win rate over the half of the sample
  that got looked at is not a win rate over the sample.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "translation_ab_audit_result_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_two_sided_sign_p(left: int, right: int) -> float | None:
    total = int(left) + int(right)
    if total <= 0:
        return None
    tail = min(int(left), int(right))
    probability = sum(math.comb(total, k) for k in range(tail + 1)) / (2**total)
    return min(1.0, 2.0 * probability)


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> tuple[float, float] | None:
    """Wilson 95% interval: usable at the sample sizes a human audit produces."""
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    spread = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return round(max(0.0, centre - spread), 4), round(min(1.0, centre + spread), 4)


def build(answers: list[dict[str, Any]], verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    answer_ids = [str(row["row_id"]) for row in answers]
    if len(set(answer_ids)) != len(answer_ids):
        raise ValueError("duplicate row_id in answers")
    verdict_ids = [str(row.get("row_id") or "") for row in verdicts]
    if len(set(verdict_ids)) != len(verdict_ids):
        raise ValueError("duplicate row_id in verdicts")
    unknown = sorted(set(verdict_ids) - set(answer_ids))
    if unknown:
        raise ValueError(f"verdict for unknown row: {unknown[0]}")

    by_id = {str(row["row_id"]): row for row in answers}
    verdict_by_id = {str(row["row_id"]): row for row in verdicts}
    arm_names = sorted({str(row["arm_1"]) for row in answers} | {str(row["arm_2"]) for row in answers})
    if len(arm_names) != 2:
        raise ValueError(f"expected exactly two arms, found {arm_names}")

    resolved: list[dict[str, Any]] = []
    for row_id in answer_ids:
        answer = by_id[row_id]
        verdict = verdict_by_id.get(row_id, {"verdict": "unreviewed"})
        value = str(verdict.get("verdict") or "unreviewed")
        if value == "arm_1_better":
            winner = str(answer["arm_1"])
        elif value == "arm_2_better":
            winner = str(answer["arm_2"])
        else:
            winner = value
        resolved.append(
            {
                "row_id": row_id,
                "cue_index": answer.get("cue_index"),
                "start_s": answer.get("start_s"),
                "ja": answer.get("ja"),
                "arm_1": answer["arm_1"],
                "arm_2": answer["arm_2"],
                "verdict": value,
                "winner": winner,
                "note": str(verdict.get("note") or ""),
            }
        )

    counts = Counter(row["winner"] for row in resolved)
    first, second = arm_names
    decisive = counts[first] + counts[second]
    reviewed = sum(1 for row in resolved if row["verdict"] != "unreviewed")
    return {
        "schema": RESULT_SCHEMA,
        "arms": arm_names,
        "cards": len(resolved),
        "reviewed": reviewed,
        "unreviewed": len(resolved) - reviewed,
        "wins": {first: counts[first], second: counts[second]},
        "equivalent_good": counts["equivalent_good"],
        "equivalent_bad": counts["equivalent_bad"],
        "unsure": counts["unsure"],
        "decisive": decisive,
        # Keyed by arm rather than by "candidate": neither arm is privileged
        # here, and a share whose owner has to be inferred gets misread.
        "win_share_decisive": {
            arm: (round(counts[arm] / decisive, 4) if decisive else None)
            for arm in arm_names
        },
        "win_share_ci95": {
            arm: wilson_interval(counts[arm], decisive) for arm in arm_names
        },
        "sign_test_p": exact_two_sided_sign_p(counts[first], counts[second]),
        "leading_arm_counts": dict(Counter(str(row["arm_1"]) for row in answers)),
        "rows": resolved,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--verdicts", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    result = build(_rows(Path(args.answers)), _rows(Path(args.verdicts)))
    printable = {key: value for key, value in result.items() if key != "rows"}
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        printable["written_to"] = str(output)
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
