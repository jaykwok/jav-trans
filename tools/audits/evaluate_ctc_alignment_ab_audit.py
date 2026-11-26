#!/usr/bin/env python3
"""Reveal a completed blinded CTC boundary A/B and summarize preferences."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "ctc_alignment_ab_audit_result_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_two_sided_sign_p(candidate: int, baseline: int) -> float | None:
    n = int(candidate) + int(baseline)
    if n <= 0:
        return None
    tail = min(int(candidate), int(baseline))
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2.0 * probability)


def build(answers: list[dict[str, Any]], verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    answer_ids = [str(row["row_id"]) for row in answers]
    if len(set(answer_ids)) != len(answer_ids):
        raise ValueError("duplicate row_id in answers")
    verdict_ids = [str(row.get("row_id") or "") for row in verdicts]
    if len(set(verdict_ids)) != len(verdict_ids):
        raise ValueError("duplicate row_id in verdicts")
    by_id = {str(row["row_id"]): row for row in answers}
    unknown = sorted(set(verdict_ids) - set(answer_ids))
    if unknown:
        raise ValueError(f"verdict for unknown row: {unknown[0]}")
    verdict_by_id = {str(row["row_id"]): row for row in verdicts}
    resolved: list[dict[str, Any]] = []
    for row_id in answer_ids:
        answer = by_id[row_id]
        verdict = verdict_by_id.get(row_id, {"row_id": row_id, "verdict": "unreviewed"})
        value = str(verdict.get("verdict") or "unreviewed")
        winner = value
        if value == "arm_1_better":
            winner = "candidate" if answer["arm_1"] == "model_b" else "baseline"
        elif value == "arm_2_better":
            winner = "candidate" if answer["arm_2"] == "model_b" else "baseline"
        resolved.append(
            {
                **answer,
                "verdict": value,
                "winner": winner,
                "note": str(verdict.get("note") or ""),
            }
        )

    def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(row["winner"] for row in rows)
        candidate, baseline = counts["candidate"], counts["baseline"]
        decisive = candidate + baseline
        return {
            "reviewed": len(rows),
            "candidate_better": candidate,
            "baseline_better": baseline,
            "equivalent_good": counts["equivalent_good"],
            "equivalent_bad": counts["equivalent_bad"],
            "unsure": counts["unsure"],
            "unreviewed": counts["unreviewed"],
            "candidate_win_share_decisive": round(candidate / decisive, 4) if decisive else None,
            "sign_test_p_two_sided": exact_two_sided_sign_p(candidate, baseline),
        }

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in resolved:
        groups.setdefault(f"{row['domain']}:{row['boundary']}", []).append(row)
    overall = report(resolved)
    p = overall["sign_test_p_two_sided"]
    if not resolved or overall["unreviewed"]:
        gate = "incomplete"
    elif overall["candidate_better"] > overall["baseline_better"] and p is not None and p <= 0.05:
        gate = "candidate_preferred"
    elif overall["baseline_better"] > overall["candidate_better"] and p is not None and p <= 0.05:
        gate = "baseline_preferred"
    else:
        gate = "inconclusive_or_equivalent"
    return {
        "schema": RESULT_SCHEMA,
        "rows": len(resolved),
        "overall": overall,
        "by_domain_and_boundary": {key: report(rows) for key, rows in sorted(groups.items())},
        "human_preference_gate": gate,
    }


def combine_rounds(
    answer_rounds: list[list[dict[str, Any]]],
    verdict_rounds: list[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(answer_rounds) != len(verdict_rounds):
        raise ValueError("answers/verdicts round count mismatch")
    combined_answers: list[dict[str, Any]] = []
    combined_verdicts: list[dict[str, Any]] = []
    for round_index, (answers, verdicts) in enumerate(
        zip(answer_rounds, verdict_rounds, strict=True), start=1
    ):
        prefix = f"round-{round_index}:"
        combined_answers.extend(
            {**row, "row_id": prefix + str(row["row_id"])} for row in answers
        )
        combined_verdicts.extend(
            {**row, "row_id": prefix + str(row["row_id"])} for row in verdicts
        )
    return combined_answers, combined_verdicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True, action="append")
    parser.add_argument("--verdicts", required=True, action="append")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--domain",
        action="append",
        choices=("galgame", "jav"),
        help="Only score the selected domain; repeat to include more than one.",
    )
    args = parser.parse_args()
    if len(args.answers) != len(args.verdicts):
        parser.error("repeat --answers and --verdicts the same number of times")
    answers, verdicts = combine_rounds(
        [_rows(Path(path)) for path in args.answers],
        [_rows(Path(path)) for path in args.verdicts],
    )
    if args.domain:
        domains = set(args.domain)
        answers = [row for row in answers if str(row.get("domain")) in domains]
        answer_ids = {str(row["row_id"]) for row in answers}
        verdicts = [row for row in verdicts if str(row.get("row_id")) in answer_ids]
    result = build(answers, verdicts)
    result["domains"] = sorted({str(row["domain"]) for row in answers})
    result["rounds"] = len(args.answers)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
