#!/usr/bin/env python3
"""Read the onset audit as a detection experiment, not as a raw error rate.

The audit cannot report milliseconds, so nothing here does. What it can report
is how often a cut is *heard* as chopped, at offsets of known size. That makes
the result a comparison rather than a measurement:

  * `control_early` is cut half a second before the predicted start, so nothing
    can be clipped. Its "chopped" rate is the floor - whatever an auditor calls
    chopped when it is not. If this is high the question failed and nothing
    below can be read.
  * `probe_late_400ms` and `probe_late_150ms` are cut late by known amounts.
    Their lift over the floor is the ear's sensitivity at those offsets, which
    is what makes the plan's 150 ms / 400 ms gates testable at all.
  * `aligned` is the real prediction. It is read by where it falls between the
    floor and the probes.

The inference to draw, stated so it cannot be overclaimed: if 400 ms is
detectable (probe separated from floor) and `aligned` is not separated from the
floor, then predicted onsets are not systematically late by 400 ms. That is a
bound, not a point estimate, and a probe that fails to separate means the audit
had no power at that offset rather than that the model is good.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.stats import (  # noqa: E402
    two_proportion_difference,
    wilson,
)
from tools.audits.select_alignment_onset_audit import STRATUM_OFFSETS  # noqa: E402

RESULT_SCHEMA = "alignment_onset_audit_result_v1"
# Only these two carry timing information. `non_semantic` says the opening sound
# was not a word, which is a fact about the clip rather than about the cut, so it
# cannot be scored as intact or clipped and is reported on its own.
DECISIVE = ("intact", "clipped")
NON_SEMANTIC = "non_semantic"
FLOOR_STRATUM = "control_early"
MEASURED_STRATUM = "aligned"
# Below this many decisive answers in either the floor or the measured stratum,
# the comparison is too weak to state a bound from. Chosen because at n=10 a
# two-proportion interval is wider than the effect being looked for.
MIN_DECISIVE_FOR_A_BOUND = 12


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def join(answers: list[dict], verdicts: list[dict]) -> list[dict]:
    by_id = {str(row["row_id"]): row for row in answers}
    joined: list[dict] = []
    for verdict in verdicts:
        row_id = str(verdict.get("row_id") or "")
        key = by_id.get(row_id)
        if key is None:
            raise ValueError(f"verdict for an unknown row_id: {row_id}")
        joined.append({**key, "verdict": str(verdict.get("verdict") or "unreviewed")})
    return joined


def stratum_report(rows: list[dict]) -> dict[str, Any]:
    decisive = [row for row in rows if row["verdict"] in DECISIVE]
    clipped = sum(1 for row in decisive if row["verdict"] == "clipped")
    non_semantic = sum(1 for row in rows if row["verdict"] == NON_SEMANTIC)
    answered = [
        row for row in rows if row["verdict"] in (*DECISIVE, NON_SEMANTIC, "unsure")
    ]
    rate = clipped / len(decisive) if decisive else None
    return {
        "reviewed": len(rows),
        "decisive": len(decisive),
        "non_semantic": non_semantic,
        # Share of answered clips whose opening sound was not a word. This should
        # be roughly equal across strata - it is a property of the line, not of
        # where the cut was made - so a large spread is a sign the offsets are
        # audible in a way the design did not intend.
        "non_semantic_share": round(non_semantic / len(answered), 4)
        if answered
        else None,
        "unsure": sum(1 for row in rows if row["verdict"] == "unsure"),
        "unreviewed": sum(1 for row in rows if row["verdict"] == "unreviewed"),
        "clipped": clipped,
        "clipped_rate": round(rate, 4) if rate is not None else None,
        "clipped_rate_ci95": wilson(clipped, len(decisive)) if decisive else None,
    }


def build(answers_path: Path, verdicts_path: Path) -> dict[str, Any]:
    rows = join(_rows(answers_path), _rows(verdicts_path))
    by_stratum: dict[str, list[dict]] = {}
    for row in rows:
        by_stratum.setdefault(str(row["stratum"]), []).append(row)

    strata = {
        name: {
            "offset_s": STRATUM_OFFSETS.get(name),
            **stratum_report(rows_in),
        }
        for name, rows_in in sorted(by_stratum.items())
    }

    floor = strata.get(FLOOR_STRATUM) or {}
    floor_rate = floor.get("clipped_rate")
    floor_n = floor.get("decisive") or 0

    comparisons: dict[str, Any] = {}
    for name, report in strata.items():
        if name == FLOOR_STRATUM or floor_rate is None:
            continue
        comparisons[name] = two_proportion_difference(
            report.get("clipped_rate"), report.get("decisive") or 0, floor_rate, floor_n
        )

    measured = comparisons.get(MEASURED_STRATUM) or {}
    probe_400 = comparisons.get("probe_late_400ms") or {}
    probe_150 = comparisons.get("probe_late_150ms") or {}

    # Cutting later should never be heard as chopped LESS often - unless the
    # offset is large enough to clear the first word entirely and land on the
    # next one's clean onset, which scores as intact. That is what happened at
    # +400 ms on this domain (clipped lines median 2.19 s vs intact 1.44 s), so
    # a non-monotonic ladder means the large probe measured skip-past rather
    # than detectability and must not be used to bound anything.
    ladder = [
        (report["offset_s"], report["clipped_rate"])
        for report in strata.values()
        if report.get("offset_s") is not None
        and report.get("clipped_rate") is not None
    ]
    ladder.sort()
    monotonic = all(
        earlier[1] <= later[1] + 1e-9 for earlier, later in zip(ladder, ladder[1:])
    )
    skip_past_suspected = not monotonic

    # The verdict is deliberately conservative: a bound is only claimed when the
    # probe that defines it actually separated, so a null result reads as "no
    # power here" rather than as "the model passed".
    aligned_n = (strata.get(MEASURED_STRATUM) or {}).get("decisive") or 0
    if floor_rate is None:
        verdict = "no floor stratum reviewed; nothing can be read"
    elif min(floor_n, aligned_n) < MIN_DECISIVE_FOR_A_BOUND:
        # The cost of the exclusive third option: if most openings were
        # non-semantic, the timing question was answered too few times to
        # compare. Saying so is the honest outcome; topping the sample up from
        # the unused eligible lines is the fix, not reading this one harder.
        verdict = (
            f"too few decisive answers to compare "
            f"(floor n={floor_n}, aligned n={aligned_n}, need "
            f"{MIN_DECISIVE_FOR_A_BOUND} each); most openings were likely "
            "non-semantic - draw a top-up batch rather than reading this as a result"
        )
    elif floor_rate > 0.30:
        verdict = (
            f"control_early was heard as chopped {floor_rate:.1%} of the time; "
            "the question or the listening pass failed, main result unusable"
        )
    elif measured.get("separated_from_reference") and (measured.get("difference") or 0) > 0:
        # Checked before the probe branches: this comparison only involves the
        # floor and the prediction, so skip-past contamination of a large probe
        # cannot affect it.
        verdict = (
            "aligned onsets are heard as chopped far more often than the floor; "
            "the head IS systematically late on real audio"
        )
    elif not probe_400.get("separated_from_reference"):
        verdict = (
            "a deliberate 400 ms late cut was NOT reliably heard; this audit has "
            "no power at the p90 gate and cannot bound the onset error"
        )
    elif measured.get("separated_from_reference"):
        verdict = (
            "aligned onsets are heard as chopped more often than the floor; "
            "the head IS systematically late on real audio"
        )
    else:
        bound = "400 ms"
        if probe_150.get("separated_from_reference"):
            bound = "150 ms"
        verdict = (
            f"400 ms is detectable and aligned onsets are indistinguishable from "
            f"the floor; predicted onsets are not systematically late by {bound}"
        )

    return {
        "schema": RESULT_SCHEMA,
        "total_rows": len(rows),
        "strata": strata,
        "comparisons_vs_floor": comparisons,
        "floor_stratum": FLOOR_STRATUM,
        "ladder_monotonic": monotonic,
        "skip_past_suspected": skip_past_suspected,
        "verdict": verdict,
    }


def render(result: dict[str, Any]) -> str:
    lines = ["", "对齐起点审计 · 按可听出的切分档位读", ""]
    lines.append(
        f"{'stratum':20s} {'offset':>8s} {'n(A+B)':>7s} {'非语义':>7s} "
        f"{'unsure':>7s} {'被切率':>8s}  CI95"
    )
    for name, report in result["strata"].items():
        offset = report.get("offset_s")
        rate = report.get("clipped_rate")
        ci = report.get("clipped_rate_ci95") or ["", ""]
        lines.append(
            f"{name:20s} {offset if offset is not None else '':>8} "
            f"{report['decisive']:>7d} {report.get('non_semantic', 0):>7d} "
            f"{report['unsure']:>7d} "
            f"{(f'{rate:.1%}' if rate is not None else '--'):>8s}  "
            f"[{ci[0]}, {ci[1]}]"
        )
    lines.append("")
    lines.append("与对照（control_early）之差：")
    for name, comparison in result["comparisons_vs_floor"].items():
        if not comparison:
            continue
        lines.append(
            f"  {name:20s} {comparison['difference']:+.4f} "
            f"CI95 {comparison['ci95_approx']} "
            f"{'分得开' if comparison['separated_from_reference'] else '分不开'}"
        )
    lines.append("")
    lines.append(f"结论：{result['verdict']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--verdicts", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = build(Path(args.answers), Path(args.verdicts))
    print(render(result))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
