#!/usr/bin/env python3
"""Score a teacher prompt against the human verdicts, per direction.

A single agreement rate hides the thing that matters. False keeps and false
drops are not equal errors here: a kept clip that holds no words costs ASR
compute and is filtered downstream, while a dropped clip that holds words is
never transcribed and never comes back. So the two directions are reported
separately, and the one that decides whether the teacher may be trusted is
recall on clips a human heard words in.

`unsure` is reported and never redistributed, on either side. A teacher that
answers `unsure` to a third of the corpus has not agreed with anything; folding
those into a rate would hide that.

The development and held-out halves are scored by the same code and reported
with the same fields, so the only difference between the two numbers is which
clips they came from.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

SUMMARY_SCHEMA = "word_teacher_calibration_result_v1"
DECISIVE = ("words", "no_words")

# What relabelling is worth is arithmetic, not taste. 32.0% of dropped seconds
# hold words today; a teacher with recall R leaves 32% x (1 - R). R = 0.70 takes
# it to 9.6%, a threefold cut, which is worth doing. Demanding 0.85 sounds safer
# and is not: at the ~28 word-positive clips one half of this set can hold, an
# 0.85 lower bound needs a near-flawless score, so the gate would reject teachers
# that would plainly improve the corpus.
RECALL_FLOOR = 0.70
FALSE_DROP_RATE_TODAY = 0.320
DROPPED_MINUTES = 116.3

# Recall alone is trivially gamed: a teacher answering `words` to everything
# scores 1.00. Specificity is what makes the recall gate mean something, so it
# is binding too, at the level below which the teacher is not discriminating at
# all rather than at a level tuned for cost.
SPECIFICITY_FLOOR = 0.50


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    if total <= 0:
        return None
    rate = hits / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    half = z * ((rate * (1 - rate) / total + z * z / (4 * total * total)) ** 0.5)
    half /= denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def join(items: list[dict], answers: list[dict]) -> list[dict]:
    by_id = {str(row["item_id"]): row for row in items}
    seen: set[str] = set()
    joined: list[dict] = []
    for answer in answers:
        item_id = str(answer.get("item_id") or "")
        item = by_id.get(item_id)
        if item is None:
            # An answer from the other half is not an error; runs share a file.
            continue
        if item_id in seen:
            raise ValueError(f"duplicate teacher answer for {item_id}")
        seen.add(item_id)
        joined.append({**item, "teacher": str(answer.get("label") or ""),
                       "heard": str(answer.get("heard") or "")})
    return joined


def _rate(hits: int, total: int) -> dict[str, Any]:
    interval = wilson(hits, total)
    return {
        "n": total,
        "agreed": hits,
        "rate": round(hits / total, 4) if total else None,
        "ci95": [round(interval[0], 4), round(interval[1], 4)] if interval else None,
    }


def report(joined: list[dict], *, name: str) -> dict[str, Any]:
    decided = [row for row in joined if row["teacher"] in DECISIVE]
    words = [row for row in decided if row["human"] == "words"]
    no_words = [row for row in decided if row["human"] == "no_words"]
    recall = _rate(sum(1 for r in words if r["teacher"] == "words"), len(words))
    specificity = _rate(
        sum(1 for r in no_words if r["teacher"] == "no_words"), len(no_words)
    )
    overall = _rate(
        sum(1 for r in decided if r["teacher"] == r["human"]), len(decided)
    )

    by_stratum: dict[str, Counter] = defaultdict(Counter)
    for row in decided:
        by_stratum[row["stratum"]][f"{row['human']}->{row['teacher']}"] += 1

    verdict = "unusable"
    if recall["ci95"] is None or specificity["ci95"] is None:
        basis = "缺少某一侧的人工样本，无法评估"
    elif specificity["ci95"][0] < SPECIFICITY_FLOOR:
        basis = (
            f"无词特异度下界 {specificity['ci95'][0]:.0%} 低于 "
            f"{SPECIFICITY_FLOOR:.0%}：它没有在区分，只是倾向于一律说有词，"
            "此时召回再高也没有意义"
        )
    elif recall["ci95"][0] >= RECALL_FLOOR:
        verdict = "trusted"
        basis = "有词召回的整个置信区间在门槛以上，可以用它标全量"
    elif recall["ci95"][1] < RECALL_FLOOR:
        basis = "有词召回的整个置信区间在门槛以下，用它重标改善有限，改人工分批听"
    else:
        verdict = "undecided"
        basis = "有词召回的置信区间横跨门槛，这一批定不下来"

    # What accepting this teacher would actually buy, in the units the false-drop
    # audit measured. Reported whatever the verdict, because a gate that only
    # says pass/fail hides how close the call was.
    projected: dict[str, Any] = {}
    if recall["rate"] is not None:
        residual = FALSE_DROP_RATE_TODAY * (1.0 - recall["rate"])
        projected = {
            "false_drop_rate_today": FALSE_DROP_RATE_TODAY,
            "false_drop_rate_after": round(residual, 4),
            "minutes_recovered": round(
                DROPPED_MINUTES * FALSE_DROP_RATE_TODAY * recall["rate"], 1
            ),
        }
        if specificity["rate"] is not None:
            projected["extra_minutes_kept"] = round(
                DROPPED_MINUTES
                * (1 - FALSE_DROP_RATE_TODAY)
                * (1 - specificity["rate"]),
                1,
            )

    return {
        "projected": projected,
        "schema": SUMMARY_SCHEMA,
        "half": name,
        "clips": len(joined),
        "teacher_unsure": sum(1 for row in joined if row["teacher"] == "unsure"),
        "teacher_off_vocabulary": sum(
            1 for row in joined if row["teacher"] not in (*DECISIVE, "unsure")
        ),
        "verdict": verdict,
        "basis": basis,
        "recall_floor": RECALL_FLOOR,
        "specificity_floor": SPECIFICITY_FLOOR,
        "words_recall": recall,
        "no_words_specificity": specificity,
        "overall_agreement": overall,
        "per_stratum": {k: dict(v) for k, v in sorted(by_stratum.items())},
        "disagreements": [
            {
                "item_id": row["item_id"],
                "human": row["human"],
                "teacher": row["teacher"],
                "clip_s": row["clip_duration_s"],
                "type_label": row["type_label"],
                "heard": row["heard"],
            }
            for row in decided
            if row["teacher"] != row["human"]
        ],
    }


def render(result: dict[str, Any]) -> str:
    lines = [f"=== {result['half']} ({result['clips']} 条) ==="]
    for key, label in (
        ("words_recall", "人工有词 → teacher 也说有词 (召回，决定性指标)"),
        ("no_words_specificity", "人工无词 → teacher 也说无词"),
        ("overall_agreement", "整体吻合"),
    ):
        entry = result[key]
        if entry["rate"] is None:
            lines.append(f"  {label}: 无样本")
            continue
        low, high = entry["ci95"]
        lines.append(
            f"  {label}: {entry['agreed']}/{entry['n']} = {entry['rate']:.1%}"
            f"（95% CI {low:.1%}~{high:.1%}）"
        )
    lines.append(
        f"  teacher 答 unsure {result['teacher_unsure']} 条，"
        f"词表外 {result['teacher_off_vocabulary']} 条"
    )
    projected = result.get("projected") or {}
    if projected:
        lines.append(
            f"\n若采信它重标：假删率 {projected['false_drop_rate_today']:.1%} → "
            f"{projected['false_drop_rate_after']:.1%}"
            f"，找回约 {projected['minutes_recovered']} 分钟语音"
            + (
                f"，多留约 {projected['extra_minutes_kept']} 分钟无词音频"
                if "extra_minutes_kept" in projected
                else ""
            )
        )
    lines.append(
        f"\n判定：{result['verdict']}"
        f"（门槛 召回 ≥ {result['recall_floor']:.0%}、特异度 ≥ "
        f"{result['specificity_floor']:.0%}）"
    )
    lines.append(f"  {result['basis']}")
    if result["disagreements"]:
        lines.append("\n分歧明细")
        for row in result["disagreements"]:
            lines.append(
                f"  {row['item_id']:<34} 人工 {row['human']:<9} teacher "
                f"{row['teacher']:<9} {row['clip_s']:>5.2f}s  {row['heard'][:44]}"
            )
    return "\n".join(lines)


def build(*, items: Path, answers: Path, name: str, output: Path | None) -> dict:
    result = report(join(_rows(items), _rows(answers)), name=name)
    result["items"] = str(items)
    result["answers"] = str(answers)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--name", default="development")
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(
        render(
            build(
                items=Path(args.items).expanduser().resolve(),
                answers=Path(args.answers).expanduser().resolve(),
                name=args.name,
                output=Path(args.output).expanduser().resolve()
                if args.output
                else None,
            )
        )
    )
