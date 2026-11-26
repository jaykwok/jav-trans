"""Blind meaning review for synthetic translation experiments.

Reads one or more `run_translation_quality_eval.py` reports over the same cases
and turns their outputs into a sheet a judge can label without knowing which
arm wrote what; `score` then reports, per arm, the four numbers the 2026-09-28
research asked every comparison to show side by side:

- request failures (arm status) and extra requests (retries),
- character residue (kana or empty left in the text),
- meaning errors on checkpoint lines,
- meaning errors on ordinary lines (a normal sentence made wrong).

A completed request or a clean character check is not a correct translation;
only the labels decide meaning. Judgement is per original authored sentence:
source planning may split one line into several cues, and those are read
together.

  make  --report [label=]report.json ... --out DIR   -> sheet.md, key.json, labels.json
  score --out DIR                                    -> score.json (+ table on stdout)

Fill `labels.json` with 1 (meaning right) or 0 (meaning wrong) for every item
before opening `key.json`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _checkpoint_fallback() -> dict[tuple[str, str], str]:
    """Checkpoints for reports written before cases carried them."""
    from tools.audits.translation_quality_cases import build_cases, plan_source_cases

    return {
        (case["case_id"], case["text"]): case.get("checkpoint", "")
        for case in plan_source_cases(build_cases())
    }


def _units(cases: list[dict]) -> list[dict]:
    """Group consecutive cues of one authored sentence."""
    fallback = None
    units: list[dict] = []
    for index, case in enumerate(cases):
        if "checkpoint" in case:
            checkpoint = case["checkpoint"]
        else:
            fallback = fallback if fallback is not None else _checkpoint_fallback()
            checkpoint = fallback.get((case.get("case_id", ""), case.get("text", "")), "")
        key = (case.get("case_id", ""), case.get("reference_source") or f"#{index}")
        if units and units[-1]["key"] == key:
            units[-1]["cues"].append(index)
            continue
        units.append({
            "key": key, "cues": [index], "scene": case.get("case_id", ""),
            "category": case.get("category", ""),
            "source": case.get("reference_source") or case.get("text", ""),
            "reference": case.get("reference", ""), "checkpoint": checkpoint,
        })
    return units


def _arm_metrics(arm: dict, cue_count: int) -> dict:
    timings = [
        timing for timing in arm.get("timings") or []
        if isinstance(timing, dict) and timing.get("batch_index") is not None and not timing.get("is_warmup")
    ]
    requests = sum(int(timing.get("request_count") or 0) for timing in timings)
    return {
        "status": arm.get("status"),
        "complete": len(arm.get("texts") or []) == cue_count,
        "character_residue": len(arm.get("remaining_kana_or_empty_ids") or []),
        "requests": requests,
        "extra_requests": max(0, requests - sum(1 for timing in timings if timing.get("request_count"))),
    }


def make(reports: list[str], out: Path, *, seed: int = 20260928) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty; use a fresh directory")
    loaded = []
    for spec in reports:
        label, _, path = spec.rpartition("=")
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        if report.get("synthetic_only") is not True:
            raise ValueError("only synthetic experiment reports can be reviewed with this tool")
        loaded.append((label or Path(path).parent.name, report))
    cases = loaded[0][1]["cases"]
    for _label, report in loaded[1:]:
        if [case["text"] for case in report["cases"]] != [case["text"] for case in cases]:
            raise ValueError("reports cover different cases; compare like with like")
    units = _units(cases)

    arms: dict[str, dict] = {}
    texts: dict[str, list[str]] = {}
    for label, report in loaded:
        for name, arm in report.get("arms", {}).items():
            arm_id = f"{label}:{name}" if len(loaded) > 1 else name
            if arm_id in arms:
                raise ValueError(f"duplicate arm {arm_id!r}; use distinct report labels")
            arms[arm_id] = _arm_metrics(arm, len(cases))
            if arms[arm_id]["complete"]:
                texts[arm_id] = list(arm["texts"])

    rng = random.Random(seed)
    key: dict[str, list[str]] = {}
    unit_meta: dict[str, dict] = {}
    lines: list[str] = []
    scene = None
    for unit_id, unit in enumerate(units):
        if unit["scene"] != scene:
            scene = unit["scene"]
            lines.extend(["", f"## {scene}  ({unit['category']})"])
            for other in units:
                if other["scene"] == scene:
                    lines.append(f"  - {other['source']}   ~ {other['reference']}")
        unit_meta[str(unit_id)] = {"checkpoint": bool(unit["checkpoint"]), "cues": unit["cues"]}
        outputs: dict[str, list[str]] = defaultdict(list)
        for arm_id, arm_texts in texts.items():
            outputs[" / ".join(arm_texts[index] for index in unit["cues"])].append(arm_id)
        candidates = list(outputs)
        if len(candidates) > len(LETTERS):
            raise ValueError(f"unit {unit_id} has more than {len(LETTERS)} distinct outputs; compare fewer arms")
        rng.shuffle(candidates)
        lines.append(f"  [{unit_id}] {unit['source']}")
        if unit["checkpoint"]:
            lines.append(f"      ★ 检查点：{unit['checkpoint']}")
        for letter, text in zip(LETTERS, candidates):
            item = f"{unit_id}.{letter}"
            key[item] = outputs[text]
            lines.append(f"      {item} {text}")
    out.mkdir(parents=True, exist_ok=True)
    header = [
        "# 盲评表",
        "",
        "每条候选按原句含义判 1（正确）或 0（错误），填入 labels.json；判完之前不要打开 key.json。",
        "有检查点的句子先看检查点；没有检查点的是普通句，含义被改坏同样判 0。措辞生硬但含义正确判 1。",
    ]
    (out / "sheet.md").write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    (out / "key.json").write_text(
        json.dumps({"items": key, "units": unit_meta, "arms": arms}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    (out / "labels.json").write_text(json.dumps({item: None for item in key}, indent=1), encoding="utf-8")
    return {"units": len(units), "items": len(key), "arms": sorted(arms)}


def score(out: Path) -> dict:
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    labels = json.loads((out / "labels.json").read_text(encoding="utf-8"))
    unlabeled = sorted(item for item in key["items"] if labels.get(item) not in (0, 1))
    if unlabeled:
        raise ValueError(f"{len(unlabeled)} items are not labelled 0/1, e.g. {unlabeled[:5]}")
    result: dict[str, dict] = {}
    for arm_id, metrics in key["arms"].items():
        row = {**metrics, "units": 0, "checkpoint_errors": 0, "ordinary_errors": 0}
        result[arm_id] = row
    for item, arm_ids in key["items"].items():
        unit = key["units"][item.split(".")[0]]
        for arm_id in arm_ids:
            row = result[arm_id]
            row["units"] += 1
            if labels[item] == 0:
                row["checkpoint_errors" if unit["checkpoint"] else "ordinary_errors"] += 1
    (out / "score.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def _table(result: dict) -> str:
    rows = [
        "| 组 | 状态 | 额外请求 | 字符残留 | 检查点含义错误 | 普通句被改坏 | 判定句数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm_id, row in sorted(result.items()):
        rows.append(
            f"| {arm_id} | {row['status']} | {row['extra_requests']} | {row['character_residue']} "
            f"| {row['checkpoint_errors']} | {row['ordinary_errors']} | {row['units']} |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    make_parser = sub.add_parser("make")
    make_parser.add_argument("--report", action="append", required=True, help="[label=]path to report.json")
    make_parser.add_argument("--out", required=True)
    make_parser.add_argument("--seed", type=int, default=20260928)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "make":
        print(json.dumps(make(args.report, Path(args.out), seed=args.seed), ensure_ascii=False))
    else:
        print(_table(score(Path(args.out))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
