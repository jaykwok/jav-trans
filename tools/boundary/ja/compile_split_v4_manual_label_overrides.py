#!/usr/bin/env python3
"""Compile Split v4 gate verdicts into authoritative cut/continue/ignore overrides."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def compile_overrides(
    *, gate_manifest: Path, gate_verdicts: list[Path], candidate_verdicts: Path, output: Path
) -> dict:
    manifest = {row["audit_id"]: row for row in _rows(gate_manifest)}
    gate: dict[str, dict] = {}
    for path in gate_verdicts:
        for row in _rows(path):
            if row.get("verdict") not in {None, "", "unreviewed"}:
                gate[str(row["audit_id"])] = row
    compiled: dict[tuple[str, float], dict] = {}
    for audit_id, verdict in gate.items():
        if verdict.get("verdict") != "false_cut":
            continue
        item = manifest[audit_id]
        key = (str(item["audio_id"]), round(float(item["time_s"]), 6))
        compiled[key] = {
            "schema": "split_v4_manual_label_override_v1",
            "audio_id": key[0],
            "time_s": key[1],
            "teacher_label": "continue",
            "training_label": "continue",
            "training_ignore_reason": "",
            "source_audit_id": audit_id,
            "reason": "manual_sentence_internal_false_cut",
        }
    for residual in _rows(candidate_verdicts):
        if not residual.get("complete"):
            raise ValueError(f"{residual.get('audit_id')}: candidate audit is incomplete")
        for candidate in residual["candidates"]:
            manual = str(candidate.get("manual_label") or "unreviewed")
            if manual not in {"cut", "continue", "unsure"}:
                raise ValueError(f"unsupported manual label: {manual!r}")
            key = (str(residual["audio_id"]), round(float(candidate["time_s"]), 6))
            row = {
                "schema": "split_v4_manual_label_override_v1",
                "audio_id": key[0],
                "time_s": key[1],
                "teacher_label": manual,
                "training_label": manual if manual in {"cut", "continue"} else "ignore",
                "training_ignore_reason": "manual_unsure" if manual == "unsure" else "",
                "source_audit_id": str(residual["audit_id"]),
                "reason": "manual_missing_cut_candidate_review",
            }
            previous = compiled.get(key)
            if previous and previous["teacher_label"] != row["teacher_label"]:
                raise ValueError(f"conflicting manual override for {key}")
            compiled[key] = row
    rows = sorted(compiled.values(), key=lambda row: (row["audio_id"], row["time_s"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), "utf-8")
    counts = {label: sum(row["teacher_label"] == label for row in rows) for label in ("cut", "continue", "unsure")}
    summary = {
        "schema": "split_v4_manual_label_override_summary_v1",
        "override_count": len(rows),
        "teacher_label_counts": counts,
        "training_label_counts": {
            "cut": counts["cut"], "continue": counts["continue"], "ignore": counts["unsure"]
        },
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-manifest", required=True)
    parser.add_argument("--gate-verdicts", action="append", required=True)
    parser.add_argument("--candidate-verdicts", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(compile_overrides(
        gate_manifest=Path(args.gate_manifest),
        gate_verdicts=[Path(path) for path in args.gate_verdicts],
        candidate_verdicts=Path(args.candidate_verdicts),
        output=Path(args.output),
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
