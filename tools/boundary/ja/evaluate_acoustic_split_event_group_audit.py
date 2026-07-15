#!/usr/bin/env python3
"""Compile manually grouped acoustic Split candidates into events/sub-islands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ITEM_SCHEMA = "acoustic_split_event_group_audit_v1"
VERDICT_SCHEMA = "acoustic_split_event_group_manual_verdict_v1"
SUMMARY_SCHEMA = "acoustic_split_event_group_gate_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate(*, items: Path, verdicts: Path, output_dir: Path) -> dict[str, Any]:
    item_rows = _rows(items)
    verdict_by_id = {
        str(row["sample_id"]): row
        for row in _rows(verdicts)
        if row.get("schema") == VERDICT_SCHEMA
    }
    events: list[dict[str, Any]] = []
    subislands: list[dict[str, Any]] = []
    complete_sources = 0
    unsure_links = 0
    for item in item_rows:
        if item.get("schema") != ITEM_SCHEMA:
            raise ValueError("incompatible event grouping item schema")
        sample_id = str(item["sample_id"])
        verdict = verdict_by_id.get(sample_id) or {}
        decisions = {
            str(row["link_id"]): str(row.get("decision") or "unreviewed")
            for row in verdict.get("links") or []
        }
        expected_links = [str(row["link_id"]) for row in item["links"]]
        complete = all(decisions.get(link) in {"same_event", "new_event"} for link in expected_links)
        unsure_links += sum(decisions.get(link) == "unsure" for link in expected_links)
        complete_sources += int(complete)
        if not complete:
            continue
        candidate_by_id = {
            str(candidate["candidate_id"]): candidate
            for candidate in item.get("source_candidates") or []
        }
        # Older items keep candidates only in the source audit; the builder
        # embeds the required split candidate rows below before this evaluator
        # is used in production.
        if not candidate_by_id:
            candidate_by_id = {
                str(candidate["candidate_id"]): candidate
                for run in item["runs"]
                for candidate in run.get("candidates") or []
            }
        source_events: list[dict[str, Any]] = []
        for run in item["runs"]:
            ids = [str(value) for value in run["candidate_ids"]]
            groups: list[list[str]] = [[ids[0]]]
            for left, right in zip(ids, ids[1:]):
                decision = decisions[f"{left}__{right}"]
                if decision == "same_event":
                    groups[-1].append(right)
                else:
                    groups.append([right])
            for group in groups:
                rows = [candidate_by_id[value] for value in group]
                representative = max(
                    rows, key=lambda row: float(row["proposer_probability"])
                )
                source_events.append(
                    {
                        "candidate_ids": group,
                        "representative_candidate_id": str(representative["candidate_id"]),
                        "representative_time_s": float(representative["time_s"]),
                        "representative_probability": float(
                            representative["proposer_probability"]
                        ),
                        "basin_start_s": min(float(row["time_s"]) for row in rows),
                        "basin_end_s": max(float(row["time_s"]) for row in rows),
                    }
                )
        source_events.sort(key=lambda row: row["representative_time_s"])
        for index, event in enumerate(source_events):
            event_id = f"{sample_id}__e{index:02d}"
            event["schema"] = "acoustic_split_event_v3"
            event["sample_id"] = sample_id
            event["event_id"] = event_id
            events.append(event)
        boundaries = [0.0] + [row["representative_time_s"] for row in source_events] + [float(item["duration_s"])]
        for index, (start_s, end_s) in enumerate(zip(boundaries, boundaries[1:])):
            subislands.append(
                {
                    "schema": "provisional_speech_subisland_v1",
                    "sample_id": sample_id,
                    "subisland_id": f"{sample_id}__s{index:02d}",
                    "audio": str(item["audio"]),
                    "start_s": float(start_s),
                    "end_s": float(end_s),
                    "duration_s": float(end_s) - float(start_s),
                    "left_event_id": (
                        source_events[index - 1]["event_id"] if index > 0 else None
                    ),
                    "right_event_id": (
                        source_events[index]["event_id"]
                        if index < len(source_events)
                        else None
                    ),
                    "cueqc_status": "requires_pre_inner_manual_label",
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "events.jsonl", events)
    _write(output_dir / "provisional_subislands.jsonl", subislands)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "source_count": len(item_rows),
        "complete_source_count": complete_sources,
        "unsure_link_count": unsure_links,
        "event_count": len(events),
        "provisional_subisland_count": len(subislands),
        "training_ready": complete_sources == len(item_rows) and unsure_links == 0,
        "events": str(output_dir / "events.jsonl"),
        "provisional_subislands": str(output_dir / "provisional_subislands.jsonl"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate acoustic Split event grouping audit.")
    parser.add_argument("--items", required=True)
    parser.add_argument("--verdicts", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            evaluate(
                items=Path(args.items),
                verdicts=Path(args.verdicts),
                output_dir=Path(args.output_dir),
            ),
            ensure_ascii=False,
        )
    )
