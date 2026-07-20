#!/usr/bin/env python3
"""Evaluate exact-island repairs for contaminated Scorer all-background rows."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.generate_scorer_v10_background_speech_repair_audit_html import (
    ISLAND_SCHEMA,
    LINK_SCHEMA,
    MANUAL_VERDICT_SCHEMA,
    SUMMARY_SCHEMA,
)


RESULT_SCHEMA = "speech_scorer_v10_background_speech_repair_manual_gate_v1"
DECISION_SCHEMA = "speech_scorer_v10_background_speech_repair_decision_v1"
EVENT_SCHEMA = "speech_scorer_v10_background_speech_repair_event_v1"
ISLAND_VERDICTS = {
    "target_speech_span_ok",
    "target_speech_boundary_incomplete",
    "background_or_nonsemantic",
    "unsure",
    "unreviewed",
}
LINK_VERDICTS = {
    "same_asr_unit",
    "separate_target_events",
    "unsure",
    "unreviewed",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _core_id(
    *, source_id: str, event_index: int, start_sample: int, end_sample: int
) -> str:
    payload = (
        "scorer-v10-background-speech-repair-v1\0"
        f"{source_id}\0{event_index}\0{start_sample}\0{end_sample}"
    ).encode("utf-8")
    return "scorer-v10-repair-core-" + hashlib.sha256(payload).hexdigest()


def evaluate(
    *,
    audit_summary: Path,
    audit_manifest: Path,
    manual_verdicts: Path,
    output: Path,
) -> dict[str, Any]:
    summary = json.loads(audit_summary.read_text(encoding="utf-8-sig"))
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("invalid background speech repair audit summary schema")
    if Path(str(summary.get("audit_manifest") or "")).resolve() != audit_manifest.resolve():
        raise ValueError("background speech repair manifest does not match its summary")
    if _sha256(audit_manifest) != str(summary.get("audit_manifest_sha256") or ""):
        raise ValueError("background speech repair manifest changed after page generation")
    for path_key, sha_key in (
        ("canonical_sources", "canonical_sources_sha256"),
        ("prediction_audit_manifest", "prediction_audit_manifest_sha256"),
        ("prediction_manual_verdicts", "prediction_manual_verdicts_sha256"),
    ):
        bound = Path(str(summary.get(path_key) or ""))
        if not bound.is_file() or _sha256(bound) != str(summary.get(sha_key) or ""):
            raise ValueError(f"background speech repair evidence changed: {path_key}")

    targets: dict[str, dict[str, Any]] = {}
    islands_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    links_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(audit_manifest):
        item_id = str(row.get("item_id") or "")
        if not item_id or item_id in targets:
            raise ValueError("background speech repair items require unique item_id values")
        item_type = str(row.get("item_type") or "")
        expected_schema = ISLAND_SCHEMA if item_type == "island" else LINK_SCHEMA
        if item_type not in {"island", "link"} or row.get("schema") != expected_schema:
            raise ValueError("invalid background speech repair item schema")
        targets[item_id] = row
        source_id = str(row.get("source_id") or "")
        if item_type == "island":
            islands_by_source[source_id].append(row)
        else:
            links_by_source[source_id].append(row)
    if len(targets) != int(summary.get("review_item_count") or -1):
        raise ValueError("background speech repair item count mismatch")
    if sum(len(rows) for rows in islands_by_source.values()) != int(
        summary.get("island_count") or -1
    ):
        raise ValueError("background speech repair island count mismatch")
    if sum(len(rows) for rows in links_by_source.values()) != int(
        summary.get("link_count") or -1
    ):
        raise ValueError("background speech repair link count mismatch")

    for source_id, islands in islands_by_source.items():
        islands.sort(key=lambda row: int(row["island_index"]))
        if [int(row["island_index"]) for row in islands] != list(range(len(islands))):
            raise ValueError(f"background speech repair island indexes are invalid: {source_id}")
        links = links_by_source.get(source_id, [])
        links.sort(key=lambda row: int(row["link_index"]))
        if len(links) != max(0, len(islands) - 1):
            raise ValueError(f"background speech repair link topology mismatch: {source_id}")
        for index, link in enumerate(links):
            if (
                int(link["left_island_index"]) != index
                or int(link["right_island_index"]) != index + 1
                or str(link["left_island_id"]) != str(islands[index]["item_id"])
                or str(link["right_island_id"]) != str(islands[index + 1]["item_id"])
            ):
                raise ValueError(f"background speech repair link identity mismatch: {source_id}")

    verdicts: dict[str, dict[str, Any]] = {}
    for row in _rows(manual_verdicts):
        if row.get("schema") != MANUAL_VERDICT_SCHEMA:
            raise ValueError("invalid background speech repair verdict schema")
        item_id = str(row.get("item_id") or "")
        if item_id not in targets or item_id in verdicts:
            raise ValueError(f"invalid or duplicate background speech repair verdict: {item_id}")
        target = targets[item_id]
        for field in ("item_type", "source_id"):
            if str(row.get(field) or "") != str(target.get(field) or ""):
                raise ValueError(f"background speech repair verdict {field} mismatch: {item_id}")
        verdict = str(row.get("verdict") or "unreviewed")
        allowed = ISLAND_VERDICTS if target["item_type"] == "island" else LINK_VERDICTS
        if verdict not in allowed:
            raise ValueError(f"invalid {target['item_type']} repair verdict: {verdict}")
        verdicts[item_id] = row

    missing_island_ids: list[str] = []
    unreviewed_island_ids: list[str] = []
    unsure_ids: list[str] = []
    boundary_followup_ids: list[str] = []
    target_island_ids: list[str] = []
    required_link_ids: list[str] = []
    unreviewed_required_link_ids: list[str] = []
    verdict_counts: Counter[str] = Counter()
    decisions: list[dict[str, Any]] = []
    source_without_target_ids: list[str] = []
    for source_id in sorted(islands_by_source):
        islands = islands_by_source[source_id]
        links = links_by_source[source_id]
        source_target_count = 0
        for island in islands:
            item_id = str(island["item_id"])
            verdict = str(verdicts.get(item_id, {}).get("verdict") or "missing")
            verdict_counts[verdict] += 1
            if verdict == "missing":
                missing_island_ids.append(item_id)
            elif verdict == "unreviewed":
                unreviewed_island_ids.append(item_id)
            elif verdict == "unsure":
                unsure_ids.append(item_id)
            elif verdict == "target_speech_boundary_incomplete":
                boundary_followup_ids.append(item_id)
            elif verdict == "target_speech_span_ok":
                target_island_ids.append(item_id)
                source_target_count += 1
            decisions.append(
                {
                    **island,
                    "schema": DECISION_SCHEMA,
                    "verdict": verdict,
                }
            )
        if source_target_count == 0:
            source_without_target_ids.append(source_id)
        for link in links:
            left_id = str(link["left_island_id"])
            right_id = str(link["right_island_id"])
            left = str(verdicts.get(left_id, {}).get("verdict") or "missing")
            right = str(verdicts.get(right_id, {}).get("verdict") or "missing")
            required = left == right == "target_speech_span_ok"
            item_id = str(link["item_id"])
            verdict = str(verdicts.get(item_id, {}).get("verdict") or "missing")
            if required:
                required_link_ids.append(item_id)
                verdict_counts[verdict] += 1
                if verdict in {"missing", "unreviewed"}:
                    unreviewed_required_link_ids.append(item_id)
                elif verdict == "unsure":
                    unsure_ids.append(item_id)
            decisions.append(
                {
                    **link,
                    "schema": DECISION_SCHEMA,
                    "required": required,
                    "verdict": verdict,
                }
            )

    manual_review_complete = not (
        missing_island_ids
        or unreviewed_island_ids
        or unreviewed_required_link_ids
    )
    repair_ready = (
        manual_review_complete
        and not unsure_ids
        and not boundary_followup_ids
        and not source_without_target_ids
    )

    events: list[dict[str, Any]] = []
    if repair_ready:
        for source_id in sorted(islands_by_source):
            islands = islands_by_source[source_id]
            links = links_by_source[source_id]
            current: dict[str, Any] | None = None
            source_events: list[dict[str, Any]] = []
            for index, island in enumerate(islands):
                item_id = str(island["item_id"])
                if str(verdicts[item_id]["verdict"]) != "target_speech_span_ok":
                    current = None
                    continue
                merge_previous = False
                link_id = ""
                if index > 0 and current is not None:
                    link = links[index - 1]
                    link_id = str(link["item_id"])
                    merge_previous = (
                        str(verdicts.get(link_id, {}).get("verdict") or "")
                        == "same_asr_unit"
                    )
                if merge_previous:
                    current["end_frame"] = int(island["end_frame"])
                    current["end_sample"] = int(island["end_sample"])
                    current["end_s"] = float(island["end_s"])
                    current["member_island_ids"].append(item_id)
                    current["same_unit_link_ids"].append(link_id)
                else:
                    current = {
                        "source_id": source_id,
                        "partition": str(island["partition"]),
                        "background_id": str(island["background_id"]),
                        "start_frame": int(island["start_frame"]),
                        "end_frame": int(island["end_frame"]),
                        "start_sample": int(island["start_sample"]),
                        "end_sample": int(island["end_sample"]),
                        "start_s": float(island["start_s"]),
                        "end_s": float(island["end_s"]),
                        "member_island_ids": [item_id],
                        "same_unit_link_ids": [],
                    }
                    source_events.append(current)
            for event_index, event in enumerate(source_events):
                event.update(
                    schema=EVENT_SCHEMA,
                    event_id=f"{source_id}::event{event_index:02d}",
                    event_index=event_index,
                    core_id=_core_id(
                        source_id=source_id,
                        event_index=event_index,
                        start_sample=int(event["start_sample"]),
                        end_sample=int(event["end_sample"]),
                    ),
                    label="speech",
                    label_source="manual_background_speech_repair_v1",
                )
                events.append(event)

    output.parent.mkdir(parents=True, exist_ok=True)
    decisions_path = output.with_suffix(".decisions.jsonl")
    events_path = output.with_suffix(".events.jsonl")
    decisions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decisions),
        encoding="utf-8",
    )
    events_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in events),
        encoding="utf-8",
    )
    result = {
        "schema": RESULT_SCHEMA,
        "audit_summary": str(audit_summary),
        "audit_manifest": str(audit_manifest),
        "manual_verdicts": str(manual_verdicts),
        "canonical_sources": str(summary["canonical_sources"]),
        "canonical_sources_sha256": str(summary["canonical_sources_sha256"]),
        "source_count": len(islands_by_source),
        "island_count": sum(len(rows) for rows in islands_by_source.values()),
        "target_island_count": len(target_island_ids),
        "target_island_ids": sorted(target_island_ids),
        "required_link_count": len(required_link_ids),
        "required_link_ids": sorted(required_link_ids),
        "missing_island_count": len(missing_island_ids),
        "missing_island_ids": sorted(missing_island_ids),
        "unreviewed_island_count": len(unreviewed_island_ids),
        "unreviewed_island_ids": sorted(unreviewed_island_ids),
        "unreviewed_required_link_count": len(unreviewed_required_link_ids),
        "unreviewed_required_link_ids": sorted(unreviewed_required_link_ids),
        "unsure_count": len(unsure_ids),
        "unsure_ids": sorted(unsure_ids),
        "boundary_followup_count": len(boundary_followup_ids),
        "boundary_followup_ids": sorted(boundary_followup_ids),
        "source_without_target_count": len(source_without_target_ids),
        "source_without_target_ids": sorted(source_without_target_ids),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "manual_review_complete": manual_review_complete,
        "repair_event_count": len(events),
        "repair_events": str(events_path),
        "decisions": str(decisions_path),
        "canonical_repair_ready": repair_ready,
        "training_manifest_allowed": False,
        "checkpoint_promotion_authorized": False,
    }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-summary", required=True)
    parser.add_argument("--audit-manifest", required=True)
    parser.add_argument("--manual-verdicts", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            evaluate(
                audit_summary=Path(args.audit_summary),
                audit_manifest=Path(args.audit_manifest),
                manual_verdicts=Path(args.manual_verdicts),
                output=Path(args.output),
            ),
            ensure_ascii=False,
        )
    )
