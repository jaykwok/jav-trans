#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_MANIFEST = (
    PROJECT_ROOT
    / "datasets/train/cueqc-v13-galgame-unique-composites2048-v1/"
    "source_manifest_stratified1024.jsonl"
)
DEFAULT_RUNTIME_MANIFEST = (
    PROJECT_ROOT
    / "agents/temp/20260717_191305_cueqc-v13-split-v4-runtime1024/"
    "runtime_v11_provisional.jsonl"
)
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "agents/temp/20260717_233330_cueqc-v13-binary-role-holdout-dw125/"
    "predictions.jsonl"
)
DEFAULT_AUDITED_SAFE_DROP_VERDICTS = (
    PROJECT_ROOT
    / "agents/audits/20260717_233937_cueqc-v13-binary-false-drop2/"
    "manual_verdicts.jsonl"
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            yield dict(payload)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _required_float(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"row missing numeric {key}") from exc
    return value


def load_exact_sources(path: Path) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    core_ids: set[str] = set()
    for row in iter_jsonl(path):
        source_id = str(row.get("sample_id") or "")
        if not source_id or source_id in sources:
            raise ValueError(f"source manifest has missing/duplicate sample_id: {source_id!r}")
        sample_rate = int(row.get("sample_rate") or 0)
        if sample_rate <= 0:
            raise ValueError(f"source {source_id} has invalid sample_rate")
        cores: list[dict[str, Any]] = []
        for raw_core in row.get("core_spans") or []:
            core = dict(raw_core)
            core_id = str(core.get("core_id") or "")
            if not core_id or core_id in core_ids:
                raise ValueError(f"core manifest has missing/duplicate core_id: {core_id!r}")
            start_sample = int(core.get("start_sample") or 0)
            end_sample = int(core.get("end_sample") or 0)
            if end_sample <= start_sample:
                raise ValueError(f"core {core_id} has an invalid sample span")
            core_ids.add(core_id)
            cores.append(
                {
                    "core_id": core_id,
                    "text": str(core.get("text") or ""),
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "start_s": start_sample / sample_rate,
                    "end_s": end_sample / sample_rate,
                }
            )
        if not cores:
            raise ValueError(f"source {source_id} has no exact core spans")
        sources[source_id] = {
            "source_id": source_id,
            "source_partition": str(row.get("source_partition") or ""),
            "audio": str(row.get("audio") or ""),
            "sample_rate": sample_rate,
            "cores": sorted(cores, key=lambda item: (item["start_sample"], item["end_sample"])),
        }
    return sources


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        row_id = str(row.get("row_id") or "")
        prediction = str(row.get("prediction") or "")
        if not row_id or row_id in predictions:
            raise ValueError(f"predictions have missing/duplicate row_id: {row_id!r}")
        if prediction not in {"drop", "keep"}:
            raise ValueError(f"prediction {row_id} is not binary: {prediction!r}")
        predictions[row_id] = {
            "prediction": prediction,
            "prob_drop": float(row.get("prob_drop") or 0.0),
            "prob_keep": float(row.get("prob_keep") or 0.0),
            "truth_label": str(row.get("truth_label") or ""),
        }
    return predictions


def apply_audited_safe_drop_verdicts(
    predictions: dict[str, dict[str, Any]],
    path: Path,
) -> int:
    count = 0
    for row in iter_jsonl(path):
        row_id = str(row.get("row_id") or "")
        verdict = str(row.get("verdict") or "")
        if verdict != "safe_drop":
            raise ValueError(f"display gate only accepts audited safe_drop verdicts: {verdict!r}")
        if row_id not in predictions:
            raise ValueError(f"audited safe_drop row is missing from predictions: {row_id}")
        predictions[row_id]["truth_label"] = "drop"
        predictions[row_id]["audited_safe_drop"] = True
        count += 1
    return count


def load_runtime_groups(path: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_ids: set[str] = set()
    for row in iter_jsonl(path):
        source_id = str(row.get("sample_id") or "")
        row_id = str(row.get("subisland_id") or "")
        if not source_id or not row_id or row_id in row_ids:
            raise ValueError(f"runtime has missing/duplicate identity: {source_id!r}/{row_id!r}")
        row_ids.add(row_id)
        candidate = dict(row.get("pre_asr_candidate") or {})
        candidate_features = dict(candidate.get("features") or {})
        prediction = dict(row.get("inner_edge_prediction") or {})
        start_s = _required_float(row, "start_s")
        end_s = _required_float(row, "end_s")
        if end_s <= start_s:
            raise ValueError(f"runtime row {row_id} has invalid span")
        groups[source_id].append(
            {
                "row_id": row_id,
                "source_id": source_id,
                "start_s": start_s,
                "end_s": end_s,
                "planned_island_id": str(candidate.get("planned_island_id") or ""),
                "position": int(candidate.get("position_in_planned_island") or 0),
                "inner_prediction": prediction,
                "left_edge_is_split_cut": bool(candidate_features.get("left_edge_is_split_cut")),
                "right_edge_is_split_cut": bool(candidate_features.get("right_edge_is_split_cut")),
                "left_edge_is_island_edge": bool(candidate_features.get("left_edge_is_island_edge")),
                "right_edge_is_island_edge": bool(candidate_features.get("right_edge_is_island_edge")),
            }
        )
    for rows in groups.values():
        rows.sort(key=lambda item: (item["start_s"], item["end_s"], item["row_id"]))
    return dict(groups)


def reconstruct_kept_display_chunks(
    runtime_rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    *,
    display_mode: str = "acoustic",
    display_overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> list[dict[str, Any]]:
    if display_mode not in {"acoustic", "inner_all_edges", "model"}:
        raise ValueError(f"unsupported display_mode: {display_mode}")
    if display_mode == "model" and display_overrides is None:
        raise ValueError("display_mode=model requires display_overrides")
    kept: list[dict[str, Any]] = []
    for row in runtime_rows:
        row_id = str(row["row_id"])
        decision = predictions.get(row_id)
        if decision is None:
            raise ValueError(f"runtime row is missing a CueQC prediction: {row_id}")
        if decision["prediction"] != "keep":
            continue
        inner_prediction = dict(row.get("inner_prediction") or {})
        display_start = float(row["start_s"])
        display_end = float(row["end_s"])
        if display_mode == "inner_all_edges":
            if str(inner_prediction.get("start_action") or "") == "refined":
                display_start = float(inner_prediction.get("start_s") or display_start)
            if str(inner_prediction.get("end_action") or "") == "refined":
                display_end = float(inner_prediction.get("end_s") or display_end)
        elif display_mode == "model":
            override = (display_overrides or {}).get(row_id)
            if override is None:
                raise ValueError(f"display override is missing runtime row: {row_id}")
            if bool(override.get("display_drop")):
                continue
            display_start = float(override["display_start_s"])
            display_end = float(override["display_end_s"])
        kept.append(
            {
                **dict(row),
                "display_start_s": display_start,
                "display_end_s": display_end,
                "member_row_ids": [row_id],
                "cueqc_prob_keep": float(decision.get("prob_keep") or 0.0),
                "inner_action": "not_adjacent",
            }
        )

    result: list[dict[str, Any]] = []
    for chunk in kept:
        if not result:
            result.append(chunk)
            continue
        left = result[-1]
        consecutive = (
            left["planned_island_id"]
            and left["planned_island_id"] == chunk["planned_island_id"]
            and int(chunk["position"]) == int(left["position"]) + 1
        )
        if not consecutive:
            result.append(chunk)
            continue
        if display_mode == "model":
            if float(left["display_end_s"]) <= float(chunk["display_start_s"]):
                result.append(chunk)
                continue
            left["start_s"] = min(float(left["start_s"]), float(chunk["start_s"]))
            left["end_s"] = max(float(left["end_s"]), float(chunk["end_s"]))
            left["display_start_s"] = min(
                float(left["display_start_s"]), float(chunk["display_start_s"])
            )
            left["display_end_s"] = max(
                float(left["display_end_s"]), float(chunk["display_end_s"])
            )
            left["position"] = int(chunk["position"])
            left["member_row_ids"].extend(chunk["member_row_ids"])
            left["cueqc_prob_keep"] = min(
                float(left["cueqc_prob_keep"]), float(chunk["cueqc_prob_keep"])
            )
            left["inner_action"] = "model_overlap_merge"
            continue
        left_prediction = dict(left.get("inner_prediction") or {})
        right_prediction = dict(chunk.get("inner_prediction") or {})
        left_action = str(left_prediction.get("end_action") or "abstain")
        right_action = str(right_prediction.get("start_action") or "abstain")
        left_end = float(left_prediction.get("end_s") or left["end_s"])
        right_start = float(right_prediction.get("start_s") or chunk["start_s"])
        if left_action == "refined" and right_action == "refined" and left_end <= right_start:
            left["display_end_s"] = left_end
            left["inner_action"] = "safe"
            chunk["display_start_s"] = right_start
            chunk["inner_action"] = "safe"
            result.append(chunk)
            continue
        left["start_s"] = min(float(left["start_s"]), float(chunk["start_s"]))
        left["end_s"] = max(float(left["end_s"]), float(chunk["end_s"]))
        left["display_start_s"] = min(
            float(left["display_start_s"]), float(chunk["display_start_s"])
        )
        left["display_end_s"] = max(
            float(left["display_end_s"]), float(chunk["display_end_s"])
        )
        left["position"] = int(chunk["position"])
        left["member_row_ids"].extend(chunk["member_row_ids"])
        left["cueqc_prob_keep"] = min(
            float(left["cueqc_prob_keep"]), float(chunk["cueqc_prob_keep"])
        )
        left["inner_action"] = "abstain_merge"
    return result


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def paired_display_chunk_rows(
    *,
    source_id: str,
    source_partition: str,
    cores: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    final_chunks: Sequence[Mapping[str, Any]],
    tolerance_s: float,
) -> tuple[list[dict[str, Any]], set[str]]:
    runtime_by_id = {str(row["row_id"]): row for row in runtime_rows}
    eligible_cores: list[Mapping[str, Any]] = []
    for core in cores:
        core_start, core_end = float(core["start_s"]), float(core["end_s"])
        hits = [
            row for row in runtime_rows
            if min(float(row["end_s"]), core_end) > max(float(row["start_s"]), core_start)
        ]
        if any(
            str(predictions[str(row["row_id"])].get("truth_label") or "") == "keep"
            for row in hits
        ):
            eligible_cores.append(core)
    assigned_core_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    for chunk in final_chunks:
        members = [runtime_by_id[row_id] for row_id in chunk["member_row_ids"]]
        target_spans: list[tuple[str, float, float]] = []
        for core in eligible_cores:
            core_start, core_end = float(core["start_s"]), float(core["end_s"])
            for member in members:
                member_start, member_end = float(member["start_s"]), float(member["end_s"])
                if min(member_end, core_end) <= max(member_start, core_start):
                    continue
                target_start, target_end = core_start, core_end
                if bool(member.get("left_edge_is_split_cut")) and core_start < member_start:
                    target_start = member_start
                if bool(member.get("right_edge_is_split_cut")) and core_end > member_end:
                    target_end = member_end
                target_spans.append((str(core["core_id"]), target_start, target_end))
        target_ids = list(dict.fromkeys(span[0] for span in target_spans))
        assigned_core_ids.update(target_ids)
        if target_spans:
            target_start = min(span[1] for span in target_spans)
            target_end = max(span[2] for span in target_spans)
            start_error = float(chunk["display_start_s"]) - target_start
            end_error = float(chunk["display_end_s"]) - target_end
        else:
            target_start = target_end = start_error = end_error = None
        rows.append({
            "schema": "paired_display_chunk_gate_row_v1",
            "source_id": source_id,
            "source_partition": source_partition,
            "member_row_ids": list(chunk["member_row_ids"]),
            "target_core_ids": target_ids,
            "display_start_s": float(chunk["display_start_s"]),
            "display_end_s": float(chunk["display_end_s"]),
            "target_start_s": target_start,
            "target_end_s": target_end,
            "start_error_s": start_error,
            "end_error_s": end_error,
            "start_within_tolerance": (
                start_error is not None and abs(start_error) <= tolerance_s
            ),
            "end_within_tolerance": (
                end_error is not None and abs(end_error) <= tolerance_s
            ),
            "has_semantic_target": bool(target_spans),
        })
    return rows, {str(core["core_id"]) for core in eligible_cores} - assigned_core_ids


def evaluate_exact_display_gate(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    runtime_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    predictions: Mapping[str, Mapping[str, Any]],
    tolerance_s: float,
    required_coverage: float,
    display_mode: str = "acoustic",
    display_overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if tolerance_s < 0.0:
        raise ValueError("tolerance_s must be non-negative")
    if not 0.0 < required_coverage <= 0.95:
        raise ValueError("required_coverage must be in (0, 0.95]")
    unknown_sources = sorted(set(runtime_groups) - set(sources))
    if unknown_sources:
        raise ValueError(f"runtime contains unknown sources: {unknown_sources[:3]}")

    rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    missing_reason_counts: Counter[str] = Counter()
    excluded_reason_counts: Counter[str] = Counter()
    no_core_chunk_count = 0
    paired_rows: list[dict[str, Any]] = []
    paired_missing_core_ids: set[str] = set()
    for source_id, source in sources.items():
        runtime_rows = list(runtime_groups.get(source_id, ()))
        final_chunks = reconstruct_kept_display_chunks(
            runtime_rows,
            predictions,
            display_mode=display_mode,
            display_overrides=display_overrides,
        )
        cores = list(source["cores"])
        source_paired_rows, source_missing_core_ids = paired_display_chunk_rows(
            source_id=source_id,
            source_partition=str(source.get("source_partition") or ""),
            cores=cores,
            runtime_rows=runtime_rows,
            predictions=predictions,
            final_chunks=final_chunks,
            tolerance_s=tolerance_s,
        )
        paired_rows.extend(source_paired_rows)
        paired_missing_core_ids.update(source_missing_core_ids)
        source_counts["source"] += 1
        source_counts["exact_core"] += len(cores)
        source_counts["final_kept_chunk"] += len(final_chunks)
        for core in cores:
            truth_start = float(core["start_s"])
            truth_end = float(core["end_s"])
            runtime_hits = [
                row
                for row in runtime_rows
                if min(float(row["end_s"]), truth_end)
                > max(float(row["start_s"]), truth_start)
            ]
            observed_truths = {
                str(predictions[str(row["row_id"])].get("truth_label") or "")
                for row in runtime_hits
            }
            if "keep" in observed_truths:
                eligible = True
                exclusion_reason = ""
            elif "unsure" in observed_truths:
                eligible = False
                exclusion_reason = "teacher_unsure"
            elif runtime_hits:
                eligible = False
                exclusion_reason = "teacher_drop"
            else:
                eligible = False
                exclusion_reason = "no_teacher_runtime_overlap"
            if exclusion_reason:
                excluded_reason_counts[exclusion_reason] += 1
            final_hits = [
                chunk
                for chunk in final_chunks
                if min(float(chunk["display_end_s"]), truth_end)
                > max(float(chunk["display_start_s"]), truth_start)
            ]
            if final_hits:
                display_start = min(float(chunk["display_start_s"]) for chunk in final_hits)
                display_end = max(float(chunk["display_end_s"]) for chunk in final_hits)
                start_error = display_start - truth_start
                end_error = display_end - truth_end
                missing_reason = ""
            else:
                display_start = None
                display_end = None
                start_error = None
                end_error = None
                if not runtime_hits:
                    missing_reason = "upstream_no_runtime_overlap"
                elif not any(
                    predictions[str(row["row_id"])]["prediction"] == "keep"
                    for row in runtime_hits
                ):
                    missing_reason = "cueqc_removed_all_overlaps"
                else:
                    missing_reason = "paired_inner_removed_overlap"
                if eligible:
                    missing_reason_counts[missing_reason] += 1
            rows.append(
                {
                    "schema": "display_span_exact_core_gate_row_v2",
                    "source_id": source_id,
                    "source_partition": str(source.get("source_partition") or ""),
                    "audio": str(source.get("audio") or ""),
                    "core_id": str(core["core_id"]),
                    "core_text": str(core.get("text") or ""),
                    "display_start_s": display_start,
                    "display_end_s": display_end,
                    "truth_start_s": truth_start,
                    "truth_end_s": truth_end,
                    "start_error_s": start_error,
                    "end_error_s": end_error,
                    "start_within_tolerance": (
                        start_error is not None and abs(start_error) <= tolerance_s
                    ),
                    "end_within_tolerance": (
                        end_error is not None and abs(end_error) <= tolerance_s
                    ),
                    "runtime_overlap_row_ids": [str(row["row_id"]) for row in runtime_hits],
                    "final_member_row_ids": [
                        row_id
                        for chunk in final_hits
                        for row_id in chunk["member_row_ids"]
                    ],
                    "final_chunk_count": len(final_hits),
                    "fragmented_by_runtime_split": len(final_hits) > 1,
                    "missing": not final_hits,
                    "missing_reason": missing_reason,
                    "teacher_labels_observed": sorted(observed_truths),
                    "eligible_for_display_gate": eligible,
                    "training_ignore_reason": exclusion_reason,
                }
            )

        eligible_cores = [core for core in rows if core["source_id"] == source_id and core["eligible_for_display_gate"]]
        for chunk in final_chunks:
            if not any(
                min(float(chunk["display_end_s"]), float(core["truth_end_s"]))
                > max(float(chunk["display_start_s"]), float(core["truth_start_s"]))
                for core in eligible_cores
            ):
                no_core_chunk_count += 1

    eligible_rows = [row for row in rows if row["eligible_for_display_gate"]]
    missing_rows = [row for row in eligible_rows if row["missing"]]
    covered_rows = [row for row in eligible_rows if not row["missing"]]
    fragmented_rows = [row for row in covered_rows if row["fragmented_by_runtime_split"]]
    start_errors = [abs(float(row["start_error_s"])) for row in covered_rows]
    end_errors = [abs(float(row["end_error_s"])) for row in covered_rows]
    start_pass_count = sum(bool(row["start_within_tolerance"]) for row in eligible_rows)
    end_pass_count = sum(bool(row["end_within_tolerance"]) for row in eligible_rows)
    start_coverage = start_pass_count / len(eligible_rows) if eligible_rows else 0.0
    end_coverage = end_pass_count / len(eligible_rows) if eligible_rows else 0.0
    edge_gate_pass = (
        bool(eligible_rows)
        and start_coverage >= required_coverage
        and end_coverage >= required_coverage
    )
    speech_coverage_gate_pass = not missing_rows
    paired_semantic_rows = [row for row in paired_rows if row["has_semantic_target"]]
    paired_background_rows = [row for row in paired_rows if not row["has_semantic_target"]]
    paired_start_coverage = (
        sum(bool(row["start_within_tolerance"]) for row in paired_semantic_rows)
        / len(paired_semantic_rows) if paired_semantic_rows else 0.0
    )
    paired_end_coverage = (
        sum(bool(row["end_within_tolerance"]) for row in paired_semantic_rows)
        / len(paired_semantic_rows) if paired_semantic_rows else 0.0
    )
    paired_core_assignments: Counter[str] = Counter(
        core_id for row in paired_semantic_rows for core_id in row["target_core_ids"]
    )
    paired_fragmented_core_ids = sorted(
        core_id for core_id, count in paired_core_assignments.items() if count > 1
    )
    paired_chunk_gate_pass = (
        bool(paired_semantic_rows)
        and paired_start_coverage >= required_coverage
        and paired_end_coverage >= required_coverage
        and not paired_missing_core_ids
    )
    summary = {
        "schema": "display_span_exact_core_gate_summary_v2",
        "display_mode": display_mode,
        "tolerance_s": tolerance_s,
        "required_coverage": required_coverage,
        "source_count": source_counts["source"],
        "exact_core_count": source_counts["exact_core"],
        "eligible_semantic_core_count": len(eligible_rows),
        "excluded_core_count": len(rows) - len(eligible_rows),
        "excluded_core_reason_counts": dict(sorted(excluded_reason_counts.items())),
        "final_kept_chunk_count": source_counts["final_kept_chunk"],
        "covered_core_count": len(covered_rows),
        "no_semantic_core_chunk_count": no_core_chunk_count,
        "missing_core_count": len(missing_rows),
        "missing_core_ids": [str(row["core_id"]) for row in missing_rows],
        "missing_reason_counts": dict(sorted(missing_reason_counts.items())),
        "fragmented_core_count": len(fragmented_rows),
        "start_within_tolerance_count": start_pass_count,
        "end_within_tolerance_count": end_pass_count,
        "start_within_tolerance_coverage": start_coverage,
        "end_within_tolerance_coverage": end_coverage,
        "start_abs_error_p50_s": _quantile(start_errors, 0.50),
        "start_abs_error_p95_s": _quantile(start_errors, 0.95),
        "start_abs_error_max_s": max(start_errors, default=0.0),
        "end_abs_error_p50_s": _quantile(end_errors, 0.50),
        "end_abs_error_p95_s": _quantile(end_errors, 0.95),
        "end_abs_error_max_s": max(end_errors, default=0.0),
        "start_early_adjustable_count": sum(
            float(row["start_error_s"]) < -tolerance_s for row in covered_rows
        ),
        "start_late_acoustic_truncation_count": sum(
            float(row["start_error_s"]) > tolerance_s for row in covered_rows
        ),
        "end_early_acoustic_truncation_count": sum(
            float(row["end_error_s"]) < -tolerance_s for row in covered_rows
        ),
        "end_late_adjustable_count": sum(
            float(row["end_error_s"]) > tolerance_s for row in covered_rows
        ),
        "edge_gate_pass": edge_gate_pass,
        "speech_coverage_gate_pass": speech_coverage_gate_pass,
        "gate_pass": edge_gate_pass and speech_coverage_gate_pass,
        "paired_semantic_chunk_count": len(paired_semantic_rows),
        "paired_background_display_chunk_count": len(paired_background_rows),
        "paired_start_within_tolerance_coverage": paired_start_coverage,
        "paired_end_within_tolerance_coverage": paired_end_coverage,
        "paired_missing_core_count": len(paired_missing_core_ids),
        "paired_missing_core_ids": sorted(paired_missing_core_ids),
        "paired_fragmented_core_count": len(paired_fragmented_core_ids),
        "paired_fragmented_core_ids_sample": paired_fragmented_core_ids[:50],
        "paired_chunk_gate_pass": paired_chunk_gate_pass,
        "display_refiner_required": not edge_gate_pass,
        "upstream_repair_required": not speech_coverage_gate_pass,
    }
    return rows, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest = Path(args.source_manifest).resolve()
    runtime_manifest = Path(args.runtime_manifest).resolve()
    predictions_path = Path(args.predictions).resolve()
    audited_safe_drop_verdicts = Path(args.audited_safe_drop_verdicts).resolve()
    output_dir = Path(args.output_dir).resolve()
    sources = load_exact_sources(source_manifest)
    predictions = load_predictions(predictions_path)
    audited_safe_drop_count = apply_audited_safe_drop_verdicts(
        predictions, audited_safe_drop_verdicts
    )
    runtime_groups = load_runtime_groups(runtime_manifest)
    runtime_row_ids = {
        str(row["row_id"])
        for group in runtime_groups.values()
        for row in group
    }
    missing_predictions = sorted(runtime_row_ids - set(predictions))
    extra_predictions = sorted(set(predictions) - runtime_row_ids)
    if missing_predictions or extra_predictions:
        raise ValueError(
            "runtime/prediction identities differ: "
            f"missing={len(missing_predictions)} extra={len(extra_predictions)}"
        )
    rows, summary = evaluate_exact_display_gate(
        sources=sources,
        runtime_groups=runtime_groups,
        predictions=predictions,
        tolerance_s=float(args.tolerance_ms) / 1000.0,
        required_coverage=float(args.required_coverage),
        display_mode=str(args.display_mode),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "display_span_gate_rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary.update(
        {
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": file_sha256(source_manifest),
            "runtime_manifest": str(runtime_manifest),
            "runtime_manifest_sha256": file_sha256(runtime_manifest),
            "predictions": str(predictions_path),
            "predictions_sha256": file_sha256(predictions_path),
            "audited_safe_drop_verdicts": str(audited_safe_drop_verdicts),
            "audited_safe_drop_verdicts_sha256": file_sha256(
                audited_safe_drop_verdicts
            ),
            "audited_safe_drop_count": audited_safe_drop_count,
            "rows": str(rows_path),
        }
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate final paired-edge display spans against sample-exact hardmix cores."
    )
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--runtime-manifest", default=str(DEFAULT_RUNTIME_MANIFEST))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument(
        "--audited-safe-drop-verdicts",
        default=str(DEFAULT_AUDITED_SAFE_DROP_VERDICTS),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tolerance-ms", type=float, default=300.0)
    parser.add_argument("--required-coverage", type=float, default=0.95)
    parser.add_argument(
        "--display-mode",
        choices=("acoustic", "inner_all_edges"),
        default="acoustic",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
