#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]


NUMERIC_FIELDS = (
    "duration_s",
    "raw_duration_s",
    "acoustic_duration_s",
    "refined_duration_s",
    "prev_gap_s",
    "next_gap_s",
    "speech_segment_count",
    "internal_gap_count",
    "internal_gap_max_s",
    "boundary_score",
    "scorer_speech_mean",
    "scorer_speech_p90",
    "scorer_speech_active_ratio_05",
    "scorer_split_mean",
    "scorer_split_max",
    "scorer_split_p90",
    "split_peak_count",
    "split_peak_density",
    "split_peak_top1",
    "split_peak_top1_prominence",
    "primary_cut_count",
    "primary_cut_density",
    "weak_cut_count",
    "weak_cut_density",
    "refiner_confidence_min",
    "refiner_confidence_mean",
    "refiner_start_confidence",
    "refiner_end_confidence",
    "trim_total_s",
    "trim_ratio",
    "core_duration_ratio",
    "micro_chunk_candidate",
    "below_subtitle_min_duration",
    "num_chunks_in_planned_island",
)

PRIMARY_NUMERIC_FIELDS = (
    "duration_s",
    "refined_duration_s",
    "num_chunks_in_planned_island",
    "split_peak_density",
    "primary_cut_density",
    "scorer_split_max",
    "micro_chunk_candidate",
    "below_subtitle_min_duration",
)

CATEGORY_FIELDS = (
    "audio_id",
    "duration_bucket",
    "micro_action",
    "micro_resolve_action",
    "refiner_start_source",
    "refiner_end_source",
    "refiner_safety_action",
    "planned_island_id",
)

PRIMARY_CATEGORY_FIELDS = (
    "duration_bucket",
    "micro_action",
    "micro_resolve_action",
    "refiner_start_source",
    "refiner_end_source",
)

DURATION_BUCKETS = (
    (0.20, "<0.20s"),
    (0.50, "0.20-0.50s"),
    (0.8341666667, "0.50-subtitle_min"),
    (1.50, "subtitle_min-1.50s"),
    (3.00, "1.50-3.00s"),
    (6.00, "3.00-6.00s"),
)


@dataclass
class DatasetSummary:
    name: str
    path: str
    count: int
    numeric: dict[str, list[float]]
    categories: dict[str, Counter[str]]
    audio_spans: dict[str, dict[str, float]]


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def repo_display_path(path: str | Path | None) -> str:
    if not path:
        return ""
    raw = Path(path)
    try:
        return str(raw.resolve().relative_to(PROJECT_ROOT)).replace("/", "\\")
    except ValueError:
        return str(raw)


def iter_json_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        first = ""
        while True:
            char = handle.read(1)
            if not char:
                return
            if not char.isspace():
                first = char
                break
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"JSON payload must be a list: {path}")
            for row in payload:
                if isinstance(row, Mapping):
                    yield dict(row)
            return
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            yield dict(row)


def row_value(row: Mapping[str, Any], key: str) -> Any:
    if key in row:
        return row.get(key)
    features = row.get("features")
    if isinstance(features, Mapping) and key in features:
        return features.get(key)
    return None


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def duration_from_row(row: Mapping[str, Any]) -> float | None:
    for key in ("duration_s", "refined_duration_s", "acoustic_duration_s", "raw_duration_s"):
        value = as_float(row_value(row, key))
        if value is not None:
            return value
    start = as_float(row.get("start"))
    end = as_float(row.get("end"))
    if start is not None and end is not None and end >= start:
        return end - start
    return None


def duration_bucket(duration_s: float | None) -> str:
    if duration_s is None:
        return "missing"
    prev = 0.0
    for limit, label in DURATION_BUCKETS:
        if duration_s < limit:
            return label
        prev = limit
    return f">={prev:.2f}s"


def micro_action(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("micro_resolve_action") or "").strip()
    if explicit:
        return explicit
    candidates = {
        "none": as_float(row_value(row, "micro_action_none")) or 0.0,
        "preserve": as_float(row_value(row, "micro_action_preserve")) or 0.0,
        "merge_left": as_float(row_value(row, "micro_action_merge_left")) or 0.0,
        "merge_right": as_float(row_value(row, "micro_action_merge_right")) or 0.0,
        "unknown": as_float(row_value(row, "micro_action_unknown")) or 0.0,
    }
    name, value = max(candidates.items(), key=lambda item: item[1])
    return name if value > 0.0 else "missing"


def category_value(row: Mapping[str, Any], key: str) -> str:
    if key == "duration_bucket":
        return duration_bucket(duration_from_row(row))
    if key == "micro_action":
        return micro_action(row)
    value = row.get(key)
    if value is None:
        features = row.get("features")
        if isinstance(features, Mapping):
            value = features.get(key)
    text = str(value or "").strip()
    return text if text else "missing"


def summarize_dataset(path: Path, name: str) -> DatasetSummary:
    numeric: dict[str, list[float]] = {field: [] for field in NUMERIC_FIELDS}
    categories: dict[str, Counter[str]] = {field: Counter() for field in CATEGORY_FIELDS}
    audio_spans: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "duration_sum_s": 0.0, "start_min_s": math.inf, "end_max_s": -math.inf}
    )
    count = 0
    for row in iter_json_rows(path):
        count += 1
        for field in NUMERIC_FIELDS:
            value = as_float(row_value(row, field))
            if value is None and field == "duration_s":
                value = duration_from_row(row)
            if value is not None:
                numeric[field].append(value)
        for field in CATEGORY_FIELDS:
            categories[field][category_value(row, field)] += 1
        audio_id = category_value(row, "audio_id")
        duration = duration_from_row(row) or 0.0
        start = as_float(row.get("start"))
        end = as_float(row.get("end"))
        span = audio_spans[audio_id]
        span["count"] += 1.0
        span["duration_sum_s"] += duration
        if start is not None:
            span["start_min_s"] = min(span["start_min_s"], start)
        if end is not None:
            span["end_max_s"] = max(span["end_max_s"], end)
    return DatasetSummary(
        name=name,
        path=repo_display_path(path),
        count=count,
        numeric=numeric,
        categories=categories,
        audio_spans={key: dict(value) for key, value in audio_spans.items()},
    )


def numeric_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(arr, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p10": float(quantiles[2]),
        "p25": float(quantiles[3]),
        "p50": float(quantiles[4]),
        "p75": float(quantiles[5]),
        "p90": float(quantiles[6]),
        "p95": float(quantiles[7]),
        "p99": float(quantiles[8]),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
    }


def ks_distance(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    a = np.sort(np.asarray(left, dtype=np.float64))
    b = np.sort(np.asarray(right, dtype=np.float64))
    values = np.sort(np.unique(np.concatenate([a, b])))
    if values.size == 0:
        return 0.0
    cdf_a = np.searchsorted(a, values, side="right") / float(a.size)
    cdf_b = np.searchsorted(b, values, side="right") / float(b.size)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def rel_delta(new: float | int | None, old: float | int | None) -> float:
    if old is None or new is None:
        return 0.0
    old_f = float(old)
    new_f = float(new)
    denom = max(abs(old_f), 1e-9)
    return (new_f - old_f) / denom


def compare_numeric(
    baseline: DatasetSummary,
    candidate: DatasetSummary,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for field in NUMERIC_FIELDS:
        base_stats = numeric_stats(baseline.numeric.get(field, []))
        cand_stats = numeric_stats(candidate.numeric.get(field, []))
        out[field] = {
            "baseline": base_stats,
            "candidate": cand_stats,
            "ks_distance": ks_distance(baseline.numeric.get(field, []), candidate.numeric.get(field, [])),
            "mean_delta": float(cand_stats.get("mean", 0.0)) - float(base_stats.get("mean", 0.0)),
            "p50_delta": float(cand_stats.get("p50", 0.0)) - float(base_stats.get("p50", 0.0)),
            "p90_delta": float(cand_stats.get("p90", 0.0)) - float(base_stats.get("p90", 0.0)),
            "mean_rel_delta": rel_delta(cand_stats.get("mean"), base_stats.get("mean")),
            "p50_rel_delta": rel_delta(cand_stats.get("p50"), base_stats.get("p50")),
            "p90_rel_delta": rel_delta(cand_stats.get("p90"), base_stats.get("p90")),
        }
    return out


def compare_categories(
    baseline: DatasetSummary,
    candidate: DatasetSummary,
    *,
    top_n: int = 20,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for field in CATEGORY_FIELDS:
        base = baseline.categories.get(field, Counter())
        cand = candidate.categories.get(field, Counter())
        keys = set(base) | set(cand)
        rows = []
        for key in sorted(keys):
            base_count = int(base.get(key, 0))
            cand_count = int(cand.get(key, 0))
            base_prop = base_count / baseline.count if baseline.count else 0.0
            cand_prop = cand_count / candidate.count if candidate.count else 0.0
            rows.append(
                {
                    "value": key,
                    "baseline_count": base_count,
                    "candidate_count": cand_count,
                    "baseline_prop": base_prop,
                    "candidate_prop": cand_prop,
                    "delta_prop": cand_prop - base_prop,
                }
            )
        rows.sort(key=lambda row: abs(float(row["delta_prop"])), reverse=True)
        out[field] = {
            "top_changes": rows[:top_n],
            "max_abs_delta_prop": max((abs(float(row["delta_prop"])) for row in rows), default=0.0),
        }
    return out


def audio_span_summary(summary: DatasetSummary) -> dict[str, Any]:
    per_audio = {}
    for audio_id, span in summary.audio_spans.items():
        start_min = float(span["start_min_s"])
        end_max = float(span["end_max_s"])
        timeline_s = end_max - start_min if math.isfinite(start_min) and math.isfinite(end_max) else 0.0
        per_audio[audio_id] = {
            "chunk_count": int(span["count"]),
            "chunk_duration_sum_s": float(span["duration_sum_s"]),
            "timeline_span_s": timeline_s,
            "chunks_per_minute": (float(span["count"]) / timeline_s * 60.0) if timeline_s > 0 else 0.0,
        }
    return {
        "audio_count": len(per_audio),
        "per_audio": dict(sorted(per_audio.items())),
    }


def material_change_findings(
    *,
    baseline: DatasetSummary,
    candidate: DatasetSummary,
    numeric: Mapping[str, Mapping[str, Any]],
    categories: Mapping[str, Mapping[str, Any]],
    count_threshold: float,
    ks_threshold: float,
    quantile_rel_threshold: float,
    category_pp_threshold: float,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    count_rel = rel_delta(candidate.count, baseline.count)
    if abs(count_rel) >= count_threshold:
        findings.append(
            {
                "kind": "count",
                "field": "candidate_count",
                "severity": abs(count_rel),
                "message": f"candidate count changed by {count_rel * 100.0:.2f}%",
            }
        )
    for field in PRIMARY_NUMERIC_FIELDS:
        row = numeric.get(field) or {}
        ks = float(row.get("ks_distance") or 0.0)
        p50_rel = abs(float(row.get("p50_rel_delta") or 0.0))
        p90_rel = abs(float(row.get("p90_rel_delta") or 0.0))
        if ks >= ks_threshold or p50_rel >= quantile_rel_threshold or p90_rel >= quantile_rel_threshold:
            findings.append(
                {
                    "kind": "numeric",
                    "field": field,
                    "severity": max(ks, p50_rel, p90_rel),
                    "ks_distance": ks,
                    "p50_rel_delta": float(row.get("p50_rel_delta") or 0.0),
                    "p90_rel_delta": float(row.get("p90_rel_delta") or 0.0),
                    "message": (
                        f"{field} changed: KS={ks:.3f}, "
                        f"p50_rel={float(row.get('p50_rel_delta') or 0.0) * 100.0:.2f}%, "
                        f"p90_rel={float(row.get('p90_rel_delta') or 0.0) * 100.0:.2f}%"
                    ),
                }
            )
    for field in PRIMARY_CATEGORY_FIELDS:
        row = categories.get(field) or {}
        max_delta = float(row.get("max_abs_delta_prop") or 0.0)
        if max_delta >= category_pp_threshold:
            findings.append(
                {
                    "kind": "category",
                    "field": field,
                    "severity": max_delta,
                    "message": f"{field} max bucket delta is {max_delta * 100.0:.2f}pp",
                }
            )
    findings.sort(key=lambda row: float(row["severity"]), reverse=True)
    return findings


def render_markdown(payload: Mapping[str, Any]) -> str:
    gate = payload["gate"]
    lines = [
        "# Pre-ASR Candidate Distribution Compare",
        "",
        f"- Baseline: `{payload['baseline']['path']}`",
        f"- Candidate: `{payload['candidate']['path']}`",
        f"- Baseline count: `{payload['baseline']['count']}`",
        f"- Candidate count: `{payload['candidate']['count']}`",
        f"- Gate: `{gate['decision']}`",
        f"- Material findings: `{len(gate['material_findings'])}`",
        "",
    ]
    if gate["material_findings"]:
        lines.extend(["## Material Changes", ""])
        for finding in gate["material_findings"][:20]:
            lines.append(f"- {finding['message']}")
        lines.append("")
    else:
        lines.extend(
            [
                "## Gate Interpretation",
                "",
                "- No material candidate distribution shift was detected under the configured thresholds.",
                "- Treat this as a stop sign before Omni labeling: verify the new scorer/refiner checkpoints were loaded and the workflow cache was rebuilt.",
                "",
            ]
        )

    lines.extend(
        [
            "## Numeric Fields",
            "",
            "| field | base p50 | cand p50 | p50 delta % | base p90 | cand p90 | p90 delta % | KS |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    numeric = payload["numeric"]
    for field in PRIMARY_NUMERIC_FIELDS:
        row = numeric.get(field) or {}
        base = row.get("baseline") or {}
        cand = row.get("candidate") or {}
        lines.append(
            f"| {field} | {float(base.get('p50', 0.0)):.6g} | {float(cand.get('p50', 0.0)):.6g} | "
            f"{float(row.get('p50_rel_delta', 0.0)) * 100.0:.2f} | "
            f"{float(base.get('p90', 0.0)):.6g} | {float(cand.get('p90', 0.0)):.6g} | "
            f"{float(row.get('p90_rel_delta', 0.0)) * 100.0:.2f} | "
            f"{float(row.get('ks_distance', 0.0)):.3f} |"
        )
    lines.extend(["", "## Category Changes", ""])
    categories = payload["categories"]
    for field in PRIMARY_CATEGORY_FIELDS:
        lines.append(f"### {field}")
        lines.append("")
        lines.append("| value | base % | cand % | delta pp | base n | cand n |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in (categories.get(field) or {}).get("top_changes", [])[:12]:
            lines.append(
                f"| {row['value']} | {float(row['baseline_prop']) * 100.0:.2f} | "
                f"{float(row['candidate_prop']) * 100.0:.2f} | "
                f"{float(row['delta_prop']) * 100.0:.2f} | "
                f"{int(row['baseline_count'])} | {int(row['candidate_count'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two Pre-ASR candidate distributions before spending Omni labels."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count-threshold", type=float, default=0.05)
    parser.add_argument("--ks-threshold", type=float, default=0.10)
    parser.add_argument("--quantile-rel-threshold", type=float, default=0.10)
    parser.add_argument("--category-pp-threshold", type=float, default=0.05)
    args = parser.parse_args(argv)

    baseline_path = project_path(args.baseline)
    candidate_path = project_path(args.candidate)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = summarize_dataset(baseline_path, args.baseline_name)
    candidate = summarize_dataset(candidate_path, args.candidate_name)
    numeric = compare_numeric(baseline, candidate)
    categories = compare_categories(baseline, candidate)
    findings = material_change_findings(
        baseline=baseline,
        candidate=candidate,
        numeric=numeric,
        categories=categories,
        count_threshold=float(args.count_threshold),
        ks_threshold=float(args.ks_threshold),
        quantile_rel_threshold=float(args.quantile_rel_threshold),
        category_pp_threshold=float(args.category_pp_threshold),
    )
    decision = "material_shift" if findings else "no_material_shift"
    payload = {
        "schema": "pre_asr_candidate_distribution_compare_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": {
            "name": baseline.name,
            "path": baseline.path,
            "count": baseline.count,
            "audio": audio_span_summary(baseline),
        },
        "candidate": {
            "name": candidate.name,
            "path": candidate.path,
            "count": candidate.count,
            "audio": audio_span_summary(candidate),
        },
        "thresholds": {
            "count_threshold": float(args.count_threshold),
            "ks_threshold": float(args.ks_threshold),
            "quantile_rel_threshold": float(args.quantile_rel_threshold),
            "category_pp_threshold": float(args.category_pp_threshold),
        },
        "gate": {
            "decision": decision,
            "allow_omni_labeling": bool(findings),
            "material_findings": findings,
            "note": (
                "If decision is no_material_shift, stop before Omni labeling and verify the "
                "new scorer/refiner checkpoints were actually loaded and caches rebuilt."
            ),
        },
        "numeric": numeric,
        "categories": categories,
    }
    json_path = output_dir / "distribution_compare.json"
    md_path = output_dir / "distribution_compare.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(payload) + "\n", encoding="utf-8")
    print(f"decision={decision}")
    print(f"allow_omni_labeling={str(bool(findings)).lower()}")
    print(f"summary={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
