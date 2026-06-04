#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.asr.diagnostics.diagnose_asr_alignment import diagnose_case  # noqa: E402


DEFAULT_BOUNDARY_MANIFEST = (
    "datasets/test/speech-boundary-ja/v1-11/"
    "galgame-synthetic-timeline-v5-long-gap-test64/boundary_manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = "agents/temp/speech-boundary-ja/fallback-timing-error"
DEFAULT_REPORT_TITLE = "Fallback Timing Error"
TARGET_QUALITIES = ("forced", "vad_coarse", "proportional")


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def project_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def discover_aligned_jsons(workflow_root: Path | None, explicit: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(explicit)
    if workflow_root is not None:
        candidates.extend(sorted((workflow_root / "archived").glob("*/*.aligned_segments.json")))
        candidates.extend(sorted(workflow_root.glob("*.aligned_segments.json")))
    paths: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        paths.append(resolved)
    return paths


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def interval_gap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    if interval_overlap(a_start, a_end, b_start, b_end) > 0:
        return 0.0
    if a_end <= b_start:
        return b_start - a_end
    return a_start - b_end


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def clean_quality(value: Any) -> str:
    quality = str(value or "").strip()
    return quality if quality in TARGET_QUALITIES else quality or "unknown"


def normalize_truth_segments(row: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for index, item in enumerate(row.get("actual_speech_segments") or []):
        if not isinstance(item, dict):
            continue
        start = _float(item.get("start"))
        end = _float(item.get("end"), start)
        if end <= start:
            continue
        segments.append(
            {
                "truth_index": index,
                "start": start,
                "end": end,
                "duration_s": end - start,
            }
        )
    return segments


def boundary_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        keys = {
            str(row.get("audio_id") or ""),
            Path(str(row.get("audio") or "")).stem,
        }
        for key in keys:
            if key:
                indexed[key] = row
    return indexed


def match_truth(
    *,
    start: float,
    end: float,
    truth_segments: list[dict[str, Any]],
    max_gap_s: float,
) -> tuple[dict[str, Any] | None, float, float]:
    best_truth: dict[str, Any] | None = None
    best_overlap = -1.0
    best_gap = float("inf")
    for truth in truth_segments:
        overlap = interval_overlap(start, end, truth["start"], truth["end"])
        gap = interval_gap(start, end, truth["start"], truth["end"])
        if overlap > best_overlap or (overlap == best_overlap and gap < best_gap):
            best_truth = truth
            best_overlap = overlap
            best_gap = gap
    if best_truth is None:
        return None, 0.0, float("inf")
    if best_overlap <= 0.0 and best_gap > max_gap_s:
        return None, best_overlap, best_gap
    return best_truth, max(0.0, best_overlap), best_gap


def error_row(
    *,
    unit: str,
    audio_id: str,
    aligned_path: Path,
    quality: str,
    fallback_type: str,
    fallback_subtype: str,
    pred_start: float,
    pred_end: float,
    truth: dict[str, Any],
    overlap_s: float,
    gap_s: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start_error = pred_start - truth["start"]
    end_error = pred_end - truth["end"]
    duration_error = (pred_end - pred_start) - truth["duration_s"]
    row = {
        "unit": unit,
        "audio_id": audio_id,
        "aligned_path": project_rel(aligned_path),
        "truth_index": truth["truth_index"],
        "quality": quality,
        "fallback_type": fallback_type,
        "fallback_subtype": fallback_subtype,
        "pred_start": round(pred_start, 6),
        "pred_end": round(pred_end, 6),
        "truth_start": round(truth["start"], 6),
        "truth_end": round(truth["end"], 6),
        "start_error_s": round(start_error, 6),
        "end_error_s": round(end_error, 6),
        "duration_error_s": round(duration_error, 6),
        "abs_start_error_s": round(abs(start_error), 6),
        "abs_end_error_s": round(abs(end_error), 6),
        "max_boundary_abs_error_s": round(max(abs(start_error), abs(end_error)), 6),
        "overlap_s": round(overlap_s, 6),
        "gap_s": round(gap_s, 6),
    }
    if extra:
        row.update(extra)
    return row


def load_case_predictions(
    *,
    aligned_path: Path,
    workflow_root: Path | None,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    aligned = read_json(aligned_path)
    if not isinstance(aligned, dict):
        raise ValueError(f"Expected object JSON: {aligned_path}")
    segments = [item for item in aligned.get("segments") or [] if isinstance(item, dict)]
    diagnostics, _summary = diagnose_case(aligned_path=aligned_path, workflow_root=workflow_root)
    by_chunk: dict[int, dict[str, Any]] = {}
    for row in diagnostics:
        try:
            by_chunk[int(row.get("chunk_index"))] = row
        except (TypeError, ValueError):
            continue
    predictions: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        start = _float(segment.get("start"))
        end = _float(segment.get("end"), start)
        if end <= start:
            continue
        chunk_index = None
        try:
            chunk_index = int(segment.get("source_chunk_index"))
        except (TypeError, ValueError):
            pass
        diagnostic = by_chunk.get(chunk_index) if chunk_index is not None else None
        quality = clean_quality((diagnostic or {}).get("alignment_quality"))
        predictions.append(
            {
                "segment_index": index,
                "start": start,
                "end": end,
                "duration_s": end - start,
                "text": str(segment.get("text") or ""),
                "source_chunk_index": chunk_index,
                "quality": quality,
                "fallback_type": str((diagnostic or {}).get("fallback_type") or "unknown"),
                "fallback_subtype": str((diagnostic or {}).get("fallback_subtype") or "unknown"),
            }
        )
    return predictions, by_chunk


def measure_case(
    *,
    aligned_path: Path,
    workflow_root: Path | None,
    truth_row: dict[str, Any],
    max_gap_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audio_id = str(truth_row.get("audio_id") or aligned_path.name.removesuffix(".aligned_segments.json"))
    truth_segments = normalize_truth_segments(truth_row)
    predictions, diagnostics_by_chunk = load_case_predictions(
        aligned_path=aligned_path,
        workflow_root=workflow_root,
    )
    measurements: list[dict[str, Any]] = []
    unmatched_predictions = 0
    ignored_predictions = 0

    for prediction in predictions:
        quality = prediction["quality"]
        if quality not in TARGET_QUALITIES:
            ignored_predictions += 1
            continue
        truth, overlap_s, gap_s = match_truth(
            start=prediction["start"],
            end=prediction["end"],
            truth_segments=truth_segments,
            max_gap_s=max_gap_s,
        )
        if truth is None:
            unmatched_predictions += 1
            continue
        measurements.append(
            error_row(
                unit="output_segment",
                audio_id=audio_id,
                aligned_path=aligned_path,
                quality=quality,
                fallback_type=prediction["fallback_type"],
                fallback_subtype=prediction["fallback_subtype"],
                pred_start=prediction["start"],
                pred_end=prediction["end"],
                truth=truth,
                overlap_s=overlap_s,
                gap_s=gap_s,
                extra={
                    "segment_index": prediction["segment_index"],
                    "source_chunk_index": prediction["source_chunk_index"],
                    "pred_duration_s": round(prediction["duration_s"], 6),
                    "text": prediction["text"],
                },
            )
        )

    cue_count = 0
    for truth in truth_segments:
        for quality in TARGET_QUALITIES:
            members = [
                prediction
                for prediction in predictions
                if prediction["quality"] == quality
                and interval_gap(prediction["start"], prediction["end"], truth["start"], truth["end"]) <= max_gap_s
                and (
                    interval_overlap(prediction["start"], prediction["end"], truth["start"], truth["end"]) > 0
                    or interval_gap(prediction["start"], prediction["end"], truth["start"], truth["end"]) <= max_gap_s
                )
            ]
            if not members:
                continue
            pred_start = min(item["start"] for item in members)
            pred_end = max(item["end"] for item in members)
            overlap_s = interval_overlap(pred_start, pred_end, truth["start"], truth["end"])
            gap_s = interval_gap(pred_start, pred_end, truth["start"], truth["end"])
            fallback_types = sorted({str(item["fallback_type"]) for item in members if item.get("fallback_type")})
            fallback_subtypes = sorted(
                {str(item["fallback_subtype"]) for item in members if item.get("fallback_subtype")}
            )
            measurements.append(
                error_row(
                    unit="cue",
                    audio_id=audio_id,
                    aligned_path=aligned_path,
                    quality=quality,
                    fallback_type=",".join(fallback_types) if fallback_types else "unknown",
                    fallback_subtype=",".join(fallback_subtypes) if fallback_subtypes else "unknown",
                    pred_start=pred_start,
                    pred_end=pred_end,
                    truth=truth,
                    overlap_s=overlap_s,
                    gap_s=gap_s,
                    extra={
                        "member_count": len(members),
                        "member_segment_indices": [item["segment_index"] for item in members],
                        "member_chunk_indices": sorted(
                            {
                                item["source_chunk_index"]
                                for item in members
                                if item.get("source_chunk_index") is not None
                            }
                        ),
                    },
                )
            )
            cue_count += 1

    case_summary = {
        "audio_id": audio_id,
        "aligned_path": project_rel(aligned_path),
        "truth_segment_count": len(truth_segments),
        "prediction_count": len(predictions),
        "diagnostic_chunk_count": len(diagnostics_by_chunk),
        "measured_count": len(measurements),
        "cue_measurement_count": cue_count,
        "unmatched_prediction_count": unmatched_predictions,
        "ignored_prediction_count": ignored_predictions,
        "prediction_quality_counts": dict(Counter(item["quality"] for item in predictions).most_common()),
    }
    return measurements, case_summary


def aggregate_measurements(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("unit")), str(row.get("quality")))].append(row)

    aggregates: list[dict[str, Any]] = []
    for (unit, quality), items in sorted(grouped.items()):
        start_abs = [float(item["abs_start_error_s"]) for item in items]
        end_abs = [float(item["abs_end_error_s"]) for item in items]
        max_abs = [float(item["max_boundary_abs_error_s"]) for item in items]
        start_signed = [float(item["start_error_s"]) for item in items]
        end_signed = [float(item["end_error_s"]) for item in items]
        aggregates.append(
            {
                "unit": unit,
                "quality": quality,
                "count": len(items),
                "p50_abs_start_error_s": round_or_none(percentile(start_abs, 0.50)),
                "p90_abs_start_error_s": round_or_none(percentile(start_abs, 0.90)),
                "p50_abs_end_error_s": round_or_none(percentile(end_abs, 0.50)),
                "p90_abs_end_error_s": round_or_none(percentile(end_abs, 0.90)),
                "p50_max_boundary_abs_error_s": round_or_none(percentile(max_abs, 0.50)),
                "p90_max_boundary_abs_error_s": round_or_none(percentile(max_abs, 0.90)),
                "median_signed_start_error_s": round_or_none(percentile(start_signed, 0.50)),
                "median_signed_end_error_s": round_or_none(percentile(end_signed, 0.50)),
            }
        )
    return aggregates


def decide_gate(aggregates: list[dict[str, Any]], *, gate_unit: str) -> dict[str, Any]:
    by_key = {(item["unit"], item["quality"]): item for item in aggregates}
    forced = by_key.get((gate_unit, "forced"))
    fallback_items = [
        item
        for quality in ("vad_coarse", "proportional")
        if (item := by_key.get((gate_unit, quality))) is not None
    ]
    metric = "p90_max_boundary_abs_error_s"
    if not fallback_items:
        return {
            "status": "INSUFFICIENT_DATA",
            "basis": gate_unit,
            "metric": metric,
            "reason": f"no vad_coarse/proportional {gate_unit} measurements",
        }
    if forced is None or forced.get(metric) is None:
        max_abs = max(float(item.get(metric) or 0.0) for item in fallback_items)
        if max_abs <= 0.150:
            status = "PASS_RECLASSIFICATION_CLEANUP"
        elif max_abs > 0.250:
            status = "FAIL_PHASE_1_2"
        else:
            status = "GRAY_PHASE_1_RETEST"
        return {
            "status": status,
            "basis": f"{gate_unit}_absolute_without_forced",
            "metric": metric,
            "fallback_p90_max_s": round(max_abs, 6),
            "reason": "forced bucket unavailable; compared fallback absolute p90 to R14 thresholds",
        }

    forced_p90 = float(forced.get(metric) or 0.0)
    deltas = {
        str(item["quality"]): round(float(item.get(metric) or 0.0) - forced_p90, 6)
        for item in fallback_items
    }
    max_delta = max(deltas.values())
    if max_delta <= 0.150:
        status = "PASS_RECLASSIFICATION_CLEANUP"
    elif max_delta > 0.250:
        status = "FAIL_PHASE_1_2"
    else:
        status = "GRAY_PHASE_1_RETEST"
    return {
        "status": status,
        "basis": f"{gate_unit}_delta_vs_forced",
        "metric": metric,
        "forced_p90_s": round(forced_p90, 6),
        "fallback_delta_p90_s": deltas,
        "max_delta_p90_s": round(max_delta, 6),
    }


def _fmt_s(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 1000:.1f}ms"
    except (TypeError, ValueError):
        return "-"


def build_markdown(summary: dict[str, Any]) -> str:
    gate = summary["gate"]
    report_title = str(summary.get("report_title") or DEFAULT_REPORT_TITLE)
    lines = [
        f"# {report_title}",
        "",
        "## Gate",
        "",
        f"- status: `{gate['status']}`",
        f"- basis: `{gate.get('basis', '')}`",
        f"- metric: `{gate.get('metric', '')}`",
    ]
    if "forced_p90_s" in gate:
        lines.append(f"- forced p90: `{_fmt_s(gate['forced_p90_s'])}`")
    if "fallback_p90_max_s" in gate:
        lines.append(f"- fallback p90 max: `{_fmt_s(gate['fallback_p90_max_s'])}`")
    if "max_delta_p90_s" in gate:
        lines.append(f"- max fallback-forced p90 delta: `{_fmt_s(gate['max_delta_p90_s'])}`")
    if gate.get("fallback_delta_p90_s"):
        for quality, delta in gate["fallback_delta_p90_s"].items():
            lines.append(f"- `{quality}` p90 delta: `{_fmt_s(delta)}`")
    if gate.get("reason"):
        lines.append(f"- reason: {gate['reason']}")
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- boundary manifest: `{summary['boundary_manifest']}`",
            f"- workflow root: `{summary.get('workflow_root') or ''}`",
            f"- aligned files: `{summary['aligned_file_count']}`",
            f"- cases measured: `{summary['case_count']}`",
            f"- measurements: `{summary['measurement_count']}`",
            "",
            "## Timing Error",
            "",
            "| unit | quality | count | p50 start | p90 start | p50 end | p90 end | p50 max | p90 max |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["aggregates"]:
        lines.append(
            "| `{unit}` | `{quality}` | {count} | {p50s} | {p90s} | {p50e} | {p90e} | {p50m} | {p90m} |".format(
                unit=row["unit"],
                quality=row["quality"],
                count=row["count"],
                p50s=_fmt_s(row.get("p50_abs_start_error_s")),
                p90s=_fmt_s(row.get("p90_abs_start_error_s")),
                p50e=_fmt_s(row.get("p50_abs_end_error_s")),
                p90e=_fmt_s(row.get("p90_abs_end_error_s")),
                p50m=_fmt_s(row.get("p50_max_boundary_abs_error_s")),
                p90m=_fmt_s(row.get("p90_max_boundary_abs_error_s")),
            )
        )
    lines.extend(["", "## Case Notes", ""])
    for case in summary["cases"]:
        lines.append(
            "- `{audio_id}`: truth={truth}, predictions={pred}, measured={measured}, "
            "cue_measurements={cues}, unmatched={unmatched}, ignored={ignored}, qualities={qualities}".format(
                audio_id=case["audio_id"],
                truth=case["truth_segment_count"],
                pred=case["prediction_count"],
                measured=case["measured_count"],
                cues=case["cue_measurement_count"],
                unmatched=case["unmatched_prediction_count"],
                ignored=case["ignored_prediction_count"],
                qualities=case["prediction_quality_counts"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_measurement(
    *,
    boundary_manifest: Path,
    workflow_root: Path | None,
    aligned_jsons: list[Path],
    output_dir: Path,
    max_gap_s: float,
    gate_unit: str,
    report_title: str = DEFAULT_REPORT_TITLE,
) -> dict[str, Any]:
    boundary_rows = read_jsonl(boundary_manifest)
    indexed = boundary_index(boundary_rows)
    aligned_paths = discover_aligned_jsons(workflow_root, aligned_jsons)
    if not aligned_paths:
        raise SystemExit("No aligned_segments JSON files found.")

    all_measurements: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    missing_truth: list[str] = []
    for aligned_path in aligned_paths:
        audio_id = aligned_path.name.removesuffix(".aligned_segments.json")
        truth_row = indexed.get(audio_id) or indexed.get(aligned_path.parent.name)
        if truth_row is None:
            missing_truth.append(project_rel(aligned_path))
            continue
        measurements, case_summary = measure_case(
            aligned_path=aligned_path,
            workflow_root=workflow_root,
            truth_row=truth_row,
            max_gap_s=max_gap_s,
        )
        all_measurements.extend(measurements)
        case_summaries.append(case_summary)

    aggregates = aggregate_measurements(all_measurements)
    gate = decide_gate(aggregates, gate_unit=gate_unit)
    summary = {
        "report_title": report_title,
        "boundary_manifest": project_rel(boundary_manifest),
        "workflow_root": project_rel(workflow_root) if workflow_root else "",
        "aligned_file_count": len(aligned_paths),
        "case_count": len(case_summaries),
        "measurement_count": len(all_measurements),
        "max_gap_s": max_gap_s,
        "gate_unit": gate_unit,
        "missing_truth": missing_truth,
        "aggregates": aggregates,
        "gate": gate,
        "cases": case_summaries,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "measurements.jsonl", all_measurements)
    (output_dir / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure forced-alignment fallback timing error against synthetic speech-island truth.",
    )
    parser.add_argument("--boundary-manifest", default=DEFAULT_BOUNDARY_MANIFEST)
    parser.add_argument("--workflow-root", default="")
    parser.add_argument("--aligned-json", action="append", default=[])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-gap-s",
        type=float,
        default=0.50,
        help="Maximum non-overlap gap allowed when matching a prediction to a truth island.",
    )
    parser.add_argument(
        "--gate-unit",
        choices=("cue", "output_segment"),
        default="cue",
        help=(
            "Measurement unit used for the R14 p90 delta gate. `cue` is the synthetic "
            "actual_speech_segments envelope; `output_segment` is each emitted subtitle segment."
        ),
    )
    parser.add_argument(
        "--report-title",
        default=DEFAULT_REPORT_TITLE,
        help="Markdown report title.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workflow_root = project_path(args.workflow_root) if args.workflow_root else None
    aligned_jsons = [project_path(path) for path in args.aligned_json]
    summary = run_measurement(
        boundary_manifest=project_path(args.boundary_manifest),
        workflow_root=workflow_root,
        aligned_jsons=aligned_jsons,
        output_dir=project_path(args.output_dir),
        max_gap_s=args.max_gap_s,
        gate_unit=args.gate_unit,
        report_title=args.report_title,
    )
    print(
        "measured cases={cases} measurements={measurements} gate={gate} output={output}".format(
            cases=summary["case_count"],
            measurements=summary["measurement_count"],
            gate=summary["gate"]["status"],
            output=project_rel(project_path(args.output_dir)),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
