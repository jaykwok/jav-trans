#!/usr/bin/env python3
"""Analyze how Semantic Split threshold choices affect re-exported chunks."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (PROJECT_ROOT, SRC_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

SUMMARY_SCHEMA = "semantic_split_threshold_sensitivity_summary_v1"
ROW_SCHEMA = "semantic_split_threshold_sensitivity_row_v1"


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def repo_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _feature_dirs(reexport_dir: Path) -> list[Path]:
    features_dir = reexport_dir / "features"
    if not features_dir.exists():
        raise FileNotFoundError(f"features directory not found: {features_dir}")
    return sorted(path for path in features_dir.iterdir() if path.is_dir())


def _selected_cuts(
    split_rows: list[Mapping[str, Any]],
    *,
    threshold: float,
    min_chunk_after_split_s: float,
) -> list[dict[str, Any]]:
    accepted_times = {
        round(safe_float(row.get("time_s")), 6)
        for row in split_rows
        if bool(row.get("accepted"))
    }
    eligible = []
    for row in split_rows:
        if str(row.get("label") or "") != "cut":
            continue
        p_cut = safe_float(row.get("p_cut"))
        time_s = safe_float(row.get("time_s"))
        core_start = safe_float(row.get("core_start"))
        core_end = safe_float(row.get("core_end"))
        edge_gap = min(abs(time_s - core_start), abs(time_s - core_end))
        if p_cut < threshold or edge_gap < min_chunk_after_split_s:
            continue
        eligible.append(
            {
                "time_s": time_s,
                "p_cut": p_cut,
                "accepted_current": round(time_s, 6) in accepted_times,
                "core_start": core_start,
                "core_end": core_end,
                "edge_gap_s": edge_gap,
            }
        )
    selected_by_core: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in sorted(eligible, key=lambda item: safe_float(item.get("p_cut")), reverse=True):
        key = (round(safe_float(row.get("core_start")), 6), round(safe_float(row.get("core_end")), 6))
        selected = selected_by_core.setdefault(key, [])
        time_s = safe_float(row.get("time_s"))
        if any(abs(time_s - safe_float(existing.get("time_s"))) < min_chunk_after_split_s for existing in selected):
            continue
        selected.append(row)
    result = [row for selected in selected_by_core.values() for row in selected]
    return sorted(result, key=lambda row: (row["time_s"], -row["p_cut"]))


def _chunk_hit_count(
    chunks: list[Mapping[str, Any]],
    cuts: list[Mapping[str, Any]],
    *,
    min_duration_s: float,
    min_chunk_after_split_s: float,
) -> tuple[int, int, list[dict[str, Any]]]:
    long_chunks = 0
    hit_chunks = 0
    examples: list[dict[str, Any]] = []
    for chunk in chunks:
        start = safe_float(chunk.get("start"))
        end = safe_float(chunk.get("end"), start)
        duration = safe_float(chunk.get("duration_s"), end - start)
        if duration < min_duration_s:
            continue
        long_chunks += 1
        internal = [
            cut
            for cut in cuts
            if start + min_chunk_after_split_s <= safe_float(cut.get("time_s")) <= end - min_chunk_after_split_s
        ]
        if not internal:
            continue
        hit_chunks += 1
        if len(examples) < 50:
            best = max(internal, key=lambda row: safe_float(row.get("p_cut")))
            examples.append(
                {
                    "schema": ROW_SCHEMA,
                    "candidate_id": str(chunk.get("candidate_id") or chunk.get("sample_id") or ""),
                    "window_id": str(chunk.get("audio_id") or chunk.get("window_id") or chunk.get("video_id") or ""),
                    "chunk_index": int(safe_float(chunk.get("chunk_index", chunk.get("index")), 0.0)),
                    "start": round(start, 6),
                    "end": round(end, 6),
                    "duration_s": round(duration, 6),
                    "internal_candidate_count": len(internal),
                    "best_internal_time_s": round(safe_float(best.get("time_s")), 6),
                    "best_internal_p_cut": round(safe_float(best.get("p_cut")), 6),
                    "best_internal_already_accepted": bool(best.get("accepted_current")),
                }
            )
    return long_chunks, hit_chunks, examples


def analyze_thresholds(
    *,
    reexport_dir: Path,
    thresholds: list[float],
    long_chunk_min_s: float,
    min_chunk_after_split_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_candidates = 0
    total_split_candidates = 0
    for threshold in thresholds:
        new_cut_count = 0
        selected_cut_count = 0
        long_chunk_count = 0
        long_chunk_with_internal_cut = 0
        affected_windows: set[str] = set()
        examples: list[dict[str, Any]] = []
        for feature_dir in _feature_dirs(reexport_dir):
            candidate_path = feature_dir / "pre_asr_candidates.jsonl"
            split_path = feature_dir / "semantic_split_features.jsonl"
            if not candidate_path.exists() or not split_path.exists():
                continue
            chunks = read_jsonl(candidate_path)
            split_rows = read_jsonl(split_path)
            if threshold == thresholds[0]:
                total_candidates += len(chunks)
                total_split_candidates += len(split_rows)
            cuts = _selected_cuts(
                split_rows,
                threshold=threshold,
                min_chunk_after_split_s=min_chunk_after_split_s,
            )
            selected_cut_count += len(cuts)
            new_cuts = [cut for cut in cuts if not bool(cut.get("accepted_current"))]
            new_cut_count += len(new_cuts)
            window_long, window_hits, window_examples = _chunk_hit_count(
                chunks,
                new_cuts,
                min_duration_s=long_chunk_min_s,
                min_chunk_after_split_s=min_chunk_after_split_s,
            )
            long_chunk_count += window_long
            long_chunk_with_internal_cut += window_hits
            if window_hits:
                affected_windows.add(feature_dir.name)
            examples.extend(window_examples[: max(0, 50 - len(examples))])
        rows.append(
            {
                "schema": ROW_SCHEMA,
                "threshold": float(threshold),
                "selected_cut_count": selected_cut_count,
                "new_cut_count_vs_current": new_cut_count,
                "long_chunk_min_s": float(long_chunk_min_s),
                "long_chunk_count": long_chunk_count,
                "long_chunk_with_new_internal_cut": long_chunk_with_internal_cut,
                "affected_long_chunk_window_count": len(affected_windows),
                "example_count": len(examples),
                "examples": examples,
            }
        )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "reexport_dir": repo_rel(reexport_dir),
        "thresholds": thresholds,
        "current_threshold_reference": "promoted Split v2 decision_config normal_cut_threshold=0.75",
        "long_chunk_min_s": float(long_chunk_min_s),
        "min_chunk_after_split_s": float(min_chunk_after_split_s),
        "candidate_count": total_candidates,
        "split_candidate_count": total_split_candidates,
        "rows": rows,
        "route_note": (
            "This is read-only sensitivity analysis. Lowering thresholds changes chunk boundaries "
            "and requires explicit route approval plus re-export/relabel closure."
        ),
    }
    return rows, summary


def export_threshold_sensitivity(
    *,
    reexport_dir: Path,
    output_dir: Path,
    thresholds: list[float],
    long_chunk_min_s: float,
    min_chunk_after_split_s: float,
) -> dict[str, Any]:
    rows, summary = analyze_thresholds(
        reexport_dir=reexport_dir,
        thresholds=thresholds,
        long_chunk_min_s=long_chunk_min_s,
        min_chunk_after_split_s=min_chunk_after_split_s,
    )
    rows_path = output_dir / "threshold_sensitivity.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(rows_path, rows)
    summary = {**summary, "manifest": repo_rel(rows_path), "summary": repo_rel(summary_path)}
    write_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reexport-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, action="append", dest="thresholds")
    parser.add_argument("--long-chunk-min-s", type=float, default=15.0)
    parser.add_argument("--min-chunk-after-split-s", type=float, default=1.2)
    args = parser.parse_args(argv)
    thresholds = args.thresholds or [0.70, 0.65, 0.60, 0.55, 0.50]
    if any(threshold < 0.0 or threshold > 1.0 for threshold in thresholds):
        parser.error("--threshold must be in [0, 1]")
    if args.long_chunk_min_s < 0.0:
        parser.error("--long-chunk-min-s must be non-negative")
    if args.min_chunk_after_split_s < 0.0:
        parser.error("--min-chunk-after-split-s must be non-negative")
    summary = export_threshold_sensitivity(
        reexport_dir=project_path(args.reexport_dir),
        output_dir=project_path(args.output_dir),
        thresholds=[float(threshold) for threshold in thresholds],
        long_chunk_min_s=float(args.long_chunk_min_s),
        min_chunk_after_split_s=float(args.min_chunk_after_split_s),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
