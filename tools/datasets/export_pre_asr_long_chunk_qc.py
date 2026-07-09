#!/usr/bin/env python3
"""Export QC manifests for unusually long Pre-ASR chunks."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (PROJECT_ROOT, SRC_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

SUMMARY_SCHEMA = "pre_asr_long_chunk_qc_summary_v1"
ROW_SCHEMA = "pre_asr_long_chunk_qc_row_v1"


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


def _source_windows_by_id(reexport_dir: Path) -> dict[str, dict[str, Any]]:
    path = reexport_dir / "source_windows.jsonl"
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {str(row.get("window_id") or row.get("audio_id") or "").strip(): row for row in rows}


def _split_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "split_metadata": "",
            "split_candidate_count": 0,
            "split_label_cut_count": 0,
            "split_accepted_count": 0,
            "split_accepted_cut_count": 0,
            "split_cut_not_accepted_count": 0,
            "split_cut_ge_0p5_not_accepted_count": 0,
            "split_cut_ge_0p7_not_accepted_count": 0,
            "split_top_cut_time_s": None,
            "split_top_cut_p_cut": None,
            "split_top_cut_accepted": None,
        }
    rows = read_jsonl(path)
    label_cut = [row for row in rows if str(row.get("label") or "") == "cut"]
    accepted = [row for row in rows if bool(row.get("accepted"))]
    accepted_cut = [row for row in label_cut if bool(row.get("accepted"))]
    not_accepted_cut = [row for row in label_cut if not bool(row.get("accepted"))]
    top_cut = max(label_cut, key=lambda row: safe_float(row.get("p_cut")), default=None)
    return {
        "split_metadata": repo_rel(path),
        "split_candidate_count": len(rows),
        "split_label_cut_count": len(label_cut),
        "split_accepted_count": len(accepted),
        "split_accepted_cut_count": len(accepted_cut),
        "split_cut_not_accepted_count": len(not_accepted_cut),
        "split_cut_ge_0p5_not_accepted_count": sum(
            1 for row in not_accepted_cut if safe_float(row.get("p_cut")) >= 0.5
        ),
        "split_cut_ge_0p7_not_accepted_count": sum(
            1 for row in not_accepted_cut if safe_float(row.get("p_cut")) >= 0.7
        ),
        "split_top_cut_time_s": None if top_cut is None else round(safe_float(top_cut.get("time_s")), 6),
        "split_top_cut_p_cut": None if top_cut is None else round(safe_float(top_cut.get("p_cut")), 6),
        "split_top_cut_accepted": None if top_cut is None else bool(top_cut.get("accepted")),
    }


def collect_long_chunks(
    *,
    reexport_dir: Path,
    min_duration_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    windows = _source_windows_by_id(reexport_dir)
    rows: list[dict[str, Any]] = []
    all_durations: list[float] = []
    total_candidates = 0
    per_window: Counter[str] = Counter()
    for feature_dir in _feature_dirs(reexport_dir):
        candidate_path = feature_dir / "pre_asr_candidates.jsonl"
        if not candidate_path.exists():
            continue
        split_info = _split_summary(feature_dir / "semantic_split_features.jsonl")
        for item in read_jsonl(candidate_path):
            duration = safe_float(item.get("duration_s"), safe_float(item.get("end")) - safe_float(item.get("start")))
            total_candidates += 1
            all_durations.append(duration)
            if duration < min_duration_s:
                continue
            candidate_id = str(item.get("candidate_id") or item.get("sample_id") or item.get("id") or "").strip()
            window_id = str(item.get("audio_id") or item.get("window_id") or item.get("video_id") or "").strip()
            source = windows.get(window_id, {})
            start = safe_float(item.get("start"))
            end = safe_float(item.get("end"), start + duration)
            source_start = safe_float(source.get("source_start_s"))
            row = {
                "schema": ROW_SCHEMA,
                "candidate_id": candidate_id,
                "window_id": window_id,
                "video_id": str(source.get("video_id") or item.get("video_id") or ""),
                "chunk_index": int(safe_float(item.get("chunk_index", item.get("index")), 0.0)),
                "start": round(start, 6),
                "end": round(end, 6),
                "duration_s": round(duration, 6),
                "source_start_s": round(source_start, 6),
                "source_end_s": round(safe_float(source.get("source_end_s")), 6),
                "source_video_time_start_s": round(source_start + start, 6),
                "source_video_time_end_s": round(source_start + end, 6),
                "source_video": str(source.get("source_video") or ""),
                "audio_wav": str(source.get("audio_wav") or ""),
                "request_audio_hint": "",
                "candidate_manifest": repo_rel(candidate_path),
                **split_info,
            }
            rows.append(row)
            per_window[window_id] += 1
    rows.sort(key=lambda row: (-safe_float(row.get("duration_s")), str(row.get("window_id")), int(row.get("chunk_index") or 0)))
    durations = sorted(all_durations)
    thresholds = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "reexport_dir": repo_rel(reexport_dir),
        "min_duration_s": float(min_duration_s),
        "candidate_count": total_candidates,
        "long_chunk_count": len(rows),
        "long_window_count": len(per_window),
        "duration_max_s": round(max(durations), 6) if durations else 0.0,
        "duration_p50_s": _percentile(durations, 0.50),
        "duration_p90_s": _percentile(durations, 0.90),
        "duration_p95_s": _percentile(durations, 0.95),
        "duration_p99_s": _percentile(durations, 0.99),
        "threshold_counts": {f"gt_{int(threshold)}s": sum(1 for value in durations if value > threshold) for threshold in thresholds},
        "top_candidate_id": rows[0]["candidate_id"] if rows else "",
        "top_duration_s": rows[0]["duration_s"] if rows else 0.0,
        "review_required": bool(rows),
        "route_note": (
            "Long Pre-ASR chunks are split operating-point evidence. Review before CueQC training; "
            "do not change split thresholds without an explicit route decision."
        ),
    }
    return rows, summary


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(len(sorted_values) * q))
    return round(sorted_values[index], 6)


def export_long_chunk_qc(*, reexport_dir: Path, output_dir: Path, min_duration_s: float) -> dict[str, Any]:
    rows, summary = collect_long_chunks(reexport_dir=reexport_dir, min_duration_s=min_duration_s)
    manifest_path = output_dir / "long_pre_asr_chunks.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(manifest_path, rows)
    summary = {
        **summary,
        "manifest": repo_rel(manifest_path),
        "summary": repo_rel(summary_path),
    }
    write_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reexport-dir", required=True, help="Stage D re-export directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for long chunk QC outputs.")
    parser.add_argument("--min-duration-s", type=float, default=15.0)
    args = parser.parse_args(argv)
    if args.min_duration_s < 0.0:
        parser.error("--min-duration-s must be non-negative")
    summary = export_long_chunk_qc(
        reexport_dir=project_path(args.reexport_dir),
        output_dir=project_path(args.output_dir),
        min_duration_s=float(args.min_duration_s),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
