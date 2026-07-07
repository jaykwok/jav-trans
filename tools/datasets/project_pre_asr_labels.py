#!/usr/bin/env python3
"""Project old Pre-ASR keep/drop labels onto a re-exported chunk set.

Stage D intentionally changes chunk boundaries (promoted proposer v1 + Split
v2), so candidate ids from the old Omni labels are no longer stable. This tool
uses the stable window-relative span key instead:

1. near-identical boundaries inherit directly;
2. dominant overlap inherits with a projection flag when it does not cross a
   conflicting old label;
3. failed manual-audit regions force every overlapping new chunk to
   ``ambiguous_ignore``.

Unmatched chunks are emitted separately for Omni v2 relabeling.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (PROJECT_ROOT, SRC_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

SCHEMA = "pre_asr_label_projection_v1"
LABEL_SCHEMA = "pre_asr_projected_label_v1"
UNMATCHED_SCHEMA = "pre_asr_projection_unmatched_candidate_v1"
IGNORE_LABEL = "ambiguous_ignore"
_CANDIDATE_KEEP_FIELDS = {
    "candidate_id",
    "sample_id",
    "id",
    "audio_id",
    "video_id",
    "window_id",
    "chunk_index",
    "index",
    "start",
    "end",
    "duration_s",
    "feature_schema",
    "runtime_adapter",
    "schema",
}
_WINDOW_KEEP_FIELDS = {
    "audio_wav",
    "omni_mp3_32k",
    "source_video_id",
    "source_video",
    "source_start_s",
    "source_end_s",
}


@dataclass(frozen=True)
class SpanItem:
    row: dict[str, Any]
    key: str
    window_id: str
    start: float
    end: float
    label: str = ""
    manual: bool = False


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _window_from_candidate_id(value: str) -> str:
    match = re.match(r"^preasr-(.+)-chunk\d+$", value)
    return "" if match is None else match.group(1)


def window_id(row: Mapping[str, Any]) -> str:
    for key in ("window_id", "audio_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    for key in ("candidate_id", "sample_id", "id"):
        parsed = _window_from_candidate_id(str(row.get(key) or "").strip())
        if parsed:
            return parsed
    return str(row.get("video_id") or "").strip()


def row_key(row: Mapping[str, Any]) -> str:
    for key in ("candidate_id", "sample_id", "id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return f"{window_id(row)}#chunk{int(row['chunk_index']):05d}"


def normalize_label(row: Mapping[str, Any]) -> str:
    raw = str(
        row.get("label")
        or row.get("route")
        or row.get("decision")
        or row.get("display_decision")
        or ""
    ).strip().lower()
    if raw in {"keep", "keep_for_asr", "1", "positive", "definite_keep"}:
        return "definite_keep"
    if raw in {"drop", "drop_before_asr", "0", "negative", "definite_drop"}:
        return "definite_drop"
    if raw in {"ignore", "skip", "ambiguous", "unsure", IGNORE_LABEL, "-100"}:
        return IGNORE_LABEL
    raise ValueError(f"unknown Pre-ASR label: {raw!r}")


def is_manual_label(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("override_source") or "") == "manual_audit"
        or str(row.get("label_source") or "").startswith("manual_audit:")
    )


def effective_label_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for path in paths:
        for row in _read_jsonl(path):
            key = row_key(row)
            if key not in rows_by_key:
                order.append(key)
            item = dict(row)
            item["label"] = normalize_label(item)
            item["window_id"] = window_id(item)
            rows_by_key[key] = item
    return [rows_by_key[key] for key in order]


def _float(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def _span_item(row: Mapping[str, Any], *, labeled: bool) -> SpanItem:
    label = normalize_label(row) if labeled else ""
    return SpanItem(
        row=dict(row),
        key=row_key(row),
        window_id=window_id(row),
        start=_float(row, "start"),
        end=_float(row, "end"),
        label=label,
        manual=is_manual_label(row) if labeled else False,
    )


def _overlap(a: SpanItem, b: SpanItem) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def _iou(a: SpanItem, b: SpanItem) -> float:
    overlap = _overlap(a, b)
    if overlap <= 0.0:
        return 0.0
    union = max(a.end, b.end) - min(a.start, b.start)
    return overlap / max(union, 1e-9)


def _display_decision(label: str) -> str:
    if label == "definite_keep":
        return "keep"
    if label == "definite_drop":
        return "drop"
    return IGNORE_LABEL


def _candidate_base(candidate: SpanItem) -> dict[str, Any]:
    row = candidate.row
    candidate_id = str(row.get("candidate_id") or row.get("sample_id") or candidate.key)
    payload = {
        "candidate_id": candidate_id,
        "sample_id": candidate_id,
        "audio_id": str(row.get("audio_id") or candidate.window_id),
        "video_id": str(row.get("video_id") or candidate.window_id),
        "window_id": candidate.window_id,
        "chunk_index": int(row.get("chunk_index", row.get("index", 0))),
        "start": candidate.start,
        "end": candidate.end,
        "duration_s": round(max(0.0, candidate.end - candidate.start), 6),
        "feature_schema": str(row.get("feature_schema") or row.get("schema") or ""),
        "runtime_adapter": str(row.get("runtime_adapter") or ""),
    }
    for key in _WINDOW_KEEP_FIELDS:
        if key in row:
            payload[key] = row[key]
    return payload


def _label_row(
    candidate: SpanItem,
    source: SpanItem,
    *,
    label: str,
    method: str,
    overlap_s: float,
    iou: float,
    boundary_distance_s: float,
) -> dict[str, Any]:
    source_row = source.row
    payload = {
        "schema": LABEL_SCHEMA,
        **_candidate_base(candidate),
        "label": label,
        "display_decision": _display_decision(label),
        "training_label_included": label != IGNORE_LABEL,
        "label_source": f"projected:{source_row.get('label_source') or source_row.get('override_source') or 'old_pre_asr'}",
        "projection": {
            "schema": SCHEMA,
            "method": method,
            "source_candidate_id": source.key,
            "source_window_id": source.window_id,
            "source_start": source.start,
            "source_end": source.end,
            "source_label": source.label,
            "source_label_source": str(source_row.get("label_source") or ""),
            "source_override_source": str(source_row.get("override_source") or ""),
            "source_manual_audit": bool(source.manual),
            "overlap_s": round(overlap_s, 6),
            "iou": round(iou, 6),
            "boundary_distance_s": round(boundary_distance_s, 6),
            "flagged": method == "dominant_overlap",
        },
    }
    if method == "dominant_overlap":
        payload["projection_flag"] = "dominant_overlap"
    return payload


def _unmatched_row(
    candidate: SpanItem,
    *,
    reason: str,
    best_old: SpanItem | None,
    best_iou: float,
) -> dict[str, Any]:
    payload = {
        "schema": UNMATCHED_SCHEMA,
        **_candidate_base(candidate),
        "unmatched_reason": reason,
        "best_iou": round(best_iou, 6),
    }
    if best_old is not None:
        payload["best_source_candidate_id"] = best_old.key
        payload["best_source_label"] = best_old.label
        payload["best_source_start"] = best_old.start
        payload["best_source_end"] = best_old.end
    return payload


def _boundary_distance(candidate: SpanItem, old: SpanItem) -> float:
    return abs(candidate.start - old.start) + abs(candidate.end - old.end)


def _same_label_or_no_overlap(
    candidate: SpanItem,
    chosen: SpanItem,
    overlaps: list[tuple[SpanItem, float, float]],
) -> bool:
    for old, overlap_s, _old_iou in overlaps:
        if old.key == chosen.key or overlap_s <= 0.0:
            continue
        if old.label != chosen.label:
            return False
    return True


def _choose_projection(
    candidate: SpanItem,
    old_rows: list[SpanItem],
    *,
    boundary_tolerance_s: float,
    iou_threshold: float,
) -> tuple[SpanItem | None, str, float, float, str]:
    near = [
        old
        for old in old_rows
        if abs(candidate.start - old.start) <= boundary_tolerance_s
        and abs(candidate.end - old.end) <= boundary_tolerance_s
    ]
    if near:
        labels = {old.label for old in near}
        if len(labels) > 1:
            best = min(near, key=lambda old: _boundary_distance(candidate, old))
            return best, "", _iou(candidate, best), _overlap(candidate, best), "boundary_label_conflict"
        best = min(near, key=lambda old: _boundary_distance(candidate, old))
        return best, "boundary_match", _iou(candidate, best), _overlap(candidate, best), ""

    overlaps = [
        (old, _overlap(candidate, old), _iou(candidate, old))
        for old in old_rows
        if _overlap(candidate, old) > 0.0
    ]
    if not overlaps:
        return None, "", 0.0, 0.0, "no_overlap"
    best, best_overlap, best_iou = max(overlaps, key=lambda item: item[2])
    if best_iou < iou_threshold:
        return best, "", best_iou, best_overlap, "below_iou_threshold"
    if not _same_label_or_no_overlap(candidate, best, overlaps):
        return best, "", best_iou, best_overlap, "cross_boundary_label_conflict"
    return best, "dominant_overlap", best_iou, best_overlap, ""


def project_label_rows(
    old_label_rows: list[dict[str, Any]],
    new_candidate_rows: list[dict[str, Any]],
    *,
    boundary_tolerance_s: float = 0.12,
    iou_threshold: float = 0.60,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    old_items = [_span_item(row, labeled=True) for row in old_label_rows]
    new_items = [_span_item(row, labeled=False) for row in new_candidate_rows]
    old_by_window: dict[str, list[SpanItem]] = {}
    new_by_window: dict[str, list[int]] = {}
    for old in old_items:
        old_by_window.setdefault(old.window_id, []).append(old)
    for index, candidate in enumerate(new_items):
        new_by_window.setdefault(candidate.window_id, []).append(index)

    projected: list[dict[str, Any] | None] = [None] * len(new_items)
    unmatched: list[dict[str, Any] | None] = [None] * len(new_items)
    matched_old_keys: set[str] = set()
    initial_reasons: Counter[str] = Counter()

    for index, candidate in enumerate(new_items):
        old_rows = old_by_window.get(candidate.window_id, [])
        chosen, method, best_iou, best_overlap, reason = _choose_projection(
            candidate,
            old_rows,
            boundary_tolerance_s=boundary_tolerance_s,
            iou_threshold=iou_threshold,
        )
        if method and chosen is not None:
            projected[index] = _label_row(
                candidate,
                chosen,
                label=chosen.label,
                method=method,
                overlap_s=best_overlap,
                iou=best_iou,
                boundary_distance_s=_boundary_distance(candidate, chosen),
            )
            matched_old_keys.add(chosen.key)
            continue
        initial_reasons[reason] += 1
        unmatched[index] = _unmatched_row(
            candidate,
            reason=reason,
            best_old=chosen,
            best_iou=best_iou,
        )

    manual_failed = [
        old
        for old in old_items
        if old.manual
        and old.window_id in new_by_window
        and old.key not in matched_old_keys
    ]
    manual_ignore_indexes: set[int] = set()
    for old in manual_failed:
        for index in new_by_window.get(old.window_id, []):
            candidate = new_items[index]
            overlap_s = _overlap(candidate, old)
            if overlap_s <= 0.0:
                continue
            projected[index] = _label_row(
                candidate,
                old,
                label=IGNORE_LABEL,
                method="manual_projection_failed_ignore",
                overlap_s=overlap_s,
                iou=_iou(candidate, old),
                boundary_distance_s=_boundary_distance(candidate, old),
            )
            projected[index]["label_source"] = (
                f"manual_projection_failed:{old.row.get('label_source') or old.row.get('override_source') or 'manual_audit'}"
            )
            projected[index]["projection"]["flagged"] = True
            unmatched[index] = None
            manual_ignore_indexes.add(index)

    output_labels = [row for row in projected if row is not None]
    output_unmatched = [
        row
        for index, row in enumerate(unmatched)
        if row is not None and projected[index] is None
    ]
    method_counts = Counter(
        str(row["projection"]["method"]) for row in output_labels
    )
    label_counts = Counter(str(row["label"]) for row in output_labels)
    summary = {
        "schema": "pre_asr_label_projection_summary_v1",
        "old_label_count": len(old_items),
        "old_manual_label_count": sum(1 for old in old_items if old.manual),
        "new_candidate_count": len(new_items),
        "projected_label_count": len(output_labels),
        "unmatched_candidate_count": len(output_unmatched),
        "projected_labels": dict(label_counts),
        "projection_methods": dict(method_counts),
        "dominant_overlap_flagged_count": int(method_counts["dominant_overlap"]),
        "manual_projection_failed_old_count": len(manual_failed),
        "manual_projection_failed_ignore_count": len(manual_ignore_indexes),
        "initial_unmatched_reasons": dict(initial_reasons),
        "boundary_tolerance_s": boundary_tolerance_s,
        "iou_threshold": iou_threshold,
    }
    return output_labels, output_unmatched, summary


def _compact_candidate_row(
    candidate: Mapping[str, Any],
    *,
    window: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    item = {
        key: candidate[key]
        for key in _CANDIDATE_KEEP_FIELDS
        if key in candidate
    }
    if window is not None:
        item["window_id"] = str(window["window_id"])
        item["source_video_id"] = str(window.get("video_id") or "")
        for key in (
            "source_video",
            "source_start_s",
            "source_end_s",
            "audio_wav",
            "omni_mp3_32k",
        ):
            if key in window:
                item[key] = window[key]
    return item


def _candidate_rows_from_source_windows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_windows in paths:
        for window in _read_jsonl(source_windows):
            candidate_path = project_path(str(window["pre_asr_candidates"]))
            for candidate in _read_jsonl(candidate_path):
                rows.append(_compact_candidate_row(candidate, window=window))
    return rows


def _candidate_rows_from_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_compact_candidate_row(row) for row in _read_jsonl(path))
    return rows


def run(args: argparse.Namespace) -> None:
    label_paths = [project_path(path) for path in args.old_labels]
    label_paths.extend(project_path(path) for path in (args.old_overrides or []))
    new_rows: list[dict[str, Any]] = []
    if args.new_source_windows:
        new_rows.extend(
            _candidate_rows_from_source_windows(
                project_path(path) for path in args.new_source_windows
            )
        )
    if args.new_candidates:
        new_rows.extend(
            _candidate_rows_from_paths(project_path(path) for path in args.new_candidates)
        )
    if not new_rows:
        raise ValueError("no new candidate rows loaded")

    labels, unmatched, summary = project_label_rows(
        effective_label_rows(label_paths),
        new_rows,
        boundary_tolerance_s=args.boundary_tolerance_s,
        iou_threshold=args.iou_threshold,
    )
    output_dir = project_path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT
        / "agents"
        / "temp"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pre-asr-label-projection"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "pre_asr_projected_labels.jsonl"
    unmatched_path = output_dir / "pre_asr_unmatched_for_omni.jsonl"
    summary_path = output_dir / "summary.json"
    summary.update(
        {
            "old_labels": [str(path) for path in label_paths],
            "new_source_windows": [
                str(project_path(path)) for path in (args.new_source_windows or [])
            ],
            "new_candidates": [
                str(project_path(path)) for path in (args.new_candidates or [])
            ],
            "projected_labels_path": str(labels_path.resolve()),
            "unmatched_for_omni_path": str(unmatched_path.resolve()),
        }
    )
    _write_jsonl(labels_path, labels)
    _write_jsonl(unmatched_path, unmatched)
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project old Pre-ASR labels/overrides onto new chunk candidates."
    )
    parser.add_argument("--old-labels", action="append", required=True)
    parser.add_argument(
        "--old-overrides",
        action="append",
        default=None,
        help="Optional override JSONL files applied after --old-labels.",
    )
    parser.add_argument("--new-source-windows", action="append", default=None)
    parser.add_argument("--new-candidates", action="append", default=None)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--boundary-tolerance-s", type=float, default=0.12)
    parser.add_argument("--iou-threshold", type=float, default=0.60)
    args = parser.parse_args()
    if args.boundary_tolerance_s <= 0.0:
        parser.error("--boundary-tolerance-s must be positive")
    if not 0.0 < args.iou_threshold <= 1.0:
        parser.error("--iou-threshold must be in (0, 1]")
    if not args.new_source_windows and not args.new_candidates:
        parser.error("provide --new-source-windows or --new-candidates")
    return args


if __name__ == "__main__":
    run(parse_args())
