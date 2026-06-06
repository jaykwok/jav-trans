#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(dict(payload))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def export_manifest(
    *,
    review_items_path: Path,
    source_audio_path: Path,
    output_dir: Path,
    dataset_id: str,
    max_rows: int | None,
) -> dict[str, Any]:
    items = read_jsonl(review_items_path)
    if max_rows is not None:
        items = items[:max_rows]
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        sample_id = str(item.get("sample_id") or f"{dataset_id}-{index:02d}")
        if sample_id == f"{dataset_id}-{index:02d}":
            sample_id = f"{dataset_id}-{index:02d}-{int(_as_float(item.get('timeline_start_s')) * 1000):08d}"
        left = str(item.get("left_ja") or "").strip()
        right = str(item.get("right_ja") or "").strip()
        merged = str(item.get("merged_ja") or "").strip()
        risk = str(item.get("risk_tags") or "").strip() or "no_risk_tag"
        display = (
            f"{dataset_id} #{index} / priority {item.get('review_priority', '')}\n"
            f"risk: {risk}\n\n"
            f"left: {left}\n"
            f"right: {right}\n\n"
            f"merged: {merged}"
        )
        rows.append(
            {
                "sample_id": sample_id,
                "review_type": "cue_planner_merge_extra",
                "failure_bucket": risk,
                "source_audio_path": project_rel(source_audio_path),
                "start": round(_as_float(item.get("timeline_start_s")), 3),
                "end": round(_as_float(item.get("timeline_end_s")), 3),
                "duration_s": round(
                    max(0.0, _as_float(item.get("timeline_end_s")) - _as_float(item.get("timeline_start_s"))),
                    3,
                ),
                "chunk_index": int(_as_float(item.get("left_index"))),
                "position": index,
                "display_text": display,
                "align_text": merged,
                "raw_text": merged,
                "left_ja": left,
                "right_ja": right,
                "merged_ja": merged,
                "audit_reason": f"{dataset_id} extra merge versus baseline; review readability/fallback/timing risk",
                "case_label": f"{item.get('timeline_start', '')} - {item.get('timeline_end', '')}",
                "score": item.get("score"),
                "risk_tags": risk,
                "manual_label": "",
                "manual_text": "",
                "notes": "",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "cue_planner_audio_audit_manifest.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(manifest_path, rows)
    summary = {
        "source_review_items": project_rel(review_items_path),
        "source_audio": project_rel(source_audio_path),
        "dataset_id": dataset_id,
        "rows": len(rows),
        "manifest_jsonl": project_rel(manifest_path),
    }
    write_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert cue-planner review items to an audio audit manifest.")
    parser.add_argument("--review-items", required=True, help="merge review JSONL")
    parser.add_argument("--source-audio", required=True, help="source full audio path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args(argv)
    summary = export_manifest(
        review_items_path=project_path(args.review_items),
        source_audio_path=project_path(args.source_audio),
        output_dir=project_path(args.output_dir),
        dataset_id=args.dataset_id,
        max_rows=args.max_rows,
    )
    print(f"manifest={summary['manifest_jsonl']} rows={summary['rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
