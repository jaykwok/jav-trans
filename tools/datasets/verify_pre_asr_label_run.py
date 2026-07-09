#!/usr/bin/env python3
"""Verify closure for a Pre-ASR Omni labeling run."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (PROJECT_ROOT, SRC_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

SUMMARY_SCHEMA = "pre_asr_label_run_verification_v1"


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _iter_candidate_rows(dataset_dir: Path) -> Iterable[dict[str, Any]]:
    for window in read_jsonl(dataset_dir / "source_windows.jsonl"):
        candidate_path = Path(str(window["pre_asr_candidates"]))
        if not candidate_path.is_absolute():
            candidate_path = (PROJECT_ROOT / candidate_path).resolve()
        for row in read_jsonl(candidate_path):
            yield row


def _joint_label_paths(output_dir: Path) -> list[Path]:
    joint_dir = output_dir / "joint_labels"
    if not joint_dir.exists():
        return []
    return sorted(joint_dir.glob("*.json"))


def _raw_response_count(output_dir: Path) -> int:
    raw_dir = output_dir / "raw_responses" / "pre_asr"
    if not raw_dir.exists():
        return 0
    return sum(1 for _path in raw_dir.rglob("*.json"))


def _error_count(output_dir: Path) -> int:
    errors_dir = output_dir / "errors"
    if not errors_dir.exists():
        return 0
    return sum(1 for _path in errors_dir.glob("*.json"))


def verify_run(*, dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    source_windows = read_jsonl(dataset_dir / "source_windows.jsonl")
    expected_candidate_ids = [
        str(row.get("candidate_id") or row.get("sample_id") or row.get("id") or "")
        for row in _iter_candidate_rows(dataset_dir)
    ]
    expected_candidate_set = set(expected_candidate_ids)
    joint_paths = _joint_label_paths(output_dir)
    pre_asr_labels: list[dict[str, Any]] = []
    completed_windows: set[str] = set()
    for path in joint_paths:
        payload = read_json(path)
        completed_windows.add(str(payload.get("window_id") or path.stem))
        rows = payload.get("pre_asr_labels")
        if isinstance(rows, list):
            pre_asr_labels.extend(row for row in rows if isinstance(row, Mapping))
    label_candidate_ids = [
        str(row.get("candidate_id") or row.get("sample_id") or row.get("id") or "")
        for row in pre_asr_labels
    ]
    label_counts = Counter(str(row.get("label") or "") for row in pre_asr_labels)
    fallback_counts = Counter(
        str(row.get("local_fallback") or "") for row in pre_asr_labels if row.get("local_fallback")
    )
    included_count = sum(1 for row in pre_asr_labels if bool(row.get("training_label_included")))
    duplicate_ids = sorted(
        candidate_id for candidate_id, count in Counter(label_candidate_ids).items() if candidate_id and count > 1
    )
    missing_ids = sorted(expected_candidate_set - set(label_candidate_ids))
    unexpected_ids = sorted(set(label_candidate_ids) - expected_candidate_set)
    pre_asr_labels_jsonl = output_dir / "pre_asr_labels.jsonl"
    jsonl_count = len(read_jsonl(pre_asr_labels_jsonl)) if pre_asr_labels_jsonl.exists() else 0
    complete = (
        len(completed_windows) == len(source_windows)
        and len(pre_asr_labels) == len(expected_candidate_ids)
        and not duplicate_ids
        and not missing_ids
        and not unexpected_ids
        and _error_count(output_dir) == 0
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "dataset_dir": repo_rel(dataset_dir),
        "output_dir": repo_rel(output_dir),
        "expected_window_count": len(source_windows),
        "completed_window_count": len(completed_windows),
        "expected_candidate_count": len(expected_candidate_ids),
        "pre_asr_label_count": len(pre_asr_labels),
        "pre_asr_labels_jsonl_count": jsonl_count,
        "raw_response_count": _raw_response_count(output_dir),
        "error_count": _error_count(output_dir),
        "training_label_included_count": included_count,
        "training_label_excluded_count": len(pre_asr_labels) - included_count,
        "label_counts": dict(sorted(label_counts.items())),
        "local_fallback_counts": dict(sorted(fallback_counts.items())),
        "duplicate_candidate_id_count": len(duplicate_ids),
        "missing_candidate_id_count": len(missing_ids),
        "unexpected_candidate_id_count": len(unexpected_ids),
        "duplicate_candidate_ids_sample": duplicate_ids[:20],
        "missing_candidate_ids_sample": missing_ids[:20],
        "unexpected_candidate_ids_sample": unexpected_ids[:20],
        "complete": complete,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    summary = verify_run(
        dataset_dir=project_path(args.dataset_dir),
        output_dir=project_path(args.output_dir),
    )
    if args.output_json:
        write_json(project_path(args.output_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_complete and not bool(summary["complete"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
