#!/usr/bin/env python3
"""Build a stable, video-isolated Pre-ASR manual audit corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.datasets.project_pre_asr_labels import (  # noqa: E402
    _candidate_rows_from_source_windows,
    project_label_rows,
    project_path,
)

CORPUS_SCHEMA = "pre_asr_manual_audit_corpus_v1"
VERDICT_TO_LABEL = {
    "drop": "definite_drop",
    "keep": "definite_keep",
    "unsure": "ambiguous_ignore",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(dict(row))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _write_ids(path: Path, values: Iterable[str]) -> None:
    path.write_text(
        "".join(f"{value}\n" for value in sorted(set(values)) if value),
        encoding="utf-8",
    )


def _span_key(row: Mapping[str, Any]) -> tuple[str, float, float]:
    window_id = str(row.get("window_id") or row.get("audio_id") or "").strip()
    if not window_id:
        raise ValueError(f"manual verdict row has no window_id/audio_id: {row}")
    return window_id, round(float(row["start"]), 6), round(float(row["end"]), 6)


def _stable_candidate_id(key: tuple[str, float, float]) -> str:
    raw = f"{key[0]}|{key[1]:.6f}|{key[2]:.6f}"
    return f"manual-preasr-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _partition(video_id: str, holdout_percent: int) -> str:
    bucket = int(hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "audit_holdout" if bucket < holdout_percent else "train_anchor"


def build_corpus(
    verdict_paths: list[Path],
    *,
    holdout_percent: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, float, float], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for path in verdict_paths:
        for row in _read_jsonl(path):
            verdict = str(row.get("verdict") or row.get("manual_label") or "").strip().lower()
            if verdict not in VERDICT_TO_LABEL:
                raise ValueError(f"invalid manual verdict {verdict!r} in {path}")
            grouped[_span_key(row)].append((path.parent.name, row))

    corpus: list[dict[str, Any]] = []
    conflict_count = 0
    for key in sorted(grouped):
        records = grouped[key]
        verdicts = sorted({str(row.get("verdict") or row.get("manual_label")).lower() for _, row in records})
        conflict = len(verdicts) > 1
        if conflict:
            conflict_count += 1
        verdict = "unsure" if conflict else verdicts[0]
        representative = records[-1][1]
        video_id = str(
            representative.get("video_id")
            or representative.get("source_video_id")
            or key[0].rsplit("-w", 1)[0]
        )
        label = VERDICT_TO_LABEL[verdict]
        candidate_id = _stable_candidate_id(key)
        corpus.append(
            {
                "schema": CORPUS_SCHEMA,
                "candidate_id": candidate_id,
                "sample_id": candidate_id,
                "audio_id": key[0],
                "window_id": key[0],
                "video_id": video_id,
                "chunk_index": int(representative.get("chunk_index", 0)),
                "start": key[1],
                "end": key[2],
                "duration_s": round(max(0.0, key[2] - key[1]), 6),
                "label": label,
                "display_decision": verdict,
                "training_label_included": label != "ambiguous_ignore",
                "label_source": "manual_audit:cueqc_v12_corpus",
                "override_source": "manual_audit",
                "override_reason": "conflicting_manual_verdicts" if conflict else f"manual_verdict:{verdict}",
                "manual_partition": _partition(video_id, holdout_percent),
                "audit_sources": sorted({audit_id for audit_id, _ in records}),
                "source_candidate_ids": sorted(
                    {
                        str(row.get("candidate_id") or row.get("sample_id") or "")
                        for _, row in records
                        if row.get("candidate_id") or row.get("sample_id")
                    }
                ),
                "source_verdicts": verdicts,
            }
        )

    summary = {
        "schema": "pre_asr_manual_audit_corpus_summary_v1",
        "verdict_files": [str(path) for path in verdict_paths],
        "source_verdict_rows": sum(len(items) for items in grouped.values()),
        "unique_spans": len(corpus),
        "duplicate_rows_collapsed": sum(len(items) - 1 for items in grouped.values()),
        "conflicting_span_count": conflict_count,
        "labels": dict(Counter(str(row["label"]) for row in corpus)),
        "partitions": dict(Counter(str(row["manual_partition"]) for row in corpus)),
        "partition_labels": {
            partition: dict(
                Counter(
                    str(row["label"])
                    for row in corpus
                    if row["manual_partition"] == partition
                )
            )
            for partition in ("train_anchor", "audit_holdout")
        },
        "holdout_percent": int(holdout_percent),
        "partition_key": "sha256(video_id)[:8] % 100",
    }
    return corpus, summary


def _source_windows(paths: list[Path]) -> list[dict[str, Any]]:
    return [row for path in paths for row in _read_jsonl(path)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    verdict_paths = [project_path(path) for path in args.verdicts]
    source_window_paths = [project_path(path) for path in args.new_source_windows]
    output_dir = project_path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT
        / "agents"
        / "temp"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pre-asr-manual-audit-corpus"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus, summary = build_corpus(
        verdict_paths,
        holdout_percent=int(args.holdout_percent),
    )
    corpus_path = output_dir / "manual_audit_corpus.jsonl"
    _write_jsonl(corpus_path, corpus)

    projected, unmatched, projection_summary = project_label_rows(
        corpus,
        _candidate_rows_from_source_windows(source_window_paths),
        boundary_tolerance_s=float(args.boundary_tolerance_s),
        iou_threshold=1.0,
    )
    corpus_by_id = {str(row["candidate_id"]): row for row in corpus}
    for row in projected:
        source = corpus_by_id[str(row["projection"]["source_candidate_id"])]
        row["manual_partition"] = source["manual_partition"]
        row["audit_sources"] = source["audit_sources"]
        row["manual_corpus_candidate_id"] = source["candidate_id"]

    projected_path = output_dir / "projected_manual_labels.jsonl"
    unmatched_path = output_dir / "unmatched_current_candidates.jsonl"
    _write_jsonl(projected_path, projected)
    _write_jsonl(unmatched_path, unmatched)

    exact_definite = [
        row
        for row in projected
        if row["projection"]["method"] == "boundary_match"
        and row["label"] in {"definite_keep", "definite_drop"}
    ]
    train_anchor_ids = [
        str(row["candidate_id"])
        for row in exact_definite
        if row["manual_partition"] == "train_anchor"
    ]
    holdout_candidate_ids = [
        str(row["candidate_id"])
        for row in exact_definite
        if row["manual_partition"] == "audit_holdout"
    ]
    train_audio_ids = {
        str(row["audio_id"])
        for row in exact_definite
        if row["manual_partition"] == "train_anchor"
    }
    holdout_video_ids = {
        str(row["video_id"])
        for row in corpus
        if row["manual_partition"] == "audit_holdout"
        and row["label"] in {"definite_keep", "definite_drop"}
    }
    holdout_audio_ids = {
        str(row["window_id"])
        for row in _source_windows(source_window_paths)
        if str(row.get("video_id") or "") in holdout_video_ids
    }

    train_ids_path = output_dir / "train_anchor_candidate_ids.txt"
    train_audio_path = output_dir / "force_train_audio_ids.txt"
    holdout_ids_path = output_dir / "holdout_candidate_ids.txt"
    holdout_audio_path = output_dir / "force_val_audio_ids.txt"
    _write_ids(train_ids_path, train_anchor_ids)
    _write_ids(train_audio_path, train_audio_ids)
    _write_ids(holdout_ids_path, holdout_candidate_ids)
    _write_ids(holdout_audio_path, holdout_audio_ids)

    summary.update(
        {
            "new_source_windows": [str(path) for path in source_window_paths],
            "corpus": str(corpus_path),
            "projected_manual_labels": str(projected_path),
            "projection": projection_summary,
            "exact_definite_projection_count": len(exact_definite),
            "train_anchor_candidate_count": len(train_anchor_ids),
            "train_anchor_audio_count": len(train_audio_ids),
            "holdout_candidate_count": len(holdout_candidate_ids),
            "holdout_video_count": len(holdout_video_ids),
            "holdout_audio_count": len(holdout_audio_ids),
            "train_anchor_candidate_ids": str(train_ids_path),
            "force_train_audio_ids": str(train_audio_path),
            "holdout_candidate_ids": str(holdout_ids_path),
            "force_val_audio_ids": str(holdout_audio_path),
            "unmatched_current_candidates": str(unmatched_path),
            "manual_projection_policy": "boundary_match_only; changed spans become ambiguous_ignore",
        }
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and project the stable Pre-ASR manual audit corpus."
    )
    parser.add_argument("--verdicts", action="append", required=True)
    parser.add_argument("--new-source-windows", action="append", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--holdout-percent", type=int, default=20)
    parser.add_argument("--boundary-tolerance-s", type=float, default=0.12)
    args = parser.parse_args(argv)
    if not 1 <= args.holdout_percent <= 99:
        parser.error("--holdout-percent must be in [1, 99]")
    if args.boundary_tolerance_s <= 0.0:
        parser.error("--boundary-tolerance-s must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
