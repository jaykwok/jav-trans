#!/usr/bin/env python3
"""Re-export joint-window Pre-ASR chunks with the current boundary chain.

This is Stage D.1's bridge from old Omni-labeled windows to the promoted
proposer/Split operation point. It preserves the sampled source windows and
training WAVs, but writes fresh runtime artifacts under a new output directory:

- ``source_windows.jsonl`` with updated feature/candidate paths;
- ``features/<window_id>/pre_asr_candidates.jsonl`` for label projection;
- current semantic split feature exports and boundary audit sidecars.

When multiple input datasets contain the same ``window_id``, the later dataset
argument wins. The planned invocation order is v1, v2, v3, so the latest window
manifest is the single source for duplicate windows.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (PROJECT_ROOT, SRC_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

SCHEMA = "joint_pre_asr_chunk_reexport_v1"


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


def collect_source_windows(dataset_dirs: Iterable[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_window: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    dataset_counts: Counter[str] = Counter()
    overwritten: Counter[str] = Counter()
    for dataset_dir in dataset_dirs:
        dataset = project_path(dataset_dir)
        rows = _read_jsonl(dataset / "source_windows.jsonl")
        dataset_counts[dataset.name] += len(rows)
        for row in rows:
            window_id = str(row["window_id"])
            if window_id not in by_window:
                order.append(window_id)
            else:
                overwritten[window_id] += 1
            by_window[window_id] = {
                **row,
                "source_dataset_dir": str(dataset.resolve()),
                "source_dataset_name": dataset.name,
            }
    return [by_window[window_id] for window_id in order], {
        "input_window_count": sum(dataset_counts.values()),
        "deduped_window_count": len(order),
        "input_datasets": dict(dataset_counts),
        "overwritten_duplicate_window_count": int(sum(overwritten.values())),
        "overwritten_duplicate_windows": dict(overwritten),
    }


def _reexported_row(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    feature_dir: Path,
) -> dict[str, Any]:
    updated = dict(row)
    updated.update(
        {
            "schema": SCHEMA,
            "reexport_source_schema": str(row.get("schema") or ""),
            "semantic_split_features": str(Path(str(payload["semantic_split_features"])).resolve()),
            "semantic_split_metadata": str(Path(str(payload["semantic_split_metadata"])).resolve()),
            "speech_sequence_features": str(Path(str(payload["speech_sequence_features"])).resolve()),
            "pre_asr_candidates": str(Path(str(payload["pre_asr_candidates"])).resolve()),
            "boundary_audit": str(Path(str(payload["boundary_audit"])).resolve()),
            "span_count": int(payload.get("span_count", row.get("span_count", 0)) or 0),
            "candidate_count": int(payload.get("candidate_count", row.get("candidate_count", 0)) or 0),
            "reexport_feature_dir": str(feature_dir.resolve()),
            "reexport_resumed": bool(payload.get("resumed")),
        }
    )
    return updated


def reexport_windows(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    exporter: Callable[..., Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    resumed = 0
    candidate_count = 0
    span_count = 0
    for position, row in enumerate(rows, start=1):
        window_id = str(row["window_id"])
        feature_dir = output_dir / "features" / window_id
        payload = exporter(
            wav_path=project_path(str(row["audio_wav"])),
            feature_dir=feature_dir,
        )
        updated = _reexported_row(row, payload, feature_dir=feature_dir)
        output_rows.append(updated)
        if bool(payload.get("resumed")):
            resumed += 1
        candidate_count += int(updated.get("candidate_count") or 0)
        span_count += int(updated.get("span_count") or 0)
        _write_jsonl(output_dir / "source_windows.jsonl", output_rows)
        print(
            f"reexported window={position}/{len(rows)} id={window_id} "
            f"candidates={updated['candidate_count']} resumed={updated['reexport_resumed']}",
            flush=True,
        )
    return output_rows, {
        "window_count": len(output_rows),
        "resumed_window_count": resumed,
        "span_count": span_count,
        "candidate_count": candidate_count,
    }


def run(args: argparse.Namespace) -> None:
    from boundary.gpu_safety import apply_vram_safety_cap
    from tools.datasets.prepare_joint_boundary_omni_dataset import (
        _export_boundary_features,
    )

    applied_ratio = apply_vram_safety_cap()
    output_dir = project_path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT
        / "agents"
        / "temp"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pre-asr-chunk-reexport"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows, collect_summary = collect_source_windows(
        project_path(path) for path in args.dataset_dir
    )
    selected_rows = (
        source_rows
        if args.max_windows is None
        else source_rows[: max(0, int(args.max_windows))]
    )
    output_rows, reexport_summary = reexport_windows(
        selected_rows,
        output_dir=output_dir,
        exporter=_export_boundary_features,
    )
    _write_jsonl(output_dir / "source_windows.jsonl", output_rows)
    summary = {
        "schema": "joint_pre_asr_chunk_reexport_summary_v1",
        "output_dir": str(output_dir.resolve()),
        "source_windows": str((output_dir / "source_windows.jsonl").resolve()),
        "vram_safety_ratio": applied_ratio,
        "selected_window_count": len(selected_rows),
        **collect_summary,
        **reexport_summary,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-export Pre-ASR chunk candidates for existing joint windows."
    )
    parser.add_argument("--dataset-dir", action="append", required=True)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to agents/temp/<timestamp>_pre-asr-chunk-reexport/.",
    )
    parser.add_argument("--max-windows", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
