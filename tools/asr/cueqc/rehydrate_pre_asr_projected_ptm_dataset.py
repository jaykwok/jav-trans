#!/usr/bin/env python3
"""Repool Pre-ASR chunks from the persisted variance-preserving PTM frames."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.pre_asr_cueqc import (  # noqa: E402
    PRE_ASR_CUEQC_FEATURE_NAMES,
    PRE_ASR_CUEQC_PTM_BINS,
    PRE_ASR_CUEQC_PTM_DIM,
    PRE_ASR_CUEQC_PTM_BIN_FEATURE_NAMES,
)
from boundary.sequence_features import (  # noqa: E402
    CHUNK_PROJECTED_PTM_SCHEMA,
    FrameSequenceFeatureConfig,
    FrameSequenceFeatureProvider,
)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _provider(path: Path) -> FrameSequenceFeatureProvider:
    bundle = np.load(path)
    projected = np.asarray(bundle["ptm_projected"], dtype=np.float32)
    digest = str(np.asarray(bundle["ptm_projection_digest"]).reshape(-1)[0])
    if projected.ndim != 2 or projected.shape[1] != PRE_ASR_CUEQC_PTM_DIM:
        raise ValueError(f"projected PTM shape mismatch: {path}: {projected.shape}")
    return FrameSequenceFeatureProvider(
        duration_s=projected.shape[0] * float(bundle["frame_hop_s"][0]),
        frame_hop_s=float(bundle["frame_hop_s"][0]),
        ptm=bundle["ptm"],
        mfcc=bundle["mfcc"],
        ptm_projected=projected,
        ptm_projected_digest=digest,
        config=FrameSequenceFeatureConfig(max_ptm_dims=PRE_ASR_CUEQC_PTM_DIM),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_windows_path = _project_path(args.source_windows)
    output_dir = _project_path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT
        / "agents"
        / "temp"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pre-asr-projected-ptm-rehydrate"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_root = output_dir / "features"
    feature_root.mkdir(parents=True, exist_ok=True)

    output_windows: list[dict[str, Any]] = []
    digests: Counter[str] = Counter()
    candidate_count = 0
    for window in _read_jsonl(source_windows_path):
        window_id = str(window["window_id"])
        provider = _provider(_project_path(str(window["speech_sequence_features"])))
        signature = provider.chunk_pooled_projected_ptm_signature(
            bins=PRE_ASR_CUEQC_PTM_BINS
        )
        digest = str(signature["ptm_projection_digest"])
        digests[digest] += 1
        candidates = _read_jsonl(_project_path(str(window["pre_asr_candidates"])))
        hydrated: list[dict[str, Any]] = []
        for candidate in candidates:
            pooled = provider.chunk_pooled_projected_ptm_features(
                start_s=float(candidate["start"]),
                end_s=float(candidate["end"]),
                bins=PRE_ASR_CUEQC_PTM_BINS,
            )
            item = dict(candidate)
            item.update(
                {
                    "ptm_pooling_schema": CHUNK_PROJECTED_PTM_SCHEMA,
                    "pre_asr_ptm_pooling_schema": CHUNK_PROJECTED_PTM_SCHEMA,
                    "ptm_pooling_available": True,
                    "ptm_pooling_bins": PRE_ASR_CUEQC_PTM_BINS,
                    "pre_asr_ptm_pooling_bins": PRE_ASR_CUEQC_PTM_BINS,
                    "ptm_pooling_dim": len(pooled),
                    "pre_asr_ptm_pooling_dim": len(pooled),
                    "ptm_projection_digest": digest,
                    "pre_asr_ptm_projection_digest": digest,
                    "pre_asr_ptm_pooled_features": pooled,
                    "ptm_bin_feature_names": list(PRE_ASR_CUEQC_PTM_BIN_FEATURE_NAMES),
                    "feature_names": list(PRE_ASR_CUEQC_FEATURE_NAMES),
                }
            )
            hydrated.append(item)
        target_dir = feature_root / window_id
        target_dir.mkdir(parents=True, exist_ok=True)
        candidates_path = target_dir / "pre_asr_candidates.jsonl"
        _write_jsonl(candidates_path, hydrated)
        output_window = dict(window)
        output_window["pre_asr_candidates"] = str(candidates_path.resolve())
        output_window["pre_asr_ptm_pooling_schema"] = CHUNK_PROJECTED_PTM_SCHEMA
        output_window["pre_asr_ptm_projection_digest"] = digest
        output_windows.append(output_window)
        candidate_count += len(hydrated)

    if len(digests) != 1:
        raise ValueError(f"expected one projected PTM digest, got {dict(digests)}")
    output_windows_path = output_dir / "source_windows.jsonl"
    _write_jsonl(output_windows_path, output_windows)
    digest = next(iter(digests))
    summary = {
        "schema": "pre_asr_projected_ptm_rehydrate_summary_v1",
        "source_windows": str(source_windows_path),
        "output_source_windows": str(output_windows_path),
        "window_count": len(output_windows),
        "candidate_count": candidate_count,
        "ptm_pooling_schema": CHUNK_PROJECTED_PTM_SCHEMA,
        "ptm_projection_digest": digest,
        "ptm_dim": PRE_ASR_CUEQC_PTM_DIM,
        "ptm_bins": PRE_ASR_CUEQC_PTM_BINS,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rehydrate Pre-ASR candidates with projected PTM pooling.")
    parser.add_argument("--source-windows", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
