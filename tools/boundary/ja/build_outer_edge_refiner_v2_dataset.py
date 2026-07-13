#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.ja.dataset import effective_frame_weights, read_jsonl  # noqa: E402
from boundary.ja.features import load_cached_feature  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    SPEECH_ISLAND_SCORER_LABELS,
    load_speech_island_scorer_checkpoint,
    score_semantic_speech_outputs,
)
from boundary.ja.semantic_speech_train import _class_indexes  # noqa: E402
from boundary.outer_refiner_v2 import OUTER_EDGE_REFINER_V2_FEATURE_SCHEMA  # noqa: E402


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.append(np.asarray(active, dtype=bool), False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index))
            start = None
    return result


def run(args: argparse.Namespace) -> None:
    apply_vram_safety_cap(0.95)
    records = read_jsonl(Path(args.labels))
    rows = _rows(Path(args.feature_manifest))
    if not rows or str(rows[0].get("ptm") or "") != QWEN_ASR_17B_REPO_ID:
        raise ValueError("Outer Edge Refiner v2 dataset is 1.7B-only")
    scorer = load_speech_island_scorer_checkpoint(
        args.scorer_checkpoint, device=args.device
    )
    output_dir = Path(args.output_dir)
    feature_dir = output_dir / "islands"
    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    missed_target_sources = 0
    target_index = SPEECH_ISLAND_SCORER_LABELS.index("semantic_target")
    discardable_index = SPEECH_ISLAND_SCORER_LABELS.index("discardable")
    for row_index, row in enumerate(rows):
        record = records[int(row["label_index"])]
        ptm, mfcc = load_cached_feature(Path(str(row["feature_path"])))
        total = min(
            int(ptm.shape[0]),
            int(mfcc.shape[0]),
            len(record.speech_frames),
        )
        if total <= 0:
            continue
        probabilities, semantic_projected = score_semantic_speech_outputs(
            scorer,
            ptm=ptm[:total],
            mfcc=mfcc[:total],
        )
        truth = _class_indexes(record, total=total)
        weights = np.asarray(effective_frame_weights(record)[:total], dtype=np.float32)
        predicted = np.argmax(probabilities, axis=1)
        islands = _runs(predicted != discardable_index)
        if not islands and np.any(truth == target_index):
            missed_target_sources += 1
        for island_index, (start, end) in enumerate(islands):
            frame_total = end - start
            position = (
                np.arange(frame_total, dtype=np.float32) / max(1, frame_total - 1)
            ).reshape(-1, 1)
            features = np.concatenate(
                (
                    semantic_projected[start:end],
                    mfcc[start:end],
                    probabilities[start:end],
                    position,
                ),
                axis=1,
            ).astype(np.float32)
            feature_path = feature_dir / f"row{row_index:06d}_island{island_index:03d}.npz"
            np.savez_compressed(
                feature_path,
                features=features,
                labels=truth[start:end].astype(np.int64),
                weights=weights[start:end],
                scorer_probabilities=probabilities[start:end],
            )
            metadata = dict(record.boundary_metadata or {})
            manifest_rows.append(
                {
                    "schema": OUTER_EDGE_REFINER_V2_FEATURE_SCHEMA,
                    "audio_id": record.audio_id,
                    "feature_path": str(feature_path),
                    "start_frame": start,
                    "end_frame": end,
                    "frame_hop_s": float(record.frame_hop_s),
                    "partition": str(metadata.get("source_partition") or "train"),
                    "contains_target": bool(np.any(truth[start:end] == target_index)),
                }
            )
        if args.log_every and (row_index + 1) % args.log_every == 0:
            print(
                f"outer_v2_dataset={row_index + 1}/{len(rows)} islands={len(manifest_rows)}",
                flush=True,
            )
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    summary = {
        "schema": OUTER_EDGE_REFINER_V2_FEATURE_SCHEMA,
        "source_count": len(rows),
        "island_count": len(manifest_rows),
        "target_island_count": sum(bool(row["contains_target"]) for row in manifest_rows),
        "missed_target_source_count": missed_target_sources,
        "labels": list(SPEECH_ISLAND_SCORER_LABELS),
        "decision_mode": "argmax",
        "manifest": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 1.7B full-island Outer v2 data.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
