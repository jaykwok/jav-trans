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

from boundary.ja.dataset import read_jsonl  # noqa: E402
from boundary.ja.features import load_cached_feature  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    load_speech_island_scorer_checkpoint,
    score_speech_island_probabilities_batch,
)
from boundary.outer_refiner import OUTER_EDGE_FEATURE_SCHEMA  # noqa: E402


SCALAR_NAMES = (
    "island_duration_s",
    "island_position_ratio",
    "speech_mean",
    "speech_min",
    "speech_max",
    "speech_active_ratio",
)


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _islands(probabilities: np.ndarray, *, threshold: float, dilation_frames: int) -> list[tuple[int, int]]:
    active = probabilities >= threshold
    if dilation_frames > 0:
        kernel = np.ones(2 * dilation_frames + 1, dtype=np.int16)
        active = np.convolve(active.astype(np.int16), kernel, mode="same") > 0
    result: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(np.append(active, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index))
            start = None
    return result


def _pool(array: np.ndarray, start: int, end: int, bins: int) -> np.ndarray:
    window = array[max(0, start) : min(array.shape[0], end)]
    return np.stack(
        [
            part.mean(axis=0).astype(np.float32)
            if part.shape[0]
            else np.zeros(array.shape[1], dtype=np.float32)
            for part in np.array_split(window, bins, axis=0)
        ]
    )


def _features(
    *,
    ptm: np.ndarray,
    mfcc: np.ndarray,
    speech: np.ndarray,
    start: int,
    end: int,
    context_frames: int,
    ptm_dim: int,
    frame_hop_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate((ptm[:, :ptm_dim], mfcc), axis=1).astype(np.float32)
    frames = np.concatenate(
        (
            _pool(combined, start - context_frames, start, 4),
            _pool(combined, start, start + context_frames, 4),
            _pool(combined, end - context_frames, end, 4),
            _pool(combined, end, end + context_frames, 4),
        )
    )
    window = speech[start:end]
    scalar = np.asarray(
        (
            (end - start) * frame_hop_s,
            (start + end) / max(1, 2 * speech.size),
            float(window.mean()) if window.size else 0.0,
            float(window.min()) if window.size else 0.0,
            float(window.max()) if window.size else 0.0,
            float((window >= 0.5).mean()) if window.size else 0.0,
        ),
        dtype=np.float32,
    )
    return frames, scalar


def run(args: argparse.Namespace) -> None:
    records = read_jsonl(Path(args.labels))
    rows = _rows(Path(args.feature_manifest))
    scorer = load_speech_island_scorer_checkpoint(args.scorer_checkpoint, device=args.device)
    frame_rows: list[np.ndarray] = []
    scalar_rows: list[np.ndarray] = []
    delta_rows: list[tuple[float, float]] = []
    confidence_rows: list[tuple[float, float]] = []
    partitions: list[str] = []
    metadata_rows: list[dict] = []
    selected = []
    for row_index, row in enumerate(rows):
        record = records[int(row["label_index"])]
        metadata = dict(record.boundary_metadata or {})
        frame_count = int(row.get("frame_count") or 0)
        if metadata.get("actual_speech_segments") and (
            args.max_frames <= 0 or frame_count <= args.max_frames
        ):
            selected.append((row_index, row, record, metadata))
        if args.max_rows and row_index + 1 >= args.max_rows:
            break
    selected.sort(key=lambda item: int(item[1].get("frame_count") or 0))
    batches = []
    current = []
    current_max_frames = 0
    for item in selected:
        frame_count = int(item[1].get("frame_count") or 0)
        next_max = max(current_max_frames, frame_count)
        if current and (
            len(current) >= args.batch_size
            or next_max * (len(current) + 1) > args.max_batch_frames
        ):
            batches.append(current)
            current = []
            current_max_frames = 0
        current.append(item)
        current_max_frames = max(current_max_frames, frame_count)
    if current:
        batches.append(current)
    processed = 0
    for batch in batches:
        loaded = []
        for row_index, row, record, metadata in batch:
            ptm, mfcc = load_cached_feature(Path(str(row["feature_path"])))
            total = min(ptm.shape[0], mfcc.shape[0], len(record.speech_frames))
            if total > 0:
                loaded.append(
                    (row_index, record, metadata, ptm[:total], mfcc[:total])
                )
        probabilities_batch = score_speech_island_probabilities_batch(
            scorer,
            feature_pairs=[(item[3], item[4]) for item in loaded],
        )
        for (row_index, record, metadata, ptm, mfcc), probabilities in zip(
            loaded, probabilities_batch
        ):
            actual = list(metadata["actual_speech_segments"])
            predicted = _islands(
                probabilities,
                threshold=args.threshold,
                dilation_frames=round(args.dilation_s / record.frame_hop_s),
            )
            for start, end in predicted:
                predicted_start = start * record.frame_hop_s
                predicted_end = end * record.frame_hop_s
                overlaps = [
                    item
                    for item in actual
                    if float(item["end"]) > predicted_start
                    and float(item["start"]) < predicted_end
                ]
                if overlaps:
                    true_start = min(float(item["start"]) for item in overlaps)
                    true_end = max(float(item["end"]) for item in overlaps)
                    start_delta = true_start - predicted_start
                    end_delta = true_end - predicted_end
                    confidence = (
                        float(abs(start_delta) <= args.max_delta_s),
                        float(abs(end_delta) <= args.max_delta_s),
                    )
                else:
                    start_delta = end_delta = 0.0
                    confidence = (0.0, 0.0)
                frames, scalars = _features(
                    ptm=ptm,
                    mfcc=mfcc,
                    speech=probabilities,
                    start=start,
                    end=end,
                    context_frames=round(args.context_s / record.frame_hop_s),
                    ptm_dim=args.ptm_dim,
                    frame_hop_s=record.frame_hop_s,
                )
                frame_rows.append(frames)
                scalar_rows.append(scalars)
                delta_rows.append(
                    (
                        float(np.clip(start_delta, -args.max_delta_s, args.max_delta_s)),
                        float(np.clip(end_delta, -args.max_delta_s, args.max_delta_s)),
                    )
                )
                confidence_rows.append(confidence)
                partitions.append(str(metadata.get("source_partition") or "train"))
                metadata_rows.append(
                    {
                        "audio_id": record.audio_id,
                        "predicted_start": predicted_start,
                        "predicted_end": predicted_end,
                        "start_delta_s": start_delta,
                        "end_delta_s": end_delta,
                        "start_confidence_target": confidence[0],
                        "end_confidence_target": confidence[1],
                    }
                )
        processed += len(batch)
        if args.log_every and processed % args.log_every < args.batch_size:
            print(
                f"outer_refiner_dataset_rows={processed}/{len(selected)} "
                f"items={len(frame_rows)}",
                flush=True,
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frame_features=np.stack(frame_rows),
        scalar_features=np.stack(scalar_rows),
        target_deltas=np.asarray(delta_rows, dtype=np.float32),
        confidence_targets=np.asarray(confidence_rows, dtype=np.float32),
        partitions=np.asarray(partitions),
    )
    with output.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema": OUTER_EDGE_FEATURE_SCHEMA,
        "count": len(frame_rows),
        "matched_start": int(sum(item[0] for item in confidence_rows)),
        "matched_end": int(sum(item[1] for item in confidence_rows)),
        "scalar_names": list(SCALAR_NAMES),
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build speech-island outer edge dataset.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--dilation-s", type=float, default=0.2)
    parser.add_argument("--context-s", type=float, default=0.6)
    parser.add_argument("--max-delta-s", type=float, default=0.5)
    parser.add_argument("--ptm-dim", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batch-frames", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
