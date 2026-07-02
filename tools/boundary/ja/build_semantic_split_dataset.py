#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boundary.ja.dataset import read_jsonl  # noqa: E402
from boundary.ja.features import load_cached_feature  # noqa: E402
from boundary.sequence_features import SPLIT_CANDIDATE_SCALAR_NAMES  # noqa: E402
from boundary.split_model import SEMANTIC_SPLIT_FEATURE_SCHEMA  # noqa: E402


LABELS = {"cut": 0, "continue": 1, "unsure": 2}


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values, (10.0, 90.0))
    if high <= low + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _acoustic_scores(mfcc: np.ndarray) -> np.ndarray:
    energy = _normalize(mfcc[:, 0])
    delta = np.zeros(mfcc.shape[0], dtype=np.float32)
    if mfcc.shape[0] > 1:
        delta[1:] = np.mean(np.abs(np.diff(mfcc, axis=0)), axis=1)
    return np.clip(0.7 * (1.0 - energy) + 0.3 * _normalize(delta), 0.0, 1.0)


def _pool(array: np.ndarray, start: int, end: int, bins: int) -> np.ndarray:
    window = array[max(0, start) : min(array.shape[0], end)]
    result: list[np.ndarray] = []
    for part in np.array_split(window, bins, axis=0):
        result.append(
            part.mean(axis=0).astype(np.float32)
            if part.shape[0]
            else np.zeros(array.shape[1], dtype=np.float32)
        )
    return np.stack(result)


def _candidate_features(
    *,
    ptm: np.ndarray,
    mfcc: np.ndarray,
    speech: np.ndarray,
    acoustic: np.ndarray,
    core_start: float,
    core_end: float,
    time_s: float,
    frame_hop_s: float,
    ptm_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame = int(round(time_s / frame_hop_s))
    left = int(round(1.6 / frame_hop_s))
    gap = int(round(0.3 / frame_hop_s))
    combined = np.concatenate((ptm[:, :ptm_dim], mfcc), axis=1).astype(np.float32)
    frames = np.concatenate(
        (
            _pool(combined, frame - left, frame - gap, 8),
            _pool(combined, frame - gap, frame + gap, 4),
            _pool(combined, frame + gap, frame + left, 8),
        ),
        axis=0,
    )
    local = acoustic[max(0, frame - gap) : min(acoustic.size, frame + gap + 1)]
    score = float(acoustic[min(max(frame, 0), acoustic.size - 1)])
    prominence = score - float(local.min()) if local.size else 0.0
    speech_local = speech[max(0, frame - gap) : min(speech.size, frame + gap + 1)]
    speech_valley = 1.0 - float(speech_local.mean()) if speech_local.size else 0.0
    core_duration = core_end - core_start
    left_speech = speech[max(0, frame - left) : frame]
    right_speech = speech[frame : min(speech.size, frame + left)]
    scalar = np.asarray(
        (
            score,
            prominence,
            speech_valley,
            score + prominence + speech_valley,
            core_duration,
            time_s - core_start,
            core_end - time_s,
            (time_s - core_start) / max(core_duration, 1e-6),
            float(left_speech.mean()) if left_speech.size else 0.0,
            float(right_speech.mean()) if right_speech.size else 0.0,
            float(speech_local.mean()) if speech_local.size else 0.0,
            float((left_speech >= 0.5).mean()) if left_speech.size else 0.0,
            float((right_speech >= 0.5).mean()) if right_speech.size else 0.0,
        ),
        dtype=np.float32,
    )
    return frames, scalar


def run(args: argparse.Namespace) -> None:
    records = read_jsonl(Path(args.labels))
    rows = _rows(Path(args.feature_manifest))
    rng = np.random.default_rng(args.seed)
    frame_rows: list[np.ndarray] = []
    scalar_rows: list[np.ndarray] = []
    labels: list[int] = []
    partitions: list[str] = []
    metadata_rows: list[dict] = []
    cut_frame_rows: list[np.ndarray] = []
    cut_scalar_rows: list[np.ndarray] = []
    cut_targets: list[float] = []
    cut_partitions: list[str] = []
    type_seen: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for row in rows:
        record = records[int(row["label_index"])]
        metadata = dict(record.boundary_metadata or {})
        example_type = str(metadata.get("native_example_type") or "")
        if example_type not in {"long_speech_chain", "split_stress", "positive_speech_timeline"}:
            continue
        if example_type == "positive_speech_timeline" and type_seen[example_type] >= args.max_positive_rows:
            continue
        type_seen[example_type] += 1
        ptm, mfcc = load_cached_feature(Path(str(row["feature_path"])))
        total = min(ptm.shape[0], mfcc.shape[0], len(record.speech_frames))
        if total < 10:
            continue
        ptm = ptm[:total]
        mfcc = mfcc[:total]
        speech = np.asarray(record.speech_frames[:total], dtype=np.float32)
        acoustic = _acoustic_scores(mfcc)
        core_start = 0.0
        core_end = min(record.duration_s, total * record.frame_hop_s)
        cuts = list(metadata.get("utterance_boundaries") or metadata.get("cut_point_segments") or [])
        actual = list(metadata.get("actual_speech_segments") or [])
        candidates: list[tuple[str, float, str]] = []
        for cut in cuts:
            cut_time = float(cut["time_s"])
            if core_start + 1.0 <= cut_time <= core_end - 1.0:
                candidates.append(("cut", cut_time, "utterance_boundary"))
                for offset in (-0.24, -0.12, 0.12, 0.24):
                    proposal_time = cut_time + offset
                    cut_frames, cut_scalars = _candidate_features(
                        ptm=ptm,
                        mfcc=mfcc,
                        speech=speech,
                        acoustic=acoustic,
                        core_start=core_start,
                        core_end=core_end,
                        time_s=proposal_time,
                        frame_hop_s=record.frame_hop_s,
                        ptm_dim=args.ptm_dim,
                    )
                    cut_frame_rows.append(cut_frames)
                    cut_scalar_rows.append(cut_scalars)
                    cut_targets.append(-offset)
                    cut_partitions.append(str(metadata.get("source_partition") or "train"))
                offset = float(rng.choice((-0.45, 0.45)))
                unsure_time = cut_time + offset
                if core_start + 1.0 <= unsure_time <= core_end - 1.0:
                    candidates.append(("unsure", unsure_time, "offset_boundary"))
        spans = actual or [{"start": core_start, "end": core_end}]
        for span in spans:
            start = float(span["start"]) + 1.0
            end = float(span["end"]) - 1.0
            if end <= start:
                continue
            lower = max(0, int(round(start / record.frame_hop_s)))
            upper = min(total, int(round(end / record.frame_hop_s)))
            if upper <= lower:
                continue
            valley_frame = lower + int(np.argmax(acoustic[lower:upper]))
            candidates.append(
                ("continue", valley_frame * record.frame_hop_s, "intra_utterance_valley")
            )
        for label, time_s, reason in candidates:
            frame_features, scalar_features = _candidate_features(
                ptm=ptm,
                mfcc=mfcc,
                speech=speech,
                acoustic=acoustic,
                core_start=core_start,
                core_end=core_end,
                time_s=time_s,
                frame_hop_s=record.frame_hop_s,
                ptm_dim=args.ptm_dim,
            )
            frame_rows.append(frame_features)
            scalar_rows.append(scalar_features)
            labels.append(LABELS[label])
            partition = str(metadata.get("source_partition") or "train")
            partitions.append(partition)
            label_counts[label] += 1
            metadata_rows.append(
                {
                    "audio_id": record.audio_id,
                    "label": label,
                    "time_s": time_s,
                    "reason": reason,
                    "partition": partition,
                }
            )
        if args.max_candidates and len(labels) >= args.max_candidates:
            break
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_array = np.stack(frame_rows)
    scalar_array = np.stack(scalar_rows)
    np.savez_compressed(
        output,
        frame_features=frame_array,
        scalar_features=scalar_array,
        labels=np.asarray(labels, dtype=np.int64),
        partitions=np.asarray(partitions),
    )
    metadata_path = output.with_suffix(".jsonl")
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema": SEMANTIC_SPLIT_FEATURE_SCHEMA,
        "count": len(labels),
        "label_counts": dict(label_counts),
        "frame_shape": list(frame_array.shape),
        "scalar_names": list(SPLIT_CANDIDATE_SCALAR_NAMES),
        "source_type_counts": dict(type_seen),
        "output": str(output),
        "metadata": str(metadata_path),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.cut_refiner_output:
        cut_output = Path(args.cut_refiner_output)
        cut_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cut_output,
            frame_features=np.stack(cut_frame_rows),
            scalar_features=np.stack(cut_scalar_rows),
            target_delta_s=np.asarray(cut_targets, dtype=np.float32),
            partitions=np.asarray(cut_partitions),
        )
        cut_output.with_suffix(".summary.json").write_text(
            json.dumps(
                {
                    "schema": "cut_edge_candidate_features_v1",
                    "count": len(cut_targets),
                    "target_abs_mean_s": float(np.mean(np.abs(cut_targets))),
                    "output": str(cut_output),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate-level semantic split dataset.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cut-refiner-output", default="")
    parser.add_argument("--ptm-dim", type=int, default=128)
    parser.add_argument("--max-positive-rows", type=int, default=4000)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
