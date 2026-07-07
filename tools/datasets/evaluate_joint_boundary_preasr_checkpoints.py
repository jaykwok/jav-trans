#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for value in (PROJECT_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from asr.pre_asr_cueqc import load_checkpoint as load_pre_asr  # noqa: E402
from boundary.sequence_store import load_sequence_arrays  # noqa: E402
from boundary.split_model import load_semantic_split_verifier  # noqa: E402
from tools.asr.cueqc.compile_pre_asr_v12_features import (  # noqa: E402
    candidate_for_chunk,
    label_for_chunk,
    read_chunk_document,
    read_labels,
)


SPLIT_LABELS = ("cut", "continue", "unsure")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _classification_metrics(
    truth: list[str],
    predicted: list[str],
    labels: tuple[str, ...],
) -> dict[str, Any]:
    confusion = {
        expected: {
            actual: sum(
                left == expected and right == actual
                for left, right in zip(truth, predicted)
            )
            for actual in labels
        }
        for expected in labels
    }
    result: dict[str, Any] = {
        "count": len(truth),
        "accuracy": (
            sum(left == right for left, right in zip(truth, predicted))
            / max(1, len(truth))
        ),
        "confusion": confusion,
    }
    for label in labels:
        tp = confusion[label][label]
        expected_count = sum(confusion[label].values())
        predicted_count = sum(confusion[other][label] for other in labels)
        result[f"{label}_recall"] = tp / max(1, expected_count)
        result[f"{label}_precision"] = tp / max(1, predicted_count)
    return result


def _evaluate_split(
    *,
    dataset: Path,
    checkpoint: Path,
    device: str,
    repo_id: str,
) -> dict[str, Any]:
    bundle = load_sequence_arrays(dataset / "semantic_split" / "features.npz")
    model = load_semantic_split_verifier(
        checkpoint,
        device=device,
        expected_ptm_repo_id=repo_id,
    )
    predicted: list[str] = []
    frames = bundle["frame_features"].astype(np.float32)
    scalars = bundle["scalar_features"].astype(np.float32)
    for start in range(0, len(frames), 256):
        predicted.extend(
            decision.label
            for decision in model.decide(
                frame_features=frames[start : start + 256],
                scalar_features=scalars[start : start + 256],
            )
        )
    truth = [SPLIT_LABELS[int(value)] for value in bundle["labels"]]
    partitions = bundle["partitions"].astype(str)
    result = {
        "all": _classification_metrics(truth, predicted, SPLIT_LABELS)
    }
    for partition in ("train", "val"):
        indexes = np.flatnonzero(partitions == partition)
        result[partition] = _classification_metrics(
            [truth[int(index)] for index in indexes],
            [predicted[int(index)] for index in indexes],
            SPLIT_LABELS,
        )
    return result


def _evaluate_pre_asr(
    *,
    dataset: Path,
    checkpoint: Path,
    device: str,
    threshold: float | None,
) -> dict[str, Any]:
    windows = _read_jsonl(dataset / "source_windows.jsonl")
    split_bundle = np.load(dataset / "semantic_split" / "features.npz")
    partition_by_video: dict[str, str] = {}
    for video_id, partition in zip(
        split_bundle["video_ids"].astype(str),
        split_bundle["partitions"].astype(str),
    ):
        partition_by_video.setdefault(str(video_id), str(partition))
    labels = read_labels([str(dataset / "pre_asr" / "labels.jsonl")])
    model = load_pre_asr(checkpoint, device=device)
    if threshold is not None:
        model.drop_threshold = float(threshold)
    all_rows: list[dict[str, Any]] = []
    by_video: dict[str, list[dict[str, Any]]] = {}
    for window in windows:
        audio_id, chunks = read_chunk_document(Path(window["pre_asr_candidates"]))
        candidates = [
            candidate_for_chunk(chunks, index)
            for index in range(len(chunks))
        ]
        decisions = model.decide(candidates)
        video_id = str(window["video_id"])
        for index, (chunk, candidate, decision) in enumerate(
            zip(chunks, candidates, decisions)
        ):
            label = label_for_chunk(
                labels,
                audio_id=audio_id,
                chunk=chunk,
                index=index,
            )
            if label is None or int(label["label_index"]) not in (0, 1):
                continue
            truth = "drop" if int(label["label_index"]) == 0 else "keep"
            prediction = (
                "drop"
                if decision["route"] == "drop_before_asr"
                else "keep"
            )
            item = {
                "window_id": str(window["window_id"]),
                "video_id": video_id,
                "chunk_index": int(candidate["index"]),
                "start": float(candidate["start"]),
                "end": float(candidate["end"]),
                "duration_s": float(candidate["duration_s"]),
                "truth": truth,
                "prediction": prediction,
                "prob_drop": float(decision["prob_drop"]),
                "partition": partition_by_video.get(video_id, "train"),
            }
            all_rows.append(item)
            by_video.setdefault(video_id, []).append(item)
    truth = [row["truth"] for row in all_rows]
    predicted = [row["prediction"] for row in all_rows]
    errors = [row for row in all_rows if row["truth"] != row["prediction"]]
    result = {
        "all": _classification_metrics(
            truth,
            predicted,
            ("drop", "keep"),
        ),
        "video_count": len(by_video),
        "false_decisions": errors,
    }
    for partition in ("train", "val"):
        partition_rows = [
            row for row in all_rows if row["partition"] == partition
        ]
        result[partition] = _classification_metrics(
            [row["truth"] for row in partition_rows],
            [row["prediction"] for row in partition_rows],
            ("drop", "keep"),
        )
    return result


def run(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset_dir)
    result: dict[str, Any] = {
        "schema": "joint_boundary_preasr_checkpoint_evaluation_v1",
        "semantic_split": {},
        "pre_asr": {},
    }
    for raw_checkpoint in args.semantic_checkpoint:
        checkpoint = Path(raw_checkpoint)
        result["semantic_split"][str(checkpoint)] = _evaluate_split(
            dataset=dataset,
            checkpoint=checkpoint,
            device=args.device,
            repo_id=args.asr_repo_id,
        )
    for raw_checkpoint in args.pre_asr_checkpoint:
        checkpoint = Path(raw_checkpoint)
        thresholds = args.pre_asr_threshold or [None]
        for threshold in thresholds:
            key = (
                str(checkpoint)
                if threshold is None
                else f"{checkpoint}@threshold={threshold:g}"
            )
            result["pre_asr"][key] = _evaluate_pre_asr(
                dataset=dataset,
                checkpoint=checkpoint,
                device=args.device,
                threshold=threshold,
            )
    output = Path(args.output or dataset / "checkpoint_evaluation.json")
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="datasets/train/omni-joint-boundary-preasr-v1",
    )
    parser.add_argument("--semantic-checkpoint", action="append", default=[])
    parser.add_argument("--pre-asr-checkpoint", action="append", default=[])
    parser.add_argument(
        "--pre-asr-threshold",
        action="append",
        type=float,
        default=[],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--asr-repo-id",
        default="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    )
    parser.add_argument("--output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
