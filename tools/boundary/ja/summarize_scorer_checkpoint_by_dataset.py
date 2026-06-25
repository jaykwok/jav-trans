#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boundary.ja import read_jsonl  # noqa: E402
from boundary.ja.features import load_cached_feature  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    load_feature_frame_scorer_checkpoint,
    score_feature_frame_boundary_probabilities_batch,
)
from boundary.ja.train import scorer_v6_targets_from_record  # noqa: E402


SCHEMA = "speech_boundary_ja_scorer_dataset_output_summary_v1"


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def repo_display_path(path: str | Path | None) -> str:
    if not path:
        return ""
    raw = Path(path)
    try:
        return str(raw.resolve().relative_to(PROJECT_ROOT)).replace("/", "\\")
    except ValueError:
        return str(raw)


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"feature manifest row must be an object: {path}:{line_number}")
            rows.append(dict(row))
    return rows


def iter_frame_limited_batches(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    max_batch_frames: int,
) -> Iterable[list[dict[str, Any]]]:
    limit = max(1, int(max_batch_frames))
    current: list[dict[str, Any]] = []
    current_max_frames = 0
    for row in rows:
        try:
            frame_count = int(row.get("frame_count") or 0)
        except (TypeError, ValueError):
            frame_count = 0
        projected_count = len(current) + 1
        projected_max = max(current_max_frames, frame_count)
        projected_frames = projected_count * projected_max
        if current and (len(current) >= batch_size or projected_frames > limit):
            yield current
            current = []
            current_max_frames = 0
        current.append(row)
        current_max_frames = max(current_max_frames, frame_count)
    if current:
        yield current


def local_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _as_float_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    return np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float32).reshape(-1)


def distribution(values: Iterable[float] | np.ndarray) -> dict[str, float | int]:
    data = _as_float_array(values)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


class GroupAccumulator:
    def __init__(self) -> None:
        self.example_count = 0
        self.frame_count = 0
        self.speech_frame_count = 0
        self.split_frame_count = 0
        self.values: dict[str, list[np.ndarray]] = defaultdict(list)

    def add_values(self, name: str, values: np.ndarray) -> None:
        data = np.asarray(values, dtype=np.float32).reshape(-1)
        if data.size:
            self.values[name].append(data)

    def add_example(
        self,
        *,
        speech_probs: np.ndarray,
        split_probs: np.ndarray,
        speech_labels: np.ndarray,
        split_labels: np.ndarray,
        speech_threshold: float,
        split_threshold: float,
    ) -> None:
        total = min(
            int(speech_probs.size),
            int(split_probs.size),
            int(speech_labels.size),
            int(split_labels.size),
        )
        if total <= 0:
            return
        speech_probs = np.asarray(speech_probs[:total], dtype=np.float32)
        split_probs = np.asarray(split_probs[:total], dtype=np.float32)
        speech_labels = np.asarray(speech_labels[:total], dtype=np.float32) > 0.5
        split_labels = np.asarray(split_labels[:total], dtype=np.float32) > 0.5
        non_speech_labels = ~speech_labels

        self.example_count += 1
        self.frame_count += total
        self.speech_frame_count += int(np.sum(speech_labels))
        self.split_frame_count += int(np.sum(split_labels))

        self.add_values("speech_prob_all", speech_probs)
        self.add_values("split_prob_all", split_probs)
        self.add_values("speech_prob_on_speech", speech_probs[speech_labels])
        self.add_values("speech_prob_on_non_speech", speech_probs[non_speech_labels])
        self.add_values("split_prob_on_split", split_probs[split_labels])
        self.add_values("split_prob_on_non_split", split_probs[~split_labels])
        self.add_values("speech_hit_on_speech", (speech_probs[speech_labels] >= speech_threshold).astype(np.float32))
        self.add_values(
            "speech_false_positive_on_non_speech",
            (speech_probs[non_speech_labels] >= speech_threshold).astype(np.float32),
        )
        self.add_values("split_hit_on_split", (split_probs[split_labels] >= split_threshold).astype(np.float32))
        self.add_values("split_predicted_positive", (split_probs >= split_threshold).astype(np.float32))

    def summarize(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "example_count": int(self.example_count),
            "frame_count": int(self.frame_count),
            "speech_frame_count": int(self.speech_frame_count),
            "split_frame_count": int(self.split_frame_count),
            "speech_frame_ratio": (self.speech_frame_count / self.frame_count) if self.frame_count else 0.0,
            "split_frame_ratio": (self.split_frame_count / self.frame_count) if self.frame_count else 0.0,
            "distributions": {},
        }
        distributions: dict[str, Any] = {}
        for name, chunks in sorted(self.values.items()):
            values = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            distributions[name] = distribution(values)
        payload["distributions"] = distributions
        return payload


def summarize_checkpoint_by_dataset(
    *,
    checkpoint: Path,
    labels: Path,
    feature_manifest: Path,
    output: Path,
    device: str,
    batch_size: int,
    max_batch_frames: int,
    max_examples: int,
    speech_threshold: float,
    split_threshold: float,
    split_boundary_radius_frames: int,
    split_boundary_sigma_frames: float,
    split_target_mode: str,
) -> dict[str, Any]:
    records = read_jsonl(labels)
    rows = read_jsonl_rows(feature_manifest)
    if max_examples > 0:
        rows = rows[:max_examples]
    bundle = load_feature_frame_scorer_checkpoint(checkpoint, device=device)
    groups: dict[str, GroupAccumulator] = defaultdict(GroupAccumulator)
    processed = 0

    for batch_rows in iter_frame_limited_batches(
        rows,
        batch_size=max(1, int(batch_size)),
        max_batch_frames=max_batch_frames,
    ):
        feature_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        batch_records = []
        for row in batch_rows:
            label_index = int(row["label_index"])
            record = records[label_index]
            ptm, mfcc = load_cached_feature(project_path(str(row["feature_path"])))
            feature_pairs.append((ptm, mfcc))
            batch_records.append(record)
        outputs = score_feature_frame_boundary_probabilities_batch(bundle, feature_pairs=feature_pairs)
        if str(device).startswith("cuda"):
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
        for record, (speech_probs, split_probs) in zip(batch_records, outputs):
            frame_total = min(int(speech_probs.size), len(record.speech_frames))
            if frame_total <= 0:
                continue
            speech_labels, split_labels = scorer_v6_targets_from_record(
                record,
                frame_count=frame_total,
                split_boundary_radius_frames=split_boundary_radius_frames,
                split_boundary_sigma_frames=split_boundary_sigma_frames,
                split_target_mode=split_target_mode,
            )
            example_type = str((record.boundary_metadata or {}).get("native_example_type") or "unknown")
            for group_name in ("ALL", example_type):
                groups[group_name].add_example(
                    speech_probs=speech_probs[:frame_total],
                    split_probs=split_probs[:frame_total],
                    speech_labels=speech_labels,
                    split_labels=split_labels[:frame_total],
                    speech_threshold=speech_threshold,
                    split_threshold=split_threshold,
                )
            processed += 1

    summary = {
        "schema": SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": repo_display_path(checkpoint),
        "checkpoint_signature": bundle.signature(),
        "labels": repo_display_path(labels),
        "feature_manifest": repo_display_path(feature_manifest),
        "processed_examples": int(processed),
        "total_feature_rows": int(len(rows)),
        "thresholds": {
            "speech": float(speech_threshold),
            "split": float(split_threshold),
        },
        "target_config": {
            "split_boundary_radius_frames": int(split_boundary_radius_frames),
            "split_boundary_sigma_frames": float(split_boundary_sigma_frames),
            "split_target_mode": str(split_target_mode),
            "batch_size": int(batch_size),
            "max_batch_frames": int(max_batch_frames),
        },
        "groups": {name: groups[name].summarize() for name in sorted(groups)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize SpeechBoundary-JA scorer v6 checkpoint outputs by v6-native dataset group."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batch-frames", type=int, default=4096)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--speech-threshold", type=float, default=0.5)
    parser.add_argument("--split-threshold", type=float, default=0.5)
    parser.add_argument("--split-boundary-radius-frames", type=int, default=1)
    parser.add_argument("--split-boundary-sigma-frames", type=float, default=1.0)
    parser.add_argument("--split-target-mode", choices=["hard", "gaussian"], default="gaussian")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_batch_frames <= 0:
        parser.error("--max-batch-frames must be positive")
    if args.max_examples < 0:
        parser.error("--max-examples must be non-negative")
    if args.split_boundary_radius_frames < 0:
        parser.error("--split-boundary-radius-frames must be non-negative")
    if args.split_boundary_sigma_frames <= 0.0:
        parser.error("--split-boundary-sigma-frames must be positive")
    if not args.output:
        args.output = (
            PROJECT_ROOT
            / "agents"
            / "temp"
            / f"{local_timestamp()}_scorer-checkpoint-dataset-summary"
            / "summary.json"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_checkpoint_by_dataset(
        checkpoint=project_path(args.checkpoint),
        labels=project_path(args.labels),
        feature_manifest=project_path(args.feature_manifest),
        output=project_path(args.output),
        device=str(args.device),
        batch_size=int(args.batch_size),
        max_batch_frames=int(args.max_batch_frames),
        max_examples=int(args.max_examples),
        speech_threshold=float(args.speech_threshold),
        split_threshold=float(args.split_threshold),
        split_boundary_radius_frames=int(args.split_boundary_radius_frames),
        split_boundary_sigma_frames=float(args.split_boundary_sigma_frames),
        split_target_mode=str(args.split_target_mode),
    )
    print(f"summary={repo_display_path(project_path(args.output))}")
    groups = summary.get("groups", {})
    all_group = groups.get("ALL", {})
    distributions = all_group.get("distributions", {}) if isinstance(all_group, Mapping) else {}
    speech_hit = distributions.get("speech_hit_on_speech", {}).get("mean", 0.0)
    split_hit = distributions.get("split_hit_on_split", {}).get("mean", 0.0)
    print(
        "processed={processed} speech_recall={speech:.4f} split_recall={split:.4f}".format(
            processed=summary.get("processed_examples", 0),
            speech=float(speech_hit or 0.0),
            split=float(split_hit or 0.0),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
