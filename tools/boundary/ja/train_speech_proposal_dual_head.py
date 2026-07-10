#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for _root in (SRC_ROOT, PROJECT_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from asr.backends.qwen import qwen_asr_repo_tag  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.ja.dataset import effective_frame_weights, read_jsonl  # noqa: E402
from boundary.ja.dual_head import (  # noqa: E402
    SPEECH_PROPOSAL_DUAL_HEAD_MODEL_ARCH,
    SPEECH_PROPOSAL_DUAL_HEAD_NORMALIZATION_SCHEMA,
    build_dual_head_checkpoint,
    build_dual_head_model,
)
from boundary.ja.model import load_speech_island_scorer_checkpoint  # noqa: E402
from boundary.ja.features import load_cached_feature  # noqa: E402
from boundary.ja.proposal import load_boundary_proposal_checkpoint  # noqa: E402
from tools.boundary.ja.build_runtime_semantic_split_dataset import (  # noqa: E402
    semantic_split_truth_boundaries,
)
from tools.boundary.ja.train_boundary_proposal_scorer import (  # noqa: E402
    boundary_target_frames,
)


def _manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _partition(record: Any) -> str:
    return str((record.boundary_metadata or {}).get("source_partition") or "train")


def _trim_process_working_set() -> None:
    if sys.platform == "win32":
        ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())


def _memory_usage_mib() -> tuple[float, float]:
    import psutil

    process_rss = psutil.Process().memory_info().rss / (1024 ** 2)
    memory = psutil.virtual_memory()
    physical_used = (memory.total - memory.available) / (1024 ** 2)
    return process_rss, physical_used


def _crop(
    arrays: Sequence[np.ndarray], *, max_frames: int, rng: np.random.Generator
) -> list[np.ndarray]:
    if arrays[0].shape[0] <= max_frames:
        return [np.asarray(value) for value in arrays]
    start = int(rng.integers(0, arrays[0].shape[0] - max_frames + 1))
    return [np.asarray(value[start : start + max_frames]) for value in arrays]


class PackedFeatureCache:
    """Disk-backed full-PTM cache for a deterministic Stage F schedule."""

    def __init__(self, cache_dir: Path) -> None:
        index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
        self.raw_ptm_dim = int(index["raw_ptm_dim"])
        self.mfcc_dim = int(index["mfcc_dim"])
        self.total_frames = int(index["total_frames"])
        self.entries = {
            str(key): (int(value["offset"]), int(value["frame_count"]))
            for key, value in dict(index["entries"]).items()
        }
        self.ptm_path = cache_dir / "ptm.float32.dat"
        self.mfcc_path = cache_dir / "mfcc.float32.dat"

    def __getitem__(self, path_text: str) -> tuple[np.ndarray, np.ndarray]:
        offset, count = self.entries[path_text]
        ptm = np.fromfile(
            self.ptm_path,
            dtype=np.float32,
            count=count * self.raw_ptm_dim,
            offset=offset * self.raw_ptm_dim * np.dtype(np.float32).itemsize,
        ).reshape(count, self.raw_ptm_dim)
        mfcc = np.fromfile(
            self.mfcc_path,
            dtype=np.float32,
            count=count * self.mfcc_dim,
            offset=offset * self.mfcc_dim * np.dtype(np.float32).itemsize,
        ).reshape(count, self.mfcc_dim)
        return ptm, mfcc


def _pack_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    cache_dir: Path,
    raw_ptm_dim: int,
    mfcc_dim: int,
    workers: int,
) -> PackedFeatureCache:
    frame_counts: dict[str, int] = {}
    for row in rows:
        path_text = str(row["feature_path"])
        frame_counts[path_text] = int(row["frame_count"])
    paths = sorted(frame_counts)
    digest = hashlib.sha256(
        json.dumps(
            {
                "paths": [[path, frame_counts[path]] for path in paths],
                "raw_ptm_dim": raw_ptm_dim,
                "mfcc_dim": mfcc_dim,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    index_path = cache_dir / "index.json"
    if index_path.exists():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if str(existing.get("source_digest") or "") != digest:
            raise ValueError("packed feature cache source digest mismatch")
        print("dual_head_pack=reused", flush=True)
        return PackedFeatureCache(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, int]] = {}
    offset = 0
    for path_text in paths:
        count = frame_counts[path_text]
        entries[path_text] = {"offset": offset, "frame_count": count}
        offset += count
    total_frames = offset
    ptm_path = cache_dir / "ptm.float32.dat"
    mfcc_path = cache_dir / "mfcc.float32.dat"
    with ptm_path.open("wb") as handle:
        handle.truncate(total_frames * raw_ptm_dim * np.dtype(np.float32).itemsize)
    with mfcc_path.open("wb") as handle:
        handle.truncate(total_frames * mfcc_dim * np.dtype(np.float32).itemsize)

    def load(path_text: str) -> tuple[str, np.ndarray, np.ndarray]:
        ptm, mfcc = load_cached_feature(Path(path_text))
        expected = frame_counts[path_text]
        if ptm.shape != (expected, raw_ptm_dim):
            raise ValueError(
                f"feature PTM shape {ptm.shape} != {(expected, raw_ptm_dim)}: {path_text}"
            )
        if mfcc.shape != (expected, mfcc_dim):
            raise ValueError(f"feature MFCC shape {mfcc.shape} != {(expected, mfcc_dim)}")
        return path_text, ptm, mfcc

    completed = 0
    worker_count = max(1, workers)
    with (
        ThreadPoolExecutor(max_workers=worker_count) as pool,
        ptm_path.open("r+b") as ptm_out,
        mfcc_path.open("r+b") as mfcc_out,
    ):
        for batch_start in range(0, len(paths), worker_count * 2):
            batch = paths[batch_start : batch_start + worker_count * 2]
            for path_text, ptm, mfcc in pool.map(load, batch):
                entry = entries[path_text]
                start = entry["offset"]
                end = start + entry["frame_count"]
                ptm_out.seek(start * raw_ptm_dim * np.dtype(np.float32).itemsize)
                np.asarray(ptm, dtype=np.float32).tofile(ptm_out)
                mfcc_out.seek(start * mfcc_dim * np.dtype(np.float32).itemsize)
                np.asarray(mfcc, dtype=np.float32).tofile(mfcc_out)
                completed += 1
                if completed % 250 == 0 or completed == len(paths):
                    print(
                        f"dual_head_pack={completed}/{len(paths)} "
                        f"written_gib={(end * (raw_ptm_dim + mfcc_dim) * 4) / (1024 ** 3):.3f}",
                        flush=True,
                    )
    index_path.write_text(
        json.dumps(
            {
                "schema": "speech_proposal_dual_head_packed_features_v1",
                "source_digest": digest,
                "raw_ptm_dim": raw_ptm_dim,
                "mfcc_dim": mfcc_dim,
                "total_frames": total_frames,
                "entries": entries,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return PackedFeatureCache(cache_dir)


def _speech_arrays(
    row: Mapping[str, Any],
    records: list[Any],
    cache: PackedFeatureCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    record = records[int(row["label_index"])]
    ptm, mfcc = cache[str(row["feature_path"])]
    weights = np.asarray(effective_frame_weights(record), dtype=np.float32)
    count = min(len(record.speech_frames), len(weights), ptm.shape[0], mfcc.shape[0])
    labels = np.asarray(record.speech_frames[:count], dtype=np.float32)
    return ptm[:count], mfcc[:count], labels, weights[:count]


def _proposal_arrays(
    row: Mapping[str, Any],
    records: list[Any],
    cache: PackedFeatureCache,
    *,
    radius_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    record = records[int(row["label_index"])]
    ptm, mfcc = cache[str(row["feature_path"])]
    count = min(int(ptm.shape[0]), int(mfcc.shape[0]))
    boundaries = [
        float(item["time_s"])
        for item in semantic_split_truth_boundaries(dict(record.boundary_metadata or {}))
        if 0.0 < float(item["time_s"]) < record.duration_s
    ]
    targets = boundary_target_frames(
        boundary_times_s=boundaries,
        frame_count=count,
        frame_hop_s=record.frame_hop_s,
        radius_frames=radius_frames,
    )
    return ptm[:count], mfcc[:count], targets


def initialize_dual_head_from_teachers(
    dual: Any,
    speech_model: Any,
    proposal_model: Any,
    speech_normalization: Mapping[str, Any],
) -> None:
    import torch

    dual.encoder.proj.load_state_dict(speech_model.proj.state_dict())
    dual.encoder.backbone.load_state_dict(speech_model.backbone.state_dict())
    dual.encoder.norm.load_state_dict(speech_model.norm.state_dict())
    mean = np.asarray(speech_normalization["feature_mean"], dtype=np.float32)
    std = np.maximum(
        np.asarray(speech_normalization["feature_std"], dtype=np.float32), 1e-6
    )
    projected_dim = int(dual.projected_ptm_dim)
    if projected_dim != int(speech_model.model_config["input_dim"]) - dual.mfcc_dim:
        raise ValueError("dual-head projected PTM dimension differs from speech teacher")
    with torch.no_grad():
        dual.ptm_projector.weight.zero_()
        device = dual.ptm_projector.weight.device
        diagonal = torch.arange(projected_dim, device=device)
        dual.ptm_projector.weight[diagonal, diagonal] = torch.from_numpy(
            1.0 / std[:projected_dim]
        ).to(device)
        dual.ptm_projector.bias.copy_(
            torch.from_numpy(-mean[:projected_dim] / std[:projected_dim]).to(device)
        )
        dual.encoder.head.weight[0].copy_(speech_model.head.weight[0])
        dual.encoder.head.bias[0].copy_(speech_model.head.bias[0])
        dual.encoder.head.weight[1].copy_(proposal_model.head.weight[0])
        dual.encoder.head.bias[1].copy_(proposal_model.head.bias[0])


def _teacher_features(
    ptm: np.ndarray,
    mfcc: np.ndarray,
    *,
    ptm_dim: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    features = np.concatenate((ptm[:, :ptm_dim], mfcc), axis=1).astype(np.float32)
    return (features - mean) / std


def _focal_loss(
    logits: Any,
    targets: Any,
    *,
    positive_weight: float,
    negative_weight: float,
    focal_gamma: float,
) -> Any:
    import torch
    import torch.nn.functional as F

    raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probabilities, 1.0 - probabilities)
    weights = torch.where(
        targets > 0.5,
        torch.full_like(targets, positive_weight),
        torch.full_like(targets, negative_weight),
    )
    return (raw * torch.pow(1.0 - pt, focal_gamma) * weights).sum() / weights.sum().clamp_min(1e-6)


def _metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
        "positive_count": int(truth.sum()),
        "predicted_positive_count": int(predicted.sum()),
    }


def _evaluate_speech(
    *,
    dual: Any,
    teacher: Any,
    records: list[Any],
    rows: Sequence[Mapping[str, Any]],
    feature_cache: PackedFeatureCache,
    teacher_mean: np.ndarray,
    teacher_std: np.ndarray,
    device: Any,
    teacher_ptm_dim: int,
    threshold: float,
) -> dict[str, Any]:
    import torch

    truth_rows: list[np.ndarray] = []
    dual_rows: list[np.ndarray] = []
    teacher_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for row in rows:
            ptm, mfcc, labels, _weights = _speech_arrays(
                row, records, feature_cache
            )
            dual_prob = (
                torch.sigmoid(
                    dual(
                        torch.from_numpy(np.array(ptm, copy=True)).unsqueeze(0).to(device),
                        torch.from_numpy(np.array(mfcc, copy=True)).unsqueeze(0).to(device),
                    )[0, :, 0]
                )
                .cpu()
                .numpy()
            )
            teacher_input = _teacher_features(
                ptm,
                mfcc,
                ptm_dim=teacher_ptm_dim,
                mean=teacher_mean,
                std=teacher_std,
            )
            teacher_prob = (
                torch.sigmoid(
                    teacher.model(
                        torch.from_numpy(teacher_input).unsqueeze(0).to(device)
                    )[0, :, 0]
                )
                .cpu()
                .numpy()
            )
            truth_rows.append(labels >= 0.5)
            dual_rows.append(dual_prob)
            teacher_rows.append(teacher_prob)
    truth = np.concatenate(truth_rows)
    dual_prob = np.concatenate(dual_rows)
    teacher_prob = np.concatenate(teacher_rows)
    return {
        "dual": _metrics(truth, dual_prob >= threshold),
        "teacher": _metrics(truth, teacher_prob >= threshold),
        "teacher_agreement": float(
            np.mean((dual_prob >= threshold) == (teacher_prob >= threshold))
        ),
        "mean_abs_probability_delta": float(np.mean(np.abs(dual_prob - teacher_prob))),
        "frame_count": int(truth.size),
    }


def _evaluate_proposal(
    *,
    dual: Any,
    teacher: Any,
    records: list[Any],
    rows: Sequence[Mapping[str, Any]],
    feature_cache: PackedFeatureCache,
    teacher_mean: np.ndarray,
    teacher_std: np.ndarray,
    device: Any,
    teacher_ptm_dim: int,
    radius_frames: int,
    threshold: float,
) -> dict[str, Any]:
    import torch

    truth_rows: list[np.ndarray] = []
    dual_rows: list[np.ndarray] = []
    teacher_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for row in rows:
            ptm, mfcc, labels = _proposal_arrays(
                row, records, feature_cache, radius_frames=radius_frames
            )
            dual_input = (
                torch.from_numpy(np.array(ptm, copy=True)).unsqueeze(0).to(device),
                torch.from_numpy(np.array(mfcc, copy=True)).unsqueeze(0).to(device),
            )
            teacher_features = _teacher_features(
                ptm,
                mfcc,
                ptm_dim=teacher_ptm_dim,
                mean=teacher_mean,
                std=teacher_std,
            )
            dual_prob = torch.sigmoid(dual(*dual_input)[0, :, 1]).cpu().numpy()
            teacher_prob = (
                torch.sigmoid(
                    teacher.model(
                        torch.from_numpy(teacher_features).unsqueeze(0).to(device)
                    )[0, :, 0]
                )
                .cpu()
                .numpy()
            )
            truth_rows.append(labels >= 0.5)
            dual_rows.append(dual_prob)
            teacher_rows.append(teacher_prob)
    truth = np.concatenate(truth_rows)
    dual_prob = np.concatenate(dual_rows)
    teacher_prob = np.concatenate(teacher_rows)
    return {
        "dual": _metrics(truth, dual_prob >= threshold),
        "teacher": _metrics(truth, teacher_prob >= threshold),
        "teacher_agreement": float(
            np.mean((dual_prob >= threshold) == (teacher_prob >= threshold))
        ),
        "mean_abs_probability_delta": float(np.mean(np.abs(dual_prob - teacher_prob))),
        "frame_count": int(truth.size),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    vram_safety_ratio = apply_vram_safety_cap()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    speech_teacher = load_speech_island_scorer_checkpoint(
        args.speech_checkpoint, device=device
    )
    proposal_teacher = load_boundary_proposal_checkpoint(
        args.proposal_checkpoint, device=device
    )
    if speech_teacher.model_config["input_dim"] != proposal_teacher.model_config["input_dim"]:
        raise ValueError("teacher input dimensions differ")
    if speech_teacher.metadata.get("ptm_repo_id") != proposal_teacher.metadata.get("ptm_repo_id"):
        raise ValueError("teacher PTM repos differ")

    speech_records = read_jsonl(Path(args.speech_labels))
    proposal_records = read_jsonl(Path(args.proposal_labels))
    speech_rows = _manifest(Path(args.speech_feature_manifest))
    proposal_rows = [
        row
        for row in _manifest(Path(args.proposal_feature_manifest))
        if proposal_records[int(row["label_index"])].boundary_metadata
    ]
    speech_train = [row for row in speech_rows if _partition(speech_records[int(row["label_index"])]) != "val"]
    speech_val = [row for row in speech_rows if _partition(speech_records[int(row["label_index"])]) == "val"]
    proposal_train = [row for row in proposal_rows if _partition(proposal_records[int(row["label_index"])]) != "val"]
    proposal_val = [row for row in proposal_rows if _partition(proposal_records[int(row["label_index"])]) == "val"]
    if not all((speech_train, speech_val, proposal_train, proposal_val)):
        raise ValueError("both tasks require non-empty train and val partitions")

    base = speech_teacher.model_config
    speech_ptm_dims = {int(row["ptm_dim"]) for row in speech_rows}
    proposal_ptm_dims = {int(row["ptm_dim"]) for row in proposal_rows}
    if len(speech_ptm_dims) != 1 or len(proposal_ptm_dims) != 1:
        raise ValueError("dual-head manifests must be hydrated to one raw PTM dimension")
    raw_ptm_dim = next(iter(speech_ptm_dims))
    if raw_ptm_dim != next(iter(proposal_ptm_dims)):
        raise ValueError("speech/proposal raw PTM dimensions differ")
    if int(base["ptm_dim"]) >= raw_ptm_dim:
        raise ValueError("dual-head training requires full PTM beyond the teacher prefix")
    mfcc_dim = int(base["mfcc_dim"])
    speech_mean = np.asarray(
        speech_teacher.normalization["feature_mean"], dtype=np.float32
    )
    speech_std = np.maximum(
        np.asarray(speech_teacher.normalization["feature_std"], dtype=np.float32),
        1e-6,
    )
    dual_normalization = {
        "schema": SPEECH_PROPOSAL_DUAL_HEAD_NORMALIZATION_SCHEMA,
        "mfcc_mean": speech_mean[int(base["ptm_dim"]) :].tolist(),
        "mfcc_std": speech_std[int(base["ptm_dim"]) :].tolist(),
    }
    model_config = {
        "raw_ptm_dim": raw_ptm_dim,
        "projected_ptm_dim": int(base["ptm_dim"]),
        "mfcc_dim": mfcc_dim,
        "input_dim": raw_ptm_dim + mfcc_dim,
        "encoder_input_dim": int(base["input_dim"]),
        "hidden_size": int(base["hidden_size"]),
        "num_layers": int(base["num_layers"]),
        "state_size": int(base.get("state_size", 32)),
        "num_heads": int(base.get("num_heads", 4)),
        "n_groups": int(base.get("n_groups", 2)),
        "chunk_size": int(base.get("chunk_size", 8)),
        "conv_kernel": int(base.get("conv_kernel", 4)),
        "bidirectional": bool(base.get("bidirectional", True)),
        "model_arch": SPEECH_PROPOSAL_DUAL_HEAD_MODEL_ARCH,
        "output_dim": 2,
    }
    dual = build_dual_head_model(model_config, dual_normalization).to(device)
    initialize_dual_head_from_teachers(
        dual,
        speech_teacher.model,
        proposal_teacher.model,
        speech_teacher.normalization,
    )
    speech_teacher.model.eval()
    proposal_teacher.model.eval()
    proposal_mean = np.asarray(
        proposal_teacher.normalization["feature_mean"], dtype=np.float32
    )
    proposal_std = np.maximum(
        np.asarray(proposal_teacher.normalization["feature_std"], dtype=np.float32),
        1e-6,
    )
    schedule_rng = np.random.default_rng(args.seed)
    crop_rng = np.random.default_rng(args.seed + 1)
    torch.manual_seed(args.seed)
    schedule: list[tuple[str, dict[str, Any]]] = []
    for _step in range(args.max_steps):
        if schedule_rng.random() < args.proposal_ratio:
            schedule.append(
                (
                    "proposal",
                    proposal_train[int(schedule_rng.integers(0, len(proposal_train)))],
                )
            )
        else:
            schedule.append(
                (
                    "speech",
                    speech_train[int(schedule_rng.integers(0, len(speech_train)))],
                )
            )
    preload_rows = [row for _task, row in schedule]
    preload_rows.extend(speech_val[: args.max_eval_windows])
    preload_rows.extend(proposal_val[: args.max_eval_windows])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache = _pack_features(
        preload_rows,
        cache_dir=output_dir / "packed-feature-cache",
        raw_ptm_dim=model_config["raw_ptm_dim"],
        mfcc_dim=model_config["mfcc_dim"],
        workers=args.preload_workers,
    )
    _trim_process_working_set()
    process_rss_peak_mib, physical_ram_used_peak_mib = _memory_usage_mib()
    import psutil

    physical_ram_cap_mib = (
        psutil.virtual_memory().total * vram_safety_ratio / (1024 ** 2)
    )
    optimizer = torch.optim.AdamW(dual.parameters(), lr=args.learning_rate)
    losses: list[float] = []
    task_counts = {"speech": 0, "proposal": 0}
    started = time.monotonic()
    dual.train()
    for step, (task, row) in enumerate(schedule, start=1):
        proposal_task = task == "proposal"
        if proposal_task:
            task_counts["proposal"] += 1
            ptm, mfcc, labels = _proposal_arrays(
                row,
                proposal_records,
                feature_cache,
                radius_frames=args.boundary_radius_frames,
            )
            ptm, mfcc, labels = _crop(
                [ptm, mfcc, labels],
                max_frames=args.max_train_frames,
                rng=crop_rng,
            )
            dual_ptm = torch.from_numpy(np.array(ptm, copy=True)).unsqueeze(0).to(device)
            dual_mfcc = (
                torch.from_numpy(np.array(mfcc, copy=True)).unsqueeze(0).to(device)
            )
            teacher_input = torch.from_numpy(
                _teacher_features(
                    ptm,
                    mfcc,
                    ptm_dim=int(proposal_teacher.model_config["ptm_dim"]),
                    mean=proposal_mean,
                    std=proposal_std,
                )
            ).unsqueeze(0).to(device)
            targets = torch.from_numpy(labels).unsqueeze(0).unsqueeze(-1).to(device)
            logits = dual(dual_ptm, dual_mfcc)[..., 1:2]
            with torch.no_grad():
                teacher_prob = torch.sigmoid(proposal_teacher.model(teacher_input))
            supervised = _focal_loss(
                logits,
                targets,
                positive_weight=args.proposal_positive_weight,
                negative_weight=1.0,
                focal_gamma=args.focal_gamma,
            )
            distill = F.binary_cross_entropy_with_logits(logits, teacher_prob)
            loss = supervised + args.distill_weight * distill
        else:
            task_counts["speech"] += 1
            ptm, mfcc, labels, weights = _speech_arrays(
                row, speech_records, feature_cache
            )
            ptm, mfcc, labels, weights = _crop(
                [ptm, mfcc, labels, weights],
                max_frames=args.max_train_frames,
                rng=crop_rng,
            )
            dual_ptm = torch.from_numpy(np.array(ptm, copy=True)).unsqueeze(0).to(device)
            dual_mfcc = (
                torch.from_numpy(np.array(mfcc, copy=True)).unsqueeze(0).to(device)
            )
            teacher_input = torch.from_numpy(
                _teacher_features(
                    ptm,
                    mfcc,
                    ptm_dim=int(speech_teacher.model_config["ptm_dim"]),
                    mean=speech_mean,
                    std=speech_std,
                )
            ).unsqueeze(0).to(device)
            targets = torch.from_numpy(labels).unsqueeze(0).unsqueeze(-1).to(device)
            frame_weights = (
                torch.from_numpy(weights).unsqueeze(0).unsqueeze(-1).to(device)
            )
            logits = dual(dual_ptm, dual_mfcc)[..., 0:1]
            with torch.no_grad():
                teacher_prob = torch.sigmoid(speech_teacher.model(teacher_input))
            supervised_raw = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            probabilities = torch.sigmoid(logits)
            pt = torch.where(targets > 0.5, probabilities, 1.0 - probabilities)
            class_weights = torch.where(
                targets > 0.5,
                torch.full_like(targets, args.speech_positive_weight),
                torch.full_like(targets, args.speech_negative_weight),
            )
            effective = frame_weights * class_weights
            supervised = (
                supervised_raw * torch.pow(1.0 - pt, args.focal_gamma) * effective
            ).sum() / effective.sum().clamp_min(1e-6)
            distill = F.binary_cross_entropy_with_logits(logits, teacher_prob)
            loss = supervised + args.distill_weight * distill
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if args.log_every and step % args.log_every == 0:
            _trim_process_working_set()
            process_rss_mib, physical_ram_used_mib = _memory_usage_mib()
            process_rss_peak_mib = max(process_rss_peak_mib, process_rss_mib)
            physical_ram_used_peak_mib = max(
                physical_ram_used_peak_mib, physical_ram_used_mib
            )
            if physical_ram_used_mib > physical_ram_cap_mib:
                raise MemoryError(
                    f"physical RAM soft OOM: {physical_ram_used_mib:.1f} MiB "
                    f"> {physical_ram_cap_mib:.1f} MiB"
                )
            print(
                f"dual_head_train={step}/{args.max_steps} "
                f"loss={losses[-1]:.6f} avg={np.mean(losses[-args.log_every:]):.6f} "
                f"speech={task_counts['speech']} proposal={task_counts['proposal']} "
                f"rss_mib={process_rss_mib:.1f} ram_used_mib={physical_ram_used_mib:.1f} "
                f"elapsed_s={time.monotonic() - started:.1f}",
                flush=True,
            )

    dual.eval()
    speech_eval = _evaluate_speech(
        dual=dual,
        teacher=speech_teacher,
        records=speech_records,
        rows=speech_val[: args.max_eval_windows],
        feature_cache=feature_cache,
        teacher_mean=speech_mean,
        teacher_std=speech_std,
        device=device,
        teacher_ptm_dim=int(speech_teacher.model_config["ptm_dim"]),
        threshold=args.speech_threshold,
    )
    proposal_eval = _evaluate_proposal(
        dual=dual,
        teacher=proposal_teacher,
        records=proposal_records,
        rows=proposal_val[: args.max_eval_windows],
        feature_cache=feature_cache,
        teacher_mean=proposal_mean,
        teacher_std=proposal_std,
        device=device,
        teacher_ptm_dim=int(proposal_teacher.model_config["ptm_dim"]),
        radius_frames=args.boundary_radius_frames,
        threshold=args.proposal_threshold,
    )
    repo_id = str(speech_teacher.metadata.get("ptm_repo_id") or "")
    checkpoint_path = output_dir / (
        f"speech_proposal_dual_head_v1.{qwen_asr_repo_tag(repo_id)}.pt"
    )
    metadata = {
        "ptm_repo_id": repo_id,
        "trained_steps": args.max_steps,
        "speech_teacher_checkpoint": args.speech_checkpoint,
        "speech_teacher_sha256": speech_teacher.sha256,
        "proposal_teacher_checkpoint": args.proposal_checkpoint,
        "proposal_teacher_sha256": proposal_teacher.sha256,
        "speech_labels": args.speech_labels,
        "speech_feature_manifest": args.speech_feature_manifest,
        "proposal_labels": args.proposal_labels,
        "proposal_feature_manifest": args.proposal_feature_manifest,
        "boundary_radius_frames": args.boundary_radius_frames,
        "ptm_projection_init": "speech_teacher_normalized_prefix_identity",
        "vram_safety_ratio": vram_safety_ratio,
    }
    torch.save(
        build_dual_head_checkpoint(
            model=dual,
            model_config=model_config,
            normalization=dual_normalization,
            metadata=metadata,
        ),
        checkpoint_path,
    )
    metrics = {
        "schema": "speech_proposal_dual_head_train_v1",
        "checkpoint": str(checkpoint_path),
        "loss": float(np.mean(losses)),
        "steps": args.max_steps,
        "task_counts": task_counts,
        "speech": speech_eval,
        "proposal": proposal_eval,
        "vram_safety_ratio": vram_safety_ratio,
        "shared_vram_budget": False,
        "process_rss_peak_mib": process_rss_peak_mib,
        "physical_ram_used_peak_mib": physical_ram_used_peak_mib,
        "physical_ram_cap_mib": physical_ram_cap_mib,
    }
    if device.type == "cuda":
        metrics["cuda_peak_allocated_mib"] = float(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )
        metrics["cuda_peak_reserved_mib"] = float(
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        )
        metrics["dedicated_vram_cap_mib"] = float(
            torch.cuda.get_device_properties(device).total_memory
            * vram_safety_ratio
            / (1024 ** 2)
        )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "command.txt").write_text(
        " ".join(sys.argv) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stage F speech/proposal dual head.")
    parser.add_argument("--speech-checkpoint", required=True)
    parser.add_argument("--proposal-checkpoint", required=True)
    parser.add_argument("--speech-labels", required=True)
    parser.add_argument("--speech-feature-manifest", required=True)
    parser.add_argument("--proposal-labels", required=True)
    parser.add_argument("--proposal-feature-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--proposal-ratio", type=float, default=0.5)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--speech-positive-weight", type=float, default=2.0)
    parser.add_argument("--speech-negative-weight", type=float, default=8.0)
    parser.add_argument("--proposal-positive-weight", type=float, default=30.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--boundary-radius-frames", type=int, default=1)
    parser.add_argument("--max-train-frames", type=int, default=1024)
    parser.add_argument("--max-eval-windows", type=int, default=256)
    parser.add_argument("--preload-workers", type=int, default=6)
    parser.add_argument("--speech-threshold", type=float, default=0.5)
    parser.add_argument("--proposal-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args(argv)
    if not 0.0 < args.proposal_ratio < 1.0:
        parser.error("--proposal-ratio must be in (0, 1)")
    return args


if __name__ == "__main__":
    run(parse_args())
