#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.backends.qwen import qwen_asr_repo_id, qwen_asr_repo_tag  # noqa: E402
from asr.pre_asr_cueqc import (  # noqa: E402
    PRE_ASR_CUEQC_DEFAULT_DROP_THRESHOLD,
    PRE_ASR_CUEQC_FEATURE_SCHEMA,
    PRE_ASR_CUEQC_IGNORE_LABEL,
    PRE_ASR_CUEQC_MODEL_ARCH,
    PRE_ASR_CUEQC_PTM_BINS,
    PRE_ASR_CUEQC_PTM_DIM,
    PRE_ASR_CUEQC_RUNTIME_ADAPTER,
    PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES,
    PRE_ASR_CUEQC_SCHEMA,
    PreAsrCueQCMambaV10,
    make_model_config,
)
from tools.asr.cueqc.compile_pre_asr_v10_features import (  # noqa: E402
    FEATURE_BUNDLE_SCHEMA,
    project_path,
    repo_display_path,
)


METRICS_SCHEMA = "cueqc_pre_asr_mamba_v10_train_metrics"
DEFAULT_SWEEP_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)


def default_checkpoint_name(asr_repo_id: str) -> str:
    return f"cueqc_pre_asr_mamba_v10_binary.{qwen_asr_repo_tag(asr_repo_id)}.pt"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_feature_bundle(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("feature bundle must be a mapping")
    if payload.get("schema") != FEATURE_BUNDLE_SCHEMA:
        raise ValueError(f"unsupported feature bundle schema: {payload.get('schema')!r}")
    if tuple(payload.get("feature_names") or ()) != PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES:
        raise ValueError("feature bundle feature_names do not match Pre-ASR CueQC v10 runtime")
    if payload.get("feature_schema") != PRE_ASR_CUEQC_FEATURE_SCHEMA:
        raise ValueError("feature bundle feature_schema mismatch")
    if payload.get("runtime_adapter") != PRE_ASR_CUEQC_RUNTIME_ADAPTER:
        raise ValueError("feature bundle runtime_adapter mismatch")
    if int(payload.get("ptm_bins") or 0) != PRE_ASR_CUEQC_PTM_BINS:
        raise ValueError("feature bundle ptm_bins mismatch")
    if int(payload.get("ptm_dim") or 0) != PRE_ASR_CUEQC_PTM_DIM:
        raise ValueError("feature bundle ptm_dim mismatch")
    return dict(payload)


def _valid_flat(
    probs: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    durations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = (mask > 0) & (y != PRE_ASR_CUEQC_IGNORE_LABEL)
    return probs[valid], y[valid], durations[valid]


def classification_metrics(
    probs: np.ndarray,
    y: np.ndarray,
    durations: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    if probs.size == 0:
        return {
            "drop_precision": 0.0,
            "drop_recall": 0.0,
            "drop_f1": 0.0,
            "semantic_keep_recall": 0.0,
            "false_drop_rate": 0.0,
            "false_drop_count": 0.0,
            "false_drop_duration_s": 0.0,
            "false_keep_count": 0.0,
            "false_keep_duration_s": 0.0,
            "asr_time_saved_s": 0.0,
            "asr_time_saved_ratio": 0.0,
            "drop_duration_s": 0.0,
            "drop_chunk_ratio": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 0.0,
        }
    p_drop = probs[:, 0]
    pred_drop = p_drop >= threshold
    true_drop = y == 0
    true_keep = y == 1
    tp = int(np.sum(pred_drop & true_drop))
    fp = int(np.sum(pred_drop & true_keep))
    fn = int(np.sum(~pred_drop & true_drop))
    tn = int(np.sum(~pred_drop & true_keep))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    keep_recall = tn / (tn + fp) if tn + fp else 0.0
    false_drop_duration = float(np.sum(durations[pred_drop & true_keep]))
    false_keep_duration = float(np.sum(durations[~pred_drop & true_drop]))
    drop_duration = float(np.sum(durations[pred_drop]))
    total_duration = float(np.sum(durations))
    return {
        "drop_precision": precision,
        "drop_recall": recall,
        "drop_f1": f1,
        "semantic_keep_recall": keep_recall,
        "false_drop_rate": fp / max(1, int(np.sum(true_keep))),
        "false_drop_count": float(fp),
        "false_drop_duration_s": false_drop_duration,
        "false_keep_count": float(fn),
        "false_keep_duration_s": false_keep_duration,
        "asr_time_saved_s": drop_duration,
        "asr_time_saved_ratio": drop_duration / total_duration if total_duration > 0.0 else 0.0,
        "drop_duration_s": drop_duration,
        "drop_chunk_ratio": float(np.mean(pred_drop)) if pred_drop.size else 0.0,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _duration_matrix(bundle: Mapping[str, Any], scalar: Any) -> np.ndarray:
    features = tuple(str(item) for item in bundle.get("feature_names") or ())
    name = "refined_duration_s" if "refined_duration_s" in features else "duration_s"
    try:
        index = features.index(name)
    except ValueError:
        return np.zeros(tuple(scalar.shape[:2]), dtype=np.float32)
    return scalar[:, :, index].detach().cpu().numpy().astype(np.float32)


def _threshold_sweep(probs: np.ndarray, y: np.ndarray, mask: np.ndarray, durations: np.ndarray) -> dict[str, Any]:
    valid_probs, valid_y, valid_durations = _valid_flat(probs, y, mask, durations)
    return {
        f"{threshold:.2f}": classification_metrics(
            valid_probs,
            valid_y,
            valid_durations,
            threshold=threshold,
        )
        for threshold in DEFAULT_SWEEP_THRESHOLDS
    }


def train(
    *,
    features_path: Path,
    output_dir: Path,
    asr_repo_id: str,
    hidden_size: int,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str,
    drop_threshold: float,
    keep_class_weight: float,
    drop_class_weight: float,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    bundle = load_feature_bundle(features_path)
    scalar = bundle["scalar_features"].float()
    ptm_bins = bundle["ptm_bins"].float()
    bin_mask = bundle["bin_mask"].float()
    chunk_mask = bundle["chunk_mask"].float()
    y = bundle["labels"].long()
    if scalar.ndim != 3 or scalar.shape[2] != len(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES):
        raise ValueError("scalar feature tensor shape mismatch")
    if ptm_bins.ndim != 4 or ptm_bins.shape[2:] != (PRE_ASR_CUEQC_PTM_BINS, PRE_ASR_CUEQC_PTM_DIM):
        raise ValueError("ptm bin tensor shape mismatch")
    if y.shape != chunk_mask.shape or y.shape != scalar.shape[:2]:
        raise ValueError("label tensor shape mismatch")
    selected_repo = qwen_asr_repo_id(asr_repo_id)
    bundle_repo = qwen_asr_repo_id(str(bundle.get("asr_repo_id") or selected_repo))
    if bundle_repo != selected_repo:
        raise ValueError(f"feature bundle asr_repo_id={bundle_repo!r} does not match {selected_repo!r}")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    group_count = int(scalar.shape[0])
    order = rng.permutation(group_count)
    val_count = max(1, int(round(group_count * 0.15))) if group_count >= 8 else max(1, group_count // 4)
    val_idx = torch.as_tensor(order[:val_count], dtype=torch.long)
    train_idx = torch.as_tensor(order[val_count:] if val_count < group_count else order, dtype=torch.long)

    train_valid = chunk_mask[train_idx].bool()
    if not torch.any((y[train_idx] != PRE_ASR_CUEQC_IGNORE_LABEL) & train_valid):
        raise ValueError("training split has no definite keep/drop labels")
    scalar_train = scalar[train_idx][train_valid]
    mean = scalar_train.mean(dim=0)
    std = scalar_train.std(dim=0).clamp_min(1e-6)
    scalar_norm = (scalar - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    scalar_norm = torch.nan_to_num(scalar_norm, nan=0.0, posinf=0.0, neginf=0.0)

    normalized_device = device.strip().lower()
    if normalized_device == "auto":
        normalized_device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(normalized_device)
    model_config = make_model_config(
        {
            "ptm_dim": PRE_ASR_CUEQC_PTM_DIM,
            "scalar_dim": len(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES),
            "hidden_size": hidden_size,
        }
    )
    model = PreAsrCueQCMambaV10(**model_config).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = torch.tensor(
        [float(drop_class_weight), float(keep_class_weight)],
        dtype=torch.float32,
        device=dev,
    )
    train_idx = train_idx.to(dev)
    ptm_bins_d = ptm_bins.to(dev)
    scalar_d = scalar_norm.to(dev)
    chunk_mask_d = chunk_mask.to(dev)
    bin_mask_d = bin_mask.to(dev)
    y_d = y.to(dev)
    batch_size = max(1, int(batch_size))
    for _step in range(max(1, int(steps))):
        sample = torch.randint(0, train_idx.shape[0], (min(batch_size, train_idx.shape[0]),), device=dev)
        group_ids = train_idx[sample]
        logits = model(
            ptm_bins_d[group_ids],
            scalar_d[group_ids],
            chunk_mask=chunk_mask_d[group_ids],
            bin_mask=bin_mask_d[group_ids],
        )
        loss = F.cross_entropy(
            logits.reshape(-1, 2),
            y_d[group_ids].reshape(-1),
            weight=class_weights,
            ignore_index=PRE_ASR_CUEQC_IGNORE_LABEL,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    with torch.inference_mode():
        logits_all = model(
            ptm_bins_d,
            scalar_d,
            chunk_mask=chunk_mask_d,
            bin_mask=bin_mask_d,
        )
        probs_all = torch.softmax(logits_all, dim=-1).float().cpu().numpy()
    y_np = y.numpy()
    mask_np = chunk_mask.numpy()
    durations = _duration_matrix(bundle, scalar)
    val_mask = np.zeros((group_count,), dtype=bool)
    val_mask[val_idx.numpy()] = True
    val_probs, val_y, val_durations = _valid_flat(
        probs_all[val_mask],
        y_np[val_mask],
        mask_np[val_mask],
        durations[val_mask],
    )
    all_probs, all_y, all_durations = _valid_flat(probs_all, y_np, mask_np, durations)
    created_at = datetime.now().isoformat(timespec="seconds")
    metrics = {
        "schema": METRICS_SCHEMA,
        "created_at": created_at,
        "features": repo_display_path(features_path),
        "feature_sha256": file_sha256(features_path),
        "asr_repo_id": selected_repo,
        "train_group_count": int(train_idx.shape[0]),
        "val_group_count": int(val_idx.shape[0]),
        "all_group_count": group_count,
        "class_counts": {
            "drop": int(np.sum(y_np == 0)),
            "keep": int(np.sum(y_np == 1)),
            "ambiguous_ignore": int(np.sum((y_np == PRE_ASR_CUEQC_IGNORE_LABEL) & (mask_np > 0))),
        },
        "class_weights": {
            "drop": float(drop_class_weight),
            "keep": float(keep_class_weight),
        },
        "drop_threshold": float(drop_threshold),
        "threshold_sweep": _threshold_sweep(probs_all, y_np, mask_np, durations),
        "val": classification_metrics(val_probs, val_y, val_durations, threshold=drop_threshold),
        "all": classification_metrics(all_probs, all_y, all_durations, threshold=drop_threshold),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / default_checkpoint_name(selected_repo)
    checkpoint = {
        "schema": PRE_ASR_CUEQC_SCHEMA,
        "arch": PRE_ASR_CUEQC_MODEL_ARCH,
        "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
        "runtime_adapter": PRE_ASR_CUEQC_RUNTIME_ADAPTER,
        "feature_names": list(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES),
        "model_config": model_config,
        "feature_mean": mean.cpu().numpy().astype(np.float32).tolist(),
        "feature_std": std.cpu().numpy().astype(np.float32).tolist(),
        "decision_config": {
            "drop_threshold": float(drop_threshold),
            "hard_keep_veto": True,
            "hard_drop_rule": True,
            "keep_veto": True,
        },
        "metadata": {
            "asr_repo_id": selected_repo,
            "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
            "runtime_adapter": PRE_ASR_CUEQC_RUNTIME_ADAPTER,
            "feature_bundle": repo_display_path(features_path),
            "feature_bundle_sha256": file_sha256(features_path),
            "trained_steps": int(steps),
            "created_at": created_at,
            "ignore_label": PRE_ASR_CUEQC_IGNORE_LABEL,
        },
        "model_state_dict": model.cpu().state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    metrics["checkpoint"] = repo_display_path(checkpoint_path)
    metrics["checkpoint_sha256"] = file_sha256(checkpoint_path)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pre-ASR CueQC v10 hierarchical Mamba2 binary checkpoint.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asr-repo-id", required=True)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--drop-threshold", type=float, default=PRE_ASR_CUEQC_DEFAULT_DROP_THRESHOLD)
    parser.add_argument("--drop-class-weight", type=float, default=1.0)
    parser.add_argument("--keep-class-weight", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 0.0 <= args.drop_threshold <= 1.0:
        parser.error("--drop-threshold must be between 0 and 1")
    if args.drop_class_weight <= 0.0 or args.keep_class_weight <= 0.0:
        parser.error("class weights must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = train(
        features_path=project_path(args.features),
        output_dir=project_path(args.output_dir),
        asr_repo_id=str(args.asr_repo_id),
        hidden_size=int(args.hidden_size),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        device=str(args.device),
        drop_threshold=float(args.drop_threshold),
        keep_class_weight=float(args.keep_class_weight),
        drop_class_weight=float(args.drop_class_weight),
    )
    print(
        "checkpoint={checkpoint} val_drop_f1={f1:.4f} val_keep_recall={keep:.4f}".format(
            checkpoint=metrics["checkpoint"],
            f1=float(metrics["val"]["drop_f1"]),
            keep=float(metrics["val"]["semantic_keep_recall"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
