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
    PRE_ASR_CUEQC_FEATURE_NAMES,
    PRE_ASR_CUEQC_FEATURE_SCHEMA,
    PRE_ASR_CUEQC_SCHEMA,
)
from tools.asr.cueqc.compile_pre_asr_v6_features import FEATURE_BUNDLE_SCHEMA, project_path, repo_display_path  # noqa: E402


METRICS_SCHEMA = "cueqc_pre_asr_mamba_v6_train_metrics"


def default_checkpoint_name(asr_repo_id: str) -> str:
    return f"cueqc_pre_asr_mamba_v6_binary.{qwen_asr_repo_tag(asr_repo_id)}.pt"


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
    if tuple(payload.get("feature_names") or ()) != PRE_ASR_CUEQC_FEATURE_NAMES:
        raise ValueError("feature bundle feature_names do not match Pre-ASR CueQC v6 runtime")
    if payload.get("feature_schema") != PRE_ASR_CUEQC_FEATURE_SCHEMA:
        raise ValueError("feature bundle feature_schema mismatch")
    return dict(payload)


def make_model(input_dim: int, hidden_size: int):
    from torch import nn

    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, 2),
    )


def classification_metrics(probs: np.ndarray, y: np.ndarray, *, threshold: float) -> dict[str, float]:
    p_drop = probs[:, 0]
    pred_drop = p_drop >= threshold
    true_drop = y == 0
    tp = int(np.sum(pred_drop & true_drop))
    fp = int(np.sum(pred_drop & ~true_drop))
    fn = int(np.sum(~pred_drop & true_drop))
    tn = int(np.sum(~pred_drop & ~true_drop))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    keep_recall = tn / (tn + fp) if tn + fp else 0.0
    return {
        "drop_precision": precision,
        "drop_recall": recall,
        "drop_f1": f1,
        "keep_recall": keep_recall,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
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
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    bundle = load_feature_bundle(features_path)
    x = bundle["x"].float()
    y = bundle["y"].long()
    if x.ndim != 2 or x.shape[1] != len(PRE_ASR_CUEQC_FEATURE_NAMES):
        raise ValueError("feature tensor shape mismatch")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("label tensor shape mismatch")
    selected_repo = qwen_asr_repo_id(asr_repo_id)
    bundle_repo = qwen_asr_repo_id(str(bundle.get("asr_repo_id") or selected_repo))
    if bundle_repo != selected_repo:
        raise ValueError(f"feature bundle asr_repo_id={bundle_repo!r} does not match {selected_repo!r}")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    count = int(x.shape[0])
    order = rng.permutation(count)
    val_count = max(1, int(round(count * 0.15))) if count >= 8 else max(1, count // 4)
    val_idx = torch.as_tensor(order[:val_count], dtype=torch.long)
    train_idx = torch.as_tensor(order[val_count:] if val_count < count else order, dtype=torch.long)

    mean = x[train_idx].mean(dim=0)
    std = x[train_idx].std(dim=0).clamp_min(1e-6)
    x_norm = (x - mean) / std
    x_norm = torch.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)

    normalized_device = device.strip().lower()
    if normalized_device == "auto":
        normalized_device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(normalized_device)
    model = make_model(x.shape[1], hidden_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_x = x_norm[train_idx].to(dev)
    train_y = y[train_idx].to(dev)
    class_counts = torch.bincount(train_y.cpu(), minlength=2).float()
    class_weights = (
        float(train_y.shape[0]) / (2.0 * class_counts.clamp_min(1.0))
    ).to(dev)
    batch_size = max(1, int(batch_size))
    for _step in range(max(1, int(steps))):
        sample = torch.randint(0, train_x.shape[0], (min(batch_size, train_x.shape[0]),), device=dev)
        logits = model(train_x[sample])
        loss = F.cross_entropy(logits, train_y[sample], weight=class_weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    with torch.inference_mode():
        val_probs = torch.softmax(model(x_norm[val_idx].to(dev)), dim=-1).cpu().numpy()
        all_probs = torch.softmax(model(x_norm.to(dev)), dim=-1).cpu().numpy()
    metrics = {
        "schema": METRICS_SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "features": repo_display_path(features_path),
        "feature_sha256": file_sha256(features_path),
        "asr_repo_id": selected_repo,
        "train_count": int(train_idx.shape[0]),
        "val_count": int(val_idx.shape[0]),
        "all_count": count,
        "class_counts": {
            "drop": int(torch.sum(y == 0).item()),
            "keep": int(torch.sum(y == 1).item()),
        },
        "train_class_weights": {
            "drop": float(class_weights[0].detach().cpu().item()),
            "keep": float(class_weights[1].detach().cpu().item()),
        },
        "drop_threshold": float(drop_threshold),
        "val": classification_metrics(val_probs, y[val_idx].numpy(), threshold=drop_threshold),
        "all": classification_metrics(all_probs, y.numpy(), threshold=drop_threshold),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / default_checkpoint_name(selected_repo)
    checkpoint = {
        "schema": PRE_ASR_CUEQC_SCHEMA,
        "feature_names": list(PRE_ASR_CUEQC_FEATURE_NAMES),
        "model_config": {
            "input_dim": len(PRE_ASR_CUEQC_FEATURE_NAMES),
            "hidden_size": int(hidden_size),
        },
        "normalization": {
            "mean": mean.cpu().numpy().astype(np.float32).tolist(),
            "std": std.cpu().numpy().astype(np.float32).tolist(),
        },
        "decision_config": {
            "drop_threshold": float(drop_threshold),
        },
        "metadata": {
            "asr_repo_id": selected_repo,
            "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
            "feature_bundle": repo_display_path(features_path),
            "feature_bundle_sha256": file_sha256(features_path),
            "trained_steps": int(steps),
            "created_at": metrics["created_at"],
        },
        "state_dict": model.cpu().state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    metrics["checkpoint"] = repo_display_path(checkpoint_path)
    metrics["checkpoint_sha256"] = file_sha256(checkpoint_path)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pre-ASR CueQC v6 binary checkpoint.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asr-repo-id", required=True)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--drop-threshold", type=float, default=0.90)
    args = parser.parse_args(argv)
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 0.0 <= args.drop_threshold <= 1.0:
        parser.error("--drop-threshold must be between 0 and 1")
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
    )
    print(
        "checkpoint={checkpoint} val_drop_f1={f1:.4f} val_keep_recall={keep:.4f}".format(
            checkpoint=metrics["checkpoint"],
            f1=float(metrics["val"]["drop_f1"]),
            keep=float(metrics["val"]["keep_recall"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
