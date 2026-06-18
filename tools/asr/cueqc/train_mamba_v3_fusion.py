#!/usr/bin/env python3
"""Train CueQC Mamba v3-Fusion — display (keep/drop) classifier.

Consumes the ASR-internals feature bundle from ``extract_features_v3_fusion.py``
and trains a ``CueQCMambaV3Fusion`` (ASR Mamba2 + token Mamba2 + decoder/structured
fusion). Binary CrossEntropy with keep up-weighted.

Data split is a **fixed test movie** (``--test-audio-id``): all samples from one
video id form the held-out test set; the rest are train. Normalization (z-score)
is computed on the **train split only** so test leakage cannot happen. A small
inner validation split (random) is used for early monitoring; the headline
metrics are evaluated on the fixed test movie.

Label convention (v3): ``drop = 0``, ``keep = 1``. ``logits[:, 0]`` = drop.
Checkpoint schema: ``cueqc_mamba_checkpoint_v3_fusion``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.cueqc_model import CueQCMambaV3Fusion  # noqa: E402
from asr.backends.qwen import qwen_asr_repo_tag  # noqa: E402

CHECKPOINT_SCHEMA = "cueqc_mamba_checkpoint_v3_fusion"
METRICS_SCHEMA = "cueqc_mamba_train_metrics_v3_fusion"
DECISION_VERSION = "cueqc_display_binary_v1"


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 500
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    seed: int = 13
    device: str = "auto"
    val_ratio: float = 0.15
    drop_threshold: float = 0.85
    hidden_size: int = 128
    num_layers: int = 2
    state_size: int = 32
    num_heads: int = 4
    head_dim: int = 64
    n_groups: int = 4
    chunk_size: int = 64
    mlp_dim: int = 64
    dropout: float = 0.2
    test_audio_id: str = ""
    keep_class_weight: float = 1.5
    label_smoothing: float = 0.1
    eval_every: int = 25
    early_stop_patience: int = 6


def _resolve_device(requested: str) -> torch.device:
    normalized = (requested or "auto").strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    return torch.device(normalized)


def _pad_batch(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of [T_i, D] tensors to [B, max_T, D] + mask [B, max_T]."""
    max_t = max(int(s.shape[0]) for s in seqs)
    d = int(seqs[0].shape[1])
    out = torch.zeros((len(seqs), max_t, d), dtype=torch.float32)
    mask = torch.zeros((len(seqs), max_t), dtype=torch.float32)
    for i, s in enumerate(seqs):
        t = int(s.shape[0])
        out[i, :t] = s
        mask[i, :t] = 1.0
    return out, mask


def _collate(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    asr_frames, asr_masks = _pad_batch([s["asr_frames"] for s in samples])
    token_trace, token_masks = _pad_batch([s["token_trace"] for s in samples])
    decoder = torch.stack([s["decoder_stats"] for s in samples])
    structured = torch.stack([s["structured"] for s in samples])
    return {
        "asr_frames": asr_frames,
        "asr_mask": asr_masks,
        "token_trace": token_trace,
        "token_mask": token_masks,
        "decoder_stats": decoder,
        "structured": structured,
    }


def _normalize_split(arrays: list[np.ndarray], mean: np.ndarray, std: np.ndarray) -> list[np.ndarray]:
    out = []
    for a in arrays:
        a = (a - mean) / std
        a = np.where(np.isfinite(a), a, 0.0).astype(np.float32)
        out.append(a)
    return out


def _compute_stats(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Feature-wise mean/std from a list of [*, D] arrays (flattened across samples)."""
    stacked = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrays], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


@torch.no_grad()
def evaluate(
    model: CueQCMambaV3Fusion,
    samples: list[dict[str, Any]],
    *,
    asr_mean: np.ndarray,
    asr_std: np.ndarray,
    token_mean: np.ndarray,
    token_std: np.ndarray,
    decoder_mean: np.ndarray,
    decoder_std: np.ndarray,
    structured_mean: np.ndarray,
    structured_std: np.ndarray,
    device: torch.device,
    drop_threshold: float,
) -> dict[str, float]:
    if not samples:
        return {"n": 0}
    # Apply normalization on numpy arrays.
    norm_samples = []
    for s in samples:
        norm_samples.append({
            "asr_frames": torch.from_numpy(_normalize_split([s["asr_frames"].numpy()], asr_mean, asr_std)[0]),
            "token_trace": torch.from_numpy(_normalize_split([s["token_trace"].numpy()], token_mean, token_std)[0]),
            "decoder_stats": torch.from_numpy(_normalize_split([s["decoder_stats"].numpy(), ], decoder_mean, decoder_std)[0]),
            "structured": torch.from_numpy(_normalize_split([s["structured"].numpy(), ], structured_mean, structured_std)[0]),
        })
    model.eval()
    batch = _collate(norm_samples)
    logits = model(
        asr_frames=batch["asr_frames"].to(device),
        asr_mask=batch["asr_mask"].to(device),
        token_trace=batch["token_trace"].to(device),
        token_mask=batch["token_mask"].to(device),
        decoder_stats=batch["decoder_stats"].to(device),
        structured=batch["structured"].to(device),
    )
    probs = torch.softmax(logits, dim=-1)
    p_drop = probs[:, 0]
    # Decision: drop only if p_drop >= threshold (conservative), else keep.
    preds = torch.where(p_drop >= drop_threshold, torch.tensor(0), torch.tensor(1)).to(logits.device)
    labels = torch.tensor([s["__label__"] for s in samples], device=logits.device)

    n = labels.shape[0]
    keep_mask = labels == 1
    drop_mask = labels == 0
    n_keep = int(keep_mask.sum().item())
    n_drop = int(drop_mask.sum().item())

    acc = float((preds == labels).float().mean().item()) if n else 0.0
    keep_recall = float((preds[keep_mask] == 1).float().mean().item()) if n_keep else 0.0
    drop_recall = float((preds[drop_mask] == 0).float().mean().item()) if n_drop else 0.0
    # precision over predicted-drop
    pred_drop = preds == 0
    drop_precision = float((labels[pred_drop] == 0).float().mean().item()) if pred_drop.sum() else 0.0
    # false drop = true keep predicted drop / true keep
    false_drop = float((preds[keep_mask] == 0).float().sum().item()) if n_keep else 0.0
    false_drop_rate = float(false_drop / n_keep) if n_keep else 0.0

    return {
        "n": n,
        "accuracy": round(acc, 4),
        "keep_recall": round(keep_recall, 4),
        "drop_precision": round(drop_precision, 4),
        "drop_recall": round(drop_recall, 4),
        "false_drop_rate": round(false_drop_rate, 4),
        "n_keep": n_keep,
        "n_drop": n_drop,
    }


def _checkpoint_name(feature_config: dict[str, Any], explicit: str = "") -> str:
    if explicit.strip():
        return explicit.strip()
    asr_model_id = str(feature_config.get("asr_model_id") or "").strip()
    return f"cueqc_mamba_v3_fusion.{qwen_asr_repo_tag(asr_model_id)}.pt"


def train(
    config: TrainConfig,
    *,
    features_path: Path,
    output_dir: Path,
    checkpoint_name: str = "",
) -> dict[str, Any]:
    bundle = torch.load(features_path, map_location="cpu", weights_only=False)
    if bundle.get("schema") != "cueqc_mamba_v3_fusion_features":
        raise ValueError(f"unexpected feature schema: {bundle.get('schema')!r}")
    fcfg = bundle["feature_config"]
    asr_model_id = str(fcfg.get("asr_model_id") or "").strip()
    if not asr_model_id:
        raise ValueError("CueQC feature_config.asr_model_id is required for repo-id checkpoint binding")
    samples = bundle["samples"]
    labels = bundle["labels"].tolist()
    meta = bundle["meta"]
    n = len(samples)
    if n == 0:
        raise RuntimeError("no samples in feature bundle")
    if any(int(label) not in {0, 1} for label in labels):
        raise RuntimeError("feature bundle contains unlabeled samples; run predict_v3_fusion.py instead")
    # Stamp labels onto samples for evaluation convenience.
    for s, lb, m in zip(samples, labels, meta):
        s["__label__"] = lb
        s["audio_id"] = m["audio_id"]

    # ---- Split: fixed test movie vs train ----
    test_audio_id = config.test_audio_id.strip()
    if test_audio_id:
        test_idx = [i for i, s in enumerate(samples) if s["audio_id"] == test_audio_id]
        train_idx = [i for i, s in enumerate(samples) if s["audio_id"] != test_audio_id]
        if not test_idx:
            raise RuntimeError(f"--test-audio-id {test_audio_id!r} matched no samples; "
                               f"available audio_ids: {sorted({s['audio_id'] for s in samples})}")
    else:
        # Fallback: random split (warn in metrics).
        perm = list(range(n))
        random.Random(config.seed).shuffle(perm)
        test_n = max(1, int(n * config.val_ratio))
        test_idx = perm[:test_n]
        train_idx = perm[test_n:]
    if not train_idx:
        raise RuntimeError("train split is empty after removing test movie")

    # ---- Inner validation (random) from train for monitoring ----
    random.Random(config.seed).shuffle(train_idx)
    val_n = max(1, int(len(train_idx) * config.val_ratio))
    val_idx = train_idx[:val_n]
    train_only_idx = train_idx[val_n:]

    # ---- Normalization from TRAIN ONLY ----
    train_samples = [samples[i] for i in train_only_idx]
    asr_mean, asr_std = _compute_stats([s["asr_frames"].numpy() for s in train_samples])
    token_mean, token_std = _compute_stats([s["token_trace"].numpy() for s in train_samples])
    decoder_mean, decoder_std = _compute_stats([s["decoder_stats"].numpy().reshape(1, -1) for s in train_samples])
    structured_mean, structured_std = _compute_stats([s["structured"].numpy().reshape(1, -1) for s in train_samples])

    # Pre-normalize train samples for training loop.
    def normalize_sample(s: dict[str, Any]) -> dict[str, Any]:
        return {
            "asr_frames": torch.from_numpy(_normalize_split([s["asr_frames"].numpy()], asr_mean, asr_std)[0]),
            "token_trace": torch.from_numpy(_normalize_split([s["token_trace"].numpy()], token_mean, token_std)[0]),
            "decoder_stats": torch.from_numpy(_normalize_split([s["decoder_stats"].numpy().reshape(1, -1)], decoder_mean, decoder_std)[0]).reshape(-1),
            "structured": torch.from_numpy(_normalize_split([s["structured"].numpy().reshape(1, -1)], structured_mean, structured_std)[0]).reshape(-1),
        }

    norm_train = [(normalize_sample(samples[i]), samples[i]["__label__"]) for i in train_only_idx]

    device = _resolve_device(config.device)
    model = CueQCMambaV3Fusion(
        asr_dim=fcfg["asr_dim"],
        token_dim=fcfg["token_dim"],
        decoder_dim=fcfg["decoder_dim"],
        structured_dim=fcfg["structured_dim"],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        state_size=config.state_size,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        n_groups=config.n_groups,
        chunk_size=config.chunk_size,
        mlp_dim=config.mlp_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # keep up-weighted (class index 1).
    class_weights = torch.tensor([1.0, float(config.keep_class_weight)], device=device)

    bs = min(config.batch_size, len(norm_train))
    # Label smoothing combats logit blow-up (cold-start cluster-broadcast labels
    # let the model memorize; smoothing keeps softmax probabilities away from 0/1).
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)
    norm_kwargs = dict(
        asr_mean=asr_mean, asr_std=asr_std,
        token_mean=token_mean, token_std=token_std,
        decoder_mean=decoder_mean, decoder_std=decoder_std,
        structured_mean=structured_mean, structured_std=structured_std,
        device=device, drop_threshold=config.drop_threshold,
    )
    val_samples_eval = [samples[i] for i in val_idx]
    best_val_score = -1.0
    best_state = None
    best_step = 0
    patience_left = config.early_stop_patience
    last_loss = 0.0
    model.train()
    rng = random.Random(config.seed)
    for step in range(config.max_steps):
        rng.shuffle(norm_train)
        epoch_losses = []
        for start in range(0, len(norm_train), bs):
            batch_samples = [norm_train[i][0] for i in range(start, min(start + bs, len(norm_train)))]
            batch_labels = torch.tensor([norm_train[i][1] for i in range(start, min(start + bs, len(norm_train)))], device=device)
            b = _collate(batch_samples)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                asr_frames=b["asr_frames"].to(device),
                asr_mask=b["asr_mask"].to(device),
                token_trace=b["token_trace"].to(device),
                token_mask=b["token_mask"].to(device),
                decoder_stats=b["decoder_stats"].to(device),
                structured=b["structured"].to(device),
            )
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        last_loss = sum(epoch_losses) / len(epoch_losses)
        # Periodic validation: pick the checkpoint that maximizes keep_recall
        # while keeping false_drop_rate low (score = keep_recall - false_drop_rate).
        if (step + 1) % config.eval_every == 0 or step == config.max_steps - 1:
            v = evaluate(model, val_samples_eval, **norm_kwargs)
            score = v.get("keep_recall", 0.0) - v.get("false_drop_rate", 0.0)
            print(f"  step {step + 1}/{config.max_steps} train_loss={last_loss:.4e} "
                  f"val_keep_recall={v.get('keep_recall', 0):.3f} val_fdr={v.get('false_drop_rate', 0):.3f} score={score:.3f}")
            if score > best_val_score:
                best_val_score = score
                best_step = step + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = config.early_stop_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"  early stop at step {step + 1} (best step {best_step}, score {best_val_score:.3f})")
                    break
    # Restore best checkpoint (max keep_recall - false_drop_rate on val).
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  restored best checkpoint from step {best_step} (val score {best_val_score:.3f})")

    # ---- Evaluation (final: on best checkpoint) ----
    test_eval = evaluate(model, [samples[i] for i in test_idx], **norm_kwargs)
    val_eval = evaluate(model, [samples[i] for i in val_idx], **norm_kwargs)

    # ---- Checkpoint ----
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file_name = _checkpoint_name(fcfg, checkpoint_name)
    if Path(checkpoint_file_name).name != checkpoint_file_name:
        raise ValueError("--checkpoint-name must be a file name, not a path")
    checkpoint_path = output_dir / checkpoint_file_name
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "version": 3,
        "decision_version": DECISION_VERSION,
        "model_config": dict(model.model_config),
        "feature_config": {
            "feature_source": fcfg["feature_source"],
            "asr_model_id": fcfg.get("asr_model_id", ""),
            "asr_model_spec": fcfg.get("asr_model_spec", ""),
            "uses_bge": fcfg["uses_bge"],
            "text_embedding": fcfg["text_embedding"],
            "asr_dim": fcfg["asr_dim"],
            "token_dim": fcfg["token_dim"],
            "decoder_dim": fcfg["decoder_dim"],
            "structured_dim": fcfg["structured_dim"],
            "token_feature_names": fcfg["token_feature_names"],
            "decoder_feature_names": fcfg["decoder_feature_names"],
            "structured_feature_names": fcfg["structured_feature_names"],
        },
        "metadata": {
            "asr_repo_id": asr_model_id,
            "asr_model_spec": str(fcfg.get("asr_model_spec") or ""),
        },
        "normalization": {
            "asr_mean": torch.from_numpy(asr_mean),
            "asr_std": torch.from_numpy(asr_std),
            "token_mean": torch.from_numpy(token_mean),
            "token_std": torch.from_numpy(token_std),
            "decoder_mean": torch.from_numpy(decoder_mean),
            "decoder_std": torch.from_numpy(decoder_std),
            "structured_mean": torch.from_numpy(structured_mean),
            "structured_std": torch.from_numpy(structured_std),
        },
        "decision_config": {
            "drop_threshold": float(config.drop_threshold),
            "fallback_policy": "keep",
            "apply_enabled_default": True,
        },
        "label_config": {"drop": 0, "keep": 1},
        "state_dict": model.to("cpu").state_dict(),
        "metrics": {
            "test_audio_id": test_audio_id,
            "test_n": test_eval.get("n", 0),
            "test_accuracy": test_eval.get("accuracy", 0.0),
            "test_keep_recall": test_eval.get("keep_recall", 0.0),
            "test_drop_precision": test_eval.get("drop_precision", 0.0),
            "test_drop_recall": test_eval.get("drop_recall", 0.0),
            "test_false_drop_rate": test_eval.get("false_drop_rate", 0.0),
            "val_accuracy": val_eval.get("accuracy", 0.0),
            "val_keep_recall": val_eval.get("keep_recall", 0.0),
            "val_false_drop_rate": val_eval.get("false_drop_rate", 0.0),
            "last_loss": last_loss,
            "train_count": len(train_only_idx),
            "val_count": len(val_idx),
        },
        "train_config": asdict(config),
        "features_path": str(features_path),
    }
    torch.save(checkpoint, checkpoint_path)

    metrics = {
        "schema": METRICS_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "records": n,
        "test_audio_id": test_audio_id,
        "split_mode": "fixed_test_movie" if test_audio_id else "random",
        "test": test_eval,
        "val": val_eval,
        "labels": {"keep": int(sum(1 for lb in labels if lb == 1)), "drop": int(sum(1 for lb in labels if lb == 0))},
        "last_loss": last_loss,
        "train_config": asdict(config),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(f"checkpoint={checkpoint_path}")
    print(f"metrics={metrics_path}")
    print(f"test: {test_eval}")
    print(f"val:  {val_eval}")
    print(f"last_loss={last_loss:.4f}")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CueQC Mamba v3-Fusion.")
    p.add_argument("--features", required=True, help="cueqc_train_features_v3_fusion.pt")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--checkpoint-name",
        default="",
        help="Checkpoint file name. Default appends feature_config.asr_model_id repo id tag.",
    )
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", dest="learning_rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default="auto")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--drop-threshold", type=float, default=0.85)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--state-size", type=int, default=32)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--n-groups", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--mlp-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--test-audio-id", default="", help="video id held out as the test movie")
    p.add_argument("--keep-class-weight", type=float, default=1.5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--early-stop-patience", type=int, default=6)
    args = p.parse_args(argv)
    if not 0.0 <= args.val_ratio < 1.0:
        p.error("--val-ratio must be in [0, 1)")
    if not 0.5 <= args.drop_threshold <= 1.0:
        p.error("--drop-threshold must be in [0.5, 1.0]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fields = {
        "max_steps", "batch_size", "learning_rate", "weight_decay", "seed", "device",
        "val_ratio", "drop_threshold", "hidden_size", "num_layers", "state_size",
        "num_heads", "head_dim", "n_groups", "chunk_size", "mlp_dim", "dropout",
        "test_audio_id", "keep_class_weight", "label_smoothing", "eval_every",
        "early_stop_patience",
    }
    config = TrainConfig(**{k: getattr(args, k) for k in fields})
    train(
        config,
        features_path=Path(args.features),
        output_dir=Path(args.output_dir),
        checkpoint_name=args.checkpoint_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
