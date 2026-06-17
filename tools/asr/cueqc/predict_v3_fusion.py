#!/usr/bin/env python3
"""Run CueQC v3-Fusion inference on a feature bundle and export pseudo labels."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.cueqc_model import CueQCMambaV3Fusion  # noqa: E402


def _normalize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if mean.size != arr.shape[-1] or std.size != arr.shape[-1]:
        raise ValueError(f"normalization shape mismatch: arr={arr.shape[-1]} mean={mean.size} std={std.size}")
    out = (arr - mean) / np.maximum(std, 1e-6)
    return np.where(np.isfinite(out), out, 0.0).astype(np.float32)


def _pad_batch(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_t = max(int(seq.shape[0]) for seq in seqs)
    dim = int(seqs[0].shape[1])
    out = torch.zeros((len(seqs), max_t, dim), dtype=torch.float32)
    mask = torch.zeros((len(seqs), max_t), dtype=torch.float32)
    for index, seq in enumerate(seqs):
        steps = int(seq.shape[0])
        out[index, :steps] = seq
        mask[index, :steps] = 1.0
    return out, mask


def _resolve_device(requested: str) -> torch.device:
    normalized = (requested or "auto").strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    return torch.device(normalized)


def _model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> CueQCMambaV3Fusion:
    if checkpoint.get("schema") != "cueqc_mamba_checkpoint_v3_fusion":
        raise ValueError(f"unexpected checkpoint schema: {checkpoint.get('schema')!r}")
    model_config = dict(checkpoint.get("model_config") or {})
    valid = {
        "asr_dim", "token_dim", "decoder_dim", "structured_dim",
        "hidden_size", "num_layers", "state_size", "num_heads", "head_dim",
        "n_groups", "chunk_size", "mlp_dim", "dropout",
    }
    model = CueQCMambaV3Fusion(**{key: value for key, value in model_config.items() if key in valid})
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def _normalization(checkpoint: dict[str, Any]) -> dict[str, np.ndarray]:
    norm = checkpoint.get("normalization") or {}
    return {
        key: np.asarray(norm.get(key, []), dtype=np.float32)
        for key in (
            "asr_mean", "asr_std",
            "token_mean", "token_std",
            "decoder_mean", "decoder_std",
            "structured_mean", "structured_std",
        )
    }


def _sample_to_tensors(sample: dict[str, Any], norm: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "asr_frames": torch.from_numpy(_normalize(sample["asr_frames"].numpy(), norm["asr_mean"], norm["asr_std"])),
        "token_trace": torch.from_numpy(_normalize(sample["token_trace"].numpy(), norm["token_mean"], norm["token_std"])),
        "decoder_stats": torch.from_numpy(
            _normalize(sample["decoder_stats"].numpy().reshape(1, -1), norm["decoder_mean"], norm["decoder_std"])[0]
        ),
        "structured": torch.from_numpy(
            _normalize(sample["structured"].numpy().reshape(1, -1), norm["structured_mean"], norm["structured_std"])[0]
        ),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    device = _resolve_device(args.device)
    features = torch.load(args.features, map_location="cpu", weights_only=False)
    if features.get("schema") != "cueqc_mamba_v3_fusion_features":
        raise ValueError(f"unexpected feature schema: {features.get('schema')!r}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _model_from_checkpoint(checkpoint, device)
    norm = _normalization(checkpoint)
    decision = checkpoint.get("decision_config") or {}
    drop_threshold = float(args.drop_threshold if args.drop_threshold is not None else decision.get("drop_threshold", 0.85))
    keep_threshold = float(args.keep_threshold)

    samples: list[dict[str, Any]] = list(features.get("samples") or [])
    labels = features.get("labels")
    label_values = labels.tolist() if labels is not None else [-1] * len(samples)
    meta: list[dict[str, Any]] = list(features.get("meta") or [{} for _ in samples])
    batch_size = max(1, int(args.batch_size))

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch_samples = [_sample_to_tensors(sample, norm) for sample in samples[start : start + batch_size]]
            asr_frames, asr_mask = _pad_batch([sample["asr_frames"] for sample in batch_samples])
            token_trace, token_mask = _pad_batch([sample["token_trace"] for sample in batch_samples])
            decoder_stats = torch.stack([sample["decoder_stats"] for sample in batch_samples])
            structured = torch.stack([sample["structured"] for sample in batch_samples])
            logits = model(
                asr_frames=asr_frames.to(device),
                asr_mask=asr_mask.to(device),
                token_trace=token_trace.to(device),
                token_mask=token_mask.to(device),
                decoder_stats=decoder_stats.to(device),
                structured=structured.to(device),
            )
            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
            for offset, prob in enumerate(probs):
                index = start + offset
                p_drop = float(prob[0])
                p_keep = float(prob[1])
                if p_drop >= drop_threshold:
                    display = "drop"
                    confidence = p_drop
                else:
                    display = "keep"
                    confidence = p_keep
                item_meta = meta[index] if index < len(meta) and isinstance(meta[index], dict) else {}
                rows.append({
                    "schema": "cueqc_prediction_v3_fusion_v1",
                    "sample_id": item_meta.get("sample_id", ""),
                    "audio_id": item_meta.get("audio_id", ""),
                    "video_id": item_meta.get("video_id", ""),
                    "chunk_index": item_meta.get("chunk_index"),
                    "cluster_id": item_meta.get("cluster_id", ""),
                    "start": item_meta.get("start"),
                    "end": item_meta.get("end"),
                    "text": item_meta.get("text", ""),
                    "label": int(label_values[index]) if index < len(label_values) else -1,
                    "display_hint": display,
                    "confidence": round(confidence, 6),
                    "display_prob_drop": round(p_drop, 6),
                    "display_prob_keep": round(p_keep, 6),
                    "drop_threshold": drop_threshold,
                    "keep_threshold": keep_threshold,
                    "model_version": "cueqc_mamba_v3_fusion",
                    "decision_version": "cueqc_display_binary_v1",
                })

    pseudo_rows = [
        {
            **row,
            "schema": "cueqc_pseudo_label_v3_fusion_v1",
            "targets": {
                "display_decision": row["display_hint"],
                "display_label": 0 if row["display_hint"] == "drop" else 1,
            },
            "label_meta": {
                "label_source": "cueqc_v3_fusion_pseudo",
                "confidence": row["confidence"],
                "display_prob_drop": row["display_prob_drop"],
                "display_prob_keep": row["display_prob_keep"],
            },
        }
        for row in rows
        if (row["display_hint"] == "drop" and row["display_prob_drop"] >= drop_threshold)
        or (row["display_hint"] == "keep" and row["display_prob_keep"] >= keep_threshold)
    ]

    output_dir = Path(args.output_dir)
    predictions_path = output_dir / "cueqc_predictions.jsonl"
    pseudo_path = output_dir / "cueqc_pseudo_labels.high_conf.jsonl"
    summary_path = output_dir / "summary.json"
    _write_jsonl(predictions_path, rows)
    _write_jsonl(pseudo_path, pseudo_rows)

    counts = Counter(str(row["display_hint"]) for row in rows)
    pseudo_counts = Counter(str(row["display_hint"]) for row in pseudo_rows)
    summary = {
        "schema": "cueqc_prediction_summary_v3_fusion_v1",
        "features": str(args.features),
        "checkpoint": str(args.checkpoint),
        "predictions_path": str(predictions_path),
        "pseudo_labels_path": str(pseudo_path),
        "records": len(rows),
        "pseudo_records": len(pseudo_rows),
        "counts": dict(counts.most_common()),
        "pseudo_counts": dict(pseudo_counts.most_common()),
        "drop_threshold": drop_threshold,
        "keep_threshold": keep_threshold,
        "batch_size": batch_size,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(f"predictions={predictions_path}")
    print(f"pseudo_labels={pseudo_path}")
    print(f"summary={summary_path}")
    print(f"records={len(rows)} pseudo_records={len(pseudo_rows)} counts={dict(counts)} pseudo_counts={dict(pseudo_counts)}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict CueQC v3-Fusion keep/drop labels from extracted features.")
    parser.add_argument("--features", required=True, help="cueqc_mamba_v3_fusion_features .pt bundle")
    parser.add_argument("--checkpoint", required=True, help="cueqc_mamba_v3_fusion.pt checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--drop-threshold", type=float, default=None)
    parser.add_argument("--keep-threshold", type=float, default=0.95)
    args = parser.parse_args(argv)
    if args.drop_threshold is not None and not 0.5 <= args.drop_threshold <= 1.0:
        parser.error("--drop-threshold must be in [0.5, 1.0]")
    if not 0.5 <= args.keep_threshold <= 1.0:
        parser.error("--keep-threshold must be in [0.5, 1.0]")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
