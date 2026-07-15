#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID, qwen_asr_repo_tag  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.inner_refiner_v1 import (  # noqa: E402
    FullSubislandInnerEdgeNetwork,
    build_inner_edge_refiner_v1_checkpoint,
)
from boundary.ja.model import SPEECH_ISLAND_SCORER_LABELS  # noqa: E402
from boundary.outer_refiner_v2 import decode_outer_edge_probabilities  # noqa: E402
from tools.boundary.ja.train_outer_edge_refiner_v2 import (  # noqa: E402
    _adaptive_sampling_probabilities,
    _edge_categories_by_audio_id,
    _load,
    _normalization,
    _normalized,
    _rows,
)


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    if args.ptm_repo_id != QWEN_ASR_17B_REPO_ID:
        raise ValueError("Inner Edge Refiner v1 training is 1.7B-only")
    apply_vram_safety_cap(0.95)
    rows = _rows(Path(args.dataset_manifest))
    train_rows = [row for row in rows if row.get("partition") == "train"]
    val_rows = [row for row in rows if row.get("partition") != "train"]
    if not train_rows:
        raise ValueError("Inner Edge Refiner v1 has no train rows")
    if not val_rows:
        rng = np.random.default_rng(args.seed)
        val_indexes = set(
            int(index)
            for index in rng.permutation(len(train_rows))[: max(1, len(train_rows) // 10)]
        )
        val_rows = [row for index, row in enumerate(train_rows) if index in val_indexes]
        train_rows = [row for index, row in enumerate(train_rows) if index not in val_indexes]
    categories_by_id = _edge_categories_by_audio_id(
        synthetic_details=Path(args.synthetic_details),
        negative_manifest=Path(args.negative_manifest),
    )
    sampling_probabilities, edge_category_counts, effective_sample_size = (
        _adaptive_sampling_probabilities(train_rows, categories_by_id)
    )
    normalization = _normalization(train_rows)
    first, _labels, _weights = _load(train_rows[0])
    expected_input_dim = args.raw_ptm_dim + args.mfcc_dim + 1
    if int(first.shape[1]) != expected_input_dim:
        raise ValueError(
            f"Inner v1 feature dim {first.shape[1]} != expected {expected_input_dim}"
        )
    model_config = {
        "ptm_input_dim": args.raw_ptm_dim,
        "ptm_projected_dim": args.projected_ptm_dim,
        "mfcc_dim": args.mfcc_dim,
        "position_dim": 1,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "state_size": 32,
        "num_heads": 4,
        "head_dim": 64,
        "n_groups": 2,
        "conv_kernel": 4,
        "chunk_size": 8,
        "bidirectional": True,
        "output_dim": len(SPEECH_ISLAND_SCORER_LABELS),
    }
    device = torch.device(args.device)
    model = FullSubislandInnerEdgeNetwork(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    class_weights = torch.tensor(
        [args.discardable_weight, args.semantic_target_weight, args.unsure_weight],
        dtype=torch.float32,
        device=device,
    )
    rng = np.random.default_rng(args.seed)
    losses: list[float] = []
    started = time.monotonic()
    for step in range(args.max_steps):
        row = train_rows[int(rng.choice(len(train_rows), p=sampling_probabilities))]
        features, labels, weights = _load(row)
        logits = model(
            torch.from_numpy(_normalized(features, normalization)).unsqueeze(0).to(device)
        )[0]
        label_tensor = torch.from_numpy(labels).to(device)
        effective = torch.from_numpy(weights).to(device) * class_weights[label_tensor]
        raw_loss = F.cross_entropy(logits, label_tensor, reduction="none")
        probabilities = torch.softmax(logits, dim=-1)
        pt = probabilities.gather(1, label_tensor.unsqueeze(1)).squeeze(1)
        loss = (
            raw_loss * torch.pow(1.0 - pt, args.focal_gamma) * effective
        ).sum() / effective.sum().clamp_min(1e-6)
        target_frames = torch.nonzero(
            label_tensor == SPEECH_ISLAND_SCORER_LABELS.index("semantic_target"),
            as_tuple=False,
        ).flatten()
        if target_frames.numel() > 0:
            edge_indexes = torch.unique(torch.stack((target_frames[0], target_frames[-1])))
            edge_loss = F.cross_entropy(logits[edge_indexes], label_tensor[edge_indexes])
            loss = loss + float(args.edge_loss_weight) * edge_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if args.log_every and (step + 1) % args.log_every == 0:
            print(
                f"inner_v1_train={step + 1}/{args.max_steps} "
                f"loss={losses[-1]:.6f} elapsed_s={time.monotonic() - started:.1f}",
                flush=True,
            )
    target_index = SPEECH_ISLAND_SCORER_LABELS.index("semantic_target")
    predicted_covered = edge_total = known_clipped = 0
    true_positive = false_positive = false_negative = 0
    start_errors: list[float] = []
    end_errors: list[float] = []
    with torch.inference_mode():
        for row in val_rows:
            features, labels, _weights = _load(row)
            logits = model(
                torch.from_numpy(_normalized(features, normalization)).unsqueeze(0).to(device)
            )[0]
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            truth_target = np.flatnonzero(labels == target_index)
            if truth_target.size == 0:
                continue
            edge_total += 1
            prediction = decode_outer_edge_probabilities(
                probabilities,
                raw_start_s=0.0,
                raw_end_s=float(len(labels)),
                frame_hop_s=1.0,
            )
            if prediction.start_action == "refined" or prediction.end_action == "refined":
                predicted_covered += 1
            predicted_target = np.argmax(probabilities, axis=1) == target_index
            truth_mask = labels == target_index
            true_positive += int(np.sum(predicted_target & truth_mask))
            false_positive += int(np.sum(predicted_target & ~truth_mask))
            false_negative += int(np.sum(~predicted_target & truth_mask))
            truth_start = int(truth_target[0])
            truth_end = int(truth_target[-1]) + 1
            start_errors.append(abs(float(prediction.start_s) - truth_start))
            end_errors.append(abs(float(prediction.end_s) - truth_end))
            if prediction.start_s > truth_start or prediction.end_s < truth_end:
                known_clipped += 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / (
        f"inner_edge_refiner_v1.{qwen_asr_repo_tag(args.ptm_repo_id)}.pt"
    )
    torch.save(
        build_inner_edge_refiner_v1_checkpoint(
            model=model,
            model_config=model_config,
            feature_config={
                "raw_ptm_dim": args.raw_ptm_dim,
                "learned_ptm_projected_dim": args.projected_ptm_dim,
                "ptm_projection": "checkpoint_learned_linear",
                "mfcc_dim": args.mfcc_dim,
                "relative_position_dim": 1,
                "frame_hop_s": args.frame_hop_s,
                "context": "cueqc_retained_provisional_subisland",
            },
            normalization=normalization,
            metadata={
                "ptm_repo_id": args.ptm_repo_id,
                "dataset_manifest": str(Path(args.dataset_manifest)),
                "trained_steps": args.max_steps,
                "edge_sampling": "sqrt_inverse_frequency_multilabel_v1",
                "edge_category_counts": edge_category_counts,
                "edge_sampling_effective_sample_size": effective_sample_size,
            },
        ),
        checkpoint_path,
    )
    metrics = {
        "schema": "inner_edge_refiner_v1_train_metrics_v1",
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "loss": float(np.mean(losses)),
        "paired_edge_coverage": predicted_covered / max(edge_total, 1),
        "known_tail_clipping_count": known_clipped,
        "semantic_target_precision": true_positive / max(true_positive + false_positive, 1),
        "semantic_target_recall": true_positive / max(true_positive + false_negative, 1),
        "start_mae_s": float(np.mean(start_errors)) * args.frame_hop_s,
        "end_mae_s": float(np.mean(end_errors)) * args.frame_hop_s,
        "ptm_projection": "checkpoint_learned_linear_2048_to_128",
        "edge_loss_weight": float(args.edge_loss_weight),
        "edge_sampling": "sqrt_inverse_frequency_multilabel_v1",
        "edge_category_counts": edge_category_counts,
        "edge_sampling_effective_sample_size": effective_sample_size,
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 1.7B Inner Edge Refiner v1.")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--synthetic-details", required=True)
    parser.add_argument("--negative-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ptm-repo-id", default=QWEN_ASR_17B_REPO_ID)
    parser.add_argument("--raw-ptm-dim", type=int, default=2048)
    parser.add_argument("--projected-ptm-dim", type=int, default=128)
    parser.add_argument("--mfcc-dim", type=int, default=40)
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--discardable-weight", type=float, default=1.0)
    parser.add_argument("--semantic-target-weight", type=float, default=3.0)
    parser.add_argument("--unsure-weight", type=float, default=1.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--edge-loss-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
