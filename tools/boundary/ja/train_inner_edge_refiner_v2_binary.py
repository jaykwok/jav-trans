#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID, qwen_asr_repo_tag  # noqa: E402
from boundary.binary_edge_refiner import (  # noqa: E402
    BINARY_EDGE_IGNORE_INDEX,
    BinaryFrameEdgeNetwork,
    canonical_to_binary_labels,
    decode_binary_edge_logits,
)
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.inner_refiner_v2 import build_inner_edge_refiner_v2_checkpoint  # noqa: E402
from tools.boundary.ja.edge_frame_dataset import (  # noqa: E402
    load_edge_row,
    normalize_edge_features,
    read_edge_rows,
)


PARTITIONS = ("train", "val", "test")
INNER_INPUT_DISTRIBUTION = "post_cueqc_v13_keep_subislands"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_save_atomic(payload: object, path: Path) -> None:
    import torch

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def validate_dataset_rows(rows: list[dict]) -> dict[str, object]:
    if not rows:
        raise ValueError("Inner v2 dataset is empty")
    source_partitions: dict[str, set[str]] = defaultdict(set)
    core_partitions: dict[str, set[str]] = defaultdict(set)
    core_counts: Counter[str] = Counter()
    partition_counts: Counter[str] = Counter()
    cueqc_shas: set[str] = set()
    subisland_ids: set[str] = set()
    for row in rows:
        source_id = str(row.get("source_id") or "").strip()
        core_id = str(row.get("core_id") or "").strip()
        subisland_id = str(row.get("subisland_id") or row.get("row_id") or "").strip()
        partition = str(row.get("partition") or "").strip()
        if not source_id or not core_id or not subisland_id:
            raise ValueError("Inner v2 rows require source_id/core_id/subisland_id")
        if subisland_id in subisland_ids:
            raise ValueError("Inner v2 provisional subisland identity is duplicated")
        subisland_ids.add(subisland_id)
        if partition not in PARTITIONS:
            raise ValueError(f"Inner v2 row has invalid partition: {partition!r}")
        if row.get("input_distribution") != INNER_INPUT_DISTRIBUTION:
            raise ValueError("Inner v2 rows must be actual post-CueQC v13 keep subislands")
        if row.get("cueqc_label") != "keep":
            raise ValueError("Inner v2 training rows require CueQC keep decisions")
        if row.get("training_manifest_allowed") is not True:
            raise ValueError("Inner v2 rows require an approved training manifest gate")
        cueqc_sha = str(row.get("cueqc_checkpoint_sha256") or "").lower()
        if len(cueqc_sha) != 64 or any(ch not in "0123456789abcdef" for ch in cueqc_sha):
            raise ValueError("Inner v2 rows require the exact CueQC checkpoint SHA256")
        cueqc_shas.add(cueqc_sha)
        source_partitions[source_id].add(partition)
        core_partitions[core_id].add(partition)
        core_counts[core_id] += 1
        partition_counts[partition] += 1
    if any(len(values) != 1 for values in source_partitions.values()):
        raise ValueError("Inner v2 source identity crosses dataset partitions")
    if any(len(values) != 1 for values in core_partitions.values()):
        raise ValueError("Inner v2 core identity crosses dataset partitions")
    if max(core_counts.values(), default=0) > 1:
        raise ValueError("Inner v2 requires each semantic core at most once")
    if any(partition_counts[name] <= 0 for name in PARTITIONS):
        raise ValueError("Inner v2 requires fixed train/val/test partitions")
    if len(cueqc_shas) != 1:
        raise ValueError("Inner v2 dataset mixes CueQC checkpoint identities")
    return {
        "source_count": len(source_partitions),
        "core_count": len(core_partitions),
        "partition_counts": dict(partition_counts),
        "cueqc_checkpoint_sha256": next(iter(cueqc_shas)),
    }


def load_source_features(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        if "ptm" not in source.files or "mfcc" not in source.files:
            raise ValueError(f"Inner v2 source features require ptm/mfcc arrays: {path}")
        ptm = np.asarray(source["ptm"], dtype=np.float32)
        mfcc = np.asarray(source["mfcc"], dtype=np.float32)
    if ptm.ndim != 2 or mfcc.ndim != 2 or ptm.shape[0] <= 0:
        raise ValueError(f"Inner v2 source features must be non-empty 2D arrays: {path}")
    if ptm.shape[0] != mfcc.shape[0]:
        raise ValueError(f"Inner v2 source PTM/MFCC frame counts differ: {path}")
    if not np.isfinite(ptm).all() or not np.isfinite(mfcc).all():
        raise ValueError(f"Inner v2 source features contain non-finite values: {path}")
    return ptm, mfcc


def load_binary(
    row: dict,
    *,
    expected_ptm_dim: int | None = None,
    expected_mfcc_dim: int | None = None,
):
    if row.get("label_path"):
        ptm, mfcc = load_source_features(str(row["source_feature_path"]))
        if expected_ptm_dim is not None and ptm.shape[1] != expected_ptm_dim:
            raise ValueError(f"Inner v2 raw PTM width mismatch: {row.get('row_id')}")
        if expected_mfcc_dim is not None and mfcc.shape[1] != expected_mfcc_dim:
            raise ValueError(f"Inner v2 MFCC width mismatch: {row.get('row_id')}")
        with np.load(row["label_path"], allow_pickle=False) as payload:
            if "labels" not in payload.files:
                raise ValueError(f"Inner v2 label payload is missing labels: {row.get('row_id')}")
            labels = payload["labels"].astype(np.int64)
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        if labels.ndim != 1 or start < 0 or end <= start or end > ptm.shape[0]:
            raise ValueError(f"Inner v2 feature/label coordinates are invalid: {row.get('row_id')}")
        ptm = ptm[start:end]
        mfcc = mfcc[start:end]
        lengths = {len(ptm), len(mfcc), len(labels)}
        if len(lengths) != 1 or not labels.size:
            raise ValueError(f"Inner v2 feature/label slice mismatch: {row.get('row_id')}")
        total = len(labels)
        position = (
            np.arange(total, dtype=np.float32) / max(1, total - 1)
        ).reshape(-1, 1)
        auxiliary = [position]
        if row.get("acoustic_start_frame") is not None:
            acoustic_start = int(row["acoustic_start_frame"]) - start
            acoustic_end = int(row["acoustic_end_frame"]) - start
            acoustic_length = max(1, acoustic_end - acoustic_start)
            acoustic_position = (
                (np.arange(total, dtype=np.float32) - acoustic_start)
                / max(1, acoustic_length - 1)
            )
            auxiliary.append(np.clip(acoustic_position, -1.0, 2.0).reshape(-1, 1))
        features = np.concatenate((ptm[:total], mfcc[:total], *auxiliary), axis=1)
        weights = np.ones(total, dtype=np.float32)
        binary = canonical_to_binary_labels(labels[:total])
    else:
        features, canonical, weights = load_edge_row(row)
        features = np.asarray(features, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        binary = canonical_to_binary_labels(canonical)
        if expected_ptm_dim is not None and expected_mfcc_dim is not None:
            minimum_width = expected_ptm_dim + expected_mfcc_dim + 1
            if features.ndim != 2 or features.shape[1] < minimum_width:
                raise ValueError(f"Inner v2 frame feature width mismatch: {row.get('row_id')}")
    if features.ndim != 2 or features.shape[0] <= 0:
        raise ValueError(f"Inner v2 frame features must be non-empty 2D: {row.get('row_id')}")
    if binary.ndim != 1 or weights.ndim != 1:
        raise ValueError(f"Inner v2 labels/weights must be 1D: {row.get('row_id')}")
    if len({features.shape[0], binary.shape[0], weights.shape[0]}) != 1:
        raise ValueError(f"Inner v2 feature/label/weight lengths differ: {row.get('row_id')}")
    if not np.isfinite(features).all() or not np.isfinite(weights).all():
        raise ValueError(f"Inner v2 features or weights contain non-finite values: {row.get('row_id')}")
    if np.any(weights < 0.0):
        raise ValueError(f"Inner v2 source weights must be non-negative: {row.get('row_id')}")
    return features, binary, weights


def compute_normalization(rows: list[dict]) -> dict[str, list[float]]:
    first, _labels, _weights = load_binary(rows[0])
    feature_sum = np.zeros(first.shape[1], dtype=np.float64)
    square_sum = np.zeros(first.shape[1], dtype=np.float64)
    frame_count = 0
    for row in rows:
        features, labels, weights = load_binary(row)
        if features.shape[1] != feature_sum.shape[0]:
            raise ValueError("Inner v2 feature width changes across dataset rows")
        valid = (labels != BINARY_EDGE_IGNORE_INDEX) & (weights > 0.0)
        values = features[valid].astype(np.float64)
        feature_sum += values.sum(axis=0)
        square_sum += np.square(values).sum(axis=0)
        frame_count += len(values)
    if frame_count <= 0:
        raise ValueError("Inner v2 train partition has no weighted definite frames")
    mean = feature_sum / frame_count
    variance = square_sum / frame_count - np.square(mean)
    return {
        "feature_mean": mean.astype(np.float32).tolist(),
        "feature_std": np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32).tolist(),
    }


def evaluate(model, rows, normalization, device, tolerance_frames: int) -> dict:
    import torch

    start_hits = end_hits = count = all_background = 0
    start_errors, end_errors = [], []
    tp = fp = fn = 0
    model.eval()
    with torch.inference_mode():
        for row in rows:
            features, labels, weights = load_binary(row)
            valid = (labels != BINARY_EDGE_IGNORE_INDEX) & (weights > 0.0)
            truth = np.flatnonzero((labels == 1) & valid)
            if not truth.size:
                continue
            logits = model(torch.from_numpy(normalize_edge_features(features, normalization)).unsqueeze(0).to(device))[0].cpu().numpy()
            predicted = np.argmax(logits, axis=1)
            tp += int(np.sum((predicted[valid] == 1) & (labels[valid] == 1)))
            fp += int(np.sum((predicted[valid] == 1) & (labels[valid] == 0)))
            fn += int(np.sum((predicted[valid] == 0) & (labels[valid] == 1)))
            try:
                start, end = decode_binary_edge_logits(logits, raw_start_s=0.0, raw_end_s=float(len(labels)), frame_hop_s=1.0)
                ps, pe = int(round(start)), int(round(end))
            except ValueError:
                all_background += 1
                count += 1
                continue
            start_error, end_error = abs(ps - int(truth[0])), abs(pe - int(truth[-1] + 1))
            start_errors.append(start_error); end_errors.append(end_error)
            start_hits += int(start_error <= tolerance_frames); end_hits += int(end_error <= tolerance_frames); count += 1
    return {
        "count": count,
        "start_coverage": start_hits / max(count, 1), "end_coverage": end_hits / max(count, 1),
        "start_mae_frames": float(np.mean(start_errors)) if start_errors else float("inf"),
        "end_mae_frames": float(np.mean(end_errors)) if end_errors else float("inf"),
        "semantic_precision": tp / max(tp + fp, 1), "semantic_recall": tp / max(tp + fn, 1),
        "all_background_count": all_background,
    }


def run(args: argparse.Namespace) -> dict:
    import torch
    import torch.nn.functional as F

    apply_vram_safety_cap(0.95)
    if args.max_steps <= 0 or args.eval_interval <= 0:
        raise ValueError("Inner v2 max_steps/eval_interval must be positive")
    if args.frame_hop_s <= 0.0 or args.tolerance_s < 0.0:
        raise ValueError("Inner v2 frame_hop_s must be positive and tolerance non-negative")
    _seed_everything(int(args.seed), torch)
    rows = read_edge_rows(Path(args.dataset_manifest))
    dataset_summary = validate_dataset_rows(rows)
    train_rows = [row for row in rows if str(row.get("partition")) == "train"]
    val_rows = [row for row in rows if str(row.get("partition")) == "val"]
    test_rows = [row for row in rows if str(row.get("partition")) == "test"]
    if not train_rows or not val_rows or not test_rows:
        raise ValueError("Inner v2 requires fixed train/val/test partitions")
    expected_width: int | None = None
    for row in rows:
        features, _labels, _weights = load_binary(
            row,
            expected_ptm_dim=args.raw_ptm_dim,
            expected_mfcc_dim=args.mfcc_dim,
        )
        if expected_width is None:
            expected_width = int(features.shape[1])
        elif int(features.shape[1]) != expected_width:
            raise ValueError("Inner v2 feature width changes across dataset rows")
    normalization = compute_normalization(train_rows)
    first, _labels, _weights = load_binary(
        train_rows[0],
        expected_ptm_dim=args.raw_ptm_dim,
        expected_mfcc_dim=args.mfcc_dim,
    )
    position_dim = int(first.shape[1]) - args.raw_ptm_dim - args.mfcc_dim
    model_config = {
        "ptm_input_dim": args.raw_ptm_dim, "ptm_projected_dim": args.projected_ptm_dim,
        "mfcc_dim": args.mfcc_dim, "position_dim": position_dim, "hidden_size": args.hidden_size,
        "num_layers": args.num_layers, "state_size": 32, "num_heads": 4, "head_dim": 64,
        "n_groups": 2, "conv_kernel": 4, "chunk_size": 8, "bidirectional": True, "output_dim": 2,
    }
    if position_dim <= 0:
        raise ValueError("Inner v2 frame feature dimension mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Inner v2 requested CUDA but CUDA is unavailable")
    model = BinaryFrameEdgeNetwork(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    train_order = np.empty(0, dtype=np.int64)
    train_position = 0
    counts = Counter()
    for row in rows:
        if row.get("label_path"):
            with np.load(row["label_path"]) as payload:
                canonical = payload["labels"].astype(np.int64)
        else:
            _f, canonical, _w = load_edge_row(row)
        counts.update(
            background=int(np.sum(canonical == 0)),
            semantic_target=int(np.sum(canonical == 1)),
            unsure=int(np.sum(canonical == 2)),
        )
    best_score = (-1.0, -1.0, -1.0); best_step = 0; best_state = None; started = time.monotonic()
    for step in range(1, args.max_steps + 1):
        if train_position >= len(train_order):
            train_order = rng.permutation(len(train_rows))
            train_position = 0
        row = train_rows[int(train_order[train_position])]
        train_position += 1
        features, labels, source_weights = load_binary(row)
        valid = (labels != BINARY_EDGE_IGNORE_INDEX) & (source_weights > 0.0)
        if not np.any(valid):
            continue
        model.train()
        logits = model(torch.from_numpy(normalize_edge_features(features, normalization)).unsqueeze(0).to(device))[0]
        target = torch.from_numpy(labels).to(device)
        ce = F.cross_entropy(logits, target, reduction="none", ignore_index=BINARY_EDGE_IGNORE_INDEX)
        class_weights = np.where(
            labels == 0, float(args.background_weight), float(args.semantic_weight)
        ).astype(np.float32)
        weights = torch.from_numpy(
            source_weights.astype(np.float32) * class_weights
        ).to(device)
        valid_t = (target != BINARY_EDGE_IGNORE_INDEX) & (weights > 0.0)
        loss = (ce[valid_t] * weights[valid_t]).sum() / weights[valid_t].sum().clamp_min(1e-6)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step % args.eval_interval == 0 or step == args.max_steps:
            val = evaluate(model, val_rows, normalization, device, int(round(args.tolerance_s / args.frame_hop_s)))
            score = (float(val["all_background_count"] == 0), min(val["start_coverage"], val["end_coverage"]), val["semantic_recall"])
            print(json.dumps({"step": step, "loss": float(loss.detach()), "val": val}), flush=True)
            if score > best_score:
                best_score, best_step, best_state = score, step, copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Inner v2 produced no checkpoint")
    model.load_state_dict(best_state)
    tolerance_frames = int(round(args.tolerance_s / args.frame_hop_s))
    train = evaluate(model, train_rows, normalization, device, tolerance_frames)
    val = evaluate(model, val_rows, normalization, device, tolerance_frames)
    test = evaluate(model, test_rows, normalization, device, tolerance_frames)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / f"inner_edge_refiner_v2.{qwen_asr_repo_tag(QWEN_ASR_17B_REPO_ID)}.pt"
    checkpoint_payload = build_inner_edge_refiner_v2_checkpoint(
        model=model, model_config=model_config,
        feature_config={"raw_ptm_dim": args.raw_ptm_dim, "learned_ptm_projected_dim": args.projected_ptm_dim, "mfcc_dim": args.mfcc_dim, "relative_position_dim": position_dim, "frame_hop_s": args.frame_hop_s, "acoustic_refinement": True},
        normalization=normalization,
        metadata={"ptm_repo_id": QWEN_ASR_17B_REPO_ID, "dataset_manifest": args.dataset_manifest, "dataset_manifest_sha256": _sha256(Path(args.dataset_manifest)), "dataset_summary": dataset_summary, "input_distribution": INNER_INPUT_DISTRIBUTION, "upstream_cueqc_checkpoint_sha256": dataset_summary["cueqc_checkpoint_sha256"], "trained_steps": args.max_steps, "best_step": best_step, "canonical_label_counts": dict(counts), "excluded_training_count": int(counts["unsure"]), "training_initialization": "random", "checkpoint_selection": "val_inner_acoustic_edge_tolerance_v2", "evaluation_tolerance_s": float(args.tolerance_s), "evaluation_tolerance_frames": tolerance_frames, "class_weights": {"background": float(args.background_weight), "semantic_core": float(args.semantic_weight)}, "acoustic_refinement": True, "feeds_asr": True, "promotion_ready": False},
    )
    _torch_save_atomic(checkpoint_payload, checkpoint)
    numeric_gate_pass = min(val["start_coverage"], val["end_coverage"], test["start_coverage"], test["end_coverage"]) >= 0.95 and not train["all_background_count"] and not val["all_background_count"] and not test["all_background_count"]
    summary = {"schema": "inner_edge_refiner_v2_binary_training_summary_v2", "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint), "best_step": best_step, "train": train, "val": val, "test": test, "dataset": dataset_summary, "dataset_manifest_sha256": _sha256(Path(args.dataset_manifest)), "upstream_cueqc_checkpoint_sha256": dataset_summary["cueqc_checkpoint_sha256"], "canonical_label_counts": dict(counts), "excluded_training_count": int(counts["unsure"]), "evaluation_tolerance_s": float(args.tolerance_s), "evaluation_tolerance_frames": tolerance_frames, "numeric_gate_pass": bool(numeric_gate_pass), "gate_pass": False, "promotion_ready": False, "manual_zero_clipping_gate": "required_before_promotion", "acoustic_refinement": True, "feeds_asr": True, "elapsed_s": time.monotonic() - started}
    _write_json_atomic(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False)); return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 1.7B binary acoustic Inner Edge Refiner v2.")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--raw-ptm-dim", type=int, default=2048); parser.add_argument("--projected-ptm-dim", type=int, default=128); parser.add_argument("--mfcc-dim", type=int, default=40); parser.add_argument("--frame-hop-s", type=float, default=0.02); parser.add_argument("--max-steps", type=int, default=1500); parser.add_argument("--eval-interval", type=int, default=100); parser.add_argument("--learning-rate", type=float, default=5e-5); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--background-weight", type=float, default=5.1); parser.add_argument("--semantic-weight", type=float, default=1.0); parser.add_argument("--hidden-size", type=int, default=128); parser.add_argument("--num-layers", type=int, default=2); parser.add_argument("--tolerance-s", type=float, default=0.3); parser.add_argument("--seed", type=int, default=17); parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
