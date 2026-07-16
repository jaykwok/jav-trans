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

from asr.backends.qwen import (  # noqa: E402
    DEFAULT_INNER_EDGE_REFINER_CHECKPOINT_BY_REPO,
    DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO,
    QWEN_ASR_17B_REPO_ID,
    checkpoint_path_for_repo_env,
    qwen_asr_repo_id,
    qwen_asr_repo_tag,
)
from asr.pre_asr_cueqc import (  # noqa: E402
    PRE_ASR_CUEQC_ARTIFACT,
    PRE_ASR_CUEQC_FEATURE_SCHEMA,
    PRE_ASR_CUEQC_IGNORE_LABEL,
    PRE_ASR_CUEQC_MODEL_PTM_TOKENS,
    PRE_ASR_CUEQC_MODEL_ARCH,
    PRE_ASR_CUEQC_PTM_DIM,
    PRE_ASR_CUEQC_RUNTIME_ADAPTER,
    PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES,
    PRE_ASR_CUEQC_SCHEMA,
    PreAsrCueQCNetwork,
    make_model_config,
)
from tools.asr.cueqc.pre_asr_feature_compiler import (  # noqa: E402
    FEATURE_BUNDLE_SCHEMA,
    project_path,
    repo_display_path,
)
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402


METRICS_SCHEMA = "cueqc_pre_asr_semantic_chunk_v13_train_metrics"


def default_checkpoint_name(asr_repo_id: str) -> str:
    return f"pre_asr_cueqc_v13.{qwen_asr_repo_tag(asr_repo_id)}.pt"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def active_semantic_split_checkpoint(asr_repo_id: str) -> Path:
    return Path(
        checkpoint_path_for_repo_env(
            repo_id=asr_repo_id,
            mapping_env="SEMANTIC_SPLIT_MODEL_PATH_BY_REPO",
            default_mapping=DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO,
        )
    )


def active_inner_edge_refiner_checkpoint(asr_repo_id: str) -> Path:
    return Path(
        checkpoint_path_for_repo_env(
            repo_id=asr_repo_id,
            mapping_env="INNER_EDGE_REFINER_MODEL_PATH_BY_REPO",
            default_mapping=DEFAULT_INNER_EDGE_REFINER_CHECKPOINT_BY_REPO,
        )
    )


def load_feature_bundle(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("feature bundle must be a mapping")
    if payload.get("schema") != FEATURE_BUNDLE_SCHEMA:
        raise ValueError(f"unsupported feature bundle schema: {payload.get('schema')!r}")
    if tuple(payload.get("feature_names") or ()) != PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES:
        raise ValueError("feature bundle feature_names do not match Pre-ASR CueQC v13 runtime")
    if payload.get("feature_schema") != PRE_ASR_CUEQC_FEATURE_SCHEMA:
        raise ValueError("feature bundle feature_schema mismatch")
    if payload.get("runtime_adapter") != PRE_ASR_CUEQC_RUNTIME_ADAPTER:
        raise ValueError("feature bundle runtime_adapter mismatch")
    ptm_tensor = payload.get("ptm_bins")
    ptm_bin_count = payload.get("ptm_bin_count")
    if ptm_bin_count is None and hasattr(ptm_tensor, "shape") and len(ptm_tensor.shape) >= 3:
        ptm_bin_count = int(ptm_tensor.shape[2])
    if int(ptm_bin_count or 0) != PRE_ASR_CUEQC_MODEL_PTM_TOKENS:
        raise ValueError("feature bundle ptm_bins mismatch")
    ptm_dim = payload.get("ptm_dim")
    if ptm_dim is None and hasattr(ptm_tensor, "shape") and len(ptm_tensor.shape) >= 4:
        ptm_dim = int(ptm_tensor.shape[3])
    if int(ptm_dim or 0) != PRE_ASR_CUEQC_PTM_DIM:
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
) -> dict[str, float]:
    if probs.size == 0:
        return {
            "drop_precision": 0.0,
            "drop_recall": 0.0,
            "drop_f1": 0.0,
            "semantic_keep_recall": 0.0,
            "unsure_recall": 0.0,
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
    predicted = np.argmax(probs, axis=1)
    pred_drop = predicted == 0
    pred_keep = predicted == 1
    true_drop = y == 0
    true_keep = y == 1
    true_unsure = y == 2
    tp = int(np.sum(pred_drop & true_drop))
    fp = int(np.sum(pred_drop & ~true_drop))
    fn = int(np.sum(~pred_drop & true_drop))
    tn = int(np.sum(pred_keep & true_keep))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    keep_total = int(np.sum(true_keep))
    keep_recall = tn / keep_total if keep_total else 0.0
    unsure_total = int(np.sum(true_unsure))
    unsure_recall = (
        int(np.sum((predicted == 2) & true_unsure)) / unsure_total
        if unsure_total
        else 0.0
    )
    false_drop_duration = float(np.sum(durations[pred_drop & true_keep]))
    false_keep_duration = float(np.sum(durations[~pred_drop & true_drop]))
    drop_duration = float(np.sum(durations[pred_drop]))
    total_duration = float(np.sum(durations))
    return {
        "drop_precision": precision,
        "drop_recall": recall,
        "drop_f1": f1,
        "semantic_keep_recall": keep_recall,
        "unsure_recall": unsure_recall,
        "false_drop_rate": int(np.sum(pred_drop & true_keep)) / max(1, keep_total),
        "false_drop_count": float(np.sum(pred_drop & true_keep)),
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


def _class_counts(y: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    valid = mask > 0
    return {
        "drop": int(np.sum((y == 0) & valid)),
        "keep": int(np.sum((y == 1) & valid)),
        "unsure": int(np.sum((y == 2) & valid)),
        "ambiguous_ignore": int(np.sum((y == PRE_ASR_CUEQC_IGNORE_LABEL) & valid)),
    }


def _group_label_counts(y: np.ndarray, mask: np.ndarray, group_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group_index in range(y.shape[0]):
        group = dict(group_rows[group_index]) if group_index < len(group_rows) else {}
        out.append(
            {
                "group_index": group_index,
                "audio_id": str(group.get("audio_id") or ""),
                "planned_island_id": str(group.get("planned_island_id") or ""),
                **_class_counts(y[group_index : group_index + 1], mask[group_index : group_index + 1]),
            }
        )
    return out


def _split_label_masks(
    *,
    y: np.ndarray,
    chunk_mask: np.ndarray,
    group_rows: list[Mapping[str, Any]],
    split_mode: str,
    val_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    valid = (chunk_mask > 0) & ((y == 0) | (y == 1) | (y == 2))
    train = np.zeros_like(valid, dtype=bool)
    val = np.zeros_like(valid, dtype=bool)
    if split_mode == "role_holdout":
        for group_index, group in enumerate(group_rows):
            role = str(group.get("dataset_role") or "").strip().lower()
            target = (
                val
                if role in {"semantic", "val", "validation", "holdout"}
                else train
            )
            target[group_index] = valid[group_index]
    elif split_mode in {"group", "video_group"}:
        group_count = int(y.shape[0])
        if split_mode == "video_group":
            video_ids = [
                str(group.get("video_id") or group.get("audio_id") or index)
                for index, group in enumerate(group_rows)
            ]
            unique_videos = sorted(set(video_ids))
            order = rng.permutation(len(unique_videos))
            val_count = (
                max(1, int(round(len(unique_videos) * val_ratio)))
                if len(unique_videos) >= 2
                else 0
            )
            val_videos = {unique_videos[int(item)] for item in order[:val_count]}
            val_groups = {
                index for index, video_id in enumerate(video_ids) if video_id in val_videos
            }
            fallback_order = rng.permutation(group_count)
        else:
            order = rng.permutation(group_count)
            val_count = max(1, int(round(group_count * val_ratio))) if group_count >= 2 else 0
            val_groups = set(int(item) for item in order[:val_count])
            fallback_order = order
        for group_index in range(group_count):
            target = val if group_index in val_groups else train
            target[group_index] = valid[group_index]
        if not np.any(train & valid) and np.any(valid):
            first = int(fallback_order[-1])
            train[first] = valid[first]
            val[first] = False
    elif split_mode == "chunk_stratified":
        for group_index in range(y.shape[0]):
            for label_index in (0, 1, 2):
                positions = np.flatnonzero(valid[group_index] & (y[group_index] == label_index))
                if positions.size == 0:
                    continue
                shuffled = rng.permutation(positions)
                if shuffled.size <= 1:
                    val_count = 0
                else:
                    val_count = int(round(shuffled.size * val_ratio))
                    val_count = min(shuffled.size - 1, max(1, val_count))
                if val_count:
                    val[group_index, shuffled[:val_count]] = True
                    train[group_index, shuffled[val_count:]] = True
                else:
                    train[group_index, shuffled] = True
    else:
        raise ValueError(f"unsupported split_mode: {split_mode!r}")
    if not np.any(train & valid):
        raise ValueError("training split has no CueQC labels")
    if not np.any(val & valid):
        raise ValueError("validation split has no CueQC labels")
    train_group_count = int(np.sum(np.any(train, axis=1)))
    val_group_count = int(np.sum(np.any(val, axis=1)))
    summary = {
        "mode": split_mode,
        "val_ratio": float(val_ratio),
        "train_group_count": train_group_count,
        "val_group_count": val_group_count,
        "all_group_count": int(y.shape[0]),
        "train_counts": _class_counts(y, train),
        "val_counts": _class_counts(y, val),
        "all_counts": _class_counts(y, chunk_mask),
        "groups_train": _group_label_counts(y, train, group_rows),
        "groups_val": _group_label_counts(y, val, group_rows),
    }
    return train, val, summary


def _balanced_anchor_positions(train_mask: np.ndarray, y: np.ndarray, device: Any) -> dict[int, Any]:
    import torch

    positions: dict[int, Any] = {}
    for label_index in (0, 1, 2):
        raw = np.argwhere(train_mask & (y == label_index))
        if raw.size:
            positions[label_index] = torch.as_tensor(raw, dtype=torch.long, device=device)
    if not positions:
        raise ValueError("training split has no label positions")
    return positions


def _matching_group_indexes(
    group_rows: list[Mapping[str, Any]],
    audio_ids: list[str],
) -> set[int]:
    wanted = {str(item).strip() for item in audio_ids if str(item).strip()}
    if not wanted:
        return set()
    return {
        index
        for index, group in enumerate(group_rows)
        if str(group.get("audio_id") or "") in wanted
    }


def _matching_candidate_positions(
    group_rows: list[Mapping[str, Any]],
    candidate_ids: list[str],
) -> set[tuple[int, int]]:
    wanted = {str(item).strip() for item in candidate_ids if str(item).strip()}
    if not wanted:
        return set()
    positions = {
        (group_index, chunk_index)
        for group_index, group in enumerate(group_rows)
        for chunk_index, row_id in enumerate(group.get("row_ids") or ())
        if str(row_id) in wanted
    }
    found = {
        str((group_rows[group_index].get("row_ids") or [])[chunk_index])
        for group_index, chunk_index in positions
    }
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"anchor candidate_ids not found in feature bundle: {missing}")
    return positions


def _boost_anchor_positions(
    positions_by_label: dict[int, Any],
    *,
    group_indexes: set[int],
    candidate_positions: set[tuple[int, int]] | None = None,
    boost: int,
) -> dict[int, Any]:
    candidate_positions = candidate_positions or set()
    if (not group_indexes and not candidate_positions) or int(boost) <= 1:
        return positions_by_label
    import torch

    out: dict[int, Any] = {}
    for label, positions in positions_by_label.items():
        selected = torch.zeros(
            positions.shape[0],
            dtype=torch.bool,
            device=positions.device,
        )
        for group_index in group_indexes:
            selected |= positions[:, 0] == int(group_index)
        for group_index, chunk_index in candidate_positions:
            selected |= (positions[:, 0] == int(group_index)) & (
                positions[:, 1] == int(chunk_index)
            )
        hard_positions = positions[selected]
        if hard_positions.numel():
            positions = torch.cat(
                [positions, *([hard_positions] * (int(boost) - 1))],
                dim=0,
            )
        out[int(label)] = positions
    return out


def _sample_balanced_anchors(
    *,
    positions_by_label: Mapping[int, Any],
    batch_size: int,
    device: Any,
) -> Any:
    import torch

    available = sorted(int(label) for label in positions_by_label)
    anchors = []
    for offset in range(max(1, int(batch_size))):
        if len(available) == 1:
            label = available[0]
        else:
            label = available[int(torch.randint(0, len(available), (1,), device=device).item())]
        positions = positions_by_label[label]
        index = int(torch.randint(0, positions.shape[0], (1,), device=device).item())
        anchors.append(positions[index])
    return torch.stack(anchors, dim=0)


def _window_batch(
    *,
    group_ids: Any,
    ptm_bins: Any,
    scalar: Any,
    chunk_mask: Any,
    bin_mask: Any,
    y: Any,
    sequence_window_size: int,
) -> tuple[Any, Any, Any, Any, Any]:
    if sequence_window_size <= 0 or int(ptm_bins.shape[1]) <= sequence_window_size:
        return (
            ptm_bins[group_ids],
            scalar[group_ids],
            chunk_mask[group_ids],
            bin_mask[group_ids],
            y[group_ids],
        )
    import torch

    window = min(int(sequence_window_size), int(ptm_bins.shape[1]))
    ptm_rows = []
    scalar_rows = []
    chunk_mask_rows = []
    bin_mask_rows = []
    y_rows = []
    for raw_group_id in group_ids.detach().cpu().tolist():
        group_id = int(raw_group_id)
        length = int(chunk_mask[group_id].sum().detach().cpu().item())
        if length <= window:
            start = 0
        else:
            start = int(torch.randint(0, length - window + 1, (1,), device=group_ids.device).item())
        end = start + window
        ptm_rows.append(ptm_bins[group_id, start:end])
        scalar_rows.append(scalar[group_id, start:end])
        chunk_mask_rows.append(chunk_mask[group_id, start:end])
        bin_mask_rows.append(bin_mask[group_id, start:end])
        y_rows.append(y[group_id, start:end])
    return (
        torch.stack(ptm_rows, dim=0),
        torch.stack(scalar_rows, dim=0),
        torch.stack(chunk_mask_rows, dim=0),
        torch.stack(bin_mask_rows, dim=0),
        torch.stack(y_rows, dim=0),
    )


def _window_batch_from_anchors(
    *,
    anchor_positions: Any,
    ptm_bins: Any,
    scalar: Any,
    chunk_mask: Any,
    bin_mask: Any,
    y: Any,
    sequence_window_size: int,
) -> tuple[Any, Any, Any, Any, Any]:
    import torch

    if sequence_window_size <= 0 or int(ptm_bins.shape[1]) <= sequence_window_size:
        group_ids = anchor_positions[:, 0]
        target_rows = torch.full_like(
            y[group_ids],
            PRE_ASR_CUEQC_IGNORE_LABEL,
        )
        for row_index, (_group_id, chunk_index) in enumerate(
            anchor_positions.detach().cpu().tolist()
        ):
            target_rows[row_index, int(chunk_index)] = y[
                int(_group_id), int(chunk_index)
            ]
        return (
            ptm_bins[group_ids],
            scalar[group_ids],
            chunk_mask[group_ids],
            bin_mask[group_ids],
            target_rows,
        )
    window = min(int(sequence_window_size), int(ptm_bins.shape[1]))
    ptm_rows = []
    scalar_rows = []
    chunk_mask_rows = []
    bin_mask_rows = []
    y_rows = []
    for raw_group_id, raw_chunk_index in anchor_positions.detach().cpu().tolist():
        group_id = int(raw_group_id)
        chunk_index = int(raw_chunk_index)
        length = int(chunk_mask[group_id].sum().detach().cpu().item())
        if length <= window:
            start = 0
        else:
            low = max(0, chunk_index - window + 1)
            high = min(chunk_index, length - window)
            if low <= high:
                start = int(torch.randint(low, high + 1, (1,), device=anchor_positions.device).item())
            else:
                start = min(max(0, chunk_index - window // 2), length - window)
        end = start + window
        ptm_rows.append(ptm_bins[group_id, start:end])
        scalar_rows.append(scalar[group_id, start:end])
        chunk_mask_rows.append(chunk_mask[group_id, start:end])
        bin_mask_rows.append(bin_mask[group_id, start:end])
        target = torch.full_like(
            y[group_id, start:end],
            PRE_ASR_CUEQC_IGNORE_LABEL,
        )
        target[chunk_index - start] = y[group_id, chunk_index]
        y_rows.append(target)
    return (
        torch.stack(ptm_rows, dim=0),
        torch.stack(scalar_rows, dim=0),
        torch.stack(chunk_mask_rows, dim=0),
        torch.stack(bin_mask_rows, dim=0),
        torch.stack(y_rows, dim=0),
    )


def _predict_logits_windowed(
    *,
    model: Any,
    ptm_bins: Any,
    scalar: Any,
    chunk_mask: Any,
    bin_mask: Any,
    sequence_window_size: int,
) -> Any:
    if sequence_window_size <= 0 or int(ptm_bins.shape[1]) <= sequence_window_size:
        return model(ptm_bins, scalar, chunk_mask=chunk_mask, bin_mask=bin_mask)
    import torch

    group_count, max_chunks = tuple(chunk_mask.shape)
    logits_all = torch.zeros((group_count, max_chunks, 2), dtype=torch.float32, device=ptm_bins.device)
    counts = torch.zeros((group_count, max_chunks, 1), dtype=torch.float32, device=ptm_bins.device)
    window = min(int(sequence_window_size), int(max_chunks))
    for group_index in range(int(group_count)):
        length = int(chunk_mask[group_index].sum().detach().cpu().item())
        if length <= 0:
            continue
        for start in range(0, length, window):
            end = min(length, start + window)
            logits = model(
                ptm_bins[group_index : group_index + 1, start:end],
                scalar[group_index : group_index + 1, start:end],
                chunk_mask=chunk_mask[group_index : group_index + 1, start:end],
                bin_mask=bin_mask[group_index : group_index + 1, start:end],
            )
            logits_all[group_index, start:end] += logits[0].float()
            counts[group_index, start:end] += 1.0
    return logits_all / counts.clamp_min(1.0)


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
    keep_class_weight: float,
    drop_class_weight: float,
    unsure_class_weight: float,
    sequence_window_size: int,
    temporal_residual_scale: float,
    split_mode: str,
    val_ratio: float,
    focal_gamma: float,
    init_checkpoint: Path | None,
    force_train_audio_ids: list[str],
    anchor_boost_audio_ids: list[str],
    anchor_boost: int,
    semantic_split_checkpoint: Path | None,
    inner_edge_refiner_checkpoint: Path | None,
    force_val_audio_ids: list[str] | None = None,
    anchor_boost_candidate_ids: list[str] | None = None,
    valid_prefix_temporal: bool = False,
    ptm_encoder_mode: str = "summary_mlp",
    semantic_auxiliary: bool = False,
    semantic_aux_loss_weight: float = 0.0,
    late_fusion: bool = False,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    apply_vram_safety_cap(0.95)
    bundle = load_feature_bundle(features_path)
    scalar = bundle["scalar_features"].float()
    ptm_bins = bundle["ptm_bins"].float()
    bin_mask = bundle["bin_mask"].float()
    chunk_mask = bundle["chunk_mask"].float()
    y = bundle["labels"].long()
    if scalar.ndim != 3 or scalar.shape[2] != len(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES):
        raise ValueError("scalar feature tensor shape mismatch")
    if ptm_bins.ndim != 4 or ptm_bins.shape[2:] != (
        PRE_ASR_CUEQC_MODEL_PTM_TOKENS,
        PRE_ASR_CUEQC_PTM_DIM,
    ):
        raise ValueError("ptm bin tensor shape mismatch")
    if y.shape != chunk_mask.shape or y.shape != scalar.shape[:2]:
        raise ValueError("label tensor shape mismatch")
    selected_repo = qwen_asr_repo_id(asr_repo_id)
    if selected_repo != QWEN_ASR_17B_REPO_ID:
        raise ValueError("CueQC v13 training is restricted to the 1.7B repo")
    split_checkpoint_path = semantic_split_checkpoint or active_semantic_split_checkpoint(selected_repo)
    if not split_checkpoint_path.exists():
        raise FileNotFoundError(f"semantic split checkpoint not found: {split_checkpoint_path}")
    split_checkpoint_sha256 = file_sha256(split_checkpoint_path)
    inner_checkpoint_path = (
        inner_edge_refiner_checkpoint
        or active_inner_edge_refiner_checkpoint(selected_repo)
    )
    if not inner_checkpoint_path.exists():
        raise FileNotFoundError(
            f"inner edge refiner checkpoint not found: {inner_checkpoint_path}"
        )
    inner_checkpoint_sha256 = file_sha256(inner_checkpoint_path)
    bundle_repo = qwen_asr_repo_id(str(bundle.get("asr_repo_id") or selected_repo))
    if bundle_repo != selected_repo:
        raise ValueError(f"feature bundle asr_repo_id={bundle_repo!r} does not match {selected_repo!r}")
    ptm_projection_digest = str(bundle.get("ptm_projection_digest") or "")
    ptm_pooling_schemas = [str(item) for item in bundle.get("ptm_pooling_schemas") or ()]

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    group_count = int(scalar.shape[0])
    y_np = y.numpy()
    mask_np = chunk_mask.numpy()
    group_rows = [dict(item) for item in (bundle.get("groups") or []) if isinstance(item, Mapping)]
    train_label_mask, val_label_mask, split_summary = _split_label_masks(
        y=y_np,
        chunk_mask=mask_np,
        group_rows=group_rows,
        split_mode=split_mode,
        val_ratio=val_ratio,
        rng=rng,
    )
    force_train_groups = _matching_group_indexes(group_rows, force_train_audio_ids)
    force_val_groups = _matching_group_indexes(group_rows, force_val_audio_ids or [])
    overlap_groups = force_train_groups & force_val_groups
    if overlap_groups:
        overlap_ids = sorted(
            str(group_rows[index].get("audio_id") or "") for index in overlap_groups
        )
        raise ValueError(f"audio_ids forced into both train and validation: {overlap_ids}")
    for group_index in force_train_groups:
        valid = (mask_np[group_index] > 0) & (
            (y_np[group_index] == 0)
            | (y_np[group_index] == 1)
            | (y_np[group_index] == 2)
        )
        train_label_mask[group_index, valid] = True
        val_label_mask[group_index, valid] = False
    for group_index in force_val_groups:
        valid = (mask_np[group_index] > 0) & (
            (y_np[group_index] == 0)
            | (y_np[group_index] == 1)
            | (y_np[group_index] == 2)
        )
        train_label_mask[group_index, valid] = False
        val_label_mask[group_index, valid] = True
    if force_train_groups or force_val_groups:
        if not np.any(train_label_mask):
            raise ValueError("forced split leaves no training labels")
        if not np.any(val_label_mask):
            raise ValueError("forced split leaves no validation labels")
        split_summary.update(
            {
                "forced_train_audio_ids": sorted(
                    str(group_rows[index].get("audio_id") or "")
                    for index in force_train_groups
                ),
                "forced_val_audio_ids": sorted(
                    str(group_rows[index].get("audio_id") or "")
                    for index in force_val_groups
                ),
                "train_group_count": int(np.sum(np.any(train_label_mask, axis=1))),
                "val_group_count": int(np.sum(np.any(val_label_mask, axis=1))),
                "train_counts": _class_counts(y_np, train_label_mask),
                "val_counts": _class_counts(y_np, val_label_mask),
                "groups_train": _group_label_counts(
                    y_np, train_label_mask, group_rows
                ),
                "groups_val": _group_label_counts(
                    y_np, val_label_mask, group_rows
                ),
            }
        )

    train_label_mask_t = torch.from_numpy(train_label_mask)
    init_payload: dict[str, Any] | None = None
    if init_checkpoint is not None:
        raw_init = torch.load(
            init_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(raw_init, Mapping):
            raise ValueError("init checkpoint must be a mapping")
        init_payload = dict(raw_init)
        if init_payload.get("schema") != PRE_ASR_CUEQC_SCHEMA:
            raise ValueError("init checkpoint schema mismatch")
        if init_payload.get("arch") != PRE_ASR_CUEQC_MODEL_ARCH:
            raise ValueError("init checkpoint architecture mismatch")
        if tuple(init_payload.get("feature_names") or ()) != PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES:
            raise ValueError("init checkpoint feature_names mismatch")
        init_metadata = init_payload.get("metadata")
        if isinstance(init_metadata, Mapping):
            init_repo = qwen_asr_repo_id(
                str(init_metadata.get("asr_repo_id") or selected_repo)
            )
            if init_repo != selected_repo:
                raise ValueError(
                    f"init checkpoint asr_repo_id={init_repo!r} does not match "
                    f"{selected_repo!r}"
                )
        mean = torch.as_tensor(
            init_payload.get("feature_mean"),
            dtype=torch.float32,
        )
        std = torch.as_tensor(
            init_payload.get("feature_std"),
            dtype=torch.float32,
        )
        if mean.shape != (len(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES),):
            raise ValueError("init checkpoint feature_mean shape mismatch")
        if std.shape != mean.shape:
            raise ValueError("init checkpoint feature_std shape mismatch")
        std = std.clamp_min(1e-6)
    else:
        scalar_train = scalar[train_label_mask_t]
        mean = scalar_train.mean(dim=0)
        std = scalar_train.std(dim=0).clamp_min(1e-6)
    scalar_norm = (scalar - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    scalar_norm = torch.nan_to_num(scalar_norm, nan=0.0, posinf=0.0, neginf=0.0)

    normalized_device = device.strip().lower()
    if normalized_device == "auto":
        normalized_device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(normalized_device)
    if init_payload is not None:
        model_config = make_model_config(init_payload.get("model_config"))
    else:
        model_config = make_model_config(
            {
                "ptm_dim": PRE_ASR_CUEQC_PTM_DIM,
                "scalar_dim": len(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES),
                "hidden_size": hidden_size,
                "temporal_residual_scale": temporal_residual_scale,
                "valid_prefix_temporal": bool(valid_prefix_temporal),
                "ptm_encoder_mode": str(ptm_encoder_mode),
                "semantic_auxiliary": bool(semantic_auxiliary),
                "late_fusion": bool(late_fusion),
                "num_classes": 3,
            }
        )
    model = PreAsrCueQCNetwork(**model_config).to(dev)
    if init_payload is not None:
        model.load_state_dict(init_payload["model_state_dict"], strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = torch.tensor(
        [
            float(drop_class_weight),
            float(keep_class_weight),
            float(unsure_class_weight),
        ],
        dtype=torch.float32,
        device=dev,
    )
    ptm_bins_d = ptm_bins.to(dev)
    scalar_d = scalar_norm.to(dev)
    chunk_mask_d = chunk_mask.to(dev)
    bin_mask_d = bin_mask.to(dev)
    y_train = y.clone()
    y_train[~train_label_mask_t] = PRE_ASR_CUEQC_IGNORE_LABEL
    y_train_d = y_train.to(dev)
    positions_by_label = _balanced_anchor_positions(train_label_mask, y_np, dev)
    anchor_boost_groups = _matching_group_indexes(
        group_rows,
        anchor_boost_audio_ids,
    )
    anchor_boost_candidates = _matching_candidate_positions(
        group_rows,
        anchor_boost_candidate_ids or [],
    )
    positions_by_label = _boost_anchor_positions(
        positions_by_label,
        group_indexes=anchor_boost_groups,
        candidate_positions=anchor_boost_candidates,
        boost=anchor_boost,
    )
    batch_size = max(1, int(batch_size))
    for _step in range(max(1, int(steps))):
        anchor_positions = _sample_balanced_anchors(
            positions_by_label=positions_by_label,
            batch_size=batch_size,
            device=dev,
        )
        batch_ptm, batch_scalar, batch_chunk_mask, batch_bin_mask, batch_y = _window_batch_from_anchors(
            anchor_positions=anchor_positions,
            ptm_bins=ptm_bins_d,
            scalar=scalar_d,
            chunk_mask=chunk_mask_d,
            bin_mask=bin_mask_d,
            y=y_train_d,
            sequence_window_size=sequence_window_size,
        )
        model_output = model(
            batch_ptm,
            batch_scalar,
            chunk_mask=batch_chunk_mask,
            bin_mask=batch_bin_mask,
            return_auxiliary=bool(semantic_auxiliary),
        )
        if isinstance(model_output, tuple):
            logits, auxiliary = model_output
        else:
            logits = model_output
            auxiliary = {}
        flat_logits = logits.reshape(-1, 3)
        flat_targets = batch_y.reshape(-1)
        active = flat_targets != PRE_ASR_CUEQC_IGNORE_LABEL
        active_logits = flat_logits[active]
        active_targets = flat_targets[active]
        raw_loss = F.cross_entropy(
            active_logits,
            active_targets,
            weight=class_weights,
            reduction="none",
        )
        pt = torch.softmax(active_logits, dim=-1).gather(
            1, active_targets.unsqueeze(1)
        ).squeeze(1)
        loss = (raw_loss * torch.pow(1.0 - pt, focal_gamma)).mean()
        if semantic_auxiliary and semantic_aux_loss_weight > 0.0:
            semantic_logits = auxiliary["semantic_logits"].reshape(-1, 3)[active]
            semantic_raw_loss = F.cross_entropy(
                semantic_logits,
                active_targets,
                weight=class_weights,
                reduction="none",
            )
            semantic_pt = torch.softmax(semantic_logits, dim=-1).gather(
                1, active_targets.unsqueeze(1)
            ).squeeze(1)
            semantic_loss = (
                semantic_raw_loss * torch.pow(1.0 - semantic_pt, focal_gamma)
            ).mean()
            loss = loss + float(semantic_aux_loss_weight) * semantic_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    with torch.inference_mode():
        logits_all = _predict_logits_windowed(
            model=model,
            ptm_bins=ptm_bins_d,
            scalar=scalar_d,
            chunk_mask=chunk_mask_d,
            bin_mask=bin_mask_d,
            sequence_window_size=sequence_window_size,
        )
        probs_all = torch.softmax(logits_all, dim=-1).float().cpu().numpy()
    durations = _duration_matrix(bundle, scalar)
    train_probs, train_y, train_durations = _valid_flat(
        probs_all,
        y_np,
        train_label_mask.astype(np.float32),
        durations,
    )
    val_probs, val_y, val_durations = _valid_flat(
        probs_all,
        y_np,
        val_label_mask.astype(np.float32),
        durations,
    )
    all_probs, all_y, all_durations = _valid_flat(probs_all, y_np, mask_np, durations)
    created_at = datetime.now().isoformat(timespec="seconds")
    metrics = {
        "schema": METRICS_SCHEMA,
        "created_at": created_at,
        "features": repo_display_path(features_path),
        "feature_sha256": file_sha256(features_path),
        "asr_repo_id": selected_repo,
        "semantic_split_checkpoint": repo_display_path(split_checkpoint_path),
        "semantic_split_weights_sha256": split_checkpoint_sha256,
        "inner_edge_refiner_checkpoint": repo_display_path(inner_checkpoint_path),
        "inner_edge_refiner_weights_sha256": inner_checkpoint_sha256,
        "ptm_projection_digest": ptm_projection_digest,
        "ptm_pooling_schemas": ptm_pooling_schemas,
        "split": split_summary,
        "train_group_count": int(split_summary["train_group_count"]),
        "val_group_count": int(split_summary["val_group_count"]),
        "all_group_count": group_count,
        "class_counts": _class_counts(y_np, mask_np),
        "class_weights": {
            "drop": float(drop_class_weight),
            "keep": float(keep_class_weight),
            "unsure": float(unsure_class_weight),
        },
        "focal_gamma": float(focal_gamma),
        "valid_prefix_temporal": bool(model_config["valid_prefix_temporal"]),
        "ptm_encoder_mode": str(model_config["ptm_encoder_mode"]),
        "semantic_auxiliary": bool(model_config["semantic_auxiliary"]),
        "semantic_aux_loss_weight": float(semantic_aux_loss_weight),
        "late_fusion": bool(model_config["late_fusion"]),
        "sequence_window_size": int(sequence_window_size),
        "init_checkpoint": (
            repo_display_path(init_checkpoint)
            if init_checkpoint is not None
            else None
        ),
        "force_train_audio_ids": sorted(
            str(group_rows[index].get("audio_id") or "")
            for index in force_train_groups
        ),
        "force_val_audio_ids": sorted(
            str(group_rows[index].get("audio_id") or "")
            for index in force_val_groups
        ),
        "anchor_boost_audio_ids": sorted(
            str(group_rows[index].get("audio_id") or "")
            for index in anchor_boost_groups
        ),
        "anchor_boost_candidate_ids": sorted(
            str((group_rows[group_index].get("row_ids") or [])[chunk_index])
            for group_index, chunk_index in anchor_boost_candidates
        ),
        "anchor_boost": int(anchor_boost),
        "model_config": model_config,
        "decision_mode": "argmax",
        "train": classification_metrics(train_probs, train_y, train_durations),
        "val": classification_metrics(val_probs, val_y, val_durations),
        "all": classification_metrics(all_probs, all_y, all_durations),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / default_checkpoint_name(selected_repo)
    checkpoint = {
        "schema": PRE_ASR_CUEQC_SCHEMA,
        "arch": PRE_ASR_CUEQC_MODEL_ARCH,
        "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
        "runtime_adapter": PRE_ASR_CUEQC_RUNTIME_ADAPTER,
        "feature_names": list(PRE_ASR_CUEQC_SCALAR_FEATURE_NAMES),
        "semantic_split_weights_sha256": split_checkpoint_sha256,
        "inner_edge_refiner_weights_sha256": inner_checkpoint_sha256,
        "model_config": model_config,
        "feature_mean": mean.cpu().numpy().astype(np.float32).tolist(),
        "feature_std": std.cpu().numpy().astype(np.float32).tolist(),
        "decision_config": {
            "decision_mode": "argmax",
            "hard_keep_veto": False,
            "hard_drop_rule": False,
            "keep_veto": False,
            "model_only": True,
            "inference_window_size": int(sequence_window_size),
        },
        "metadata": {
            "artifact": dict(PRE_ASR_CUEQC_ARTIFACT),
            "asr_repo_id": selected_repo,
            "feature_schema": PRE_ASR_CUEQC_FEATURE_SCHEMA,
            "runtime_adapter": PRE_ASR_CUEQC_RUNTIME_ADAPTER,
            "feature_bundle": repo_display_path(features_path),
            "feature_bundle_sha256": file_sha256(features_path),
            "semantic_split_checkpoint": repo_display_path(split_checkpoint_path),
            "semantic_split_weights_sha256": split_checkpoint_sha256,
            "inner_edge_refiner_checkpoint": repo_display_path(inner_checkpoint_path),
            "inner_edge_refiner_weights_sha256": inner_checkpoint_sha256,
            "ptm_projection_digest": ptm_projection_digest,
            "ptm_pooling_schemas": ptm_pooling_schemas,
            "trained_steps": int(steps),
            "sequence_window_size": int(sequence_window_size),
            "split_mode": str(split_mode),
            "val_ratio": float(val_ratio),
            "created_at": created_at,
            "ignore_label": PRE_ASR_CUEQC_IGNORE_LABEL,
            "init_checkpoint": (
                repo_display_path(init_checkpoint)
                if init_checkpoint is not None
                else ""
            ),
            "init_checkpoint_sha256": (
                file_sha256(init_checkpoint)
                if init_checkpoint is not None
                else ""
            ),
            "force_train_audio_ids": sorted(
                str(group_rows[index].get("audio_id") or "")
                for index in force_train_groups
            ),
            "force_val_audio_ids": sorted(
                str(group_rows[index].get("audio_id") or "")
                for index in force_val_groups
            ),
            "anchor_boost_audio_ids": sorted(
                str(group_rows[index].get("audio_id") or "")
                for index in anchor_boost_groups
            ),
            "anchor_boost_candidate_ids": sorted(
                str((group_rows[group_index].get("row_ids") or [])[chunk_index])
                for group_index, chunk_index in anchor_boost_candidates
            ),
            "anchor_boost": int(anchor_boost),
            "valid_prefix_temporal": bool(model_config["valid_prefix_temporal"]),
            "ptm_encoder_mode": str(model_config["ptm_encoder_mode"]),
            "semantic_auxiliary": bool(model_config["semantic_auxiliary"]),
            "semantic_aux_loss_weight": float(semantic_aux_loss_weight),
            "late_fusion": bool(model_config["late_fusion"]),
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
    parser = argparse.ArgumentParser(description="Train Pre-ASR CueQC v13 hierarchical Mamba2 argmax checkpoint.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asr-repo-id", required=True)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--temporal-residual-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--drop-class-weight", type=float, default=1.0)
    parser.add_argument("--keep-class-weight", type=float, default=2.0)
    parser.add_argument("--unsure-class-weight", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--valid-prefix-temporal",
        action="store_true",
        help="Use padding-invariant valid-prefix masking and bidirectional reversal.",
    )
    parser.add_argument(
        "--ptm-encoder-mode",
        choices=("summary_mlp", "token_attention"),
        default="summary_mlp",
    )
    parser.add_argument("--semantic-auxiliary", action="store_true")
    parser.add_argument("--semantic-aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--late-fusion", action="store_true")
    parser.add_argument(
        "--init-checkpoint",
        help="Fine-tune from an existing compatible v13 checkpoint and preserve its normalization.",
    )
    parser.add_argument(
        "--semantic-split-checkpoint",
        help="Semantic Split checkpoint whose sha256 must be embedded in the v13 checkpoint; defaults to the active repo mapping.",
    )
    parser.add_argument(
        "--inner-edge-refiner-checkpoint",
        help="Inner Edge Refiner checkpoint whose sha256 must be embedded in the v13 checkpoint; defaults to the active repo mapping.",
    )
    parser.add_argument(
        "--force-train-audio-id",
        action="append",
        default=[],
        help="Keep all definite labels for this audio_id in training rather than validation.",
    )
    parser.add_argument(
        "--force-train-audio-id-file",
        action="append",
        default=[],
        help="Text file containing one force-train audio_id per line.",
    )
    parser.add_argument(
        "--force-val-audio-id",
        action="append",
        default=[],
        help="Keep all definite labels for this audio_id in validation.",
    )
    parser.add_argument(
        "--force-val-audio-id-file",
        action="append",
        default=[],
        help="Text file containing one force-validation audio_id per line.",
    )
    parser.add_argument(
        "--anchor-boost-audio-id",
        action="append",
        default=[],
        help="Oversample labeled anchors belonging to this audio_id.",
    )
    parser.add_argument(
        "--anchor-boost-candidate-id",
        action="append",
        default=[],
        help="Oversample only the labeled anchor with this candidate_id.",
    )
    parser.add_argument(
        "--anchor-boost-candidate-id-file",
        action="append",
        default=[],
        help="Text file containing one anchor candidate_id per line.",
    )
    parser.add_argument("--anchor-boost", type=int, default=1)
    parser.add_argument(
        "--split-mode",
        choices=("chunk_stratified", "group", "video_group", "role_holdout"),
        default="chunk_stratified",
        help="chunk_stratified samples train/test chunks within every group and label class; group keeps the old group-level split.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--sequence-window-size",
        type=int,
        default=512,
        help="Train and evaluate long planned-island sequences in fixed-size chunk windows; 0 disables windowing.",
    )
    args = parser.parse_args(argv)
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if (
        args.drop_class_weight <= 0.0
        or args.keep_class_weight <= 0.0
        or args.unsure_class_weight <= 0.0
    ):
        parser.error("class weights must be positive")
    if args.focal_gamma < 0.0:
        parser.error("--focal-gamma must be non-negative")
    if args.semantic_aux_loss_weight < 0.0:
        parser.error("--semantic-aux-loss-weight must be non-negative")
    if args.semantic_aux_loss_weight > 0.0 and not args.semantic_auxiliary:
        parser.error("--semantic-aux-loss-weight requires --semantic-auxiliary")
    if args.late_fusion and not args.semantic_auxiliary:
        parser.error("--late-fusion requires --semantic-auxiliary")
    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio must be in (0, 1)")
    if args.sequence_window_size < 0:
        parser.error("--sequence-window-size must be non-negative")
    if args.anchor_boost <= 0:
        parser.error("--anchor-boost must be positive")
    return args


def _read_id_files(paths: list[str]) -> list[str]:
    values: list[str] = []
    for raw_path in paths:
        path = project_path(raw_path)
        values.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return values


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
        keep_class_weight=float(args.keep_class_weight),
        drop_class_weight=float(args.drop_class_weight),
        unsure_class_weight=float(args.unsure_class_weight),
        sequence_window_size=int(args.sequence_window_size),
        temporal_residual_scale=float(args.temporal_residual_scale),
        split_mode=str(args.split_mode),
        val_ratio=float(args.val_ratio),
        focal_gamma=float(args.focal_gamma),
        init_checkpoint=(
            project_path(args.init_checkpoint)
            if args.init_checkpoint
            else None
        ),
        semantic_split_checkpoint=(
            project_path(args.semantic_split_checkpoint)
            if args.semantic_split_checkpoint
            else None
        ),
        inner_edge_refiner_checkpoint=(
            project_path(args.inner_edge_refiner_checkpoint)
            if args.inner_edge_refiner_checkpoint
            else None
        ),
        force_train_audio_ids=[
            *args.force_train_audio_id,
            *_read_id_files(list(args.force_train_audio_id_file)),
        ],
        anchor_boost_audio_ids=list(args.anchor_boost_audio_id),
        anchor_boost=int(args.anchor_boost),
        force_val_audio_ids=[
            *args.force_val_audio_id,
            *_read_id_files(list(args.force_val_audio_id_file)),
        ],
        anchor_boost_candidate_ids=[
            *args.anchor_boost_candidate_id,
            *_read_id_files(list(args.anchor_boost_candidate_id_file)),
        ],
        valid_prefix_temporal=bool(args.valid_prefix_temporal),
        ptm_encoder_mode=str(args.ptm_encoder_mode),
        semantic_auxiliary=bool(args.semantic_auxiliary),
        semantic_aux_loss_weight=float(args.semantic_aux_loss_weight),
        late_fusion=bool(args.late_fusion),
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
