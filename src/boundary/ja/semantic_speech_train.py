from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from boundary.ja.dataset import LabelRecord, effective_frame_weights
from boundary.ja.features import load_cached_feature
from boundary.ja.model import (
    SPEECH_ISLAND_MEMBERSHIP_LABELS,
    SPEECH_ISLAND_SCORER_CONTENT_DIM,
    SPEECH_ISLAND_SCORER_LABELS,
    SPEECH_ISLAND_SCORER_MODEL_ARCH,
    SPEECH_ISLAND_SCORER_OUTPUT_DIM,
    SPEECH_ISLAND_SCORER_OUTPUT_HEADS,
    SPEECH_ISLAND_SCORER_SCHEMA,
    build_speech_island_scorer_checkpoint,
    build_speech_island_scorer_model,
    load_speech_island_scorer_checkpoint,
)


@dataclass(frozen=True)
class SemanticSpeechTrainConfig:
    max_steps: int = 3000
    learning_rate: float = 2e-4
    seed: int = 13
    device: str = "cuda"
    hidden_size: int = 128
    num_layers: int = 2
    state_size: int = 32
    num_heads: int = 4
    n_groups: int = 2
    conv_kernel: int = 4
    chunk_size: int = 8
    bidirectional: bool = True
    discardable_weight: float = 1.0
    semantic_target_weight: float = 3.0
    unsure_weight: float = 1.5
    membership_outside_weight: float = 1.0
    membership_inside_weight: float = 2.0
    membership_unsure_weight: float = 1.5
    membership_loss_weight: float = 1.0
    focal_gamma: float = 2.0
    eval_ratio: float = 0.1
    max_train_frames: int = 1024
    max_eval_frames: int = 1024
    max_eval_windows: int = 512
    log_every: int = 100
    raw_ptm_dim: int = 2048
    projected_ptm_dim: int = 128


@dataclass(frozen=True)
class SemanticSpeechTrainMetrics:
    schema: str
    steps: int
    loss: float
    eval_loss: float
    accuracy: float
    semantic_target_precision: float
    semantic_target_recall: float
    discardable_recall: float
    unsure_recall: float
    retained_speech_recall: float
    membership_inside_recall: float
    membership_outside_recall: float
    membership_unsure_recall: float
    train_windows: int
    eval_windows: int
    checkpoint: str
    metrics_path: str


def _class_indexes(record: LabelRecord, *, total: int) -> np.ndarray:
    metadata = dict(record.boundary_metadata or {})
    values = list(metadata.get("semantic_class_frames") or ())
    if not values:
        raise ValueError(
            f"semantic_class_frames missing for {record.audio_id!r}; "
            "v9 must not train from legacy binary speech_frames"
        )
    label_to_index = {label: index for index, label in enumerate(SPEECH_ISLAND_SCORER_LABELS)}
    indexes: list[int] = []
    for value in values[:total]:
        if isinstance(value, str):
            indexes.append(label_to_index[value])
        else:
            index = int(value)
            if index < 0 or index >= len(SPEECH_ISLAND_SCORER_LABELS):
                raise ValueError(f"invalid semantic speech class index: {index}")
            indexes.append(index)
    if len(indexes) < total:
        raise ValueError(
            f"semantic_class_frames length {len(indexes)} is below feature length {total}"
        )
    return np.asarray(indexes, dtype=np.int64)


def _membership_indexes(record: LabelRecord, *, total: int) -> np.ndarray:
    metadata = dict(record.boundary_metadata or {})
    values = list(metadata.get("semantic_membership_frames") or ())
    if not values:
        raise ValueError(
            f"semantic_membership_frames missing for {record.audio_id!r}; "
            "v9 grouping must not be derived from content-class runs"
        )
    label_to_index = {
        label: index for index, label in enumerate(SPEECH_ISLAND_MEMBERSHIP_LABELS)
    }
    indexes: list[int] = []
    for value in values[:total]:
        if isinstance(value, str):
            indexes.append(label_to_index[value])
        else:
            index = int(value)
            if index < 0 or index >= len(SPEECH_ISLAND_MEMBERSHIP_LABELS):
                raise ValueError(f"invalid semantic source membership index: {index}")
            indexes.append(index)
    if len(indexes) < total:
        raise ValueError(
            f"semantic_membership_frames length {len(indexes)} is below feature length {total}"
        )
    return np.asarray(indexes, dtype=np.int64)


def semantic_training_arrays(
    row: Mapping[str, Any],
    records: list[LabelRecord],
    *,
    raw_ptm_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    record = records[int(row["label_index"])]
    ptm, mfcc = load_cached_feature(Path(str(row["feature_path"])))
    base_weights = np.asarray(effective_frame_weights(record), dtype=np.float32)
    total = min(
        len(record.speech_frames), len(base_weights), ptm.shape[0], mfcc.shape[0]
    )
    if ptm.shape[1] != raw_ptm_dim:
        raise ValueError(
            f"training feature raw_ptm_dim={ptm.shape[1]} does not match {raw_ptm_dim}"
        )
    labels = _class_indexes(record, total=total)
    membership_labels = _membership_indexes(record, total=total)
    metadata = dict(record.boundary_metadata or {})
    content_weights = base_weights[:total].copy()
    semantic_weights = metadata.get("semantic_class_weights")
    if semantic_weights is not None:
        values = np.asarray(list(semantic_weights)[:total], dtype=np.float32)
        if values.size != total:
            raise ValueError("semantic_class_weights length must match semantic_class_frames")
        content_weights *= values
    membership_weights = base_weights[:total].copy()
    configured_membership_weights = metadata.get("semantic_membership_weights")
    if configured_membership_weights is not None:
        values = np.asarray(
            list(configured_membership_weights)[:total], dtype=np.float32
        )
        if values.size != total:
            raise ValueError(
                "semantic_membership_weights length must match semantic_membership_frames"
            )
        membership_weights *= values
    return (
        np.asarray(ptm[:total], dtype=np.float32),
        np.asarray(mfcc[:total], dtype=np.float32),
        labels,
        content_weights,
        membership_labels,
        membership_weights,
    )


def _crop(*arrays, max_frames: int, rng, random: bool):
    if max_frames <= 0 or arrays[0].shape[0] <= max_frames:
        return tuple(arrays)
    limit = arrays[0].shape[0] - max_frames
    start = int(rng.integers(0, limit + 1)) if random else limit // 2
    end = start + max_frames
    return tuple(array[start:end] for array in arrays)


def _partition_rows(
    rows: list[dict[str, Any]],
    records: list[LabelRecord],
    *,
    eval_ratio: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    unspecified: list[dict[str, Any]] = []
    for row in rows:
        record = records[int(row["label_index"])]
        partition = str((record.boundary_metadata or {}).get("source_partition") or "")
        if partition in {"val", "validation", "held_out", "held-out"}:
            val.append(row)
        elif partition == "train":
            train.append(row)
        else:
            unspecified.append(row)
    if unspecified:
        order = list(rng.permutation(len(unspecified)))
        eval_count = max(1, int(round(len(unspecified) * eval_ratio)))
        val.extend(unspecified[index] for index in order[:eval_count])
        train.extend(unspecified[index] for index in order[eval_count:])
    if not train or not val:
        raise ValueError("semantic speech training requires non-empty train and validation rows")
    return train, val


def _evaluate(*, model, records, rows, config, device) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    confusion = np.zeros(
        (len(SPEECH_ISLAND_SCORER_LABELS), len(SPEECH_ISLAND_SCORER_LABELS)),
        dtype=np.int64,
    )
    membership_confusion = np.zeros(
        (
            len(SPEECH_ISLAND_MEMBERSHIP_LABELS),
            len(SPEECH_ISLAND_MEMBERSHIP_LABELS),
        ),
        dtype=np.int64,
    )
    loss_sum = 0.0
    weight_sum = 0.0
    with torch.inference_mode():
        for row in rows:
            (
                ptm,
                mfcc,
                labels,
                weights,
                membership_labels,
                membership_weights,
            ) = semantic_training_arrays(
                row, records, raw_ptm_dim=config.raw_ptm_dim
            )
            (
                ptm,
                mfcc,
                labels,
                weights,
                membership_labels,
                membership_weights,
            ) = _crop(
                ptm,
                mfcc,
                labels,
                weights,
                membership_labels,
                membership_weights,
                max_frames=config.max_eval_frames,
                rng=None,
                random=False,
            )
            all_logits = model(
                torch.from_numpy(ptm).unsqueeze(0).to(device),
                torch.from_numpy(mfcc).unsqueeze(0).to(device),
            )[0]
            logits = all_logits[:, :SPEECH_ISLAND_SCORER_CONTENT_DIM]
            membership_logits = all_logits[:, SPEECH_ISLAND_SCORER_CONTENT_DIM:]
            label_tensor = torch.from_numpy(labels).to(device)
            membership_label_tensor = torch.from_numpy(membership_labels).to(device)
            raw_loss = F.cross_entropy(logits, label_tensor, reduction="none")
            membership_raw_loss = F.cross_entropy(
                membership_logits, membership_label_tensor, reduction="none"
            )
            loss_sum += float(
                (
                    raw_loss * torch.from_numpy(weights).to(device)
                    + config.membership_loss_weight
                    * membership_raw_loss
                    * torch.from_numpy(membership_weights).to(device)
                )
                .sum()
                .cpu()
            )
            weight_sum += float(
                weights.sum()
                + config.membership_loss_weight * membership_weights.sum()
            )
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            membership_predictions = (
                torch.argmax(membership_logits, dim=-1).cpu().numpy()
            )
            active = weights > 0.0
            for truth, predicted in zip(labels[active], predictions[active]):
                confusion[int(truth), int(predicted)] += 1
            membership_active = membership_weights > 0.0
            for truth, predicted in zip(
                membership_labels[membership_active],
                membership_predictions[membership_active],
            ):
                membership_confusion[int(truth), int(predicted)] += 1
    recall = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    precision = np.diag(confusion) / np.maximum(confusion.sum(axis=0), 1)
    target = SPEECH_ISLAND_SCORER_LABELS.index("semantic_target")
    discard = SPEECH_ISLAND_SCORER_LABELS.index("discardable")
    unsure = SPEECH_ISLAND_SCORER_LABELS.index("unsure")
    membership_recall = np.diag(membership_confusion) / np.maximum(
        membership_confusion.sum(axis=1), 1
    )
    membership_outside = SPEECH_ISLAND_MEMBERSHIP_LABELS.index("outside")
    membership_inside = SPEECH_ISLAND_MEMBERSHIP_LABELS.index("inside")
    membership_unsure = SPEECH_ISLAND_MEMBERSHIP_LABELS.index("unsure")
    retained_truth = confusion[target].sum() + confusion[unsure].sum()
    retained_hits = (
        confusion[target, target]
        + confusion[target, unsure]
        + confusion[unsure, target]
        + confusion[unsure, unsure]
    )
    return {
        "loss": loss_sum / max(weight_sum, 1e-6),
        "accuracy": float(np.diag(confusion).sum() / max(confusion.sum(), 1)),
        "semantic_target_precision": float(precision[target]),
        "semantic_target_recall": float(recall[target]),
        "discardable_recall": float(recall[discard]),
        "unsure_recall": float(recall[unsure]),
        "retained_speech_recall": float(retained_hits / max(retained_truth, 1)),
        "membership_inside_recall": float(membership_recall[membership_inside]),
        "membership_outside_recall": float(membership_recall[membership_outside]),
        "membership_unsure_recall": float(membership_recall[membership_unsure]),
    }


def _initialize_from_legacy(
    model,
    legacy_model,
    *,
    legacy_mean: np.ndarray,
    legacy_std: np.ndarray,
    projected_ptm_dim: int,
) -> None:
    import torch

    model.encoder.proj.load_state_dict(legacy_model.proj.state_dict())
    model.encoder.backbone.load_state_dict(legacy_model.backbone.state_dict())
    model.encoder.norm.load_state_dict(legacy_model.norm.state_dict())
    target_index = SPEECH_ISLAND_SCORER_LABELS.index("semantic_target")
    discardable_index = SPEECH_ISLAND_SCORER_LABELS.index("discardable")
    unsure_index = SPEECH_ISLAND_SCORER_LABELS.index("unsure")
    membership_offset = SPEECH_ISLAND_SCORER_CONTENT_DIM
    inside_index = membership_offset + SPEECH_ISLAND_MEMBERSHIP_LABELS.index("inside")
    outside_index = membership_offset + SPEECH_ISLAND_MEMBERSHIP_LABELS.index("outside")
    membership_unsure_index = (
        membership_offset + SPEECH_ISLAND_MEMBERSHIP_LABELS.index("unsure")
    )
    with torch.no_grad():
        model.ptm_projector.weight.zero_()
        diagonal = torch.arange(projected_ptm_dim, device=model.ptm_projector.weight.device)
        model.ptm_projector.weight[diagonal, diagonal] = torch.from_numpy(
            1.0 / legacy_std[:projected_ptm_dim]
        ).to(model.ptm_projector.weight.device)
        model.ptm_projector.bias.copy_(
            torch.from_numpy(
                -legacy_mean[:projected_ptm_dim]
                / legacy_std[:projected_ptm_dim]
            ).to(model.ptm_projector.bias.device)
        )
        model.encoder.head.weight.zero_()
        model.encoder.head.bias.zero_()
        legacy_weight = legacy_model.head.weight[0]
        legacy_bias = legacy_model.head.bias[0]
        model.encoder.head.weight[target_index].copy_(legacy_weight * 0.5)
        model.encoder.head.bias[target_index].copy_(legacy_bias * 0.5)
        model.encoder.head.weight[discardable_index].copy_(legacy_weight * -0.5)
        model.encoder.head.bias[discardable_index].copy_(legacy_bias * -0.5)
        model.encoder.head.weight[unsure_index].zero_()
        model.encoder.head.bias[unsure_index].zero_()
        model.encoder.head.weight[inside_index].copy_(legacy_weight * 0.5)
        model.encoder.head.bias[inside_index].copy_(legacy_bias * 0.5)
        model.encoder.head.weight[outside_index].copy_(legacy_weight * -0.5)
        model.encoder.head.bias[outside_index].copy_(legacy_bias * -0.5)
        model.encoder.head.weight[membership_unsure_index].zero_()
        model.encoder.head.bias[membership_unsure_index].zero_()


def train_semantic_speech_scorer(
    *,
    records: list[LabelRecord],
    feature_manifest_rows: Iterable[Mapping[str, Any]],
    output_dir: Path,
    config: SemanticSpeechTrainConfig,
    labels_path: str,
    feature_manifest_path: str,
    checkpoint_name: str,
    warm_start_checkpoint: str,
) -> SemanticSpeechTrainMetrics:
    import torch
    import torch.nn.functional as F

    rows = [dict(row) for row in feature_manifest_rows]
    if not rows:
        raise ValueError("at least one feature manifest row is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    train_rows, eval_rows = _partition_rows(
        rows, records, eval_ratio=config.eval_ratio, rng=rng
    )
    raw_ptm_dim = int(config.raw_ptm_dim)
    projected_ptm_dim = int(config.projected_ptm_dim)
    mfcc_dim = int(rows[0]["mfcc_dim"])
    legacy = load_speech_island_scorer_checkpoint(
        warm_start_checkpoint,
        device="cpu",
    )
    if legacy.ptm_dim != projected_ptm_dim or legacy.mfcc_dim != mfcc_dim:
        raise ValueError("semantic scorer warm-start dimensions do not match")
    legacy_mean = np.asarray(legacy.normalization["feature_mean"], dtype=np.float32)
    legacy_std = np.maximum(
        np.asarray(legacy.normalization["feature_std"], dtype=np.float32),
        1e-6,
    )
    model_config = {
        "raw_ptm_dim": raw_ptm_dim,
        "projected_ptm_dim": projected_ptm_dim,
        "mfcc_dim": mfcc_dim,
        "input_dim": raw_ptm_dim + mfcc_dim,
        "projection_type": "task_aware_linear",
        "mfcc_mean": legacy_mean[projected_ptm_dim:].tolist(),
        "mfcc_std": legacy_std[projected_ptm_dim:].tolist(),
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "state_size": config.state_size,
        "num_heads": config.num_heads,
        "n_groups": config.n_groups,
        "conv_kernel": config.conv_kernel,
        "chunk_size": config.chunk_size,
        "bidirectional": config.bidirectional,
        "model_arch": SPEECH_ISLAND_SCORER_MODEL_ARCH,
        "output_dim": SPEECH_ISLAND_SCORER_OUTPUT_DIM,
    }
    device = torch.device(config.device)
    model = build_speech_island_scorer_model(
        schema=SPEECH_ISLAND_SCORER_SCHEMA,
        model_config=model_config,
    ).to(device)
    _initialize_from_legacy(
        model,
        legacy.model,
        legacy_mean=legacy_mean,
        legacy_std=legacy_std,
        projected_ptm_dim=projected_ptm_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    class_weights = torch.tensor(
        [
            config.discardable_weight,
            config.semantic_target_weight,
            config.unsure_weight,
        ],
        dtype=torch.float32,
        device=device,
    )
    membership_class_weights = torch.tensor(
        [
            config.membership_outside_weight,
            config.membership_inside_weight,
            config.membership_unsure_weight,
        ],
        dtype=torch.float32,
        device=device,
    )
    losses: list[float] = []
    started = time.monotonic()
    for step in range(config.max_steps):
        row = train_rows[step % len(train_rows)]
        (
            ptm,
            mfcc,
            labels,
            weights,
            membership_labels,
            membership_weights,
        ) = semantic_training_arrays(
            row, records, raw_ptm_dim=raw_ptm_dim
        )
        (
            ptm,
            mfcc,
            labels,
            weights,
            membership_labels,
            membership_weights,
        ) = _crop(
            ptm,
            mfcc,
            labels,
            weights,
            membership_labels,
            membership_weights,
            max_frames=config.max_train_frames,
            rng=rng,
            random=True,
        )
        all_logits = model(
            torch.from_numpy(ptm).unsqueeze(0).to(device),
            torch.from_numpy(mfcc).unsqueeze(0).to(device),
        )[0]
        logits = all_logits[:, :SPEECH_ISLAND_SCORER_CONTENT_DIM]
        membership_logits = all_logits[:, SPEECH_ISLAND_SCORER_CONTENT_DIM:]
        label_tensor = torch.from_numpy(labels).to(device)
        membership_label_tensor = torch.from_numpy(membership_labels).to(device)
        frame_weights = torch.from_numpy(weights).to(device)
        membership_frame_weights = torch.from_numpy(membership_weights).to(device)
        raw_loss = F.cross_entropy(logits, label_tensor, reduction="none")
        probabilities = torch.softmax(logits, dim=-1)
        pt = probabilities.gather(1, label_tensor.unsqueeze(1)).squeeze(1)
        effective = frame_weights * class_weights[label_tensor]
        content_loss = (
            raw_loss * torch.pow(1.0 - pt, config.focal_gamma) * effective
        ).sum() / effective.sum().clamp_min(1e-6)
        membership_raw_loss = F.cross_entropy(
            membership_logits, membership_label_tensor, reduction="none"
        )
        membership_probabilities = torch.softmax(membership_logits, dim=-1)
        membership_pt = membership_probabilities.gather(
            1, membership_label_tensor.unsqueeze(1)
        ).squeeze(1)
        membership_effective = (
            membership_frame_weights
            * membership_class_weights[membership_label_tensor]
        )
        membership_loss = (
            membership_raw_loss
            * torch.pow(1.0 - membership_pt, config.focal_gamma)
            * membership_effective
        ).sum() / membership_effective.sum().clamp_min(1e-6)
        loss = content_loss + config.membership_loss_weight * membership_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if config.log_every and (step + 1) % config.log_every == 0:
            print(
                f"semantic_speech_train={step + 1}/{config.max_steps} "
                f"loss={losses[-1]:.6f} avg_loss={np.mean(losses):.6f} "
                f"elapsed_s={time.monotonic() - started:.1f}",
                flush=True,
            )
    evaluation = _evaluate(
        model=model,
        records=records,
        rows=eval_rows[: config.max_eval_windows],
        config=config,
        device=device,
    )
    checkpoint_path = output_dir / checkpoint_name
    torch.save(
        build_speech_island_scorer_checkpoint(
            model=model,
            model_config=model_config,
            normalization={"schema": "task_aware_full_ptm_projection_v1"},
            metadata={
                "operating_point": "semantic-speech-high-recall-argmax-v9",
                "ptm_repo_id": str(rows[0].get("ptm") or ""),
                "labels": list(SPEECH_ISLAND_SCORER_OUTPUT_HEADS),
                "content_labels": list(SPEECH_ISLAND_SCORER_LABELS),
                "membership_labels": list(SPEECH_ISLAND_MEMBERSHIP_LABELS),
                "labels_path": labels_path,
                "feature_manifest": feature_manifest_path,
                "trained_steps": config.max_steps,
                "config": asdict(config),
                "warm_start_checkpoint": warm_start_checkpoint,
                "warm_start_sha256": legacy.sha256,
            },
        ),
        checkpoint_path,
    )
    metrics_path = output_dir / "train_metrics.json"
    metrics = SemanticSpeechTrainMetrics(
        schema=SPEECH_ISLAND_SCORER_SCHEMA,
        steps=config.max_steps,
        loss=float(np.mean(losses)),
        eval_loss=float(evaluation["loss"]),
        accuracy=float(evaluation["accuracy"]),
        semantic_target_precision=float(evaluation["semantic_target_precision"]),
        semantic_target_recall=float(evaluation["semantic_target_recall"]),
        discardable_recall=float(evaluation["discardable_recall"]),
        unsure_recall=float(evaluation["unsure_recall"]),
        retained_speech_recall=float(evaluation["retained_speech_recall"]),
        membership_inside_recall=float(evaluation["membership_inside_recall"]),
        membership_outside_recall=float(evaluation["membership_outside_recall"]),
        membership_unsure_recall=float(evaluation["membership_unsure_recall"]),
        train_windows=len(train_rows),
        eval_windows=min(len(eval_rows), config.max_eval_windows),
        checkpoint=str(checkpoint_path),
        metrics_path=str(metrics_path),
    )
    metrics_path.write_text(
        json.dumps(asdict(metrics), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics
