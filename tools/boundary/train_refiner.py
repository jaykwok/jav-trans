#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from asr.backends.qwen import qwen_asr_repo_tag
from boundary.backbones import TRANSFORMERS_MAMBA2_BACKBONE, BoundarySequenceClassifier
from boundary.refiner import (
    BOUNDARY_REFINER_OUTPUT_DIM,
    DEFAULT_BOUNDARY_DELTA_MAX_S,
    build_learned_refiner_checkpoint,
    load_edge_sequence_refiner_v6_checkpoint,
)
from boundary.sequence_features import (
    FRAME_SEQUENCE_FEATURE_SCHEMA,
    feature_extraction_hash,
    validate_sequence_features,
)

@dataclass(frozen=True)
class TrainRefinerConfig:
    max_steps: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 13
    device: str = "auto"
    val_ratio: float = 0.10
    hidden_size: int = 64
    num_layers: int = 1
    state_size: int = 16
    num_heads: int = 4
    n_groups: int = 2
    chunk_size: int = 8
    bidirectional: bool = True
    log_interval_steps: int = 20
    boundary_delta_max_s: float = DEFAULT_BOUNDARY_DELTA_MAX_S
    start_delta_loss_weight: float = 1.00
    end_delta_loss_weight: float = 0.60
    init_checkpoint: str = ""
    preserve_init_normalization: bool = False
    freeze_backbone: bool = False
    tensor_cache_path: str = ""
    loader_log_interval_rows: int = 0
    checkpoint_name: str = ""


@dataclass(frozen=True)
class LoadedRefinerDataset:
    features: torch.Tensor
    boundary_delta_targets: torch.Tensor
    boundary_delta_weights: torch.Tensor
    mask: torch.Tensor
    feature_names: tuple[str, ...]
    feature_metadata: dict[str, Any]
    train_schema: str
    first_row: dict[str, Any]


@dataclass(frozen=True)
class DatasetScan:
    row_count: int
    max_len: int
    feature_names: tuple[str, ...]
    feature_metadata: dict[str, Any]
    train_schema: str
    first_row: dict[str, Any]


def train_refiner(
    *,
    dataset_paths: Sequence[Path],
    output_dir: Path,
    config: TrainRefinerConfig = TrainRefinerConfig(),
) -> dict[str, Any]:
    _validate_config(config)
    loaded = _load_or_build_dataset_tensors(
        dataset_paths,
        tensor_cache_path=Path(config.tensor_cache_path) if config.tensor_cache_path else None,
        log_interval_rows=config.loader_log_interval_rows,
    )
    features = loaded.features
    boundary_delta_targets = loaded.boundary_delta_targets
    boundary_delta_weights = loaded.boundary_delta_weights
    mask = loaded.mask
    feature_names = loaded.feature_names
    feature_metadata = loaded.feature_metadata

    train_idx, val_idx = _split_indexes(features.shape[0], val_ratio=config.val_ratio, seed=config.seed)
    train_mask = mask[train_idx]
    train_row_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_row_mask[train_idx] = True
    active_train_mask = train_row_mask[:, None] & mask
    active_train_features = features[active_train_mask]
    feature_mean = active_train_features.mean(dim=0)
    feature_std = active_train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    del active_train_features
    normalization_source = "train_dataset"
    if config.preserve_init_normalization:
        if not config.init_checkpoint:
            raise ValueError("--preserve-init-normalization requires --init-checkpoint")
        init_mean, init_std = _load_initial_checkpoint_normalization(
            Path(config.init_checkpoint),
            expected_feature_names=feature_names,
        )
        feature_mean = init_mean
        feature_std = init_std
        normalization_source = "init_checkpoint"
    features.sub_(feature_mean).div_(feature_std)

    device = _resolve_device(config.device)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    model = BoundarySequenceClassifier(
        input_dim=len(feature_names),
        backbone=TRANSFORMERS_MAMBA2_BACKBONE,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_dim=BOUNDARY_REFINER_OUTPUT_DIM,
        state_size=config.state_size,
        num_heads=config.num_heads,
        n_groups=config.n_groups,
        chunk_size=config.chunk_size,
        bidirectional=config.bidirectional,
    ).to(device)
    init_metadata: dict[str, Any] = {}
    if config.init_checkpoint:
        init_metadata = _load_initial_checkpoint(
            model,
            Path(config.init_checkpoint),
            expected_input_dim=len(feature_names),
        )
    if config.freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)

    train_dataset = Subset(
        TensorDataset(features, boundary_delta_targets, boundary_delta_weights, mask),
        train_idx,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=True,
        drop_last=False,
    )
    delta_criterion = nn.SmoothL1Loss(reduction="none")
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("no trainable parameters remain after applying freeze options")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    step = 0
    last_loss = math.nan
    while step < config.max_steps:
        for (
            batch_features,
            batch_boundary_delta_targets,
            batch_boundary_delta_weights,
            batch_mask,
        ) in train_loader:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            batch_mask = batch_mask.to(device)
            logits = model(batch_features.to(device), attention_mask=batch_mask.long())
            delta_pred = torch.tanh(logits[..., :2]) * float(config.boundary_delta_max_s)
            delta_target = batch_boundary_delta_targets.to(device)
            delta_weight = batch_boundary_delta_weights.to(device)
            start_delta_loss = _weighted_delta_loss(
                delta_pred[..., 0],
                delta_target[..., 0],
                weights=delta_weight[..., 0],
                mask=batch_mask,
                criterion=delta_criterion,
            )
            end_delta_loss = _weighted_delta_loss(
                delta_pred[..., 1],
                delta_target[..., 1],
                weights=delta_weight[..., 1],
                mask=batch_mask,
                criterion=delta_criterion,
            )
            loss = (
                float(config.start_delta_loss_weight) * start_delta_loss
                + float(config.end_delta_loss_weight) * end_delta_loss
            )
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            if config.log_interval_steps > 0 and step % config.log_interval_steps == 0:
                print(f"step={step} loss={last_loss:.6f}", flush=True)
            if step >= config.max_steps:
                break

    metrics = {
        "schema": "boundary_refiner_train_metrics_v1",
        "dataset_paths": [str(path) for path in dataset_paths],
        "output_dir": str(output_dir),
        "config": asdict(config),
        "feature_names": list(feature_names),
        "feature_metadata": feature_metadata,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "train_items": int(mask[train_idx].sum().item()),
        "val_items": int(mask[val_idx].sum().item()) if val_idx else 0,
        "last_train_loss": last_loss,
        "train": _evaluate(
            model,
            features,
            boundary_delta_targets,
            boundary_delta_weights,
            mask,
            indexes=train_idx,
            device=device,
            boundary_delta_max_s=config.boundary_delta_max_s,
            batch_size=config.batch_size,
        ),
        "val": _evaluate(
            model,
            features,
            boundary_delta_targets,
            boundary_delta_weights,
            mask,
            indexes=val_idx,
            device=device,
            boundary_delta_max_s=config.boundary_delta_max_s,
            batch_size=config.batch_size,
        )
        if val_idx
        else {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    checkpoint = build_learned_refiner_checkpoint(
        model=model,
        feature_names=feature_names,
        feature_mean=tuple(float(value) for value in feature_mean.tolist()),
        feature_std=tuple(float(value) for value in feature_std.tolist()),
        metadata={
            "trainer": "tools/boundary/train_refiner.py",
            "dataset_paths": [str(path) for path in dataset_paths],
            "train_schema": loaded.train_schema,
            "runtime_adapter": "edge_sequence_v1",
            "edge_policy": {
                "boundary_delta_max_s": config.boundary_delta_max_s,
                "start_delta_loss_weight": config.start_delta_loss_weight,
                "end_delta_loss_weight": config.end_delta_loss_weight,
                "delta_loss": "smooth_l1",
                "source": "scorer_v4_island_edges",
            },
            "boundary_delta_max_s": config.boundary_delta_max_s,
            "init_checkpoint": init_metadata,
            "normalization_source": normalization_source,
            "preserve_init_normalization": config.preserve_init_normalization,
            "freeze_backbone": config.freeze_backbone,
            **feature_metadata,
        },
    )
    checkpoint_name = config.checkpoint_name.strip()
    if not checkpoint_name:
        ptm_repo_id = str(feature_metadata.get("ptm_repo_id") or "").strip()
        checkpoint_name = (
            f"boundary_edge_refiner_v6.{qwen_asr_repo_tag(ptm_repo_id)}.pt"
            if ptm_repo_id
            else "boundary_edge_refiner_v6.pt"
        )
    if Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("checkpoint_name must be a file name, not a path")
    checkpoint_path = output_dir / checkpoint_name
    torch.save(checkpoint, checkpoint_path)
    metrics["checkpoint"] = str(checkpoint_path)

    refiner = load_edge_sequence_refiner_v6_checkpoint(
        checkpoint_path,
        backbone_override=TRANSFORMERS_MAMBA2_BACKBONE,
    )
    smoke_sequence = _row_sequence(loaded.first_row, feature_names)
    smoke_decisions = refiner.decide_sequence(smoke_sequence)
    metrics["loader_smoke"] = {
        "signature": refiner.signature(),
        "decision_count": len(smoke_decisions),
        "first_decision": (
            {
                "source": smoke_decisions[0].source,
                "start_refine_delta_s": smoke_decisions[0].start_refine_delta_s,
                "end_refine_delta_s": smoke_decisions[0].end_refine_delta_s,
            }
            if smoke_decisions
            else None
        ),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"checkpoint={checkpoint_path}")
    print(f"metrics={metrics_path}")
    print(f"items={metrics['train_items'] + metrics['val_items']}")
    return metrics


def _load_dataset_tensors(paths: Sequence[Path]) -> LoadedRefinerDataset:
    return _load_dataset_tensors_from_jsonl(paths, log_interval_rows=0)


def _load_or_build_dataset_tensors(
    paths: Sequence[Path],
    *,
    tensor_cache_path: Path | None,
    log_interval_rows: int,
) -> LoadedRefinerDataset:
    if tensor_cache_path is not None and tensor_cache_path.exists():
        try:
            print(f"loading_tensor_cache={tensor_cache_path}", flush=True)
            return _load_tensor_cache(tensor_cache_path, dataset_paths=paths)
        except ValueError as exc:
            print(f"ignoring_tensor_cache={tensor_cache_path} reason={exc}", flush=True)
    loaded = _load_dataset_tensors_from_jsonl(paths, log_interval_rows=log_interval_rows)
    if tensor_cache_path is not None:
        print(f"writing_tensor_cache={tensor_cache_path}", flush=True)
        _write_tensor_cache(tensor_cache_path, loaded, dataset_paths=paths)
    return loaded


def _load_dataset_tensors_from_jsonl(
    paths: Sequence[Path],
    *,
    log_interval_rows: int,
) -> LoadedRefinerDataset:
    scan = _scan_dataset(paths, log_interval_rows=log_interval_rows)
    features = torch.zeros((scan.row_count, scan.max_len, len(scan.feature_names)), dtype=torch.float32)
    boundary_delta_targets = torch.zeros((scan.row_count, scan.max_len, 2), dtype=torch.float32)
    boundary_delta_weights = torch.zeros((scan.row_count, scan.max_len, 2), dtype=torch.float32)
    mask = torch.zeros((scan.row_count, scan.max_len), dtype=torch.bool)

    row_index = 0
    for row in _iter_dataset_rows(paths):
        sequence = _row_sequence_array(row, scan.feature_names)
        boundary_delta_sequence = _row_boundary_delta_target_sequence(
            row,
            expected_len=int(sequence.shape[0]),
        )
        boundary_delta_weight_sequence = _row_boundary_delta_weight_sequence(
            row,
            expected_len=int(sequence.shape[0]),
        )
        length = int(sequence.shape[0])
        features[row_index, :length].copy_(torch.from_numpy(sequence))
        boundary_delta_targets[row_index, :length].copy_(
            torch.tensor(boundary_delta_sequence, dtype=torch.float32)
        )
        boundary_delta_weights[row_index, :length].copy_(
            torch.tensor(boundary_delta_weight_sequence, dtype=torch.float32)
        )
        mask[row_index, :length] = True
        row_index += 1
        if log_interval_rows > 0 and row_index % log_interval_rows == 0:
            print(f"loaded_dataset_rows={row_index}/{scan.row_count}", flush=True)
    if row_index != scan.row_count:
        raise ValueError(f"dataset changed while loading: expected {scan.row_count}, got {row_index}")
    return LoadedRefinerDataset(
        features=features,
        boundary_delta_targets=boundary_delta_targets,
        boundary_delta_weights=boundary_delta_weights,
        mask=mask,
        feature_names=scan.feature_names,
        feature_metadata=scan.feature_metadata,
        train_schema=scan.train_schema,
        first_row=scan.first_row,
    )


def _load_tensor_cache(path: Path, *, dataset_paths: Sequence[Path] | None = None) -> LoadedRefinerDataset:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"tensor cache must be a mapping: {path}")
    if str(payload.get("schema") or "") != "boundary_refiner_tensor_cache_v6":
        raise ValueError(f"unsupported tensor cache schema: {payload.get('schema')!r}")
    if dataset_paths is not None:
        cached_sources = payload.get("dataset_sources")
        if cached_sources != _dataset_source_fingerprints(dataset_paths):
            raise ValueError("dataset source fingerprint mismatch")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("tensor cache missing tensors")
    features = _tensor_from_cache(tensors, "features", ndim=3, dtype=torch.float32)
    boundary_delta_targets = _tensor_from_cache(
        tensors,
        "boundary_delta_targets",
        ndim=3,
        dtype=torch.float32,
    )
    boundary_delta_weights = _tensor_from_cache(
        tensors,
        "boundary_delta_weights",
        ndim=3,
        dtype=torch.float32,
    )
    mask = _tensor_from_cache(tensors, "mask", ndim=2, dtype=torch.bool)
    row_count, max_len, feature_dim = features.shape
    if boundary_delta_targets.shape != (row_count, max_len, 2):
        raise ValueError("tensor cache boundary_delta_targets shape mismatch")
    if boundary_delta_weights.shape != (row_count, max_len, 2):
        raise ValueError("tensor cache boundary_delta_weights shape mismatch")
    if mask.shape != (row_count, max_len):
        raise ValueError("tensor cache mask shape mismatch")
    feature_names = tuple(str(name) for name in payload.get("feature_names") or ())
    if len(feature_names) != feature_dim:
        raise ValueError("tensor cache feature_names length mismatch")
    feature_metadata = dict(payload.get("feature_metadata") or {})
    first_row = dict(payload.get("first_row") or {})
    if not first_row:
        raise ValueError("tensor cache missing first_row")
    return LoadedRefinerDataset(
        features=features,
        boundary_delta_targets=boundary_delta_targets,
        boundary_delta_weights=boundary_delta_weights,
        mask=mask,
        feature_names=feature_names,
        feature_metadata=feature_metadata,
        train_schema=str(payload.get("train_schema") or ""),
        first_row=first_row,
    )


def _tensor_from_cache(
    tensors: Mapping[str, Any],
    name: str,
    *,
    ndim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = tensors.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"tensor cache missing tensor: {name}")
    if value.ndim != ndim:
        raise ValueError(f"tensor cache {name} ndim mismatch: {value.ndim} != {ndim}")
    return value.to(device="cpu", dtype=dtype, copy=False).contiguous()


def _write_tensor_cache(
    path: Path,
    loaded: LoadedRefinerDataset,
    *,
    dataset_paths: Sequence[Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "boundary_refiner_tensor_cache_v6",
            "dataset_sources": _dataset_source_fingerprints(dataset_paths),
            "feature_names": list(loaded.feature_names),
            "feature_metadata": loaded.feature_metadata,
            "train_schema": loaded.train_schema,
            "first_row": loaded.first_row,
            "tensors": {
                "features": loaded.features.contiguous(),
                "boundary_delta_targets": loaded.boundary_delta_targets.contiguous(),
                "boundary_delta_weights": loaded.boundary_delta_weights.contiguous(),
                "mask": loaded.mask.contiguous(),
            },
        },
        path,
    )


def _dataset_source_fingerprints(paths: Sequence[Path]) -> list[dict[str, Any]]:
    fingerprints: list[dict[str, Any]] = []
    for path in paths:
        stat = path.stat()
        fingerprints.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return fingerprints


def _load_initial_checkpoint(
    model: BoundarySequenceClassifier,
    checkpoint_path: Path,
    *,
    expected_input_dim: int,
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"init checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("init checkpoint must be a mapping")
    schema = str(payload.get("schema") or "")
    if schema != "boundary_edge_refiner_v6":
        raise ValueError(f"init checkpoint schema must be boundary_edge_refiner_v6, got {schema!r}")
    model_config = dict(payload.get("model_config") or {})
    if int(model_config.get("input_dim", -1)) != int(expected_input_dim):
        raise ValueError(
            "init checkpoint input_dim mismatch: "
            f"{model_config.get('input_dim')} != {expected_input_dim}"
        )
    if int(model_config.get("output_dim", -1)) != BOUNDARY_REFINER_OUTPUT_DIM:
        raise ValueError(
            "init checkpoint output_dim mismatch: "
            f"{model_config.get('output_dim')} != {BOUNDARY_REFINER_OUTPUT_DIM}"
        )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("init checkpoint missing state_dict")
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise ValueError(
            "init checkpoint state_dict mismatch: "
            f"missing={list(missing)} unexpected={list(unexpected)}"
        )
    metadata = dict(payload.get("metadata") or {})
    return {
        "path": str(checkpoint_path),
        "schema": schema,
        "model_config": model_config,
        "metadata": metadata,
    }


def _load_initial_checkpoint_normalization(
    checkpoint_path: Path,
    *,
    expected_feature_names: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"init checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("init checkpoint must be a mapping")
    checkpoint_feature_names = tuple(str(name) for name in payload.get("feature_names") or ())
    if checkpoint_feature_names != tuple(expected_feature_names):
        raise ValueError("init checkpoint feature_names do not match training dataset")
    mean = _normalization_tensor(payload.get("feature_mean"), expected_len=len(checkpoint_feature_names), name="feature_mean")
    std = _normalization_tensor(payload.get("feature_std"), expected_len=len(checkpoint_feature_names), name="feature_std")
    return mean, std.clamp_min(1e-6)


def _normalization_tensor(raw: object, *, expected_len: int, name: str) -> torch.Tensor:
    if not isinstance(raw, (list, tuple)) or len(raw) != expected_len:
        raise ValueError(f"init checkpoint {name} length mismatch")
    values = torch.tensor([float(value) for value in raw], dtype=torch.float32)
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError(f"init checkpoint {name} must be finite")
    return values


def _scan_dataset(paths: Sequence[Path], *, log_interval_rows: int = 0) -> DatasetScan:
    row_count = 0
    max_len = 0
    feature_names: tuple[str, ...] | None = None
    first_row: dict[str, Any] | None = None
    train_schemas: set[str] = set()
    feature_schema_values: set[str] = set()
    feature_hash_values: set[str] = set()
    ptm_repo_values: set[str] = set()
    dataset_source_values: set[str] = set()
    scorer_checkpoint_values: set[str] = set()
    expected_signature: dict[str, Any] | None = None

    for row in _iter_dataset_rows(paths):
        _validate_v6_edge_row(row, row_index=row_count)
        if first_row is None:
            first_row = dict(row)
        row_names = _row_feature_names(row, feature_names)
        if feature_names is None:
            feature_names = row_names
        elif row_names != feature_names:
            raise ValueError(f"feature_names mismatch at row {row_count}: {row_names} != {feature_names}")
        sequence_len = _row_sequence_length(row, feature_names)
        _row_boundary_delta_target_sequence(row, expected_len=sequence_len)
        _row_boundary_delta_weight_sequence(row, expected_len=sequence_len)
        max_len = max(max_len, sequence_len)
        train_schemas.add(str(row.get("schema") or ""))
        if row.get("feature_schema"):
            feature_schema_values.add(str(row["feature_schema"]))
        if row.get("feature_schema_hash"):
            feature_hash_values.add(str(row["feature_schema_hash"]))
        metadata = row.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("ptm_repo_id"):
            ptm_repo_values.add(str(metadata["ptm_repo_id"]))
        dataset_source = (
            metadata.get("dataset_source") if isinstance(metadata, Mapping) else row.get("dataset_source")
        )
        if dataset_source:
            dataset_source_values.add(str(dataset_source))
        scorer_checkpoint = metadata.get("scorer_checkpoint") if isinstance(metadata, Mapping) else None
        if isinstance(scorer_checkpoint, Mapping):
            scorer_checkpoint_values.add(
                json.dumps(dict(scorer_checkpoint), ensure_ascii=False, sort_keys=True)
            )
        signature = row.get("feature_signature")
        if isinstance(signature, Mapping):
            signature_dict = dict(signature)
            if expected_signature is None:
                expected_signature = signature_dict
            elif signature_dict != expected_signature:
                raise ValueError(f"feature_signature mismatch at row {row_count}")
        row_count += 1
        if log_interval_rows > 0 and row_count % log_interval_rows == 0:
            print(f"scanned_dataset_rows={row_count}", flush=True)

    if row_count == 0 or feature_names is None or first_row is None:
        raise ValueError("boundary refiner dataset is empty")
    if max_len <= 0:
        raise ValueError("boundary refiner dataset has no sequence items")
    return DatasetScan(
        row_count=row_count,
        max_len=max_len,
        feature_names=feature_names,
        feature_metadata=_feature_metadata_from_scan(
            feature_names=feature_names,
            schema_values=feature_schema_values,
            hash_values=feature_hash_values,
            expected_signature=expected_signature,
            ptm_repo_values=ptm_repo_values,
            dataset_source_values=dataset_source_values,
            scorer_checkpoint_values=scorer_checkpoint_values,
        ),
        train_schema=_train_schema_from_values(train_schemas),
        first_row=first_row,
    )


def _iter_dataset_rows(paths: Sequence[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _row_feature_names(row: Mapping[str, Any], fallback: tuple[str, ...] | None = None) -> tuple[str, ...]:
    raw = row.get("feature_names")
    if raw:
        names = tuple(str(name) for name in raw)
    elif fallback is not None:
        names = fallback
    else:
        raise ValueError("boundary refiner v6 rows require feature_names")
    if not names:
        raise ValueError("dataset feature_names must not be empty")
    return names


def _train_schema(rows: Sequence[Mapping[str, Any]]) -> str:
    schemas = {str(row.get("schema") or "") for row in rows}
    return _train_schema_from_values(schemas)


def _train_schema_from_values(schemas: set[str]) -> str:
    if schemas == {"boundary_edge_refiner_dataset_v6"}:
        return "boundary_edge_refiner_dataset_v6"
    raise ValueError(f"unsupported boundary refiner dataset schema values: {sorted(schemas)}")


def _validate_v6_edge_row(row: Mapping[str, Any], *, row_index: int) -> None:
    schema = str(row.get("schema") or "")
    if schema != "boundary_edge_refiner_dataset_v6":
        raise ValueError(
            "boundary refiner training only accepts "
            f"'boundary_edge_refiner_dataset_v6', got {schema!r} at row {row_index}"
        )
    stale_fields = (
        "sequence_labels",
        "sequence_context_targets",
        "merge_positive",
        "split_negative",
        "boundary_merge_prob",
        "boundary_split_prob",
        "boundary_decision_merge",
        "merge_label",
        "merge_weight",
        "split_label",
        "split_weight",
    )
    present = [field for field in stale_fields if field in row]
    if present:
        raise ValueError(
            "boundary refiner v6 rows must not contain old merge/split/context fields "
            f"at row {row_index}: {present}"
        )
    metadata = row.get("metadata")
    metadata_source = metadata.get("dataset_source") if isinstance(metadata, Mapping) else ""
    dataset_source = str(row.get("dataset_source") or metadata_source or "")
    if dataset_source != "scorer_v4_predicted_island_edges":
        raise ValueError(
            "boundary refiner v6 rows must use "
            f"'scorer_v4_predicted_island_edges', got {dataset_source!r} at row {row_index}"
        )
    scorer_checkpoint = metadata.get("scorer_checkpoint") if isinstance(metadata, Mapping) else None
    if not isinstance(scorer_checkpoint, Mapping):
        raise ValueError(f"boundary refiner v6 rows require metadata.scorer_checkpoint at row {row_index}")


def _feature_metadata_from_scan(
    *,
    feature_names: tuple[str, ...],
    schema_values: set[str],
    hash_values: set[str],
    expected_signature: Mapping[str, Any] | None,
    ptm_repo_values: set[str],
    dataset_source_values: set[str],
    scorer_checkpoint_values: set[str],
) -> dict[str, Any]:
    if schema_values and schema_values != {FRAME_SEQUENCE_FEATURE_SCHEMA}:
        raise ValueError(f"unsupported sequence feature schema: {sorted(schema_values)}")
    if len(hash_values) > 1:
        raise ValueError("mixed feature_schema_hash values are not allowed")
    if not ptm_repo_values:
        raise ValueError("boundary refiner v6 rows require metadata.ptm_repo_id")
    if len(ptm_repo_values) > 1:
        raise ValueError(f"mixed ptm_repo_id values are not allowed: {sorted(ptm_repo_values)}")
    if len(dataset_source_values) > 1:
        raise ValueError(f"mixed dataset_source values are not allowed: {sorted(dataset_source_values)}")
    if len(scorer_checkpoint_values) > 1:
        raise ValueError("mixed scorer_checkpoint metadata values are not allowed")
    ptm_repo_id = next(iter(ptm_repo_values))
    extra_metadata: dict[str, Any] = {}
    if dataset_source_values:
        extra_metadata["dataset_source"] = next(iter(dataset_source_values))
    if scorer_checkpoint_values:
        extra_metadata["scorer_checkpoint"] = json.loads(next(iter(scorer_checkpoint_values)))
    if expected_signature is not None:
        feature_schema_hash = next(iter(hash_values)) if hash_values else _hash_from_signature(expected_signature)
        metadata = {
            "feature_schema": str(expected_signature.get("feature_schema") or FRAME_SEQUENCE_FEATURE_SCHEMA),
            "feature_schema_hash": feature_schema_hash,
            "feature_signature": dict(expected_signature),
            "feature_dim": len(feature_names),
            "ptm_repo_id": ptm_repo_id,
            **extra_metadata,
        }
        return metadata
    if schema_values or hash_values:
        raise ValueError("sequence feature rows require feature_signature metadata")
    metadata = {
        "feature_dim": len(feature_names),
        "ptm_repo_id": ptm_repo_id,
        **extra_metadata,
    }
    return metadata


def _hash_from_signature(signature: Mapping[str, Any]) -> str:
    feature_config = signature.get("feature_config")
    feature_names = signature.get("feature_names")
    if not isinstance(feature_config, Mapping) or not isinstance(feature_names, list):
        raise ValueError("feature_signature must contain feature_config and feature_names")
    from boundary.sequence_features import FrameSequenceFeatureConfig

    config = FrameSequenceFeatureConfig(
        left_context_s=float(feature_config["left_context_s"]),
        right_context_s=float(feature_config["right_context_s"]),
        max_ptm_dims=int(feature_config["max_ptm_dims"]),
        include_mfcc=bool(feature_config["include_mfcc"]),
    )
    return feature_extraction_hash(
        config=config,
        feature_names=[str(name) for name in feature_names],
    )


def _row_sequence_length(row: Mapping[str, Any], feature_names: tuple[str, ...]) -> int:
    raw_sequence = row.get("sequence_features")
    if isinstance(raw_sequence, list) and raw_sequence:
        first = raw_sequence[0]
        if not isinstance(first, list) or len(first) != len(feature_names):
            validate_sequence_features(
                raw_sequence,
                feature_names=feature_names,
                expected_feature_names=row.get("feature_names") or feature_names,
            )
        return len(raw_sequence)
    raise ValueError("boundary refiner v6 rows require sequence_features")


def _row_sequence_array(row: Mapping[str, Any], feature_names: tuple[str, ...]) -> np.ndarray:
    raw_sequence = row.get("sequence_features")
    if isinstance(raw_sequence, list) and raw_sequence:
        return validate_sequence_features(
            raw_sequence,
            feature_names=feature_names,
            expected_feature_names=row.get("feature_names") or feature_names,
        ).astype(np.float32, copy=False)
    raise ValueError("boundary refiner v6 rows require sequence_features")


def _row_sequence(row: Mapping[str, Any], feature_names: tuple[str, ...]) -> list[list[float]]:
    return _row_sequence_array(row, feature_names).astype(float).tolist()


def _row_boundary_delta_target_sequence(row: Mapping[str, Any], *, expected_len: int) -> list[list[float]]:
    raw = row.get("sequence_boundary_delta_targets")
    if isinstance(raw, list) and raw:
        if len(raw) != expected_len:
            raise ValueError("sequence_boundary_delta_targets length must match sequence_features length")
        targets: list[list[float]] = []
        for item in raw:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(
                    "sequence_boundary_delta_targets items must be [start_delta_s, end_delta_s]"
                )
            start_delta, end_delta = float(item[0]), float(item[1])
            if not math.isfinite(start_delta) or not math.isfinite(end_delta):
                raise ValueError("sequence_boundary_delta_targets must not contain NaN or inf")
            targets.append([start_delta, end_delta])
        return targets
    raise ValueError("boundary refiner v6 rows require sequence_boundary_delta_targets")


def _row_boundary_delta_weight_sequence(row: Mapping[str, Any], *, expected_len: int) -> list[list[float]]:
    raw = row.get("sequence_boundary_delta_weights")
    if raw is None:
        return [[1.0, 1.0] for _ in range(expected_len)]
    if isinstance(raw, list) and raw:
        if len(raw) != expected_len:
            raise ValueError("sequence_boundary_delta_weights length must match sequence_features length")
        weights: list[list[float]] = []
        for item in raw:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(
                    "sequence_boundary_delta_weights items must be [start_weight, end_weight]"
                )
            start_weight, end_weight = float(item[0]), float(item[1])
            if (
                not math.isfinite(start_weight)
                or not math.isfinite(end_weight)
                or start_weight < 0.0
                or end_weight < 0.0
            ):
                raise ValueError("sequence_boundary_delta_weights must be finite non-negative values")
            weights.append([start_weight, end_weight])
        return weights
    raise ValueError("sequence_boundary_delta_weights must be omitted or a non-empty list")


def _weighted_delta_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor,
    mask: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    active_weights = weights * mask.to(dtype=weights.dtype)
    denominator = active_weights.sum().clamp_min(1.0)
    losses = criterion(pred, target)
    return (losses * active_weights).sum() / denominator


def _split_indexes(count: int, *, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indexes = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indexes)
    val_count = int(round(count * val_ratio)) if count > 1 else 0
    val_count = min(max(0, val_count), max(0, count - 1))
    return indexes[val_count:], indexes[:val_count]


def _evaluate(
    model: BoundarySequenceClassifier,
    features: torch.Tensor,
    boundary_delta_targets: torch.Tensor,
    boundary_delta_weights: torch.Tensor,
    mask: torch.Tensor,
    *,
    indexes: Sequence[int] | None = None,
    device: torch.device,
    boundary_delta_max_s: float,
    batch_size: int = 256,
) -> dict[str, Any]:
    if features.numel() == 0:
        return {}
    if indexes is None:
        indexes = list(range(features.shape[0]))
    if not indexes:
        return {}
    model.eval()
    count = 0
    start_weighted_abs_error = 0.0
    end_weighted_abs_error = 0.0
    start_weight_sum = 0.0
    end_weight_sum = 0.0
    max_abs_error = 0.0
    start_error_parts: list[torch.Tensor] = []
    end_error_parts: list[torch.Tensor] = []
    start_weight_parts: list[torch.Tensor] = []
    end_weight_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(indexes), max(1, batch_size)):
            batch_indexes = list(indexes[start : start + max(1, batch_size)])
            batch_mask = mask[batch_indexes]
            if not bool(batch_mask.any().item()):
                continue
            logits = model(
                features[batch_indexes].to(device),
                attention_mask=batch_mask.to(device).long(),
            ).detach().cpu()
            pred = torch.tanh(logits[..., :2]) * float(boundary_delta_max_s)
            target = boundary_delta_targets[batch_indexes]
            weights = boundary_delta_weights[batch_indexes]
            active = batch_mask.unsqueeze(-1)
            abs_error = torch.abs(pred - target) * active
            start_weights = weights[..., 0] * batch_mask.to(dtype=weights.dtype)
            end_weights = weights[..., 1] * batch_mask.to(dtype=weights.dtype)
            start_weighted_abs_error += float((abs_error[..., 0] * start_weights).sum().item())
            end_weighted_abs_error += float((abs_error[..., 1] * end_weights).sum().item())
            start_weight_sum += float(start_weights.sum().item())
            end_weight_sum += float(end_weights.sum().item())
            count += int(batch_mask.sum().item())
            start_active = batch_mask & (start_weights > 0.0)
            end_active = batch_mask & (end_weights > 0.0)
            if bool(start_active.any().item()):
                start_error_parts.append((pred[..., 0] - target[..., 0])[start_active].detach().cpu())
                start_weight_parts.append(start_weights[start_active].detach().cpu())
            if bool(end_active.any().item()):
                end_error_parts.append((pred[..., 1] - target[..., 1])[end_active].detach().cpu())
                end_weight_parts.append(end_weights[end_active].detach().cpu())
            if bool(batch_mask.any().item()):
                max_abs_error = max(max_abs_error, float(abs_error[active.expand_as(abs_error)].max().item()))
    if count == 0:
        return {}
    start_error_stats = _boundary_delta_error_stats(
        _cat_or_empty(start_error_parts),
        _cat_or_empty(start_weight_parts),
    )
    end_error_stats = _boundary_delta_error_stats(
        _cat_or_empty(end_error_parts),
        _cat_or_empty(end_weight_parts),
    )
    return {
        "count": count,
        "start_delta_mae_s": start_weighted_abs_error / max(1e-9, start_weight_sum),
        "end_delta_mae_s": end_weighted_abs_error / max(1e-9, end_weight_sum),
        "start_weight_sum": start_weight_sum,
        "end_weight_sum": end_weight_sum,
        "max_abs_delta_error_s": max_abs_error,
        "start_delta_error": start_error_stats,
        "end_delta_error": end_error_stats,
    }


def _cat_or_empty(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    if not parts:
        return torch.empty((0,), dtype=torch.float32)
    return torch.cat([part.to(dtype=torch.float32).flatten() for part in parts], dim=0)


def _boundary_delta_error_stats(
    errors: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, Any]:
    errors = errors.detach().cpu().to(dtype=torch.float32).flatten()
    weights = weights.detach().cpu().to(dtype=torch.float32).flatten()
    if errors.numel() != weights.numel():
        raise ValueError("errors and weights must have the same length")
    if errors.numel() == 0:
        return {
            "count": 0,
            "weight_sum": 0.0,
        }
    finite = torch.isfinite(errors) & torch.isfinite(weights) & (weights > 0.0)
    errors = errors[finite]
    weights = weights[finite]
    if errors.numel() == 0:
        return {
            "count": 0,
            "weight_sum": 0.0,
        }
    abs_errors = errors.abs()
    weight_sum = weights.sum().clamp_min(1e-9)
    return {
        "count": int(errors.numel()),
        "weight_sum": float(weights.sum().item()),
        "mean_error_s": float((errors * weights).sum().item() / float(weight_sum.item())),
        "mae_s": float((abs_errors * weights).sum().item() / float(weight_sum.item())),
        "signed_error_p05_s": _quantile(errors, 0.05),
        "signed_error_p10_s": _quantile(errors, 0.10),
        "signed_error_p50_s": _quantile(errors, 0.50),
        "signed_error_p90_s": _quantile(errors, 0.90),
        "signed_error_p95_s": _quantile(errors, 0.95),
        "abs_error_p50_s": _quantile(abs_errors, 0.50),
        "abs_error_p90_s": _quantile(abs_errors, 0.90),
        "abs_error_p95_s": _quantile(abs_errors, 0.95),
        "abs_error_p99_s": _quantile(abs_errors, 0.99),
        "max_abs_error_s": float(abs_errors.max().item()),
    }


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.to(dtype=torch.float32), float(q)).item())


def _resolve_device(requested: str) -> torch.device:
    normalized = (requested or "auto").strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA training but torch.cuda.is_available() is false")
    return torch.device(normalized)


def _validate_config(config: TrainRefinerConfig) -> None:
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if not 0.0 <= config.val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if config.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if config.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if config.state_size <= 0:
        raise ValueError("state_size must be positive")
    if config.num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if config.n_groups <= 0:
        raise ValueError("n_groups must be positive")
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if config.log_interval_steps < 0:
        raise ValueError("log_interval_steps must be non-negative")
    if config.boundary_delta_max_s <= 0.0:
        raise ValueError("boundary_delta_max_s must be positive")
    if config.start_delta_loss_weight < 0.0:
        raise ValueError("start_delta_loss_weight must be non-negative")
    if config.end_delta_loss_weight < 0.0:
        raise ValueError("end_delta_loss_weight must be non-negative")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a transformers.Mamba2Model Boundary Refiner v6 from scorer island edge samples."
    )
    parser.add_argument("--dataset", action="append", required=True, help="Gap dataset JSONL. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--state-size", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--n-groups", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--unidirectional", action="store_true")
    parser.add_argument("--log-interval-steps", type=int, default=20)
    parser.add_argument("--boundary-delta-max-s", type=float, default=DEFAULT_BOUNDARY_DELTA_MAX_S)
    parser.add_argument("--start-delta-loss-weight", type=float, default=1.00)
    parser.add_argument("--end-delta-loss-weight", type=float, default=0.60)
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Optional boundary_edge_refiner_v6 checkpoint used to initialize weights.",
    )
    parser.add_argument(
        "--preserve-init-normalization",
        action="store_true",
        help="Reuse feature_mean/std from --init-checkpoint instead of recomputing them from the fine-tune dataset.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the Mamba2 backbone and train only the output head.",
    )
    parser.add_argument(
        "--tensor-cache-path",
        default="",
        help=(
            "Optional .pt tensor cache. If missing, JSONL is streamed once into this cache; "
            "if present, training loads tensors directly."
        ),
    )
    parser.add_argument(
        "--loader-log-interval-rows",
        type=int,
        default=0,
        help="Print JSONL scan/load progress every N rows; 0 disables loader progress logs.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="",
        help="Checkpoint file name. Default appends the dataset metadata.ptm_repo_id repo id tag.",
    )
    args = parser.parse_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_refiner(
        dataset_paths=[Path(path) for path in args.dataset],
        output_dir=Path(args.output_dir),
        config=TrainRefinerConfig(
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            val_ratio=args.val_ratio,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            state_size=args.state_size,
            num_heads=args.num_heads,
            n_groups=args.n_groups,
            chunk_size=args.chunk_size,
            bidirectional=not args.unidirectional,
            log_interval_steps=args.log_interval_steps,
            boundary_delta_max_s=args.boundary_delta_max_s,
            start_delta_loss_weight=args.start_delta_loss_weight,
            end_delta_loss_weight=args.end_delta_loss_weight,
            init_checkpoint=args.init_checkpoint,
            preserve_init_normalization=args.preserve_init_normalization,
            freeze_backbone=args.freeze_backbone,
            tensor_cache_path=args.tensor_cache_path,
            loader_log_interval_rows=args.loader_log_interval_rows,
            checkpoint_name=args.checkpoint_name,
        ),
    )


if __name__ == "__main__":
    main()
