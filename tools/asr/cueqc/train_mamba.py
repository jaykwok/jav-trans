#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.cueqc import CUEQC_FEATURE_SCHEMA_VERSION, CUEQC_TAXONOMY_VERSION, numeric_feature_vector

TRANSFORMERS_MAMBA2_BACKBONE = "transformers.Mamba2Model"


CONTENT_LABELS = ("dialogue", "non_dialogue", "mixed", "uncertain")
DISPLAY_LABELS = ("keep", "drop", "compact", "review")
ALIGNMENT_LABELS = ("align", "skip_align_fallback")
QC_LABELS = ("keep", "review", "reject")


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 13
    device: str = "auto"
    hidden_size: int = 64
    num_layers: int = 1
    state_size: int = 16
    num_heads: int = 4
    n_groups: int = 2
    chunk_size: int = 8


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def _feature_row(record: Mapping[str, Any]) -> dict[str, Any]:
    features = record.get("features") if isinstance(record.get("features"), Mapping) else {}
    return {
        "duration_s": record.get("duration_s", 0.0),
        "text_features": features.get("text_features", {}),
        "qc": features.get("qc", {}),
        "adjacency": features.get("adjacency", {}),
    }


def _target_index(value: str, labels: tuple[str, ...]) -> int:
    try:
        return labels.index(value)
    except ValueError:
        return labels.index(labels[-1])


def train(dataset_path: Path, output_dir: Path, config: TrainConfig) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from boundary.backbones import (
            TRANSFORMERS_MAMBA2_BACKBONE as RUNTIME_MAMBA2_BACKBONE,
            BoundarySequenceClassifier,
        )
    except Exception as exc:  # pragma: no cover - exercised in minimal envs
        raise RuntimeError(
            "CueQC Mamba training requires torch, transformers, and the boundary Mamba2 backbone"
        ) from exc

    rows = read_jsonl(dataset_path)
    if not rows:
        raise ValueError("empty CueQC training dataset")
    vectors = [numeric_feature_vector(_feature_row(row)) for row in rows]
    feature_dim = len(vectors[0])
    targets = []
    for row in rows:
        raw_targets = row.get("targets") if isinstance(row.get("targets"), Mapping) else {}
        targets.append(
            [
                _target_index(str(raw_targets.get("content_type") or "uncertain"), CONTENT_LABELS),
                _target_index(str(raw_targets.get("display_hint") or "review"), DISPLAY_LABELS),
                _target_index(str(raw_targets.get("alignment_policy") or "align"), ALIGNMENT_LABELS),
                _target_index(str(raw_targets.get("qc_decision") or "review"), QC_LABELS),
            ]
        )
    features = torch.tensor(vectors, dtype=torch.float32).unsqueeze(1)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    mean = features.mean(dim=(0, 1))
    std = features.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    features = (features - mean) / std
    device = _resolve_device(config.device, torch)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = BoundarySequenceClassifier(
        input_dim=feature_dim,
        backbone=RUNTIME_MAMBA2_BACKBONE,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_dim=sum(
            len(labels)
            for labels in (CONTENT_LABELS, DISPLAY_LABELS, ALIGNMENT_LABELS, QC_LABELS)
        ),
        state_size=config.state_size,
        num_heads=config.num_heads,
        n_groups=config.n_groups,
        chunk_size=config.chunk_size,
        bidirectional=True,
    ).to(device)
    dataset = TensorDataset(features, target_tensor)
    loader = DataLoader(dataset, batch_size=min(config.batch_size, len(dataset)), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    spans = _head_spans()
    last_loss = 0.0
    for step in range(config.max_steps):
        for batch_features, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features.to(device), attention_mask=torch.ones(batch_features.shape[:2], dtype=torch.long, device=device))
            logits = logits[:, 0, :] if logits.ndim == 3 else logits
            loss = (
                criterion(logits[:, spans["content"][0] : spans["content"][1]], batch_targets[:, 0].to(device))
                + criterion(logits[:, spans["display"][0] : spans["display"][1]], batch_targets[:, 1].to(device))
                + criterion(logits[:, spans["alignment"][0] : spans["alignment"][1]], batch_targets[:, 2].to(device))
                + criterion(logits[:, spans["qc"][0] : spans["qc"][1]], batch_targets[:, 3].to(device))
            )
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            break
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "cueqc_mamba.pt"
    checkpoint = {
        "schema": "cueqc_mamba_checkpoint_v1",
        "feature_schema_version": CUEQC_FEATURE_SCHEMA_VERSION,
        "taxonomy_version": CUEQC_TAXONOMY_VERSION,
        "model": model.to("cpu").state_dict(),
        "model_config": dict(model.model_config),
        "feature_mean": [float(value) for value in mean.tolist()],
        "feature_std": [float(value) for value in std.tolist()],
        "heads": {
            "content_type": list(CONTENT_LABELS),
            "display_hint": list(DISPLAY_LABELS),
            "alignment_policy": list(ALIGNMENT_LABELS),
            "qc_decision": list(QC_LABELS),
        },
        "head_spans": spans,
        "train_config": asdict(config),
        "dataset_path": str(dataset_path),
    }
    torch.save(checkpoint, checkpoint_path)
    metrics = {
        "schema": "cueqc_mamba_train_metrics_v1",
        "checkpoint": str(checkpoint_path),
        "records": len(rows),
        "feature_dim": feature_dim,
        "last_loss": last_loss,
        "taxonomy_version": CUEQC_TAXONOMY_VERSION,
        "feature_schema_version": CUEQC_FEATURE_SCHEMA_VERSION,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"checkpoint={checkpoint_path}")
    print(f"metrics={metrics_path}")
    return metrics


def _head_spans() -> dict[str, tuple[int, int]]:
    cursor = 0
    spans: dict[str, tuple[int, int]] = {}
    for name, labels in (
        ("content", CONTENT_LABELS),
        ("display", DISPLAY_LABELS),
        ("alignment", ALIGNMENT_LABELS),
        ("qc", QC_LABELS),
    ):
        spans[name] = (cursor, cursor + len(labels))
        cursor += len(labels)
    return spans


def _resolve_device(requested: str, torch_module) -> Any:
    normalized = (requested or "auto").strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch_module.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    return torch_module.device(normalized)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train first CueQC Mamba multi-head classifier.")
    parser.add_argument("--dataset", required=True, help="cueqc_train.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--state-size", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--n-groups", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train(
        Path(args.dataset),
        Path(args.output_dir),
        TrainConfig(
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            state_size=args.state_size,
            num_heads=args.num_heads,
            n_groups=args.n_groups,
            chunk_size=args.chunk_size,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
