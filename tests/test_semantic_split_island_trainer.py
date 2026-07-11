from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from boundary.split_model import load_semantic_split_verifier
from tools.boundary.ja.train_semantic_split_island_model import (
    build_lr_scheduler,
    calibrate_thresholds,
    island_batches,
    load_island_dataset,
    run,
    split_group_names,
)


SCALAR_DIM = 13
FRAME_DIM = 6
BINS = 20


def _write_dataset(path: Path, *, islands: int, val_islands: int) -> None:
    rng = np.random.default_rng(11)
    frames: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    labels: list[int] = []
    partitions: list[str] = []
    group_ids: list[str] = []
    times: list[float] = []
    roles: list[int] = []
    pairs: list[int] = []
    omni: list[list[float]] = []
    dataset_roles: list[str] = []
    pair_counter = 0
    for island in range(islands):
        partition = "val" if island < val_islands else "train"
        count = int(rng.integers(3, 6))
        cut_positions = {0} if island % 2 == 0 else set()
        pair_positions: tuple[int, int] | None = None
        if island % 4 == 0 and count >= 3:
            pair_positions = (0, 2)
            cut_positions = {0, 2}
        for position in range(count):
            is_cut = position in cut_positions
            base = 2.0 if is_cut else -2.0
            frames.append(
                (base + rng.normal(0, 0.3, size=(BINS, FRAME_DIM))).astype(np.float32)
            )
            scalar = rng.normal(0, 0.3, size=SCALAR_DIM).astype(np.float32)
            scalar[0] = base
            scalar[4] = 8.0  # core_duration_s -> normal regime
            scalars.append(scalar)
            labels.append(0 if is_cut else 1)
            partitions.append(partition)
            group_ids.append(f"synthetic|island{island:03d}")
            times.append(float(position))
            roles.append(1 if is_cut else 0)
            if pair_positions and position in pair_positions:
                pairs.append(pair_counter)
            else:
                pairs.append(-1)
            omni.append([1.0, 1.0, 0.0] if is_cut else [-1.0, -1.0, -1.0])
            dataset_roles.append("smoke")
        if pair_positions:
            pair_counter += 1
    np.savez(
        path,
        frame_features=np.stack(frames),
        scalar_features=np.stack(scalars),
        labels=np.asarray(labels, dtype=np.int64),
        partitions=np.asarray(partitions),
        group_ids=np.asarray(group_ids),
        times_s=np.asarray(times, dtype=np.float32),
        structural_roles=np.asarray(roles, dtype=np.int64),
        pair_ids=np.asarray(pairs, dtype=np.int64),
        omni_aux=np.asarray(omni, dtype=np.float32),
        dataset_roles=np.asarray(dataset_roles),
    )


def test_island_batches_respect_candidate_cap() -> None:
    groups = {
        "a": np.arange(4),
        "b": np.arange(4),
        "c": np.arange(4),
    }
    batches = island_batches(
        ["a", "b", "c"], groups, batch_islands=8, max_batch_candidates=8
    )
    assert [len(batch) for batch in batches] == [2, 1]


def test_cosine_scheduler_uses_lr_multipliers() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    scheduler = build_lr_scheduler(
        optimizer,
        schedule="cosine",
        warmup_steps=2,
        max_steps=6,
    )

    assert scheduler is not None
    assert optimizer.param_groups[0]["lr"] == 1e-4
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 2e-4
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.0


def test_trainer_smoke_produces_calibrated_v2_checkpoint(tmp_path: Path) -> None:
    dataset = tmp_path / "sequence.npz"
    _write_dataset(dataset, islands=16, val_islands=4)
    output_dir = tmp_path / "train-out"
    run(
        argparse.Namespace(
            dataset=str(dataset),
            output_dir=str(output_dir),
            ptm_repo_id="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
            ptm_dim=4,
            max_steps=24,
            eval_every=12,
            batch_islands=4,
            max_batch_candidates=64,
            learning_rate=5e-3,
            hidden_size=16,
            candidate_layers=1,
            island_layers=1,
            dropout=0.0,
            cut_weight=2.0,
            continue_weight=1.0,
            unsure_gate_weight=0.0,
            focal_gamma=1.5,
            label_aux_weight=0.3,
            role_aux_weight=0.3,
            omni_aux_weight=0.2,
            pair_loss_weight=0.5,
            cut_island_ratio=0.7,
            calibration_min_precision=0.5,
            calibration_domain_prefix="real_",
            offset_weight=0.3,
            extra_context_scales="",
            ptm_projection="",
            ptm_projector_dim=0,
            ptm_projector_residual=False,
            freeze_ptm_projector=False,
            freeze_backbone=False,
            init_checkpoint="",
            lr_schedule="constant",
            warmup_steps=0,
            seed=13,
            device="cpu",
            log_every=0,
        )
    )
    checkpoint = (
        output_dir
        / "semantic_split_model_v2.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf.pt"
    )
    assert checkpoint.exists()
    metrics = json.loads((output_dir / "metrics.json").read_text("utf-8"))
    assert "decision_config" in metrics
    assert "normal_cut_threshold" in metrics["decision_config"]
    assert "smoke" in metrics["val_at_calibrated_thresholds"]["domains"]
    assert metrics["val_at_calibrated_thresholds"]["complete_pair_count"] >= 1
    # No real_ domains here, so pooled_real falls back to all domains.
    pooled = metrics["val_at_calibrated_thresholds"]["pooled_real"]
    assert pooled["domains"] == ["smoke"]
    assert 0.0 <= pooled["cut_f1"] <= 1.0
    if "cut_f1_ci95" in pooled:
        low, high = pooled["cut_f1_ci95"]
        assert low <= high

    verifier = load_semantic_split_verifier(checkpoint, device="cpu")
    data = load_island_dataset(dataset)
    _train, val = split_group_names(data)
    indexes = data["groups"][val[0]]
    decisions = verifier.decide_islands(
        island_frame_features=[data["frames"][indexes]],
        island_scalar_features=[data["scalars"][indexes]],
    )
    assert len(decisions[0]) == int(indexes.size)


def test_calibrate_thresholds_prefers_high_f1_at_min_precision(tmp_path: Path) -> None:
    dataset = tmp_path / "sequence.npz"
    _write_dataset(dataset, islands=8, val_islands=2)
    data = load_island_dataset(dataset)
    gate_by_row = {
        index: (0.9 if int(data["labels"][index]) == 0 else 0.2)
        for name in data["groups"]
        for index in data["groups"][name].tolist()
    }
    decision = calibrate_thresholds(
        data,
        gate_by_row,
        short_core_max_s=6.0,
        min_precision=0.85,
    )
    assert 0.3 <= decision["normal_cut_threshold"] <= 0.9


def test_calibrate_thresholds_domain_prefix_ignores_other_domains(tmp_path: Path) -> None:
    dataset = tmp_path / "sequence.npz"
    _write_dataset(dataset, islands=8, val_islands=2)
    data = load_island_dataset(dataset)
    # Every third row belongs to the deployment domain; give it clean scores
    # so a low threshold works, while the noisy majority domain would push the
    # pooled calibration to the top of the sweep.
    roles = data["dataset_roles"].copy()
    gate_by_row: dict[int, float] = {}
    for name in data["groups"]:
        for index in data["groups"][name].tolist():
            if index % 3 == 0:
                roles[index] = "real_anchor_v3"
                gate_by_row[index] = 0.6 if int(data["labels"][index]) == 0 else 0.1
            else:
                roles[index] = "hardmix_structural"
                gate_by_row[index] = 0.9 if int(data["labels"][index]) == 0 else 0.85
    data["dataset_roles"] = roles

    filtered = calibrate_thresholds(
        data,
        gate_by_row,
        short_core_max_s=6.0,
        min_precision=0.85,
        domain_prefix="real_",
    )
    pooled = calibrate_thresholds(
        data,
        gate_by_row,
        short_core_max_s=6.0,
        min_precision=0.85,
    )
    assert filtered["normal_cut_threshold"] <= 0.6
    assert pooled["normal_cut_threshold"] > filtered["normal_cut_threshold"]

    # An unmatched prefix falls back to pooled rows instead of the defaults.
    fallback = calibrate_thresholds(
        data,
        gate_by_row,
        short_core_max_s=6.0,
        min_precision=0.85,
        domain_prefix="nonexistent_",
    )
    assert fallback["normal_cut_threshold"] == pooled["normal_cut_threshold"]
