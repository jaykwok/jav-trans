#!/usr/bin/env python3
"""Train the random-init 1.7B Scorer v11 candidate-membership model."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from boundary.ja.candidate_training import candidate_boundary_heatmap_loss  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    CANDIDATE_ISLAND_SCORER_V11_CAPACITY_PROFILES,
    CANDIDATE_ISLAND_SCORER_V11_COMPACT_CAPACITY_PROFILE,
    CANDIDATE_ISLAND_SCORER_V11_DATASET_CONTRACT,
    CANDIDATE_ISLAND_SCORER_V11_FEATURE_CACHE_GATE_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_FULL_CAPACITY_PROFILE,
    CANDIDATE_ISLAND_SCORER_V11_HEATMAP_AUXILIARY,
    CANDIDATE_ISLAND_SCORER_V11_HEATMAP_MODEL_ARCH,
    CANDIDATE_ISLAND_SCORER_V11_HEATMAP_SCHEMA,
    CANDIDATE_ISLAND_SCORER_V11_HEATMAP_SIGMA_FRAMES,
    CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM,
    CANDIDATE_ISLAND_SCORER_V11_RAW_PTM_DIM,
    CANDIDATE_ISLAND_SCORER_V11_TRAINING_ROW_SCHEMA,
    CandidateIslandHeatmapScorerNetwork,
    CandidateIslandScorerNetwork,
    build_speech_island_scorer_checkpoint,
)
from pipeline.memory_safety import (  # noqa: E402
    reset_shared_vram_baseline,
    runtime_memory_snapshot,
)


SUMMARY_SCHEMA = "candidate_island_scorer_v11_training_summary_v1"
IGNORE_INDEX = -100
PARTITIONS = {"train", "val", "test"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def validate_training_inputs(
    *, dataset_manifest: Path, feature_cache_gate: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    dataset_manifest = dataset_manifest.resolve()
    feature_cache_gate = feature_cache_gate.resolve()
    rows = _read_jsonl(dataset_manifest)
    gate = _read_json(feature_cache_gate)
    if gate.get("schema") != CANDIDATE_ISLAND_SCORER_V11_FEATURE_CACHE_GATE_SCHEMA:
        raise ValueError("Scorer v11 trainer requires the current feature-cache gate")
    if gate.get("boundary_serialization_contract_id") != (
        ACOUSTIC_BINARY_V12_CONTRACT.contract_id
    ):
        raise ValueError("Scorer v11 trainer requires the central boundary contract")
    if not bool(gate.get("training_manifest_allowed")):
        raise ValueError("Scorer v11 feature cache is not approved for training")
    manifest_sha = _sha256(dataset_manifest)
    if gate.get("dataset_manifest_sha256") != manifest_sha:
        raise ValueError("Scorer v11 dataset manifest SHA256 mismatch")
    if _resolve(str(gate.get("dataset_manifest") or "")) != dataset_manifest:
        raise ValueError("Scorer v11 gate points to another dataset manifest")
    signed_manifest = _resolve(str(gate.get("signed_feature_manifest") or ""))
    signed_sha = _sha256(signed_manifest)
    if gate.get("signed_feature_manifest_sha256") != signed_sha:
        raise ValueError("Scorer v11 signed feature manifest SHA256 mismatch")

    row_ids: set[str] = set()
    source_partition: dict[str, str] = {}
    source_core_key: dict[str, tuple[str, ...]] = {}
    core_owner: dict[str, tuple[str, str]] = {}
    partition_counts: Counter[str] = Counter()
    owner_by_source: dict[str, list[tuple[int, int]]] = {}
    provenance_fields = {
        "canonical_sources_sha256": set(),
        "signed_feature_manifest_sha256": set(),
        "feature_config_sha256": set(),
        "raw_feature_manifest_sha256": set(),
    }
    for row in rows:
        if row.get("schema") != CANDIDATE_ISLAND_SCORER_V11_TRAINING_ROW_SCHEMA:
            raise ValueError("Scorer v11 trainer rejects legacy training rows")
        if row.get("boundary_serialization_contract_id") != (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError("Scorer v11 row has the wrong central contract")
        row_id = str(row.get("row_id") or "")
        if not row_id or row_id in row_ids:
            raise ValueError(f"duplicate or missing Scorer v11 row_id: {row_id!r}")
        row_ids.add(row_id)
        source_id = str(row.get("source_id") or "")
        partition = str(row.get("partition") or "")
        if not source_id or partition not in PARTITIONS:
            raise ValueError("Scorer v11 row requires source_id and fixed partition")
        previous_partition = source_partition.setdefault(source_id, partition)
        if previous_partition != partition:
            raise ValueError(f"Scorer v11 source crosses partitions: {source_id}")
        core_ids = tuple(str(value) for value in row.get("core_ids") or ())
        if not core_ids:
            raise ValueError(f"Scorer v11 source has no core identity: {source_id}")
        previous_core_key = source_core_key.setdefault(source_id, core_ids)
        if previous_core_key != core_ids:
            raise ValueError(f"Scorer v11 source changes core identity: {source_id}")
        for core_id in core_ids:
            owner = (source_id, partition)
            previous_owner = core_owner.setdefault(core_id, owner)
            if previous_owner != owner:
                raise ValueError(f"Scorer v11 core is reused or crosses partitions: {core_id}")
        synthetic = bool(row.get("synthetic_composite"))
        if synthetic and partition != "train":
            raise ValueError("Scorer v11 synthetic composites are train-only")
        if partition in {"val", "test"} and row.get("input_distribution") != (
            "real_workflow_source_windows"
        ):
            raise ValueError("Scorer v11 held-out rows must be real workflow windows")
        start = int(row.get("window_start_frame", -1))
        end = int(row.get("window_end_frame", -1))
        owner_start = int(row.get("owner_start_frame", -1))
        owner_end = int(row.get("owner_end_frame", -1))
        if not (0 <= start <= owner_start < owner_end <= end <= int(row.get("source_frame_count") or 0)):
            raise ValueError(f"invalid Scorer v11 window ownership: {row_id}")
        if int(row.get("owner_local_start", -1)) != owner_start - start or int(
            row.get("owner_local_end", -1)
        ) != owner_end - start:
            raise ValueError(f"Scorer v11 local ownership mismatch: {row_id}")
        if int(row.get("context_window_frames") or 0) != int(
            CANDIDATE_ISLAND_SCORER_V11_DATASET_CONTRACT["context_window_frames"]
        ):
            raise ValueError("Scorer v11 context window contract mismatch")
        owner_by_source.setdefault(source_id, []).append((owner_start, owner_end))
        partition_counts[partition] += 1
        for key in provenance_fields:
            provenance_fields[key].add(str(row.get(key) or ""))

    if not rows or set(partition_counts) != PARTITIONS:
        raise ValueError(f"Scorer v11 requires non-empty train/val/test: {dict(partition_counts)}")
    for source_id, intervals in owner_by_source.items():
        ordered = sorted(intervals)
        expected = 0
        for start, end in ordered:
            if start != expected:
                raise ValueError(f"Scorer v11 owner coverage has gap/overlap: {source_id}")
            expected = end
        source_rows = [row for row in rows if row["source_id"] == source_id]
        if expected != int(source_rows[0]["source_frame_count"]):
            raise ValueError(f"Scorer v11 owner coverage misses source tail: {source_id}")
    for key, values in provenance_fields.items():
        if len(values) != 1 or len(next(iter(values))) != 64:
            raise ValueError(f"Scorer v11 rows have inconsistent {key}")
    if next(iter(provenance_fields["signed_feature_manifest_sha256"])) != signed_sha:
        raise ValueError("Scorer v11 rows point to another signed feature manifest")
    provenance = {
        key: next(iter(values)) for key, values in provenance_fields.items()
    }
    return rows, gate, provenance


class SourceArrayCache:
    def __init__(self, max_sources: int = 4) -> None:
        self.max_sources = max(1, int(max_sources))
        self._values: OrderedDict[tuple[str, str], dict[str, np.ndarray]] = OrderedDict()

    def load(self, row: dict[str, Any]) -> dict[str, np.ndarray]:
        key = (str(row["feature_path"]), str(row["label_path"]))
        cached = self._values.pop(key, None)
        if cached is not None:
            self._values[key] = cached
            return cached
        feature_path = _resolve(key[0])
        label_path = _resolve(key[1])
        with np.load(feature_path, allow_pickle=False) as payload:
            ptm = np.asarray(payload["ptm"], dtype=np.float32)
            mfcc = np.asarray(payload["mfcc"], dtype=np.float32)
        with np.load(label_path, allow_pickle=False) as payload:
            labels = np.asarray(payload["training_labels"], dtype=np.int64)
            start_heatmap = np.asarray(payload["start_heatmap"], dtype=np.float32)
            end_heatmap = np.asarray(payload["end_heatmap"], dtype=np.float32)
            boundary_valid = np.asarray(payload["boundary_valid"], dtype=np.bool_)
        expected = int(row["source_frame_count"])
        if ptm.shape != (expected, CANDIDATE_ISLAND_SCORER_V11_RAW_PTM_DIM):
            raise ValueError(f"Scorer v11 PTM shape mismatch: {row['row_id']}")
        if mfcc.shape != (expected, CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM):
            raise ValueError(f"Scorer v11 MFCC shape mismatch: {row['row_id']}")
        for value in (labels, start_heatmap, end_heatmap, boundary_valid):
            if value.shape != (expected,):
                raise ValueError(f"Scorer v11 label shape mismatch: {row['row_id']}")
        result = {
            "ptm": ptm,
            "mfcc": mfcc,
            "labels": labels,
            "start_heatmap": start_heatmap,
            "end_heatmap": end_heatmap,
            "boundary_valid": boundary_valid,
        }
        self._values[key] = result
        while len(self._values) > self.max_sources:
            self._values.popitem(last=False)
        return result

    def clear(self) -> None:
        self._values.clear()


def load_candidate_window(row: dict[str, Any], cache: SourceArrayCache) -> dict[str, Any]:
    source = cache.load(row)
    start = int(row["window_start_frame"])
    end = int(row["window_end_frame"])
    length = end - start
    owner = np.zeros(length, dtype=np.bool_)
    owner[int(row["owner_local_start"]) : int(row["owner_local_end"])] = True
    return {
        "row": row,
        "ptm": np.ascontiguousarray(source["ptm"][start:end]),
        "mfcc": np.ascontiguousarray(source["mfcc"][start:end]),
        "labels": np.ascontiguousarray(source["labels"][start:end]),
        "owner": owner,
        "start_heatmap": np.ascontiguousarray(source["start_heatmap"][start:end]),
        "end_heatmap": np.ascontiguousarray(source["end_heatmap"][start:end]),
        "boundary_valid": np.ascontiguousarray(source["boundary_valid"][start:end]),
    }


def compute_mfcc_normalization(
    rows: Sequence[dict[str, Any]], cache: SourceArrayCache
) -> dict[str, list[float]]:
    total = 0
    total_sum = np.zeros(CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM, dtype=np.float64)
    total_sq = np.zeros(CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM, dtype=np.float64)
    for row in rows:
        item = load_candidate_window(row, cache)
        valid = item["owner"] & (item["labels"] != IGNORE_INDEX)
        values = item["mfcc"][valid].astype(np.float64, copy=False)
        total += int(values.shape[0])
        total_sum += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
    if total <= 0:
        raise ValueError("Scorer v11 train normalization has no definite owner frames")
    mean = total_sum / total
    variance = np.maximum(total_sq / total - np.square(mean), 1e-12)
    return {
        "mfcc_mean": mean.astype(np.float32).tolist(),
        "mfcc_std": np.sqrt(variance).astype(np.float32).tolist(),
    }


def _pack_batches(
    rows: Sequence[dict[str, Any]], *, max_padded_frames: int
) -> list[list[dict[str, Any]]]:
    if max_padded_frames <= 0:
        raise ValueError("max_padded_frames must be positive")
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_max = 0
    for row in rows:
        length = int(row["window_end_frame"]) - int(row["window_start_frame"])
        proposed_max = max(current_max, length)
        if current and proposed_max * (len(current) + 1) > max_padded_frames:
            batches.append(current)
            current = []
            current_max = 0
        current.append(row)
        current_max = max(current_max, length)
    if current:
        batches.append(current)
    return batches


def _collate(items: Sequence[dict[str, Any]], torch, device) -> dict[str, Any]:
    lengths = [int(item["ptm"].shape[0]) for item in items]
    frames = max(lengths)
    batch = len(items)
    ptm = torch.zeros((batch, frames, CANDIDATE_ISLAND_SCORER_V11_RAW_PTM_DIM), dtype=torch.float32)
    mfcc = torch.zeros((batch, frames, CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM), dtype=torch.float32)
    labels = torch.full((batch, frames), IGNORE_INDEX, dtype=torch.long)
    attention = torch.zeros((batch, frames), dtype=torch.long)
    owner = torch.zeros((batch, frames), dtype=torch.bool)
    start_heatmap = torch.zeros((batch, frames), dtype=torch.float32)
    end_heatmap = torch.zeros((batch, frames), dtype=torch.float32)
    boundary_valid = torch.zeros((batch, frames), dtype=torch.bool)
    for index, item in enumerate(items):
        length = lengths[index]
        ptm[index, :length] = torch.from_numpy(item["ptm"])
        mfcc[index, :length] = torch.from_numpy(item["mfcc"])
        labels[index, :length] = torch.from_numpy(item["labels"])
        attention[index, :length] = 1
        owner[index, :length] = torch.from_numpy(item["owner"])
        start_heatmap[index, :length] = torch.from_numpy(item["start_heatmap"])
        end_heatmap[index, :length] = torch.from_numpy(item["end_heatmap"])
        boundary_valid[index, :length] = torch.from_numpy(item["boundary_valid"])
    return {
        "ptm": ptm.to(device),
        "mfcc": mfcc.to(device),
        "labels": labels.to(device),
        "attention": attention.to(device),
        "owner": owner.to(device),
        "start_heatmap": start_heatmap.to(device),
        "end_heatmap": end_heatmap.to(device),
        "boundary_valid": boundary_valid.to(device),
    }


def _model_config(args: argparse.Namespace, normalization: dict[str, list[float]]) -> dict[str, Any]:
    capacity_profile = str(args.capacity_profile)
    if capacity_profile not in CANDIDATE_ISLAND_SCORER_V11_CAPACITY_PROFILES:
        raise ValueError(f"unknown Scorer v11 capacity profile: {capacity_profile!r}")
    capacity = CANDIDATE_ISLAND_SCORER_V11_CAPACITY_PROFILES[capacity_profile]
    config = {
        "raw_ptm_dim": CANDIDATE_ISLAND_SCORER_V11_RAW_PTM_DIM,
        "projected_ptm_dim": int(capacity["projected_ptm_dim"]),
        "mfcc_dim": CANDIDATE_ISLAND_SCORER_V11_MFCC_DIM,
        "capacity_profile": capacity_profile,
        "hidden_size": int(capacity["hidden_size"]),
        "num_layers": int(capacity["num_layers"]),
        "state_size": int(capacity["state_size"]),
        "num_heads": int(capacity["num_heads"]),
        "head_dim": int(capacity["head_dim"]),
        "n_groups": int(capacity["n_groups"]),
        "conv_kernel": int(capacity["conv_kernel"]),
        "chunk_size": int(capacity["chunk_size"]),
        "bidirectional": True,
        "valid_prefix_bidirectional": True,
        "context_window_frames": CANDIDATE_ISLAND_SCORER_V11_DATASET_CONTRACT[
            "context_window_frames"
        ],
        "context_overlap_frames": CANDIDATE_ISLAND_SCORER_V11_DATASET_CONTRACT[
            "context_overlap_frames"
        ],
        "window_ownership": CANDIDATE_ISLAND_SCORER_V11_DATASET_CONTRACT[
            "window_ownership"
        ],
        "output_dim": 2,
        "mfcc_mean": normalization["mfcc_mean"],
        "mfcc_std": normalization["mfcc_std"],
    }
    if args.variant == "heatmap_aux":
        config.update(
            {
                "model_arch": CANDIDATE_ISLAND_SCORER_V11_HEATMAP_MODEL_ARCH,
                "boundary_heatmap_sigma_frames": CANDIDATE_ISLAND_SCORER_V11_HEATMAP_SIGMA_FRAMES,
                "boundary_auxiliary": CANDIDATE_ISLAND_SCORER_V11_HEATMAP_AUXILIARY,
            }
        )
    else:
        config["model_arch"] = str(capacity["model_arch"])
    return config


def _loss(model, batch, *, variant: str, heatmap_weight: float, torch):
    valid = batch["owner"] & (batch["labels"] != IGNORE_INDEX)
    if not bool(torch.any(valid)):
        raise ValueError("Scorer v11 batch has no definite owner frames")
    if variant == "heatmap_aux":
        outputs = model.forward_outputs(
            batch["ptm"], batch["mfcc"], attention_mask=batch["attention"]
        )
        logits = outputs["class_logits"]
    else:
        outputs = None
        logits = model(batch["ptm"], batch["mfcc"], attention_mask=batch["attention"])
    main = torch.nn.functional.cross_entropy(logits[valid], batch["labels"][valid])
    auxiliary = torch.zeros((), dtype=main.dtype, device=main.device)
    if outputs is not None and heatmap_weight > 0.0:
        boundary_valid = valid & batch["boundary_valid"]
        auxiliary = candidate_boundary_heatmap_loss(
            start_logits=outputs["start_boundary_logits"],
            end_logits=outputs["end_boundary_logits"],
            start_targets=batch["start_heatmap"],
            end_targets=batch["end_heatmap"],
            valid_mask=boundary_valid,
        )
    return main + heatmap_weight * auxiliary, main, auxiliary, logits, valid


def _evaluate(
    model,
    rows: Sequence[dict[str, Any]],
    *,
    cache: SourceArrayCache,
    max_padded_frames: int,
    variant: str,
    heatmap_weight: float,
    torch,
    device,
) -> dict[str, Any]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    loss_sum = 0.0
    batches = 0
    model.eval()
    with torch.no_grad():
        for rows_batch in _pack_batches(rows, max_padded_frames=max_padded_frames):
            batch = _collate([load_candidate_window(row, cache) for row in rows_batch], torch, device)
            loss, _main, _aux, logits, valid = _loss(
                model,
                batch,
                variant=variant,
                heatmap_weight=heatmap_weight,
                torch=torch,
            )
            target = batch["labels"][valid].detach().cpu().numpy()
            predicted = torch.argmax(logits, dim=-1)[valid].detach().cpu().numpy()
            for truth, prediction in zip(target, predicted, strict=True):
                confusion[int(truth), int(prediction)] += 1
            loss_sum += float(loss.detach().cpu())
            batches += 1
            del batch, logits, valid, loss
    total = int(confusion.sum())
    inside_total = int(confusion[1].sum())
    outside_total = int(confusion[0].sum())
    return {
        "loss": loss_sum / max(batches, 1),
        "frame_accuracy": float(np.trace(confusion) / max(total, 1)),
        "inside_candidate_recall": float(confusion[1, 1] / max(inside_total, 1)),
        "outside_candidate_recall": float(confusion[0, 0] / max(outside_total, 1)),
        "prediction_drop_truth_keep_frames": int(confusion[1, 0]),
        "confusion_truth_by_prediction": confusion.tolist(),
        "definite_owner_frame_count": total,
    }


def _release_cuda(torch, device) -> dict[str, Any]:
    before = {"allocated_bytes": 0, "reserved_bytes": 0}
    after = dict(before)
    if device.type == "cuda":
        before = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        after = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        }
    return {"before_cleanup": before, "after_cleanup": after}


def _memory_snapshot(device, *, stage: str) -> dict[str, Any]:
    import torch

    snapshot = runtime_memory_snapshot(require_shared_vram=device.type == "cuda")
    if device.type != "cuda":
        snapshot = {
            key: value
            for key, value in snapshot.items()
            if not key.startswith("shared_vram")
        }
        snapshot.update(
            shared_vram_mb=0.0,
            shared_vram_monitor="not_applicable_cpu_stage",
        )
    else:
        snapshot.update(
            cuda_allocated_mb=round(torch.cuda.memory_allocated(device) / 2**20, 3),
            cuda_reserved_mb=round(torch.cuda.memory_reserved(device) / 2**20, 3),
            cuda_max_allocated_mb=round(
                torch.cuda.max_memory_allocated(device) / 2**20, 3
            ),
            cuda_max_reserved_mb=round(
                torch.cuda.max_memory_reserved(device) / 2**20, 3
            ),
        )
    if snapshot["physical_ram_used_mb"] > snapshot["physical_ram_budget_mb"]:
        raise MemoryError("Scorer v11 exceeded the 95% physical RAM budget")
    if device.type == "cuda" and float(snapshot.get("shared_vram_mb") or 0.0) > 0.0:
        raise MemoryError(
            "Scorer v11 shared VRAM spill is a soft OOM: "
            f"shared_vram_mb={float(snapshot.get('shared_vram_mb') or 0.0):.3f}"
        )
    snapshot["stage"] = stage
    return snapshot


def run(args: argparse.Namespace) -> dict[str, Any]:
    import psutil
    import torch

    if args.variant not in {"baseline", "heatmap_aux"}:
        raise ValueError("Scorer v11 variant must be baseline or heatmap_aux")
    capacity_profile = str(args.capacity_profile)
    if capacity_profile not in CANDIDATE_ISLAND_SCORER_V11_CAPACITY_PROFILES:
        raise ValueError(f"unknown Scorer v11 capacity profile: {capacity_profile!r}")
    capacity = CANDIDATE_ISLAND_SCORER_V11_CAPACITY_PROFILES[capacity_profile]
    if (
        args.variant == "heatmap_aux"
        and capacity_profile != CANDIDATE_ISLAND_SCORER_V11_FULL_CAPACITY_PROFILE
    ):
        raise ValueError(
            "Scorer v11 heatmap A/B is only defined after the full-capacity baseline; "
            "do not mix heatmap and capacity axes"
        )
    if int(args.max_padded_frames) > int(capacity["max_padded_frames"]):
        raise ValueError(
            "Scorer v11 max_padded_frames exceeds the verified no-spill capacity "
            f"for {capacity_profile}: {capacity['max_padded_frames']}"
        )
    if float(args.class_weight_outside) != 1.0 or float(args.class_weight_inside) != 1.0:
        raise ValueError("Scorer v11 neutral baseline requires class weights 1/1")
    if args.variant == "baseline" and float(args.heatmap_weight) != 0.0:
        raise ValueError("baseline Scorer v11 forbids auxiliary loss")
    if args.variant == "heatmap_aux" and float(args.heatmap_weight) <= 0.0:
        raise ValueError("heatmap_aux requires a positive pre-registered weight")
    if str(args.device).lower() == "cpu" and not bool(args.smoke):
        raise ValueError("CPU is allowed only for Scorer v11 plumbing smoke, never full training")
    if str(args.device).lower().startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but unavailable; CPU fallback is forbidden")

    dataset_manifest = Path(args.dataset_manifest).resolve()
    gate_path = Path(args.feature_cache_gate).resolve()
    rows, gate, provenance = validate_training_inputs(
        dataset_manifest=dataset_manifest, feature_cache_gate=gate_path
    )
    by_partition = {
        partition: [row for row in rows if row["partition"] == partition]
        for partition in sorted(PARTITIONS)
    }
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device)
    process = psutil.Process()
    physical_ram = int(psutil.virtual_memory().total)
    physical_ram_budget = int(physical_ram * 0.95)
    ram_budget: dict[str, Any] = {
        "physical_total_bytes": physical_ram,
        "physical_budget_bytes": physical_ram_budget,
        "physical_fraction_limit": 0.95,
        "process_rss_before_bytes": int(process.memory_info().rss),
    }
    if ram_budget["process_rss_before_bytes"] > physical_ram_budget:
        raise MemoryError("Scorer v11 process already exceeds the 95% physical RAM budget")
    gpu_budget: dict[str, Any] = {"device": str(device), "physical_fraction_limit": 0.95}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        total_memory = int(properties.total_memory)
        budget = int(total_memory * 0.95)
        torch.cuda.set_per_process_memory_fraction(0.95, device=device)
        gpu_budget.update(
            {
                "physical_total_bytes": total_memory,
                "physical_budget_bytes": budget,
                "shared_vram_counted_as_available": False,
                "shared_vram_spill_policy": "soft_oom_abort",
            }
        )
        gc.collect()
        torch.cuda.empty_cache()
        shared_vram_baseline = reset_shared_vram_baseline(required=True)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        shared_vram_baseline = {
            "shared_vram_mb": 0.0,
            "shared_vram_monitor": "not_applicable_cpu_stage",
        }
    memory_snapshots: list[dict[str, Any]] = [
        {**shared_vram_baseline, "stage": "execution_baseline"}
    ]

    cache = SourceArrayCache(max_sources=int(args.source_cache_size))
    normalization = compute_mfcc_normalization(by_partition["train"], cache)
    config = _model_config(args, normalization)
    if args.variant == "heatmap_aux":
        model = CandidateIslandHeatmapScorerNetwork(**config)
        schema = CANDIDATE_ISLAND_SCORER_V11_HEATMAP_SCHEMA
    else:
        model = CandidateIslandScorerNetwork(**config)
        schema = str(capacity["schema"])
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    training_steps = 0
    training_losses: list[float] = []
    model.train()
    for _epoch in range(int(args.epochs)):
        shuffled = list(by_partition["train"])
        random.shuffle(shuffled)
        for rows_batch in _pack_batches(shuffled, max_padded_frames=int(args.max_padded_frames)):
            batch = _collate([load_candidate_window(row, cache) for row in rows_batch], torch, device)
            if int(process.memory_info().rss) > physical_ram_budget:
                raise MemoryError("Scorer v11 exceeded the 95% physical RAM budget")
            if device.type == "cuda" and int(torch.cuda.memory_reserved(device)) > int(
                gpu_budget["physical_budget_bytes"]
            ):
                raise MemoryError("Scorer v11 exceeded the 95% physical VRAM budget")
            memory_snapshots.append(
                _memory_snapshot(device, stage=f"train_step_{training_steps + 1}")
            )
            optimizer.zero_grad(set_to_none=True)
            loss, _main, _aux, _logits, _valid = _loss(
                model,
                batch,
                variant=args.variant,
                heatmap_weight=float(args.heatmap_weight),
                torch=torch,
            )
            loss.backward()
            if float(args.gradient_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip_norm))
            optimizer.step()
            training_losses.append(float(loss.detach().cpu()))
            training_steps += 1
            del batch, loss, _main, _aux, _logits, _valid
            if int(args.max_steps) > 0 and training_steps >= int(args.max_steps):
                break
        if int(args.max_steps) > 0 and training_steps >= int(args.max_steps):
            break
    metrics = {
        partition: _evaluate(
            model,
            by_partition[partition],
            cache=cache,
            max_padded_frames=int(args.max_padded_frames),
            variant=args.variant,
            heatmap_weight=float(args.heatmap_weight),
            torch=torch,
            device=device,
        )
        for partition in ("val", "test")
    }
    memory_snapshots.append(_memory_snapshot(device, stage="evaluation_complete"))
    numeric_gate_pass = (
        not bool(args.smoke)
        and metrics["val"]["inside_candidate_recall"] >= 0.95
        and metrics["test"]["inside_candidate_recall"] >= 0.95
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / (
        f"scorer-v11-{capacity_profile}-{args.variant}.pt"
    )
    metadata = {
        "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
        "dataset_manifest": _display(dataset_manifest),
        "dataset_manifest_sha256": _sha256(dataset_manifest),
        "feature_manifest": str(gate["signed_feature_manifest"]),
        "signed_feature_manifest_sha256": provenance["signed_feature_manifest_sha256"],
        "canonical_sources_sha256": provenance["canonical_sources_sha256"],
        "feature_cache_gate": _display(gate_path),
        "feature_cache_gate_sha256": _sha256(gate_path),
        "feature_config_sha256": provenance["feature_config_sha256"],
        "training_initialization": "random",
        "training_variant": args.variant,
        "capacity_profile": capacity_profile,
        "capacity_ab_axis": "ptm_adapter_and_temporal_capacity_profile",
        "class_weights": {"outside_candidate": 1.0, "inside_candidate": 1.0},
        "heatmap_auxiliary_weight": float(args.heatmap_weight),
        "numeric_gate_maximum_requirement": 0.95,
        "numeric_gate_pass": numeric_gate_pass,
        "manual_zero_clipping_gate": "pending",
        "manual_zero_true_speech_deletion_gate": "pending",
        "promotion_allowed": False,
        "smoke": bool(args.smoke),
    }
    payload = build_speech_island_scorer_checkpoint(
        model=model,
        model_config=config,
        normalization=normalization,
        metadata=metadata,
        schema=schema,
    )
    torch.save(payload, checkpoint_path)
    del payload, optimizer, model
    cache.clear()
    del cache
    gc.collect()
    lifecycle = _release_cuda(torch, device)
    ram_budget["process_rss_after_cleanup_bytes"] = int(process.memory_info().rss)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "variant": args.variant,
        "capacity_profile": capacity_profile,
        "capacity_contract": dict(capacity),
        "smoke": bool(args.smoke),
        "checkpoint": _display(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset_manifest": _display(dataset_manifest),
        "dataset_manifest_sha256": _sha256(dataset_manifest),
        "feature_cache_gate": _display(gate_path),
        "feature_cache_gate_sha256": _sha256(gate_path),
        "training_steps": training_steps,
        "training_loss_mean": float(np.mean(training_losses)) if training_losses else None,
        "metrics": metrics,
        "numeric_gate_maximum_requirement": 0.95,
        "numeric_gate_pass": numeric_gate_pass,
        "manual_gate_status": "pending",
        "promotion_allowed": False,
        "gpu_budget": gpu_budget,
        "ram_budget": ram_budget,
        "stage_lifecycle": lifecycle,
        "memory_snapshots": memory_snapshots,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--feature-cache-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=("baseline", "heatmap_aux"), default="baseline")
    parser.add_argument(
        "--capacity-profile",
        choices=(
            CANDIDATE_ISLAND_SCORER_V11_FULL_CAPACITY_PROFILE,
            CANDIDATE_ISLAND_SCORER_V11_COMPACT_CAPACITY_PROFILE,
        ),
        default=CANDIDATE_ISLAND_SCORER_V11_FULL_CAPACITY_PROFILE,
    )
    parser.add_argument("--heatmap-weight", type=float, default=0.0)
    parser.add_argument("--class-weight-outside", type=float, default=1.0)
    parser.add_argument("--class-weight-inside", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=117)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-padded-frames", type=int, default=2000)
    parser.add_argument("--source-cache-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
