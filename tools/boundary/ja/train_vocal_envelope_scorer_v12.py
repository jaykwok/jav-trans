#!/usr/bin/env python3
"""Train one Scorer v12 CE/run-balanced/adjacency arm."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.ja.vocal_envelope_training import (  # noqa: E402
    compute_vocal_envelope_losses,
    source_metrics,
)
from boundary.ja.vocal_envelope_v12 import (  # noqa: E402
    VOCAL_ENVELOPE_SCORER_V12_CANONICAL_LABEL_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX,
    VOCAL_ENVELOPE_SCORER_V12_SCHEMA,
    VOCAL_ENVELOPE_SCORER_V12_TRAINING_ROW_SCHEMA,
    VocalEnvelopeScorerV12Network,
    build_vocal_envelope_scorer_v12_checkpoint,
    vocal_envelope_v12_model_config,
)
from pipeline.memory_safety import reset_shared_vram_baseline, runtime_memory_snapshot  # noqa: E402

CONTRACT_ID = "boundary_acoustic_binary_v12"
GATE_SCHEMA = "vocal_envelope_scorer_v12_feature_cache_gate_v1"
SUMMARY_SCHEMA = "vocal_envelope_scorer_v12_training_summary_v1"
PROGRESS_SCHEMA = "vocal_envelope_scorer_v12_training_progress_v1"
SOURCE_PREDICTION_SCHEMA = "vocal_envelope_scorer_v12_source_prediction_v1"
SEED = 117
HARD_RECALL = 0.95


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resolve_gate_file(gate_path: Path, value: Any) -> Path:
    raw = Path(str(value or ""))
    resolved = raw.resolve() if raw.is_absolute() else (gate_path.parent / raw).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _atomic_torch_save(torch, path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _memory(torch, device, *, stage: str) -> dict[str, Any]:
    snapshot = runtime_memory_snapshot(require_shared_vram=device.type == "cuda")
    if device.type == "cuda":
        snapshot.update(
            cuda_allocated_mb=round(torch.cuda.memory_allocated(device) / 2**20, 3),
            cuda_reserved_mb=round(torch.cuda.memory_reserved(device) / 2**20, 3),
            cuda_max_allocated_mb=round(torch.cuda.max_memory_allocated(device) / 2**20, 3),
            cuda_max_reserved_mb=round(torch.cuda.max_memory_reserved(device) / 2**20, 3),
        )
    else:
        snapshot.update(shared_vram_mb=0.0, shared_vram_monitor="not_applicable_cpu_stage")
    if float(snapshot.get("physical_ram_used_mb") or 0.0) > float(snapshot.get("physical_ram_budget_mb") or math.inf):
        raise MemoryError("Scorer v12 exceeded the 95% physical RAM budget")
    if device.type == "cuda" and float(snapshot.get("shared_vram_mb") or 0.0) > 0.0:
        raise MemoryError("Scorer v12 shared VRAM spill is a soft OOM")
    snapshot["stage"] = stage
    return snapshot


class SourceArrayCache:
    def __init__(self, limit: int = 3) -> None:
        self.limit = max(1, int(limit))
        self.values: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source_id = str(row["source_id"])
        if source_id in self.values:
            value = self.values.pop(source_id)
            self.values[source_id] = value
            return value
        feature = Path(str(row["feature_path"]))
        label = Path(str(row["label_path"]))
        if _sha256(feature) != str(row["feature_sha256"]) or _sha256(label) != str(row["label_sha256"]):
            raise ValueError(f"Scorer v12 cached source SHA mismatch: {source_id}")
        with np.load(feature) as payload:
            ptm = np.asarray(payload["ptm"], dtype=np.float32)
            mfcc = np.asarray(payload["mfcc"], dtype=np.float32)
        with np.load(label) as payload:
            labels = np.asarray(payload["labels"], dtype=np.int64)
        expected = int(row["source_frame_count"])
        if ptm.shape != (expected, 2048) or mfcc.shape != (expected, 40) or labels.shape != (expected,):
            raise ValueError(f"Scorer v12 cached source geometry mismatch: {source_id}")
        value = (ptm, mfcc, labels)
        self.values[source_id] = value
        while len(self.values) > self.limit:
            self.values.popitem(last=False)
        return value

    def clear(self) -> None:
        self.values.clear()


def _validate(gate_path: Path, rows_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gate = _read_json(gate_path)
    rows = _read_jsonl(rows_path)
    if (
        gate.get("schema") != GATE_SCHEMA
        or gate.get("boundary_serialization_contract_id") != CONTRACT_ID
        or gate.get("status") != "approved_for_training"
        or gate.get("training_allowed") is not True
    ):
        raise ValueError("Scorer v12 trainer requires an approved v12 feature gate")
    referenced = {
        "canonical_sources": "canonical_sources_sha256",
        "raw_feature_manifest": "raw_feature_manifest_sha256",
        "signed_feature_manifest": "signed_feature_manifest_sha256",
        "training_windows": "training_windows_sha256",
    }
    resolved_references: dict[str, Path] = {}
    for field, sha_field in referenced.items():
        resolved = _resolve_gate_file(gate_path, gate.get(field))
        if _sha256(resolved) != str(gate.get(sha_field) or ""):
            raise ValueError(f"Scorer v12 gate {field} SHA mismatch")
        resolved_references[field] = resolved
    if resolved_references["training_windows"] != rows_path.resolve():
        raise ValueError("Scorer v12 gate points to another training manifest")
    seen_rows: set[str] = set()
    source_partition: dict[str, str] = {}
    core_source: dict[str, str] = {}
    source_rows: dict[str, list[dict[str, Any]]] = {}
    source_identity: dict[str, tuple[Any, ...]] = {}
    partition_counts = Counter()
    for row in rows:
        if row.get("schema") != VOCAL_ENVELOPE_SCORER_V12_TRAINING_ROW_SCHEMA:
            raise ValueError("Scorer v12 trainer rejects legacy/v11 rows")
        if row.get("boundary_serialization_contract_id") != CONTRACT_ID or row.get("canonical_label_schema") != VOCAL_ENVELOPE_SCORER_V12_CANONICAL_LABEL_SCHEMA:
            raise ValueError("Scorer v12 row contract mismatch")
        row_id, source_id, partition = str(row.get("row_id") or ""), str(row.get("source_id") or ""), str(row.get("partition") or "")
        if not row_id or row_id in seen_rows or not source_id or partition not in {"train", "val", "test"}:
            raise ValueError("Scorer v12 row identity/partition is invalid")
        seen_rows.add(row_id)
        previous = source_partition.setdefault(source_id, partition)
        if previous != partition:
            raise ValueError(f"Scorer v12 source crosses partitions: {source_id}")
        cores = [str(value) for value in list(row.get("core_ids") or ())]
        if len(cores) != 1:
            raise ValueError(f"Scorer v12 row needs one core: {row_id}")
        core_previous = core_source.setdefault(cores[0], source_id)
        if core_previous != source_id:
            raise ValueError(f"Scorer v12 core is reused by multiple sources: {cores[0]}")
        if row.get("synthetic_composite") is True and partition != "train":
            raise ValueError("Scorer v12 synthetic rows are train-only")
        if row.get("canonical_sources_sha256") != gate.get("canonical_sources_sha256") or row.get("raw_feature_manifest_sha256") != gate.get("raw_feature_manifest_sha256"):
            raise ValueError(f"Scorer v12 row source manifest binding mismatch: {row_id}")
        if row.get("signed_feature_manifest_sha256") != gate.get("signed_feature_manifest_sha256"):
            raise ValueError(f"Scorer v12 row signed manifest binding mismatch: {row_id}")
        frame_count = int(row.get("source_frame_count") or 0)
        window_start, window_end = int(row.get("window_start_frame") or 0), int(row.get("window_end_frame") or 0)
        owner_start, owner_end = int(row.get("owner_start_frame") or 0), int(row.get("owner_end_frame") or 0)
        local_start, local_end = int(row.get("owner_local_start") or 0), int(row.get("owner_local_end") or 0)
        if not (0 <= window_start <= owner_start < owner_end <= window_end <= frame_count):
            raise ValueError(f"Scorer v12 row window/owner geometry is invalid: {row_id}")
        if local_start != owner_start - window_start or local_end != owner_end - window_start:
            raise ValueError(f"Scorer v12 row local ownership mismatch: {row_id}")
        identity = (
            partition,
            tuple(cores),
            frame_count,
            str(row.get("feature_path") or ""),
            str(row.get("feature_sha256") or ""),
            str(row.get("label_path") or ""),
            str(row.get("label_sha256") or ""),
        )
        if source_id in source_identity and source_identity[source_id] != identity:
            raise ValueError(f"Scorer v12 source identity changes across windows: {source_id}")
        source_identity[source_id] = identity
        source_rows.setdefault(source_id, []).append(row)
        partition_counts[partition] += 1
    if any(partition_counts[name] <= 0 for name in ("train", "val", "test")):
        raise ValueError("Scorer v12 requires train/val/test rows")
    if int(gate.get("window_count") or 0) != len(rows):
        raise ValueError("Scorer v12 gate window count mismatch")
    if int(gate.get("source_count") or 0) != len(source_rows):
        raise ValueError("Scorer v12 gate source count mismatch")
    for source_id, group in source_rows.items():
        cursor = 0
        for row in sorted(group, key=lambda item: int(item["owner_start_frame"])):
            start, end = int(row["owner_start_frame"]), int(row["owner_end_frame"])
            if start != cursor:
                raise ValueError(f"Scorer v12 owner frames must cover each source exactly once: {source_id}")
            cursor = end
        if cursor != int(group[0]["source_frame_count"]):
            raise ValueError(f"Scorer v12 owner frames miss source tail: {source_id}")
    return gate, rows


def _window(row: Mapping[str, Any], cache: SourceArrayCache) -> dict[str, Any]:
    ptm, mfcc, labels = cache.get(row)
    start, end = int(row["window_start_frame"]), int(row["window_end_frame"])
    owner_start, owner_end = int(row["owner_local_start"]), int(row["owner_local_end"])
    return {
        "ptm": ptm[start:end], "mfcc": mfcc[start:end], "labels": labels[start:end],
        "owner_start": owner_start, "owner_end": owner_end,
    }


def _pack(rows: list[dict[str, Any]], *, max_padded_frames: int, max_rows: int) -> list[list[dict[str, Any]]]:
    if max_padded_frames <= 0 or max_rows <= 0:
        raise ValueError("Scorer v12 batch limits must be positive")
    output: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    longest = 0
    for row in rows:
        length = int(row["window_end_frame"]) - int(row["window_start_frame"])
        if length <= 0 or length > max_padded_frames:
            raise ValueError(f"Scorer v12 row exceeds batch frame budget: {row.get('row_id')}")
        proposed_longest = max(longest, length)
        if current and (len(current) >= max_rows or proposed_longest * (len(current) + 1) > max_padded_frames):
            output.append(current)
            current, longest = [], 0
        current.append(row)
        longest = max(longest, length)
    if current:
        output.append(current)
    return output


def _collate(items: list[dict[str, Any]], torch, device) -> dict[str, Any]:
    frames = max(len(item["labels"]) for item in items)
    batch = len(items)
    ptm = torch.zeros((batch, frames, 2048), dtype=torch.float32, device=device)
    mfcc = torch.zeros((batch, frames, 40), dtype=torch.float32, device=device)
    labels = torch.full((batch, frames), VOCAL_ENVELOPE_SCORER_V12_IGNORE_INDEX, dtype=torch.long, device=device)
    attention = torch.zeros((batch, frames), dtype=torch.bool, device=device)
    owner = torch.zeros((batch, frames), dtype=torch.bool, device=device)
    for index, item in enumerate(items):
        length = len(item["labels"])
        ptm[index, :length] = torch.from_numpy(item["ptm"]).to(device)
        mfcc[index, :length] = torch.from_numpy(item["mfcc"]).to(device)
        labels[index, :length] = torch.from_numpy(item["labels"]).to(device)
        attention[index, :length] = True
        owner[index, item["owner_start"] : item["owner_end"]] = True
    return {"ptm": ptm, "mfcc": mfcc, "labels": labels, "attention": attention, "owner": owner}


def _normalization(train_rows: list[dict[str, Any]], cache: SourceArrayCache) -> dict[str, list[float]]:
    seen: set[str] = set()
    count = 0
    total = np.zeros(40, dtype=np.float64)
    total_sq = np.zeros(40, dtype=np.float64)
    for row in train_rows:
        source_id = str(row["source_id"])
        if source_id in seen:
            continue
        seen.add(source_id)
        _, mfcc, _ = cache.get(row)
        total += mfcc.sum(axis=0, dtype=np.float64)
        total_sq += np.square(mfcc, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += len(mfcc)
    if count <= 0:
        raise ValueError("Scorer v12 normalization has no train frames")
    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 1e-6)
    return {"mfcc_mean": mean.tolist(), "mfcc_std": np.sqrt(variance).tolist()}


def _runs(values: np.ndarray, target: int) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    start = None
    for index in range(len(values) + 1):
        active = index < len(values) and int(values[index]) == target
        if active and start is None:
            start = index
        elif not active and start is not None:
            output.append((start, index))
            start = None
    return output


def _evaluate(model, rows: list[dict[str, Any]], cache: SourceArrayCache, torch, device, *, arm: str, max_padded_frames: int, max_rows: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row["source_id"]), []).append(row)
    prediction_buffers = {source_id: np.full(int(group[0]["source_frame_count"]), -1, dtype=np.int64) for source_id, group in by_source.items()}
    loss_sums = Counter()
    batch_count = 0
    ordered = sorted(rows, key=lambda row: int(row["window_end_frame"]) - int(row["window_start_frame"]), reverse=True)
    with torch.inference_mode():
        for batch_rows in _pack(ordered, max_padded_frames=max_padded_frames, max_rows=max_rows):
            items = [_window(row, cache) for row in batch_rows]
            batch = _collate(items, torch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(batch["ptm"], batch["mfcc"], attention_mask=batch["attention"])
                losses = compute_vocal_envelope_losses(logits, batch["labels"], batch["owner"], arm=arm)
            predicted = logits.argmax(dim=-1).detach().cpu().numpy()
            for index, row in enumerate(batch_rows):
                start, end = int(row["owner_start_frame"]), int(row["owner_end_frame"])
                local_start, local_end = int(row["owner_local_start"]), int(row["owner_local_end"])
                prediction_buffers[str(row["source_id"])][start:end] = predicted[index, local_start:local_end]
            for key in ("main_loss", "run_loss", "adjacency_loss"):
                loss_sums[key] += float(losses[key].detach().cpu())
            batch_count += 1
            del batch, logits, losses, predicted, items
    predictions: list[dict[str, Any]] = []
    totals = Counter()
    macro = Counter()
    all_vocal_total = all_vocal_success = all_nonvocal_total = all_nonvocal_success = 0
    vocal_runs = continuous_runs = complete_run_deletions = 0
    internal_holes = 0
    prediction_run_count = 0
    for source_id, group in by_source.items():
        _, _, labels = cache.get(group[0])
        predicted = prediction_buffers[source_id]
        if np.any(predicted < 0):
            raise RuntimeError(f"Scorer v12 evaluation missed source frames: {source_id}")
        definite = np.isin(labels, [0, 1])
        vocal = labels == 1
        nonvocal = labels == 0
        totals["vocal_truth"] += int(vocal.sum())
        totals["vocal_hit"] += int(np.sum(vocal & (predicted == 1)))
        totals["nonvocal_truth"] += int(nonvocal.sum())
        totals["nonvocal_hit"] += int(np.sum(nonvocal & (predicted == 0)))
        metrics = source_metrics(labels, predicted)
        if math.isfinite(metrics.vocal_recall): macro["vocal_recall_sum"] += metrics.vocal_recall; macro["vocal_source_count"] += 1
        if math.isfinite(metrics.non_vocal_recall): macro["nonvocal_recall_sum"] += metrics.non_vocal_recall; macro["nonvocal_source_count"] += 1
        if metrics.all_vocal_keep is not None:
            all_vocal_total += 1; all_vocal_success += int(metrics.all_vocal_keep)
        if metrics.all_non_vocal_full_drop is not None:
            all_nonvocal_total += 1; all_nonvocal_success += int(metrics.all_non_vocal_full_drop)
        source_vocal_runs = len(_runs(labels, 1))
        vocal_runs += source_vocal_runs
        continuous_runs += int(round(metrics.vocal_continuity * source_vocal_runs)) if source_vocal_runs else 0
        complete_run_deletions += metrics.complete_vocal_run_deletion_count
        internal_holes += metrics.internal_hole_count
        prediction_run_count += metrics.prediction_run_count
        predictions.append({
            "schema": SOURCE_PREDICTION_SCHEMA, "boundary_serialization_contract_id": CONTRACT_ID,
            "source_id": source_id, "partition": group[0]["partition"],
            "source_kind": group[0].get("source_kind", ""),
            "frame_count": len(labels), "frame_hop_s": 0.02,
            "labels": labels.tolist(), "predicted_labels": predicted.tolist(),
            "vocal_recall": metrics.vocal_recall, "non_vocal_recall": metrics.non_vocal_recall,
            "vocal_continuity": metrics.vocal_continuity,
            "internal_hole_count": metrics.internal_hole_count,
            "vocal_prediction_run_count": metrics.prediction_run_count,
            "complete_vocal_run_deletion_count": metrics.complete_vocal_run_deletion_count,
            "all_vocal_keep": metrics.all_vocal_keep,
            "all_non_vocal_full_drop": metrics.all_non_vocal_full_drop,
        })
    result = {
        "main_loss": loss_sums["main_loss"] / max(batch_count, 1),
        "run_loss": loss_sums["run_loss"] / max(batch_count, 1),
        "adjacency_loss": loss_sums["adjacency_loss"] / max(batch_count, 1),
        "vocal_recall": totals["vocal_hit"] / max(totals["vocal_truth"], 1),
        "non_vocal_recall": totals["nonvocal_hit"] / max(totals["nonvocal_truth"], 1),
        "vocal_source_macro_recall": macro["vocal_recall_sum"] / max(macro["vocal_source_count"], 1),
        "non_vocal_source_macro_recall": macro["nonvocal_recall_sum"] / max(macro["nonvocal_source_count"], 1),
        "vocal_continuity": continuous_runs / max(vocal_runs, 1),
        "vocal_truth_run_count": vocal_runs,
        "complete_vocal_run_deletion_count": complete_run_deletions,
        "internal_hole_count": internal_holes,
        "vocal_prediction_run_count": prediction_run_count,
        "all_vocal_source_count": all_vocal_total,
        "all_vocal_source_keep_recall": all_vocal_success / all_vocal_total if all_vocal_total else None,
        "all_nonvocal_source_count": all_nonvocal_total,
        "all_nonvocal_source_full_drop_recall": all_nonvocal_success / all_nonvocal_total if all_nonvocal_total else None,
        "source_count": len(by_source),
    }
    return result, predictions


def _selection(metrics: Mapping[str, Any]) -> tuple[int, float, int, float]:
    vocal = float(metrics["vocal_recall"])
    continuity = float(metrics["vocal_continuity"])
    zero_delete = int(metrics["complete_vocal_run_deletion_count"]) == 0
    safety = vocal >= HARD_RECALL and continuity >= HARD_RECALL and zero_delete
    all_vocal = metrics.get("all_vocal_source_keep_recall")
    safety = safety and (all_vocal is None or float(all_vocal) >= HARD_RECALL)
    controls = [float(metrics["non_vocal_recall"]), continuity]
    for key in ("all_vocal_source_keep_recall", "all_nonvocal_source_full_drop_recall"):
        if metrics.get(key) is not None:
            controls.append(float(metrics[key]))
    return int(safety), min(controls), -int(metrics["internal_hole_count"]), vocal


def _numeric_gate(metrics: Mapping[str, Any]) -> bool:
    if float(metrics["vocal_recall"]) < HARD_RECALL:
        return False
    if float(metrics["non_vocal_recall"]) < HARD_RECALL:
        return False
    if float(metrics["vocal_continuity"]) < HARD_RECALL:
        return False
    if int(metrics["complete_vocal_run_deletion_count"]) != 0:
        return False
    for key in ("all_vocal_source_keep_recall", "all_nonvocal_source_full_drop_recall"):
        value = metrics.get(key)
        if value is not None and float(value) < HARD_RECALL:
            return False
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Scorer v12 training requested CUDA but CUDA is unavailable")
    applied_ratio = apply_vram_safety_cap(0.95) if device.type == "cuda" else None
    gate_path, rows_path = Path(args.feature_gate).resolve(), Path(args.training_windows).resolve()
    gate, rows = _validate(gate_path, rows_path)
    by_partition = {name: [row for row in rows if row["partition"] == name] for name in ("train", "val", "test")}
    cache = SourceArrayCache(args.cache_sources)
    normalization = _normalization(by_partition["train"], cache)
    config = vocal_envelope_v12_model_config(**normalization)
    model = VocalEnvelopeScorerV12Network(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Scorer v12 training output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    checkpoint_path = output / f"scorer-v12-arm-{args.arm.upper()}.pt"
    predictions_path = output / "val_source_predictions.jsonl"
    train_batches_template = _pack(by_partition["train"], max_padded_frames=args.max_batch_frames, max_rows=args.max_batch_rows)
    total_steps = min(args.max_steps, len(train_batches_template) * args.epochs) if args.max_steps > 0 else len(train_batches_template) * args.epochs
    step = 0; best_key = (-1, -1.0, -10**9, -1.0); best_epoch = 0; best_metrics = None; patience = 0
    memory_log: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        reset_shared_vram_baseline(required=True)
        torch.cuda.reset_peak_memory_stats(device)
    _write_json(progress_path, {"schema": PROGRESS_SCHEMA, "status": "running", "arm": args.arm.upper(), "seed": SEED, "step": 0, "total": total_steps, "main_loss": None, "run_loss": None, "adjacency_loss": None, "val_vocal_recall": None, "val_non_vocal_recall": None, "vocal_continuity": None, "internal_hole_count": None, "vocal_prediction_run_count": None, "complete_vocal_run_deletion_count": None, "all_vocal_source_keep_recall": None, "all_nonvocal_source_full_drop_recall": None})
    try:
        for epoch in range(1, args.epochs + 1):
            shuffled = list(by_partition["train"])
            random.Random(SEED + epoch).shuffle(shuffled)
            batches = _pack(shuffled, max_padded_frames=args.max_batch_frames, max_rows=args.max_batch_rows)
            model.train()
            epoch_losses = Counter(); epoch_batches = 0
            for batch_rows in batches:
                if args.max_steps > 0 and step >= args.max_steps:
                    break
                items = [_window(row, cache) for row in batch_rows]
                batch = _collate(items, torch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(batch["ptm"], batch["mfcc"], attention_mask=batch["attention"])
                    losses = compute_vocal_envelope_losses(logits, batch["labels"], batch["owner"], arm=args.arm)
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                step += 1; epoch_batches += 1
                for key in ("main_loss", "run_loss", "adjacency_loss"):
                    epoch_losses[key] += float(losses[key].detach().cpu())
                if step == 1 or step % args.progress_every == 0 or step == total_steps:
                    memory = _memory(torch, device, stage=f"train-step-{step}")
                    memory_log.append(memory)
                    payload = {"schema": PROGRESS_SCHEMA, "status": "running", "arm": args.arm.upper(), "seed": SEED, "epoch": epoch, "step": step, "total": total_steps, "main_loss": epoch_losses["main_loss"] / epoch_batches, "run_loss": epoch_losses["run_loss"] / epoch_batches, "adjacency_loss": epoch_losses["adjacency_loss"] / epoch_batches, "val_vocal_recall": None if best_metrics is None else best_metrics["vocal_recall"], "val_non_vocal_recall": None if best_metrics is None else best_metrics["non_vocal_recall"], "vocal_continuity": None if best_metrics is None else best_metrics["vocal_continuity"], "internal_hole_count": None if best_metrics is None else best_metrics["internal_hole_count"], "vocal_prediction_run_count": None if best_metrics is None else best_metrics["vocal_prediction_run_count"], "complete_vocal_run_deletion_count": None if best_metrics is None else best_metrics["complete_vocal_run_deletion_count"], "all_vocal_source_keep_recall": None if best_metrics is None else best_metrics["all_vocal_source_keep_recall"], "all_nonvocal_source_full_drop_recall": None if best_metrics is None else best_metrics["all_nonvocal_source_full_drop_recall"], "memory": memory, "elapsed_s": round(time.perf_counter() - started, 3)}
                    _write_json(progress_path, payload)
                    print(f"step={step}/{total_steps} arm={args.arm.upper()} main_loss={payload['main_loss']:.6f} run_loss={payload['run_loss']:.6f} adjacency_loss={payload['adjacency_loss']:.6f}", flush=True)
                del batch, logits, losses, items
            val_metrics, val_predictions = _evaluate(model, by_partition["val"], cache, torch, device, arm=args.arm, max_padded_frames=args.max_batch_frames, max_rows=args.max_batch_rows)
            key = _selection(val_metrics)
            print(f"epoch={epoch} val_vocal_recall={val_metrics['vocal_recall']:.4f} val_non_vocal_recall={val_metrics['non_vocal_recall']:.4f} vocal_continuity={val_metrics['vocal_continuity']:.4f} internal_holes={val_metrics['internal_hole_count']}", flush=True)
            if key > best_key:
                best_key, best_epoch, best_metrics, patience = key, epoch, val_metrics, 0
                metadata = {
                    "ptm_repo_id": QWEN_ASR_17B_REPO_ID,
                    "dataset_manifest": str(rows_path), "dataset_manifest_sha256": _sha256(rows_path),
                    "feature_manifest": str(gate["signed_feature_manifest"]), "signed_feature_manifest_sha256": str(gate["signed_feature_manifest_sha256"]),
                    "canonical_sources_sha256": str(gate["canonical_sources_sha256"]),
                    "feature_cache_gate": str(gate_path), "feature_cache_gate_sha256": _sha256(gate_path),
                    "feature_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest(),
                    "training_initialization": "random", "seed": SEED, "loss_arm": args.arm.upper(),
                    "loss_contract": {"main_ce": 1.0, "run_balanced_ce": 0.5 if args.arm.upper() in {"B", "C"} else 0.0, "adjacency": 0.25 if args.arm.upper() == "C" else 0.0},
                    "best_epoch": epoch, "trained_steps": step, "val_metrics": val_metrics,
                }
                _atomic_torch_save(
                    torch,
                    checkpoint_path,
                    build_vocal_envelope_scorer_v12_checkpoint(
                        model=model,
                        model_config=config,
                        normalization=normalization,
                        metadata=metadata,
                    ),
                )
                with predictions_path.open("w", encoding="utf-8") as handle:
                    for row in val_predictions:
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            else:
                patience += 1
            _write_json(progress_path, {"schema": PROGRESS_SCHEMA, "status": "running", "arm": args.arm.upper(), "seed": SEED, "epoch": epoch, "step": step, "total": total_steps, "main_loss": epoch_losses["main_loss"] / max(epoch_batches, 1), "run_loss": epoch_losses["run_loss"] / max(epoch_batches, 1), "adjacency_loss": epoch_losses["adjacency_loss"] / max(epoch_batches, 1), "val_vocal_recall": val_metrics["vocal_recall"], "val_non_vocal_recall": val_metrics["non_vocal_recall"], "vocal_continuity": val_metrics["vocal_continuity"], "internal_hole_count": val_metrics["internal_hole_count"], "vocal_prediction_run_count": val_metrics["vocal_prediction_run_count"], "all_vocal_source_keep_recall": val_metrics["all_vocal_source_keep_recall"], "all_nonvocal_source_full_drop_recall": val_metrics["all_nonvocal_source_full_drop_recall"], "complete_vocal_run_deletion_count": val_metrics["complete_vocal_run_deletion_count"], "best_epoch": best_epoch, "early_stopping_patience": patience, "elapsed_s": round(time.perf_counter() - started, 3)})
            if patience >= args.early_stopping_patience or (args.max_steps > 0 and step >= args.max_steps):
                break
        if not checkpoint_path.is_file() or best_metrics is None:
            raise RuntimeError("Scorer v12 training did not produce a best checkpoint")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        test_metrics, test_predictions = _evaluate(model, by_partition["test"], cache, torch, device, arm=args.arm, max_padded_frames=args.max_batch_frames, max_rows=args.max_batch_rows)
        test_predictions_path = output / "test_source_predictions.jsonl"
        with test_predictions_path.open("w", encoding="utf-8") as handle:
            for row in test_predictions:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        numeric_gate = _numeric_gate(best_metrics)
        summary = {"schema": SUMMARY_SCHEMA, "boundary_serialization_contract_id": CONTRACT_ID, "scorer_schema": VOCAL_ENVELOPE_SCORER_V12_SCHEMA, "arm": args.arm.upper(), "seed": SEED, "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256(checkpoint_path), "best_epoch": best_epoch, "trained_steps": step, "val_metrics": best_metrics, "test_metrics": test_metrics, "numeric_gate_passed": numeric_gate, "manual_gate_required": True, "promotion_allowed": False, "test_ran_after_val_selection": True, "vram_safety_ratio": applied_ratio, "memory_log": memory_log, "val_source_predictions": str(predictions_path), "test_source_predictions": str(test_predictions_path)}
        _write_json(output / "summary.json", summary)
        _write_json(progress_path, {"schema": PROGRESS_SCHEMA, "status": "completed", "arm": args.arm.upper(), "seed": SEED, "step": step, "total": total_steps, "best_epoch": best_epoch, "val_vocal_recall": best_metrics["vocal_recall"], "val_non_vocal_recall": best_metrics["non_vocal_recall"], "vocal_continuity": best_metrics["vocal_continuity"], "internal_hole_count": best_metrics["internal_hole_count"], "vocal_prediction_run_count": best_metrics["vocal_prediction_run_count"], "complete_vocal_run_deletion_count": best_metrics["complete_vocal_run_deletion_count"], "all_vocal_source_keep_recall": best_metrics["all_vocal_source_keep_recall"], "all_nonvocal_source_full_drop_recall": best_metrics["all_nonvocal_source_full_drop_recall"], "numeric_gate_passed": numeric_gate, "summary": str(output / "summary.json"), "elapsed_s": round(time.perf_counter() - started, 3)})
        return summary
    except Exception as error:
        failed = _read_json(progress_path) if progress_path.is_file() else {}
        failed.update(
            {
                "schema": PROGRESS_SCHEMA,
                "status": "failed",
                "arm": args.arm.upper(),
                "seed": SEED,
                "step": step,
                "total": total_steps,
                "error": f"{type(error).__name__}: {error}",
                "elapsed_s": round(time.perf_counter() - started, 3),
            }
        )
        _write_json(progress_path, failed)
        raise
    finally:
        cache.clear(); optimizer.zero_grad(set_to_none=True)
        del optimizer, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass
            cleanup = _memory(torch, device, stage="post-release")
            if progress_path.is_file():
                progress = _read_json(progress_path); progress["post_release_memory"] = cleanup; _write_json(progress_path, progress)
            summary_path = output / "summary.json"
            if summary_path.is_file():
                completed_summary = _read_json(summary_path)
                completed_summary["post_release_memory"] = cleanup
                _write_json(summary_path, completed_summary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-gate", required=True)
    parser.add_argument("--training-windows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm", choices=("A", "B", "C", "a", "b", "c"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-batch-frames", type=int, default=1000)
    parser.add_argument("--max-batch-rows", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--cache-sources", type=int, default=3)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
