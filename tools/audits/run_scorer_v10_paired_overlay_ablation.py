#!/usr/bin/env python3
"""Run a diagnostic-only paired overlay ablation for Scorer v10."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from audio.loading import load_audio_16k_mono  # noqa: E402
from boundary.gpu_safety import apply_vram_safety_cap  # noqa: E402
from boundary.ja.features import FeatureConfig, build_ptm_feature_extractor, extract_mfcc  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    load_speech_island_scorer_checkpoint,
    score_binary_speech_class_probabilities,
)
from pipeline.memory_safety import reset_shared_vram_baseline, runtime_memory_snapshot  # noqa: E402
from tools.boundary.ja.build_feature_cache import (  # noqa: E402
    _combine_workflow_window_features,
    _extract_ptm_window_features,
    _workflow_window_starts,
)

SCHEMA = "speech_scorer_v10_paired_overlay_ablation_v1"
CONTRACT_ID = "boundary_acoustic_binary_v12"
SAMPLE_RATE = 16000
FRAME_HOP_S = 0.02


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_audio(path: Path) -> np.ndarray:
    audio, rate = load_audio_16k_mono(str(path))
    if rate != SAMPLE_RATE or not len(audio):
        raise ValueError(f"invalid 16 kHz audio: {path}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def _rms(audio: np.ndarray) -> float:
    values = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def _crop_or_tile(source: np.ndarray, samples: int, seed: int) -> tuple[np.ndarray, int]:
    if not len(source):
        raise ValueError("overlay audio is empty")
    tiled = np.tile(source, int(np.ceil(samples / len(source))) + 1)
    maximum = max(0, len(tiled) - samples)
    offset = int(np.random.default_rng(seed).integers(0, maximum + 1)) if maximum else 0
    return np.ascontiguousarray(tiled[offset : offset + samples], dtype=np.float32), offset


def _mix(clean: np.ndarray, overlay: np.ndarray, speech_mask: np.ndarray, snr_db: float) -> tuple[np.ndarray, dict[str, float]]:
    semantic_rms = max(_rms(clean[speech_mask]), 1e-8)
    overlay_rms = max(_rms(overlay[speech_mask]), 1e-8)
    scale = semantic_rms / (10.0 ** (snr_db / 20.0)) / overlay_rms
    unbounded = clean + overlay * scale
    peak = float(np.max(np.abs(unbounded), initial=0.0))
    limiter_gain = min(1.0, 0.98 / peak) if peak else 1.0
    mixed = np.ascontiguousarray(unbounded * limiter_gain, dtype=np.float32)
    achieved = 20.0 * np.log10(semantic_rms / max(overlay_rms * scale, 1e-8))
    return mixed, {
        "target_snr_db": float(snr_db),
        "achieved_snr_db": float(achieved),
        "overlay_scale": float(scale),
        "limiter_gain": float(limiter_gain),
    }


def _overlay_kind(row: dict[str, Any]) -> str:
    explicit = str(row.get("eval_type") or "")
    if explicit:
        return explicit
    text = " ".join([str(row.get("background_type") or ""), *(row.get("omni_flags") or [])]).lower()
    if "breath" in text:
        return "breathing"
    if "kiss" in text:
        return "kissing"
    if "moan" in text or "groan" in text:
        return "moaning"
    return "non_speech"


def select_bases(canonical: list[dict[str, Any]], predictions: list[dict[str, Any]], per_partition: int) -> list[dict[str, Any]]:
    prediction = {str(row["source_id"]): row for row in predictions}
    selected: list[dict[str, Any]] = []
    for partition in ("train", "val", "test"):
        candidates = [
            row for row in canonical
            if row.get("partition") == partition
            and row.get("row_role") == "speech"
            and row.get("additive_overlay") is None
            and str(row.get("source_id")) in prediction
        ]
        candidates.sort(key=lambda row: (
            prediction[str(row["source_id"])].get("fragmented_truth_run_count", 0),
            str(row["source_id"]),
        ))
        if len(candidates) < per_partition:
            raise ValueError(f"not enough clean {partition} bases")
        selected.extend(candidates[:per_partition])
    return selected


def build_variants(*, bases: list[dict[str, Any]], negatives: list[dict[str, Any]], output_dir: Path, snrs: list[float], overlay_types: list[str], seed: int) -> list[dict[str, Any]]:
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in negatives:
        audio = Path(str(row.get("audio") or ""))
        if audio.is_file():
            pools[(str(row.get("source_partition") or ""), _overlay_kind(row))].append(row)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    variants: list[dict[str, Any]] = []
    for base_index, base in enumerate(bases):
        clean = _load_audio(Path(str(base["audio"])))
        mask = np.zeros(len(clean), dtype=bool)
        fixed_context = []
        for span in base["canonical_spans"]:
            start, end = int(span["start_sample"]), int(span["end_sample"])
            fixed_context.append({key: span.get(key) for key in ("start_sample", "end_sample", "label", "core_id", "background_id")})
            if span["label"] == "speech":
                mask[start:end] = True
        identity = {
            "source_id": base["source_id"], "partition": base["partition"],
            "core_ids": base["core_ids"], "sample_count": len(clean), "canonical_spans": fixed_context,
        }
        pair_hash = _stable_hash(identity)
        axes: list[tuple[str, float | None, dict[str, Any] | None]] = [("clean", None, None)]
        for overlay_type in overlay_types:
            pool = pools[(str(base["partition"]), overlay_type)]
            if not pool:
                raise ValueError(f"no {base['partition']} overlay pool for {overlay_type}")
            source = pool[base_index % len(pool)]
            axes.extend((overlay_type, snr, source) for snr in snrs)
        for variant_index, (overlay_type, snr, source) in enumerate(axes):
            audio = clean
            overlay_detail = None
            if source is not None and snr is not None:
                raw = _load_audio(Path(str(source["audio"])))
                crop_seed = seed + base_index * 1009 + overlay_types.index(overlay_type) * 101
                overlay, offset = _crop_or_tile(raw, len(clean), crop_seed)
                audio, mix = _mix(clean, overlay, mask, snr)
                overlay_detail = {
                    "type": overlay_type, "audio_id": source.get("audio_id"),
                    "source_offset_sample": offset, "crop_seed": crop_seed, **mix,
                }
            variant_id = f"pair-{base_index:03d}-{variant_index:02d}-{overlay_type}-{snr if snr is not None else 'na'}"
            path = audio_dir / f"{variant_id}.wav"
            sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
            variants.append({
                "schema": SCHEMA, "diagnostic_only": True, "training_manifest_allowed": False,
                "boundary_serialization_contract_id": CONTRACT_ID,
                "variant_id": variant_id, "pair_hash": pair_hash, "fixed_identity": identity,
                "partition": base["partition"], "base_source_id": base["source_id"],
                "audio": str(path), "sample_count": len(audio), "duration_s": len(audio) / SAMPLE_RATE,
                "overlay_type": overlay_type, "overlay_snr_db": snr, "overlay": overlay_detail,
            })
    return variants


def _truth_labels(variant: dict[str, Any], frame_count: int) -> np.ndarray:
    labels = np.zeros(frame_count, dtype=np.int64)
    for span in variant["fixed_identity"]["canonical_spans"]:
        if span["label"] != "speech":
            continue
        start = max(0, int(np.floor(int(span["start_sample"]) / SAMPLE_RATE / FRAME_HOP_S)))
        end = min(frame_count, int(np.ceil(int(span["end_sample"]) / SAMPLE_RATE / FRAME_HOP_S)))
        labels[start:end] = 1
    return labels


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    truth_runs = _runs(truth == 1)
    fragmented = 0
    internal_gaps = 0
    run_count = 0
    for start, end in truth_runs:
        overlaps = _runs(predicted[start:end] == 1)
        run_count += len(overlaps)
        fragmented += int(len(overlaps) > 1)
        internal_gaps += int(sum(max(0, right[0] - left[1]) for left, right in zip(overlaps, overlaps[1:])))
    speech = truth == 1
    return {
        "truth_run_count": int(len(truth_runs)), "fragmented_truth_run_count": int(fragmented),
        "prediction_run_count_within_truth": int(run_count), "internal_gap_frames": int(internal_gaps),
        "speech_recall": float(np.mean(predicted[speech] == 1)) if np.any(speech) else 1.0,
        "continuity": float((len(truth_runs) - fragmented) / max(len(truth_runs), 1)),
    }


def extract_and_score(args: argparse.Namespace, variants: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    import torch

    apply_vram_safety_cap()
    if args.device.startswith("cuda"):
        torch.cuda.init()
        warmup = torch.ones(1, device=args.device)
        warmup.add_(1.0)
        torch.cuda.synchronize()
        del warmup
        torch.cuda.empty_cache()
        gc.collect()
        reset_shared_vram_baseline(required=True)
    memory_snapshots = []
    config = FeatureConfig(ptm=args.ptm, frame_hop_s=FRAME_HOP_S, window_s=30.0, overlap_s=5.0,
                           n_mfcc=40, n_fft=400, feature_dim=2048, device=args.device,
                           dtype="bfloat16", model_path=args.model_path, download=False,
                           attention="sdpa", language="Japanese")
    windows: list[dict[str, Any]] = []
    owners: list[list[int]] = []
    for variant_index, row in enumerate(variants):
        audio = _load_audio(Path(row["audio"]))
        indices = []
        for window_index, start in enumerate(_workflow_window_starts(sample_count=len(audio), sample_rate=SAMPLE_RATE, window_s=30.0, overlap_s=5.0)):
            chunk = np.ascontiguousarray(audio[start : min(len(audio), start + 30 * SAMPLE_RATE)], dtype=np.float32)
            indices.append(len(windows))
            windows.append({"window_index": window_index, "start_sample": start, "audio": chunk,
                            "mfcc": extract_mfcc(chunk, sample_rate=SAMPLE_RATE, config=config)})
        owners.append(indices)
    extractor = build_ptm_feature_extractor(config)
    try:
        _extract_ptm_window_features(
            ptm_extractor=extractor,
            window_audios=[windows[0]["audio"]],
            sample_rate=SAMPLE_RATE,
            ptm_window_batch_size=1,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            reset_shared_vram_baseline(required=True)
        memory_snapshots.append(runtime_memory_snapshot(require_shared_vram=args.device.startswith("cuda")))
        ptm_all, _ = _extract_ptm_window_features(ptm_extractor=extractor,
            window_audios=[row["audio"] for row in windows], sample_rate=SAMPLE_RATE,
            ptm_window_batch_size=args.batch_size)
    finally:
        extractor.close()
    bundles = []
    for row, indices in zip(variants, owners):
        local_windows = [windows[index] for index in indices]
        local_ptm = [ptm_all[index] for index in indices]
        bundles.append(_combine_workflow_window_features(windows=local_windows, ptm_features=local_ptm,
            duration_s=row["duration_s"], sample_rate=SAMPLE_RATE, config=config))
    del extractor, ptm_all, windows
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    memory_snapshots.append(runtime_memory_snapshot(require_shared_vram=args.device.startswith("cuda")))
    scorer = load_speech_island_scorer_checkpoint(args.checkpoint, device=args.device)
    results = []
    feature_dir = output_dir / "features"
    if args.save_features:
        feature_dir.mkdir(parents=True, exist_ok=True)
    for row, bundle in zip(variants, bundles):
        ptm = np.asarray(bundle["ptm"][:, :2048], dtype=np.float32)
        mfcc = np.asarray(bundle["mfcc"], dtype=np.float32)
        feature_path = ""
        if args.save_features:
            path = feature_dir / f"{row['variant_id']}.npz"
            np.savez_compressed(path, ptm=ptm, mfcc=mfcc)
            feature_path = str(path)
        truth = _truth_labels(row, len(ptm))
        probabilities = score_binary_speech_class_probabilities(scorer, ptm=ptm, mfcc=mfcc)
        predicted = np.argmax(probabilities, axis=1)
        results.append({**row, **_metrics(truth, predicted),
                        "mean_speech_probability": float(np.mean(probabilities[:, 1])),
                        "frame_count": len(ptm), "feature_path": feature_path})
    del scorer, bundles
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    memory_snapshots.append(runtime_memory_snapshot(require_shared_vram=args.device.startswith("cuda")))
    (output_dir / "memory_snapshots.json").write_text(
        json.dumps(memory_snapshots, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for snapshot in memory_snapshots:
        if float(snapshot["physical_ram_used_mb"]) > float(snapshot["physical_ram_budget_mb"]):
            raise MemoryError("paired overlay ablation exceeded the 0.95 physical RAM budget")
        if args.device.startswith("cuda") and float(snapshot.get("shared_vram_mb") or 0.0) > 0.0:
            raise MemoryError("paired overlay ablation shared VRAM spill is a soft OOM")
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    clean = {row["pair_hash"]: row for row in results if row["overlay_type"] == "clean"}
    deltas = []
    for row in results:
        if row["overlay_type"] == "clean":
            continue
        base = clean[row["pair_hash"]]
        deltas.append({
            "pair_hash": row["pair_hash"], "partition": row["partition"],
            "overlay_type": row["overlay_type"], "overlay_snr_db": row["overlay_snr_db"],
            "continuity_delta": row["continuity"] - base["continuity"],
            "internal_gap_frames_delta": row["internal_gap_frames"] - base["internal_gap_frames"],
            "prediction_run_count_delta": row["prediction_run_count_within_truth"] - base["prediction_run_count_within_truth"],
            "speech_recall_delta": row["speech_recall"] - base["speech_recall"],
        })
    grouped = {}
    for key in sorted({(row["overlay_type"], row["overlay_snr_db"]) for row in deltas}):
        values = [row for row in deltas if (row["overlay_type"], row["overlay_snr_db"]) == key]
        grouped[f"{key[0]}@{key[1]:g}dB"] = {
            metric: float(np.mean([row[metric] for row in values]))
            for metric in ("continuity_delta", "internal_gap_frames_delta", "prediction_run_count_delta", "speech_recall_delta")
        }
    return {"schema": SCHEMA, "diagnostic_only": True, "training_manifest_allowed": False,
            "variant_count": len(results), "pair_count": len(clean), "partition_counts": dict(Counter(row["partition"] for row in clean.values())),
            "paired_deltas": deltas, "grouped_mean_deltas": grouped}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.base_manifest:
        bases = _rows(Path(args.base_manifest))
    else:
        bases = select_bases(_rows(Path(args.canonical_sources)), _rows(Path(args.predictions)), args.per_partition)
    overlay_types = args.overlay_type or ["breathing", "kissing", "non_speech"]
    snrs = args.snr_db or [10.0, 14.0, 18.0]
    variants = build_variants(bases=bases, negatives=_rows(Path(args.negative_manifest)), output_dir=output_dir,
                              snrs=snrs, overlay_types=overlay_types, seed=args.seed)
    _write_jsonl(output_dir / "variants.jsonl", variants)
    results = extract_and_score(args, variants, output_dir)
    _write_jsonl(output_dir / "results.jsonl", results)
    summary = summarize(results)
    summary.update({"checkpoint": args.checkpoint, "canonical_sources": args.canonical_sources,
                    "base_manifest": args.base_manifest,
                    "predictions": args.predictions, "variants": str(output_dir / "variants.jsonl"),
                    "results": str(output_dir / "results.jsonl"),
                    "memory_snapshots": str(output_dir / "memory_snapshots.json")})
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-sources", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--base-manifest", default="")
    parser.add_argument("--negative-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ptm", default="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf")
    parser.add_argument("--overlay-type", action="append")
    parser.add_argument("--snr-db", action="append", type=float)
    parser.add_argument("--per-partition", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-features", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
