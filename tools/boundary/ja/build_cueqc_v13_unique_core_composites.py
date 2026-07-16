#!/usr/bin/env python3
"""Build unique-core Galgame composites for CueQC v13 runtime replay."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from audio.loading import load_audio_16k_mono  # noqa: E402
from tools.boundary.ja.build_galgame_synthetic_timeline import (  # noqa: E402
    crop_or_tile_audio,
)


SCHEMA = "cueqc_v13_unique_core_composite_v1"
SUMMARY_SCHEMA = "cueqc_v13_unique_core_composite_summary_v1"
SAMPLE_RATE = 16000


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = load_audio_16k_mono(str(path))
    if sample_rate != SAMPLE_RATE or not len(audio):
        raise ValueError(f"invalid 16 kHz audio: {path}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def _rms(audio: np.ndarray) -> float:
    values = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def _limit(audio: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.ascontiguousarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    gain = 0.98 / peak if peak > 0.98 else 1.0
    return np.ascontiguousarray(values * gain, dtype=np.float32), float(gain)


def _partition(index: int, count: int) -> str:
    ratio = (index + 0.5) / max(1, count)
    if ratio < 0.85:
        return "train"
    if ratio < 0.95:
        return "val"
    return "test"


def _gap_duration_quantiles(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    values = np.asarray(payload.get("durations_s") or [], dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if not values.size:
        raise ValueError("real gap duration pool is empty")
    return {
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "q40": float(np.quantile(values, 0.40)),
    }


def _snr_values(path: Path) -> np.ndarray:
    values: list[float] = []
    for row in _rows(path):
        detail = row.get("background_mix") or (row.get("augmentation") or {}).get(
            "background_mix"
        )
        if not isinstance(detail, dict) or detail.get("enabled") is False:
            continue
        value = detail.get("snr_db")
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    if not values:
        raise ValueError("SNR reference has no enabled values")
    return np.asarray(values, dtype=np.float64)


def _negative_pools(path: Path) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(path):
        audio = Path(str(row.get("audio") or ""))
        if not audio.exists():
            continue
        pools[str(row.get("source_partition") or "train")].append(row)
    for partition in ("train", "val", "test"):
        if not pools[partition]:
            raise ValueError(f"negative manifest has no {partition} rows")
    return dict(pools)


def _is_vocal_negative(row: dict[str, Any]) -> bool:
    values = {
        str(row.get("background_type") or "").lower(),
        *(str(value).lower() for value in row.get("omni_flags") or []),
    }
    tokens = (
        "breath",
        "moan",
        "kiss",
        "cry",
        "sob",
        "movement",
        "background speech",
        "crowd",
    )
    return any(token in value for value in values for token in tokens)


def _negative_clip(
    row: dict[str, Any], *, samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    source = _load_audio(Path(str(row["audio"])))
    clipped, offset = crop_or_tile_audio(source, samples=samples, rng=rng)
    return clipped, {
        "audio_id": str(row.get("audio_id") or Path(str(row["audio"])).stem),
        "audio": str(row["audio"]),
        "background_type": str(row.get("background_type") or "negative"),
        "omni_flags": list(row.get("omni_flags") or []),
        "source_offset_s": offset / SAMPLE_RATE,
        "duration_s": len(clipped) / SAMPLE_RATE,
    }


def _mix_overlay(
    clean: np.ndarray,
    overlay: np.ndarray,
    *,
    core_spans: list[dict[str, Any]],
    target_snr_db: float,
) -> tuple[np.ndarray, dict[str, float]]:
    semantic = np.concatenate(
        [clean[int(span["start_sample"]) : int(span["end_sample"])] for span in core_spans]
    )
    semantic_rms = max(_rms(semantic), 1e-6)
    overlay_rms = max(_rms(overlay), 1e-6)
    target_overlay_rms = semantic_rms / (10.0 ** (target_snr_db / 20.0))
    scale = target_overlay_rms / overlay_rms
    mixed, limiter_gain = _limit(clean + overlay * scale)
    return mixed, {
        "target_snr_db": float(target_snr_db),
        "achieved_snr_db": float(
            20.0 * np.log10(semantic_rms / max(overlay_rms * scale, 1e-6))
        ),
        "overlay_scale": float(scale),
        "limiter_gain": float(limiter_gain),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    cores = _rows(Path(args.semantic_cores))
    rng.shuffle(cores)
    required_cores = int(args.sample_count) * 2
    if required_cores > len(cores):
        raise ValueError(
            f"two-core dataset requires {required_cores} unique cores; got {len(cores)}"
        )
    selected = cores[:required_cores]
    core_ids = [str(row["audio_id"]) for row in selected]
    if len(set(core_ids)) != len(core_ids):
        raise ValueError("semantic core inventory contains duplicate audio_id values")

    negative_pools = _negative_pools(Path(args.negative_manifest))
    vocal_pools = {
        partition: [row for row in pool if _is_vocal_negative(row)]
        for partition, pool in negative_pools.items()
    }
    if any(not pool for pool in vocal_pools.values()):
        raise ValueError("negative manifest lacks partitioned vocal negatives")
    for pool in negative_pools.values():
        rng.shuffle(pool)
    for pool in vocal_pools.values():
        rng.shuffle(pool)
    negative_cursor = Counter()
    vocal_cursor = Counter()
    gap_quantiles = _gap_duration_quantiles(Path(args.gap_durations))
    negative_duration_quantiles = {
        partition: np.quantile(
            np.asarray(
                [float(row.get("duration_s") or 0.0) for row in pool],
                dtype=np.float64,
            ),
            [0.25, 0.50, 0.75],
        )
        for partition, pool in vocal_pools.items()
    }
    snr_values = _snr_values(Path(args.snr_reference))
    snr_quantiles = np.quantile(snr_values, [0.15, 0.35, 0.50, 0.65, 0.85])

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for sample_index in range(int(args.sample_count)):
        partition = _partition(sample_index, int(args.sample_count))
        pair = selected[sample_index * 2 : sample_index * 2 + 2]
        core_audio = [_load_audio(Path(str(row["audio"]))) for row in pair]
        gap_key = ("q10", "q25", "q40")[sample_index % 3]
        gap_samples = max(1, int(round(gap_quantiles[gap_key] * SAMPLE_RATE)))
        gaps: list[np.ndarray] = []
        gap_sources: list[dict[str, Any]] = []
        for _ in range(2):
            pool = negative_pools[partition]
            cursor = int(negative_cursor[partition])
            row = pool[cursor % len(pool)]
            negative_cursor[partition] += 1
            clip, detail = _negative_clip(row, samples=gap_samples, rng=rng)
            gaps.append(clip)
            gap_sources.append(detail)

        vocal_pool = vocal_pools[partition]
        vocal_index = int(vocal_cursor[partition])
        vocal_row = vocal_pool[vocal_index % len(vocal_pool)]
        vocal_cursor[partition] += 1
        negative_duration = float(
            negative_duration_quantiles[partition][sample_index % 3]
        )
        negative_samples = max(1, int(round(negative_duration * SAMPLE_RATE)))
        negative_unit, negative_unit_source = _negative_clip(
            vocal_row, samples=negative_samples, rng=rng
        )

        first_end = len(core_audio[0])
        negative_start = first_end + len(gaps[0])
        negative_end = negative_start + len(negative_unit)
        second_start = negative_end + len(gaps[1])
        clean, clean_gain = _limit(
            np.concatenate(
                (core_audio[0], gaps[0], negative_unit, gaps[1], core_audio[1])
            )
        )
        core_spans = [
            {
                "core_id": str(pair[0]["audio_id"]),
                "text": str(pair[0].get("text") or ""),
                "source_audio": str(pair[0]["audio"]),
                "start_sample": 0,
                "end_sample": first_end,
                "start_s": 0.0,
                "end_s": first_end / SAMPLE_RATE,
            },
            {
                "core_id": str(pair[1]["audio_id"]),
                "text": str(pair[1].get("text") or ""),
                "source_audio": str(pair[1]["audio"]),
                "start_sample": second_start,
                "end_sample": second_start + len(core_audio[1]),
                "start_s": second_start / SAMPLE_RATE,
                "end_s": (second_start + len(core_audio[1])) / SAMPLE_RATE,
            },
        ]
        overlay_enabled = sample_index % 2 == 1
        overlay_detail: dict[str, Any] | None = None
        mixed = clean
        if overlay_enabled:
            pool = negative_pools[partition]
            cursor = int(negative_cursor[partition])
            overlay_row = pool[cursor % len(pool)]
            negative_cursor[partition] += 1
            overlay, source_detail = _negative_clip(
                overlay_row, samples=len(clean), rng=rng
            )
            target_snr = float(snr_quantiles[(sample_index // 2) % len(snr_quantiles)])
            mixed, mix_detail = _mix_overlay(
                clean,
                overlay,
                core_spans=core_spans,
                target_snr_db=target_snr,
            )
            overlay_detail = {"source": source_detail, "mix": mix_detail}

        sample_id = f"cueqc-v13-gal-unique-{sample_index:06d}"
        audio_path = audio_dir / f"{sample_id}.wav"
        sf.write(str(audio_path), mixed, SAMPLE_RATE, subtype="PCM_16")
        rows.append(
            {
                "schema": SCHEMA,
                "sample_id": sample_id,
                "audio": str(audio_path),
                "sample_rate": SAMPLE_RATE,
                "sample_count": int(len(mixed)),
                "duration_s": len(mixed) / SAMPLE_RATE,
                "source_partition": partition,
                "pipeline_entry_stage": "semantic_speech_scorer",
                "scorer_required": True,
                "core_spans": core_spans,
                "negative_unit_span": {
                    "start_sample": negative_start,
                    "end_sample": negative_end,
                    "start_s": negative_start / SAMPLE_RATE,
                    "end_s": negative_end / SAMPLE_RATE,
                    "source": negative_unit_source,
                    "role": "cueqc_drop_target_if_isolated_by_runtime_split",
                },
                "inter_unit_gaps": {
                    "left_start_sample": first_end,
                    "left_end_sample": negative_start,
                    "right_start_sample": negative_end,
                    "right_end_sample": second_start,
                    "duration_source": gap_key,
                    "sources": gap_sources,
                },
                "clean_limiter_gain": clean_gain,
                "additive_overlay": overlay_detail,
                "label_contract": "new_runtime_chunk_intersection_with_exact_unique_semantic_core_spans_v1",
            }
        )

    manifest = output_dir / "source_manifest.jsonl"
    _write_jsonl(manifest, rows)
    used = Counter(
        str(core["core_id"]) for row in rows for core in row["core_spans"]
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "seed": int(args.seed),
        "source_count": len(rows),
        "semantic_core_count": sum(used.values()),
        "unique_semantic_core_count": len(used),
        "max_core_use_count": max(used.values(), default=0),
        "core_reuse_policy": "each_core_at_most_once_v1",
        "gap_duration_quantiles_s": gap_quantiles,
        "central_negative_unit": "partitioned_real_vocal_negative_v1",
        "additive_overlay_count": sum(row["additive_overlay"] is not None for row in rows),
        "partition_counts": dict(Counter(row["source_partition"] for row in rows)),
        "manifest": str(manifest),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-cores", required=True)
    parser.add_argument("--negative-manifest", required=True)
    parser.add_argument("--gap-durations", required=True)
    parser.add_argument("--snr-reference", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False))
