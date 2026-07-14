#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.loading import load_audio_16k_mono  # noqa: E402
from tools.boundary.ja.build_galgame_synthetic_timeline import (  # noqa: E402
    crop_or_tile_audio,
)


SCHEMA = "semantic_split_multicore_additive_overlay_smoke_v2"
SUMMARY_SCHEMA = "semantic_split_multicore_additive_overlay_smoke_summary_v2"
OVERLAY_SCHEMA = "semantic_core_additive_overlay_v1"
SAMPLE_RATE = 16000


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_span(path: Path, start_s: float, end_s: float) -> tuple[np.ndarray, int, int]:
    audio, sample_rate = load_audio_16k_mono(str(path))
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected 16 kHz normalized audio: {path}")
    start = max(0, int(round(start_s * sample_rate)))
    end = min(len(audio), int(round(end_s * sample_rate)))
    if end <= start:
        raise ValueError(f"empty semantic span: {path} {start_s:.3f}-{end_s:.3f}")
    return np.ascontiguousarray(audio[start:end], dtype=np.float32), start, end


def load_semantic_cores(path: Path) -> list[dict[str, Any]]:
    cores: list[dict[str, Any]] = []
    for source in _rows(path):
        units = {str(unit["unit_id"]): unit for unit in source["text_units"]}
        for alignment in source["semantic_alignments"]:
            if alignment["status"] != "matched":
                continue
            unit_id = str(alignment["unit_id"])
            unit = units[unit_id]
            audio, source_start_sample, source_end_sample = _load_span(
                Path(source["audio"]),
                float(alignment["start_s"]),
                float(alignment["end_s"]),
            )
            cores.append(
                {
                    "core_id": f"{source['sample_id']}:{unit_id}",
                    "source_sample_id": str(source["sample_id"]),
                    "source_audio": str(source["audio"]),
                    "source_start_s": source_start_sample / SAMPLE_RATE,
                    "source_end_s": source_end_sample / SAMPLE_RATE,
                    "teacher_start_s": float(alignment["start_s"]),
                    "teacher_end_s": float(alignment["end_s"]),
                    "source_start_sample": source_start_sample,
                    "source_end_sample": source_end_sample,
                    "text": str(unit["text"]),
                    "audio": audio,
                }
            )
    cores.sort(key=lambda row: row["core_id"])
    if len(cores) < 4:
        raise ValueError("multi-core smoke requires at least four matched semantic cores")
    return cores


def load_gap_quantiles(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    values = np.asarray(payload.get("durations_s") or [], dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise ValueError("gap duration pool is empty")
    return {
        "q25": float(np.quantile(values, 0.25)),
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "source": str(path),
        "count": int(values.size),
    }


def load_snr_quantiles(path: Path) -> dict[str, Any]:
    values: list[float] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            detail = row.get("background_mix") or (row.get("augmentation") or {}).get(
                "background_mix"
            )
            if (
                not isinstance(detail, dict)
                or detail.get("enabled") is False
                or detail.get("skipped")
            ):
                continue
            value = detail.get("snr_db")
            if value is not None and np.isfinite(float(value)):
                values.append(float(value))
    if not values:
        raise ValueError("SNR reference manifest has no enabled background mixes")
    array = np.asarray(values, dtype=np.float64)
    return {
        "q15": float(np.quantile(array, 0.15)),
        "q35": float(np.quantile(array, 0.35)),
        "q50": float(np.quantile(array, 0.50)),
        "q65": float(np.quantile(array, 0.65)),
        "q85": float(np.quantile(array, 0.85)),
        "source": str(path),
        "count": int(array.size),
        "derivation": "empirical_hardmix_background_snr_quantiles_v1",
    }


def _negative_matches(row: dict[str, Any], flags: set[str]) -> bool:
    values = {
        str(row.get("background_type") or "").lower(),
        *(str(value).lower() for value in row.get("omni_flags") or []),
    }
    return any(any(flag in value for value in values) for flag in flags)


def select_overlay_rows(
    path: Path, *, min_duration_s: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [
        row
        for row in _rows(path)
        if Path(str(row.get("audio") or "")).exists()
        and str(row.get("source_partition") or "train") == "train"
        and float(row.get("duration_s") or 0.0) >= float(min_duration_s)
    ]
    music = sorted(
        (row for row in rows if _negative_matches(row, {"music", "bgm"})),
        key=lambda row: (
            -float(row.get("duration_s") or 0.0),
            str(row.get("audio_id") or row["audio"]),
        ),
    )
    vocal = sorted(
        (
            row
            for row in rows
            if _negative_matches(row, {"breathing", "moaning", "kissing"})
        ),
        key=lambda row: (
            -float(row.get("duration_s") or 0.0),
            str(row.get("audio_id") or row["audio"]),
        ),
    )
    if len(music) < 2 or not vocal:
        raise ValueError("overlay manifest needs two music rows and one vocal row")
    return [music[0], music[1], vocal[0]], {
        "manifest": str(path),
        "music_audio_ids": [str(music[0]["audio_id"]), str(music[1]["audio_id"])],
        "music_durations_s": [float(music[0]["duration_s"]), float(music[1]["duration_s"])],
        "vocal_audio_id": str(vocal[0]["audio_id"]),
        "vocal_duration_s": float(vocal[0]["duration_s"]),
        "selection": "longest_train_assets_per_overlay_type_v1",
    }


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64, copy=False)))))


def _limit(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _overlay_audio(
    row: dict[str, Any], *, duration_s: float, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    audio, sample_rate = load_audio_16k_mono(str(row["audio"]))
    if sample_rate != SAMPLE_RATE:
        raise ValueError("negative audio must normalize to 16 kHz")
    values, offset = crop_or_tile_audio(
        audio,
        samples=max(1, int(round(duration_s * sample_rate))),
        rng=rng,
    )
    return values, {
        "audio_id": str(row.get("audio_id") or Path(str(row["audio"])).stem),
        "source_audio": str(row["audio"]),
        "source_partition": str(row.get("source_partition") or "train"),
        "background_type": str(row.get("background_type") or "unknown"),
        "omni_flags": [str(value) for value in row.get("omni_flags") or []],
        "source_offset_sample": int(offset),
        "source_offset_s": offset / SAMPLE_RATE,
        "source_sample_count": int(len(audio)),
        "source_duration_s": len(audio) / SAMPLE_RATE,
        "rendered_sample_count": int(len(values)),
        "rendered_duration_s": len(values) / SAMPLE_RATE,
        "tiled": bool(len(audio) < len(values)),
    }


def _semantic_rms(audio: np.ndarray, core_spans: list[dict[str, Any]]) -> float:
    pieces = [
        audio[int(core["start_sample"]) : int(core["end_sample"])]
        for core in core_spans
        if int(core["end_sample"]) > int(core["start_sample"])
    ]
    return _rms(np.concatenate(pieces)) if pieces else _rms(audio)


def _mix_additive_overlay(
    clean: np.ndarray,
    overlay: np.ndarray,
    *,
    core_spans: list[dict[str, Any]],
    target_snr_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    clean = np.ascontiguousarray(clean, dtype=np.float32)
    overlay = np.ascontiguousarray(overlay, dtype=np.float32)
    if clean.shape != overlay.shape:
        raise ValueError("clean and overlay audio must have identical sample counts")
    semantic_rms = _semantic_rms(clean, core_spans)
    source_overlay_rms = _rms(overlay)
    if semantic_rms <= 1e-8 or source_overlay_rms <= 1e-8:
        raise ValueError("clean semantic core and overlay must both have non-zero RMS")
    target_overlay_rms = semantic_rms / (10.0 ** (target_snr_db / 20.0))
    overlay_scale = target_overlay_rms / source_overlay_rms
    scaled_overlay = overlay * overlay_scale
    mix = clean + scaled_overlay
    peak_before = float(np.max(np.abs(mix))) if mix.size else 0.0
    limiter_gain = min(1.0, 0.98 / peak_before) if peak_before > 0.0 else 1.0
    clean_component = np.ascontiguousarray(clean * limiter_gain, dtype=np.float32)
    overlay_component = np.ascontiguousarray(scaled_overlay * limiter_gain, dtype=np.float32)
    mixed = np.ascontiguousarray(clean_component + overlay_component, dtype=np.float32)
    achieved_snr_db = 20.0 * np.log10(
        _semantic_rms(clean_component, core_spans) / max(_rms(overlay_component), 1e-12)
    )
    return clean_component, overlay_component, mixed, {
        "target_snr_db": float(target_snr_db),
        "achieved_snr_db": float(achieved_snr_db),
        "semantic_reference_rms": float(semantic_rms),
        "source_overlay_rms": float(source_overlay_rms),
        "overlay_scale": float(overlay_scale),
        "mix_peak_before_limiter": peak_before,
        "shared_limiter_gain": float(limiter_gain),
    }


def _switched_overlay(
    first: np.ndarray,
    second: np.ndarray,
    *,
    crossfade_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if first.shape != second.shape:
        raise ValueError("switched overlays must have identical sample counts")
    size = first.size
    midpoint = size // 2
    fade = min(max(1, int(round(crossfade_s * SAMPLE_RATE))), midpoint, size - midpoint)
    start = midpoint - fade // 2
    end = start + fade
    bed = first.copy()
    phase = np.linspace(0.0, np.pi / 2.0, fade, dtype=np.float32)
    bed[start:end] = first[start:end] * np.cos(phase) + second[start:end] * np.sin(phase)
    bed[end:] = second[end:]
    return np.ascontiguousarray(bed, dtype=np.float32), {
        "switch_sample": int(midpoint),
        "switch_s": midpoint / SAMPLE_RATE,
        "crossfade_sample_count": int(fade),
        "crossfade_s": fade / SAMPLE_RATE,
    }


def _core_copy(core: dict[str, Any], *, start_sample: int, end_sample: int) -> dict[str, Any]:
    return {
        "core_id": core["core_id"],
        "text": core["text"],
        "start_sample": start_sample,
        "end_sample": end_sample,
        "start_s": start_sample / SAMPLE_RATE,
        "end_s": end_sample / SAMPLE_RATE,
        "source_audio": core["source_audio"],
        "core_library_audio": core.get("core_library_audio"),
        "source_start_sample": core["source_start_sample"],
        "source_end_sample": core["source_end_sample"],
        "source_start_s": core["source_start_s"],
        "source_end_s": core["source_end_s"],
    }


def _safe_composite(
    *,
    sample_id: str,
    cores: list[dict[str, Any]],
    gaps: list[tuple[str, np.ndarray, dict[str, Any]]],
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(gaps) != len(cores) - 1:
        raise ValueError("safe composite needs one gap between adjacent cores")
    parts: list[np.ndarray] = []
    core_spans: list[dict[str, Any]] = []
    gap_spans: list[dict[str, Any]] = []
    cursor = 0
    for index, core in enumerate(cores):
        values = np.asarray(core["audio"], dtype=np.float32)
        start = cursor
        cursor += values.size
        core_spans.append(_core_copy(core, start_sample=start, end_sample=cursor))
        parts.append(values)
        if index >= len(gaps):
            continue
        kind, gap, detail = gaps[index]
        gap_start = cursor
        cursor += gap.size
        gap_spans.append(
            {
                "gap_id": f"{sample_id}:g{index:02d}",
                "kind": kind,
                "start_sample": gap_start,
                "end_sample": cursor,
                "start_s": gap_start / SAMPLE_RATE,
                "end_s": cursor / SAMPLE_RATE,
                "source": detail,
            }
        )
        parts.append(gap)
    audio = _limit(np.concatenate(parts))
    events = []
    for index, (left, right, gap) in enumerate(zip(core_spans, core_spans[1:], gap_spans)):
        events.append(
            {
                "event_id": f"{sample_id}:e{index:02d}",
                "left_core_id": left["core_id"],
                "right_core_id": right["core_id"],
                "semantic_decision": "cut",
                "representative_sample": (int(left["end_sample"]) + int(right["start_sample"])) // 2,
                "representative_s": ((int(left["end_sample"]) + int(right["start_sample"])) // 2) / SAMPLE_RATE,
                "event_interval_start_sample": int(left["end_sample"]),
                "event_interval_end_sample": int(right["start_sample"]),
                "event_interval_start_s": float(left["end_s"]),
                "event_interval_end_s": float(right["start_s"]),
                "inner_target": {
                    "status": "safe",
                    "left_speech_end_sample": int(left["end_sample"]),
                    "right_speech_start_sample": int(right["start_sample"]),
                    "left_speech_end_s": float(left["end_s"]),
                    "right_speech_start_s": float(right["start_s"]),
                    "removed_gap_start_sample": int(gap["start_sample"]),
                    "removed_gap_end_sample": int(gap["end_sample"]),
                    "removed_gap_start_s": float(gap["start_s"]),
                    "removed_gap_end_s": float(gap["end_s"]),
                    "gap_kind": gap["kind"],
                    "reason": "独立 semantic cores 之间存在已知可移除的非语义区间。",
                },
            }
        )
    return audio, {
        "core_spans": core_spans,
        "gap_spans": gap_spans,
        "semantic_events": events,
        "continue_control": False,
    }


def _overlap_composite(
    *, sample_id: str, left: dict[str, Any], right: dict[str, Any], overlap_s: float
) -> tuple[np.ndarray, dict[str, Any]]:
    left_audio = np.asarray(left["audio"], dtype=np.float32)
    right_audio = np.asarray(right["audio"], dtype=np.float32)
    overlap = min(
        max(1, int(round(overlap_s * SAMPLE_RATE))),
        left_audio.size - 1,
        right_audio.size - 1,
    )
    phase = np.linspace(0.0, np.pi / 2.0, overlap, dtype=np.float32)
    mixed = left_audio[-overlap:] * np.cos(phase) + right_audio[:overlap] * np.sin(phase)
    audio = _limit(np.concatenate((left_audio[:-overlap], mixed, right_audio[overlap:])))
    left_span = _core_copy(left, start_sample=0, end_sample=left_audio.size)
    right_start = left_audio.size - overlap
    right_span = _core_copy(
        right,
        start_sample=right_start,
        end_sample=right_start + right_audio.size,
    )
    event = {
        "event_id": f"{sample_id}:e00",
        "left_core_id": left_span["core_id"],
        "right_core_id": right_span["core_id"],
        "semantic_decision": "cut",
        "representative_sample": (int(left_span["end_sample"]) + int(right_span["start_sample"])) // 2,
        "representative_s": ((int(left_span["end_sample"]) + int(right_span["start_sample"])) // 2) / SAMPLE_RATE,
        "event_interval_start_sample": int(right_span["start_sample"]),
        "event_interval_end_sample": int(left_span["end_sample"]),
        "event_interval_start_s": right_span["start_s"],
        "event_interval_end_s": left_span["end_s"],
        "inner_target": {
            "status": "abstain",
            "left_speech_end_sample": None,
            "right_speech_start_sample": None,
            "left_speech_end_s": None,
            "right_speech_start_s": None,
            "removed_gap_start_sample": None,
            "removed_gap_end_sample": None,
            "removed_gap_start_s": None,
            "removed_gap_end_s": None,
            "gap_kind": "overlap",
            "reason": "独立 semantic cores 发生重叠，没有连续安全区，不得强制切。",
        },
    }
    return audio, {
        "core_spans": [left_span, right_span],
        "gap_spans": [
            {
                "gap_id": f"{sample_id}:g00",
                "kind": "overlap",
                "start_sample": int(right_span["start_sample"]),
                "end_sample": int(left_span["end_sample"]),
                "start_s": right_span["start_s"],
                "end_s": left_span["end_s"],
                "source": {"overlap_s": overlap / SAMPLE_RATE},
            }
        ],
        "semantic_events": [event],
        "continue_control": False,
    }


def build_smoke(
    *,
    semantic_labels: Path,
    overlay_manifest: Path,
    gap_duration_pool: Path,
    snr_reference_manifest: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cores = load_semantic_cores(semantic_labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    core_dir = output_dir / "semantic_cores"
    core_dir.mkdir(parents=True, exist_ok=True)
    core_library_rows: list[dict[str, Any]] = []
    for core in cores:
        filename = core["core_id"].replace(":", "__") + ".wav"
        core_path = core_dir / filename
        sf.write(str(core_path), core["audio"], SAMPLE_RATE, subtype="PCM_16")
        core["core_library_audio"] = str(core_path)
        core_library_rows.append(
            {
                "schema": "semantic_core_seed_library_v1",
                "core_id": core["core_id"],
                "audio": str(core_path),
                "sample_rate": SAMPLE_RATE,
                "sample_count": int(len(core["audio"])),
                "duration_s": len(core["audio"]) / SAMPLE_RATE,
                "text": core["text"],
                "source_audio": core["source_audio"],
                "source_start_sample": core["source_start_sample"],
                "source_end_sample": core["source_end_sample"],
                "source_start_s": core["source_start_s"],
                "source_end_s": core["source_end_s"],
                "teacher_start_s": core["teacher_start_s"],
                "teacher_end_s": core["teacher_end_s"],
            }
        )
    core_library_path = output_dir / "semantic_core_library.jsonl"
    _write_jsonl(core_library_path, core_library_rows)
    quantiles = load_gap_quantiles(gap_duration_pool)
    overlay_rows, overlay_selection = select_overlay_rows(
        overlay_manifest,
        min_duration_s=quantiles["q50"],
    )
    snr_quantiles = load_snr_quantiles(snr_reference_manifest)
    music_a, music_b, vocal = overlay_rows
    core_rms = float(np.median([_rms(core["audio"]) for core in cores]))
    longest_index = max(range(len(cores)), key=lambda index: len(cores[index]["audio"]))
    long_core = cores[longest_index]
    remaining = [core for index, core in enumerate(cores) if index != longest_index]
    shortest_index = min(
        range(len(remaining)), key=lambda index: len(remaining[index]["audio"])
    )
    short_core = remaining[shortest_index]
    medium_cores = [
        core for index, core in enumerate(remaining) if index != shortest_index
    ]
    medium_left, medium_right = medium_cores[:2]

    silence = rng.normal(
        0.0,
        max(core_rms * 0.003, 1e-6),
        max(1, int(round(quantiles["q25"] * SAMPLE_RATE))),
    ).astype(np.float32)
    short_room = rng.normal(
        0.0,
        max(core_rms * 0.006, 1e-6),
        max(1, int(round(quantiles["q25"] * SAMPLE_RATE))),
    ).astype(np.float32)
    long_room = rng.normal(
        0.0,
        max(core_rms * 0.006, 1e-6),
        max(1, int(round(quantiles["q50"] * SAMPLE_RATE))),
    ).astype(np.float32)

    recipes: list[dict[str, Any]] = []

    def add_overlay_recipe(
        *,
        sample_id: str,
        focus: str,
        clean: np.ndarray,
        truth: dict[str, Any],
        axes: dict[str, Any],
        source_rows: list[dict[str, Any]],
        snr_quantile: str,
        switch: bool = False,
    ) -> None:
        source_details: list[dict[str, Any]] = []
        beds: list[np.ndarray] = []
        for source_row in source_rows:
            bed, source_detail = _overlay_audio(
                source_row,
                duration_s=len(clean) / SAMPLE_RATE,
                rng=rng,
            )
            beds.append(bed)
            source_details.append(source_detail)
        switch_detail = None
        if switch:
            overlay, switch_detail = _switched_overlay(
                beds[0],
                beds[1],
                crossfade_s=quantiles["q25"],
            )
        else:
            overlay = beds[0]
        clean_component, overlay_component, mixed, mix_detail = _mix_additive_overlay(
            clean,
            overlay,
            core_spans=truth["core_spans"],
            target_snr_db=float(snr_quantiles[snr_quantile]),
        )
        recipes.append(
            {
                "sample_id": sample_id,
                "audit_focus": focus,
                "clean": clean_component,
                "overlay_audio": overlay_component,
                "mixed": mixed,
                "truth": truth,
                "sampling_axes": {
                    **axes,
                    "overlay_axis": "switch" if switch else "continuous",
                    "snr_quantile": snr_quantile,
                },
                "overlay": {
                    "schema": OVERLAY_SCHEMA,
                    "mode": "additive_full_duration",
                    "start_sample": 0,
                    "end_sample": int(len(clean)),
                    "start_s": 0.0,
                    "end_s": len(clean) / SAMPLE_RATE,
                    "sources": source_details,
                    "snr_reference": {
                        "quantile": snr_quantile,
                        "manifest": str(snr_reference_manifest),
                        "derivation": snr_quantiles["derivation"],
                    },
                    "mix": mix_detail,
                    "switch": switch_detail,
                    "semantic_timeline_effect": "none",
                    "overlay_transitions_create_semantic_events": False,
                },
            }
        )

    clean, truth = _safe_composite(
        sample_id="ov01_music_over_two_core_safe",
        cores=[medium_left, medium_right],
        gaps=[("low_room_tone", silence, {"duration_quantile": "q25"})],
    )
    add_overlay_recipe(
        sample_id="ov01_music_over_two_core_safe",
        focus="BGM 与两个 semantic cores 全程同时可听；core 语义必须完整，gap 仍是可移除的 semantic-safe 区间。",
        clean=clean,
        truth=truth,
        axes={"core_count": 2, "gap_axis": "low_room_tone", "speaker_axis": "cross_source"},
        source_rows=[music_a],
        snr_quantile="q85",
    )

    clean, truth = _safe_composite(
        sample_id="ov02_music_over_three_core_two_safe",
        cores=[short_core, medium_left, medium_right],
        gaps=[
            ("low_room_tone", long_room, {"duration_quantile": "q50"}),
            ("low_room_tone", short_room, {"duration_quantile": "q25"}),
        ],
    )
    add_overlay_recipe(
        sample_id="ov02_music_over_three_core_two_safe",
        focus="BGM 覆盖三个 semantic cores 与两个 gap；两个 Split events 都来自语义 core 关系，而不是背景变化。",
        clean=clean,
        truth=truth,
        axes={"core_count": 3, "gap_axis": "two_room_gaps", "speaker_axis": "cross_source"},
        source_rows=[music_b],
        snr_quantile="q50",
    )

    clean, truth = _safe_composite(
        sample_id="ov03_vocal_over_two_core_safe",
        cores=[medium_right, short_core],
        gaps=[("low_room_tone", long_room, {"duration_quantile": "q50"})],
    )
    add_overlay_recipe(
        sample_id="ov03_vocal_over_two_core_safe",
        focus="definite-drop 呼吸/呻吟与两个 semantic cores 同时可听；不得把它当字幕语义，且不能遮坏清楚台词。",
        clean=clean,
        truth=truth,
        axes={"core_count": 2, "gap_axis": "room_under_vocal_overlay", "speaker_axis": "cross_source"},
        source_rows=[vocal],
        snr_quantile="q15",
    )

    clean, truth = _overlap_composite(
        sample_id="ov04_music_over_overlap_abstain",
        left=medium_left,
        right=medium_right,
        overlap_s=0.18,
    )
    add_overlay_recipe(
        sample_id="ov04_music_over_overlap_abstain",
        focus="两个 semantic cores 在 BGM 下发生重叠；Split 有语义 event，但 Inner 无安全区，必须 abstain。",
        clean=clean,
        truth=truth,
        axes={"core_count": 2, "gap_axis": "overlap", "speaker_axis": "cross_source"},
        source_rows=[music_a],
        snr_quantile="q65",
    )

    single = np.asarray(long_core["audio"], dtype=np.float32)
    truth = {
        "core_spans": [_core_copy(long_core, start_sample=0, end_sample=len(single))],
        "gap_spans": [],
        "semantic_events": [],
        "continue_control": True,
    }
    add_overlay_recipe(
        sample_id="ov05_bgm_switch_over_single_core_continue",
        focus="单一 maximal semantic core 与两种 BGM 同时可听；BGM crossfade 不得产生 Split event。",
        clean=single,
        truth=truth,
        axes={"core_count": 1, "gap_axis": "none", "speaker_axis": "single_source"},
        source_rows=[music_a, music_b],
        snr_quantile="q35",
        switch=True,
    )

    clean_dir = output_dir / "audio" / "clean"
    overlay_dir = output_dir / "audio" / "overlay"
    mixed_dir = output_dir / "audio" / "mixed"
    for directory in (clean_dir, overlay_dir, mixed_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for recipe in recipes:
        sample_id = str(recipe["sample_id"])
        clean_path = clean_dir / f"{sample_id}.wav"
        overlay_path = overlay_dir / f"{sample_id}.wav"
        mixed_path = mixed_dir / f"{sample_id}.wav"
        sf.write(str(clean_path), recipe["clean"], SAMPLE_RATE, subtype="PCM_16")
        sf.write(str(overlay_path), recipe["overlay_audio"], SAMPLE_RATE, subtype="PCM_16")
        sf.write(str(mixed_path), recipe["mixed"], SAMPLE_RATE, subtype="PCM_16")
        manifest_rows.append(
            {
                "schema": SCHEMA,
                "sample_id": sample_id,
                "clean_audio": str(clean_path),
                "overlay_audio": str(overlay_path),
                "mixed_audio": str(mixed_path),
                "sample_rate": SAMPLE_RATE,
                "sample_count": int(len(recipe["mixed"])),
                "duration_s": len(recipe["mixed"]) / SAMPLE_RATE,
                "audit_focus": recipe["audit_focus"],
                "sampling_axes": recipe["sampling_axes"],
                "semantic_core_source": str(semantic_labels),
                "overlay_manifest": str(overlay_manifest),
                "overlay": recipe["overlay"],
                **recipe["truth"],
            }
        )
    manifest_path = output_dir / "multicore_composite_smoke.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "seed": int(seed),
        "sample_count": len(manifest_rows),
        "core_count_values": sorted({int(row["sampling_axes"]["core_count"]) for row in manifest_rows}),
        "gap_axes": [str(row["sampling_axes"]["gap_axis"]) for row in manifest_rows],
        "semantic_event_count": sum(len(row["semantic_events"]) for row in manifest_rows),
        "inner_safe_count": sum(
            event["inner_target"]["status"] == "safe"
            for row in manifest_rows
            for event in row["semantic_events"]
        ),
        "inner_abstain_count": sum(
            event["inner_target"]["status"] == "abstain"
            for row in manifest_rows
            for event in row["semantic_events"]
        ),
        "continue_control_count": sum(bool(row["continue_control"]) for row in manifest_rows),
        "gap_duration_quantiles": quantiles,
        "snr_quantiles": snr_quantiles,
        "target_snr_quantiles": [row["sampling_axes"]["snr_quantile"] for row in manifest_rows],
        "overlay_selection": overlay_selection,
        "overlay_mode": "additive_full_duration",
        "all_semantic_cores_have_simultaneous_overlay": True,
        "semantic_core_library": str(core_library_path),
        "semantic_core_count": len(core_library_rows),
        "coordinate_contract": "dual_coordinates_float_seconds_model_interface_exact_sample_offsets_v1",
        "training_ready": False,
        "training_blocker": "fixed five-sample manual audit is required before proposer replay or dataset expansion",
        "manifest": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fixed five-sample semantic Split/Inner additive-overlay smoke."
    )
    parser.add_argument("--semantic-labels", required=True)
    parser.add_argument("--overlay-manifest", required=True)
    parser.add_argument("--gap-duration-pool", required=True)
    parser.add_argument("--snr-reference-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build_smoke(
                semantic_labels=Path(args.semantic_labels),
                overlay_manifest=Path(args.overlay_manifest),
                gap_duration_pool=Path(args.gap_duration_pool),
                snr_reference_manifest=Path(args.snr_reference_manifest),
                output_dir=Path(args.output_dir),
                seed=int(args.seed),
            ),
            ensure_ascii=False,
        )
    )
