#!/usr/bin/env python3
"""Build exact-composition train-only sources for Scorer v11.

The semantic composite itself is one continuous ``inside_candidate`` island,
including its short internal gaps, vocal negatives, and optional overlay. Only
clear non-vocal material outside that island is labelled ``outside_candidate``.
Isolated human vocal controls are also positive candidates and are left for
CueQC to drop later in the real workflow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from boundary.contracts import ACOUSTIC_BINARY_V12_CONTRACT  # noqa: E402
from boundary.ja.model import (  # noqa: E402
    CANDIDATE_ISLAND_SCORER_V11_SYNTHETIC_TRAIN_SOURCE_SCHEMA,
)
from audio.loading import load_audio_16k_mono  # noqa: E402
from tools.boundary.ja.build_galgame_synthetic_timeline import (  # noqa: E402
    crop_or_tile_audio,
)


SUMMARY_SCHEMA = "candidate_island_scorer_v11_train_source_build_summary_v1"
COMPOSITE_SCHEMA = "cueqc_v13_unique_core_composite_v1"
V10_CANONICAL_SCHEMA = "speech_scorer_v10_canonical_source_v1"
PARTITION_SCHEMA = "candidate_island_scorer_v11_partition_manifest_v1"
OUTSIDE_CONSENSUS_SCHEMA = "candidate_island_scorer_v11_outside_consensus_v1"
SAMPLE_RATE = 16000
FRAME_SAMPLES = 320
FRAME_HOP_S = 0.02
BRACKET_DURATIONS_S = (0.5, 1.0, 2.0, 3.0)
DEFAULT_OUTSIDE_CONTROL_COUNT = 320
OUTSIDE_CONTROL_SEGMENT_DURATIONS_S = (2.0, 3.0, 4.0, 5.0, 6.0)

OUTSIDE_BACKGROUND_TYPES = {
    "ambient_noise",
    "clicking",
    "environmental_noise",
    "environmental_noise+silence",
    "footsteps+train_noise",
    "heavy_machinery",
    "impact_sound",
    "movement",
    "music",
    "music_end",
    "music_start",
    "noise",
    "noise_only",
    "rustling",
    "short_noise",
    "short_silence",
    "silence",
    "vehicle_noise",
}
VOCAL_TOKENS = (
    "breath",
    "cry",
    "groan",
    "grunt",
    "kiss",
    "moan",
    "sob",
    "speech",
    "vocal",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def _stable_key(seed: int, value: str) -> tuple[str, str]:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(), value


def _load_audio(path: Path) -> np.ndarray:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if values.ndim > 1:
        values = np.mean(values, axis=1)
    if sample_rate != SAMPLE_RATE or not len(values):
        raise ValueError(f"Scorer v11 train source requires non-empty mono 16k audio: {path}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Scorer v11 train source contains non-finite audio: {path}")
    return np.ascontiguousarray(values, dtype=np.float32)


def _load_source_audio(path: Path) -> np.ndarray:
    values, sample_rate = load_audio_16k_mono(str(path))
    if sample_rate != SAMPLE_RATE or not len(values):
        raise ValueError(f"Scorer v11 source requires non-empty mono 16k audio: {path}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Scorer v11 source contains non-finite audio: {path}")
    return np.ascontiguousarray(values, dtype=np.float32)


def _normalize_video_id(value: str) -> str:
    normalized = str(value).removeprefix("scorer-v10-background-")
    normalized = normalized.removeprefix("preasr-")
    normalized = re.sub(r"-chunk\d+$", "", normalized)
    return re.sub(r"-w\d+$", "", normalized)


def _background_video_ids(row: dict[str, Any]) -> set[str]:
    values = {
        _normalize_video_id(str(value))
        for value in row.get("background_source_video_ids") or ()
        if str(value)
    }
    for key in ("background_id", "source_id"):
        value = str(row.get(key) or "")
        if value:
            values.add(_normalize_video_id(value))
    return {value for value in values if value}


def _preasr_video_id(audio_id: str) -> str:
    return _normalize_video_id(audio_id)


def _heldout_video_ids(partition_rows: Sequence[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for row in partition_rows:
        if row.get("schema") != PARTITION_SCHEMA:
            raise ValueError("wrong Scorer v11 partition manifest schema")
        if row.get("boundary_serialization_contract_id") != (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError("wrong central Boundary contract in partition manifest")
        if row.get("partition") in {"val", "test"}:
            video_id = _normalize_video_id(str(row.get("video_id") or ""))
            if not video_id:
                raise ValueError("held-out partition row has no video identity")
            result.add(video_id)
    if not result:
        raise ValueError("Scorer v11 train build requires frozen held-out video identities")
    return result


def _outside_consensus_ids(rows: Sequence[dict[str, Any]]) -> tuple[set[str], Counter[str]]:
    clear: set[str] = set()
    decisions: Counter[str] = Counter()
    seen: set[str] = set()
    for row in rows:
        if row.get("schema") != OUTSIDE_CONSENSUS_SCHEMA:
            raise ValueError("wrong Scorer v11 outside consensus schema")
        if row.get("boundary_serialization_contract_id") != (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError("outside consensus uses another Boundary contract")
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in seen:
            raise ValueError("outside consensus source identity is missing or duplicated")
        seen.add(source_id)
        decision = str(row.get("decision") or "")
        if decision not in {"clear_outside", "unsure"}:
            raise ValueError(f"invalid Scorer v11 outside consensus decision: {decision}")
        decisions[decision] += 1
        allowed = bool(row.get("training_manifest_allowed"))
        training_label = int(row.get("training_label"))
        if decision == "clear_outside":
            if not allowed or training_label != 0:
                raise ValueError("clear outside consensus must explicitly allow label 0")
            clear.add(source_id)
        elif allowed or training_label != -100:
            raise ValueError("unsure outside consensus must be ignore=-100")
    if not clear:
        raise ValueError("Scorer v11 outside consensus has no clear outside sources")
    return clear, decisions


def _background_pools(
    rows: Sequence[dict[str, Any]], *, heldout_video_ids: set[str], seed: int,
    clear_outside_source_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    outside: list[dict[str, Any]] = []
    vocal: list[dict[str, Any]] = []
    for row in rows:
        if row.get("schema") != V10_CANONICAL_SCHEMA:
            raise ValueError("wrong background inventory source schema")
        if row.get("boundary_serialization_contract_id") != (
            ACOUSTIC_BINARY_V12_CONTRACT.contract_id
        ):
            raise ValueError("background inventory uses another Boundary contract")
        if row.get("partition") != "train" or row.get("row_role") != "all_background":
            continue
        if _background_video_ids(row) & heldout_video_ids:
            continue
        audio = _resolve(str(row.get("audio") or ""))
        if not audio.exists():
            continue
        eligible.append(row)
        background_type = str(row.get("background_type") or "").strip().lower()
        flags = {
            background_type,
            *(str(value).strip().lower() for value in row.get("omni_flags") or ()),
        }
        has_vocal_flag = any(
            token in value for token in VOCAL_TOKENS for value in flags
        )
        if has_vocal_flag:
            vocal.append(row)
        elif (
            str(row["source_id"]) in clear_outside_source_ids
            and background_type in OUTSIDE_BACKGROUND_TYPES
        ):
            outside.append(row)
    outside.sort(key=lambda row: _stable_key(seed, str(row["source_id"])))
    vocal.sort(key=lambda row: _stable_key(seed + 1, str(row["source_id"])))
    eligible.sort(key=lambda row: _stable_key(seed + 2, str(row["source_id"])))
    if not outside:
        raise ValueError("no held-out-disjoint three-way-confirmed non-vocal outside sources")
    if not vocal:
        raise ValueError("no held-out-disjoint isolated vocal candidate sources")
    return outside, vocal, eligible


def _rms(audio: np.ndarray) -> float:
    values = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def _limit(audio: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.ascontiguousarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    gain = 0.98 / peak if peak > 0.98 else 1.0
    return np.ascontiguousarray(values * gain, dtype=np.float32), float(gain)


def _mix_overlay(
    clean: np.ndarray,
    overlay: np.ndarray,
    *,
    core_spans: Sequence[dict[str, Any]],
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


def _is_vocal_detail(detail: dict[str, Any]) -> bool:
    values = {
        str(detail.get("background_type") or "").strip().lower(),
        *(str(value).strip().lower() for value in detail.get("omni_flags") or ()),
    }
    return any(token in value for token in VOCAL_TOKENS for value in values)


def _replacement_row(
    detail: dict[str, Any],
    *,
    eligible_pool: Sequence[dict[str, Any]],
    selection_key: str,
) -> dict[str, Any]:
    background_type = str(detail.get("background_type") or "").strip().lower()
    same_type = [
        row
        for row in eligible_pool
        if str(row.get("background_type") or "").strip().lower() == background_type
    ]
    candidates = same_type
    if not candidates:
        candidates = [
            row for row in eligible_pool if _is_vocal_detail(row) == _is_vocal_detail(detail)
        ]
    if not candidates:
        candidates = list(eligible_pool)
    if not candidates:
        raise ValueError("Scorer v11 has no train-disjoint component replacement pool")
    digest = hashlib.sha256(selection_key.encode("utf-8")).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


def _clip_inside_component(
    detail: dict[str, Any],
    *,
    samples: int,
    heldout_video_ids: set[str],
    eligible_pool: Sequence[dict[str, Any]],
    selection_key: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    original_audio_id = str(detail.get("audio_id") or "")
    original_path = _resolve(str(detail.get("audio") or ""))
    original_video_id = _preasr_video_id(original_audio_id or original_path.stem)
    replaced = original_video_id in heldout_video_ids or not original_path.exists()
    selected: dict[str, Any]
    if replaced:
        selected = _replacement_row(
            detail, eligible_pool=eligible_pool, selection_key=selection_key
        )
        selected_path = _resolve(str(selected["audio"]))
        source = _load_audio(selected_path)
        clipped, offset = crop_or_tile_audio(source, samples=samples, rng=rng)
        selected_detail = {
            "audio_id": str(selected.get("background_id") or selected["source_id"]),
            "audio": _display(selected_path),
            "audio_sha256": _sha256(selected_path),
            "background_type": str(selected.get("background_type") or ""),
            "omni_flags": list(selected.get("omni_flags") or ()),
            "source_offset_sample": int(offset),
        }
    else:
        source = _load_audio(original_path)
        if len(source) >= samples:
            offset = min(
                max(0, int(round(float(detail.get("source_offset_s") or 0.0) * SAMPLE_RATE))),
                len(source) - samples,
            )
            clipped = np.ascontiguousarray(source[offset : offset + samples], dtype=np.float32)
        else:
            clipped, offset = crop_or_tile_audio(source, samples=samples, rng=rng)
        selected_detail = {
            "audio_id": original_audio_id or original_path.stem,
            "audio": _display(original_path),
            "audio_sha256": _sha256(original_path),
            "background_type": str(detail.get("background_type") or ""),
            "omni_flags": list(detail.get("omni_flags") or ()),
            "source_offset_sample": int(offset),
        }
    selected_video_id = _preasr_video_id(str(selected_detail["audio_id"]))
    if selected_video_id in heldout_video_ids:
        raise ValueError(
            "Scorer v11 component selection still references held-out video: "
            f"{selected_video_id}"
        )
    selected_detail["video_id"] = selected_video_id
    return np.ascontiguousarray(clipped, dtype=np.float32), {
        "heldout_component_replaced": replaced,
        "original_audio_id": original_audio_id,
        "original_video_id": original_video_id,
        "original_background_type": str(detail.get("background_type") or ""),
        "selected": selected_detail,
        "output_sample_count": int(len(clipped)),
    }


def _rebuild_semantic_candidate(
    source: dict[str, Any],
    *,
    heldout_video_ids: set[str],
    eligible_pool: Sequence[dict[str, Any]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    core_spans = list(source.get("core_spans") or ())
    if len(core_spans) != 2:
        raise ValueError("Scorer v11 fixed composite requires exactly two semantic cores")
    core_audio: list[np.ndarray] = []
    core_provenance: list[dict[str, Any]] = []
    for core in core_spans:
        path = _resolve(str(core.get("source_audio") or ""))
        values = _load_source_audio(path)
        expected = int(core["end_sample"]) - int(core["start_sample"])
        if len(values) != expected:
            raise ValueError(
                "Scorer v11 semantic core length differs from frozen composition: "
                f"core_id={core.get('core_id')}, expected={expected}, actual={len(values)}"
            )
        core_audio.append(values)
        core_provenance.append(
            {
                "core_id": str(core["core_id"]),
                "audio": _display(path),
                "audio_sha256": _sha256(path),
                "sample_count": len(values),
            }
        )

    gaps = dict(source.get("inter_unit_gaps") or {})
    gap_details = list(gaps.get("sources") or ())
    negative = dict(source.get("negative_unit_span") or {})
    if len(gap_details) != 2 or not negative.get("source"):
        raise ValueError("Scorer v11 fixed composite lacks exact internal components")
    left_samples = int(gaps["left_end_sample"]) - int(gaps["left_start_sample"])
    right_samples = int(gaps["right_end_sample"]) - int(gaps["right_start_sample"])
    negative_samples = int(negative["end_sample"]) - int(negative["start_sample"])
    sample_id = str(source["sample_id"])
    left, left_detail = _clip_inside_component(
        dict(gap_details[0]),
        samples=left_samples,
        heldout_video_ids=heldout_video_ids,
        eligible_pool=eligible_pool,
        selection_key=f"{sample_id}:left-gap",
        rng=rng,
    )
    unit, unit_detail = _clip_inside_component(
        dict(negative["source"]),
        samples=negative_samples,
        heldout_video_ids=heldout_video_ids,
        eligible_pool=eligible_pool,
        selection_key=f"{sample_id}:negative-unit",
        rng=rng,
    )
    right, right_detail = _clip_inside_component(
        dict(gap_details[1]),
        samples=right_samples,
        heldout_video_ids=heldout_video_ids,
        eligible_pool=eligible_pool,
        selection_key=f"{sample_id}:right-gap",
        rng=rng,
    )
    clean, clean_gain = _limit(
        np.concatenate((core_audio[0], left, unit, right, core_audio[1]))
    )
    rebuilt_spans = [
        {"start_sample": 0, "end_sample": len(core_audio[0])},
        {
            "start_sample": len(core_audio[0]) + left_samples + negative_samples + right_samples,
            "end_sample": len(clean),
        },
    ]
    overlay = source.get("additive_overlay")
    mixed = clean
    overlay_detail: dict[str, Any] | None = None
    if overlay:
        overlay_audio, overlay_source = _clip_inside_component(
            dict(overlay["source"]),
            samples=len(clean),
            heldout_video_ids=heldout_video_ids,
            eligible_pool=eligible_pool,
            selection_key=f"{sample_id}:overlay",
            rng=rng,
        )
        target_snr = float((overlay.get("mix") or {})["target_snr_db"])
        mixed, mix_detail = _mix_overlay(
            clean, overlay_audio, core_spans=rebuilt_spans, target_snr_db=target_snr
        )
        overlay_detail = {"source": overlay_source, "mix": mix_detail}
    if len(mixed) != int(source["sample_count"]):
        raise ValueError(
            "Scorer v11 rebuilt candidate changed frozen sample count: "
            f"sample_id={sample_id}, expected={source['sample_count']}, actual={len(mixed)}"
        )
    components = [left_detail, unit_detail, right_detail]
    if overlay_detail:
        components.append(dict(overlay_detail["source"]))
    return np.ascontiguousarray(mixed, dtype=np.float32), {
        "source_sample_id": sample_id,
        "original_composite_audio": str(source.get("audio") or ""),
        "original_composite_audio_sha256": _sha256(_resolve(str(source["audio"]))),
        "core_sources": core_provenance,
        "internal_components": {
            "left_gap": left_detail,
            "negative_unit": unit_detail,
            "right_gap": right_detail,
        },
        "overlay": overlay_detail,
        "clean_limiter_gain": clean_gain,
        "heldout_component_replacement_count": sum(
            bool(detail.get("heldout_component_replaced")) for detail in components
        ),
        "internal_gap_policy": "inside_candidate_preserve_continuity_v1",
    }


def _clip_background(
    row: dict[str, Any], *, samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    path = _resolve(str(row["audio"]))
    source = _load_audio(path)
    values, offset = crop_or_tile_audio(source, samples=samples, rng=rng)
    return np.ascontiguousarray(values, dtype=np.float32), {
        "source_id": str(row["source_id"]),
        "background_type": str(row.get("background_type") or ""),
        "audio": _display(path),
        "audio_sha256": _sha256(path),
        "source_offset_sample": int(offset),
        "output_sample_count": int(len(values)),
        "crop_or_tile": "tile" if samples > len(source) else "crop",
    }


def _candidate_spans(
    *, frame_count: int, left_samples: int, candidate_samples: int
) -> list[dict[str, Any]]:
    inside_start = left_samples // FRAME_SAMPLES
    inside_end = (left_samples + candidate_samples + FRAME_SAMPLES - 1) // FRAME_SAMPLES
    if not (0 < inside_start < inside_end < frame_count):
        raise ValueError("invalid Scorer v11 candidate bracket geometry")
    return [
        {"label": "outside_candidate", "start_frame": 0, "end_frame": inside_start},
        {
            "label": "inside_candidate",
            "start_frame": inside_start,
            "end_frame": inside_end,
        },
        {
            "label": "outside_candidate",
            "start_frame": inside_end,
            "end_frame": frame_count,
        },
    ]


def _write_candidate_source(
    *,
    source_id: str,
    source_kind: str,
    candidate: np.ndarray,
    candidate_provenance: dict[str, Any],
    core_ids: Sequence[str],
    outside_pool: Sequence[dict[str, Any]],
    output_path: Path,
    source_index: int,
    rng: np.random.Generator,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    candidate = np.ascontiguousarray(candidate, dtype=np.float32)
    if not len(candidate) or not np.all(np.isfinite(candidate)):
        raise ValueError("Scorer v11 candidate audio must be non-empty and finite")
    left_s = BRACKET_DURATIONS_S[source_index % len(BRACKET_DURATIONS_S)]
    right_s = BRACKET_DURATIONS_S[(source_index * 3 + 1) % len(BRACKET_DURATIONS_S)]
    left_samples = int(round(left_s * SAMPLE_RATE / FRAME_SAMPLES)) * FRAME_SAMPLES
    right_samples = int(round(right_s * SAMPLE_RATE / FRAME_SAMPLES)) * FRAME_SAMPLES
    right_samples += (-len(candidate)) % FRAME_SAMPLES
    left_row = outside_pool[(source_index * 2) % len(outside_pool)]
    right_row = outside_pool[(source_index * 2 + 1) % len(outside_pool)]
    left, left_provenance = _clip_background(left_row, samples=left_samples, rng=rng)
    right, right_provenance = _clip_background(right_row, samples=right_samples, rng=rng)
    audio = np.ascontiguousarray(np.concatenate((left, candidate, right)), dtype=np.float32)
    if len(audio) % FRAME_SAMPLES:
        raise ValueError("Scorer v11 synthetic train audio is not frame aligned")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, SAMPLE_RATE, subtype="PCM_16")
    written = _load_audio(output_path)
    if len(written) != len(audio):
        raise ValueError("Scorer v11 synthetic train write changed sample count")
    frame_count = len(written) // FRAME_SAMPLES
    return {
        "schema": CANDIDATE_ISLAND_SCORER_V11_SYNTHETIC_TRAIN_SOURCE_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "source_id": source_id,
        "partition": "train",
        "source_kind": source_kind,
        "synthetic_composite": True,
        "input_distribution": "train_exact_candidate_context_composite_v1",
        "audio": _display(output_path),
        "audio_sha256": _sha256(output_path),
        "sample_rate": SAMPLE_RATE,
        "sample_count": int(len(written)),
        "duration_s": len(written) / SAMPLE_RATE,
        "frame_count": frame_count,
        "frame_hop_s": FRAME_HOP_S,
        "core_ids": [str(value) for value in core_ids],
        "canonical_spans": _candidate_spans(
            frame_count=frame_count,
            left_samples=left_samples,
            candidate_samples=len(candidate),
        ),
        "candidate_sample_span": {
            "start_sample": left_samples,
            "end_sample": left_samples + len(candidate),
        },
        "candidate_source": {**candidate_provenance, "sample_count": int(len(candidate))},
        "outside_brackets": {
            "left": left_provenance,
            "right": right_provenance,
        },
        "composition_provenance": provenance,
        "training_manifest_allowed": True,
    }


def _write_outside_control(
    *,
    outside_pool: Sequence[dict[str, Any]],
    output_path: Path,
    source_index: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if not outside_pool:
        raise ValueError("Scorer v11 outside control pool is empty")
    components: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    for component_index, seconds in enumerate(OUTSIDE_CONTROL_SEGMENT_DURATIONS_S):
        row = outside_pool[
            (source_index * len(OUTSIDE_CONTROL_SEGMENT_DURATIONS_S) + component_index)
            % len(outside_pool)
        ]
        values, detail = _clip_background(
            row, samples=int(seconds * SAMPLE_RATE), rng=rng
        )
        components.append(values)
        provenance.append(detail)
    values = np.ascontiguousarray(np.concatenate(components), dtype=np.float32)
    samples = int(sum(OUTSIDE_CONTROL_SEGMENT_DURATIONS_S) * SAMPLE_RATE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, values, SAMPLE_RATE, subtype="PCM_16")
    written = _load_audio(output_path)
    if len(written) != samples or samples % FRAME_SAMPLES:
        raise ValueError("Scorer v11 outside control write changed frame geometry")
    source_id = f"scorer-v11-outside-control-{source_index:04d}"
    return {
        "schema": CANDIDATE_ISLAND_SCORER_V11_SYNTHETIC_TRAIN_SOURCE_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "source_id": source_id,
        "partition": "train",
        "source_kind": "clear_nonvocal_all_background",
        "synthetic_composite": True,
        "input_distribution": "train_exact_candidate_context_composite_v1",
        "audio": _display(output_path),
        "audio_sha256": _sha256(output_path),
        "sample_rate": SAMPLE_RATE,
        "sample_count": int(len(written)),
        "duration_s": len(written) / SAMPLE_RATE,
        "frame_count": len(written) // FRAME_SAMPLES,
        "frame_hop_s": FRAME_HOP_S,
        "core_ids": [f"background-control-instance::{source_id}"],
        "canonical_spans": [
            {
                "label": "outside_candidate",
                "start_frame": 0,
                "end_frame": len(written) // FRAME_SAMPLES,
            }
        ],
        "outside_control_sources": provenance,
        "outside_control_composition": "train_nonvocal_mosaic_20s_v1",
        "training_manifest_allowed": True,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest = Path(args.source_manifest).resolve()
    background_inventory = Path(args.background_inventory).resolve()
    partition_manifest = Path(args.heldout_partition_manifest).resolve()
    outside_consensus_manifest = Path(args.outside_consensus_manifest).resolve()
    for path in (
        source_manifest,
        background_inventory,
        partition_manifest,
        outside_consensus_manifest,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    source_manifest_sha = _sha256(source_manifest)
    background_inventory_sha = _sha256(background_inventory)
    partition_manifest_sha = _sha256(partition_manifest)
    outside_consensus_manifest_sha = _sha256(outside_consensus_manifest)
    composites = _read_jsonl(source_manifest)
    train_composites = [row for row in composites if row.get("source_partition") == "train"]
    if not train_composites:
        raise ValueError("Scorer v11 train source manifest has no train composites")
    if any(row.get("schema") != COMPOSITE_SCHEMA for row in train_composites):
        raise ValueError("wrong Scorer v11 semantic composite input schema")
    max_semantic_sources = args.max_semantic_sources
    if max_semantic_sources is not None:
        max_semantic_sources = int(max_semantic_sources)
        if max_semantic_sources <= 0:
            raise ValueError("--max-semantic-sources must be positive")
        train_composites = train_composites[:max_semantic_sources]
    core_ids = [
        str(core.get("core_id") or "")
        for row in train_composites
        for core in row.get("core_spans") or ()
    ]
    if not core_ids or any(not value for value in core_ids) or len(set(core_ids)) != len(core_ids):
        raise ValueError("Scorer v11 semantic train cores must be non-empty and unique")
    heldout_ids = _heldout_video_ids(_read_jsonl(partition_manifest))
    clear_outside_source_ids, outside_consensus_decisions = _outside_consensus_ids(
        _read_jsonl(outside_consensus_manifest)
    )
    outside_pool, vocal_pool, eligible_pool = _background_pools(
        _read_jsonl(background_inventory),
        heldout_video_ids=heldout_ids,
        seed=int(args.seed),
        clear_outside_source_ids=clear_outside_source_ids,
    )
    vocal_source_count = int(args.vocal_source_count)
    outside_control_count = int(args.outside_control_count)
    if vocal_source_count < 0 or outside_control_count < 0:
        raise ValueError("Scorer v11 train source counts must be non-negative")
    if len(vocal_pool) < vocal_source_count:
        raise ValueError("not enough held-out-disjoint isolated vocal sources")
    if outside_control_count and not outside_pool:
        raise ValueError("no held-out-disjoint clear non-vocal controls")

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, Any]] = []
    overlay_counts: Counter[str] = Counter()
    heldout_component_replacements = 0
    for index, source in enumerate(train_composites):
        candidate, candidate_provenance = _rebuild_semantic_candidate(
            source,
            heldout_video_ids=heldout_ids,
            eligible_pool=eligible_pool,
            rng=rng,
        )
        heldout_component_replacements += int(
            candidate_provenance["heldout_component_replacement_count"]
        )
        overlay = source.get("additive_overlay")
        overlay_counts["overlay" if overlay else "clean"] += 1
        rows.append(
            _write_candidate_source(
                source_id=f"scorer-v11-semantic-{source['sample_id']}",
                source_kind="semantic_composite_candidate",
                candidate=candidate,
                candidate_provenance=candidate_provenance,
                core_ids=[str(core["core_id"]) for core in source["core_spans"]],
                outside_pool=outside_pool,
                output_path=audio_dir / f"semantic-{index:04d}.wav",
                source_index=index,
                rng=rng,
                provenance={
                    "source_manifest": _display(source_manifest),
                    "source_manifest_sha256": source_manifest_sha,
                    "source_row_schema": COMPOSITE_SCHEMA,
                    "source_sample_id": str(source["sample_id"]),
                    "internal_gap_policy": "inside_candidate_preserve_continuity_v1",
                    "additive_overlay": overlay,
                },
            )
        )

    vocal_selection = vocal_pool[:vocal_source_count]
    for local_index, source in enumerate(vocal_selection):
        global_index = len(rows)
        vocal_path = _resolve(str(source["audio"]))
        rows.append(
            _write_candidate_source(
                source_id=f"scorer-v11-isolated-vocal-{local_index:04d}",
                source_kind="isolated_human_vocal_candidate",
                candidate=_load_audio(vocal_path),
                candidate_provenance={
                    "audio": _display(vocal_path),
                    "audio_sha256": _sha256(vocal_path),
                    "inventory_source_id": str(source["source_id"]),
                },
                core_ids=[f"isolated-vocal::{source['source_id']}"],
                outside_pool=outside_pool,
                output_path=audio_dir / f"isolated-vocal-{local_index:04d}.wav",
                source_index=global_index,
                rng=rng,
                provenance={
                    "background_inventory": _display(background_inventory),
                    "background_inventory_sha256": background_inventory_sha,
                    "inventory_source_id": str(source["source_id"]),
                    "inventory_background_type": str(source.get("background_type") or ""),
                    "duty_mapping": "human_vocal_inside_candidate_then_cueqc_v1",
                },
            )
        )

    for local_index in range(outside_control_count):
        rows.append(
            _write_outside_control(
                outside_pool=outside_pool,
                output_path=audio_dir / f"outside-control-{local_index:04d}.wav",
                source_index=local_index,
                rng=rng,
            )
        )

    seen_core: dict[str, str] = {}
    frame_counts: Counter[str] = Counter()
    source_kind_counts: Counter[str] = Counter()
    for row in rows:
        source_kind_counts[str(row["source_kind"])] += 1
        for span in row["canonical_spans"]:
            frame_counts[str(span["label"])] += int(span["end_frame"]) - int(
                span["start_frame"]
            )
        for core_id in row["core_ids"]:
            previous = seen_core.setdefault(str(core_id), str(row["source_id"]))
            if previous != str(row["source_id"]):
                raise ValueError(f"Scorer v11 train core is reused: {core_id}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "synthetic_train_sources.jsonl"
    _write_jsonl(manifest_path, rows)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "boundary_serialization_contract_id": ACOUSTIC_BINARY_V12_CONTRACT.contract_id,
        "source_manifest": _display(source_manifest),
        "source_manifest_sha256": source_manifest_sha,
        "background_inventory": _display(background_inventory),
        "background_inventory_sha256": background_inventory_sha,
        "heldout_partition_manifest": _display(partition_manifest),
        "heldout_partition_manifest_sha256": partition_manifest_sha,
        "outside_consensus_manifest": _display(outside_consensus_manifest),
        "outside_consensus_manifest_sha256": outside_consensus_manifest_sha,
        "outside_consensus_decision_counts": dict(sorted(outside_consensus_decisions.items())),
        "outside_omni_only_truth_allowed": False,
        "heldout_video_ids": sorted(heldout_ids),
        "source_count": len(rows),
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "semantic_core_count": len(core_ids),
        "unique_core_identity_count": len(seen_core),
        "max_core_use_count": 1,
        "overlay_counts": dict(sorted(overlay_counts.items())),
        "heldout_component_replacement_count": heldout_component_replacements,
        "outside_background_pool_count": len(outside_pool),
        "outside_control_source_reuse_allowed": True,
        "outside_control_duration_s": float(
            sum(OUTSIDE_CONTROL_SEGMENT_DURATIONS_S)
        ),
        "outside_control_component_count": len(
            OUTSIDE_CONTROL_SEGMENT_DURATIONS_S
        ),
        "isolated_vocal_pool_count": len(vocal_pool),
        "train_disjoint_component_pool_count": len(eligible_pool),
        "canonical_frame_counts": dict(sorted(frame_counts.items())),
        "synthetic_train_sources": _display(manifest_path),
        "synthetic_train_sources_sha256": _sha256(manifest_path),
        "synthetic_composites_train_only": True,
        "heldout_audio_used": False,
        "training_manifest_allowed": True,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--background-inventory", required=True)
    parser.add_argument("--heldout-partition-manifest", required=True)
    parser.add_argument("--outside-consensus-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocal-source-count", type=int, default=256)
    parser.add_argument(
        "--outside-control-count", type=int, default=DEFAULT_OUTSIDE_CONTROL_COUNT
    )
    parser.add_argument("--max-semantic-sources", type=int)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
