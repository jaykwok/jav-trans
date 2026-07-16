#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.loading import load_audio_16k_mono  # noqa: E402
from boundary.ja import TeacherSegment, build_supervised_record, write_jsonl  # noqa: E402


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"manifest must be a JSON list or JSONL: {path}")
        return [dict(row) for row in payload if isinstance(row, Mapping)]

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if not isinstance(row, Mapping):
            raise ValueError(f"manifest JSONL row must be an object: {path}:{line_number}")
        rows.append(dict(row))
    return rows


def valid_source_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid = []
    skipped = []
    for index, row in enumerate(rows):
        if row.get("error"):
            skipped.append({"index": index, "reason": "source_error", "error": str(row.get("error"))})
            continue
        audio = row.get("audio")
        if not audio:
            skipped.append({"index": index, "reason": "missing_audio"})
            continue
        audio_path = Path(str(audio))
        if not audio_path.exists():
            skipped.append({"index": index, "reason": "audio_not_found", "audio": str(audio_path)})
            continue
        valid.append(row)
    return valid, skipped


def require_source_schema(
    rows: list[dict[str, Any]], required_schema: str | None
) -> None:
    if not required_schema:
        return
    incompatible = sum(str(row.get("schema") or "") != required_schema for row in rows)
    if incompatible:
        raise ValueError(
            f"source manifest has {incompatible} rows outside required schema "
            f"{required_schema!r}"
        )


def summarize_source_usage(detail_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(source.get("source_audio_id") or "")
        for row in detail_rows
        for source in row.get("sources") or []
        if str(source.get("source_audio_id") or "")
    )
    return {
        "source_core_use_count": sum(counts.values()),
        "unique_source_core_count": len(counts),
        "max_source_core_use_count": max(counts.values(), default=0),
    }


def load_valid_manifest_rows(paths: list[str] | None, *, role: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for manifest_path in paths or []:
        path = Path(manifest_path)
        rows, row_skipped = valid_source_rows(load_manifest_rows(path))
        for row in rows:
            updated = dict(row)
            updated["_manifest"] = str(path)
            valid.append(updated)
        for row in row_skipped:
            updated = dict(row)
            updated["manifest"] = str(path)
            updated["role"] = role
            skipped.append(updated)
    return valid, skipped


def load_excluded_source_audio_ids(paths: list[str] | None) -> set[str]:
    excluded: set[str] = set()
    for manifest_path in paths or []:
        for row in load_manifest_rows(Path(manifest_path)):
            metadata = dict(row.get("boundary_metadata") or {})
            excluded.update(
                str(value)
                for value in (
                    row.get("source_audio_ids")
                    or metadata.get("source_audio_ids")
                    or []
                )
                if str(value)
            )
    return excluded


def crop_or_tile_audio(
    audio: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    if samples <= 0:
        return np.zeros(0, dtype=np.float32), 0
    if audio.size <= 0:
        raise ValueError("audio is empty")
    values = np.ascontiguousarray(audio, dtype=np.float32)
    if values.size >= samples:
        offset = int(rng.integers(0, values.size - samples + 1)) if values.size > samples else 0
        return np.ascontiguousarray(values[offset : offset + samples], dtype=np.float32), offset
    repeats = int(math.ceil(samples / values.size))
    tiled = np.tile(values, repeats)[:samples]
    return np.ascontiguousarray(tiled, dtype=np.float32), 0


def load_negative_audio(
    row: Mapping[str, Any],
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    audio_path = Path(str(row["audio"]))
    audio, sample_rate = load_audio_16k_mono(str(audio_path))
    if sample_rate != 16000:
        raise ValueError(f"expected 16kHz audio after normalization, got {sample_rate}")
    clipped, offset = crop_or_tile_audio(audio, samples=samples, rng=rng)
    detail = {
        "audio_id": str(row.get("audio_id") or audio_path.stem),
        "audio": str(audio_path),
        "source": str(row.get("source") or "negative"),
        "background_type": str(row.get("source") or "negative"),
        "manifest": str(row.get("_manifest") or ""),
        "source_offset_s": offset / sample_rate,
        "duration_s": clipped.size / sample_rate,
    }
    return clipped, detail


def clipped_speech_audio(
    row: Mapping[str, Any],
    *,
    rng: np.random.Generator,
    trim_head_s: float,
    trim_tail_s: float,
    max_speech_s: float,
    min_speech_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    audio_path = Path(str(row["audio"]))
    audio, sample_rate = load_audio_16k_mono(str(audio_path))
    if sample_rate != 16000:
        raise ValueError(f"expected 16kHz audio after normalization, got {sample_rate}")
    start_sample = max(0, int(round(max(0.0, trim_head_s) * sample_rate)))
    end_sample = max(start_sample, len(audio) - int(round(max(0.0, trim_tail_s) * sample_rate)))
    trimmed = np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)
    if trimmed.size < max(1, int(round(min_speech_s * sample_rate))):
        raise ValueError("speech clip is shorter than --min-speech-s after trim")
    if max_speech_s > 0.0:
        max_samples = max(1, int(round(max_speech_s * sample_rate)))
        if trimmed.size > max_samples:
            offset = int(rng.integers(0, trimmed.size - max_samples + 1))
            trimmed = np.ascontiguousarray(trimmed[offset : offset + max_samples], dtype=np.float32)
            start_sample += offset
            end_sample = start_sample + max_samples
    return trimmed, {
        "source_audio_id": str(row.get("audio_id") or audio_path.stem),
        "source_audio": str(audio_path),
        "source_input": str(row.get("input") or ""),
        "source_text": str(row.get("text") or ""),
        "source_start_s": start_sample / sample_rate,
        "source_end_s": end_sample / sample_rate,
        "duration_s": trimmed.size / sample_rate,
    }


def choose_speech_row(
    *,
    source_rows: list[dict[str, Any]],
    source_cursor: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
    used_source_indices: set[int],
) -> tuple[dict[str, Any] | None, int, int | None]:
    if not source_rows:
        return None, source_cursor, None
    if not args.randomize_speech_order:
        if source_cursor >= len(source_rows):
            if not args.reuse_sources:
                return None, source_cursor, None
            source_cursor = 0
            if args.shuffle:
                rng.shuffle(source_rows)
        row = source_rows[source_cursor]
        return row, source_cursor + 1, source_cursor

    available_indices = [
        index
        for index in range(len(source_rows))
        if args.reuse_sources or index not in used_source_indices
    ]
    if not available_indices:
        return None, source_cursor, None

    selected_index = int(available_indices[int(rng.integers(0, len(available_indices)))])
    return source_rows[selected_index], source_cursor, selected_index


def speech_group_key(row: Mapping[str, Any]) -> str:
    """Identity of the source recording a speech unit came from."""

    return str(row.get("input") or row.get("source_video") or row.get("audio_id") or "")


def choose_same_source_row(
    *,
    source_rows: list[dict[str, Any]],
    previous_group_key: str,
    used_source_indices: set[int],
    reuse_sources: bool,
    rng: np.random.Generator,
) -> tuple[dict[str, Any] | None, int | None]:
    """Pick the next speech unit from the SAME source recording when possible.

    In the real domain consecutive utterances usually share a speaker/channel;
    without this every hardmix boundary co-occurs with a recording change.
    """

    if not previous_group_key:
        return None, None
    indices = [
        index
        for index, row in enumerate(source_rows)
        if speech_group_key(row) == previous_group_key
        and (reuse_sources or index not in used_source_indices)
    ]
    if not indices:
        return None, None
    selected = int(indices[int(rng.integers(0, len(indices)))])
    return source_rows[selected], selected


def stable_source_partition(
    row: Mapping[str, Any],
    *,
    train_ratio: float,
    val_ratio: float,
) -> str:
    identity = str(row.get("audio_id") or row.get("audio") or row.get("input") or "")
    fraction = int(hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12], 16) / float(
        16**12
    )
    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + val_ratio:
        return "val"
    return "test"


def boundary_type_from_gap(
    gap_s: float,
    *,
    cut_point_max_gap_s: float,
    long_gap_boundary_min_s: float,
) -> str:
    if gap_s <= cut_point_max_gap_s:
        return "cut_point"
    if gap_s >= long_gap_boundary_min_s:
        return "long_gap"
    return "ambiguous_gap"


def choose_real_negative_gap(
    *,
    rows: list[dict[str, Any]],
    samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not rows:
        raise ValueError("no negative rows available")
    first_index = int(rng.integers(0, len(rows)))
    errors: list[str] = []
    for attempt in range(min(len(rows), 8)):
        index = (first_index + attempt) % len(rows)
        row = rows[index]
        try:
            audio, detail = load_negative_audio(row, samples=samples, rng=rng)
            detail["row_index"] = index
            return audio, detail
        except Exception as exc:
            errors.append(f"{index}:{exc}")
    raise ValueError("; ".join(errors) or "failed to load real negative gap")


def synthesize_gap(
    *,
    samples: int,
    sample_rate: int,
    index: int,
    rng: np.random.Generator,
    noise_rms: float,
    hum_rms: float,
    randomize_mode: bool = False,
) -> tuple[np.ndarray, str]:
    if samples <= 0:
        return np.zeros(0, dtype=np.float32), "empty"
    mode = int(rng.integers(0, 4)) if randomize_mode else index % 4
    if mode == 0:
        return np.zeros(samples, dtype=np.float32), "silence"
    if mode == 1:
        return rng.normal(0.0, max(0.0, noise_rms), samples).astype(np.float32), "white_noise"
    if mode == 2:
        time = np.arange(samples, dtype=np.float32) / sample_rate
        frequency = 80.0 + float(rng.integers(0, 80))
        return (max(0.0, hum_rms) * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32), "hum"
    base = rng.normal(0.0, max(0.0, noise_rms), samples).astype(np.float32)
    envelope = np.linspace(0.1, 1.0, samples, dtype=np.float32)
    return (base * envelope).astype(np.float32), "fade_noise"


def build_gap(
    *,
    samples: int,
    sample_rate: int,
    index: int,
    rng: np.random.Generator,
    noise_rms: float,
    hum_rms: float,
    negative_rows: list[dict[str, Any]],
    negative_gap_prob: float,
    randomize_synthetic_mode: bool = False,
    preferred_flags: list[str] | None = None,
    preferred_flag_prob: float = 0.0,
) -> tuple[np.ndarray, str, dict[str, Any] | None]:
    if samples <= 0:
        return np.zeros(0, dtype=np.float32), "empty", None
    if negative_rows and negative_gap_prob > 0.0 and float(rng.random()) < negative_gap_prob:
        candidate_rows = negative_rows
        preferred_applied = False
        # Positional prior: right after speech, breath/moan-like drops dominate
        # in the real domain; music/mechanical beds fill long runs.
        if (
            preferred_flags
            and preferred_flag_prob > 0.0
            and float(rng.random()) < preferred_flag_prob
        ):
            filtered = [
                row
                for row in negative_rows
                if any(
                    flag in str(row.get("background_type") or "")
                    for flag in preferred_flags
                )
            ]
            if filtered:
                candidate_rows = filtered
                preferred_applied = True
        try:
            audio, detail = choose_real_negative_gap(rows=candidate_rows, samples=samples, rng=rng)
            detail["preferred_flags_applied"] = preferred_applied
            return audio, "real_negative", detail
        except Exception as exc:
            detail = {"error": str(exc), "fallback": "synthetic"}
    else:
        detail = None
    audio, mode = synthesize_gap(
        samples=samples,
        sample_rate=sample_rate,
        index=index,
        rng=rng,
        noise_rms=noise_rms,
        hum_rms=hum_rms,
        randomize_mode=randomize_synthetic_mode,
    )
    return audio, mode, detail


def expand_segments(
    segments: list[TeacherSegment],
    *,
    duration_s: float,
    dilation_s: float,
) -> list[TeacherSegment]:
    if dilation_s <= 0.0:
        return list(segments)
    expanded = [
        TeacherSegment(
            start=max(0.0, segment.start - dilation_s),
            end=min(duration_s, segment.end + dilation_s),
            score=segment.score,
        )
        for segment in segments
    ]
    merged: list[TeacherSegment] = []
    for segment in sorted(expanded, key=lambda item: (item.start, item.end)):
        if segment.end <= segment.start:
            continue
        if not merged or segment.start > merged[-1].end:
            merged.append(segment)
        else:
            previous = merged[-1]
            merged[-1] = TeacherSegment(
                start=previous.start,
                end=max(previous.end, segment.end),
                score=previous.score if previous.score is not None else segment.score,
            )
    return merged


def align_unit_loudness(
    audio: np.ndarray,
    *,
    rng: np.random.Generator,
    target_rms: float,
    jitter_db: float,
    max_scale_db: float = 20.0,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Scale one timeline unit toward the example-level loudness target.

    Adjacent hardmix units come from different recordings, so without this every
    boundary co-occurs with a loudness step the model can exploit as a shortcut.
    """

    if audio.size <= 0 or target_rms <= 0.0:
        return audio, None
    unit_rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64, copy=False)))))
    if unit_rms <= 1e-6:
        return audio, None
    jitter = float(rng.uniform(-jitter_db, jitter_db)) if jitter_db > 0.0 else 0.0
    scale_db = 20.0 * float(np.log10(target_rms / unit_rms)) + jitter
    scale_db = float(np.clip(scale_db, -max_scale_db, max_scale_db))
    scale = 10.0 ** (scale_db / 20.0)
    scaled = np.clip(audio * scale, -1.0, 1.0).astype(np.float32, copy=False)
    return scaled, {
        "unit_rms": unit_rms,
        "target_rms": target_rms,
        "scale_db": scale_db,
    }


def empirical_gap_samples(
    *,
    rng: np.random.Generator,
    sample_rate: int,
    duration_pool_s: list[float],
    min_s: float,
    max_s: float,
    jitter_ratio: float = 0.15,
) -> int:
    """Sample a gap duration from a real-domain pause pool (with jitter)."""

    if not duration_pool_s:
        raise ValueError("duration_pool_s must not be empty")
    value = float(duration_pool_s[int(rng.integers(0, len(duration_pool_s)))])
    if jitter_ratio > 0.0:
        value *= float(rng.uniform(1.0 - jitter_ratio, 1.0 + jitter_ratio))
    value = min(max(value, max(0.0, min_s)), max(min_s, max_s))
    return max(0, int(round(value * sample_rate)))


def mix_background_audio(
    audio: np.ndarray,
    *,
    background_rows: list[dict[str, Any]],
    rng: np.random.Generator,
    snr_db_min: float,
    snr_db_max: float,
    switch_prob: float = 0.0,
    speech_segments: list[TeacherSegment] | None = None,
    sample_rate: int = 16000,
    switch_crossfade_s: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if audio.size <= 0:
        return audio, {"skipped": "empty_audio"}
    background_row_index = int(rng.integers(0, len(background_rows)))
    background, detail = load_negative_audio(
        background_rows[background_row_index],
        samples=audio.size,
        rng=rng,
    )
    switch_detail: dict[str, Any] | None = None
    # A mid-example bed switch placed INSIDE a speech unit teaches the model
    # that a background change is not a semantic boundary.
    if (
        switch_prob > 0.0
        and speech_segments
        and len(background_rows) > 1
        and float(rng.random()) < switch_prob
    ):
        fade_samples = max(1, int(round(switch_crossfade_s * sample_rate)))
        eligible = [
            segment
            for segment in speech_segments
            if (segment.end - segment.start) * sample_rate > fade_samples * 1.5
        ]
        if eligible:
            segment = eligible[int(rng.integers(0, len(eligible)))]
            margin_s = switch_crossfade_s * 0.75
            lower = segment.start + margin_s
            upper = max(lower, segment.end - margin_s)
            switch_s = float(rng.uniform(lower, upper)) if upper > lower else lower
            switch_sample = int(round(switch_s * sample_rate))
            second_index = int(rng.integers(0, len(background_rows)))
            if second_index == background_row_index:
                second_index = (second_index + 1) % len(background_rows)
            second, second_detail = load_negative_audio(
                background_rows[second_index],
                samples=audio.size,
                rng=rng,
            )
            fade_start = max(0, switch_sample - fade_samples // 2)
            fade_end = min(audio.size, fade_start + fade_samples)
            if fade_end > fade_start:
                ramp = np.linspace(0.0, 1.0, fade_end - fade_start, dtype=np.float32)
                merged = background.copy()
                merged[fade_start:fade_end] = (
                    background[fade_start:fade_end] * np.cos(ramp * np.pi / 2.0)
                    + second[fade_start:fade_end] * np.sin(ramp * np.pi / 2.0)
                )
                merged[fade_end:] = second[fade_end:]
                background = merged
                switch_detail = {
                    "switch_s": switch_s,
                    "crossfade_s": (fade_end - fade_start) / sample_rate,
                    "second_row_index": second_index,
                    "second_audio_id": second_detail.get("audio_id"),
                    "inside_speech": True,
                }
    snr_db = float(rng.uniform(snr_db_min, snr_db_max)) if snr_db_max > snr_db_min else float(snr_db_max)
    signal_rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64, copy=False)))))
    noise_rms = float(np.sqrt(np.mean(np.square(background.astype(np.float64, copy=False)))))
    if signal_rms <= 1e-8 or noise_rms <= 1e-8:
        return audio, {"skipped": "low_rms", "snr_db": snr_db, **detail}
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    scaled = background * (target_noise_rms / max(noise_rms, 1e-12))
    mixed = np.clip(audio + scaled, -1.0, 1.0).astype(np.float32, copy=False)
    detail.update(
        {
            "row_index": background_row_index,
            "snr_db": snr_db,
            "signal_rms": signal_rms,
            "background_rms": noise_rms,
            "target_background_rms": target_noise_rms,
        }
    )
    if switch_detail:
        detail["switch"] = switch_detail
    return mixed, detail


def apply_random_gain(
    audio: np.ndarray,
    *,
    rng: np.random.Generator,
    gain_db_min: float,
    gain_db_max: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if audio.size <= 0:
        return audio, {"skipped": "empty_audio"}
    if gain_db_min == 0.0 and gain_db_max == 0.0:
        return audio, {"enabled": False}
    gain_db = float(rng.uniform(gain_db_min, gain_db_max)) if gain_db_max > gain_db_min else float(gain_db_max)
    scale = 10.0 ** (gain_db / 20.0)
    peak_before = float(np.max(np.abs(audio))) if audio.size else 0.0
    scaled = audio.astype(np.float32, copy=False) * scale
    peak_scaled = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    peak_limit_scale = 1.0
    if peak_scaled > 0.99:
        peak_limit_scale = 0.99 / max(peak_scaled, 1e-12)
        scaled = scaled * peak_limit_scale
    return scaled.astype(np.float32, copy=False), {
        "enabled": True,
        "gain_db": gain_db,
        "scale": scale,
        "peak_before": peak_before,
        "peak_scaled": peak_scaled,
        "peak_limit_scale": peak_limit_scale,
    }


def apply_filter_aug(
    audio: np.ndarray,
    *,
    sample_rate: int,
    rng: np.random.Generator,
    filter_prob: float,
    filter_mode: str,
    lowpass_min_hz: float,
    lowpass_max_hz: float,
    bandpass_low_min_hz: float,
    bandpass_low_max_hz: float,
    bandpass_high_min_hz: float,
    bandpass_high_max_hz: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if audio.size <= 0:
        return audio, {"skipped": "empty_audio"}
    if filter_prob <= 0.0 or filter_mode == "none" or float(rng.random()) >= filter_prob:
        return audio, {"enabled": False}
    try:
        from scipy import signal
    except Exception as exc:
        return audio, {"enabled": False, "skipped": "scipy_unavailable", "error": str(exc)}

    mode = filter_mode
    if mode == "random":
        mode = "lowpass" if float(rng.random()) < 0.5 else "bandpass"
    nyquist = sample_rate / 2.0
    try:
        if mode == "lowpass":
            cutoff = float(rng.uniform(lowpass_min_hz, lowpass_max_hz))
            cutoff = min(max(100.0, cutoff), nyquist - 100.0)
            sos = signal.butter(4, cutoff / nyquist, btype="lowpass", output="sos")
            filtered = signal.sosfilt(sos, audio).astype(np.float32, copy=False)
            return filtered, {"enabled": True, "mode": mode, "cutoff_hz": cutoff}
        if mode == "bandpass":
            low = float(rng.uniform(bandpass_low_min_hz, bandpass_low_max_hz))
            high = float(rng.uniform(bandpass_high_min_hz, bandpass_high_max_hz))
            low = min(max(20.0, low), nyquist - 200.0)
            high = min(max(low + 100.0, high), nyquist - 50.0)
            sos = signal.butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
            filtered = signal.sosfilt(sos, audio).astype(np.float32, copy=False)
            return filtered, {"enabled": True, "mode": mode, "low_hz": low, "high_hz": high}
    except Exception as exc:
        return audio, {"enabled": False, "skipped": "filter_error", "error": str(exc), "mode": mode}
    return audio, {"enabled": False, "skipped": "unsupported_mode", "mode": mode}


def apply_codec_aug(
    audio: np.ndarray,
    *,
    rng: np.random.Generator,
    codec_prob: float,
    codec_aug: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if audio.size <= 0:
        return audio, {"skipped": "empty_audio"}
    if codec_prob <= 0.0 or codec_aug == "none" or float(rng.random()) >= codec_prob:
        return audio, {"enabled": False}
    mode = codec_aug
    if mode == "random":
        mode = "pcm16" if float(rng.random()) < 0.5 else "mulaw"
    values = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    if mode == "pcm16":
        quantized = np.round(values * 32767.0) / 32767.0
        return quantized.astype(np.float32, copy=False), {"enabled": True, "mode": mode}
    if mode == "mulaw":
        mu = 255.0
        encoded = np.sign(values) * np.log1p(mu * np.abs(values)) / np.log1p(mu)
        quantized = np.round((encoded + 1.0) * 127.5) / 127.5 - 1.0
        decoded = np.sign(quantized) * ((1.0 + mu) ** np.abs(quantized) - 1.0) / mu
        return decoded.astype(np.float32, copy=False), {"enabled": True, "mode": mode, "mu": mu}
    return audio, {"enabled": False, "skipped": "unsupported_mode", "mode": mode}


def choose_overlap_speech(
    *,
    rows: list[dict[str, Any]],
    rng: np.random.Generator,
    trim_head_s: float,
    trim_tail_s: float,
    max_speech_s: float,
    min_speech_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not rows:
        raise ValueError("no speech rows available")
    first_index = int(rng.integers(0, len(rows)))
    errors: list[str] = []
    for attempt in range(min(len(rows), 8)):
        index = (first_index + attempt) % len(rows)
        row = rows[index]
        try:
            audio, detail = clipped_speech_audio(
                row,
                rng=rng,
                trim_head_s=trim_head_s,
                trim_tail_s=trim_tail_s,
                max_speech_s=max_speech_s,
                min_speech_s=min_speech_s,
            )
            detail["row_index"] = index
            return audio, detail
        except Exception as exc:
            errors.append(f"{index}:{exc}")
    raise ValueError("; ".join(errors) or "failed to load overlap speech")


def mix_overlap_speech(
    audio: np.ndarray,
    *,
    speech_rows: list[dict[str, Any]],
    speech_segments: list[TeacherSegment],
    rng: np.random.Generator,
    sample_rate: int,
    probability: float,
    snr_db_min: float,
    snr_db_max: float,
    trim_head_s: float,
    trim_tail_s: float,
    max_speech_s: float,
    min_speech_s: float,
) -> tuple[np.ndarray, list[TeacherSegment], list[dict[str, Any]], dict[str, Any]]:
    if audio.size <= 0 or probability <= 0.0 or not speech_rows or float(rng.random()) >= probability:
        return audio, speech_segments, [], {"enabled": False}
    try:
        overlap_audio, detail = choose_overlap_speech(
            rows=speech_rows,
            rng=rng,
            trim_head_s=trim_head_s,
            trim_tail_s=trim_tail_s,
            max_speech_s=max_speech_s,
            min_speech_s=min_speech_s,
        )
    except Exception as exc:
        return audio, speech_segments, [], {"enabled": False, "skipped": "speech_load_error", "error": str(exc)}
    if overlap_audio.size <= 0:
        return audio, speech_segments, [], {"enabled": False, "skipped": "empty_overlap"}
    if overlap_audio.size > audio.size:
        overlap_audio = overlap_audio[: audio.size]
    max_start = max(0, audio.size - overlap_audio.size)
    start_sample = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    end_sample = start_sample + int(overlap_audio.size)
    base_slice = audio[start_sample:end_sample]
    signal_rms = float(np.sqrt(np.mean(np.square(base_slice.astype(np.float64, copy=False)))))
    overlap_rms = float(np.sqrt(np.mean(np.square(overlap_audio.astype(np.float64, copy=False)))))
    snr_db = float(rng.uniform(snr_db_min, snr_db_max)) if snr_db_max > snr_db_min else float(snr_db_max)
    if signal_rms <= 1e-8 or overlap_rms <= 1e-8:
        scale = 0.25
    else:
        scale = (signal_rms / (10.0 ** (snr_db / 20.0))) / max(overlap_rms, 1e-12)
    mixed = audio.copy()
    mixed[start_sample:end_sample] = mixed[start_sample:end_sample] + overlap_audio * scale
    mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32, copy=False)
    segment = TeacherSegment(start=start_sample / sample_rate, end=end_sample / sample_rate, score=1.0)
    detail.update(
        {
            "synthetic_start_s": segment.start,
            "synthetic_end_s": segment.end,
            "snr_db": snr_db,
            "scale": scale,
            "overlap": True,
        }
    )
    overlap_segment = {
        "start": segment.start,
        "end": segment.end,
        "source_audio_id": detail.get("source_audio_id"),
        "snr_db": snr_db,
    }
    return mixed, [*speech_segments, segment], [overlap_segment], {
        "enabled": True,
        "segments": [overlap_segment],
        "source": detail,
    }


def gap_samples(
    *,
    rng: np.random.Generator,
    sample_rate: int,
    min_s: float,
    max_s: float,
) -> int:
    if max_s <= 0.0:
        return 0
    lower = max(0.0, min_s)
    upper = max(lower, max_s)
    duration_s = float(rng.uniform(lower, upper)) if upper > lower else upper
    return max(0, int(round(duration_s * sample_rate)))


def internal_gap_samples(
    *,
    rng: np.random.Generator,
    sample_rate: int,
    min_s: float,
    max_s: float,
    touch_gap_prob: float,
    short_gap_prob: float,
    short_gap_max_s: float,
) -> tuple[int, str]:
    touch_probability = max(0.0, min(1.0, float(touch_gap_prob)))
    short_probability = max(0.0, min(1.0, float(short_gap_prob)))
    draw = float(rng.random())
    if draw < touch_probability:
        return 0, "touch"
    if draw < touch_probability + short_probability:
        upper = max(0.0, float(short_gap_max_s))
        duration_s = float(rng.uniform(0.0, upper)) if upper > 0.0 else 0.0
        return max(0, int(round(duration_s * sample_rate))), "short"
    return (
        gap_samples(
            rng=rng,
            sample_rate=sample_rate,
            min_s=min_s,
            max_s=max_s,
        ),
        "regular",
    )


def sample_half_open_int(
    rng: np.random.Generator,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Sample an integer from [minimum, maximum)."""

    if minimum < 0:
        raise ValueError("minimum must be non-negative")
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")
    return int(rng.integers(minimum, maximum))


def sample_binary_hardmix_layout(
    rng: np.random.Generator,
    *,
    speech_count_min: int,
    speech_count_max: int,
    edge_noise_count_min: int,
    edge_noise_count_max: int,
    inter_noise_count_min: int,
    inter_noise_count_max: int,
) -> dict[str, Any]:
    """Build a 1=speech/0=noise layout with independently sampled zero runs."""

    speech_count = sample_half_open_int(
        rng,
        minimum=speech_count_min,
        maximum=speech_count_max,
    )
    leading_noise_count = sample_half_open_int(
        rng,
        minimum=edge_noise_count_min,
        maximum=edge_noise_count_max,
    )
    trailing_noise_count = sample_half_open_int(
        rng,
        minimum=edge_noise_count_min,
        maximum=edge_noise_count_max,
    )
    inter_noise_counts = [
        sample_half_open_int(
            rng,
            minimum=inter_noise_count_min,
            maximum=inter_noise_count_max,
        )
        for _ in range(max(0, speech_count - 1))
    ]
    tokens: list[int] = [0] * leading_noise_count
    for speech_index in range(speech_count):
        tokens.append(1)
        if speech_index < len(inter_noise_counts):
            tokens.extend([0] * inter_noise_counts[speech_index])
    tokens.extend([0] * trailing_noise_count)
    return {
        "speech_count": speech_count,
        "leading_noise_count": leading_noise_count,
        "inter_noise_counts": inter_noise_counts,
        "trailing_noise_count": trailing_noise_count,
        "tokens": tokens,
        "pattern": "".join(str(token) for token in tokens),
    }


def crossfade_samples(
    *,
    rng: np.random.Generator,
    sample_rate: int,
    min_ms: float,
    max_ms: float,
) -> int:
    if max_ms <= 0.0:
        return 0
    lower = max(0.0, min_ms)
    upper = max(lower, max_ms)
    duration_ms = float(rng.uniform(lower, upper)) if upper > lower else upper
    return max(0, int(round(duration_ms * sample_rate / 1000.0)))


def crossfade_gains(samples: int, *, curve: str) -> tuple[np.ndarray, np.ndarray]:
    if samples <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    t = np.linspace(0.0, 1.0, samples, endpoint=False, dtype=np.float32)
    if curve == "linear":
        return (1.0 - t).astype(np.float32), t.astype(np.float32)
    if curve != "equal_power":
        raise ValueError(f"unsupported crossfade curve: {curve}")
    return (
        np.cos(0.5 * np.pi * t).astype(np.float32),
        np.sin(0.5 * np.pi * t).astype(np.float32),
    )


def append_timeline_part(
    audio: np.ndarray,
    part: np.ndarray,
    *,
    kind: str,
    previous_kind: str | None,
    rng: np.random.Generator,
    sample_rate: int,
    crossfade_ms_min: float,
    crossfade_ms_max: float,
    crossfade_curve: str,
) -> tuple[np.ndarray, int, int, dict[str, Any] | None]:
    if part.size <= 0:
        return audio, int(audio.size), int(audio.size), None
    values = np.ascontiguousarray(part, dtype=np.float32)
    requested = crossfade_samples(
        rng=rng,
        sample_rate=sample_rate,
        min_ms=crossfade_ms_min,
        max_ms=crossfade_ms_max,
    )
    overlap_samples = min(requested, int(audio.size), int(values.size))
    if overlap_samples <= 0:
        start_sample = int(audio.size)
        combined = np.concatenate([audio, values]).astype(np.float32, copy=False)
        return combined, start_sample, int(start_sample + values.size), None

    fade_out, fade_in = crossfade_gains(overlap_samples, curve=crossfade_curve)
    transition_start = int(audio.size - overlap_samples)
    transition_end = int(audio.size)
    overlap = audio[-overlap_samples:] * fade_out + values[:overlap_samples] * fade_in
    combined = np.concatenate(
        [audio[:-overlap_samples], overlap.astype(np.float32, copy=False), values[overlap_samples:]]
    ).astype(np.float32, copy=False)
    transition = {
        "start_s": transition_start / sample_rate,
        "end_s": transition_end / sample_rate,
        "samples": overlap_samples,
        "duration_s": overlap_samples / sample_rate,
        "crossfade_ms": overlap_samples * 1000.0 / sample_rate,
        "curve": crossfade_curve,
        "left_kind": previous_kind or "unknown",
        "right_kind": kind,
    }
    return combined, transition_start, int(transition_start + values.size), transition


def build_synthetic_timeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    source_rows, skipped = valid_source_rows(load_manifest_rows(Path(args.manifest)))
    require_source_schema(source_rows, getattr(args, "require_source_schema", None))
    source_rows_before_exclusion = len(source_rows)
    excluded_source_audio_ids = load_excluded_source_audio_ids(
        getattr(args, "exclude_source_manifest", None)
    )
    excluded_source_audio_ids.update(
        str(value)
        for value in (getattr(args, "exclude_source_audio_id", None) or [])
        if str(value)
    )
    source_rows = [
        row
        for row in source_rows
        if str(row.get("audio_id") or Path(str(row["audio"])).stem)
        not in excluded_source_audio_ids
    ]
    negative_rows, negative_skipped = load_valid_manifest_rows(args.negative_manifest, role="negative_gap")
    background_rows, background_skipped = load_valid_manifest_rows(args.background_manifest, role="background")
    skipped.extend(negative_skipped)
    skipped.extend(background_skipped)
    if args.shuffle:
        rng.shuffle(source_rows)
        rng.shuffle(negative_rows)
        rng.shuffle(background_rows)
    if args.limit is not None:
        source_rows = source_rows[: args.limit]
    if not source_rows:
        raise ValueError("no valid Galgame source rows available")
    source_rows_by_partition: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for row in source_rows:
        source_rows_by_partition[
            stable_source_partition(
                row,
                train_ratio=args.partition_train_ratio,
                val_ratio=args.partition_val_ratio,
            )
        ].append(row)
    negative_rows_by_partition: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for row in negative_rows:
        explicit_partition = str(row.get("source_partition") or "")
        partition = (
            explicit_partition
            if explicit_partition in negative_rows_by_partition
            else stable_source_partition(
                {
                    "audio_id": (
                        row.get("video_id")
                        or row.get("audio_id")
                        or row.get("audio")
                    )
                },
                train_ratio=args.partition_train_ratio,
                val_ratio=args.partition_val_ratio,
            )
        )
        negative_rows_by_partition[partition].append(row)

    sample_rate = 16000
    gap_duration_pool: list[float] = []
    if getattr(args, "gap_duration_samples", ""):
        pool_payload = json.loads(
            Path(args.gap_duration_samples).read_text(encoding="utf-8")
        )
        raw_pool = (
            pool_payload.get("durations_s")
            if isinstance(pool_payload, dict)
            else pool_payload
        )
        gap_duration_pool = [
            float(value) for value in (raw_pool or []) if float(value) > 0.0
        ]
        if not gap_duration_pool:
            raise ValueError(
                f"--gap-duration-samples has no positive durations: {args.gap_duration_samples}"
            )
    post_speech_noise_flags = [
        flag.strip()
        for flag in str(args.post_speech_noise_flags or "").split(",")
        if flag.strip()
    ]
    records = []
    manifest_rows = []
    boundary_rows = []
    detail_rows = []
    gap_mode_counts: Counter[str] = Counter()
    background_mix_count = 0
    background_skip_count = 0
    overlap_mix_count = 0
    gain_aug_count = 0
    filter_aug_count = 0
    codec_aug_count = 0
    internal_gap_policy_counts: Counter[str] = Counter()
    source_cursor = 0
    used_source_indices: set[int] = set()
    source_cursors_by_partition = {"train": 0, "val": 0, "test": 0}
    used_source_indices_by_partition = {
        "train": set(),
        "val": set(),
        "test": set(),
    }
    nominal_speech_count = args.speech_clips_per_example
    if args.timeline_pattern_mode == "binary_hardmix":
        nominal_speech_count = max(
            1,
            int(
                round(
                    (
                        args.hardmix_speech_count_min
                        + args.hardmix_speech_count_max
                        - 1
                    )
                    / 2.0
                )
            ),
        )
    synthetic_count = min(args.count, math.ceil(len(source_rows) / nominal_speech_count))
    if args.reuse_sources:
        synthetic_count = args.count
    for output_index in range(synthetic_count):
        audio = np.zeros(0, dtype=np.float32)
        speech_segments: list[TeacherSegment] = []
        source_details: list[dict[str, Any]] = []
        transition_regions: list[dict[str, Any]] = []
        previous_kind: str | None = None
        previous_speech_segment: TeacherSegment | None = None
        utterance_boundaries: list[dict[str, Any]] = []
        semantic_split_boundaries: list[dict[str, Any]] = []
        cut_point_segments: list[dict[str, Any]] = []
        source_partition = "train"
        example_source_rows = source_rows
        example_negative_rows = negative_rows
        if args.timeline_pattern_mode == "binary_hardmix":
            source_partition = str(
                rng.choice(
                    ("train", "val", "test"),
                    p=(
                        args.partition_train_ratio,
                        args.partition_val_ratio,
                        1.0 - args.partition_train_ratio - args.partition_val_ratio,
                    ),
                )
            )
            example_source_rows = source_rows_by_partition[source_partition]
            if not example_source_rows:
                raise ValueError(
                    f"no source rows assigned to partition {source_partition!r}"
                )
            example_negative_rows = negative_rows_by_partition[source_partition]
            if negative_rows and not example_negative_rows:
                raise ValueError(
                    f"no negative rows assigned to partition {source_partition!r}"
                )
        example_target_rms = 10.0 ** (
            float(
                rng.uniform(
                    args.unit_loudness_target_db_min,
                    args.unit_loudness_target_db_max,
                )
            )
            / 20.0
        )
        apply_unit_loudness = (
            args.timeline_pattern_mode == "binary_hardmix"
            and float(rng.random()) < args.unit_loudness_align_prob
        )
        previous_speech_group_key = ""
        same_source_adjacent_count = 0

        def hardmix_gap_samples_fn() -> int:
            if gap_duration_pool:
                return empirical_gap_samples(
                    rng=rng,
                    sample_rate=sample_rate,
                    duration_pool_s=gap_duration_pool,
                    min_s=args.hardmix_noise_duration_min_s,
                    max_s=args.hardmix_noise_duration_max_s,
                )
            return gap_samples(
                rng=rng,
                sample_rate=sample_rate,
                min_s=args.hardmix_noise_duration_min_s,
                max_s=args.hardmix_noise_duration_max_s,
            )

        def align_negative_gap(
            gap_audio: np.ndarray, mode: str
        ) -> tuple[np.ndarray, dict[str, Any] | None]:
            if not apply_unit_loudness or mode != "real_negative":
                return gap_audio, None
            offset_db = float(
                rng.uniform(
                    args.negative_loudness_offset_db_min,
                    args.negative_loudness_offset_db_max,
                )
            )
            return align_unit_loudness(
                gap_audio,
                rng=rng,
                target_rms=example_target_rms * (10.0 ** (offset_db / 20.0)),
                jitter_db=0.0,
            )
        if args.timeline_pattern_mode == "binary_hardmix":
            hardmix_layout = sample_binary_hardmix_layout(
                rng,
                speech_count_min=args.hardmix_speech_count_min,
                speech_count_max=args.hardmix_speech_count_max,
                edge_noise_count_min=args.hardmix_edge_noise_count_min,
                edge_noise_count_max=args.hardmix_edge_noise_count_max,
                inter_noise_count_min=args.hardmix_inter_noise_count_min,
                inter_noise_count_max=args.hardmix_inter_noise_count_max,
            )
            speech_count = int(hardmix_layout["speech_count"])
            leading_noise_count = int(hardmix_layout["leading_noise_count"])
            inter_noise_counts = list(hardmix_layout["inter_noise_counts"])
            trailing_noise_count = int(hardmix_layout["trailing_noise_count"])
        else:
            speech_count = args.speech_clips_per_example
            leading_noise_count = 1
            inter_noise_counts = [1] * max(0, speech_count - 1)
            trailing_noise_count = 1
            hardmix_layout = {
                "speech_count": speech_count,
                "leading_noise_count": leading_noise_count,
                "inter_noise_counts": inter_noise_counts,
                "trailing_noise_count": trailing_noise_count,
                "tokens": [
                    0,
                    *[
                        token
                        for speech_index in range(speech_count)
                        for token in ([1, 0] if speech_index < speech_count - 1 else [1])
                    ],
                    0,
                ],
                "pattern": "",
            }
            hardmix_layout["pattern"] = "".join(
                str(token) for token in hardmix_layout["tokens"]
            )
        gap_index = output_index * (
            speech_count
            + leading_noise_count
            + sum(inter_noise_counts)
            + trailing_noise_count
            + 1
        )
        next_gap_index = gap_index

        for noise_unit_index in range(leading_noise_count):
            if args.timeline_pattern_mode == "binary_hardmix":
                leading_samples = hardmix_gap_samples_fn()
            else:
                leading_samples = gap_samples(
                    rng=rng,
                    sample_rate=sample_rate,
                    min_s=args.leading_gap_min_s,
                    max_s=args.leading_gap_max_s,
                )
            leading_gap, mode, gap_detail = build_gap(
                samples=leading_samples,
                sample_rate=sample_rate,
                index=next_gap_index,
                rng=rng,
                noise_rms=args.noise_rms,
                hum_rms=args.hum_rms,
                negative_rows=example_negative_rows,
                negative_gap_prob=args.negative_gap_prob,
                randomize_synthetic_mode=(
                    args.timeline_pattern_mode == "binary_hardmix"
                ),
            )
            next_gap_index += 1
            leading_gap, gap_loudness = align_negative_gap(leading_gap, mode)
            if leading_gap.size:
                audio, gap_start_sample, gap_end_sample, transition = append_timeline_part(
                    audio,
                    leading_gap,
                    kind="gap",
                    previous_kind=previous_kind,
                    rng=rng,
                    sample_rate=sample_rate,
                    crossfade_ms_min=args.crossfade_ms_min,
                    crossfade_ms_max=args.crossfade_ms_max,
                    crossfade_curve=args.crossfade_curve,
                )
                previous_kind = "gap"
                gap_mode_counts[mode] += 1
                gap_row = {
                    "gap": "leading",
                    "noise_unit_index": noise_unit_index,
                    "gap_type": mode,
                    "mode": mode,
                    "synthetic_start_s": gap_start_sample / sample_rate,
                    "synthetic_end_s": gap_end_sample / sample_rate,
                }
                if gap_loudness:
                    gap_row["loudness_align"] = gap_loudness
                if gap_detail:
                    gap_row.update(gap_detail)
                source_details.append(gap_row)
                if transition:
                    transition_regions.append(transition)

        for speech_index in range(speech_count):
            example_source_cursor = source_cursor
            example_used_source_indices = used_source_indices
            if args.timeline_pattern_mode == "binary_hardmix":
                example_source_cursor = source_cursors_by_partition[source_partition]
                example_used_source_indices = used_source_indices_by_partition[
                    source_partition
                ]
            row = None
            source_index = None
            if (
                args.timeline_pattern_mode == "binary_hardmix"
                and previous_speech_group_key
                and float(rng.random()) < args.same_source_adjacent_prob
            ):
                row, source_index = choose_same_source_row(
                    source_rows=example_source_rows,
                    previous_group_key=previous_speech_group_key,
                    used_source_indices=example_used_source_indices,
                    reuse_sources=args.reuse_sources,
                    rng=rng,
                )
                if row is not None:
                    same_source_adjacent_count += 1
            if row is None:
                row, source_cursor, source_index = choose_speech_row(
                    source_rows=example_source_rows,
                    source_cursor=example_source_cursor,
                    args=args,
                    rng=rng,
                    used_source_indices=example_used_source_indices,
                )
                if args.timeline_pattern_mode == "binary_hardmix":
                    source_cursors_by_partition[source_partition] = source_cursor
            if row is None:
                break
            if source_index is not None:
                example_used_source_indices.add(source_index)
            try:
                speech_audio, source_detail = clipped_speech_audio(
                    row,
                    rng=rng,
                    trim_head_s=args.trim_head_s,
                    trim_tail_s=args.trim_tail_s,
                    max_speech_s=args.max_speech_s,
                    min_speech_s=args.min_speech_s,
                )
            except Exception as exc:
                skipped.append(
                    {
                        "index": source_index,
                        "audio_id": str(row.get("audio_id") or ""),
                        "reason": "speech_load_error",
                        "error": str(exc),
                    }
                )
                continue
            previous_speech_group_key = speech_group_key(row)
            if apply_unit_loudness:
                speech_audio, speech_loudness = align_unit_loudness(
                    speech_audio,
                    rng=rng,
                    target_rms=example_target_rms,
                    jitter_db=args.unit_loudness_jitter_db,
                )
                if speech_loudness:
                    source_detail["loudness_align"] = speech_loudness
            audio, speech_start_sample, speech_end_sample, transition = append_timeline_part(
                audio,
                speech_audio,
                kind="speech",
                previous_kind=previous_kind,
                rng=rng,
                sample_rate=sample_rate,
                crossfade_ms_min=args.crossfade_ms_min,
                crossfade_ms_max=args.crossfade_ms_max,
                crossfade_curve=args.crossfade_curve,
            )
            previous_kind = "speech"
            if transition:
                transition_regions.append(transition)
            current_segment = TeacherSegment(
                start=speech_start_sample / sample_rate,
                end=speech_end_sample / sample_rate,
                score=1.0,
            )
            speech_segments.append(current_segment)
            if previous_speech_segment is not None:
                turn_gap_s = current_segment.start - previous_speech_segment.end
                boundary_time_s = (previous_speech_segment.end + current_segment.start) / 2.0
                boundary_type = boundary_type_from_gap(
                    turn_gap_s,
                    cut_point_max_gap_s=args.cut_point_max_gap_s,
                    long_gap_boundary_min_s=args.long_gap_boundary_min_s,
                )
                turn = {
                    "index": len(utterance_boundaries),
                    "time_s": boundary_time_s,
                    "previous_speech_end_s": previous_speech_segment.end,
                    "next_speech_start_s": current_segment.start,
                    "gap_s": turn_gap_s,
                    "boundary_type": boundary_type,
                    "inter_noise_unit_count": int(
                        inter_noise_counts[speech_index - 1]
                    ),
                    "inter_noise_modes": [
                        str(item.get("mode") or "")
                        for item in source_details
                        if item.get("gap") == f"middle-{speech_index - 1}"
                    ],
                }
                utterance_boundaries.append(turn)
                if int(turn["inter_noise_unit_count"]) <= 0:
                    semantic_split_boundaries.append(
                        {
                            **turn,
                            "structural_role": "speech_to_speech",
                        }
                    )
                else:
                    semantic_split_boundaries.extend(
                        (
                            {
                                **turn,
                                "time_s": previous_speech_segment.end,
                                "structural_role": "speech_to_noise",
                            },
                            {
                                **turn,
                                "time_s": current_segment.start,
                                "structural_role": "noise_to_speech",
                            },
                        )
                    )
                if turn_gap_s <= args.cut_point_max_gap_s:
                    cut_point_segments.append(
                        {
                            "start": boundary_time_s,
                            "end": boundary_time_s,
                            "time_s": boundary_time_s,
                            "gap_s": turn_gap_s,
                        }
                    )
            previous_speech_segment = current_segment
            source_detail.update(
                {
                    "synthetic_start_s": speech_start_sample / sample_rate,
                    "synthetic_end_s": speech_end_sample / sample_rate,
                    "source_index": source_index,
                }
            )
            source_details.append(source_detail)

            if speech_index < speech_count - 1:
                for noise_unit_index in range(inter_noise_counts[speech_index]):
                    if args.timeline_pattern_mode == "binary_hardmix":
                        middle_samples = hardmix_gap_samples_fn()
                        gap_policy = "hardmix_unit"
                    else:
                        middle_samples, gap_policy = internal_gap_samples(
                            rng=rng,
                            sample_rate=sample_rate,
                            min_s=args.gap_min_s,
                            max_s=args.gap_max_s,
                            touch_gap_prob=args.touch_gap_prob,
                            short_gap_prob=args.short_gap_prob,
                            short_gap_max_s=args.short_gap_max_s,
                        )
                    internal_gap_policy_counts[gap_policy] += 1
                    middle_gap, mode, gap_detail = build_gap(
                        samples=middle_samples,
                        sample_rate=sample_rate,
                        index=next_gap_index,
                        rng=rng,
                        noise_rms=args.noise_rms,
                        hum_rms=args.hum_rms,
                        negative_rows=example_negative_rows,
                        negative_gap_prob=args.negative_gap_prob,
                        randomize_synthetic_mode=(
                            args.timeline_pattern_mode == "binary_hardmix"
                        ),
                        preferred_flags=(
                            post_speech_noise_flags if noise_unit_index == 0 else None
                        ),
                        preferred_flag_prob=args.post_speech_noise_flag_prob,
                    )
                    next_gap_index += 1
                    middle_gap, gap_loudness = align_negative_gap(middle_gap, mode)
                    if middle_gap.size:
                        audio, gap_start_sample, gap_end_sample, transition = append_timeline_part(
                            audio,
                            middle_gap,
                            kind="gap",
                            previous_kind=previous_kind,
                            rng=rng,
                            sample_rate=sample_rate,
                            crossfade_ms_min=args.crossfade_ms_min,
                            crossfade_ms_max=args.crossfade_ms_max,
                            crossfade_curve=args.crossfade_curve,
                        )
                        previous_kind = "gap"
                        gap_mode_counts[mode] += 1
                        gap_row = {
                            "gap": f"middle-{speech_index}",
                            "noise_unit_index": noise_unit_index,
                            "gap_type": mode,
                            "gap_policy": gap_policy,
                            "mode": mode,
                            "synthetic_start_s": gap_start_sample / sample_rate,
                            "synthetic_end_s": gap_end_sample / sample_rate,
                        }
                        if gap_loudness:
                            gap_row["loudness_align"] = gap_loudness
                        if gap_detail:
                            gap_row.update(gap_detail)
                        source_details.append(gap_row)
                        if transition:
                            transition_regions.append(transition)

        for noise_unit_index in range(trailing_noise_count):
            if args.timeline_pattern_mode == "binary_hardmix":
                trailing_samples = hardmix_gap_samples_fn()
            else:
                trailing_samples = gap_samples(
                    rng=rng,
                    sample_rate=sample_rate,
                    min_s=args.trailing_gap_min_s,
                    max_s=args.trailing_gap_max_s,
                )
            trailing_gap, mode, gap_detail = build_gap(
                samples=trailing_samples,
                sample_rate=sample_rate,
                index=next_gap_index,
                rng=rng,
                noise_rms=args.noise_rms,
                hum_rms=args.hum_rms,
                negative_rows=example_negative_rows,
                negative_gap_prob=args.negative_gap_prob,
                randomize_synthetic_mode=(
                    args.timeline_pattern_mode == "binary_hardmix"
                ),
                preferred_flags=(
                    post_speech_noise_flags if noise_unit_index == 0 else None
                ),
                preferred_flag_prob=args.post_speech_noise_flag_prob,
            )
            next_gap_index += 1
            trailing_gap, gap_loudness = align_negative_gap(trailing_gap, mode)
            if trailing_gap.size:
                audio, gap_start_sample, gap_end_sample, transition = append_timeline_part(
                    audio,
                    trailing_gap,
                    kind="gap",
                    previous_kind=previous_kind,
                    rng=rng,
                    sample_rate=sample_rate,
                    crossfade_ms_min=args.crossfade_ms_min,
                    crossfade_ms_max=args.crossfade_ms_max,
                    crossfade_curve=args.crossfade_curve,
                )
                previous_kind = "gap"
                gap_mode_counts[mode] += 1
                gap_row = {
                    "gap": "trailing",
                    "noise_unit_index": noise_unit_index,
                    "gap_type": mode,
                    "mode": mode,
                    "synthetic_start_s": gap_start_sample / sample_rate,
                    "synthetic_end_s": gap_end_sample / sample_rate,
                }
                if gap_loudness:
                    gap_row["loudness_align"] = gap_loudness
                if gap_detail:
                    gap_row.update(gap_detail)
                source_details.append(gap_row)
                if transition:
                    transition_regions.append(transition)

        if not speech_segments or audio.size <= 0:
            continue
        audio, speech_segments, overlap_segments, overlap_detail = mix_overlap_speech(
            audio,
            speech_rows=source_rows,
            speech_segments=speech_segments,
            rng=rng,
            sample_rate=sample_rate,
            probability=args.overlap_speech_prob,
            snr_db_min=args.overlap_snr_db_min,
            snr_db_max=args.overlap_snr_db_max,
            trim_head_s=args.trim_head_s,
            trim_tail_s=args.trim_tail_s,
            max_speech_s=args.overlap_max_speech_s,
            min_speech_s=args.min_speech_s,
        )
        if overlap_detail.get("enabled"):
            overlap_mix_count += 1
        background_detail: dict[str, Any] | None = None
        if background_rows and args.background_mix_prob > 0.0 and float(rng.random()) < args.background_mix_prob:
            try:
                audio, background_detail = mix_background_audio(
                    audio,
                    background_rows=background_rows,
                    rng=rng,
                    snr_db_min=args.background_snr_db_min,
                    snr_db_max=args.background_snr_db_max,
                    switch_prob=args.background_switch_prob,
                    speech_segments=speech_segments,
                    sample_rate=sample_rate,
                )
                if background_detail.get("skipped"):
                    background_skip_count += 1
                else:
                    background_mix_count += 1
            except Exception as exc:
                background_skip_count += 1
                background_detail = {"error": str(exc)}
        audio, gain_detail = apply_random_gain(
            audio,
            rng=rng,
            gain_db_min=args.gain_db_min,
            gain_db_max=args.gain_db_max,
        )
        if gain_detail.get("enabled"):
            gain_aug_count += 1
        audio, filter_detail = apply_filter_aug(
            audio,
            sample_rate=sample_rate,
            rng=rng,
            filter_prob=args.filter_prob,
            filter_mode=args.filter_mode,
            lowpass_min_hz=args.lowpass_min_hz,
            lowpass_max_hz=args.lowpass_max_hz,
            bandpass_low_min_hz=args.bandpass_low_min_hz,
            bandpass_low_max_hz=args.bandpass_low_max_hz,
            bandpass_high_min_hz=args.bandpass_high_min_hz,
            bandpass_high_max_hz=args.bandpass_high_max_hz,
        )
        if filter_detail.get("enabled"):
            filter_aug_count += 1
        audio, codec_detail = apply_codec_aug(
            audio,
            rng=rng,
            codec_prob=args.codec_prob,
            codec_aug=args.codec_aug,
        )
        if codec_detail.get("enabled"):
            codec_aug_count += 1
        audio = np.clip(audio, -1.0, 1.0)
        audio_id = f"{args.audio_id_prefix}-{output_index:06d}"
        audio_path = audio_dir / f"{audio_id}.wav"
        sf.write(str(audio_path), audio, sample_rate)
        duration_s = len(audio) / sample_rate
        label_segments = expand_segments(
            speech_segments,
            duration_s=duration_s,
            dilation_s=args.speech_label_dilation_s,
        )
        record = build_supervised_record(
            audio_id=audio_id,
            source=args.source,
            duration_s=duration_s,
            text=args.text_separator.join(
                str(item.get("source_text") or "") for item in source_details if item.get("source_text")
            ),
            speech_segments=label_segments,
            frame_hop_s=args.frame_hop_s,
        )
        boundary_metadata = {
            "source_partition": source_partition,
            "timeline_pattern_mode": args.timeline_pattern_mode,
            "timeline_pattern": hardmix_layout["pattern"],
            "timeline_tokens": hardmix_layout["tokens"],
            "timeline_layout": hardmix_layout,
            "source_audio_ids": [
                str(item.get("source_audio_id")) for item in source_details if item.get("source_audio_id")
            ],
            "utterance_boundaries": utterance_boundaries,
            "semantic_split_boundaries": semantic_split_boundaries,
            "cut_point_segments": cut_point_segments,
            "actual_speech_segments": [
                {"start": segment.start, "end": segment.end} for segment in speech_segments
            ],
            "speech_label_dilation_s": args.speech_label_dilation_s,
            "realism": {
                "unit_loudness_aligned": bool(apply_unit_loudness),
                "target_rms": example_target_rms,
                "same_source_adjacent_count": same_source_adjacent_count,
                "gap_duration_source": (
                    "empirical_pool" if gap_duration_pool else "uniform"
                ),
                "background": background_detail or {},
            },
        }
        record = type(record)(
            audio_id=record.audio_id,
            source=record.source,
            duration_s=record.duration_s,
            text=record.text,
            teacher_segments=record.teacher_segments,
            frame_hop_s=record.frame_hop_s,
            speech_frames=record.speech_frames,
            label_quality=record.label_quality,
            frame_weights=record.frame_weights,
            boundary_metadata=boundary_metadata,
        )
        records.append(record)
        manifest_rows.append(
            {
                "audio_id": audio_id,
                "audio": str(audio_path),
                "duration_s": duration_s,
                "sample_rate": sample_rate,
                "source": args.source,
                "source_partition": source_partition,
                "timeline_pattern_mode": args.timeline_pattern_mode,
                "timeline_pattern": hardmix_layout["pattern"],
                "timeline_tokens": hardmix_layout["tokens"],
                "timeline_layout": hardmix_layout,
                "label_quality": record.label_quality,
                "input": ",".join(
                    str(item.get("source_input") or "") for item in source_details if item.get("source_input")
                ),
                "source_audio_ids": [
                    str(item.get("source_audio_id")) for item in source_details if item.get("source_audio_id")
                ],
                "utterance_boundaries": utterance_boundaries,
                "semantic_split_boundaries": semantic_split_boundaries,
                "cut_point_segments": cut_point_segments,
                "speech_segments": [{"start": segment.start, "end": segment.end} for segment in label_segments],
                "actual_speech_segments": [{"start": segment.start, "end": segment.end} for segment in speech_segments],
                "speech_label_dilation_s": args.speech_label_dilation_s,
                "transition_regions": transition_regions,
                "overlap_segments": overlap_segments,
                "augmentation": {
                    "background_mix": background_detail,
                    "overlap_speech": overlap_detail,
                    "gain": gain_detail,
                    "filter": filter_detail,
                    "codec_aug": codec_detail,
                },
            }
        )
        boundary_rows.append(
            {
                "audio_id": audio_id,
                "audio": str(audio_path),
                "duration_s": duration_s,
                "sample_rate": sample_rate,
                "source": args.source,
                "source_partition": source_partition,
                "timeline_pattern_mode": args.timeline_pattern_mode,
                "timeline_pattern": hardmix_layout["pattern"],
                "timeline_tokens": hardmix_layout["tokens"],
                "timeline_layout": hardmix_layout,
                "text": record.text,
                "frame_hop_s": record.frame_hop_s,
                "speech_frames": record.speech_frames,
                "speech_segments": [{"start": segment.start, "end": segment.end} for segment in label_segments],
                "actual_speech_segments": [{"start": segment.start, "end": segment.end} for segment in speech_segments],
                "speech_label_dilation_s": args.speech_label_dilation_s,
                "source_audio_ids": [
                    str(item.get("source_audio_id")) for item in source_details if item.get("source_audio_id")
                ],
                "utterance_boundaries": utterance_boundaries,
                "semantic_split_boundaries": semantic_split_boundaries,
                "cut_point_segments": cut_point_segments,
                "sources": source_details,
                "background_mix": background_detail,
                "transition_regions": transition_regions,
                "overlap_segments": overlap_segments,
                "augmentation": {
                    "background_mix": background_detail,
                    "overlap_speech": overlap_detail,
                    "gain": gain_detail,
                    "filter": filter_detail,
                    "codec_aug": codec_detail,
                },
            }
        )
        detail_rows.append(
            {
                "audio_id": audio_id,
                "source_partition": source_partition,
                "timeline_pattern_mode": args.timeline_pattern_mode,
                "timeline_pattern": hardmix_layout["pattern"],
                "timeline_tokens": hardmix_layout["tokens"],
                "timeline_layout": hardmix_layout,
                "duration_s": duration_s,
                "speech_segments": [{"start": segment.start, "end": segment.end} for segment in label_segments],
                "actual_speech_segments": [{"start": segment.start, "end": segment.end} for segment in speech_segments],
                "utterance_boundaries": utterance_boundaries,
                "semantic_split_boundaries": semantic_split_boundaries,
                "cut_point_segments": cut_point_segments,
                "sources": source_details,
                "background_mix": background_detail,
                "transition_regions": transition_regions,
                "overlap_segments": overlap_segments,
                "augmentation": {
                    "background_mix": background_detail,
                    "overlap_speech": overlap_detail,
                    "gain": gain_detail,
                    "filter": filter_detail,
                    "codec_aug": codec_detail,
                },
            }
        )

    labels_path = output_dir / args.output_jsonl
    manifest_path = output_dir / args.output_manifest
    boundary_manifest_path = output_dir / "boundary_manifest.jsonl"
    details_path = output_dir / "synthetic_timeline_details.jsonl"
    skipped_path = output_dir / "synthetic_timeline_skipped.json"
    summary_path = output_dir / "synthetic_timeline_summary.json"
    write_jsonl(labels_path, records)
    manifest_path.write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_jsonl(boundary_manifest_path, boundary_rows)
    with details_path.open("w", encoding="utf-8") as handle:
        for row in detail_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    skipped_path.write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    total_frames = sum(len(record.speech_frames) for record in records)
    speech_frames = sum(sum(int(value) for value in record.speech_frames) for record in records)
    cut_point_count = sum(len(row.get("cut_point_segments") or []) for row in boundary_rows)
    utterance_boundary_count = sum(len(row.get("utterance_boundaries") or []) for row in boundary_rows)
    source_usage = summarize_source_usage(detail_rows)
    summary = {
        "manifest": str(Path(args.manifest)),
        "records": len(records),
        "source_rows": len(source_rows),
        "source_rows_before_exclusion": source_rows_before_exclusion,
        "excluded_source_audio_id_count": len(excluded_source_audio_ids),
        "negative_rows": len(negative_rows),
        "background_rows": len(background_rows),
        "skipped": len(skipped),
        "duration_s_total": sum(record.duration_s for record in records),
        "speech_frame_ratio": (speech_frames / total_frames) if total_frames else 0.0,
        "label_quality_counts": dict(sorted(Counter(record.label_quality for record in records).items())),
        "gap_mode_counts": dict(sorted(gap_mode_counts.items())),
        "internal_gap_policy_counts": dict(sorted(internal_gap_policy_counts.items())),
        "background_mix_count": background_mix_count,
        "background_skip_count": background_skip_count,
        "overlap_mix_count": overlap_mix_count,
        "gain_aug_count": gain_aug_count,
        "filter_aug_count": filter_aug_count,
        "codec_aug_count": codec_aug_count,
        "random_speech_order_enabled": bool(args.randomize_speech_order),
        "utterance_boundary_count": utterance_boundary_count,
        "cut_point_segment_count": cut_point_count,
        **source_usage,
        "unused_source_core_count": len(source_rows)
        - source_usage["unique_source_core_count"],
        "labels": str(labels_path),
        "output_manifest": str(manifest_path),
        "boundary_manifest": str(boundary_manifest_path),
        "details": str(details_path),
        "skipped_report": str(skipped_path),
        "config": {
            "count": args.count,
            "timeline_pattern_mode": args.timeline_pattern_mode,
            "speech_clips_per_example": args.speech_clips_per_example,
            "hardmix_speech_count": [
                args.hardmix_speech_count_min,
                args.hardmix_speech_count_max,
            ],
            "hardmix_edge_noise_count": [
                args.hardmix_edge_noise_count_min,
                args.hardmix_edge_noise_count_max,
            ],
            "hardmix_inter_noise_count": [
                args.hardmix_inter_noise_count_min,
                args.hardmix_inter_noise_count_max,
            ],
            "hardmix_noise_duration_s": [
                args.hardmix_noise_duration_min_s,
                args.hardmix_noise_duration_max_s,
            ],
            "partition_train_ratio": args.partition_train_ratio,
            "partition_val_ratio": args.partition_val_ratio,
            "seed": args.seed,
            "shuffle": args.shuffle,
            "reuse_sources": args.reuse_sources,
            "randomize_speech_order": args.randomize_speech_order,
            "require_source_schema": getattr(args, "require_source_schema", None),
            "exclude_source_manifest": list(
                getattr(args, "exclude_source_manifest", None) or []
            ),
            "cut_point_max_gap_s": args.cut_point_max_gap_s,
            "long_gap_boundary_min_s": args.long_gap_boundary_min_s,
            "trim_head_s": args.trim_head_s,
            "trim_tail_s": args.trim_tail_s,
            "max_speech_s": args.max_speech_s,
            "min_speech_s": args.min_speech_s,
            "gap_min_s": args.gap_min_s,
            "gap_max_s": args.gap_max_s,
            "touch_gap_prob": args.touch_gap_prob,
            "short_gap_prob": args.short_gap_prob,
            "short_gap_max_s": args.short_gap_max_s,
            "leading_gap_min_s": args.leading_gap_min_s,
            "leading_gap_max_s": args.leading_gap_max_s,
            "trailing_gap_min_s": args.trailing_gap_min_s,
            "trailing_gap_max_s": args.trailing_gap_max_s,
            "noise_rms": args.noise_rms,
            "hum_rms": args.hum_rms,
            "negative_manifest": args.negative_manifest,
            "negative_gap_prob": args.negative_gap_prob,
            "background_manifest": args.background_manifest,
            "background_mix_prob": args.background_mix_prob,
            "background_snr_db_min": args.background_snr_db_min,
            "background_snr_db_max": args.background_snr_db_max,
            "speech_label_dilation_s": args.speech_label_dilation_s,
            "crossfade_ms_min": args.crossfade_ms_min,
            "crossfade_ms_max": args.crossfade_ms_max,
            "crossfade_curve": args.crossfade_curve,
            "gain_db_min": args.gain_db_min,
            "gain_db_max": args.gain_db_max,
            "filter_prob": args.filter_prob,
            "filter_mode": args.filter_mode,
            "lowpass_min_hz": args.lowpass_min_hz,
            "lowpass_max_hz": args.lowpass_max_hz,
            "bandpass_low_min_hz": args.bandpass_low_min_hz,
            "bandpass_low_max_hz": args.bandpass_low_max_hz,
            "bandpass_high_min_hz": args.bandpass_high_min_hz,
            "bandpass_high_max_hz": args.bandpass_high_max_hz,
            "codec_prob": args.codec_prob,
            "codec_aug": args.codec_aug,
            "overlap_speech_prob": args.overlap_speech_prob,
            "overlap_snr_db_min": args.overlap_snr_db_min,
            "overlap_snr_db_max": args.overlap_snr_db_max,
            "overlap_max_speech_s": args.overlap_max_speech_s,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"labels={labels_path}")
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exact-timeline supervised VAD clips by concatenating Galgame speech islands and synthetic gaps."
    )
    parser.add_argument("--manifest", required=True, help="hf_audio_manifest.json from materialized Galgame audio.")
    parser.add_argument(
        "--require-source-schema",
        help="Reject source rows outside this approved inventory schema.",
    )
    parser.add_argument(
        "--exclude-source-manifest",
        action="append",
        help=(
            "JSON/JSONL generated dataset manifest whose source_audio_ids are "
            "excluded before sampling. Repeatable."
        ),
    )
    parser.add_argument("--exclude-source-audio-id", action="append")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--limit", type=int, help="Optional source row limit before synthesis.")
    parser.add_argument(
        "--timeline-pattern-mode",
        choices=("alternating", "binary_hardmix"),
        default="alternating",
        help=(
            "alternating preserves the legacy one-gap layout; binary_hardmix samples "
            "1=speech/0=noise sequences with independent, consecutive noise units."
        ),
    )
    parser.add_argument("--speech-clips-per-example", type=int, default=2)
    parser.add_argument("--hardmix-speech-count-min", type=int, default=1)
    parser.add_argument(
        "--hardmix-speech-count-max",
        type=int,
        default=5,
        help="Exclusive upper bound for speech units in binary_hardmix mode.",
    )
    parser.add_argument("--hardmix-edge-noise-count-min", type=int, default=0)
    parser.add_argument(
        "--hardmix-edge-noise-count-max",
        type=int,
        default=5,
        help="Exclusive upper bound for leading/trailing noise units.",
    )
    parser.add_argument("--hardmix-inter-noise-count-min", type=int, default=0)
    parser.add_argument(
        "--hardmix-inter-noise-count-max",
        type=int,
        default=5,
        help="Exclusive upper bound for each inter-speech zero run.",
    )
    parser.add_argument("--hardmix-noise-duration-min-s", type=float, default=0.05)
    parser.add_argument(
        "--hardmix-noise-duration-max-s",
        type=float,
        default=2.0,
        help="Upper duration bound for every independent zero unit.",
    )
    parser.add_argument("--partition-train-ratio", type=float, default=0.85)
    parser.add_argument("--partition-val-ratio", type=float, default=0.10)
    parser.add_argument("--reuse-sources", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--randomize-speech-order",
        action="store_true",
        help="Sample speech islands randomly within each synthetic example instead of taking manifest rows sequentially.",
    )
    parser.add_argument(
        "--cut-point-max-gap-s",
        type=float,
        default=0.12,
        help="Adjacent speech islands with gap <= this are recorded as cut_point utterance boundaries.",
    )
    parser.add_argument(
        "--long-gap-boundary-min-s",
        dest="long_gap_boundary_min_s",
        type=float,
        default=0.5,
        help="Adjacent speech islands with gap >= this are recorded as long_gap utterance boundaries.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--source", default="galgame-synthetic-timeline-v5-long-gap")
    parser.add_argument("--audio-id-prefix", default="galgame-synthv5-lg")
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--trim-head-s", type=float, default=0.0)
    parser.add_argument("--trim-tail-s", type=float, default=0.0)
    parser.add_argument("--max-speech-s", type=float, default=8.0, help="Crop speech islands to this length; <=0 disables.")
    parser.add_argument("--min-speech-s", type=float, default=0.05)
    parser.add_argument("--gap-min-s", type=float, default=1.0)
    parser.add_argument("--gap-max-s", type=float, default=6.0)
    parser.add_argument(
        "--touch-gap-prob",
        type=float,
        default=0.0,
        help="Probability that an internal speech-island gap is exactly zero samples.",
    )
    parser.add_argument(
        "--short-gap-prob",
        type=float,
        default=0.0,
        help="Probability that an internal speech-island gap is sampled from [0, short-gap-max-s].",
    )
    parser.add_argument(
        "--short-gap-max-s",
        type=float,
        default=0.12,
        help="Upper bound for short internal gaps used to train cut_point utterance boundaries.",
    )
    parser.add_argument("--leading-gap-min-s", type=float, default=0.5)
    parser.add_argument("--leading-gap-max-s", type=float, default=4.0)
    parser.add_argument("--trailing-gap-min-s", type=float, default=0.5)
    parser.add_argument("--trailing-gap-max-s", type=float, default=4.0)
    parser.add_argument("--noise-rms", type=float, default=0.01)
    parser.add_argument("--hum-rms", type=float, default=0.02)
    parser.add_argument(
        "--negative-manifest",
        action="append",
        help="Optional negative clip manifest JSON. Repeatable; used for real non-speech gaps.",
    )
    parser.add_argument("--negative-gap-prob", type=float, default=0.9)
    parser.add_argument(
        "--background-manifest",
        action="append",
        help="Optional negative clip manifest JSON. Repeatable; mixed under the final audio at SNR.",
    )
    parser.add_argument("--background-mix-prob", type=float, default=1.0)
    parser.add_argument("--background-snr-db-min", type=float, default=8.0)
    parser.add_argument("--background-snr-db-max", type=float, default=22.0)
    parser.add_argument(
        "--background-switch-prob",
        type=float,
        default=0.35,
        help=(
            "Probability of a mid-example background bed switch, crossfaded "
            "INSIDE a speech unit so background change is decorrelated from "
            "boundaries."
        ),
    )
    parser.add_argument(
        "--unit-loudness-align-prob",
        type=float,
        default=1.0,
        help=(
            "binary_hardmix only: probability that all units of an example are "
            "RMS-aligned to one example-level target, removing the loudness "
            "step every boundary otherwise carries."
        ),
    )
    parser.add_argument("--unit-loudness-jitter-db", type=float, default=2.0)
    parser.add_argument("--unit-loudness-target-db-min", type=float, default=-26.0)
    parser.add_argument("--unit-loudness-target-db-max", type=float, default=-18.0)
    parser.add_argument(
        "--negative-loudness-offset-db-min",
        type=float,
        default=-10.0,
        help="Real-negative 0-units are aligned to target + uniform offset (dB).",
    )
    parser.add_argument("--negative-loudness-offset-db-max", type=float, default=2.0)
    parser.add_argument(
        "--same-source-adjacent-prob",
        type=float,
        default=0.5,
        help=(
            "binary_hardmix only: probability the next speech unit comes from "
            "the SAME source recording as the previous one."
        ),
    )
    parser.add_argument(
        "--gap-duration-samples",
        default="",
        help=(
            "Optional JSON file (list of seconds, or {\"durations_s\": [...]}) "
            "with real-domain pause durations; hardmix noise unit durations are "
            "then sampled from this pool instead of uniform."
        ),
    )
    parser.add_argument(
        "--post-speech-noise-flags",
        default="breath,moan,sigh,gasp,pant,kiss",
        help=(
            "Comma-separated background_type keywords preferred for the FIRST "
            "noise unit right after a speech unit."
        ),
    )
    parser.add_argument(
        "--post-speech-noise-flag-prob",
        type=float,
        default=0.6,
    )
    parser.add_argument("--crossfade-ms-min", type=float, default=5.0)
    parser.add_argument("--crossfade-ms-max", type=float, default=30.0)
    parser.add_argument("--crossfade-curve", choices=("equal_power", "linear"), default="equal_power")
    parser.add_argument("--gain-db-min", type=float, default=-3.0)
    parser.add_argument("--gain-db-max", type=float, default=3.0)
    parser.add_argument("--filter-prob", type=float, default=0.25)
    parser.add_argument("--filter-mode", choices=("none", "random", "lowpass", "bandpass"), default="random")
    parser.add_argument("--lowpass-min-hz", type=float, default=3000.0)
    parser.add_argument("--lowpass-max-hz", type=float, default=7200.0)
    parser.add_argument("--bandpass-low-min-hz", type=float, default=80.0)
    parser.add_argument("--bandpass-low-max-hz", type=float, default=300.0)
    parser.add_argument("--bandpass-high-min-hz", type=float, default=3000.0)
    parser.add_argument("--bandpass-high-max-hz", type=float, default=7200.0)
    parser.add_argument("--codec-prob", type=float, default=0.05)
    parser.add_argument("--codec-aug", choices=("none", "random", "pcm16", "mulaw"), default="random")
    parser.add_argument("--overlap-speech-prob", type=float, default=0.12)
    parser.add_argument("--overlap-snr-db-min", type=float, default=0.0)
    parser.add_argument("--overlap-snr-db-max", type=float, default=10.0)
    parser.add_argument("--overlap-max-speech-s", type=float, default=2.0)
    parser.add_argument(
        "--speech-label-dilation-s",
        type=float,
        default=0.08,
        help="Dilate speech labels on both sides without changing audio, useful for high-recall boundary training.",
    )
    parser.add_argument("--text-separator", default=" ")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "speech-boundary-ja" / "galgame-synthetic-timeline"))
    parser.add_argument("--output-jsonl", default="labels.jsonl")
    parser.add_argument("--output-manifest", default="manifest.json")
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.speech_clips_per_example <= 0:
        parser.error("--speech-clips-per-example must be positive")
    for minimum_name, maximum_name in (
        ("hardmix_speech_count_min", "hardmix_speech_count_max"),
        ("hardmix_edge_noise_count_min", "hardmix_edge_noise_count_max"),
        ("hardmix_inter_noise_count_min", "hardmix_inter_noise_count_max"),
    ):
        minimum = getattr(args, minimum_name)
        maximum = getattr(args, maximum_name)
        if minimum < 0:
            parser.error(f"--{minimum_name.replace('_', '-')} must be non-negative")
        if maximum <= minimum:
            parser.error(
                f"--{maximum_name.replace('_', '-')} must be greater than "
                f"--{minimum_name.replace('_', '-')}"
            )
    if args.hardmix_speech_count_min <= 0:
        parser.error("--hardmix-speech-count-min must be positive")
    if not 0.0 < args.partition_train_ratio < 1.0:
        parser.error("--partition-train-ratio must be in (0, 1)")
    if not 0.0 < args.partition_val_ratio < 1.0:
        parser.error("--partition-val-ratio must be in (0, 1)")
    if args.partition_train_ratio + args.partition_val_ratio >= 1.0:
        parser.error("--partition-train-ratio + --partition-val-ratio must be < 1")
    if args.frame_hop_s <= 0.0:
        parser.error("--frame-hop-s must be positive")
    if args.min_speech_s <= 0.0:
        parser.error("--min-speech-s must be positive")
    for name in (
        "trim_head_s",
        "trim_tail_s",
        "cut_point_max_gap_s",
        "long_gap_boundary_min_s",
        "gap_min_s",
        "gap_max_s",
        "touch_gap_prob",
        "short_gap_prob",
        "short_gap_max_s",
        "leading_gap_min_s",
        "leading_gap_max_s",
        "trailing_gap_min_s",
        "trailing_gap_max_s",
        "noise_rms",
        "hum_rms",
        "negative_gap_prob",
        "background_mix_prob",
        "speech_label_dilation_s",
        "crossfade_ms_min",
        "crossfade_ms_max",
        "filter_prob",
        "lowpass_min_hz",
        "lowpass_max_hz",
        "bandpass_low_min_hz",
        "bandpass_low_max_hz",
        "bandpass_high_min_hz",
        "bandpass_high_max_hz",
        "codec_prob",
        "overlap_speech_prob",
        "overlap_max_speech_s",
        "hardmix_noise_duration_min_s",
        "hardmix_noise_duration_max_s",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("negative_gap_prob", "background_mix_prob", "filter_prob", "codec_prob", "overlap_speech_prob"):
        if getattr(args, name) > 1.0:
            parser.error(f"--{name.replace('_', '-')} must be <= 1")
    if args.gap_max_s < args.gap_min_s:
        parser.error("--gap-max-s must be >= --gap-min-s")
    if args.hardmix_noise_duration_max_s <= args.hardmix_noise_duration_min_s:
        parser.error(
            "--hardmix-noise-duration-max-s must be greater than "
            "--hardmix-noise-duration-min-s"
        )
    if not 0.0 <= args.touch_gap_prob <= 1.0:
        parser.error("--touch-gap-prob must be in [0, 1]")
    if not 0.0 <= args.short_gap_prob <= 1.0:
        parser.error("--short-gap-prob must be in [0, 1]")
    if args.touch_gap_prob + args.short_gap_prob > 1.0:
        parser.error("--touch-gap-prob + --short-gap-prob must be <= 1")
    if args.short_gap_max_s < 0.0:
        parser.error("--short-gap-max-s must be non-negative")
    if args.leading_gap_max_s < args.leading_gap_min_s:
        parser.error("--leading-gap-max-s must be >= --leading-gap-min-s")
    if args.trailing_gap_max_s < args.trailing_gap_min_s:
        parser.error("--trailing-gap-max-s must be >= --trailing-gap-min-s")
    if args.background_snr_db_max < args.background_snr_db_min:
        parser.error("--background-snr-db-max must be >= --background-snr-db-min")
    if args.crossfade_ms_max < args.crossfade_ms_min:
        parser.error("--crossfade-ms-max must be >= --crossfade-ms-min")
    if args.gain_db_max < args.gain_db_min:
        parser.error("--gain-db-max must be >= --gain-db-min")
    if args.lowpass_max_hz < args.lowpass_min_hz:
        parser.error("--lowpass-max-hz must be >= --lowpass-min-hz")
    if args.bandpass_low_max_hz < args.bandpass_low_min_hz:
        parser.error("--bandpass-low-max-hz must be >= --bandpass-low-min-hz")
    if args.bandpass_high_max_hz < args.bandpass_high_min_hz:
        parser.error("--bandpass-high-max-hz must be >= --bandpass-high-min-hz")
    if args.overlap_snr_db_max < args.overlap_snr_db_min:
        parser.error("--overlap-snr-db-max must be >= --overlap-snr-db-min")
    return args


if __name__ == "__main__":
    build_synthetic_timeline(parse_args())
