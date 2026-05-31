#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.loading import load_audio_16k_mono  # noqa: E402
from vad.fusionvad_ja import TeacherSegment, build_supervised_record, write_jsonl  # noqa: E402


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"manifest must be a JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


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
) -> tuple[np.ndarray, str]:
    if samples <= 0:
        return np.zeros(0, dtype=np.float32), "empty"
    mode = index % 4
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
) -> tuple[np.ndarray, str, dict[str, Any] | None]:
    if samples <= 0:
        return np.zeros(0, dtype=np.float32), "empty", None
    if negative_rows and negative_gap_prob > 0.0 and float(rng.random()) < negative_gap_prob:
        try:
            audio, detail = choose_real_negative_gap(rows=negative_rows, samples=samples, rng=rng)
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
    )
    return audio, mode, detail


def expand_segments(
    segments: list[TeacherSegment],
    *,
    duration_s: float,
    pad_s: float,
) -> list[TeacherSegment]:
    if pad_s <= 0.0:
        return list(segments)
    expanded = [
        TeacherSegment(
            start=max(0.0, segment.start - pad_s),
            end=min(duration_s, segment.end + pad_s),
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


def mix_background_audio(
    audio: np.ndarray,
    *,
    background_rows: list[dict[str, Any]],
    rng: np.random.Generator,
    snr_db_min: float,
    snr_db_max: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if audio.size <= 0:
        return audio, {"skipped": "empty_audio"}
    background_row_index = int(rng.integers(0, len(background_rows)))
    background, detail = load_negative_audio(
        background_rows[background_row_index],
        samples=audio.size,
        rng=rng,
    )
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

    sample_rate = 16000
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
    source_cursor = 0
    synthetic_count = min(args.count, math.ceil(len(source_rows) / args.speech_clips_per_example))
    if args.reuse_sources:
        synthetic_count = args.count
    for output_index in range(synthetic_count):
        audio = np.zeros(0, dtype=np.float32)
        speech_segments: list[TeacherSegment] = []
        source_details: list[dict[str, Any]] = []
        transition_regions: list[dict[str, Any]] = []
        previous_kind: str | None = None
        gap_index = output_index * (args.speech_clips_per_example + 2)

        leading_samples = gap_samples(
            rng=rng,
            sample_rate=sample_rate,
            min_s=args.leading_gap_min_s,
            max_s=args.leading_gap_max_s,
        )
        leading_gap, mode, gap_detail = build_gap(
            samples=leading_samples,
            sample_rate=sample_rate,
            index=gap_index,
            rng=rng,
            noise_rms=args.noise_rms,
            hum_rms=args.hum_rms,
            negative_rows=negative_rows,
            negative_gap_prob=args.negative_gap_prob,
        )
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
                "gap_type": mode,
                "mode": mode,
                "synthetic_start_s": gap_start_sample / sample_rate,
                "synthetic_end_s": gap_end_sample / sample_rate,
            }
            if gap_detail:
                gap_row.update(gap_detail)
            source_details.append(gap_row)
            if transition:
                transition_regions.append(transition)

        for speech_index in range(args.speech_clips_per_example):
            if source_cursor >= len(source_rows):
                if not args.reuse_sources:
                    break
                source_cursor = 0
                if args.shuffle:
                    rng.shuffle(source_rows)
            row = source_rows[source_cursor]
            source_cursor += 1
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
                        "index": source_cursor - 1,
                        "audio_id": str(row.get("audio_id") or ""),
                        "reason": "speech_load_error",
                        "error": str(exc),
                    }
                )
                continue
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
            speech_segments.append(
                TeacherSegment(
                    start=speech_start_sample / sample_rate,
                    end=speech_end_sample / sample_rate,
                    score=1.0,
                )
            )
            source_detail.update(
                {
                    "synthetic_start_s": speech_start_sample / sample_rate,
                    "synthetic_end_s": speech_end_sample / sample_rate,
                }
            )
            source_details.append(source_detail)

            if speech_index < args.speech_clips_per_example - 1:
                middle_samples = gap_samples(
                    rng=rng,
                    sample_rate=sample_rate,
                    min_s=args.gap_min_s,
                    max_s=args.gap_max_s,
                )
                middle_gap, mode, gap_detail = build_gap(
                    samples=middle_samples,
                    sample_rate=sample_rate,
                    index=gap_index + speech_index + 1,
                    rng=rng,
                    noise_rms=args.noise_rms,
                    hum_rms=args.hum_rms,
                    negative_rows=negative_rows,
                    negative_gap_prob=args.negative_gap_prob,
                )
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
                        "gap_type": mode,
                        "mode": mode,
                        "synthetic_start_s": gap_start_sample / sample_rate,
                        "synthetic_end_s": gap_end_sample / sample_rate,
                    }
                    if gap_detail:
                        gap_row.update(gap_detail)
                    source_details.append(gap_row)
                    if transition:
                        transition_regions.append(transition)

        trailing_samples = gap_samples(
            rng=rng,
            sample_rate=sample_rate,
            min_s=args.trailing_gap_min_s,
            max_s=args.trailing_gap_max_s,
        )
        trailing_gap, mode, gap_detail = build_gap(
            samples=trailing_samples,
            sample_rate=sample_rate,
            index=gap_index + args.speech_clips_per_example + 1,
            rng=rng,
            noise_rms=args.noise_rms,
            hum_rms=args.hum_rms,
            negative_rows=negative_rows,
            negative_gap_prob=args.negative_gap_prob,
        )
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
                "gap_type": mode,
                "mode": mode,
                "synthetic_start_s": gap_start_sample / sample_rate,
                "synthetic_end_s": gap_end_sample / sample_rate,
            }
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
            pad_s=args.speech_label_pad_s,
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
        records.append(record)
        manifest_rows.append(
            {
                "audio_id": audio_id,
                "audio": str(audio_path),
                "duration_s": duration_s,
                "sample_rate": sample_rate,
                "source": args.source,
                "label_quality": record.label_quality,
                "input": ",".join(
                    str(item.get("source_input") or "") for item in source_details if item.get("source_input")
                ),
                "source_audio_ids": [
                    str(item.get("source_audio_id")) for item in source_details if item.get("source_audio_id")
                ],
                "speech_segments": [{"start": segment.start, "end": segment.end} for segment in label_segments],
                "actual_speech_segments": [{"start": segment.start, "end": segment.end} for segment in speech_segments],
                "speech_label_pad_s": args.speech_label_pad_s,
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
                "text": record.text,
                "frame_hop_s": record.frame_hop_s,
                "speech_frames": record.speech_frames,
                "speech_segments": [{"start": segment.start, "end": segment.end} for segment in label_segments],
                "actual_speech_segments": [{"start": segment.start, "end": segment.end} for segment in speech_segments],
                "speech_label_pad_s": args.speech_label_pad_s,
                "source_audio_ids": [
                    str(item.get("source_audio_id")) for item in source_details if item.get("source_audio_id")
                ],
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
                "duration_s": duration_s,
                "speech_segments": [{"start": segment.start, "end": segment.end} for segment in label_segments],
                "actual_speech_segments": [{"start": segment.start, "end": segment.end} for segment in speech_segments],
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
    summary = {
        "manifest": str(Path(args.manifest)),
        "records": len(records),
        "source_rows": len(source_rows),
        "negative_rows": len(negative_rows),
        "background_rows": len(background_rows),
        "skipped": len(skipped),
        "duration_s_total": sum(record.duration_s for record in records),
        "speech_frame_ratio": (speech_frames / total_frames) if total_frames else 0.0,
        "label_quality_counts": dict(sorted(Counter(record.label_quality for record in records).items())),
        "gap_mode_counts": dict(sorted(gap_mode_counts.items())),
        "background_mix_count": background_mix_count,
        "background_skip_count": background_skip_count,
        "overlap_mix_count": overlap_mix_count,
        "gain_aug_count": gain_aug_count,
        "filter_aug_count": filter_aug_count,
        "codec_aug_count": codec_aug_count,
        "labels": str(labels_path),
        "output_manifest": str(manifest_path),
        "boundary_manifest": str(boundary_manifest_path),
        "details": str(details_path),
        "skipped_report": str(skipped_path),
        "config": {
            "count": args.count,
            "speech_clips_per_example": args.speech_clips_per_example,
            "seed": args.seed,
            "shuffle": args.shuffle,
            "reuse_sources": args.reuse_sources,
            "trim_head_s": args.trim_head_s,
            "trim_tail_s": args.trim_tail_s,
            "max_speech_s": args.max_speech_s,
            "min_speech_s": args.min_speech_s,
            "gap_min_s": args.gap_min_s,
            "gap_max_s": args.gap_max_s,
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
            "speech_label_pad_s": args.speech_label_pad_s,
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
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--limit", type=int, help="Optional source row limit before synthesis.")
    parser.add_argument("--speech-clips-per-example", type=int, default=2)
    parser.add_argument("--reuse-sources", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
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
    parser.add_argument("--negative-gap-prob", type=float, default=0.75)
    parser.add_argument(
        "--background-manifest",
        action="append",
        help="Optional negative clip manifest JSON. Repeatable; mixed under the final audio at SNR.",
    )
    parser.add_argument("--background-mix-prob", type=float, default=0.5)
    parser.add_argument("--background-snr-db-min", type=float, default=8.0)
    parser.add_argument("--background-snr-db-max", type=float, default=22.0)
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
        "--speech-label-pad-s",
        type=float,
        default=0.08,
        help="Expand speech labels on both sides without changing audio, useful for high-recall boundary training.",
    )
    parser.add_argument("--text-separator", default=" ")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "galgame-synthetic-timeline"))
    parser.add_argument("--output-jsonl", default="labels.jsonl")
    parser.add_argument("--output-manifest", default="manifest.json")
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.speech_clips_per_example <= 0:
        parser.error("--speech-clips-per-example must be positive")
    if args.frame_hop_s <= 0.0:
        parser.error("--frame-hop-s must be positive")
    if args.min_speech_s <= 0.0:
        parser.error("--min-speech-s must be positive")
    for name in (
        "trim_head_s",
        "trim_tail_s",
        "gap_min_s",
        "gap_max_s",
        "leading_gap_min_s",
        "leading_gap_max_s",
        "trailing_gap_min_s",
        "trailing_gap_max_s",
        "noise_rms",
        "hum_rms",
        "negative_gap_prob",
        "background_mix_prob",
        "speech_label_pad_s",
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
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("negative_gap_prob", "background_mix_prob", "filter_prob", "codec_prob", "overlap_speech_prob"):
        if getattr(args, name) > 1.0:
            parser.error(f"--{name.replace('_', '-')} must be <= 1")
    if args.gap_max_s < args.gap_min_s:
        parser.error("--gap-max-s must be >= --gap-min-s")
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
