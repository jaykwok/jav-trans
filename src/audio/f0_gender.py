import logging
import os
from collections import Counter

import librosa
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from audio.loading import load_audio_16k_mono

logger = logging.getLogger(__name__)

F0_THRESHOLD_HZ = float(os.getenv("F0_THRESHOLD_HZ", "160.0"))
F0_NAN_RATIO_THRESHOLD = float(os.getenv("F0_NAN_RATIO_THRESHOLD", "0.6"))
F0_GENDER_CARRYOVER_ENABLED = os.getenv(
    "F0_GENDER_CARRYOVER_ENABLED", "1"
).strip().lower() in {"1", "true", "yes", "on"}
F0_GENDER_CARRYOVER_MAX_GAP_S = float(os.getenv("F0_GENDER_CARRYOVER_MAX_GAP_S", "10.0"))
F0_GENDER_CARRYOVER_MAX_SEGMENT_S = float(
    os.getenv("F0_GENDER_CARRYOVER_MAX_SEGMENT_S", "8.0")
)


def _classify_f0(
    values,
    *,
    threshold_hz: float,
    nan_ratio_threshold: float = F0_NAN_RATIO_THRESHOLD,
) -> str | None:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return None
    valid = np.isfinite(arr)
    nan_ratio = 1.0 - (float(valid.sum()) / float(arr.size))
    if nan_ratio > nan_ratio_threshold or not valid.any():
        return None
    median_f0 = float(np.nanmedian(arr[valid]))
    if not np.isfinite(median_f0):
        return None
    return "M" if median_f0 < threshold_hz else "F"


def _load_audio(audio_path, _load_fn=None):
    if _load_fn is not None:
        return _load_fn(str(audio_path), sr=16000, mono=True)
    return load_audio_16k_mono(str(audio_path))


def _normalize_audio(y, sr: int) -> tuple[np.ndarray, int]:
    audio = np.asarray(y, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if sr != 16000:
        divisor = int(np.gcd(int(sr), 16000))
        audio = signal.resample_poly(
            audio,
            16000 // divisor,
            int(sr) // divisor,
        ).astype(np.float32, copy=False)
        sr = 16000
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)


def _run_pyin(clip, *, pyin_fn, sr: int, frame_length: int, hop_length: int):
    if len(clip) < frame_length:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=np.float32),
        )
    try:
        return pyin_fn(
            clip,
            fmin=50,
            fmax=500,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    except Exception:
        try:
            return pyin_fn(clip, fmin=50, fmax=500, sr=sr)
        except TypeError:
            return pyin_fn(clip, 50, 500)


def _frame_times(frame_count: int, *, hop_length: int, sr: int) -> np.ndarray:
    if frame_count <= 0:
        return np.zeros(0, dtype=np.float32)
    return (np.arange(frame_count, dtype=np.float32) * float(hop_length)) / float(sr)


def _f0_for_time_range(
    f0,
    times,
    start: float,
    end: float,
    *,
    fallback_to_all: bool = False,
):
    values = np.asarray(f0, dtype=np.float32)
    frame_times = np.asarray(times, dtype=np.float32)
    if values.size == 0 or frame_times.size == 0:
        return values[:0]
    mask = (frame_times >= float(start)) & (frame_times <= float(end))
    selected = values[mask]
    if selected.size == 0 and fallback_to_all:
        return values
    return selected


def _word_gender(
    word: dict,
    segment_start: float,
    f0,
    times,
    *,
    threshold_hz: float,
) -> str | None:
    try:
        word_start = max(0.0, float(word.get("start", segment_start)) - segment_start)
        word_end = max(word_start, float(word.get("end", word_start)) - segment_start)
    except (TypeError, ValueError):
        return None
    return _classify_f0(
        _f0_for_time_range(f0, times, word_start, word_end),
        threshold_hz=threshold_hz,
    )


def _majority_gender(words: list[dict]) -> str | None:
    genders = [word.get("gender") for word in words if word.get("gender") is not None]
    if not genders:
        return None
    counts = Counter(genders)
    return str(counts.most_common(1)[0][0])


def _fallback_words_for_segment(segment: dict) -> list[dict]:
    text = str(segment.get("text", "")).strip()
    if not text:
        return []

    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    duration = max(0.0, end - start)
    compact_chars = [char for char in text if not char.isspace()]
    if not compact_chars:
        return []

    words: list[dict] = []
    char_count = len(compact_chars)
    for idx, char in enumerate(compact_chars):
        word_start = start + duration * (idx / char_count)
        word_end = start + duration * ((idx + 1) / char_count)
        words.append({"start": word_start, "end": word_end, "word": char})
    return words


def _segment_time(segment: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(segment.get(key, default))
    except (TypeError, ValueError):
        return default


def _segment_duration(segment: dict) -> float:
    start = _segment_time(segment, "start")
    end = _segment_time(segment, "end", start)
    return max(0.0, end - start)


def _nearest_gender_anchor(
    segments: list[dict],
    original_genders: list,
    start_index: int,
    step: int,
) -> tuple[int, str] | None:
    index = start_index
    while 0 <= index < len(segments):
        gender = original_genders[index]
        if gender in {"M", "F"}:
            return index, str(gender)
        index += step
    return None


def _anchor_gap_s(
    segments: list[dict],
    index: int,
    left_index: int,
    right_index: int,
) -> float:
    left_end = _segment_time(segments[left_index], "end")
    right_start = _segment_time(segments[right_index], "start")
    if right_start >= left_end:
        return right_start - left_end

    segment = segments[index]
    left_segment_gap = max(
        0.0,
        _segment_time(segment, "start") - _segment_time(segments[left_index], "end"),
    )
    right_segment_gap = max(
        0.0,
        _segment_time(segments[right_index], "start") - _segment_time(segment, "end"),
    )
    return max(left_segment_gap, right_segment_gap)


def _apply_gender_carry_over(
    segments,
    *,
    enabled: bool,
    max_gap_s: float,
    max_segment_s: float,
):
    if not enabled:
        return segments

    result = [dict(segment) for segment in segments]
    original_genders = [segment.get("gender") for segment in result]
    for index, gender in enumerate(original_genders):
        if gender is not None:
            continue
        segment = result[index]
        if _segment_duration(segment) > max_segment_s:
            continue

        left_anchor = _nearest_gender_anchor(result, original_genders, index - 1, -1)
        right_anchor = _nearest_gender_anchor(result, original_genders, index + 1, 1)
        if left_anchor is None or right_anchor is None:
            continue

        left_index, left_gender = left_anchor
        right_index, right_gender = right_anchor
        if left_gender != right_gender:
            continue
        if _anchor_gap_s(result, index, left_index, right_index) > max_gap_s:
            continue
        segment["gender"] = left_gender
    return result


def detect_gender_f0_word_level(
    audio_path,
    segments,
    *,
    window_ms=300,
    hop_ms=100,
    voiced_flag_gate=True,
    median_filter_frames=9,
    min_span_ms=500,
    f0_threshold_hz=160.0,
    _load_fn=None,
    _pyin_fn=None,
) -> list[dict]:
    """Detect segment and word-level acoustic gender using pYIN F0."""

    _ = min_span_ms
    threshold_hz = float(f0_threshold_hz if f0_threshold_hz is not None else F0_THRESHOLD_HZ)
    pyin_fn = _pyin_fn if _pyin_fn is not None else librosa.pyin

    try:
        y, sr = _load_audio(audio_path, _load_fn=_load_fn)
        y, sr = _normalize_audio(y, int(sr))
    except Exception as exc:
        logger.warning("f0_gender: failed to load audio %s: %s", audio_path, exc)
        return list(segments)

    result: list[dict] = []
    for seg in segments:
        s = dict(seg)
        try:
            start = max(0.0, float(s.get("start", 0.0)))
            end = max(start, float(s.get("end", start)))
            start_i = max(0, min(len(y), int(start * sr)))
            end_i = max(start_i, min(len(y), int(end * sr)))
            clip = y[start_i:end_i]

            original_words = list(s.get("words") or [])
            words = [dict(word) for word in original_words]
            fallback_words = False
            if not words:
                words = _fallback_words_for_segment(s)
                fallback_words = True

            if len(clip) < 160:
                for word in words:
                    word["gender"] = None
                s["gender"] = None
                if original_words:
                    s["words"] = words
                result.append(s)
                continue

            hop_length = max(1, int(sr * float(hop_ms) / 1000.0))
            frame_length = max(hop_length, int(sr * float(window_ms) / 1000.0))
            if len(clip) < frame_length:
                for word in words:
                    word["gender"] = None
                s["gender"] = None
                if original_words:
                    s["words"] = words
                result.append(s)
                continue
            f0, voiced_flag, _ = _run_pyin(
                clip,
                pyin_fn=pyin_fn,
                sr=sr,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            f0 = np.asarray(f0, dtype=np.float32)
            voiced_flag = np.asarray(voiced_flag, dtype=bool)

            if f0.size == 0:
                for word in words:
                    word["gender"] = None
                s["gender"] = None
                if original_words:
                    s["words"] = words
                result.append(s)
                continue

            if voiced_flag_gate and voiced_flag.size == f0.size:
                f0 = f0.copy()
                f0[~voiced_flag] = np.nan

            filter_size = max(1, int(median_filter_frames))
            if filter_size > 1:
                valid = np.isfinite(f0)
                fill_value = float(np.nanmedian(f0[valid])) if valid.any() else np.nan
                if np.isfinite(fill_value):
                    filtered_input = np.where(valid, f0, fill_value)
                    f0 = median_filter(filtered_input, size=filter_size).astype(np.float32)
                    if voiced_flag_gate and voiced_flag.size == f0.size:
                        f0[~voiced_flag] = np.nan

            times = _frame_times(f0.size, hop_length=hop_length, sr=sr)
            if fallback_words:
                s["gender"] = _classify_f0(
                    _f0_for_time_range(f0, times, 0.0, end - start, fallback_to_all=True),
                    threshold_hz=threshold_hz,
                )
            else:
                for word in words:
                    word["gender"] = _word_gender(
                        word,
                        start,
                        f0,
                        times,
                        threshold_hz=threshold_hz,
                    )
                s["words"] = words
                s["gender"] = _majority_gender(words)
        except Exception as exc:
            logger.warning("f0_gender: segment %s error: %s", s.get("start"), exc)
            s["gender"] = None
            if s.get("words"):
                s["words"] = [dict(word, gender=None) for word in s.get("words") or []]
        result.append(s)
    return _apply_gender_carry_over(
        result,
        enabled=F0_GENDER_CARRYOVER_ENABLED,
        max_gap_s=F0_GENDER_CARRYOVER_MAX_GAP_S,
        max_segment_s=F0_GENDER_CARRYOVER_MAX_SEGMENT_S,
    )


def detect_gender_f0(
    audio_path,
    segments: list[dict],
    *,
    threshold_hz: float | None = None,
    nan_ratio_threshold: float | None = None,
    _load_fn=None,
    _pyin_fn=None,
) -> list[dict]:
    _ = nan_ratio_threshold
    return detect_gender_f0_word_level(
        audio_path,
        segments,
        f0_threshold_hz=threshold_hz if threshold_hz is not None else F0_THRESHOLD_HZ,
        _load_fn=_load_fn,
        _pyin_fn=_pyin_fn,
    )
