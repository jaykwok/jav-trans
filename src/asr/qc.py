import logging
import os
import re
import statistics
from typing import Callable


# check_logprob_quality returns {"verdict": "ok|warn|reject", "reason": str|None, "metrics": dict}.
# Callers that persist it on text results should use an asr_qc_ prefix.
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}
_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in _FALSE_VALUES:
        return False
    if value in _TRUE_VALUES:
        return True
    return default


def asr_qc_enabled() -> bool:
    return _env_bool("ASR_QC_ENABLED", True)


ASR_QC_ENABLED = asr_qc_enabled()
ASR_QC_REPETITION_THRESHOLD = max(
    1,
    int(os.getenv("ASR_QC_REPETITION_THRESHOLD", "10")),
)

_REPEAT_MIN_RUN = max(2, int(os.getenv("ASR_QC_REPEAT_MIN_RUN", "4")))
_REPEAT_MIN_RATIO = float(os.getenv("ASR_QC_REPEAT_MIN_RATIO", "0.40"))
_HALLUCINATION_MIN_CHARS = max(
    1, int(os.getenv("ASR_QC_HALLUCINATION_MIN_CHARS", "50"))
)
_DENSITY_MIN_CHARS = max(1, int(os.getenv("ASR_QC_DENSITY_MIN_CHARS", "30")))
_LOW_INFO_DURATION_S = float(os.getenv("ASR_QC_LOW_INFO_DURATION", "6.0"))
_LOW_INFO_MAX_CHARS = max(0, int(os.getenv("ASR_QC_LOW_INFO_MAX_CHARS", "5")))
_EMPTY_DURATION_S = float(os.getenv("ASR_QC_EMPTY_DURATION", "1.0"))
_LONG_LOW_VALUE_DURATION_S = float(
    os.getenv("ASR_QC_LONG_LOW_VALUE_DURATION", "6.0")
)
_NONLEXICAL_REPEAT_MAX_DURATION_S = float(
    os.getenv("ASR_QC_NONLEXICAL_REPEAT_MAX_DURATION", "12.0")
)
_NONLEXICAL_REPEAT_MAX_UNIT_LEN = max(
    1,
    int(os.getenv("ASR_QC_NONLEXICAL_REPEAT_MAX_UNIT_LEN", "4")),
)
_NONLEXICAL_REPEAT_MAX_CPS = float(
    os.getenv("ASR_QC_NONLEXICAL_REPEAT_MAX_CPS", "8.0")
)
_REPETITION_TRUNCATE_KEEP_RUNS = max(
    1,
    int(os.getenv("ASR_QC_REPETITION_TRUNCATE_KEEP_RUNS", "3")),
)
_DIALOGUE_MARKER_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u9fff]|[!?！？。…「」『』、]"
)
_KANA_RE = re.compile(r"^[\u3040-\u30ff]+$")

_logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _adaptive_hard_max_chars_per_sec() -> float:
    return _env_float("ASR_QC_ADAPTIVE_HARD_MAX_CHARS_PER_SEC", 14.0)


def _adaptive_logprob_threshold(result: dict, base_threshold: float) -> dict:
    no_speech_prob = _optional_float(result.get("no_speech_prob"))
    compression_ratio = _optional_float(result.get("compression_ratio"))
    duration_s = _optional_float(result.get("duration_s"))
    compact_chars = int(_optional_float(result.get("compact_chars")) or 0)
    chars_per_sec = _optional_float(result.get("chars_per_sec")) or 0.0
    low_value = bool(result.get("low_value"))
    text = str(result.get("text") or result.get("raw_text") or "")
    repeat = result.get("max_repeat") if isinstance(result.get("max_repeat"), dict) else {}
    repeat_run = int(_optional_float(repeat.get("run")) or 0)
    repeat_ratio = float(_optional_float(repeat.get("ratio")) or 0.0)
    nonlexical_repetition = bool(result.get("nonlexical_repetition"))

    min_threshold = _env_float("ASR_QC_ADAPTIVE_MIN_LOGPROB", -0.95)
    max_threshold = _env_float("ASR_QC_ADAPTIVE_MAX_LOGPROB", -0.55)
    if min_threshold > max_threshold:
        min_threshold, max_threshold = max_threshold, min_threshold

    hard_reasons: list[str] = []
    if no_speech_prob is not None and no_speech_prob >= _env_float(
        "ASR_QC_ADAPTIVE_HARD_NOSPEECH_THRESHOLD",
        0.5,
    ):
        hard_reasons.append("high_no_speech")
    if compression_ratio is not None and compression_ratio >= _env_float(
        "ASR_QC_ADAPTIVE_HARD_COMPRESSION_THRESHOLD",
        2.0,
    ):
        hard_reasons.append("high_compression")
    if compact_chars >= _DENSITY_MIN_CHARS and chars_per_sec >= _env_float(
        "ASR_QC_ADAPTIVE_HARD_MAX_CHARS_PER_SEC",
        14.0,
    ):
        hard_reasons.append("abnormal_char_density")
    if (
        not nonlexical_repetition
        and repeat_run >= _REPEAT_MIN_RUN
        and repeat_ratio >= _env_float(
            "ASR_QC_ADAPTIVE_HARD_REPEAT_RATIO",
            0.45,
        )
    ):
        hard_reasons.append("repeat_ngram_loop")

    if no_speech_prob is not None and no_speech_prob < 0.01:
        speech_confidence_relax = 0.10
    elif no_speech_prob is not None and no_speech_prob < 0.10:
        speech_confidence_relax = 0.05
    else:
        speech_confidence_relax = 0.0

    has_dialogue_marker = bool(_DIALOGUE_MARKER_RE.search(text))
    if compact_chars >= 16 and 1.2 <= chars_per_sec <= 6.0:
        semantic_text_relax = 0.10
    elif compact_chars >= 8 and 0.8 <= chars_per_sec <= 8.0:
        semantic_text_relax = 0.06
    elif compact_chars >= 3 and has_dialogue_marker:
        semantic_text_relax = 0.03
    else:
        semantic_text_relax = 0.0

    hallucination_risk_penalty = 0.0
    if low_value and (duration_s or 0.0) >= _LONG_LOW_VALUE_DURATION_S:
        hallucination_risk_penalty += 0.12
    if repeat_run >= 3 and not nonlexical_repetition:
        hallucination_risk_penalty += 0.10
    if compression_ratio is not None and compression_ratio >= 1.70:
        hallucination_risk_penalty += 0.08
    if chars_per_sec < 0.6:
        hallucination_risk_penalty += 0.08

    threshold = _clamp(
        base_threshold
        - speech_confidence_relax
        - semantic_text_relax
        + hallucination_risk_penalty,
        min_threshold,
        max_threshold,
    )
    video_threshold = _optional_float(result.get("video_logprob_threshold"))
    if video_threshold is not None:
        threshold = min(threshold, video_threshold)
    return {
        "threshold": threshold,
        "base_threshold": base_threshold,
        "video_threshold": video_threshold,
        "min_threshold": min_threshold,
        "max_threshold": max_threshold,
        "speech_confidence_relax": speech_confidence_relax,
        "semantic_text_relax": semantic_text_relax,
        "hallucination_risk_penalty": round(hallucination_risk_penalty, 6),
        "hard_reject_reasons": hard_reasons,
        "has_dialogue_marker": has_dialogue_marker,
    }


def check_logprob_quality(result: dict) -> dict:
    """
    Returns {'verdict': 'ok'/'warn'/'reject', 'reason': str|None, 'metrics': dict}.
    Skips any check where the signal is None.
    Reads thresholds dynamically so tests can override via monkeypatch.setenv.
    """
    nospeech_threshold = _env_float("ASR_QC_ADAPTIVE_HARD_NOSPEECH_THRESHOLD", 0.5)
    logprob_threshold = _env_float("ASR_QC_ADAPTIVE_BASE_LOGPROB", -0.7)
    compression_threshold = _env_float(
        "ASR_QC_ADAPTIVE_HARD_COMPRESSION_THRESHOLD",
        2.0,
    )

    no_speech_prob = _optional_float(result.get("no_speech_prob"))
    avg_logprob = _optional_float(result.get("avg_logprob"))
    compression_ratio = _optional_float(result.get("compression_ratio"))
    adaptive = _adaptive_logprob_threshold(result, logprob_threshold)
    logprob_threshold = float(adaptive["threshold"])

    reject_reasons: list[str] = []
    warn_reasons: list[str] = []
    hard_reasons = list(adaptive.get("hard_reject_reasons") or [])
    if hard_reasons:
        reject_reasons.extend(hard_reasons)
    else:
        if no_speech_prob is not None and no_speech_prob > nospeech_threshold:
            reject_reasons.append("high_no_speech")
        if compression_ratio is not None and compression_ratio > compression_threshold:
            reject_reasons.append("high_compression")
    if avg_logprob is not None and avg_logprob < logprob_threshold:
        reject_reasons.append("low_logprob")

    if reject_reasons:
        verdict = "reject"
        reason = ",".join(reject_reasons + warn_reasons)
    elif warn_reasons:
        verdict = "warn"
        reason = ",".join(warn_reasons)
    else:
        verdict = "ok"
        reason = None

    return {
        "verdict": verdict,
        "reason": reason,
        "metrics": {
            "avg_logprob": avg_logprob,
            "compression_ratio": compression_ratio,
            "no_speech_prob": no_speech_prob,
            "logprob_threshold": logprob_threshold,
            "compression_threshold": compression_threshold,
            "nospeech_threshold": nospeech_threshold,
            "review_uncertain_enabled": True,
            "precision_policy": "adaptive",
            "adaptive": adaptive,
        },
    }


def build_asr_manifest(segments: list[dict]) -> dict:
    total = len(segments)
    empty_count = sum(
        1 for segment in segments if not str(segment.get("text", "") or "").strip()
    )
    empty_ratio = empty_count / max(1, total)
    return {
        "segment_count": total,
        "empty_text_count": empty_count,
        "empty_text_ratio": round(empty_ratio, 6),
        "qc_empty_threshold": _env_float("QC_EMPTY_THRESHOLD", 0.05),
    }


def asr_qc_gate(segments: list[dict], headless: bool = False) -> bool:
    manifest = build_asr_manifest(segments)
    ratio = float(manifest["empty_text_ratio"])
    threshold = float(manifest["qc_empty_threshold"])
    if ratio <= threshold:
        return True

    empty_count = int(manifest["empty_text_count"])
    total = int(manifest["segment_count"])
    warning = (
        "[asr-qc] empty_text_ratio="
        f"{ratio:.1%} ({empty_count}/{total}) exceeds threshold {threshold:.1%}"
    )
    print(f"[WARN] {warning}")

    if _env_bool("QC_IGNORE_EMPTY", False):
        print("[WARN] QC_IGNORE_EMPTY=1; continuing despite high ASR empty-text rate")
        return True

    if headless:
        print("[WARN] Headless mode blocked translation; set QC_IGNORE_EMPTY=1 to continue")
        return False

    answer = input("ASR empty-text rate is high. Continue translation? [y/N]: ")
    return answer.strip().lower() in {"y", "yes"}

_COMPACT_RE = re.compile(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u9fff]+")
_PUNCT_RE = re.compile(
    r"[\s\u3000\u3001\u3002\uff01\uff1f\uff0c\uff0e,.!?~\u301c"
    r"\u30fc\u2010-\u2015\u300c\u300d\u300e\u300f()\[\]{}]+"
)
_MOJIBAKE_RE = re.compile(
    r"[\ufffd\u00c2\u00c3\u00c4\u00a4\u00a5\u00a6\u00a7]"
)


def _compact_text(text: str) -> str:
    return _COMPACT_RE.sub("", text or "")


def _strip_punctuation(text: str) -> str:
    return _PUNCT_RE.sub("", text or "")


def _preview(text: str, limit: int = 80) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _collapse_repeated_noise(text: str) -> str:
    cleaned = re.sub(r"[ \t]+", " ", (text or "").strip())
    return cleaned.strip()


def _find_max_repeat(compact: str) -> dict:
    if not compact:
        return {
            "unit": "",
            "unit_len": 0,
            "run": 0,
            "chars": 0,
            "ratio": 0.0,
        }

    best = {
        "unit": "",
        "unit_len": 0,
        "run": 1,
        "chars": 0,
        "ratio": 0.0,
    }
    max_unit_len = min(12, max(1, len(compact) // 2))

    for unit_len in range(1, max_unit_len + 1):
        i = 0
        while i <= len(compact) - unit_len:
            unit = compact[i : i + unit_len]
            run = 1
            j = i + unit_len
            while j + unit_len <= len(compact) and compact[j : j + unit_len] == unit:
                run += 1
                j += unit_len

            repeated_chars = run * unit_len
            ratio = repeated_chars / max(1, len(compact))
            if run > 1 and (run, ratio, repeated_chars) > (
                int(best["run"]),
                float(best["ratio"]),
                int(best["chars"]),
            ):
                best = {
                    "unit": unit,
                    "unit_len": unit_len,
                    "run": run,
                    "chars": repeated_chars,
                    "ratio": round(ratio, 4),
                }

            i += max(1, repeated_chars if run > 1 else 1)

    return best


def _find_repeated_span(text: str) -> dict:
    compact_to_text: list[int] = []
    compact_chars: list[str] = []
    for index, char in enumerate(text or ""):
        if _COMPACT_RE.fullmatch(char):
            continue
        compact_to_text.append(index)
        compact_chars.append(char)

    compact = "".join(compact_chars)
    repeat = _find_max_repeat(compact)
    unit = str(repeat.get("unit") or "")
    run = int(repeat.get("run") or 0)
    if not unit or run <= 1:
        return {
            "unit": "",
            "run": 0,
            "compact_start": 0,
            "compact_end": 0,
            "text_start": 0,
            "text_end": 0,
        }

    best_start = compact.find(unit * run)
    if best_start < 0:
        return {
            "unit": unit,
            "run": run,
            "compact_start": 0,
            "compact_end": 0,
            "text_start": 0,
            "text_end": 0,
        }
    compact_end = best_start + len(unit) * run
    text_start = compact_to_text[best_start] if best_start < len(compact_to_text) else 0
    text_end_index = compact_end - 1
    text_end = (
        compact_to_text[text_end_index] + 1
        if 0 <= text_end_index < len(compact_to_text)
        else len(text or "")
    )
    return {
        "unit": unit,
        "run": run,
        "compact_start": best_start,
        "compact_end": compact_end,
        "text_start": text_start,
        "text_end": text_end,
    }


def _repetition_repair_suggestion(text: str, repeat: dict) -> dict:
    unit = str(repeat.get("unit") or "")
    run = int(repeat.get("run") or 0)
    unit_len = int(repeat.get("unit_len") or 0)
    ratio = float(repeat.get("ratio") or 0.0)
    repeated_loop = (
        unit
        and run >= _REPEAT_MIN_RUN
        and ratio >= _REPEAT_MIN_RATIO
        and (unit_len > 1 or run >= _REPEAT_MIN_RUN * 2)
    )
    if not repeated_loop:
        return {
            "action": "none",
            "reason": "",
            "unit": unit,
            "run": run,
            "keep_runs": 0,
            "suggested_text": text or "",
            "changed": False,
        }

    keep_runs = min(run, _REPETITION_TRUNCATE_KEEP_RUNS)
    span = _find_repeated_span(text or "")
    start = int(span.get("text_start") or 0)
    end = int(span.get("text_end") or 0)
    if end <= start:
        suggested = text or ""
    else:
        suggested = (text or "")[:start] + unit * keep_runs + (text or "")[end:]
        suggested = re.sub(r"\s+", " ", suggested).strip()
    return {
        "action": "truncate_repetition",
        "reason": "repeat_ngram_loop",
        "unit": unit,
        "run": run,
        "keep_runs": keep_runs,
        "text_start": start,
        "text_end": end,
        "suggested_text": suggested,
        "changed": suggested != (text or ""),
    }


def _repeated_vocalization_profile(
    text: str,
    repeat: dict,
    *,
    duration_s: float,
    chars_per_sec: float,
) -> dict:
    compact = _strip_punctuation(text)
    unit = str(repeat.get("unit") or "")
    unit_len = int(repeat.get("unit_len") or 0)
    run = int(repeat.get("run") or 0)
    ratio = float(repeat.get("ratio") or 0.0)
    has_kanji = any("\u3400" <= char <= "\u9fff" for char in compact)
    has_latin_or_digit = any(char.isascii() and char.isalnum() for char in compact)
    kana_only = bool(compact) and bool(_KANA_RE.fullmatch(compact))
    unit_kana_only = bool(unit) and bool(_KANA_RE.fullmatch(unit))
    unique_chars = len(set(compact))
    preserve_candidate = (
        kana_only
        and unit_kana_only
        and not has_kanji
        and not has_latin_or_digit
        and unit_len <= _NONLEXICAL_REPEAT_MAX_UNIT_LEN
        and unique_chars <= max(4, unit_len + 2)
        and duration_s <= _NONLEXICAL_REPEAT_MAX_DURATION_S
        and chars_per_sec <= _NONLEXICAL_REPEAT_MAX_CPS
        and run >= _REPEAT_MIN_RUN
        and ratio >= _REPEAT_MIN_RATIO
    )
    return {
        "preserve_candidate": preserve_candidate,
        "compact_chars": len(compact),
        "unique_chars": unique_chars,
        "kana_only": kana_only,
        "unit_kana_only": unit_kana_only,
        "has_kanji": has_kanji,
        "has_latin_or_digit": has_latin_or_digit,
        "unit_len": unit_len,
        "run": run,
        "ratio": round(ratio, 4),
        "duration_s": round(duration_s, 3),
        "chars_per_sec": round(chars_per_sec, 3),
        "policy": "preserve_with_review" if preserve_candidate else "standard_repeat_qc",
    }


def _text_density_profile(text: str, *, duration_s: float) -> dict:
    normalized = _collapse_repeated_noise(text)
    compact = _strip_punctuation(normalized)
    unique_chars = len(set(compact))
    kana_chars = sum(1 for char in compact if "\u3040" <= char <= "\u30ff")
    has_kanji = any("\u3400" <= char <= "\u9fff" for char in compact)
    has_latin = any("A" <= char <= "Z" or "a" <= char <= "z" for char in compact)
    if not compact:
        level = "empty_or_punctuation"
    elif has_latin:
        level = "normal_dialogue"
    elif duration_s >= _LOW_INFO_DURATION_S and len(compact) <= _LOW_INFO_MAX_CHARS:
        level = "long_sparse_text"
    elif len(compact) <= 2:
        level = "short_vocalization_candidate"
    elif unique_chars <= 2 and len(compact) >= 4 and not has_kanji:
        level = "repeated_vocalization_candidate"
    elif kana_chars == len(compact) and len(compact) <= 5 and not has_kanji:
        level = "short_kana_dialogue_candidate"
    else:
        level = "normal_dialogue"
    return {
        "level": level,
        "compact_chars": len(compact),
        "unique_chars": unique_chars,
        "duration_s": round(duration_s, 3),
        "has_kanji": has_kanji,
        "has_latin": has_latin,
        "action": "preserve_with_review" if level != "normal_dialogue" else "preserve",
    }


def _duration_for(chunk: dict, text_result: dict) -> float:
    duration = text_result.get("duration")
    if duration is None:
        duration = chunk.get("duration")
    if duration is None:
        start = float(chunk.get("start", 0.0))
        end = float(chunk.get("end", start))
        duration = max(0.0, end - start)
    return max(0.0, float(duration or 0.0))


def _adaptive_video_logprob_threshold(enriched_results: list[dict]) -> float | None:
    hard_nospeech = _env_float(
        "ASR_QC_ADAPTIVE_HARD_NOSPEECH_THRESHOLD",
        0.5,
    )
    hard_compression = _env_float(
        "ASR_QC_ADAPTIVE_HARD_COMPRESSION_THRESHOLD",
        2.0,
    )
    hard_cps = _env_float(
        "ASR_QC_ADAPTIVE_HARD_MAX_CHARS_PER_SEC",
        14.0,
    )
    values: list[float] = []
    for result in enriched_results:
        avg_logprob = _optional_float(result.get("avg_logprob"))
        if avg_logprob is None:
            continue
        no_speech_prob = _optional_float(result.get("no_speech_prob"))
        compression_ratio = _optional_float(result.get("compression_ratio"))
        compact_chars = int(_optional_float(result.get("compact_chars")) or 0)
        chars_per_sec = _optional_float(result.get("chars_per_sec")) or 0.0
        repeat = result.get("max_repeat") if isinstance(result.get("max_repeat"), dict) else {}
        repeat_run = int(_optional_float(repeat.get("run")) or 0)
        repeat_ratio = float(_optional_float(repeat.get("ratio")) or 0.0)
        vocalization_repetition = _repeated_vocalization_profile(
            str(result.get("text") or result.get("raw_text") or ""),
            repeat,
            duration_s=float(result.get("duration_s") or 0.0),
            chars_per_sec=chars_per_sec,
        )
        if no_speech_prob is not None and no_speech_prob >= hard_nospeech:
            continue
        if compression_ratio is not None and compression_ratio >= hard_compression:
            continue
        if compact_chars >= _DENSITY_MIN_CHARS and chars_per_sec >= hard_cps:
            continue
        if (
            not vocalization_repetition["preserve_candidate"]
            and repeat_run >= _REPEAT_MIN_RUN
            and repeat_ratio >= 0.45
        ):
            continue
        values.append(avg_logprob)

    if len(values) < 5:
        return None

    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)
    raw_threshold = median - _env_float("ASR_QC_ADAPTIVE_VIDEO_MAD_MULTIPLIER", 1.8) * mad
    return _clamp(
        raw_threshold,
        _env_float("ASR_QC_ADAPTIVE_MIN_LOGPROB", -0.95),
        _env_float("ASR_QC_ADAPTIVE_VIDEO_MAX_LOGPROB", -0.70),
    )


def evaluate_asr_chunk_qc(
    chunk: dict,
    text_result: dict,
    *,
    is_low_value_text: Callable[[str], bool] | None = None,
) -> dict:
    raw_text = str(text_result.get("raw_text", "") or "")
    text = str(text_result.get("text", "") or raw_text)
    duration = _duration_for(chunk, text_result)
    compact = _strip_punctuation(text)
    context_compact = _compact_text(text)
    repeat = _find_max_repeat(context_compact)
    chars_per_sec = len(compact) / duration if duration > 0 else 0.0
    low_value = bool(is_low_value_text(text)) if is_low_value_text else False
    mojibake = bool(_MOJIBAKE_RE.search(text))
    enriched_result = {
        **text_result,
        "text": text,
        "raw_text": raw_text,
        "duration_s": duration,
        "compact_chars": len(compact),
        "chars_per_sec": chars_per_sec,
        "low_value": low_value,
        "mojibake": mojibake,
        "max_repeat": repeat,
    }

    reasons: list[str] = []
    severity = "ok"
    asr_generation = text_result.get("asr_generation")
    generation_error_kind = ""
    if isinstance(asr_generation, dict):
        generation_error_kind = str(asr_generation.get("error_kind") or "").strip()

    if generation_error_kind:
        reasons.append(f"generation_{generation_error_kind}")
        severity = "reject" if generation_error_kind in {"timeout", "worker_error"} else "warn"

    if not compact and duration >= _EMPTY_DURATION_S:
        reasons.append("empty_text_for_speech_chunk")
        if severity == "ok":
            severity = "warn"

    if mojibake:
        reasons.append("mojibake")
        severity = "reject"

    repeat_run = int(repeat["run"])
    repeat_ratio = float(repeat["ratio"])
    repeat_unit_len = int(repeat["unit_len"])
    repetition_repair = _repetition_repair_suggestion(text, repeat)
    text_density = _text_density_profile(text, duration_s=duration)
    vocalization_repetition = _repeated_vocalization_profile(
        text,
        repeat,
        duration_s=duration,
        chars_per_sec=chars_per_sec,
    )
    enriched_result["nonlexical_repetition"] = bool(
        vocalization_repetition["preserve_candidate"]
    )
    repeated_loop = (
        repeat_run >= _REPEAT_MIN_RUN
        and repeat_ratio >= _REPEAT_MIN_RATIO
        and (repeat_unit_len > 1 or repeat_run >= _REPEAT_MIN_RUN * 2)
    )
    if repeated_loop:
        if vocalization_repetition["preserve_candidate"]:
            reasons.append("repeated_nonlexical_vocalization")
            if severity == "ok":
                severity = "warn"
        else:
            reasons.append("repeat_ngram_loop")
            severity = "reject"

    if (
        not vocalization_repetition["preserve_candidate"]
        and len(compact) >= _HALLUCINATION_MIN_CHARS
        and repeat_ratio >= _REPEAT_MIN_RATIO
    ):
        reasons.append("hallucination_cap_like")
        severity = "reject"

    if (
        len(compact) >= _DENSITY_MIN_CHARS
        and chars_per_sec > _adaptive_hard_max_chars_per_sec()
    ):
        reasons.append("abnormal_char_density")
        severity = "reject"

    if duration >= _LOW_INFO_DURATION_S and len(compact) <= _LOW_INFO_MAX_CHARS and low_value:
        reasons.append("long_sparse_text")
        if severity == "ok":
            severity = "warn"
    elif duration >= _LONG_LOW_VALUE_DURATION_S and low_value:
        reasons.append("long_low_value_text")
        if severity == "ok":
            severity = "warn"

    signal_qc = check_logprob_quality(enriched_result)
    if signal_qc["verdict"] != "ok":
        signal_reasons = [
            reason
            for reason in str(signal_qc.get("reason") or signal_qc["verdict"]).split(",")
            if reason
        ]
        for reason in signal_reasons:
            if reason not in reasons:
                reasons.append(reason)
        if signal_qc["verdict"] == "reject":
            severity = "reject"
        elif severity == "ok":
            severity = "warn"
        _logger.warning(
            "[asr-qc] signal_verdict=%s reason=%s text=%s",
            signal_qc["verdict"],
            signal_qc["reason"],
            _preview(text),
        )

    return {
        "ok": severity == "ok",
        "severity": severity,
        "reasons": reasons,
        "metrics": {
            "duration_s": round(duration, 3),
            "compact_chars": len(compact),
            "chars_per_sec": round(chars_per_sec, 3),
            "low_value": low_value,
            "mojibake": mojibake,
            "max_repeat": repeat,
            "repetition_repair": repetition_repair,
            "text_density": text_density,
            "vocalization_repetition": vocalization_repetition,
            "signal_quality": signal_qc,
            "generation": asr_generation if isinstance(asr_generation, dict) else {},
        },
        "text_preview": _preview(text),
        "signal_qc": signal_qc,
    }


def evaluate_asr_text_results_qc(
    chunks: list[dict],
    text_results: list[dict],
    *,
    is_low_value_text: Callable[[str], bool] | None = None,
) -> dict:
    qc_policy = {
        "repetition_check": "on",
        "repetition_threshold": ASR_QC_REPETITION_THRESHOLD,
        "precision_policy": "adaptive",
        "review_uncertain_enabled": True,
    }

    if not asr_qc_enabled():
        return {
            "enabled": False,
            "chunk_count": len(chunks),
            "reject_count": 0,
            "warning_count": 0,
            "generation_error_count": 0,
            "generation_overflow_count": 0,
            "timeout_count": 0,
            "quarantined_count": 0,
            "empty_text_for_speech_count": 0,
            "review_uncertain_count": 0,
            "review_uncertain_items": [],
            "items": [],
            "rejected_indices": [],
            **qc_policy,
        }

    items: list[dict] = []
    rejected_indices: list[int] = []
    warning_count = 0
    generation_error_count = 0
    generation_overflow_count = 0
    timeout_count = 0
    quarantined_count = 0
    empty_text_for_speech_count = 0
    enriched_results_for_video: list[dict] = []

    for chunk, text_result in zip(chunks, text_results):
        raw_text = str(text_result.get("raw_text", "") or "")
        text = str(text_result.get("text", "") or raw_text)
        duration = _duration_for(chunk, text_result)
        compact = _strip_punctuation(text)
        context_compact = _compact_text(text)
        enriched_results_for_video.append(
            {
                **text_result,
                "text": text,
                "raw_text": raw_text,
                "duration_s": duration,
                "compact_chars": len(compact),
                "chars_per_sec": len(compact) / duration if duration > 0 else 0.0,
                "max_repeat": _find_max_repeat(context_compact),
            }
        )
    video_logprob_threshold = _adaptive_video_logprob_threshold(enriched_results_for_video)

    for index, (chunk, text_result) in enumerate(zip(chunks, text_results)):
        if video_logprob_threshold is not None:
            text_result = {**text_result, "video_logprob_threshold": video_logprob_threshold}
        qc = evaluate_asr_chunk_qc(
            chunk,
            text_result,
            is_low_value_text=is_low_value_text,
        )
        item = {
            "position": index,
            "chunk_index": chunk.get("index", index + 1),
            "start": float(chunk.get("start", 0.0)),
            "end": float(chunk.get("end", 0.0)),
            **qc,
        }
        items.append(item)
        reasons = set(qc.get("reasons") or [])
        if "empty_text_for_speech_chunk" in reasons:
            empty_text_for_speech_count += 1
        asr_generation = text_result.get("asr_generation")
        error_kind = ""
        if isinstance(asr_generation, dict):
            error_kind = str(asr_generation.get("error_kind") or "").strip()
        if error_kind:
            generation_error_count += 1
            if error_kind == "overflow":
                generation_overflow_count += 1
            if error_kind == "timeout":
                timeout_count += 1
            if error_kind in {"quarantined", "timeout", "oom", "worker_error"}:
                failure_kind = ""
                if isinstance(asr_generation, dict):
                    failure_kind = str(asr_generation.get("failure_kind") or "").strip()
                if error_kind == "quarantined" or failure_kind:
                    quarantined_count += 1
        if qc["severity"] == "reject":
            rejected_indices.append(index)
        elif qc["severity"] == "warn":
            warning_count += 1

    return {
        "enabled": True,
        "chunk_count": len(chunks),
        "reject_count": len(rejected_indices),
        "warning_count": warning_count,
        "generation_error_count": generation_error_count,
        "generation_overflow_count": generation_overflow_count,
        "timeout_count": timeout_count,
        "quarantined_count": quarantined_count,
        "empty_text_for_speech_count": empty_text_for_speech_count,
        "review_uncertain_count": 0,
        "review_uncertain_items": [],
        "items": items,
        "rejected_indices": rejected_indices,
        **qc_policy,
    }


def _review_reasons_for_qc_item(item: dict) -> list[str]:
    severity = str(item.get("severity") or "").strip().lower()
    reasons = [str(reason) for reason in (item.get("reasons") or []) if reason]
    signal_qc = item.get("signal_qc")
    signal_verdict = ""
    if isinstance(signal_qc, dict):
        signal_verdict = str(signal_qc.get("verdict") or "").strip().lower()

    review_reasons: list[str] = []
    for reason in reasons:
        if reason.startswith("generation_"):
            review_reasons.append(reason)
        elif reason in {"long_sparse_text", "long_low_value_text"}:
            review_reasons.append(reason)
    if not review_reasons:
        if signal_verdict == "reject":
            review_reasons.append("signal_reject")
        elif severity == "reject":
            review_reasons.append("qc_reject")
    return list(dict.fromkeys(review_reasons))


def collect_adaptive_precision_review(
    chunks: list[dict],
    text_results: list[dict],
    qc_report: dict,
) -> tuple[list[dict], dict, list[str]]:
    if not asr_qc_enabled():
        return text_results, qc_report, []

    items = list(qc_report.get("items") or [])
    if not items:
        updated_report = dict(qc_report)
        updated_report["review_uncertain_enabled"] = True
        return text_results, updated_report, []

    items_by_position = {
        int(item.get("position", index)): item
        for index, item in enumerate(items)
        if isinstance(item, dict)
    }
    review_items: list[dict] = []
    log_lines: list[str] = []

    for index, text_result in enumerate(text_results):
        item = items_by_position.get(index)
        if item is None:
            continue
        review_reasons = _review_reasons_for_qc_item(item)
        if not review_reasons:
            continue
        chunk = chunks[index] if index < len(chunks) else {}
        review_item = {
            "position": index,
            "chunk_index": item.get("chunk_index", chunk.get("index", index + 1)),
            "start": item.get("start", chunk.get("start", 0.0)),
            "end": item.get("end", chunk.get("end", 0.0)),
            "reasons": review_reasons,
            "original_text": str(text_result.get("text") or ""),
            "original_raw_text": str(
                text_result.get("raw_text") or text_result.get("text") or ""
            ),
            "text_preview": item.get("text_preview", ""),
            "metrics": item.get("metrics") or {},
        }
        review_items.append(review_item)
        log_lines.append(
            "ASR Adaptive Precision review chunk {chunk_index}: reasons={reasons}, text={text}".format(
                chunk_index=review_item["chunk_index"],
                reasons=",".join(review_reasons),
                text=review_item["text_preview"],
            )
        )

    if not review_items:
        updated_report = dict(qc_report)
        updated_report["review_uncertain_enabled"] = True
        return text_results, updated_report, []

    updated_report = dict(qc_report)
    existing = list(updated_report.get("review_uncertain_items") or [])
    updated_report["review_uncertain_items"] = existing + review_items
    updated_report["review_uncertain_count"] = len(updated_report["review_uncertain_items"])
    updated_report["review_uncertain_enabled"] = True
    updated_report["precision_policy"] = "adaptive"
    return text_results, updated_report, log_lines


def format_qc_log_items(report: dict, limit: int = 8) -> list[str]:
    lines: list[str] = []
    interesting = [
        item
        for item in report.get("items", [])
        if item.get("severity") in {"reject", "warn"}
    ]

    for item in interesting[:limit]:
        metrics = item.get("metrics", {})
        repeat = metrics.get("max_repeat", {})
        reasons = ",".join(item.get("reasons", [])) or "none"
        generation = metrics.get("generation") or {}
        generation_suffix = ""
        if generation.get("error_kind"):
            generation_suffix = f", generation_error={generation.get('error_kind')}"
        lines.append(
            "ASR QC chunk {chunk_index}: severity={severity}, reasons={reasons}, "
            "duration={duration_s}s, chars={compact_chars}, cps={chars_per_sec}, "
            "repeat={unit}x{run}{generation_suffix}, text={text_preview}".format(
                chunk_index=item.get("chunk_index"),
                severity=item.get("severity"),
                reasons=reasons,
                duration_s=metrics.get("duration_s", 0.0),
                compact_chars=metrics.get("compact_chars", 0),
                chars_per_sec=metrics.get("chars_per_sec", 0.0),
                unit=repeat.get("unit", ""),
                run=repeat.get("run", 0),
                generation_suffix=generation_suffix,
                text_preview=item.get("text_preview", ""),
            )
        )

    remaining = len(interesting) - len(lines)
    if remaining > 0:
        lines.append(f"ASR QC: {remaining} additional flagged chunks omitted from log")

    return lines
