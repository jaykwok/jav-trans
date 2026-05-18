import logging
import os
import re
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


def asr_recovery_enabled() -> bool:
    return _env_bool("ASR_RECOVERY_ENABLED", False)


def asr_precision_mode() -> str:
    value = os.getenv("ASR_PRECISION_MODE", "normal").strip().lower()
    if value in {"strict", "precision", "conservative"}:
        return "strict"
    return "normal"


def asr_drop_uncertain_enabled() -> bool:
    return asr_precision_mode() == "strict" or _env_bool(
        "ASR_DROP_UNCERTAIN_ENABLED",
        False,
    )


ASR_QC_ENABLED = asr_qc_enabled()
ASR_RECOVERY_ENABLED = asr_recovery_enabled()
ASR_QC_REPETITION_THRESHOLD = max(
    1,
    int(os.getenv("ASR_QC_REPETITION_THRESHOLD", "10")),
)

_REPEAT_MIN_RUN = max(2, int(os.getenv("ASR_QC_REPEAT_MIN_RUN", "4")))
_REPEAT_MIN_RATIO = float(os.getenv("ASR_QC_REPEAT_MIN_RATIO", "0.40"))
_HALLUCINATION_MIN_CHARS = max(
    1, int(os.getenv("ASR_QC_HALLUCINATION_MIN_CHARS", "50"))
)
_MAX_CHARS_PER_SEC = float(os.getenv("ASR_QC_MAX_CHARS_PER_SEC", "8.0"))
_DENSITY_MIN_CHARS = max(1, int(os.getenv("ASR_QC_DENSITY_MIN_CHARS", "30")))
_LOW_INFO_DURATION_S = float(os.getenv("ASR_QC_LOW_INFO_DURATION", "6.0"))
_LOW_INFO_MAX_CHARS = max(0, int(os.getenv("ASR_QC_LOW_INFO_MAX_CHARS", "5")))
_LOW_INFO_RECOVERY_ENABLED = _env_bool("ASR_QC_LOW_INFO_RECOVERY", False)
_EMPTY_DURATION_S = float(os.getenv("ASR_QC_EMPTY_DURATION", "1.0"))
_LONG_LOW_VALUE_DURATION_S = float(
    os.getenv("ASR_QC_LONG_LOW_VALUE_DURATION", "6.0")
)

_logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def check_logprob_quality(result: dict) -> dict:
    """
    Returns {'verdict': 'ok'/'warn'/'reject', 'reason': str|None, 'metrics': dict}.
    Skips any check where the signal is None.
    Reads thresholds dynamically so tests can override via monkeypatch.setenv.
    """
    strict_drop = asr_drop_uncertain_enabled()
    nospeech_threshold = _env_float("ASR_QC_NOSPEECH_THRESHOLD", 0.6)
    logprob_threshold = _env_float("ASR_QC_LOGPROB_THRESHOLD", -1.0)
    compression_threshold = _env_float("ASR_QC_COMPRESSION_THRESHOLD", 2.4)
    if strict_drop:
        nospeech_threshold = _env_float(
            "ASR_QC_STRICT_NOSPEECH_THRESHOLD",
            0.5,
        )
        logprob_threshold = _env_float(
            "ASR_QC_STRICT_LOGPROB_THRESHOLD",
            -0.7,
        )
        compression_threshold = _env_float(
            "ASR_QC_STRICT_COMPRESSION_THRESHOLD",
            2.0,
        )

    no_speech_prob = _optional_float(result.get("no_speech_prob"))
    avg_logprob = _optional_float(result.get("avg_logprob"))
    compression_ratio = _optional_float(result.get("compression_ratio"))

    reject_reasons: list[str] = []
    warn_reasons: list[str] = []
    if no_speech_prob is not None and no_speech_prob > nospeech_threshold:
        reject_reasons.append("high_no_speech")
    if compression_ratio is not None and compression_ratio > compression_threshold:
        reject_reasons.append("high_compression")
    if avg_logprob is not None and avg_logprob < logprob_threshold:
        if strict_drop:
            reject_reasons.append("low_logprob")
        else:
            warn_reasons.append("low_logprob")

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
            "drop_uncertain_enabled": strict_drop,
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


def _duration_for(chunk: dict, text_result: dict) -> float:
    duration = text_result.get("duration")
    if duration is None:
        duration = chunk.get("duration")
    if duration is None:
        start = float(chunk.get("start", 0.0))
        end = float(chunk.get("end", start))
        duration = max(0.0, end - start)
    return max(0.0, float(duration or 0.0))


def evaluate_asr_chunk_qc(
    chunk: dict,
    text_result: dict,
    *,
    is_low_value_text: Callable[[str], bool] | None = None,
    is_context_leak: Callable[[str], bool] | None = None,
) -> dict:
    raw_text = str(text_result.get("raw_text", "") or "")
    text = str(text_result.get("text", "") or raw_text)
    duration = _duration_for(chunk, text_result)
    compact = _strip_punctuation(text)
    context_compact = _compact_text(text)
    repeat = _find_max_repeat(context_compact)
    chars_per_sec = len(compact) / duration if duration > 0 else 0.0
    low_value = bool(is_low_value_text(text)) if is_low_value_text else False
    context_leak = bool(is_context_leak(text)) if is_context_leak else False
    mojibake = bool(_MOJIBAKE_RE.search(text))

    reasons: list[str] = []
    severity = "ok"
    asr_generation = text_result.get("asr_generation")
    generation_error_kind = ""
    if isinstance(asr_generation, dict):
        generation_error_kind = str(asr_generation.get("error_kind") or "").strip()

    if generation_error_kind:
        reasons.append(f"generation_{generation_error_kind}")
        severity = "recover" if generation_error_kind in {"timeout", "worker_error"} else "warn"

    if not compact and duration >= _EMPTY_DURATION_S:
        reasons.append("empty_text_for_speech_chunk")
        if severity == "ok":
            severity = "warn"

    if mojibake:
        reasons.append("mojibake")
        severity = "recover"

    if context_leak:
        reasons.append("context_leak")
        severity = "recover"

    repeat_run = int(repeat["run"])
    repeat_ratio = float(repeat["ratio"])
    repeat_unit_len = int(repeat["unit_len"])
    repeated_loop = (
        repeat_run >= _REPEAT_MIN_RUN
        and repeat_ratio >= _REPEAT_MIN_RATIO
        and (repeat_unit_len > 1 or repeat_run >= _REPEAT_MIN_RUN * 2)
    )
    if repeated_loop:
        reasons.append("repeat_ngram_loop")
        severity = "recover"

    if len(compact) >= _HALLUCINATION_MIN_CHARS and repeat_ratio >= _REPEAT_MIN_RATIO:
        reasons.append("hallucination_cap_like")
        severity = "recover"

    if len(compact) >= _DENSITY_MIN_CHARS and chars_per_sec > _MAX_CHARS_PER_SEC:
        reasons.append("abnormal_char_density")
        severity = "recover"

    if duration >= _LOW_INFO_DURATION_S and len(compact) <= _LOW_INFO_MAX_CHARS and low_value:
        reasons.append("long_low_information_chunk")
        if _LOW_INFO_RECOVERY_ENABLED:
            severity = "recover"
        elif severity == "ok":
            severity = "warn"
    elif duration >= _LONG_LOW_VALUE_DURATION_S and low_value:
        reasons.append("long_low_value_text")
        if severity == "ok":
            severity = "warn"

    signal_qc = check_logprob_quality(text_result)
    if signal_qc["verdict"] != "ok":
        reasons.append(str(signal_qc.get("reason") or signal_qc["verdict"]))
        if severity == "ok":
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
            "context_leak": context_leak,
            "mojibake": mojibake,
            "max_repeat": repeat,
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
    is_context_leak: Callable[[str], bool] | None = None,
    backend=None,
) -> dict:
    accepts_contexts = bool(getattr(backend, "accepts_contexts", True))
    context_leak_callback = is_context_leak if accepts_contexts else None
    context_leak_check = "on" if accepts_contexts else "skipped(backend_ignores_contexts)"
    qc_policy = {
        "context_leak_check": context_leak_check,
        "repetition_check": "on",
        "repetition_threshold": ASR_QC_REPETITION_THRESHOLD,
        "precision_mode": asr_precision_mode(),
        "drop_uncertain_enabled": asr_drop_uncertain_enabled(),
    }

    if not asr_qc_enabled():
        return {
            "enabled": False,
            "chunk_count": len(chunks),
            "recoverable_count": 0,
            "warning_count": 0,
            "generation_error_count": 0,
            "generation_overflow_count": 0,
            "timeout_count": 0,
            "quarantined_count": 0,
            "empty_text_for_speech_count": 0,
            "dropped_uncertain_count": 0,
            "dropped_uncertain_items": [],
            "items": [],
            "recoverable_indices": [],
            **qc_policy,
        }

    items: list[dict] = []
    recoverable_indices: list[int] = []
    warning_count = 0
    generation_error_count = 0
    generation_overflow_count = 0
    timeout_count = 0
    quarantined_count = 0
    empty_text_for_speech_count = 0

    for index, (chunk, text_result) in enumerate(zip(chunks, text_results)):
        qc = evaluate_asr_chunk_qc(
            chunk,
            text_result,
            is_low_value_text=is_low_value_text,
            is_context_leak=context_leak_callback,
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
        if qc["severity"] == "recover":
            recoverable_indices.append(index)
        elif qc["severity"] == "warn":
            warning_count += 1

    return {
        "enabled": True,
        "chunk_count": len(chunks),
        "recoverable_count": len(recoverable_indices),
        "warning_count": warning_count,
        "generation_error_count": generation_error_count,
        "generation_overflow_count": generation_overflow_count,
        "timeout_count": timeout_count,
        "quarantined_count": quarantined_count,
        "empty_text_for_speech_count": empty_text_for_speech_count,
        "dropped_uncertain_count": 0,
        "dropped_uncertain_items": [],
        "items": items,
        "recoverable_indices": recoverable_indices,
        **qc_policy,
    }


def _drop_reasons_for_qc_item(item: dict) -> list[str]:
    severity = str(item.get("severity") or "").strip().lower()
    reasons = [str(reason) for reason in (item.get("reasons") or []) if reason]
    signal_qc = item.get("signal_qc")
    signal_verdict = ""
    if isinstance(signal_qc, dict):
        signal_verdict = str(signal_qc.get("verdict") or "").strip().lower()

    drop_reasons: list[str] = []
    if signal_verdict == "reject":
        drop_reasons.append("signal_reject")
    if severity == "recover":
        drop_reasons.append("recoverable_qc")
    for reason in reasons:
        if reason.startswith("generation_"):
            drop_reasons.append(reason)
        elif reason in {"long_low_information_chunk", "long_low_value_text"}:
            drop_reasons.append(reason)
    return list(dict.fromkeys(drop_reasons))


def _dropped_text_result(text_result: dict, item: dict, drop_reasons: list[str]) -> dict:
    dropped = dict(text_result)
    original_text = str(text_result.get("text") or "")
    original_raw_text = str(text_result.get("raw_text") or original_text)
    dropped["text"] = ""
    dropped["raw_text"] = ""
    dropped["segments"] = []
    dropped["asr_dropped"] = {
        "policy": "strict_precision",
        "reasons": drop_reasons,
        "original_text": original_text,
        "original_raw_text": original_raw_text,
        "qc": item,
    }
    log = list(text_result.get("log", []))
    log.append(
        "ASR strict precision drop: reasons={reasons}, text={text}".format(
            reasons=",".join(drop_reasons),
            text=_preview(original_text or original_raw_text),
        )
    )
    dropped["log"] = log
    return dropped


def apply_strict_precision_filter(
    chunks: list[dict],
    text_results: list[dict],
    qc_report: dict,
) -> tuple[list[dict], dict, list[str]]:
    if not asr_drop_uncertain_enabled() or not asr_qc_enabled():
        return text_results, qc_report, []

    items = list(qc_report.get("items") or [])
    if not items:
        return text_results, qc_report, []

    items_by_position = {
        int(item.get("position", index)): item
        for index, item in enumerate(items)
        if isinstance(item, dict)
    }
    updated_results = list(text_results)
    dropped_items: list[dict] = []
    log_lines: list[str] = []

    for index, text_result in enumerate(text_results):
        item = items_by_position.get(index)
        if item is None:
            continue
        drop_reasons = _drop_reasons_for_qc_item(item)
        if not drop_reasons:
            continue
        updated_results[index] = _dropped_text_result(text_result, item, drop_reasons)
        chunk = chunks[index] if index < len(chunks) else {}
        dropped_item = {
            "position": index,
            "chunk_index": item.get("chunk_index", chunk.get("index", index + 1)),
            "start": item.get("start", chunk.get("start", 0.0)),
            "end": item.get("end", chunk.get("end", 0.0)),
            "reasons": drop_reasons,
            "original_text": str(text_result.get("text") or ""),
            "original_raw_text": str(
                text_result.get("raw_text") or text_result.get("text") or ""
            ),
            "text_preview": item.get("text_preview", ""),
            "metrics": item.get("metrics") or {},
        }
        dropped_items.append(dropped_item)
        log_lines.append(
            "ASR Strict Precision drop chunk {chunk_index}: reasons={reasons}, text={text}".format(
                chunk_index=dropped_item["chunk_index"],
                reasons=",".join(drop_reasons),
                text=dropped_item["text_preview"],
            )
        )

    if not dropped_items:
        return text_results, qc_report, []

    updated_report = dict(qc_report)
    existing = list(updated_report.get("dropped_uncertain_items") or [])
    updated_report["dropped_uncertain_items"] = existing + dropped_items
    updated_report["dropped_uncertain_count"] = len(updated_report["dropped_uncertain_items"])
    updated_report["drop_uncertain_enabled"] = True
    updated_report["precision_mode"] = asr_precision_mode()
    return updated_results, updated_report, log_lines


def format_qc_log_items(report: dict, limit: int = 8) -> list[str]:
    lines: list[str] = []
    interesting = [
        item
        for item in report.get("items", [])
        if item.get("severity") in {"recover", "warn"}
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
