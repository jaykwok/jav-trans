import os
import re
from typing import Callable


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


ASR_QC_ENABLED = _env_bool("ASR_QC_ENABLED", True)
ASR_RECOVERY_ENABLED = _env_bool("ASR_RECOVERY_ENABLED", False)
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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


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

    if not compact and duration >= _EMPTY_DURATION_S:
        reasons.append("empty_text_for_speech_chunk")
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
        },
        "text_preview": _preview(text),
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
    }

    if not ASR_QC_ENABLED:
        return {
            "enabled": False,
            "chunk_count": len(chunks),
            "recoverable_count": 0,
            "warning_count": 0,
            "items": [],
            "recoverable_indices": [],
            **qc_policy,
        }

    items: list[dict] = []
    recoverable_indices: list[int] = []
    warning_count = 0

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

        if qc["severity"] == "recover":
            recoverable_indices.append(index)
        elif qc["severity"] == "warn":
            warning_count += 1

    return {
        "enabled": True,
        "chunk_count": len(chunks),
        "recoverable_count": len(recoverable_indices),
        "warning_count": warning_count,
        "items": items,
        "recoverable_indices": recoverable_indices,
        **qc_policy,
    }


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
        lines.append(
            "ASR QC chunk {chunk_index}: severity={severity}, reasons={reasons}, "
            "duration={duration_s}s, chars={compact_chars}, cps={chars_per_sec}, "
            "repeat={unit}x{run}, text={text_preview}".format(
                chunk_index=item.get("chunk_index"),
                severity=item.get("severity"),
                reasons=reasons,
                duration_s=metrics.get("duration_s", 0.0),
                compact_chars=metrics.get("compact_chars", 0),
                chars_per_sec=metrics.get("chars_per_sec", 0.0),
                unit=repeat.get("unit", ""),
                run=repeat.get("run", 0),
                text_preview=item.get("text_preview", ""),
            )
        )

    remaining = len(interesting) - len(lines)
    if remaining > 0:
        lines.append(f"ASR QC: {remaining} additional flagged chunks omitted from log")

    return lines


