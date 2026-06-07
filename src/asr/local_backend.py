import concurrent.futures
import gc
import inspect
import multiprocessing as mp
import os
import re
import time
import uuid
import wave
from pathlib import Path
from typing import Callable

from utils.model_paths import resolve_model_spec
from asr.backends.qwen import (
    active_qwen_asr_model_id,
    active_qwen_asr_model_path,
    current_qwen_asr_backend,
    qwen_asr_default_batch_size,
)
from asr.prealign import (
    clean_text_for_aligner,
    normalize_display_text,
    prepare_text_for_alignment,
    strip_alignment_punctuation,
)
from asr.timestamp_fallback import build_word_timestamps_fallback

ASR_MODEL_ID = active_qwen_asr_model_id()
ALIGNER_MODEL_ID = os.getenv("ALIGNER_MODEL_ID", "Qwen/Qwen3-ForcedAligner-0.6B")
ASR_MODEL_PATH = active_qwen_asr_model_path()
ALIGNER_MODEL_PATH = os.getenv("ALIGNER_MODEL_PATH", "").strip()
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "Japanese").strip() or "Japanese"
ASR_CONTEXT = os.getenv("ASR_CONTEXT", "").strip()


def _resolve_asr_batch_size() -> int:
    raw = os.getenv("ASR_BATCH_SIZE", "auto").strip().lower()
    if raw in {"", "auto"}:
        return max(1, qwen_asr_default_batch_size(current_qwen_asr_backend()))
    return max(1, int(raw))


ASR_BATCH_SIZE = _resolve_asr_batch_size()
ASR_MAX_NEW_TOKENS = max(64, int(os.getenv("ASR_MAX_NEW_TOKENS", "128")))
TRANSCRIPTION_MAX_NEW_TOKENS = max(
    32,
    int(os.getenv("TRANSCRIPTION_MAX_NEW_TOKENS", str(ASR_MAX_NEW_TOKENS))),
)
TRANSCRIPTION_TIMEOUT_S = float(os.getenv("TRANSCRIPTION_TIMEOUT_S", "180"))
ASR_DTYPE = os.getenv("ASR_DTYPE", "auto").strip().lower()
ASR_ATTN = os.getenv("ASR_ATTENTION", "auto").strip().lower()
ASR_REPETITION_PENALTY = float(os.getenv("ASR_REPETITION_PENALTY", "1.05"))
ASR_FORCE_LANGUAGE = os.getenv("ASR_FORCE_LANGUAGE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
ALIGNER_BATCH_SIZE = max(
    1,
    int(os.getenv("ALIGNER_BATCH_SIZE", str(ASR_BATCH_SIZE))),
)
ALIGN_LONG_CHUNK_BATCH_SIZE = max(
    1,
    int(os.getenv("ALIGN_LONG_CHUNK_BATCH_SIZE", "1")),
)
ASR_NATIVE_MIN_SPAN_MS = float(os.getenv("ASR_NATIVE_MIN_SPAN_MS", "80"))
ASR_NATIVE_MAX_ZERO_RATIO = float(os.getenv("ASR_NATIVE_MAX_ZERO_RATIO", "0.55"))
ASR_NATIVE_MAX_REPEAT_RATIO = float(os.getenv("ASR_NATIVE_MAX_REPEAT_RATIO", "0.65"))
ASR_NATIVE_MAX_CHARS_PER_ITEM = float(os.getenv("ASR_NATIVE_MAX_CHARS_PER_ITEM", "12.0"))
_ALIGNMENT_MAX_COVERAGE_RATIO = float(
    os.getenv("ALIGNMENT_MAX_COVERAGE_RATIO", "0.05")
)
_ALIGNMENT_MAX_CPS = float(os.getenv("ALIGNMENT_MAX_CPS", "50.0"))
ALIGNMENT_HYBRID_FORCE_MAX_CHUNK_S = float(os.getenv("ALIGNMENT_HYBRID_FORCE_MAX_CHUNK", "24.0"))
ALIGNMENT_HYBRID_FORCE_MIN_TEXT_LEN = max(
    1,
    int(os.getenv("ALIGNMENT_HYBRID_FORCE_MIN_TEXT_LEN", "20")),
)
_ASR_SUBPROCESS_KILL_GRACE_S = float(os.getenv("ASR_SUBPROCESS_KILL_GRACE_S", "5"))
_ASR_SUBPROCESS_READY_TIMEOUT_S = float(
    os.getenv("ASR_SUBPROCESS_READY_TIMEOUT_S", "600")
)


def _resolve_timestamp_mode() -> str:
    raw = os.getenv("ALIGNMENT_TIMESTAMP_MODE", "forced").strip().lower()
    valid_modes = {"forced", "native", "hybrid"}
    if raw not in valid_modes:
        raise ValueError(
            f"Unsupported ALIGNMENT_TIMESTAMP_MODE={raw!r}; "
            f"expected one of {sorted(valid_modes)}"
        )
    return raw


ALIGNMENT_TIMESTAMP_MODE = _resolve_timestamp_mode()


def _get_wav_duration(audio_path: str) -> float:
    with wave.open(audio_path, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
    return frames / rate if rate else 0.0


def _get_wav_duration_or_zero(audio_path: str) -> float:
    try:
        return _get_wav_duration(audio_path)
    except Exception:
        return 0.0


def _notify(on_stage: Callable[[str], None] | None, message: str) -> None:
    if on_stage:
        on_stage(message)


def _payload_has_timeout_log(payload: dict) -> bool:
    log = payload.get("log", [])
    if isinstance(log, str):
        return "TIMEOUT:" in log
    return any("TIMEOUT:" in str(entry) for entry in log)


def _clear_cuda_cache(device: str) -> None:
    if not device.startswith("cuda"):
        return
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _detect_dtype(device: str):
    import torch

    if ASR_DTYPE == "float32":
        return torch.float32
    if ASR_DTYPE == "float16":
        return torch.float16
    if ASR_DTYPE in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _detect_attention(device: str) -> str:
    if ASR_ATTN != "auto":
        return ASR_ATTN
    if not device.startswith("cuda"):
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def _clean_master_text(text: str) -> str:
    return normalize_display_text(text)


def restore_timestamps_to_original(
    original_text: str,
    cleaned_text: str,
    word_dicts: list[dict],
) -> list[dict]:
    try:
        if not original_text or not cleaned_text or not word_dicts:
            return word_dicts

        original = original_text.strip()
        cleaned = cleaned_text.strip()
        if not original or not cleaned:
            return word_dicts

        prealign = prepare_text_for_alignment(original)
        if prealign.align_text == cleaned and prealign.align_to_display_spans:
            restored: list[dict] = []
            cursor = 0
            spans = prealign.align_to_display_spans
            display_text = prealign.display_text
            mapped_items: list[tuple[dict, int, int]] = []
            for word in word_dicts:
                token = str(word.get("word", ""))
                if not token:
                    mapped_items.append((dict(word), 0, 0))
                    continue

                clean_start = cleaned.find(token, cursor)
                if clean_start < 0:
                    clean_start = min(cursor, len(cleaned))
                clean_end = min(len(cleaned), max(clean_start + 1, clean_start + len(token)))
                cursor = clean_end

                covered = [
                    span
                    for span in spans
                    if span.align_start < clean_end and span.align_end > clean_start
                ]
                restored_word = dict(word)
                if covered:
                    display_start = min(span.display_start for span in covered)
                    display_end = max(span.display_end for span in covered)
                    mapped_items.append((restored_word, display_start, display_end))
                else:
                    mapped_items.append((restored_word, 0, 0))
            valid_indices = [i for i, (_word, start, end) in enumerate(mapped_items) if end > start]
            for ordinal, (word, display_start, display_end) in enumerate(mapped_items):
                if display_end <= display_start:
                    word["word"] = str(word.get("word", ""))
                    restored.append(word)
                    continue
                if valid_indices and ordinal == valid_indices[0]:
                    display_start = 0
                next_valid = next((i for i in valid_indices if i > ordinal), None)
                if next_valid is not None:
                    display_end = mapped_items[next_valid][1]
                elif valid_indices and ordinal == valid_indices[-1]:
                    display_end = len(display_text)
                word["word"] = display_text[display_start:display_end] or str(word.get("word", ""))
                restored.append(word)
            return restored

        ratio = len(original) / max(1, len(cleaned))
        restored: list[dict] = []
        cursor = 0
        for word in word_dicts:
            token = str(word.get("word", ""))
            if not token:
                restored.append(dict(word))
                continue

            clean_start = cleaned.find(token, cursor)
            if clean_start < 0:
                clean_start = min(cursor, len(cleaned))
            clean_end = min(len(cleaned), max(clean_start + 1, clean_start + len(token)))
            cursor = clean_end

            original_start = min(len(original), int(clean_start * ratio))
            original_end = min(len(original), max(original_start + 1, int(clean_end * ratio)))
            restored_word = dict(word)
            restored_word["word"] = original[original_start:original_end] or token
            restored.append(restored_word)

        return restored
    except Exception:
        return word_dicts


def _strip_punctuation(text: str) -> str:
    return strip_alignment_punctuation(text)


def _compact_text_len(text: str) -> int:
    return len(_strip_punctuation(text))


def _first_token_id(value) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            token_id = _first_token_id(item)
            if token_id is not None:
                return token_id
    return None


def _iter_generation_configs(model) -> list:
    configs = []

    def add_config(config) -> None:
        if config is not None and not any(config is existing for existing in configs):
            configs.append(config)

    for candidate in (
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "thinker", None),
    ):
        add_config(getattr(candidate, "generation_config", None))
        add_config(getattr(candidate, "config", None))
    return configs


def _normalize_deterministic_generation_config(model) -> None:
    generation_configs = _iter_generation_configs(model)
    fallback_eos_token_id = None
    for generation_config in generation_configs:
        fallback_eos_token_id = _first_token_id(
            getattr(generation_config, "eos_token_id", None)
        )
        if fallback_eos_token_id is not None:
            break

    for generation_config in generation_configs:
        if not bool(getattr(generation_config, "do_sample", False)) and getattr(
            generation_config, "temperature", None
        ) is not None:
            generation_config.temperature = None

        if getattr(generation_config, "pad_token_id", None) is None:
            eos_token_id = _first_token_id(
                getattr(generation_config, "eos_token_id", None)
            ) or fallback_eos_token_id
            if eos_token_id is not None:
                generation_config.pad_token_id = eos_token_id


def _apply_generation_safety(model) -> None:
    _normalize_deterministic_generation_config(model)
    if ASR_REPETITION_PENALTY <= 1.0:
        return
    try:
        model.model.thinker.generation_config.repetition_penalty = ASR_REPETITION_PENALTY
    except Exception:
        pass


def _get_attr(obj, attr: str):
    if hasattr(obj, attr):
        return getattr(obj, attr)
    if isinstance(obj, dict):
        return obj.get(attr)
    return None


def _callable_accepts_kwarg(func, name: str) -> bool:
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _asr_max_new_tokens() -> int:
    try:
        return max(64, int(os.getenv("ASR_MAX_NEW_TOKENS", str(ASR_MAX_NEW_TOKENS))))
    except (TypeError, ValueError):
        return ASR_MAX_NEW_TOKENS


def _transcription_max_new_tokens() -> int:
    try:
        fallback = str(_asr_max_new_tokens())
        return max(32, int(os.getenv("TRANSCRIPTION_MAX_NEW_TOKENS", fallback)))
    except (TypeError, ValueError):
        return TRANSCRIPTION_MAX_NEW_TOKENS


def _transcription_timeout_s() -> float:
    try:
        return float(os.getenv("TRANSCRIPTION_TIMEOUT_S", str(TRANSCRIPTION_TIMEOUT_S)))
    except (TypeError, ValueError):
        return TRANSCRIPTION_TIMEOUT_S


def _asr_language() -> str:
    return os.getenv("ASR_LANGUAGE", ASR_LANGUAGE).strip() or "Japanese"


def _asr_force_language() -> bool:
    return os.getenv("ASR_FORCE_LANGUAGE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _asr_context() -> str:
    return os.getenv("ASR_CONTEXT", ASR_CONTEXT).strip()


def _qwen_generation_metadata(
    *,
    error_kind: str | None = None,
    error_detail: str = "",
    worker_mode: str = "inproc",
) -> dict:
    return {
        "backend": current_qwen_asr_backend(),
        "model_id": active_qwen_asr_model_id(),
        "configured_max_new_tokens": _transcription_max_new_tokens(),
        "model_max_target_positions": None,
        "policy": "qwen_transcribe_limit",
        "worker_mode": worker_mode,
        "error_kind": error_kind,
        "error_detail": error_detail,
    }


def merge_master_with_timestamps(master_text: str, timestamps) -> list[dict]:
    if not master_text or not master_text.strip():
        return []

    if not timestamps:
        return [{"word": master_text.strip(), "start": 0.0, "end": 0.0}]

    result: list[dict] = []
    master_pos = 0

    for ts in timestamps:
        ts_word = _get_attr(ts, "text")
        ts_start = _get_attr(ts, "start_time")
        ts_end = _get_attr(ts, "end_time")

        if not ts_word:
            continue

        word_start = master_text.find(ts_word, master_pos)
        if word_start == -1:
            result.append(
                {
                    "word": str(ts_word),
                    "start": float(ts_start) if ts_start is not None else 0.0,
                    "end": float(ts_end) if ts_end is not None else 0.0,
                }
            )
            continue

        word_end = word_start + len(ts_word)
        if word_start > master_pos:
            gap = master_text[master_pos:word_start]
            if result:
                result[-1]["word"] += gap
            else:
                ts_word = gap + ts_word

        result.append(
            {
                "word": str(ts_word),
                "start": float(ts_start) if ts_start is not None else 0.0,
                "end": float(ts_end) if ts_end is not None else 0.0,
            }
        )
        master_pos = word_end

    if master_pos < len(master_text) and result:
        result[-1]["word"] += master_text[master_pos:]

    return result


def normalize_word_dicts(words: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for word in words:
        token = str(word.get("word", "")).strip()
        if not token:
            continue
        start = float(word.get("start", 0.0))
        end = float(word.get("end", 0.0))
        if end < start:
            end = start
        normalized.append({"start": start, "end": end, "word": token})
    normalized.sort(key=lambda item: (item["start"], item["end"]))
    return normalized


def _log_forced_alignment_stats(
    word_dicts: list[dict],
    *,
    chunk_dur: float = 0.0,
) -> None:
    if not word_dicts:
        return

    durations_ms = [
        max(0.0, float(word["end"]) - float(word["start"])) * 1000.0
        for word in word_dicts
    ]
    word_count = len(word_dicts)
    avg_word_dur = sum(durations_ms) / word_count
    min_dur = min(durations_ms)
    print(
        f"[align] chunk_dur={chunk_dur:.1f}s word_count={word_count} "
        f"avg_word_dur={avg_word_dur:.0f}ms min_dur={min_dur:.0f}ms"
    )


def looks_like_word_timing_failure(
    words: list[dict],
    *,
    min_span_ms: float = ASR_NATIVE_MIN_SPAN_MS,
    max_zero_ratio: float = ASR_NATIVE_MAX_ZERO_RATIO,
    max_repeat_ratio: float = ASR_NATIVE_MAX_REPEAT_RATIO,
    scene_duration_sec: float | None = None,
    max_coverage_ratio: float = _ALIGNMENT_MAX_COVERAGE_RATIO,
    max_cps: float = _ALIGNMENT_MAX_CPS,
) -> bool:
    return bool(
        word_timing_failure_reasons(
            words,
            min_span_ms=min_span_ms,
            max_zero_ratio=max_zero_ratio,
            max_repeat_ratio=max_repeat_ratio,
            scene_duration_sec=scene_duration_sec,
            max_coverage_ratio=max_coverage_ratio,
            max_cps=max_cps,
        )
    )


def word_timing_failure_reasons(
    words: list[dict],
    *,
    min_span_ms: float = ASR_NATIVE_MIN_SPAN_MS,
    max_zero_ratio: float = ASR_NATIVE_MAX_ZERO_RATIO,
    max_repeat_ratio: float = ASR_NATIVE_MAX_REPEAT_RATIO,
    scene_duration_sec: float | None = None,
    max_coverage_ratio: float = _ALIGNMENT_MAX_COVERAGE_RATIO,
    max_cps: float = _ALIGNMENT_MAX_CPS,
) -> list[str]:
    if not words:
        return []
    if len(words) < 2 and scene_duration_sec is None:
        return []

    tiny_span_count = 0
    zero_or_negative_count = 0
    repeated_count = 0
    prev_word = None
    min_start = float("inf")
    max_end = float("-inf")
    char_count = 0

    for word in words:
        start = float(word.get("start", 0.0))
        end = float(word.get("end", 0.0))
        token = _strip_punctuation(word.get("word", ""))
        min_start = min(min_start, start)
        max_end = max(max_end, end)
        char_count += len(token)
        span_ms = max(0.0, end - start) * 1000.0
        if span_ms <= min_span_ms:
            tiny_span_count += 1
        if end <= start:
            zero_or_negative_count += 1
        if token and prev_word and token == prev_word:
            repeated_count += 1
        if token:
            prev_word = token

    total = len(words)
    reasons: list[str] = []
    if tiny_span_count / total >= max_zero_ratio:
        reasons.append("word_timing_tiny_span_heavy")
    if zero_or_negative_count / total >= max_zero_ratio:
        reasons.append("word_timing_zero_heavy")
    if repeated_count / max(1, total - 1) >= max_repeat_ratio:
        reasons.append("word_timing_repeat_heavy")

    if scene_duration_sec is not None and scene_duration_sec > 0:
        word_span = max(0.0, max_end - min_start)
        if word_span > 0:
            if word_span / scene_duration_sec < max_coverage_ratio:
                reasons.append("word_timing_low_coverage")
            if char_count > 0 and char_count / word_span > max_cps:
                reasons.append("word_timing_high_cps")

    return reasons


def _native_timestamp_issue(words: list[dict], text: str) -> str:
    if not words:
        return "无原生时间戳"

    compact_text_len = len(_strip_punctuation(text))
    compact_word_len = sum(len(_strip_punctuation(word["word"])) for word in words)
    avg_chars_per_item = compact_word_len / max(1, len(words))

    if compact_text_len >= 18 and len(words) <= 1:
        return "原生时间戳过粗"
    if compact_text_len >= 24 and avg_chars_per_item > ASR_NATIVE_MAX_CHARS_PER_ITEM:
        return f"原生时间戳颗粒度过粗(avg_chars={avg_chars_per_item:.1f})"
    if looks_like_word_timing_failure(words):
        return "原生时间戳存在密集/零时长异常"
    return ""


def align_text_to_words(
    audio_path: str,
    text: str,
    language: str,
    *,
    aligner_handle,
) -> tuple[list[dict], str]:
    normalized_path = str(Path(audio_path).resolve())
    master_text = _clean_master_text((text or "").strip())
    detected_language = (language or ASR_LANGUAGE or "Japanese").strip()

    if not master_text:
        return [], "empty"

    align_result = None
    try:
        align_results = aligner_handle.align(
            audio=[normalized_path],
            text=[master_text],
            language=[detected_language],
        )
        align_result = next(iter(align_results or []), None)
        if align_result is None:
            return [], "forced_aligner"
        merged_words = merge_master_with_timestamps(
            master_text,
            getattr(align_result, "items", None),
        )
        return normalize_word_dicts(merged_words), "forced_aligner"
    finally:
        if align_result is not None:
            try:
                del align_result
            except Exception:
                pass


def _extract_timestamp_items(asr_result) -> list | None:
    timestamps = _get_attr(asr_result, "time_stamps")
    if timestamps is None:
        return None
    items = _get_attr(timestamps, "items")
    if items is not None:
        return items
    if isinstance(timestamps, list):
        return timestamps
    return None


class WorkerTimeoutError(RuntimeError):
    def __init__(self, detail: str = "ASR worker timed out"):
        super().__init__(detail)
        self.kind = "timeout"
        self.detail = detail


class WorkerError(RuntimeError):
    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail

class LocalAsrBackend:
    is_subprocess = False
    accepts_contexts = True

    def __init__(self, device: str):
        self.device = device if device.startswith("cuda") else "cpu"
        self.dtype = _detect_dtype(self.device)
        self.attention = _detect_attention(self.device)
        self.model = None
        self.forced_aligner = None
        self.timestamp_mode = ALIGNMENT_TIMESTAMP_MODE
        self.request_batch_size = ASR_BATCH_SIZE
        self.align_batch_size = ALIGNER_BATCH_SIZE

    def load(self, on_stage: Callable[[str], None] | None = None) -> None:
        from qwen_asr import Qwen3ASRModel

        if self.model is not None:
            return

        _notify(on_stage, "加载本地 ASR 模型...")
        model_spec = resolve_model_spec(
            active_qwen_asr_model_path() or None,
            active_qwen_asr_model_id(),
            download=True,
        )
        model_kwargs = {
            "dtype": self.dtype,
            "device_map": self.device,
            "max_inference_batch_size": ASR_BATCH_SIZE,
            "max_new_tokens": _asr_max_new_tokens(),
        }

        if self.attention and self.attention != "sdpa":
            model_kwargs["attn_implementation"] = self.attention

        self.model = Qwen3ASRModel.from_pretrained(model_spec, **model_kwargs)
        _apply_generation_safety(self.model)

    def _ensure_forced_aligner(self, on_stage: Callable[[str], None] | None = None):
        from qwen_asr import Qwen3ForcedAligner

        if self.forced_aligner is not None:
            return self.forced_aligner

        _notify(on_stage, "加载 Forced Aligner...")
        aligner_spec = resolve_model_spec(
            ALIGNER_MODEL_PATH or None,
            ALIGNER_MODEL_ID,
            download=True,
        )
        model_kwargs = {
            "dtype": self.dtype,
            "device_map": self.device,
        }
        if self.attention and self.attention != "sdpa":
            model_kwargs["attn_implementation"] = self.attention
        self.forced_aligner = Qwen3ForcedAligner.from_pretrained(aligner_spec, **model_kwargs)
        _normalize_deterministic_generation_config(self.forced_aligner)
        return self.forced_aligner

    def unload_model(self, on_stage: Callable[[str], None] | None = None) -> None:
        if self.model is None:
            return
        _notify(on_stage, "卸载 ASR 文本模型...")
        try:
            del self.model
        except Exception:
            pass
        self.model = None
        _clear_cuda_cache(self.device)

    def unload_forced_aligner(self, on_stage: Callable[[str], None] | None = None) -> None:
        if self.forced_aligner is None:
            return
        _notify(on_stage, "卸载 Forced Aligner...")
        try:
            del self.forced_aligner
        except Exception:
            pass
        self.forced_aligner = None
        _clear_cuda_cache(self.device)

    def close(self) -> None:
        self.unload_model()
        self.unload_forced_aligner()

    def _forced_align_words(
        self,
        normalized_path: str,
        master_text: str,
        detected_language: str,
        on_stage: Callable[[str], None] | None = None,
    ) -> tuple[list[dict], str]:
        forced_aligner = self._ensure_forced_aligner(on_stage=on_stage)
        _notify(on_stage, "Alignment 强制对齐中...")
        try:
            cleaned = clean_text_for_aligner(master_text)
            if not cleaned:
                return [], "empty"
            word_dicts, alignment_mode = align_text_to_words(
                normalized_path,
                cleaned,
                detected_language,
                aligner_handle=forced_aligner,
            )
            if word_dicts and cleaned != master_text:
                word_dicts = restore_timestamps_to_original(
                    master_text,
                    cleaned,
                    word_dicts,
                )
            _log_forced_alignment_stats(
                word_dicts,
                chunk_dur=_get_wav_duration_or_zero(normalized_path),
            )
            return word_dicts, alignment_mode
        finally:
            _clear_cuda_cache(self.device)

    def _forced_align_words_batch(
        self,
        items: list[tuple[str, str, str]],
        on_stage: Callable[[str], None] | None = None,
    ) -> list[list[dict]]:
        if not items:
            return []

        forced_aligner = self._ensure_forced_aligner(on_stage=on_stage)
        results: list[list[dict]] = []
        batch_size = ALIGN_LONG_CHUNK_BATCH_SIZE

        for batch_start in range(0, len(items), batch_size):
            batch_items = items[batch_start : batch_start + batch_size]
            _notify(on_stage, "Alignment 强制对齐中...")
            align_results = None
            try:
                batch_outputs: list[list[dict] | None] = [None] * len(batch_items)
                aligner_jobs: list[tuple[int, str, str, str, str]] = []
                for local_idx, (normalized_path, master_text, language) in enumerate(batch_items):
                    cleaned = clean_text_for_aligner(master_text)
                    if not cleaned:
                        batch_outputs[local_idx] = []
                        continue
                    aligner_jobs.append(
                        (local_idx, normalized_path, master_text, cleaned, language)
                    )

                if aligner_jobs:
                    align_results = forced_aligner.align(
                        audio=[
                            normalized_path
                            for _idx, normalized_path, _master_text, _cleaned, _language in aligner_jobs
                        ],
                        text=[
                            cleaned
                            for _idx, _normalized_path, _master_text, cleaned, _language in aligner_jobs
                        ],
                        language=[
                            language
                            for _idx, _normalized_path, _master_text, _cleaned, language in aligner_jobs
                        ],
                    )
                align_result_items = list(align_results or [])

                for (
                    local_idx,
                    _normalized_path,
                    master_text,
                    cleaned,
                    _language,
                ), align_result in zip(aligner_jobs, align_result_items):
                    merged_words = merge_master_with_timestamps(
                        cleaned,
                        getattr(align_result, "items", None),
                    )
                    restored_words = normalize_word_dicts(merged_words)
                    if restored_words and cleaned != master_text:
                        restored_words = restore_timestamps_to_original(
                            master_text,
                            cleaned,
                            restored_words,
                        )
                    batch_outputs[local_idx] = normalize_word_dicts(restored_words)
                    _log_forced_alignment_stats(
                        batch_outputs[local_idx] or [],
                        chunk_dur=_get_wav_duration_or_zero(_normalized_path),
                    )
                if len(align_result_items) < len(aligner_jobs):
                    for local_idx, *_rest in aligner_jobs[len(align_result_items) :]:
                        batch_outputs[local_idx] = []

                results.extend(output or [] for output in batch_outputs)
            finally:
                if align_results is not None:
                    try:
                        del align_results
                    except Exception:
                        pass
                _clear_cuda_cache(self.device)

        return results

    def align_text_to_words(
        self,
        audio_path: str,
        text: str,
        language: str | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> tuple[dict, list[str]]:
        log: list[str] = []
        normalized_path = str(Path(audio_path).resolve())
        duration = _get_wav_duration(normalized_path)
        raw_master_text = (text or "").strip()
        master_text = _clean_master_text(raw_master_text)
        detected_language = (language or ASR_LANGUAGE or "Japanese").strip()

        log.append(f"Alignment 输入文本长度: {len(raw_master_text)}")
        if master_text != raw_master_text:
            log.append(f"Alignment 清洗后文本长度: {len(master_text)}")

        if not master_text:
            return {
                "words": [],
                "text": "",
                "raw_text": raw_master_text,
                "alignment_mode": "empty",
                "duration": duration,
                "language": detected_language,
            }, log

        alignment_mode = "forced_aligner"
        align_error = ""
        fallback_meta: dict | None = None

        try:
            word_dicts, alignment_mode = self._forced_align_words(
                normalized_path,
                master_text,
                detected_language,
                on_stage=on_stage,
            )
        except Exception as exc:
            align_error = str(exc)
            word_dicts, alignment_mode, fallback_meta = build_word_timestamps_fallback(
                master_text,
                0.0,
                duration,
                audio_path=normalized_path,
            )
            word_dicts = normalize_word_dicts(word_dicts)

        log.append(f"Alignment 词数: {len(word_dicts)}")
        if align_error:
            log.append(f"Alignment 异常: {align_error}")
        if fallback_meta is not None:
            coarse_label = {
                "nonlexical": "Alignment 非词粗时间轴语音区间",
                "align_text_empty": "Alignment align_text 为空粗时间轴语音区间",
            }.get(alignment_mode, "Alignment VAD 回退语音区间")
            error_label = {
                "nonlexical": "Alignment 非词粗时间轴异常",
                "align_text_empty": "Alignment align_text 为空粗时间轴异常",
            }.get(alignment_mode, "Alignment VAD 回退异常")
            if fallback_meta.get("speech_span_count", 0):
                log.append(f"{coarse_label}: {fallback_meta['speech_span_count']}")
            elif fallback_meta.get("vad_error"):
                log.append(f"{error_label}: {fallback_meta['vad_error']}")
        log.append(f"Alignment 模式: {alignment_mode}")
        return {
            "words": word_dicts,
            "text": master_text,
            "raw_text": raw_master_text,
            "alignment_mode": alignment_mode,
            "duration": duration,
            "language": detected_language,
        }, log

    def _build_text_result(
        self,
        normalized_path: str,
        asr_result,
        language_hint: str | None,
    ) -> tuple[dict, list[str]]:
        duration = _get_wav_duration(normalized_path)
        detected_language = (asr_result.language or language_hint or "Japanese").strip()
        raw_master_text = (asr_result.text or "").strip()
        master_text = _clean_master_text(raw_master_text)

        log = [
            f"ASR 语言: {detected_language}",
            f"ASR 原始文本长度: {len(raw_master_text)}",
        ]
        if master_text != raw_master_text:
            log.append(f"ASR 清洗后文本长度: {len(master_text)}")
        log.append("ASR 输出模式: text_only")

        return {
            "text": master_text,
            "raw_text": raw_master_text,
            "duration": duration,
            "language": detected_language,
            "normalized_path": normalized_path,
            "asr_generation": _qwen_generation_metadata(),
        }, log

    def transcribe_texts(
        self,
        audio_paths: list[str],
        contexts: list[str] | None = None,
        initial_prompts: list[str | None] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> list[dict]:
        if self.model is None:
            self.load(on_stage=on_stage)
        if not audio_paths:
            return []
        if initial_prompts is not None and len(initial_prompts) != len(audio_paths):
            raise ValueError(
                f"initial_prompt count mismatch: audio_paths={len(audio_paths)}, initial_prompts={len(initial_prompts)}"
            )

        normalized_paths = [str(Path(audio_path).resolve()) for audio_path in audio_paths]
        language_hint = _asr_language() if _asr_force_language() else None
        request_contexts = contexts if contexts is not None else [_asr_context()] * len(normalized_paths)
        if len(request_contexts) != len(normalized_paths):
            raise ValueError(
                f"context count mismatch: audio_paths={len(normalized_paths)}, contexts={len(request_contexts)}"
            )

        _notify(on_stage, "ASR 文本转录中...")
        transcribe_kwargs = {
            "context": request_contexts,
            "language": language_hint,
            "return_time_stamps": False,
        }
        if _callable_accepts_kwarg(self.model.transcribe, "max_new_tokens"):
            transcribe_kwargs["max_new_tokens"] = _transcription_max_new_tokens()

        asr_results = None
        executor = None
        timed_out = False
        try:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                self.model.transcribe,
                normalized_paths,
                **transcribe_kwargs,
            )
            try:
                timeout_s = _transcription_timeout_s()
                asr_results = future.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                timed_out = True
                future.cancel()
                _notify(
                    on_stage,
                    f"[WARN] ASR 超时 ({_transcription_timeout_s()}s)，跳过当前批次",
                )
                return [
                    {
                        "text": "",
                        "raw_text": "",
                        "duration": _get_wav_duration(path),
                        "language": language_hint or "Japanese",
                        "normalized_path": path,
                        "asr_generation": _qwen_generation_metadata(
                            error_kind="timeout",
                            error_detail=f"skipped after {_transcription_timeout_s()}s",
                        ),
                        "log": [
                            f"TIMEOUT: skipped after {_transcription_timeout_s()}s"
                        ],
                    }
                    for path in normalized_paths
                ]

            payloads: list[dict] = []
            for normalized_path, asr_result in zip(normalized_paths, asr_results):
                payload, payload_log = self._build_text_result(
                    normalized_path,
                    asr_result,
                    language_hint,
                )
                payload_log.append(f"ASR 文本生成上限: {_transcription_max_new_tokens()}")
                payload["log"] = payload_log
                payloads.append(payload)
        finally:
            if executor is not None:
                executor.shutdown(wait=not timed_out, cancel_futures=True)
            if asr_results is not None:
                try:
                    del asr_results
                except Exception:
                    pass
            _clear_cuda_cache(self.device)
        return payloads

    def _should_force_align_text(self, master_text: str, raw_master_text: str, log: list[str]) -> bool:
        prealign = prepare_text_for_alignment(raw_master_text or master_text)
        if prealign.empty_after_cleaning:
            compact_display_len = _compact_text_len(prealign.display_text)
            if compact_display_len <= 0:
                log.append("Alignment 策略: nonlexical_fallback")
                log.append("Alignment 非词文本: 跳过 forced aligner，保留显示文本并使用粗时间轴")
            else:
                log.append("Alignment 策略: align_text_empty_fallback")
                log.append("Alignment align_text 为空: 跳过 forced aligner，保留显示文本并使用粗时间轴")
            return False
        log.append("Alignment 策略: forced_aligner")
        return True

    def _fallback_alignment_mode_for_text(self, master_text: str, raw_master_text: str) -> str:
        prealign = prepare_text_for_alignment(raw_master_text or master_text)
        if prealign.empty_after_cleaning and _compact_text_len(prealign.display_text) <= 0:
            return "nonlexical"
        if prealign.empty_after_cleaning:
            return "align_text_empty"
        return ""

    def _alignment_fallback_window_for_text_result(
        self,
        text_result: dict,
        duration: float,
    ) -> tuple[float, float, str]:
        full_start = 0.0
        full_end = max(0.0, float(duration))
        try:
            start = float(text_result.get("alignment_fallback_start_s"))
            end = float(text_result.get("alignment_fallback_end_s"))
        except (TypeError, ValueError):
            return full_start, full_end, "chunk"

        start = max(full_start, min(full_end, start))
        end = max(start, min(full_end, end))
        if end - start < 0.05:
            return full_start, full_end, "chunk"
        source = str(text_result.get("alignment_fallback_source") or "chunk").strip()
        return start, end, source or "chunk"

    def _build_finalize_output(
        self,
        *,
        word_dicts: list[dict],
        master_text: str,
        raw_master_text: str,
        alignment_mode: str,
        duration: float,
        detected_language: str,
        log: list[str],
        align_error: str = "",
        fallback_meta: dict | None = None,
        fallback_window_source: str = "",
    ) -> tuple[dict, list[str]]:
        log.append(f"Alignment 词数: {len(word_dicts)}")
        if align_error:
            log.append(f"Alignment 异常: {align_error}")
        if fallback_window_source == "speech_core":
            log.append("Alignment 回退窗口: speech_core")
        if fallback_meta is not None:
            if fallback_meta.get("speech_span_count", 0):
                log.append(f"Alignment VAD 回退语音区间: {fallback_meta['speech_span_count']}")
            elif fallback_meta.get("vad_error"):
                log.append(f"Alignment VAD 回退异常: {fallback_meta['vad_error']}")
        log.append(f"Alignment 模式: {alignment_mode}")
        return {
            "words": word_dicts,
            "text": master_text,
            "raw_text": raw_master_text,
            "alignment_mode": alignment_mode,
            "duration": duration,
            "language": detected_language,
        }, log

    def finalize_text_results(
        self,
        text_results: list[dict],
        on_stage: Callable[[str], None] | None = None,
    ) -> list[tuple[dict, list[str]]]:
        if not text_results:
            return []

        finalized: list[tuple[dict, list[str]] | None] = [None] * len(text_results)
        forced_jobs: list[dict] = []

        for idx, text_result in enumerate(text_results):
            log: list[str] = list(text_result.get("log", []))
            normalized_path = str(text_result["normalized_path"])
            duration = float(text_result["duration"])
            detected_language = str(text_result["language"]).strip() or "Japanese"
            raw_master_text = str(text_result.get("raw_text", "")).strip()
            master_text = str(text_result.get("text", "")).strip()
            fallback_start, fallback_end, fallback_window_source = (
                self._alignment_fallback_window_for_text_result(text_result, duration)
            )

            if not master_text:
                finalized[idx] = (
                    {
                        "words": [],
                        "text": "",
                        "raw_text": raw_master_text,
                        "alignment_mode": "empty",
                        "duration": duration,
                        "language": detected_language,
                    },
                    log,
                )
                continue

            if self._should_force_align_text(master_text, raw_master_text, log):
                forced_jobs.append(
                    {
                        "idx": idx,
                        "normalized_path": normalized_path,
                        "master_text": master_text,
                        "detected_language": detected_language,
                        "duration": duration,
                        "raw_master_text": raw_master_text,
                        "log": log,
                        "fallback_start": fallback_start,
                        "fallback_end": fallback_end,
                        "fallback_window_source": fallback_window_source,
                    }
                )
                continue

            word_dicts, alignment_mode, fallback_meta = build_word_timestamps_fallback(
                master_text or raw_master_text,
                fallback_start,
                fallback_end,
                audio_path=normalized_path,
            )
            explicit_mode = self._fallback_alignment_mode_for_text(master_text, raw_master_text)
            if explicit_mode:
                alignment_mode = explicit_mode
            word_dicts = normalize_word_dicts(word_dicts)
            finalized[idx] = self._build_finalize_output(
                word_dicts=word_dicts,
                master_text=master_text,
                raw_master_text=raw_master_text,
                alignment_mode=alignment_mode,
                duration=duration,
                detected_language=detected_language,
                log=log,
                fallback_meta=fallback_meta,
                fallback_window_source=fallback_window_source,
            )

        for batch_start in range(0, len(forced_jobs), self.align_batch_size):
            batch_jobs = forced_jobs[batch_start : batch_start + self.align_batch_size]
            batch_inputs = []
            for job in batch_jobs:
                batch_inputs.append(
                    (
                        job["normalized_path"],
                        job["raw_master_text"] or job["master_text"],
                        job["detected_language"],
                    )
                )
            try:
                batch_word_dicts = self._forced_align_words_batch(batch_inputs, on_stage=on_stage)
                for job, word_dicts in zip(batch_jobs, batch_word_dicts):
                    if not word_dicts:
                        fallback_words, fallback_mode, fallback_meta = build_word_timestamps_fallback(
                            job["master_text"],
                            job["fallback_start"],
                            job["fallback_end"],
                            audio_path=job["normalized_path"],
                        )
                        finalized[job["idx"]] = self._build_finalize_output(
                            word_dicts=normalize_word_dicts(fallback_words),
                            master_text=job["master_text"],
                            raw_master_text=job["raw_master_text"],
                            alignment_mode=fallback_mode,
                            duration=job["duration"],
                            detected_language=job["detected_language"],
                            log=job["log"],
                            align_error="forced aligner returned empty words",
                            fallback_meta=fallback_meta,
                            fallback_window_source=job["fallback_window_source"],
                        )
                        continue

                    finalized[job["idx"]] = self._build_finalize_output(
                        word_dicts=word_dicts,
                        master_text=job["master_text"],
                        raw_master_text=job["raw_master_text"],
                        alignment_mode="forced_aligner",
                        duration=job["duration"],
                        detected_language=job["detected_language"],
                        log=job["log"],
                    )
            except Exception:
                for job in batch_jobs:
                    align_error = ""
                    fallback_meta: dict | None = None
                    try:
                        word_dicts, alignment_mode = self._forced_align_words(
                            job["normalized_path"],
                            job["raw_master_text"] or job["master_text"],
                            job["detected_language"],
                            on_stage=on_stage,
                        )
                        if not word_dicts:
                            align_error = "forced aligner returned empty words"
                            word_dicts, alignment_mode, fallback_meta = build_word_timestamps_fallback(
                                job["master_text"],
                                job["fallback_start"],
                                job["fallback_end"],
                                audio_path=job["normalized_path"],
                            )
                            word_dicts = normalize_word_dicts(word_dicts)
                    except Exception as exc:
                        align_error = str(exc)
                        word_dicts, alignment_mode, fallback_meta = build_word_timestamps_fallback(
                            job["master_text"],
                            job["fallback_start"],
                            job["fallback_end"],
                            audio_path=job["normalized_path"],
                        )
                        word_dicts = normalize_word_dicts(word_dicts)

                    finalized[job["idx"]] = self._build_finalize_output(
                        word_dicts=word_dicts,
                        master_text=job["master_text"],
                        raw_master_text=job["raw_master_text"],
                        alignment_mode=alignment_mode,
                        duration=job["duration"],
                        detected_language=job["detected_language"],
                        log=job["log"],
                        align_error=align_error,
                        fallback_meta=fallback_meta,
                        fallback_window_source=job["fallback_window_source"],
                    )

        return [item for item in finalized if item is not None]

    def finalize_text_result(
        self,
        text_result: dict,
        on_stage: Callable[[str], None] | None = None,
    ) -> tuple[dict, list[str]]:
        return self.finalize_text_results([text_result], on_stage=on_stage)[0]

    def transcribe_to_words(
        self,
        audio_path: str,
        on_stage: Callable[[str], None] | None = None,
    ) -> tuple[dict, list[str]]:
        text_result = self.transcribe_texts([audio_path], on_stage=on_stage)[0]
        self.unload_model(on_stage=on_stage)
        return self.finalize_text_result(text_result, on_stage=on_stage)


class SubprocessAsrBackend:
    """Run ASR text inference in a killable child process; keep alignment local.

    The public API mirrors LocalAsrBackend. finalize_text_results runs in the
    parent process by lazily creating an in-process LocalAsrBackend for the
    forced aligner only.
    """

    is_subprocess = True
    accepts_contexts = True

    def __init__(self, device: str):
        self.device = device if device.startswith("cuda") else "cpu"
        self.request_batch_size = ASR_BATCH_SIZE
        self.align_batch_size = ALIGNER_BATCH_SIZE
        self.timestamp_mode = ALIGNMENT_TIMESTAMP_MODE
        self.kill_grace_s = _ASR_SUBPROCESS_KILL_GRACE_S
        self.ready_timeout_s = _ASR_SUBPROCESS_READY_TIMEOUT_S
        self.model = None
        self.forced_aligner = None
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._conn = None
        self._align_backend: LocalAsrBackend | None = None

    def load(self, on_stage: Callable[[str], None] | None = None) -> None:
        self._ensure_worker(on_stage=on_stage)

    def _ensure_worker(self, on_stage: Callable[[str], None] | None = None) -> None:
        if self._process is not None and self._process.is_alive() and self._conn is not None:
            return
        self._start_worker(on_stage=on_stage)

    def _start_worker(self, on_stage: Callable[[str], None] | None = None) -> None:
        from asr.worker import main as worker_main

        self._close_conn()
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=worker_main,
            args=(child_conn, {"device": self.device}),
            daemon=False,
        )

        _notify(on_stage, "启动 ASR 子进程...")
        process.start()
        child_conn.close()
        self._process = process
        self._conn = parent_conn

        deadline = time.monotonic() + self.ready_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill_child()
                raise WorkerError(
                    "crash",
                    f"worker ready timeout after {self.ready_timeout_s}s",
                )
            if parent_conn.poll(min(0.5, remaining)):
                break
            if process.exitcode is not None:
                exitcode = process.exitcode
                self._kill_child()
                raise WorkerError("crash", f"worker exited before ready: {exitcode}")

        try:
            message = parent_conn.recv()
        except EOFError as exc:
            exitcode = process.exitcode
            self._kill_child()
            raise WorkerError("crash", f"worker exited before ready: {exitcode}") from exc

        if not isinstance(message, dict):
            self._kill_child()
            raise WorkerError("protocol_error", "ready message is not a dict")

        if message.get("op") == "error":
            kind = str(message.get("kind") or "crash")
            detail = str(message.get("detail") or "worker failed during startup")
            self._kill_child()
            raise WorkerError(kind, detail)

        if message.get("op") != "ready":
            self._kill_child()
            raise WorkerError("protocol_error", f"unexpected ready message: {message!r}")

        pid = message.get("pid")
        if not isinstance(pid, int):
            self._kill_child()
            raise WorkerError("protocol_error", f"invalid ready pid: {pid!r}")

        _notify(on_stage, f"ASR 子进程就绪 pid={pid}")

    def _close_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None

    def _kill_child(self) -> None:
        process = self._process
        self._process = None

        if process is not None:
            try:
                if process.is_alive():
                    process.terminate()
                    process.join(self.kill_grace_s)
                if process.is_alive():
                    process.kill()
                    process.join(5)
                if process.is_alive():
                    process.join(1)
            except Exception:
                pass

        self._close_conn()
        _clear_cuda_cache(self.device)

    def _restart_worker(self, on_stage: Callable[[str], None] | None = None) -> None:
        self._kill_child()
        self._start_worker(on_stage=on_stage)

    def _raise_after_worker_restart(
        self,
        exc: WorkerTimeoutError | WorkerError,
        cause: BaseException | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> None:
        try:
            if cause is not None:
                raise exc from cause
            raise exc
        finally:
            try:
                self._restart_worker(on_stage=on_stage)
            except Exception as restart_exc:
                raise WorkerError(
                    "crash",
                    f"worker respawn failed: {restart_exc!r}",
                ) from exc

    def unload_model(self, on_stage: Callable[[str], None] | None = None) -> None:
        if self._process is None:
            return

        _notify(on_stage, "关闭 ASR 子进程...")
        conn = self._conn
        process = self._process
        try:
            if conn is not None and process is not None and process.is_alive():
                conn.send({"op": "shutdown"})
                process.join(5)
        except Exception:
            pass
        finally:
            if process is not None and process.is_alive():
                self._kill_child()
            else:
                self._process = None
                self._close_conn()
                _clear_cuda_cache(self.device)

    def unload_forced_aligner(self, on_stage: Callable[[str], None] | None = None) -> None:
        if self._align_backend is not None:
            self._align_backend.unload_forced_aligner(on_stage=on_stage)
        self.forced_aligner = None

    def close(self) -> None:
        self.unload_model()
        if self._align_backend is not None:
            self._align_backend.close()
            self._align_backend = None

    def _ensure_align_backend(self) -> LocalAsrBackend:
        if self._align_backend is None:
            self._align_backend = LocalAsrBackend(self.device)
        return self._align_backend

    def transcribe_texts(
        self,
        audio_paths: list[str],
        contexts: list[str] | None = None,
        initial_prompts: list[str | None] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> list[dict]:
        if not audio_paths:
            return []
        if initial_prompts is not None and len(initial_prompts) != len(audio_paths):
            raise ValueError(
                f"initial_prompt count mismatch: audio_paths={len(audio_paths)}, initial_prompts={len(initial_prompts)}"
            )

        self._ensure_worker(on_stage=on_stage)
        assert self._conn is not None

        if contexts is None:
            request_contexts = [_asr_context()] * len(audio_paths)
        elif len(contexts) != len(audio_paths):
            raise ValueError(
                f"context count mismatch: audio_paths={len(audio_paths)}, contexts={len(contexts)}"
            )
        else:
            request_contexts = contexts

        job_id = uuid.uuid4().hex[:8]
        chunks = [
            {
                "path": str(Path(audio_path).resolve()),
                "context": context,
                "index": idx,
            }
            for idx, (audio_path, context) in enumerate(
                zip(audio_paths, request_contexts)
            )
        ]

        try:
            self._conn.send({"op": "transcribe", "job_id": job_id, "chunks": chunks})
        except (BrokenPipeError, EOFError, OSError) as exc:
            exitcode = self._process.exitcode if self._process is not None else None
            failure = WorkerError(
                "crash",
                f"worker send failed exitcode={exitcode}: {exc!r}",
            )
            self._raise_after_worker_restart(failure, cause=exc, on_stage=on_stage)
        timeout_s = _transcription_timeout_s()
        deadline = time.monotonic() + timeout_s

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = f"worker timeout after {timeout_s}s"
                self._raise_after_worker_restart(
                    WorkerTimeoutError(detail),
                    on_stage=on_stage,
                )
            if not self._conn.poll(min(1.0, remaining)):
                exitcode = self._process.exitcode if self._process is not None else None
                if exitcode is not None:
                    self._raise_after_worker_restart(
                        WorkerError(
                            "crash",
                            f"worker exited before result exitcode={exitcode}",
                        ),
                        on_stage=on_stage,
                    )
                continue

            try:
                message = self._conn.recv()
            except EOFError as exc:
                exitcode = self._process.exitcode if self._process is not None else None
                failure = WorkerError(
                    "crash",
                    f"worker pipe closed exitcode={exitcode}",
                )
                self._raise_after_worker_restart(failure, cause=exc, on_stage=on_stage)

            if not isinstance(message, dict):
                self._raise_after_worker_restart(
                    WorkerError("protocol_error", "worker message is not a dict"),
                    on_stage=on_stage,
                )

            if str(message.get("job_id") or "") != job_id:
                continue

            op = message.get("op")
            if op == "result":
                results = message.get("results")
                if not isinstance(results, list) or len(results) != len(audio_paths):
                    self._raise_after_worker_restart(
                        WorkerError(
                            "protocol_error",
                            "worker result count does not match request",
                        ),
                        on_stage=on_stage,
                    )
                if any(
                    isinstance(result, dict) and _payload_has_timeout_log(result)
                    for result in results
                ):
                    self._raise_after_worker_restart(
                        WorkerTimeoutError("worker returned TIMEOUT payload"),
                        on_stage=on_stage,
                    )
                for result in results:
                    if isinstance(result, dict) and isinstance(
                        result.get("asr_generation"),
                        dict,
                    ):
                        result["asr_generation"]["worker_mode"] = "subprocess"
                return results

            if op == "error":
                kind = str(message.get("kind") or "crash")
                detail = str(message.get("detail") or "worker error")
                self._raise_after_worker_restart(
                    WorkerError(kind, detail),
                    on_stage=on_stage,
                )

            self._raise_after_worker_restart(
                WorkerError("protocol_error", f"unexpected worker op: {op}"),
                on_stage=on_stage,
            )

    def finalize_text_results(
        self,
        text_results: list[dict],
        on_stage: Callable[[str], None] | None = None,
    ) -> list[tuple[dict, list[str]]]:
        return self._ensure_align_backend().finalize_text_results(
            text_results,
            on_stage=on_stage,
        )

    def finalize_text_result(
        self,
        text_result: dict,
        on_stage: Callable[[str], None] | None = None,
    ) -> tuple[dict, list[str]]:
        return self.finalize_text_results([text_result], on_stage=on_stage)[0]

    def transcribe_to_words(
        self,
        audio_path: str,
        on_stage: Callable[[str], None] | None = None,
    ) -> tuple[dict, list[str]]:
        text_result = self.transcribe_texts([audio_path], on_stage=on_stage)[0]
        self.unload_model(on_stage=on_stage)
        return self.finalize_text_result(text_result, on_stage=on_stage)


def transcribe_to_words(
    audio_path: str,
    device: str,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[dict, list[str]]:
    backend = LocalAsrBackend(device)
    try:
        log = [f"ASR backend: {current_qwen_asr_backend()}"]
        result, extra_log = backend.transcribe_to_words(audio_path, on_stage=on_stage)
        return result, log + extra_log
    finally:
        backend.close()



