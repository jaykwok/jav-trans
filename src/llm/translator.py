import json
import contextlib
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from openai import OpenAI

from core.config import load_config
from llm import cache as translation_cache
from llm import patch as llm_patch
from llm import prompt as prompt_module


load_config()

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_CLIENT: OpenAI | None = None
_CLIENT_KEY: tuple[str, str] = ("", "")
_CLIENT_LOCK = threading.Lock()
_RETRY_CONTEXT = threading.local()
_cache_lock = threading.Lock()
PROMPT_VERSION = prompt_module.PROMPT_VERSION
_LEADING_SPEAKER_RE = prompt_module._LEADING_SPEAKER_RE
_SYSTEM_PROMPT_FULL = prompt_module._SYSTEM_PROMPT_FULL
_SYSTEM_PROMPT_COMPACT = prompt_module._SYSTEM_PROMPT_COMPACT
_JSON_OUTPUT_LABEL = prompt_module._JSON_OUTPUT_LABEL
_normalize_source_text = prompt_module._normalize_source_text
_warn_translation_cache = translation_cache._warn_translation_cache


class RetryableTranslationFormatError(RuntimeError):
    pass


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set in config.py or .env")
    return value


OPENAI_COMPATIBILITY_BASE_URL = (
    os.getenv("OPENAI_COMPATIBILITY_BASE_URL", "").strip() or None
)
API_KEY = os.getenv("API_KEY", "").strip() or None
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "").strip()
LLM_API_FORMAT = os.getenv("LLM_API_FORMAT", "chat").strip().lower() or "chat"
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "xhigh").strip() or "xhigh"
CHARACTER_FULL_NAME_REFERENCE = (
    os.getenv("ASR_CONTEXT", "").strip()
)

TRANSLATION_MAX_TOKENS = max(0, int(os.getenv("TRANSLATION_MAX_TOKENS", "384000")))
COMPACT_SYSTEM_PROMPT = os.getenv("COMPACT_SYSTEM_PROMPT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TRANSLATION_API_RETRIES = max(1, int(os.getenv("TRANSLATION_API_RETRIES", "4")))
TRANSLATION_BATCH_REPAIR_RETRIES = max(
    1,
    int(os.getenv("TRANSLATION_BATCH_REPAIR_RETRIES", "2")),
)
TRANSLATION_API_BACKOFF_BASE_S = max(
    0.1,
    float(os.getenv("TRANSLATION_API_BACKOFF_BASE_S", "1.5")),
)
TRANSLATION_API_BACKOFF_MAX_S = max(
    TRANSLATION_API_BACKOFF_BASE_S,
    float(os.getenv("TRANSLATION_API_BACKOFF_MAX_S", "20")),
)
TRANSLATION_PREFIX_WARMUP = os.getenv(
    "TRANSLATION_PREFIX_WARMUP", "1"
).strip().lower() in {"1", "true", "yes", "on"}
TRANSLATION_FULL_JSON_PREFIX_MAX_CHARS = max(
    0,
    int(os.getenv("TRANSLATION_FULL_JSON_PREFIX_MAX_CHARS", "180000")),
)
TRANSLATION_REPAIR_ENABLED = os.getenv(
    "TRANSLATION_REPAIR_ENABLED", "1"
).strip().lower() in {"1", "true", "yes", "on"}
TRANSLATION_REPAIR_MAX_IDS = max(
    0,
    int(os.getenv("TRANSLATION_REPAIR_MAX_IDS", "12")),
)
TRANSLATION_REPAIR_CONTEXT_RADIUS = max(
    1,
    int(os.getenv("TRANSLATION_REPAIR_CONTEXT_RADIUS", "1")),
)
TRANSLATION_REPAIR_LENGTH_RATIO_MIN = max(
    0.0,
    float(os.getenv("TRANSLATION_REPAIR_LENGTH_RATIO_MIN", "0.25")),
)
TRANSLATION_REPAIR_LENGTH_RATIO_MAX = max(
    TRANSLATION_REPAIR_LENGTH_RATIO_MIN,
    float(os.getenv("TRANSLATION_REPAIR_LENGTH_RATIO_MAX", "4.0")),
)

def _normalize_llm_api_format(value: str | None, fallback: str = "chat") -> str:
    normalized = (value or fallback or "chat").strip().lower()
    return normalized if normalized in {"chat", "responses"} else "chat"


def _llm_api_format(api_format: str | None = None) -> str:
    if api_format is not None:
        return _normalize_llm_api_format(api_format, LLM_API_FORMAT)
    value = os.getenv("LLM_API_FORMAT", LLM_API_FORMAT)
    return _normalize_llm_api_format(value)


def _normalize_reasoning_effort(value: str | None, fallback: str = "xhigh") -> str:
    normalized = (value or fallback or "xhigh").strip().lower()
    if normalized in {"medium", "xhigh"}:
        return normalized
    return fallback if fallback in {"medium", "xhigh"} else "xhigh"


def _get_client() -> OpenAI:
    global _CLIENT, _CLIENT_KEY
    current_key = os.getenv("API_KEY", "").strip() or None
    current_url = os.getenv("OPENAI_COMPATIBILITY_BASE_URL", "").strip() or None
    key_tuple = (current_key or "", current_url or "")
    with _CLIENT_LOCK:
        if _CLIENT is None or key_tuple != _CLIENT_KEY:
            _CLIENT = OpenAI(api_key=current_key, base_url=current_url)
            _CLIENT_KEY = key_tuple
    return _CLIENT


def _current_retry_events() -> list[dict] | None:
    events = getattr(_RETRY_CONTEXT, "events", None)
    return events if isinstance(events, list) else None


_load_translation_cache = translation_cache._load_translation_cache
_translation_cache_jsonl_path = translation_cache._translation_cache_jsonl_path
_translation_cache_legacy_json_path = translation_cache._translation_cache_legacy_json_path
_read_translation_cache_json = translation_cache._read_translation_cache_json
_read_translation_cache_jsonl = translation_cache._read_translation_cache_jsonl
_rewrite_translation_cache_jsonl = translation_cache._rewrite_translation_cache_jsonl
_save_cache_entry = translation_cache._save_cache_entry


def _compute_prompt_signature(
    extra_glossary: str = "",
    *,
    glossary: str = "",
    target_lang: str = "简体中文",
    character_reference: str = "",
) -> str:
    return translation_cache._compute_prompt_signature(
        extra_glossary,
        glossary=glossary,
        target_lang=target_lang,
        character_reference=character_reference,
        prompt_version=PROMPT_VERSION,
        model_name=os.getenv("LLM_MODEL_NAME", LLM_MODEL_NAME).strip(),
        compact_system_prompt=COMPACT_SYSTEM_PROMPT,
    )


def _translation_cache_key(
    batch_index: int,
    batch_segments: list[dict],
    *,
    extra_glossary: str = "",
    glossary: str = "",
    target_lang: str = "简体中文",
    character_reference: str = "",
) -> str:
    return translation_cache._translation_cache_key(
        batch_index,
        batch_segments,
        extra_glossary=extra_glossary,
        glossary=glossary,
        target_lang=target_lang,
        character_reference=character_reference,
        prompt_version=PROMPT_VERSION,
        model_name=os.getenv("LLM_MODEL_NAME", LLM_MODEL_NAME).strip(),
        compact_system_prompt=COMPACT_SYSTEM_PROMPT,
    )


def _get_nested_value(value, *path: str):
    current = value
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _coerce_optional_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _extract_usage_metrics(usage) -> dict:
    cached_tokens = _coerce_optional_int(
        _get_nested_value(usage, "prompt_tokens_details", "cached_tokens")
    )
    cache_hit_tokens = _coerce_optional_int(
        _get_nested_value(usage, "prompt_cache_hit_tokens")
    )
    cache_miss_tokens = _coerce_optional_int(
        _get_nested_value(usage, "prompt_cache_miss_tokens")
    )
    metrics = {
        "cached_tokens": cached_tokens,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_miss_tokens": cache_miss_tokens,
    }
    return metrics


def _emit_usage(on_usage: Callable[[dict], None] | None, usage) -> None:
    if on_usage is None or usage is None:
        return
    metrics = _extract_usage_metrics(usage)
    if not any(value is not None for value in metrics.values()):
        return
    try:
        on_usage(metrics)
    except Exception:
        return


def _merge_usage_metrics(usages: list[dict]) -> dict:
    merged = {
        "cached_tokens": None,
        "cache_hit_tokens": None,
        "cache_miss_tokens": None,
    }
    for usage in usages:
        for key in merged:
            value = _coerce_optional_int(usage.get(key))
            if value is None:
                continue
            merged[key] = value if merged[key] is None else merged[key] + value
    return merged


def _filter_global_glossary_terms(raw_terms) -> list[dict]:
    if not isinstance(raw_terms, list):
        return []
    filtered: list[dict] = []
    banned_re = re.compile(r"[,\u3001\u3002\uff0c\uff1f?？\s]")
    for item in raw_terms:
        if not isinstance(item, dict):
            continue
        ja = str(item.get("ja", "")).strip()
        zh = str(item.get("zh", "")).strip()
        if not ja or not zh:
            continue
        if len(ja) > 8 or len(zh) > 8:
            continue
        if banned_re.search(ja) or banned_re.search(zh):
            continue
        filtered.append({"ja": ja, "zh": zh})
        if len(filtered) >= 15:
            break
    return filtered


def _format_global_glossary_terms(
    terms: list[dict],
    *,
    glossary: str = "",
) -> str:
    lines = []
    seen: set[str] = set()
    for item in terms:
        ja = str(item.get("ja", "")).strip()
        zh = str(item.get("zh", "")).strip()
        if not ja or not zh or ja in seen:
            continue
        if glossary and ja in glossary:
            continue
        seen.add(ja)
        lines.append(f"{ja} \u2192 {zh}")
    return "\n".join(lines)


def _global_glossary_cache_path(translation_cache_path: str) -> str:
    cache_path = Path(translation_cache_path)
    return str(cache_path.with_name("translation_global_glossary.json"))


def extract_global_glossary(
    all_ja_texts: list[str],
    cache_path: str,
    *,
    api_format: str | None = None,
) -> list[dict]:
    if not cache_path:
        return []
    path = Path(cache_path)
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            terms = payload.get("terms") if isinstance(payload, dict) else payload
            return _filter_global_glossary_terms(terms)
    except Exception as exc:
        print(f"[WARN] failed to load translation global glossary cache: {exc}")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        source_text = "\n".join(str(text or "") for text in all_ja_texts)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是字幕术语提取器。请从全片日文字幕中提取 10-20 个反复出现的核心词，"
                    "范围包括代词、人名、性器官词、高频形容词。给出推荐中文翻译。"
                    '只返回合法 JSON：{"terms":[{"ja":"...","zh":"..."}]}。'
                ),
            },
            {"role": "user", "content": f"【全片日文字幕】\n{source_text}"},
        ]
        chat_kwargs = {"expected_count": 0, "reasoning_effort": "medium"}
        if api_format is not None:
            chat_kwargs["api_format"] = api_format
        raw_output = _chat(messages, **chat_kwargs)
        parsed = json.loads(_strip_reasoning_artifacts(raw_output))
        terms = _filter_global_glossary_terms(parsed.get("terms") if isinstance(parsed, dict) else None)
        tmp_path = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
        tmp_path.write_text(
            json.dumps({"terms": terms}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        return terms
    except Exception as exc:
        print(f"[WARN] failed to extract translation global glossary: {exc}")
        return []


def _test_crash_translation_batch() -> int:
    raw_value = os.getenv("_TEST_CRASH_TRANSLATION_BATCH", "").strip()
    if not raw_value:
        return 0
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 0


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def generate_global_context(
    segments: list[dict],
    current_index: int | None = None,
    batch_size: int = 0,
    max_chars: int | None = None,
) -> str:
    del current_index, batch_size

    lines = []
    for idx, seg in enumerate(segments):
        text = _normalize_source_text(seg.get("text", ""))
        if not text:
            continue
        start = _safe_float(seg.get("start"))
        end = _safe_float(seg.get("end"))
        lines.append(f"{idx:04d} [{start:.2f}->{end:.2f}] {text}")

    context = "\n".join(lines)
    if max_chars is not None and max_chars > 0:
        return context[:max_chars]
    return context


def translate_segments(
    segments: list[dict],
    batch_size: int = 100,
    global_context: str | None = None,
    max_workers: int = 1,
    cache_path: str = "",
    target_lang: str = "简体中文",
    glossary: str = "",
    character_reference: str | None = None,
    reasoning_effort: str | None = None,
    api_format: str | None = None,
    on_batch_done=None,
    on_progress: Callable[[dict], None] | None = None,
) -> tuple[list[str], list[dict], list[dict]]:
    if not segments:
        return [], [], []

    effective_batch_size = max(0, int(batch_size))
    effective_max_workers = max(1, int(max_workers))
    effective_cache_path = cache_path or ""
    effective_target_lang = (target_lang or "简体中文").strip() or "简体中文"
    effective_glossary = (glossary or "").strip()
    effective_character_reference = (
        CHARACTER_FULL_NAME_REFERENCE
        if character_reference is None
        else (character_reference or "").strip()
    )
    previous_retry_events = getattr(_RETRY_CONTEXT, "events", None)
    retry_events: list[dict] = []
    _RETRY_CONTEXT.events = retry_events
    try:
        if effective_batch_size > 0 and len(segments) > effective_batch_size:
            zh_texts, timings = _translate_segments_batched(
                segments,
                batch_size=effective_batch_size,
                max_workers=effective_max_workers,
                global_context=global_context,
                cache_path=effective_cache_path,
                target_lang=effective_target_lang,
                glossary=effective_glossary,
                character_reference=effective_character_reference,
                reasoning_effort=reasoning_effort,
                api_format=api_format,
                on_batch_done=on_batch_done,
                on_progress=on_progress,
            )
        else:
            zh_texts, timings = _translate_segments_single_request(
                segments,
                global_context=global_context,
                target_lang=effective_target_lang,
                glossary=effective_glossary,
                character_reference=effective_character_reference,
                reasoning_effort=reasoning_effort,
                api_format=api_format,
                on_batch_done=on_batch_done,
                on_progress=on_progress,
            )
        zh_texts, repair_timing = _apply_translation_repair_pass(
            segments,
            zh_texts,
            target_lang=effective_target_lang,
            glossary=effective_glossary,
            character_reference=effective_character_reference,
            reasoning_effort=reasoning_effort,
            api_format=api_format,
            on_progress=on_progress,
        )
        if repair_timing is not None:
            timings.append(repair_timing)
        return zh_texts, timings, list(retry_events)
    finally:
        if previous_retry_events is None:
            with contextlib.suppress(AttributeError):
                delattr(_RETRY_CONTEXT, "events")
        else:
            _RETRY_CONTEXT.events = previous_retry_events


def _chat_with_reasoning(
    messages: list[dict],
    *,
    expected_count: int,
    reasoning_effort: str | None = None,
    api_format: str | None = None,
    on_progress: Callable[[dict], None] | None = None,
    on_usage: Callable[[dict], None] | None = None,
) -> str:
    chat_kwargs = {
        "expected_count": expected_count,
        "on_progress": on_progress,
    }
    if reasoning_effort is not None:
        chat_kwargs["reasoning_effort"] = reasoning_effort
    if api_format is not None:
        chat_kwargs["api_format"] = api_format
    if on_usage is not None:
        chat_kwargs["on_usage"] = on_usage
    return _chat(
        messages,
        **chat_kwargs,
    )


def _translate_segments_single_request(
    segments: list[dict],
    *,
    global_context: str | None = None,
    target_lang: str,
    glossary: str,
    character_reference: str,
    reasoning_effort: str | None = None,
    api_format: str | None = None,
    on_batch_done=None,
    on_progress: Callable[[dict], None] | None = None,
) -> tuple[list[str], list[dict]]:
    started = time.perf_counter()
    full_context = (
        global_context
        if global_context is not None
        else generate_global_context(segments)
    )
    source_payload = _serialize_segments(segments)
    expected_count = len(segments)
    request_usages: list[dict] = []

    messages = _build_translation_messages(
        source_payload=source_payload,
        expected_count=expected_count,
        target_lang=target_lang,
        glossary=glossary,
        character_reference=character_reference,
    )
    missing_indexes: list[int] = []
    zh_texts: list[str | None] = [None] * expected_count
    for attempt in range(TRANSLATION_API_RETRIES):
        _emit_progress(on_progress, {"phase": "reset", "attempt": attempt})
        try:
            raw_output = _chat_with_reasoning(
                messages,
                expected_count=expected_count,
                reasoning_effort=reasoning_effort,
                api_format=api_format,
                on_progress=on_progress,
                on_usage=request_usages.append,
            )
            zh_texts = _parse_translation_output(raw_output, expected_count)
            missing_indexes = _missing_indexes(zh_texts)
            if missing_indexes:
                raise RetryableTranslationFormatError(
                    f"{_JSON_OUTPUT_LABEL} returned incomplete translations: "
                    f"{len(missing_indexes)} missing of {expected_count}; "
                    f"missing ids={missing_indexes[:50]}"
                )
            break
        except RetryableTranslationFormatError as exc:
            if attempt < TRANSLATION_API_RETRIES - 1:
                _request_backoff_sleep(attempt, exc)
                continue
            raise RuntimeError(
                "Single-request translation returned invalid or incomplete JSON after "
                f"{TRANSLATION_API_RETRIES} attempts: {exc}"
            ) from exc

    timing = {
        "start_index": 0,
        "segment_count": expected_count,
        "elapsed_s": time.perf_counter() - started,
        "mode": "single_request_full_context",
        "request_count": 1,
        "source_payload_chars": len(source_payload),
        "global_context_chars": len(full_context),
        "missing_count": 0,
        "missing_indexes": [],
        **_merge_usage_metrics(request_usages),
    }
    if on_batch_done:
        on_batch_done(timing)
    return [text or "" for text in zh_texts], [timing]


def _translate_segments_batched(
    segments: list[dict],
    *,
    batch_size: int,
    max_workers: int,
    global_context: str | None = None,
    cache_path: str = "",
    target_lang: str,
    glossary: str,
    character_reference: str,
    reasoning_effort: str | None = None,
    api_format: str | None = None,
    on_batch_done=None,
    on_progress: Callable[[dict], None] | None = None,
) -> tuple[list[str], list[dict]]:
    started = time.perf_counter()
    batches = _split_into_batches(segments, batch_size)
    expected_total = len(segments)
    extra_glossary = ""
    if cache_path:
        glossary_terms = extract_global_glossary(
            [str(seg.get("text", "")) for seg in segments],
            _global_glossary_cache_path(cache_path),
            api_format=api_format,
        )
        extra_glossary = _format_global_glossary_terms(
            glossary_terms,
            glossary=glossary,
        )
    full_context = (
        global_context
        if global_context is not None
        else generate_global_context(segments)
    )
    full_source_payload = _serialize_segments(segments, compact=True)
    use_full_json_prefix = (
        TRANSLATION_FULL_JSON_PREFIX_MAX_CHARS <= 0
        or len(full_source_payload) <= TRANSLATION_FULL_JSON_PREFIX_MAX_CHARS
    )
    prefix_mode = "full_json_prefix" if use_full_json_prefix else "summary_fallback"
    progress_callbacks, _ = _make_aggregated_progress_callback(
        len(batches),
        expected_total,
        on_progress,
    )
    diagnostic_progress_lock = threading.Lock()

    def emit_batch_diagnostic(payload: dict) -> None:
        if on_progress is None:
            return
        with diagnostic_progress_lock:
            _emit_progress(on_progress, payload)

    zh_texts: list[str | None] = [None] * expected_total
    timings_by_batch: dict[int, dict] = {}
    translation_cache = (
        _load_translation_cache(cache_path)
        if cache_path
        else {}
    )
    pending_batches: list[tuple[int, list[dict]]] = []
    warmup_timing: dict | None = None

    for batch_index, batch_segments in enumerate(batches):
        batch_key = _translation_cache_key(
            batch_index,
            batch_segments,
            extra_glossary=extra_glossary,
            glossary=glossary,
            target_lang=target_lang,
            character_reference=character_reference,
        )
        cached_texts = translation_cache.get(batch_key)
        if isinstance(cached_texts, list) and len(cached_texts) == len(batch_segments):
            print(f"[translation-cache] restored batch {batch_index} cache_key={batch_key}")
            start_index = batch_index * batch_size
            for offset, text in enumerate(cached_texts):
                zh_texts[start_index + offset] = str(text or "")
            timing = {
                "batch_index": batch_index,
                "start_index": start_index,
                "segment_count": len(batch_segments),
                "elapsed_s": 0.0,
                "mode": "translation_cache_hit",
                "request_count": 0,
                "source_payload_chars": 0,
                "global_context_chars": len(full_context),
                "prefix_mode": prefix_mode,
                "requested_ids": list(range(start_index, start_index + len(batch_segments))),
                "is_warmup": False,
                **_merge_usage_metrics([]),
                "missing_count": 0,
                "missing_indexes": [],
                "cache_hit": True,
            }
            timings_by_batch[batch_index] = timing
            _emit_progress(
                progress_callbacks[batch_index],
                {
                    "phase": "done",
                    "translated": len(batch_segments),
                    "expected": len(batch_segments),
                },
            )
            if on_batch_done:
                on_batch_done(timing)
        else:
            pending_batches.append((batch_index, batch_segments))

    if pending_batches and use_full_json_prefix and TRANSLATION_PREFIX_WARMUP:
        warmup_started = time.perf_counter()
        warmup_usages: list[dict] = []
        warmup_messages = _build_batch_messages(
            [],
            full_context,
            0,
            character_reference,
            0,
            batch_index=0,
            extra_glossary=extra_glossary,
            target_lang=target_lang,
            glossary=glossary,
            full_source_payload=full_source_payload,
            requested_ids=[],
            warmup=True,
        )
        try:
            _chat_with_reasoning(
                warmup_messages,
                expected_count=0,
                reasoning_effort=reasoning_effort,
                api_format=api_format,
                on_usage=warmup_usages.append,
            )
            warmup_timing = {
                "batch_index": None,
                "start_index": 0,
                "segment_count": 0,
                "elapsed_s": time.perf_counter() - warmup_started,
                "mode": "translation_prefix_warmup",
                "request_count": 1,
                "source_payload_chars": 0,
                "global_context_chars": len(full_source_payload),
                "prefix_mode": prefix_mode,
                "requested_ids": [],
                "is_warmup": True,
                "missing_count": 0,
                "missing_indexes": [],
                **_merge_usage_metrics(warmup_usages),
            }
        except Exception as exc:
            warmup_timing = {
                "batch_index": None,
                "start_index": 0,
                "segment_count": 0,
                "elapsed_s": time.perf_counter() - warmup_started,
                "mode": "translation_prefix_warmup_failed",
                "request_count": 1,
                "source_payload_chars": 0,
                "global_context_chars": len(full_source_payload),
                "prefix_mode": prefix_mode,
                "requested_ids": [],
                "is_warmup": True,
                "missing_count": 0,
                "missing_indexes": [],
                "error": str(exc)[:500],
                **_merge_usage_metrics(warmup_usages),
            }
            print(f"[WARN] translation prefix warmup failed: {exc}", flush=True)

    def run_batch(batch_index: int, batch_segments: list[dict]) -> tuple[int, list[str | None], dict]:
        batch_started = time.perf_counter()
        batch_started_ts = time.time()
        worker_thread = threading.current_thread()
        worker_thread_id = threading.get_ident()
        worker_thread_name = worker_thread.name
        start_index = batch_index * batch_size
        expected_count = len(batch_segments)
        source_payload = _serialize_segments(batch_segments, start_index=start_index)
        expected_ids = list(range(start_index, start_index + expected_count))
        trace_base = {
            "diagnostic": True,
            "batch_index": batch_index,
            "start_index": start_index,
            "segment_count": expected_count,
            "thread_id": worker_thread_id,
            "thread_name": worker_thread_name,
            "started_ts": batch_started_ts,
            "requested_ids": expected_ids,
        }
        emit_batch_diagnostic({"phase": "batch_start", **trace_base})
        messages = _build_batch_messages(
            batch_segments,
            full_context,
            start_index,
            character_reference,
            expected_count,
            batch_index=batch_index,
            extra_glossary=extra_glossary,
            target_lang=target_lang,
            glossary=glossary,
            full_source_payload=full_source_payload if use_full_json_prefix else None,
            requested_ids=expected_ids,
        )
        batch_results: list[str | None] = [None] * expected_total
        missing_indexes: list[int] = []
        progress_callback = progress_callbacks[batch_index]
        request_count = 0
        pending_ids = list(expected_ids)
        pending_segments = list(batch_segments)
        request_usages: list[dict] = []
        first_token_ts: float | None = None
        active_request_index = 0
        active_requested_ids = list(expected_ids)

        def trace_progress(evt: dict) -> None:
            nonlocal first_token_ts
            payload = dict(evt)
            phase = payload.get("phase")
            if phase in {"thinking", "translating", "done"} and first_token_ts is None:
                first_token_ts = time.time()
                emit_batch_diagnostic(
                    {
                        "phase": "batch_first_token",
                        **trace_base,
                        "first_token_ts": first_token_ts,
                        "source_phase": phase,
                        "request_index": active_request_index,
                        "requested_ids": list(active_requested_ids),
                    }
                )
            _emit_progress(progress_callback, payload)

        attempts_for_pending = 0
        retry_limit_for_pending = TRANSLATION_API_RETRIES
        last_retry_error: RetryableTranslationFormatError | None = None

        while True:
            if attempts_for_pending >= retry_limit_for_pending:
                raise RuntimeError(
                    "Batch translation returned invalid or incomplete JSON after "
                    f"{request_count} attempts: batch={batch_index}, "
                    f"start_index={start_index}, size={expected_count}, "
                    f"pending_ids={pending_ids[:50]}, error={last_retry_error}"
                ) from last_retry_error

            requested_ids = list(pending_ids)
            active_request_index = request_count
            active_requested_ids = list(requested_ids)
            trace_progress({"phase": "reset", "attempt": request_count})
            try:
                if request_count == 0:
                    request_messages = messages
                    request_expected_count = expected_count
                    request_source_payload = source_payload
                else:
                    request_expected_count = len(pending_segments)
                    request_source_payload = _serialize_segments(
                        pending_segments,
                        explicit_ids=pending_ids,
                    )
                    request_messages = _build_batch_messages(
                        pending_segments,
                        full_context,
                        0,
                        character_reference,
                        request_expected_count,
                        batch_index=batch_index,
                        extra_glossary=extra_glossary,
                        target_lang=target_lang,
                        glossary=glossary,
                        source_payload_override=request_source_payload,
                        full_source_payload=full_source_payload if use_full_json_prefix else None,
                        requested_ids=pending_ids,
                    )
                request_count += 1
                raw_output = _chat_with_reasoning(
                    request_messages,
                    expected_count=request_expected_count,
                    reasoning_effort=reasoning_effort,
                    api_format=api_format,
                    on_progress=trace_progress,
                    on_usage=request_usages.append,
                )
                parsed = _parse_partial_translation_output_by_global_id(
                    raw_output,
                    expected_ids=pending_ids,
                    total_count=expected_total,
                )
                for idx in pending_ids:
                    if parsed[idx]:
                        batch_results[idx] = parsed[idx]
                missing_indexes = [
                    index
                    for index in expected_ids
                    if batch_results[index] is None
                ]
                if not missing_indexes:
                    break

                pending_ids = list(missing_indexes)
                pending_segments = [
                    batch_segments[index - start_index]
                    for index in pending_ids
                ]
                last_retry_error = RetryableTranslationFormatError(
                    f"{_JSON_OUTPUT_LABEL} returned incomplete batch translations: "
                    f"{len(missing_indexes)} missing of {expected_count}; "
                    f"missing ids={missing_indexes[:50]}"
                )
                if len(missing_indexes) < len(requested_ids):
                    attempts_for_pending = 0
                    retry_limit_for_pending = TRANSLATION_BATCH_REPAIR_RETRIES
                else:
                    attempts_for_pending += 1
            except RetryableTranslationFormatError as exc:
                last_retry_error = exc
                attempts_for_pending += 1

            if attempts_for_pending < retry_limit_for_pending:
                sleep_attempt = max(0, attempts_for_pending - 1)
                _request_backoff_sleep(sleep_attempt, last_retry_error)
                continue

            raise RuntimeError(
                "Batch translation returned invalid or incomplete JSON after "
                f"{request_count} attempts: batch={batch_index}, "
                f"start_index={start_index}, size={expected_count}, "
                f"pending_ids={pending_ids[:50]}, error={last_retry_error}"
            ) from last_retry_error

        batch_elapsed_s = time.perf_counter() - batch_started
        batch_finished_ts = time.time()
        if first_token_ts is None:
            first_token_ts = batch_finished_ts
        emit_batch_diagnostic(
            {
                "phase": "batch_finish",
                **trace_base,
                "first_token_ts": first_token_ts,
                "finished_ts": batch_finished_ts,
                "elapsed_s": batch_elapsed_s,
                "request_count": request_count,
                "missing_count": len(missing_indexes),
                "missing_indexes": missing_indexes,
            }
        )

        timing = {
            "batch_index": batch_index,
            "start_index": start_index,
            "segment_count": expected_count,
            "elapsed_s": batch_elapsed_s,
            "mode": "batched_full_context",
            "request_count": request_count,
            "source_payload_chars": len(source_payload),
            "global_context_chars": len(full_source_payload) if use_full_json_prefix else len(full_context),
            "prefix_mode": prefix_mode,
            "requested_ids": expected_ids,
            "is_warmup": False,
            "started_ts": batch_started_ts,
            "first_token_ts": first_token_ts,
            "finished_ts": batch_finished_ts,
            "worker_thread_id": worker_thread_id,
            "worker_thread_name": worker_thread_name,
            **_merge_usage_metrics(request_usages),
            "missing_count": len(missing_indexes),
            "missing_indexes": missing_indexes,
        }
        return batch_index, batch_results, timing

    if pending_batches:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(pending_batches))) as executor:
            pending_by_index = {
                batch_index: batch for batch_index, batch in pending_batches
            }
            futures = {
                executor.submit(run_batch, batch_index, batch): batch_index
                for batch_index, batch in pending_batches
            }
            try:
                for future in as_completed(futures):
                    batch_index, batch_results, timing = future.result()
                    timings_by_batch[batch_index] = timing
                    start_index = int(timing["start_index"])
                    segment_count = int(timing["segment_count"])
                    local_texts: list[str] = []
                    for offset in range(segment_count):
                        global_index = start_index + offset
                        text = batch_results[global_index] or ""
                        zh_texts[global_index] = text
                        local_texts.append(text)
                    if cache_path:
                        batch_key = _translation_cache_key(
                            batch_index,
                            pending_by_index[batch_index],
                            extra_glossary=extra_glossary,
                            glossary=glossary,
                            target_lang=target_lang,
                            character_reference=character_reference,
                        )
                        _save_cache_entry(
                            cache_path,
                            batch_key,
                            local_texts,
                            _cache_lock,
                        )
                        print(f"[translation-cache] saved batch {batch_index} cache_key={batch_key}")
                        if _test_crash_translation_batch() == batch_index + 1:
                            for pending_future in futures:
                                if pending_future is not future:
                                    pending_future.cancel()
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise SystemExit(1)
                    if on_batch_done:
                        on_batch_done(timing)
            except Exception:
                for pending_future in futures:
                    pending_future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    missing = _missing_indexes(zh_texts)
    if missing:
        raise RuntimeError(
            "Batched translation finished with missing translations: "
            f"{len(missing)} missing; ids={missing[:50]}"
        )

    timings = []
    if warmup_timing is not None:
        timings.append(warmup_timing)
    timings.extend(timings_by_batch[index] for index in sorted(timings_by_batch))
    timings.append(
        {
            "start_index": 0,
            "segment_count": expected_total,
            "elapsed_s": time.perf_counter() - started,
            "mode": "batched_full_context_total",
            "request_count": len(pending_batches),
            "batch_size": batch_size,
            "max_workers": max_workers,
            "cache_hit_count": len(batches) - len(pending_batches),
            "prefix_mode": prefix_mode,
            "is_warmup": False,
            "requested_ids": [],
            "missing_count": 0,
            "missing_indexes": [],
        }
    )
    return [text or "" for text in zh_texts], timings


_FEMALE_SOURCE_TERMS = ("まんこ", "マンコ", "おまんこ")
_SUSPICIOUS_ASR_TERMS = (
    "マンゴー",
    "ウィンナー",
    "おまけ",
    "私の国",
    "国に",
    "こよく",
    "きゅうしてください",
)
_FORBIDDEN_FEMALE_TRANSLATIONS = ("阴道", "陰道", "肉棒", "芒果", "香肠", "香腸")
_FRAGMENT_REPAIR_SOURCE_MARKERS = (
    "精子",
    "まんこ",
    "マンコ",
    "おまんこ",
    "マンゴー",
    "ウィンナー",
    "おまけ",
    "私の国",
    "国に",
    "こよく",
    "きゅうしてください",
    "ちん",
    "チン",
)
_LOW_CONFIDENCE_REPAIR_REASONS = {
    "low_confidence",
    "low_translation_confidence",
}


def _apply_translation_repair_pass(
    segments: list[dict],
    zh_texts: list[str],
    *,
    target_lang: str,
    glossary: str,
    character_reference: str,
    reasoning_effort: str | None = None,
    api_format: str | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> tuple[list[str], dict | None]:
    repair_ids, reasons = _select_translation_repair_ids(segments, zh_texts)
    if not TRANSLATION_REPAIR_ENABLED or not repair_ids:
        return zh_texts, None

    if TRANSLATION_REPAIR_MAX_IDS <= 0:
        return zh_texts, None
    repair_ids = repair_ids[:TRANSLATION_REPAIR_MAX_IDS]
    started = time.perf_counter()
    request_usages: list[dict] = []
    _emit_progress(
        on_progress,
        {
            "phase": "repair_start",
            "repair_ids": repair_ids,
            "candidate_count": len(repair_ids),
        },
    )
    try:
        messages = _build_repair_messages(
            segments,
            zh_texts,
            repair_ids,
            reasons,
            target_lang=target_lang,
            glossary=glossary,
            character_reference=character_reference,
        )
        raw_output = _chat_with_reasoning(
            messages,
            expected_count=len(repair_ids),
            reasoning_effort=reasoning_effort,
            api_format=api_format,
            on_usage=request_usages.append,
        )
        parsed = _parse_translation_output_by_global_id(
            raw_output,
            expected_ids=repair_ids,
            total_count=len(segments),
        )
        repaired_texts = list(zh_texts)
        repaired_count = 0
        for idx in repair_ids:
            if parsed[idx]:
                repaired_texts[idx] = parsed[idx] or repaired_texts[idx]
                repaired_count += 1
        timing = {
            "mode": "translation_repair_pass",
            "start_index": min(repair_ids),
            "segment_count": repaired_count,
            "elapsed_s": time.perf_counter() - started,
            "request_count": 1,
            "repair_ids": repair_ids,
            "candidate_count": len(repair_ids),
            "missing_count": len(repair_ids) - repaired_count,
            "missing_indexes": [
                idx for idx in repair_ids if parsed[idx] is None
            ],
            **_merge_usage_metrics(request_usages),
        }
        _emit_progress(
            on_progress,
            {
                "phase": "repair_done",
                "repair_ids": repair_ids,
                "repaired": repaired_count,
                "expected": len(repair_ids),
            },
        )
        return repaired_texts, timing
    except Exception as exc:
        timing = {
            "mode": "translation_repair_failed",
            "start_index": min(repair_ids),
            "segment_count": 0,
            "elapsed_s": time.perf_counter() - started,
            "request_count": 1,
            "repair_ids": repair_ids,
            "candidate_count": len(repair_ids),
            "missing_count": len(repair_ids),
            "missing_indexes": repair_ids,
            "error": str(exc)[:500],
            **_merge_usage_metrics(request_usages),
        }
        print(f"[WARN] translation repair failed: {exc}", flush=True)
        _emit_progress(
            on_progress,
            {
                "phase": "repair_failed",
                "repair_ids": repair_ids,
                "error": str(exc)[:200],
            },
        )
        return zh_texts, timing


def _select_translation_repair_ids(
    segments: list[dict],
    zh_texts: list[str],
) -> tuple[list[int], dict[int, list[str]]]:
    repair_ids: list[int] = []
    reasons: dict[int, list[str]] = {}
    for idx, seg in enumerate(segments):
        source = _repair_source_text(seg)
        target = _repair_translation_text(seg, zh_texts, idx)
        local_reasons = _translation_repair_reasons(
            segments,
            zh_texts,
            idx,
            source,
            target,
        )
        if _has_translation_length_mismatch(source, target):
            local_reasons.append("length_mismatch")
        if not local_reasons:
            continue
        repair_ids.append(idx)
        reasons[idx] = list(dict.fromkeys(local_reasons))
    repair_ids.sort(key=lambda idx: (_repair_candidate_priority(reasons[idx]), idx))
    return repair_ids, reasons


def _repair_source_text(seg: dict) -> str:
    return _normalize_source_text(
        seg.get("source")
        or seg.get("ja_text")
        or seg.get("text")
        or seg.get("ja")
        or ""
    )


def _repair_translation_text(seg: dict, zh_texts: list[str], idx: int) -> str:
    if idx < len(zh_texts) and zh_texts[idx] is not None:
        return str(zh_texts[idx]).strip()
    return str(seg.get("translation") or "").strip()


def _has_translation_length_mismatch(source: str, target: str) -> bool:
    ratio = len(target) / max(len(source), 1)
    return (
        ratio < TRANSLATION_REPAIR_LENGTH_RATIO_MIN
        or ratio > TRANSLATION_REPAIR_LENGTH_RATIO_MAX
    )


def _repair_candidate_priority(local_reasons: list[str]) -> int:
    if "length_mismatch" in local_reasons:
        return 0
    if any(reason in _LOW_CONFIDENCE_REPAIR_REASONS for reason in local_reasons):
        return 1
    return 0


def _translation_repair_reasons(
    segments: list[dict],
    zh_texts: list[str],
    idx: int,
    source: str,
    target: str,
) -> list[str]:
    del zh_texts

    reasons: list[str] = []
    has_female_source = any(term in source for term in _FEMALE_SOURCE_TERMS)
    if has_female_source and any(term in target for term in _FORBIDDEN_FEMALE_TRANSLATIONS):
        reasons.append("female_term_drift")

    has_mango = "マンゴー" in source
    if has_mango and _looks_like_sexual_asr_context(segments, idx):
        if "小穴" not in target or _looks_like_fragment_translation(target):
            reasons.append("suspicious_mango_asr")

    has_wiener = "ウィンナー" in source
    if has_wiener and (
        "広げ" in source
        or "肉棒" in target
        or "香肠" in target
        or "香腸" in target
    ):
        reasons.append("suspicious_wiener_asr")

    if (
        "おまけ" in source
        and ("さらけ" in source or _looks_like_sexual_asr_context(segments, idx))
        and ("露" in target or "赠" in target or _looks_like_fragment_translation(target))
    ):
        reasons.append("suspicious_omake_asr")

    if (
        ("私の国" in source or "国に" in source)
        and _looks_like_sexual_asr_context(segments, idx)
        and any(term in target for term in ("国家", "国", "我国"))
    ):
        reasons.append("suspicious_kuni_asr")

    if (
        "こよく" in source
        and _looks_like_sexual_asr_context(segments, idx)
        and ("说" in target or "説" in target or _looks_like_fragment_translation(target))
    ):
        reasons.append("suspicious_koyoku_asr")

    if (
        "きゅう" in source
        and "してください" in source
        and any(term in target for term in ("吸", "吮"))
    ):
        reasons.append("suspicious_kyuu_asr")

    if (
        any(marker in source for marker in _FRAGMENT_REPAIR_SOURCE_MARKERS)
        and _looks_like_fragment_source(source)
        and _looks_like_fragment_translation(target)
    ):
        reasons.append("fragment_translation")

    return reasons


def _looks_like_sexual_asr_context(segments: list[dict], idx: int) -> bool:
    start = max(0, idx - 1)
    end = min(len(segments), idx + 2)
    joined = "\n".join(
        _normalize_source_text(seg.get("text", ""))
        for seg in segments[start:end]
    )
    return any(
        marker in joined
        for marker in (
            "精子",
            "射",
            "入れ",
            "入っ",
            "気持ち",
            "イク",
            "イッ",
            "行く",
            "エロ",
            "まんこ",
            "マンコ",
            "おまんこ",
        )
    )


def _looks_like_fragment_source(source: str) -> bool:
    stripped = source.strip()
    if not stripped or re.search(r"[。！？!?]$", stripped):
        return False
    return bool(re.search(r"(さ|に|を|で|て|が|の|へ)$", stripped))


def _looks_like_fragment_translation(target: str) -> bool:
    stripped = target.strip()
    if not stripped:
        return True
    if re.search(r"[。！？!?]$", stripped):
        return False
    if len(stripped) <= 10:
        return True
    return bool(re.search(r"[，、,]\s*[^，、,。！？!?]{1,6}$", stripped))


def _build_repair_messages(
    segments: list[dict],
    zh_texts: list[str],
    repair_ids: list[int],
    reasons: dict[int, list[str]],
    *,
    target_lang: str,
    glossary: str,
    character_reference: str,
) -> list[dict]:
    system_prompt = _build_system_prompt(
        len(repair_ids),
        character_reference,
        target_lang=target_lang,
        glossary=glossary,
    )
    system_prompt += (
        "\n\n这是翻译后局部修复任务。只修复 requested_ids 中的译文；"
        "必须利用 reason 字段、相邻字幕和 current_zh 修复明显 ASR 同音误听、上下文漂移、术语漂移和被切断的半句。"
        "reason 只是问题类别提示，不是固定译文；最终译文必须服从原文、上下文和既定术语。"
        "性器官术语继续统一为肉棒/小穴，不要漂移成其他书面或错误译法。"
    )
    context_items = _build_repair_context_items(
        segments,
        zh_texts,
        repair_ids,
        reasons,
    )
    user_content = "\n\n".join(
        [
            "【翻译修复任务】",
            f"requested_ids = {_format_requested_ids(repair_ids)}",
            "只返回 requested_ids 中列出的 id，恰好返回这些 id，不要返回 context_only 项。",
            "每个 text 只能是修复后的中文字幕；不要解释原因。",
            "【局部上下文 JSON】",
            json.dumps(context_items, ensure_ascii=False, indent=2),
            '输出 JSON：{"translations":[{"id":0,"text":"..."}]}',
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _build_repair_context_items(
    segments: list[dict],
    zh_texts: list[str],
    repair_ids: list[int],
    reasons: dict[int, list[str]],
) -> list[dict]:
    indexes: set[int] = set()
    radius = TRANSLATION_REPAIR_CONTEXT_RADIUS
    for idx in repair_ids:
        indexes.update(range(max(0, idx - radius), min(len(segments), idx + radius + 1)))

    items = []
    repair_id_set = set(repair_ids)
    for idx in sorted(indexes):
        seg = segments[idx]
        items.append(
            {
                "id": idx,
                "role": "repair" if idx in repair_id_set else "context_only",
                "reason": _public_repair_reasons(reasons.get(idx, [])),
                "start": _safe_float(seg.get("start")),
                "end": _safe_float(seg.get("end")),
                "ja": _repair_source_text(seg),
                "current_zh": _repair_translation_text(seg, zh_texts, idx),
            }
        )
    return items


def _public_repair_reasons(local_reasons: list[str]) -> list[str]:
    public: list[str] = []
    for reason in local_reasons:
        if reason == "female_term_drift":
            public.append("female_term_drift")
        elif reason == "fragment_translation":
            public.append("fragment_translation")
        elif reason == "length_mismatch":
            public.append("length_mismatch")
        elif reason.startswith("suspicious_"):
            public.append("asr_homophone_or_context_drift")
        else:
            public.append("translation_drift")
    return list(dict.fromkeys(public))


def _split_into_batches(segments: list[dict], batch_size: int) -> list[list[dict]]:
    if not segments:
        return []
    if batch_size <= 0:
        return [segments]
    return [segments[index : index + batch_size] for index in range(0, len(segments), batch_size)]


_serialize_segments = prompt_module._serialize_segments
_build_full_segments_summary = prompt_module._build_full_segments_summary


def _build_system_prompt(
    expected_count: int,
    character_reference: str,
    *,
    target_lang: str,
    glossary: str,
    compact: bool = False,
    extra_glossary: str = "",
) -> str:
    return prompt_module._build_system_prompt(
        expected_count,
        character_reference,
        target_lang=target_lang,
        glossary=glossary,
        compact=compact,
        extra_glossary=extra_glossary,
        full_template=_SYSTEM_PROMPT_FULL,
        compact_template=_SYSTEM_PROMPT_COMPACT,
    )


def _build_translation_messages(
    source_payload: str,
    expected_count: int,
    compact_system_prompt: bool = False,
    extra_glossary: str = "",
    target_lang: str = "简体中文",
    glossary: str = "",
    character_reference: str | None = None,
) -> list[dict]:
    effective_character_reference = (
        CHARACTER_FULL_NAME_REFERENCE
        if character_reference is None
        else (character_reference or "").strip()
    )
    system_prompt = _build_system_prompt(
        expected_count,
        effective_character_reference,
        target_lang=target_lang,
        glossary=glossary,
        compact=compact_system_prompt,
        extra_glossary=extra_glossary,
    )
    return prompt_module._build_translation_messages(
        source_payload=source_payload,
        expected_count=expected_count,
        compact_system_prompt=compact_system_prompt,
        extra_glossary=extra_glossary,
        target_lang=target_lang,
        glossary=glossary,
        character_reference=effective_character_reference,
        system_prompt=system_prompt,
    )


_format_requested_ids = prompt_module._format_requested_ids
_build_requested_ids_task = prompt_module._build_requested_ids_task


def _build_batch_messages(
    batch_segments: list[dict],
    full_segments_summary: str | list[dict],
    batch_offset: int,
    character_reference: str,
    expected_count: int,
    batch_index: int = 0,
    extra_glossary: str = "",
    target_lang: str = "简体中文",
    glossary: str = "",
    source_payload_override: str | None = None,
    full_source_payload: str | None = None,
    requested_ids: list[int] | None = None,
    warmup: bool = False,
) -> list[dict]:
    return prompt_module._build_batch_messages(
        batch_segments,
        full_segments_summary,
        batch_offset,
        character_reference,
        expected_count,
        batch_index=batch_index,
        extra_glossary=extra_glossary,
        target_lang=target_lang,
        glossary=glossary,
        source_payload_override=source_payload_override,
        full_source_payload=full_source_payload,
        requested_ids=requested_ids,
        warmup=warmup,
        compact_system_prompt_enabled=COMPACT_SYSTEM_PROMPT,
    )


_build_character_name_guidance = prompt_module._build_character_name_guidance


def _is_retryable_api_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True

    name = type(exc).__name__.lower()
    return any(
        marker in name
        for marker in (
            "ratelimit",
            "timeout",
            "connection",
            "serviceunavailable",
            "internalserver",
        )
    )


def _request_backoff_delay(attempt: int) -> float:
    return min(
        TRANSLATION_API_BACKOFF_MAX_S,
        TRANSLATION_API_BACKOFF_BASE_S * (2**attempt),
    )


def _record_api_retry_event(exc: Exception, attempt: int, delay_s: float) -> None:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)

    event = {
        "attempt": attempt + 1,
        "delay_s": delay_s,
        "status_code": status_code,
        "error_type": type(exc).__name__,
        "message": str(exc)[:500],
    }
    local_events = _current_retry_events()
    if local_events is not None:
        local_events.append(event)


def _request_backoff_sleep(attempt: int, exc: Exception) -> None:
    delay = _request_backoff_delay(attempt)
    _record_api_retry_event(exc, attempt, delay)
    time.sleep(delay)


def _create_chat_completion(request: dict):
    last_error: Exception | None = None

    for attempt in range(TRANSLATION_API_RETRIES):
        try:
            return _get_client().chat.completions.create(**request)
        except Exception as exc:
            last_error = exc
            if not _is_retryable_api_error(exc):
                raise

            if attempt < TRANSLATION_API_RETRIES - 1:
                _request_backoff_sleep(attempt, exc)

    if last_error is not None:
        raise last_error
    raise RuntimeError("chat completion failed without an exception")


def _create_response(request: dict):
    last_error: Exception | None = None

    for attempt in range(TRANSLATION_API_RETRIES):
        try:
            return _get_client().responses.create(**request)
        except Exception as exc:
            last_error = exc
            if not _is_retryable_api_error(exc):
                raise

            if attempt < TRANSLATION_API_RETRIES - 1:
                _request_backoff_sleep(attempt, exc)

    if last_error is not None:
        raise last_error
    raise RuntimeError("response creation failed without an exception")


def _emit_progress(
    on_progress: Callable[[dict], None] | None,
    payload: dict,
) -> None:
    if on_progress is None:
        return
    try:
        on_progress(payload)
    except Exception:
        pass


def _make_aggregated_progress_callback(
    num_batches: int,
    expected_total: int,
    on_progress: Callable[[dict], None] | None,
) -> tuple[list[Callable[[dict], None]], None]:
    lock = threading.Lock()
    batch_states: dict[int, dict] = {}
    last_emit_ts = 0.0
    done_emitted = False

    def emit(payload: dict, *, force: bool = False) -> None:
        nonlocal last_emit_ts
        now = time.monotonic()
        if not force and now - last_emit_ts < 0.25:
            return
        last_emit_ts = now
        _emit_progress(on_progress, payload)

    def build_payload() -> tuple[dict, bool] | None:
        nonlocal done_emitted
        if not batch_states:
            return None

        translated = sum(
            int(state.get("translated", 0))
            for state in batch_states.values()
            if state.get("phase") in {"translating", "done"}
        )
        if (
            num_batches > 0
            and len(batch_states) >= num_batches
            and all(state.get("phase") == "done" for state in batch_states.values())
        ):
            if done_emitted:
                return None
            done_emitted = True
            return (
                {"phase": "done", "translated": expected_total, "expected": expected_total},
                True,
            )

        if any(state.get("phase") == "translating" for state in batch_states.values()):
            return (
                {
                    "phase": "translating",
                    "translated": translated,
                    "expected": expected_total,
                    "content_chars": sum(
                        int(state.get("content_chars", 0))
                        for state in batch_states.values()
                    ),
                },
                False,
            )

        if any(state.get("phase") == "thinking" for state in batch_states.values()):
            return (
                {
                    "phase": "thinking",
                    "reasoning_chars": max(
                        int(state.get("reasoning_chars", 0))
                        for state in batch_states.values()
                    ),
                },
                False,
            )

        if any(state.get("phase") == "reset" for state in batch_states.values()):
            return (
                {
                    "phase": "reset",
                    "attempt": max(
                        int(state.get("attempt", 0))
                        for state in batch_states.values()
                    ),
                },
                True,
            )
        return None

    def make_wrapper(batch_id: int) -> Callable[[dict], None]:
        def wrapper(evt: dict) -> None:
            try:
                payload: tuple[dict, bool] | None
                with lock:
                    batch_states[batch_id] = dict(evt)
                    payload = build_payload()
                if payload is None:
                    return
                emit(payload[0], force=payload[1])
            except Exception:
                pass

        return wrapper

    return [make_wrapper(batch_id) for batch_id in range(num_batches)], None


def _count_translation_markers(
    *,
    piece: str,
    id_scan_tail: str,
    id_marker: str,
) -> tuple[int, str]:
    scan_text = id_scan_tail + piece
    tail_len = len(id_scan_tail)
    count = sum(
        1
        for match in re.finditer(re.escape(id_marker), scan_text)
        if match.end() > tail_len
    )
    return count, scan_text[-(len(id_marker) - 1) :]


def _emit_stream_content_progress(
    *,
    piece: str,
    state: dict,
    expected_count: int,
    maybe_emit: Callable[[dict], None],
) -> None:
    state["final_content"].append(piece)
    state["content_chars"] += len(piece)
    count, state["id_scan_tail"] = _count_translation_markers(
        piece=piece,
        id_scan_tail=state["id_scan_tail"],
        id_marker=state["id_marker"],
    )
    state["translated_count"] += count
    maybe_emit(
        {
            "phase": "translating",
            "translated": state["translated_count"],
            "expected": expected_count,
            "content_chars": state["content_chars"],
        }
    )


def _build_responses_input(
    messages: list[dict],
    *,
    string_content: bool = False,
) -> list[dict]:
    response_input: list[dict] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content", "")
        if string_content:
            response_input.append({"role": role, "content": str(content)})
            continue
        response_input.append(
            {
                "role": role,
                "content": [
                    {
                        "type": "input_text",
                        "text": str(content),
                    }
                ],
            }
        )
    return response_input


def _extract_response_output_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "".join(parts)


def _response_event_type(event) -> str:
    value = getattr(event, "type", "")
    if value:
        return str(value)
    if isinstance(event, dict):
        return str(event.get("type", ""))
    return ""


def _response_event_delta(event) -> str:
    value = getattr(event, "delta", None)
    if value is None and isinstance(event, dict):
        value = event.get("delta")
    return value if isinstance(value, str) else ""


def _response_event_response(event):
    value = getattr(event, "response", None)
    if value is None and isinstance(event, dict):
        value = event.get("response")
    return value


def _response_incomplete_reason(response) -> str:
    details = getattr(response, "incomplete_details", None)
    if isinstance(details, dict):
        return str(details.get("reason", ""))
    return str(getattr(details, "reason", "") or "")


def _chat(
    messages: list[dict],
    expected_count: int = 0,
    on_progress: Callable[[dict], None] | None = None,
    reasoning_effort: str | None = None,
    api_format: str | None = None,
    on_usage: Callable[[dict], None] | None = None,
) -> str:
    if _llm_api_format(api_format) == "responses":
        return _chat_responses(
            messages,
            expected_count=expected_count,
            on_progress=on_progress,
            reasoning_effort=reasoning_effort,
            on_usage=on_usage,
        )
    return _chat_completions(
        messages,
        expected_count=expected_count,
        on_progress=on_progress,
        reasoning_effort=reasoning_effort,
        on_usage=on_usage,
    )


def _chat_completions(
    messages: list[dict],
    expected_count: int = 0,
    on_progress: Callable[[dict], None] | None = None,
    reasoning_effort: str | None = None,
    on_usage: Callable[[dict], None] | None = None,
) -> str:
    model_name = os.getenv("LLM_MODEL_NAME", LLM_MODEL_NAME).strip()
    if not model_name:
        raise RuntimeError("请先在「翻译设置」中获取并选择翻译模型，再提交任务")
    request = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "response_format": {"type": "json_object"},
        "reasoning_effort": _normalize_reasoning_effort(
            reasoning_effort or os.getenv("LLM_REASONING_EFFORT", LLM_REASONING_EFFORT)
        ),
        "extra_body": {"thinking": {"type": "enabled"}},
        "stream_options": {"include_usage": True},
    }
    if TRANSLATION_MAX_TOKENS > 0:
        request["max_tokens"] = TRANSLATION_MAX_TOKENS

    try:
        response_stream = _create_chat_completion(request)
    except Exception as exc:
        if "stream_options" not in request or "stream_options" not in str(exc):
            raise
        request = dict(request)
        request.pop("stream_options", None)
        response_stream = _create_chat_completion(request)

    finish_reason = None
    reasoning_chars = 0
    last_emit = 0.0
    debounce_s = 0.25
    stream_state = {
        "final_content": [],
        "content_chars": 0,
        "translated_count": 0,
        "id_scan_tail": "",
        "id_marker": '"id":',
    }

    def maybe_emit(payload: dict, *, force: bool = False) -> None:
        nonlocal last_emit
        now = time.monotonic()
        if not force and now - last_emit < debounce_s:
            return
        last_emit = now
        _emit_progress(on_progress, payload)

    for chunk in response_stream:
        _emit_usage(on_usage, getattr(chunk, "usage", None))
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning_content = getattr(delta, "reasoning_content", None)
        if reasoning_content:
            reasoning_chars += len(reasoning_content)
            maybe_emit(
                {"phase": "thinking", "reasoning_chars": reasoning_chars}
            )
        if hasattr(delta, "content") and delta.content:
            _emit_stream_content_progress(
                piece=delta.content,
                state=stream_state,
                expected_count=expected_count,
                maybe_emit=maybe_emit,
            )
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    if finish_reason == "length":
        raise RetryableTranslationFormatError(
            f"{_JSON_OUTPUT_LABEL} response was cut off by max_tokens; "
            "increase TRANSLATION_MAX_TOKENS."
        )

    final_content_str = "".join(stream_state["final_content"])
    if not final_content_str.strip():
        raise RetryableTranslationFormatError(f"{_JSON_OUTPUT_LABEL} returned empty content.")
    _emit_progress(
        on_progress,
        {
            "phase": "done",
            "translated": stream_state["translated_count"],
            "expected": expected_count,
        },
    )
    return final_content_str.strip()


def _chat_responses(
    messages: list[dict],
    expected_count: int = 0,
    on_progress: Callable[[dict], None] | None = None,
    reasoning_effort: str | None = None,
    on_usage: Callable[[dict], None] | None = None,
) -> str:
    model_name = os.getenv("LLM_MODEL_NAME", LLM_MODEL_NAME).strip()
    if not model_name:
        raise RuntimeError("请先在「翻译设置」中获取并选择翻译模型，再提交任务")

    effective_reasoning_effort = (
        _normalize_reasoning_effort(
            reasoning_effort
            or os.getenv("LLM_REASONING_EFFORT", LLM_REASONING_EFFORT)
        )
    )
    api_format = "responses"
    use_micu_grok_patch = llm_patch.is_micu_grok_responses_request(
        model_name=model_name,
        api_format=api_format,
    )
    if use_micu_grok_patch:
        request = llm_patch.build_micu_grok_responses_request(
            messages=messages,
            model_name=model_name,
            max_tokens=TRANSLATION_MAX_TOKENS,
            reasoning_effort=effective_reasoning_effort,
        )
    else:
        request = {
            "model": model_name,
            "input": _build_responses_input(messages),
            "stream": True,
            "reasoning": {
                "effort": effective_reasoning_effort
            },
            "text": {"format": {"type": "json_object"}},
        }
        if TRANSLATION_MAX_TOKENS > 0:
            request["max_output_tokens"] = TRANSLATION_MAX_TOKENS

    response_stream = (
        llm_patch.create_micu_grok_response_stream(
            request,
            api_key=_required_env("API_KEY"),
            base_url=_required_env("OPENAI_COMPATIBILITY_BASE_URL"),
            api_retries=TRANSLATION_API_RETRIES,
            is_retryable_api_error=_is_retryable_api_error,
            backoff_sleep=_request_backoff_sleep,
        )
        if use_micu_grok_patch
        else _create_response(request)
    )

    completed_response = None
    incomplete_response = None
    failed_error = None
    reasoning_chars = 0
    last_emit = 0.0
    debounce_s = 0.25
    stream_state = {
        "final_content": [],
        "content_chars": 0,
        "translated_count": 0,
        "id_scan_tail": "",
        "id_marker": '"id":',
    }

    def maybe_emit(payload: dict, *, force: bool = False) -> None:
        nonlocal last_emit
        now = time.monotonic()
        if not force and now - last_emit < debounce_s:
            return
        last_emit = now
        _emit_progress(on_progress, payload)

    for event in response_stream:
        event_type = _response_event_type(event)
        if event_type == "response.output_text.delta":
            piece = _response_event_delta(event)
            if piece:
                _emit_stream_content_progress(
                    piece=piece,
                    state=stream_state,
                    expected_count=expected_count,
                    maybe_emit=maybe_emit,
                )
            continue

        if event_type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            reasoning_piece = _response_event_delta(event)
            if reasoning_piece:
                reasoning_chars += len(reasoning_piece)
                maybe_emit(
                    {"phase": "thinking", "reasoning_chars": reasoning_chars}
                )
            continue

        if event_type == "response.completed":
            completed_response = _response_event_response(event)
            _emit_usage(on_usage, _get_nested_value(completed_response, "usage"))
            continue

        if event_type == "response.incomplete":
            incomplete_response = _response_event_response(event)
            continue

        if event_type in {"response.failed", "response.error"}:
            failed_error = event

    if failed_error is not None:
        raise RetryableTranslationFormatError(
            f"OpenAI Responses API failed: {failed_error}"
        )

    if incomplete_response is not None:
        reason = _response_incomplete_reason(incomplete_response)
        if reason == "max_output_tokens":
            raise RetryableTranslationFormatError(
                "OpenAI Responses API response was cut off by max_output_tokens; "
                "increase TRANSLATION_MAX_TOKENS."
            )
        raise RetryableTranslationFormatError(
            f"OpenAI Responses API returned incomplete response: {reason or 'unknown'}"
        )

    final_content_str = "".join(stream_state["final_content"])
    if not final_content_str.strip() and completed_response is not None:
        final_content_str = _extract_response_output_text(completed_response)
    if not final_content_str.strip():
        raise RetryableTranslationFormatError("OpenAI Responses API returned empty content.")

    _emit_progress(
        on_progress,
        {
            "phase": "done",
            "translated": stream_state["translated_count"],
            "expected": expected_count,
        },
    )
    return final_content_str.strip()


def _strip_reasoning_artifacts(raw_output: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", raw_output or "")
    close_tag = "</think>"
    close_idx = cleaned.lower().rfind(close_tag)
    if close_idx != -1:
        cleaned = cleaned[close_idx + len(close_tag) :]
    return cleaned.strip()


def _parse_translation_output(
    raw_output: str,
    expected_count: int,
) -> list[str | None]:
    raw_output = _strip_reasoning_artifacts(raw_output)
    if not raw_output.strip():
        raise RetryableTranslationFormatError(
            f"{_JSON_OUTPUT_LABEL} returned empty content."
        )

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise RetryableTranslationFormatError(
            f"{_JSON_OUTPUT_LABEL} response was not valid JSON."
        ) from exc
    return _extract_translations_from_json(parsed, expected_count)


def _parse_translation_output_by_global_id(
    raw_output: str,
    *,
    expected_ids: list[int],
    total_count: int,
) -> list[str | None]:
    raw_output = _strip_reasoning_artifacts(raw_output)
    if not raw_output.strip():
        raise RetryableTranslationFormatError(
            f"{_JSON_OUTPUT_LABEL} returned empty content."
        )

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise RetryableTranslationFormatError(
            f"{_JSON_OUTPUT_LABEL} response was not valid JSON."
        ) from exc

    if not isinstance(parsed, dict) or not isinstance(parsed.get("translations"), list):
        raise RetryableTranslationFormatError(
            f'{_JSON_OUTPUT_LABEL} response must be {{"translations":[...]}} .'
        )

    expected_id_set = set(expected_ids)
    translations = parsed["translations"]
    if len(translations) != len(expected_ids):
        raise RetryableTranslationFormatError(
            f"{_JSON_OUTPUT_LABEL} returned wrong batch translation count: "
            f"{len(translations)} of {len(expected_ids)}."
        )

    results: list[str | None] = [None] * total_count
    seen_ids: set[int] = set()
    for item in translations:
        if not isinstance(item, dict):
            raise RetryableTranslationFormatError(
                f"{_JSON_OUTPUT_LABEL} translations must contain objects."
            )
        idx = _coerce_int(item.get("id"))
        if idx is None or idx not in expected_id_set or idx >= total_count:
            raise RetryableTranslationFormatError(
                f"{_JSON_OUTPUT_LABEL} returned invalid batch translation id: {item.get('id')!r}."
            )
        if idx in seen_ids:
            raise RetryableTranslationFormatError(
                f"{_JSON_OUTPUT_LABEL} returned duplicate translation id: {idx}."
            )
        seen_ids.add(idx)
        results[idx] = _normalize_translation_text(item.get("text"))

    return results


def _parse_partial_translation_output_by_global_id(
    raw_output: str,
    *,
    expected_ids: list[int],
    total_count: int,
) -> list[str | None]:
    raw_output = _strip_reasoning_artifacts(raw_output)
    if not raw_output.strip():
        raise RetryableTranslationFormatError(
            f"{_JSON_OUTPUT_LABEL} returned empty content."
        )

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise RetryableTranslationFormatError(
            f"{_JSON_OUTPUT_LABEL} response was not valid JSON."
        ) from exc

    if not isinstance(parsed, dict) or not isinstance(parsed.get("translations"), list):
        raise RetryableTranslationFormatError(
            f'{_JSON_OUTPUT_LABEL} response must be {{"translations":[...]}} .'
        )

    expected_id_set = set(expected_ids)
    results: list[str | None] = [None] * total_count
    seen_ids: set[int] = set()
    for item in parsed["translations"]:
        if not isinstance(item, dict):
            continue
        idx = _coerce_int(item.get("id"))
        if idx is None or idx not in expected_id_set or idx >= total_count:
            raise RetryableTranslationFormatError(
                f"{_JSON_OUTPUT_LABEL} returned invalid batch translation id: {item.get('id')!r}."
            )
        if idx in seen_ids:
            raise RetryableTranslationFormatError(
                f"{_JSON_OUTPUT_LABEL} returned duplicate translation id: {idx}."
            )
        normalized = _normalize_translation_text(item.get("text"))
        if normalized is None:
            continue
        seen_ids.add(idx)
        results[idx] = normalized

    return results


def _extract_translations_from_json(data, expected_count: int) -> list[str | None]:
    if not isinstance(data, dict) or not isinstance(data.get("translations"), list):
        raise RetryableTranslationFormatError(
            f'{_JSON_OUTPUT_LABEL} response must be {{"translations":[...]}} .'
        )

    translations = data["translations"]
    if len(translations) != expected_count:
        raise RetryableTranslationFormatError(
            f"{_JSON_OUTPUT_LABEL} returned wrong translation count: "
            f"{len(translations)} of {expected_count}."
        )

    results: list[str | None] = [None] * expected_count
    seen_ids: set[int] = set()
    for item in translations:
        if not isinstance(item, dict):
            raise RetryableTranslationFormatError(
                f"{_JSON_OUTPUT_LABEL} translations must contain objects."
            )

        idx = _coerce_int(item.get("id"))
        if idx is None or idx < 0 or idx >= expected_count:
            raise RetryableTranslationFormatError(
                f"{_JSON_OUTPUT_LABEL} returned invalid translation id: {item.get('id')!r}."
            )
        if idx in seen_ids:
            raise RetryableTranslationFormatError(
                f"{_JSON_OUTPUT_LABEL} returned duplicate translation id: {idx}."
            )

        seen_ids.add(idx)
        results[idx] = _normalize_translation_text(item.get("text"))

    return results


def _coerce_int(value) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _normalize_translation_text(text) -> str | None:
    if text is None:
        return None

    cleaned = str(text).strip()
    if not cleaned:
        return None

    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"^Translation>\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^Original>\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^['\"“”‘’]+|['\"“”‘’]+$", "", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n") if line.strip())
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    cleaned = _LEADING_SPEAKER_RE.sub("", cleaned, count=1)
    if "\n" in cleaned:
        cleaned = cleaned.replace("\n", "\\n")
    return cleaned or None


def _missing_indexes(values: list[str | None]) -> list[int]:
    return [idx for idx, value in enumerate(values) if value is None or value == ""]
