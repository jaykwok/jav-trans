import json
import contextlib
import os
import re
import threading
from typing import Callable

from core.config import DEFAULT_REASONING_EFFORT, REASONING_EFFORTS, load_config
from llm import cache as translation_cache
from llm import engine as engine_module
from llm import global_glossary
from llm import repair as repair_module
from llm import profiles as profiles_module
from llm.profiles import json_v3
from llm.profiles.base import ProfileContext
from llm import settings as llm_settings
from llm import transport_util
from llm.backends import get_backend, selected_backend_name
from llm.backends import openai_compat as openai_transport
from llm.glossary import normalize_glossary_text
from llm import prompt as prompt_module
from llm.errors import (
    ResponseTruncatedError,
    RetryableTranslationFormatError,
    # Re-exported, not used here: callers and tests catch
    # `translator.TranslationCancelledError`, which is the name the cancel path
    # is documented under even though the raise sites live in the transports.
    TranslationCancelledError,  # noqa: F401
)


load_config()

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
# Shared threading.local instance — tests and worker code set attributes on the
# object, never rebind the name, so aliasing keeps one recording scope.
_RETRY_CONTEXT = transport_util._RETRY_CONTEXT
_cache_lock = threading.Lock()
PROMPT_VERSION = prompt_module.PROMPT_VERSION
_LEADING_ROLE_LABEL_RE = prompt_module._LEADING_ROLE_LABEL_RE
_SYSTEM_PROMPT_FULL = prompt_module._SYSTEM_PROMPT_FULL
_SYSTEM_PROMPT_COMPACT = prompt_module._SYSTEM_PROMPT_COMPACT
_JSON_OUTPUT_LABEL = prompt_module._JSON_OUTPUT_LABEL
_normalize_source_text = prompt_module._normalize_source_text

_cancel_requested = transport_util._cancel_requested
_raise_if_cancelled = transport_util._raise_if_cancelled


OPENAI_COMPATIBILITY_BASE_URL = llm_settings.OPENAI_COMPATIBILITY_BASE_URL
API_KEY = llm_settings.API_KEY
LLM_MODEL_NAME = llm_settings.LLM_MODEL_NAME
LLM_REASONING_EFFORT = llm_settings.LLM_REASONING_EFFORT

DEFAULT_TARGET_LANG = llm_settings.DEFAULT_TARGET_LANG

_env_float = llm_settings._env_float
_env_int_clamped = llm_settings._env_int_clamped

TRANSLATION_MAX_TOKENS = llm_settings.TRANSLATION_MAX_TOKENS
TRANSLATION_TEMPERATURE = llm_settings.TRANSLATION_TEMPERATURE
TRANSLATION_TOP_P = llm_settings.TRANSLATION_TOP_P
TRANSLATION_BATCH_SIZE = llm_settings.TRANSLATION_BATCH_SIZE
COMPACT_SYSTEM_PROMPT = llm_settings.COMPACT_SYSTEM_PROMPT
TRANSLATION_API_RETRIES = llm_settings.TRANSLATION_API_RETRIES
TRANSLATION_BATCH_REPAIR_RETRIES = llm_settings.TRANSLATION_BATCH_REPAIR_RETRIES
TRANSLATION_BATCH_MAX_REQUESTS = llm_settings.TRANSLATION_BATCH_MAX_REQUESTS
TRANSLATION_API_BACKOFF_BASE_S = llm_settings.TRANSLATION_API_BACKOFF_BASE_S
TRANSLATION_API_BACKOFF_MAX_S = llm_settings.TRANSLATION_API_BACKOFF_MAX_S
TRANSLATION_PREFIX_WARMUP = llm_settings.TRANSLATION_PREFIX_WARMUP
TRANSLATION_FULL_JSON_PREFIX_MAX_CHARS = (
    llm_settings.TRANSLATION_FULL_JSON_PREFIX_MAX_CHARS
)
TRANSLATION_REPAIR_MAX_IDS = llm_settings.TRANSLATION_REPAIR_MAX_IDS
TRANSLATION_REPAIR_CONTEXT_RADIUS = llm_settings.TRANSLATION_REPAIR_CONTEXT_RADIUS
TRANSLATION_REPAIR_LENGTH_RATIO_MIN = llm_settings.TRANSLATION_REPAIR_LENGTH_RATIO_MIN
TRANSLATION_REPAIR_LENGTH_RATIO_MAX = llm_settings.TRANSLATION_REPAIR_LENGTH_RATIO_MAX

_GLOSSARY_OUTPUT_SCHEMA = global_glossary._GLOSSARY_OUTPUT_SCHEMA

# Distinguishes "caller said nothing" from "caller said no schema". The line
# contract needs the second, and collapsing them would hand Hy-MT2 a JSON
# grammar - the configuration that scored 152/300 untranslated against 6/300
# on its own template.
_SCHEMA_UNSET: dict = {"__schema__": "unset"}


def _translation_output_schema() -> dict:
    """The JSON contract's schema, owned by the profile that defines it.

    This used to be a second literal copy living here, which meant
    `profile.schema` was never actually read by anything on the request path.
    """
    from llm.profiles.json_v3 import TRANSLATION_OUTPUT_SCHEMA

    return json.loads(json.dumps(TRANSLATION_OUTPUT_SCHEMA))


_responses_text_format = openai_transport._responses_text_format


def _normalize_reasoning_effort(
    value: str | None, fallback: str = DEFAULT_REASONING_EFFORT
) -> str:
    return llm_settings._normalize_reasoning_effort(value, fallback)


_normalize_openai_compat_base_url = openai_transport._normalize_openai_compat_base_url
_get_client = openai_transport._get_client


def _current_retry_events() -> list[dict] | None:
    return transport_util._current_retry_events()


_load_translation_cache = translation_cache._load_translation_cache
_translation_cache_jsonl_path = translation_cache._translation_cache_jsonl_path
_read_translation_cache_jsonl = translation_cache._read_translation_cache_jsonl
_save_cache_entry = translation_cache._save_cache_entry
_translation_memory_jsonl_path = translation_cache._translation_memory_jsonl_path
_load_translation_memory = translation_cache._load_translation_memory
_read_translation_memory_jsonl = translation_cache._read_translation_memory_jsonl
_save_memory_entries = translation_cache._save_memory_entries


def _translation_model_identity() -> str:
    backend_name = selected_backend_name()
    return get_backend(backend_name).cache_identity()


def _effective_reasoning_effort(override: str | None = None) -> str:
    """The thinking tier a request will actually carry, for cache keys.

    Resolved the same way `_chat_with_reasoning` resolves it, because a key that
    disagrees with the request is worse than no key. Empty for a backend that has
    no thinking to configure, which keeps the tier out of its signatures
    entirely - local and llamacpp caches are unaffected by this.

    `supports_reasoning` is on `BaseTranslationBackend` but not on the
    `TranslationBackend` protocol, so a duck-typed backend need not have it. Such
    a backend is assumed to think: including the tier in its keys can only cost
    reuse, while omitting it would reinstate the bug this function exists for.
    """
    supports = getattr(get_backend(selected_backend_name()), "supports_reasoning", None)
    if callable(supports) and not supports():
        return ""
    return _normalize_reasoning_effort(
        override or os.getenv("LLM_REASONING_EFFORT", LLM_REASONING_EFFORT)
    )


def _effective_prompt_version() -> str:
    # The active profile produces different text for the same source lines, so
    # profiles must not share cache/memory entries. The profile signature
    # (id@version) is the version string folded into every cache/memory key.
    return profiles_module.select_profile().cache_signature()


def _translation_context_char_limit() -> int:
    """Conservative prompt-context budget before per-batch JSON is added."""
    return TRANSLATION_FULL_JSON_PREFIX_MAX_CHARS


def _compute_prompt_signature(
    extra_glossary: str = "",
    *,
    glossary: str = "",
    target_lang: str = "简体中文",
    character_reference: str = "",
    reasoning_effort: str | None = None,
    prefix_mode: str = "",
) -> str:
    return translation_cache._compute_prompt_signature(
        extra_glossary,
        glossary=glossary,
        target_lang=target_lang,
        character_reference=character_reference,
        prompt_version=_effective_prompt_version(),
        model_name=_translation_model_identity(),
        compact_system_prompt=COMPACT_SYSTEM_PROMPT,
        reasoning_effort=_effective_reasoning_effort(reasoning_effort),
        prefix_mode=prefix_mode,
    )


def _translation_cache_key(
    batch_index: int,
    batch_segments: list[dict],
    *,
    extra_glossary: str = "",
    glossary: str = "",
    target_lang: str = "简体中文",
    character_reference: str = "",
    reasoning_effort: str | None = None,
    prefix_mode: str = "",
) -> str:
    return translation_cache._translation_cache_key(
        batch_index,
        batch_segments,
        extra_glossary=extra_glossary,
        glossary=glossary,
        target_lang=target_lang,
        character_reference=character_reference,
        prompt_version=_effective_prompt_version(),
        model_name=_translation_model_identity(),
        compact_system_prompt=COMPACT_SYSTEM_PROMPT,
        reasoning_effort=_effective_reasoning_effort(reasoning_effort),
        prefix_mode=prefix_mode,
    )


def _translation_memory_key(
    source_text: str,
    extra_glossary: str = "",
    *,
    glossary: str = "",
    target_lang: str = "简体中文",
    character_reference: str = "",
    reasoning_effort: str | None = None,
) -> str:
    return translation_cache._translation_memory_key(
        source_text,
        extra_glossary,
        glossary=glossary,
        target_lang=target_lang,
        character_reference=character_reference,
        prompt_version=_effective_prompt_version(),
        model_name=_translation_model_identity(),
        reasoning_effort=_effective_reasoning_effort(reasoning_effort),
    )


def _translation_memory_source_is_cacheable(source_text: str) -> bool:
    return translation_cache._translation_memory_source_is_cacheable(source_text)


_get_nested_value = transport_util._get_nested_value
_coerce_optional_int = transport_util._coerce_optional_int
_extract_usage_metrics = transport_util._extract_usage_metrics
_emit_usage = transport_util._emit_usage
_merge_usage_metrics = transport_util._merge_usage_metrics


_filter_global_glossary_terms = global_glossary._filter_global_glossary_terms
_format_global_glossary_terms = global_glossary._format_global_glossary_terms
_global_glossary_cache_path_for_texts = global_glossary._global_glossary_cache_path_for_texts


def _glossary_chat_factory():
    # Late-bound module-global _chat so the test seam on this module still
    # intercepts glossary extraction requests.
    def _glossary_chat(messages: list[dict], **chat_kwargs) -> str:
        # Fixed at the bottom tier rather than following the job's: this is one
        # term-extraction request per film whose output feeds the prompt, so
        # thinking about it buys nothing the translation pass will not redo.
        return _chat(messages, reasoning_effort="none", **chat_kwargs)

    return _glossary_chat


def _resolve_translation_extra_glossary(
    segments: list[dict],
    cache_path: str,
    glossary: str,
    *,
    cancel_event: threading.Event | None,
    messages: list[dict] | None = None,
) -> tuple[str, bool]:
    return global_glossary.resolve_extra_glossary(
        segments,
        cache_path,
        glossary,
        chat=_glossary_chat_factory(),
        cancel_event=cancel_event,
        messages=messages,
    )


def extract_global_glossary(
    all_ja_texts: list[str],
    cache_path: str,
    *,
    cancel_event: threading.Event | None = None,
) -> list[dict]:
    return global_glossary.extract_global_glossary(
        all_ja_texts,
        cache_path,
        chat=_glossary_chat_factory(),
        cancel_event=cancel_event,
    )


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
    max_chars: int | None = None,
) -> str:
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
    global_context: str | None = None,
    max_workers: int = 1,
    cache_path: str = "",
    target_lang: str = "简体中文",
    glossary: str = "",
    character_reference: str | None = None,
    reasoning_effort: str | None = None,
    on_batch_done=None,
    on_progress: Callable[[dict], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[list[str], list[dict], list[dict]]:
    _raise_if_cancelled(cancel_event)
    if not segments:
        return [], [], []

    effective_max_workers = max(1, int(max_workers))
    backend_name = selected_backend_name()
    if backend_name == "llamacpp":
        # More client workers than server slots just queue inside llama-server
        # and inflate per-request latency past the watchdog timeouts.
        effective_max_workers = min(
            effective_max_workers,
            _env_int_clamped("LLAMACPP_PARALLEL", 2, 1, 16),
        )
    # Selected before sizing because a model contract may impose a stricter
    # hard cap (Hy-MT2 is deliberately one cue per request).
    profile = profiles_module.select_profile()
    effective_batch_size = _auto_translation_batch_size(len(segments))
    profile_batch_cap = profile.max_batch_size()
    if profile_batch_cap is not None:
        effective_batch_size = min(effective_batch_size, max(1, int(profile_batch_cap)))
    if effective_batch_size > 0:
        effective_max_workers = _auto_translation_workers(
            -(-len(segments) // effective_batch_size),  # ceil
            effective_max_workers,
        )
    effective_cache_path = cache_path or ""
    effective_target_lang = (target_lang or DEFAULT_TARGET_LANG).strip() or DEFAULT_TARGET_LANG
    effective_glossary = normalize_glossary_text(glossary)
    effective_character_reference = (character_reference or "").strip()
    _warn_about_inert_context(
        profile,
        glossary=effective_glossary,
        character_reference=effective_character_reference,
    )
    previous_retry_events = getattr(_RETRY_CONTEXT, "events", None)
    retry_events: list[dict] = []
    _RETRY_CONTEXT.events = retry_events
    try:
        _raise_if_cancelled(cancel_event)
        full_context = (
            global_context
            if global_context is not None
            else generate_global_context(segments)
        )
        context_char_limit = _translation_context_char_limit()
        if context_char_limit > 0 and len(full_context) > context_char_limit:
            full_context = full_context[:context_char_limit]
        full_source_payload = _serialize_segments(segments, compact=True)
        use_full_json_prefix = (
            context_char_limit <= 0
            or len(full_source_payload) <= context_char_limit
        )
        # Ordered after the payload on purpose: when the film is sent as one
        # JSON prefix, the extraction is issued on exactly that prefix, so the
        # request that computes the terms is also the request that warms the
        # provider's cache for every batch behind it.
        extra_glossary_value = ""
        prefix_already_warmed = False
        if profile.wants_extra_glossary:
            extra_glossary_value, prefix_already_warmed = (
                _resolve_translation_extra_glossary(
                    segments,
                    effective_cache_path,
                    effective_glossary,
                    cancel_event=cancel_event,
                    messages=(
                        prompt_module.build_glossary_extraction_messages(
                            full_source_payload=full_source_payload,
                            target_lang=effective_target_lang,
                            glossary=effective_glossary,
                            character_reference=effective_character_reference,
                        )
                        if use_full_json_prefix
                        else None
                    ),
                )
            )

        def _engine_chat(messages: list[dict], **chat_kwargs) -> str:
            # Late global lookup keeps the _chat/_chat_with_reasoning test
            # seams on this module working for engine-driven requests.
            #
            # Defaults, not overrides: the repair pass reissues at an escalated
            # tier and passes its own `reasoning_effort`. Binding the job's here
            # would both collide with that argument and silently undo the
            # escalation for any caller that got past the collision.
            chat_kwargs.setdefault("reasoning_effort", reasoning_effort)
            return _chat_with_reasoning(messages, **chat_kwargs)

        zh_texts, timings, worker_retry_events = engine_module.run_batched(
            segments,
            profile=profile,
            backend_name=backend_name,
            chat=_engine_chat,
            backoff_sleep=_call_request_backoff_sleep,
            crash_probe=_test_crash_translation_batch,
            batch_size=effective_batch_size,
            max_workers=effective_max_workers,
            api_retries=TRANSLATION_API_RETRIES,
            batch_repair_retries=TRANSLATION_BATCH_REPAIR_RETRIES,
            batch_max_requests=TRANSLATION_BATCH_MAX_REQUESTS,
            # The extraction request already sent this exact prefix, so warming
            # it again buys the same cache miss twice.
            prefix_warmup=(
                TRANSLATION_PREFIX_WARMUP
                and not (prefix_already_warmed and use_full_json_prefix)
            ),
            extra_glossary=extra_glossary_value,
            full_context=full_context,
            full_source_payload=full_source_payload,
            use_full_json_prefix=use_full_json_prefix,
            cache_path=effective_cache_path,
            cache_lock=_cache_lock,
            target_lang=effective_target_lang,
            glossary=effective_glossary,
            character_reference=effective_character_reference,
            prompt_version=_effective_prompt_version(),
            model_identity=_translation_model_identity(),
            compact_system_prompt=COMPACT_SYSTEM_PROMPT,
            reasoning_effort=_effective_reasoning_effort(reasoning_effort),
            on_batch_done=on_batch_done,
            on_progress=on_progress,
            cancel_event=cancel_event,
        )
        retry_events.extend(worker_retry_events)
        _raise_if_cancelled(cancel_event)

        def _persist_repaired_translation_cache(repaired_texts: list[str]) -> None:
            # Write repaired texts back under the same batch_key entries the
            # translation phase used, so subsequent runs reuse the repair.
            if not effective_cache_path or not segments:
                return
            # Reads the on-disk extraction written above, so this never issues a
            # request; the boolean is discarded for the same reason.
            extra_glossary, _ = _resolve_translation_extra_glossary(
                segments,
                effective_cache_path,
                effective_glossary,
                cancel_event=cancel_event,
            )
            cache_kwargs = {
                "extra_glossary": extra_glossary,
                "glossary": effective_glossary,
                "target_lang": effective_target_lang,
                "character_reference": effective_character_reference,
                # Same two inputs the engine keyed on, or the repaired text is
                # written under a key nothing ever reads again.
                "reasoning_effort": reasoning_effort,
                "prefix_mode": engine_module.prefix_mode_label(use_full_json_prefix),
            }
            for b_index, b_segments in enumerate(
                _split_into_batches(segments, effective_batch_size)
            ):
                start = b_index * effective_batch_size
                local_texts = [
                    repaired_texts[start + off]
                    if start + off < len(repaired_texts)
                    else ""
                    for off in range(len(b_segments))
                ]
                batch_key = _translation_cache_key(b_index, b_segments, **cache_kwargs)
                _save_cache_entry(
                    effective_cache_path, batch_key, local_texts, _cache_lock
                )

        if profile.wants_repair_pass:
            zh_texts, repair_timing = repair_module.apply_repair_pass(
                segments,
                zh_texts,
                chat=_engine_chat,
                profile=profile,
                batch_size=effective_batch_size,
                reasoning_effort=_effective_reasoning_effort(reasoning_effort),
                target_lang=effective_target_lang,
                glossary=effective_glossary,
                character_reference=effective_character_reference,
                on_progress=on_progress,
                cancel_event=cancel_event,
                cache_writer=_persist_repaired_translation_cache,
            )
            _raise_if_cancelled(cancel_event)
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
    max_tokens: int | None = None,
    on_progress: Callable[[dict], None] | None = None,
    on_usage: Callable[[dict], None] | None = None,
    cancel_event: threading.Event | None = None,
    bounded_response_schema: dict | None = None,
    response_schema: dict | None = _SCHEMA_UNSET,
) -> str:
    _raise_if_cancelled(cancel_event)
    effective_effort = _normalize_reasoning_effort(
        reasoning_effort or os.getenv("LLM_REASONING_EFFORT", LLM_REASONING_EFFORT)
    )
    chat_kwargs = {
        "expected_count": expected_count,
        "on_progress": on_progress,
        "cancel_event": cancel_event,
        "reasoning_effort": effective_effort,
    }
    if max_tokens is not None:
        chat_kwargs["max_tokens"] = max_tokens
    if bounded_response_schema is not None:
        chat_kwargs["bounded_response_schema"] = bounded_response_schema
    if response_schema is not _SCHEMA_UNSET:
        chat_kwargs["response_schema"] = response_schema
    if on_usage is not None:
        chat_kwargs["on_usage"] = on_usage
    try:
        return _chat(
            messages,
            **chat_kwargs,
        )
    except RetryableTranslationFormatError as exc:
        # A stream that dies mid-reasoning is the top tier's characteristic
        # failure - it is the tier that spends longest before emitting anything
        # a parser can use. One retry one tier down is cheaper than the batch
        # repair loop and usually enough; every other tier re-raises, because
        # there the interruption is not about how long the model was thinking.
        top_tier = REASONING_EFFORTS[-1]
        if effective_effort != top_tier or not _is_stream_interrupted_error(exc):
            raise
        fallback_effort = REASONING_EFFORTS[-2]
        _record_api_retry_event(
            exc, 0, 0.0, note=f"fallback_reasoning_effort_{fallback_effort}"
        )
        fallback_kwargs = dict(chat_kwargs)
        fallback_kwargs["reasoning_effort"] = fallback_effort
        return _chat(messages, **fallback_kwargs)


# Repair-pass implementation lives in llm.repair; aliases keep the module's
# public test surface stable.
_select_translation_repair_ids = repair_module._select_translation_repair_ids
_has_source_echo = repair_module._has_source_echo
_has_remaining_japanese_kana = repair_module._has_remaining_japanese_kana
_repair_source_text = repair_module._repair_source_text
_repair_translation_text = repair_module._repair_translation_text
_has_translation_length_mismatch = repair_module._has_translation_length_mismatch
_build_repair_messages = repair_module._build_repair_messages
_build_repair_context_items = repair_module._build_repair_context_items
_public_repair_reasons = repair_module._public_repair_reasons


# The engine's partition, not a second copy of it. The repair pass writes its
# corrected text back under `_translation_cache_key(b_index, ...)`, so it has to
# reproduce exactly the batches the engine translated - two implementations of
# the same five lines is one edit away from writing every repaired translation
# under a key nothing ever reads.
_split_into_batches = engine_module._split_into_batches


def _auto_translation_batch_size(segment_count: int) -> int:
    """How many cues one request can be trusted to return - nothing else.

    Deliberately independent of the worker count. The old rule sized batches as
    `ceil(count / (workers x 2))` to balance the pool, which quietly made
    concurrency a billing control: reasoning is charged per request and barely
    scales with batch size, so raising workers from 4 to 16 on a 1,595-cue film
    turned 8 requests into 32 and roughly quadrupled the thinking bill for the
    same work. Two knobs that each move both cost and parallelism cannot be
    tuned against each other; now `TRANSLATION_BATCH_SIZE` owns cost and
    `max_workers` owns parallelism.

    The remaining question is what a single request can actually finish, and the
    answer is not "as many as fit in the token budget". Measured on sample-v
    over four full runs at 200 cues per request, 7 of 32 requests came back
    incomplete - always a dropped contiguous tail (9, 50, 100, 100 and 184
    missing of 200) or an id outside the requested range. The output budget was
    42,495 tokens against a 31,486-token worst case, so nothing was truncated:
    the model simply stops early, and it does so more often the more items it is
    holding. That failure is expensive in the one currency that matters here,
    because the abandoned request's thinking is charged in full and the reissue
    thinks again from scratch.

    So the size is a reliability constant, and it is the caller's to set.
    """
    count = max(0, int(segment_count))
    if count <= 0:
        return 0
    return min(count, TRANSLATION_BATCH_SIZE)


def _auto_translation_workers(batch_count: int, max_workers: int) -> int:
    """Parallelism follows the batches, instead of the batches following it.

    Spawning more workers than there are batches does not make anything faster;
    it only used to look useful because the batch size was derived from the
    worker count, so asking for more workers manufactured more batches.
    """
    return max(1, min(int(max_workers), max(1, int(batch_count))))


_serialize_segments = prompt_module._serialize_segments


def _build_system_prompt(
    character_reference: str,
    *,
    target_lang: str,
    glossary: str,
    compact: bool = False,
    extra_glossary: str = "",
) -> str:
    return prompt_module._build_system_prompt(
        character_reference,
        target_lang=target_lang,
        glossary=glossary,
        compact=compact,
        extra_glossary=extra_glossary,
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
    effective_character_reference = (character_reference or "").strip()
    system_prompt = _build_system_prompt(
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


def _build_batch_messages(
    batch_segments: list[dict],
    full_segments_summary: str,
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
_build_glossary_extraction_messages = prompt_module.build_glossary_extraction_messages


_is_retryable_api_error = transport_util._is_retryable_api_error
_is_stream_interrupted_error = transport_util._is_stream_interrupted_error
_stream_interrupted_format_error = transport_util._stream_interrupted_format_error
_request_backoff_delay = transport_util._request_backoff_delay
_record_api_retry_event = transport_util._record_api_retry_event
_interruptible_sleep = transport_util._interruptible_sleep
_request_backoff_sleep = transport_util._request_backoff_sleep


def _call_request_backoff_sleep(
    attempt: int,
    exc: Exception,
    *,
    cancel_event: threading.Event | None = None,
) -> None:
    # Reads the translator-global _request_backoff_sleep so tests that
    # monkeypatch it keep intercepting the batched path during migration.
    if cancel_event is None:
        _request_backoff_sleep(attempt, exc)
        return
    try:
        _request_backoff_sleep(attempt, exc, cancel_event=cancel_event)
    except TypeError as type_error:
        if "cancel_event" not in str(type_error):
            raise
        _request_backoff_sleep(attempt, exc)


_emit_progress = transport_util._emit_progress


_make_aggregated_progress_callback = engine_module._make_aggregated_progress_callback


def _warn_about_inert_context(profile, *, glossary: str, character_reference: str) -> None:
    """Say so when the chosen contract cannot carry a setting the user filled in.

    Accepting a glossary and silently ignoring it is the "配置项写了没人读"
    failure this repo hunts elsewhere; the local per-line contract genuinely
    cannot use one (every context layer measured worse on Hy-MT2), but the user
    typed it into the UI and is entitled to know where it went.
    """
    reporter = getattr(profile, "warn_about_inert_context", None)
    if reporter is None:
        return
    inert = reporter(
        ProfileContext(glossary=glossary, character_reference=character_reference)
    )
    if inert:
        print(
            f"[translation] 本地逐句翻译不使用：{'、'.join(inert)}"
            "（这些设置只在 OpenAI 兼容 API 后端生效）",
            flush=True,
        )


def _chat(
    messages: list[dict],
    expected_count: int = 0,
    on_progress: Callable[[dict], None] | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
    on_usage: Callable[[dict], None] | None = None,
    cancel_event: threading.Event | None = None,
    response_schema: dict | None = _SCHEMA_UNSET,
    response_schema_name: str = "subtitle_translations",
    bounded_response_schema: dict | None = None,
) -> str:
    _raise_if_cancelled(cancel_event)
    if response_schema is _SCHEMA_UNSET:
        response_schema = _translation_output_schema()
    # The configured value is a ceiling for API models; a caller-supplied budget
    # is an upper bound on what this particular reply can legitimately need, and
    # is what stops a local model looping to the ceiling.
    effective_max_tokens = min(
        TRANSLATION_MAX_TOKENS, max_tokens if max_tokens is not None else TRANSLATION_MAX_TOKENS
    )

    def _dispatch(budget: int) -> str:
        backend_name = selected_backend_name()
        if backend_name != "openai":
            backend = get_backend(backend_name)
            return backend.chat_completion(
                messages,
                temperature=TRANSLATION_TEMPERATURE,
                top_p=TRANSLATION_TOP_P,
                max_tokens=budget,
                # Local-only, and deliberately not passed to the OpenAI
                # transports below: their strict structured-output mode
                # validates against a fixed keyword allowlist that has no
                # `maxLength` for strings, so sending it is a 400 at request
                # time rather than a tighter grammar. Nothing is lost - the
                # repetition loop this bounds was only ever measured on local
                # GGUF models, never on an API one.
                response_format=bounded_response_schema or response_schema,
                reasoning_effort=reasoning_effort,
                expected_count=expected_count,
                cancel_event=cancel_event,
                on_progress=on_progress,
                on_usage=on_usage,
            )
        return _chat_responses(
            messages,
            expected_count=expected_count,
            on_progress=on_progress,
            reasoning_effort=reasoning_effort,
            on_usage=on_usage,
            cancel_event=cancel_event,
            response_schema=response_schema,
            response_schema_name=response_schema_name,
            max_tokens=budget,
        )

    try:
        return _dispatch(effective_max_tokens)
    except ResponseTruncatedError as truncated:
        # The budget is an arithmetic bound on a legitimate translation, so
        # hitting it means either the bound was too tight for this batch or the
        # model is looping - and nothing here can tell which. One escalation
        # settles the first case; the second costs one extra request and fails
        # anyway, which is what a terminal failure did immediately while also
        # discarding every other batch of the film.
        retry_budget = min(
            TRANSLATION_MAX_TOKENS,
            int(truncated.limit * llm_settings.TRANSLATION_TRUNCATION_RETRY_FACTOR),
        )
        if retry_budget <= truncated.limit:
            raise
        _emit_progress(
            on_progress,
            {
                "phase": "output_truncated",
                "diagnostic": True,
                "limit": truncated.limit,
                "retry_limit": retry_budget,
                "expected": expected_count,
            },
        )
        try:
            return _dispatch(retry_budget)
        except ResponseTruncatedError as again:
            raise ResponseTruncatedError(
                f"{again}. Retried once at {retry_budget} tokens and it was cut "
                f"off again, so this looks like a runaway reply rather than a "
                f"tight budget. The budget comes from "
                f"TRANSLATION_OUTPUT_CHAR_RATIO (per request), not "
                f"TRANSLATION_MAX_TOKENS.",
                limit=retry_budget,
            ) from again


_chat_responses = openai_transport._chat_responses


# JSON-contract parsing lives in llm.profiles.json_v3; these aliases keep the
# repair pass and the module's public test surface on the same implementations.
_strip_reasoning_artifacts = json_v3._strip_reasoning_artifacts
_parse_translation_output = json_v3._parse_translation_output
_parse_translation_output_by_global_id = json_v3._parse_translation_output_by_global_id
_parse_partial_translation_output_by_global_id = json_v3._parse_partial_translation_output_by_global_id
_extract_translations_from_json = json_v3._extract_translations_from_json
_coerce_int = json_v3._coerce_int
_normalize_translation_text = json_v3._normalize_translation_text
_missing_indexes = json_v3._missing_indexes
