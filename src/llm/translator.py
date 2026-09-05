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
from llm import max_tokens_limits
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
from utils import hf_progress
from llm.errors import (
    MaxTokensRejectedError,
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


_format_global_glossary_terms = global_glossary._format_global_glossary_terms
_global_glossary_cache_path_for_texts = global_glossary._global_glossary_cache_path_for_texts
resolve_settled_glossary = global_glossary.resolve_settled_glossary


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
            _env_int_clamped("LLAMACPP_PARALLEL", 8, 1, 16),
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

        job_id_for_worker_threads = hf_progress.current_job_id()

        def _engine_chat(messages: list[dict], **chat_kwargs) -> str:
            # run_batched dispatches this from a ThreadPoolExecutor pool, and a
            # fresh worker thread has no `core.events` thread-local of its own
            # -- a GGUF download triggered from here (llamacpp backend, first
            # call starts the server) would otherwise emit model_download
            # events with an empty job_id, which the frontend silently drops
            # instead of showing a progress bar.
            hf_progress.propagate_job_id_to_current_thread(job_id_for_worker_threads)

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
            prefix_warmup=TRANSLATION_PREFIX_WARMUP,
            # No settled rendering exists before the base pass has translated
            # anything - see `global_glossary`. The repair pass gets one,
            # derived from this pass's own output.
            extra_glossary="",
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
            cache_kwargs = {
                # The base pass always ran with no settled rendering (there was
                # none to give it yet), so its cache keys carry "" here too -
                # this has to match or the repair is written under a key the
                # base pass's own entries never used.
                "extra_glossary": "",
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
            # Derived from this pass's own output, so it only exists once
            # there is something to measure - see `global_glossary`.
            settled_glossary_value = resolve_settled_glossary(
                segments, zh_texts, effective_cache_path, effective_glossary
            )
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
                extra_glossary=settled_glossary_value,
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


# Below this a budget is too small to translate a batch with, so a further
# halving would trade one useless request for another.
_MIN_MAX_TOKENS_RETRY = 4096


def _refusal_note(exc: BaseException, limit: int = 240) -> str:
    """The provider's own words, on one line and trimmed.

    The ladder used to log only its own numbers, so a refusal that was then
    retried successfully left no record anywhere of *what the endpoint said*.
    The classifier deciding whether those words are worth learning from is built
    out of a handful of observed messages, and the ones it cannot place are the
    ones worth collecting - which needs them in the job log, not only in the
    exception of a batch that happened to fail.
    """
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else f"{text[:limit]}…"


def _usable_retry_budget(candidate: int, *, below: int, above: int) -> bool:
    """Whether `candidate` is worth spending a request on.

    `below` is the value just refused: a retry has to be smaller or it collects
    the same refusal. `above` is the largest budget a truncated reply has
    already outgrown during this call, and it is the constraint that costs money
    to get wrong - at or under it the reply is cut off again, and unlike a
    refusal that request *generates*, so it is billed in full for an answer
    already known to be unusable.
    """
    return above < candidate < below and candidate >= _MIN_MAX_TOKENS_RETRY


def _endpoint_identity() -> tuple[str, str] | None:
    """`(base_url, model)` for the API backend, or None when it is not in use.

    Local backends never refuse a `max_tokens`, and keying a learned limit on
    their empty base URL would let one endpoint's ceiling answer for another.
    """
    if selected_backend_name() != "openai":
        return None
    model = os.getenv("LLM_MODEL_NAME", llm_settings.LLM_MODEL_NAME).strip()
    if not model:
        return None
    return os.getenv("OPENAI_COMPATIBILITY_BASE_URL", "").strip(), model


_clamp_warned: set[str] = set()


def _resolved_max_tokens(desired: int | None, limits) -> int:
    """The budget the caller wants, with "no preference" turned into a number.

    `desired=None` is the prefix warmup and anything else with no arithmetic
    bound of its own: it means "as much as this endpoint allows", which is the
    named ceiling when there is one and the configured fallback only otherwise.
    That is the *whole* job of the fallback - it stands in for a missing budget,
    never for a ceiling, so an explicit budget is never trimmed to it.
    """
    if desired is not None:
        return max(1, int(desired))
    return int(getattr(limits, "exact_ceiling", None) or TRANSLATION_MAX_TOKENS)


def _local_max_tokens_budget(desired: int | None) -> int:
    """The no-endpoint case, where the configured value really is a ceiling.

    A local model cannot refuse a `max_tokens`, so there is nothing to learn
    from and nothing to fall back to - `TRANSLATION_MAX_TOKENS` is the runaway
    backstop it has always been, and a caller-supplied budget may only lower it.
    """
    return min(_resolved_max_tokens(desired, None), TRANSLATION_MAX_TOKENS)


class _EndpointCapability:
    """What one call knows about its endpoint's `max_tokens` ceiling.

    The live copy is this object; the cache file is where it is left for the
    next call. That order is the point. Reading the state back from disk after
    every refusal made the probe ladder depend on the cache being writable, and
    a cache that is not - a read-only directory, a full disk - reads back as
    "nothing known", where the next step is `budget_for(nothing, sent - 1)`,
    i.e. `sent - 1`. The ladder then walks down one token per round trip and the
    request fails at a budget the endpoint would have taken two halvings later.
    A capability cache is an optimisation; a request must not need it to work.
    """

    def __init__(self, identity: tuple[str, str] | None) -> None:
        self.identity = identity
        self.limits = (
            max_tokens_limits.load_limits(*identity)
            if identity is not None
            else max_tokens_limits.EndpointLimits()
        )
        # Ceiling-side observations wait here until this call generates at a
        # smaller budget. See `_corroborate`.
        self._staged: list[tuple[str, int]] = []
        self._corroborated = False

    def _stage(self, kind: str, value: int) -> None:
        """Hold a refusal, or write it if this call has already been convinced."""
        if self.identity is None:
            return
        if self._corroborated:
            self._write(kind, value)
        else:
            self._staged.append((kind, value))

    def _write(self, kind: str, value: int) -> None:
        if kind == "exact_ceiling":
            max_tokens_limits.record_exact_ceiling(*self.identity, value)
        else:
            max_tokens_limits.record_rejection(*self.identity, value)

    def _corroborate(self) -> None:
        """A budget under the refusal just generated, so the refusal can be kept.

        Nothing expires any more, so the cache has no way to walk back a number
        that should not have gone in. The check that replaces it is behavioural:
        refuse high, generate low is what a real ceiling looks like, and a
        message whose wording was merely read as a ceiling will not produce it.
        Until that happens the refusal steers this call and nothing else.

        The merge keeps every floor strictly under `rejected_at` and at or under
        `exact_ceiling`, so any recorded success is by construction the one this
        is asking about - `131072` accepted against a named `131072` corroborates
        it exactly as much as `32768` accepted under a refused `65536` does.
        """
        if self._corroborated or not (
            self.limits.rejected_at or self.limits.exact_ceiling
        ):
            return
        self._corroborated = True
        for kind, value in self._staged:
            self._write(kind, value)
        self._staged.clear()

    def _observe(
        self,
        *,
        exact_ceiling: int | None = None,
        rejection: int | None = None,
        success: int | None = None,
    ) -> None:
        self.limits = max_tokens_limits.merge_observation(
            self.limits,
            exact_ceiling=exact_ceiling,
            rejection=rejection,
            success=success,
        )

    def record_exact_ceiling(self, ceiling: int, *, persist: bool = True) -> None:
        """The endpoint named its ceiling. The one kind that clamps."""
        if ceiling <= 0:
            return
        self._observe(exact_ceiling=int(ceiling))
        if persist:
            self._stage("exact_ceiling", int(ceiling))

    def record_rejection(self, sent: int, *, persist: bool = True) -> None:
        """`sent` was refused, so everything at or above it is out.

        `persist=False` keeps a refusal that is about this request rather than
        about the endpoint - a combined input+output limit - inside this call.
        It still drives the bisection, because a smaller budget really does fit;
        it just never becomes a fact about an endpoint that will see shorter
        prompts than this one.
        """
        if sent <= 0:
            return
        self._observe(rejection=int(sent))
        if persist:
            self._stage("rejection", int(sent))

    def record_success(self, accepted: int, *, persist: bool = True) -> None:
        """`accepted` went out and was generated against. A lower bound.

        `persist=False` is the first-try case: the endpoint took the number, so
        this call now knows a floor, but writing "at least this much works" on
        every batch would be a disk write per request to store something no
        refusal has bounded. The memory half is never optional - it is what the
        rest of this call bisects against.
        """
        if accepted <= 0:
            return
        self._observe(success=int(accepted))
        # Before the floor itself: whatever was refused above it is what makes
        # this number worth having on disk at all.
        self._corroborate()
        if persist and self.identity is not None:
            max_tokens_limits.record_success(*self.identity, int(accepted))

    def budget(self, desired: int | None, *, warn: bool = False) -> int:
        """What to actually ask for, given what this endpoint has said so far."""
        if self.identity is None:
            return _local_max_tokens_budget(desired)
        resolved = _resolved_max_tokens(desired, self.limits)
        budget = max_tokens_limits.budget_for(self.limits, resolved)
        if warn and budget < resolved and desired is not None:
            self._warn_clamped(resolved, budget)
        return budget

    def _warn_clamped(self, resolved: int, budget: int) -> None:
        """Shrinking a computed budget is the silent half of this: the request
        simply goes out with less room, and if the reply is then cut off the
        truncation escalation cannot raise it back past the same line. Said once
        per endpoint per process - a per-batch line would be noise."""
        key = max_tokens_limits.endpoint_key(*self.identity)
        if key in _clamp_warned:
            return
        _clamp_warned.add(key)
        source = (
            f"端点上限 {self.limits.exact_ceiling}"
            if self.limits.exact_ceiling
            else (
                f"端点已拒绝过 {self.limits.rejected_at}"
                if self.limits.rejected_at
                else f"TRANSLATION_MAX_TOKENS={TRANSLATION_MAX_TOKENS}"
            )
        )
        print(
            f"[WARN] 本次翻译预算 {resolved} 被压到 {budget}（{source}）。"
            "回复被切断时将无法再向上重试；如需更大预算请调高 "
            "TRANSLATION_MAX_TOKENS 或调小批次/推理配额。",
            flush=True,
        )

    def next_probe_after(self, sent: int) -> int:
        """The next value to try after `sent` was refused.

        Read off the bracket this call has already narrowed, so the step is the
        midpoint of what is still possible. Only an endpoint that has said
        nothing at all - a local backend - falls back to halving, which is the
        same thing with no information.
        """
        if not self.limits.known_anything:
            return sent // 2
        candidate = max_tokens_limits.budget_for(self.limits, sent - 1)
        return candidate if 0 < candidate < sent else sent // 2


def _plain_max_tokens_budget(desired: int | None) -> int:
    """`_max_tokens_budget` without the warning. Safe to call for a preview."""
    return _EndpointCapability(_endpoint_identity()).budget(desired)


def _max_tokens_budget(desired: int | None) -> int:
    """What to actually ask for, and say so the first time it is less.

    Reads the cache, so this answers for a request that has not started yet.
    Within one `_chat` the same question goes to that call's own capability
    state instead, which knows what this one cannot: what the endpoint has said
    in the last few seconds.
    """
    return _EndpointCapability(_endpoint_identity()).budget(desired, warn=True)


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
    # A caller-supplied budget is an upper bound on what this particular reply
    # can legitimately need, and is what stops a local model looping. What the
    # endpoint will accept of it is a separate question, answered by whatever it
    # has already refused - and by `TRANSLATION_MAX_TOKENS` only until it does.
    # One state for the whole call, so the escalation below sizes itself from
    # what the requests above it just learned rather than from the cache.
    capability = _EndpointCapability(_endpoint_identity())
    effective_max_tokens = capability.budget(max_tokens, warn=True)

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

    # Retry state for the whole call, not for one trip down the ladder. The
    # truncation escalation runs the ladder a second time, and while these lived
    # inside it that second run started fresh: a full adjustment quota again,
    # another known-good fallback, and - because the ladder only ever walks
    # *down* - a licence to spend them arriving back at the budget that had just
    # been truncated. That last request generates, and generates the same
    # cut-off reply, so the escalation paid twice to learn nothing.
    #
    # Counted as sends, not as retries: the decrement happens before the check
    # and the request that opened the ladder is already inside it, so 3 buys two
    # further adjustments and the third refusal goes to the fallback. One call
    # therefore sends at most five times at this layer - the first request, two
    # adjustments, one known-good, one escalation.
    probes_left = 3
    fell_back = False
    truncated_at = 0

    def _dispatch_learning_ceiling(budget: int) -> str:
        """Send `budget`; if the endpoint refuses the *number*, find one it takes.

        Nothing was generated when a refusal fires - the request was rejected
        before it ran - so those retries cost round trips and no output tokens.
        (Whether a provider bills for a rejected request is the provider's
        business; not receiving tokens locally is not proof that nobody
        charged.) Two halvings is the floor of that: 65536 -> 32768 -> 16384
        covers the whole range of ceilings actually published for these models,
        and a third would be sending a budget too small to translate with anyway.

        The one request past that ladder is not a probe and is not free: it is a
        full generation at a budget this endpoint has already answered at. It
        exists because a capability question must never be the reason a request
        that could have been completed fails instead - but it is spent once per
        call, and never at a size a truncated reply has already outgrown.
        """
        nonlocal probes_left, fell_back, truncated_at
        # This request is already a bisection step if the endpoint has a bracket
        # and has never named its ceiling. Whatever it does is then news, win or
        # lose, and the bracket has to move either way or the same midpoint gets
        # probed on every request forever.
        probing = bool(capability.limits.rejected_at) and not capability.limits.exact_ceiling
        sent = budget
        refused = False

        def _learn_accepted(value: int) -> None:
            """A lower bound, recorded as one. Together with the rejections it
            brackets the real ceiling, so the next request that wants more
            probes a *new* midpoint. Recording only after a refusal was the bug:
            a midpoint that succeeded first try left the bracket untouched and
            every later request re-probed the same number.

            The two halves are gated differently, which is the thing that was
            wrong here twice. Memory always: an unknown endpoint that truncates
            at 40000 and then refuses 80000 has told this call that 40000 works,
            and without it the bisection starts from zero, computes 40000 again
            and is - correctly - blocked for being a budget already outgrown, so
            a request that 60000 would have finished fails instead. Disk only
            once something has been refused, because writing "at least this much
            works" on every first-try success is a disk write per batch to store
            what no refusal has bounded."""
            capability.record_success(value, persist=refused or probing)

        while True:
            try:
                answer = _dispatch(sent)
            except MaxTokensRejectedError as rejected:
                refused = True
                named = rejected.limit
                # A refusal that is about this prompt's size rather than the
                # endpoint's still steers this call; it just is not written down.
                learnable = getattr(rejected, "learnable", True)
                if named is not None and named < sent:
                    capability.record_exact_ceiling(named, persist=learnable)
                else:
                    capability.record_rejection(sent, persist=learnable)
                probes_left -= 1
                next_budget: int | None = None
                source = ""
                if probes_left > 0:
                    if named is not None and named < sent:
                        next_budget = named
                        source = "parsed"
                    else:
                        # Recomputed from the bracket this refusal just
                        # narrowed, not halved off the refused value: halving
                        # from a midpoint probe would drop below a budget
                        # already known to work.
                        next_budget = capability.next_probe_after(sent)
                        source = "probed"
                    if not _usable_retry_budget(
                        next_budget, below=sent, above=truncated_at
                    ):
                        next_budget = None
                if next_budget is None and not fell_back:
                    # Out of probes, or the next probe is not a usable budget.
                    # A value already known good is neither, and failing while
                    # holding one fails a translation the endpoint would have
                    # done: with a real ceiling of 33000 the ladder spends
                    # 49152, 40960 and 36864 on the bracket and never tries the
                    # 32768 sitting in `known_good`.
                    known_good = capability.limits.known_good or 0
                    if _usable_retry_budget(
                        known_good, below=sent, above=truncated_at
                    ):
                        next_budget = known_good
                        source = "known-good"
                        fell_back = True
                if next_budget is None:
                    raise
                _emit_progress(
                    on_progress,
                    {
                        "phase": "max_tokens_rejected",
                        "diagnostic": True,
                        "sent": sent,
                        "retry_max_tokens": next_budget,
                        "source": source,
                        "learned": learnable,
                    },
                )
                print(
                    f"[WARN] endpoint rejected max_tokens={sent}, retrying at "
                    f"{next_budget} ({source}{'' if learnable else ', not learned'})"
                    f": {_refusal_note(rejected)}",
                    flush=True,
                )
                sent = next_budget
                continue
            except ResponseTruncatedError:
                # Generating until the budget ran out is the endpoint accepting
                # the number - the reply died on its own bound, not on the
                # parameter. The escalation below is the caller that has to know
                # that: without it the bracket never moves, the retry recomputes
                # the identical midpoint, and the `retry_budget <= limit` guard
                # then fails a batch that had room left above it. It is also the
                # floor every later budget in this call has to clear.
                truncated_at = max(truncated_at, sent)
                _learn_accepted(sent)
                raise
            _learn_accepted(sent)
            return answer

    try:
        return _dispatch_learning_ceiling(effective_max_tokens)
    except ResponseTruncatedError as truncated:
        # The budget is an arithmetic bound on a legitimate translation, so
        # hitting it means either the bound was too tight for this batch or the
        # model is looping - and nothing here can tell which. One escalation
        # settles the first case; the second costs one extra request and fails
        # anyway, which is what a terminal failure did immediately while also
        # discarding every other batch of the film.
        # Through the same capability state as the first attempt: the escalation
        # is exactly the case that wants more room, so it must be free to ask
        # for it wherever the endpoint has not actually said no.
        retry_budget = capability.budget(
            int(truncated.limit * llm_settings.TRANSLATION_TRUNCATION_RETRY_FACTOR),
            warn=True,
        )
        if retry_budget <= truncated_at:
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
            return _dispatch_learning_ceiling(retry_budget)
        except ResponseTruncatedError as again:
            # `again.limit` is what the retry actually went out with, which is
            # `retry_budget` only when the ladder did not have to step down from
            # it. Reporting the budget that was *aimed* for names a number no
            # request ever carried.
            raise ResponseTruncatedError(
                f"{again}. Retried once at {again.limit} tokens and it was cut "
                f"off again, so this looks like a runaway reply rather than a "
                f"tight budget. The budget comes from "
                f"TRANSLATION_OUTPUT_CHAR_RATIO (per request), not "
                f"TRANSLATION_MAX_TOKENS.",
                limit=again.limit,
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
