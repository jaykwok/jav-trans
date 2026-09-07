import json
import contextlib
import os
import re
import threading
from typing import Callable

from core.config import DEFAULT_REASONING_EFFORT, REASONING_EFFORTS, load_config
from core.typed_config import env_int
from llm import cache as translation_cache
from llm import engine as engine_module
from llm import global_glossary
from llm import repair as repair_module
from llm import profiles as profiles_module
from llm import request_config
from llm.run_context import RunContext
from llm.session import TranslationSession
from llm.profiles import json_v3
from llm.profiles.base import ProfileContext
from llm.context import SourceContext
from llm import settings as llm_settings
from llm import token_budget
from llm import transport_util
from llm import backends as backends_module
from llm.backends import (
    get_backend,
    selected_backend_name,
    task_backend,
    task_backend_name,
)
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
    # The instance this task leased, not "whatever answers to the selected name
    # right now": a cache identity computed from a different instance than the
    # one that will answer the request is a key that does not describe its value.
    return task_backend().cache_identity()


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
    supports = getattr(task_backend(), "supports_reasoning", None)
    if callable(supports) and not supports():
        return ""
    return _normalize_reasoning_effort(
        override or request_config.getenv("LLM_REASONING_EFFORT", LLM_REASONING_EFFORT)
    )


def _effective_prompt_version() -> str:
    # The active profile produces different text for the same source lines, so
    # profiles must not share cache/memory entries. The profile signature
    # (id@version) is the version string folded into every cache/memory key.
    #
    # The task's own profile when there is one, rather than a fresh selection.
    # Selecting again here is how a cache key came to name a *different*
    # contract than the one that produced the text it keys: detection reads
    # configuration the settings panel can rewrite mid-run, so the second
    # selection is not required to agree with the first.
    profile = request_config.current_profile()
    if profile is not None:
        return profile.cache_signature()
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
    context_signature: str = "",
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
        context_signature=context_signature,
    )


def _translation_memory_key(
    source_text: str,
    extra_glossary: str = "",
    *,
    glossary: str = "",
    target_lang: str = "简体中文",
    character_reference: str = "",
    reasoning_effort: str | None = None,
    context_signature: str = "",
    occurrence_id: int | None = None,
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
        context_signature=context_signature,
        occurrence_id=occurrence_id,
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
    backend_name = task_backend_name()
    if backend_name == "llamacpp":
        # More client workers than server slots just queue inside llama-server
        # and inflate per-request latency past the watchdog timeouts.
        effective_max_workers = min(
            effective_max_workers,
            env_int("LLAMACPP_PARALLEL", 8, minimum=1, maximum=16),
        )
    # Selected before sizing because a model contract may impose a stricter
    # hard cap (Hy-MT2 is deliberately one cue per request), and selected
    # exactly once - from the leased backend and the task's frozen endpoint
    # snapshot, not from live configuration. Both halves matter: a second
    # selection later disagreed with this one, and either selection reading the
    # settings panel disagreed with the model the requests actually went to.
    profile = profiles_module.select_profile(
        backend=backend_name,
        model_name=request_config.getenv("LLM_MODEL_NAME", ""),
    )
    effective_batch_size = _auto_translation_batch_size(len(segments))
    profile_batch_cap = profile.max_batch_size()
    if profile_batch_cap is not None:
        effective_batch_size = min(effective_batch_size, max(1, int(profile_batch_cap)))
    if effective_batch_size > 0:
        effective_max_workers = _auto_translation_workers(
            len(_split_into_batches(segments, effective_batch_size)),
            effective_max_workers,
        )
    effective_cache_path = cache_path or ""
    effective_target_lang = (target_lang or DEFAULT_TARGET_LANG).strip() or DEFAULT_TARGET_LANG
    effective_glossary = normalize_glossary_text(glossary)
    effective_character_reference = (character_reference or "").strip()
    # Everything both passes have to agree about, settled once. The repair pass
    # takes this object rather than eight parallel arguments it could be handed
    # inconsistently - see `llm.session`.
    session = TranslationSession(
        backend_name=backend_name,
        profile=profile,
        batch_size=effective_batch_size,
        max_workers=effective_max_workers,
        cache_path=effective_cache_path,
        target_lang=effective_target_lang,
        glossary=effective_glossary,
        character_reference=effective_character_reference,
        reasoning_effort=_effective_reasoning_effort(reasoning_effort),
    )
    _warn_about_inert_context(
        profile,
        glossary=effective_glossary,
        character_reference=effective_character_reference,
    )
    previous_retry_events = getattr(_RETRY_CONTEXT, "events", None)
    retry_events: list[dict] = []
    _RETRY_CONTEXT.events = retry_events
    # `session.bound()` binds the profile for both passes: the cache-key helpers
    # are reached from places that never see an argument, and the repair pass
    # has to write its corrected text back under the key the batch pass used.
    try:
        with session.bound():
            _raise_if_cancelled(cancel_event)
            full_context = (
                global_context
                if global_context is not None
                else generate_global_context(segments)
            )
            external_context = (
                full_context if global_context is not None and full_context != generate_global_context(segments) else ""
            )
            source_context = SourceContext.build(segments, external_context=external_context)
            model_identity = _translation_model_identity()
            context_signature = engine_module.cache_scope(
                source_context, profile, model_identity, backend_name=backend_name
            )
            context_char_limit = _translation_context_char_limit()
            if context_char_limit > 0 and len(full_context) > context_char_limit:
                full_context = full_context[:context_char_limit]
            full_source_payload = _serialize_segments(segments, compact=True)
            use_full_json_prefix = (
                context_char_limit <= 0
                or len(full_source_payload) <= context_char_limit
            )

            # Captured on the thread that holds the run, adopted on whichever pool
            # thread ends up making the request. See `llm.run_context` for what each
            # of these decides and what goes wrong when one of them is left behind.
            run_context = RunContext.capture()

            def _engine_chat(messages: list[dict], **chat_kwargs) -> str:
                # run_batched dispatches this from a ThreadPoolExecutor pool, and a
                # fresh worker thread holds none of the run's state.
                run_context.adopt()

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
                # Recurring translations are observed after the first pass,
                # never promoted to mandatory terminology.
                extra_glossary="",
                full_context=full_context,
                full_source_payload=full_source_payload,
                use_full_json_prefix=use_full_json_prefix,
                cache_path=effective_cache_path,
                cache_lock=_cache_lock,
                target_lang=effective_target_lang,
                glossary=effective_glossary,
                character_reference=effective_character_reference,
                # This profile's signature, not a fresh selection: the key has to
                # name the contract that produced the text it keys.
                prompt_version=profile.cache_signature(),
                model_identity=model_identity,
                compact_system_prompt=COMPACT_SYSTEM_PROMPT,
                reasoning_effort=_effective_reasoning_effort(reasoning_effort),
                on_batch_done=on_batch_done,
                on_progress=on_progress,
                cancel_event=cancel_event,
                source_context=source_context,
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
                    "context_signature": context_signature,
                }
                start = 0
                memory_entries: list[tuple[str, str]] = []
                for b_index, b_segments in enumerate(
                    _split_into_batches(segments, effective_batch_size)
                ):
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
                    for offset, (segment, text) in enumerate(zip(b_segments, local_texts)):
                        source = str(segment.get("text", ""))
                        if text and _translation_memory_source_is_cacheable(source):
                            key = _translation_memory_key(
                                source, glossary=effective_glossary,
                                target_lang=effective_target_lang,
                                character_reference=effective_character_reference,
                                reasoning_effort=reasoning_effort,
                                context_signature=context_signature,
                                occurrence_id=start + offset,
                            )
                            memory_entries.append((key, text))
                    start += len(b_segments)
                if memory_entries:
                    _save_memory_entries(effective_cache_path, memory_entries, _cache_lock)

            if profile.wants_repair_pass:
                # Derived from this pass's own output, so it only exists once
                # there is something to measure - see `global_glossary`.
                # Record recurrence for inspection, never as a translation rule.
                resolve_settled_glossary(
                    segments, zh_texts, effective_cache_path, effective_glossary
                )
                zh_texts, repair_timing = repair_module.apply_repair_pass(
                    segments,
                    zh_texts,
                    chat=_engine_chat,
                    session=session,
                    extra_glossary="",
                    on_progress=on_progress,
                    cancel_event=cancel_event,
                    cache_writer=_persist_repaired_translation_cache,
                )
                _raise_if_cancelled(cancel_event)
                if repair_timing is not None:
                    timings.append(repair_timing)
                from llm.semantic_review import apply_semantic_review

                zh_texts, review_timing = apply_semantic_review(
                    segments, zh_texts, chat=_engine_chat, session=session,
                    source_context=source_context, model_identity=model_identity,
                    on_progress=on_progress, cancel_event=cancel_event,
                    cache_writer=_persist_repaired_translation_cache,
                )
                if review_timing is not None:
                    timings.append(review_timing)
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
        reasoning_effort
        or request_config.getenv("LLM_REASONING_EFFORT", LLM_REASONING_EFFORT)
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

    The local contract supports matching user terms and nearby source context.
    Character references still cannot be carried by its native template.
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


def _endpoint_capability() -> token_budget.EndpointCapability:
    """This task's endpoint, as `llm.token_budget` sees it.

    The one thing the budget module cannot work out for itself: which backend
    the *lease* points at. Re-reading the selected name instead would let a task
    holding a claim on one instance file what it learns under another.
    """
    return token_budget.EndpointCapability(
        token_budget.endpoint_identity(task_backend_name())
    )


def _max_tokens_budget(desired: int | None) -> int:
    """What to actually ask for, and say so the first time it is less."""
    return _endpoint_capability().budget(desired, warn=True)


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
    capability = _endpoint_capability()
    effective_max_tokens = capability.budget(max_tokens, warn=True)

    def _dispatch(budget: int) -> str:
        # Resolved through the lease: re-reading the selected name here meant a
        # task could hold a claim on one instance and send its requests to
        # another, which the holder of *that* one was then free to close.
        if task_backend_name() != "openai":
            backend = task_backend()
            profile = request_config.current_profile()
            sampling = profile.sampling_parameters() if profile is not None and task_backend_name() == "llamacpp" else {}
            model_options = {}
            if task_backend_name() == "llamacpp" and sampling:
                model_options["sampling_parameters"] = sampling
            return backend.chat_completion(
                messages,
                temperature=float(sampling.get("temperature", TRANSLATION_TEMPERATURE)),
                top_p=float(sampling.get("top_p", TRANSLATION_TOP_P)),
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
                **model_options,
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
