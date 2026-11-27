"""Translation orchestration engine.

One batching loop parameterized by a TranslationProfile and a backend
instance. Owns batch planning, cache/memory reuse, the retry ladder, rolling
history scheduling, progress aggregation and timing records. Profiles supply
messages/parsing; backends execute requests. Nothing here imports
llm.translator.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable

from core.cancellation import PipelineCancelledError
from core.diagnostics import format_exception_chain
from llm import cache as translation_cache
from llm import settings as llm_settings
from llm import transport_util
from llm import zh_variant
from llm.context import SourceContext, split_batches
from llm.errors import (
    ContentPolicyRefusalError,
    ResponseTruncatedError,
    RetryableTranslationFormatError,
    TranslationCancelledError,
)
from llm.profiles.base import ProfileContext, TranslationProfile

logger = logging.getLogger(__name__)
_cancel_requested = transport_util._cancel_requested
_raise_if_cancelled = transport_util._raise_if_cancelled
_emit_progress = transport_util._emit_progress
_CONTEXT_LIMIT_ERROR_RE = re.compile(
    r"(?:exceed_context_size|exceeds?\s+(?:the\s+)?(?:available\s+)?context|"
    r"context\s+size|n_ctx|prompt\s+is\s+too\s+long|input\s+is\s+too\s+long|"
    r"request\s+exceeds)",
    re.IGNORECASE,
)


def _split_into_batches(
    segments: list[dict],
    batch_size: int,
    *,
    max_source_chars: int | None = None,
) -> list[list[dict]]:
    char_cap = (
        llm_settings.TRANSLATION_BATCH_MAX_SOURCE_CHARS
        if max_source_chars is None
        else max_source_chars
    )
    return split_batches(segments, batch_size, max_source_chars=char_cap)


def cache_scope(
    source_context,
    profile: TranslationProfile,
    model_identity: str,
    *,
    backend_name: str = "openai",
) -> str:
    """Include every source occurrence and actual sampling recipe in reuse."""
    sampling = {
        "temperature": llm_settings.TRANSLATION_TEMPERATURE,
        "top_p": llm_settings.TRANSLATION_TOP_P,
        **(profile.sampling_parameters() if backend_name == "llamacpp" else {}),
    }
    payload = json.dumps(
        [source_context.signature, model_identity, sampling],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class _SiblingBatchAborted(Exception):
    """A batch that stopped because another batch had already failed the call.

    Internal to `run_batched` and never surfaced: it exists so the abort a
    failure causes cannot be confused with the failure itself, nor with the
    user cancelling the job.
    """

    def __init__(self, batch_index: int) -> None:
        super().__init__(f"batch {batch_index} aborted alongside a failed sibling")
        self.batch_index = batch_index


class _CombinedCancel:
    """`is_set()` over several cancel sources at once.

    The batch pool needs a stop signal of its own - one batch failing must not
    be reported to the job as "cancelled" - but the transports below take a
    single event. They only ever ask it `is_set()`, so an OR view is enough and
    keeps the job's own event untouched.
    """

    __slots__ = ("_events",)

    def __init__(self, *events) -> None:
        self._events = [event for event in events if event is not None]

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)


def _batch_cost(batch_segments: list[dict]) -> int:
    """Rough cost of a batch: how much Japanese it has to translate."""
    return sum(len(str(seg.get("text", ""))) for seg in batch_segments)


def _submission_order(
    pending_batches: list[tuple[int, list[dict]]],
) -> list[tuple[int, list[dict]]]:
    """Longest batch first (LPT scheduling) to minimize makespan."""
    return sorted(
        pending_batches, key=lambda item: (-_batch_cost(item[1]), item[0])
    )


def _make_aggregated_progress_callback(
    num_batches: int,
    expected_total: int,
    on_progress: Callable[[dict], None] | None,
) -> tuple[list[Callable[[dict], None]], None]:
    lock = threading.Lock()
    batch_states: dict[int, dict] = {}
    last_emit_ts = 0.0
    done_emitted = False

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
            nonlocal last_emit_ts
            try:
                payload_to_emit: dict | None = None
                with lock:
                    batch_states[batch_id] = dict(evt)
                    built = build_payload()
                    if built is not None:
                        data, force = built
                        now = time.monotonic()
                        if force or now - last_emit_ts >= 0.25:
                            last_emit_ts = now
                            payload_to_emit = data
                if payload_to_emit is not None:
                    _emit_progress(on_progress, payload_to_emit)
            except (TranslationCancelledError, PipelineCancelledError):
                raise
            except Exception as exc:
                logger.warning("Aggregated progress callback failed: %s", exc, exc_info=True)

        return wrapper

    return [make_wrapper(batch_id) for batch_id in range(num_batches)], None


def _normalize_text(text) -> str | None:
    # Line profiles reuse the JSON contract's output normalizer so cached and
    # fresh texts go through identical cleanup.
    from llm.profiles import json_v3

    return json_v3._normalize_translation_text(text)


def _profile_bounded_schema_supports_ids(profile: TranslationProfile) -> bool:
    try:
        params = inspect.signature(profile.bounded_schema).parameters
    except (TypeError, ValueError):
        return False
    has_varkw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    return "ids" in params or has_varkw


def _bounded_schema_for_request(
    profile: TranslationProfile,
    request_segments: list[dict],
    *,
    ids: list[int],
    supports_ids: bool | None = None,
) -> dict | None:
    accepts_ids = (
        _profile_bounded_schema_supports_ids(profile)
        if supports_ids is None
        else supports_ids
    )
    if ids and accepts_ids:
        return profile.bounded_schema(request_segments, ids=list(ids))
    return profile.bounded_schema(request_segments)


def prefix_mode_label(use_full_json_prefix: bool, *, windowed: bool = False) -> str:
    """Name for the prompt shape a run uses.

    Folded into the batch cache key, and reported in the timings - the two must
    be the same string, which is why neither side spells it itself. Different
    prompt shapes must never share cached translations.
    """
    if use_full_json_prefix:
        return "full_json_prefix"
    return "windowed_context" if windowed else "summary_fallback"


class _BatchRunState:
    """Cohesive per-run state shared across cache preload, warmup, and workers."""

    def __init__(
        self,
        segments: list[dict],
        *,
        profile: TranslationProfile,
        backend_name: str,
        chat: Callable[..., str],
        backoff_sleep: Callable[..., None],
        crash_probe: Callable[[], int],
        batch_size: int,
        api_retries: int,
        batch_repair_retries: int,
        batch_max_requests: int,
        prefix_warmup: bool,
        extra_glossary: str,
        full_context: str,
        full_source_payload: str,
        use_full_json_prefix: bool,
        cache_path: str,
        cache_lock: threading.Lock,
        target_lang: str,
        glossary: str,
        character_reference: str,
        prompt_version: str,
        model_identity: str,
        compact_system_prompt: bool,
        reasoning_effort: str,
        on_batch_done,
        on_progress: Callable[[dict], None] | None,
        cancel_event: threading.Event | None,
        source_context: SourceContext | None,
        max_source_chars: int | None,
        focus_radius: int,
        focus_context_chars: int,
        batches: list[list[dict]] | None = None,
    ) -> None:
        self.segments = segments
        self.profile = profile
        self.backend_name = backend_name
        self.chat = chat
        self.backoff_sleep = backoff_sleep
        self.crash_probe = crash_probe
        self.batch_size = batch_size
        self.api_retries = api_retries
        self.batch_repair_retries = batch_repair_retries
        self.batch_max_requests = batch_max_requests
        self.prefix_warmup = prefix_warmup
        self.extra_glossary = extra_glossary
        self.full_context = full_context
        self.full_source_payload = full_source_payload
        self.use_full_json_prefix = use_full_json_prefix
        self.cache_path = cache_path
        self.cache_lock = cache_lock
        self.target_lang = target_lang
        self.glossary = glossary
        self.character_reference = character_reference
        self.prompt_version = prompt_version
        self.model_identity = model_identity
        self.compact_system_prompt = compact_system_prompt
        self.reasoning_effort = reasoning_effort
        self.on_batch_done = on_batch_done
        self.on_progress = on_progress
        self.cancel_event = cancel_event
        self.focus_radius = focus_radius
        self.focus_context_chars = focus_context_chars

        self.source_context = source_context or SourceContext.build(
            segments, external_context=full_context
        )
        self.batches = (
            batches
            if batches is not None
            else _split_into_batches(
                segments, batch_size, max_source_chars=max_source_chars
            )
        )
        self.context_signature = cache_scope(
            self.source_context, profile, model_identity, backend_name=backend_name
        )
        self.batch_starts: list[int] = []
        offset = 0
        for batch in self.batches:
            self.batch_starts.append(offset)
            offset += len(batch)
        self.expected_total = len(segments)
        self.prefix_mode = prefix_mode_label(
            use_full_json_prefix,
            windowed=not use_full_json_prefix and not (full_context or "").strip(),
        )
        self.prompt_sig = translation_cache._compute_prompt_signature(
            extra_glossary,
            glossary=glossary,
            target_lang=target_lang,
            character_reference=character_reference,
            prompt_version=prompt_version,
            model_name=model_identity,
            compact_system_prompt=compact_system_prompt,
            reasoning_effort=reasoning_effort,
            prefix_mode=self.prefix_mode,
            context_signature=self.context_signature,
        )
        self.memory_sig = translation_cache._compute_translation_memory_signature(
            extra_glossary,
            glossary=glossary,
            target_lang=target_lang,
            character_reference=character_reference,
            prompt_version=prompt_version,
            model_name=model_identity,
            reasoning_effort=reasoning_effort,
            context_signature=self.context_signature,
        )
        self.bounded_schema_supports_ids = _profile_bounded_schema_supports_ids(profile)
        self.progress_callbacks, _ = _make_aggregated_progress_callback(
            len(self.batches),
            self.expected_total,
            on_progress,
        )
        self.diagnostic_progress_lock = threading.Lock()
        self.zh_texts: list[str | None] = [None] * self.expected_total
        self.to_target_variant = zh_variant.converter_for(target_lang)
        self.timings_by_batch: dict[int, dict] = {}
        self.review_lock = threading.Lock()
        self.review_by_index: dict[int, dict] = {}
        self.blanked_empty_ids: set[int] = set()
        self.pending_by_index: dict[int, list[dict]] = {}
        self.worker_retry_events: list[dict] = []
        self.warmup_timing: dict | None = None

        self.batch_abort = threading.Event()
        self.worker_cancel = _CombinedCancel(cancel_event, self.batch_abort)
        self.failure_lock = threading.Lock()
        self.root_failure: BaseException | None = None
        self.failed_retry_events_lock = threading.Lock()
        self.failed_retry_events: list[dict] = []

    def emit_diagnostic(self, payload: dict) -> None:
        if self.on_progress is None:
            return
        with self.diagnostic_progress_lock:
            _emit_progress(self.on_progress, payload)

    def profile_context(
        self,
        *,
        batch_index: int = 0,
        warmup: bool = False,
        focus_radius: int | None = None,
        focus_context_chars: int | None = None,
    ) -> ProfileContext:
        return ProfileContext(
            target_lang=self.target_lang,
            glossary=self.glossary,
            extra_glossary=self.extra_glossary,
            character_reference=self.character_reference,
            global_context=self.full_context,
            full_source_payload=(
                self.full_source_payload if self.use_full_json_prefix else None
            ),
            total_count=self.expected_total,
            compact_system_prompt=self.compact_system_prompt,
            batch_index=batch_index,
            warmup=warmup,
            source_context=self.source_context,
            focus_radius=self.focus_radius if focus_radius is None else focus_radius,
            focus_context_chars=(
                self.focus_context_chars
                if focus_context_chars is None
                else focus_context_chars
            ),
        )

    def to_variant(self, text: str) -> str:
        return (
            self.to_target_variant(text)
            if text and self.to_target_variant is not None
            else text
        )

    def final_text(self, text) -> str:
        """Restore path for cached and remembered lines."""
        return self.to_variant(_normalize_text(text) or "")

    def batch_key_for(self, batch_index: int, batch_segments: list[dict]) -> str:
        return translation_cache._translation_cache_key(
            batch_index,
            batch_segments,
            prompt_sig=self.prompt_sig,
        )

    def memory_key_for(self, source_text: str, occurrence_id: int) -> str:
        return translation_cache._translation_memory_key(
            source_text,
            occurrence_id=occurrence_id,
            memory_sig=self.memory_sig,
        )

    def valid_cached(self, source: str, text) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        try:
            self.profile.validate_translation(source, text, self.target_lang)
            return True
        except RetryableTranslationFormatError:
            return False

    def record_review(self, index: int, entry: dict) -> None:
        with self.review_lock:
            self.review_by_index[index] = entry

    def record_blanked_empty(self, index: int) -> None:
        with self.review_lock:
            self.blanked_empty_ids.add(index)

    def batch_review(self, start: int, count: int) -> list[dict]:
        with self.review_lock:
            return [
                {"offset": index - start, **self.review_by_index[index]}
                for index in range(start, start + count)
                if index in self.review_by_index
            ]

    def review_field(self, ids) -> dict:
        with self.review_lock:
            entries = [
                {"id": index, **self.review_by_index[index]}
                for index in ids
                if index in self.review_by_index
            ]
        return {"translation_review": entries} if entries else {}

    def persist_batch(self, batch_index: int, batch_results: list[str | None]) -> None:
        """Commit a finished batch on its worker thread so paid replies survive failure."""
        segments = self.pending_by_index[batch_index]
        start_index = self.batch_starts[batch_index]
        local_texts: list[str] = []
        memory_entries: list[tuple[str, str]] = []
        memory_review: dict[str, dict] = {}
        with self.review_lock:
            review_snapshot = {
                start_index + offset: dict(self.review_by_index[start_index + offset])
                for offset in range(len(segments))
                if (start_index + offset) in self.review_by_index
            }
            has_blanked_empty = any(
                (start_index + offset) in self.blanked_empty_ids
                for offset in range(len(segments))
            )
        for offset in range(len(segments)):
            global_index = start_index + offset
            text = batch_results[global_index] or self.zh_texts[global_index] or ""
            # Each index belongs to exactly one batch, so no two workers write the same slot.
            self.zh_texts[global_index] = text
            local_texts.append(text)
            source_text = str(segments[offset].get("text", ""))
            if (
                self.cache_path
                and text
                and translation_cache._translation_memory_source_is_cacheable(
                    source_text
                )
            ):
                memory_key = self.memory_key_for(source_text, global_index)
                memory_entries.append((memory_key, text))
                if global_index in review_snapshot:
                    memory_review[memory_key] = review_snapshot[global_index]
        if not self.cache_path:
            return
        batch_key = self.batch_key_for(batch_index, segments)
        if not has_blanked_empty:
            batch_review = [
                {"offset": idx - start_index, **entry}
                for idx, entry in sorted(review_snapshot.items())
            ]
            translation_cache._save_cache_entry(
                self.cache_path,
                batch_key,
                local_texts,
                self.cache_lock,
                review=batch_review,
            )
        if memory_entries:
            translation_cache._save_memory_entries(
                self.cache_path,
                memory_entries,
                self.cache_lock,
                reviews=memory_review,
            )
        if not has_blanked_empty:
            print(f"[translation-cache] saved batch {batch_index} cache_key={batch_key}")


def _preload_cached_batches(
    state: _BatchRunState,
) -> tuple[list[tuple[int, list[dict]]], int, int]:
    """Restore exact batch hits and per-line translation memory before scheduling."""
    cache_reviews: dict[str, list] = {}
    memory_reviews: dict[str, dict] = {}
    cache_map = (
        translation_cache._load_translation_cache(state.cache_path, cache_reviews)
        if state.cache_path
        else {}
    )
    memory_map = (
        translation_cache._load_translation_memory(state.cache_path, memory_reviews)
        if state.cache_path
        else {}
    )
    pending_batches: list[tuple[int, list[dict]]] = []
    exact_cache_hit_count = 0
    translation_memory_hit_count = 0

    for batch_index, batch_segments in enumerate(state.batches):
        _raise_if_cancelled(state.cancel_event)
        batch_key = state.batch_key_for(batch_index, batch_segments)
        cached_texts = cache_map.get(batch_key)
        start_index = state.batch_starts[batch_index]
        if (
            isinstance(cached_texts, list)
            and len(cached_texts) == len(batch_segments)
            and all(
                state.valid_cached(str(seg.get("text", "")), text)
                for seg, text in zip(batch_segments, cached_texts)
            )
        ):
            exact_cache_hit_count += 1
            print(f"[translation-cache] restored batch {batch_index} cache_key={batch_key}")
            for offset, text in enumerate(cached_texts):
                state.zh_texts[start_index + offset] = state.final_text(text)
            for record in cache_reviews.get(batch_key) or []:
                offset = record.get("offset") if isinstance(record, dict) else None
                if isinstance(offset, int) and 0 <= offset < len(batch_segments):
                    state.record_review(
                        start_index + offset,
                        {k: v for k, v in record.items() if k != "offset"},
                    )
            timing = {
                "batch_index": batch_index,
                "start_index": start_index,
                "segment_count": len(batch_segments),
                "elapsed_s": 0.0,
                "mode": "translation_cache_hit",
                "request_count": 0,
                "source_payload_chars": 0,
                "global_context_chars": len(state.full_context),
                "prefix_mode": state.prefix_mode,
                "requested_ids": list(range(start_index, start_index + len(batch_segments))),
                "is_warmup": False,
                **transport_util._merge_usage_metrics([]),
                "missing_count": 0,
                "missing_indexes": [],
                "cache_hit": True,
                "cache_hit_type": "exact_batch",
                "translation_memory_hit_count": 0,
                **state.review_field(range(start_index, start_index + len(batch_segments))),
            }
            state.timings_by_batch[batch_index] = timing
            _emit_progress(
                state.progress_callbacks[batch_index],
                {
                    "phase": "done",
                    "translated": len(batch_segments),
                    "expected": len(batch_segments),
                },
            )
            if state.on_batch_done:
                _raise_if_cancelled(state.cancel_event)
                state.on_batch_done(timing)
            continue

        memory_hit_ids: list[int] = []
        for offset, seg in enumerate(batch_segments):
            source_text = str(seg.get("text", ""))
            if not state.cache_path or not translation_cache._translation_memory_source_is_cacheable(
                source_text
            ):
                continue
            memory_key = state.memory_key_for(source_text, start_index + offset)
            memory_text = memory_map.get(memory_key)
            if state.valid_cached(source_text, memory_text):
                global_index = start_index + offset
                state.zh_texts[global_index] = state.final_text(memory_text)
                memory_hit_ids.append(global_index)
                if memory_reviews.get(memory_key):
                    state.record_review(global_index, dict(memory_reviews[memory_key]))

        if memory_hit_ids:
            translation_memory_hit_count += len(memory_hit_ids)
            print(
                "[translation-memory] restored "
                f"batch={batch_index} ids={memory_hit_ids[:20]}"
            )

        if len(memory_hit_ids) == len(batch_segments):
            local_texts = [
                state.zh_texts[start_index + offset] or ""
                for offset in range(len(batch_segments))
            ]
            if state.cache_path:
                batch_review = state.batch_review(start_index, len(batch_segments))
                translation_cache._save_cache_entry(
                    state.cache_path,
                    batch_key,
                    local_texts,
                    state.cache_lock,
                    review=batch_review,
                )
            timing = {
                "batch_index": batch_index,
                "start_index": start_index,
                "segment_count": len(batch_segments),
                "elapsed_s": 0.0,
                "mode": "translation_memory_hit",
                "request_count": 0,
                "source_payload_chars": 0,
                "global_context_chars": len(state.full_context),
                "prefix_mode": state.prefix_mode,
                "requested_ids": list(range(start_index, start_index + len(batch_segments))),
                "is_warmup": False,
                **transport_util._merge_usage_metrics([]),
                "missing_count": 0,
                "missing_indexes": [],
                "cache_hit": True,
                "cache_hit_type": "translation_memory",
                "translation_memory_hit_count": len(memory_hit_ids),
                **state.review_field(range(start_index, start_index + len(batch_segments))),
            }
            state.timings_by_batch[batch_index] = timing
            _emit_progress(
                state.progress_callbacks[batch_index],
                {
                    "phase": "done",
                    "translated": len(batch_segments),
                    "expected": len(batch_segments),
                },
            )
            if state.on_batch_done:
                _raise_if_cancelled(state.cancel_event)
                state.on_batch_done(timing)
            continue

        if memory_hit_ids:
            _emit_progress(
                state.progress_callbacks[batch_index],
                {
                    "phase": "translating",
                    "translated": len(memory_hit_ids),
                    "expected": len(batch_segments),
                },
            )
        pending_batches.append((batch_index, batch_segments))

    state.pending_by_index = {idx: batch for idx, batch in pending_batches}
    return pending_batches, exact_cache_hit_count, translation_memory_hit_count


def _run_prefix_warmup(
    state: _BatchRunState,
    pending_batches: list[tuple[int, list[dict]]],
) -> dict | None:
    """Prime provider prefix cache before parallel API batches land."""
    if not (
        pending_batches
        and len(pending_batches) > 1
        and state.prefix_warmup
        and state.backend_name == "openai"
        and state.profile.schema is not None
    ):
        return None
    warmup_started = time.perf_counter()
    warmup_usages: list[dict] = []
    warmup_messages = state.profile.build_messages(
        [], ids=[], ctx=state.profile_context(warmup=True)
    )
    global_chars = (
        len(state.full_source_payload)
        if state.use_full_json_prefix
        else len(state.full_context)
    )
    try:
        _raise_if_cancelled(state.cancel_event)
        state.chat(
            warmup_messages,
            expected_count=0,
            on_usage=warmup_usages.append,
            cancel_event=state.cancel_event,
        )
        return {
            "batch_index": None,
            "start_index": 0,
            "segment_count": 0,
            "elapsed_s": time.perf_counter() - warmup_started,
            "mode": "translation_prefix_warmup",
            "request_count": 1,
            "source_payload_chars": 0,
            "global_context_chars": global_chars,
            "prefix_mode": state.prefix_mode,
            "requested_ids": [],
            "is_warmup": True,
            "missing_count": 0,
            "missing_indexes": [],
            **transport_util._merge_usage_metrics(warmup_usages),
        }
    except Exception as exc:
        if isinstance(exc, TranslationCancelledError):
            raise
        logger.warning("Translation prefix warmup failed: %s", format_exception_chain(exc))
        return {
            "batch_index": None,
            "start_index": 0,
            "segment_count": 0,
            "elapsed_s": time.perf_counter() - warmup_started,
            "mode": "translation_prefix_warmup_failed",
            "request_count": 1,
            "source_payload_chars": 0,
            "global_context_chars": global_chars,
            "prefix_mode": state.prefix_mode,
            "requested_ids": [],
            "is_warmup": True,
            "missing_count": 0,
            "missing_indexes": [],
            "error": str(exc)[:500],
            **transport_util._merge_usage_metrics(warmup_usages),
        }


def _execute_single_batch(
    state: _BatchRunState,
    batch_index: int,
    batch_segments: list[dict],
) -> tuple[int, list[str | None], dict, list[dict]]:
    _raise_if_cancelled(state.worker_cancel)
    # Worker threads do not inherit the caller's thread-local retry events,
    # so bind a per-batch container here and merge it back on the main thread.
    batch_retry_events: list[dict] = []
    transport_util._RETRY_CONTEXT.events = batch_retry_events
    batch_started = time.perf_counter()
    batch_started_ts = time.time()
    worker_thread = threading.current_thread()
    worker_thread_id = threading.get_ident()
    worker_thread_name = worker_thread.name
    start_index = state.batch_starts[batch_index]
    expected_count = len(batch_segments)
    all_batch_ids = list(range(start_index, start_index + expected_count))
    requested_segments: list[dict] = []
    expected_ids: list[int] = []
    batch_results: list[str | None] = [None] * state.expected_total
    for offset, seg in enumerate(batch_segments):
        global_index = start_index + offset
        cached_text = state.zh_texts[global_index]
        if cached_text is not None:
            batch_results[global_index] = cached_text
            continue
        requested_segments.append(seg)
        expected_ids.append(global_index)
    source_payload = state.profile.serialize_source(requested_segments, ids=expected_ids)
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
    state.emit_diagnostic({"phase": "batch_start", **trace_base})
    messages = state.profile.build_messages(
        requested_segments,
        ids=expected_ids,
        ctx=state.profile_context(batch_index=batch_index),
    )
    missing_indexes: list[int] = []
    progress_callback = state.progress_callbacks[batch_index]
    request_count = 0
    pending_ids = list(expected_ids)
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
            state.emit_diagnostic(
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
    retry_limit_for_pending = state.api_retries
    last_retry_error: Exception | None = None
    request_span_limit = max(1, len(expected_ids))
    effective_focus_radius = state.focus_radius
    effective_focus_context_chars = state.focus_context_chars
    blanked_empty_ids: list[int] = []
    single_cue_parsed_empty_count = 0
    output_grammar: str | None = None
    constrained_after = ""

    def constrain_output(exc: RetryableTranslationFormatError) -> None:
        nonlocal output_grammar, constrained_after
        if output_grammar is not None or state.backend_name != "llamacpp":
            return
        output_grammar = state.profile.retry_grammar(exc, state.target_lang)
        if output_grammar is not None:
            constrained_after = str(getattr(exc, "reason", "") or type(exc).__name__)
            state.emit_diagnostic(
                {
                    "phase": "batch_output_constrained",
                    **trace_base,
                    "pending_count": len(pending_ids),
                    "request_index": request_count,
                    "error": str(exc)[:300],
                }
            )

    def narrow_request_span(
        current_request_len: int | None = None,
        *,
        shrink_context: bool = False,
    ) -> bool:
        nonlocal request_span_limit, effective_focus_radius, effective_focus_context_chars
        shrank = False
        effective_current = (
            min(request_span_limit, current_request_len)
            if current_request_len is not None and current_request_len > 0
            else request_span_limit
        )
        if effective_current > 1:
            previous = effective_current
            request_span_limit = max(1, effective_current // 2)
            shrank = True
            state.emit_diagnostic(
                {
                    "phase": "batch_span_narrowed",
                    **trace_base,
                    "from_span": previous,
                    "to_span": request_span_limit,
                    "pending_count": len(pending_ids),
                    "request_index": request_count,
                    "error": str(last_retry_error)[:300],
                }
            )
        else:
            request_span_limit = 1
        if shrink_context and (
            effective_focus_context_chars > 0 or effective_focus_radius > 0
        ):
            if request_span_limit <= 1 or effective_focus_context_chars <= 250:
                effective_focus_context_chars = 0
                effective_focus_radius = 0
            else:
                effective_focus_context_chars = effective_focus_context_chars // 2
                effective_focus_radius = max(1, effective_focus_radius // 2)
            shrank = True
        return shrank

    while True:
        _raise_if_cancelled(state.worker_cancel)
        if request_count >= state.batch_max_requests:
            raise RuntimeError(
                "Batch translation exceeded hard request cap "
                f"({state.batch_max_requests}): batch={batch_index}, "
                f"start_index={start_index}, size={expected_count}, "
                f"pending_ids={pending_ids[:50]}, error={last_retry_error}"
            ) from last_retry_error
        if attempts_for_pending >= retry_limit_for_pending:
            raise RuntimeError(
                "Batch translation got no usable reply after "
                f"{request_count} attempts: batch={batch_index}, "
                f"start_index={start_index}, size={expected_count}, "
                f"pending_ids={pending_ids[:50]}, error={last_retry_error}"
            ) from last_retry_error

        requested_ids = list(pending_ids[:request_span_limit])
        pending_before_request = len(pending_ids)
        active_request_index = request_count
        active_requested_ids = list(requested_ids)
        trace_progress({"phase": "reset", "attempt": request_count})
        try:
            _raise_if_cancelled(state.worker_cancel)
            request_segments = [
                batch_segments[index - start_index] for index in requested_ids
            ]
            if request_count == 0 and requested_ids == expected_ids:
                request_messages = messages
            else:
                request_messages = state.profile.build_messages(
                    request_segments,
                    ids=requested_ids,
                    ctx=state.profile_context(
                        batch_index=batch_index,
                        focus_radius=effective_focus_radius,
                        focus_context_chars=effective_focus_context_chars,
                    ),
                )
            request_expected_count = len(request_segments)
            request_count += 1
            request_kwargs: dict = {}
            token_budget = state.profile.response_token_budget(
                request_segments,
                reasoning_effort=state.reasoning_effort,
            )
            if token_budget is not None:
                request_kwargs["max_tokens"] = token_budget
            bounded = _bounded_schema_for_request(
                state.profile,
                request_segments,
                ids=requested_ids,
                supports_ids=state.bounded_schema_supports_ids,
            )
            if bounded is not None:
                request_kwargs["bounded_response_schema"] = bounded
            request_kwargs["response_schema"] = state.profile.schema
            if output_grammar is not None:
                request_kwargs["output_grammar"] = output_grammar
            raw_output = state.chat(
                request_messages,
                expected_count=request_expected_count,
                on_progress=trace_progress,
                on_usage=request_usages.append,
                cancel_event=state.worker_cancel,
                **request_kwargs,
            )
            # No cancel check here on purpose: the reply is already generated and
            # paid for, so parse and commit it before unwinding on an abort.
            parsed = state.profile.parse_response(raw_output, ids=requested_ids)
            for idx in requested_ids:
                if parsed.get(idx):
                    state.profile.validate_translation(
                        str(batch_segments[idx - start_index].get("text", "")),
                        parsed[idx],
                        state.target_lang,
                    )
                    batch_results[idx] = state.to_variant(parsed[idx])
                    if output_grammar is not None:
                        state.record_review(
                            idx,
                            {
                                "reason": "constrained_retry",
                                "rejected": constrained_after,
                                "requests": request_count,
                            },
                        )
            missing_indexes = [
                index for index in all_batch_ids if batch_results[index] is None
            ]
            if not missing_indexes:
                break

            pending_ids = list(missing_indexes)
            last_retry_error = RetryableTranslationFormatError(
                "translation returned incomplete batch translations: "
                f"{len(missing_indexes)} missing of {expected_count}; "
                f"missing ids={missing_indexes[:50]}"
            )
            if (
                len(missing_indexes) < pending_before_request
                and state.profile.supports_partial_reissue
            ):
                # Narrowed span or partial progress succeeded: move straight on
                # to the remaining ids without artificial backoff or retry logs.
                attempts_for_pending = 0
                retry_limit_for_pending = state.batch_repair_retries
                single_cue_parsed_empty_count = 0
                continue
            attempts_for_pending += 1
            if len(requested_ids) == 1:
                single_cue_parsed_empty_count += 1
            else:
                shrank = narrow_request_span(len(requested_ids), shrink_context=False)
                if shrank and attempts_for_pending >= retry_limit_for_pending:
                    retry_limit_for_pending = max(
                        retry_limit_for_pending, attempts_for_pending + 1
                    )
        except ContentPolicyRefusalError as refusal:
            raise ContentPolicyRefusalError(
                f"{refusal} (batch={batch_index}, start_index={start_index}, "
                f"size={expected_count}, requested_ids={requested_ids[:50]})"
            ) from refusal
        except ResponseTruncatedError as exc:
            last_retry_error = exc
            shrank = narrow_request_span(len(requested_ids), shrink_context=True)
            if not shrank:
                raise
            attempts_for_pending += 1
            # Let truncation descent halve down to span=1 + zero neighbour context
            # (still strictly bounded by batch_max_requests).
            retry_limit_for_pending = max(
                retry_limit_for_pending, attempts_for_pending + 1
            )
        except RetryableTranslationFormatError as exc:
            last_retry_error = exc
            is_ctx_err = bool(_CONTEXT_LIMIT_ERROR_RE.search(str(exc)))
            shrank = narrow_request_span(len(requested_ids), shrink_context=True)
            attempts_for_pending += 1
            if is_ctx_err and shrank:
                # Let context-overflow descent reach span=1 + zero neighbour
                # context (still strictly bounded by batch_max_requests).
                retry_limit_for_pending = max(
                    retry_limit_for_pending, attempts_for_pending + 1
                )
            constrain_output(exc)

        if attempts_for_pending < retry_limit_for_pending:
            sleep_attempt = max(0, attempts_for_pending - 1)
            state.backoff_sleep(
                sleep_attempt,
                last_retry_error,
                cancel_event=state.worker_cancel,
            )
            continue

        if len(requested_ids) == 1 and single_cue_parsed_empty_count > 0:
            # Valid structured JSON was returned for this single cue on every
            # retry, but its translation stayed empty (typically untranslatable
            # vocalisation/onomatopoeia fragments). Blank it with a warning
            # instead of throwing away the whole batch and film; the repair pass
            # still gets one more chance to fill it before SRT export skips it.
            empty_idx = requested_ids[0]
            logger.warning(
                "Translation batch %d cue %d returned empty translation after %d "
                "parsed attempts; blanking cue instead of aborting the run",
                batch_index,
                empty_idx,
                single_cue_parsed_empty_count,
            )
            batch_results[empty_idx] = ""
            blanked_empty_ids.append(empty_idx)
            state.record_blanked_empty(empty_idx)
            missing_indexes = [
                index for index in all_batch_ids if batch_results[index] is None
            ]
            if not missing_indexes:
                break
            pending_ids = list(missing_indexes)
            attempts_for_pending = 0
            retry_limit_for_pending = state.batch_repair_retries
            single_cue_parsed_empty_count = 0
            continue

        raise RuntimeError(
            "Batch translation got no usable reply after "
            f"{request_count} attempts: batch={batch_index}, "
            f"start_index={start_index}, size={expected_count}, "
            f"pending_ids={pending_ids[:50]}, error={last_retry_error}"
        ) from last_retry_error

    batch_elapsed_s = time.perf_counter() - batch_started
    batch_finished_ts = time.time()
    if first_token_ts is None:
        first_token_ts = batch_finished_ts
    state.emit_diagnostic(
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
        "global_context_chars": (
            len(state.full_source_payload)
            if state.use_full_json_prefix
            else len(state.full_context)
        ),
        "prefix_mode": state.prefix_mode,
        "requested_ids": expected_ids,
        "is_warmup": False,
        "started_ts": batch_started_ts,
        "first_token_ts": first_token_ts,
        "finished_ts": batch_finished_ts,
        "worker_thread_id": worker_thread_id,
        "worker_thread_name": worker_thread_name,
        **transport_util._merge_usage_metrics(request_usages),
        "missing_count": len(missing_indexes),
        "missing_indexes": missing_indexes,
        "empty_translation_count": len(blanked_empty_ids),
        "empty_translation_ids": blanked_empty_ids,
        "translation_memory_hit_count": expected_count - len(expected_ids),
        "cache_hit_type": "mixed" if len(expected_ids) < expected_count else "miss",
        **state.review_field(all_batch_ids),
    }
    state.persist_batch(batch_index, batch_results)
    return batch_index, batch_results, timing, batch_retry_events


def _execute_batch_guarded(
    state: _BatchRunState,
    batch_index: int,
    batch_segments: list[dict],
) -> tuple[int, list[str | None], dict, list[dict]]:
    try:
        return _execute_single_batch(state, batch_index, batch_segments)
    except TranslationCancelledError:
        batch_events = transport_util._current_retry_events() or []
        if batch_events:
            with state.failed_retry_events_lock:
                state.failed_retry_events.extend(batch_events)
        if state.batch_abort.is_set() and not _cancel_requested(state.cancel_event):
            raise _SiblingBatchAborted(batch_index) from None
        raise
    except BaseException as exc:
        batch_events = transport_util._current_retry_events() or []
        if batch_events:
            with state.failed_retry_events_lock:
                state.failed_retry_events.extend(batch_events)
        error_chain = format_exception_chain(exc)
        logger.error(
            "Translation batch %d failed (start_index=%d size=%d retries=%d): %s",
            batch_index,
            state.batch_starts[batch_index],
            len(batch_segments),
            len(batch_events),
            error_chain,
            exc_info=True,
        )
        state.emit_diagnostic(
            {
                "phase": "batch_failed",
                "diagnostic": True,
                "batch_index": batch_index,
                "start_index": state.batch_starts[batch_index],
                "segment_count": len(batch_segments),
                "error_type": type(exc).__name__,
                "error": error_chain[:300],
                "retry_count": len(batch_events),
            }
        )
        with state.failure_lock:
            if state.root_failure is None:
                state.root_failure = exc
            else:
                logger.warning(
                    "Secondary batch %d failure after root error: %s",
                    batch_index,
                    error_chain,
                )
        state.batch_abort.set()
        raise


def _dispatch_pending_batches(
    state: _BatchRunState,
    pending_batches: list[tuple[int, list[dict]]],
    max_workers: int,
) -> None:
    if not pending_batches:
        return
    executor = ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(pending_batches))))
    futures: dict = {}
    try:
        _raise_if_cancelled(state.cancel_event)
        futures = {
            executor.submit(_execute_batch_guarded, state, batch_index, batch): batch_index
            for batch_index, batch in _submission_order(pending_batches)
        }

        def harvest(result, source_future) -> None:
            batch_index, _batch_results, timing, batch_retry_events = result
            state.worker_retry_events.extend(batch_retry_events)
            state.timings_by_batch[batch_index] = timing
            if state.cache_path and state.crash_probe() == batch_index + 1:
                for pending_future in futures:
                    if pending_future is not source_future:
                        pending_future.cancel()
                raise SystemExit(1)
            if state.on_batch_done:
                _raise_if_cancelled(state.cancel_event)
                state.on_batch_done(timing)

        remaining = set(futures)
        while remaining:
            _raise_if_cancelled(state.cancel_event)
            done, remaining = wait(
                remaining,
                timeout=0.1,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue
            harvested: list[tuple] = []
            failure: BaseException | None = None
            for future in sorted(done, key=lambda item: futures[item]):
                try:
                    harvested.append((future.result(), future))
                except BaseException as exc:
                    if failure is None:
                        failure = exc
            for result, source_future in harvested:
                harvest(result, source_future)
            if failure is not None:
                raise failure
    except BaseException as exc:
        state.batch_abort.set()
        for pending_future in futures:
            pending_future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        with state.failure_lock:
            root = state.root_failure
        with state.failed_retry_events_lock:
            combined_retries = [*state.worker_retry_events, *state.failed_retry_events]
        partial_timings: list[dict] = []
        if state.warmup_timing is not None:
            partial_timings.append(state.warmup_timing)
        partial_timings.extend(
            state.timings_by_batch[index] for index in sorted(state.timings_by_batch)
        )
        diagnostics_snapshot = {
            "translation_request_timings": partial_timings,
            "translation_api_retry_events": combined_retries,
        }
        if _cancel_requested(state.cancel_event) or root is None or root is exc:
            if isinstance(exc, _SiblingBatchAborted):
                err = RuntimeError(
                    "a batch stood down for a sibling failure that was never recorded"
                )
                err._translation_diagnostics = diagnostics_snapshot  # type: ignore[attr-defined]
                raise err from exc
            try:
                exc._translation_diagnostics = diagnostics_snapshot  # type: ignore[attr-defined]
            except (AttributeError, TypeError):
                pass
            raise
        try:
            root._translation_diagnostics = diagnostics_snapshot  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass
        raise root from root.__cause__
    else:
        executor.shutdown(wait=True)


def run_batched(
    segments: list[dict],
    *,
    profile: TranslationProfile,
    backend_name: str,
    chat: Callable[..., str],
    backoff_sleep: Callable[..., None],
    crash_probe: Callable[[], int],
    batch_size: int,
    max_workers: int,
    api_retries: int,
    batch_repair_retries: int,
    batch_max_requests: int,
    prefix_warmup: bool,
    extra_glossary: str,
    full_context: str,
    full_source_payload: str,
    use_full_json_prefix: bool,
    cache_path: str,
    cache_lock: threading.Lock,
    target_lang: str,
    glossary: str,
    character_reference: str,
    prompt_version: str,
    model_identity: str,
    compact_system_prompt: bool,
    reasoning_effort: str = "",
    on_batch_done=None,
    on_progress: Callable[[dict], None] | None = None,
    cancel_event: threading.Event | None = None,
    source_context=None,
    max_source_chars: int | None = None,
    focus_radius: int = 2,
    focus_context_chars: int = 3000,
    prepare_backend: Callable[[Any], Any] | None = None,
    max_workers_cap: int | None = None,
    batches: list[list[dict]] | None = None,
) -> tuple[list[str], list[dict], list[dict]]:
    """Free-parallel batch loop for id-addressed profiles (the JSON contract).

    Batches carry global segment ids, so they can run in any order across a
    worker pool. Retry ladder per batch: full re-request on format errors,
    partial reissue of only the missing ids when the profile supports it, and
    a hard request cap as the backstop. ``chat``/``backoff_sleep``/``crash_probe``
    come from the caller so transport dispatch and test seams stay outside the
    engine.
    """
    _raise_if_cancelled(cancel_event)
    started = time.perf_counter()

    state = _BatchRunState(
        segments,
        profile=profile,
        backend_name=backend_name,
        chat=chat,
        backoff_sleep=backoff_sleep,
        crash_probe=crash_probe,
        batch_size=batch_size,
        api_retries=api_retries,
        batch_repair_retries=batch_repair_retries,
        batch_max_requests=batch_max_requests,
        prefix_warmup=prefix_warmup,
        extra_glossary=extra_glossary,
        full_context=full_context,
        full_source_payload=full_source_payload,
        use_full_json_prefix=use_full_json_prefix,
        cache_path=cache_path,
        cache_lock=cache_lock,
        target_lang=target_lang,
        glossary=glossary,
        character_reference=character_reference,
        prompt_version=prompt_version,
        model_identity=model_identity,
        compact_system_prompt=compact_system_prompt,
        reasoning_effort=reasoning_effort,
        on_batch_done=on_batch_done,
        on_progress=on_progress,
        cancel_event=cancel_event,
        source_context=source_context,
        max_source_chars=max_source_chars,
        focus_radius=focus_radius,
        focus_context_chars=focus_context_chars,
        batches=batches,
    )

    pending_batches, exact_cache_hit_count, translation_memory_hit_count = (
        _preload_cached_batches(state)
    )
    _raise_if_cancelled(cancel_event)

    effective_max_workers = max(1, int(max_workers))
    if pending_batches and prepare_backend is not None:
        active_plan = prepare_backend(cancel_event)
        if active_plan is not None:
            plan_parallel = getattr(active_plan, "parallel", None)
            if isinstance(plan_parallel, int) and plan_parallel > 0:
                worker_cap = (
                    max(1, int(max_workers_cap))
                    if max_workers_cap is not None
                    else effective_max_workers
                )
                effective_max_workers = max(1, min(worker_cap, plan_parallel))
            plan_radius = getattr(active_plan, "focus_radius", None)
            if isinstance(plan_radius, int) and plan_radius > 0:
                state.focus_radius = plan_radius
            plan_ctx_chars = getattr(active_plan, "focus_context_chars", None)
            if isinstance(plan_ctx_chars, int) and plan_ctx_chars > 0:
                state.focus_context_chars = plan_ctx_chars

    state.warmup_timing = _run_prefix_warmup(state, pending_batches)
    _dispatch_pending_batches(state, pending_batches, effective_max_workers)

    _raise_if_cancelled(cancel_event)
    missing = [
        idx
        for idx, value in enumerate(state.zh_texts)
        if value is None or (value == "" and idx not in state.blanked_empty_ids)
    ]
    if missing:
        raise RuntimeError(
            "Batched translation finished with missing translations: "
            f"{len(missing)} missing; ids={missing[:50]}"
        )

    timings: list[dict] = []
    if state.warmup_timing is not None:
        timings.append(state.warmup_timing)
    timings.extend(
        state.timings_by_batch[index] for index in sorted(state.timings_by_batch)
    )
    timings.append(
        {
            "start_index": 0,
            "segment_count": state.expected_total,
            "elapsed_s": time.perf_counter() - started,
            "mode": "batched_full_context_total",
            "request_count": len(pending_batches),
            "batch_size": batch_size,
            "max_workers": effective_max_workers,
            "cache_hit_count": exact_cache_hit_count,
            "translation_memory_hit_count": translation_memory_hit_count,
            "prefix_mode": state.prefix_mode,
            "is_warmup": False,
            "requested_ids": [],
            "missing_count": 0,
            "missing_indexes": [],
        }
    )
    return [text or "" for text in state.zh_texts], timings, state.worker_retry_events
