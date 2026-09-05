"""Translation orchestration engine.

One batching loop parameterized by a TranslationProfile and a backend
instance. Owns batch planning, cache/memory reuse, the retry ladder, rolling
history scheduling, progress aggregation and timing records. Profiles supply
messages/parsing; backends execute requests. Nothing here imports
llm.translator.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable

from llm import cache as translation_cache
from llm import transport_util
from llm import zh_variant
from llm.errors import (
    ContentPolicyRefusalError,
    RetryableTranslationFormatError,
    TranslationCancelledError,
)
from llm.profiles.base import ProfileContext, TranslationProfile

_cancel_requested = transport_util._cancel_requested
_raise_if_cancelled = transport_util._raise_if_cancelled
_emit_progress = transport_util._emit_progress


def _split_into_batches(segments: list[dict], batch_size: int) -> list[list[dict]]:
    if not segments:
        return []
    if batch_size <= 0:
        return [segments]
    return [
        segments[index : index + batch_size]
        for index in range(0, len(segments), batch_size)
    ]


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
    """Rough cost of a batch: how much Japanese it has to translate.

    Latency here is dominated by output tokens, and output length tracks source
    length closely enough to rank batches by. Nothing depends on this being
    accurate - it only decides who starts first.
    """
    return sum(len(str(seg.get("text", ""))) for seg in batch_segments)


def _submission_order(
    pending_batches: list[tuple[int, list[dict]]],
) -> list[tuple[int, list[dict]]]:
    """Longest batch first.

    Batches are id-addressed and independent, so the order they run in is free -
    but the order they *start* in decides the makespan. Submitted in index
    order, the largest batch can be the last one picked up, and the whole
    translation then ends one full large batch after the pool went idle.
    Longest-processing-time-first is the standard fix and is within 4/3 of
    optimal for identical workers; here the batches are far from equal, because
    a batch of twelve long lines and a batch of twelve grunts both count twelve.
    """
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


def _normalize_text(text) -> str | None:
    # Line profiles reuse the JSON contract's output normalizer so cached and
    # fresh texts go through identical cleanup.
    from llm.profiles import json_v3

    return json_v3._normalize_translation_text(text)


def prefix_mode_label(use_full_json_prefix: bool) -> str:
    """Name for the prompt shape a run uses.

    Folded into the batch cache key, and reported in the timings - the two must
    be the same string, which is why neither side spells it itself. The two
    shapes send different prompts for the same batch, so a key without it serves
    the other shape's translation.
    """
    return "full_json_prefix" if use_full_json_prefix else "summary_fallback"


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
    batches = _split_into_batches(segments, batch_size)
    expected_total = len(segments)
    prefix_mode = prefix_mode_label(use_full_json_prefix)
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

    def _ctx(*, batch_index: int = 0, warmup: bool = False) -> ProfileContext:
        return ProfileContext(
            target_lang=target_lang,
            glossary=glossary,
            extra_glossary=extra_glossary,
            character_reference=character_reference,
            global_context=full_context,
            full_source_payload=full_source_payload if use_full_json_prefix else None,
            total_count=expected_total,
            compact_system_prompt=compact_system_prompt,
            batch_index=batch_index,
            warmup=warmup,
        )

    zh_texts: list[str | None] = [None] * expected_total
    # Applied to cache and memory hits as well as fresh output. Both keys already
    # carry target_lang, so a hit is same-variant by construction - but entries
    # written before this pass existed are not, and the conversion is idempotent,
    # so running it on every path costs nothing and repairs those in place.
    to_target_variant = zh_variant.converter_for(target_lang)

    def _to_target_variant(text: str) -> str:
        return to_target_variant(text) if text and to_target_variant is not None else text

    def _final_text(text) -> str:
        """Restore path for cached and remembered lines: same normalization as
        before, plus the variant conversion fresh output gets."""
        return _to_target_variant(_normalize_text(text) or "")

    timings_by_batch: dict[int, dict] = {}
    cache_map = translation_cache._load_translation_cache(cache_path) if cache_path else {}
    memory_map = translation_cache._load_translation_memory(cache_path) if cache_path else {}
    pending_batches: list[tuple[int, list[dict]]] = []
    worker_retry_events: list[dict] = []
    warmup_timing: dict | None = None
    exact_cache_hit_count = 0
    translation_memory_hit_count = 0

    def _batch_key_for(batch_index: int, batch_segments: list[dict]) -> str:
        return translation_cache._translation_cache_key(
            batch_index,
            batch_segments,
            extra_glossary=extra_glossary,
            glossary=glossary,
            target_lang=target_lang,
            character_reference=character_reference,
            prompt_version=prompt_version,
            model_name=model_identity,
            compact_system_prompt=compact_system_prompt,
            reasoning_effort=reasoning_effort,
            prefix_mode=prefix_mode,
        )

    def _memory_key_for(source_text: str) -> str:
        return translation_cache._translation_memory_key(
            source_text,
            extra_glossary,
            glossary=glossary,
            target_lang=target_lang,
            character_reference=character_reference,
            prompt_version=prompt_version,
            model_name=model_identity,
            reasoning_effort=reasoning_effort,
        )

    for batch_index, batch_segments in enumerate(batches):
        _raise_if_cancelled(cancel_event)
        batch_key = _batch_key_for(batch_index, batch_segments)
        cached_texts = cache_map.get(batch_key)
        start_index = batch_index * batch_size
        if isinstance(cached_texts, list) and len(cached_texts) == len(batch_segments):
            exact_cache_hit_count += 1
            print(f"[translation-cache] restored batch {batch_index} cache_key={batch_key}")
            for offset, text in enumerate(cached_texts):
                zh_texts[start_index + offset] = _final_text(text)
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
                **transport_util._merge_usage_metrics([]),
                "missing_count": 0,
                "missing_indexes": [],
                "cache_hit": True,
                "cache_hit_type": "exact_batch",
                "translation_memory_hit_count": 0,
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
                _raise_if_cancelled(cancel_event)
                on_batch_done(timing)
            continue

        memory_hit_ids: list[int] = []
        for offset, seg in enumerate(batch_segments):
            source_text = str(seg.get("text", ""))
            if not cache_path or not translation_cache._translation_memory_source_is_cacheable(
                source_text
            ):
                continue
            memory_text = memory_map.get(_memory_key_for(source_text))
            if isinstance(memory_text, str) and memory_text.strip():
                global_index = start_index + offset
                zh_texts[global_index] = _final_text(memory_text)
                memory_hit_ids.append(global_index)

        if memory_hit_ids:
            translation_memory_hit_count += len(memory_hit_ids)
            print(
                "[translation-memory] restored "
                f"batch={batch_index} ids={memory_hit_ids[:20]}"
            )

        if len(memory_hit_ids) == len(batch_segments):
            local_texts = [
                zh_texts[start_index + offset] or ""
                for offset in range(len(batch_segments))
            ]
            if cache_path:
                translation_cache._save_cache_entry(
                    cache_path, batch_key, local_texts, cache_lock
                )
                cache_map[batch_key] = local_texts
            timing = {
                "batch_index": batch_index,
                "start_index": start_index,
                "segment_count": len(batch_segments),
                "elapsed_s": 0.0,
                "mode": "translation_memory_hit",
                "request_count": 0,
                "source_payload_chars": 0,
                "global_context_chars": len(full_context),
                "prefix_mode": prefix_mode,
                "requested_ids": list(range(start_index, start_index + len(batch_segments))),
                "is_warmup": False,
                **transport_util._merge_usage_metrics([]),
                "missing_count": 0,
                "missing_indexes": [],
                "cache_hit": True,
                "cache_hit_type": "translation_memory",
                "translation_memory_hit_count": len(memory_hit_ids),
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
                _raise_if_cancelled(cancel_event)
                on_batch_done(timing)
            continue

        if memory_hit_ids:
            _emit_progress(
                progress_callbacks[batch_index],
                {
                    "phase": "translating",
                    "translated": len(memory_hit_ids),
                    "expected": len(batch_segments),
                },
            )
        pending_batches.append((batch_index, batch_segments))

    pending_by_index = {batch_index: batch for batch_index, batch in pending_batches}

    _raise_if_cancelled(cancel_event)
    # Warmup exists to prime the provider prefix cache before PARALLEL batches
    # land; with a single pending batch it is a pure extra request.
    if pending_batches and len(pending_batches) > 1 and prefix_warmup and backend_name == "openai":
        warmup_started = time.perf_counter()
        warmup_usages: list[dict] = []
        warmup_messages = profile.build_messages([], ids=[], ctx=_ctx(warmup=True))
        try:
            _raise_if_cancelled(cancel_event)
            chat(
                warmup_messages,
                expected_count=0,
                on_usage=warmup_usages.append,
                cancel_event=cancel_event,
            )
            warmup_timing = {
                "batch_index": None,
                "start_index": 0,
                "segment_count": 0,
                "elapsed_s": time.perf_counter() - warmup_started,
                "mode": "translation_prefix_warmup",
                "request_count": 1,
                "source_payload_chars": 0,
                "global_context_chars": (
                    len(full_source_payload)
                    if use_full_json_prefix
                    else len(full_context)
                ),
                "prefix_mode": prefix_mode,
                "requested_ids": [],
                "is_warmup": True,
                "missing_count": 0,
                "missing_indexes": [],
                **transport_util._merge_usage_metrics(warmup_usages),
            }
        except Exception as exc:
            if isinstance(exc, TranslationCancelledError):
                raise
            warmup_timing = {
                "batch_index": None,
                "start_index": 0,
                "segment_count": 0,
                "elapsed_s": time.perf_counter() - warmup_started,
                "mode": "translation_prefix_warmup_failed",
                "request_count": 1,
                "source_payload_chars": 0,
                "global_context_chars": (
                    len(full_source_payload)
                    if use_full_json_prefix
                    else len(full_context)
                ),
                "prefix_mode": prefix_mode,
                "requested_ids": [],
                "is_warmup": True,
                "missing_count": 0,
                "missing_indexes": [],
                "error": str(exc)[:500],
                **transport_util._merge_usage_metrics(warmup_usages),
            }
            print(f"[WARN] translation prefix warmup failed: {exc}", flush=True)

    # Set when one batch has already failed the whole call. Workers check it
    # alongside the job's cancel event, so a batch that is mid-request - or one
    # a freed worker has just picked up off the queue - stops instead of
    # streaming on into a run nobody is listening to any more.
    batch_abort = threading.Event()
    worker_cancel = _CombinedCancel(cancel_event, batch_abort)
    # The failure that started the abort, kept apart from the sibling
    # cancellations it causes so the run can report what actually broke.
    failure_lock = threading.Lock()
    root_failure: BaseException | None = None

    def persist_batch(batch_index: int, batch_results: list[str | None]) -> None:
        """Commit a finished batch: fill in its cues and write its cache.

        Called on the worker, before the batch is handed back. It used to run on
        the main thread after harvesting, which meant a batch that had been
        generated and paid for was thrown away whenever the main thread unwound
        before getting to it - and no amount of sweeping already-finished
        futures closes that, because one can always finish just after the sweep.
        Committing where the work was done removes the window instead of
        narrowing it.
        """
        segments = pending_by_index[batch_index]
        start_index = batch_index * batch_size
        local_texts: list[str] = []
        memory_entries: list[tuple[str, str]] = []
        for offset in range(len(segments)):
            global_index = start_index + offset
            text = batch_results[global_index] or zh_texts[global_index] or ""
            # Each index belongs to exactly one batch, so no two workers ever
            # write the same slot.
            zh_texts[global_index] = text
            local_texts.append(text)
            source_text = str(segments[offset].get("text", ""))
            if (
                cache_path
                and text
                and translation_cache._translation_memory_source_is_cacheable(
                    source_text
                )
            ):
                memory_entries.append((_memory_key_for(source_text), text))
        if not cache_path:
            return
        batch_key = _batch_key_for(batch_index, segments)
        translation_cache._save_cache_entry(
            cache_path, batch_key, local_texts, cache_lock
        )
        cache_map[batch_key] = local_texts
        if memory_entries:
            translation_cache._save_memory_entries(
                cache_path, memory_entries, cache_lock
            )
            for memory_key, memory_text in memory_entries:
                memory_map[memory_key] = memory_text
        print(f"[translation-cache] saved batch {batch_index} cache_key={batch_key}")

    def run_batch(
        batch_index: int, batch_segments: list[dict]
    ) -> tuple[int, list[str | None], dict, list[dict]]:
        _raise_if_cancelled(worker_cancel)
        # Worker threads do not inherit the caller's thread-local retry events,
        # so retry recording would silently no-op. Bind a per-batch container
        # here and merge it back on the main thread.
        batch_retry_events: list[dict] = []
        transport_util._RETRY_CONTEXT.events = batch_retry_events
        batch_started = time.perf_counter()
        batch_started_ts = time.time()
        worker_thread = threading.current_thread()
        worker_thread_id = threading.get_ident()
        worker_thread_name = worker_thread.name
        start_index = batch_index * batch_size
        expected_count = len(batch_segments)
        all_batch_ids = list(range(start_index, start_index + expected_count))
        requested_segments: list[dict] = []
        expected_ids: list[int] = []
        batch_results: list[str | None] = [None] * expected_total
        for offset, seg in enumerate(batch_segments):
            global_index = start_index + offset
            cached_text = zh_texts[global_index]
            if cached_text is not None:
                batch_results[global_index] = cached_text
                continue
            requested_segments.append(seg)
            expected_ids.append(global_index)
        source_payload = profile.serialize_source(requested_segments, ids=expected_ids)
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
        messages = profile.build_messages(
            requested_segments,
            ids=expected_ids,
            ctx=_ctx(batch_index=batch_index),
        )
        missing_indexes: list[int] = []
        progress_callback = progress_callbacks[batch_index]
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
        retry_limit_for_pending = api_retries
        last_retry_error: RetryableTranslationFormatError | None = None
        # How many ids one request may ask for. Reissuing the shape that just
        # failed is what the retry budget used to buy: sample-b batch 24
        # (2026-08-13) asked for 54 ids, got 54 objects back numbered 1297-1350
        # instead of 1296-1349, and did that four times before failing the film.
        # A whole-batch id shift is a capacity symptom, so the budget now buys
        # SMALLER requests instead - the same move the ASR stage makes on OOM.
        # It descends only on failure and never grows back within a batch,
        # because whatever made the model lose the id sequence is still true.
        request_span_limit = max(1, len(expected_ids))

        def narrow_request_span() -> None:
            """Halve what the next request asks for, and say so."""
            nonlocal request_span_limit
            if request_span_limit <= 1:
                return
            previous = request_span_limit
            request_span_limit = max(1, request_span_limit // 2)
            emit_batch_diagnostic(
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

        while True:
            _raise_if_cancelled(worker_cancel)
            if request_count >= batch_max_requests:
                raise RuntimeError(
                    "Batch translation exceeded hard request cap "
                    f"({batch_max_requests}): batch={batch_index}, "
                    f"start_index={start_index}, size={expected_count}, "
                    f"pending_ids={pending_ids[:50]}, error={last_retry_error}"
                ) from last_retry_error
            if attempts_for_pending >= retry_limit_for_pending:
                raise RuntimeError(
                    "Batch translation returned invalid or incomplete JSON after "
                    f"{request_count} attempts: batch={batch_index}, "
                    f"start_index={start_index}, size={expected_count}, "
                    f"pending_ids={pending_ids[:50]}, error={last_retry_error}"
                ) from last_retry_error

            # One request covers at most `request_span_limit` of what is still
            # pending; the rest stays pending and the loop comes back for it via
            # the ordinary missing-ids path.
            requested_ids = list(pending_ids[:request_span_limit])
            pending_before_request = len(pending_ids)
            active_request_index = request_count
            active_requested_ids = list(requested_ids)
            trace_progress({"phase": "reset", "attempt": request_count})
            try:
                _raise_if_cancelled(worker_cancel)
                request_segments = [
                    batch_segments[index - start_index] for index in requested_ids
                ]
                if request_count == 0 and requested_ids == expected_ids:
                    request_messages = messages
                else:
                    request_messages = profile.build_messages(
                        request_segments,
                        ids=requested_ids,
                        ctx=_ctx(batch_index=batch_index),
                    )
                request_expected_count = len(request_segments)
                request_count += 1
                request_kwargs: dict = {}
                # Sized per request, not per batch: a partial reissue asks for
                # only the missing ids, so it must not carry the whole batch's
                # budget.
                token_budget = profile.response_token_budget(
                    request_segments,
                    reasoning_effort=reasoning_effort,
                )
                if token_budget is not None:
                    request_kwargs["max_tokens"] = token_budget
                bounded = profile.bounded_schema(request_segments)
                if bounded is not None:
                    request_kwargs["bounded_response_schema"] = bounded
                # Always explicit, including when it is None: the line contract
                # needs *no* schema, and a caller that simply omits the argument
                # gets the JSON one by default. Sending a grammar to Hy-MT2 is
                # the 152/300 failure, so "unset" and "none" must not collapse.
                request_kwargs["response_schema"] = profile.schema
                raw_output = chat(
                    request_messages,
                    expected_count=request_expected_count,
                    on_progress=trace_progress,
                    on_usage=request_usages.append,
                    cancel_event=worker_cancel,
                    **request_kwargs,
                )
                # No cancel check here on purpose. The reply is already
                # generated and already billed, so parsing it costs nothing the
                # run has not paid for, and a batch completed by this reply then
                # reaches `persist_batch` and survives. Standing down on the line
                # above threw that away and made the resume buy it a second time.
                # The abort is still honoured one parse later: an incomplete
                # batch loops back to the check at the top.
                parsed = profile.parse_response(raw_output, ids=requested_ids)
                for idx in requested_ids:
                    if parsed.get(idx):
                        # Convert only. The profiles already normalized this, and
                        # re-running the normalizer here could turn a truthy but
                        # degenerate reply into "" - which this loop's own
                        # `is None` missing-check would then count as answered.
                        batch_results[idx] = _to_target_variant(parsed[idx])
                missing_indexes = [
                    index
                    for index in all_batch_ids
                    if batch_results[index] is None
                ]
                if not missing_indexes:
                    break

                pending_ids = list(missing_indexes)
                last_retry_error = RetryableTranslationFormatError(
                    "translation returned incomplete batch translations: "
                    f"{len(missing_indexes)} missing of {expected_count}; "
                    f"missing ids={missing_indexes[:50]}"
                )
                # Progress is measured against what was still pending, not
                # against what this request asked for. With a narrowed span the
                # two differ, and comparing to the request would score a
                # perfectly good half-batch as a failed attempt.
                if (
                    len(missing_indexes) < pending_before_request
                    and profile.supports_partial_reissue
                ):
                    attempts_for_pending = 0
                    retry_limit_for_pending = batch_repair_retries
                else:
                    attempts_for_pending += 1
                    narrow_request_span()
            except ContentPolicyRefusalError as refusal:
                # Terminal by design (see the exception), so the only thing to
                # add is which cues were on the wire - without it the message
                # names a filter verdict and nothing to act on.
                raise ContentPolicyRefusalError(
                    f"{refusal} (batch={batch_index}, start_index={start_index}, "
                    f"size={expected_count}, requested_ids={requested_ids[:50]})"
                ) from refusal
            except RetryableTranslationFormatError as exc:
                last_retry_error = exc
                attempts_for_pending += 1
                narrow_request_span()

            if attempts_for_pending < retry_limit_for_pending:
                sleep_attempt = max(0, attempts_for_pending - 1)
                backoff_sleep(
                    sleep_attempt,
                    last_retry_error,
                    cancel_event=worker_cancel,
                )
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
            **transport_util._merge_usage_metrics(request_usages),
            "missing_count": len(missing_indexes),
            "missing_indexes": missing_indexes,
            "translation_memory_hit_count": expected_count - len(expected_ids),
            "cache_hit_type": "mixed" if len(expected_ids) < expected_count else "miss",
        }
        persist_batch(batch_index, batch_results)
        return batch_index, batch_results, timing, batch_retry_events

    def run_batch_guarded(
        batch_index: int, batch_segments: list[dict]
    ) -> tuple[int, list[str | None], dict, list[dict]]:
        nonlocal root_failure
        try:
            return run_batch(batch_index, batch_segments)
        except TranslationCancelledError:
            if batch_abort.is_set() and not _cancel_requested(cancel_event):
                # Standing down because a sibling already failed. That is a
                # consequence, not a reason, and the two must not look alike:
                # the main thread harvests batches in index order, so a low
                # sibling's cancellation would otherwise be the first exception
                # it sees and the run would report a cancellation for a failure
                # nobody could see.
                raise _SiblingBatchAborted(batch_index) from None
            raise
        except BaseException as exc:
            with failure_lock:
                if root_failure is None:
                    root_failure = exc
            # Raise the flag on the failing thread, not on the main one. The
            # worker that just failed is the very worker the pool hands the
            # next queued batch to, and it does that before the main thread
            # has even been scheduled to see the exception - which is how a
            # dead run started a fresh batch 1 ms after failing.
            batch_abort.set()
            raise

    if pending_batches:
        executor = ThreadPoolExecutor(max_workers=min(max_workers, len(pending_batches)))
        # Bound before the try: the cancel check below can raise before any
        # batch is submitted, and the handler walks this.
        futures: dict = {}
        try:
            _raise_if_cancelled(cancel_event)
            futures = {
                executor.submit(run_batch_guarded, batch_index, batch): batch_index
                for batch_index, batch in _submission_order(pending_batches)
            }
            def harvest(result, source_future) -> None:
                """Main-thread bookkeeping for a batch the worker already committed."""
                batch_index, _batch_results, timing, batch_retry_events = result
                worker_retry_events.extend(batch_retry_events)
                timings_by_batch[batch_index] = timing
                if cache_path and crash_probe() == batch_index + 1:
                    for pending_future in futures:
                        if pending_future is not source_future:
                            pending_future.cancel()
                    raise SystemExit(1)
                if on_batch_done:
                    _raise_if_cancelled(cancel_event)
                    on_batch_done(timing)

            remaining = set(futures)
            while remaining:
                _raise_if_cancelled(cancel_event)
                done, remaining = wait(
                    remaining,
                    timeout=0.1,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    continue
                # Successes first, failure last. Nothing is at stake for the
                # translations any more - each worker committed its own batch
                # before handing it back - but the timings and `on_batch_done`
                # of a batch that finished alongside a failure are still worth
                # recording before the run unwinds.
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
            # Order matters. `cancel()` only drops a future still queued, and
            # `shutdown(wait=False)` does not interrupt a worker already inside
            # a request - so on 2026-09-04 one batch failing left three others
            # streaming, and the worker it freed picked the next queued batch
            # off and started translating it 1 ms later. Both kept billing long
            # after the job was reported failed. The flag is what the workers
            # can actually see, so it goes first.
            batch_abort.set()
            for pending_future in futures:
                pending_future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            with failure_lock:
                root = root_failure
            if _cancel_requested(cancel_event) or root is None or root is exc:
                if isinstance(exc, _SiblingBatchAborted):
                    raise RuntimeError(
                        "a batch stood down for a sibling failure that was "
                        "never recorded"
                    ) from exc
                raise
            # `exc` is a sibling standing down, or a second batch that fell over
            # after the first. Either way it is downstream of `root`, and `root`
            # is the only one that says why the run ended.
            raise root
        else:
            executor.shutdown(wait=True)

    _raise_if_cancelled(cancel_event)
    missing = [
        idx for idx, value in enumerate(zh_texts) if value is None or value == ""
    ]
    if missing:
        raise RuntimeError(
            "Batched translation finished with missing translations: "
            f"{len(missing)} missing; ids={missing[:50]}"
        )

    timings: list[dict] = []
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
            "cache_hit_count": exact_cache_hit_count,
            "translation_memory_hit_count": translation_memory_hit_count,
            "prefix_mode": prefix_mode,
            "is_warmup": False,
            "requested_ids": [],
            "missing_count": 0,
            "missing_indexes": [],
        }
    )
    return [text or "" for text in zh_texts], timings, worker_retry_events
