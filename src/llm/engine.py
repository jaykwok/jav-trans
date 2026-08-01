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
from llm import settings as llm_settings
from llm import transport_util
from llm.errors import RetryableTranslationFormatError, TranslationCancelledError
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


def _request_kwargs(profile: TranslationProfile, backend, batch_size: int) -> dict:
    """Sampling/schema kwargs gated on backend capabilities."""
    kwargs: dict = {}
    sampling = profile.sampling(batch_size) or {}
    for key in ("temperature", "top_p", "max_tokens"):
        if key in sampling:
            kwargs[key] = sampling[key]
    supports_schema = False
    try:
        supports_schema = bool(backend.supports_json_schema())
    except Exception:
        supports_schema = False
    if profile.schema is not None and supports_schema:
        kwargs["response_format"] = profile.schema
    return kwargs


def run_line_profile(
    segments: list[dict],
    *,
    profile: TranslationProfile,
    backend,
    batch_size: int,
    shard_limit: int,
    cache_path: str,
    target_lang: str,
    glossary: str,
    character_reference: str,
    cache_lock: threading.Lock,
    on_batch_done=None,
    on_progress: Callable[[dict], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[list[str], list[dict]]:
    """Line-oriented loop for history-carrying profiles (Sakura/GalTransl).

    N source lines in, N translated lines out, with a rolling history block
    for pronoun continuity. Batches are grouped into contiguous shards; shards
    run in parallel, batches inside a shard run in order so the history is
    real previous output rather than whatever thread finished first. The
    auto-extracted extra glossary is skipped on purpose -- extracting it needs
    the JSON contract, which these model families cannot speak.
    """
    normalize = _normalize_text
    history_limit = int(profile.history_limit or 0)
    batches = _split_into_batches(segments, batch_size)
    expected_total = len(segments)
    progress_callbacks, _ = _make_aggregated_progress_callback(
        len(batches),
        expected_total,
        on_progress,
    )
    cache_map = translation_cache._load_translation_cache(cache_path) if cache_path else {}
    memory_map = translation_cache._load_translation_memory(cache_path) if cache_path else {}
    zh_texts: list[str | None] = [None] * expected_total
    timings_by_batch: dict[int, dict] = {}
    timings_lock = threading.Lock()

    prompt_version = profile.cache_signature()
    model_identity = backend.cache_identity()

    def _memory_key_for(source_text: str) -> str:
        return translation_cache._translation_memory_key(
            source_text,
            "",
            glossary=glossary,
            target_lang=target_lang,
            character_reference=character_reference,
            prompt_version=prompt_version,
            model_name=model_identity,
        )

    def _ctx(history: list[str]) -> ProfileContext:
        return ProfileContext(
            target_lang=target_lang,
            glossary=glossary,
            character_reference=character_reference,
            history=tuple(history),
            total_count=expected_total,
        )

    def request_lines(
        line_segments: list[dict],
        ids: list[int],
        history: list[str],
        usages: list[dict],
    ) -> list[str]:
        messages = profile.build_messages(line_segments, ids=ids, ctx=_ctx(history))
        format_attempts = 0
        api_attempts = 0
        while True:
            _raise_if_cancelled(cancel_event)
            try:
                reply = backend.chat_completion(
                    messages,
                    expected_count=len(ids),
                    cancel_event=cancel_event,
                    on_usage=usages.append,
                    **_request_kwargs(profile, backend, len(ids)),
                )
                by_id = profile.parse_response(reply, ids=ids)
                return [str(by_id.get(idx) or "") for idx in ids]
            except RetryableTranslationFormatError:
                format_attempts += 1
                if format_attempts >= 2:
                    raise
            except Exception as exc:
                if not transport_util._is_retryable_api_error(exc):
                    raise
                api_attempts += 1
                if api_attempts > llm_settings.TRANSLATION_API_RETRIES:
                    raise
                transport_util._interruptible_sleep(
                    transport_util._request_backoff_delay(api_attempts), cancel_event
                )

    def translate_single_line(
        segment: dict,
        seg_id: int,
        history: list[str],
        usages: list[dict],
    ) -> str:
        try:
            return request_lines([segment], [seg_id], history, usages)[0]
        except RetryableTranslationFormatError:
            # Stubborn multi-line reply for a single line: keep the first
            # non-empty line instead of failing the whole job.
            messages = profile.build_messages([segment], ids=[seg_id], ctx=_ctx(history))
            reply = backend.chat_completion(
                messages,
                expected_count=1,
                cancel_event=cancel_event,
                on_usage=usages.append,
                **{**_request_kwargs(profile, backend, 1), "max_tokens": 1024},
            )
            for candidate in str(reply or "").splitlines():
                if candidate.strip():
                    return candidate.strip()
            return ""

    def run_shard(shard_batches: list[tuple[int, list[dict]]]) -> None:
        history: list[str] = []
        for batch_index, batch_segments in shard_batches:
            _raise_if_cancelled(cancel_event)
            batch_started = time.perf_counter()
            start_index = batch_index * batch_size
            ids = list(range(start_index, start_index + len(batch_segments)))
            batch_key = translation_cache._translation_cache_key(
                batch_index,
                batch_segments,
                extra_glossary="",
                glossary=glossary,
                target_lang=target_lang,
                character_reference=character_reference,
                prompt_version=prompt_version,
                model_name=model_identity,
                compact_system_prompt=llm_settings.COMPACT_SYSTEM_PROMPT,
            )
            usages: list[dict] = []
            request_count = 0
            memory_hits = 0
            cached_texts = cache_map.get(batch_key)
            if isinstance(cached_texts, list) and len(cached_texts) == len(batch_segments):
                final_texts = [normalize(text) or "" for text in cached_texts]
                mode = "translation_cache_hit"
            else:
                sources = [str(seg.get("text", "")) for seg in batch_segments]
                pending: list[str | None] = [None] * len(sources)
                for offset, source_text in enumerate(sources):
                    if not cache_path or not translation_cache._translation_memory_source_is_cacheable(
                        source_text
                    ):
                        continue
                    memory_text = memory_map.get(_memory_key_for(source_text))
                    if isinstance(memory_text, str) and memory_text.strip():
                        pending[offset] = normalize(memory_text) or ""
                        memory_hits += 1
                missing = [i for i, text in enumerate(pending) if text is None]
                if missing:
                    request_count += 1
                    try:
                        translated = request_lines(
                            [batch_segments[i] for i in missing],
                            [ids[i] for i in missing],
                            list(history),
                            usages,
                        )
                    except RetryableTranslationFormatError:
                        translated = []
                        for i in missing:
                            request_count += 1
                            translated.append(
                                translate_single_line(
                                    batch_segments[i], ids[i], list(history), usages
                                )
                            )
                    for i, text in zip(missing, translated):
                        pending[i] = normalize(text) or ""
                final_texts = [text or "" for text in pending]
                # 1:1 invariant: the cue plan is frozen before translation, so
                # every batch must produce exactly one text per input line.
                if len(final_texts) != len(batch_segments):
                    raise RuntimeError(
                        "translation profile violated the 1:1 line contract: "
                        f"{len(final_texts)} texts for {len(batch_segments)} lines"
                    )
                mode = "line_batch"
                translation_cache._save_cache_entry(
                    cache_path, batch_key, final_texts, cache_lock
                )
                if cache_path:
                    memory_entries = [
                        (_memory_key_for(str(seg.get("text", ""))), text)
                        for seg, text in zip(batch_segments, final_texts)
                        if text
                        and translation_cache._translation_memory_source_is_cacheable(
                            str(seg.get("text", ""))
                        )
                    ]
                    translation_cache._save_memory_entries(
                        cache_path, memory_entries, cache_lock
                    )
            for offset, text in enumerate(final_texts):
                zh_texts[start_index + offset] = text
            if history_limit > 0:
                history.extend(text for text in final_texts if text)
                del history[:-history_limit]
            timing = {
                "batch_index": batch_index,
                "start_index": start_index,
                "segment_count": len(batch_segments),
                "elapsed_s": round(time.perf_counter() - batch_started, 3),
                "mode": mode,
                "prompt_profile": profile.cache_signature(),
                "request_count": request_count,
                "cache_hit": mode == "translation_cache_hit",
                "translation_memory_hit_count": memory_hits,
                **transport_util._merge_usage_metrics(usages),
            }
            with timings_lock:
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

    indexed_batches = list(enumerate(batches))
    shard_count = max(1, min(int(shard_limit), len(indexed_batches))) if indexed_batches else 0
    if not indexed_batches:
        return [], []
    per_shard = (len(indexed_batches) + shard_count - 1) // shard_count
    shards = [
        indexed_batches[i * per_shard : (i + 1) * per_shard]
        for i in range(shard_count)
    ]
    shards = [shard for shard in shards if shard]
    if len(shards) == 1:
        run_shard(shards[0])
    else:
        with ThreadPoolExecutor(max_workers=len(shards)) as pool:
            futures = [pool.submit(run_shard, shard) for shard in shards]
            for future in futures:
                future.result()

    ordered_timings = [timings_by_batch[i] for i in sorted(timings_by_batch)]
    return [text if text is not None else "" for text in zh_texts], ordered_timings


def _normalize_text(text) -> str | None:
    # Line profiles reuse the JSON contract's output normalizer so cached and
    # fresh texts go through identical cleanup.
    from llm.profiles import json_v3

    return json_v3._normalize_translation_text(text)


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
                zh_texts[start_index + offset] = _normalize_text(text) or ""
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
                zh_texts[global_index] = _normalize_text(memory_text) or ""
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

    def run_batch(
        batch_index: int, batch_segments: list[dict]
    ) -> tuple[int, list[str | None], dict, list[dict]]:
        _raise_if_cancelled(cancel_event)
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
        pending_segments = list(requested_segments)
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

        while True:
            _raise_if_cancelled(cancel_event)
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

            requested_ids = list(pending_ids)
            active_request_index = request_count
            active_requested_ids = list(requested_ids)
            trace_progress({"phase": "reset", "attempt": request_count})
            try:
                _raise_if_cancelled(cancel_event)
                if request_count == 0:
                    request_messages = messages
                    request_expected_count = len(requested_segments)
                else:
                    request_expected_count = len(pending_segments)
                    request_messages = profile.build_messages(
                        pending_segments,
                        ids=pending_ids,
                        ctx=_ctx(batch_index=batch_index),
                    )
                request_count += 1
                raw_output = chat(
                    request_messages,
                    expected_count=request_expected_count,
                    on_progress=trace_progress,
                    on_usage=request_usages.append,
                    cancel_event=cancel_event,
                )
                _raise_if_cancelled(cancel_event)
                parsed = profile.parse_response(raw_output, ids=pending_ids)
                for idx in pending_ids:
                    if parsed.get(idx):
                        batch_results[idx] = parsed[idx]
                missing_indexes = [
                    index
                    for index in all_batch_ids
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
                    "translation returned incomplete batch translations: "
                    f"{len(missing_indexes)} missing of {expected_count}; "
                    f"missing ids={missing_indexes[:50]}"
                )
                if len(missing_indexes) < len(requested_ids) and profile.supports_partial_reissue:
                    attempts_for_pending = 0
                    retry_limit_for_pending = batch_repair_retries
                else:
                    attempts_for_pending += 1
            except RetryableTranslationFormatError as exc:
                last_retry_error = exc
                attempts_for_pending += 1

            if attempts_for_pending < retry_limit_for_pending:
                sleep_attempt = max(0, attempts_for_pending - 1)
                backoff_sleep(
                    sleep_attempt,
                    last_retry_error,
                    cancel_event=cancel_event,
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
        return batch_index, batch_results, timing, batch_retry_events

    if pending_batches:
        executor = ThreadPoolExecutor(max_workers=min(max_workers, len(pending_batches)))
        try:
            _raise_if_cancelled(cancel_event)
            pending_by_index = {
                batch_index: batch for batch_index, batch in pending_batches
            }
            futures = {
                executor.submit(run_batch, batch_index, batch): batch_index
                for batch_index, batch in pending_batches
            }
            try:
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
                    for future in sorted(done, key=lambda item: futures[item]):
                        _raise_if_cancelled(cancel_event)
                        batch_index, batch_results, timing, batch_retry_events = future.result()
                        worker_retry_events.extend(batch_retry_events)
                        timings_by_batch[batch_index] = timing
                        start_index = int(timing["start_index"])
                        segment_count = int(timing["segment_count"])
                        local_texts: list[str] = []
                        memory_entries: list[tuple[str, str]] = []
                        for offset in range(segment_count):
                            global_index = start_index + offset
                            text = batch_results[global_index] or zh_texts[global_index] or ""
                            zh_texts[global_index] = text
                            local_texts.append(text)
                            source_text = str(
                                pending_by_index[batch_index][offset].get("text", "")
                            )
                            if (
                                cache_path
                                and text
                                and translation_cache._translation_memory_source_is_cacheable(
                                    source_text
                                )
                            ):
                                memory_entries.append((_memory_key_for(source_text), text))
                        if cache_path:
                            batch_key = _batch_key_for(
                                batch_index,
                                pending_by_index[batch_index],
                            )
                            translation_cache._save_cache_entry(
                                cache_path,
                                batch_key,
                                local_texts,
                                cache_lock,
                            )
                            cache_map[batch_key] = local_texts
                            if memory_entries:
                                translation_cache._save_memory_entries(
                                    cache_path,
                                    memory_entries,
                                    cache_lock,
                                )
                                for memory_key, memory_text in memory_entries:
                                    memory_map[memory_key] = memory_text
                            print(f"[translation-cache] saved batch {batch_index} cache_key={batch_key}")
                            if crash_probe() == batch_index + 1:
                                for pending_future in futures:
                                    if pending_future is not future:
                                        pending_future.cancel()
                                raise SystemExit(1)
                        if on_batch_done:
                            _raise_if_cancelled(cancel_event)
                            on_batch_done(timing)
            except BaseException:
                raise
        except BaseException:
            for pending_future in futures:
                pending_future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
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
