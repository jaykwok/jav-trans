# Translation batching and concurrency logic

import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable

from llm.backends import get_backend
from llm.cache import (
    _load_translation_cache,
    _load_translation_memory,
    _save_cache_entry,
    _save_memory_entries,
    _translation_cache_key,
    _translation_memory_key,
    _translation_memory_source_is_cacheable,
)
from llm.prompt import (
    _build_batch_messages,
    _serialize_segments,
    generate_global_context,
)


TRANSLATION_BATCH_SIZE = 64
TRANSLATION_API_RETRIES = 4
TRANSLATION_BATCH_REPAIR_RETRIES = 2
TRANSLATION_BATCH_MAX_REQUESTS = 12
COMPACT_SYSTEM_PROMPT = False


class TranslationCancelledError(RuntimeError):
    pass


class RetryableTranslationFormatError(RuntimeError):
    pass


def _cancel_requested(cancel_event) -> bool:
    try:
        return bool(cancel_event is not None and cancel_event.is_set())
    except Exception:
        return False


def _raise_if_cancelled(cancel_event) -> None:
    if _cancel_requested(cancel_event):
        raise TranslationCancelledError("任务已取消")


def translate_segments_batched(
    segments: list[dict],
    *,
    batch_size: int,
    max_workers: int,
    global_context: str | None = None,
    cache_path: str = "",
    target_lang: str,
    glossary: str,
    character_reference: str,
    extra_glossary: str = "",
    reasoning_effort: str | None = None,
    api_format: str | None = None,
    on_batch_done=None,
    on_progress: Callable[[dict], None] | None = None,
    cancel_event=None,
) -> tuple[list[str], list[dict], list[dict]]:
    """批量翻译字幕段"""
    _raise_if_cancelled(cancel_event)

    started = time.perf_counter()
    batches = _split_into_batches(segments, batch_size)
    expected_total = len(segments)

    full_context = (
        global_context if global_context is not None else generate_global_context(segments)
    )

    full_source_payload = _serialize_segments(segments, compact=True)
    use_full_json_prefix = len(full_source_payload) <= 180000

    zh_texts: list[str | None] = [None] * expected_total
    timings_by_batch: dict[int, dict] = {}
    worker_retry_events: list[dict] = []

    translation_cache = _load_translation_cache(cache_path) if cache_path else {}
    translation_memory = _load_translation_memory(cache_path) if cache_path else {}

    pending_batches: list[tuple[int, list[dict]]] = []
    exact_cache_hit_count = 0
    translation_memory_hit_count = 0
    _cache_lock = threading.Lock()

    # 第一遍：检查缓存命中
    for batch_index, batch_segments in enumerate(batches):
        _raise_if_cancelled(cancel_event)
        start_index = batch_index * batch_size

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
            exact_cache_hit_count += 1
            for offset, text in enumerate(cached_texts):
                zh_texts[start_index + offset] = text or ""

            timing = {
                "batch_index": batch_index,
                "start_index": start_index,
                "segment_count": len(batch_segments),
                "elapsed_s": 0.0,
                "mode": "translation_cache_hit",
                "cache_hit": True,
            }
            timings_by_batch[batch_index] = timing
            if on_batch_done:
                on_batch_done(timing)
        else:
            # 检查 translation memory
            memory_hit_ids: list[int] = []
            for offset, seg in enumerate(batch_segments):
                source_text = str(seg.get("text", ""))
                if not cache_path or not _translation_memory_source_is_cacheable(source_text):
                    continue

                memory_key = _translation_memory_key(
                    source_text,
                    extra_glossary,
                    glossary=glossary,
                    target_lang=target_lang,
                    character_reference=character_reference,
                )
                memory_text = translation_memory.get(memory_key)
                if isinstance(memory_text, str) and memory_text.strip():
                    global_index = start_index + offset
                    zh_texts[global_index] = memory_text or ""
                    memory_hit_ids.append(global_index)

            if len(memory_hit_ids) == len(batch_segments):
                translation_memory_hit_count += len(memory_hit_ids)
                local_texts = [
                    zh_texts[start_index + offset] or "" for offset in range(len(batch_segments))
                ]
                if cache_path:
                    _save_cache_entry(cache_path, batch_key, local_texts, _cache_lock)

                timing = {
                    "batch_index": batch_index,
                    "start_index": start_index,
                    "segment_count": len(batch_segments),
                    "elapsed_s": 0.0,
                    "mode": "translation_memory_hit",
                    "cache_hit": True,
                }
                timings_by_batch[batch_index] = timing
                if on_batch_done:
                    on_batch_done(timing)
            else:
                pending_batches.append((batch_index, batch_segments))

    # 第二遍：并发翻译未命中批次
    if pending_batches:
        backend = get_backend()

        def run_batch(batch_index: int, batch_segments: list[dict]) -> tuple[int, list[str], dict]:
            """执行单个批次翻译"""
            _raise_if_cancelled(cancel_event)
            batch_started = time.perf_counter()
            start_index = batch_index * batch_size

            # 过滤已有 memory 命中的
            requested_segments: list[dict] = []
            expected_ids: list[int] = []
            for offset, seg in enumerate(batch_segments):
                global_index = start_index + offset
                if zh_texts[global_index] is None:
                    requested_segments.append(seg)
                    expected_ids.append(global_index)

            source_payload = _serialize_segments(requested_segments, explicit_ids=expected_ids)

            messages = _build_batch_messages(
                requested_segments,
                full_context,
                character_reference,
                len(requested_segments),
                batch_index=batch_index,
                extra_glossary=extra_glossary,
                target_lang=target_lang,
                glossary=glossary,
                source_payload_override=source_payload,
                full_source_payload=full_source_payload if use_full_json_prefix else None,
                requested_ids=expected_ids,
                compact_system_prompt_enabled=COMPACT_SYSTEM_PROMPT,
            )

            # 重试循环
            batch_results: list[str | None] = [None] * expected_total
            request_count = 0
            last_error = None

            for attempt in range(TRANSLATION_API_RETRIES):
                _raise_if_cancelled(cancel_event)
                if request_count >= TRANSLATION_BATCH_MAX_REQUESTS:
                    break

                try:
                    request_count += 1
                    raw_output = backend.chat_completion(
                        messages,
                        temperature=0.6,
                        top_p=0.9,
                        max_tokens=384000,
                        reasoning_effort=reasoning_effort,
                        cancel_event=cancel_event,
                        on_progress=on_progress,
                    )

                    _raise_if_cancelled(cancel_event)
                    parsed = _parse_translation_output(raw_output, len(requested_segments))

                    for idx, text in zip(expected_ids, parsed):
                        if text:
                            batch_results[idx] = text

                    break

                except Exception as exc:
                    last_error = exc
                    if attempt < TRANSLATION_API_RETRIES - 1:
                        time.sleep(min(20.0, 1.5 * (2**attempt)))

            if last_error and all(batch_results[i] is None for i in expected_ids):
                raise RuntimeError(
                    f"Batch {batch_index} translation failed after {request_count} attempts"
                ) from last_error

            # 填充结果
            local_texts: list[str] = []
            for offset in range(len(batch_segments)):
                global_index = start_index + offset
                text = batch_results[global_index] or zh_texts[global_index] or ""
                local_texts.append(text)

            timing = {
                "batch_index": batch_index,
                "start_index": start_index,
                "segment_count": len(batch_segments),
                "elapsed_s": time.perf_counter() - batch_started,
                "mode": "batched_translation",
                "request_count": request_count,
            }

            return batch_index, local_texts, timing

        # 并发执行
        executor = ThreadPoolExecutor(max_workers=min(max_workers, len(pending_batches)))
        try:
            futures = {
                executor.submit(run_batch, batch_index, batch): batch_index
                for batch_index, batch in pending_batches
            }

            remaining = set(futures)
            while remaining:
                _raise_if_cancelled(cancel_event)
                done, remaining = wait(remaining, timeout=0.1, return_when=FIRST_COMPLETED)

                for future in done:
                    batch_index, local_texts, timing = future.result()
                    timings_by_batch[batch_index] = timing

                    start_index = batch_index * batch_size
                    for offset, text in enumerate(local_texts):
                        zh_texts[start_index + offset] = text

                    # 保存缓存
                    if cache_path:
                        batch_key = _translation_cache_key(
                            batch_index,
                            [batch for idx, batch in pending_batches if idx == batch_index][0],
                            extra_glossary=extra_glossary,
                            glossary=glossary,
                            target_lang=target_lang,
                            character_reference=character_reference,
                        )
                        _save_cache_entry(cache_path, batch_key, local_texts, _cache_lock)

                    if on_batch_done:
                        on_batch_done(timing)

        finally:
            executor.shutdown(wait=True)

    # 汇总
    timings = [timings_by_batch[i] for i in sorted(timings_by_batch)]
    timings.append(
        {
            "mode": "batched_total",
            "segment_count": expected_total,
            "elapsed_s": time.perf_counter() - started,
            "cache_hit_count": exact_cache_hit_count,
            "translation_memory_hit_count": translation_memory_hit_count,
        }
    )

    return [text or "" for text in zh_texts], timings, worker_retry_events


def _split_into_batches(segments: list[dict], batch_size: int) -> list[list[dict]]:
    """将字幕段切分成批次"""
    if not segments:
        return []
    if batch_size <= 0:
        return [segments]
    return [segments[i : i + batch_size] for i in range(0, len(segments), batch_size)]


def _parse_translation_output(raw_output: str, expected_count: int) -> list[str | None]:
    """解析翻译输出"""
    import json
    import re

    # 移除 thinking block
    think_block_re = re.compile(r"<think>.*?</think>", re.S | re.I)
    raw_output = think_block_re.sub("", raw_output or "").strip()

    if not raw_output:
        return [None] * expected_count

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return [None] * expected_count

    if not isinstance(parsed, dict) or not isinstance(parsed.get("translations"), list):
        return [None] * expected_count

    translations = parsed["translations"]
    results: list[str | None] = [None] * expected_count

    for item in translations:
        if not isinstance(item, dict):
            continue

        idx = item.get("id")
        if not isinstance(idx, int) or idx < 0 or idx >= expected_count:
            continue

        text = str(item.get("text", "")).strip()
        if text:
            results[idx] = text

    return results
