# Refactored translation module - unified entry point

import os
import time
import threading
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
    _compute_prompt_signature,
)
from llm.prompt import (
    _build_translation_messages,
    _serialize_segments,
    generate_global_context,
    PROMPT_VERSION,
)
from llm.repair import apply_translation_repair_pass
from llm.batching import translate_segments_batched


# 从环境变量读取配置
TRANSLATION_BATCH_SIZE = int(os.getenv("TRANSLATION_BATCH_SIZE", "64"))
DEFAULT_TARGET_LANG = "简体中文"


class TranslationCancelledError(RuntimeError):
    pass


class RetryableTranslationFormatError(RuntimeError):
    pass


def _cancel_requested(cancel_event: threading.Event | None) -> bool:
    try:
        return bool(cancel_event is not None and cancel_event.is_set())
    except Exception:
        return False


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if _cancel_requested(cancel_event):
        raise TranslationCancelledError("任务已取消")


def translate_segments(
    segments: list[dict],
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
    cancel_event: threading.Event | None = None,
) -> tuple[list[str], list[dict], list[dict]]:
    """
    翻译字幕段

    Args:
        segments: 字幕段列表，每个元素包含 text、start、end
        global_context: 全局上下文（可选，自动生成）
        max_workers: 最大并发线程数
        cache_path: 缓存路径
        target_lang: 目标语言
        glossary: 术语表
        character_reference: 人物参考名
        reasoning_effort: 推理强度（medium/xhigh）
        api_format: API 格式（chat/responses）
        on_batch_done: 批次完成回调
        on_progress: 进度回调
        cancel_event: 取消事件

    Returns:
        (翻译结果列表, 时间统计列表, 重试事件列表)
    """
    _raise_if_cancelled(cancel_event)

    if not segments:
        return [], [], []

    effective_max_workers = max(1, int(max_workers))
    effective_batch_size = _auto_translation_batch_size(len(segments), effective_max_workers)
    effective_cache_path = cache_path or ""
    effective_target_lang = (target_lang or DEFAULT_TARGET_LANG).strip() or DEFAULT_TARGET_LANG
    effective_glossary = _normalize_glossary_text(glossary)
    effective_character_reference = (character_reference or "").strip()

    # 提取全局术语表
    extra_glossary = ""
    if effective_cache_path:
        extra_glossary = _resolve_translation_extra_glossary(
            segments,
            effective_cache_path,
            effective_glossary,
            api_format=api_format,
            cancel_event=cancel_event,
        )

    retry_events: list[dict] = []

    # 决定翻译策略
    if effective_batch_size > 0 and len(segments) > effective_batch_size:
        # 批量翻译
        zh_texts, timings, worker_retry_events = translate_segments_batched(
            segments,
            batch_size=effective_batch_size,
            max_workers=effective_max_workers,
            global_context=global_context,
            cache_path=effective_cache_path,
            target_lang=effective_target_lang,
            glossary=effective_glossary,
            character_reference=effective_character_reference,
            extra_glossary=extra_glossary,
            reasoning_effort=reasoning_effort,
            api_format=api_format,
            on_batch_done=on_batch_done,
            on_progress=on_progress,
            cancel_event=cancel_event,
        )
        retry_events.extend(worker_retry_events)
    else:
        # 单次请求翻译
        zh_texts, timings = _translate_segments_single_request(
            segments,
            global_context=global_context,
            cache_path=effective_cache_path,
            target_lang=effective_target_lang,
            glossary=effective_glossary,
            character_reference=effective_character_reference,
            extra_glossary=extra_glossary,
            reasoning_effort=reasoning_effort,
            api_format=api_format,
            on_batch_done=on_batch_done,
            on_progress=on_progress,
            cancel_event=cancel_event,
        )

    _raise_if_cancelled(cancel_event)

    # 应用翻译修复 pass
    def _persist_repaired_translation_cache(repaired_texts: list[str]) -> None:
        """持久化修复后的翻译缓存"""
        if not effective_cache_path or not segments:
            return

        cache_kwargs = {
            "extra_glossary": extra_glossary,
            "glossary": effective_glossary,
            "target_lang": effective_target_lang,
            "character_reference": effective_character_reference,
        }

        if effective_batch_size > 0 and len(segments) > effective_batch_size:
            # 分批保存
            for b_index, b_segments in enumerate(_split_into_batches(segments, effective_batch_size)):
                start = b_index * effective_batch_size
                local_texts = [
                    repaired_texts[start + off] if start + off < len(repaired_texts) else ""
                    for off in range(len(b_segments))
                ]
                batch_key = _translation_cache_key(b_index, b_segments, **cache_kwargs)
                _save_cache_entry(effective_cache_path, batch_key, local_texts, threading.Lock())
        else:
            # 整体保存
            batch_key = _translation_cache_key(0, segments, **cache_kwargs)
            _save_cache_entry(effective_cache_path, batch_key, repaired_texts, threading.Lock())

    zh_texts, repair_timing = apply_translation_repair_pass(
        segments,
        zh_texts,
        target_lang=effective_target_lang,
        glossary=effective_glossary,
        character_reference=effective_character_reference,
        reasoning_effort=reasoning_effort,
        api_format=api_format,
        on_progress=on_progress,
        cancel_event=cancel_event,
        cache_writer=_persist_repaired_translation_cache,
    )

    _raise_if_cancelled(cancel_event)

    if repair_timing is not None:
        timings.append(repair_timing)

    return zh_texts, timings, retry_events


def _translate_segments_single_request(
    segments: list[dict],
    *,
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
    cancel_event: threading.Event | None = None,
) -> tuple[list[str], list[dict]]:
    """单次请求翻译所有字幕段"""
    _raise_if_cancelled(cancel_event)

    started = time.perf_counter()
    full_context = (
        global_context if global_context is not None else generate_global_context(segments)
    )
    source_payload = _serialize_segments(segments)
    expected_count = len(segments)

    # 检查缓存
    batch_key = ""
    translation_cache = _load_translation_cache(cache_path) if cache_path else {}
    translation_memory = _load_translation_memory(cache_path) if cache_path else {}

    if cache_path:
        batch_key = _translation_cache_key(
            0,
            segments,
            extra_glossary=extra_glossary,
            glossary=glossary,
            target_lang=target_lang,
            character_reference=character_reference,
        )

        cached_texts = translation_cache.get(batch_key)
        if isinstance(cached_texts, list) and len(cached_texts) == expected_count:
            timing = {
                "start_index": 0,
                "segment_count": expected_count,
                "elapsed_s": time.perf_counter() - started,
                "mode": "translation_cache_hit",
                "cache_hit": True,
            }
            if on_batch_done:
                on_batch_done(timing)
            return list(cached_texts), [timing]

        # 检查 translation memory
        memory_texts: list[str | None] = []
        memory_hit_count = 0
        for seg in segments:
            source_text = str(seg.get("text", ""))
            memory_text = None

            if _translation_memory_source_is_cacheable(source_text):
                memory_key = _translation_memory_key(
                    source_text,
                    extra_glossary,
                    glossary=glossary,
                    target_lang=target_lang,
                    character_reference=character_reference,
                )
                cached_memory_text = translation_memory.get(memory_key)
                if isinstance(cached_memory_text, str) and cached_memory_text.strip():
                    memory_text = cached_memory_text
                    memory_hit_count += 1

            memory_texts.append(memory_text)

        if memory_hit_count == expected_count:
            final_texts = [text or "" for text in memory_texts]
            _save_cache_entry(cache_path, batch_key, final_texts, threading.Lock())

            timing = {
                "start_index": 0,
                "segment_count": expected_count,
                "elapsed_s": time.perf_counter() - started,
                "mode": "translation_memory_hit",
                "cache_hit": True,
            }
            if on_batch_done:
                on_batch_done(timing)
            return final_texts, [timing]

    # 构建翻译请求
    messages = _build_translation_messages(
        source_payload=source_payload,
        expected_count=expected_count,
        extra_glossary=extra_glossary,
        target_lang=target_lang,
        glossary=glossary,
        character_reference=character_reference,
    )

    backend = get_backend()

    # 执行翻译（带重试）
    for attempt in range(4):
        _raise_if_cancelled(cancel_event)

        try:
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
            zh_texts = _parse_translation_output(raw_output, expected_count)
            missing_indexes = [i for i, text in enumerate(zh_texts) if not text]

            if not missing_indexes:
                break

        except Exception as exc:
            if attempt < 3:
                time.sleep(min(20.0, 1.5 * (2**attempt)))
                continue
            raise RuntimeError(f"Translation failed after {attempt + 1} attempts") from exc

    timing = {
        "start_index": 0,
        "segment_count": expected_count,
        "elapsed_s": time.perf_counter() - started,
        "mode": "single_request_full_context",
    }

    if on_batch_done:
        on_batch_done(timing)

    final_texts = [text or "" for text in zh_texts]

    # 保存缓存
    if cache_path and batch_key:
        _save_cache_entry(cache_path, batch_key, final_texts, threading.Lock())

        # 保存 translation memory
        memory_entries: list[tuple[str, str]] = []
        for seg, text in zip(segments, final_texts):
            source_text = str(seg.get("text", ""))
            if text and _translation_memory_source_is_cacheable(source_text):
                memory_key = _translation_memory_key(
                    source_text,
                    extra_glossary,
                    glossary=glossary,
                    target_lang=target_lang,
                    character_reference=character_reference,
                )
                memory_entries.append((memory_key, text))

        if memory_entries:
            _save_memory_entries(cache_path, memory_entries, threading.Lock())

    return final_texts, [timing]


def _auto_translation_batch_size(segment_count: int, max_workers: int) -> int:
    """自动计算批次大小"""
    count = max(0, int(segment_count))
    if count <= 0:
        return 0
    return min(count, TRANSLATION_BATCH_SIZE)


def _split_into_batches(segments: list[dict], batch_size: int) -> list[list[dict]]:
    """将字幕段切分成批次"""
    if not segments:
        return []
    if batch_size <= 0:
        return [segments]
    return [segments[i : i + batch_size] for i in range(0, len(segments), batch_size)]


def _normalize_glossary_text(text: str) -> str:
    """标准化术语表文本"""
    from llm.glossary import normalize_glossary_text
    return normalize_glossary_text(text)


def _resolve_translation_extra_glossary(
    segments: list[dict],
    cache_path: str,
    glossary: str,
    *,
    api_format: str | None,
    cancel_event: threading.Event | None,
) -> str:
    """提取全局术语表"""
    # 这个功能暂时保持简单实现，后续可以增强
    return ""


def _parse_translation_output(raw_output: str, expected_count: int) -> list[str | None]:
    """解析翻译输出"""
    import json
    import re

    # 移除 thinking block
    think_block_re = re.compile(r"<think>.*?</think>", re.S | re.I)
    raw_output = think_block_re.sub("", raw_output or "").strip()

    if not raw_output:
        raise RetryableTranslationFormatError("LLM returned empty content.")

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise RetryableTranslationFormatError("LLM response was not valid JSON.") from exc

    if not isinstance(parsed, dict) or not isinstance(parsed.get("translations"), list):
        raise RetryableTranslationFormatError(
            'LLM response must be {"translations":[...]} .'
        )

    translations = parsed["translations"]
    if len(translations) != expected_count:
        raise RetryableTranslationFormatError(
            f"LLM returned wrong translation count: {len(translations)} of {expected_count}."
        )

    results: list[str | None] = [None] * expected_count
    seen_ids: set[int] = set()

    for item in translations:
        if not isinstance(item, dict):
            raise RetryableTranslationFormatError("LLM translations must contain objects.")

        idx = item.get("id")
        if not isinstance(idx, int) or idx < 0 or idx >= expected_count:
            raise RetryableTranslationFormatError(f"LLM returned invalid translation id: {idx}.")

        if idx in seen_ids:
            raise RetryableTranslationFormatError(f"LLM returned duplicate translation id: {idx}.")

        seen_ids.add(idx)
        text = str(item.get("text", "")).strip()
        results[idx] = text if text else None

    return results


# 保持向后兼容的导出
__all__ = [
    "translate_segments",
    "generate_global_context",
    "PROMPT_VERSION",
    "TranslationCancelledError",
    "RetryableTranslationFormatError",
]
