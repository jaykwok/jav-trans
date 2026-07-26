# Translation repair logic

import json
import time
from typing import Callable

from llm.backends import get_backend


# 修复配置
TRANSLATION_REPAIR_MAX_IDS = 12
TRANSLATION_REPAIR_CONTEXT_RADIUS = 1
TRANSLATION_REPAIR_LENGTH_RATIO_MIN = 0.25
TRANSLATION_REPAIR_LENGTH_RATIO_MAX = 4.0


class TranslationCancelledError(RuntimeError):
    pass


def _cancel_requested(cancel_event) -> bool:
    try:
        return bool(cancel_event is not None and cancel_event.is_set())
    except Exception:
        return False


def _raise_if_cancelled(cancel_event) -> None:
    if _cancel_requested(cancel_event):
        raise TranslationCancelledError("任务已取消")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_source_text(text: str) -> str:
    import re
    cleaned = re.sub(r"[ \t]+", " ", (text or "").strip())
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n") if line.strip())
    return cleaned.strip()


def apply_translation_repair_pass(
    segments: list[dict],
    zh_texts: list[str],
    *,
    target_lang: str,
    glossary: str,
    character_reference: str,
    reasoning_effort: str | None = None,
    api_format: str | None = None,
    on_progress: Callable[[dict], None] | None = None,
    cancel_event=None,
    cache_writer: Callable[[list[str]], None] | None = None,
) -> tuple[list[str], dict | None]:
    """应用翻译修复 pass"""
    _raise_if_cancelled(cancel_event)

    repair_ids, reasons = _select_translation_repair_ids(segments, zh_texts)
    if not repair_ids:
        return zh_texts, None

    if TRANSLATION_REPAIR_MAX_IDS <= 0:
        return zh_texts, None

    repair_ids = repair_ids[:TRANSLATION_REPAIR_MAX_IDS]
    started = time.perf_counter()

    _emit_progress(
        on_progress,
        {
            "phase": "repair_start",
            "repair_ids": repair_ids,
            "candidate_count": len(repair_ids),
        },
    )

    try:
        _raise_if_cancelled(cancel_event)

        messages = _build_repair_messages(
            segments,
            zh_texts,
            repair_ids,
            reasons,
            target_lang=target_lang,
            glossary=glossary,
            character_reference=character_reference,
        )

        backend = get_backend()
        raw_output = backend.chat_completion(
            messages,
            temperature=0.6,
            top_p=0.9,
            max_tokens=16384,
            reasoning_effort=reasoning_effort,
            cancel_event=cancel_event,
            on_progress=on_progress,
        )

        _raise_if_cancelled(cancel_event)

        parsed = _parse_translation_output_by_global_id(
            raw_output,
            expected_ids=repair_ids,
            total_count=len(segments),
        )

        repaired_texts = list(zh_texts)
        repaired_count = 0
        for idx in repair_ids:
            _raise_if_cancelled(cancel_event)
            if parsed[idx]:
                repaired_texts[idx] = parsed[idx] or repaired_texts[idx]
                repaired_count += 1

        if repaired_count > 0 and cache_writer is not None:
            try:
                cache_writer(repaired_texts)
            except TranslationCancelledError:
                raise
            except Exception as exc:
                print(f"[WARN] failed to persist repaired translation cache: {exc}", flush=True)

        timing = {
            "mode": "translation_repair_pass",
            "start_index": min(repair_ids),
            "segment_count": repaired_count,
            "elapsed_s": time.perf_counter() - started,
            "request_count": 1,
            "repair_ids": repair_ids,
            "candidate_count": len(repair_ids),
            "missing_count": len(repair_ids) - repaired_count,
            "missing_indexes": [idx for idx in repair_ids if parsed[idx] is None],
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
        if isinstance(exc, TranslationCancelledError):
            raise

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
    """选择需要修复的翻译 ID"""
    repair_ids: list[int] = []
    reasons: dict[int, list[str]] = {}

    for idx, seg in enumerate(segments):
        source = _repair_source_text(seg)
        target = _repair_translation_text(seg, zh_texts, idx)
        local_reasons: list[str] = []

        if _has_translation_length_mismatch(source, target):
            local_reasons.append("length_mismatch")

        if not local_reasons:
            continue

        repair_ids.append(idx)
        reasons[idx] = list(dict.fromkeys(local_reasons))

    repair_ids.sort()
    return repair_ids, reasons


def _repair_source_text(seg: dict) -> str:
    return _normalize_source_text(
        seg.get("source") or seg.get("ja_text") or seg.get("text") or seg.get("ja") or ""
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
    """构建修复请求消息"""
    from llm.prompt import _build_system_prompt

    system_prompt = _build_system_prompt(
        character_reference,
        target_lang=target_lang,
        glossary=glossary,
    )
    system_prompt += (
        "\n\n这是翻译后局部修复任务。只修复 requested_ids 中的译文；"
        "只处理 reason 字段指出的译文长度异常，保持原字幕文本含义，不要根据上下文推测或改写源文。"
    )

    context_items = _build_repair_context_items(segments, zh_texts, repair_ids, reasons)

    user_content = "\n\n".join(
        [
            "【翻译修复任务】",
            f"requested_ids = {json.dumps(repair_ids, ensure_ascii=False)}",
            "只返回 requested_ids 中列出的 id，不要返回 context_only 项。",
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
    """构建修复上下文项"""
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
                "reason": reasons.get(idx, []),
                "start": _safe_float(seg.get("start")),
                "end": _safe_float(seg.get("end")),
                "ja": _repair_source_text(seg),
                "current_zh": _repair_translation_text(seg, zh_texts, idx),
            }
        )

    return items


def _parse_translation_output_by_global_id(
    raw_output: str,
    *,
    expected_ids: list[int],
    total_count: int,
) -> list[str | None]:
    """解析翻译输出（按全局 ID）"""
    import re

    # 移除 thinking block
    think_block_re = re.compile(r"<think>.*?</think>", re.S | re.I)
    raw_output = think_block_re.sub("", raw_output or "")

    if not raw_output.strip():
        return [None] * total_count

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return [None] * total_count

    if not isinstance(parsed, dict) or not isinstance(parsed.get("translations"), list):
        return [None] * total_count

    expected_id_set = set(expected_ids)
    results: list[str | None] = [None] * total_count
    seen_ids: set[int] = set()

    for item in parsed["translations"]:
        if not isinstance(item, dict):
            continue

        idx = item.get("id")
        if not isinstance(idx, int) or idx not in expected_id_set or idx >= total_count:
            continue

        if idx in seen_ids:
            continue

        text = str(item.get("text", "")).strip()
        if text:
            results[idx] = text
            seen_ids.add(idx)

    return results


def _emit_progress(on_progress: Callable[[dict], None] | None, payload: dict) -> None:
    """发送进度事件"""
    if on_progress is None:
        return
    try:
        on_progress(payload)
    except Exception:
        pass
