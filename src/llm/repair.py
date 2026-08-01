"""Post-translation repair pass (JSON contract).

Optional stage a profile opts into via ``wants_repair_pass``: scan the
finished translation for suspicious lines (currently a single detector:
target/source length ratio out of band), then reissue only those ids with a
few lines of local context. The repair prompt and reply share the JSON
``{"translations":[...]}`` contract, so this stage only runs for profiles
that speak it. The ``chat`` callable comes from the caller — transport
dispatch and reasoning/api-format handling stay outside.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Callable

from llm import prompt as prompt_module
from llm import settings as llm_settings
from llm import transport_util
from llm.errors import TranslationCancelledError
from llm.profiles import json_v3

_raise_if_cancelled = transport_util._raise_if_cancelled
_emit_progress = transport_util._emit_progress


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def apply_repair_pass(
    segments: list[dict],
    zh_texts: list[str],
    *,
    chat: Callable[..., str],
    target_lang: str,
    glossary: str,
    character_reference: str,
    on_progress: Callable[[dict], None] | None = None,
    cancel_event: threading.Event | None = None,
    cache_writer: Callable[[list[str]], None] | None = None,
) -> tuple[list[str], dict | None]:
    _raise_if_cancelled(cancel_event)
    repair_ids, reasons = _select_translation_repair_ids(segments, zh_texts)
    if not repair_ids:
        return zh_texts, None

    if llm_settings.TRANSLATION_REPAIR_MAX_IDS <= 0:
        return zh_texts, None
    repair_ids = repair_ids[: llm_settings.TRANSLATION_REPAIR_MAX_IDS]
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
        raw_output = chat(
            messages,
            expected_count=len(repair_ids),
            on_usage=request_usages.append,
            cancel_event=cancel_event,
        )
        _raise_if_cancelled(cancel_event)
        parsed = json_v3._parse_translation_output_by_global_id(
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
            # Persist repaired texts back into the translation cache so a re-run
            # doesn't pay for the same repair again. Idempotent: _save_cache_entry
            # appends and reads dedupe by last-write-wins per batch_key.
            try:
                cache_writer(repaired_texts)
            except TranslationCancelledError:
                raise
            except Exception as exc:
                print(
                    f"[WARN] failed to persist repaired translation cache: {exc}",
                    flush=True,
                )
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
            **transport_util._merge_usage_metrics(request_usages),
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
            **transport_util._merge_usage_metrics(request_usages),
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
    return prompt_module._normalize_source_text(
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
        ratio < llm_settings.TRANSLATION_REPAIR_LENGTH_RATIO_MIN
        or ratio > llm_settings.TRANSLATION_REPAIR_LENGTH_RATIO_MAX
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
    system_prompt = prompt_module._build_system_prompt(
        character_reference,
        target_lang=target_lang,
        glossary=glossary,
    )
    system_prompt += (
        "\n\n这是翻译后局部修复任务。只修复 requested_ids 中的译文；"
        "只处理 reason 字段指出的译文长度异常，保持原字幕文本含义，不要根据上下文推测或改写源文。"
        "reason 只是问题类别提示，不是固定译文；最终译文必须服从原文和既定术语。"
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
            f"requested_ids = {prompt_module._format_requested_ids(repair_ids)}",
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
    radius = llm_settings.TRANSLATION_REPAIR_CONTEXT_RADIUS
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
        if reason == "length_mismatch":
            public.append("length_mismatch")
        else:
            public.append("translation_quality")
    return list(dict.fromkeys(public))
