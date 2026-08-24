"""Post-translation repair pass (JSON contract).

Optional stage a profile opts into via ``wants_repair_pass``: scan the finished
translation for lines a local detector can call suspicious - source echo,
residual Japanese kana, a glossary term the translation did not use,
target/source length ratio out of band - then reissue only those ids, with a few
lines of local context and one tier more thinking than the base pass used.

That escalation is the second half of the cost cascade. Reasoning is a
per-request cost that barely scales with batch size, so buying it for a whole
film is the expensive way to fix the minority of lines that actually need it;
buying it only for flagged ids is the cheap way. It is also what makes the
``none`` tier usable at all: thinking-off is fast and correct on most lines and
echoes the Japanese source on some, which is exactly a detector's job.

The detector set is what bounds how cheap the base pass can safely be: a tier
only saves money if everything it gets wrong is something a detector can see.

The repair prompt and reply share the JSON ``{"translations":[...]}`` contract,
so this stage only runs for profiles that speak it. The ``chat`` callable comes
from the caller — transport dispatch and api-format handling stay outside.
"""

from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from typing import Callable

from llm import prompt as prompt_module
from llm import settings as llm_settings
from llm import transport_util
from llm.glossary import parse_glossary_pairs
from llm.errors import (
    ResponseTruncatedError,
    RetryableTranslationFormatError,
    TranslationCancelledError,
)
from llm.profiles import json_v3
from llm.profiles.base import TranslationProfile

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
    profile: TranslationProfile,
    batch_size: int,
    reasoning_effort: str,
    target_lang: str,
    glossary: str,
    character_reference: str,
    on_progress: Callable[[dict], None] | None = None,
    cancel_event: threading.Event | None = None,
    cache_writer: Callable[[list[str]], None] | None = None,
) -> tuple[list[str], dict | None]:
    """Reissue flagged ids at the repair tier, by default one above the base.

    `reasoning_effort` is the base pass's tier, not this pass's; the repair tier
    is derived from it so the two can never be configured out of step, unless
    `TRANSLATION_REPAIR_REASONING_EFFORT` pins it.

    Candidate ids are collected across the whole film before any request goes
    out. That aggregation is the cost property: a detector firing once in each
    base batch must not buy one fixed reasoning preamble per batch. Invalid or
    truncated replies split the group in half, which reuses the engine's "ask
    for less" recovery rule instead of reissuing an identical bad shape.
    """
    _raise_if_cancelled(cancel_event)
    repair_ids, reasons = _select_translation_repair_ids(segments, zh_texts, glossary)
    if not repair_ids:
        return zh_texts, None
    if llm_settings.TRANSLATION_REPAIR_MAX_IDS <= 0:
        return zh_texts, None
    repair_ids = repair_ids[: llm_settings.TRANSLATION_REPAIR_MAX_IDS]

    started = time.perf_counter()
    repaired_texts = list(zh_texts)
    request_usages: list[dict] = []
    repaired_ids: set[int] = set()
    format_split_count = 0
    request_count = 0
    request_cap = max(1, int(llm_settings.TRANSLATION_BATCH_MAX_REQUESTS))
    initial_span = max(1, int(batch_size))
    repair_effort = llm_settings._repair_reasoning_effort(reasoning_effort)

    _emit_progress(
        on_progress,
        {
            "phase": "repair_start",
            "repair_ids": repair_ids,
            "candidate_count": len(repair_ids),
            "reasoning_effort": repair_effort,
        },
    )

    def request_group(group: list[int]) -> list[int]:
        nonlocal request_count, format_split_count
        if not group:
            return []
        if request_count >= request_cap:
            return list(group)
        _raise_if_cancelled(cancel_event)
        request_segments = [segments[idx] for idx in group]
        messages = _build_repair_messages(
            segments,
            repaired_texts,
            group,
            reasons,
            target_lang=target_lang,
            glossary=glossary,
            character_reference=character_reference,
        )
        request_kwargs: dict = {
            "response_schema": profile.schema,
            "reasoning_effort": repair_effort,
        }
        token_budget = profile.response_token_budget(
            request_segments,
            reasoning_effort=repair_effort,
        )
        if token_budget is not None:
            request_kwargs["max_tokens"] = token_budget
        bounded = profile.bounded_schema(request_segments)
        if bounded is not None:
            request_kwargs["bounded_response_schema"] = bounded

        request_count += 1
        try:
            raw_output = chat(
                messages,
                expected_count=len(group),
                on_usage=request_usages.append,
                cancel_event=cancel_event,
                **request_kwargs,
            )
            parsed = json_v3._parse_partial_translation_output_by_global_id(
                raw_output,
                expected_ids=group,
                total_count=len(segments),
            )
        except (RetryableTranslationFormatError, ResponseTruncatedError):
            if len(group) <= 1:
                return list(group)
            format_split_count += 1
            midpoint = max(1, len(group) // 2)
            return request_group(group[:midpoint]) + request_group(group[midpoint:])

        missing: list[int] = []
        for idx in group:
            _raise_if_cancelled(cancel_event)
            text = parsed[idx]
            if text:
                repaired_texts[idx] = text
                repaired_ids.add(idx)
            else:
                missing.append(idx)
        if not missing:
            return []
        if len(missing) == len(group) and len(group) > 1:
            format_split_count += 1
            midpoint = max(1, len(group) // 2)
            return request_group(group[:midpoint]) + request_group(group[midpoint:])
        return request_group(missing)

    def lingering_echoes() -> list[int]:
        return [
            idx
            for idx in repair_ids
            if _has_source_echo(
                _repair_source_text(segments[idx]),
                _repair_translation_text(segments[idx], repaired_texts, idx),
            )
        ]

    unresolved: list[int] = []
    request_error: Exception | None = None
    try:
        for offset in range(0, len(repair_ids), initial_span):
            unresolved.extend(request_group(repair_ids[offset : offset + initial_span]))
        # Exact source echo is the correctness failure that retired the old
        # no-thinking tier. Anything still echoing gets one smaller second
        # chance, because a shorter group is the lever that fixed the rest.
        retry_span = max(1, initial_span // 2)
        echoes = lingering_echoes()
        for offset in range(0, len(echoes), retry_span):
            unresolved.extend(request_group(echoes[offset : offset + retry_span]))
    except TranslationCancelledError:
        raise
    except Exception as exc:
        request_error = exc

    # Outside the handler on purpose: a provider error must not be able to skip
    # this gate. Returning the Japanese source as the translation is the known
    # 10.1% untranslated-film regression, and the caller caches what it gets
    # back, so it is worth failing the run over. Residual kana and length flags
    # stay diagnostics - they are just as often proper names or deliberate
    # subtitle compression.
    final_echoes = lingering_echoes()
    if final_echoes:
        detail = f" after {request_error}" if request_error is not None else ""
        raise RuntimeError(
            f"translation still echoes the Japanese source for {len(final_echoes)} "
            f"cues after repair at effort={repair_effort}{detail}; "
            f"ids={final_echoes[:50]}"
        )

    if request_error is not None:
        print(f"[WARN] translation repair failed: {request_error}", flush=True)
        _emit_progress(
            on_progress,
            {
                "phase": "repair_failed",
                "repair_ids": repair_ids,
                "error": str(request_error)[:200],
            },
        )
        return zh_texts, {
            "mode": "translation_repair_failed",
            "start_index": min(repair_ids),
            "segment_count": 0,
            "elapsed_s": time.perf_counter() - started,
            "request_count": request_count,
            "repair_ids": repair_ids,
            "candidate_count": len(repair_ids),
            "missing_count": len(repair_ids),
            "missing_indexes": repair_ids,
            "error": str(request_error)[:500],
            **transport_util._merge_usage_metrics(request_usages),
        }

    if repaired_ids and cache_writer is not None:
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
        "segment_count": len(repaired_ids),
        "elapsed_s": time.perf_counter() - started,
        "request_count": request_count,
        "repair_ids": repair_ids,
        "candidate_count": len(repair_ids),
        "reasoning_effort": repair_effort,
        "format_split_count": format_split_count,
        "missing_count": len(set(unresolved)),
        "missing_indexes": sorted(set(unresolved)),
        **transport_util._merge_usage_metrics(request_usages),
    }
    _emit_progress(
        on_progress,
        {
            "phase": "repair_done",
            "repair_ids": repair_ids,
            "repaired": len(repaired_ids),
            "expected": len(repair_ids),
        },
    )
    return repaired_texts, timing


_KANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_CHECK_FOLD_RE = re.compile(r"[\W_]+", re.UNICODE)


def _fold_translation_check(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return _CHECK_FOLD_RE.sub("", normalized)


def _has_source_echo(source: str, target: str) -> bool:
    folded_source = _fold_translation_check(source)
    return bool(folded_source) and _fold_translation_check(target) == folded_source


def _has_remaining_japanese_kana(target: str) -> bool:
    return bool(_KANA_RE.search(str(target or "")))


def _glossary_violations(source: str, target: str, pairs: list[tuple[str, str]]) -> bool:
    """Source uses a mapped term and the translation does not use its mapping.

    Deliberately one-directional and literal: the glossary is the one place the
    user states an exact word they want, so "the mapped target string is absent"
    is a complete test. No attempt is made to judge the substitute.
    """
    for glossary_source, glossary_target in pairs:
        if glossary_source and glossary_source in source and glossary_target not in target:
            return True
    return False


def _select_translation_repair_ids(
    segments: list[dict],
    zh_texts: list[str],
    glossary: str = "",
) -> tuple[list[int], dict[int, list[str]]]:
    """Lines a local check can call wrong, with why.

    Every detector is cheap and text-only by design - the pass has to decide
    what to escalate without spending a model call to find out, or the saving it
    exists for is gone. They differ in how much they prove: an exact source echo
    or a missed glossary term is definitely wrong, while kana and length are
    correlates.

    The glossary check was added 2026-08-24 after measuring sample-v: at
    `reasoning_effort=low` the base pass rendered 5 of 37 ちんぽ cues as 鸡巴
    rather than the configured 肉棒, and **none** of them were flagged - a term
    substitution is not an echo, contains no kana, and does not change the
    length. Glossary compliance was 83.8% against 97.3% on a non-thinking base
    pass, i.e. thinking paraphrases away from an injected term list. Without
    this detector the cheaper tier silently costs the one part of the output the
    user stated exactly.
    """
    pairs = parse_glossary_pairs(glossary) if glossary else []
    repair_ids: list[int] = []
    reasons: dict[int, list[str]] = {}
    for idx, seg in enumerate(segments):
        source = _repair_source_text(seg)
        target = _repair_translation_text(seg, zh_texts, idx)
        local_reasons: list[str] = []
        if _has_source_echo(source, target):
            local_reasons.append("source_echo")
        if _has_remaining_japanese_kana(target):
            local_reasons.append("japanese_remaining")
        if pairs and _glossary_violations(source, target, pairs):
            local_reasons.append("glossary_violation")
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
        "只处理 reason 字段指出的源文回显、日文假名残留、术语表未生效或译文长度异常，"
        "保持原字幕文本含义，不要根据上下文推测或改写源文。"
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
        elif reason == "source_echo":
            public.append("source_echo")
        elif reason == "japanese_remaining":
            public.append("japanese_remaining")
        elif reason == "glossary_violation":
            public.append("glossary_violation")
        else:
            public.append("translation_quality")
    return list(dict.fromkeys(public))
