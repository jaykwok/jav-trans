"""Post-translation repair pass (JSON contract).

Optional stage a profile opts into via ``wants_repair_pass``: scan the finished
translation for empty output, Japanese source echo, residual kana and missing
user terminology. Reissue only those ids, with neighbouring cues and confirmed
continuation units. Length ratios and variation between repeated lines do not
prove a translation is wrong; optional semantic review handles that uncertainty.

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
from the caller — the transport stays outside.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import re
import threading
import time
import unicodedata
from typing import Callable

from core.diagnostics import format_exception_chain
from llm import prompt as prompt_module
from llm import settings as llm_settings
from llm import transport_util
from llm import zh_variant
from llm.glossary import parse_glossary_pairs
from llm.errors import (
    TERMINAL_TRANSLATION_ERRORS,
    ResponseTruncatedError,
    RetryableTranslationFormatError,
    TranslationCancelledError,
)
from llm.profiles import json_v3
from llm.session import TranslationSession
from llm.context import SourceContext, cue_item

from llm.output_checks import _KANA as _KANA_RE, is_chinese_target

logger = logging.getLogger(__name__)

_raise_if_cancelled = transport_util._raise_if_cancelled
_emit_progress = transport_util._emit_progress


def _callable_accepts_two_positional(fn: Callable) -> bool:
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return True
    positional = [
        p
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    return len(positional) >= 2


def _inspect_bounded_schema_flags(profile) -> tuple[bool, bool]:
    try:
        params = inspect.signature(profile.bounded_schema).parameters
    except (TypeError, ValueError):
        return (False, False)
    has_varkw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    return ("ids" in params or has_varkw, "forbid_kana" in params or has_varkw)


def apply_repair_pass(
    segments: list[dict],
    zh_texts: list[str],
    *,
    chat: Callable[..., str],
    session: TranslationSession,
    extra_glossary: str = "",
    on_progress: Callable[[dict], None] | None = None,
    cancel_event: threading.Event | None = None,
    cache_writer: Callable[..., None] | None = None,
    source_context: SourceContext | None = None,
) -> tuple[list[str], dict | None]:
    """Reissue flagged ids cheap-first, escalating only what is still wrong.

    Stage 1 reissues every flagged id at `none` - not a blind retry of the
    original batch request, but the repair prompt, which names the exact
    defect (`reason`) with source-backed local context. Most
    flagged ids resolve here without paying for reasoning at all. Stage 2
    re-runs the same local detectors against the stage-1 output and only pays
    `escalated_effort` (derived from the base pass's tier via
    `_repair_reasoning_effort`, unless `TRANSLATION_REPAIR_REASONING_EFFORT`
    pins it) for ids that are still wrong. A final small-span second chance
    for lingering source echo - the correctness failure that retired the old
    always-`none` tier - runs at the escalated tier, unchanged from before.

    Candidate ids are collected across the whole film before any request goes
    out. That aggregation is the cost property: a detector firing once in each
    base batch must not buy one fixed reasoning preamble per batch. Invalid or
    truncated replies split the group in half, which reuses the engine's "ask
    for less" recovery rule instead of reissuing an identical bad shape.

    Stage 1 failing outright - some endpoints reject `none` reasoning as a
    request-level error rather than a bad reply - does not skip stage 2:
    a provider rejecting the cheap tier says nothing about the escalated one,
    which is a different request shape (a real reasoning budget).
    """
    _raise_if_cancelled(cancel_event)
    # Read from the session, never from live settings: this pass reissues text
    # the base pass produced, and the two have to be the same run.
    profile = session.profile
    glossary = session.glossary
    target_lang = session.target_lang
    character_reference = session.character_reference
    to_target_variant = zh_variant.converter_for(target_lang)

    def _to_target_variant(text: str) -> str:
        return (
            to_target_variant(text)
            if text and to_target_variant is not None
            else text
        )

    glossary_pairs = (
        [(src, _to_target_variant(dst)) for src, dst in parse_glossary_pairs(glossary)]
        if glossary
        else []
    )
    repair_ids, reasons = _select_translation_repair_ids(
        segments, zh_texts, glossary, glossary_pairs=glossary_pairs
    )
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
    initial_span = max(1, int(session.batch_size))
    batch_max_requests = max(1, int(llm_settings.TRANSLATION_BATCH_MAX_REQUESTS))
    request_cap = max(
        batch_max_requests,
        math.ceil(len(repair_ids) / max(1, initial_span)) * 4 + 8,
    )
    none_effort = "none"
    escalated_effort = session.repair_reasoning_effort if session.reasoning_effort else "none"
    repair_source_context = source_context or SourceContext.build(segments)
    schema_supports_ids, schema_supports_forbid_kana = _inspect_bounded_schema_flags(profile)
    cache_writer_takes_ids = (
        _callable_accepts_two_positional(cache_writer)
        if cache_writer is not None
        else False
    )

    _emit_progress(
        on_progress,
        {
            "phase": "repair_start",
            "repair_ids": repair_ids,
            "candidate_count": len(repair_ids),
            "reasoning_effort": none_effort,
            "escalated_reasoning_effort": escalated_effort,
        },
    )

    local_kana_guard = (
        session.backend_name == "llamacpp"
        and is_chinese_target(target_lang)
        and not any(_has_remaining_japanese_kana(dst) for _, dst in glossary_pairs)
    )

    def _bounded_schema_for_repair(
        request_segments: list[dict],
        *,
        ids: list[int],
        forbid_kana: bool = False,
    ) -> dict | None:
        kwargs: dict = {}
        if ids and schema_supports_ids:
            kwargs["ids"] = list(ids)
        if forbid_kana and schema_supports_forbid_kana:
            kwargs["forbid_kana"] = True
        return profile.bounded_schema(request_segments, **kwargs)

    def request_group(
        group: list[int],
        effort: str,
        *,
        forbid_kana: bool = False,
    ) -> list[int]:
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
            extra_glossary=extra_glossary,
            compact=(session.backend_name == "llamacpp"),
            source_context=repair_source_context,
        )
        request_kwargs: dict = {
            "response_schema": profile.schema,
            "reasoning_effort": effort,
        }
        token_budget = profile.response_token_budget(
            request_segments,
            reasoning_effort=effort,
        )
        if token_budget is not None:
            request_kwargs["max_tokens"] = token_budget
        bounded = _bounded_schema_for_repair(
            request_segments, ids=group, forbid_kana=forbid_kana
        )
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
            return request_group(
                group[:midpoint], effort, forbid_kana=forbid_kana
            ) + request_group(group[midpoint:], effort, forbid_kana=forbid_kana)

        missing: list[int] = []
        for idx in group:
            _raise_if_cancelled(cancel_event)
            raw_text = parsed[idx]
            if raw_text:
                source = _repair_source_text(segments[idx])
                previous = _repair_translation_text(segments[idx], repaired_texts, idx)
                raw_is_echo = _has_source_echo(source, raw_text)
                converted_text = raw_text if raw_is_echo else _to_target_variant(raw_text)
                is_echo = raw_is_echo or _has_source_echo(source, converted_text)
                if (
                    is_echo
                    and previous.strip()
                    and not _has_source_echo(source, previous)
                ):
                    logger.warning(
                        "Ignoring repair reply for cue %d that regressed into source echo",
                        idx,
                    )
                    continue
                repaired_texts[idx] = converted_text
                if not is_echo:
                    repaired_ids.add(idx)
            else:
                missing.append(idx)
        if not missing:
            return []
        if len(missing) == len(group):
            if len(group) <= 1:
                return list(group)
            format_split_count += 1
            midpoint = max(1, len(group) // 2)
            return request_group(
                group[:midpoint], effort, forbid_kana=forbid_kana
            ) + request_group(group[midpoint:], effort, forbid_kana=forbid_kana)
        return request_group(missing, effort, forbid_kana=forbid_kana)

    def lingering_echoes() -> list[int]:
        return [
            idx
            for idx in range(len(segments))
            if _has_source_echo(
                _repair_source_text(segments[idx]),
                _repair_translation_text(segments[idx], repaired_texts, idx),
            )
        ]

    def recheck(candidates: list[int]) -> list[int]:
        """Ids the detector suite still flags after a repair attempt.

        Updates `reasons` in place so the next request describes the CURRENT
        defect - a cue fixed for terminology but freshly echoing after stage 1
        must not carry a stale defect label into stage 2.
        """
        flagged: list[int] = []
        for idx in candidates:
            source = _repair_source_text(segments[idx])
            raw_target = _repair_translation_text(segments[idx], repaired_texts, idx)
            target = (
                raw_target
                if _has_source_echo(source, raw_target)
                else _to_target_variant(raw_target)
            )
            local_reasons = _detect_repair_reasons(
                source, target, glossary_pairs
            )
            if local_reasons:
                reasons[idx] = list(dict.fromkeys(local_reasons))
                flagged.append(idx)
        return flagged

    def persist_repairs(texts: list[str], ids: set[int] | None = None) -> None:
        """Write what was actually fixed back into the translation cache.

        Runs on the degraded path as well as the clean one: a repair that
        succeeded was paid for, and a later request failing says nothing about
        it. Idempotent - `_save_cache_entry` appends and reads dedupe by
        last-write-wins per batch_key.
        """
        target_ids = set(repaired_ids if ids is None else ids)
        if not target_ids or cache_writer is None:
            return
        try:
            if cache_writer_takes_ids:
                cache_writer(texts, target_ids)
            else:
                cache_writer(texts)
        except TranslationCancelledError:
            raise
        except Exception as exc:
            print(
                f"[WARN] failed to persist repaired translation cache: {exc}",
                flush=True,
            )

    unresolved: list[int] = []
    escalate_candidates: list[int] = []
    request_error: Exception | None = None

    # Stage 1 gets its own try/except, separate from stage 2's below: a
    # request-level failure here (the endpoint refuses `none` outright, a
    # timeout, ...) must not skip stage 2. Escalation is a different request
    # shape - a real reasoning budget - and may well work where `none`
    # cannot. Whatever stage 1 fixed before failing is already recorded in
    # `repaired_texts`; `recheck` below re-derives what is still wrong from
    # that state, so nothing needs to be threaded through by hand.
    none_tier_unresolved: list[int] = []
    try:
        for offset in range(0, len(repair_ids), initial_span):
            none_tier_unresolved.extend(
                request_group(repair_ids[offset : offset + initial_span], none_effort)
            )
    except TERMINAL_TRANSLATION_ERRORS:
        # Cancellation, a lease that is gone, a content refusal: each of these
        # says the next request cannot succeed either, and stage 2's escalation
        # is still a request. Persist what stage 1 did fix, then let the type
        # the backend raised reach the caller unchanged.
        persist_repairs(repaired_texts, repaired_ids)
        raise
    except Exception as exc:
        # A request-level failure at this tier almost always repeats for
        # every remaining chunk, so stop retrying stage 1 rather than burn it
        # on doomed requests - whatever it never reached falls through to
        # stage 2 below exactly like ids stage 1 tried and failed to fix.
        error_chain = format_exception_chain(exc)
        logger.warning(
            "Translation repair stage 1 failed (effort=%s): %s",
            none_effort,
            error_chain,
            exc_info=True,
        )
        _emit_progress(
            on_progress,
            {
                "phase": "repair_stage1_failed",
                "error_type": type(exc).__name__,
                "error": error_chain[:300],
            },
        )

    try:
        # Stage 2: only what stage 1 did not actually fix, escalated. Ids that
        # never got a reply (hit the request cap, or stage 1 failed outright)
        # go straight in too - a different tier is worth one more try before
        # giving up on them.
        escalate_candidates = sorted(set(recheck(repair_ids)) | set(none_tier_unresolved))
        retry_span = max(1, initial_span // 2)
        for offset in range(0, len(escalate_candidates), retry_span):
            chunk = escalate_candidates[offset : offset + retry_span]
            chunk_forbid_kana = local_kana_guard and any(
                reason in {"source_echo", "japanese_remaining", "empty_translation"}
                for idx in chunk
                for reason in reasons.get(idx, [])
            )
            unresolved.extend(
                request_group(chunk, escalated_effort, forbid_kana=chunk_forbid_kana)
            )

        # Exact source echo is the correctness failure that retired the old
        # no-thinking tier. Anything still echoing gets one smaller second
        # chance, because a shorter group is the lever that fixed the rest.
        echoes = lingering_echoes()
        for offset in range(0, len(echoes), retry_span):
            unresolved.extend(
                request_group(
                    echoes[offset : offset + retry_span],
                    escalated_effort,
                    forbid_kana=local_kana_guard,
                )
            )
    except TERMINAL_TRANSLATION_ERRORS:
        persist_repairs(repaired_texts, repaired_ids)
        raise
    except Exception as exc:
        request_error = exc
        logger.warning(
            "Translation repair stage 2 failed (effort=%s): %s",
            escalated_effort,
            format_exception_chain(exc),
            exc_info=True,
        )

    # Outside the handler on purpose: a provider error must not be able to skip
    # this gate. Returning the Japanese source as the translation is the known
    # 10.1% untranslated-film regression, and the caller caches what it gets
    # back, so it is worth failing the run over. Residual kana and glossary flags
    # stay diagnostics - they are just as often proper names or deliberate
    # subtitle compression.
    final_echoes = lingering_echoes()
    if final_echoes:
        persist_repairs(repaired_texts, repaired_ids)
        detail = (
            f" after {format_exception_chain(request_error)}"
            if request_error is not None
            else ""
        )
        raise RuntimeError(
            f"translation still echoes the Japanese source for {len(final_echoes)} "
            f"cues after repair at effort={escalated_effort}{detail}; "
            f"ids={final_echoes[:50]}"
        ) from request_error

    if request_error is not None:
        # `repaired_texts`, not `zh_texts`. The echo gate above was applied to
        # the repaired array, so returning the original one would return
        # something that was never checked - and specifically, would put back
        # the Japanese source lines the earlier requests had already fixed and
        # been billed for. Verification, return value and cache are the same
        # array on every exit from this function; ids the detectors still flag
        # are reported as unresolved rather than being hidden behind a rollback.
        still_flagged = recheck(repair_ids)
        persist_repairs(repaired_texts, repaired_ids)
        error_chain = format_exception_chain(request_error)
        logger.warning(
            "Translation repair failed (%s): %s",
            type(request_error).__name__,
            error_chain,
        )
        _emit_progress(
            on_progress,
            {
                "phase": "repair_failed",
                "repair_ids": repair_ids,
                "repaired": len(repaired_ids),
                "unresolved": len(still_flagged),
                "error_type": type(request_error).__name__,
                "error": error_chain[:300],
            },
        )
        return repaired_texts, {
            "mode": "translation_repair_failed",
            "start_index": min(repair_ids),
            "segment_count": len(repaired_ids),
            "elapsed_s": time.perf_counter() - started,
            "request_count": request_count,
            "repair_ids": repair_ids,
            "candidate_count": len(repair_ids),
            "missing_count": len(still_flagged),
            "missing_indexes": still_flagged,
            "error_type": type(request_error).__name__,
            "error": error_chain[:1000],
            **transport_util._merge_usage_metrics(request_usages),
        }

    persist_repairs(repaired_texts, repaired_ids)
    timing = {
        "mode": "translation_repair_pass",
        "start_index": min(repair_ids),
        "segment_count": len(repaired_ids),
        "elapsed_s": time.perf_counter() - started,
        "request_count": request_count,
        "repair_ids": repair_ids,
        "candidate_count": len(repair_ids),
        # Kept for callers that read a single tier; the two-stage cascade
        # below is the fuller picture.
        "reasoning_effort": escalated_effort,
        "none_tier_reasoning_effort": none_effort,
        "none_tier_resolved_count": len(repair_ids) - len(escalate_candidates),
        "escalated_count": len(escalate_candidates),
        "escalated_ids": escalate_candidates,
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


# _KANA_RE is imported from llm.output_checks (fullwidth + halfwidth kana).
_CHECK_FOLD_RE = re.compile(r"[\W_]+", re.UNICODE)


def _fold_translation_check(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return _CHECK_FOLD_RE.sub("", normalized)


def _has_source_echo(source: str, target: str) -> bool:
    folded_source = _fold_translation_check(source)
    # Shared Han words and Latin names can be valid unchanged translations.
    return bool(_KANA_RE.search(source)) and bool(folded_source) and _fold_translation_check(target) == folded_source


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


def _detect_repair_reasons(
    source: str,
    target: str,
    glossary_pairs: list[tuple[str, str]],
) -> list[str]:
    """The detector suite, factored out so the post-none-tier recheck in
    `apply_repair_pass` shares one implementation with the initial selection
    below instead of drifting into a second copy."""
    local_reasons: list[str] = []
    if not target.strip():
        local_reasons.append("empty_translation")
    if _has_source_echo(source, target):
        local_reasons.append("source_echo")
    if _has_remaining_japanese_kana(target):
        local_reasons.append("japanese_remaining")
    if glossary_pairs and _glossary_violations(source, target, glossary_pairs):
        local_reasons.append("glossary_violation")
    return local_reasons


def _select_translation_repair_ids(
    segments: list[dict],
    zh_texts: list[str],
    glossary: str = "",
    *,
    glossary_pairs: list[tuple[str, str]] | None = None,
) -> tuple[list[int], dict[int, list[str]]]:
    """Lines with an actionable output defect, with why.

    Every detector is cheap and text-only by design - the pass has to decide
    what to escalate without spending a model call to find out, or the saving it
    exists for is gone. They differ in how much they prove: an exact source echo
    or a missed glossary term is definitely wrong, while kana and length are
    correlates. A length ratio alone is never a reason to rewrite a cue.

    The glossary check was added 2026-08-24 after measuring sample-v: at
    `reasoning_effort=low` the base pass rendered 5 of 37 ちんぽ cues as 鸡巴
    rather than the configured 肉棒, and **none** of them were flagged - a term
    substitution is not an echo, contains no kana, and does not change the
    length. Glossary compliance was 83.8% against 97.3% on a non-thinking base
    pass, i.e. thinking paraphrases away from an injected term list. Without
    this detector the cheaper tier silently costs the one part of the output the
    user stated exactly.
    """
    pairs = (
        list(glossary_pairs)
        if glossary_pairs is not None
        else (parse_glossary_pairs(glossary) if glossary else [])
    )
    repair_ids: list[int] = []
    reasons: dict[int, list[str]] = {}
    for idx, seg in enumerate(segments):
        source = _repair_source_text(seg)
        target = _repair_translation_text(seg, zh_texts, idx)
        local_reasons = _detect_repair_reasons(source, target, pairs)
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


def _build_repair_messages(
    segments: list[dict],
    zh_texts: list[str],
    repair_ids: list[int],
    reasons: dict[int, list[str]],
    *,
    target_lang: str,
    glossary: str,
    character_reference: str,
    compact: bool = False,
    extra_glossary: str = "",
    source_context: SourceContext | None = None,
) -> list[dict]:
    effective_target_lang = (
        (target_lang or llm_settings.DEFAULT_TARGET_LANG).strip()
        or llm_settings.DEFAULT_TARGET_LANG
    )
    system_prompt = prompt_module._build_system_prompt(
        character_reference,
        target_lang=effective_target_lang,
        glossary=glossary,
        compact=compact,
    )
    system_prompt += (
        "\n\n这是翻译后局部修复任务。只修复 requested_ids 中的译文；"
        "只处理 reason 字段指出的空译文、源文回显、日文假名残留或术语表未生效，"
        "保持原字幕文本含义，不要根据上下文推测或改写源文。"
        "reason 只是问题类别提示，不是固定译文；最终译文必须服从原文和既定术语。"
        "保持否定、条件、施受关系和真实重复的作用，不为了通过长度检查添加或删除含义。"
    )
    ctx = source_context or SourceContext.build(segments)
    focus = ctx.focus(
        repair_ids, radius=llm_settings.TRANSLATION_REPAIR_CONTEXT_RADIUS
    )
    context_items = _build_repair_context_items(
        segments,
        zh_texts,
        repair_ids,
        reasons,
        source_context=ctx,
        focus=focus,
    )
    user_parts = [
        "【翻译修复任务】",
        f"requested_ids = {prompt_module._format_requested_ids(repair_ids)}",
        "只返回 requested_ids 中列出的 id，恰好返回这些 id，不要返回 context_only 项。",
        f"每个 text 只能是修复后的{effective_target_lang}字幕；不要解释原因。",
    ]
    # Legacy caller-supplied examples remain nonbinding; the main pipeline
    # never promotes repeated whole-line renderings into user terminology.
    extra_block = prompt_module._build_extra_glossary_block(extra_glossary)
    if extra_block:
        user_parts.append(extra_block)
    user_parts.extend(
        [
            "【局部上下文 JSON】",
            json.dumps(context_items, ensure_ascii=False, separators=(",", ":")),
            "【完整语义单元】\n"
            + json.dumps(
                focus["semantic_units"], ensure_ascii=False, separators=(",", ":")
            ),
            '输出 JSON：{"translations":[{"id":0,"text":"..."}]}',
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _build_repair_context_items(
    segments: list[dict],
    zh_texts: list[str],
    repair_ids: list[int],
    reasons: dict[int, list[str]],
    *,
    source_context: SourceContext | None = None,
    focus: dict | None = None,
) -> list[dict]:
    resolved_focus = focus or (source_context or SourceContext.build(segments)).focus(
        repair_ids, radius=llm_settings.TRANSLATION_REPAIR_CONTEXT_RADIUS
    )
    indexes = {row["id"] for row in resolved_focus["items"]}

    items = []
    repair_id_set = set(repair_ids)
    for idx in sorted(indexes):
        seg = segments[idx]
        items.append(
            {
                **cue_item(seg, idx),
                "role": "repair" if idx in repair_id_set else "context_only",
                "reason": _public_repair_reasons(reasons.get(idx, [])),
                "ja": _repair_source_text(seg),
                "current_zh": _repair_translation_text(seg, zh_texts, idx),
            }
        )
    return items


def _public_repair_reasons(local_reasons: list[str]) -> list[str]:
    public: list[str] = []
    for reason in local_reasons:
        if reason == "empty_translation":
            public.append("empty_translation")
        elif reason == "source_echo":
            public.append("source_echo")
        elif reason == "japanese_remaining":
            public.append("japanese_remaining")
        elif reason == "glossary_violation":
            public.append("glossary_violation")
        else:
            public.append("translation_quality")
    return list(dict.fromkeys(public))
