"""Post-translation repair pass (JSON contract).

Optional stage a profile opts into via ``wants_repair_pass``: scan the finished
translation for lines a local detector can call suspicious - source echo,
residual Japanese kana, a glossary term the translation did not use, a
rendering that drifted from this same film's own settled index (see
``global_glossary``), target/source length ratio out of band - then reissue
only those ids, with a few lines of local context and one tier more thinking
than the base pass used.

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
    extra_glossary: str = "",
    on_progress: Callable[[dict], None] | None = None,
    cancel_event: threading.Event | None = None,
    cache_writer: Callable[[list[str]], None] | None = None,
) -> tuple[list[str], dict | None]:
    """Reissue flagged ids cheap-first, escalating only what is still wrong.

    Stage 1 reissues every flagged id at `none` - not a blind retry of the
    original batch request, but the repair prompt, which names the exact
    defect (`reason`) and now carries the film's settled term glossary
    (`extra_glossary`), neither of which the original batch request had. Most
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
    """
    _raise_if_cancelled(cancel_event)
    glossary_pairs = parse_glossary_pairs(glossary) if glossary else []
    # Full source line -> the rendering the base pass used most often for it,
    # derived from this same film's own output (see `global_glossary`). Reused
    # here as a second detector so a cue that drifted from the film's own norm
    # gets caught the same cheap, text-only way a glossary violation does.
    settled_pairs = dict(parse_glossary_pairs(extra_glossary)) if extra_glossary else {}
    repair_ids, reasons = _select_translation_repair_ids(
        segments, zh_texts, glossary, settled_pairs
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
    request_cap = max(1, int(llm_settings.TRANSLATION_BATCH_MAX_REQUESTS))
    initial_span = max(1, int(batch_size))
    none_effort = "none"
    escalated_effort = llm_settings._repair_reasoning_effort(reasoning_effort)

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

    def request_group(group: list[int], effort: str) -> list[int]:
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
            return request_group(group[:midpoint], effort) + request_group(group[midpoint:], effort)

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
            return request_group(group[:midpoint], effort) + request_group(group[midpoint:], effort)
        return request_group(missing, effort)

    def lingering_echoes() -> list[int]:
        return [
            idx
            for idx in repair_ids
            if _has_source_echo(
                _repair_source_text(segments[idx]),
                _repair_translation_text(segments[idx], repaired_texts, idx),
            )
        ]

    def recheck(candidates: list[int]) -> list[int]:
        """Ids the detector suite still flags after a repair attempt.

        Updates `reasons` in place so the next request describes the CURRENT
        defect - a cue fixed for length but freshly echoing after stage 1
        must not carry a stale `length_mismatch` label into stage 2.
        """
        flagged: list[int] = []
        for idx in candidates:
            source = _repair_source_text(segments[idx])
            target = _repair_translation_text(segments[idx], repaired_texts, idx)
            local_reasons = _detect_repair_reasons(
                source, target, glossary_pairs, settled_pairs
            )
            if local_reasons:
                reasons[idx] = list(dict.fromkeys(local_reasons))
                flagged.append(idx)
        return flagged

    unresolved: list[int] = []
    escalate_candidates: list[int] = []
    request_error: Exception | None = None
    try:
        # Stage 1: every flagged id, cheap.
        none_tier_unresolved: list[int] = []
        for offset in range(0, len(repair_ids), initial_span):
            none_tier_unresolved.extend(
                request_group(repair_ids[offset : offset + initial_span], none_effort)
            )

        # Stage 2: only what stage 1 did not actually fix, escalated. Ids that
        # never got a reply (hit the request cap) go straight in too - a
        # different tier is worth one more try before giving up on them.
        escalate_candidates = sorted(set(recheck(repair_ids)) | set(none_tier_unresolved))
        retry_span = max(1, initial_span // 2)
        for offset in range(0, len(escalate_candidates), retry_span):
            unresolved.extend(
                request_group(escalate_candidates[offset : offset + retry_span], escalated_effort)
            )

        # Exact source echo is the correctness failure that retired the old
        # no-thinking tier. Anything still echoing gets one smaller second
        # chance, because a shorter group is the lever that fixed the rest.
        echoes = lingering_echoes()
        for offset in range(0, len(echoes), retry_span):
            unresolved.extend(request_group(echoes[offset : offset + retry_span], escalated_effort))
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
            f"cues after repair at effort={escalated_effort}{detail}; "
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


def _inconsistent_rendering(
    source: str, target: str, settled_pairs: dict[str, str]
) -> bool:
    """This exact line has a dominant rendering elsewhere in the film, and this
    cue used something else.

    Exact-line match, not substring containment like `_glossary_violations`: the
    index is built from whole recurring lines (see `global_glossary`), so a
    partial match would compare the wrong unit and false-positive on short
    lines embedded in longer ones.
    """
    settled_zh = settled_pairs.get(source)
    return bool(settled_zh) and target.strip() != settled_zh


def _detect_repair_reasons(
    source: str,
    target: str,
    glossary_pairs: list[tuple[str, str]],
    settled_pairs: dict[str, str] | None = None,
) -> list[str]:
    """The detector suite, factored out so the post-none-tier recheck in
    `apply_repair_pass` shares one implementation with the initial selection
    below instead of drifting into a second copy."""
    local_reasons: list[str] = []
    if _has_source_echo(source, target):
        local_reasons.append("source_echo")
    if _has_remaining_japanese_kana(target):
        local_reasons.append("japanese_remaining")
    if glossary_pairs and _glossary_violations(source, target, glossary_pairs):
        local_reasons.append("glossary_violation")
    if settled_pairs and _inconsistent_rendering(source, target, settled_pairs):
        local_reasons.append("inconsistent_rendering")
    if _has_translation_length_mismatch(source, target):
        local_reasons.append("length_mismatch")
    return local_reasons


def _select_translation_repair_ids(
    segments: list[dict],
    zh_texts: list[str],
    glossary: str = "",
    settled_pairs: dict[str, str] | None = None,
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
        local_reasons = _detect_repair_reasons(source, target, pairs, settled_pairs)
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
    extra_glossary: str = "",
) -> list[dict]:
    system_prompt = prompt_module._build_system_prompt(
        character_reference,
        target_lang=target_lang,
        glossary=glossary,
    )
    system_prompt += (
        "\n\n这是翻译后局部修复任务。只修复 requested_ids 中的译文；"
        "只处理 reason 字段指出的源文回显、日文假名残留、术语表未生效、"
        "译名与全片同一句台词不一致或译文长度异常，"
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
    user_parts = [
        "【翻译修复任务】",
        f"requested_ids = {prompt_module._format_requested_ids(repair_ids)}",
        "只返回 requested_ids 中列出的 id，恰好返回这些 id，不要返回 context_only 项。",
        "每个 text 只能是修复后的中文字幕；不要解释原因。",
    ]
    # Same block the base pass got, threaded through so a `none`-tier repair
    # request has the film's already-settled terms to fall back on instead of
    # having to infer them from three lines of local context alone - this is
    # most of what makes a cheap first attempt viable.
    extra_block = prompt_module._build_extra_glossary_block(extra_glossary)
    if extra_block:
        user_parts.append(extra_block)
        user_parts.append("注意：必须严格使用上面 <glossary> 标签内的术语表翻译。")
    user_parts.extend(
        [
            "【局部上下文 JSON】",
            json.dumps(context_items, ensure_ascii=False, indent=2),
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
        elif reason == "inconsistent_rendering":
            public.append("inconsistent_rendering")
        else:
            public.append("translation_quality")
    return list(dict.fromkeys(public))
