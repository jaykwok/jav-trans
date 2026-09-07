"""Bounded, evidence-based semantic review with a separate candidate check.

The reviewer is allowed to abstain. Its self-reported confidence is never an
acceptance condition: an edit needs a literal source quote, usable output, and
a fresh source-to-two-candidates comparison. This is still a model judgement,
so review is opt-in and its changes are retained for inspection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from uuid import uuid4

from llm import settings, transport_util
from llm.context import SourceContext, source_text
from llm.errors import TERMINAL_TRANSLATION_ERRORS, RetryableTranslationFormatError
from llm.output_checks import invalid_line_output
from llm.session import TranslationSession
from llm.glossary import parse_glossary_pairs

VERSION = "semantic-review-v2"
logger = logging.getLogger(__name__)
_RISK = re.compile(r"ない|なく|なかった|ません|ずに|わけ|とは限|だけ|しか|なら|たら|ても|けれど|はず|かもしれ|くれる|もら|あげ|より|[0-9０-９]")
_ISSUES = ("none", "negation", "condition", "roles", "omission", "addition", "quantity", "meaning")


def _schema(root: str, fields: dict) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "properties": {root: {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": fields, "required": list(fields),
        }}}, "required": [root],
    }


REVIEW_SCHEMA = _schema("reviews", {
    "id": {"type": "integer"},
    "issue": {"type": "string", "enum": list(_ISSUES)},
    "source_quote": {"type": "string"},
    "translation_quote": {"type": "string"},
    "reason": {"type": "string"},
    "suggestion": {"type": "string"},
})
VERIFY_SCHEMA = _schema("verdicts", {
    "id": {"type": "integer"},
    "verdict": {"type": "string", "enum": ["a", "b", "equivalent", "uncertain"]},
    "source_quote": {"type": "string"},
})


def select_candidates(
    segments: list[dict], limit: int, *, sample_every: int = 40, translations: list[str] | None = None
) -> list[int]:
    """Spread the budget over the film and reserve some for unflagged controls."""
    if limit <= 0:
        return []
    from llm.repair import _has_translation_length_mismatch

    risky = [
        index for index, seg in enumerate(segments)
        if _RISK.search(source_text(seg))
        or seg.get("continues_from_previous") or seg.get("continues_into_next")
        or (translations is not None and index < len(translations)
            and _has_translation_length_mismatch(source_text(seg), translations[index]))
    ]
    risk_set = set(risky)
    controls = [
        index for index in range(max(0, sample_every - 1), len(segments), max(1, sample_every))
        if index not in risk_set
    ] if sample_every > 0 else []
    control_budget = min(len(controls), max(1, limit // 5)) if controls else 0
    risk_budget = min(len(risky), limit - control_budget)

    def spread(values: list[int], count: int) -> list[int]:
        if count >= len(values):
            return values
        return [values[index * len(values) // count] for index in range(count)] if count else []

    chosen = spread(risky, risk_budget) + spread(controls, min(limit - risk_budget, len(controls)))
    return sorted(chosen)


def _parse(text: str, root: str, ids: list[int]) -> dict[int, dict]:
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict) or set(payload) != {root}:
            raise ValueError("unexpected root")
        rows = payload[root]
        if not isinstance(rows, list):
            raise ValueError("expected array")
        schema = REVIEW_SCHEMA if root == "reviews" else VERIFY_SCHEMA
        fields = schema["properties"][root]["items"]["properties"]
        by_id: dict[int, dict] = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != set(fields):
                raise ValueError("unexpected fields")
            index = row["id"]
            if type(index) is not int or index not in ids or index in by_id:
                raise ValueError("unexpected or duplicate id")
            for name, spec in fields.items():
                if spec["type"] == "string" and not isinstance(row[name], str):
                    raise ValueError("expected string")
                if "enum" in spec and row[name] not in spec["enum"]:
                    raise ValueError("unexpected enum")
            if root == "reviews" and row["issue"] == "none" and row["suggestion"].strip():
                raise ValueError("abstention must not include a rewrite")
            by_id[index] = row
        if set(by_id) != set(ids):
            raise ValueError("missing id")
        return by_id
    except (ValueError, KeyError, TypeError) as exc:
        # Only our structural diagnosis is recorded, never the model response.
        detail = "invalid JSON" if isinstance(exc, json.JSONDecodeError) else str(exc)
        raise RetryableTranslationFormatError(f"invalid semantic review response: {detail}") from exc


def _review_groups(context: SourceContext, candidates: list[int], limit: int, size: int = 16) -> list[list[int]]:
    """Review complete continuation units once, within both request and run caps.

    An oversized unit is left for first-pass/repair rather than reviewing and
    committing separate halves in different requests.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    used = 0
    for unit_id in dict.fromkeys(context.unit_for_id[index] for index in candidates):
        unit = context.units[unit_id]
        if len(unit) > size or used + len(unit) > limit:
            continue
        if len(current) + len(unit) > size:
            groups.append(current)
            current = []
        current.extend(unit)
        used += len(unit)
    if current:
        groups.append(current)
    return groups


def _source_evidence(context: SourceContext, index: int) -> str:
    unit = context.units[context.unit_for_id[index]]
    return "".join(context.rows[member]["ja"] for member in unit)


def validate_suggestion(row: dict, *, source: str, current: str, target_lang: str) -> bool:
    if row.get("issue") not in _ISSUES[1:]:
        return False
    quote, target_quote = row.get("source_quote"), row.get("translation_quote")
    candidate, reason = row.get("suggestion"), row.get("reason")
    if not all(isinstance(value, str) for value in (quote, target_quote, candidate, reason)):
        return False
    if not quote.strip() or quote not in source or len(quote) > 500:
        return False
    if target_quote not in current or (not target_quote.strip() and row["issue"] != "omission"):
        return False
    if not reason.strip() or len(reason) > 600 or not candidate.strip() or candidate.strip() == current.strip():
        return False
    if len(candidate) > max(64, len(source) * 4):
        return False
    return invalid_line_output(source, candidate, target_lang) is None


def _messages(context: SourceContext, texts: list[str], ids: list[int], session: TranslationSession) -> list[dict]:
    focus = context.focus(ids)
    for row in focus["items"]:
        row["current_zh"] = texts[row["id"]]
    return [
        {"role": "system", "content": (
            f"审核日语到{session.target_lang}字幕的语义准确性。不是润色任务。"
            "只检查否定、条件、施受关系、漏掉或添加含义、数量和明确误译。"
            "短句可随语境变化；保留原句语气，不统一普通整句，不纠正或补写 ASR 原文。"
            "结合 semantic_units 判断半句，不把相邻 cue 已表达的内容误报为漏译。"
            "只审 requested_ids；每个 id 返回一个 reviews 项。无明确错误或证据不足时 issue=none，"
            "其余字符串留空。有错时 source_quote 必须逐字摘自该句或它所属完整语义单元，"
            "translation_quote 摘自当前译文（漏译可为空），reason 简述可核实的语义差异，"
            "suggestion 只替换本 id，不把完整语义单元重复写进每条字幕。只输出 JSON。"
            + ("\n用户词汇表：\n" + session.glossary if session.glossary else "")
        )},
        {"role": "user", "content": json.dumps(
            {"requested_ids": ids, **focus}, ensure_ascii=False, separators=(",", ":")
        )},
    ]


def _verification(context: SourceContext, texts: list[str], proposals: dict[int, dict], session: TranslationSession):
    ids = sorted(proposals)
    focus = context.focus(ids)
    candidate_label: dict[int, str] = {}
    for row in focus["items"]:
        index = row["id"]
        if index not in proposals:
            row["translation"] = texts[index]
            continue
        candidate_label[index] = "a" if context.unit_for_id[index] % 2 else "b"
        old, new = texts[index], proposals[index]["suggestion"].strip()
        row["a"], row["b"] = (new, old) if candidate_label[index] == "a" else (old, new)
    messages = [
        {"role": "system", "content": (
            f"依据日语原文独立比较两份{session.target_lang}字幕。不要猜哪份是新译文。"
            "只在一份存在明确语义错误、另一份修正它且没有引入新错误时选 a 或 b；"
            "同义表达或只是风格不同选 equivalent，依据不足选 uncertain。"
            "检查否定、条件、施受关系、数量及新增或遗漏信息；考虑 semantic_units 的其他半句。"
            "每个 requested id 返回一个 verdicts 项，包含 id、verdict、source_quote；"
            "选择 a/b 时必须逐字引用支持判断的本句或完整语义单元原文。只输出 JSON。"
            + ("\n用户词汇表：\n" + session.glossary if session.glossary else "")
        )},
        {"role": "user", "content": json.dumps({"requested_ids": ids, **focus}, ensure_ascii=False)},
    ]
    return messages, candidate_label


def _cache_key(context: SourceContext, texts: list[str], session: TranslationSession, model_identity: str) -> str:
    payload = json.dumps([
        VERSION, context.signature, texts, model_identity, session.target_lang,
        session.glossary, session.profile.cache_signature(), session.repair_reasoning_effort,
        settings.TRANSLATION_SEMANTIC_REVIEW_MAX_IDS,
        settings.TRANSLATION_TEMPERATURE, settings.TRANSLATION_TOP_P,
    ], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_semantic_review(
    segments: list[dict], texts: list[str], *, chat, session: TranslationSession,
    source_context: SourceContext, model_identity: str, on_progress=None,
    cancel_event=None, cache_writer=None,
) -> tuple[list[str], dict | None]:
    if not settings.TRANSLATION_SEMANTIC_REVIEW_ENABLED or session.profile.schema is None:
        return texts, None
    transport_util._raise_if_cancelled(cancel_event)
    candidates = select_candidates(
        segments, settings.TRANSLATION_SEMANTIC_REVIEW_MAX_IDS, translations=texts
    )
    groups = _review_groups(source_context, candidates, settings.TRANSLATION_SEMANTIC_REVIEW_MAX_IDS)
    candidates = [index for group in groups for index in group]
    if not candidates:
        return texts, None
    started = time.perf_counter()
    cache_key = _cache_key(source_context, texts, session, model_identity)
    cache_path = Path(session.cache_path).with_suffix(".review.json") if session.cache_path else None
    if cache_path is not None:
        try:
            saved = json.loads(cache_path.read_text(encoding="utf-8"))
            if saved.get("key") == cache_key and saved.get("complete") is True:
                return texts, {"mode": "semantic_review_cache_hit", "request_count": 0, "elapsed_s": 0.0}
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    result = list(texts)
    changes: list[dict] = []
    usages: list[dict] = []
    reviewed: list[int] = []
    failures: list[str] = []
    failure_details: list[dict] = []
    format_retries: list[dict] = []
    request_count = 0
    proposed_count = 0
    retry_requests_left = 4

    def send(messages, ids, schema):
        nonlocal request_count
        transport_util._raise_if_cancelled(cancel_event)
        request_count += 1
        bounded_schema = json.loads(json.dumps(schema))
        root = next(iter(bounded_schema["properties"]))
        array = bounded_schema["properties"][root]
        array.update(minItems=len(ids), maxItems=len(ids))
        array["items"]["properties"]["id"]["enum"] = list(ids)
        response = chat(
            messages, expected_count=len(ids), response_schema=bounded_schema,
            reasoning_effort=session.repair_reasoning_effort,
            max_tokens=8192, on_usage=usages.append, cancel_event=cancel_event,
        )
        transport_util._raise_if_cancelled(cancel_event)
        return response

    def request_rows(ids, root, messages_for):
        """Recover shape failures by splitting between units, with a run cap.

        Only four additional requests are allowed across review and comparison.
        Endpoint errors and terminal failures never enter this recovery path.
        """
        nonlocal retry_requests_left
        schema = REVIEW_SCHEMA if root == "reviews" else VERIFY_SCHEMA
        try:
            return _parse(send(messages_for(ids), ids, schema), root, ids)
        except RetryableTranslationFormatError as exc:
            cuts = [
                position for position in range(1, len(ids))
                if source_context.unit_for_id[ids[position - 1]] != source_context.unit_for_id[ids[position]]
            ]
            if not cuts or retry_requests_left < 2:
                raise
            midpoint = min(cuts, key=lambda cut: abs(cut - len(ids) / 2))
            retry_requests_left -= 2
            format_retries.append({"stage": root, "ids": list(ids), "reason": str(exc)})
            left = request_rows(ids[:midpoint], root, messages_for)
            right = request_rows(ids[midpoint:], root, messages_for)
            return {**left, **right}

    def persist():
        if changes and cache_writer is not None:
            cache_writer(result)

    try:
        for ids in groups:
            transport_util._emit_progress(on_progress, {"phase": "semantic_review", "reviewed": len(reviewed), "expected": len(candidates)})
            try:
                stage = "reviews"
                rows = request_rows(ids, "reviews", lambda targets: _messages(source_context, result, targets, session))
                reviewed.extend(ids)
                proposals = {
                    index: row for index, row in rows.items()
                    if validate_suggestion(
                        row, source=_source_evidence(source_context, index),
                        current=result[index], target_lang=session.target_lang,
                    )
                    and all(
                        ja not in source_text(segments[index]) or zh in row["suggestion"]
                        for ja, zh in parse_glossary_pairs(session.glossary)
                    )
                }
                proposed_count += len(proposals)
                if not proposals:
                    continue
                stage = "verdicts"
                _, labels = _verification(source_context, result, proposals, session)
                verdicts = request_rows(
                    sorted(proposals), "verdicts",
                    lambda targets: _verification(
                        source_context, result, {index: proposals[index] for index in targets}, session,
                    )[0],
                )
                approved: set[int] = set()
                for index in proposals:
                    verdict = verdicts[index]
                    evidence = verdict.get("source_quote")
                    if verdict.get("verdict") != labels[index] or not isinstance(evidence, str) or not evidence.strip():
                        continue
                    if evidence not in _source_evidence(source_context, index):
                        continue
                    approved.add(index)
                # A continuation group is one meaning. Do not commit half of a
                # proposed rewrite when the other half failed verification.
                for index, proposal in proposals.items():
                    siblings = set(source_context.units[source_context.unit_for_id[index]]) & proposals.keys()
                    if not siblings.issubset(approved):
                        continue
                    candidate = proposal["suggestion"].strip()
                    changes.append({"id": index, "before": result[index], "after": candidate, "issue": proposal["issue"], "source_quote": verdicts[index]["source_quote"]})
                    result[index] = candidate
                persist()
            except TERMINAL_TRANSLATION_ERRORS:
                raise
            except RetryableTranslationFormatError as exc:
                failures.append(type(exc).__name__)
                failure_details.append({"stage": stage, "ids": list(ids), "reason": str(exc)})
                # A bad response shape says nothing about the next independent
                # source group. It stays bounded by the original candidate cap.
                continue
            except Exception as exc:
                failures.append(type(exc).__name__)
                failure_details.append({"stage": stage, "ids": list(ids), "reason": type(exc).__name__})
                # Endpoint-level failure should not buy the same doomed request
                # for every remaining group. The first-pass result stays usable.
                break
    except TERMINAL_TRANSLATION_ERRORS:
        try:
            persist()
        except Exception:
            # A cache failure cannot replace a cancellation, invalidated lease
            # or provider refusal with an unrelated filesystem exception.
            logger.warning("Could not persist semantic-review progress before termination")
        raise

    timing = {
        "mode": "semantic_review", "request_count": request_count,
        "elapsed_s": time.perf_counter() - started, "candidate_ids": candidates,
        "reviewed_ids": reviewed, "proposed_count": proposed_count,
        "accepted_count": len(changes), "changes": changes, "failures": failures,
        "failure_details": failure_details, "format_retries": format_retries,
        **transport_util._merge_usage_metrics(usages),
    }
    if cache_path is not None and not failures:
        staged = cache_path.with_name(f".{cache_path.name}.{uuid4().hex}.tmp")
        try:
            # Persist the key of the returned, cached text, not its superseded
            # input. Otherwise every resume pays to review its own edits again.
            payload = {"key": _cache_key(source_context, result, session, model_identity), "complete": True, "report": timing}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            staged.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            staged.replace(cache_path)
        except OSError:
            pass
        finally:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                pass
    transport_util._emit_progress(on_progress, {"phase": "semantic_review_done", "reviewed": len(reviewed), "accepted": len(changes), "failed": bool(failures)})
    return result, timing
