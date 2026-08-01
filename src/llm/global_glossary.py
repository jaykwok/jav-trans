"""Whole-film glossary pre-extraction (the "extra glossary").

One full-source request that mines 10-20 recurring terms (pronouns, names,
anatomy, high-frequency adjectives) with suggested translations, cached next
to the translation cache keyed by a digest of all source lines. Profiles opt
in via ``wants_extra_glossary``; the result is folded into every batch prompt
and into cache/memory keys. The ``chat`` callable comes from the caller.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Callable

from llm import transport_util
from llm.errors import TranslationCancelledError
from llm.glossary import parse_glossary_pairs
from llm.profiles import json_v3

_raise_if_cancelled = transport_util._raise_if_cancelled

_GLOSSARY_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ja": {"type": "string"},
                    "zh": {"type": "string"},
                },
                "required": ["ja", "zh"],
            },
        },
    },
    "required": ["terms"],
}


def _filter_global_glossary_terms(raw_terms) -> list[dict]:
    if not isinstance(raw_terms, list):
        return []
    filtered: list[dict] = []
    banned_re = re.compile(r"[,、。，？?？\s]")
    for item in raw_terms:
        if not isinstance(item, dict):
            continue
        ja = str(item.get("ja", "")).strip()
        zh = str(item.get("zh", "")).strip()
        if not ja or not zh:
            continue
        if len(ja) > 8 or len(zh) > 8:
            continue
        if banned_re.search(ja) or banned_re.search(zh):
            continue
        filtered.append({"ja": ja, "zh": zh})
        if len(filtered) >= 15:
            break
    return filtered


def _format_global_glossary_terms(
    terms: list[dict],
    *,
    glossary: str = "",
) -> str:
    lines = []
    seen: set[str] = set()
    # Parse the project glossary into its ja keys and match exactly. Treating
    # the whole glossary text as a haystack (substring `in`) would drop a
    # global term whenever its ja appeared as a substring of any glossary pair
    # (e.g. glossary "肉-肉棒" wrongly suppresses both "肉" and "肉棒").
    glossary_ja_keys = {ja for ja, _zh in parse_glossary_pairs(glossary)}
    for item in terms:
        ja = str(item.get("ja", "")).strip()
        zh = str(item.get("zh", "")).strip()
        if not ja or not zh or ja in seen:
            continue
        if ja in glossary_ja_keys:
            continue
        seen.add(ja)
        lines.append(f"{ja}-{zh}")
    return "\n".join(lines)


def _global_glossary_cache_path_for_texts(
    translation_cache_path: str,
    all_ja_texts: list[str],
) -> str:
    cache_path = Path(translation_cache_path)
    source_sig = hashlib.sha1(
        "\n".join(str(text or "") for text in all_ja_texts).encode("utf-8")
    ).hexdigest()[:12]
    return str(cache_path.with_name(f"translation_global_glossary.{source_sig}.json"))


def resolve_extra_glossary(
    segments: list[dict],
    cache_path: str,
    glossary: str,
    *,
    chat: Callable[..., str],
    cancel_event: threading.Event | None,
) -> str:
    if not cache_path:
        return ""
    all_ja_texts = [str(seg.get("text", "")) for seg in segments]
    glossary_terms = extract_global_glossary(
        all_ja_texts,
        _global_glossary_cache_path_for_texts(cache_path, all_ja_texts),
        chat=chat,
        cancel_event=cancel_event,
    )
    return _format_global_glossary_terms(glossary_terms, glossary=glossary)


def extract_global_glossary(
    all_ja_texts: list[str],
    cache_path: str,
    *,
    chat: Callable[..., str],
    cancel_event: threading.Event | None = None,
) -> list[dict]:
    _raise_if_cancelled(cancel_event)
    if not cache_path:
        return []
    path = Path(cache_path)
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            terms = payload.get("terms") if isinstance(payload, dict) else payload
            return _filter_global_glossary_terms(terms)
    except Exception as exc:
        print(f"[WARN] failed to load translation global glossary cache: {exc}")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        source_text = "\n".join(str(text or "") for text in all_ja_texts)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是字幕术语提取器。请从全片日文字幕中提取 10-20 个反复出现的核心词，"
                    "范围包括代词、人名、性器官词、高频形容词。给出推荐中文翻译。"
                    '只返回合法 JSON：{"terms":[{"ja":"...","zh":"..."}]}。'
                ),
            },
            {"role": "user", "content": f"【全片日文字幕】\n{source_text}"},
        ]
        raw_output = chat(
            messages,
            expected_count=0,
            cancel_event=cancel_event,
            response_schema=_GLOSSARY_OUTPUT_SCHEMA,
            response_schema_name="translation_glossary",
        )
        _raise_if_cancelled(cancel_event)
        parsed = json.loads(json_v3._strip_reasoning_artifacts(raw_output))
        terms = _filter_global_glossary_terms(
            parsed.get("terms") if isinstance(parsed, dict) else None
        )
        tmp_path = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
        tmp_path.write_text(
            json.dumps({"terms": terms}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        return terms
    except Exception as exc:
        if isinstance(exc, TranslationCancelledError):
            raise
        print(f"[WARN] failed to extract translation global glossary: {exc}")
        return []
