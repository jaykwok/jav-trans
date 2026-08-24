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


_KANA_RE = re.compile(r"[぀-ヿㇰ-ㇿ]")
_HAN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def _is_usable_target(zh: str) -> bool:
    """A target that is not Chinese teaches the wrong lesson, so it is dropped.

    These pairs are injected back into the batch prompt as settled translations,
    so a target carrying kana, or written in Latin letters, is an instruction to
    leave Japanese in the subtitle. Structural rather than prompt-dependent on
    purpose: the same model that was told to answer in Chinese returned `ジェイ-Jay`
    and `シルス-Sirusu` anyway, and the resulting run echoed 239 cues verbatim.
    """
    if _KANA_RE.search(zh):
        return False
    return bool(_HAN_RE.search(zh))


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
        if not _is_usable_target(zh):
            continue
        filtered.append({"ja": ja, "zh": zh})
        if len(filtered) >= 15:
            break
    return filtered


# Mosaic characters stand in for one censored kana (ち○ぽ, おま○こ). They are a
# spelling of the same word, so a term written with them has to be compared as
# "any one character here" or the variant walks straight past every check.
_MOSAIC_RE = re.compile(r"[○〇●◯\*＊]")


def _spellings_match(needle: str, window: str) -> bool:
    """Same length, char for char, a mosaic on either side matching anything."""
    return len(needle) == len(window) and all(
        left == right or _MOSAIC_RE.match(left) or _MOSAIC_RE.match(right)
        for left, right in zip(needle, window)
    )


def _term_variant_of(ja: str, glossary_ja: str) -> bool:
    """One of these two spellings occurs inside the other.

    Scanned by hand rather than by regex because the mosaic can sit on either
    side: `おち○ぽ` has to match the glossary's `ちんぽ`, and building the pattern
    from just one of the two strings only ever wildcards that one.
    """
    for needle, haystack in ((glossary_ja, ja), (ja, glossary_ja)):
        if not needle or not haystack or len(needle) > len(haystack):
            continue
        span = len(needle)
        if any(
            _spellings_match(needle, haystack[start : start + span])
            for start in range(len(haystack) - span + 1)
        ):
            return True
    return False


def _format_global_glossary_terms(
    terms: list[dict],
    *,
    glossary: str = "",
) -> str:
    """Extracted terms, minus the ones that would argue with the user's glossary.

    The project glossary is the only place the user states an exact word, so an
    extracted term that disagrees with it has to lose. Suppression covers
    spelling variants of the same word, not just exact keys: measured on
    sample-v 2026-08-24 with a glossary of `ちんぽ-肉棒`, the extractor mined the
    film's own wording and returned `ちんぽ-鸡巴` **plus** `ちんちん-鸡巴`,
    `おちんぽ-鸡巴`, `ち○ぽ-鸡巴`, `おち○ちん-鸡巴`. Exact-key matching dropped two
    of them and injected the rest, so the prompt told the model 肉棒 and 鸡巴 for
    the same word in the same breath - and the model generalised across the
    family: 6 of 37 cues came back 鸡巴. Thinking made it worse, because
    deliberating between two stated rules is what picking one looks like.

    Only conflicting mappings are dropped; a variant that agrees with the
    glossary is kept, since it reinforces rather than competes. The residual
    risk is over-suppression when a glossary key is a short substring of an
    unrelated compound - that costs one extracted hint, against a measured 16%
    failure rate on exactly the terms the user asked for by name.
    """
    lines = []
    seen: set[str] = set()
    glossary_pairs = parse_glossary_pairs(glossary)
    glossary_ja_keys = {ja for ja, _zh in glossary_pairs}
    for item in terms:
        ja = str(item.get("ja", "")).strip()
        zh = str(item.get("zh", "")).strip()
        if not ja or not zh or ja in seen:
            continue
        # An exact key is already stated in the prompt by the glossary itself.
        if ja in glossary_ja_keys:
            continue
        if any(
            _term_variant_of(ja, glossary_ja) and zh != glossary_zh
            for glossary_ja, glossary_zh in glossary_pairs
        ):
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
    messages: list[dict] | None = None,
) -> tuple[str, bool]:
    """Returns the formatted block and whether a request was actually issued.

    The caller needs the second value to decide about prefix warmup: when this
    did send a request it already warmed the provider's prefix, and a separate
    warmup would just buy the same prefix twice. A cache hit sends nothing, so
    the warmup still has work to do.
    """
    if not cache_path:
        return "", False
    all_ja_texts = [str(seg.get("text", "")) for seg in segments]
    glossary_terms, issued_request = _extract_global_glossary(
        all_ja_texts,
        _global_glossary_cache_path_for_texts(cache_path, all_ja_texts),
        chat=chat,
        cancel_event=cancel_event,
        messages=messages,
    )
    return (
        _format_global_glossary_terms(glossary_terms, glossary=glossary),
        issued_request,
    )


def _fallback_extraction_messages(all_ja_texts: list[str]) -> list[dict]:
    """Standalone shape, for callers with no translation prefix to share."""
    source_text = "\n".join(str(text or "") for text in all_ja_texts)
    return [
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


def _extract_global_glossary(
    all_ja_texts: list[str],
    cache_path: str,
    *,
    chat: Callable[..., str],
    cancel_event: threading.Event | None = None,
    messages: list[dict] | None = None,
) -> tuple[list[dict], bool]:
    _raise_if_cancelled(cancel_event)
    if not cache_path:
        return [], False
    path = Path(cache_path)
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            terms = payload.get("terms") if isinstance(payload, dict) else payload
            return _filter_global_glossary_terms(terms), False
    except Exception as exc:
        print(f"[WARN] failed to load translation global glossary cache: {exc}")

    issued_request = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        request_messages = messages or _fallback_extraction_messages(all_ja_texts)
        issued_request = True
        raw_output = chat(
            request_messages,
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
        return terms, issued_request
    except Exception as exc:
        if isinstance(exc, TranslationCancelledError):
            raise
        print(f"[WARN] failed to extract translation global glossary: {exc}")
        # A request that failed still warmed the prefix if it reached the
        # provider, but that cannot be told apart from one that never left, so
        # the warmup is left to run rather than skipped on a maybe.
        return [], False


def extract_global_glossary(
    all_ja_texts: list[str],
    cache_path: str,
    *,
    chat: Callable[..., str],
    cancel_event: threading.Event | None = None,
    messages: list[dict] | None = None,
) -> list[dict]:
    terms, _issued = _extract_global_glossary(
        all_ja_texts,
        cache_path,
        chat=chat,
        cancel_event=cancel_event,
        messages=messages,
    )
    return terms
