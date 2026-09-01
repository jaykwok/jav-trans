"""Post-translation settled-rendering index (the "extra glossary").

Used to guess this with a dedicated LLM call before translation started: give
it the whole film and ask it to name 10-20 recurring terms with a suggested
Chinese rendering. Retired 2026-09-01 after two measured failures traced to
the same root cause - the guess was fiction until the base pass actually ran:

* Unmetered: the extraction request's usage was never threaded through
  ``on_usage``, so its tokens never appeared in a film's cost total.
* Wrong: asked to invent a rendering for the character name "オナ美" before any
  line of the film had been translated, the model returned "小穴" (the female
  anatomy term) - and that guess then rode into the base pass as a "settled"
  translation every batch was told to keep.

Nothing here calls the model anymore. ``derive_settled_glossary`` runs after
the base pass instead of before it, over lines that already have a real
translation: when the same Japanese line was sent more than once (this
genre's dialogue repeats constantly - moans, stock phrases, scene-transition
lines) and the base pass rendered it the same way most of the time, that
majority rendering is a fact about what the film actually says, not a guess
about what it might. Profiles opt in via ``wants_repair_pass`` (there being
no earlier point at which this index could exist, it never reaches the base
pass - only the repair pass, as extra context and as a new self-consistency
detector). The on-disk artifact keeps the pre-2026-09-01 filename and
``{"terms":[{"ja":...,"zh":...}]}`` shape so ``pipeline/quality.py``'s reader
needs no change; only who writes it, and when, moved.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from llm import prompt as prompt_module
from llm.glossary import parse_glossary_pairs

_KANA_RE = re.compile(r"[぀-ヿㇰ-ㇿ]")
_HAN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def _is_usable_target(zh: str) -> bool:
    """A target that is not Chinese teaches the wrong lesson, so it is dropped.

    These pairs are injected back into the repair prompt as settled
    translations, so a target carrying kana, or written in Latin letters, is an
    instruction to leave Japanese in the subtitle. Structural rather than
    prompt-dependent on purpose: a model told to answer in Chinese has still
    returned bare romanised names for a whole line before now.
    """
    if _KANA_RE.search(zh):
        return False
    return bool(_HAN_RE.search(zh))


# `parse_glossary_pairs` splits items on "," / "，" / "\n" and each item on the
# first "-", so a candidate line carrying any of those characters would either
# be torn into the wrong number of items or split at the wrong point. Short
# extracted terms rarely contained one; full dialogue lines - which is what
# this module produces now - occasionally do (romanised names use a space,
# not a dash, but a stray "-" or mid-line comma is still possible).
_UNSAFE_PAIR_CHARS_RE = re.compile(r"[,，\n\r\-]")


def _is_safe_pair_field(text: str) -> bool:
    return bool(text) and not _UNSAFE_PAIR_CHARS_RE.search(text)


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
    """Settled renderings, minus the ones that would argue with the user's glossary.

    The project glossary is the only place the user states an exact word, so a
    settled rendering that disagrees with it has to lose. Suppression covers
    spelling variants of the same word, not just exact keys - see
    `_term_variant_of` - because the source line carrying the disagreement is
    still the film's own wording, mosaic censoring included.

    Only conflicting mappings are dropped; a variant that agrees with the
    glossary is kept, since it reinforces rather than competes.
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


# A group smaller than this proves nothing about which rendering is the norm.
_MIN_GROUP_SIZE = 2
# Below this share, the group disagrees with itself about how to render the
# line - exactly the case with no settled answer to hand back, so it is left
# alone rather than picking an arbitrary winner.
_MIN_DOMINANT_SHARE = 0.6
_MAX_SETTLED_TERMS = 20
_MAX_LINE_LEN = 40


def derive_settled_glossary(
    all_ja_texts: list[str],
    zh_texts: list[str],
    *,
    glossary: str = "",
) -> list[dict]:
    """The dominant rendering for every Japanese line translated more than once.

    Pure text statistics over the base pass's own output - no request, no
    guess. A line is included only when it recurred (``_MIN_GROUP_SIZE``) and
    the base pass rendered it the same way most of the time
    (``_MIN_DOMINANT_SHARE``); a line the base pass itself rendered
    inconsistently has no settled answer to report, so it is skipped rather
    than picking whichever rendering happened to be more common by one.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for ja, zh in zip(all_ja_texts, zh_texts):
        key = prompt_module._normalize_source_text(ja)
        zh_clean = str(zh or "").strip()
        if not key or not zh_clean:
            continue
        if len(key) > _MAX_LINE_LEN or not _is_safe_pair_field(key):
            continue
        groups[key].append(zh_clean)

    candidates: list[tuple[int, str, str]] = []
    for ja, renderings in groups.items():
        if len(renderings) < _MIN_GROUP_SIZE:
            continue
        dominant_zh, dominant_count = Counter(renderings).most_common(1)[0]
        if dominant_count / len(renderings) < _MIN_DOMINANT_SHARE:
            continue
        if not _is_safe_pair_field(dominant_zh) or not _is_usable_target(dominant_zh):
            continue
        candidates.append((dominant_count, ja, dominant_zh))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [
        {"ja": ja, "zh": zh} for _count, ja, zh in candidates[:_MAX_SETTLED_TERMS]
    ]


def _write_settled_glossary(path: str, terms: list[dict]) -> None:
    try:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"terms": terms}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[WARN] failed to write translation settled glossary: {exc}")


def resolve_settled_glossary(
    segments: list[dict],
    zh_texts: list[str],
    cache_path: str,
    glossary: str,
) -> str:
    """The formatted repair-context block, derived from what the film actually says.

    Writes the same on-disk artifact the pre-extraction used to (path keyed by
    a digest of the source text, same ``{"terms":[...]}`` shape) so
    `pipeline/quality.py`'s reader keeps working unchanged - only the content
    is now measured rather than guessed, and it exists only once there is
    something to measure.
    """
    all_ja_texts = [str(seg.get("text", "")) for seg in segments]
    terms = derive_settled_glossary(all_ja_texts, zh_texts, glossary=glossary)
    if cache_path:
        _write_settled_glossary(
            _global_glossary_cache_path_for_texts(cache_path, all_ja_texts), terms
        )
    return _format_global_glossary_terms(terms, glossary=glossary)
