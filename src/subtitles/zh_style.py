"""Chinese (Simplified) subtitle text style per the Netflix CHS Timed Text Style Guide.

This is the presentation layer: translation caches keep the raw LLM output and
every rule here is applied at SRT write time. All functions are pure and
idempotent, so re-running a write over already-normalized text is a no-op.

The rule set implemented (guide sections in parentheses):
- no commas or periods; a single space replaces them (Punctuation)
- 、 only mid-sentence, never at the end of a line or subtitle (Punctuation)
- ellipsis is a single U+2026, never runs of dots or U+22EF (Punctuation)
- full-width ？！, and no ?!/??/!! combinations (Punctuation)
- half-width Arabic numerals (Numbers)
- full-width double quotes (Quotes)
- no italics or other markup (Formatting)
- 16 full-width units per line, 2 lines max, bottom-heavy pyramid (Line Treatment)
"""

from __future__ import annotations

import re


# Weight table mirrors subtitles.writer._count_text_units: a full-width glyph
# occupies one column, ASCII roughly half. Spaces count here (unlike the
# reading-speed counter) because a space substitutes a comma and still takes
# display width on the line.
ASCII_ALNUM_WEIGHT = 0.55
ASCII_OTHER_WEIGHT = 0.35

_MARKUP_TAG_RE = re.compile(r"</?\s*(?:i|b|u|em|strong|font)(?:\s[^>]*)?>|\{\\[^}]*\}", re.IGNORECASE)
_FULLWIDTH_DIGIT_TABLE = {ord(full): ord("0") + offset for offset, full in enumerate("０１２３４５６７８９")}
_ELLIPSIS_SOURCE_RE = re.compile(r"(?:[…⋯‥]|\.{3,}|。{2,}|・{3,}|·{3,})+")
_ELLIPSIS_RUN_RE = re.compile(r"…{2,}")
_TERMINAL_COMBO_RE = re.compile(r"[？！]{2,}")
_NUMERIC_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")
_NUMERIC_COMMA_SENTINEL = "\x00"
_COMMA_PERIOD_RE = re.compile(r"[，,。]")
_HALFWIDTH_QUOTE_RE = re.compile(r'"([^"\n]*)"')
_SPACE_RUN_RE = re.compile(r"[ \t　]+")
_TRAILING_ENUM_COMMA_RE = re.compile(r"、+$")


def zh_display_units(text: str, *, ascii_alnum_weight: float = ASCII_ALNUM_WEIGHT) -> float:
    units = 0.0
    for char in str(text or ""):
        if char == "\n":
            continue
        if char.isascii() and char.isalnum():
            units += ascii_alnum_weight
        elif char.isascii():
            units += ASCII_OTHER_WEIGHT
        else:
            units += 1.0
    return units


def _normalize_zh_line(line: str) -> str:
    line = _SPACE_RUN_RE.sub(" ", line).strip()
    line = _TRAILING_ENUM_COMMA_RE.sub("", line)
    return line.strip()


def normalize_zh_subtitle_text(text: str) -> str:
    normalized = str(text or "")
    normalized = _MARKUP_TAG_RE.sub("", normalized)
    normalized = normalized.translate(_FULLWIDTH_DIGIT_TABLE)
    # Ellipsis first: dot runs must collapse before the single 。→space rule
    # can see a genuine period, and 、、、-style runs are not ellipses so they
    # fall through to the trailing-、 strip instead.
    normalized = _ELLIPSIS_SOURCE_RE.sub("…", normalized)
    normalized = _ELLIPSIS_RUN_RE.sub("…", normalized)
    normalized = normalized.replace("?", "？").replace("!", "！")
    normalized = _TERMINAL_COMBO_RE.sub(lambda match: match.group(0)[0], normalized)
    # Thousands separators stay (half-width numerals rule); every other comma
    # and every period becomes the guide's single space.
    normalized = _NUMERIC_COMMA_RE.sub(_NUMERIC_COMMA_SENTINEL, normalized)
    normalized = _COMMA_PERIOD_RE.sub(" ", normalized)
    normalized = normalized.replace(_NUMERIC_COMMA_SENTINEL, ",")
    normalized = _HALFWIDTH_QUOTE_RE.sub("“\\1”", normalized)
    lines = [_normalize_zh_line(line) for line in normalized.replace("\\n", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


# Break-point preference for the two-line pyramid: a space is the guide's own
# break point (it is the replaced comma or period, and General Requirements asks
# for the break "after punctuation marks"), so it is free; after sentence
# punctuation is likewise free, between hiragana→kanji readable, anywhere else
# expensive. Splitting inside an ASCII word or number is effectively forbidden.
_BREAK_AFTER_FREE = "！？…”》」』"
_BREAK_BEFORE_BAD = "！？…、”》」』ー～"
# 、 is not a break point: the guide allows it mid-sentence but not "at the end
# of a line or subtitle", which is exactly what breaking after one produces.
_BREAK_AFTER_BANNED = "、"


def _zh_break_cost(text: str, position: int) -> float:
    previous = text[position - 1]
    following = text[position]
    if following in _BREAK_BEFORE_BAD:
        return 4.0
    if previous.isascii() and previous.isalnum() and following.isascii() and following.isalnum():
        return 6.0
    if previous in _BREAK_AFTER_BANNED:
        return 4.0
    if previous in _BREAK_AFTER_FREE:
        return 0.0
    if previous.isspace():
        return 0.0
    if "ぁ" <= previous <= "ゟ" and "一" <= following <= "鿿":
        return 0.5
    return 1.0


def wrap_zh_subtitle_text(
    text: str,
    *,
    line_max_units: float = 16.0,
    ascii_alnum_weight: float = ASCII_ALNUM_WEIGHT,
) -> str:
    """Wrap normalized zh text into at most two lines of ≤16 display units.

    A single subtitle never becomes three lines: past 32 units the two lines
    simply run long and QC reports them, matching the guide's own "when
    unavoidable" allowance.
    """
    flat = _SPACE_RUN_RE.sub(" ", str(text or "").replace("\\n", "\n").replace("\n", " ")).strip()
    if not flat:
        return ""
    if line_max_units <= 0 or zh_display_units(flat, ascii_alnum_weight=ascii_alnum_weight) <= line_max_units:
        return flat

    total_units = zh_display_units(flat, ascii_alnum_weight=ascii_alnum_weight)
    best_position = None
    best_cost = float("inf")
    prefix_units = 0.0
    for position in range(1, len(flat)):
        char = flat[position - 1]
        if char.isascii() and char.isalnum():
            prefix_units += ascii_alnum_weight
        elif char.isascii():
            prefix_units += ASCII_OTHER_WEIGHT
        else:
            prefix_units += 1.0
        top_units = prefix_units
        bottom_units = total_units - prefix_units
        cost = _zh_break_cost(flat, position)
        # Bottom-heavy pyramid: a heavier top line costs, a heavier bottom is
        # merely imbalance. Fragment lines (≤2 units) are all but forbidden.
        if top_units > bottom_units:
            cost += (top_units - bottom_units) * 0.6
        else:
            cost += (bottom_units - top_units) * 0.15
        if top_units <= 2.0 or bottom_units <= 2.0:
            cost += 8.0
        cost += max(0.0, top_units - line_max_units) * 3.0
        cost += max(0.0, bottom_units - line_max_units) * 3.0
        if cost < best_cost:
            best_cost = cost
            best_position = position

    if best_position is None:
        return flat
    top = flat[:best_position].strip()
    bottom = flat[best_position:].strip()
    # Only the boundary this function creates is its to repair: `、` is banned at
    # a line end, and every other line end in `flat` was already normalized.
    # Breaking here costs 4.0 above, so this fires only when every alternative
    # break was worse still.
    top = _TRAILING_ENUM_COMMA_RE.sub("", top).strip()
    if not top or not bottom:
        return flat
    return f"{top}\n{bottom}"


_BANNED_DISPLAY_RE = re.compile(r"[，。,⋯‥]|\.{3,}|…{2,}|[?!]|[？！]{2,}|</?\s*i\s*>", re.IGNORECASE)
# The one positional rule in the set: 、 is legal mid-sentence and banned "at the
# end of a line or subtitle", so it is counted by where it sits, not by glyph.
_LINE_FINAL_ENUM_COMMA_RE = re.compile(r"、[ \t　]*(?:\n|$)")


def count_banned_punctuation(text: str) -> int:
    """Occurrences of punctuation the guide forbids in final zh output (QC hook)."""
    source = str(text or "")
    matches = len(_LINE_FINAL_ENUM_COMMA_RE.findall(source))
    for match in _BANNED_DISPLAY_RE.finditer(source):
        if (
            match.group(0) == ","
            and match.start() > 0
            and match.end() < len(source)
            and source[match.start() - 1].isdigit()
            and source[match.end()].isdigit()
        ):
            continue
        matches += 1
    return matches
