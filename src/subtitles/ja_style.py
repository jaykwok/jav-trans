"""Japanese subtitle text style per the Netflix Japanese Timed Text Style Guide.

Parallel to `zh_style`, and deliberately a separate module: the two guides
disagree on the points that matter most here. CHS keeps 、 mid-sentence and
wraps at 16 units; the Japanese guide bans 、 and 。 outright and wraps at 13.
Rendering Japanese through the CHS pass gets both wrong.

Like `zh_style`, this is the presentation layer: caches and prompts keep the raw
ASR text and every rule here is applied at SRT write time. All functions are
pure and idempotent, so re-running a write over already-styled text is a no-op.

The rule set implemented (guide sections in parentheses):
- 13 full-width units per line, horizontal subtitles (I.5)
- full-width counts 1, half-width counts 0.5, spaces included (I.5)
- maximum two lines (I.14)
- no 。 or 、; 、 becomes a half-width space, 。 a full-width space (I.17)
- ？ and ！ are full-width, and are followed by a full-width space when a new
  sentence starts on the same line (I.17)

Not implemented, because it cannot be detected from the text: I.17's exception
for an official title that legitimately contains punctuation (e.g. a book or
film title). Such a title loses its commas here like any other text.

The guide's vertical-subtitle limit of 11 units is not used - SRT is horizontal.
"""

from __future__ import annotations

import re
import unicodedata


# I.5 counts by rendered width. `east_asian_width` reports W (wide) and F
# (fullwidth) for kana, kanji and the ideographic space, and Na/H/N for ASCII
# and half-width kana. Ambiguous (A) covers glyphs like … and § that render
# full-width in a Japanese face, so they count as 1 rather than 0.5.
_FULLWIDTH_EAW = frozenset({"W", "F", "A"})

HALFWIDTH_SPACE = " "
FULLWIDTH_SPACE = "　"

_MARKUP_TAG_RE = re.compile(r"</?\s*(?:i|b|u|em|strong|font)(?:\s[^>]*)?>|\{\\[^}]*\}", re.IGNORECASE)
_TERMINAL_COMBO_RE = re.compile(r"[？！]{2,}")
_SPACE_RUN_RE = re.compile(r"[ \t　]+")
# I.17: a full-width space follows ？ or ！ only "when a new sentence starts on
# the same line", so the lookahead requires something after it on that line.
_TERMINAL_MARK_RE = re.compile(r"([？！])(?=[^\s\n])")


def ja_display_units(text: str) -> float:
    units = 0.0
    for char in str(text or ""):
        if char == "\n":
            continue
        units += 1.0 if unicodedata.east_asian_width(char) in _FULLWIDTH_EAW else 0.5
    return units


def _collapse_spaces(line: str) -> str:
    """One space per run, full-width if the run contained one.

    Adjacent replaced punctuation (`。、` -> `　 `) would otherwise stack into
    a gap two or three units wide. The wider mark wins because it came from the
    stronger break: a sentence end outranks a clause end.
    """
    def replace(match: re.Match[str]) -> str:
        return FULLWIDTH_SPACE if FULLWIDTH_SPACE in match.group(0) else HALFWIDTH_SPACE

    return _SPACE_RUN_RE.sub(replace, line)


def _normalize_ja_line(line: str) -> str:
    # Strip after collapsing: a sentence-final 。 has just become a trailing
    # full-width space, and a line may not end on one.
    return _collapse_spaces(line).strip().strip(FULLWIDTH_SPACE).strip()


def normalize_ja_subtitle_text(text: str) -> str:
    normalized = str(text or "")
    normalized = _MARKUP_TAG_RE.sub("", normalized)
    normalized = normalized.replace("?", "？").replace("!", "！")
    normalized = _TERMINAL_COMBO_RE.sub(lambda match: match.group(0)[0], normalized)
    # I.17's two replacements. Order does not matter - neither produces the
    # other's input - but both must run before spaces are collapsed.
    normalized = normalized.replace("、", HALFWIDTH_SPACE)
    normalized = normalized.replace("。", FULLWIDTH_SPACE)
    normalized = _TERMINAL_MARK_RE.sub(r"\1" + FULLWIDTH_SPACE, normalized)
    lines = [_normalize_ja_line(line) for line in normalized.replace("\\n", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


# Break-point preference for the two-line wrap. A space is the guide's own
# break point - after I.17 it is a replaced 、 or 。 - so it is free, as is a
# terminal mark. The rest is 禁則処理: a line may not open on a small kana, a
# long-vowel mark, an iteration mark or a closing bracket, and may not close on
# an opening bracket.
_BREAK_AFTER_FREE = "？！…」』】》〉”"
_BREAK_BEFORE_BAD = "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮーー～？！…、。」』】》〉”々ゝゞ"
_BREAK_AFTER_BAD = "「『【《〈“"


def _is_katakana(char: str) -> bool:
    return "ァ" <= char <= "ヶ"


def _ja_break_cost(text: str, position: int) -> float:
    previous = text[position - 1]
    following = text[position]
    if following in _BREAK_BEFORE_BAD:
        return 4.0
    if previous in _BREAK_AFTER_BAD:
        return 4.0
    if previous.isascii() and previous.isalnum() and following.isascii() and following.isalnum():
        return 6.0
    # A katakana run is one loanword, so cutting it is cutting a word - the same
    # thing as splitting `Alexander`, and priced the same.
    if _is_katakana(previous) and _is_katakana(following):
        return 6.0
    if previous.isspace():
        return 0.0
    if previous in _BREAK_AFTER_FREE:
        return 0.0
    # The one boundary a script transition actually evidences: a trailing kana
    # (particle or okurigana) followed by a new content word.
    if "ぁ" <= previous <= "ゟ" and "一" <= following <= "鿿":
        return 0.5
    # Everywhere else there is no boundary evidence at all - Japanese is unspaced
    # and nothing here tokenizes it. Priced above the balance term's reach (0.6
    # per unit of imbalance) so that a real break point wins even when it leaves
    # the two lines lopsided; `わかりま/した` beating a full-width space is the
    # failure this prevents.
    return 3.0


def wrap_ja_subtitle_text(text: str, *, line_max_units: float = 13.0) -> str:
    """Wrap normalized ja text into at most two lines of ≤13 display units.

    A subtitle never becomes three lines (I.14). Past 26 units the two lines
    simply run long and QC reports them - the same trade `zh_style` makes, and
    the same reason: silently dropping text to satisfy a width is worse than
    reporting a line that is too wide.
    """
    flat = _collapse_spaces(str(text or "").replace("\\n", "\n").replace("\n", HALFWIDTH_SPACE)).strip()
    if not flat:
        return ""
    if line_max_units <= 0 or ja_display_units(flat) <= line_max_units:
        return flat

    total_units = ja_display_units(flat)
    best_position = None
    best_cost = float("inf")
    prefix_units = 0.0
    for position in range(1, len(flat)):
        prefix_units += ja_display_units(flat[position - 1])
        top_units = prefix_units
        bottom_units = total_units - prefix_units
        cost = _ja_break_cost(flat, position)
        # Bottom-heavy: a heavier top line costs, a heavier bottom is merely
        # imbalance. Fragment lines (≤2 units) are all but forbidden. This
        # weighting is inherited from `zh_style` rather than taken from the
        # Japanese guide, which gives examples rather than a rule.
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
    top = flat[:best_position].strip().strip(FULLWIDTH_SPACE).strip()
    bottom = flat[best_position:].strip().strip(FULLWIDTH_SPACE).strip()
    if not top or not bottom:
        return flat
    return f"{top}\n{bottom}"


# I.17 bans these outright, so unlike the CHS counter there is no positional
# rule here - the glyph anywhere in rendered output is the defect.
_BANNED_JA_RE = re.compile(r"[、。?!]|[？！]{2,}|</?\s*i\s*>", re.IGNORECASE)


def count_banned_ja_punctuation(text: str) -> int:
    """Occurrences of punctuation the guide forbids in final ja output (QC hook)."""
    return len(_BANNED_JA_RE.findall(str(text or "")))
