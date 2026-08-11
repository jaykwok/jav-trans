"""Decide whether a subtitle cue is nothing but non-semantic vocalisation.

**Why this is a filter and not a model change.** The local ASR transcribes moans
as text, and forced alignment cannot refuse text it is handed - every character
gets at least one frame by construction. So the only place this can be answered
is after transcription, on the words themselves.

**Why a character class alone does not work.** The first version of this module
classified by character set, and on a real film it deleted `イッちゃう、イッちゃう
…はぁ、はぁ、はぁ` and `はぁはぁ…れろちんぽ、れろちんぽ…` - because `ち`, `ゅ`,
`ぽ` and `ぐ` had to be in the class to catch `ちゅっ` and `んぐっ`, and those same
kana spell ordinary words. Vocalisation and speech draw on one inventory; no
partition of the kana separates them.

So the test is a decomposition instead. A cue is vocalisation only if the whole
of it can be consumed by

  * `_CORE_CHARS` - kana that carry no lexical content standing alone, and
  * `_VOCAL_MORPHEMES` - an explicit list of vocalisation syllables that need
    letters the core cannot have.

Anything left over means a word was in there, and the cue is kept. That makes the
list an allow-list of noise rather than a deny-list of letters: adding `ちゅ` can
only ever let `ちゅ` through, never `ちんぽ`.

**The asymmetry is deliberate.** A false drop deletes something the speaker said
and the viewer can never recover it; a false keep leaves one `あっ` on screen. So
every ambiguity resolves toward keeping: `_PROTECTED` is consulted first and also
per comma-separated part, laughter is kept, and anything that fails to decompose
is kept.
"""
from __future__ import annotations

import unicodedata

# Kana that carry no lexical content standing alone. Deliberately does NOT
# include the consonant rows - see the module docstring for what that cost.
_CORE_CHARS = set(
    "あぁいぃうぅえぇおぉっん"
    "はひふへほ"  # panting: はぁ, ふぅ, へぇ - `はい`/`へえ` are in _PROTECTED
    "アァイィウゥエェオォッン"
    "ハヒフヘホ"
    "ー〜～"
)

# Vocalisation syllables that need letters `_CORE_CHARS` cannot contain. Matched
# longest-first, and only ever as whole units, so they cannot combine with a
# stray kana to swallow a word.
_VOCAL_MORPHEMES = (
    "ちゅぱ", "ちゅる", "ちゅぷ", "ぐちゅ", "ぴちゃ", "じゅる", "ぬぷ",
    "ちゅ", "れろ", "ぺろ", "んぐ", "んむ", "ちゅっ", "ずず", "じゅ",
)

# Both the ASCII and the full-width forms. This filter runs BEFORE the render
# stage normalises punctuation, so it sees `ぅ...ちゅぷっ、` where the finished
# subtitle reads `ぅ…ちゅぷっ`. With only the full-width forms listed, a real film
# flagged 120 cues and dropped 52; with these added it flags 349 and drops 224.
# The trap is that the finished SRT barely shows it - measured on the rendered
# file the two sets differ by 5 cues, because by then the ellipses are already
# `…`. The gap only exists at the point where the filter actually runs.
_DECORATION = set("、。！？!?…‥・「」『』（）()♪ 　　" + ".,-~\"'`;:")

# Short utterances built only from the above that are nevertheless real speech.
# Explicit, because nothing orthographic separates them from a moan: `うん` and
# `んん` differ by meaning, not by sound class. Laughter is here on purpose - it
# is non-semantic but communicative, and a viewer notices when a laugh vanishes.
_PROTECTED = frozenset(
    {
        "うん", "ううん", "うんうん", "うんっ", "うーん", "ううーん",
        "はい", "はいはい", "はーい", "いいえ", "いえ",
        "ええ", "えー", "えっ", "え",
        "ええと", "えっと", "えーと", "うーんと",
        "ふふ", "ふふっ", "ふふふ", "ふふふふ", "へへ", "へへっ",
        "ほほ", "はは", "ははっ", "あはは", "うふふ", "えへへ",
    }
)

_MORPHEMES_LONGEST_FIRST = tuple(
    sorted(_VOCAL_MORPHEMES, key=len, reverse=True)
)


def _strip_decoration(text: str) -> str:
    return "".join(
        ch for ch in str(text or "") if ch not in _DECORATION and not ch.isspace()
    )


def _carries_lexical_content(text: str) -> bool:
    """Kanji, latin or digits settle it immediately, without decomposition."""
    for ch in text:
        if ch.isdigit() or (ch.isascii() and ch.isalpha()):
            return True
        if "CJK UNIFIED" in unicodedata.name(ch, ""):
            return True
    return False


def _decomposes_into_vocalisation(body: str) -> bool:
    index = 0
    length = len(body)
    while index < length:
        for morpheme in _MORPHEMES_LONGEST_FIRST:
            if body.startswith(morpheme, index):
                index += len(morpheme)
                break
        else:
            if body[index] in _CORE_CHARS:
                index += 1
            else:
                # A letter neither the core nor the noise list accounts for.
                # Something was said here.
                return False
    return True


def _parts(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    for ch in str(text or ""):
        if ch in _DECORATION or ch.isspace():
            if current:
                parts.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def is_non_semantic_vocalisation(text: str) -> bool:
    """True when the cue is only vocalisation and may be dropped.

    Empty text is not vocalisation - it is nothing, and what an empty cue means
    is the caller's decision, not this module's.
    """
    body = _strip_decoration(text)
    if not body:
        return False
    if body in _PROTECTED or _carries_lexical_content(body):
        return False
    # `あ、うん` still contains the reply, so the allow-list is consulted on the
    # parts as well as on the whole.
    if any(part in _PROTECTED for part in _parts(text)):
        return False
    return _decomposes_into_vocalisation(body)


def block_text(block: dict) -> str:
    """The Japanese text of a cue, under whichever key currently holds it.

    Cues carry `ja_text` through the layout stage and only gain a plain `text`
    later. Reading one key alone silently classified every cue as empty, and an
    empty cue is not vocalisation - so the filter reported zero on a film where
    it should have found hundreds, and looked like it was working.
    """
    for key in ("ja_text", "text"):
        value = str(block.get(key) or "").strip()
        if value:
            return value
    return ""


def drop_vocalisation_runs(
    blocks: list[dict],
    *,
    min_run: int = 2,
) -> tuple[list[dict], dict]:
    """Remove runs of consecutive cues that are only vocalisation.

    **Only runs.** A lone vocalisation cue between two lines of dialogue is far
    more likely to be a real reaction - a gasp answering something said - than a
    fragment of a moaning passage, and the lexicon is not good enough to tell
    them apart on text alone. Requiring neighbours substitutes context for
    lexicon: it is one rule rather than a growing list of special cases, and it
    degrades safely, because a gap in the word list simply breaks the run.

    Returns the surviving blocks and a diagnostics dict. Nothing is filtered when
    `min_run` is below 1.
    """
    flags = [
        bool(is_non_semantic_vocalisation(block_text(block))) for block in blocks
    ]
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flagged in enumerate(flags):
        if flagged and start is None:
            start = index
        elif not flagged and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(flags)))

    doomed: set[int] = set()
    dropped_runs = 0
    for begin, end in runs:
        if end - begin >= max(1, int(min_run)):
            doomed.update(range(begin, end))
            dropped_runs += 1

    kept = [block for index, block in enumerate(blocks) if index not in doomed]
    diagnostics = {
        "vocalisation_cues_flagged": sum(flags),
        "vocalisation_cues_dropped": len(doomed),
        "vocalisation_runs_dropped": dropped_runs,
        # Flagged but left alone because they stood by themselves. Worth seeing:
        # if this grows large the run threshold is doing the real work, not the
        # word list.
        "vocalisation_cues_kept_as_isolated": sum(flags) - len(doomed),
        "vocalisation_min_run": max(1, int(min_run)),
    }
    return kept, diagnostics
