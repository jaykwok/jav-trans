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

from collections import Counter
from dataclasses import dataclass
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
#
# `いい` and its lengthenings were the expensive omission. It is spelled from one
# core kana repeated, so it decomposed, and this module is shared with the CTC
# training-target builder - so the same verdict both deleted the cue in
# production and taught the alignment head that a breathy `いい` is blank. On the
# archived NSFW strip manifest it was removed 169 times and kept 0 times (`いい`
# 113, `いいっ` 56), while `いいよ`/`いいの` survived only because they carry a
# kana the core set does not have. In this domain it is a high-frequency word.
#
# `いっ`/`うっ`/`おっ` are deliberately NOT here: `イッ(く)` truncations and bare
# grunts have no clean lexical boundary, and the acoustic verdict is the right
# instrument for them, not the word list.
_PROTECTED = frozenset(
    {
        "うん", "ううん", "うんうん", "うんっ", "うーん", "ううーん",
        "はい", "はいはい", "はーい", "いいえ", "いえ",
        "ええ", "えー", "えっ", "え",
        "ええと", "えっと", "えーと", "うーんと",
        "いい", "いいー", "おい", "おーい", "おう",
        "ふーん", "ふうん", "へぇ", "へえ",
        "ふふ", "ふふっ", "ふふふ", "ふふふふ", "へへ", "へへっ",
        "ほほ", "はは", "ははっ", "あはは", "うふふ", "えへへ",
    }
)

_MORPHEMES_LONGEST_FIRST = tuple(
    sorted(_VOCAL_MORPHEMES, key=len, reverse=True)
)


def _is_protected(fragment: str) -> bool:
    """Allow-list membership, with an emphatic trailing `っ` forgiven.

    The list was enumerated by hand and the emphatic forms fell through the
    gaps: `はい` was protected but `はいっ` was stripped 48 times in the archived
    NSFW targets, `いえ` but not `いえっ`. Written out, the exceptions would
    double the list and the next one would be missed the same way.

    Only a trailing `っ` is forgiven, and only after the entry itself failed to
    match. That cannot promote anything new on its own: `いっ`, `うっ`, `おっ`
    reduce to `い`, `う`, `お`, which are not protected and must not be - a bare
    grunt and a truncated `イッ(く)` have no lexical boundary between them, and
    the acoustic verdict is the right instrument for those, not this list.
    """
    if fragment in _PROTECTED:
        return True
    return fragment.endswith("っ") and fragment[:-1] in _PROTECTED


def _strip_decoration(text: str) -> str:
    return "".join(
        ch for ch in str(text or "") if ch not in _DECORATION and not ch.isspace()
    )


def is_decoration_only(text: str) -> bool:
    """True when nothing but punctuation, spaces and music marks remains.

    Public because the writer needs the same reading: a cue that renders to
    `…♪` has nothing for a viewer to read, and the answer has to come from the
    one `_DECORATION` set rather than a second list that drifts away from it.

    Note this is not the vocalisation verdict. `_cue_verdict` keeps such a cue
    (empty body, nothing to judge) because it decides whether kana are moaning;
    whether an unreadable cue should reach the screen is the writer's question.
    """
    return bool(str(text or "").strip()) and not _strip_decoration(text)


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


def _indexed_parts(text: str) -> list[tuple[int, int, str]]:
    """`(start, end, body)` for each decoration-free run, in text order.

    The positions are what makes a sub-cue decision possible: every character
    of a cue carries its own measured time, so a character range is also a time
    range. `_parts` is this without the bookkeeping.
    """
    parts: list[tuple[int, int, str]] = []
    current: list[str] = []
    begin = 0
    for index, ch in enumerate(str(text or "")):
        if ch in _DECORATION or ch.isspace():
            if current:
                parts.append((begin, index, "".join(current)))
                current = []
        else:
            if not current:
                begin = index
            current.append(ch)
    if current:
        parts.append((begin, len(str(text or "")), "".join(current)))
    return parts


def _parts(text: str) -> list[str]:
    return [body for _, _, body in _indexed_parts(text)]


def is_non_semantic_vocalisation(text: str) -> bool:
    """True when the cue is only vocalisation and may be dropped.

    Empty text is not vocalisation - it is nothing, and what an empty cue means
    is the caller's decision, not this module's.
    """
    body = _strip_decoration(text)
    if not body:
        return False
    if _is_protected(body) or _carries_lexical_content(body):
        return False
    # `あ、うん` still contains the reply, so the allow-list is consulted on the
    # parts as well as on the whole.
    if any(_is_protected(part) for part in _parts(text)):
        return False
    return _decomposes_into_vocalisation(body)


@dataclass(frozen=True)
class CueAcoustics:
    """The frame head's three-class reading over one cue's own span.

    `speech_max_run_s` is not derivable from the shares and is the reason this is
    a record rather than three numbers: one second of speech followed by five of
    moaning averages to the same shares as six seconds of moaning, and only the
    longest contiguous run of speech tells them apart.
    """

    silence: float
    vocalisation: float
    speech: float
    speech_max_run_s: float


@dataclass(frozen=True)
class CueVerdict:
    drop: bool
    reason: str


# Reason codes. They reach the QC report and the cue records, so they are part of
# the contract rather than log strings.
REASON_PROTECTED = "protected"
REASON_NO_ACOUSTICS = "no_acoustics"
REASON_TEXT_RUN = "vocal_text_in_run"
REASON_VOCAL_AUDIO = "vocal_text_vocal_audio"
REASON_KANA_VOCAL_AUDIO = "kana_text_vocal_audio"
REASON_VOCAL_TEXT_SPEECH_AUDIO = "vocal_text_speech_audio"
REASON_LEXICAL_VOCAL_AUDIO = "lexical_text_vocal_audio"
REASON_KEPT = "kept"

# Verdicts that survive but want a human. Both are cases where text and
# acoustics disagree in the direction that keeps the cue, which is the safe
# direction to be wrong in and the interesting one to look at.
_MARKED_REASONS = frozenset(
    {REASON_VOCAL_TEXT_SPEECH_AUDIO, REASON_LEXICAL_VOCAL_AUDIO}
)


def classify_cue(
    text: str,
    acoustics: CueAcoustics | None,
    *,
    vocal_speech_max: float = 0.10,
    vocal_speech_run_max_s: float = 0.30,
    kana_speech_max: float = 0.05,
    kana_vocalisation_min: float = 0.60,
    vocal_text_speech_min: float = 0.30,
) -> CueVerdict:
    """The acoustic half of the verdict, for one cue on its own.

    **Additive, not gating.** This only ever ADDS a reason to drop; a cue the
    text rule already removes is not re-examined here. Measured on eight films,
    requiring acoustic confirmation before honouring the text rule put 457 cues
    back on screen, and sampling them showed `あっ、あっ、あっ`,
    `あんっ!あんっ!` - plain moaning that the acoustics simply failed to
    confirm. Text evidence for a run of pure-vocalisation cues is already strong;
    the acoustics are here to reach what the text cannot see, which is the
    isolated cue and the onomatopoeia no allow-list spells.

    `protected` wins over everything, including the acoustics. Nothing in the
    sound separates `ふふふ` from a moan or `いい` from a sigh - that is why the
    allow-list exists - and an early version of this evaluation omitted the check
    and duly deleted both.

    Returning `keep` with a reason rather than a bare False: the marked cases are
    the ones a human should look at, and a verdict nobody can read is a detector
    running for nothing.
    """
    body = _strip_decoration(text)
    if not body:
        return CueVerdict(False, REASON_KEPT)
    if _is_protected(body) or any(_is_protected(part) for part in _parts(text)):
        return CueVerdict(False, REASON_PROTECTED)
    if acoustics is None:
        # A v1 head, or a cue with no acoustic extent. The caller falls back to
        # the text rule alone, which is the shipped behaviour.
        return CueVerdict(False, REASON_NO_ACOUSTICS)

    vocal_text = is_non_semantic_vocalisation(text)
    lexical = _carries_lexical_content(body)
    if vocal_text:
        if (
            acoustics.speech < vocal_speech_max
            and acoustics.speech_max_run_s < vocal_speech_run_max_s
        ):
            return CueVerdict(True, REASON_VOCAL_AUDIO)
        if acoustics.speech >= vocal_text_speech_min:
            # Breathy speech the ASR wrote in core kana, or a word the lexicon
            # does not protect. Kept and marked; never dropped on text alone.
            return CueVerdict(False, REASON_VOCAL_TEXT_SPEECH_AUDIO)
        return CueVerdict(False, REASON_KEPT)
    if not lexical:
        # Kana that would not decompose - the onomatopoeia the allow-list misses.
        if (
            acoustics.speech < kana_speech_max
            and acoustics.vocalisation > kana_vocalisation_min
        ):
            return CueVerdict(True, REASON_KANA_VOCAL_AUDIO)
        return CueVerdict(False, REASON_KEPT)
    # Kanji, latin or digits. Never dropped here: the ASR hallucinating a word
    # over moaning is real - one was found by ear in the 60-cue audit - but the
    # frame head cannot separate it from ordinary dialogue at a usable rate.
    # Marking every cue where vocalisation outweighs speech would flag 26.8% of
    # all kanji-bearing cues on eight films, and the case that was found sits
    # inside that bulk at p_sp=0.358. So the flag stays narrow and is a pointer
    # for review, not a filter.
    if (
        acoustics.speech < kana_speech_max
        and acoustics.vocalisation > kana_vocalisation_min
    ):
        return CueVerdict(False, REASON_LEXICAL_VOCAL_AUDIO)
    return CueVerdict(False, REASON_KEPT)


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


def _safe_seconds(primary, fallback) -> float:
    for value in (primary, fallback):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed == parsed:  # not NaN
            return parsed
    return 0.0


def mixed_cue_split_point(
    acoustics: CueAcoustics | None,
    start_s: float,
    end_s: float,
    *,
    minimum_vocalisation_s: float = 1.0,
    minimum_speech_s: float = 0.5,
) -> bool:
    """Whether a cue holds both a real stretch of speech and a long moan.

    Symptom (g): `お兄さまの...!あっ、あんっ!` is one second of speech and five of
    moaning, and the shares alone cannot see that - a span average reads the
    same as six seconds of moaning. `speech_max_run_s` is what distinguishes
    them, which is why it is carried per cue rather than derived.

    This is only the gate. It says a cue is worth examining; `split_mixed_cue`
    decides whether anything may actually be removed, and answers that with the
    same joint verdict applied to the candidate fragment.
    """
    if acoustics is None or end_s <= start_s:
        return False
    span = end_s - start_s
    vocalisation_s = acoustics.vocalisation * span
    return (
        vocalisation_s >= minimum_vocalisation_s
        and acoustics.speech_max_run_s >= minimum_speech_s
    )


def _word_texts(block: dict) -> list[str] | None:
    """The cue's per-character word list, if it reconstructs the cue text.

    Every character the aligner emitted carries its own measured start and end,
    including the punctuation an acoustic-only head gives zero width. So the
    concatenation is the cue text exactly, and when it is not - a cue whose text
    was rewritten after alignment, or one assembled without words - there is no
    character-to-time map and nothing here may be cut.
    """
    words = block.get("words")
    if not isinstance(words, list) or not words:
        return None
    texts = [str(word.get("word") or "") for word in words]
    if "".join(texts) != block_text(block):
        return None
    return texts


def _span_of_characters(
    block: dict, texts: list[str], begin: int, end: int
) -> tuple[float, float] | None:
    """Measured seconds covering text[begin:end], ignoring zero-width marks.

    Punctuation gets a zero-width span from an acoustic-only head, so including
    it would put the edge of the range at a moment nothing was heard.
    """
    words = block.get("words") or []
    position = 0
    starts: list[float] = []
    ends: list[float] = []
    for word, text in zip(words, texts):
        length = len(text)
        head, tail = position, position + length
        position = tail
        if tail <= begin or head >= end:
            continue
        start = _safe_seconds(word.get("start"), None)
        finish = _safe_seconds(word.get("end"), None)
        if finish <= start:
            continue
        starts.append(start)
        ends.append(finish)
    if not starts:
        return None
    return min(starts), max(ends)


@dataclass(frozen=True)
class CueSplit:
    """What `split_mixed_cue` decided to take off one cue."""

    text: str
    removed_prefix: str
    removed_suffix: str
    removed_seconds: float
    start_s: float
    end_s: float
    # Where the removed audio actually was. Carried rather than derived: a
    # listening audit of what a split deleted needs the seconds, and the cue it
    # is attached to no longer covers them by the time anyone can ask.
    prefix_span: tuple[float, float] | None = None
    suffix_span: tuple[float, float] | None = None


def split_mixed_cue(
    block: dict,
    acoustic_reader,
    *,
    vocal_speech_max: float = 0.10,
    vocal_speech_run_max_s: float = 0.30,
    kana_speech_max: float = 0.05,
    kana_vocalisation_min: float = 0.60,
    vocal_text_speech_min: float = 0.30,
    minimum_removed_s: float = 1.0,
) -> CueSplit | None:
    """Take a purely-vocal head or tail off a cue that also contains speech.

    **The removal criterion is not new.** A fragment may be taken off only when
    the same joint verdict that governs whole cues - the text decomposition plus
    `classify_cue` on that fragment's own re-measured frame classes - returns
    `drop` for it standing alone. So this cannot delete anything the filter would
    have kept as a cue; it only stops requiring that the noise arrive in a cue of
    its own. `_PROTECTED` still wins, because `classify_cue` consults it first.

    Three further guards, in the module's usual direction:

      * only a maximal run of parts at one end, so nothing is cut out of the
        middle of a sentence;
      * the remainder must still say something, so a cue is never emptied here -
        emptying it is the whole-cue rule's decision and it has its own evidence;
      * the fragment must be at least `minimum_removed_s` long, because the
        symptom being treated is a cue whose moaning dominates its screen time,
        not a trailing `あっ`.

    Returns None when nothing may be removed, which is the common case.
    """
    if acoustic_reader is None:
        return None
    text = block_text(block)
    texts = _word_texts(block)
    if not text or texts is None:
        return None
    parts = _indexed_parts(text)
    if len(parts) < 2:
        return None

    # The maximal vocal run at each end. The walk stops at the first part that
    # says something, so a word can never be inside a candidate fragment.
    head_count = 0
    while head_count < len(parts) and is_non_semantic_vocalisation(parts[head_count][2]):
        head_count += 1
    tail_count = 0
    while (
        tail_count < len(parts) - head_count
        and is_non_semantic_vocalisation(parts[-1 - tail_count][2])
    ):
        tail_count += 1
    if head_count == 0 and tail_count == 0:
        return None
    if head_count + tail_count >= len(parts):
        # Every part is vocal, so this is not a mixed cue at all and the
        # whole-cue rule owns it.
        return None

    def _removable(begin: int, end: int) -> tuple[float, float] | None:
        fragment = text[begin:end]
        if not _strip_decoration(fragment):
            return None
        span = _span_of_characters(block, texts, begin, end)
        if span is None or span[1] - span[0] < minimum_removed_s:
            return None
        shares = acoustic_reader(span[0], span[1])
        if not isinstance(shares, dict):
            return None
        acoustics = block_acoustics({"acoustic_classes": shares})
        verdict = classify_cue(
            fragment,
            acoustics,
            vocal_speech_max=vocal_speech_max,
            vocal_speech_run_max_s=vocal_speech_run_max_s,
            kana_speech_max=kana_speech_max,
            kana_vocalisation_min=kana_vocalisation_min,
            vocal_text_speech_min=vocal_text_speech_min,
        )
        return span if verdict.drop else None

    keep_begin, keep_end = 0, len(text)
    removed_prefix = removed_suffix = ""
    prefix_span = suffix_span = None
    removed_seconds = 0.0
    if head_count:
        # Up to the start of the first surviving part, so the decoration between
        # them goes with the fragment rather than opening the kept text.
        boundary = parts[head_count][0]
        span = _removable(0, boundary)
        if span is not None:
            keep_begin = boundary
            removed_prefix = text[:boundary]
            prefix_span = span
            removed_seconds += span[1] - span[0]
    if tail_count:
        boundary = parts[len(parts) - tail_count][0]
        if boundary > keep_begin:
            span = _removable(boundary, len(text))
            if span is not None:
                keep_end = boundary
                removed_suffix = text[boundary:]
                suffix_span = span
                removed_seconds += span[1] - span[0]
    if keep_begin == 0 and keep_end == len(text):
        return None

    kept = text[keep_begin:keep_end]
    if not _strip_decoration(kept):
        return None
    span = _span_of_characters(block, texts, keep_begin, keep_end)
    if span is None:
        return None
    return CueSplit(
        text=kept,
        removed_prefix=removed_prefix,
        removed_suffix=removed_suffix,
        removed_seconds=removed_seconds,
        start_s=span[0],
        end_s=span[1],
        prefix_span=prefix_span,
        suffix_span=suffix_span,
    )


def _apply_split(block: dict, split: CueSplit) -> dict:
    """Rewrite one cue around what survived, times included.

    The display window keeps the lead-in and linger the layout gave it and moves
    with the measured span, so the cue can only ever shrink - a split cannot
    create an overlap with a neighbour it did not already have.
    """
    updated = dict(block)
    texts = _word_texts(block) or []
    words = list(block.get("words") or [])
    prefix, suffix = len(split.removed_prefix), len(split.removed_suffix)
    if texts:
        total = len(block_text(block))
        position = 0
        kept_words = []
        for word, text in zip(words, texts):
            head, tail = position, position + len(text)
            position = tail
            if tail <= prefix or head >= total - suffix:
                continue
            kept_words.append(word)
        updated["words"] = kept_words
    for key in ("ja_text", "text"):
        if str(block.get(key) or "").strip():
            updated[key] = split.text
            break

    old_start = _safe_seconds(block.get("acoustic_start"), block.get("start"))
    old_end = _safe_seconds(block.get("acoustic_end"), block.get("end"))
    lead_in = max(0.0, old_start - _safe_seconds(block.get("display_start"), old_start))
    linger = max(0.0, _safe_seconds(block.get("display_end"), old_end) - old_end)
    updated["start"] = split.start_s
    updated["end"] = split.end_s
    updated["acoustic_start"] = split.start_s
    updated["acoustic_end"] = split.end_s
    updated["acoustic_duration"] = max(0.0, split.end_s - split.start_s)
    if block.get("display_start") is not None:
        updated["display_start"] = split.start_s - lead_in
    if block.get("display_end") is not None:
        updated["display_end"] = split.end_s + linger
        updated["display_duration"] = max(
            0.0, (split.end_s + linger) - (split.start_s - lead_in)
        )
    # The removed audio is exactly the bridge these flags were claiming across,
    # so the same rule as a dropped run applies: unknown is not "continues".
    if split.removed_prefix:
        updated["continues_from_previous"] = False
    if split.removed_suffix:
        updated["continues_into_next"] = False
    updated["vocalisation_split"] = {
        "removed_prefix": split.removed_prefix,
        "removed_suffix": split.removed_suffix,
        "removed_seconds": round(split.removed_seconds, 3),
        "removed_spans": [
            [round(span[0], 3), round(span[1], 3)]
            for span in (split.prefix_span, split.suffix_span)
            if span is not None
        ],
    }
    return updated


def block_acoustics(block: dict) -> CueAcoustics | None:
    """The frame-class reading a cue carries, if the head that made it had one.

    Absent on every cue produced by a v1 head, which is why the whole joint
    verdict has to degrade rather than fail: a promoted head outlives the code
    that trained it, and rolling back to the previous one is the response to a
    bad promotion.
    """
    shares = block.get("acoustic_classes")
    if not isinstance(shares, dict):
        return None
    try:
        return CueAcoustics(
            silence=float(shares["silence"]),
            vocalisation=float(shares["vocalisation"]),
            speech=float(shares["speech"]),
            speech_max_run_s=float(shares.get("speech_max_run_s") or 0.0),
        )
    except (KeyError, TypeError, ValueError):
        return None


def drop_vocalisation_runs(
    blocks: list[dict],
    *,
    min_run: int = 2,
    use_acoustics: bool = True,
    vocal_speech_max: float = 0.10,
    vocal_speech_run_max_s: float = 0.30,
    kana_speech_max: float = 0.05,
    kana_vocalisation_min: float = 0.60,
    vocal_text_speech_min: float = 0.30,
    split_mixed_cues: bool = True,
    acoustic_reader=None,
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

    # The acoustic half, added to the text rule rather than gating it. Every cue
    # the run rule already condemned keeps its verdict; this only reaches the
    # ones text cannot see - the isolated moan and the onomatopoeia no
    # allow-list spells.
    reasons: dict[int, str] = {index: REASON_TEXT_RUN for index in doomed}
    marks: dict[int, str] = {}
    splits: dict[int, CueSplit] = {}
    acoustic_verdicts: Counter[str] = Counter()
    acoustic_dropped = 0
    acoustics_seen = 0
    mixed_cues = 0
    for index, block in enumerate(blocks):
        if not use_acoustics or index in doomed:
            continue
        acoustics = block_acoustics(block)
        if acoustics is not None:
            acoustics_seen += 1
        verdict = classify_cue(
            block_text(block),
            acoustics,
            vocal_speech_max=vocal_speech_max,
            vocal_speech_run_max_s=vocal_speech_run_max_s,
            kana_speech_max=kana_speech_max,
            kana_vocalisation_min=kana_vocalisation_min,
            vocal_text_speech_min=vocal_text_speech_min,
        )
        acoustic_verdicts[verdict.reason] += 1
        if verdict.drop:
            doomed.add(index)
            reasons[index] = verdict.reason
            acoustic_dropped += 1
            continue
        if verdict.reason in _MARKED_REASONS:
            marks[index] = verdict.reason
        if not mixed_cue_split_point(
            acoustics,
            _safe_seconds(block.get("acoustic_start"), block.get("start")),
            _safe_seconds(block.get("acoustic_end"), block.get("end")),
        ):
            continue
        mixed_cues += 1
        if not split_mixed_cues:
            continue
        split = split_mixed_cue(
            block,
            acoustic_reader,
            vocal_speech_max=vocal_speech_max,
            vocal_speech_run_max_s=vocal_speech_run_max_s,
            kana_speech_max=kana_speech_max,
            kana_vocalisation_min=kana_vocalisation_min,
            vocal_text_speech_min=vocal_text_speech_min,
        )
        if split is not None:
            splits[index] = split

    # A cue that neighboured a dropped run must stop claiming it continues into
    # or out of one. The flags reach the translation prompt as `cont_prev` /
    # `cont_next`, and after the run is gone they assert that two cues join
    # across a passage that is no longer in the file - measured on eight films,
    # 442 of 513 dropped runs left such a claim behind, over gaps of 20s and
    # more. Continuity across the removed audio is unknown, and unknown must not
    # be reported as continuing.
    kept: list[dict] = []
    continuity_cleared = 0
    for index, block in enumerate(blocks):
        if index in doomed:
            continue
        after_drop = (index - 1) in doomed
        before_drop = (index + 1) in doomed
        if (after_drop and block.get("continues_from_previous")) or (
            before_drop and block.get("continues_into_next")
        ):
            block = dict(block)
            continuity_cleared += 1
            if after_drop:
                block["continues_from_previous"] = False
            if before_drop:
                block["continues_into_next"] = False
        # The marked verdicts travel with the cue, not only in the counts. These
        # are the cues a human should look at - breathy speech the lexicon does
        # not protect, and the ASR hallucinating a word over moaning - and a
        # verdict nobody downstream can read is a detector running for nothing.
        marked = marks.get(index)
        if marked:
            block = dict(block)
            block["vocalisation_verdict"] = marked
        split = splits.get(index)
        if split is not None:
            block = _apply_split(block, split)
        kept.append(block)

    diagnostics = {
        "vocalisation_cues_flagged": sum(flags),
        "vocalisation_cues_dropped": len(doomed),
        "vocalisation_runs_dropped": dropped_runs,
        "vocalisation_continuity_flags_cleared": continuity_cleared,
        # Flagged but left alone because they stood by themselves. Worth seeing:
        # if this grows large the run threshold is doing the real work, not the
        # word list.
        # Flagged by TEXT and left alone. Under the joint verdict this no longer
        # equals "survived": the acoustics reach most of these, so the two counts
        # are reported separately rather than one being inferred from the other.
        "vocalisation_cues_kept_as_isolated": sum(
            1
            for index, flagged in enumerate(flags)
            if flagged and index not in doomed
        ),
        "vocalisation_min_run": max(1, int(min_run)),
        "vocalisation_acoustics_available": acoustics_seen,
        "vocalisation_cues_dropped_by_acoustics": acoustic_dropped,
        "vocalisation_acoustic_verdicts": dict(acoustic_verdicts),
        "vocalisation_cues_marked": len(marks),
        # Cues holding both a real stretch of speech and a long moan, and how
        # many of those the fragment-level verdict actually agreed to trim. The
        # two are reported apart on purpose: the gap between them is the share of
        # symptom (g) that the evidence does not support acting on, and it is the
        # number to watch if the split ever looks too eager.
        "vocalisation_mixed_cues_detected": mixed_cues,
        "vocalisation_mixed_cues_split": len(splits),
        "vocalisation_split_removed_seconds": round(
            sum(split.removed_seconds for split in splits.values()), 3
        ),
        "vocalisation_split_removed_chars": sum(
            len(split.removed_prefix) + len(split.removed_suffix)
            for split in splits.values()
        ),
    }
    return kept, diagnostics
