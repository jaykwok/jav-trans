from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


_STRIP_PUNCT_RE = re.compile(r"[。！？…、,.!?・「」『』（）()【】\[\]\s~〜ー-]+")
_DECORATION_RE = re.compile(r"[♡♥❤💕💖💗💘♪♫♬★☆※]+")
_LAUGH_RE = re.compile(r"[wWｗＷ]+")
_PUNCT_RE = re.compile(r"[。！？…、,.!?・「」『』（）()【】\[\]~〜～]+")
_KANA_CHAR_RE = re.compile(r"[ぁ-ゖァ-ヺ]")
_KANA_REPEAT_RE = re.compile(r"([ぁ-ゖァ-ヺ])\1{2,}")
_LONG_VOWEL_RE = re.compile(r"([ーｰ])\1+")
_SPACE_RE = re.compile(r"[ \t]+")
_REPEATED_PHRASE_RE = re.compile(r"([ぁ-ゖァ-ヺ一-龯]{2,8})(?:[、。！？…\s]*\1){2,}")
_FLAG_ORDER = {
    "removed_decoration": 0,
    "removed_laugh_marker": 1,
    "removed_punctuation": 2,
    "compacted_kana_repeat": 3,
    "compacted_long_vowel": 4,
    "compacted_repeated_phrase": 5,
    "empty_after_alignment_cleaning": 6,
}


@dataclass(frozen=True)
class CharSpan:
    align_start: int
    align_end: int
    display_start: int
    display_end: int


@dataclass(frozen=True)
class PrealignText:
    raw_text: str
    display_text: str
    align_text: str
    changed: bool
    removed_for_alignment: bool
    empty_after_cleaning: bool
    flags: list[str] = field(default_factory=list)
    align_to_display_spans: list[CharSpan] = field(default_factory=list)

    @property
    def raw_len(self) -> int:
        return len(self.raw_text)

    @property
    def display_len(self) -> int:
        return len(self.display_text)

    @property
    def align_len(self) -> int:
        return len(self.align_text)


def strip_alignment_punctuation(text: str) -> str:
    return _STRIP_PUNCT_RE.sub("", text or "")


def _ordered_flags(flags: list[str]) -> list[str]:
    unique = list(dict.fromkeys(flags))
    return sorted(unique, key=lambda item: (_FLAG_ORDER.get(item, 100), item))


def normalize_display_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = cleaned.replace("\r", " ").replace("\n", " ").strip()
    cleaned = _SPACE_RE.sub(" ", cleaned)

    return cleaned.strip()


def _append_char(
    chars: list[str],
    spans: list[CharSpan],
    char: str,
    *,
    display_start: int,
    display_end: int,
) -> None:
    align_start = len(chars)
    chars.append(char)
    spans.append(
        CharSpan(
            align_start=align_start,
            align_end=align_start + 1,
            display_start=display_start,
            display_end=max(display_start + 1, display_end),
        )
    )


def _append_segment(
    chars: list[str],
    spans: list[CharSpan],
    text: str,
    *,
    display_start: int,
    display_end: int,
) -> None:
    for offset, char in enumerate(text):
        _append_char(
            chars,
            spans,
            char,
            display_start=min(display_start + offset, max(display_start, display_end - 1)),
            display_end=display_end if offset == len(text) - 1 else display_start + offset + 1,
        )


def _normalize_for_alignment(display_text: str) -> tuple[str, list[CharSpan], list[str]]:
    flags: list[str] = []
    chars: list[str] = []
    spans: list[CharSpan] = []
    index = 0
    while index < len(display_text):
        char = display_text[index]

        if _DECORATION_RE.fullmatch(char):
            flags.append("removed_decoration")
            index += 1
            continue
        if _LAUGH_RE.fullmatch(char):
            start = index
            while index < len(display_text) and _LAUGH_RE.fullmatch(display_text[index]):
                index += 1
            if index - start >= 2:
                flags.append("removed_laugh_marker")
                continue
            _append_char(chars, spans, char, display_start=start, display_end=index)
            continue
        if _PUNCT_RE.fullmatch(char):
            flags.append("removed_punctuation")
            index += 1
            continue
        if char.isspace():
            if chars and chars[-1] != " ":
                _append_char(chars, spans, " ", display_start=index, display_end=index + 1)
            index += 1
            continue

        if _KANA_CHAR_RE.fullmatch(char):
            start = index
            while index < len(display_text) and display_text[index] == char:
                index += 1
            repeat_len = index - start
            keep_len = min(repeat_len, 2)
            if repeat_len > keep_len:
                flags.append("compacted_kana_repeat")
            _append_segment(
                chars,
                spans,
                char * keep_len,
                display_start=start,
                display_end=index,
            )
            continue

        if char in {"ー", "ｰ"}:
            start = index
            while index < len(display_text) and display_text[index] in {"ー", "ｰ"}:
                index += 1
            if index - start > 1:
                flags.append("compacted_long_vowel")
            _append_char(chars, spans, char, display_start=start, display_end=index)
            continue

        _append_char(chars, spans, char, display_start=index, display_end=index + 1)
        index += 1

    align_text = "".join(chars).strip()
    if align_text != "".join(chars):
        leading = len("".join(chars)) - len("".join(chars).lstrip())
        trailing_end = leading + len(align_text)
        spans = spans[leading:trailing_end]
        chars = list(align_text)
        spans = [
            CharSpan(
                align_start=i,
                align_end=i + 1,
                display_start=span.display_start,
                display_end=span.display_end,
            )
            for i, span in enumerate(spans)
        ]
    return align_text, spans, list(dict.fromkeys(flags))


def _compact_repeated_phrases_for_alignment(
    align_text: str,
    spans: list[CharSpan],
) -> tuple[str, list[CharSpan], bool]:
    changed = False
    for _ in range(2):
        match = _REPEATED_PHRASE_RE.search(align_text)
        if not match:
            break
        token = match.group(1)
        replacement = f"{token}{token}"
        start, end = match.span()
        if len(replacement) >= end - start:
            break
        display_start = spans[start].display_start if start < len(spans) else start
        display_end = spans[end - 1].display_end if end - 1 < len(spans) else end
        replacement_spans = [
            CharSpan(
                align_start=start + offset,
                align_end=start + offset + 1,
                display_start=min(display_start + offset, max(display_start, display_end - 1)),
                display_end=display_end if offset == len(replacement) - 1 else display_start + offset + 1,
            )
            for offset in range(len(replacement))
        ]
        align_text = align_text[:start] + replacement + align_text[end:]
        spans = spans[:start] + replacement_spans + spans[end:]
        spans = [
            CharSpan(
                align_start=index,
                align_end=index + 1,
                display_start=span.display_start,
                display_end=span.display_end,
            )
            for index, span in enumerate(spans)
        ]
        changed = True
    return align_text, spans, changed


def prepare_text_for_alignment(text: str) -> PrealignText:
    raw_text = text or ""
    display_text = normalize_display_text(raw_text)
    align_text, spans, flags = _normalize_for_alignment(display_text)
    align_text, spans, phrase_compacted = _compact_repeated_phrases_for_alignment(
        align_text,
        spans,
    )
    if phrase_compacted:
        flags.append("compacted_repeated_phrase")
    align_text = _SPACE_RE.sub(" ", align_text).strip()

    if not display_text:
        return PrealignText(
            raw_text=raw_text,
            display_text=display_text,
            align_text="",
            changed=raw_text != display_text,
            removed_for_alignment=False,
            empty_after_cleaning=True,
            flags=["empty_input"] if raw_text == "" else ["empty_display_text"],
        )

    empty_after_cleaning = not bool(
        strip_alignment_punctuation(_DECORATION_RE.sub("", _LAUGH_RE.sub("", align_text)))
    )
    if empty_after_cleaning:
        align_text = ""
        spans = []
        flags.append("empty_after_alignment_cleaning")

    return PrealignText(
        raw_text=raw_text,
        display_text=display_text,
        align_text=align_text,
        changed=raw_text != display_text or display_text != align_text,
        removed_for_alignment=display_text != align_text,
        empty_after_cleaning=empty_after_cleaning,
        flags=_ordered_flags(flags),
        align_to_display_spans=spans,
    )


def clean_text_for_aligner(text: str) -> str:
    return prepare_text_for_alignment(text).align_text
