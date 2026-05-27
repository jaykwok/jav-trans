from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


_STRIP_PUNCT_RE = re.compile(r"[。！？…、,.!?・「」『』（）()【】\[\]\s~〜ー-]+")
_DECORATION_RE = re.compile(r"[♡♥❤💕💖💗💘♪♫♬★☆※]+")
_LAUGH_RE = re.compile(r"[wWｗＷ]+")
_PUNCT_RE = re.compile(r"[。！？…、,.!?・「」『』（）()【】\[\]~〜～]+")
_KANA_REPEAT_RE = re.compile(r"([ぁ-ゖァ-ヺ])\1{2,}")
_LONG_VOWEL_RE = re.compile(r"([ーｰ])\1+")
_SPACE_RE = re.compile(r"[ \t]+")
_REPEATED_PHRASE_RE = re.compile(r"([ぁ-ゖァ-ヺ一-龯]{2,8})(?:[、。！？…\s]*\1){2,}")


@dataclass(frozen=True)
class PrealignText:
    raw_text: str
    display_text: str
    align_text: str
    changed: bool
    removed_for_alignment: bool
    empty_after_cleaning: bool
    flags: list[str] = field(default_factory=list)

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


def normalize_display_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = cleaned.replace("\r", " ").replace("\n", " ").strip()
    cleaned = _SPACE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"(.)\1{4,}", r"\1\1", cleaned)

    for _ in range(2):
        updated = _REPEATED_PHRASE_RE.sub(r"\1、\1", cleaned)
        if updated == cleaned:
            break
        cleaned = updated

    return cleaned.strip()


def prepare_text_for_alignment(text: str) -> PrealignText:
    raw_text = text or ""
    display_text = normalize_display_text(raw_text)
    flags: list[str] = []
    align_text = display_text

    if not align_text:
        return PrealignText(
            raw_text=raw_text,
            display_text=display_text,
            align_text="",
            changed=raw_text != display_text,
            removed_for_alignment=False,
            empty_after_cleaning=True,
            flags=["empty_input"] if raw_text == "" else ["empty_display_text"],
        )

    if _DECORATION_RE.search(align_text):
        flags.append("removed_decoration")
        align_text = _DECORATION_RE.sub("", align_text)
    if _LAUGH_RE.search(align_text):
        flags.append("removed_laugh_marker")
        align_text = _LAUGH_RE.sub("", align_text)
    if _PUNCT_RE.search(align_text):
        flags.append("removed_punctuation")
        align_text = _PUNCT_RE.sub("", align_text)

    repeat_compacted = _KANA_REPEAT_RE.sub(r"\1\1", align_text)
    if repeat_compacted != align_text:
        flags.append("compacted_kana_repeat")
        align_text = repeat_compacted

    vowel_compacted = _LONG_VOWEL_RE.sub(r"\1", align_text)
    if vowel_compacted != align_text:
        flags.append("compacted_long_vowel")
        align_text = vowel_compacted

    align_text = _SPACE_RE.sub(" ", align_text).strip()
    empty_after_cleaning = not bool(
        strip_alignment_punctuation(_DECORATION_RE.sub("", _LAUGH_RE.sub("", align_text)))
    )
    if empty_after_cleaning:
        align_text = ""
        flags.append("empty_after_alignment_cleaning")

    return PrealignText(
        raw_text=raw_text,
        display_text=display_text,
        align_text=align_text,
        changed=raw_text != display_text or display_text != align_text,
        removed_for_alignment=display_text != align_text,
        empty_after_cleaning=empty_after_cleaning,
        flags=list(dict.fromkeys(flags)),
    )


def clean_text_for_aligner(text: str) -> str:
    return prepare_text_for_alignment(text).align_text
