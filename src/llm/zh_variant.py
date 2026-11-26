"""Force translated subtitles into the Chinese variant the user picked.

The web page offers 简体中文 / 繁體中文 / English, and that choice is already in
every prompt - but a prompt is a request, not a guarantee. The local default
(Hy-MT2) answers in 繁體 for a fair share of lines regardless of what the prompt
asked for, and API models drift the same way on shorter cues. Neither failure is
visible to any existing check: the line is a correct, fluent translation, just in
the wrong script, so it passes the passthrough/empty/length guards and lands on
screen.

Conversion is a post-filter rather than a stronger prompt because it is the only
form that cannot fail silently: OpenCC is deterministic, idempotent, and leaves
anything that is not Han text alone. Running it on output that is already in the
requested variant is a no-op.

The mappings are direction-asymmetric in difficulty, which is why the reference
OpenCC dictionaries are used instead of a character table: 繁->简 is mostly
many-to-one, but 简->繁 needs phrase context (头发 -> 頭髮 yet 发现 -> 發現;
干净 -> 乾淨 yet 干什么 -> 幹什麼). A per-character map gets those backwards.

Known and accepted: t2s converts Han characters wherever it finds them, so a
Japanese word that survived untranslated can come out mixed, e.g. 発見 -> 発见
(発 is shinjitai and has no entry, 見 does). That only reaches the screen when a
line was left untranslated, which the profiles already treat as a hard failure.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# Order matters: 繁體/正體 must be tested before the generic "chinese" fallbacks,
# and zh-hant before zh, so a prefix match cannot claim the wrong side.
_TRADITIONAL = re.compile(r"繁體|繁体|正體|正体|zh[-_]?(hant|tw|hk|mo)\b|traditional", re.I)
_SIMPLIFIED = re.compile(r"简体|簡體|zh[-_]?(hans|cn|sg)\b|simplified", re.I)

# `s2tw`, not `s2t`: OpenCC's generic traditional target writes the mainland
# variant forms (裏面, 這裏), while readers who pick 繁體中文 are overwhelmingly
# TW/HK and expect 裡面, 這裡. Stopping short of `s2twp` is deliberate - that one
# also swaps vocabulary (軟件 -> 軟體), which changes the translator's word
# choice rather than the script, and this pass has no mandate to do that.
# `t2s` needs no such split: it folds 裡面 and 裏面 alike back to 里面.
_CONFIG_BY_VARIANT = {"simplified": "t2s", "traditional": "s2tw"}

_converters: dict[str, Callable[[str], str]] = {}
_unavailable = False


def target_variant(target_lang: str) -> str | None:
    """`"simplified"`, `"traditional"`, or None when the target is not Chinese.

    None is the answer for English and for anything unrecognised. Guessing a
    variant for an unknown label would rewrite output the user never asked to
    have rewritten, so an unreadable setting means "leave it alone".
    """
    label = str(target_lang or "").strip()
    if not label:
        return None
    if _TRADITIONAL.search(label):
        return "traditional"
    if _SIMPLIFIED.search(label):
        return "simplified"
    return None


def converter_for(target_lang: str) -> Callable[[str], str] | None:
    """A text -> text converter for `target_lang`, or None to leave text as is.

    Built once per variant and reused: OpenCC parses its dictionaries on
    construction, which is far too expensive to repeat per subtitle line.
    """
    global _unavailable

    variant = target_variant(target_lang)
    if variant is None or _unavailable:
        return None
    if variant in _converters:
        return _converters[variant]
    try:
        from opencc import OpenCC
    except ImportError:
        # A venv built before this dependency existed. Degrading to the previous
        # behaviour beats failing a translation run over a cosmetic pass.
        _unavailable = True
        print(
            "[translation] 未安装 opencc-python-reimplemented，跳过简繁转换；"
            "请重新执行 uv sync"
        )
        return None
    converter = OpenCC(_CONFIG_BY_VARIANT[variant])
    _converters[variant] = converter.convert
    return converter.convert


def convert(text: str, target_lang: str) -> str:
    """Convenience wrapper for one-off calls; prefer `converter_for` in loops."""
    convert_fn = converter_for(target_lang)
    return convert_fn(text) if convert_fn and text else text
