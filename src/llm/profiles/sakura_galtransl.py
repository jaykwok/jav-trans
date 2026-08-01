"""Sakura / GalTransl line-oriented profile.

GalTransl-style models are narrow translation models: they expect the exact
prompt template published on the model card and answer with translated lines
only -- no JSON, no ids. Feeding them the generic JSON batch contract produces
garbage, so this profile replaces it wholesale.

Wire contract (Sakura-GalTransl v3 series, from the official model card):

    system: 你是一个视觉小说翻译模型，可以通顺地使用给定的术语表以指定的风格将日文
            翻译成简体中文，并联系上下文正确使用人称代词，注意不要混淆使役态和被动
            态的主语和宾语，不要擅自添加原文中没有的特殊符号，也不要擅自增加或减少
            换行。
    user:   [历史翻译：<previous target lines>]
            参考以下术语表（可为空，格式为src->dst #备注）：
            <glossary lines>
            根据以上术语表的对应关系和备注，结合历史剧情和上下文，将下面的文本从日文
            翻译成简体中文：
            <N source lines>

The model preserves line count ("不要擅自增加或减少换行"), which is the whole
parse contract: N lines in, N lines out. Recommended sampling is
temperature=0.3 / top_p=0.8 (model card).
"""

from __future__ import annotations

import os

from llm.errors import RetryableTranslationFormatError
from llm.glossary import parse_glossary_pairs
from llm.profiles.base import ProfileContext, TranslationProfile

PROFILE_NAME = "sakura_galtransl_v3"

GALTRANSL_SYSTEM_PROMPT = (
    "你是一个视觉小说翻译模型，可以通顺地使用给定的术语表以指定的风格将日文翻译成简体中文，"
    "并联系上下文正确使用人称代词，注意不要混淆使役态和被动态的主语和宾语，"
    "不要擅自添加原文中没有的特殊符号，也不要擅自增加或减少换行。"
)

# Model-card recommended sampling for the GalTransl v3 series.
SAKURA_TEMPERATURE = 0.3
SAKURA_TOP_P = 0.8


def _env_int_clamped(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def sakura_history_limit() -> int:
    return _env_int_clamped("SAKURA_HISTORY_LINES", 8, 0, 24)


def glossary_to_sakura(glossary_text: str | None) -> str:
    """Project glossary lines (``src-dst``) -> GalTransl ``src->dst`` lines."""
    return "\n".join(
        f"{source}->{target}" for source, target in parse_glossary_pairs(glossary_text)
    )


def build_sakura_messages(
    source_lines: list[str],
    *,
    glossary_text: str = "",
    history_lines: list[str] | None = None,
) -> list[dict]:
    if not source_lines:
        raise ValueError("build_sakura_messages requires at least one source line")
    parts: list[str] = []
    history = [line.strip() for line in (history_lines or []) if line.strip()]
    if history:
        parts.append("历史翻译：" + "\n".join(history))
    glossary_block = glossary_to_sakura(glossary_text)
    parts.append("参考以下术语表（可为空，格式为src->dst #备注）：")
    if glossary_block:
        parts.append(glossary_block)
    parts.append(
        "根据以上术语表的对应关系和备注，结合历史剧情和上下文，将下面的文本从日文翻译成简体中文："
    )
    # Source lines are single-line by construction (the ASR text normalizer
    # folds newlines); flatten defensively so one bad line cannot desync the
    # line-count contract for the whole batch.
    parts.append("\n".join(" ".join(str(line).splitlines()) or "…" for line in source_lines))
    return [
        {"role": "system", "content": GALTRANSL_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return "\n".join(lines[1:]).strip()


def parse_sakura_response(text: str, expected_count: int) -> list[str]:
    """Split a Sakura reply into exactly ``expected_count`` lines.

    Anything else is a retryable format error -- the caller decides whether to
    retry the batch or fall back to line-by-line requests.
    """
    cleaned = _strip_code_fence(str(text or ""))
    lines = [line.strip() for line in cleaned.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    while lines and not lines[0]:
        lines.pop(0)
    if len(lines) != expected_count:
        raise RetryableTranslationFormatError(
            f"Sakura reply has {len(lines)} lines, expected {expected_count}"
        )
    return lines


class SakuraGalTranslProfile(TranslationProfile):
    id = "sakura_galtransl"
    version = "v3"

    needs_history = True
    line_capable = True
    wants_repair_pass = False
    wants_extra_glossary = False
    supports_partial_reissue = False
    schema = None

    @property
    def history_limit(self) -> int:  # type: ignore[override]
        return sakura_history_limit()

    def sampling(self, batch_size: int) -> dict:
        return {
            "temperature": SAKURA_TEMPERATURE,
            "top_p": SAKURA_TOP_P,
            # Line-oriented replies are short; a generous per-line budget with
            # a hard ceiling keeps runaway generations bounded.
            "max_tokens": min(4096, 160 * max(1, batch_size) + 256),
        }

    def build_messages(
        self,
        segments: list[dict],
        *,
        ids: list[int],
        ctx: ProfileContext,
    ) -> list[dict]:
        del ids
        source_lines = [str(seg.get("text", "")) for seg in segments]
        return build_sakura_messages(
            source_lines,
            glossary_text=ctx.glossary,
            history_lines=list(ctx.history),
        )

    def parse_response(
        self,
        text: str,
        *,
        ids: list[int],
    ) -> dict[int, str | None]:
        lines = parse_sakura_response(text, len(ids))
        return dict(zip(ids, lines))
