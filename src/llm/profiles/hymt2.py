"""Line-oriented Hy-MT2 with the model card's native context/term templates.

The old generic batch contract was measured on the former 1.8B
default using 300 real ASR cues, scored by kana in the output (a Chinese subtitle
line cannot contain kana, so kana there is untranslated source handed back):

    bare per-line template          6 / 300
    + project system prompt v2.9   26 / 300
    + glossary / character block   30 / 300
    + neighbouring-line context    60 / 300
    JSON batch contract           152 / 300, plus 88 lines echoed verbatim

Those results do not measure the current 7B's native background and terminology
templates. We still ask for one free-text target, never JSON or a whole-film
prompt. Only nearby source evidence and terms actually present in the target
are carried. Official reference:
https://huggingface.co/tencent/Hy-MT2-7B-GGUF/blob/main/README.md
"""

from __future__ import annotations

import logging
import re

from llm.errors import RetryableTranslationFormatError, UntranslatedOutputError
from llm.profiles.base import ProfileContext, TranslationProfile
from llm.profiles.json_v3 import _normalize_translation_text
from llm.glossary import parse_glossary_pairs
from llm.output_checks import (
    KANA_REASONS, NO_KANA_GRAMMAR, invalid_line_output, is_chinese_target,
)
from llm.context import source_text
from llm import zh_variant

logger = logging.getLogger(__name__)

# Model-card templates. Context is source-only and bounded per request.
LINE_PROMPT = (
    "将以下文本翻译为{target_lang}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
)
CONTEXT_PROMPT = (
    "【背景信息】\n{background}\n\n"
    "请结合背景信息将以下文本翻译为{target_lang}。\n\n【待翻译文本】\n{text}"
)

# The card asks a Chinese prompt for the language's Chinese name from its own
# table, and the UI labels are not those names. Measured 2026-09-28 on the four
# prompts issue #2 failed on: "简体中文" echoed 73/120 samples, "中文" failed
# 22/120; first requests over that whole film, 9 vs 3 rejections of 798. The
# script itself is still enforced afterwards by `zh_variant`, from the UI label.
_CARD_NAME_BY_VARIANT = {"simplified": "中文", "traditional": "繁体中文"}
_CARD_NAME_BY_LABEL = {"english": "英语", "英文": "英语"}


def card_target_name(target_lang: str) -> str:
    label = (target_lang or "简体中文").strip() or "简体中文"
    variant = zh_variant.target_variant(label)
    if variant is not None:
        return _CARD_NAME_BY_VARIANT[variant]
    return _CARD_NAME_BY_LABEL.get(label.lower(), label)

_THINK_BLOCK_RE = re.compile(r"(?s)^.*</think>")
# Cheap per-character bound on the reply, same reasoning as the JSON profile's
# budget: Chinese is denser than kana, so a translation is essentially never
# longer than its source. Wider here (3.0 rather than 1.5) because a single
# short cue has no batch to average against - `ん` legitimately becomes
# `嗯嗯嗯…`.
_LINE_CHAR_RATIO = 3.0
_MIN_TOKEN_BUDGET = 64
_MAX_TOKEN_BUDGET = 512


class HyMt2Profile(TranslationProfile):
    """One preplanned source cue per request with native context and terms."""

    id = "hymt2"
    version = "hymt2-context-v6"

    # No repair pass and no partial reissue: both are batch concepts. A request
    # here is one cue, so a bad reply is retried as a whole by the engine's
    # normal path rather than repaired in place.
    wants_repair_pass = False
    supports_partial_reissue = False
    # Free-form text. Constraining this model with a grammar is what produced
    # the 152/300 failure above.
    schema = None

    def sampling_parameters(self) -> dict[str, float | int]:
        # The model card calls this repetition_penalty; llama.cpp's request
        # field is repeat_penalty. Do not silently inherit an API recipe.
        return {"temperature": 0.7, "top_p": 0.6, "top_k": 20, "repeat_penalty": 1.05}

    def validate_translation(self, source: str, target: str, target_lang: str) -> None:
        reason = invalid_line_output(source, target, target_lang)
        if reason:
            raise UntranslatedOutputError(
                f"Hy-MT2 reply is not a {target_lang} translation: {reason}", reason=reason
            )

    def retry_grammar(self, rejected: BaseException, target_lang: str) -> str | None:
        # Some cues this model hands back as Japanese on most samples: issue #2's
        # film died on `こんにちは。`, echoed on 26/40 native samples of its two
        # prompts, and `んく` echoed 20/20, so four identical retries could not
        # save it. Excluding kana from the grammar answered all 80 samples of the
        # four failing prompts (2026-09-28). The first request stays native; only
        # a reply the check below already refused is replaced.
        if (
            isinstance(rejected, UntranslatedOutputError)
            and rejected.reason in KANA_REASONS
            and is_chinese_target(target_lang)
        ):
            return NO_KANA_GRAMMAR
        return None

    def max_batch_size(self) -> int | None:
        return 1

    def response_token_budget(
        self,
        segments: list[dict],
        *,
        reasoning_effort: str = "",
    ) -> int | None:
        # Hy-MT2 is a translation model with no reasoning mode, so the effort is
        # accepted for one uniform profile API and deliberately ignored here.
        del reasoning_effort
        if not segments:
            return None
        source_chars = sum(len(str(seg.get("text", ""))) for seg in segments)
        budget = int(source_chars * _LINE_CHAR_RATIO)
        return max(_MIN_TOKEN_BUDGET, min(_MAX_TOKEN_BUDGET, budget))

    def serialize_source(
        self,
        segments: list[dict],
        *,
        ids: list[int],
        compact: bool = False,
    ) -> str:
        del ids, compact
        return "\n".join(str(seg.get("text", "")) for seg in segments)

    def build_messages(
        self,
        segments: list[dict],
        *,
        ids: list[int],
        ctx: ProfileContext,
    ) -> list[dict]:
        if len(segments) != 1 or len(ids) != 1:
            raise ValueError(f"{self.id} requires exactly one source cue per request")
        text = source_text(segments[0])
        target_lang = card_target_name(ctx.target_lang)
        background: list[str] = []
        if ctx.source_context is not None:
            focus = ctx.source_context.focus(ids, radius=3, context_chars=240)
            previous = [row["ja"] for row in focus["items"] if row["id"] < ids[0]]
            following = [row["ja"] for row in focus["items"] if row["id"] > ids[-1]]
            if previous:
                background.append("前文：" + "\n".join(previous))
            if following:
                background.append("后文：" + "\n".join(following))
        terms = [(ja, zh) for ja, zh in parse_glossary_pairs(ctx.glossary) if ja in text]
        terminology = ""
        if terms:
            terminology = "参考下面的翻译：\n" + "\n".join(
                f"{ja} 翻译成 {zh}" for ja, zh in sorted(terms, key=lambda pair: -len(pair[0]))[:8]
            ) + "\n"
        if background:
            prompt = CONTEXT_PROMPT.format(
                background="\n".join(background), target_lang=target_lang, text=text
            )
        else:
            prompt = LINE_PROMPT.format(target_lang=target_lang, text=text)
        return [
            {
                "role": "user",
                "content": terminology + prompt,
            }
        ]

    def parse_response(
        self,
        text: str,
        *,
        ids: list[int],
    ) -> dict[int, str | None]:
        if not ids:
            return {}
        if len(ids) != 1:
            raise RetryableTranslationFormatError("Hy-MT2 accepts one source cue per response")
        cleaned = _THINK_BLOCK_RE.sub("", text or "").strip()
        normalized = _normalize_translation_text(cleaned)
        if normalized is None:
            # Empty is an error, never a translation. The removed line-oriented
            # path wrote "" here and returned successfully, which is the one way
            # this pipeline could put an untranslated cue on screen without
            # failing the job.
            raise RetryableTranslationFormatError(
                "Hy-MT2 returned an empty translation for one cue."
            )
        return {ids[0]: normalized}

    def warn_about_inert_context(self, ctx: ProfileContext) -> list[str]:
        """Name the job settings this contract cannot carry.

        Accepting a glossary and then ignoring it is the "配置项写了没人读"
        failure this repo already hunts elsewhere; the user filled that box in
        the UI and is entitled to know it does not reach the model.
        """
        inert: list[str] = []
        if (ctx.extra_glossary or "").strip():
            inert.append("自动重复句译法")
        if (ctx.character_reference or "").strip():
            inert.append("角色参考")
        if ctx.source_context is None and ((ctx.global_context or "").strip() or ctx.full_source_payload):
            inert.append("全片上下文")
        return inert
