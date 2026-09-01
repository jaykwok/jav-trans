"""Line-oriented contract for Hy-MT2, the local translation default.

Hy-MT2 is a single-sentence translation model, and this profile exists because
that is not a style preference. The contract was measured on the former 1.8B
default using 300 real ASR cues, scored by kana in the output (a Chinese subtitle
line cannot contain kana, so kana there is untranslated source handed back):

    bare per-line template          6 / 300
    + project system prompt v2.9   26 / 300
    + glossary / character block   30 / 300
    + neighbouring-line context    60 / 300
    JSON batch contract           152 / 300, plus 88 lines echoed verbatim

Every addition to the prompt costs another notch, and the batch contract costs
an order of magnitude: 63% of the lines it *did* return were wrong while
`missing=0` and the JSON parsed, so neither the engine nor the post-gate had any
reason to complain. Anything that deviates from the model card's template is out
of distribution for this model.

So this profile deliberately sends the bare template and nothing else. What that
gives up is real and is not silently swallowed - `warn_about_inert_context`
reports the glossary and character reference as inapplicable rather than
accepting settings it will not use.

Pairing it with the local backend rather than the JSON contract also costs less
than it looks: the JSON contract's full-transcript prefix cannot fit an 8GB
card's context budget anyway, so on local hardware that layer was never
available to begin with.
"""

from __future__ import annotations

import logging
import re

from llm.errors import RetryableTranslationFormatError
from llm.profiles.base import ProfileContext, TranslationProfile
from llm.profiles.json_v3 import _normalize_translation_text

logger = logging.getLogger(__name__)

# The model card's own Default Translation template. Kept verbatim on purpose:
# every measured deviation made it worse.
LINE_PROMPT = (
    "将以下文本翻译为{target_lang}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
)

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
    """One cue per request, bare template, no schema."""

    id = "hymt2"
    version = "hymt2-line-v1"

    # No repair pass and no partial reissue: both are batch concepts. A request
    # here is one cue, so a bad reply is retried as a whole by the engine's
    # normal path rather than repaired in place.
    wants_repair_pass = False
    supports_partial_reissue = False
    # Free-form text. Constraining this model with a grammar is what produced
    # the 152/300 failure above.
    schema = None

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
        if len(segments) != 1:
            # Reachable only if `max_batch_size` stopped being honoured. Failing
            # here is the point: a bare reply carries no ids, so a two-cue
            # request would be silently mis-assigned rather than detected.
            raise ValueError(
                f"{self.id} translates one cue per request, got {len(segments)}"
            )
        text = str(segments[0].get("text", ""))
        target_lang = (ctx.target_lang or "简体中文").strip() or "简体中文"
        return [
            {
                "role": "user",
                "content": LINE_PROMPT.format(target_lang=target_lang, text=text),
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
        if (ctx.glossary or "").strip() or (ctx.extra_glossary or "").strip():
            inert.append("术语表")
        if (ctx.character_reference or "").strip():
            inert.append("角色参考")
        if (ctx.global_context or "").strip() or ctx.full_source_payload:
            inert.append("全片上下文")
        return inert
