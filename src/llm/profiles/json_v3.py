"""Generic JSON batch-contract profile (the default for API / capable models).

Owns the ``{"translations":[{"id":..,"text":..}]}`` wire contract: message
construction delegates to ``llm.prompt`` (system prompt v2.9, full-JSON prefix,
``requested_ids`` incremental batches) and all reply parsing/normalization
lives here.
"""

from __future__ import annotations

import json
import re

from llm import prompt as prompt_module
from llm.errors import RetryableTranslationFormatError
from llm.output_checks import _KANA_CLASS
from llm.profiles.base import ProfileContext, TranslationProfile

_JSON_OUTPUT_LABEL = prompt_module._JSON_OUTPUT_LABEL
_LEADING_ROLE_LABEL_RE = prompt_module._LEADING_ROLE_LABEL_RE
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
# JSON-safe kana exclusion regex for llama.cpp GBNF: also excludes raw quotes,
# backslashes and ASCII control chars so the sampler cannot break out of a JSON
# string value and trigger llama-server peg-native 500 parse failures.
_NO_KANA_JSON_STRING_PATTERN = f'^[^{_KANA_CLASS}"\\\\\\u0000-\\u001F]+$'

_WRAP_QUOTE_PAIRS: tuple[tuple[str, str], ...] = (
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
)

TRANSLATION_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["id", "text"],
            },
        },
    },
    "required": ["translations"],
}


def _strip_reasoning_artifacts(raw_output: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", raw_output or "")
    close_tag = "</think>"
    close_idx = cleaned.lower().rfind(close_tag)
    if close_idx != -1:
        cleaned = cleaned[close_idx + len(close_tag) :]
    return cleaned.strip()


def _parse_json_payload(raw_output: str):
    raw_output = _strip_reasoning_artifacts(raw_output)
    if not raw_output.strip():
        raise RetryableTranslationFormatError(
            f"{_JSON_OUTPUT_LABEL} returned empty content."
        )
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise RetryableTranslationFormatError(
            f"{_JSON_OUTPUT_LABEL} response was not valid JSON."
        ) from exc


def _parse_partial_translation_output_by_global_id(
    raw_output: str,
    *,
    expected_ids: list[int],
    total_count: int,
) -> list[str | None]:
    parsed = _parse_json_payload(raw_output)

    if not isinstance(parsed, dict) or not isinstance(parsed.get("translations"), list):
        raise RetryableTranslationFormatError(
            f'{_JSON_OUTPUT_LABEL} response must be {{"translations":[...]}} .'
        )

    expected_id_set = set(expected_ids)
    results: list[str | None] = [None] * total_count
    seen_ids: set[int] = set()
    for item in parsed["translations"]:
        if not isinstance(item, dict):
            continue
        idx = _coerce_int(item.get("id"))
        if idx is None or idx not in expected_id_set or idx >= total_count:
            raise RetryableTranslationFormatError(
                f"{_JSON_OUTPUT_LABEL} returned invalid batch translation id: {item.get('id')!r}."
            )
        if idx in seen_ids:
            raise RetryableTranslationFormatError(
                f"{_JSON_OUTPUT_LABEL} returned duplicate translation id: {idx}."
            )
        normalized = _normalize_translation_text(item.get("text"))
        if normalized is None:
            continue
        seen_ids.add(idx)
        results[idx] = normalized

    return results


def _coerce_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _strip_paired_outer_quotes(text: str) -> str:
    cleaned = text
    while len(cleaned) >= 2:
        stripped = False
        for left, right in _WRAP_QUOTE_PAIRS:
            if cleaned.startswith(left) and cleaned.endswith(right):
                inner = cleaned[len(left) : len(cleaned) - len(right)].strip()
                if left != right:
                    if right in inner and (
                        left not in inner or inner.index(right) < inner.index(left)
                    ):
                        break
                elif left in inner:
                    break
                cleaned = inner
                stripped = True
                break
        if not stripped:
            break
    return cleaned


def _normalize_translation_text(text) -> str | None:
    if text is None:
        return None

    cleaned = str(text).strip()
    if not cleaned:
        return None

    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"^Translation>\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^Original>\s*", "", cleaned, flags=re.I)
    cleaned = _strip_paired_outer_quotes(cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n") if line.strip())
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    cleaned = _LEADING_ROLE_LABEL_RE.sub("", cleaned, count=1)
    if "\n" in cleaned:
        cleaned = cleaned.replace("\n", "\\n")
    return cleaned or None


# One `{"id":123,"text":"…"},` of scaffolding per translated line, plus the
# `{"translations":[…]}` wrapper. Counted in characters and spent as tokens:
# for this contract that is conservative in both directions, since the ASCII
# structure is well under one token per character and CJK text is around 0.7.
_STRUCTURE_TOKENS_PER_ITEM = 28
_WRAPPER_TOKENS = 32
_MIN_TOKEN_BUDGET = 96

# Floor for the per-field character bound, so a batch of very short cues still
# leaves room for a legitimately expansive rendering of one of them (`ん` ->
# `嗯嗯嗯…`). Measured on 1362 clean translations: no output ever exceeded its
# source by more than 7 characters in absolute terms, so 32 is far clear of it.
_MIN_TEXT_MAX_LENGTH = 32


def _reasoning_token_allowance(reasoning_effort: str | None = None) -> int:
    """Extra `max_tokens` room for the reasoning stream at this effort.

    Resolved from the passed effort rather than the environment: a job carries
    its own setting (the Web 「推理强度」 selector, `ctx.llm_reasoning_effort`),
    and reading the env here would budget for whatever the process was started
    with - which is exactly how the A/B arms would have been mis-sized.

    `none` is the tier that does not think, so it gets nothing - which is most
    of why it is the cheap one. `low` and `high` share one allowance because
    the measured demand does not separate them cleanly: on DeepSeek `low` spent
    7,860/14,034/9,383 reasoning characters on 8/24/54-cue batches against
    `high`'s 2,058/18,393/20,231, i.e. sometimes more. The base is sized over
    the worst of both.
    """
    from llm import request_config
    from llm import settings as llm_settings

    if reasoning_effort == "":
        raw_effort = "none"
    elif reasoning_effort is None:
        raw_effort = request_config.getenv(
            "LLM_REASONING_EFFORT", llm_settings.LLM_REASONING_EFFORT
        )
    else:
        raw_effort = reasoning_effort
    effort = llm_settings._normalize_reasoning_effort(raw_effort)
    if effort == "none":
        return 0
    return max(0, int(llm_settings.TRANSLATION_REASONING_TOKEN_ALLOWANCE))


class JsonProfile(TranslationProfile):
    id = "json"
    version = prompt_module.PROMPT_VERSION

    wants_repair_pass = True
    supports_partial_reissue = True
    schema = TRANSLATION_OUTPUT_SCHEMA

    def sampling_parameters(self) -> dict[str, float | int]:
        # Native Hy-MT3-7B model-card sampling for llama.cpp; API transports
        # ignore this hook and keep deployment sampling defaults.
        return {"temperature": 0.7, "top_p": 0.6, "top_k": 20, "repeat_penalty": 1.05}

    def response_token_budget(
        self,
        segments: list[dict],
        *,
        reasoning_effort: str | None = None,
    ) -> int | None:
        """`max_tokens` for this request: the visible reply, plus the thinking.

        The first two terms model what the answer is made of. The third exists
        because this number is sent as `max_tokens`, which on a reasoning
        request is spent on the reasoning stream first: at `high` an 8-cue batch
        got a 469-token budget while the model spent 2,058 characters thinking,
        so it was cut off every time and the film died after one doubling.
        Reasoning is driven far more by the effort than by how much text was
        handed over - `high` spends about the same on 24 cues as on 54 - so the
        allowance is flat per tier rather than a second ratio, and it is zero at
        `none`. See `settings.TRANSLATION_REASONING_TOKEN_ALLOWANCE`.
        """
        if not segments:
            return None
        from llm import settings as llm_settings

        source_chars = sum(len(str(seg.get("text", ""))) for seg in segments)
        body = source_chars * llm_settings.TRANSLATION_OUTPUT_CHAR_RATIO
        structure = _STRUCTURE_TOKENS_PER_ITEM * len(segments) + _WRAPPER_TOKENS
        thinking = _reasoning_token_allowance(reasoning_effort)
        return max(_MIN_TOKEN_BUDGET, int(body + structure + thinking))

    def bounded_schema(
        self,
        segments: list[dict],
        *,
        ids: list[int] | None = None,
        forbid_kana: bool = False,
    ) -> dict | None:
        """`maxLength` on `text`, sized by the longest source line in the batch.

        `minItems`/`maxItems` already pin how many objects come back, which is
        why a runaway still satisfies the grammar: the model emits the right
        twelve objects and writes `嗯嗯嗯…` into one of them until it hits
        `max_tokens`. Bounding the field closes that, and unlike a token budget
        it cannot produce broken JSON - the grammar simply stops offering the
        sampler a token that would overflow, so the string closes and the object
        completes.

        The bound is the *longest* line in the batch rather than each line's own
        length, because one `items` schema applies to every element. That costs
        nothing in practice: over 4000 simulated 12-line batches drawn from 1362
        clean translations, the worst observed
        `max(len(translation)) / max(len(source))` was 1.025, so the 1.5 ratio
        leaves a 1.46x margin on the tightest case measured.
        """
        if not segments:
            return None
        from llm import settings as llm_settings

        longest = max((len(str(seg.get("text", ""))) for seg in segments), default=0)
        limit = max(
            _MIN_TEXT_MAX_LENGTH,
            int(longest * llm_settings.TRANSLATION_OUTPUT_CHAR_RATIO),
        )
        id_prop: dict = {"type": "integer"}
        text_prop: dict = {"type": "string", "maxLength": limit}
        if forbid_kana:
            text_prop["pattern"] = _NO_KANA_JSON_STRING_PATTERN
        translations_prop: dict = {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": id_prop,
                    "text": text_prop,
                },
                "required": ["id", "text"],
            },
        }
        if ids:
            translations_prop["minItems"] = len(ids)
            translations_prop["maxItems"] = len(ids)
            id_prop["enum"] = [int(idx) for idx in ids]
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "translations": translations_prop,
            },
            "required": ["translations"],
        }

    def serialize_source(
        self,
        segments: list[dict],
        *,
        ids: list[int],
        compact: bool = False,
    ) -> str:
        return prompt_module._serialize_segments(
            segments,
            explicit_ids=list(ids),
            compact=compact,
        )

    def build_messages(
        self,
        segments: list[dict],
        *,
        ids: list[int],
        ctx: ProfileContext,
    ) -> list[dict]:
        needs_source_payload = ctx.full_source_payload is None and not (
            ctx.source_context is not None and ids and not ctx.warmup
        )
        source_payload_override = (
            self.serialize_source(segments, ids=ids) if needs_source_payload else ""
        )
        return prompt_module._build_batch_messages(
            segments,
            ctx.global_context,
            ctx.character_reference,
            len(segments),
            batch_index=ctx.batch_index,
            extra_glossary=ctx.extra_glossary,
            target_lang=ctx.target_lang,
            glossary=ctx.glossary,
            source_payload_override=source_payload_override,
            full_source_payload=ctx.full_source_payload,
            requested_ids=list(ids),
            source_context=ctx.source_context,
            focus_radius=ctx.focus_radius,
            focus_context_chars=ctx.focus_context_chars,
            warmup=ctx.warmup,
            compact_system_prompt_enabled=ctx.compact_system_prompt,
        )

    def parse_response(
        self,
        text: str,
        *,
        ids: list[int],
    ) -> dict[int, str | None]:
        total = (max(ids) + 1) if ids else 0
        values = _parse_partial_translation_output_by_global_id(
            text,
            expected_ids=list(ids),
            total_count=total,
        )
        return {idx: values[idx] for idx in ids}
