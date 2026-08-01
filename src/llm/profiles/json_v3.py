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
from llm.profiles.base import ProfileContext, TranslationProfile

_JSON_OUTPUT_LABEL = prompt_module._JSON_OUTPUT_LABEL
_LEADING_ROLE_LABEL_RE = prompt_module._LEADING_ROLE_LABEL_RE
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)

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


def _parse_translation_output(
    raw_output: str,
    expected_count: int,
) -> list[str | None]:
    parsed = _parse_json_payload(raw_output)
    return _extract_translations_from_json(parsed, expected_count)


def _parse_translation_output_by_global_id(
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
    translations = parsed["translations"]
    if len(translations) != len(expected_ids):
        raise RetryableTranslationFormatError(
            f"{_JSON_OUTPUT_LABEL} returned wrong batch translation count: "
            f"{len(translations)} of {len(expected_ids)}."
        )

    results: list[str | None] = [None] * total_count
    seen_ids: set[int] = set()
    for item in translations:
        if not isinstance(item, dict):
            raise RetryableTranslationFormatError(
                f"{_JSON_OUTPUT_LABEL} translations must contain objects."
            )
        idx = _coerce_int(item.get("id"))
        if idx is None or idx not in expected_id_set or idx >= total_count:
            raise RetryableTranslationFormatError(
                f"{_JSON_OUTPUT_LABEL} returned invalid batch translation id: {item.get('id')!r}."
            )
        if idx in seen_ids:
            raise RetryableTranslationFormatError(
                f"{_JSON_OUTPUT_LABEL} returned duplicate translation id: {idx}."
            )
        seen_ids.add(idx)
        results[idx] = _normalize_translation_text(item.get("text"))

    return results


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


def _extract_translations_from_json(data, expected_count: int) -> list[str | None]:
    if not isinstance(data, dict) or not isinstance(data.get("translations"), list):
        raise RetryableTranslationFormatError(
            f'{_JSON_OUTPUT_LABEL} response must be {{"translations":[...]}} .'
        )

    translations = data["translations"]
    if len(translations) != expected_count:
        raise RetryableTranslationFormatError(
            f"{_JSON_OUTPUT_LABEL} returned wrong translation count: "
            f"{len(translations)} of {expected_count}."
        )

    results: list[str | None] = [None] * expected_count
    seen_ids: set[int] = set()
    for item in translations:
        if not isinstance(item, dict):
            raise RetryableTranslationFormatError(
                f"{_JSON_OUTPUT_LABEL} translations must contain objects."
            )

        idx = _coerce_int(item.get("id"))
        if idx is None or idx < 0 or idx >= expected_count:
            raise RetryableTranslationFormatError(
                f"{_JSON_OUTPUT_LABEL} returned invalid translation id: {item.get('id')!r}."
            )
        if idx in seen_ids:
            raise RetryableTranslationFormatError(
                f"{_JSON_OUTPUT_LABEL} returned duplicate translation id: {idx}."
            )

        seen_ids.add(idx)
        results[idx] = _normalize_translation_text(item.get("text"))

    return results


def _coerce_int(value) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _normalize_translation_text(text) -> str | None:
    if text is None:
        return None

    cleaned = str(text).strip()
    if not cleaned:
        return None

    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"^Translation>\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^Original>\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^['\"“”‘’]+|['\"“”‘’]+$", "", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n") if line.strip())
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    cleaned = _LEADING_ROLE_LABEL_RE.sub("", cleaned, count=1)
    if "\n" in cleaned:
        cleaned = cleaned.replace("\n", "\\n")
    return cleaned or None


def _missing_indexes(values: list[str | None]) -> list[int]:
    return [idx for idx, value in enumerate(values) if value is None or value == ""]


class JsonProfile(TranslationProfile):
    id = "json"
    version = prompt_module.PROMPT_VERSION

    needs_history = False
    line_capable = False
    wants_repair_pass = True
    wants_extra_glossary = True
    supports_partial_reissue = True
    schema = TRANSLATION_OUTPUT_SCHEMA

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
        return prompt_module._build_batch_messages(
            segments,
            ctx.global_context,
            ctx.character_reference,
            len(segments),
            batch_index=ctx.batch_index,
            extra_glossary=ctx.extra_glossary,
            target_lang=ctx.target_lang,
            glossary=ctx.glossary,
            source_payload_override=self.serialize_source(segments, ids=ids),
            full_source_payload=ctx.full_source_payload,
            requested_ids=list(ids),
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
