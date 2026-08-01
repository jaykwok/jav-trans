"""Translation profile abstraction.

A profile owns one model family's prompt contract: how to build messages for a
batch of subtitle lines and how to parse the reply back into per-id texts. The
orchestration engine is written once and parameterized by these objects; adding
support for a new fine-tuned model is one new profile module, never an engine
edit.

Hard invariant: translation is 1:1. ``parse_response`` must yield exactly one
text per requested id (or raise ``RetryableTranslationFormatError``); the cue
plan is frozen before translation and profiles can never merge or split lines.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProfileContext:
    """Job-level context threaded into message construction."""

    target_lang: str = "简体中文"
    glossary: str = ""
    extra_glossary: str = ""
    character_reference: str = ""
    global_context: str = ""
    # Full-transcript JSON payload for prefix-cache-friendly prompts (JSON
    # contract only; None disables the prefix).
    full_source_payload: str | None = None
    # Rolling translated history for line-oriented profiles (most recent last).
    history: tuple[str, ...] = ()
    total_count: int = 0
    compact_system_prompt: bool = False
    # Per-request scheduling info the engine fills in: which batch this is and
    # whether the request only warms the provider prefix cache.
    batch_index: int = 0
    warmup: bool = False


class TranslationProfile(abc.ABC):
    """One model family's prompt/parse contract."""

    id: str = ""
    version: str = ""

    # Scheduling: contiguous shards with sequential in-shard execution so the
    # rolling history is real, instead of free parallel batches.
    needs_history: bool = False
    history_limit: int = 0
    # Retry ladder: after batch-level format retries are exhausted, may the
    # engine fall back to one-line-per-request?
    line_capable: bool = False
    # Orchestration stages the profile opts into.
    wants_repair_pass: bool = False
    wants_extra_glossary: bool = False
    # Can the profile re-request only the missing ids of a partial reply?
    supports_partial_reissue: bool = False
    # JSON schema for structured decoding, passed to backends that report
    # supports_json_schema(); None means free-form text.
    schema: dict | None = None

    def cache_signature(self) -> str:
        """Folded into every cache/memory key; bump ``version`` on any change
        that alters output for identical input."""
        return f"{self.id}@{self.version}"

    def sampling(self, batch_size: int) -> dict:
        """Sampling overrides (temperature/top_p/max_tokens) for a batch."""
        return {}

    def serialize_source(
        self,
        segments: list[dict],
        *,
        ids: list[int],
        compact: bool = False,
    ) -> str:
        """Source payload string for prompts/metrics; "" if not applicable."""
        del segments, ids, compact
        return ""

    @abc.abstractmethod
    def build_messages(
        self,
        segments: list[dict],
        *,
        ids: list[int],
        ctx: ProfileContext,
    ) -> list[dict]:
        """Messages for one batch; ``ids`` are global segment indexes."""

    @abc.abstractmethod
    def parse_response(
        self,
        text: str,
        *,
        ids: list[int],
    ) -> dict[int, str | None]:
        """Map reply text to ``{global_id: normalized_text}``.

        Raise ``RetryableTranslationFormatError`` on contract violations.
        Profiles with ``supports_partial_reissue`` may return a subset / None
        values; strict line profiles must cover every id.
        """
