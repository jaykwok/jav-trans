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
from dataclasses import dataclass


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

    def max_batch_size(self) -> int | None:
        """Hard cap on how many cues may share one request, or None for no cap.

        A cap, not a preference: the line-oriented contract returns one bare
        translation with no ids in it, so two cues in one request cannot be told
        apart afterwards. `translate_segments` applies this after the shared
        cues-per-worker sizing rule; this is a model contract, not a selectable
        scheduling mode.
        """
        return None

    def response_token_budget(
        self,
        segments: list[dict],
        *,
        reasoning_effort: str = "",
    ) -> int | None:
        """Upper bound on reply length for these segments, or None for no bound.

        Not a tuning knob: it is how long the answer *can* be, so a model stuck
        in a repetition loop stops at the bound rather than at the configured
        ceiling. Profiles own it because only the profile knows what structure
        it asked the model to emit around the translations.

        `reasoning_effort` is passed because the bound goes out as `max_tokens`,
        which on a reasoning request also has to cover the thinking the answer
        is not made of. Whether there *is* any thinking is read off the same
        argument - the `none` tier needs no allowance - so a caller cannot size
        a budget for a mode the request is not in. See `json_v3`.
        """
        del segments, reasoning_effort
        return None

    def bounded_schema(self, segments: list[dict]) -> dict | None:
        """`schema` narrowed to what these segments can legitimately produce.

        The static `schema` pins the shape but not the size, so a model can sit
        inside the grammar and still write one field forever. Returning None
        means "the static schema is already tight enough".

        Separate from `response_token_budget` because they fail differently: a
        token budget truncates the reply mid-string and leaves unparseable JSON,
        while a bound in the grammar makes the runaway *unrepresentable* - the
        sampler cannot pick a token that would exceed it, so the reply is still
        valid JSON. Only the second one is a correctness fix.
        """
        del segments
        return None

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
