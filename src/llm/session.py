"""One film's translation, as one object both of its passes share.

A translation run is two passes over the same material: the batch pass, and the
repair pass that reissues the cues a detector flagged. They have to agree about
a lot - which profile shapes the prompt and the cache key, which model instance
answers, which thinking tier the base pass paid for, which glossary and target
language the text was produced under, how large a request may get. None of that
is a per-pass decision, and none of it was represented as one thing: the repair
pass took eight parallel keyword arguments, each of which the caller could get
wrong independently, and the only thing keeping them consistent was that one
call site happened to pass the right values.

The cost of "independently" is not hypothetical for the neighbouring case: the
profile used to be selected twice, and the second selection disagreed with the
first, so the cache key described a prompt the request had not used. The session
is what makes that shape impossible here - a pass takes the session or it takes
nothing, and everything derived from it (the escalated repair tier, the endpoint
capability, the profile binding) is derived once.

It is a value: nothing about the run can change through it, and holding one does
not hold a resource. The lease, the cancel event and the endpoint snapshot are
not in it - those are thread-scoped and travel through `llm.run_context`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from llm import request_config
from llm import settings as llm_settings
from llm import token_budget
from llm.profiles.base import TranslationProfile


@dataclass(frozen=True)
class TranslationSession:
    """Everything both passes of one translation run must agree about."""

    backend_name: str
    profile: TranslationProfile
    batch_size: int
    max_workers: int
    cache_path: str
    target_lang: str
    glossary: str
    character_reference: str
    # The tier the base pass actually carried, already resolved - not the
    # caller's request and not what the settings say now. The repair pass
    # escalates *from* this, so reading it live would let a settings change
    # midway through a film silently change what the repair pass is escalating.
    reasoning_effort: str

    @property
    def repair_reasoning_effort(self) -> str:
        """The tier the repair pass reissues at, derived from the base tier.

        The base tier floored at `low` - `none` has to escalate, above it the
        escalation was measured and did not pay - unless
        `TRANSLATION_REPAIR_REASONING_EFFORT` pins a tier. Derived here so the
        two passes cannot end up with different answers to "floored from what?".
        """
        return llm_settings._repair_reasoning_effort(self.reasoning_effort)

    def endpoint_identity(self) -> str:
        """What the learned `max_tokens` ceiling is keyed by for this run.

        The capability state itself lives in `llm.token_budget` and is applied
        where a request is built, which is the only place that knows the reply's
        own size. What matters here is that both passes key it the same way: a
        settings change mid-film must not make the repair pass size its requests
        against a ceiling another endpoint taught us.
        """
        return token_budget.endpoint_identity(self.backend_name)

    @contextmanager
    def bound(self) -> Iterator["TranslationSession"]:
        """Bind the profile for the duration of a pass.

        Bound rather than only passed down: the cache-key helpers are reached
        from places that never see an argument - the repair pass writes its
        corrected text back through the same key builder the batch pass used,
        and that key is derived from the profile.
        """
        previous = request_config.current_profile()
        request_config.bind_profile(self.profile)
        try:
            yield self
        finally:
            request_config.bind_profile(previous)


__all__ = ["TranslationSession"]
