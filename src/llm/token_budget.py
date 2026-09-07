"""What an endpoint will accept as `max_tokens`, and what to ask it for.

This is the arithmetic and the learned state, with nothing of the translation
engine in it: it knows the configured fallback, the per-endpoint limits on disk
and how the two combine, and nothing about batches, prompts, retries or which
backend object a task holds. Callers name the endpoint; this module answers.

That split exists for a reason beyond tidiness. `preflight` runs *before* a
translation to say whether the budget will be clamped, and the only place that
could answer was inside the engine it precedes - so the import graph closed into
`backends → openai_compat → preflight → translator → backends`, the one
multi-module cycle in `src/`. Lazy imports kept it from failing at startup,
which is not the same as it being right: the question "how much room does this
endpoint give" has no dependency on the thing that spends it.
"""

from __future__ import annotations

from llm import max_tokens_limits
from llm import request_config
from llm import settings as llm_settings

# The fallback ceiling, used only where the endpoint has said nothing. Bound at
# import like every other setting; tests rebind this name, which is why the code
# below reads it through the module rather than capturing it in a default.
CONFIGURED_MAX_TOKENS = llm_settings.TRANSLATION_MAX_TOKENS

# One warning per endpoint per process. A per-batch line would be noise.
_clamp_warned: set[str] = set()


def endpoint_identity(backend_name: str) -> tuple[str, str] | None:
    """`(base_url, model)` for an API backend, or None for a local one.

    Local backends never refuse a `max_tokens`, and keying a learned limit on
    their empty base URL would let one endpoint's ceiling answer for another.

    Read from the task's frozen configuration, like the request itself: a limit
    learned from a refusal is a fact about the endpoint that refused it, and
    filing it under whatever the settings page names a moment later would teach
    one endpoint's ceiling to a different one.
    """
    if backend_name != "openai":
        return None
    model = request_config.getenv("LLM_MODEL_NAME", llm_settings.LLM_MODEL_NAME).strip()
    if not model:
        return None
    return request_config.getenv("OPENAI_COMPATIBILITY_BASE_URL", "").strip(), model


def resolved_max_tokens(desired: int | None, limits) -> int:
    """The budget the caller wants, with "no preference" turned into a number.

    `desired=None` is the prefix warmup and anything else with no arithmetic
    bound of its own: it means "as much as this endpoint allows", which is the
    named ceiling when there is one and the configured fallback only otherwise.
    That is the *whole* job of the fallback - it stands in for a missing budget,
    never for a ceiling, so an explicit budget is never trimmed to it.
    """
    if desired is not None:
        return max(1, int(desired))
    return int(getattr(limits, "exact_ceiling", None) or CONFIGURED_MAX_TOKENS)


def local_budget(desired: int | None) -> int:
    """The no-endpoint case, where the configured value really is a ceiling.

    A local model cannot refuse a `max_tokens`, so there is nothing to learn
    from and nothing to fall back to - `TRANSLATION_MAX_TOKENS` is the runaway
    backstop it has always been, and a caller-supplied budget may only lower it.
    """
    return min(resolved_max_tokens(desired, None), CONFIGURED_MAX_TOKENS)


class EndpointCapability:
    """What one call knows about its endpoint's `max_tokens` ceiling.

    The live copy is this object; the cache file is where it is left for the
    next call. That order is the point. Reading the state back from disk after
    every refusal made the probe ladder depend on the cache being writable, and
    a cache that is not - a read-only directory, a full disk - reads back as
    "nothing known", where the next step is `budget_for(nothing, sent - 1)`,
    i.e. `sent - 1`. The ladder then walks down one token per round trip and the
    request fails at a budget the endpoint would have taken two halvings later.
    A capability cache is an optimisation; a request must not need it to work.
    """

    def __init__(self, identity: tuple[str, str] | None) -> None:
        self.identity = identity
        self.limits = (
            max_tokens_limits.load_limits(*identity)
            if identity is not None
            else max_tokens_limits.EndpointLimits()
        )
        # Ceiling-side observations wait here until this call generates at a
        # smaller budget. See `_corroborate`.
        self._staged: list[tuple[str, int]] = []
        self._corroborated = False

    def _stage(self, kind: str, value: int) -> None:
        """Hold a refusal, or write it if this call has already been convinced."""
        if self.identity is None:
            return
        if self._corroborated:
            self._write(kind, value)
        else:
            self._staged.append((kind, value))

    def _write(self, kind: str, value: int) -> None:
        if kind == "exact_ceiling":
            max_tokens_limits.record_exact_ceiling(*self.identity, value)
        else:
            max_tokens_limits.record_rejection(*self.identity, value)

    def _corroborate(self) -> None:
        """A budget under the refusal just generated, so the refusal can be kept.

        Nothing expires any more, so the cache has no way to walk back a number
        that should not have gone in. The check that replaces it is behavioural:
        refuse high, generate low is what a real ceiling looks like, and a
        message whose wording was merely read as a ceiling will not produce it.
        Until that happens the refusal steers this call and nothing else.

        The merge keeps every floor strictly under `rejected_at` and at or under
        `exact_ceiling`, so any recorded success is by construction the one this
        is asking about - `131072` accepted against a named `131072` corroborates
        it exactly as much as `32768` accepted under a refused `65536` does.
        """
        if self._corroborated or not (
            self.limits.rejected_at or self.limits.exact_ceiling
        ):
            return
        self._corroborated = True
        for kind, value in self._staged:
            self._write(kind, value)
        self._staged.clear()

    def _observe(
        self,
        *,
        exact_ceiling: int | None = None,
        rejection: int | None = None,
        success: int | None = None,
    ) -> None:
        self.limits = max_tokens_limits.merge_observation(
            self.limits,
            exact_ceiling=exact_ceiling,
            rejection=rejection,
            success=success,
        )

    def record_exact_ceiling(self, ceiling: int, *, persist: bool = True) -> None:
        """The endpoint named its ceiling. The one kind that clamps."""
        if ceiling <= 0:
            return
        self._observe(exact_ceiling=int(ceiling))
        if persist:
            self._stage("exact_ceiling", int(ceiling))

    def record_rejection(self, sent: int, *, persist: bool = True) -> None:
        """`sent` was refused, so everything at or above it is out.

        `persist=False` keeps a refusal that is about this request rather than
        about the endpoint - a combined input+output limit - inside this call.
        It still drives the bisection, because a smaller budget really does fit;
        it just never becomes a fact about an endpoint that will see shorter
        prompts than this one.
        """
        if sent <= 0:
            return
        self._observe(rejection=int(sent))
        if persist:
            self._stage("rejection", int(sent))

    def record_success(self, accepted: int, *, persist: bool = True) -> None:
        """`accepted` went out and was generated against. A lower bound.

        `persist=False` is the first-try case: the endpoint took the number, so
        this call now knows a floor, but writing "at least this much works" on
        every batch would be a disk write per request to store something no
        refusal has bounded. The memory half is never optional - it is what the
        rest of this call bisects against.
        """
        if accepted <= 0:
            return
        self._observe(success=int(accepted))
        # Before the floor itself: whatever was refused above it is what makes
        # this number worth having on disk at all.
        self._corroborate()
        if persist and self.identity is not None:
            max_tokens_limits.record_success(*self.identity, int(accepted))

    def budget(self, desired: int | None, *, warn: bool = False) -> int:
        """What to actually ask for, given what this endpoint has said so far."""
        if self.identity is None:
            return local_budget(desired)
        resolved = resolved_max_tokens(desired, self.limits)
        budget = max_tokens_limits.budget_for(self.limits, resolved)
        if warn and budget < resolved and desired is not None:
            self._warn_clamped(resolved, budget)
        return budget

    def _warn_clamped(self, resolved: int, budget: int) -> None:
        """Shrinking a computed budget is the silent half of this: the request
        simply goes out with less room, and if the reply is then cut off the
        truncation escalation cannot raise it back past the same line. Said once
        per endpoint per process - a per-batch line would be noise."""
        key = max_tokens_limits.endpoint_key(*self.identity)
        if key in _clamp_warned:
            return
        _clamp_warned.add(key)
        source = (
            f"端点上限 {self.limits.exact_ceiling}"
            if self.limits.exact_ceiling
            else (
                f"端点已拒绝过 {self.limits.rejected_at}"
                if self.limits.rejected_at
                else f"TRANSLATION_MAX_TOKENS={CONFIGURED_MAX_TOKENS}"
            )
        )
        print(
            f"[WARN] 本次翻译预算 {resolved} 被压到 {budget}（{source}）。"
            "回复被切断时将无法再向上重试；如需更大预算请调高 "
            "TRANSLATION_MAX_TOKENS 或调小批次/推理配额。",
            flush=True,
        )

    def next_probe_after(self, sent: int) -> int:
        """The next value to try after `sent` was refused.

        Read off the bracket this call has already narrowed, so the step is the
        midpoint of what is still possible. Only an endpoint that has said
        nothing at all - a local backend - falls back to halving, which is the
        same thing with no information.
        """
        if not self.limits.known_anything:
            return sent // 2
        candidate = max_tokens_limits.budget_for(self.limits, sent - 1)
        return candidate if 0 < candidate < sent else sent // 2


def plain_budget(desired: int | None, *, backend_name: str) -> int:
    """The budget with no warning printed. Safe to call for a preview.

    The warning-emitting variant would print at startup and then mark the
    endpoint as already-warned, so the run-time clamp - the one that sees the
    real source text - would never say anything.
    """
    return EndpointCapability(endpoint_identity(backend_name)).budget(desired)


def budget_with_warning(desired: int | None, *, backend_name: str) -> int:
    """What to actually ask for, and say so the first time it is less.

    Reads the cache, so this answers for a request that has not started yet.
    Within one call the same question goes to that call's own capability state
    instead, which knows what this one cannot: what the endpoint has said during
    this very request.
    """
    return EndpointCapability(endpoint_identity(backend_name)).budget(desired, warn=True)
