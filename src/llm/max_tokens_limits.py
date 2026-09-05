"""What `max_tokens` an endpoint accepts, learned from what it refuses.

The legal ceiling is a property of `(endpoint, model)` and only the endpoint
knows it. Checked 2026-09-05 against a public model catalogue, the *same*
GLM-5.3-Flash weights are published as 128,000 on one host, 131,072 on another
and 1,048,576 on a third - and the id this project sends (a bare model name to
whatever `OPENAI_COMPATIBILITY_BASE_URL` points at) is not even a key any such
table can be looked up by. So no table ships here.

What is recorded is three different kinds of knowledge, kept apart because they
are not equally strong:

* `exact_ceiling` - the endpoint named this number. It is the ceiling, so it
  clamps.
* `rejected_at` - the smallest number known to be refused. Says nothing about
  what *is* allowed, only that everything at or above it is not.
* `known_good` - the largest number known to have been accepted. A lower bound.

Collapsing the last two into one "learned limit" is the bug this shape exists to
avoid: with a real ceiling of 50,000, sending 65,536 gets refused and 32,768
succeeds, and calling 32,768 "the limit" pins the endpoint a third below its
actual capability for as long as the entry lives. Held apart, the same two
observations instead say `50,000 ∈ [32768, 65536)`, and the next request that
wants more probes the middle rather than accepting the first number that worked.

Entries are persisted because a refusal costs a round trip, and they do not
expire. The cache key is `(base_url, model)`, so a model that gets a different
cap is a *different key*: within one key the number is set when the deployment
ships and stays put until it is retired. Thirty-day expiry was guarding against
a provider quietly raising the cap on an id already in use, and paying for that
by throwing away everything learned, every month, on every endpoint.

What expiry was also doing, silently, was limiting the damage of a number that
should never have been learned. That job moved to where it belongs: a refusal is
written down only once the same call has *generated* at a smaller budget (see
`translator._EndpointCapability`), so what survives forever is a bracket the
endpoint's own behaviour demonstrated, not one inferred from the wording of a
single error message. Only evidence about the *endpoint* is written down at all;
a refusal that is really about one prompt's size stays in the call that saw it.

The one case this gets wrong is an endpoint whose id is served by more than one
upstream - OpenRouter picks a provider per request - where the first bracket
learned can under-book a more generous one. That direction is safe: it costs
capability, never a failed request, and the bisection still climbs toward
whatever the answering provider allows.

Every path is best effort: a capability cache is an optimisation, and no parse
error or failed write in here may cost a translation that would otherwise have
succeeded. Which is why what a call learns lives in the caller's own
`EndpointLimits` (see `translator._EndpointCapability`) and only passes through
here on its way to the next call - a request must not need this file to have
been writable.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.runtime_paths import runtime_path

# v4 has no expiry at all, and a stored refusal now carries a stronger claim
# than a v3 one did (see `record_rejection`), so older files are dropped rather
# than read. Neither v2 nor v3 ever left this working tree.
LIMITS_SCHEMA = "translation_endpoint_max_tokens_v4"
_LOCK = threading.RLock()
_MAX_ENTRIES = 16
# Stop bisecting once the bracket is this tight. The room still unclaimed is
# then worth less than the rejected round trip it would take to claim it.
_CONVERGED_GAP = 1024


@dataclass(frozen=True)
class EndpointLimits:
    exact_ceiling: int | None = None
    rejected_at: int | None = None
    # The best floor still in evidence, and the last one observed. Two slots for
    # one kind of knowledge because they age and die separately: a success at
    # 30,000 neither re-proves a 50,000 nor is disproved when a refusal at
    # 40,000 kills it.
    known_good: int | None = None
    known_good_recent: int | None = None

    @property
    def known_anything(self) -> bool:
        return any(
            value is not None
            for value in (
                self.exact_ceiling,
                self.rejected_at,
                self.known_good,
                self.known_good_recent,
            )
        )


def limits_path() -> Path:
    raw = os.getenv(
        "TRANSLATION_MAX_TOKENS_CACHE_PATH",
        "tmp/cache/translation_max_tokens.json",
    ).strip()
    return runtime_path(raw or "tmp/cache/translation_max_tokens.json")


def endpoint_key(base_url: str, model: str) -> str:
    """Hashed identity. The base URL is not stored: a private relay's host has
    no business sitting in a cache file that gets pasted into bug reports."""
    encoded = json.dumps(
        {"base_url": base_url.strip(), "model": model.strip()},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _timestamp(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _empty_payload() -> dict[str, Any]:
    return {"schema": LIMITS_SCHEMA, "limits": {}}


def _load_payload() -> dict[str, Any]:
    try:
        payload = json.loads(limits_path().read_text(encoding="utf-8"))
    except Exception:
        return _empty_payload()
    if not isinstance(payload, dict) or payload.get("schema") != LIMITS_SCHEMA:
        return _empty_payload()
    if not isinstance(payload.get("limits"), dict):
        payload["limits"] = {}
    return payload


def _write_payload(payload: dict[str, Any]) -> None:
    target = limits_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(target)


def _entry_limits(entry: Any) -> EndpointLimits:
    if not isinstance(entry, dict):
        return EndpointLimits()
    floors = [
        _positive_int(entry.get(field))
        for field in ("known_good", "known_good_recent")
    ]
    return EndpointLimits(
        exact_ceiling=_positive_int(entry.get("exact_ceiling")),
        rejected_at=_positive_int(entry.get("rejected_at")),
        # Best floor on record, and the last one recorded. Two slots, and the
        # merge rules decide between them - see `merge_observation`.
        known_good=max((value for value in floors if value), default=None),
        known_good_recent=floors[1],
    )


def load_limits(base_url: str, model: str) -> EndpointLimits:
    """Everything known about this endpoint's ceiling. Never raises."""
    try:
        if not model.strip():
            return EndpointLimits()
        with _LOCK:
            payload = _load_payload()
        return _entry_limits(payload["limits"].get(endpoint_key(base_url, model)))
    except Exception:
        return EndpointLimits()


def budget_for(limits: EndpointLimits, desired: int) -> int:
    """How much to ask for, given what this endpoint has already said.

    `desired` is what the caller actually wants; resolving "no preference" into
    a number is the caller's job, because the fallback is a stand-in for a
    *budget*, not a claim about the endpoint. Nothing known therefore means the
    request goes out exactly as asked - it is the refusal path's job to discover
    a ceiling, and clamping to a fallback here is how the truncation escalation
    used to be silently capped at a number nobody had measured.

    Only an exact ceiling clamps. A bracket instead picks the next bisection
    step, so a refusal narrows it rather than pinning the endpoint at the first
    number that happened to succeed.
    """
    desired = max(1, int(desired))
    if limits.exact_ceiling:
        return min(desired, limits.exact_ceiling)
    if not limits.rejected_at:
        return desired
    known_good = limits.known_good or 0
    if desired <= known_good:
        return desired
    if limits.rejected_at - known_good <= _CONVERGED_GAP:
        return min(desired, known_good) if known_good else min(desired, limits.rejected_at - 1)
    probe = (
        (known_good + limits.rejected_at) // 2
        if known_good
        else limits.rejected_at // 2
    )
    return min(desired, max(probe, known_good, 1))


def merge_observation(
    current: EndpointLimits,
    *,
    exact_ceiling: int | None = None,
    rejection: int | None = None,
    success: int | None = None,
) -> EndpointLimits:
    """Fold one observation into what was already known, keeping it consistent.

    A rejection is a lower ceiling, a success is a higher floor, and either can
    contradict the other side of the bracket - an endpoint whose cap moved, or a
    hand-edited file. The contradicted half is dropped rather than kept, because
    two numbers that cannot both be true make the bisection nonsense.

    Which half goes is decided by age, not by strength: the observation being
    folded in just happened, and everything else in `current` is at most a month
    of cache. So a refusal at or below a stored `exact_ceiling` retires the
    ceiling and starts a new bracket - keeping it instead left every retry
    clamped to a number the endpoint had *just* said no to, which walks the
    ladder down one token per round trip and never reaches a budget that works.

    That has to run in both directions or it is not an age rule at all. A reply
    generated at 50,000 disproves a cached "everything from 40,000 up is
    refused" exactly as flatly, and dropping the *success* there - which is what
    happens when the halves are ranked instead of dated - throws away the one
    number this call has actually seen work and keeps the one it just contradicted.
    """
    exact = current.exact_ceiling
    rejected = current.rejected_at
    best = current.known_good
    recent = current.known_good_recent
    if exact_ceiling:
        # The endpoint named a number. Whatever a bisection had guessed around
        # it is superseded, in either direction.
        exact = exact_ceiling
        if rejected and rejected <= exact:
            rejected = None
    if rejection:
        rejected = rejection if rejected is None else min(rejected, rejection)
        if exact and exact_ceiling is None and rejection <= exact:
            exact = None
    if success:
        recent = success
        best = success if best is None else max(best, success)
        if rejected and rejection is None and success >= rejected:
            rejected = None
        if exact and exact_ceiling is None and success > exact:
            exact = None

    def _survives(value: int | None) -> bool:
        """Whether a floor is still compatible with the ceiling side.

        Applied to each slot on its own, which is the point of having two. A
        refusal at 40,000 disproves a 50,000 that was accepted and says nothing
        at all about a 30,000 that was; dropping the pair together threw away a
        floor nothing had contradicted and sent the next bisection back to
        `rejected_at // 2`, which is a *smaller* budget than one this endpoint
        had already generated at - and being too small costs a truncated,
        billed reply rather than a free refusal.
        """
        if not value:
            return False
        if rejected and value >= rejected:
            return False
        if exact and value > exact:
            return False
        return True

    # Whatever is left of a contradiction here is old against old - a
    # half-written or hand-edited file, since every fresh observation above
    # resolved its own conflicts. Keeping the refusal is the conservative half:
    # asking too high costs a refused round trip, asking too low costs a probe.
    best = max((value for value in (best, recent) if _survives(value)), default=None)
    recent = recent if _survives(recent) else None
    return EndpointLimits(
        exact_ceiling=exact,
        rejected_at=rejected,
        known_good=best,
        known_good_recent=recent,
    )


def _record(
    base_url: str,
    model: str,
    *,
    exact_ceiling: int | None = None,
    rejection: int | None = None,
    success: int | None = None,
) -> None:
    """Read, merge and write under one lock. Swallows everything.

    One critical section rather than the load-then-update pair this used to be:
    with two, concurrent batches could each read the same bracket and the slower
    write would put back the weaker observation.
    """
    try:
        if not model.strip():
            return
        now = time.time()
        key = endpoint_key(base_url, model)
        with _LOCK:
            payload = _load_payload()
            merged = merge_observation(
                _entry_limits(payload["limits"].get(key)),
                exact_ceiling=exact_ceiling,
                rejection=rejection,
                success=success,
            )
            entry: dict[str, Any] = {
                "model": model.strip(),
                # Last touched, for eviction only. Nothing reads it as evidence
                # about any of the numbers below.
                "updated_at": round(now, 3),
            }
            for field in (
                "exact_ceiling",
                "rejected_at",
                "known_good",
                "known_good_recent",
            ):
                value = getattr(merged, field)
                if value:
                    entry[field] = value
            payload["limits"][key] = entry
            limits = payload["limits"]
            if len(limits) > _MAX_ENTRIES:
                payload["limits"] = dict(
                    sorted(
                        limits.items(),
                        key=lambda item: _timestamp(
                            item[1].get("updated_at")
                            if isinstance(item[1], dict)
                            else None
                        ),
                        reverse=True,
                    )[:_MAX_ENTRIES]
                )
            _write_payload(payload)
    except Exception as exc:
        print(f"[WARN] endpoint max_tokens cache not updated: {exc!r}", flush=True)


def record_exact_ceiling(base_url: str, model: str, ceiling: int) -> None:
    """The endpoint named its ceiling. This is the one kind that clamps."""
    if ceiling > 0:
        _record(base_url, model, exact_ceiling=int(ceiling))


def record_rejection(base_url: str, model: str, sent: int) -> None:
    """`sent` was refused, so everything at or above it is out.

    Written here means written for good - there is no expiry to walk it back.
    Which is why the caller does not call this the moment a refusal arrives:
    `translator._EndpointCapability` holds it until the same call has generated
    at a smaller budget, so a permanent bracket needs the endpoint to have
    *behaved* like one, not merely to have said something a phrase list read as
    a ceiling.
    """
    if sent > 0:
        _record(base_url, model, rejection=int(sent))


def record_success(base_url: str, model: str, accepted: int) -> None:
    """`accepted` went through. A lower bound on the ceiling, never the ceiling."""
    if accepted > 0:
        _record(base_url, model, success=int(accepted))


__all__ = [
    "EndpointLimits",
    "budget_for",
    "endpoint_key",
    "limits_path",
    "load_limits",
    "merge_observation",
    "record_exact_ceiling",
    "record_rejection",
    "record_success",
]
