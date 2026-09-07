"""Who may use the GPU right now - a different question from who still has to
clean up after themselves.

``core.resources`` answers the second one: a process we asked to stop, and that
has not confirmed it is gone. That ledger was doing double duty as the device's
admission rule, and the two questions do not have the same answer.
``gpu_blocked()`` is true only for a record in ``stopping`` state, so a
perfectly healthy llama-server holding six gigabytes of VRAM said nothing at
all about whether the next ASR stage could start - and the arbitration that did
exist ran in one direction only: loading a local model stops the ASR worker
first, while starting ASR never looked at the local model at all.

So the permit lives here, beside the ledger rather than inside it:

* **The ledger still says "released".** A record in ``stopping`` that has not
  confirmed still owns its VRAM, so no permit is granted while
  :func:`core.resources.gpu_blocked` is true. An unconfirmed cleanup cannot be
  overtaken by a new claim.
* **The permit says "in use".** One holder at a time, for as long as that
  holder is actually about to occupy the device.

A stage refuses to start beside any *live* GPU owner it is not itself
responsible for stopping. Default-deny, and deliberately asymmetric:

``ASR`` waits for a resident llama-server to go away. Nothing stops one on
ASR's behalf - it is shared between jobs and may be mid-generation for one of
them.

``LOCAL_LLM`` does not wait for a live ASR worker, because it stops it itself:
that is what ``_release_asr_worker_vram`` is for. It only needs the permit, so
that no transcription can start inside the window between "the ASR child was
stopped" and "the server is registered as the new owner of the card".

Stated as exemptions rather than as conflicts on purpose. The failure mode of
the opposite default is a second multi-GB model loaded onto an occupied card,
and a subsystem added later should inherit the safe answer instead of the
silent one - which is the exact shape of bug this project keeps re-finding in
each new copy of GPU bookkeeping.

A stage that stays resident does not release its permit - it **hands it over**
to the identity it just registered, in one lock section (:func:`handover`).
That step is the whole protocol, because the alternative is two steps with a
gap in them:

    ASR checks, and finds the card free
    -> the local start takes the permit, registers its server, releases
    -> ASR grants itself the permit on the strength of that earlier check,
       and transcribes next to a resident multi-GB model.

Exclusivity of the permit does not close that, and neither does moving the
registry read inside this module's lock: the gap is between "the resident
resource exists" and "the permit is free", and only making those one atomic
transition removes it. So a claim is created *while* its creator still holds
the permit, and admission sees a claim from the same instant the permit is
gone. The claim then ends by itself when the registry stops reporting that
identity as holding the card, which is why nothing here has to be released by
hand: the registry - with its watchdog and its manual retry - already owns that
lifetime, and a permit this module owned alone could be stranded by a close
that never confirmed.

The registry is still scanned as a second, weaker rule: a live GPU-holding
record of a kind this stage may not displace also blocks, claim or no claim.
That is the safety net for a resource registered outside the protocol; the
claim is what makes the common path race-free.

A permit is owned by the caller that holds the handle, never by the thread that
took it. Waiting therefore happens outside every other subsystem's lock and is
always cancellable: waiting for another task to finish is exactly the wait a
user's 取消 has to be able to reach, and holding a lock across it is how that
wait became unreachable in the first place.

API translation takes no permit at all. It holds no VRAM and must keep
overlapping with ASR the way it does today.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass
from typing import Any

from core import resources

# The registry `kind` each stage registers its process under. Spelled once here
# so the conflict table below and the subsystems cannot drift apart.
ASR = "asr_worker"
LOCAL_LLM = "local_server"

# The live GPU owners each stage may start beside - everything else makes it
# wait. A stage always tolerates its own kind (the ASR child stays resident
# between jobs by design), and `LOCAL_LLM` additionally tolerates the ASR
# worker because stopping it is part of its own startup. Anything not named
# here blocks, including a kind added after this table was written.
_DISPLACEABLE_LIVE_KINDS: dict[str, frozenset[str]] = {
    ASR: frozenset({ASR}),
    LOCAL_LLM: frozenset({LOCAL_LLM, ASR}),
}

# How long a waiter sleeps between attempts. A release notifies, so this only
# bounds how fast a waiter notices a cancel or a change in the cleanup ledger.
_POLL_S = 0.1


class GpuAdmissionCancelled(RuntimeError):
    """The caller was cancelled while waiting for the device."""


class GpuAdmissionTimeout(RuntimeError):
    """The wait ran out. Ends admission only - it releases nothing."""


class GpuHandoverError(RuntimeError):
    """A handover was attempted with a permit that is no longer held."""


@dataclass(frozen=True)
class GpuPermit:
    kind: str
    description: str
    serial: int
    since: float

    def view(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            # Never a command line or a media path - same rule as the registry.
            "description": self.description,
            "serial": self.serial,
            "since": self.since,
            "held_s": max(0.0, time.time() - self.since),
        }


@dataclass(frozen=True)
class GpuClaim:
    """A permit that became a resident resource.

    Named by registry identity, so it ends exactly when that identity stops
    holding the card - and by `(resource_id, generation)`, so a claim raised for
    one server can never describe the one that replaced it.
    """

    kind: str
    description: str
    resource_id: str
    generation: int
    since: float

    def view(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "description": self.description,
            "resource_id": self.resource_id,
            "generation": self.generation,
            "since": self.since,
            "held_s": max(0.0, time.time() - self.since),
        }


_LOCK = threading.Lock()
_FREE = threading.Condition(_LOCK)
_HOLDER: GpuPermit | None = None
_CLAIMS: dict[tuple[str, int], GpuClaim] = {}
_SERIAL = itertools.count(1)


def _cancelled(cancel) -> bool:
    """Accept either a `threading.Event` or a plain `() -> bool` predicate.

    The two stages that take permits already carry one each, and neither shape
    is worth adapting at every call site. A predicate that raises is treated as
    "not cancelled": failing to read a cancel flag must not by itself abort a
    stage that was never asked to stop.
    """
    if cancel is None:
        return False
    try:
        is_set = getattr(cancel, "is_set", None)
        if callable(is_set):
            return bool(is_set())
        if callable(cancel):
            return bool(cancel())
    except Exception:
        return False
    return False


def _tolerated(kind: str) -> frozenset[str]:
    return _DISPLACEABLE_LIVE_KINDS.get(kind, frozenset({kind}))


def _prune_claims() -> None:
    """Drop claims whose resource no longer holds the card. Call under ``_FREE``.

    A claim lives exactly as long as the registry reports its identity as
    occupying the GPU, so nothing has to release one by hand - and a close that
    never confirmed cannot strand one here, because that case keeps the record
    and is answered by `resources.gpu_blocked()` instead.
    """
    for key, claim in list(_CLAIMS.items()):
        if not resources.occupies_gpu(claim.resource_id, claim.generation):
            del _CLAIMS[key]


def _claim_conflict(kind: str) -> str:
    """A resident claim ``kind`` refuses to run beside, described, or ""."""
    _prune_claims()
    tolerated = _tolerated(kind)
    for claim in _CLAIMS.values():
        if claim.kind not in tolerated:
            return f"{claim.kind} {claim.description}"
    return ""


def _live_conflict(kind: str) -> str:
    """The safety net: a live GPU record of a kind ``kind`` may not displace.

    Weaker than a claim - a registration is not synchronised with this module,
    so this read can be overtaken by one that lands a moment later. It is here
    for a resource registered outside the handover protocol, where blocking one
    check late still beats never blocking at all.
    """
    tolerated = _tolerated(kind)
    for record in resources.live():
        if record["holds_gpu"] and record["kind"] not in tolerated:
            return f"{record['kind']} pid={record['pid']}"
    return ""


def acquire(
    kind: str,
    *,
    cancel=None,
    timeout_s: float | None = None,
    description: str = "",
) -> GpuPermit:
    """Wait for exclusive use of the device, cancellably.

    ``timeout_s=None`` waits indefinitely, which is the right default for both
    callers: the thing being waited for is another task's GPU stage, and the
    honest answer is "until it finishes or you cancel". A deadline here would
    end the *wait*, never the other task's ownership.
    """
    global _HOLDER
    deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
    while True:
        if _cancelled(cancel):
            raise GpuAdmissionCancelled(
                f"GPU 使用许可等待已取消（{kind}）。"
            )
        with _FREE:
            # Every read that decides this grant happens inside this section,
            # never before it. They take the registry's own short lock and
            # nothing in `core.resources` ever takes `_FREE`, so there is no
            # cycle; a check taken outside is a check that can go stale before
            # it is acted on. `_claim_conflict` is the one that closes the
            # handover race, because a claim appears in the same lock section
            # that gives the permit up (see `handover`). The `_live_conflict`
            # scan behind it is the weaker safety net for records registered
            # outside the protocol.
            blocked = resources.gpu_blocked()
            conflict = ""
            if not blocked:
                conflict = _claim_conflict(kind) or _live_conflict(kind)
            if _HOLDER is None and not blocked and not conflict:
                _HOLDER = GpuPermit(
                    kind=kind,
                    description=description,
                    serial=next(_SERIAL),
                    since=time.time(),
                )
                return _HOLDER
            if deadline is not None and time.monotonic() >= deadline:
                held = _HOLDER.kind if _HOLDER is not None else ""
                reason = (
                    "有未确认退出的 GPU 子进程"
                    if blocked
                    else f"{conflict} 仍在占用显存"
                    if conflict
                    else f"{held or '另一个阶段'} 正在使用 GPU"
                )
                raise GpuAdmissionTimeout(
                    f"等待 GPU 使用许可超过 {timeout_s:g}s（{kind}）：{reason}。"
                )
            _FREE.wait(_POLL_S)


def release(permit: GpuPermit | None) -> bool:
    """Give the device back. A stale permit releases nothing.

    Matched by serial rather than by kind: two stages of the same kind can
    follow each other, and a late release naming only "asr_worker" would hand
    away the permit its successor is holding.
    """
    global _HOLDER
    if permit is None:
        return False
    with _FREE:
        if _HOLDER is None or _HOLDER.serial != permit.serial:
            return False
        _HOLDER = None
        _FREE.notify_all()
        return True


def handover(
    permit: GpuPermit,
    *,
    resource_id: str,
    generation: int,
    description: str = "",
) -> GpuClaim:
    """Turn exclusive use into a resident claim on a registered identity.

    One lock section, because the two halves are what the race lived between:
    "the resident resource is visible to admission" has to become true at the
    same instant "the permit is free" does. Register the resource first, then
    call this - never register, release, and hope the next check is late enough.

    The claim ends by itself once the registry stops reporting that identity as
    holding the card, so there is no matching release to forget and no way for a
    close that never confirmed to strand one here.
    """
    global _HOLDER
    with _FREE:
        if _HOLDER is None or _HOLDER.serial != permit.serial:
            raise GpuHandoverError(
                "GPU 使用许可已经不在本次调用手上，无法交接给常驻资源"
                f"（{permit.kind} #{permit.serial}）。"
            )
        claim = GpuClaim(
            kind=permit.kind,
            description=description or permit.description,
            resource_id=resource_id,
            generation=generation,
            since=permit.since,
        )
        _CLAIMS[(resource_id, generation)] = claim
        _HOLDER = None
        _FREE.notify_all()
        return claim


def holder() -> GpuPermit | None:
    with _FREE:
        return _HOLDER


def blocked_for(kind: str) -> bool:
    """Would :func:`acquire` have to wait right now?

    The queue reads this to hold a job at 等待 GPU instead of letting it enter
    the ASR stage and park inside the transcription call.
    """
    with _FREE:
        return bool(
            _HOLDER is not None
            or resources.gpu_blocked()
            or _claim_conflict(kind)
            or _live_conflict(kind)
        )


def claims() -> list[dict[str, Any]]:
    """Resident claims that are still holding the card."""
    with _FREE:
        _prune_claims()
        return [claim.view() for claim in _CLAIMS.values()]


def snapshot() -> dict[str, Any]:
    """Permit and claim state for the diagnostics panel."""
    current = holder()
    return {
        "held": current is not None,
        "holder": current.view() if current is not None else None,
        "resident": claims(),
        "asr_blocked": blocked_for(ASR),
    }


def _reset_for_tests() -> None:
    """Drop the permit and every claim. Tests only - it releases no device."""
    global _HOLDER
    with _FREE:
        _HOLDER = None
        _CLAIMS.clear()
        _FREE.notify_all()
