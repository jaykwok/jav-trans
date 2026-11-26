"""What "the GPU is free" has to mean before a stage may start.

The cleanup ledger answers a narrower question - has a process we asked to stop
confirmed it is gone - and it was standing in for this one. A healthy owner is
invisible to it, so a resident llama-server read as a free card; and the only
arbitration that existed ran one way, since loading a local model stops the ASR
worker while starting ASR never looked at the local model at all.

These pin the permit's four properties: exclusivity, the asymmetry, the
cancellable wait, and that an unconfirmed cleanup still closes admission.
"""

from __future__ import annotations

import threading
import time

import pytest

from core import gpu_admission, resources


@pytest.fixture(autouse=True)
def _clean_admission():
    yield
    gpu_admission._reset_for_tests()
    resources.forget_all()


class _FakeProcess:
    def __init__(self, pid: int = 1) -> None:
        self.pid = pid


def _register_live(kind: str, pid: int = 1) -> str:
    resource_id = resources.new_resource_id(kind)
    resources.register(
        resource_id=resource_id,
        generation=1,
        kind=kind,
        description=kind,
        process=_FakeProcess(pid),
        stopper=lambda process, handle: resources.StopOutcome(process_state="gone"),
        holds_gpu=True,
    )
    return resource_id


def test_two_stages_never_hold_the_card_at_once():
    first = gpu_admission.acquire(gpu_admission.ASR)
    with pytest.raises(gpu_admission.GpuAdmissionTimeout):
        gpu_admission.acquire(gpu_admission.LOCAL_LLM, timeout_s=0.2)

    assert gpu_admission.release(first) is True
    second = gpu_admission.acquire(gpu_admission.LOCAL_LLM, timeout_s=0.5)
    assert second.kind == gpu_admission.LOCAL_LLM


def test_the_other_order_is_refused_too():
    # Both orders, because the bug was directional: whichever stage started
    # second used to be the one that got in.
    held = gpu_admission.acquire(gpu_admission.LOCAL_LLM)
    with pytest.raises(gpu_admission.GpuAdmissionTimeout):
        gpu_admission.acquire(gpu_admission.ASR, timeout_s=0.2)
    gpu_admission.release(held)


def test_a_waiter_stops_waiting_when_the_task_is_cancelled():
    # The whole point of holding this wait outside every other subsystem's lock:
    # what is being waited for is another job's GPU stage, which can run for
    # minutes, so 取消 has to be able to reach the waiter.
    held = gpu_admission.acquire(gpu_admission.ASR)
    cancel = threading.Event()
    outcome: list[str] = []

    def waiter() -> None:
        try:
            gpu_admission.acquire(gpu_admission.LOCAL_LLM, cancel=cancel)
        except gpu_admission.GpuAdmissionCancelled:
            outcome.append("cancelled")

    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    time.sleep(0.2)
    assert outcome == []  # still waiting; nothing was killed to get past it

    cancel.set()
    thread.join(3.0)
    assert outcome == ["cancelled"]
    assert gpu_admission.holder() is held  # the running stage kept the card


def test_a_live_local_server_holds_asr_off_the_card():
    # `gpu_blocked()` is false here - nothing was asked to stop. That is exactly
    # the state in which a job used to be dispatched to load a second multi-GB
    # model beside the first.
    _register_live(gpu_admission.LOCAL_LLM, pid=42)

    assert resources.gpu_blocked() is False
    assert gpu_admission.blocked_for(gpu_admission.ASR) is True
    with pytest.raises(gpu_admission.GpuAdmissionTimeout):
        gpu_admission.acquire(gpu_admission.ASR, timeout_s=0.2)


def test_a_live_asr_worker_does_not_hold_the_local_model_off():
    # Deliberately not symmetric: the local start *stops* the ASR worker itself
    # (`_release_asr_worker_vram`), so waiting for it would be waiting for
    # something only this caller is going to do.
    _register_live(gpu_admission.ASR, pid=7)

    assert gpu_admission.blocked_for(gpu_admission.LOCAL_LLM) is False
    permit = gpu_admission.acquire(gpu_admission.LOCAL_LLM, timeout_s=0.5)
    assert permit.kind == gpu_admission.LOCAL_LLM


def test_the_asr_worker_does_not_block_itself():
    # It stays resident between jobs on purpose; a stage always tolerates a live
    # owner of its own kind.
    _register_live(gpu_admission.ASR, pid=3)

    assert gpu_admission.blocked_for(gpu_admission.ASR) is False


def test_an_unrecognised_gpu_owner_blocks_rather_than_being_ignored():
    # Default-deny. Listing conflicts instead of exemptions means a subsystem
    # added later is admitted beside anything until somebody remembers to add
    # it to a table - which is the shape of bug this module exists to end.
    _register_live("some_future_gpu_stage", pid=11)

    assert resources.gpu_blocked() is False
    for kind in (gpu_admission.ASR, gpu_admission.LOCAL_LLM):
        assert gpu_admission.blocked_for(kind) is True


def test_an_unconfirmed_cleanup_still_closes_admission():
    resources.register(
        resource_id="worker",
        generation=1,
        kind=gpu_admission.ASR,
        description="worker",
        process=_FakeProcess(9),
        stopper=lambda process, handle: resources.StopOutcome(process_state="alive"),
        holds_gpu=True,
        state="stopping",
        process_state="unknown",
    )

    assert resources.gpu_blocked() is True
    for kind in (gpu_admission.ASR, gpu_admission.LOCAL_LLM):
        with pytest.raises(gpu_admission.GpuAdmissionTimeout):
            gpu_admission.acquire(kind, timeout_s=0.2)


def test_a_registry_scan_cannot_go_stale_before_the_permit_is_granted(monkeypatch):
    """Check and grant are one decision, or they are no decision at all.

    Reading the ledger before taking the lock leaves a window the other side
    fits through whole: ASR scans and finds the card free -> the local start
    takes the permit, registers its resident server and releases it -> ASR
    grants itself the permit on the strength of a scan that predates all three,
    and transcribes next to a resident multi-GB model. Exclusive `_HOLDER` does
    not close that; the scan has to happen inside the same section as the grant.
    """
    scanning = threading.Event()
    paused_once = threading.Event()
    real_conflict = gpu_admission._live_conflict

    def paused(kind: str) -> str:
        answer = real_conflict(kind)
        if threading.current_thread().name == "probe-asr" and not paused_once.is_set():
            paused_once.set()
            scanning.set()
            time.sleep(0.5)  # hold this decision open
        return answer

    monkeypatch.setattr(gpu_admission, "_live_conflict", paused)
    granted: list[gpu_admission.GpuPermit] = []

    def asr() -> None:
        granted.append(gpu_admission.acquire(gpu_admission.ASR, timeout_s=3.0))

    thread = threading.Thread(target=asr, name="probe-asr", daemon=True)
    thread.start()
    assert scanning.wait(3.0)

    # The local side's whole handover, attempted inside that window.
    started = time.monotonic()
    with pytest.raises(gpu_admission.GpuAdmissionTimeout):
        gpu_admission.acquire(gpu_admission.LOCAL_LLM, timeout_s=0.05)
    # It did not merely lose a race - it could not run at all until the ASR
    # side's decision had been committed.
    assert time.monotonic() - started >= 0.4

    thread.join(3.0)
    assert [permit.kind for permit in granted] == [gpu_admission.ASR]


def test_the_permit_and_the_claim_change_hands_at_the_same_instant():
    """The handover is one transition, not "register, then release".

    Two steps leave an instant where the permit is free and the resident
    resource is not yet what holds anyone off - and an admission check taken
    before the first step can be acted on after the second. Being exclusive
    about `_HOLDER` does not help: the gap is not in the permit, it is between
    the permit and the resource.
    """
    permit = gpu_admission.acquire(gpu_admission.LOCAL_LLM, description="llama-server")
    resource_id = _register_live(gpu_admission.LOCAL_LLM, pid=42)
    assert gpu_admission.holder() is not None  # the permit is what holds ASR off

    claim = gpu_admission.handover(permit, resource_id=resource_id, generation=1)

    # ...and now the claim is, with nothing observable in between.
    assert gpu_admission.holder() is None
    assert claim.kind == gpu_admission.LOCAL_LLM
    assert [entry["resource_id"] for entry in gpu_admission.claims()] == [resource_id]
    assert gpu_admission.blocked_for(gpu_admission.ASR) is True
    with pytest.raises(gpu_admission.GpuAdmissionTimeout):
        gpu_admission.acquire(gpu_admission.ASR, timeout_s=0.2)


def test_a_claim_ends_with_the_record_it_names():
    # No matching release to forget, and none to be lost: the registry already
    # owns that lifetime, watchdog and 重试清理 included, so a close that only
    # confirms much later cannot strand a claim.
    permit = gpu_admission.acquire(gpu_admission.LOCAL_LLM)
    resource_id = _register_live(gpu_admission.LOCAL_LLM, pid=42)
    gpu_admission.handover(permit, resource_id=resource_id, generation=1)

    assert resources.request_stop(resource_id, 1).released is True

    assert gpu_admission.claims() == []
    assert gpu_admission.blocked_for(gpu_admission.ASR) is False
    assert gpu_admission.acquire(gpu_admission.ASR, timeout_s=0.5).kind == gpu_admission.ASR


def test_a_claim_raised_for_one_generation_cannot_describe_its_replacement():
    # Same rule as the registry's: identity is `(resource_id, generation)`, so a
    # claim for a retired server says nothing about the one that took its place.
    permit = gpu_admission.acquire(gpu_admission.LOCAL_LLM)
    resource_id = _register_live(gpu_admission.LOCAL_LLM, pid=42)
    gpu_admission.handover(permit, resource_id=resource_id, generation=1)
    resources.request_stop(resource_id, 1)

    assert gpu_admission.claims() == []


def test_a_permit_that_is_no_longer_held_cannot_hand_over():
    permit = gpu_admission.acquire(gpu_admission.LOCAL_LLM)
    gpu_admission.release(permit)

    with pytest.raises(gpu_admission.GpuHandoverError):
        gpu_admission.handover(permit, resource_id="sample", generation=1)
    assert gpu_admission.claims() == []


def test_a_stale_release_never_frees_its_successors_permit():
    # Two stages of the same kind follow each other all the time, and a release
    # naming only "asr_worker" would hand away the permit the next one holds.
    first = gpu_admission.acquire(gpu_admission.ASR)
    gpu_admission.release(first)
    second = gpu_admission.acquire(gpu_admission.ASR)

    assert gpu_admission.release(first) is False
    assert gpu_admission.holder() is second
