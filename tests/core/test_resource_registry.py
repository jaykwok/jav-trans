"""The four things the resource lifecycle has to guarantee.

Every round of review found the same class of bug in a different subsystem's
private copy of this machinery. These pin the guarantees at the one place that
implements them now, so a future subsystem gets them by construction:

* a late cleanup cannot touch the resource that replaced its target,
* a cleanup never targets a live resource somebody is using,
* a stop that has finished but not committed cannot be claimed again,
* nothing gets past GPU admission while an owner has not confirmed it let go.
"""

from __future__ import annotations

import threading
import time

import pytest

from core import resources


class _FakeProcess:
    def __init__(self, pid: int = 100, immortal: bool = False) -> None:
        self.pid = pid
        self.alive = True
        self.stops = 0
        self._immortal = immortal

    def stop(self) -> str:
        self.stops += 1
        if not self._immortal:
            self.alive = False
        return "alive" if self.alive else "gone"


def _stopper(process, job_handle):
    return resources.StopOutcome(process_state=process.stop(), job_handle=job_handle)


@pytest.fixture(autouse=True)
def _quiet_watchdog(monkeypatch):
    # The backoff loop would race the assertions with its own retries; it is
    # exercised through the ASR path instead.
    monkeypatch.setattr(resources, "_start_watchdog", lambda: None)
    yield
    resources.forget_all()


def _register(generation: int, process, **kwargs):
    resources.register(
        resource_id="worker",
        generation=generation,
        kind="asr_worker",
        description="worker",
        process=process,
        stopper=_stopper,
        holds_gpu=True,
        **kwargs,
    )


def test_a_late_cleanup_cannot_reach_the_generation_that_replaced_it():
    first = _FakeProcess(pid=1)
    _register(1, first)
    assert resources.request_stop("worker", 1).released is True

    second = _FakeProcess(pid=2)
    _register(2, second)

    # The queued request from before finally runs.
    result = resources.request_stop("worker", 1)

    assert result.stopped is True  # nothing of generation 1 is here
    assert second.stops == 0  # ...and generation 2 was never touched
    assert second.alive is True
    assert resources.is_registered("worker", 2) is True


def test_scheduled_stop_blocks_admission_before_owner_enters_close():
    process = _FakeProcess()
    _register(1, process)
    resources.schedule_stop("worker", 1)
    assert process.stops == 0  # scheduling is a non-blocking state transition
    assert resources.gpu_blocked()
    assert resources.pending()[0]["generation"] == 1
    resources.schedule_stop("worker", 99)  # a stale/unknown identity does nothing
    assert resources.retry_pending()["released"] == 1
    assert process.stops == 1
    assert not resources.gpu_blocked()


def test_an_identity_is_never_handed_out_twice():
    # Owners used to name themselves, and each numbered its own generations from
    # zero: two instances created one after the other claimed the same identity
    # for their first process. Uniqueness belongs to the registry.
    first = resources.new_resource_id("llamacpp-server")
    second = resources.new_resource_id("llamacpp-server")
    assert first != second

    _register(1, _FakeProcess(pid=1))
    with pytest.raises(resources.ResourceIdentityError):
        # Overwriting drops the only reference to the process still filed here.
        _register(1, _FakeProcess(pid=2))
    assert resources.live()[0]["pid"] == 1


def test_a_committed_result_never_evicts_the_record_that_replaced_it():
    # The key is where a record is filed, not proof of which resource it is.
    hold = threading.Event()

    def slow_stopper(process, job_handle):
        hold.wait(3.0)
        return resources.StopOutcome(process_state="gone", job_handle=None)

    stale = _FakeProcess(pid=1)
    resources.register(
        resource_id="worker",
        generation=1,
        kind="asr_worker",
        description="worker",
        process=stale,
        stopper=slow_stopper,
        holds_gpu=True,
    )
    stopping = threading.Thread(target=resources.request_stop, args=("worker", 1))
    stopping.start()
    time.sleep(0.05)

    # While that stop is in flight, its record is dropped and the identity is
    # re-registered by hand (a caller that ignores ResourceIdentityError).
    resources.forget_all()
    live = _FakeProcess(pid=2)
    _register(1, live)

    hold.set()
    stopping.join(3.0)

    assert resources.is_registered("worker", 1) is True
    assert resources.live()[0]["pid"] == 2


def test_a_retry_never_targets_a_live_resource():
    live = _FakeProcess(pid=1)
    _register(1, live)
    stuck = _FakeProcess(pid=2, immortal=True)
    resources.register(
        resource_id="other-worker",
        generation=1,
        kind="asr_worker",
        description="worker",
        process=stuck,
        stopper=_stopper,
        holds_gpu=True,
        state="stopping",
        process_state="alive",
    )

    summary = resources.retry_pending()

    assert summary["retried"] == 1  # only the one already waiting for cleanup
    assert stuck.stops == 1
    assert live.stops == 0
    assert live.alive is True


def test_a_finished_stop_cannot_be_claimed_again_before_it_commits(monkeypatch):
    # Closing happens outside the lock (it blocks), so the claim is what stands
    # in for "this result is not in yet". The dangerous window is *after* the
    # close returns and before the result is written back: releasing the claim
    # there (in a `finally`, as it used to be) let a second retry take the record
    # and act on the handle that had just been closed.
    closes: list[int] = []
    finished = threading.Event()
    second: list[resources.StopResult] = []

    def stopper(process, job_handle):
        closes.append(job_handle)
        finished.set()
        return resources.StopOutcome(process_state="gone", job_handle=None)

    resources.register(
        resource_id="worker",
        generation=1,
        kind="asr_worker",
        description="worker",
        process=_FakeProcess(),
        job_handle=303,
        stopper=stopper,
        holds_gpu=True,
    )

    class _HookedLock:
        """Lets a competing retry in once, the first time the lock is released
        after the close returned - i.e. into whatever gap exists there."""

        def __init__(self, real):
            self._real = real
            self._fired = False

        def __enter__(self):
            return self._real.__enter__()

        def __exit__(self, *exc_info):
            released = self._real.__exit__(*exc_info)
            if finished.is_set() and not self._fired:
                self._fired = True
                other = threading.Thread(
                    target=lambda: second.append(resources.request_stop("worker", 1))
                )
                other.start()
                other.join(3.0)
            return released

    monkeypatch.setattr(resources, "_LOCK", _HookedLock(resources._LOCK))

    result = resources.request_stop("worker", 1)

    assert result.released is True
    assert closes == [303]  # the handle was handed out exactly once
    # There is no half-way state to catch: the record is either still claimed or
    # already committed and gone. What must never happen is a second claim on a
    # record whose handle has been closed but whose result is not written yet.
    assert second and second[0].detail in {
        "cleanup already in progress",
        "not registered",
    }
    assert resources.is_registered("worker", 1) is False


def test_two_threads_retrying_stop_a_resource_once():
    hold = threading.Event()
    calls: list[int] = []

    def slow_stopper(process, job_handle):
        calls.append(1)
        hold.wait(3.0)
        return resources.StopOutcome(process_state="gone", job_handle=None)

    resources.register(
        resource_id="worker",
        generation=1,
        kind="asr_worker",
        description="worker",
        process=_FakeProcess(immortal=True),
        stopper=slow_stopper,
        holds_gpu=True,
        state="stopping",
        process_state="alive",
    )

    first = threading.Thread(target=resources.retry_pending)
    first.start()
    deadline = time.monotonic() + 3.0
    while not calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert calls == [1]

    second = threading.Thread(target=resources.retry_pending)
    second.start()
    second.join(3.0)
    assert second.is_alive() is False  # it found the record claimed

    hold.set()
    first.join(3.0)
    assert calls == [1]


def test_admission_is_closed_until_every_gpu_owner_confirms():
    asr = _FakeProcess(pid=1, immortal=True)
    _register(1, asr, state="stopping", process_state="alive")
    resources.register(
        resource_id="llamacpp-server",
        generation=1,
        kind="local_server",
        description="llama-server",
        process=_FakeProcess(pid=2, immortal=True),
        stopper=_stopper,
        holds_gpu=True,
        state="stopping",
        process_state="alive",
    )

    assert resources.gpu_blocked() is True
    assert resources.snapshot()["state"] == "cleanup_failed"

    # The ASR worker confirming is not enough while the other owner has not:
    # admission is one question asked of everything holding the card, not each
    # subsystem refusing to start its own replacement.
    assert resources.request_stop("worker", 1).stopped is False
    asr.alive = False
    assert resources.request_stop("worker", 1).released is True
    assert resources.is_registered("worker", 1) is False
    assert resources.gpu_blocked() is True  # llama-server is still resident


def test_a_leaked_handle_does_not_hold_the_card():
    # The process inside that job is confirmed gone, so it holds no VRAM. It is
    # a handle to close, tracked and retried on its own - blocking every GPU
    # stage on it would be a cost nobody could justify.
    resources.register(
        resource_id="worker",
        generation=1,
        kind="asr_worker",
        description="worker",
        process=_FakeProcess(),
        job_handle=4242,
        stopper=lambda process, handle: resources.StopOutcome(
            process_state="gone", job_handle=handle, detail="handle would not close"
        ),
        holds_gpu=True,
    )

    result = resources.request_stop("worker", 1)

    assert result.stopped is True
    assert result.released is False  # the handle is still open
    assert resources.gpu_blocked() is False
    snapshot = resources.snapshot()
    assert snapshot["state"] == "cleaning"
    assert snapshot["handle_open"] is True


def test_a_stopper_that_raises_is_an_unknown_not_an_exit():
    def angry_stopper(process, job_handle):
        raise OSError("access denied")

    resources.register(
        resource_id="worker",
        generation=1,
        kind="asr_worker",
        description="worker",
        process=_FakeProcess(),
        stopper=angry_stopper,
        holds_gpu=True,
    )

    result = resources.request_stop("worker", 1)

    assert result.stopped is False
    assert result.process_state == "unknown"
    assert resources.gpu_blocked() is True  # unknown never frees the card
    assert resources.is_registered("worker", 1) is True


def test_confirm_gone_keeps_a_record_that_still_holds_a_handle():
    resources.register(
        resource_id="worker",
        generation=1,
        kind="asr_worker",
        description="worker",
        process=_FakeProcess(),
        job_handle=99,
        stopper=_stopper,
        holds_gpu=True,
    )

    assert resources.confirm_gone("worker", 1) is False
    assert resources.is_registered("worker", 1) is True
    assert resources.gpu_blocked() is False  # the process is gone
    assert resources.snapshot()["open_job_handles"] == 1
