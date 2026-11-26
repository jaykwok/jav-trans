"""A child we could not stop is a resource state, not a one-off error.

Refusing to start a replacement worker was only half of it: the refusal was
invisible, nothing retried it, and every later job failed against the same busy
GPU one by one. These pin the other half - the state is observable, retried on
its own bounded backoff, recoverable by hand, and released only when the child
is *confirmed* gone.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from core import resources
from pipeline import gpu_worker


class _FakeChild:
    def __init__(self, *, immortal=False, liveness_error=False, pid=4321):
        self.alive = True
        self.pid = pid
        self.exitcode = None
        self._immortal = immortal
        self._liveness_error = liveness_error

    def start(self):
        self.alive = True

    def is_alive(self):
        if self._liveness_error:
            raise OSError("handle is invalid")
        return self.alive

    def terminate(self):
        if not self._immortal:
            self.alive = False

    def kill(self):
        if not self._immortal:
            self.alive = False

    def join(self, timeout=None):
        return None


class _FakeConn:
    def __init__(self, messages=()):
        self._messages = list(messages)
        self.closed = False

    def poll(self, timeout=None):
        return bool(self._messages)

    def recv(self):
        return self._messages.pop(0)

    def send(self, message):
        return None

    def close(self):
        self.closed = True


class _FakeCtx:
    """Just enough multiprocessing to run the real `_start_worker`."""

    def __init__(self, process):
        self._process = process

    def Pipe(self, duplex=True):
        return _FakeConn([{"op": "ready"}]), _FakeConn()

    def Process(self, **kwargs):
        return self._process


def test_failed_cleanup_becomes_a_blocking_resource_state(monkeypatch):
    monkeypatch.setattr(gpu_worker, "close_kill_on_close_job", lambda _handle: True)
    # The watchdog would race the assertions with its own retry; the backoff is
    # exercised separately below.
    monkeypatch.setattr(resources, "_start_watchdog", lambda: None)
    client = gpu_worker._GpuWorkerClient()
    client._process = _FakeChild(immortal=True)

    assert client._kill_child() is False

    status = gpu_worker.asr_gpu_cleanup_status()
    assert gpu_worker.asr_gpu_blocked() is True
    assert status["state"] == "cleanup_failed"
    assert status["pid"] == 4321
    assert status["attempts"] == 1
    assert status["last_error"]
    assert status["next_retry_in_s"] > 0


def test_a_liveness_check_that_errors_is_unknown_not_exited(monkeypatch):
    # `is_alive()` raising used to read as "already gone", which is the one
    # answer that lets a caller start a second worker on the same card.
    monkeypatch.setattr(gpu_worker, "close_kill_on_close_job", lambda _handle: True)
    monkeypatch.setattr(resources, "_start_watchdog", lambda: None)
    client = gpu_worker._GpuWorkerClient()
    child = _FakeChild(liveness_error=True)
    client._process = child

    assert client._child_liveness(child) == "unknown"
    assert client._kill_child() is False
    assert client._process is child
    assert client.has_unreleased_child() is True
    assert gpu_worker.asr_gpu_blocked() is True
    with pytest.raises(gpu_worker.GpuWorkerError) as exc_info:
        client._start_worker()
    assert exc_info.value.kind == "cleanup_failed"


def test_a_handle_that_will_not_close_stays_with_its_own_record(monkeypatch):
    # A raw job handle is an integer. Clearing the attribute after a failed
    # CloseHandle leaks it, and because the job is kill-on-close, the tree
    # inside it then has no way of ever being reached again. The record it was
    # registered with keeps it - and that record never moves to another child.
    monkeypatch.setattr(gpu_worker, "close_kill_on_close_job", lambda _handle: False)
    client = gpu_worker._GpuWorkerClient()
    client._process = _FakeChild()
    client._job_handle = 987654

    assert client._kill_child() is True  # the child itself did exit
    kept = gpu_worker.unclosed_job_handles()
    assert len(kept) == 1
    assert kept[0]["resource_id"] == client._resource_id()
    assert kept[0]["process_state"] == "gone"  # holds no GPU, only a handle
    assert gpu_worker.asr_gpu_blocked() is False
    assert gpu_worker.asr_gpu_cleanup_status()["handle_open"] is True

    closed: list[int] = []
    monkeypatch.setattr(
        gpu_worker,
        "close_kill_on_close_job",
        lambda handle: (closed.append(handle), True)[1],
    )
    assert gpu_worker.retry_asr_gpu_cleanup()["handle_open"] is False
    assert closed == [987654]  # the integer itself was never dropped


def test_confirmed_exit_releases_the_gpu_and_unblocks_dispatch(monkeypatch):
    monkeypatch.setattr(gpu_worker, "close_kill_on_close_job", lambda _handle: True)
    monkeypatch.setattr(resources, "_start_watchdog", lambda: None)
    client = gpu_worker._GpuWorkerClient()
    child = _FakeChild(immortal=True)
    client._process = child

    assert client._kill_child() is False
    assert gpu_worker.asr_gpu_blocked() is True

    # The child finally exits - a manual 重试清理 must notice and restore the queue.
    child._immortal = False
    child.alive = False
    status = gpu_worker.retry_asr_gpu_cleanup()

    assert status["state"] == "released"
    assert status["blocked"] is False
    assert gpu_worker.asr_gpu_blocked() is False
    assert client._process is None


def test_shutdown_keeps_a_worker_it_could_not_stop(monkeypatch):
    """Dropping the reference on a failed close hands the GPU back by accident.

    `shutdown_global_worker()` used to clear `_GLOBAL_WORKER` regardless of the
    outcome. The undead child stayed on the card, the next client had no child
    of its own, and *its* empty cleanup then cleared the blocked state - so the
    queue resumed and a second GPU process started beside the first. This is not
    only the exit path: loading the local translation model calls it too.
    """
    monkeypatch.setattr(gpu_worker, "close_kill_on_close_job", lambda _handle: True)
    monkeypatch.setattr(resources, "_start_watchdog", lambda: None)
    client = gpu_worker._GpuWorkerClient()
    child = _FakeChild(immortal=True)
    client._process = child
    client._child_generation = 1
    monkeypatch.setattr(gpu_worker, "_GLOBAL_WORKER", client)

    assert gpu_worker.shutdown_global_worker() is False
    assert gpu_worker._GLOBAL_WORKER is client  # still ours to retry
    assert gpu_worker.asr_gpu_blocked() is True

    # And a client that owns nothing cannot release someone else's child.
    newcomer = gpu_worker._GpuWorkerClient()
    assert newcomer._kill_child() is True  # it has nothing of its own to stop
    assert gpu_worker.asr_gpu_blocked() is True
    assert gpu_worker.asr_gpu_cleanup_status()["pid"] == 4321


def test_a_stuck_handle_survives_its_client_being_replaced(monkeypatch):
    # "Does not block the GPU" and "no longer anyone's responsibility" are
    # different claims. The child exiting settles the first; only closing the
    # handle settles the second - and the second must not depend on the client,
    # which is free to be replaced at any time.
    monkeypatch.setattr(gpu_worker, "close_kill_on_close_job", lambda _handle: False)
    client = gpu_worker._GpuWorkerClient()
    client._process = _FakeChild()
    client._job_handle = 987654
    monkeypatch.setattr(gpu_worker, "_GLOBAL_WORKER", client)

    assert client._kill_child() is True  # the child did exit: GPU is free
    assert gpu_worker.asr_gpu_blocked() is False
    assert gpu_worker.asr_gpu_cleanup_status()["handle_open"] is True
    # The client has nothing left to hold, so it may be replaced...
    assert gpu_worker._get_global_worker() is not client
    # ...and the handle is still reachable afterwards.
    assert gpu_worker.asr_gpu_cleanup_status()["open_job_handles"] == 1

    monkeypatch.setattr(gpu_worker, "close_kill_on_close_job", lambda _handle: True)
    assert gpu_worker.retry_asr_gpu_cleanup()["handle_open"] is False


@pytest.mark.skipif(os.name != "nt", reason="Job Objects are Windows-only")
def test_retrying_an_old_handle_never_closes_the_live_workers_job(monkeypatch):
    """The retry entry point must not follow a client that has moved on.

    A client that could not close its handle used to stay registered as the
    holder, then bind a *new* job for its next child. 重试清理 walked that
    registration and closed whatever the client held at that moment - the live
    worker's kill-on-close job, taking its process tree with it. Records are per
    child now, and a live one is never a cleanup target.
    """
    import asr.local_backend as local_backend

    closed: list[int] = []
    monkeypatch.setattr(
        gpu_worker,
        "close_kill_on_close_job",
        lambda handle: (closed.append(handle), False)[1],
    )
    monkeypatch.setattr(local_backend, "_create_kill_on_close_job_object", lambda: 202)
    monkeypatch.setattr(
        local_backend, "_assign_process_to_job_object", lambda job, process: None
    )

    client = gpu_worker._GpuWorkerClient()
    client._process = _FakeChild()  # exits when asked
    client._job_handle = 101  # ...but this handle will not close
    new_child = _FakeChild(pid=555)
    client._ctx = _FakeCtx(new_child)

    client._start_worker()

    assert client._job_handle == 202  # the live worker's job
    kept = gpu_worker.unclosed_job_handles()
    assert [record["generation"] for record in kept] == [0]  # only the old child

    closed.clear()
    monkeypatch.setattr(
        gpu_worker,
        "close_kill_on_close_job",
        lambda handle: (closed.append(handle), True)[1],
    )
    gpu_worker.retry_asr_gpu_cleanup()

    assert closed == [101]  # never 202
    assert client._job_handle == 202
    assert new_child.alive is True


def test_a_stale_cleanup_request_cannot_reach_the_next_generation(monkeypatch):
    """Identity is `(resource_id, generation)` and it never moves.

    The watchdog and the manual button used to capture an owner and call into a
    cleanup that re-read "the current child" when it finally ran. Between the
    two, the stuck child can exit and a second one start - and the retry then
    terminated a worker in the middle of somebody's job.
    """
    monkeypatch.setattr(gpu_worker, "close_kill_on_close_job", lambda _handle: True)
    monkeypatch.setattr(resources, "_start_watchdog", lambda: None)
    client = gpu_worker._GpuWorkerClient()
    first = _FakeChild(immortal=True)
    client._process = first
    client._child_generation = 1

    assert client._kill_child() is False
    # What a queued retry carries: the identity it was raised for.
    stale = (client._resource_id(), 1)
    assert gpu_worker.asr_gpu_blocked() is True

    # The first child finally exits and the client takes a second one.
    first._immortal = False
    first.alive = False
    second = _FakeChild(pid=777)
    client._ctx = _FakeCtx(second)
    client._start_worker()
    assert client._process is second
    assert client._child_generation == 2

    # Now the queued retry runs.
    result = resources.request_stop(*stale)
    assert result.stopped is True  # nothing of that generation is here
    assert second.alive is True  # ...and this one was never touched
    assert client._process is second
    assert client.is_alive() is True


def test_two_concurrent_retries_close_a_handle_once(monkeypatch):
    # The pending list is copied under the lock and closed outside it (closing
    # can block), so without an exclusive claim both callers close the same
    # integer - and the second close lands on whatever it names by then.
    closed: list[int] = []
    hold = threading.Event()

    def slow_close(handle):
        closed.append(handle)
        hold.wait(3.0)
        return True

    monkeypatch.setattr(resources, "_start_watchdog", lambda: None)
    monkeypatch.setattr(gpu_worker, "close_kill_on_close_job", lambda _handle: False)
    client = gpu_worker._GpuWorkerClient()
    client._process = _FakeChild()
    client._job_handle = 303
    client._ensure_registered_locked(client._process, 0)
    resources.request_stop(client._resource_id(), 0)  # child exits, handle stuck
    assert gpu_worker.asr_gpu_cleanup_status()["open_job_handles"] == 1
    monkeypatch.setattr(gpu_worker, "close_kill_on_close_job", slow_close)

    first = threading.Thread(target=gpu_worker.retry_asr_gpu_cleanup)
    first.start()
    deadline = time.monotonic() + 3.0
    while not closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert closed == [303]  # the first retry is inside CloseHandle right now

    second = threading.Thread(target=gpu_worker.retry_asr_gpu_cleanup)
    second.start()
    second.join(3.0)
    assert second.is_alive() is False  # it found the record claimed and moved on

    hold.set()
    first.join(3.0)
    assert closed == [303]
    assert gpu_worker.unclosed_job_handles() == []


def test_cleanup_retry_backoff_is_bounded(monkeypatch):
    monkeypatch.setattr(gpu_worker, "close_kill_on_close_job", lambda _handle: True)
    monkeypatch.setattr(resources, "_start_watchdog", lambda: None)
    client = gpu_worker._GpuWorkerClient()
    client._process = _FakeChild(immortal=True)

    delays = []
    for _ in range(8):
        client._kill_child()
        delays.append(gpu_worker.asr_gpu_cleanup_status()["next_retry_in_s"])

    assert delays[0] < delays[1] < delays[2]  # backs off...
    assert max(delays) <= resources.RETRY_MAX_S  # ...but never further
    assert gpu_worker.asr_gpu_cleanup_status()["attempts"] == 8
