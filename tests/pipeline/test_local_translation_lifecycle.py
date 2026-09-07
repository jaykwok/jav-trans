from __future__ import annotations

import asyncio
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

import main
from core import resources
from llm import backends as llm_backends
from llm.errors import TranslationCancelledError


@pytest.fixture(autouse=True)
def output_identity(monkeypatch, tmp_path):
    # Backend lifetime tests stub computation; isolate output ownership too.
    lease = main.artifact_store.claim_output(
        tmp_path / "sample-a.srt", project_root=tmp_path,
        job_id="sample-job", run_id="sample-run",
    )
    monkeypatch.setattr(main, "_output_lease", lambda *args, **kwargs: lease)


@pytest.fixture
def clean_local_backend_slot(monkeypatch):
    """Empty the shared llamacpp slot afterwards, however the test ended.

    A test that starts fake servers can leave a stuck instance in it, and
    `monkeypatch` cannot undo a key it never saw - the factory puts it there
    mid-test. Left behind, the next task's teardown closes it, that close fails,
    and the resulting pending record blocks GPU admission for every later test.

    In the finalizer rather than at the end of the test body: a failing assertion
    would skip the cleanup and pollute the rest of the run - which is exactly how
    this was found. Depends on `monkeypatch` so that this runs first and any
    `setitem` restore still gets the last word.
    """
    yield
    llm_backends._BACKEND_INSTANCES.pop("llamacpp", None)


def _translating_artifacts() -> SimpleNamespace:
    """Stands in for a run that will actually send something to a backend.

    The entry point now decides whether to lease at all, so "there is something
    to translate" has to be stated rather than assumed: a bare object() would
    describe a Japanese-only run, which takes no lease and would make every
    lease assertion below vacuously true.
    """
    return SimpleNamespace(segments=[{"start": 0.0, "end": 1.0, "text": "日本語"}], job_id="sample-job", srt_path="sample-a.srt")


def _translating_ctx() -> SimpleNamespace:
    # `subtitle_mode` as well as `skip_translation`: the entry point builds an
    # ExecutionPlan, which states how the output is spelled alongside whether
    # anything is translated at all.
    return SimpleNamespace(skip_translation=False, subtitle_mode="zh", job_id="sample-job")


class _FakeBackend:
    def __init__(self, closed: list[str], label: str) -> None:
        self._closed = closed
        self._label = label

    def close(self) -> bool:
        self._closed.append(self._label)
        return True


@pytest.mark.parametrize("fails", [False, True])
def test_translation_task_always_closes_local_backend(monkeypatch, fails):
    closed_backends: list[str] = []
    closed_loggers: list[object] = []
    artifacts = _translating_artifacts()

    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setattr(main, "_reopen_snapshot_run_logger", lambda _artifacts: None)
    monkeypatch.setattr(
        main,
        "_close_artifacts_logger",
        lambda value: closed_loggers.append(value),
    )
    monkeypatch.setitem(
        llm_backends._BACKEND_INSTANCES,
        "llamacpp",
        _FakeBackend(closed_backends, "llamacpp"),
    )

    def run_impl(*_args, **_kwargs):
        if fails:
            raise RuntimeError("translation failed")
        return ["done.srt"]

    monkeypatch.setattr(main, "_run_translation_and_write_impl", run_impl)

    if fails:
        with pytest.raises(RuntimeError, match="translation failed"):
            main.run_translation_and_write("video.mp4", artifacts, ctx=_translating_ctx())
    else:
        assert main.run_translation_and_write(
            "video.mp4", artifacts, ctx=_translating_ctx()
        ) == ["done.srt"]

    assert closed_backends == ["llamacpp"]
    assert closed_loggers == [artifacts]
    assert llm_backends.backend_lease_count("llamacpp") == 0


def test_finished_task_does_not_close_a_backend_another_task_is_using(monkeypatch):
    # Two local translations run in parallel on the same process-wide instance.
    # The first to finish used to reset_backend("llamacpp") unconditionally,
    # closing the llama-server the second one was still translating through.
    closed_backends: list[str] = []
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setattr(main, "_reopen_snapshot_run_logger", lambda _artifacts: None)
    monkeypatch.setattr(main, "_close_artifacts_logger", lambda _value: None)
    monkeypatch.setitem(
        llm_backends._BACKEND_INSTANCES,
        "llamacpp",
        _FakeBackend(closed_backends, "llamacpp"),
    )

    with llm_backends.backend_lease("llamacpp"):  # the job still translating
        monkeypatch.setattr(
            main,
            "_run_translation_and_write_impl",
            lambda *_args, **_kwargs: ["done.srt"],
        )
        main.run_translation_and_write(
            "video.mp4", _translating_artifacts(), ctx=_translating_ctx()
        )
        assert closed_backends == []
        assert llm_backends.backend_lease_count("llamacpp") == 1

    assert closed_backends == ["llamacpp"]


def test_cancelling_a_local_request_reports_drain_instead_of_killing_the_server():
    """HTTP abort is not proof that the model stopped generating.

    The server is shared with whatever other job holds a lease, so a cancel must
    not kill it. What it may do - and must do - is stop claiming the card is
    free, because the slot can still be busy and the VRAM is still gone.
    """
    from llm.backends.llamacpp_server import LlamaCppServerBackend
    from llm.errors import TranslationCancelledError

    backend = LlamaCppServerBackend()
    backend._port = 8080
    backend._proc = SimpleNamespace(poll=lambda: None, kill=lambda: None)

    class _HangingCompletions:
        # Async on purpose: it stands in for a socket read, which is what the
        # cancel has to reach.
        async def create(self, **_kwargs):
            await asyncio.sleep(30)

    class _HangingClient:
        class chat:  # noqa: N801 - mirrors the openai client shape
            completions = _HangingCompletions()

    backend._make_async_client = lambda _port: _HangingClient()
    assert backend.drain_status()["draining"] is False

    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    with pytest.raises(TranslationCancelledError):
        backend._request_completion({"model": "m"}, cancel)

    assert backend.drain_status()["draining"] is True
    # The shared process is untouched: cancelling A cannot end B's server.
    assert backend._proc is not None


def test_another_request_finishing_does_not_prove_the_cancelled_one_stopped():
    # llama-server runs several slots in parallel and the responses carry no id
    # to correlate them by, so B succeeding says nothing about A. Clearing the
    # unknown on B's success was a guess dressed up as a confirmation.
    from llm.backends.llamacpp_server import LlamaCppServerBackend

    backend = LlamaCppServerBackend()
    backend._port = 8080
    backend._proc = SimpleNamespace(poll=lambda: None, kill=lambda: None)
    backend._draining_since = 1000.0

    class _AnswerNow:
        async def create(self, **_kwargs):
            return SimpleNamespace(choices=[], usage=None)

    class _Client:
        class chat:  # noqa: N801 - mirrors the openai client shape
            completions = _AnswerNow()

    backend._make_async_client = lambda _port: _Client()
    backend._request_completion({"model": "m"}, None)

    assert backend.drain_status()["draining"] is True

    # Only the process going away settles it.
    backend._proc = SimpleNamespace(poll=lambda: 0, kill=lambda: None)
    assert backend.drain_status()["draining"] is False


def test_status_never_waits_for_the_model_lifecycle_lock():
    """The page polls this every 5s on the event loop.

    `_lock` is held across model load and health checks - up to the 300s startup
    timeout - so taking it here stalled the whole web app, cancel endpoint
    included, for as long as a load took.
    """
    from llm.backends.llamacpp_server import LlamaCppServerBackend

    backend = LlamaCppServerBackend()
    backend._proc = SimpleNamespace(poll=lambda: None, kill=lambda: None)
    holding = threading.Event()
    release = threading.Event()

    def hold_the_lifecycle_lock():
        with backend._lock:
            holding.set()
            release.wait(5.0)

    holder = threading.Thread(target=hold_the_lifecycle_lock, daemon=True)
    holder.start()
    try:
        assert holding.wait(2.0)
        started = time.perf_counter()
        backend.drain_status()
        elapsed = time.perf_counter() - started
    finally:
        release.set()
        holder.join(5.0)

    assert elapsed < 0.1


def test_loading_the_local_model_refuses_while_the_asr_child_is_stuck(monkeypatch):
    # Freeing the ASR worker's VRAM is a precondition for loading a multi-GB
    # GGUF, not a nice-to-have: swallowing its failure is how a second GPU
    # process gets started next to one that would not die.
    from llm.backends import llamacpp_server
    from pipeline import gpu_worker

    monkeypatch.setattr(gpu_worker, "close_kill_on_close_job", lambda _handle: True)
    from core import resources

    monkeypatch.setattr(resources, "_start_watchdog", lambda: None)

    class _Immortal:
        pid = 4321
        exitcode = None

        def is_alive(self):
            return True

        def terminate(self):
            return None

        kill = terminate

        def join(self, timeout=None):
            return None

    client = gpu_worker._GpuWorkerClient()
    client._process = _Immortal()
    monkeypatch.setattr(gpu_worker, "_GLOBAL_WORKER", client)

    with pytest.raises(RuntimeError, match="拒绝启动本地翻译服务器"):
        llamacpp_server._release_asr_worker_vram()


class _StubbornServer:
    """A llama-server that ignores kill() until `dead` is set."""

    def __init__(self, pid: int = 9911) -> None:
        self.pid = pid
        self.dead = False
        self.kills = 0

    def poll(self):
        return 0 if self.dead else None

    def kill(self):
        self.kills += 1

    def wait(self, timeout=None):
        if self.dead:
            return 0
        raise subprocess.TimeoutExpired("llama-server", timeout or 0)


class _ObedientServer(_StubbornServer):
    def kill(self):
        self.kills += 1
        self.dead = True


class _FakeOsBoundary:
    """Everything `_ensure_server` touches outside this process, and nothing else.

    The point is to drive the *real* startup path - the one that allocates the
    identity, registers the resource and binds the job handle - because that is
    where the identity bugs live. Registering records by hand in a test is
    exactly how the last one stayed hidden.
    """

    def __init__(self, monkeypatch, tmp_path, *, server=_ObedientServer, closes=True):
        from llm.backends import llamacpp_server

        self.procs: list = []
        self.closed_handles: list = []
        self.factory = server
        self._handle_seq = 700

        class _Subprocess:
            TimeoutExpired = subprocess.TimeoutExpired
            STDOUT = subprocess.STDOUT

            @staticmethod
            def Popen(command, **_kwargs):  # noqa: N802 - mirrors subprocess
                proc = server(pid=4000 + len(self.procs))
                self.procs.append(proc)
                return proc

        def _bind(proc):
            self._handle_seq += 1
            return self._handle_seq

        def _close(handle):
            self.closed_handles.append(handle)
            return closes

        monkeypatch.setattr(llamacpp_server, "subprocess", _Subprocess)
        monkeypatch.setattr(llamacpp_server, "_bind_kill_on_close_job", _bind)
        monkeypatch.setattr(llamacpp_server, "close_kill_on_close_job", _close)
        monkeypatch.setattr(
            llamacpp_server, "resolve_server_executable", lambda: "llama-server.exe"
        )
        monkeypatch.setattr(
            llamacpp_server, "resolve_gguf_model_path", lambda **_kwargs: "model.gguf"
        )
        monkeypatch.setattr(llamacpp_server, "_pick_free_port", lambda: 8080)
        monkeypatch.setattr(llamacpp_server, "_release_asr_worker_vram", lambda: None)
        monkeypatch.setattr(llamacpp_server, "server_environment", dict)
        monkeypatch.setattr(llamacpp_server, "probe_compute_devices", lambda *_: ["gpu0"])
        monkeypatch.setattr(
            llamacpp_server.LlamaCppServerBackend,
            "_server_log_path",
            lambda _self: tmp_path / "llamacpp_server.log",
        )
        monkeypatch.setattr(
            llamacpp_server.LlamaCppServerBackend, "_health_ok", lambda _self, _port: True
        )
        monkeypatch.setattr(resources, "_start_watchdog", lambda: None)


def test_two_backend_instances_never_claim_the_same_server_identity(monkeypatch, tmp_path):
    """The identity has to be unique across *instances*, not just within one.

    `resource_id` was the constant `"llamacpp-server"` and every new instance
    started its generation at 0, so instance B's first server had precisely the
    identity instance A's first server had. Registering it overwrote A's pending
    record, and A's queued cleanup - now pointing at B's process - killed the
    server B was translating through.
    """
    from llm.backends.llamacpp_server import LlamaCppServerBackend

    # The handle refuses to close: that is what leaves a pending record behind
    # after the process itself is gone.
    os_boundary = _FakeOsBoundary(monkeypatch, tmp_path, closes=False)

    first = LlamaCppServerBackend()
    first._ensure_server()
    first_identity = (first._resource_id(), first._server_generation)
    assert first.close() is True  # the process confirmed; its handle did not
    assert resources.is_registered(*first_identity) is True

    second = LlamaCppServerBackend()
    second._ensure_server()
    second_identity = (second._resource_id(), second._server_generation)

    assert second_identity != first_identity
    assert resources.is_registered(*first_identity) is True  # not overwritten

    # The cleanup queued for the first instance finally runs.
    resources.request_stop(*first_identity)

    running = os_boundary.procs[1]
    assert second._proc is running
    assert running.kills == 0
    assert running.poll() is None
    assert resources.is_registered(*second_identity) is True


class TestWaitingOutAModelLoad:
    """A model load is minutes, and `_lock` used to be held across all of it.

    Everything that touches the instance queued behind it: a sibling task on an
    uncancellable lock wait - the same shape as the ASR-lock bug one layer up -
    and 关闭, which is what the page calls when the user changes the setting.
    The load now happens outside the lock, and "a start is in progress" is
    state a waiter can watch rather than a lock it disappears into.
    """

    @staticmethod
    def _loading_backend(monkeypatch, tmp_path):
        from llm.backends import llamacpp_server

        _FakeOsBoundary(monkeypatch, tmp_path)
        ready = threading.Event()
        monkeypatch.setattr(
            llamacpp_server.LlamaCppServerBackend,
            "_health_ok",
            lambda _self, _port: ready.is_set(),
        )
        return llamacpp_server.LlamaCppServerBackend(), ready

    def test_a_second_task_can_be_cancelled_while_it_waits(self, monkeypatch, tmp_path):
        backend, ready = self._loading_backend(monkeypatch, tmp_path)
        cancel = threading.Event()
        loader_done = threading.Event()
        waiter: list = []

        def load():
            try:
                backend._ensure_server()
            finally:
                loader_done.set()

        def arrive_during_the_load():
            try:
                backend._ensure_server(cancel_event=cancel)
                waiter.append("started")
            except BaseException as exc:
                waiter.append(exc)

        loader = threading.Thread(target=load, daemon=True)
        loader.start()
        try:
            second = threading.Thread(target=arrive_during_the_load, daemon=True)
            second.start()
            time.sleep(0.2)  # it is waiting out the load by now
            assert waiter == []

            cancel.set()
            second.join(3.0)

            assert len(waiter) == 1
            assert isinstance(waiter[0], TranslationCancelledError)
        finally:
            ready.set()
            assert loader_done.wait(5.0)
            loader.join(5.0)
            backend.close()

    def test_a_sibling_can_be_cancelled_while_the_model_is_resolved(
        self, monkeypatch, tmp_path
    ):
        """Resolving the GGUF is a 4.6GB download on a first run.

        Under `_lock` it was worse than the load itself: a sibling task did not
        even reach the condition it was supposed to wait on - it blocked on the
        lock, where its cancel event could not reach it, until the download
        finished.
        """
        from llm.backends import llamacpp_server

        _FakeOsBoundary(monkeypatch, tmp_path)
        resolving = threading.Event()
        release = threading.Event()

        def resolve(**_kwargs):
            resolving.set()
            assert release.wait(5.0)
            return "model.gguf"

        monkeypatch.setattr(llamacpp_server, "resolve_gguf_model_path", resolve)
        backend = llamacpp_server.LlamaCppServerBackend()
        cancel = threading.Event()
        waiter: list = []
        starter_done = threading.Event()

        def start():
            try:
                backend._ensure_server()
            finally:
                starter_done.set()

        def arrive_during_the_download():
            try:
                backend._ensure_server(cancel_event=cancel)
                waiter.append("started")
            except BaseException as exc:
                waiter.append(exc)

        starter = threading.Thread(target=start, daemon=True)
        starter.start()
        try:
            assert resolving.wait(3.0)
            second = threading.Thread(target=arrive_during_the_download, daemon=True)
            second.start()
            time.sleep(0.2)
            assert waiter == []

            cancel.set()
            second.join(3.0)

            assert len(waiter) == 1
            assert isinstance(waiter[0], TranslationCancelledError)
        finally:
            release.set()
            assert starter_done.wait(5.0)
            starter.join(5.0)
            backend.close()

    def test_the_starter_stops_waiting_on_a_download_that_will_not_stop(
        self, monkeypatch, tmp_path
    ):
        # The progress hook asks huggingface_hub to abort, but what it does with
        # that is not a contract we own. The wait is cancellable either way.
        from llm.backends import llamacpp_server

        _FakeOsBoundary(monkeypatch, tmp_path)
        resolving = threading.Event()
        release = threading.Event()

        def never_returns(**_kwargs):
            resolving.set()
            assert release.wait(10.0)
            return "model.gguf"

        monkeypatch.setattr(llamacpp_server, "resolve_gguf_model_path", never_returns)
        backend = llamacpp_server.LlamaCppServerBackend()
        cancel = threading.Event()
        outcome: list = []

        def start():
            try:
                backend._ensure_server(cancel_event=cancel)
                outcome.append("started")
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=start, daemon=True)
        thread.start()
        try:
            assert resolving.wait(3.0)
            cancel.set()
            thread.join(3.0)

            assert not thread.is_alive()
            assert isinstance(outcome[0], TranslationCancelledError)
        finally:
            release.set()

    def test_the_download_reports_under_the_run_that_asked_for_it(
        self, monkeypatch, tmp_path
    ):
        """A new thread inherits no run.

        `core.events`'s job_id is thread-local, so the resolver thread starts
        with none, and hf_progress's module-level fallback is *shared* - it is
        whatever job most recently started an ASR download. Unattributed
        progress is merely lost; progress stamped with a concurrent job's id
        passes every check the receiving end makes and shows a translation
        model's download inside that other job.
        """
        from core import events
        from llm.backends import llamacpp_server
        from utils import hf_progress

        _FakeOsBoundary(monkeypatch, tmp_path)
        monkeypatch.setattr(events, "_thread_local", threading.local())
        events.set_current_run("sample-a", "run-a")
        events._thread_local.video = "sample-a.mp4"
        # Another job is downloading an ASR model right now, and that is what
        # the shared fallback says.
        monkeypatch.setattr(hf_progress, "_override_job_id", "sample-b")
        monkeypatch.setattr(hf_progress, "_override_run_id", "run-b")
        captured: list[dict] = []
        monkeypatch.setattr(events, "emit", lambda event: captured.append(event))

        def resolve(*, cancel_event=None):
            with hf_progress.tqdm_class(cancel_event)(
                total=100, initial=0, unit="B", unit_scale=True,
                desc="sample-model.gguf", name="huggingface_hub.http_get",
                disable=True,
            ) as bar:
                bar.update(100)
            return "model.gguf"

        monkeypatch.setattr(llamacpp_server, "resolve_gguf_model_path", resolve)
        backend = llamacpp_server.LlamaCppServerBackend()

        assert backend._resolve_model_path(threading.Event()) == "model.gguf"

        assert captured
        assert {
            (event["job_id"], event["run_id"], event["video"]) for event in captured
        } == {("sample-a", "run-a", "sample-a.mp4")}

    def test_closing_does_not_queue_behind_the_load(self, monkeypatch, tmp_path):
        backend, ready = self._loading_backend(monkeypatch, tmp_path)
        outcome: list = []
        loader_done = threading.Event()

        def load():
            try:
                backend._ensure_server()
                outcome.append("started")
            except BaseException as exc:
                outcome.append(exc)
            finally:
                loader_done.set()

        loader = threading.Thread(target=load, daemon=True)
        loader.start()
        try:
            time.sleep(0.2)  # inside the load
            started = time.perf_counter()
            stopped = backend.close()
            elapsed = time.perf_counter() - started

            assert stopped is True
            assert elapsed < 1.0
            # And the load it interrupted says so, rather than reporting a
            # startup failure of a process someone else stopped.
            assert loader_done.wait(5.0)
            assert isinstance(outcome[0], RuntimeError)
            assert "被关闭或替换" in str(outcome[0])
        finally:
            ready.set()
            loader.join(5.0)


def test_a_local_server_that_will_not_die_is_not_reported_as_stopped():
    """A swallowed `TimeoutExpired` made a failed stop look like a clean one.

    `_teardown_locked` cleared `_proc` up front and ignored the wait timeout, so
    the caller could not tell the two apart: the instance left the registry, the
    draining flag was cleared as if the process had gone, and the next
    translation loaded a second multi-GB model beside the first.
    """
    from llm.backends.llamacpp_server import LlamaCppServerBackend

    backend = LlamaCppServerBackend()
    server = _StubbornServer()
    backend._proc = server
    backend._port = 8080
    backend._server_generation = 1
    backend._draining_since = 1000.0

    assert backend.close() is False
    assert backend._proc is server  # still ours to retry
    assert server.kills >= 1
    status = backend.cleanup_status()
    assert status["stuck"] is True
    assert status["pid"] == 9911
    assert status["last_error"]
    # The unknown is not settled by a stop that did not confirm.
    assert backend.drain_status()["draining"] is True

    # ...and nothing may start beside it.
    with pytest.raises(RuntimeError, match="拒绝启动新的本地翻译服务器"):
        backend._ensure_server()

    # The process finally exits: a retry notices and everything clears. The
    # retry goes through the one cleanup entry point, naming the record.
    server.dead = True
    assert resources.request_stop(
        backend._resource_id(), status["generation"]
    ).released is True
    assert backend._proc is None
    assert backend.cleanup_status()["stuck"] is False
    assert backend.drain_status()["draining"] is False


def test_a_backend_that_could_not_stop_keeps_its_place_in_the_registry(monkeypatch):
    # Forgetting the instance strands the process: nothing owns it, nothing can
    # retry it, and the next get_backend() builds a fresh instance that starts a
    # second server next to the one still holding the VRAM.
    from llm.backends.llamacpp_server import LlamaCppServerBackend

    backend = LlamaCppServerBackend()
    server = _StubbornServer()
    backend._proc = server
    backend._server_generation = 1
    monkeypatch.setitem(llm_backends._BACKEND_INSTANCES, "llamacpp", backend)

    with llm_backends.backend_lease("llamacpp"):
        pass  # the last user leaves - this is where the close happens

    assert server.kills >= 1
    assert llm_backends._BACKEND_INSTANCES.get("llamacpp") is backend
    assert llm_backends.get_backend("llamacpp") is backend  # no second instance
    assert llm_backends.local_backend_cleanup_status()["stuck"] is True

    server.dead = True
    assert llm_backends.retry_local_backend_cleanup() is True
    assert "llamacpp" not in llm_backends._BACKEND_INSTANCES


def test_retry_cleanup_never_stops_a_server_another_job_is_using(monkeypatch):
    """重试清理 is "retry a recorded stuck resource", not "stop what is running".

    It used to call the backend's cleanup unconditionally, and only *afterwards*
    look at the leases - so a user pressing the button (for a drain notice, or
    for an unrelated media orphan) shut down the healthy shared server the other
    job was still translating through.
    """
    from llm.backends.llamacpp_server import LlamaCppServerBackend

    backend = LlamaCppServerBackend()
    healthy = _StubbornServer(pid=4242)
    backend._proc = healthy
    backend._port = 8080
    backend._server_generation = 1
    resources.register(
        resource_id=backend._resource_id(),
        generation=1,
        kind="local_server",
        description="llama-server",
        process=healthy,
        stopper=backend._stop_server,
        holds_gpu=True,
    )
    monkeypatch.setitem(llm_backends._BACKEND_INSTANCES, "llamacpp", backend)

    with llm_backends.backend_lease("llamacpp"):  # job B is still translating
        assert llm_backends.backend_lease_count("llamacpp") == 1
        assert llm_backends.retry_local_backend_cleanup() is True
        assert healthy.kills == 0  # nothing was stopped
        assert backend._proc is healthy
        assert backend._port == 8080
        assert llm_backends.backend_lease_count("llamacpp") == 1


def test_a_late_retry_cannot_stop_the_server_that_replaced_the_stuck_one():
    # Reading a status and acting on it are two moments; a generation can change
    # in between, which is why the check happens again inside the lifecycle lock.
    from llm.backends.llamacpp_server import LlamaCppServerBackend

    backend = LlamaCppServerBackend()
    first = _StubbornServer(pid=1111)
    backend._proc = first
    backend._server_generation = 1

    assert backend.close() is False
    stale = (backend._resource_id(), backend.cleanup_status()["generation"])
    assert stale[1] == 1

    # The stuck one finally exits and a second server takes its place.
    first.dead = True
    assert resources.request_stop(*stale).released is True
    second = _StubbornServer(pid=2222)
    backend._proc = second
    backend._server_generation = 2
    backend._port = 8080
    # Registered exactly as `_ensure_server` does, so the stale request has a
    # live record of the same resource id that it could wrongly reach.
    resources.register(
        resource_id=backend._resource_id(),
        generation=2,
        kind="local_server",
        description="llama-server",
        process=second,
        stopper=backend._stop_server,
        holds_gpu=True,
    )

    # The queued retry from before now runs.
    assert resources.request_stop(*stale).stopped is True
    assert second.kills == 0
    assert backend._proc is second
    assert resources.is_registered(backend._resource_id(), 2) is True


def test_a_stuck_local_server_holds_the_gpu_against_asr_too(monkeypatch):
    """One admission check, asked of every owner of the card.

    Each subsystem refused to start *its own* replacement, but nothing stopped
    the other one: a llama-server that could not be killed still had the model
    resident, and the ASR queue - which only ever asked `asr_gpu_blocked()` -
    started a worker straight into it.
    """
    from llm.backends import llamacpp_server
    from llm.backends.llamacpp_server import LlamaCppServerBackend
    from web import pipeline_manager as pm

    monkeypatch.setattr(resources, "_start_watchdog", lambda: None)
    backend = LlamaCppServerBackend()
    server = _StubbornServer()
    backend._proc = server
    backend._server_generation = 1
    monkeypatch.setitem(llm_backends._BACKEND_INSTANCES, "llamacpp", backend)

    assert backend.close() is False
    assert pm.asr_gpu.asr_gpu_blocked() is False  # ASR itself has nothing pending
    assert pm.gpu_admission_blocked() is True
    status = pm.gpu_cleanup_status()
    assert status["gpu_blocked"] is True
    assert status["pending_resources"][0]["kind"] == "local_server"

    # A handle nobody could close is a different claim: the process is gone, so
    # it holds no VRAM and must not pause the queue.
    monkeypatch.setattr(llamacpp_server, "close_kill_on_close_job", lambda _h: False)
    server.dead = True
    resources.attach_handle(backend._resource_id(), 1, 4242)
    resources.request_stop(backend._resource_id(), 1)

    assert pm.gpu_admission_blocked() is False
    status = pm.gpu_cleanup_status()
    assert status["handle_open"] is True  # ...but it is still tracked
    assert status["state"] == "cleaning"


def test_a_started_server_hands_the_card_over_rather_than_giving_it_back(
    monkeypatch, tmp_path
):
    """A *healthy* server is what the cleanup ledger cannot see.

    The test above is about one that could not be stopped. This one starts
    normally, so nothing is pending anywhere - and the queue still must not
    dispatch an ASR stage into it. Startup takes exclusive use of the card and
    then hands it to the identity it registered, in one step: registering and
    then releasing would leave an instant where the permit is free and the
    resident server is not yet holding anyone off.
    """
    from llm.backends.llamacpp_server import LlamaCppServerBackend
    from core import gpu_admission
    from web import pipeline_manager as pm

    _FakeOsBoundary(monkeypatch, tmp_path)
    backend = LlamaCppServerBackend()
    backend._ensure_server()
    identity = (backend._resource_id(), backend._server_generation)

    assert resources.gpu_blocked() is False  # nothing was asked to stop
    assert gpu_admission.holder() is None  # the permit was not kept
    assert [claim["resource_id"] for claim in gpu_admission.claims()] == [identity[0]]
    assert pm.gpu_admission_blocked() is True

    assert backend.close() is True

    assert gpu_admission.claims() == []  # the claim ended with its record
    assert pm.gpu_admission_blocked() is False


def test_a_cancelled_request_retires_the_generation_instead_of_trusting_it():
    # Cancelling the HTTP request proves this process stopped reading, not that
    # the server stopped generating. So that generation stops taking new work -
    # otherwise a steady stream of arriving tasks keeps it alive forever and the
    # unknown never ends.
    from llm.backends.llamacpp_server import LlamaCppServerBackend
    from llm.errors import TranslationCancelledError

    backend = LlamaCppServerBackend()
    backend._port = 8080
    backend._server_generation = 1
    proc = SimpleNamespace(poll=lambda: None, kill=lambda: None)
    backend._proc = proc

    class _Hanging:
        async def create(self, **_kwargs):
            await asyncio.sleep(30)

    class _Client:
        class chat:  # noqa: N801 - mirrors the openai client shape
            completions = _Hanging()

    backend._make_async_client = lambda _port: _Client()
    assert backend.generation_retiring() is False

    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    with pytest.raises(TranslationCancelledError):
        backend._request_completion({"model": "m"}, cancel)

    assert backend.generation_retiring() is True
    assert backend._proc is proc  # B's server is not killed for A's cancel


def test_a_new_task_waits_out_a_retiring_generation_instead_of_joining_it(monkeypatch):
    from llm.backends.llamacpp_server import LlamaCppServerBackend

    monkeypatch.setattr(llm_backends, "_RETIRE_WAIT_S", 5.0)
    backend = LlamaCppServerBackend()
    server = _ObedientServer()
    backend._proc = server
    backend._port = 8080
    backend._server_generation = 1
    monkeypatch.setitem(llm_backends._BACKEND_INSTANCES, "llamacpp", backend)

    joined = threading.Event()
    released = threading.Event()

    def newcomer():
        with llm_backends.backend_lease("llamacpp"):
            joined.set()
            released.wait(5.0)

    with llm_backends.backend_lease("llamacpp"):  # job B, still translating
        backend._retiring = True  # ...until one of its requests is cancelled
        thread = threading.Thread(target=newcomer, daemon=True)
        thread.start()
        # C must not attach to the generation that is on its way out...
        assert joined.wait(0.5) is False
        assert server.kills == 0  # ...and B is left alone while it works

    # B lets go: the generation is closed and C is free to start a new one.
    assert joined.wait(5.0) is True
    assert server.kills == 1
    # C is translating through a new instance, not the retired one.
    assert llm_backends._BACKEND_INSTANCES.get("llamacpp") is not backend
    released.set()
    thread.join(5.0)


def _retiring_local_backend(monkeypatch):
    """A local instance whose current generation was retired by a cancel."""
    from llm.backends.llamacpp_server import LlamaCppServerBackend

    backend = LlamaCppServerBackend()
    backend._proc = _ObedientServer()
    backend._port = 8080
    backend._server_generation = 1
    monkeypatch.setitem(llm_backends._BACKEND_INSTANCES, "llamacpp", backend)
    return backend


def test_cancelling_a_task_ends_its_wait_for_a_retiring_generation(monkeypatch):
    """The wait happens before the request layer exists, so it needs the event.

    `run_translation_and_write` blocks here on its way into the translation, and
    the request-level hard cancel cannot reach code that has not started yet: a
    cancelled task sat out the whole retirement wait and then entered the
    implementation anyway, holding a lease it should never have been granted.
    """
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setattr(llm_backends, "_RETIRE_WAIT_S", 0.5)
    monkeypatch.setattr(main, "_reopen_snapshot_run_logger", lambda _artifacts: None)
    monkeypatch.setattr(main, "_close_artifacts_logger", lambda _value: None)
    entered: list = []
    monkeypatch.setattr(
        main,
        "_run_translation_and_write_impl",
        lambda *_args, **kwargs: entered.append(kwargs.get("cancel_event")) or [],
    )
    backend = _retiring_local_backend(monkeypatch)
    cancel = threading.Event()
    outcome: list = []

    def cancelled_task():
        try:
            main.run_translation_and_write(
                "video.mp4",
                _translating_artifacts(),
                ctx=_translating_ctx(),
                cancel_event=cancel,
            )
            outcome.append("returned")
        except main.PipelineCancelledError:
            outcome.append("cancelled")
        except BaseException as exc:  # pragma: no cover - diagnostic
            outcome.append(exc)

    with llm_backends.backend_lease("llamacpp"):  # job B, still translating
        backend._retiring = True
        thread = threading.Thread(target=cancelled_task, daemon=True)
        thread.start()
        time.sleep(0.05)  # it is inside the wait by now
        cancel.set()
        thread.join(2.0)

        assert outcome == ["cancelled"]  # the wait ended on the cancel
        assert entered == []  # and never entered the translation
        # A cancelled waiter takes nothing: B is still the only holder.
        assert llm_backends.backend_lease_count("llamacpp") == 1


def test_an_api_task_does_not_wait_for_a_local_servers_retirement(monkeypatch):
    # The entry point leased "llamacpp" unconditionally, so a job translating
    # through the API could be held up for a resource it never touches.
    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")
    monkeypatch.setattr(llm_backends, "_RETIRE_WAIT_S", 30.0)
    monkeypatch.setattr(main, "_reopen_snapshot_run_logger", lambda _artifacts: None)
    monkeypatch.setattr(main, "_close_artifacts_logger", lambda _value: None)
    monkeypatch.setattr(
        main, "_run_translation_and_write_impl", lambda *_args, **_kwargs: ["done.srt"]
    )
    backend = _retiring_local_backend(monkeypatch)
    done = threading.Event()

    with llm_backends.backend_lease("llamacpp"):  # a local job, still translating
        backend._retiring = True
        thread = threading.Thread(
            target=lambda: (
                main.run_translation_and_write(
                    "video.mp4", _translating_artifacts(), ctx=_translating_ctx()
                ),
                done.set(),
            ),
            daemon=True,
        )
        thread.start()
        assert done.wait(2.0) is True
        assert llm_backends.backend_lease_count("openai") == 0
        assert llm_backends.backend_lease_count("llamacpp") == 1

    thread.join(5.0)


@pytest.mark.parametrize(
    "artifacts, ctx",
    [
        (
            _translating_artifacts(),
            SimpleNamespace(skip_translation=True, subtitle_mode="zh", job_id="sample-job"),
        ),
        (
            SimpleNamespace(segments=[], job_id="sample-job", srt_path="sample-a.srt"),
            _translating_ctx(),
        ),
    ],
    ids=["japanese-only", "nothing-recognised"],
)
def test_a_run_with_nothing_to_translate_waits_for_no_backend(
    monkeypatch, artifacts, ctx
):
    """A lease is a wait, so taking one is a decision, not a formality.

    Japanese-only output and an empty transcript never send a request, yet both
    used to queue behind a retiring local server on the way in - and could fail
    on its timeout, before the writer that was their entire job had a chance to
    run.
    """
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setattr(llm_backends, "_RETIRE_WAIT_S", 30.0)
    monkeypatch.setattr(main, "_reopen_snapshot_run_logger", lambda _artifacts: None)
    monkeypatch.setattr(main, "_close_artifacts_logger", lambda _value: None)
    monkeypatch.setattr(
        main, "_run_translation_and_write_impl", lambda *_args, **_kwargs: ["sample.ja.srt"]
    )
    backend = _retiring_local_backend(monkeypatch)
    written: list[list[str]] = []
    done = threading.Event()

    with llm_backends.backend_lease("llamacpp"):  # a local job, still translating
        backend._retiring = True
        thread = threading.Thread(
            target=lambda: (
                written.append(
                    main.run_translation_and_write("video.mp4", artifacts, ctx=ctx)
                ),
                done.set(),
            ),
            daemon=True,
        )
        thread.start()

        assert done.wait(2.0) is True
        assert written == [["sample.ja.srt"]]
        # No second holder: it went straight to the writer.
        assert llm_backends.backend_lease_count("llamacpp") == 1

    thread.join(5.0)


def test_a_retirement_outlasting_the_warning_still_refuses_new_tasks(monkeypatch):
    # The wait used to expire into "继续复用当前实例", which is the one outcome
    # retirement forbids: with tasks arriving steadily the abandoned generation
    # was never released. The timeout may warn; it may not hand the server out.
    monkeypatch.setattr(llm_backends, "_RETIRE_WAIT_S", 0.2)
    backend = _retiring_local_backend(monkeypatch)
    joined = threading.Event()
    released = threading.Event()

    def newcomer():
        with llm_backends.backend_lease("llamacpp"):
            joined.set()
            released.wait(5.0)

    with llm_backends.backend_lease("llamacpp"):  # job B, still translating
        backend._retiring = True
        thread = threading.Thread(target=newcomer, daemon=True)
        thread.start()
        # Several warning intervals later, it is still waiting.
        assert joined.wait(1.0) is False
        assert llm_backends.backend_lease_count("llamacpp") == 1
        assert backend._proc.kills == 0

    assert joined.wait(5.0) is True
    released.set()
    thread.join(5.0)


class _RecordingBackend:
    def __init__(self, label: str, calls: list[str]) -> None:
        self.label = label
        self._calls = calls

    def chat_completion(self, messages, **_kwargs) -> str:
        self._calls.append(self.label)
        return '{"translations": []}'

    def cache_identity(self) -> str:
        return self.label

    def name(self) -> str:
        return "llamacpp"

    def close(self) -> bool:
        return True


def test_a_task_never_sends_requests_to_an_instance_it_does_not_hold(monkeypatch):
    """Holding A and requesting through B is the same bug, one layer down.

    The lease binds an instance, but every request used to re-resolve the
    backend by name. Reset the settings mid-task and the task kept running while
    its requests went to the new instance - one it held no claim on, and which
    the next task to finish was free to close underneath it.
    """
    from llm import translator

    calls: list[str] = []
    leased = _RecordingBackend("A", calls)
    replacement = _RecordingBackend("B", calls)

    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setattr(main, "_reopen_snapshot_run_logger", lambda _artifacts: None)
    monkeypatch.setattr(main, "_close_artifacts_logger", lambda _value: None)
    monkeypatch.setitem(llm_backends._BACKEND_INSTANCES, "llamacpp", leased)
    monkeypatch.setitem(
        llm_backends._BACKEND_REGISTRY, "llamacpp", lambda: replacement
    )

    def translate_after_a_settings_reset(*_args, **_kwargs):
        # The user switches the local model in 翻译设置 while this task runs.
        llm_backends.reset_backend("llamacpp")
        translator._chat([{"role": "user", "content": "x"}])
        return ["done.srt"]

    monkeypatch.setattr(
        main, "_run_translation_and_write_impl", translate_after_a_settings_reset
    )

    with pytest.raises(llm_backends.BackendLeaseInvalidatedError):
        main.run_translation_and_write(
            "video.mp4", _translating_artifacts(), ctx=_translating_ctx()
        )

    assert calls == []  # B never saw a request from a task holding A
    assert llm_backends.backend_lease_count("llamacpp") == 0


def test_batch_workers_translate_through_the_leased_instance(monkeypatch):
    # Batch translation dispatches from a thread pool, and a fresh thread holds
    # no lease of its own: without carrying it over, every worker falls back to
    # resolving the backend by name - the lookup the lease exists to replace.
    import json

    from llm import translator

    calls: list[str] = []
    leased = _RecordingBackend("A", calls)
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setitem(llm_backends._BACKEND_INSTANCES, "llamacpp", leased)

    seen: list = []
    seen_lock = threading.Lock()

    def fake_chat(messages, expected_count=0, **_kwargs):
        with seen_lock:
            seen.append(llm_backends.current_lease())
        return json.dumps(
            {"translations": [{"id": index, "text": "zh"} for index in range(1, expected_count + 1)]},
            ensure_ascii=False,
        )

    monkeypatch.setattr(translator, "_chat_with_reasoning", fake_chat)
    monkeypatch.setattr(translator, "_auto_translation_batch_size", lambda *_args: 1)
    segments = [
        {"start": 0.0, "end": 1.0, "text": "こんにちは"},
        {"start": 1.0, "end": 2.0, "text": "テストです"},
    ]

    with llm_backends.backend_lease("llamacpp"):
        lease = llm_backends.current_lease()
        translator.translate_segments(segments, global_context="", max_workers=2)

    assert seen, "the engine never dispatched"
    assert all(held == lease for held in seen)
    assert all(held is not None and held[1] is leased for held in seen)


def test_a_reset_after_the_lease_check_cannot_restart_the_server(monkeypatch, tmp_path):
    """The dangerous window is between resolving the instance and calling it.

    `_dispatch` resolves the backend, then calls it. A reset landing in between
    left the caller holding an object that was merely absent from a dictionary -
    and using it meant `_ensure_server()`, which happily started a replacement
    for the server that had just been closed.
    """
    from llm.backends.llamacpp_server import LlamaCppServerBackend
    from llm.errors import BackendLeaseInvalidatedError

    os_boundary = _FakeOsBoundary(monkeypatch, tmp_path)
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    backend = LlamaCppServerBackend()
    monkeypatch.setitem(llm_backends._BACKEND_INSTANCES, "llamacpp", backend)
    monkeypatch.setitem(
        llm_backends._BACKEND_REGISTRY, "llamacpp", LlamaCppServerBackend
    )

    with llm_backends.backend_lease("llamacpp"):
        backend._ensure_server()  # this task's server is up
        assert len(os_boundary.procs) == 1

        resolved = llm_backends.task_backend()  # the lease check passes...
        llm_backends.reset_backend("llamacpp")  # ...and the settings change here

        with pytest.raises(BackendLeaseInvalidatedError):
            resolved.chat_completion([{"role": "user", "content": "x"}])

    assert len(os_boundary.procs) == 1  # no second server was started
    assert os_boundary.procs[0].poll() is not None  # and the first one is gone


def test_a_task_gets_a_fresh_instance_once_the_cleanup_finally_succeeds(
    monkeypatch, tmp_path, clean_local_backend_slot
):
    """The whole recovery, end to end.

    A failed close leaves the instance in the slot on purpose - it owns the
    process nobody else can reach. But it is permanently invalidated, so once
    the background retry confirms that process gone, leaving it there meant the
    next task leased an instance that could only raise: the page said the GPU
    was released and the job failed anyway.
    """
    from llm.backends.llamacpp_server import LlamaCppServerBackend

    os_boundary = _FakeOsBoundary(monkeypatch, tmp_path, server=_StubbornServer)
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    built: list = []

    def factory():
        instance = LlamaCppServerBackend()
        built.append(instance)
        return instance

    monkeypatch.setitem(llm_backends._BACKEND_REGISTRY, "llamacpp", factory)
    monkeypatch.delitem(llm_backends._BACKEND_INSTANCES, "llamacpp", raising=False)

    # A normal task: it starts a server and lets go at the end.
    with llm_backends.backend_lease("llamacpp") as backend:
        backend._ensure_server()
    stuck = os_boundary.procs[0]

    # The close could not confirm, so the instance stays as its owner.
    assert built == [backend]
    assert stuck.kills >= 1
    assert backend.is_invalidated() is True
    assert llm_backends._BACKEND_INSTANCES.get("llamacpp") is backend
    assert llm_backends.local_backend_cleanup_status()["stuck"] is True

    # The process finally exits and the retry confirms it.
    stuck.dead = True
    assert resources.retry_pending("local_server")["released"] == 1
    assert resources.snapshot()["state"] == "released"

    # The next task gets a working instance, not the retired one.
    with llm_backends.backend_lease("llamacpp") as second:
        assert second is not backend
        second._ensure_server()

    assert len(built) == 2
    assert len(os_boundary.procs) == 2
    assert backend.is_invalidated() is True  # never revived


def test_a_new_task_waits_while_the_stuck_instance_is_still_being_cleaned_up(
    monkeypatch, tmp_path, clean_local_backend_slot
):
    # The other half of the recovery: until the cleanup confirms, the stuck
    # instance stays - it owns the process, it is what the page reports, and it
    # is the only thing that can retry. A newcomer waits (cancellably) rather
    # than replacing it.
    from llm.backends.llamacpp_server import LlamaCppServerBackend

    os_boundary = _FakeOsBoundary(monkeypatch, tmp_path, server=_StubbornServer)
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    built: list = []

    def factory():
        instance = LlamaCppServerBackend()
        built.append(instance)
        return instance

    monkeypatch.setitem(llm_backends._BACKEND_REGISTRY, "llamacpp", factory)
    monkeypatch.delitem(llm_backends._BACKEND_INSTANCES, "llamacpp", raising=False)

    with llm_backends.backend_lease("llamacpp") as backend:
        backend._ensure_server()
    assert llm_backends.local_backend_cleanup_status()["stuck"] is True

    cancel = threading.Event()
    threading.Timer(0.05, cancel.set).start()
    with pytest.raises(main.PipelineCancelledError):
        with llm_backends.backend_lease("llamacpp", cancel):
            pass

    assert built == [backend]  # no replacement was built
    assert llm_backends._BACKEND_INSTANCES.get("llamacpp") is backend
    assert llm_backends.local_backend_cleanup_status()["stuck"] is True
    assert len(os_boundary.procs) == 1


def test_an_invalidated_instance_starts_no_server(monkeypatch, tmp_path):
    # The startup entry point, on its own: invalidation has to stop a server
    # from being spawned even when nothing else looks at the instance again.
    from llm.backends.llamacpp_server import LlamaCppServerBackend
    from llm.errors import BackendLeaseInvalidatedError

    os_boundary = _FakeOsBoundary(monkeypatch, tmp_path)
    backend = LlamaCppServerBackend()
    backend.invalidate()

    with pytest.raises(BackendLeaseInvalidatedError):
        backend._ensure_server()

    assert os_boundary.procs == []
    assert backend.is_invalidated() is True  # one-way, and it stays that way


def test_an_invalidated_instance_refuses_a_request_before_the_server(monkeypatch):
    # The request entry point, on its own: it must refuse without reaching the
    # startup path at all, which is what a backend without one would need.
    from llm.backends.llamacpp_server import LlamaCppServerBackend
    from llm.errors import BackendLeaseInvalidatedError

    reached: list[str] = []
    backend = LlamaCppServerBackend()
    monkeypatch.setattr(
        LlamaCppServerBackend,
        "_ensure_server",
        lambda _self, _cancel=None: reached.append("server"),
    )
    backend.invalidate()

    with pytest.raises(BackendLeaseInvalidatedError):
        backend.chat_completion([{"role": "user", "content": "x"}])

    assert reached == []


def test_the_claimer_that_starts_a_retirement_close_can_be_cancelled(monkeypatch):
    # Every other claimer waits on _CLOSING and can be cancelled; the one that
    # discovered the retirement used to run the close itself, so it sat inside
    # close() with its cancel event unread.
    closing = threading.Event()
    release = threading.Event()

    class _RetiringAndSlowToClose:
        def generation_retiring(self) -> bool:
            return True

        def close(self) -> bool:
            closing.set()
            release.wait(5.0)
            return True

    instance = _RetiringAndSlowToClose()
    monkeypatch.setitem(llm_backends._BACKEND_INSTANCES, "llamacpp", instance)

    cancel = threading.Event()
    threading.Timer(0.05, cancel.set).start()
    started = time.perf_counter()
    with pytest.raises(main.PipelineCancelledError):
        with llm_backends.backend_lease("llamacpp", cancel):
            pass
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0  # it waited on the close, it did not perform it
    assert closing.is_set() is True  # ...which was started all the same
    assert llm_backends.backend_lease_count("llamacpp") == 0

    # Cancelling the wait does not abort a cleanup that has begun.
    release.set()
    deadline = time.monotonic() + 5.0
    while llm_backends._CLOSING.get("llamacpp") is not None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert llm_backends._CLOSING.get("llamacpp") is None
    assert llm_backends._BACKEND_INSTANCES.get("llamacpp") is not instance


def test_a_cancel_that_arrives_while_the_lock_is_held_stops_the_claim(monkeypatch):
    """Acquiring the registry lock can take as long as a close.

    The claim checked the cancel event, then blocked on the lock, and granted a
    lease as soon as it got in - so a task cancelled during that wait came back
    holding a resource nobody was going to use.
    """
    cancel = threading.Event()

    class _CancelWhileWaiting:
        # Stands in for "the cancel arrived while this thread waited for the
        # lock": it fires exactly once, on the way in.
        def __init__(self, real):
            self._real = real
            self._armed = True

        def __enter__(self):
            entered = self._real.__enter__()
            if self._armed:
                self._armed = False
                cancel.set()
            return entered

        def __exit__(self, *exc_info):
            return self._real.__exit__(*exc_info)

    monkeypatch.setattr(
        llm_backends, "_REGISTRY_LOCK", _CancelWhileWaiting(llm_backends._REGISTRY_LOCK)
    )

    with pytest.raises(main.PipelineCancelledError):
        with llm_backends.backend_lease("llamacpp", cancel):
            pass

    assert llm_backends.backend_lease_count("llamacpp") == 0


def test_a_slow_close_does_not_make_a_claim_uncancellable(monkeypatch):
    # The close waits for a process to exit. Held under the registry lock, every
    # claim queued behind it on a mutex - and a mutex ignores cancel events.
    closing = threading.Event()
    release = threading.Event()

    class _SlowToClose:
        def close(self) -> bool:
            closing.set()
            release.wait(5.0)
            return True

    monkeypatch.setitem(llm_backends._BACKEND_INSTANCES, "llamacpp", _SlowToClose())

    def take_and_release():
        with llm_backends.backend_lease("llamacpp"):
            pass  # the last holder leaves: the close starts here

    closer = threading.Thread(target=take_and_release, daemon=True)
    closer.start()
    assert closing.wait(2.0) is True

    cancel = threading.Event()
    threading.Timer(0.05, cancel.set).start()
    started = time.perf_counter()
    with pytest.raises(main.PipelineCancelledError):
        with llm_backends.backend_lease("llamacpp", cancel):
            pass
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0  # it did not sit on the mutex until the close finished
    assert llm_backends.backend_lease_count("llamacpp") == 0
    release.set()
    closer.join(5.0)


@pytest.mark.parametrize("answer", [None, "ok", 1])
def test_a_backend_that_does_not_confirm_its_close_is_kept(monkeypatch, answer):
    """`close()` has to *say* whether it worked.

    `None` used to count as success - the same "silence means fine" that let a
    llama-server survive a teardown the caller recorded as clean. A backend with
    nothing to stop simply does not define `close()`.
    """

    class _Vague:
        def __init__(self) -> None:
            self.calls = 0

        def close(self):
            self.calls += 1
            return answer

    instance = _Vague()
    monkeypatch.setitem(llm_backends._BACKEND_INSTANCES, "llamacpp", instance)

    with llm_backends.backend_lease("llamacpp"):
        pass

    assert instance.calls == 1
    assert llm_backends._BACKEND_INSTANCES.get("llamacpp") is instance


def test_release_without_a_lease_never_closes_another_holder(monkeypatch):
    closed_backends: list[str] = []
    monkeypatch.setitem(
        llm_backends._BACKEND_INSTANCES,
        "llamacpp",
        _FakeBackend(closed_backends, "llamacpp"),
    )

    with llm_backends.backend_lease("llamacpp"):
        assert llm_backends._release_backend_lease("openai") is False
        assert closed_backends == []
