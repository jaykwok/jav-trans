"""What a worker thread has to be told, and what happens when it is not.

Batch translation dispatches from a thread pool, and a fresh worker holds none
of the run's state: not the job it belongs to, not the instance the task leased,
not the endpoint snapshot that says which model that instance is pointed at, not
the prompt profile the cache key is derived from. Each of those has produced its
own bug when it was the one left behind, so they travel as one value.

The second half is what the accessor does when nothing was carried. Falling back
to "resolve the backend by name" is right for a caller that never took a lease -
the CLI, a preflight check - and is exactly wrong for a thread inside a task,
where it means translating through whatever the settings panel currently says.
"""

from __future__ import annotations

import threading

import pytest

from core import events
from llm import backends
from llm import request_config
from llm import settings as llm_settings
from llm.errors import BackendLeaseInvalidatedError
from llm.run_context import RunContext
from utils import hf_progress


class _FakeBackend:
    """A registered instance with no lifecycle hooks, so nothing is managed."""

    def __init__(self, name: str) -> None:
        self.name = name

    def cache_identity(self) -> str:
        return self.name


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setenv("TRANSLATION_BACKEND", "run-context-test")
    instance = _FakeBackend("run-context-test")
    backends.register_backend("run-context-test", lambda: instance, replace=True)
    yield instance
    with backends._REGISTRY_LOCK:
        backends._BACKEND_REGISTRY.pop("run-context-test", None)
        backends._BACKEND_INSTANCES.pop("run-context-test", None)
    backends.bind_lease(None)
    request_config.bind(None)
    request_config.bind_profile(None)


@pytest.fixture(autouse=True)
def _clean_thread_state():
    yield
    backends.bind_lease(None)
    request_config.bind(None)
    request_config.bind_profile(None)
    hf_progress.adopt_identity(("", "", ""))


def _capture_on_a_worker(context: RunContext) -> RunContext:
    """Adopt `context` on a fresh thread and report what that thread then holds."""
    seen: list[RunContext] = []

    def run() -> None:
        context.adopt()
        seen.append(RunContext.capture())

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(5)
    assert seen, "worker did not finish"
    return seen[0]


def test_the_whole_run_travels_to_a_worker_thread(registered, monkeypatch):
    monkeypatch.setattr(llm_settings, "TRANSLATION_PROMPT_PROFILE", "auto", raising=False)
    hf_progress.adopt_identity(("job-a", "run-a", "sample-a.mp4"))
    profile = object()

    with backends.backend_lease("run-context-test"):
        request_config.bind_profile(profile)
        context = RunContext.capture()

        adopted = _capture_on_a_worker(context)

        assert adopted.identity == ("job-a", "run-a", "sample-a.mp4")
        assert adopted.lease == context.lease
        assert adopted.config is context.config
        assert adopted.profile is profile


def test_a_worker_that_was_told_nothing_is_refused(registered):
    """The silent version of this is the bug: the worker resolves the backend by
    name, gets whatever the settings currently point at, and translates."""
    failures: list[BaseException] = []

    def run() -> None:
        try:
            backends.task_backend()
        except BaseException as exc:  # noqa: BLE001 - the assertion is the type
            failures.append(exc)

    with backends.backend_lease("run-context-test"):
        worker = threading.Thread(target=run)
        worker.start()
        worker.join(5)

    assert failures and isinstance(failures[0], BackendLeaseInvalidatedError)
    assert "RunContext" in str(failures[0])


def test_the_name_is_refused_on_the_same_terms(registered):
    failures: list[BaseException] = []

    def run() -> None:
        try:
            backends.task_backend_name()
        except BaseException as exc:  # noqa: BLE001
            failures.append(exc)

    with backends.backend_lease("run-context-test"):
        worker = threading.Thread(target=run)
        worker.start()
        worker.join(5)

    assert failures and isinstance(failures[0], BackendLeaseInvalidatedError)


def test_a_caller_that_never_took_a_lease_still_resolves_by_name(registered):
    """The CLI and the preflight checks have no task and no lease, and asking
    the settings which backend is selected is the correct answer for them."""
    assert backends.task_backend_name() == "run-context-test"
    assert backends.task_backend() is registered


def test_a_context_can_be_borrowed_and_given_back(registered):
    hf_progress.adopt_identity(("job-a", "run-a", "sample-a.mp4"))
    outer = RunContext.capture()
    borrowed = RunContext(
        identity=("job-b", "run-b", "sample-b.mp4"),
        lease=None,
        config=None,
        profile=None,
    )

    with borrowed.bound():
        assert hf_progress.current_identity() == ("job-b", "run-b", "sample-b.mp4")

    assert RunContext.capture() == outer


def test_capturing_on_a_thread_that_holds_nothing_is_empty():
    context = RunContext.capture()

    assert context.identity == ("", "", "")
    assert context.lease is None
    assert context.config is None
    # And adopting it is a no-op rather than an error: it is what a worker
    # started outside a task carries.
    context.adopt()
    assert str(getattr(events._thread_local, "job_id", "") or "") == ""
