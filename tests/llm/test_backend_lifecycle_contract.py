import threading

import pytest

from llm import backends
from llm.backends.base import ManagedTranslationBackend
from llm.errors import BackendUnavailableError


class ManagedStub:
    def __init__(self):
        self.invalidated = False

    def invalidate(self):
        self.invalidated = True

    def is_invalidated(self):
        return self.invalidated

    def cleanup_status(self):
        return {"stuck": True}

    def generation_retiring(self):
        return False

    def close(self):
        return False


@pytest.mark.parametrize("missing", ["invalidate", "is_invalidated", "cleanup_status", "generation_retiring", "close"])
def test_incomplete_lifecycle_is_rejected_before_caching(monkeypatch, missing):
    instance = ManagedStub()
    setattr(instance, missing, None)
    monkeypatch.setitem(backends._BACKEND_REGISTRY, "incomplete-test", lambda: instance)
    with pytest.raises(TypeError, match=missing):
        backends.get_backend("incomplete-test")
    assert "incomplete-test" not in backends._BACKEND_INSTANCES


def test_process_backend_cannot_be_constructed_without_lifecycle():
    class Incomplete(ManagedTranslationBackend):
        def name(self):
            return "incomplete"

        def chat_completion(self, *args, **kwargs):
            return ""

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


@pytest.mark.parametrize("state", ["closing", "retiring", "stuck"])
def test_admission_deadline_never_replaces_or_leases_old_instance(monkeypatch, state):
    instance = ManagedStub()
    monkeypatch.setenv("LOCAL_BACKEND_WAIT_TIMEOUT_S", "0.03")
    monkeypatch.setattr(backends, "_BACKEND_INSTANCES", {"deadline-test": instance})
    monkeypatch.setattr(backends, "_BACKEND_LEASES", {})
    monkeypatch.setattr(backends, "_CLOSING", {})
    if state == "closing":
        backends._CLOSING["deadline-test"] = instance
    elif state == "retiring":
        instance.generation_retiring = lambda: True
        backends._BACKEND_LEASES["deadline-test"] = [instance]
    else:
        instance.invalidate()
    count = backends.backend_lease_count("deadline-test")
    with pytest.raises(BackendUnavailableError):
        with backends.backend_lease("deadline-test", threading.Event()):
            pytest.fail("must not claim")
    assert backends._BACKEND_INSTANCES["deadline-test"] is instance
    assert backends.backend_lease_count("deadline-test") == count
    if state == "closing":
        assert backends._CLOSING["deadline-test"] is instance


def test_unreadable_invalidation_is_not_permission_to_run(monkeypatch):
    class Broken(ManagedStub):
        def is_invalidated(self):
            raise RuntimeError("unavailable")

    instance = Broken()
    instance.cleanup_status = lambda: {"stuck": False}  # live, not cleaning
    monkeypatch.setitem(backends._BACKEND_INSTANCES, "unreadable-test", instance)
    with pytest.raises(BackendUnavailableError):
        with backends.backend_lease("unreadable-test"):
            pytest.fail("must not grant")
    assert backends._BACKEND_INSTANCES["unreadable-test"] is instance


def test_incomplete_cleanup_snapshot_is_unknown():
    instance = ManagedStub()
    instance.cleanup_status = lambda: {}
    assert backends._instance_cleanup_pending(instance)


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_admission_deadline_must_be_finite_positive(monkeypatch, value):
    from core.typed_config import configuration_problems

    monkeypatch.setenv("LOCAL_BACKEND_WAIT_TIMEOUT_S", value)
    assert backends._admission_timeout_s() == 600.0
    assert any("LOCAL_BACKEND_WAIT_TIMEOUT_S" in problem for problem in configuration_problems())
    monkeypatch.setenv("LOCAL_BACKEND_WAIT_TIMEOUT_S", "1.5")
    assert backends._admission_timeout_s() == 1.5
    assert not any("LOCAL_BACKEND_WAIT_TIMEOUT_S" in problem for problem in configuration_problems())


@pytest.fixture
def isolated_registry(monkeypatch):
    for name in ("_BACKEND_INSTANCES", "_BACKEND_LEASES", "_CLOSING", "_CLOSE_OPERATIONS"):
        monkeypatch.setattr(backends, name, {})
    monkeypatch.setenv("LOCAL_BACKEND_CLOSE_WAIT_S", "0.03")


def test_hung_close_is_bounded_coalesced_and_keeps_owner(monkeypatch, isolated_registry):
    release = threading.Event()
    entered = threading.Event()
    calls = []
    instance = ManagedStub()
    def close():
        calls.append(1)
        entered.set()
        release.wait(5)
        return True
    instance.close = close
    backends._BACKEND_INSTANCES["close-test"] = instance
    monkeypatch.setitem(backends._BACKEND_REGISTRY, "close-test", lambda: instance)
    operation = None
    try:
        with backends.backend_lease("close-test"):
            pass
        assert entered.is_set()
        assert backends._CLOSING["close-test"] is instance
        operation = backends._CLOSE_OPERATIONS["close-test"]
        assert not operation.done.is_set()
        # The same operation is joined, never duplicated.
        assert backends._start_close("close-test", instance) is operation
        assert not backends._finish_close("close-test", instance)
        assert calls == [1]
        with pytest.raises(BackendUnavailableError):
            backends.get_backend("close-test")
    finally:
        release.set()
        operation = operation or backends._CLOSE_OPERATIONS.get("close-test")
        if operation:
            assert operation.done.wait(2)
    assert "close-test" not in backends._CLOSING
    assert "close-test" not in backends._CLOSE_OPERATIONS


def test_late_lease_release_does_not_close_reset_instance_twice(monkeypatch, isolated_registry):
    instance = ManagedStub()
    calls = []
    instance.close = lambda: calls.append(1) or True
    backends._BACKEND_INSTANCES["late-test"] = instance
    monkeypatch.setitem(backends._BACKEND_REGISTRY, "late-test", lambda: instance)
    with backends.backend_lease("late-test"):
        backends.reset_backend("late-test")
    assert calls == [1]


def test_failed_close_thread_start_retains_reservation(monkeypatch, isolated_registry):
    instance = ManagedStub()
    backends._BACKEND_INSTANCES["start-test"] = instance
    def fail_start(_self):
        raise RuntimeError("cannot start thread")
    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="cannot start thread"):
        backends.reset_backend("start-test")
    assert backends._CLOSING["start-test"] is instance
    with pytest.raises(BackendUnavailableError):
        backends.get_backend("start-test")
