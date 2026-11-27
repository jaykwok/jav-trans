"""Port allocation, exclusive socket binding, and server readiness checks."""

from __future__ import annotations

import socket
import sys
import types
from types import SimpleNamespace

import pytest

import launcher


def _hold(port: int) -> socket.socket:
    held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        held.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    held.bind(("127.0.0.1", port))
    held.listen(1)
    return held


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_a_free_port_is_returned_unchanged() -> None:
    port = _free_port()
    assert launcher._first_free_port(port) == port


def test_a_held_port_steps_forward() -> None:
    port = _free_port()
    held = _hold(port)
    try:
        assert launcher._first_free_port(port) > port
    finally:
        held.close()


def test_it_walks_past_a_run_of_held_ports() -> None:
    base = _free_port()
    held = []
    try:
        for offset in range(3):
            try:
                held.append(_hold(base + offset))
            except OSError:
                pass
        resolved = launcher._first_free_port(base)
        assert resolved >= base
        assert all(resolved != sock.getsockname()[1] for sock in held)
    finally:
        for sock in held:
            sock.close()


def test_giving_up_returns_the_preferred_port() -> None:
    """With nowhere to go it must still return a number: the caller reports a
    bind failure far more clearly than this helper could."""
    port = _free_port()
    held = _hold(port)
    try:
        assert launcher._first_free_port(port, attempts=1) == port
    finally:
        held.close()


def test_bind_free_socket_skips_active_wildcard_listener_and_keeps_exclusive_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P1-2: ports already answering connect_ex (such as a 0.0.0.0 listener on
    Windows) are skipped, and the returned socket stays exclusively bound."""
    base = _free_port()
    monkeypatch.setattr(launcher, "_port_accepts_connection", lambda candidate: candidate == base)
    sock, bound_port = launcher._bind_free_socket(base, attempts=5)
    try:
        assert bound_port > base
        assert sock.getsockname() == ("127.0.0.1", bound_port)
        # On Windows SO_EXCLUSIVEADDRUSE prevents a second bind on the same port.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as competitor:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                competitor.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            with pytest.raises(OSError):
                competitor.bind(("127.0.0.1", bound_port))
    finally:
        sock.close()


def test_resolve_events_port_avoids_collision_and_upper_boundary() -> None:
    assert launcher._resolve_events_port(2233, 2234, finder=lambda p: p) == 2234
    assert launcher._resolve_events_port(2233, 2233, finder=lambda p: p) == 2234
    events_at_max = launcher._resolve_events_port(65535, 65535, finder=lambda p: p)
    assert 1 <= events_at_max < 65535


def test_run_server_passes_bound_socket_and_publishes_events_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`JAV_SUB_EVENTS_PORT` must be overwritten with the resolved port and the
    pre-bound socket must be passed directly into `uvicorn.Server.run(sockets=[sock])`."""
    captured: dict[str, object] = {}

    class _FakeConfig:
        def __init__(self, app, *, host, port, log_level):
            captured["app"] = app
            captured["host"] = host
            captured["port"] = port
            captured["log_level"] = log_level

    class _FakeServer:
        def __init__(self, config):
            self.config = config
            self.started = False
            self.should_exit = False

        def run(self, *, sockets=None):
            captured["sockets"] = sockets
            self.started = True

    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.Config = _FakeConfig
    fake_uvicorn.Server = _FakeServer
    fake_web_app = types.ModuleType("web.app")
    fake_web_app.create_app = lambda: "fake-app"
    fake_updater = types.ModuleType("core.updater")
    fake_updater.register_update_exit_callback = lambda cb: captured.setdefault("cb", cb)

    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setitem(sys.modules, "web.app", fake_web_app)
    monkeypatch.setitem(sys.modules, "core.updater", fake_updater)
    monkeypatch.setenv("JAV_SUB_EVENTS_PORT", "9999")
    monkeypatch.setattr(launcher, "PORT", 2240)
    monkeypatch.setattr(launcher, "EVENTS_PORT", 2241)

    sock, _port = launcher._bind_free_socket(_free_port())
    launcher._run_server(sock)

    assert launcher.os.environ["JAV_SUB_EVENTS_PORT"] == "2241"
    assert captured["sockets"] == [sock]
    assert sock.fileno() == -1  # closed cleanly in finally


def test_wait_for_server_relies_on_started_flag_not_unrelated_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P1-2: even if another process listens on `port`, _wait_for_server stays
    False until our own uvicorn Server marks `started = True`."""
    held = _hold(_free_port())
    port = int(held.getsockname()[1])
    try:
        launcher._server_error.clear()
        launcher._server_failed.clear()
        monkeypatch.setattr(launcher, "_uvicorn_server", SimpleNamespace(started=False))
        assert launcher._wait_for_server(port, timeout_s=0.06) is False
        monkeypatch.setattr(launcher, "_uvicorn_server", SimpleNamespace(started=True))
        assert launcher._wait_for_server(port, timeout_s=0.1) is True
    finally:
        held.close()
