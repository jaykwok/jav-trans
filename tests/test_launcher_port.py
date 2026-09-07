"""A default port is a number someone else also picked.

The launcher used to bind whatever `JAV_TRANS_PORT` said and hope. When
something already held it, `uvicorn.run` raised inside a daemon thread while
the main thread slept 1.5s and then opened the window anyway - pointing at
whatever *was* listening on that port. Walking forward to the next free port is
what Jupyter does with 8888.
"""

from __future__ import annotations

import socket
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_launcher_probe", PROJECT_ROOT / "launcher.py"
)


def _first_free_port(preferred: int, attempts: int = 20) -> int:
    """Reimport the function without executing the launcher's module body.

    Importing launcher.py would start resolving ports and pull in webview at
    import time; the function under test is self-contained, so it is read out
    of the source instead.
    """
    source = (PROJECT_ROOT / "launcher.py").read_text(encoding="utf-8")
    start = source.index("def _first_free_port(")
    end = source.index("\nPORT = ", start)
    namespace: dict = {}
    exec(compile(source[start:end], "launcher.py", "exec"), namespace)  # noqa: S102
    return namespace["_first_free_port"](preferred, attempts)


def _hold(port: int) -> socket.socket:
    held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    held.bind(("127.0.0.1", port))
    held.listen(1)
    return held


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_a_free_port_is_returned_unchanged() -> None:
    port = _free_port()
    assert _first_free_port(port) == port


def test_a_held_port_steps_forward() -> None:
    port = _free_port()
    held = _hold(port)
    try:
        assert _first_free_port(port) > port
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
        resolved = _first_free_port(base)
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
        assert _first_free_port(port, attempts=1) == port
    finally:
        held.close()


def test_the_two_defaults_do_not_collide(monkeypatch) -> None:
    from core.typed_config import env_int

    source = (PROJECT_ROOT / "launcher.py").read_text(encoding="utf-8")
    start = source.index("\nPORT = ")
    declarations = compile(source[start:source.index("\n\n", start)], "launcher.py", "exec")
    namespace = {"env_int": env_int, "_first_free_port": lambda preferred: preferred}
    monkeypatch.delenv("JAV_TRANS_PORT", raising=False)
    monkeypatch.delenv("JAV_TRANS_EVENTS_PORT", raising=False)
    exec(declarations, namespace)
    assert (namespace["PORT"], namespace["EVENTS_PORT"]) == (2233, 2234)
    # Equal requested ports at the upper boundary must also remain distinct,
    # without ever passing an out-of-range number to bind().
    monkeypatch.setenv("JAV_TRANS_PORT", "65535")
    monkeypatch.setenv("JAV_TRANS_EVENTS_PORT", "65535")
    exec(declarations, namespace)
    assert namespace["PORT"] == 65535
    assert 1 <= namespace["EVENTS_PORT"] < 65535


def test_the_resolved_events_port_is_published_not_defaulted() -> None:
    """`setdefault` would leave the app dialling the port that was already
    taken - the one case this whole helper exists for."""
    source = (PROJECT_ROOT / "launcher.py").read_text(encoding="utf-8")
    assert 'os.environ["JAV_TRANS_EVENTS_PORT"] = str(EVENTS_PORT)' in source
    assert 'setdefault("JAV_TRANS_EVENTS_PORT"' not in source
