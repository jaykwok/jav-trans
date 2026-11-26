"""取消必须在静默期也生效，并且只影响被取消的那个请求。

A translation request is silent for most of its life: the model thinks for
minutes before the first token and goes quiet again mid-stream. The old
transport only checked the cancel event between stream events, so 取消 during
those gaps did nothing at all - the worker thread was parked inside a socket
read and only noticed when the provider finally spoke.

These run a real HTTP server on localhost and stop it at the three places that
matter (before the response headers, before the first event, mid-stream), so
they test the transport actually in use rather than a mock of it.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from llm.backends import openai_compat
from llm.errors import TranslationCancelledError


_HEADERS = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Type: text/event-stream\r\n"
    b"Cache-Control: no-cache\r\n"
    b"Connection: keep-alive\r\n"
    b"Transfer-Encoding: chunked\r\n"
    b"\r\n"
)


def _sse(payload: dict) -> bytes:
    body = (
        f"event: {payload['type']}\r\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\r\n\r\n"
    ).encode("utf-8")
    return b"%x\r\n%s\r\n" % (len(body), body)


_DELTA = {
    "type": "response.output_text.delta",
    "sequence_number": 1,
    "item_id": "msg_1",
    "output_index": 0,
    "content_index": 0,
    "delta": '{"translations":[{"id":0,"text":"甲"}]}',
}
_COMPLETED = {
    "type": "response.completed",
    "sequence_number": 2,
    "response": {"id": "resp_1", "usage": {"input_tokens": 1, "output_tokens": 1}},
}


class _FakeResponsesServer:
    """Serves /v1/responses and stops wherever the test asks it to.

    Records when the client closed the connection, which is how these tests tell
    "the request was abandoned" from "the request was merely ignored".
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.closed_at: dict[str, float] = {}
        self.requests = 0
        # Signalled once a connection is parked at the stall point this mode is
        # named after. Sleeping a fixed 0.6s instead only *probably* got there:
        # on a loaded machine the cancel could land while the server was still
        # reading the request, which is a different test than the one intended
        # (and made the "did the client close the socket?" assertion flaky).
        self._reached_lock = threading.Lock()
        self._reached: dict[str, threading.Event] = {}
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def _reached_event(self, mode: str) -> threading.Event:
        with self._reached_lock:
            return self._reached.setdefault(mode, threading.Event())

    def wait_until_reached(self, mode: str, timeout: float = 10.0) -> bool:
        """Block until a request is parked at `mode`'s stall point."""
        return self._reached_event(mode).wait(timeout)

    def _accept_loop(self) -> None:
        self._sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except (TimeoutError, OSError):
                continue
            thread = threading.Thread(target=self._serve, args=(conn,), daemon=True)
            self._threads.append(thread)
            thread.start()

    def _serve(self, conn: socket.socket) -> None:
        conn.settimeout(0.2)
        body = self._read_request(conn)
        if body is None:
            return
        self.requests += 1
        mode = self.mode
        # A body carrying "job-b" is the request that must be unaffected.
        if b"job-b" in body:
            mode = "complete"
        try:
            if mode == "before_headers":
                pass  # the client is left waiting for the status line
            elif mode == "before_first_event":
                conn.sendall(_HEADERS)
            elif mode == "mid_stream":
                conn.sendall(_HEADERS)
                conn.sendall(_sse(_DELTA))
            elif mode == "complete":
                conn.sendall(_HEADERS)
                conn.sendall(_sse(_DELTA))
                conn.sendall(_sse(_COMPLETED))
                conn.sendall(b"0\r\n\r\n")
                self._reached_event(mode).set()
                conn.close()
                return
            # Everything this mode was going to send has been sent: the request
            # is now in the silence the test is about.
            self._reached_event(mode).set()
            self._wait_for_close(conn, mode)
        except OSError:
            self.closed_at.setdefault(mode, time.monotonic())
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _read_request(self, conn: socket.socket) -> bytes | None:
        """Headers *and* body: which job this is, is in the body."""
        raw = b""
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                chunk = conn.recv(65536)
            except TimeoutError:
                continue
            except OSError:
                return None
            if not chunk:
                return None
            raw += chunk
            head, _, payload = raw.partition(b"\r\n\r\n")
            if not _:
                continue
            length = 0
            for line in head.split(b"\r\n"):
                name, _sep, value = line.partition(b":")
                if name.lower() == b"content-length":
                    length = int(value.strip() or 0)
            if len(payload) >= length:
                return raw
        return None

    def _wait_for_close(self, conn: socket.socket, mode: str) -> None:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not self._stop.is_set():
            try:
                if conn.recv(1) == b"":
                    self.closed_at[mode] = time.monotonic()
                    return
            except TimeoutError:
                continue
            except OSError:
                self.closed_at[mode] = time.monotonic()
                return

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass


@pytest.fixture
def fake_api(monkeypatch):
    servers: list[_FakeResponsesServer] = []

    def start(mode: str) -> _FakeResponsesServer:
        server = _FakeResponsesServer(mode)
        servers.append(server)
        monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", server.base_url)
        monkeypatch.setenv("API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL_NAME", "fake-model")
        return server

    yield start
    for server in servers:
        server.close()


def _translate(cancel_event, marker: str = "job-a") -> str:
    return openai_compat._chat_responses(
        messages=[{"role": "user", "content": marker}],
        expected_count=1,
        cancel_event=cancel_event,
    )


@pytest.mark.parametrize(
    "mode",
    ["before_headers", "before_first_event", "mid_stream"],
)
def test_cancel_reaches_a_request_stopped_anywhere(fake_api, mode):
    server = fake_api(mode)
    cancel = threading.Event()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_translate, cancel)
        # Cancel into the silence - which is exactly where the old transport was
        # deaf - but only once the server says the request is actually parked
        # there. A wall-clock sleep could not tell "in the silence" from "still
        # connecting".
        assert server.wait_until_reached(mode)
        assert not future.done()
        started = time.monotonic()
        cancel.set()
        with pytest.raises(TranslationCancelledError):
            future.result(timeout=10)
        elapsed = time.monotonic() - started

    # Seconds, not the SDK's 600s read timeout - and not because a timeout was
    # shortened: the socket itself was closed.
    assert elapsed < 5.0
    deadline = time.monotonic() + 5.0
    while mode not in server.closed_at and time.monotonic() < deadline:
        time.sleep(0.05)
    assert mode in server.closed_at  # the request released its connection
    # And it was not reissued: a cancelled request is never retried, however the
    # cleanup went.
    assert server.requests == 1


def test_cancel_before_the_request_starts_takes_effect_immediately(fake_api):
    server = fake_api("before_headers")
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(TranslationCancelledError):
        _translate(cancel)

    assert server.requests == 0  # nothing was ever sent


def test_cancelling_one_request_leaves_the_other_alone(fake_api):
    server = fake_api("before_first_event")
    cancel_a = threading.Event()

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(_translate, cancel_a, "job-a")
        future_b = pool.submit(_translate, None, "job-b")
        assert server.wait_until_reached("before_first_event")  # A is parked
        cancel_a.set()

        with pytest.raises(TranslationCancelledError):
            future_a.result(timeout=10)
        # B shares nothing with A but the process: no shared client is closed,
        # so its answer arrives whole.
        assert json.loads(future_b.result(timeout=10))["translations"][0]["id"] == 0
