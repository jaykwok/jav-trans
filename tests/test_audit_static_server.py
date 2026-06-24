from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tools.audits.audit_nav import write_audit_index, write_latest_audit_entry
from tools.audits.serve_static import CLIENT_DISCONNECT_ERRORS, make_handler, parse_range_header, resolve_url_path


def _write(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")
    return path


@contextmanager
def _server(*, root: Path, audit_root: Path, rm_root: Path):
    handler = make_handler(root=root, audit_root=audit_root, rm_root=rm_root)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _read(url: str, *, headers: dict[str, str] | None = None):
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=5) as response:
        return response.status, dict(response.headers), response.read()


def test_parse_range_header_supports_full_and_suffix_ranges():
    byte_range = parse_range_header("bytes=2-5", 10)
    assert byte_range is not None
    assert (byte_range.start, byte_range.end, byte_range.length) == (2, 5, 4)

    suffix_range = parse_range_header("bytes=-3", 10)
    assert suffix_range is not None
    assert (suffix_range.start, suffix_range.end, suffix_range.length) == (7, 9, 3)


def test_resolve_url_path_rejects_traversal(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()

    try:
        resolve_url_path(root, "/..%2Fsecret.txt")
    except PermissionError:
        pass
    else:
        raise AssertionError("path traversal was accepted")


def test_static_server_supports_range_requests(tmp_path: Path):
    root = tmp_path / "root"
    audit_root = root / "agents" / "audits"
    rm_root = root / "agents" / "rm" / "audit-deletions"
    _write(root / "agents" / "audits" / "sample" / "audio.wav", b"0123456789")

    with _server(root=root, audit_root=audit_root, rm_root=rm_root) as base_url:
        status, headers, body = _read(
            f"{base_url}/agents/audits/sample/audio.wav",
            headers={"Range": "bytes=2-5"},
        )
        assert status == 206
        assert body == b"2345"
        assert headers["Content-Range"] == "bytes 2-5/10"
        assert headers["Accept-Ranges"] == "bytes"

        status, headers, body = _read(
            f"{base_url}/agents/audits/sample/audio.wav",
            headers={"Range": "bytes=-3"},
        )
        assert status == 206
        assert body == b"789"
        assert headers["Content-Range"] == "bytes 7-9/10"

        try:
            _read(f"{base_url}/agents/audits/sample/audio.wav", headers={"Range": "bytes=99-100"})
        except HTTPError as exc:
            assert exc.code == 416
            assert exc.headers["Content-Range"] == "bytes */10"
        else:
            raise AssertionError("invalid range did not return 416")


def test_static_server_treats_windows_range_abort_as_client_disconnect():
    assert isinstance(ConnectionAbortedError(), CLIENT_DISCONNECT_ERRORS)


def test_static_server_delete_api_moves_audit_and_rebuilds_nav(tmp_path: Path):
    root = tmp_path / "root"
    audit_root = root / "agents" / "audits"
    rm_root = root / "agents" / "rm" / "audit-deletions"
    latest = _write(audit_root / "latest" / "index.html", "<!doctype html><title>latest</title>")
    old = _write(audit_root / "old" / "index.html", "<!doctype html><title>old</title>")
    write_latest_audit_entry(audit_root=audit_root, latest_html=latest, title="Latest")
    write_audit_index(audit_root=audit_root, latest_html=latest, latest_title="Latest")

    with _server(root=root, audit_root=audit_root, rm_root=rm_root) as base_url:
        request = Request(
            f"{base_url}/__audit_api__/delete-audit",
            data=json.dumps({"href": "old/index.html"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))

    assert payload["ok"] is True
    assert payload["deleted_href"] == "old/index.html"
    assert payload["latest_href"] == "latest/index.html"
    assert not old.exists()
    moved = list(rm_root.glob("*-old"))
    assert len(moved) == 1
    assert (moved[0] / "index.html").exists()
    assert "old/index.html" not in (audit_root / "index.html").read_text(encoding="utf-8")
