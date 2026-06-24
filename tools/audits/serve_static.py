from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from tools.audits.audit_nav import AUDIT_RM_ROOT, AUDIT_ROOT, PROJECT_ROOT, delete_audit_entry


API_DELETE_AUDIT = "/__audit_api__/delete-audit"
BODY_LIMIT_BYTES = 64 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
CLIENT_DISCONNECT_ERRORS: tuple[type[OSError], ...] = (
    BrokenPipeError,
    ConnectionAbortedError,
    ConnectionResetError,
)


mimetypes.add_type("application/javascript; charset=utf-8", ".js")
mimetypes.add_type("application/json; charset=utf-8", ".json")
mimetypes.add_type("application/jsonl; charset=utf-8", ".jsonl")
mimetypes.add_type("text/css; charset=utf-8", ".css")
mimetypes.add_type("text/html; charset=utf-8", ".html")
mimetypes.add_type("text/plain; charset=utf-8", ".srt")
mimetypes.add_type("text/vtt; charset=utf-8", ".vtt")
mimetypes.add_type("audio/wav", ".wav")
mimetypes.add_type("audio/flac", ".flac")
mimetypes.add_type("audio/mpeg", ".mp3")
mimetypes.add_type("video/mp4", ".mp4")


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def parse_range_header(header: str | None, size: int) -> ByteRange | None:
    if not header:
        return None
    unit, separator, raw_range = header.strip().partition("=")
    if unit.lower() != "bytes" or separator != "=" or "," in raw_range:
        raise ValueError("unsupported Range header")
    start_text, dash, end_text = raw_range.partition("-")
    if dash != "-" or (not start_text and not end_text):
        raise ValueError("invalid Range header")
    if size <= 0:
        raise ValueError("range requested for empty file")

    if start_text:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    else:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise ValueError("invalid suffix range")
        start = max(size - suffix_length, 0)
        end = size - 1

    if start < 0 or start >= size:
        raise ValueError("range start outside file")
    end = min(end, size - 1)
    if end < start:
        raise ValueError("range end before start")
    return ByteRange(start=start, end=end)


def resolve_url_path(root: Path, request_path: str) -> Path:
    raw_path = urlsplit(request_path).path
    if raw_path == "/":
        return root.resolve()
    decoded = unquote(raw_path).replace("\\", "/")
    relative = decoded.lstrip("/")
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise PermissionError(f"path escapes server root: {decoded}") from exc
    return target


def guess_content_type(path: Path) -> str:
    content_type, _encoding = mimetypes.guess_type(str(path))
    if content_type:
        return content_type
    return "application/octet-stream"


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def make_handler(
    *,
    root: Path = PROJECT_ROOT,
    audit_root: Path = AUDIT_ROOT,
    rm_root: Path = AUDIT_RM_ROOT,
) -> type[BaseHTTPRequestHandler]:
    root = root.resolve()
    audit_root = audit_root.resolve()
    rm_root = rm_root.resolve()

    class AuditStaticHandler(BaseHTTPRequestHandler):
        server_version = "JAVTransAuditStatic/1.0"

        def do_OPTIONS(self) -> None:  # noqa: N802
            if urlsplit(self.path).path != API_DELETE_AUDIT:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Allow", "POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            if urlsplit(self.path).path != API_DELETE_AUDIT:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                return
            try:
                payload = self._read_json_body()
                href = payload.get("href")
                if not isinstance(href, str) or not href.strip():
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "missing href"})
                    return
                started_at = time.perf_counter()
                result = delete_audit_entry(href=href, audit_root=audit_root, rm_root=rm_root)
                elapsed_s = time.perf_counter() - started_at
                self.log_message(
                    "deleted audit href=%s moved_to=%s elapsed=%.3fs",
                    href,
                    result.get("moved_to", ""),
                    elapsed_s,
                )
                self._send_json(HTTPStatus.OK, {"ok": True, "delete_elapsed_s": round(elapsed_s, 3), **result})
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
            except (json.JSONDecodeError, ValueError, PermissionError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            except Exception as exc:  # pragma: no cover - keeps the browser API readable.
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})

        def do_GET(self) -> None:  # noqa: N802
            self._serve_static(head_only=False)

        def do_HEAD(self) -> None:  # noqa: N802
            self._serve_static(head_only=True)

        def _read_json_body(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if length > BODY_LIMIT_BYTES:
                raise ValueError("request body too large")
            body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(body.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _serve_static(self, *, head_only: bool) -> None:
            if urlsplit(self.path).path == "/":
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", "/agents/audits/index.html")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            try:
                target = resolve_url_path(root, self.path)
            except PermissionError as exc:
                self.send_error(HTTPStatus.FORBIDDEN, str(exc))
                return
            if target.is_dir():
                index = target / "index.html"
                if index.exists():
                    target = index
                else:
                    self.send_error(HTTPStatus.FORBIDDEN, "directory listing is disabled")
                    return
            if not target.exists() or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "file not found")
                return

            stat_result = target.stat()
            size = stat_result.st_size
            try:
                requested_range = parse_range_header(self.headers.get("Range"), size)
            except ValueError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            content_type = guess_content_type(target)
            if requested_range is None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Length", str(size))
                start = 0
                length = size
            else:
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {requested_range.start}-{requested_range.end}/{size}")
                self.send_header("Content-Length", str(requested_range.length))
                start = requested_range.start
                length = requested_range.length
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Last-Modified", self.date_time_string(stat_result.st_mtime))
            self.end_headers()
            if not head_only and length:
                self._send_file_range(target, start=start, length=length)

        def _send_file_range(self, path: Path, *, start: int, length: int) -> None:
            remaining = length
            with path.open("rb") as file:
                file.seek(start)
                while remaining > 0:
                    chunk = file.read(min(COPY_CHUNK_BYTES, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except CLIENT_DISCONNECT_ERRORS:
                        break
                    remaining -= len(chunk)

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

    return AuditStaticHandler


class AuditThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve local JAVTrans audit pages without live-server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8080, help="Bind port.")
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="Static root; defaults to project root.")
    parser.add_argument("--audit-root", default=str(AUDIT_ROOT), help="Audit root for delete API.")
    parser.add_argument("--rm-root", default=str(AUDIT_RM_ROOT), help="Destination root for deleted audits.")
    parser.add_argument("--open", action="store_true", help="Open the audit navigation page in the default browser.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    audit_root = Path(args.audit_root).resolve()
    rm_root = Path(args.rm_root).resolve()
    handler = make_handler(root=root, audit_root=audit_root, rm_root=rm_root)
    server = AuditThreadingHTTPServer((args.host, args.port), handler)
    base_url = f"http://{args.host}:{server.server_port}"
    print(f"Audit navigation: {base_url}/agents/audits/index.html", flush=True)
    print(f"Latest audit entry: {base_url}/agents/audits/latest-audit.html", flush=True)
    print("Delete API: enabled at /__audit_api__/delete-audit", flush=True)
    print("Range requests: enabled", flush=True)
    if args.open:
        webbrowser.open(f"{base_url}/agents/audits/index.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAudit static server stopped.", file=sys.stderr)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
