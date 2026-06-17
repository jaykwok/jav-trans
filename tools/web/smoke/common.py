from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = {"done", "failed", "cancelled"}


def timestamp_prefix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_run_dir(slug: str) -> Path:
    return Path("agents") / "temp" / f"{timestamp_prefix()}_{slug}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def http_json(method: str, url: str, payload: Any | None = None, *, timeout_s: float = 30.0) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code}: {detail}") from exc
    if not raw.strip():
        return None
    return json.loads(raw)


def is_port_open(host: str, port: int, *, timeout_s: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        return sock.connect_ex((host, int(port))) == 0


def wait_for_port(host: str, port: int, *, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if is_port_open(host, port):
            return True
        time.sleep(0.5)
    return False


def compact_progress(state: dict[str, Any]) -> str:
    progress = state.get("progress") if isinstance(state.get("progress"), dict) else {}
    extra = progress.get("extra") if isinstance(progress.get("extra"), dict) else {}
    current = progress.get("current", extra.get("current", ""))
    total = progress.get("total", extra.get("total", ""))
    elapsed = extra.get("elapsed_s", "")
    label = extra.get("label") or state.get("current_stage") or progress.get("stage") or ""
    parts = [
        f"status={state.get('status')}",
        f"stage={state.get('current_stage')}",
    ]
    if label:
        parts.append(f"label={label}")
    if current != "" or total != "":
        parts.append(f"progress={current}/{total}")
    if elapsed != "":
        parts.append(f"elapsed_s={elapsed}")
    return " ".join(str(part) for part in parts)
