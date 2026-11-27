"""First-run installer: build the runtime environment, then hand off to the app.

The release used to carry torch and the ASR weights inside the PyInstaller
bundle, which made a ~6 GB download for every user and a fresh ~6 GB upload for
every patch. Both are fetchable at run time, so the release can instead be the
launcher plus source, and this script is what turns that into a working install.

**Standard library only, and that is not negotiable.** This runs before any
dependency exists; importing anything from site-packages here would be the same
chicken-and-egg it exists to solve.

The proxy is the reason for the console prompt. Settings written by the web page
land in `config/config.json` as PROXY_PROTOCOL/HOST/PORT, and every later run reads
them from there - but on the very first run there is no `config/config.json` and no
web page yet, so the console is the only place left to ask. The keys written here
are exactly the ones `core.config.network_proxy_url_from_env` reads, so one answer
serves both.

Nothing here decides whether the network is "fast enough". It range-reads the
real torch wheel named in `uv.lock`, learns the wheel's true size from the
`Content-Range` header, converts the measured rate into how long that download
would actually take, and lets the user decide from that number.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import html.parser
import http.client
import ipaddress
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path, PurePosixPath

_FROZEN = bool(getattr(sys, "frozen", False))
ROOT = (Path(sys.executable) if _FROZEN else Path(__file__)).resolve().parent
CONFIG_PATH = ROOT / "config" / "config.json"
LEGACY_ENV_PATH = ROOT / ".env"
VENV_PATH = ROOT / ".venv"
LOCK_PATH = ROOT / "uv.lock"
# In .venv so that deleting the environment also invalidates the record of it.
STAMP_PATH = VENV_PATH / ".jav-sub-install-stamp"
# Touched by launcher.py once its server answers and its window is up. That is
# the first moment at which hiding the console cannot swallow a startup failure.
# Default launches isolate this path per PID; READY_PATH stays as a test hook.
_DEFAULT_READY_PATH = ROOT / "tmp" / ".app-ready"
READY_PATH = _DEFAULT_READY_PATH

# Import roots that have to exist on disk for the app to start, mapped to the
# distribution that provides each. A tripwire, not a manifest: what breaks an
# environment that used to work is someone deleting a folder to reclaim disk
# space, and a handful of stat() calls catches that on every launch. Anything
# subtler is left to the import probe, which costs a torch import and therefore
# only runs in --doctor and after a launch that failed.
CRITICAL_IMPORT_ROOTS = {
    "torch": "torch",
    "torchcodec": "torchcodec",
    "transformers": "transformers",
    "numpy": "numpy",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "webview": "pywebview",
    "openai": "openai",
}
# Files that travel in the archive. uv can rebuild .venv from these; it cannot
# rebuild these, so a missing one is reported as "re-extract", never "repair".
PAYLOAD_PATHS = ("launcher.py", "pyproject.toml", "src")

# Every file the archive carries, written at build time. Upgrading by extracting
# a new release over the old folder replaces files but never removes one, so a
# module or FFmpeg DLL the new release dropped would stay behind - and a stale
# package directory shadows a new module of the same name. The index is what lets
# a launch tell "shipped by this release" from "left over by an older one".
RELEASE_INDEX_NAME = "release-files.json"
RELEASE_INDEX_SCHEMA = "jav_sub_release_files_v1"
RELEASE_MANIFEST_SCHEMA = "release_manifest_v2"
# Stale files are moved here, not deleted: someone may have edited one.
LEFTOVERS_DIR = Path("tmp") / "upgrade-leftovers"

# Online self-update handoff between launcher.py/Web UI and bootstrap.py.
UPDATE_EXIT_CODE = 42
DEFAULT_UPDATE_REPO = "jaykwok/jav-sub"
UPDATE_ARCHIVE_NAME = "jav-sub-windows-x64.zip"
UPDATE_SHA256_NAME = "jav-sub-windows-x64.zip.sha256"
UPDATE_MANIFEST_NAME = "release-manifest.json"
PENDING_UPDATE_SCHEMA = "jav_sub_pending_update_v1"
UPDATES_DIR = Path("tmp") / "updates"
PENDING_UPDATE_FILE_NAME = "pending-update.json"
_UPDATE_SEMVER_RE = re.compile(r"^[vV]?(\d+)\.(\d+)\.(\d+)$")
_UPDATE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UPDATE_SHA256_LINE_RE = re.compile(r"^\s*([0-9a-fA-F]{64})(?:\s+\*?(.+?))?\s*$")
_PROTECTED_UPDATE_TOP_DIRS = frozenset(
    {
        ".venv",
        ".python",
        "models",
        "config",
        "outputs",
        "cache",
        "logs",
        "tmp",
        ".git",
    }
)
_REQUIRED_PAYLOAD_ENTRIES = (
    "jav-sub.exe",
    "launcher.py",
    "pyproject.toml",
    "uv.lock",
    RELEASE_INDEX_NAME,
)
_MAX_ARCHIVE_MEMBERS = 10_000
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_GITHUB_ALLOWED_REDIRECT_SUFFIXES = (
    "github.com",
    "githubusercontent.com",
    "githubassets.com",
)
_WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{idx}" for idx in range(1, 10)),
        *(f"LPT{idx}" for idx in range(1, 10)),
    }
)
_LAUNCHER_INSTANCE_DIR_NAME = "launcher"
_WORKSPACE_GATE_NAME = "workspace.gate"
_REQUIRED_FFMPEG_DLL_PATTERNS = ("avcodec-*.dll", "avformat-*.dll", "avutil-*.dll")

# Must match core.config._PROXY_ENV_KEYS / _NO_PROXY_ENV_KEYS. Both cases are
# set because parts of the stack read only the lowercase names.
PROXY_ENV_KEYS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)
NO_PROXY_ENV_KEYS = ("NO_PROXY", "no_proxy")
_LOOPBACK_NO_PROXY = ("127.0.0.1", "localhost", "::1")
SUPPORTED_PROXY_PROTOCOLS = ("http", "https", "socks5")
DEFAULT_PROXY_PORT = "7890"

# torch is by far the largest download, and the CUDA index is a different host
# from PyPI, so it is the only honest thing to measure against.
FALLBACK_PROBE_URL = "https://download.pytorch.org/whl/cu130/torch/"
FALLBACK_TOTAL_BYTES = 1_900_000_000
PROBE_BYTES = 4 * 1024 * 1024
PROBE_MAX_S = 8.0
PROBE_TIMEOUT_S = 20.0

_WHEEL_URL = re.compile(r'"(https?://[^"]+\.whl)"')

# uv is published to PyPI as a wheel that carries the binary, which is a better
# source than its GitHub release for this audience: PyPI is the one host the
# install already depends on, and it is mirrored where GitHub is not.
UV_PYPI_INDEX = "https://pypi.org/simple/uv/"
_UV_WHEEL_FILENAME_RE = re.compile(r"^uv-([0-9]+(?:\.[0-9]+)*)-py3-none-win_amd64\.whl$")


_BOOTSTRAP_LOG_ENABLED = False
_BOOTSTRAP_LOG_LOCK = threading.Lock()
_BOOTSTRAP_LOG_MAX_BYTES = 2 * 1024 * 1024

_NTSTATUS_NAMES = {
    0x80000003: "STATUS_BREAKPOINT",
    0xC0000005: "STATUS_ACCESS_VIOLATION",
    0xC000001D: "STATUS_ILLEGAL_INSTRUCTION",
    0xC00000FD: "STATUS_STACK_OVERFLOW",
    0xC0000135: "STATUS_DLL_NOT_FOUND",
    0xC0000139: "STATUS_ENTRYPOINT_NOT_FOUND",
    0xC0000142: "STATUS_DLL_INIT_FAILED",
    0xC0000374: "STATUS_HEAP_CORRUPTION",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",
}


def _format_exit_code(code: int | None) -> str:
    if code is None:
        return "None"
    try:
        value = int(code)
    except (TypeError, ValueError):
        return str(code)
    if os.name == "nt":
        unsigned = None
        if value >= 0x80000000:
            unsigned = value & 0xFFFFFFFF
        elif value < 0:
            candidate = value & 0xFFFFFFFF
            if candidate >= 0x80000000:
                unsigned = candidate
        if unsigned is not None:
            label = _NTSTATUS_NAMES.get(unsigned)
            if label:
                return f"{value} (0x{unsigned:08X} {label})"
            return f"{value} (0x{unsigned:08X})"
    return str(value)


def _bootstrap_log_path() -> Path:
    return ROOT / "logs" / "bootstrap.log"


def _append_bootstrap_log(message: str) -> None:
    if not _BOOTSTRAP_LOG_ENABLED:
        return
    path = _bootstrap_log_path()
    try:
        with _BOOTSTRAP_LOG_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size > _BOOTSTRAP_LOG_MAX_BYTES:
                backup = path.with_suffix(".log.1")
                try:
                    if backup.exists():
                        backup.unlink()
                    path.replace(backup)
                except OSError:
                    pass
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with path.open("a", encoding="utf-8", errors="replace") as writer:
                for line in str(message).splitlines() or [""]:
                    writer.write(f"{stamp} {line}\n")
    except OSError:
        pass


def log(message: str = "") -> None:
    print(message, flush=True)
    _append_bootstrap_log(message)


def enable_utf8_console() -> None:
    """Make the console able to show the Chinese prompts below.

    A frozen exe inherits the console code page, which on a non-Chinese Windows
    is cp437 - printing would raise UnicodeEncodeError and take the installer
    down with it. Best-effort: if either half fails, `errors="replace"` on the
    reconfigure still keeps output flowing.
    """
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:  # noqa: BLE001 - console tweaks must never be fatal
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def interactive() -> bool:
    try:
        return bool(sys.stdin is not None and sys.stdin.isatty())
    except (AttributeError, OSError, ValueError):
        return False


# --- console window ---------------------------------------------------------

_SW_HIDE = 0
_SW_SHOW = 5
_console_hidden = False


def _console_window() -> int:
    if os.name != "nt":
        return 0
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception:  # noqa: BLE001 - a window handle is never worth crashing for
        return 0


def _apply_console_visibility(visible: bool) -> bool:
    hwnd = _console_window()
    if not hwnd:
        return False
    try:
        import ctypes

        ctypes.windll.user32.ShowWindow(hwnd, _SW_SHOW if visible else _SW_HIDE)
    except Exception:  # noqa: BLE001
        return False
    return True


def set_console_visible(visible: bool) -> None:
    """Hide (or bring back) the console this exe was started with.

    The console exists for the install and for errors; once the app window is up
    it is just a black rectangle the user has to ignore. Frozen builds only: in a
    source checkout this handle belongs to the developer's own terminal, and
    hiding a window they opened for other work would be hostile.
    """
    global _console_hidden
    if not _FROZEN or _console_hidden == (not visible):
        return
    if _apply_console_visibility(visible):
        _console_hidden = not visible


def pause_before_exit() -> None:
    """Hold the window open on failure.

    Double-clicked from Explorer the console dies with the process, so an error
    message would only flash - which is how a broken install looks like "nothing
    happens" instead of like a diagnosis.
    """
    if not _FROZEN or not interactive():
        return
    try:
        input("\n按回车关闭这个窗口 ...")
    except (EOFError, KeyboardInterrupt):
        pass


# --- config/config.json -----------------------------------------------------


def _stringify_config_value(value: object) -> str | None:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return None


def _read_config_file_strict(path: Path | None = None) -> dict[str, str]:
    """Parse `config/config.json`, raising on unreadable, truncated, or malformed content.

    A missing file returns ``{}``, and an explicit ``{}`` object returns ``{}``.
    An existing 0-byte / whitespace-only file is treated as truncated/corrupt so
    model cache cleanup never misinterprets a truncated config as empty.
    """
    target = path or CONFIG_PATH
    if not target.exists():
        return {}
    try:
        text = target.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("config/config.json 不是有效的 UTF-8 编码") from exc
    if not text.strip():
        raise ValueError("config/config.json 内容为空（可能已被截断）")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"config/config.json 格式损坏（{exc}）") from exc
    if not isinstance(loaded, dict):
        raise ValueError("config/config.json 顶层不是 JSON 对象")
    values: dict[str, str] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            continue
        coerced = _stringify_config_value(value)
        if coerced is not None:
            values[key] = coerced
    return values


def read_config_file(path: Path | None = None) -> dict[str, str]:
    """Parse `config/config.json` leniently. Malformed content is never fatal."""
    try:
        return _read_config_file_strict(path)
    except (OSError, ValueError):
        return {}


def _replace_atomically(
    src: Path,
    dst: Path,
    *,
    retries: int = 6,
    base_delay_s: float = 0.02,
) -> None:
    """Replace `dst` with `src`, retrying transient Windows `PermissionError`."""
    attempts = max(1, int(retries))
    delay = max(0.0, float(base_delay_s))
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(delay * (2**attempt))


def _write_json_atomic(target: Path, payload: object) -> None:
    """Atomically write JSON to `target` via a same-directory uuid temp file + fsync."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_atomically(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def update_config_file(updates: dict[str, str], path: Path | None = None) -> None:
    """Merge keys into `config/config.json`, preserving all existing settings.

    The web settings page owns this file too, so rewriting it wholesale would
    throw away the user's API keys on the next launch. Refuses to overwrite a
    non-UTF-8, truncated/empty, or malformed JSON file rather than silently
    discarding it.
    """
    target = path or CONFIG_PATH
    try:
        existing = _read_config_file_strict(target)
    except ValueError as exc:
        raise ValueError(f"{exc}，已拒绝覆盖以保护现有配置。") from exc

    merged: dict[str, str] = dict(existing)
    for key, value in updates.items():
        merged[str(key)] = "" if value is None else str(value)

    _write_json_atomic(target, merged)


def _no_proxy_value_with_loopback(existing: str) -> str:
    """Merge loopback hosts into a NO_PROXY list, preserving user entries."""
    entries = [item.strip() for item in str(existing or "").split(",")]
    entries = [item for item in entries if item]
    if "*" in entries:
        return ",".join(entries)
    lowered = {item.lower() for item in entries}
    for host in _LOOPBACK_NO_PROXY:
        if host not in lowered:
            entries.append(host)
            lowered.add(host)
    return ",".join(entries)


def _format_proxy_host(raw_host: str) -> str:
    """Wrap a bare IPv6 literal in brackets while keeping other hosts intact."""
    host = raw_host.strip()
    if not host or "://" in host:
        return host
    userinfo, sep, host_part = host.rpartition("@")
    if ":" in host_part and not (host_part.startswith("[") and host_part.endswith("]")):
        try:
            ipaddress.IPv6Address(host_part)
        except ValueError:
            return host
        host_part = f"[{host_part}]"
    return f"{userinfo}@{host_part}" if sep else host_part


def proxy_url_from(values: dict[str, str]) -> str:
    """Same rule as core.config.network_proxy_url_from_env, restated here
    because importing src/ would mean importing its dependencies."""
    protocol = (values.get("PROXY_PROTOCOL") or "http").strip().lower() or "http"
    host = _format_proxy_host(values.get("PROXY_HOST") or "")
    port = (values.get("PROXY_PORT") or "").strip()
    if not host or not port:
        return ""
    if protocol not in SUPPORTED_PROXY_PROTOCOLS:
        protocol = "http"
    return f"{protocol}://{host}:{port}"


def _validate_proxy_url(proxy_url: str) -> tuple[dict[str, str], str]:
    raw = str(proxy_url or "").strip()
    if any(ch.isspace() or ch in "\\\x00" for ch in raw):
        raise ValueError("代理 URL 不能包含空白或非法字符")
    parsed = urllib.parse.urlsplit(raw)
    if (
        parsed.scheme not in SUPPORTED_PROXY_PROTOCOLS
        or not parsed.hostname
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("代理 URL 必须是 http://、https:// 或 socks5://地址:端口")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("代理端口必须是 1-65535 的数字") from exc
    if port is not None:
        host = parsed.netloc.rsplit(":", 1)[0]
    else:
        if parsed.netloc.endswith(":"):
            raise ValueError("代理端口必须是 1-65535 的数字")
        host = parsed.netloc
        port = {"http": 80, "https": 443, "socks5": 1080}[parsed.scheme]
    if not host or "://" in host or any(ch in host for ch in ("/", "?", "#", "\\", "\x00")):
        raise ValueError("代理地址格式无效")
    if not 1 <= port <= 65535:
        raise ValueError("代理端口必须是 1-65535 的数字")
    settings = {
        "PROXY_PROTOCOL": parsed.scheme,
        "PROXY_HOST": host,
        "PROXY_PORT": str(port),
    }
    return settings, proxy_url_from(settings)


def _parse_proxy_host_entry(raw: str) -> tuple[str, str, str]:
    """Parse user input at the proxy host prompt.

    Accepts plain host, bare/bracketed IPv6, `host:port`, or full
    `scheme://host[:port]` URLs and rejects whitespace/path/invalid port.
    Returns ``(detected_protocol, normalized_host, detected_port)``.
    """
    text = raw.strip()
    if not text:
        return "", "", ""
    if any(ch.isspace() or ch in "?#\\\x00" for ch in text):
        raise ValueError("代理地址不能包含空白、?、# 或反斜杠")
    if "://" in text:
        parsed = urllib.parse.urlsplit(text)
        scheme = parsed.scheme.lower()
        if scheme not in SUPPORTED_PROXY_PROTOCOLS:
            raise ValueError(f"不支持的代理协议 {parsed.scheme!r}，仅支持 http、https、socks5")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment or not parsed.hostname:
            raise ValueError("代理地址不能包含路径或无效主机")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("代理端口必须是 1-65535 的数字") from exc
        if port is not None:
            if not 1 <= port <= 65535:
                raise ValueError("代理端口必须是 1-65535 的数字")
            host = parsed.netloc.rsplit(":", 1)[0]
            port_str = str(port)
        else:
            if parsed.netloc.endswith(":"):
                raise ValueError("代理端口必须是 1-65535 的数字")
            host = parsed.netloc
            port_str = ""
        if not host:
            raise ValueError("代理地址无效")
        return scheme, host, port_str

    if "/" in text:
        raise ValueError("代理地址不能包含路径字符 '/'")

    userinfo, sep, host_part = text.rpartition("@")
    if host_part and ":" in host_part and not host_part.startswith("["):
        try:
            ipaddress.IPv6Address(host_part)
        except ValueError:
            pass
        else:
            formatted = f"{userinfo}@[{host_part}]" if sep else f"[{host_part}]"
            return "", formatted, ""

    parsed = urllib.parse.urlsplit(f"http://{text}")
    if not parsed.hostname or parsed.path or parsed.query or parsed.fragment:
        raise ValueError("代理地址格式无效")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("代理端口必须是 1-65535 的数字") from exc
    if port is not None:
        if not 1 <= port <= 65535:
            raise ValueError("代理端口必须是 1-65535 的数字")
        host = parsed.netloc.rsplit(":", 1)[0]
        return "", host, str(port)
    if parsed.netloc.endswith(":"):
        raise ValueError("代理端口必须是 1-65535 的数字")
    return "", parsed.netloc, ""


def apply_proxy(proxy_url: str, *, clear_existing: bool = False) -> None:
    """Export the proxy to this process and everything it spawns.

    uv and huggingface_hub both read these, so setting them here is what makes
    one console answer cover dependency install and model download alike.
    Matches `core.config.apply_network_proxy_environment`: preserves external
    shell proxy env vars when `proxy_url` is empty unless `clear_existing=True`,
    and merges loopback addresses into `NO_PROXY`/`no_proxy` when setting a proxy.
    """
    raw = str(proxy_url or "").strip()
    if raw:
        settings, normalized_url = _validate_proxy_url(raw)
        # Export the app's keys too: load_config protects process values from config/config.json.
        # In particular, an explicit --proxy must survive the hand-off to launcher.
        os.environ.update(settings)
        for key in PROXY_ENV_KEYS:
            os.environ[key] = normalized_url
        existing_no_proxy = ""
        for key in NO_PROXY_ENV_KEYS:
            existing_no_proxy = os.environ.get(key) or existing_no_proxy
        merged_no_proxy = _no_proxy_value_with_loopback(existing_no_proxy)
        for key in NO_PROXY_ENV_KEYS:
            os.environ[key] = merged_no_proxy
        return
    if clear_existing:
        os.environ.update({"PROXY_PROTOCOL": "http", "PROXY_HOST": "", "PROXY_PORT": ""})
        for key in PROXY_ENV_KEYS:
            os.environ.pop(key, None)


# --- network ----------------------------------------------------------------


def torch_wheel_url(lock_path: Path | None = None) -> str:
    """The Windows torch wheel URL recorded in `uv.lock`.

    Read as text rather than through tomllib: the lock is large, the shape of
    the wheel table has changed between uv versions, and all that is wanted is a
    representative big file to time.
    """
    try:
        text = (lock_path or LOCK_PATH).read_text(encoding="utf-8")
    except OSError:
        return ""
    block = ""
    for chunk in text.split("\n[[package]]"):
        if re.search(r'^name = "torch"$', chunk, re.MULTILINE):
            block = chunk
            break
    if not block:
        return ""
    urls = [url for url in _WHEEL_URL.findall(block) if "win_amd64" in url]
    if not urls:
        return ""
    # Prefer the ordinary build over the free-threaded one; either is the same
    # order of magnitude, but the ordinary build is what gets installed.
    ordinary = [url for url in urls if "cp314t" not in url]
    return (ordinary or urls)[0]


def _total_from_headers(response, read_fallback: int) -> int:
    content_range = response.headers.get("Content-Range", "")
    _, _, total = content_range.partition("/")
    if total.strip().isdigit():
        return int(total.strip())
    length = response.headers.get("Content-Length", "")
    if length.strip().isdigit() and response.status != 206:
        return int(length.strip())
    return read_fallback


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise OSError("SOCKS5 代理在握手完成前断开连接")
        data.extend(chunk)
    return bytes(data)


def _socks5_socket(proxy_url: str, address: tuple[str, int], timeout) -> socket.socket:
    """RFC 1928 CONNECT with proxy-side DNS; no installed packages required."""
    proxy = urllib.parse.urlsplit(proxy_url)
    host, port = address
    username = urllib.parse.unquote(proxy.username or "").encode("utf-8")
    password = urllib.parse.unquote(proxy.password or "").encode("utf-8")
    if len(username) > 255 or len(password) > 255:
        raise OSError("SOCKS5 用户名或密码过长")
    encoded_host = host.encode("idna")
    if ":" in host:
        destination = b"\x04" + socket.inet_pton(socket.AF_INET6, host)
    elif len(encoded_host) <= 255:
        destination = b"\x03" + bytes([len(encoded_host)]) + encoded_host
    else:
        raise OSError("SOCKS5 目标主机名过长")
    sock = socket.create_connection((proxy.hostname, proxy.port or 1080), timeout)
    try:
        method = 2 if proxy.username is not None else 0
        sock.sendall(bytes([5, 1, method]))
        if _recv_exact(sock, 2) != bytes([5, method]):
            raise OSError("SOCKS5 代理不接受所配置的认证方式")
        if method == 2:
            sock.sendall(b"\x01" + bytes([len(username)]) + username + bytes([len(password)]) + password)
            if _recv_exact(sock, 2) != b"\x01\x00":
                raise OSError("SOCKS5 代理认证失败")
        sock.sendall(b"\x05\x01\x00" + destination + port.to_bytes(2, "big"))
        version, status, reserved, address_type = _recv_exact(sock, 4)
        if version != 5 or reserved != 0 or status != 0:
            raise OSError(f"SOCKS5 代理连接目标失败（状态 {status}）")
        if address_type == 1:
            address_size = 4
        elif address_type == 4:
            address_size = 16
        elif address_type == 3:
            address_size = _recv_exact(sock, 1)[0]
        else:
            raise OSError("SOCKS5 代理返回无效的地址类型")
        _recv_exact(sock, address_size + 2)
        return sock
    except BaseException:
        sock.close()
        raise


class _BootstrapProxyHandler(urllib.request.ProxyHandler):
    def proxy_open(self, req, proxy, type):
        if urllib.parse.urlsplit(proxy).scheme != "socks5":
            return super().proxy_open(req, proxy, type)
        if req.host and urllib.request.proxy_bypass(req.host):
            return None

        def connection(host, **kwargs):
            cls = http.client.HTTPSConnection if type == "https" else http.client.HTTPConnection
            conn = cls(host, **kwargs)
            # Keep stdlib TLS/hostname verification; replace only the socket dialer.
            conn._create_connection = lambda address, timeout, *_args: _socks5_socket(proxy, address, timeout)
            return conn

        return urllib.request.HTTPHandler().do_open(connection, req)


def _build_proxy_handler() -> _BootstrapProxyHandler:
    all_proxy = os.environ.get("ALL_PROXY") or os.environ.get("all_proxy") or ""
    proxies: dict[str, str] = {}
    for scheme in ("http", "https"):
        value = (
            os.environ.get(f"{scheme.upper()}_PROXY")
            or os.environ.get(f"{scheme}_proxy")
            or all_proxy
        )
        if value:
            proxies[scheme] = value
    return _BootstrapProxyHandler(proxies) if proxies else _BootstrapProxyHandler()


def _opener() -> urllib.request.OpenerDirector:
    """Build a fresh opener per probe.

    urllib caches the default opener - and with it the proxy settings that were
    in the environment the first time it was used - so re-testing after the user
    enters a proxy would silently re-measure the direct connection.
    When no proxy environment variables are set, omit the explicit `{}` mapping
    so urllib keeps its OS/system proxy fallback.
    """
    return urllib.request.build_opener(_build_proxy_handler())


def measure_download(url: str = "") -> tuple[float, int] | None:
    """Return (bytes/sec, total size of the probed file), or None if unreachable.

    The clock starts at the first byte, not at connect: a cold TLS handshake to
    this host was measured at 15 s on a link that then ran at 4 MB/s, and
    charging that to throughput would recommend a proxy to someone who does not
    need one.
    """
    url = url or torch_wheel_url() or FALLBACK_PROBE_URL
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "jav-sub-setup",
            "Range": f"bytes=0-{PROBE_BYTES - 1}",
            "Accept-Encoding": "identity",
        },
    )
    read = 0
    started: float | None = None
    try:
        with _opener().open(request, timeout=PROBE_TIMEOUT_S) as response:
            while read < PROBE_BYTES:
                chunk = response.read(65536)
                if started is None:
                    started = time.monotonic()
                if not chunk:
                    break
                read += len(chunk)
                if time.monotonic() - started > PROBE_MAX_S:
                    break
            total = _total_from_headers(response, read)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if read <= 0 or started is None:
        return None
    elapsed = max(1e-6, time.monotonic() - started)
    return read / elapsed, total or FALLBACK_TOTAL_BYTES


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return "不到 1 分钟"
    if seconds < 3600:
        return f"约 {seconds / 60:.0f} 分钟"
    return f"约 {seconds / 3600:.1f} 小时"


def describe_speed(measured: tuple[float, int] | None) -> str:
    if measured is None:
        return "连接失败——直连大概率装不上，建议设置代理"
    rate, total = measured
    gib = total / 1024 ** 3
    estimate = format_duration(total / rate) if rate > 0 else "无法估算"
    return f"{rate / 1024 ** 2:.2f} MB/s（torch 约 {gib:.1f}GB，按此速度{estimate}）"


# --- uv ---------------------------------------------------------------------


def find_uv() -> Path | None:
    """Bundled copy, then a uv already on the machine, then one fetched before.

    Bundled first so a release installs with the uv it was tested against, and
    `bin/` last so that a download from a previous run is picked up without
    needing any extra path to be remembered.
    """
    suffix = ".exe" if os.name == "nt" else ""
    if (bundled := ROOT / f"uv{suffix}").is_file():
        return bundled
    if found := shutil.which("uv"):
        return Path(found)
    if (fetched := ROOT / "bin" / f"uv{suffix}").is_file():
        return fetched
    return None


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


class _SimpleIndexLinkParser(html.parser.HTMLParser):
    """Extract `<a href=...>` links and `data-yanked` flags from PEP 503 HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, bool]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = {name.lower(): value for name, value in attrs}
        href = attr_map.get("href")
        if not href:
            return
        yanked = "data-yanked" in attr_map
        self.links.append((html.unescape(href.strip()), yanked))


def _extract_fragment_sha256(fragment: str) -> str:
    raw = str(fragment or "").strip()
    if not raw:
        return ""
    for item in raw.split("&"):
        key, sep, value = item.partition("=")
        if sep and key.strip().lower() == "sha256":
            return value.strip().lower()
    return ""


def _latest_uv_wheel(
    index_html: str,
    *,
    base_url: str = UV_PYPI_INDEX,
) -> tuple[str, str]:
    """Return ``(url, sha256)`` for the newest non-yanked stable Windows uv wheel.

    Skips links carrying `data-yanked` and prerelease/non-numeric versions,
    unescapes and resolves relative URLs against `base_url`, strips any
    `#sha256=...` fragment from the returned URL, and returns the digest.
    """
    parser = _SimpleIndexLinkParser()
    parser.feed(index_html)
    parser.close()

    candidates: list[tuple[tuple[int, ...], str, str]] = []
    for href, yanked in parser.links:
        if yanked or not href:
            continue
        resolved = urllib.parse.urljoin(base_url, href)
        try:
            parsed = urllib.parse.urlsplit(resolved)
        except ValueError:
            continue
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        filename = PurePosixPath(urllib.parse.unquote(parsed.path)).name
        match = _UV_WHEEL_FILENAME_RE.fullmatch(filename)
        if not match:
            continue
        version = match.group(1)
        clean_url = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.query, "")
        )
        sha256 = _extract_fragment_sha256(parsed.fragment)
        candidates.append((_version_key(version), clean_url, sha256))

    if not candidates:
        return "", ""
    _ver_key, best_url, best_sha = max(candidates, key=lambda item: item[0])
    return best_url, best_sha


def latest_uv_wheel_url(index_html: str, *, base_url: str = UV_PYPI_INDEX) -> str:
    """Newest Windows uv wheel URL from a PyPI simple index page.

    Sorted by numeric version rather than taking the last link: the index is
    ordered by upload time, and yanked/prerelease builds are skipped.
    """
    url, _sha256 = _latest_uv_wheel(index_html, base_url=base_url)
    return url


def download_uv() -> Path | None:
    """Fetch uv from PyPI into `bin/`, reporting progress.

    Reached only when the release did not bundle uv and the machine has none.
    Runs after the proxy interview, so a user who needed a proxy for torch has
    already given one and this download uses it too.
    """
    target = ROOT / "bin" / ("uv.exe" if os.name == "nt" else "uv")
    log("\n未找到 uv，正在从 PyPI 下载（约 20MB）...")
    opener = _opener()
    try:
        with opener.open(
            urllib.request.Request(
                UV_PYPI_INDEX,
                headers={
                    "User-Agent": "jav-sub-setup",
                    "Accept": "application/vnd.pypi.simple.v1+html",
                },
            ),
            timeout=PROBE_TIMEOUT_S,
        ) as response:
            index_html = response.read().decode("utf-8", "replace")
            base_url = (
                response.geturl()
                if hasattr(response, "geturl") and callable(response.geturl)
                else UV_PYPI_INDEX
            ) or UV_PYPI_INDEX
    except (urllib.error.URLError, OSError, ValueError) as error:
        log(f"  无法读取 PyPI 索引: {error}")
        return None

    url, expected_sha = _latest_uv_wheel(index_html, base_url=base_url)
    if not url:
        log("  PyPI 索引里没有 Windows 版 uv。")
        return None

    with tempfile.TemporaryDirectory() as work:
        wheel = Path(work) / "uv.whl"
        tmp_target: Path | None = None
        try:
            actual_sha = _download_with_progress(url, wheel, opener=opener)
            if expected_sha:
                if not _UPDATE_SHA256_RE.fullmatch(expected_sha) or actual_sha != expected_sha:
                    log("  下载的 uv 校验失败（SHA-256 不匹配）。")
                    return None
            with zipfile.ZipFile(wheel) as archive:
                names = [
                    name for name in archive.namelist()
                    if name.endswith(("/uv.exe", "/uv")) and "/scripts/" in name.lower()
                ]
                if not names:
                    log("  下载的 uv 包里没有可执行文件。")
                    return None
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp_target = target.with_name(
                    f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
                )
                with archive.open(names[0]) as source, tmp_target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
                    sink.flush()
                    os.fsync(sink.fileno())
                tmp_target.chmod(0o755)
                _replace_atomically(tmp_target, target)
        except (urllib.error.URLError, OSError, ValueError, zipfile.BadZipFile) as error:
            log(f"  下载失败: {error}")
            return None
        finally:
            if tmp_target is not None:
                try:
                    tmp_target.unlink(missing_ok=True)
                except OSError:
                    pass

    log(f"  已安装 uv: {target}")
    return target


def _download_with_progress(
    url: str,
    destination: Path,
    *,
    opener=None,
    headers: dict[str, str] | None = None,
    timeout: float = PROBE_TIMEOUT_S,
) -> str:
    req_headers = {"User-Agent": "jav-sub-setup"}
    if headers:
        req_headers.update(headers)
    request = urllib.request.Request(url, headers=req_headers)
    active_opener = opener if opener is not None else _opener()
    digest = hashlib.sha256()
    with active_opener.open(request, timeout=timeout) as response:
        resp_headers = getattr(response, "headers", None)
        length = resp_headers.get("Content-Length", "") if resp_headers is not None else ""
        total = int(length) if str(length).strip().isdigit() else 0
        read = 0
        with open(destination, "wb") as sink:
            while chunk := response.read(262144):
                sink.write(chunk)
                digest.update(chunk)
                read += len(chunk)
                if total:
                    print(f"\r  {read * 100 // total}%", end="", flush=True)
    if total:
        print("\r  100%", flush=True)
    return digest.hexdigest()


def ensure_uv() -> Path | None:
    return find_uv() or download_uv()


def venv_python() -> Path:
    return VENV_PATH / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run_streaming(command: list[str]) -> int:
    """Run a command with its output going straight to this console.

    Deliberately not captured: the point of the console window is that the user
    watches uv's own download progress instead of a spinner that hides it.
    """
    log(f"\n$ {' '.join(str(part) for part in command)}\n")
    try:
        return subprocess.call([str(part) for part in command], cwd=str(ROOT), env=os.environ.copy())
    except OSError as error:
        log(f"命令无法执行: {error}")
        return 1


def _read_toml(path: Path) -> tuple[bytes, dict | None]:
    try:
        raw = path.read_bytes()
    except OSError:
        return b"<missing>", None
    try:
        return raw, tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        return raw, None


def lock_digest() -> str:
    """Identity of the dependency set, so a patched release re-syncs itself.

    The project's own version number is left out. Every release bumps it in
    pyproject.toml and in the lock's entry for the project, yet uv records the
    project as virtual - no build backend, never installed into .venv - so the
    number changes nothing there. Hashing it made every upgrade look like a
    dependency change: a sync with nothing to install, preceded by the speed
    test and the proxy question. The number is only left out when the lock says
    the project is virtual; a missing or unreadable file is hashed as it is,
    and the worst a changed digest costs is one sync.
    """
    lock_raw, lock = _read_toml(LOCK_PATH)
    project_raw, project = _read_toml(ROOT / "pyproject.toml")
    roots = [
        package
        for package in (lock or {}).get("package") or []
        if isinstance(package, dict) and package.get("source") == {"virtual": "."}
    ]
    digest = hashlib.sha256()
    if lock is None or project is None or len(roots) != 1:
        digest.update(lock_raw)
        digest.update(project_raw)
        return digest.hexdigest()
    roots[0].pop("version", None)
    if isinstance(project.get("project"), dict):
        project["project"].pop("version", None)
    for data in (lock, project):
        digest.update(json.dumps(data, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


_PYVENV_PATH_KEYS = ("home", "base-prefix", "base-exec-prefix", "base-executable")
_VENV_REBUILD_MARKER_NAME = ".jav-sub-venv-rebuild-required"


def _venv_dir(root: Path | None = None) -> Path:
    if root is None:
        return VENV_PATH
    return root / ".venv"


def _venv_rebuild_marker(root: Path | None = None) -> Path:
    return _venv_dir(root) / _VENV_REBUILD_MARKER_NAME


def _legacy_tmp_python_dir(root: Path | None = None) -> Path:
    return ((root or ROOT) / "tmp" / "python").resolve()


def _path_within(candidate: Path, base: Path) -> bool:
    try:
        resolved = candidate.resolve()
        return resolved == base or resolved.is_relative_to(base)
    except (OSError, ValueError):
        return False


def venv_uses_legacy_tmp_python(root: Path | None = None) -> bool:
    """True when `.venv/pyvenv.cfg` still references the legacy `tmp/python` tree."""
    cfg_path = _venv_dir(root) / "pyvenv.cfg"
    if not cfg_path.is_file():
        return False
    legacy_dir = _legacy_tmp_python_dir(root)
    try:
        text = cfg_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return False
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, raw_value = line.partition("=")
        if key.strip().lower() not in _PYVENV_PATH_KEYS:
            continue
        value = raw_value.strip()
        if value and _path_within(Path(value), legacy_dir):
            return True
    return False


def venv_needs_rebuild(root: Path | None = None) -> bool:
    """True when `.venv` must be removed and recreated before launching."""
    return venv_uses_legacy_tmp_python(root) or _venv_rebuild_marker(root).is_file()


def _mark_venv_needs_rebuild(root: Path | None = None, reason: str = "") -> None:
    """Persist rebuild requirement inside `.venv` and invalidate the install stamp."""
    venv_dir = _venv_dir(root)
    marker = _venv_rebuild_marker(root)
    try:
        if venv_dir.is_dir():
            marker.write_text((reason or "legacy-python-migration-failed") + "\n", encoding="utf-8")
    except OSError:
        pass
    stamp_candidates = [venv_dir / STAMP_PATH.name]
    try:
        if root is None or venv_dir.resolve() == VENV_PATH.resolve():
            stamp_candidates.append(STAMP_PATH)
    except OSError:
        pass
    for stamp in stamp_candidates:
        try:
            stamp.unlink(missing_ok=True)
        except OSError:
            pass


_VENV_PENDING_VERIFICATION_TAG = "pending-verification"


def _mark_venv_pending_verification(root: Path | None = None) -> None:
    """Persist pending-verification marker before rewriting `pyvenv.cfg`.

    Leaves `STAMP_PATH` intact so a successful verification remains a zero-sync
    upgrade, while any crash or interruption before verification completes keeps
    `venv_needs_rebuild()` True and `environment_is_current()` False.
    """
    marker = _venv_rebuild_marker(root)
    marker.write_text(f"{_VENV_PENDING_VERIFICATION_TAG}\n", encoding="utf-8")


def _clear_venv_rebuild_marker(root: Path | None = None) -> None:
    try:
        _venv_rebuild_marker(root).unlink(missing_ok=True)
    except OSError:
        pass


def _pyvenv_paths_exist(cfg_path: Path) -> bool:
    try:
        text = cfg_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return False
    saw_path = False
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, raw_value = line.partition("=")
        if key.strip().lower() not in _PYVENV_PATH_KEYS:
            continue
        value = raw_value.strip()
        if not value:
            continue
        saw_path = True
        if not Path(value).exists():
            return False
    return saw_path


def _verify_python_executable(python_exe: Path) -> bool:
    if not python_exe.is_file():
        return False
    try:
        completed = subprocess.run(
            [str(python_exe), "-c", "import sys"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def migrate_legacy_python_runtime(
    root: Path | None = None,
    *,
    verify_python=None,
) -> str:
    """Relocate a 1.8.x `.venv` bound to `tmp/python` onto `.python`.

    Returns ``"noop"`` if `.venv` does not reference `tmp/python` (or in a git
    checkout), ``"migrated"`` if `tmp/python` was moved to `.python` and the
    relocated interpreter verified healthy, or ``"needs_rebuild"`` if the venv
    must be recreated by `uv sync`. Persists a pending-verification marker
    *before* rewriting `pyvenv.cfg` and clears it only after verification
    succeeds so a crash before verification cannot bypass rebuild or checking.
    """
    base_root = root or ROOT
    if (base_root / ".git").exists():
        return "noop"
    venv_dir = _venv_dir(root)
    cfg_path = venv_dir / "pyvenv.cfg"
    python_exe = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    checker = verify_python if verify_python is not None else _verify_python_executable

    marker = _venv_rebuild_marker(root)
    if marker.is_file():
        try:
            marker_state = marker.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            marker_state = ""
        if marker_state == _VENV_PENDING_VERIFICATION_TAG and not venv_uses_legacy_tmp_python(root):
            if _pyvenv_paths_exist(cfg_path) and checker(python_exe):
                _clear_venv_rebuild_marker(root)
                return "migrated"
            _mark_venv_needs_rebuild(root, "relocated python executable verification failed")
            return "needs_rebuild"
        return "needs_rebuild"

    if not venv_uses_legacy_tmp_python(root):
        return "noop"

    legacy_dir = _legacy_tmp_python_dir(base_root)
    new_dir = (base_root / ".python").resolve()

    if not legacy_dir.is_dir():
        _mark_venv_needs_rebuild(root, "legacy tmp/python directory missing")
        return "needs_rebuild"

    try:
        if not new_dir.exists():
            shutil.move(str(legacy_dir), str(new_dir))
        else:
            for child in sorted(legacy_dir.iterdir()):
                dest = new_dir / child.name
                if not dest.exists():
                    shutil.move(str(child), str(dest))
            if legacy_dir.is_dir() and not any(legacy_dir.iterdir()):
                legacy_dir.rmdir()
    except OSError as error:
        log(f"  无法迁移旧版 Python 运行时到 .python: {error}")
        _mark_venv_needs_rebuild(root, f"move failed: {error}")
        return "needs_rebuild"

    try:
        lines = cfg_path.read_text(encoding="utf-8-sig").splitlines(keepends=True)
        rewritten: list[str] = []
        rewritten_targets: list[Path] = []
        for line in lines:
            if "=" not in line:
                rewritten.append(line)
                continue
            key, sep, raw_value = line.partition("=")
            if key.strip().lower() not in _PYVENV_PATH_KEYS:
                rewritten.append(line)
                continue
            value_stripped = raw_value.strip()
            if not value_stripped:
                rewritten.append(line)
                continue
            candidate = Path(value_stripped).resolve()
            if _path_within(candidate, legacy_dir):
                rel = candidate.relative_to(legacy_dir)
                updated_path = new_dir / rel
                rewritten_targets.append(updated_path)
                eol = "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else "")
                rewritten.append(f"{key.strip()} {sep} {updated_path}{eol}")
            else:
                rewritten.append(line)
        temporary = cfg_path.with_name(f"{cfg_path.name}.{os.getpid()}.tmp")
        temporary.write_text("".join(rewritten), encoding="utf-8")
        # Persist pending-verification BEFORE replacing pyvenv.cfg so a crash
        # before verification finishes cannot leave a rewritten venv looking ready.
        _mark_venv_pending_verification(root)
        os.replace(temporary, cfg_path)
    except OSError as error:
        log(f"  无法更新 .venv/pyvenv.cfg: {error}")
        _mark_venv_needs_rebuild(root, f"pyvenv.cfg rewrite failed: {error}")
        return "needs_rebuild"

    if any(not target.exists() for target in rewritten_targets):
        _mark_venv_needs_rebuild(root, "relocated python target missing")
        return "needs_rebuild"

    if not checker(python_exe):
        _mark_venv_needs_rebuild(root, "relocated python executable verification failed")
        return "needs_rebuild"
    _clear_venv_rebuild_marker(root)
    return "migrated"


def environment_is_current() -> bool:
    """Whether the installed environment matches the shipped lockfile.

    A stamp rather than an import probe: importing torch costs seconds and this
    runs on every launch. The stamp is written only after the imports have been
    verified once, so an interrupted sync leaves no stamp and gets redone.
    """
    if not venv_python().is_file():
        return False
    if venv_needs_rebuild():
        return False
    try:
        return STAMP_PATH.read_text(encoding="utf-8").strip() == lock_digest()
    except OSError:
        return False


_PROBE_PREFIX = "IMPORT-FAIL "


def probe_imports(names: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """Actually import each name inside the installed environment.

    The ground truth, and the only check that catches a half-deleted package: a
    dist-info left behind by a deleted `torch/lib` still satisfies `uv sync`,
    which audits metadata rather than files. Injects `ROOT/bin` into PATH and
    Windows DLL search paths so bundled FFmpeg shared DLLs are visible, and
    exercises `torchcodec._core.get_ffmpeg_library_versions()` to verify real
    DLL loading. Returns [(import name, reason)] for everything that failed.
    """
    targets = list(names) or sorted(CRITICAL_IMPORT_ROOTS)
    bin_dir = ROOT / "bin"
    env = os.environ.copy()
    if bin_dir.is_dir():
        bin_str = str(bin_dir)
        current_path = env.get("PATH", "")
        if bin_str not in current_path.split(os.pathsep):
            env["PATH"] = bin_str + (os.pathsep + current_path if current_path else "")
    script = (
        "import importlib, os\n"
        f"bin_dir = {str(bin_dir)!r}\n"
        "_dll_handles = []\n"
        "if os.name == 'nt' and os.path.isdir(bin_dir) and hasattr(os, 'add_dll_directory'):\n"
        "    try:\n"
        "        _dll_handles.append(os.add_dll_directory(bin_dir))\n"
        "    except OSError:\n"
        "        pass\n"
        f"for name in {targets!r}:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "        if name == 'torchcodec':\n"
        "            from torchcodec._core import get_ffmpeg_library_versions\n"
        "            versions = get_ffmpeg_library_versions()\n"
        "            if not isinstance(versions, dict) or not versions:\n"
        "                raise RuntimeError('torchcodec failed to load FFmpeg shared libraries')\n"
        "    except BaseException as exc:\n"
        f"        print('{_PROBE_PREFIX}' + name + ': '"
        " + type(exc).__name__ + ': ' + str(exc).replace(chr(10), ' '), flush=True)\n"
    )
    try:
        completed = subprocess.run(
            [str(venv_python()), "-c", script],
            cwd=str(ROOT), capture_output=True, env=env, timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return [(name, f"无法运行 .venv 里的 python: {error}") for name in targets]

    failures: list[tuple[str, str]] = []
    for line in completed.stdout.decode("utf-8", "replace").splitlines():
        if not line.startswith(_PROBE_PREFIX):
            continue
        name, _, reason = line[len(_PROBE_PREFIX):].partition(": ")
        failures.append((name.strip(), reason.strip()))
    if completed.returncode != 0 and not failures:
        # The interpreter itself died - a corrupt python3xx.dll, say.
        detail = completed.stderr.decode("utf-8", "replace").strip()[-800:]
        failures.append(("python", detail or f"退出码 {completed.returncode}"))
    return failures


def imports_work() -> bool:
    failures = probe_imports(("torch", "transformers"))
    for name, reason in failures:
        log(f"  {name}: {reason}")
    return not failures


# --- environment integrity --------------------------------------------------


def site_packages_dir() -> Path:
    if os.name == "nt":
        return VENV_PATH / "Lib" / "site-packages"
    found = sorted((VENV_PATH / "lib").glob("python3.*/site-packages"))
    return found[0] if found else VENV_PATH / "lib" / "site-packages"


def missing_import_roots() -> list[str]:
    """Critical import roots that are not on disk. Cheap enough for every run."""
    root = site_packages_dir()
    if not root.is_dir():
        return sorted(CRITICAL_IMPORT_ROOTS)
    return [
        name
        for name in sorted(CRITICAL_IMPORT_ROOTS)
        if not (root / name).is_dir() and not (root / f"{name}.py").is_file()
    ]


def distributions_for(import_roots) -> list[str]:
    return sorted(
        {
            CRITICAL_IMPORT_ROOTS[name]
            for name in import_roots
            if name in CRITICAL_IMPORT_ROOTS
        }
    )


def missing_payload_paths() -> list[str]:
    return [name for name in PAYLOAD_PATHS if not (ROOT / name).exists()]


def write_release_index(payload: Path) -> Path:
    """Record every file of a staged release, for `prune_stale_release_files`.

    The build calls this on the staged payload before hashing it, so the index
    travels in the archive and appears in the release manifest like any file.
    """
    index = payload / RELEASE_INDEX_NAME
    files = sorted(
        path.relative_to(payload).as_posix()
        for path in payload.rglob("*")
        if path.is_file() and path != index
    )
    index.write_text(
        json.dumps({"schema": RELEASE_INDEX_SCHEMA, "files": files}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return index


def _parse_release_index_text(text: str) -> frozenset[str] | None:
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("schema") != RELEASE_INDEX_SCHEMA:
        return None
    files = data.get("files")
    if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
        return None
    shipped = frozenset(files)
    if "launcher.py" not in shipped or not any(item.startswith("src/") for item in shipped):
        return None
    return shipped


def read_release_index(root: Path) -> frozenset[str] | None:
    """The files this release shipped, or None when there is no usable index.

    An index that does not list launcher.py and something under src/ is treated
    as absent: every path it leaves out counts as stale, so a truncated one would
    otherwise empty src/.
    """
    try:
        text = (root / RELEASE_INDEX_NAME).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return _parse_release_index_text(text)


# FFmpeg Shared is all a release puts in bin/, and only names that are
# recognisably FFmpeg's are the release's to remove. Other tools live there too -
# the uv the first run downloads, a llama.cpp build with its own DLLs - and
# moving half of one breaks it.
_FFMPEG_BINARY = re.compile(
    r"(?:avcodec|avdevice|avfilter|avformat|avutil|postproc|swresample|swscale)-\d+\.dll"
    r"|ffmpeg\.exe|ffprobe\.exe|ffplay\.exe",
    re.IGNORECASE,
)


def _is_release_binary(name: str) -> bool:
    return _FFMPEG_BINARY.fullmatch(name) is not None


# Retired local model caches shipped by earlier releases of jav-sub.
# Only explicitly retired repo families are matched here (never unknown user
# folders), and anything actively referenced in `config/config.json` is always spared.
_RETIRED_HUB_MODEL_DIR = re.compile(
    r"^models--(?:"
    r"tencent--hy-mt(?:1\.5|2)-.*"
    r"|sakurallm--.*"
    r"|qwen--qwen3(?:\.5)?-(?:4b|8b|14b).*"
    r"|jaykwok--qwen3-asr-0\.6b.*"
    r")$",
    re.IGNORECASE,
)
_RETIRED_FLAT_MODEL_DIR = re.compile(
    r"^(?:"
    r"tencent-hy-mt(?:1\.5|2)-.*"
    r"|sakurallm-.*"
    r"|qwen-qwen3(?:\.5)?-(?:4b|8b|14b).*"
    r"|jaykwok-qwen3-asr-0\.6b.*"
    r")$",
    re.IGNORECASE,
)


_MODEL_PROTECT_KEYS = (
    "LLAMACPP_MODEL_REPO",
    "LLAMACPP_GGUF_PATH",
    "ASR_BACKEND",
    "ASR_MODEL_ID",
    "ASR_MODEL_PATH",
)


def _active_model_keys(
    root: Path,
    *env_sources: dict[str, str],
) -> tuple[set[str], set[Path]]:
    """Repo slugs and explicit paths configured in `config/config.json` or process env."""
    repos: set[str] = {
        "tencent/Hy-MT3-7B-GGUF",
        "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    }
    protected_paths: set[Path] = set()
    for env in env_sources:
        for repo_key in ("LLAMACPP_MODEL_REPO", "ASR_BACKEND", "ASR_MODEL_ID"):
            value = (env.get(repo_key) or "").strip()
            if value:
                repos.add(value)
        for path_key in ("LLAMACPP_GGUF_PATH", "ASR_MODEL_PATH"):
            raw = (env.get(path_key) or "").strip()
            if not raw:
                continue
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            try:
                protected_paths.add(candidate.resolve())
            except OSError:
                pass

    protected_names: set[str] = set()
    for repo in repos:
        cleaned = repo.strip("/")
        if not cleaned:
            continue
        protected_names.add(f"models--{cleaned.replace('/', '--')}".casefold())
        protected_names.add(cleaned.replace("/", "-").casefold())

    return protected_names, protected_paths


def _path_is_protected(candidate: Path, protected_paths: set[Path]) -> bool:
    if not protected_paths:
        return False
    try:
        resolved = candidate.resolve()
    except OSError:
        return False
    for target in protected_paths:
        if target == resolved or target.is_relative_to(resolved):
            return True
    return False


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for current, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(current) / name).lstat().st_size
            except OSError:
                pass
    return total


def retired_model_cache_paths(root: Path, env: dict[str, str] | None = None) -> list[str]:
    """Retired local model cache directories under `models/` safe to reclaim on upgrade."""
    models_dir = root / "models"
    if not models_dir.is_dir():
        return []
    if env is None:
        config_path = root / "config" / "config.json"
        try:
            config_values = _read_config_file_strict(config_path)
        except (OSError, ValueError) as error:
            log(f"  无法读取 {config_path.name}（{error}），已跳过旧模型缓存清理以保护当前配置的模型")
            return []
        process_values = {
            key: os.environ[key]
            for key in _MODEL_PROTECT_KEYS
            if key in os.environ and os.environ[key].strip()
        }
        protected_names, protected_paths = _active_model_keys(root, config_values, process_values)
    else:
        protected_names, protected_paths = _active_model_keys(root, env)

    retired: list[str] = []
    hub_dir = models_dir / "hub"
    if hub_dir.is_dir():
        for entry in sorted(hub_dir.iterdir()):
            if not entry.is_dir():
                continue
            if not _RETIRED_HUB_MODEL_DIR.fullmatch(entry.name):
                continue
            if entry.name.casefold() in protected_names or _path_is_protected(entry, protected_paths):
                continue
            retired.append(f"models/hub/{entry.name}")
    for entry in sorted(models_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not _RETIRED_FLAT_MODEL_DIR.fullmatch(entry.name):
            continue
        if entry.name.casefold() in protected_names or _path_is_protected(entry, protected_paths):
            continue
        retired.append(f"models/{entry.name}")
    return retired


def prune_retired_model_caches(root: Path | None = None) -> tuple[list[str], int]:
    """Remove retired local model caches left behind after an in-place upgrade.

    Runs only for packaged upgrades (not git checkouts, requires a valid release
    index) and only deletes explicitly retired model families not referenced by
    `config/config.json`, reclaiming multi-GB disk space that moving to
    `tmp/upgrade-leftovers/` would leave occupied. Failure on any entry is logged
    and skipped.
    """
    root = root or ROOT
    if (root / ".git").exists():
        return [], 0
    if read_release_index(root) is None:
        return [], 0
    candidates = retired_model_cache_paths(root)
    if not candidates:
        return [], 0
    removed: list[str] = []
    reclaimed_bytes = 0
    for relative in candidates:
        path = root / relative
        size = _dir_size_bytes(path)
        try:
            shutil.rmtree(path)
        except OSError as error:
            log(f"  无法清理旧模型缓存 {relative}: {error}")
            continue
        removed.append(relative)
        reclaimed_bytes += size
        if relative.startswith("models/hub/"):
            lock_dir = root / "models" / "hub" / ".locks" / path.name
            if lock_dir.is_dir():
                shutil.rmtree(lock_dir, ignore_errors=True)
    return removed, reclaimed_bytes


def prune_legacy_env_file(root: Path | None = None) -> bool:
    """Remove legacy `.env` left behind by older releases after an in-place upgrade.

    Runs only for packaged upgrades (not git checkouts, requires a valid release
    index). Settings now live in `config/config.json` and old `.env` files are
    intentionally not migrated.
    """
    root = root or ROOT
    if (root / ".git").exists():
        return False
    if read_release_index(root) is None:
        return False
    legacy_env = root / ".env"
    if not legacy_env.is_file():
        return False
    try:
        legacy_env.unlink()
    except OSError as error:
        log(f"  无法清理旧版配置文件 .env: {error}")
        return False
    return True


def _path_key(relative: str) -> str:
    # Windows paths ignore case, and copying src/llm/ over an old src/LLM/ writes
    # into the existing folder under its old spelling. A case-sensitive match
    # would then move the new release's own files out. Folding on every platform
    # can only err towards keeping a file.
    return relative.casefold()


def stale_release_paths(root: Path, shipped: frozenset[str]) -> list[str]:
    """Paths under src/ and bin/ that this release did not ship.

    src/ belongs to the release entirely. An unlisted directory goes as a whole,
    bytecode cache included, because a leftover package directory would shadow a
    new module of the same name. An unlisted file goes on its own. Bytecode
    caches inside shipped directories stay: Python never imports a cached module
    whose source is gone. Names are compared the way Windows compares them.
    """
    shipped_files = {_path_key(item) for item in shipped}
    shipped_dirs = {
        _path_key(parent.as_posix()) for item in shipped for parent in PurePosixPath(item).parents
    }
    stale: list[str] = []
    for current, dirs, files in os.walk(root / "src"):
        here = Path(current).relative_to(root).as_posix()
        descend = []
        for name in sorted(dirs):
            if name.casefold() == "__pycache__":
                continue
            relative = f"{here}/{name}"
            if _path_key(relative) in shipped_dirs:
                descend.append(name)
            else:
                stale.append(relative)
        dirs[:] = descend
        stale.extend(
            f"{here}/{name}"
            for name in sorted(files)
            if _path_key(f"{here}/{name}") not in shipped_files
        )
    bin_dir = root / "bin"
    if bin_dir.is_dir():
        for path in sorted(bin_dir.iterdir()):
            relative = f"bin/{path.name}"
            if (
                path.is_file()
                and _is_release_binary(path.name)
                and _path_key(relative) not in shipped_files
            ):
                stale.append(relative)
    return stale


def prune_stale_release_files(root: Path | None = None) -> tuple[list[str], Path | None]:
    """Move what an older release left in src/ and bin/ out of the way.

    This is what makes extracting a new release over the old folder a complete
    upgrade. Nothing happens in a git checkout (src/ is the developer's) or
    without an index (an older archive). A path that cannot be moved is
    reported and skipped: a leftover costs less than a launcher that refuses to
    start.
    """
    root = root or ROOT
    if (root / ".git").exists():
        return [], None
    shipped = read_release_index(root)
    if shipped is None:
        return [], None
    stale = stale_release_paths(root, shipped)
    if not stale:
        return [], None
    base = root / LEFTOVERS_DIR / time.strftime("%Y%m%d_%H%M%S")
    target, attempt = base, 1
    while target.exists():
        target = base.with_name(f"{base.name}-{attempt}")
        attempt += 1
    moved = []
    for relative in stale:
        destination = target / relative
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root / relative), str(destination))
        except OSError as error:
            log(f"  无法移走旧版本文件 {relative}: {error}")
            continue
        moved.append(relative)
    return moved, (target if moved else None)


def enclosing_install(root: Path | None = None) -> Path | None:
    """The older install this folder was extracted into, if the parent is one.

    Extracting a release *into* the old folder instead of over it nests a second
    copy inside the first, which then installs a fresh environment and downloads
    every model again while the old ones sit one level up.
    """
    root = root or ROOT
    parent = root.parent
    if parent == root:
        return None
    looks_installed = (parent / ".venv").is_dir() or (parent / "models").is_dir()
    if (parent / "launcher.py").is_file() and (parent / "src").is_dir() and looks_installed:
        return parent
    return None


# --- online self-update (stdlib only) ---------------------------------------


def parse_semver(raw: str | None) -> tuple[int, int, int] | None:
    """Parse strict `[v]x.y.z` stable release versions; reject prereleases."""
    if not isinstance(raw, str):
        return None
    match = _UPDATE_SEMVER_RE.fullmatch(raw.strip())
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def current_project_version(root: Path | None = None) -> str:
    base = root or ROOT
    _, data = _read_toml(base / "pyproject.toml")
    project = (data or {}).get("project") if isinstance(data, dict) else None
    version = project.get("version") if isinstance(project, dict) else ""
    return version.strip() if isinstance(version, str) and version.strip() else "0.0.0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_update_relpath(raw: str) -> PurePosixPath:
    if "\x00" in raw:
        raise ValueError("更新包路径包含非法空字符")
    normalized = raw.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"更新包包含绝对路径：{raw}")
    raw_trimmed = normalized[:-1] if normalized.endswith("/") else normalized
    if not raw_trimmed:
        raise ValueError(f"更新包包含空路径或越界路径：{raw}")
    for part in raw_trimmed.split("/"):
        if part in {".", "..", ""}:
            raise ValueError(f"更新包包含越界路径：{raw}")
        if ":" in part:
            raise ValueError(f"更新包路径包含非法字符 ':'：{raw}")
        if part.endswith(" ") or part.endswith("."):
            raise ValueError(f"更新包路径段末尾包含非法空格或点号：{raw}")
        stem = part.split(".", 1)[0].rstrip(" ").upper()
        if stem in _WINDOWS_RESERVED_DEVICE_NAMES:
            raise ValueError(f"更新包包含 Windows 保留设备名路径：{raw}")
    pure = PurePosixPath(raw_trimmed)
    if not pure.parts or any(part in {".", "..", ""} for part in pure.parts):
        raise ValueError(f"更新包包含越界路径：{raw}")
    return pure


def _parse_sha256_sidecar(text: str, expected_filename: str = UPDATE_ARCHIVE_NAME) -> str | None:
    expected_cf = expected_filename.casefold()
    bare: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _UPDATE_SHA256_LINE_RE.fullmatch(stripped)
        if not match:
            continue
        digest = match.group(1).lower()
        name = (match.group(2) or "").strip()
        if name:
            if Path(name).name.casefold() == expected_cf:
                return digest
        elif bare is None:
            bare = digest
    return bare


def _allowed_update_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(url.strip())
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "https" and bool(host):
        return True
    if parsed.scheme == "http" and host in _LOOPBACK_HOSTS:
        return True
    return False


def _is_host_in_suffixes(host: str, suffixes: tuple[str, ...]) -> bool:
    normalized = host.rstrip(".").lower()
    for suffix in suffixes:
        if normalized == suffix or normalized.endswith(f".{suffix}"):
            return True
    return False


def _is_allowed_update_redirect(origin_url: str, target_url: str) -> bool:
    if not _allowed_update_url(target_url):
        return False
    try:
        origin = urllib.parse.urlsplit(origin_url.strip())
        target = urllib.parse.urlsplit(target_url.strip())
    except ValueError:
        return False
    if origin.scheme == "https" and target.scheme != "https":
        return False
    origin_host = (origin.hostname or "").rstrip(".").lower()
    target_host = (target.hostname or "").rstrip(".").lower()
    if not target_host:
        return False
    if target_host == origin_host or (
        origin_host and target_host.endswith(f".{origin_host}")
    ):
        return True
    if _is_host_in_suffixes(target_host, _GITHUB_ALLOWED_REDIRECT_SUFFIXES):
        return True
    return False


class _SafeUpdateRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject HTTPS downgrades and untrusted redirect targets during self-update."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        origin_url = req.full_url
        target_url = urllib.parse.urljoin(origin_url, str(newurl).strip())
        if not _is_allowed_update_redirect(origin_url, target_url):
            raise ValueError(
                f"更新下载重定向目标不受信任或试图协议降级（{target_url}），已拒绝。"
            )
        return super().redirect_request(req, fp, code, msg, headers, target_url)


def _update_opener():
    opener = _opener()
    if not isinstance(opener, urllib.request.OpenerDirector):
        return opener
    return urllib.request.build_opener(_build_proxy_handler(), _SafeUpdateRedirectHandler())


def _update_api_url() -> str:
    explicit = os.environ.get("JAV_SUB_UPDATE_API_URL", "").strip()
    if explicit:
        return explicit
    repo = os.environ.get("JAV_SUB_UPDATE_REPO", DEFAULT_UPDATE_REPO).strip().strip("/")
    return f"https://api.github.com/repos/{repo or DEFAULT_UPDATE_REPO}/releases/latest"


def _try_lock_file(handle) -> bool:
    """Non-blocking exclusive OS lock compatible with `launcher.py` & `core.atomic_io`."""
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock_and_close(handle) -> None:
    if handle is None:
        return
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        try:
            handle.close()
        except OSError:
            pass


@contextlib.contextmanager
def _workspace_gate(root: Path, *, wait_s: float = 5.0, poll_s: float = 0.05):
    """Acquire the shared OS mutex `tmp/launcher/workspace.gate`.

    Yields `True` while the lock is held, or `False` if busy past `wait_s`.
    Matches `launcher.py` and `core/atomic_io.py` so self-update never races
    instance registration or exit temp sweeps.
    """
    try:
        directory = root / "tmp" / _LAUNCHER_INSTANCE_DIR_NAME
        directory.mkdir(parents=True, exist_ok=True)
        handle = (directory / _WORKSPACE_GATE_NAME).open("a+", encoding="utf-8")
    except OSError:
        yield False
        return

    deadline = time.monotonic() + max(0.0, float(wait_s))
    sleep_s = max(0.005, float(poll_s))
    acquired = False
    while True:
        if _try_lock_file(handle):
            acquired = True
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(sleep_s, remaining))

    if not acquired:
        try:
            handle.close()
        except OSError:
            pass
        yield False
        return

    try:
        yield True
    finally:
        _unlock_and_close(handle)


def _launcher_slot_is_held(path: Path) -> bool:
    if os.name == "nt":
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except OSError:
            return True
        return False
    try:
        import fcntl

        with path.open("r+", encoding="utf-8") as probe:
            try:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
        path.unlink(missing_ok=True)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return False


def _scan_launcher_slots_held(root: Path) -> bool:
    instance_dir = root / "tmp" / _LAUNCHER_INSTANCE_DIR_NAME
    if not instance_dir.is_dir():
        return False
    try:
        slots = sorted(instance_dir.glob("instance-*.lock"))
    except OSError:
        return True
    return any(_launcher_slot_is_held(slot) for slot in slots)


def _other_launcher_instances_active(root: Path, *, wait_s: float = 10.0) -> bool:
    instance_dir = root / "tmp" / _LAUNCHER_INSTANCE_DIR_NAME
    if not instance_dir.is_dir():
        return False
    deadline = time.monotonic() + max(0.0, wait_s)
    while True:
        remaining = max(0.0, deadline - time.monotonic())
        gate_wait = min(remaining, 0.2) if remaining > 0 else 0.0
        with _workspace_gate(root, wait_s=gate_wait) as acquired:
            if acquired and not _scan_launcher_slots_held(root):
                return False
        if time.monotonic() >= deadline:
            return True
        time.sleep(min(0.2, max(0.01, deadline - time.monotonic())))


def _unlink_with_retry(path: Path, *, attempts: int = 8) -> bool:
    for attempt in range(max(1, attempts)):
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            if attempt >= attempts - 1:
                return False
            time.sleep(0.05 * (attempt + 1))
        except OSError:
            return False
    return not path.exists()


def _copy_file_atomic(source: Path, destination: Path, *, attempts: int = 8) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        shutil.copy2(source, temporary)
        for attempt in range(attempts):
            try:
                os.replace(temporary, destination)
                return
            except PermissionError:
                if attempt >= attempts - 1:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _replace_running_exe(staged_exe: Path, target_exe: Path) -> None:
    """Replace `jav-sub.exe` safely on Windows while this process may be running from it.

    Windows refuses direct overwrite of a running executable, but allows
    renaming it out of the way (`jav-sub.exe.old`) and placing the new
    executable at the original path. The next launch deletes `.old`.
    """
    backup: Path | None = None
    if target_exe.exists():
        candidate = target_exe.with_name(f"{target_exe.name}.old")
        if candidate.exists():
            try:
                candidate.unlink()
            except OSError:
                candidate = target_exe.with_name(f"{target_exe.name}.old.{os.getpid()}")
        for attempt in range(8):
            try:
                os.replace(target_exe, candidate)
                backup = candidate
                break
            except PermissionError:
                if attempt >= 7:
                    raise
                time.sleep(0.05 * (attempt + 1))
    try:
        _copy_file_atomic(staged_exe, target_exe)
    except BaseException:
        if backup is not None and backup.exists() and not target_exe.exists():
            try:
                os.replace(backup, target_exe)
            except OSError:
                pass
        raise


def prune_old_launcher_backups(root: Path | None = None) -> list[str]:
    """Delete `.old` exe backups from prior self-updates and any stale `tmp/updates/`."""
    base = root or ROOT
    removed: list[str] = []
    for candidate in sorted(base.glob("jav-sub.exe.old*")):
        if not candidate.is_file():
            continue
        # Right after an online update relaunch, the previous jav-sub.exe may
        # still be tearing down for a fraction of a second. Retry briefly only
        # when a backup actually exists so normal launches pay zero delay.
        for attempt in range(5):
            try:
                candidate.unlink()
                removed.append(candidate.name)
                break
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.2)
            except OSError:
                break
    updates_root = base / UPDATES_DIR
    pending = updates_root / PENDING_UPDATE_FILE_NAME
    if (
        updates_root.is_dir()
        and not pending.exists()
        and not _other_launcher_instances_active(base, wait_s=0.0)
    ):
        shutil.rmtree(updates_root, ignore_errors=True)
        if not updates_root.exists():
            removed.append(UPDATES_DIR.as_posix())
    return removed


def _read_and_validate_staged_update(
    base_root: Path,
) -> tuple[str, dict | None, Path | None, list[str]]:
    updates_root = (base_root / UPDATES_DIR).resolve()
    pending_path = updates_root / PENDING_UPDATE_FILE_NAME
    try:
        pending_text = pending_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "none", None, None, []
    except OSError as exc:
        log(f"  暂时无法读取待应用更新凭证（文件可能正被占用），已保留暂存更新包并停止本次启动：{exc}")
        return "failed", None, None, []
    except UnicodeDecodeError as exc:
        log(f"  待应用更新凭证编码无效，已清理：{exc}")
        shutil.rmtree(updates_root, ignore_errors=True)
        return "invalid", None, None, []
    try:
        descriptor = json.loads(pending_text)
    except ValueError as exc:
        log(f"  待应用更新凭证无法解析，已清理：{exc}")
        shutil.rmtree(updates_root, ignore_errors=True)
        return "invalid", None, None, []
    if not isinstance(descriptor, dict) or descriptor.get("schema") != PENDING_UPDATE_SCHEMA:
        log("  待应用更新凭证格式无效，已清理。")
        shutil.rmtree(updates_root, ignore_errors=True)
        return "invalid", None, None, []
    if descriptor.get("manifest_verified") is not True:
        log("  待应用更新未通过发行清单核验，已拒绝并清理。")
        shutil.rmtree(updates_root, ignore_errors=True)
        return "invalid", None, None, []

    target_version = str(descriptor.get("target_version") or "").strip()
    if parse_semver(target_version) is None:
        log("  待应用更新版本号无效，已清理。")
        shutil.rmtree(updates_root, ignore_errors=True)
        return "invalid", None, None, []

    raw_staged = str(descriptor.get("staged_dir") or "").strip()
    try:
        pure_staged = _validate_update_relpath(raw_staged)
        staged_dir = (base_root / Path(*pure_staged.parts)).resolve()
        staged_dir.relative_to(updates_root)
    except ValueError as exc:
        log(f"  待应用更新目录越界，已拒绝并清理：{exc}")
        shutil.rmtree(updates_root, ignore_errors=True)
        return "invalid", None, None, []

    index_path = staged_dir / RELEASE_INDEX_NAME
    try:
        index_text = index_path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        log("  待应用更新目录缺少 release-files.json，已清理。")
        shutil.rmtree(updates_root, ignore_errors=True)
        return "invalid", None, None, []
    except OSError as exc:
        log(f"  暂时无法读取暂存包清单 release-files.json（文件可能正被占用），已保留暂存更新包并停止本次启动：{exc}")
        return "failed", descriptor, None, []
    except UnicodeDecodeError:
        log("  待应用更新目录中的 release-files.json 编码无效，已清理。")
        shutil.rmtree(updates_root, ignore_errors=True)
        return "invalid", None, None, []

    shipped = _parse_release_index_text(index_text)
    if shipped is None:
        log("  待应用更新目录缺少有效的 release-files.json，已清理。")
        shutil.rmtree(updates_root, ignore_errors=True)
        return "invalid", None, None, []

    for required in _REQUIRED_PAYLOAD_ENTRIES:
        req_path = staged_dir / required
        try:
            st = req_path.stat()
        except (FileNotFoundError, NotADirectoryError):
            log(f"  待应用更新目录缺少必需文件 {required}，已清理。")
            shutil.rmtree(updates_root, ignore_errors=True)
            return "invalid", None, None, []
        except OSError as exc:
            log(f"  暂时无法检查暂存包文件 {required}（文件可能正被占用），已保留暂存更新包并停止本次启动：{exc}")
            return "failed", descriptor, None, []
        if not stat.S_ISREG(st.st_mode):
            log(f"  待应用更新目录缺少必需文件 {required}，已清理。")
            shutil.rmtree(updates_root, ignore_errors=True)
            return "invalid", None, None, []
    if not (staged_dir / "src").is_dir():
        log("  待应用更新目录缺少 src/ 目录，已清理。")
        shutil.rmtree(updates_root, ignore_errors=True)
        return "invalid", None, None, []

    pyproject_path = staged_dir / "pyproject.toml"
    try:
        pyproject_raw = pyproject_path.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        log("  待应用更新目录缺少 pyproject.toml，已清理。")
        shutil.rmtree(updates_root, ignore_errors=True)
        return "invalid", None, None, []
    except OSError as exc:
        log(f"  暂时无法读取暂存包 pyproject.toml（文件可能正被占用），已保留暂存更新包并停止本次启动：{exc}")
        return "failed", descriptor, None, []
    try:
        pyproject_data = tomllib.loads(pyproject_raw.decode("utf-8"))
        project_tbl = pyproject_data.get("project") if isinstance(pyproject_data, dict) else None
        raw_staged_ver = project_tbl.get("version") if isinstance(project_tbl, dict) else ""
        staged_version = raw_staged_ver.strip() if isinstance(raw_staged_ver, str) else ""
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        staged_version = ""
    if parse_semver(staged_version) != parse_semver(target_version):
        log(f"  待应用更新版本不匹配（{staged_version or '无效'} != {target_version}），已清理。")
        shutil.rmtree(updates_root, ignore_errors=True)
        return "invalid", None, None, []

    validated_files: list[str] = []
    seen_cf: set[str] = set()
    for rel in sorted(shipped):
        try:
            pure = _validate_update_relpath(rel)
            if pure.parts[0].casefold() in _PROTECTED_UPDATE_TOP_DIRS:
                raise ValueError(f"包含受保护目录：{rel}")
            rel_cf = pure.as_posix().casefold()
            if rel_cf in seen_cf:
                raise ValueError(f"包含重复或大小写冲突路径：{rel}")
            seen_cf.add(rel_cf)
            src = (staged_dir / Path(*pure.parts)).resolve()
            src.relative_to(staged_dir)
            try:
                st = src.stat()
            except (FileNotFoundError, NotADirectoryError) as exc:
                raise ValueError(f"缺少载荷文件：{rel}") from exc
            except OSError as exc:
                log(f"  暂时无法检查暂存包文件 {rel}（文件可能正被占用），已保留暂存更新包并停止本次启动：{exc}")
                return "failed", descriptor, None, []
            if not stat.S_ISREG(st.st_mode):
                raise ValueError(f"缺少载荷文件：{rel}")
        except ValueError as exc:
            log(f"  待应用更新载荷校验失败，已清理：{exc}")
            shutil.rmtree(updates_root, ignore_errors=True)
            return "invalid", None, None, []
        validated_files.append(rel)

    return "valid", descriptor, staged_dir, validated_files


def apply_pending_update(
    root: Path | None = None,
    *,
    wait_instances_s: float = 10.0,
) -> tuple[str, str]:
    """Apply a verified update staged under `tmp/updates/` and clean up staging.

    Returns ``(status, version)`` where ``status`` is one of:
    - ``"none"``: no pending update (or git checkout)
    - ``"invalid"``: invalid pending descriptor/payload cleaned up before touching install
    - ``"deferred"``: another launcher window is still running; staging preserved
    - ``"applied"``: overlay succeeded and `tmp/updates/` cleaned
    - ``"failed"``: a transient I/O error interrupted validation or overlay; staging
      is preserved and the caller must abort startup rather than running a partially
      updated install.
    """
    base_root = (root or ROOT).resolve()
    if (base_root / ".git").exists():
        return "none", ""
    validation_status, descriptor, staged_dir, validated_files = _read_and_validate_staged_update(
        base_root
    )
    if validation_status != "valid" or descriptor is None or staged_dir is None:
        target = str(descriptor.get("target_version") or "") if isinstance(descriptor, dict) else ""
        return validation_status, target
    target_version = str(descriptor["target_version"])

    if _other_launcher_instances_active(base_root, wait_s=wait_instances_s):
        log("\n检测到同目录还有其他 jav-sub 窗口正在运行，已暂缓覆盖升级。")
        log("请关闭全部 jav-sub 窗口后重新启动，将自动继续完成升级。")
        return "deferred", target_version

    with _workspace_gate(base_root, wait_s=wait_instances_s) as gate_acquired:
        if not gate_acquired or _scan_launcher_slots_held(base_root):
            log("\n检测到同目录还有其他 jav-sub 窗口正在运行，已暂缓覆盖升级。")
            log("请关闭全部 jav-sub 窗口后重新启动，将自动继续完成升级。")
            return "deferred", target_version

        log(f"\n=== 正在应用在线升级 -> v{target_version} ===")
        exe_rel = ""
        try:
            for rel in validated_files:
                if rel.casefold() == "jav-sub.exe":
                    exe_rel = rel
                    continue
                pure = _validate_update_relpath(rel)
                src = staged_dir / Path(*pure.parts)
                dst = base_root / Path(*pure.parts)
                _copy_file_atomic(src, dst)

            # Publish the new release-files.json once all non-exe files are in place.
            _copy_file_atomic(staged_dir / RELEASE_INDEX_NAME, base_root / RELEASE_INDEX_NAME)

            # Replace jav-sub.exe last via rename-aside on Windows.
            if exe_rel:
                _replace_running_exe(staged_dir / exe_rel, base_root / "jav-sub.exe")
        except OSError as exc:
            log(f"覆盖升级失败（文件可能正被占用）：{exc}")
            log("为避免新旧版本文件混用，已保留暂存更新包并停止本次启动。")
            log("请关闭占用程序目录的窗口或安全软件后，重新双击 jav-sub.exe 自动继续完成升级。")
            return "failed", target_version

        pending_path = base_root / UPDATES_DIR / PENDING_UPDATE_FILE_NAME
        if not _unlink_with_retry(pending_path):
            log(f"[WARN] 已完成覆盖更新，但暂时无法删除 {pending_path.name}，请稍后手动清理 tmp\\updates\\。")
        shutil.rmtree(base_root / UPDATES_DIR, ignore_errors=True)
    log(f"已成功覆盖更新至 v{target_version}，并清理升级临时缓存。")
    return "applied", target_version


def _safe_extract_update_zip(
    archive_path: Path,
    dest_dir: Path,
    *,
    max_members: int = _MAX_ARCHIVE_MEMBERS,
    max_total_bytes: int = _MAX_ARCHIVE_UNCOMPRESSED_BYTES,
) -> Path:
    extract_root = dest_dir.resolve()
    extract_root.mkdir(parents=True, exist_ok=True)
    member_limit = max(1, int(max_members))
    byte_limit = max(1, int(max_total_bytes))
    seen_files_cf: dict[str, str] = {}
    seen_dirs_cf: dict[str, str] = {}
    total_extracted_bytes = 0
    with zipfile.ZipFile(archive_path, "r") as zf:
        infos = zf.infolist()
        if not infos:
            raise ValueError("安装包为空")
        if len(infos) > member_limit:
            raise ValueError(f"安装包成员数量超过安全上限（{len(infos)} > {member_limit}）")
        for info in infos:
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError(f"安装包包含符号链接：{info.filename}")
            pure = _validate_update_relpath(info.filename)
            posix_path = pure.as_posix()
            path_cf = posix_path.casefold()

            parent_end = len(pure.parts) if info.is_dir() else len(pure.parts) - 1
            for depth in range(1, parent_end + 1):
                dir_posix = "/".join(pure.parts[:depth])
                dir_cf = dir_posix.casefold()
                if dir_cf in seen_files_cf:
                    raise ValueError(f"安装包包含文件与目录大小写冲突：{info.filename}")
                prev_dir = seen_dirs_cf.get(dir_cf)
                if prev_dir is not None and prev_dir != dir_posix:
                    raise ValueError(f"安装包包含目录大小写冲突：{prev_dir} 与 {dir_posix}")
                seen_dirs_cf[dir_cf] = dir_posix

            if not info.is_dir():
                if path_cf in seen_files_cf or path_cf in seen_dirs_cf:
                    raise ValueError(f"安装包包含重复或大小写冲突路径：{info.filename}")
                seen_files_cf[path_cf] = posix_path

            target = (extract_root / Path(*pure.parts)).resolve()
            target.relative_to(extract_root)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if info.file_size > byte_limit or total_extracted_bytes + info.file_size > byte_limit:
                raise ValueError("安装包解压体积超过安全上限")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    total_extracted_bytes += len(chunk)
                    if total_extracted_bytes > byte_limit:
                        raise ValueError("安装包解压体积超过安全上限")
                    dst.write(chunk)
    if (extract_root / RELEASE_INDEX_NAME).is_file():
        return extract_root
    children = [p for p in extract_root.iterdir() if p.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return extract_root


def _verify_manifest_payload(staged_dir: Path, manifest_payload: dict) -> None:
    if not isinstance(manifest_payload, dict) or manifest_payload.get("schema") != RELEASE_MANIFEST_SCHEMA:
        raise ValueError("release-manifest.json schema 无效")
    git_meta = manifest_payload.get("git")
    if isinstance(git_meta, dict) and git_meta.get("dirty") is True:
        raise ValueError("发行清单标记为未提交的脏构建（git.dirty=true），已拒绝安装")
    files = manifest_payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("release-manifest.json 缺少文件列表")
    checked = 0
    recorded_dist: set[str] = set()
    recorded_dist_cf: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict) or entry.get("root") != "dist":
            continue
        rel = str(entry.get("path") or "")
        expected_hash = str(entry.get("sha256") or "").lower()
        expected_size = entry.get("bytes")
        if (
            not rel
            or not _UPDATE_SHA256_RE.fullmatch(expected_hash)
            or type(expected_size) is not int
            or expected_size < 0
        ):
            raise ValueError(f"发行清单条目无效：{rel!r}")
        pure = _validate_update_relpath(rel)
        if pure.parts[0].casefold() in _PROTECTED_UPDATE_TOP_DIRS:
            raise ValueError(f"发行清单试图写入受保护目录：{rel}")
        posix_rel = pure.as_posix()
        rel_cf = posix_rel.casefold()
        if posix_rel in recorded_dist or rel_cf in recorded_dist_cf:
            raise ValueError(f"发行清单包含重复或大小写冲突路径：{rel}")
        target = (staged_dir / Path(*pure.parts)).resolve()
        target.relative_to(staged_dir)
        if not target.is_file() or target.stat().st_size != expected_size:
            raise ValueError(f"文件缺失或大小不匹配：{rel}")
        if _sha256_file(target) != expected_hash:
            raise ValueError(f"文件 SHA-256 校验失败：{rel}")
        recorded_dist.add(posix_rel)
        recorded_dist_cf.add(rel_cf)
        checked += 1
    if checked == 0:
        raise ValueError("发行清单不含有效载荷条目")
    extra_files = [
        path.relative_to(staged_dir).as_posix()
        for path in sorted(staged_dir.rglob("*"))
        if path.is_file() and path.relative_to(staged_dir).as_posix() not in recorded_dist
    ]
    if extra_files:
        raise ValueError(f"更新包包含未登记的额外文件：{extra_files[0]}")


def stage_online_update_cli(
    root: Path | None = None,
    *,
    assume_yes: bool = False,
) -> tuple[str, str]:
    """CLI fallback (`jav-sub.exe --update`): check, download, verify, and stage.

    Returns ``(status, version)`` where ``status`` is one of
    ``"staged"``, ``"up_to_date"``, ``"cancelled"``, or ``"error"``.
    """
    base_root = (root or ROOT).resolve()
    if (base_root / ".git").exists():
        log("检测到这是 git 工作副本，请直接使用 git pull 更新。")
        return "error", ""
    if _other_launcher_instances_active(base_root, wait_s=0.0):
        log("检测到同目录还有其他 jav-sub 窗口正在运行，请先关闭全部窗口后再执行升级。")
        return "error", ""

    local_ver_str = current_project_version(base_root)
    local_tuple = parse_semver(local_ver_str) or (0, 0, 0)
    api_url = _update_api_url()
    if not _allowed_update_url(api_url):
        log(f"更新接口地址不受信任（{api_url}），已拒绝。")
        return "error", ""
    log(f"当前版本：v{local_ver_str}")
    log("正在检查最新稳定版 ...")
    opener = _update_opener()
    req = urllib.request.Request(
        api_url,
        headers={
            "User-Agent": f"jav-sub-setup/{local_ver_str}",
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with opener.open(req, timeout=20.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log(f"检查更新失败：{exc}")
        return "error", ""

    if not isinstance(payload, dict) or payload.get("draft") or payload.get("prerelease"):
        log("未找到可用的正式稳定版本。")
        return "up_to_date", ""
    tag = str(payload.get("tag_name") or "").strip()
    remote_tuple = parse_semver(tag)
    if remote_tuple is None:
        log(f"最新标签 {tag!r} 不是标准 x.y.z 稳定版，已跳过。")
        return "up_to_date", ""
    remote_ver_str = f"{remote_tuple[0]}.{remote_tuple[1]}.{remote_tuple[2]}"
    if remote_tuple <= local_tuple:
        log(f"当前已是最新版本（v{local_ver_str}）。")
        return "up_to_date", remote_ver_str

    assets = payload.get("assets") or []
    archive_asset = sha_asset = manifest_asset = None
    for item in assets:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip().casefold()
        if name == UPDATE_ARCHIVE_NAME.casefold():
            archive_asset = item
        elif name == UPDATE_SHA256_NAME.casefold():
            sha_asset = item
        elif name == UPDATE_MANIFEST_NAME.casefold():
            manifest_asset = item

    archive_url = str((archive_asset or {}).get("browser_download_url") or "").strip()
    sha_url = str((sha_asset or {}).get("browser_download_url") or "").strip()
    manifest_url = str((manifest_asset or {}).get("browser_download_url") or "").strip()
    digest_raw = str((archive_asset or {}).get("digest") or "").strip().lower()
    expected_sha = (
        digest_raw[len("sha256:") :].strip()
        if digest_raw.startswith("sha256:") and _UPDATE_SHA256_RE.fullmatch(digest_raw[len("sha256:") :].strip())
        else ""
    )

    if not archive_url or not _allowed_update_url(archive_url):
        log(f"版本 {tag} 缺少有效的 Windows 安装包资产。")
        return "error", ""
    if not expected_sha and (not sha_url or not _allowed_update_url(sha_url)):
        log(f"版本 {tag} 缺少 SHA-256 校验摘要，已拒绝自动下载。")
        return "error", ""
    if not manifest_url or not _allowed_update_url(manifest_url):
        log(f"版本 {tag} 缺少发行清单（{UPDATE_MANIFEST_NAME}），已拒绝自动下载。")
        return "error", ""

    log(f"发现新版本：v{remote_ver_str}（当前 v{local_ver_str}）")
    if not assume_yes and interactive() and not ask_yes(f"现在下载并升级到 v{remote_ver_str} 吗？", default=True):
        log("已取消升级。")
        return "cancelled", remote_ver_str

    version_dir = base_root / UPDATES_DIR / f"v{remote_ver_str}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    archive_part = version_dir / f"{UPDATE_ARCHIVE_NAME}.part"
    archive_path = version_dir / UPDATE_ARCHIVE_NAME
    extract_dir = version_dir / "extract"
    staged_dir = version_dir / "staged"
    try:
        version_dir.mkdir(parents=True, exist_ok=True)
        ua_headers = {"User-Agent": f"jav-sub-setup/{local_ver_str}"}
        if sha_url and _allowed_update_url(sha_url):
            sha_req = urllib.request.Request(sha_url, headers=ua_headers)
            with opener.open(sha_req, timeout=20.0) as resp:
                sidecar_sha = _parse_sha256_sidecar(resp.read().decode("utf-8", "replace"))
            if not sidecar_sha:
                raise ValueError("无法解析 .sha256 校验文件")
            if expected_sha and expected_sha != sidecar_sha:
                raise ValueError("安装包摘要与 .sha256 文件不一致")
            expected_sha = sidecar_sha
        if not _UPDATE_SHA256_RE.fullmatch(expected_sha):
            raise ValueError("缺少有效的 64 位 SHA-256 摘要")

        m_req = urllib.request.Request(manifest_url, headers=ua_headers)
        with opener.open(m_req, timeout=20.0) as resp:
            manifest_payload = json.loads(resp.read().decode("utf-8"))

        log(f"正在下载 {UPDATE_ARCHIVE_NAME} ...")
        actual_sha = _download_with_progress(
            archive_url,
            archive_part,
            opener=opener,
            headers=ua_headers,
            timeout=60.0,
        )
        if actual_sha != expected_sha:
            raise ValueError("安装包 SHA-256 校验失败")
        _replace_atomically(archive_part, archive_path)

        log("正在解压并核验发行清单 ...")
        extracted = _safe_extract_update_zip(archive_path, extract_dir)
        if extracted != staged_dir:
            shutil.move(str(extracted), str(staged_dir))
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        for required in _REQUIRED_PAYLOAD_ENTRIES:
            if not (staged_dir / required).is_file():
                raise ValueError(f"更新包缺少必需文件：{required}")
        if not (staged_dir / "src").is_dir():
            raise ValueError("更新包缺少 src/ 目录")
        if read_release_index(staged_dir) is None:
            raise ValueError("更新包缺少有效的 release-files.json")
        if parse_semver(current_project_version(staged_dir)) != remote_tuple:
            raise ValueError("更新包内 pyproject.toml 版本与目标版本不一致")
        _verify_manifest_payload(staged_dir, manifest_payload)
        archive_path.unlink(missing_ok=True)

        pending_path = base_root / UPDATES_DIR / PENDING_UPDATE_FILE_NAME
        descriptor = {
            "schema": PENDING_UPDATE_SCHEMA,
            "from_version": local_ver_str,
            "target_version": remote_ver_str,
            "tag": tag,
            "staged_dir": staged_dir.resolve().relative_to(base_root).as_posix(),
            "zip_sha256": actual_sha,
            "manifest_verified": True,
            "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_json_atomic(pending_path, descriptor)
        return "staged", remote_ver_str
    except (urllib.error.URLError, OSError, ValueError, zipfile.BadZipFile) as exc:
        log(f"在线升级准备失败：{exc}")
        shutil.rmtree(version_dir, ignore_errors=True)
        return "error", ""


def _relaunch_updated_exe(passthrough: list[str], *, keep_console: bool = False) -> bool:
    exe = ROOT / "jav-sub.exe"
    if not exe.is_file():
        return False
    cmd = [str(exe)]
    if keep_console:
        cmd.append("--keep-console")
    cmd.extend(passthrough)
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("_MEIPASS", "_PYI_"))
    }
    try:
        subprocess.Popen(cmd, cwd=str(ROOT), env=clean_env)
        return True
    except OSError as exc:
        log(f"自动重启新版程序失败（请手动双击 jav-sub.exe）：{exc}")
        return False


def _missing_ffmpeg_dll_patterns(directory: Path) -> list[str]:
    return [
        pattern
        for pattern in _REQUIRED_FFMPEG_DLL_PATTERNS
        if not any(directory.glob(pattern))
    ]


def ffmpeg_problem() -> str:
    """Empty string when ffmpeg is usable, else what is wrong with it.

    Never repaired from here: FFmpeg ships inside the archive, so the only fix is
    to put those files back.
    """
    suffix = ".exe" if os.name == "nt" else ""
    bin_dir = ROOT / "bin"
    if all((bin_dir / f"{name}{suffix}").is_file() for name in ("ffmpeg", "ffprobe")):
        # torchcodec loads the shared libraries rather than the executables, so
        # an ffmpeg.exe whose av*.dll were deleted is still a broken install.
        if os.name == "nt":
            missing = _missing_ffmpeg_dll_patterns(bin_dir)
            if missing:
                return f"bin\\ 里缺少 FFmpeg 的 av*.dll（缺少 {', '.join(missing)}；torchcodec 需要 Shared 版）"
        return ""
    ffmpeg_found = shutil.which("ffmpeg")
    ffprobe_found = shutil.which("ffprobe")
    if ffmpeg_found and ffprobe_found:
        if os.name == "nt":
            candidate_dirs: list[Path] = []
            for exe_path in (ffmpeg_found, ffprobe_found):
                parent = Path(exe_path).parent
                if parent not in candidate_dirs:
                    candidate_dirs.append(parent)
            if not any(
                directory.is_dir() and not _missing_ffmpeg_dll_patterns(directory)
                for directory in candidate_dirs
            ):
                return (
                    "系统 PATH 中的 FFmpeg 缺少 av*.dll"
                    "（需同时具备 avcodec/avformat/avutil；torchcodec 需要 Shared 版）"
                )
        return ""
    return "找不到 ffmpeg / ffprobe"


def diagnose(*, deep: bool = False) -> dict:
    """Everything wrong with this install right now.

    `deep` adds the import probe. Without it this is only filesystem checks, so
    it is affordable on the launch path.
    """
    report: dict = {
        "payload": missing_payload_paths(),
        "ffmpeg": ffmpeg_problem(),
        "venv": "",
        "broken": [],
        "failures": [],
    }
    if not venv_python().is_file():
        report["venv"] = "缺少 .venv（运行环境还没装，或被删除了）"
        report["broken"] = sorted(set(CRITICAL_IMPORT_ROOTS.values()))
        return report

    if venv_uses_legacy_tmp_python():
        report["venv"] = ".venv 仍指向已弃用的 tmp\\python 目录，需要重建以迁至 .python"
        return report
    if venv_needs_rebuild():
        report["venv"] = ".venv 迁移至 .python 后解释器校验失败，需要重建运行环境"
        return report

    missing = missing_import_roots()
    if missing:
        report["venv"] = "运行环境不完整，缺少：" + "、".join(missing)
        report["broken"] = distributions_for(missing)
    elif not environment_is_current():
        report["venv"] = "运行环境和当前版本的依赖清单不一致"

    if deep:
        failures = probe_imports()
        if failures:
            report["failures"] = failures
            report["venv"] = report["venv"] or "依赖装着但导入失败"
            report["broken"] = sorted(
                set(report["broken"]) | set(distributions_for(name for name, _ in failures))
            )
    return report


def report_is_healthy(report: dict) -> bool:
    return not (
        report["payload"] or report["ffmpeg"] or report["venv"] or report["failures"]
    )


# --- flow -------------------------------------------------------------------


def prompt_proxy() -> dict[str, str] | None:
    """Ask for proxy settings. None means "carry on without one"."""
    log()
    log("填写本地代理（Clash / v2ray 等）的监听地址；直接回车则放弃设置。")
    try:
        raw_host = input("  代理地址 [留空跳过]: ").strip()
        if not raw_host:
            return None
        try:
            detected_protocol, host, detected_port = _parse_proxy_host_entry(raw_host)
        except ValueError as exc:
            log(f"  {exc}，跳过代理设置。")
            return None
        if detected_port:
            port = detected_port
        else:
            port = input(f"  代理端口 [{DEFAULT_PROXY_PORT}]: ").strip() or DEFAULT_PROXY_PORT
        if detected_protocol:
            protocol = detected_protocol
        else:
            protocol = input("  协议 http/https/socks5 [http]: ").strip().lower() or "http"
    except (EOFError, KeyboardInterrupt):
        return None
    if not (port.isascii() and port.isdigit()) or not 1 <= int(port) <= 65535:
        log(f"  端口 {port!r} 无效（必须是 1-65535 的数字），跳过代理设置。")
        return None
    if protocol not in SUPPORTED_PROXY_PROTOCOLS:
        log(f"  不支持的协议 {protocol!r}，按 http 处理。")
        protocol = "http"
    settings = {"PROXY_PROTOCOL": protocol, "PROXY_HOST": host, "PROXY_PORT": str(int(port))}
    try:
        validated_settings, _ = _validate_proxy_url(proxy_url_from(settings))
    except ValueError as exc:
        log(f"  代理配置无效（{exc}），跳过代理设置。")
        return None
    return validated_settings


def ask_yes(question: str, *, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{question} {suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default
    if not answer:
        return default
    return answer in {"y", "yes"}


def ensure_proxy_configured(*, assume_yes: bool) -> None:
    """Settle the proxy question before anything large is downloaded."""
    proxy_url = proxy_url_from(read_config_file())
    if proxy_url:
        try:
            apply_proxy(proxy_url)
        except ValueError as exc:
            log(f"\n[WARN] config/config.json 中的代理配置无效（{exc}），已忽略。")
        else:
            log("使用 config/config.json 中已配置的代理")
            return
    if assume_yes or not interactive():
        return

    log("\n正在测速（读取 PyTorch 官方源上的 torch wheel）...")
    log(f"  {describe_speed(measure_download())}")

    if not ask_yes("需要设置代理吗？"):
        return
    while True:
        settings = prompt_proxy()
        if not settings:
            return
        proxy_url = proxy_url_from(settings)
        try:
            apply_proxy(proxy_url)
        except ValueError as exc:
            log(f"  代理配置无效（{exc}）。")
            if ask_yes("重新填写代理吗？", default=True):
                continue
            return
        break
    try:
        update_config_file(settings)
        log("代理设置已写入 config/config.json")
    except (ValueError, OSError) as exc:
        log(f"[WARN] {exc}；本次启动仍使用所填代理，请修复 config/config.json 后重新保存。")
    log("正在用代理复测 ...")
    log(f"  {describe_speed(measure_download())}")


def is_source_checkout() -> bool:
    return (ROOT / ".git").exists()


def _sync_command(uv: Path, *, reinstall: list[str] | None = None, reinstall_all: bool = False):
    # --frozen pins the release to the versions it was built against, but it
    # fails outright without a lockfile, so a payload missing one resolves live
    # instead of refusing to install.
    command: list = [uv, "sync", "--no-dev"]
    if LOCK_PATH.is_file():
        command.insert(2, "--frozen")
    if reinstall_all:
        command.append("--reinstall")
    else:
        for package in reinstall or []:
            command.extend(["--reinstall-package", package])
    return command


def install(
    uv: Path,
    *,
    reinstall: list[str] | None = None,
    reinstall_all: bool = False,
    rebuild_venv: bool = False,
) -> int:
    log("\n=== 安装运行环境 ===")
    if (rebuild_venv or venv_needs_rebuild()) and VENV_PATH.exists():
        log("正在移除需要重建的旧版 .venv，以便在 .python 下重建...")
        try:
            shutil.rmtree(VENV_PATH)
        except OSError as error:
            log(f"无法移除旧版 .venv（请关闭其他占用窗口后重试）：{error}")
            return 1
    log("首次运行需要下载 PyTorch 等依赖，安装后约占 3.3GB 磁盘。中断后重新运行即可续装。")
    code = run_streaming(_sync_command(uv, reinstall=reinstall, reinstall_all=reinstall_all))
    if code != 0:
        return code
    log("\n正在验证环境 ...")
    if not imports_work():
        log("依赖安装完成，但 torch / transformers 无法导入。")
        return 1
    try:
        STAMP_PATH.write_text(lock_digest(), encoding="utf-8")
    except OSError:
        pass  # Costs a redundant sync next launch; not worth failing over.
    return 0


def repair(uv: Path, broken: list[str]) -> bool:
    """Reinstall what is broken, then verify by importing it.

    Targeted first: `uv sync` on its own would audit the metadata, find the
    dist-info of a package whose files were deleted, and report success without
    touching the disk - so the packages that failed are named explicitly. The
    full --reinstall is the fallback for damage we could not attribute.
    """
    log("\n=== 修复运行环境 ===")
    if venv_needs_rebuild() and VENV_PATH.exists():
        log("正在移除需要重建的旧版 .venv，以便在 .python 下重建...")
        try:
            shutil.rmtree(VENV_PATH)
        except OSError as error:
            log(f"无法移除旧版 .venv（请关闭其他占用窗口后重试）：{error}")
            return False
    if broken:
        log("将重新安装：" + "、".join(broken))
    if run_streaming(_sync_command(uv, reinstall=broken)) == 0:
        failures = probe_imports()
        if not failures:
            _write_stamp()
            return True
        log("\n定向重装后仍有导入失败：")
        for name, reason in failures:
            log(f"  {name}: {reason}")

    log("\n改为完整重装所有依赖（会比较久，已下载的包会从本地缓存复用）...")
    if run_streaming(_sync_command(uv, reinstall_all=True)) != 0:
        return False
    failures = probe_imports()
    for name, reason in failures:
        log(f"  {name}: {reason}")
    if failures:
        return False
    _write_stamp()
    return True


def _write_stamp() -> None:
    try:
        STAMP_PATH.write_text(lock_digest(), encoding="utf-8")
    except OSError:
        pass


def doctor(*, assume_yes: bool, deep: bool = False, allow_repair: bool = True) -> tuple[bool, bool]:
    """Print a checklist of what is wrong, repair what uv can.

    Returns (healthy now, repaired something).
    """
    log("\n=== 环境自检 ===")
    report = diagnose(deep=deep)

    if report["payload"]:
        log("  程序文件: 缺少 " + "、".join(report["payload"]))
    else:
        log("  程序文件: 正常")
    log(f"  FFmpeg:   {report['ffmpeg'] or '正常'}")
    log(f"  运行环境: {report['venv'] or '正常'}")
    for name, reason in report["failures"]:
        log(f"    {name}: {reason}")

    if report_is_healthy(report):
        return True, False

    if report["payload"]:
        log("\n缺少的程序文件在压缩包里，无法自动补回。")
        log("请重新解压一份完整的程序目录（.venv、.python、models、config 可以从旧目录搬过去）。")
        return False, False
    if report["ffmpeg"]:
        log("\nFFmpeg 随压缩包分发，无法自动补回。")
        log("请从压缩包里重新解压 bin\\ 目录，或安装 FFmpeg Shared 版并加入系统 PATH。")
        if not report["venv"] and not report["failures"]:
            return False, False

    if not allow_repair:
        log("\n只报告，不动这个环境（git 工作副本里请用 uv sync；")
        log("确实要按发布版的方式修复时加 --allow-source-checkout）。")
        return False, False

    # Before anything is downloaded, so a machine that needs a proxy is asked
    # once and the answer covers uv and the model downloads afterwards.
    ensure_proxy_configured(assume_yes=assume_yes)
    uv = ensure_uv()
    if uv is None:
        log("\n没有可用的 uv，无法修复依赖。手动安装后重新运行本程序:")
        log('  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"')
        return False, False

    repaired = repair(uv, list(report["broken"]))
    if repaired:
        log("\n运行环境已修复。")
    else:
        log("\n修复失败。请检查网络或代理设置后重试；实在不行删掉 .venv 再运行一次。")
    return repaired and not report["ffmpeg"], repaired


def _effective_ready_path(ready_path: Path | None = None) -> Path:
    """Return a PID-isolated ready file path unless explicitly overridden."""
    if ready_path is not None:
        return Path(ready_path)
    if READY_PATH != _DEFAULT_READY_PATH:
        return READY_PATH
    return ROOT / "tmp" / f".app-ready-{os.getpid()}"


def _watch_for_ready(stop: threading.Event, ready_path: Path | None = None) -> None:
    target = _effective_ready_path(ready_path)
    while not stop.wait(0.2):
        if target.exists():
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
            set_console_visible(False)
            return


def launch(
    extra_args: list[str],
    *,
    hide_when_ready: bool = True,
    ready_path: Path | None = None,
) -> int:
    log("\n=== 启动 jav-sub ===")
    stop = threading.Event()
    target_ready = _effective_ready_path(ready_path)
    if hide_when_ready:
        try:
            target_ready.parent.mkdir(parents=True, exist_ok=True)
            target_ready.unlink(missing_ok=True)
            os.environ["JAV_SUB_READY_FILE"] = str(target_ready)
        except OSError:
            os.environ.pop("JAV_SUB_READY_FILE", None)
        else:
            threading.Thread(
                target=_watch_for_ready,
                args=(stop, target_ready),
                daemon=True,
            ).start()
    else:
        os.environ.pop("JAV_SUB_READY_FILE", None)
    try:
        code = run_streaming([venv_python(), "launcher.py", *extra_args])
    finally:
        stop.set()
        if hide_when_ready:
            try:
                target_ready.unlink(missing_ok=True)
            except OSError:
                pass
    if code != 0:
        # Whatever went wrong has to be readable, so give the window back.
        set_console_visible(True)
    return code


def launch_with_recovery(
    extra_args: list[str],
    *,
    assume_yes: bool,
    repaired_already: bool,
    hide_when_ready: bool = True,
    allow_repair: bool = True,
) -> int:
    code = launch(extra_args, hide_when_ready=hide_when_ready)
    if code in (0, UPDATE_EXIT_CODE):
        return code

    log(f"\n程序退出，退出码 {_format_exit_code(code)}。")
    if repaired_already:
        # The environment was rebuilt minutes ago; re-diagnosing it would only
        # repeat what we just verified.
        _log_launch_failure_hints()
        return code

    healthy, repaired = doctor(assume_yes=assume_yes, deep=True, allow_repair=allow_repair)
    if not repaired:
        if healthy:
            log("\n运行环境自检没发现问题，所以问题不在依赖上：")
        _log_launch_failure_hints()
        return code

    log("\n重新启动一次 ...")
    code = launch(extra_args, hide_when_ready=hide_when_ready)
    if code not in (0, UPDATE_EXIT_CODE):
        log(f"\n重试后仍退出，退出码 {_format_exit_code(code)}。")
        _log_launch_failure_hints()
    return code


def _log_launch_failure_hints() -> None:
    log("  · 窗口打不开：安装 Microsoft Edge WebView2 运行时")
    log("    https://developer.microsoft.com/en-us/microsoft-edge/webview2/")
    log("  · 显卡相关报错：更新 NVIDIA 驱动后重启")
    log("  · 详细错误见上面的输出，以及 logs\\app-<PID>.log、logs\\bootstrap.log 和最新的 .run.log")
    log("  · 想再自检一次：jav-sub.exe --doctor")


def _finish_after_launch(
    exit_code: int,
    passthrough: list[str],
    *,
    keep_console: bool = False,
) -> int:
    if exit_code == UPDATE_EXIT_CODE:
        set_console_visible(True)
        apply_status, updated_ver = apply_pending_update()
        if apply_status != "applied":
            return 1
        log(f"正在重启已升级的 jav-sub v{updated_ver} ...")
        return 0 if _relaunch_updated_exe(passthrough, keep_console=keep_console) else 1
    return exit_code


def main(argv: list[str] | None = None) -> int:
    # Before argparse, which prints Chinese help text and exits without ever
    # reaching the body of this function.
    enable_utf8_console()

    parser = argparse.ArgumentParser(
        description="jav-sub 启动器（首次运行会先安装运行环境）"
    )
    parser.add_argument("--proxy", default="", help="代理 URL，例如 http://127.0.0.1:7890")
    parser.add_argument("--yes", action="store_true", help="不提问，直连安装")
    parser.add_argument("--reinstall", action="store_true", help="即使环境已就绪也重新安装")
    parser.add_argument("--install-only", action="store_true", help="只安装，不启动")
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="逐项检查程序文件、FFmpeg 和依赖（含实际导入），缺失就补回来",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="检查 GitHub Releases 最新稳定版并在线下载覆盖升级",
    )
    parser.add_argument(
        "--keep-console",
        action="store_true",
        help="程序窗口打开后不隐藏这个控制台（排查问题时用）",
    )
    parser.add_argument(
        "--allow-source-checkout",
        action="store_true",
        help="允许在 git 工作副本里同步环境（会按 --no-dev 移除 pytest 等开发依赖）",
    )
    args, passthrough = parser.parse_known_args(argv)

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if _FROZEN:
        # Keep a release self-contained, and put the cache on the same volume as
        # .venv so uv hardlinks out of it instead of copying every wheel twice.
        # Only when frozen: in a source checkout uv's own cache is the right one,
        # and redirecting it would re-download everything the developer has.
        os.environ.setdefault("UV_CACHE_DIR", str(ROOT / "cache" / "uv"))
        os.environ.setdefault("UV_PYTHON_INSTALL_DIR", str(ROOT / ".python"))

    prune_old_launcher_backups()

    # If a previous session staged a verified update (e.g. deferred or interrupted
    # by a file lock), apply it before touching src/ or venv. If applying fails
    # or is deferred, stop launch so we never run a mixed old/new directory.
    if (ROOT / UPDATES_DIR).is_dir():
        apply_status, updated_ver = apply_pending_update()
        if apply_status == "applied":
            log(f"已应用待完成更新 v{updated_ver}，正在重启 jav-sub ...")
            if _FROZEN:
                return 0 if _relaunch_updated_exe(passthrough, keep_console=args.keep_console) else 1
        elif apply_status in {"failed", "deferred"}:
            return 1

    if args.update:
        if args.proxy:
            try:
                apply_proxy(args.proxy)
            except ValueError:
                log("代理 URL 无效，请使用 http://、https:// 或 socks5://地址:端口（端口 1-65535）。")
                return 2
        else:
            try:
                apply_proxy(proxy_url_from(read_config_file()))
            except ValueError as exc:
                log(f"\n[WARN] config/config.json 中的代理配置无效（{exc}），本次已忽略代理设置。")
        stage_status, _ = stage_online_update_cli(assume_yes=args.yes)
        if stage_status in {"up_to_date", "cancelled"}:
            return 0
        if stage_status != "staged":
            return 1
        apply_status, updated_ver = apply_pending_update()
        if apply_status != "applied":
            return 1
        if args.install_only:
            return 0
        log(f"正在重启已升级的 jav-sub v{updated_ver} ...")
        return 0 if _relaunch_updated_exe(passthrough, keep_console=args.keep_console) else 1

    # Before anything reads src/: an upgrade extracted over the old folder is
    # only finished once what the old release left behind is out of the way.
    moved, leftovers = prune_stale_release_files()
    if moved and leftovers is not None:
        log(f"已把旧版本遗留、本版本不再使用的 {len(moved)} 个程序文件移到 "
            f"{leftovers.relative_to(ROOT)}（覆盖升级的正常现象，程序正常后可删除）")

    if prune_legacy_env_file():
        log("已清理旧版本遗留配置文件 .env（新版配置已改为 config/config.json，请在界面中重新设置）")

    removed_models, reclaimed_bytes = prune_retired_model_caches()
    if removed_models:
        reclaimed_gb = reclaimed_bytes / (1024 ** 3)
        names = "、".join(Path(item).name for item in removed_models)
        log(f"已清理旧版本不再使用的 {len(removed_models)} 个模型缓存"
            f"（释放约 {reclaimed_gb:.2f} GB）：{names}")

    python_migration = migrate_legacy_python_runtime()
    migration_needs_rebuild = python_migration == "needs_rebuild"
    if python_migration == "migrated":
        log("已将旧版 Python 运行时从 tmp\\python 迁移到 .python（现在可安全清空 tmp\\）")
    elif migration_needs_rebuild:
        log("检测到旧版 .venv 仍依赖 tmp\\python 且无法直接迁移，将自动重建运行环境到 .python")

    # A structural check on every run, not just the first: the stamp says the
    # install once completed, which stops being true the moment someone deletes a
    # folder out of .venv to reclaim disk space. Cheap (a handful of stat calls);
    # the import probe that would catch subtler damage waits for --doctor or for
    # a launch that actually failed.
    report = diagnose()
    broken_after_install = bool(report["venv"]) and not report["payload"]
    needs_install = (
        args.reinstall
        or migration_needs_rebuild
        or not environment_is_current()
        or broken_after_install
    )

    # This exe is the launcher too - every run after the first goes straight to
    # launch() - so the banner says which of the three is happening instead of
    # calling itself an installer forever.
    if args.doctor:
        log("jav-sub 环境自检")
    elif not venv_python().is_file():
        log("jav-sub 首次运行安装")
    elif report["broken"]:
        log("jav-sub 修复运行环境")
    elif needs_install:
        log("jav-sub 更新运行环境")
    else:
        log("jav-sub 启动")
    log(f"程序目录: {ROOT}")

    if args.doctor:
        # --doctor runs before the source-checkout guard below, and repairing
        # means `uv sync --no-dev`: in a checkout that would strip pytest and
        # pyinstaller out of the developer's own .venv. Report, do not touch.
        may_repair = args.allow_source_checkout or not is_source_checkout()
        healthy, repaired = doctor(assume_yes=args.yes, deep=True, allow_repair=may_repair)
        if not healthy and not repaired:
            return 1
        if args.install_only:
            return 0
        exit_code = launch_with_recovery(
            passthrough,
            assume_yes=args.yes,
            repaired_already=True,
            hide_when_ready=not args.keep_console,
            allow_repair=may_repair,
        )
        return _finish_after_launch(exit_code, passthrough, keep_console=args.keep_console)

    if not venv_python().is_file() and (nested := enclosing_install()) is not None:
        log(f"\n注意：上一级目录 {nested} 看起来是已经装好的 jav-sub。")
        log("升级应把新版压缩包里的 jav-sub 文件夹解压到旧版 jav-sub 文件夹所在的位置，")
        log("替换同名文件；像现在这样解压进旧版目录里面，会重新安装运行环境、重新下载模型。")
        if interactive() and not args.yes and not ask_yes("仍要在这里全新安装吗？"):
            log("已取消。请关闭窗口，按上面的方式重新解压。")
            return 1

    if report["payload"]:
        log("\n程序文件不完整，缺少：" + "、".join(report["payload"]))
        log("这些文件在压缩包里，无法自动补回：请把发布包重新解压并覆盖到本目录，")
        log("替换同名文件即可；.venv、.python、models、config 都会保留，不用重新下载。")
        return 1

    if needs_install and is_source_checkout() and not args.allow_source_checkout:
        # `uv sync --no-dev` would strip pytest and pyinstaller out of the
        # developer's own .venv. This installer is for releases; in a checkout
        # the developer already has `uv sync`.
        log("\n检测到这是 git 工作副本，已跳过环境同步（避免 --no-dev 删掉开发依赖）。")
        log("在工作副本里请直接用 uv sync / uv run python launcher.py；")
        log("确实要模拟发布行为时加 --allow-source-checkout。")
        return 1

    if args.proxy:
        try:
            apply_proxy(args.proxy)
        except ValueError:
            log("代理 URL 无效，请使用 http://、https:// 或 socks5://地址:端口（端口 1-65535）。")
            return 2
        log("使用命令行指定的代理")
    elif needs_install:
        ensure_proxy_configured(assume_yes=args.yes)
    else:
        # Already installed: hand the stored proxy to the app for model
        # downloads, but do not interrogate the user on every launch.
        try:
            apply_proxy(proxy_url_from(read_config_file()))
        except ValueError as exc:
            log(f"\n[WARN] config/config.json 中的代理配置无效（{exc}），本次已忽略代理设置，请在控制台重新保存。")

    if needs_install:
        if broken_after_install:
            log(f"\n{report['venv']}")
        # After the proxy interview, so that a machine with no uv at all can
        # still fetch it through the proxy the user just entered.
        uv = ensure_uv()
        if uv is None:
            log("\n没有可用的 uv，无法安装依赖。手动安装后重新运行本程序:")
            log('  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"')
            return 1
        # Deleted files leave their dist-info behind, and a plain sync audits
        # metadata: the packages we know are broken have to be named.
        code = install(uv, reinstall=list(report["broken"]), rebuild_venv=migration_needs_rebuild)
        if code != 0:
            log(f"\n安装失败（退出码 {_format_exit_code(code)}）。")
            log("修好网络或设置代理后重新运行本程序即可继续，已下载的部分会被复用。")
            return code
        log("\n环境就绪。")
    else:
        log("环境已就绪，跳过安装。")

    if report["ffmpeg"]:
        # Not fatal: the UI opens and settings work, only audio extraction dies.
        log(f"\n注意，FFmpeg 有问题：{report['ffmpeg']}")
        log("请从压缩包里重新解压 bin\\ 目录，否则转写会在提取音频这一步失败。")

    if args.install_only:
        return 0
    exit_code = launch_with_recovery(
        passthrough,
        assume_yes=args.yes,
        repaired_already=needs_install,
        hide_when_ready=not args.keep_console,
    )
    return _finish_after_launch(exit_code, passthrough, keep_console=args.keep_console)


def _run_cli(argv: list[str] | None = None) -> int:
    global _BOOTSTRAP_LOG_ENABLED
    _BOOTSTRAP_LOG_ENABLED = True
    try:
        return main(argv)
    except KeyboardInterrupt:
        set_console_visible(True)
        log("\n已由用户中断。")
        return 130
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - top-level crash guard
        set_console_visible(True)
        log(f"\n启动器发生未捕获异常：{type(exc).__name__}: {exc}")
        log(traceback.format_exc().rstrip())
        log(f"详细启动日志已写入：{_bootstrap_log_path()}")
        return 1


if __name__ == "__main__":
    exit_code = _run_cli()
    if exit_code != 0:
        pause_before_exit()
    sys.exit(exit_code)
