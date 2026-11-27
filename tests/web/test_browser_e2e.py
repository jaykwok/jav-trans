"""Real Chromium/Edge CDP end-to-end interactive UI click regression suite.

Spins up the real FastAPI/uvicorn Web console against an isolated temporary
install directory and local mock GitHub Releases server, drives headless
Microsoft Edge (the exact Chromium engine backing pywebview/WebView2 on
Windows) via Chrome DevTools Protocol, clicks every primary interactive control,
and fails on any uncaught JS exception, console error, unexpected alert dialog,
or HTTP >= 400 resource/API response.
"""

from __future__ import annotations

import asyncio
import hashlib
import http.server
import io
import json
import os
import socket
import subprocess
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pytest
import uvicorn
import websockets

import bootstrap
from core import config as core_config, updater
from llm.backends import llamacpp_runtime
from web import pipeline_manager
from web.app import create_app
from web.routes import config as config_routes


def _find_chromium_binary() -> Path | None:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _free_ports(count: int, *, exclude: set[int] | None = None) -> list[int]:
    """Allocate ``count`` distinct free localhost ports without racing reuse."""
    sockets: list[socket.socket] = []
    ports: list[int] = []
    blocked = set(exclude or ())
    try:
        while len(ports) < count:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
            if port in blocked:
                sock.close()
                continue
            blocked.add(port)
            sockets.append(sock)
            ports.append(port)
    finally:
        for sock in sockets:
            sock.close()
    return ports


def _make_mock_release_assets(version: str = "1.9.1") -> tuple[bytes, str, bytes]:
    files: dict[str, bytes] = {
        "jav-sub.exe": f"fake-exe-{version}".encode("utf-8"),
        "launcher.py": f"# launcher {version}\n".encode("utf-8"),
        "pyproject.toml": f'[project]\nname = "jav-sub"\nversion = "{version}"\n'.encode("utf-8"),
        "uv.lock": b"version = 1\n",
        "README.txt": b"readme\n",
        "src/main.py": f"VERSION = '{version}'\n".encode("utf-8"),
        "bin/ffmpeg.exe": b"ffmpeg-bytes",
    }
    files[updater.RELEASE_INDEX_NAME] = (
        json.dumps(
            {"schema": updater.RELEASE_INDEX_SCHEMA, "files": sorted(files.keys())},
            ensure_ascii=False,
            indent=1,
        )
        + "\n"
    ).encode("utf-8")

    manifest_entries = [
        {
            "path": rel,
            "root": "dist",
            "category": "payload",
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for rel, content in sorted(files.items())
    ]
    manifest_bytes = json.dumps(
        {
            "schema": updater.RELEASE_MANIFEST_SCHEMA,
            "dist_root": "jav-sub",
            "files": manifest_entries,
        }
    ).encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel, content in sorted(files.items()):
            zf.writestr(f"jav-sub/{rel}", content)
    zip_bytes = buf.getvalue()
    zip_sha = hashlib.sha256(zip_bytes).hexdigest()
    return zip_bytes, zip_sha, manifest_bytes


def _start_mock_release_server(
    version: str = "1.9.1",
) -> tuple[http.server.ThreadingHTTPServer, int, str]:
    zip_bytes, zip_sha, manifest_bytes = _make_mock_release_assets(version)
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    release_payload = json.dumps(
        {
            "tag_name": f"v{version}",
            "draft": False,
            "prerelease": False,
            "html_url": f"{base}/releases/tag/v{version}",
            "body": f"## v{version}\n\n- E2E online self-update test release",
            "published_at": "2026-11-26T12:00:00Z",
            "assets": [
                {
                    "name": updater.UPDATE_ARCHIVE_NAME,
                    "browser_download_url": f"{base}/{updater.UPDATE_ARCHIVE_NAME}",
                    "size": len(zip_bytes),
                    "digest": f"sha256:{zip_sha}",
                },
                {
                    "name": updater.UPDATE_SHA256_NAME,
                    "browser_download_url": f"{base}/{updater.UPDATE_SHA256_NAME}",
                    "size": 100,
                },
                {
                    "name": updater.UPDATE_MANIFEST_NAME,
                    "browser_download_url": f"{base}/{updater.UPDATE_MANIFEST_NAME}",
                    "size": len(manifest_bytes),
                },
            ],
        }
    ).encode("utf-8")
    routes: dict[str, tuple[str, bytes]] = {
        "/latest": ("application/json", release_payload),
        f"/{updater.UPDATE_ARCHIVE_NAME}": ("application/zip", zip_bytes),
        f"/{updater.UPDATE_SHA256_NAME}": (
            "text/plain",
            f"{zip_sha}  {updater.UPDATE_ARCHIVE_NAME}\n".encode("ascii"),
        ),
        f"/{updater.UPDATE_MANIFEST_NAME}": ("application/json", manifest_bytes),
    }

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            entry = routes.get(self.path)
            if entry is None:
                self.send_response(404)
                self.end_headers()
                return
            ctype, body = entry
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _fmt: str, *_args: Any) -> None:
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port, zip_sha


def _make_fake_install(root: Path, version: str = "1.9.0") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "jav-sub.exe").write_bytes(f"exe-{version}".encode("utf-8"))
    (root / "launcher.py").write_text(f"# launcher {version}\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "jav-sub"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "main.py").write_text(f"VERSION = '{version}'\n", encoding="utf-8")
    bootstrap.write_release_index(root)
    return root


class _CDPPage:
    """Minimal Chrome DevTools Protocol driver over `websockets`."""

    def __init__(self, ws: Any) -> None:
        self._ws = ws
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self.exceptions: list[str] = []
        self.console_errors: list[str] = []
        self.http_errors: list[tuple[int, str]] = []
        self.dialogs: list[tuple[str, str]] = []
        self.accept_confirm = True
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self._reader_task = asyncio.create_task(self._read_loop())

    def _spawn_bg(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)

        def _done(done_task: asyncio.Task[Any]) -> None:
            self._bg_tasks.discard(done_task)
            if not done_task.cancelled():
                done_task.exception()

        task.add_done_callback(_done)

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                if "id" in msg:
                    fut = self._pending.pop(msg["id"], None)
                    if fut is not None and not fut.done():
                        if "error" in msg:
                            fut.set_exception(RuntimeError(f"CDP error: {msg['error']}"))
                        else:
                            fut.set_result(msg.get("result", {}))
                    continue
                method = msg.get("method", "")
                params = msg.get("params") or {}
                if method == "Runtime.exceptionThrown":
                    details = params.get("exceptionDetails") or {}
                    text = details.get("text") or ""
                    exc_desc = (details.get("exception") or {}).get("description") or ""
                    self.exceptions.append(f"{text}: {exc_desc}".strip(": "))
                elif method == "Runtime.consoleAPICalled" and params.get("type") == "error":
                    args = [
                        str(a.get("value") if "value" in a else a.get("description", ""))
                        for a in params.get("args") or []
                    ]
                    self.console_errors.append(" ".join(args))
                elif method == "Log.entryAdded":
                    entry = params.get("entry") or {}
                    if entry.get("level") == "error":
                        self.console_errors.append(
                            f"{entry.get('text', '')} ({entry.get('url', '')})"
                        )
                elif method == "Network.responseReceived":
                    resp = params.get("response") or {}
                    status = int(resp.get("status") or 0)
                    if status >= 400:
                        self.http_errors.append((status, str(resp.get("url") or "")))
                elif method == "Page.javascriptDialogOpening":
                    dtype = str(params.get("type") or "")
                    message = str(params.get("message") or "")
                    self.dialogs.append((dtype, message))
                    accept = self.accept_confirm if dtype == "confirm" else True
                    self._spawn_bg(
                        self.send("Page.handleJavaScriptDialog", {"accept": accept})
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        req_id = self._next_id
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        await self._ws.send(json.dumps({"id": req_id, "method": method, "params": params or {}}))
        return await asyncio.wait_for(fut, timeout=15.0)

    async def eval(self, expression: str) -> Any:
        result = await self.send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": True,
                "returnByValue": True,
            },
        )
        if "exceptionDetails" in result:
            raise AssertionError(f"JS eval threw: {result['exceptionDetails']}")
        return (result.get("result") or {}).get("value")

    async def wait_until(self, expression: str, *, timeout_s: float = 10.0, desc: str = "") -> Any:
        deadline = time.monotonic() + timeout_s
        last = None
        while time.monotonic() < deadline:
            last = await self.eval(expression)
            if last:
                return last
            await asyncio.sleep(0.08)
        raise AssertionError(
            f"Timed out waiting for {desc or expression!r}; last={last!r}, "
            f"exceptions={self.exceptions}, console_errors={self.console_errors}, "
            f"http_errors={self.http_errors}"
        )

    async def close(self) -> None:
        self._reader_task.cancel()
        bg_tasks = list(self._bg_tasks)
        for task in bg_tasks:
            task.cancel()
        await asyncio.gather(self._reader_task, *bg_tasks, return_exceptions=True)


def _wait_for_cdp_ws_url(cdp_port: int, *, timeout_s: float = 15.0) -> str:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{cdp_port}/json/list"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                targets = json.loads(resp.read().decode("utf-8"))
            for target in targets:
                if target.get("type") == "page" and target.get("webSocketDebuggerUrl"):
                    return str(target["webSocketDebuggerUrl"])
        except Exception:
            pass
        time.sleep(0.15)
    raise RuntimeError(f"Headless Chromium CDP endpoint on port {cdp_port} did not start")


def _wait_for_http_ready(port: int, *, timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/api/config"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"Uvicorn server on port {port} did not become ready")


def test_web_ui_full_browser_click_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    browser = _find_chromium_binary()
    if browser is None:
        msg = "No Chromium/Edge executable available for headless browser E2E test"
        if os.getenv("JAV_SUB_CI_STRICT") == "1":
            pytest.fail(msg)
        pytest.skip(msg)

    install_dir = _make_fake_install(tmp_path / "pkg", version="1.9.0")
    sample_video = tmp_path / "sample-a.mp4"
    sample_video.write_bytes(b"synthetic-video")
    edge_profile = tmp_path / "edge-profile"
    edge_profile.mkdir()

    mock_gh_server, mock_gh_port, expected_zip_sha = _start_mock_release_server("1.9.1")
    app_port, events_port, cdp_port = _free_ports(3, exclude={mock_gh_port})

    # Isolate every writable path & runtime side effect to tmp_path.
    monkeypatch.setenv("TRANSLATION_BACKEND", "llamacpp")
    monkeypatch.setenv("JAV_SUB_EVENTS_PORT", str(events_port))
    monkeypatch.setenv("JAV_SUB_UPDATE_API_URL", f"http://127.0.0.1:{mock_gh_port}/latest")
    monkeypatch.setenv("JAV_SUB_DISABLE_UPDATE_CHECK", "0")
    monkeypatch.setenv("LLAMACPP_AUTO_DOWNLOAD", "0")
    monkeypatch.setenv("PROXY_PROTOCOL", "http")
    monkeypatch.setenv("PROXY_HOST", "")
    monkeypatch.setenv("PROXY_PORT", "")
    monkeypatch.setattr(core_config, "PROJECT_ROOT", install_dir)
    monkeypatch.setattr(core_config, "CONFIG_DIR", install_dir / "config")
    monkeypatch.setattr(core_config, "CONFIG_PATH", install_dir / "config" / "config.json")
    monkeypatch.setattr(config_routes, "PROJECT_ROOT", install_dir)
    monkeypatch.setattr(
        config_routes,
        "_probe_cuda_environment_status",
        lambda: {"ok": True, "status": "ok"},
    )
    monkeypatch.setattr(config_routes, "_CUDA_ENV_CACHE", {"ok": True, "status": "ok"})
    monkeypatch.setattr(
        llamacpp_runtime,
        "start_managed_llama_server_prefetch",
        lambda: "disabled",
    )
    monkeypatch.setattr(pipeline_manager, "PROJECT_ROOT", install_dir)
    monkeypatch.setattr(pipeline_manager, "_jobs_path", install_dir / "cache" / "web" / "jobs.json")
    monkeypatch.setattr(pipeline_manager, "_jobs", {})
    monkeypatch.setattr(pipeline_manager, "_cancel_events", {})
    monkeypatch.setattr(pipeline_manager, "_active_retry_recoveries", 0)
    monkeypatch.setattr(pipeline_manager, "_epoch", 0)

    async def _noop_workers(*_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    monkeypatch.setattr(pipeline_manager, "start_workers", _noop_workers)

    mgr = updater.UpdateManager(root=install_dir)
    monkeypatch.setattr(updater, "get_update_manager", lambda: mgr)
    monkeypatch.setattr(updater.os, "name", "nt")
    restart_calls: list[str] = []
    updater.register_update_exit_callback(lambda: restart_calls.append("exit"))

    uv_config = uvicorn.Config(
        create_app(),
        host="127.0.0.1",
        port=app_port,
        log_level="error",
    )
    uv_server = uvicorn.Server(uv_config)
    server_thread = threading.Thread(target=uv_server.run, daemon=True)
    server_thread.start()

    edge_proc = subprocess.Popen(
        [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            f"--remote-debugging-port={cdp_port}",
            f"--user-data-dir={edge_profile}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        _wait_for_http_ready(app_port)
        ws_url = _wait_for_cdp_ws_url(cdp_port)

        async def _drive_browser() -> None:
            async with websockets.connect(ws_url, max_size=8 * 1024 * 1024) as ws:
                page = _CDPPage(ws)
                try:
                    await page.send("Page.enable")
                    await page.send("Runtime.enable")
                    await page.send("Network.enable")
                    await page.send("Log.enable")
                    await page.send("Page.navigate", {"url": f"http://127.0.0.1:{app_port}/"})

                    # 1. Boot: real SSE green dot, version pill, default local backend visible.
                    await page.wait_until(
                        "document.getElementById('conn-dot')?.className === 'conn-dot ok'",
                        desc="SSE green dot (#conn-dot ok)",
                    )
                    await page.wait_until(
                        "document.getElementById('btn-update-trigger')?.textContent === 'v1.9.0'",
                        desc="header version pill v1.9.0",
                    )
                    boot_state = await page.eval(
                        """(() => ({
                            backend: document.getElementById('translation-backend').value,
                            localVisible: getComputedStyle(document.getElementById('llamacpp-backend-fields')).display !== 'none',
                            openaiHidden: getComputedStyle(document.getElementById('openai-backend-fields')).display === 'none',
                            noPresets: document.querySelectorAll('.preset-chip').length === 0,
                            noTuning: document.getElementById('section-tuning') === null,
                            submitDisabled: document.getElementById('btn-submit').disabled,
                        }))()"""
                    )
                    assert boot_state == {
                        "backend": "llamacpp",
                        "localVisible": True,
                        "openaiHidden": True,
                        "noPresets": True,
                        "noTuning": True,
                        "submitDisabled": True,
                    }

                    # 2. Expand translation settings, switch backend llamacpp -> openai -> llamacpp,
                    #    toggle API key visibility, change target language -> subtitle mode labels, save.
                    await page.eval("document.querySelector('#panel-translation > summary').click()")
                    await page.wait_until("document.getElementById('panel-translation').open === true")

                    await page.eval(
                        """(() => {
                            const sel = document.getElementById('translation-backend');
                            sel.value = 'openai';
                            sel.dispatchEvent(new Event('change', { bubbles: true }));
                        })()"""
                    )
                    assert await page.eval(
                        """(() => ({
                            openaiVisible: getComputedStyle(document.getElementById('openai-backend-fields')).display !== 'none',
                            localHidden: getComputedStyle(document.getElementById('llamacpp-backend-fields')).display === 'none',
                        }))()"""
                    ) == {"openaiVisible": True, "localHidden": True}

                    await page.eval("document.getElementById('btn-show-key').click()")
                    assert await page.eval("document.getElementById('api-key').type") == "text"
                    await page.eval("document.getElementById('btn-show-key').click()")
                    assert await page.eval("document.getElementById('api-key').type") == "password"

                    await page.eval(
                        """(() => {
                            const lang = document.getElementById('api-target-lang');
                            lang.value = 'English';
                            lang.dispatchEvent(new Event('change', { bubbles: true }));
                        })()"""
                    )
                    en_labels = await page.eval(
                        "[...document.getElementById('r-mode').options].map(o => o.textContent)"
                    )
                    assert en_labels == ["English", "双语（日文 + English）", "仅日文（不翻译）"]
                    await page.eval(
                        """(() => {
                            const lang = document.getElementById('api-target-lang');
                            lang.value = '简体中文';
                            lang.dispatchEvent(new Event('change', { bubbles: true }));
                            const sel = document.getElementById('translation-backend');
                            sel.value = 'llamacpp';
                            sel.dispatchEvent(new Event('change', { bubbles: true }));
                            document.getElementById('api-glossary').value = 'テスト-测试';
                            document.getElementById('btn-save-translation').click();
                        })()"""
                    )
                    assert await page.eval(
                        """(() => ({
                            localVisible: getComputedStyle(document.getElementById('llamacpp-backend-fields')).display !== 'none',
                            openaiHidden: getComputedStyle(document.getElementById('openai-backend-fields')).display === 'none',
                        }))()"""
                    ) == {"localVisible": True, "openaiHidden": True}
                    await page.wait_until(
                        "[...document.querySelectorAll('.toast')].some(t => t.textContent.includes('config/config.json'))",
                        desc="translation settings saved toast",
                    )
                    saved_cfg = json.loads((install_dir / "config" / "config.json").read_text(encoding="utf-8"))
                    assert saved_cfg["TRANSLATION_BACKEND"] == "llamacpp"
                    assert saved_cfg["TRANSLATION_GLOSSARY"] == "テスト-测试"

                    # 3. Simplified sidebar: no presets/tuning, merged 3-choice subtitle output with dynamic target lang,
                    #    ja_only dimming translation panel, and concurrency field visible only under OpenAI backend.
                    assert await page.eval(
                        """(() => ({
                            presets: document.querySelectorAll('.preset-chip').length,
                            sectionTuning: !!document.getElementById('section-tuning'),
                            panelTuning: !!document.getElementById('panel-tuning'),
                            skipSwitch: !!document.getElementById('r-skip-translation'),
                            saveCustom: !!document.getElementById('btn-save-custom'),
                        }))()"""
                    ) == {
                        "presets": 0,
                        "sectionTuning": False,
                        "panelTuning": False,
                        "skipSwitch": False,
                        "saveCustom": False,
                    }
                    assert await page.eval(
                        """(() => [...document.getElementById('r-mode').options].map(o => [o.value, o.textContent]))()"""
                    ) == [
                        ["zh", "简体中文"],
                        ["bilingual", "双语（日文 + 简体中文）"],
                        ["ja_only", "仅日文（不翻译）"],
                    ]
                    await page.eval(
                        """(() => {
                            const lang = document.getElementById('api-target-lang');
                            lang.value = 'English';
                            lang.dispatchEvent(new Event('change', { bubbles: true }));
                        })()"""
                    )
                    assert await page.eval(
                        """(() => [...document.getElementById('r-mode').options].map(o => [o.value, o.textContent]))()"""
                    ) == [
                        ["zh", "English"],
                        ["bilingual", "双语（日文 + English）"],
                        ["ja_only", "仅日文（不翻译）"],
                    ]
                    await page.eval(
                        """(() => {
                            const lang = document.getElementById('api-target-lang');
                            lang.value = '简体中文';
                            lang.dispatchEvent(new Event('change', { bubbles: true }));
                        })()"""
                    )
                    await page.eval(
                        """(() => {
                            const mode = document.getElementById('r-mode');
                            mode.value = 'ja_only';
                            mode.dispatchEvent(new Event('change', { bubbles: true }));
                        })()"""
                    )
                    assert await page.eval(
                        "document.getElementById('panel-translation').classList.contains('disabled')"
                    )
                    await page.eval(
                        """(() => {
                            const mode = document.getElementById('r-mode');
                            mode.value = 'zh';
                            mode.dispatchEvent(new Event('change', { bubbles: true }));
                        })()"""
                    )
                    assert not await page.eval(
                        "document.getElementById('panel-translation').classList.contains('disabled')"
                    )
                    assert await page.eval(
                        "document.getElementById('t-translation-max-workers').offsetParent === null"
                    )
                    await page.eval(
                        """(() => {
                            const sel = document.getElementById('translation-backend');
                            sel.value = 'openai';
                            sel.dispatchEvent(new Event('change', { bubbles: true }));
                        })()"""
                    )
                    assert await page.eval(
                        "document.getElementById('t-translation-max-workers').offsetParent !== null"
                    )
                    await page.eval(
                        """(() => {
                            const sel = document.getElementById('translation-backend');
                            sel.value = 'llamacpp';
                            sel.dispatchEvent(new Event('change', { bubbles: true }));
                        })()"""
                    )
                    assert await page.eval(
                        "document.getElementById('t-translation-max-workers').offsetParent === null"
                    )

                    # 4. Network proxy panel toggle on -> save -> off -> clear.
                    await page.eval("document.querySelector('#panel-proxy > summary').click()")
                    assert await page.eval("document.getElementById('proxy-status-tag').textContent") == "未启用"
                    assert await page.eval("document.getElementById('proxy-host').disabled") is True
                    await page.eval(
                        """(() => {
                            const en = document.getElementById('proxy-enabled');
                            en.checked = true;
                            en.dispatchEvent(new Event('change', { bubbles: true }));
                        })()"""
                    )
                    assert await page.eval("document.getElementById('proxy-host').disabled") is False
                    await page.eval(
                        """(() => {
                            document.getElementById('proxy-protocol').value = 'socks5';
                            document.getElementById('proxy-host').value = '127.0.0.1';
                            const port = document.getElementById('proxy-port');
                            port.value = '1080';
                            port.dispatchEvent(new Event('change', { bubbles: true }));
                        })()"""
                    )
                    await page.wait_until(
                        "[...document.querySelectorAll('.toast')].some(t => t.textContent.includes('代理设置已保存'))",
                        desc="proxy saved toast",
                    )
                    assert await page.eval("document.getElementById('proxy-status-tag').textContent") == "已启用"
                    proxy_cfg = json.loads((install_dir / "config" / "config.json").read_text(encoding="utf-8"))
                    assert proxy_cfg["PROXY_PROTOCOL"] == "socks5"
                    assert proxy_cfg["PROXY_HOST"] == "127.0.0.1"
                    assert proxy_cfg["PROXY_PORT"] == "1080"

                    await page.eval(
                        """(() => {
                            const en = document.getElementById('proxy-enabled');
                            en.checked = false;
                            en.dispatchEvent(new Event('change', { bubbles: true }));
                        })()"""
                    )
                    await page.wait_until(
                        "[...document.querySelectorAll('.toast')].some(t => t.textContent.includes('代理已关闭'))",
                        desc="proxy disabled toast",
                    )
                    assert await page.eval("document.getElementById('proxy-status-tag').textContent") == "未启用"
                    proxy_cfg_off = json.loads((install_dir / "config" / "config.json").read_text(encoding="utf-8"))
                    assert proxy_cfg_off["PROXY_HOST"] == ""
                    assert proxy_cfg_off["PROXY_PORT"] == ""

                    # 5. Drop file -> pending card -> remove -> re-drop -> submit -> cancel -> clear done + log clear.
                    await page.eval(f"window.__pywebviewDrop([{json.dumps(str(sample_video))}])")
                    assert await page.eval("document.getElementById('btn-submit').disabled") is False
                    assert await page.eval("document.querySelectorAll('.job-card-pending').length") == 1
                    await page.eval("document.querySelector('[data-remove-pending]').click()")
                    assert await page.eval("document.querySelectorAll('.job-card-pending').length") == 0
                    assert await page.eval("document.getElementById('btn-submit').disabled") is True

                    await page.eval(f"window.__pywebviewDrop([{json.dumps(str(sample_video))}])")
                    await page.eval("document.getElementById('btn-submit').click()")
                    await page.wait_until(
                        "document.querySelectorAll('.job-card .badge-queued').length === 1",
                        desc="submitted queued job card",
                    )
                    await page.eval("document.querySelector('[data-cancel]').click()")
                    await page.wait_until(
                        "document.querySelectorAll('.job-card .badge-cancelled').length === 1",
                        desc="cancelled job card",
                    )
                    await page.eval("document.getElementById('btn-clear-done').click()")
                    await page.wait_until(
                        "document.querySelectorAll('.job-card').length === 0",
                        desc="job list cleared after confirm",
                    )
                    assert any(dtype == "confirm" for dtype, _ in page.dialogs)
                    assert not any(dtype == "alert" for dtype, _ in page.dialogs), page.dialogs

                    # Log line click expands, then log-clear clears all lines.
                    assert await page.eval("document.querySelectorAll('#log-scroll .log-line').length > 0")
                    await page.eval("document.querySelector('#log-scroll .log-line').click()")
                    assert await page.eval(
                        "document.querySelector('#log-scroll .log-line').classList.contains('expanded')"
                    )
                    await page.eval("document.getElementById('log-clear').click()")
                    assert await page.eval("document.getElementById('log-scroll').children.length") == 0

                    # 6. Online Update modal: open -> check -> release notes & auto-check toggle -> Escape/close -> reload.
                    await page.eval("document.getElementById('btn-update-trigger').click()")
                    assert await page.eval("document.getElementById('update-modal').hidden") is False
                    await page.eval("document.getElementById('btn-update-check').click()")
                    await page.wait_until(
                        "document.getElementById('btn-update-trigger').textContent === '↑ 新版本 v1.9.1'",
                        desc="header pill shows v1.9.1 available",
                    )
                    modal_state = await page.eval(
                        """(() => ({
                            hasClass: document.getElementById('btn-update-trigger').classList.contains('has-update'),
                            versionInfo: document.getElementById('update-version-info').textContent,
                            notesHidden: document.getElementById('update-notes').hidden,
                            notesText: document.getElementById('update-notes').textContent,
                            startHidden: document.getElementById('btn-update-start').hidden,
                            openHidden: document.getElementById('btn-update-open').hidden,
                        }))()"""
                    )
                    assert modal_state["hasClass"] is True
                    assert "v1.9.0" in modal_state["versionInfo"] and "v1.9.1" in modal_state["versionInfo"]
                    assert modal_state["notesHidden"] is False
                    assert "E2E online self-update test release" in modal_state["notesText"]
                    assert modal_state["startHidden"] is False
                    assert modal_state["openHidden"] is False

                    # Toggle startup auto-check off -> persisted, then back on.
                    await page.eval("document.getElementById('update-auto-check').click()")
                    await page.wait_until(
                        "document.getElementById('update-auto-check').disabled === false && document.getElementById('update-auto-check').checked === false",
                        desc="auto-check toggled off",
                    )
                    assert (
                        json.loads((install_dir / "config" / "config.json").read_text(encoding="utf-8"))[
                            "JAV_SUB_DISABLE_UPDATE_CHECK"
                        ]
                        == "1"
                    )
                    await page.eval("document.getElementById('update-auto-check').click()")
                    await page.wait_until(
                        "document.getElementById('update-auto-check').disabled === false && document.getElementById('update-auto-check').checked === true",
                        desc="auto-check toggled back on",
                    )
                    assert (
                        json.loads((install_dir / "config" / "config.json").read_text(encoding="utf-8"))[
                            "JAV_SUB_DISABLE_UPDATE_CHECK"
                        ]
                        == "0"
                    )

                    # Escape key closes modal; clicking trigger reopens; close button closes.
                    await page.eval("document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))")
                    assert await page.eval("document.getElementById('update-modal').hidden") is True
                    await page.eval("document.getElementById('btn-update-trigger').click()")
                    assert await page.eval("document.getElementById('update-modal').hidden") is False
                    close_inside_modal = await page.eval(
                        """(() => {
                            const m = document.querySelector('.update-modal').getBoundingClientRect();
                            const b = document.getElementById('btn-update-close').getBoundingClientRect();
                            return b.left >= m.left && b.right <= m.right && b.top >= m.top && b.bottom <= m.bottom;
                        })()"""
                    )
                    assert close_inside_modal is True
                    await page.eval("document.getElementById('btn-update-close').click()")
                    assert await page.eval("document.getElementById('update-modal').hidden") is True

                    # 7. Real page reload: verify persistence, green SSE dot reconnect, and zero errors.
                    await page.send("Page.reload")
                    await page.wait_until(
                        "document.getElementById('conn-dot')?.className === 'conn-dot ok'",
                        desc="SSE green dot after reload",
                    )
                    await page.wait_until(
                        "document.getElementById('translation-backend')?.value === 'llamacpp' && document.getElementById('api-glossary')?.value === 'テスト-测试'",
                        desc="persisted backend and glossary after reload",
                    )

                    # 8. Click real one-click update start inside modal -> downloads, verifies manifest, stages pending-update.json, requests restart!
                    await page.eval("document.getElementById('btn-update-trigger').click()")
                    await page.wait_until("document.getElementById('btn-update-start').hidden === false")
                    await page.eval("document.getElementById('btn-update-start').click()")
                    await page.wait_until(
                        "document.getElementById('update-status-text').textContent.includes('自动覆盖重启')",
                        desc="modal reached restarting state after click",
                    )
                    assert restart_calls == ["exit"]
                    pending = install_dir / "tmp" / "updates" / updater.PENDING_UPDATE_FILE_NAME
                    assert pending.is_file()
                    desc = json.loads(pending.read_text(encoding="utf-8"))
                    assert desc["target_version"] == "1.9.1"
                    assert desc["zip_sha256"] == expected_zip_sha
                    assert desc["manifest_verified"] is True

                    # Apply the browser-staged update via bootstrap and verify overlay + cleanup!
                    apply_status, applied_ver = bootstrap.apply_pending_update(
                        install_dir,
                        wait_instances_s=0.0,
                    )
                    assert (apply_status, applied_ver) == ("applied", "1.9.1")
                    assert bootstrap.current_project_version(install_dir) == "1.9.1"
                    assert (install_dir / "jav-sub.exe").read_bytes() == b"fake-exe-1.9.1"
                    assert not (install_dir / bootstrap.UPDATES_DIR).exists()

                    # Zero runtime errors, console errors, unexpected alerts, or HTTP >= 400!
                    assert page.exceptions == [], f"Uncaught JS exceptions: {page.exceptions}"
                    assert page.console_errors == [], f"Console errors: {page.console_errors}"
                    assert page.http_errors == [], f"HTTP >= 400 responses: {page.http_errors}"
                    assert not any(dtype == "alert" for dtype, _ in page.dialogs), page.dialogs
                finally:
                    await page.close()

        asyncio.run(_drive_browser())
    finally:
        updater.register_update_exit_callback(None)
        uv_server.should_exit = True
        server_thread.join(timeout=5.0)
        mock_gh_server.shutdown()
        mock_gh_server.server_close()
        if edge_proc.poll() is None:
            edge_proc.terminate()
            try:
                edge_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                edge_proc.kill()
                edge_proc.wait(timeout=5.0)
