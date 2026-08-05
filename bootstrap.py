"""First-run installer: build the runtime environment, then hand off to the app.

The release used to carry torch and the ASR weights inside the PyInstaller
bundle, which made a ~6 GB download for every user and a fresh ~6 GB upload for
every patch. Both are fetchable at run time, so the release can instead be the
launcher plus source, and this script is what turns that into a working install.

**Standard library only, and that is not negotiable.** This runs before any
dependency exists; importing anything from site-packages here would be the same
chicken-and-egg it exists to solve.

The proxy is the reason for the console prompt. Settings written by the web page
land in `.env` as PROXY_PROTOCOL/HOST/PORT, and every later run reads them from
there - but on the very first run there is no `.env` and no web page yet, so the
console is the only place left to ask. The keys written here are exactly the ones
`core.config.network_proxy_url_from_env` reads, so one answer serves both.

Nothing here decides whether the network is "fast enough". It range-reads the
real torch wheel named in `uv.lock`, learns the wheel's true size from the
`Content-Range` header, converts the measured rate into how long that download
would actually take, and lets the user decide from that number.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

_FROZEN = bool(getattr(sys, "frozen", False))
ROOT = (Path(sys.executable) if _FROZEN else Path(__file__)).resolve().parent
ENV_PATH = ROOT / ".env"
VENV_PATH = ROOT / ".venv"
LOCK_PATH = ROOT / "uv.lock"
# In .venv so that deleting the environment also invalidates the record of it.
STAMP_PATH = VENV_PATH / ".jav-trans-install-stamp"
# Touched by launcher.py once its server answers and its window is up. That is
# the first moment at which hiding the console cannot swallow a startup failure.
READY_PATH = ROOT / "tmp" / ".app-ready"

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

# Must match core.config._PROXY_ENV_KEYS. Both cases are set because parts of
# the stack read only the lowercase names.
PROXY_ENV_KEYS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)
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
_UV_WHEEL = re.compile(
    r'href="([^"#]*uv-([0-9][^"#/]*?)-py3-none-win_amd64\.whl)(?:#[^"]*)?"'
)


def log(message: str = "") -> None:
    print(message, flush=True)


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


# --- .env -------------------------------------------------------------------


def _unquote(value: str) -> str:
    """Strip the quotes python-dotenv would strip.

    The web settings page writes this file too, and it quotes every value. Read
    literally, a proxy switched off in the UI (`PROXY_HOST=""`) came back as the
    two-character string `""` - truthy - and the next launch exported
    `HTTP_PROXY=http://"":""` to uv and to the model downloader.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1].strip()
    return value


def read_env_file(path: Path | None = None) -> dict[str, str]:
    """Parse `.env` leniently. A malformed line is skipped, never fatal."""
    values: dict[str, str] = {}
    try:
        text = (path or ENV_PATH).read_text(encoding="utf-8")
    except OSError:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = _unquote(value)
    return values


def update_env_file(updates: dict[str, str], path: Path | None = None) -> None:
    """Set keys in `.env`, preserving every other line as written.

    The web settings page owns this file too, so rewriting it wholesale would
    throw away the user's API keys on the next launch.
    """
    target = path or ENV_PATH
    try:
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        lines = []

    remaining = dict(updates)
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        key = stripped.partition("=")[0].strip() if "=" in stripped else ""
        if key in remaining and not stripped.startswith("#"):
            out.append(f"{key}={remaining.pop(key)}\n")
        else:
            out.append(line)
    if remaining:
        if out and not out[-1].endswith("\n"):
            out.append("\n")
        if out:
            out.append("\n")
        out.append("# --- network proxy (written by the first-run installer) ---\n")
        out.extend(f"{key}={value}\n" for key, value in remaining.items())

    target.write_text("".join(out), encoding="utf-8")


def proxy_url_from(values: dict[str, str]) -> str:
    """Same rule as core.config.network_proxy_url_from_env, restated here
    because importing src/ would mean importing its dependencies."""
    protocol = (values.get("PROXY_PROTOCOL") or "http").strip().lower() or "http"
    host = (values.get("PROXY_HOST") or "").strip()
    port = (values.get("PROXY_PORT") or "").strip()
    if not host or not port:
        return ""
    if protocol not in SUPPORTED_PROXY_PROTOCOLS:
        protocol = "http"
    return f"{protocol}://{host}:{port}"


def apply_proxy(proxy_url: str) -> None:
    """Export the proxy to this process and everything it spawns.

    uv and huggingface_hub both read these, so setting them here is what makes
    one console answer cover dependency install and model download alike.
    """
    for key in PROXY_ENV_KEYS:
        if proxy_url:
            os.environ[key] = proxy_url
        else:
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


def _opener() -> urllib.request.OpenerDirector:
    """Build a fresh opener per probe.

    urllib caches the default opener - and with it the proxy settings that were
    in the environment the first time it was used - so re-testing after the user
    enters a proxy would silently re-measure the direct connection.
    """
    proxies = {}
    for scheme in ("http", "https"):
        value = os.environ.get(f"{scheme.upper()}_PROXY") or os.environ.get(f"{scheme}_proxy")
        # urllib speaks no socks; uv still gets it through the environment.
        if value and not value.lower().startswith("socks"):
            proxies[scheme] = value
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxies))


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
            "User-Agent": "jav-trans-setup",
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
    return tuple(int(part) for part in re.findall(r"\d+", version)[:4])


def latest_uv_wheel_url(index_html: str) -> str:
    """Newest Windows uv wheel from a PyPI simple index page.

    Sorted by version rather than taking the last link: the index is ordered by
    upload time, and a yanked-then-rebuilt patch would otherwise win.
    """
    matches = _UV_WHEEL.findall(index_html)
    if not matches:
        return ""
    url, _version = max(matches, key=lambda item: _version_key(item[1]))
    return url


def download_uv() -> Path | None:
    """Fetch uv from PyPI into `bin/`, reporting progress.

    Reached only when the release did not bundle uv and the machine has none.
    Runs after the proxy interview, so a user who needed a proxy for torch has
    already given one and this download uses it too.
    """
    target = ROOT / "bin" / ("uv.exe" if os.name == "nt" else "uv")
    log("\n未找到 uv，正在从 PyPI 下载（约 20MB）...")
    try:
        with _opener().open(
            urllib.request.Request(
                UV_PYPI_INDEX,
                headers={
                    "User-Agent": "jav-trans-setup",
                    "Accept": "application/vnd.pypi.simple.v1+html",
                },
            ),
            timeout=PROBE_TIMEOUT_S,
        ) as response:
            index_html = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError) as error:
        log(f"  无法读取 PyPI 索引: {error}")
        return None

    url = latest_uv_wheel_url(index_html)
    if not url:
        log("  PyPI 索引里没有 Windows 版 uv。")
        return None

    with tempfile.TemporaryDirectory() as work:
        wheel = Path(work) / "uv.whl"
        try:
            _download_with_progress(url, wheel)
            with zipfile.ZipFile(wheel) as archive:
                names = [
                    name for name in archive.namelist()
                    if name.endswith(("/uv.exe", "/uv")) and "/scripts/" in name.lower()
                ]
                if not names:
                    log("  下载的 uv 包里没有可执行文件。")
                    return None
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(names[0]) as source, open(target, "wb") as sink:
                    shutil.copyfileobj(source, sink)
        except (urllib.error.URLError, OSError, ValueError, zipfile.BadZipFile) as error:
            log(f"  下载失败: {error}")
            return None

    target.chmod(0o755)
    log(f"  已安装 uv: {target}")
    return target


def _download_with_progress(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "jav-trans-setup"})
    with _opener().open(request, timeout=PROBE_TIMEOUT_S) as response:
        length = response.headers.get("Content-Length", "")
        total = int(length) if length.strip().isdigit() else 0
        read = 0
        with open(destination, "wb") as sink:
            while chunk := response.read(262144):
                sink.write(chunk)
                read += len(chunk)
                if total:
                    print(f"\r  {read * 100 // total}%", end="", flush=True)
    if total:
        print("\r  100%", flush=True)


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


def lock_digest() -> str:
    """Identity of the dependency set, so a patched release re-syncs itself."""
    digest = hashlib.sha256()
    for path in (LOCK_PATH, ROOT / "pyproject.toml"):
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()


def environment_is_current() -> bool:
    """Whether the installed environment matches the shipped lockfile.

    A stamp rather than an import probe: importing torch costs seconds and this
    runs on every launch. The stamp is written only after the imports have been
    verified once, so an interrupted sync leaves no stamp and gets redone.
    """
    if not venv_python().is_file():
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
    which audits metadata rather than files. Returns [(import name, reason)] for
    everything that failed, so the repair can name the packages to reinstall.
    """
    targets = list(names) or sorted(CRITICAL_IMPORT_ROOTS)
    script = (
        "import importlib\n"
        f"for name in {targets!r}:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except BaseException as exc:\n"
        f"        print('{_PROBE_PREFIX}' + name + ': '"
        " + type(exc).__name__ + ': ' + str(exc).replace(chr(10), ' '), flush=True)\n"
    )
    try:
        completed = subprocess.run(
            [str(venv_python()), "-c", script],
            cwd=str(ROOT), capture_output=True, timeout=600,
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


def ffmpeg_problem() -> str:
    """"" when ffmpeg is usable, else what is wrong with it.

    Never repaired from here: FFmpeg ships inside the archive, so the only fix is
    to put those files back.
    """
    suffix = ".exe" if os.name == "nt" else ""
    bin_dir = ROOT / "bin"
    if all((bin_dir / f"{name}{suffix}").is_file() for name in ("ffmpeg", "ffprobe")):
        # torchcodec loads the shared libraries rather than the executables, so
        # an ffmpeg.exe whose av*.dll were deleted is still a broken install.
        if os.name == "nt" and not any(bin_dir.glob("avcodec-*.dll")):
            return "bin\\ 里缺少 FFmpeg 的 av*.dll（torchcodec 需要 Shared 版）"
        return ""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
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
        host = input("  代理地址 [留空跳过]: ").strip()
        if not host:
            return None
        port = input(f"  代理端口 [{DEFAULT_PROXY_PORT}]: ").strip() or DEFAULT_PROXY_PORT
        protocol = input("  协议 http/https/socks5 [http]: ").strip().lower() or "http"
    except (EOFError, KeyboardInterrupt):
        return None
    if not port.isdigit():
        log(f"  端口 {port!r} 不是数字，跳过代理设置。")
        return None
    if protocol not in SUPPORTED_PROXY_PROTOCOLS:
        log(f"  不支持的协议 {protocol!r}，按 http 处理。")
        protocol = "http"
    return {"PROXY_PROTOCOL": protocol, "PROXY_HOST": host, "PROXY_PORT": port}


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
    proxy_url = proxy_url_from(read_env_file())
    if proxy_url:
        log(f"使用 .env 中已配置的代理: {proxy_url}")
        apply_proxy(proxy_url)
        return
    if assume_yes or not interactive():
        return

    log("\n正在测速（读取 PyTorch 官方源上的 torch wheel）...")
    log(f"  {describe_speed(measure_download())}")

    if not ask_yes("需要设置代理吗？"):
        return
    settings = prompt_proxy()
    if not settings:
        return
    update_env_file(settings)
    proxy_url = proxy_url_from(settings)
    apply_proxy(proxy_url)
    log(f"已写入 {ENV_PATH.name}，代理: {proxy_url}")
    if proxy_url.startswith("socks5"):
        log("（socks5 代理无法在此复测，uv 下载时仍会使用）")
        return
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


def install(uv: Path, *, reinstall: list[str] | None = None, reinstall_all: bool = False) -> int:
    log("\n=== 安装运行环境 ===")
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
        log("请重新解压一份完整的程序目录（.venv、models、.env 可以从旧目录搬过去）。")
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


def _watch_for_ready(stop: threading.Event) -> None:
    while not stop.wait(0.2):
        if READY_PATH.exists():
            READY_PATH.unlink(missing_ok=True)
            set_console_visible(False)
            return


def launch(extra_args: list[str], *, hide_when_ready: bool = True) -> int:
    log("\n=== 启动 jav-trans ===")
    stop = threading.Event()
    if hide_when_ready:
        try:
            READY_PATH.parent.mkdir(parents=True, exist_ok=True)
            READY_PATH.unlink(missing_ok=True)
            os.environ["JAV_TRANS_READY_FILE"] = str(READY_PATH)
        except OSError:
            os.environ.pop("JAV_TRANS_READY_FILE", None)
        threading.Thread(target=_watch_for_ready, args=(stop,), daemon=True).start()
    else:
        os.environ.pop("JAV_TRANS_READY_FILE", None)
    try:
        code = run_streaming([venv_python(), "launcher.py", *extra_args])
    finally:
        stop.set()
        try:
            READY_PATH.unlink(missing_ok=True)
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
    if code == 0:
        return code

    log(f"\n程序退出，退出码 {code}。")
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
    if code != 0:
        _log_launch_failure_hints()
    return code


def _log_launch_failure_hints() -> None:
    log("  · 窗口打不开：安装 Microsoft Edge WebView2 运行时")
    log("    https://developer.microsoft.com/en-us/microsoft-edge/webview2/")
    log("  · 显卡相关报错：更新 NVIDIA 驱动后重启")
    log("  · 详细错误见上面的输出，以及 tmp\\log\\ 下最新的 .run.log")
    log("  · 想再自检一次：jav-trans.exe --doctor")


def main(argv: list[str] | None = None) -> int:
    # Before argparse, which prints Chinese help text and exits without ever
    # reaching the body of this function.
    enable_utf8_console()

    parser = argparse.ArgumentParser(
        description="jav-trans 启动器（首次运行会先安装运行环境）"
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
        os.environ.setdefault("UV_CACHE_DIR", str(ROOT / "tmp" / "cache" / "uv"))
        os.environ.setdefault("UV_PYTHON_INSTALL_DIR", str(ROOT / "tmp" / "python"))

    # A structural check on every run, not just the first: the stamp says the
    # install once completed, which stops being true the moment someone deletes a
    # folder out of .venv to reclaim disk space. Cheap (a handful of stat calls);
    # the import probe that would catch subtler damage waits for --doctor or for
    # a launch that actually failed.
    report = diagnose()
    broken_after_install = bool(report["venv"]) and not report["payload"]
    needs_install = args.reinstall or not environment_is_current() or broken_after_install

    # This exe is the launcher too - every run after the first goes straight to
    # launch() - so the banner says which of the three is happening instead of
    # calling itself an installer forever.
    if args.doctor:
        log("jav-trans 环境自检")
    elif not venv_python().is_file():
        log("jav-trans 首次运行安装")
    elif report["broken"]:
        log("jav-trans 修复运行环境")
    elif needs_install:
        log("jav-trans 更新运行环境")
    else:
        log("jav-trans 启动")
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
        return launch_with_recovery(
            passthrough,
            assume_yes=args.yes,
            repaired_already=True,
            hide_when_ready=not args.keep_console,
            allow_repair=may_repair,
        )

    if report["payload"]:
        log("\n程序文件不完整，缺少：" + "、".join(report["payload"]))
        log("这些文件在压缩包里，无法自动补回：请重新解压一份完整的程序目录")
        log("（.venv、models、.env 可以从旧目录搬过去，不用重新下载）。")
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
        apply_proxy(args.proxy.strip())
        log(f"使用命令行指定的代理: {args.proxy.strip()}")
    elif needs_install:
        ensure_proxy_configured(assume_yes=args.yes)
    else:
        # Already installed: hand the stored proxy to the app for model
        # downloads, but do not interrogate the user on every launch.
        apply_proxy(proxy_url_from(read_env_file()))

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
        code = install(uv, reinstall=list(report["broken"]))
        if code != 0:
            log(f"\n安装失败（退出码 {code}）。")
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
    return launch_with_recovery(
        passthrough,
        assume_yes=args.yes,
        repaired_already=needs_install,
        hide_when_ready=not args.keep_console,
    )


if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        pause_before_exit()
    sys.exit(exit_code)
