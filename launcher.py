import atexit
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

multiprocessing.freeze_support()

_FROZEN = bool(getattr(sys, "frozen", False))
_RESOURCE_ROOT = (
    Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)).resolve()
    if _FROZEN
    else Path(__file__).resolve().parent
)
_ROOT = Path(sys.executable).resolve().parent if _FROZEN else _RESOURCE_ROOT

os.environ.setdefault("JAV_TRANS_RUNTIME_ROOT", str(_ROOT))
os.environ.setdefault("JAV_TRANS_RESOURCE_ROOT", str(_RESOURCE_ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HOME", str(_ROOT / "models"))
os.environ.setdefault("HF_HUB_CACHE", str(_ROOT / "models" / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(_ROOT / "models" / "xet"))
os.environ.setdefault("TORCH_HOME", str(_ROOT / "tmp" / "cache" / "torch"))

for _SRC in (_RESOURCE_ROOT / "src", _ROOT / "src"):
    if _SRC.exists() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

from utils.subprocess_tools import no_window_subprocess_kwargs
from utils.ffmpeg_runtime import configure_ffmpeg_shared_runtime

_BIN_DIR = _RESOURCE_ROOT / "bin"
if _BIN_DIR.exists():
    current_path = os.environ.get("PATH", "")
    bin_text = str(_BIN_DIR)
    if bin_text not in current_path.split(os.pathsep):
        os.environ["PATH"] = bin_text + (os.pathsep + current_path if current_path else "")

_FFMPEG_SHARED_DIRECTORIES = configure_ffmpeg_shared_runtime()

_APP_ICON_PATH = _RESOURCE_ROOT / "src" / "assets" / "images" / "icon.ico"
_CUDA_PROBE_CHILD = "--cuda-probe-child" in sys.argv
_SMOKE_IMPORTS = "--smoke-imports" in sys.argv


def _webview_icon_arg() -> str | None:
    return str(_APP_ICON_PATH) if _APP_ICON_PATH.exists() else None

def _first_free_port(preferred: int, attempts: int = 20) -> int:
    """`preferred` if it is free, else the next port that is.

    The console runs on a fixed default, and a default is a number someone
    else also picked: another app holding it used to surface as a traceback in
    a daemon thread while the window opened anyway - pointing at whatever was
    already listening there. Walking forward is what Jupyter does with 8888.
    """
    import socket

    for offset in range(max(1, attempts)):
        candidate = preferred + offset
        if candidate > 65535:
            break
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                continue
        return candidate
    return preferred


PORT = _first_free_port(int(os.getenv("JAV_TRANS_PORT", "2233")))
EVENTS_PORT = _first_free_port(int(os.getenv("JAV_TRANS_EVENTS_PORT", "2234")))
if EVENTS_PORT == PORT:
    EVENTS_PORT = _first_free_port(PORT + 1)

# Globs inside tmp/ that are always safe to remove (one-time run artifacts).
# Coverage note: atexit cleanup is best-effort; it only sweeps these top-level
# tmp/ entries plus tmp/web/jobs/<id>/audio/ (see below). It deliberately leaves
# tmp/cache/ (model caches, reused across runs) and tmp/web/ (jobs.json + uploads)
# untouched. The legacy names jobs_*/chunk_* never matched — runtime now writes
# intermediate per-chunk audio under tmp/chunks/ and job state under tmp/web/jobs/.
_CLEAN_GLOBS = [
    "chunks",            # tmp/chunks/<id> — intermediate per-chunk audio segments
    "asr_timeouts",      # tmp/asr_timeouts/ — ASR timeout diagnostic logs
    "pytest",            # tmp/pytest/ — pytest tmp_path artifacts
    "recovery_*",        # tmp/recovery_* — recovery state if present
    "smoke_api_*",       # tmp/smoke_api_* — smoke-test artifacts if present
]


def _cleanup_temp() -> None:
    tmp = _ROOT / "tmp"
    if not tmp.exists():
        return

    # 1. One-time job/chunk dirs at tmp root
    for pattern in _CLEAN_GLOBS:
        for p in tmp.glob(pattern):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                p.unlink(missing_ok=True)

    # 2. Inside tmp/web/jobs/<id>/ - remove audio/ subdirs (large chunks),
    #    keep everything else (checkpoints, aligned_segments.json)
    jobs_root = tmp / "web" / "jobs"
    if jobs_root.exists():
        for job_dir in jobs_root.iterdir():
            if job_dir.is_dir():
                audio_dir = job_dir / "audio"
                if audio_dir.exists():
                    shutil.rmtree(audio_dir, ignore_errors=True)


def _version_tuple(value: str | None) -> tuple[int, ...]:
    parts: list[int] = []
    for token in str(value or "").split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _find_nvidia_smi() -> str:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", "C:\\Program Files"))
        candidate = candidate / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
        if candidate.exists():
            return str(candidate)
    return ""


def _probe_nvidia_smi() -> dict:
    exe = _find_nvidia_smi()
    if not exe:
        return {"available": False, "error": "nvidia-smi not found"}
    result: dict = {"available": True, "path": exe}
    try:
        summary = subprocess.run(
            [exe],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            **no_window_subprocess_kwargs(),
        )
        result["returncode"] = summary.returncode
        if summary.returncode != 0:
            result["error"] = (summary.stderr or summary.stdout).strip()
        match = None
        for line in (summary.stdout or "").splitlines():
            if "CUDA Version:" in line:
                match = line
                break
        if match:
            import re

            found = re.search(r"CUDA Version:\s*([0-9.]+)", match)
            if found:
                result["cuda_version"] = found.group(1)
    except Exception as exc:  # noqa: BLE001 - diagnostic path
        result["available"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    try:
        query = subprocess.run(
            [exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            **no_window_subprocess_kwargs(),
        )
        devices = []
        for line in (query.stdout or "").splitlines():
            if not line.strip():
                continue
            name, _, driver = line.partition(",")
            devices.append(
                {
                    "name": name.strip(),
                    "driver_version": driver.strip(),
                }
            )
        if devices:
            result["devices"] = devices
            result["driver_version"] = devices[0].get("driver_version", "")
    except Exception as exc:  # noqa: BLE001 - optional diagnostic
        result["query_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _cuda_probe_payload() -> dict:
    payload: dict = {
        "status": "error",
        "ok": False,
        "code": "unknown",
        "message": "CUDA 环境检测失败。",
        "nvidia_smi": _probe_nvidia_smi(),
    }
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - user-facing diagnostic
        payload.update(
            {
                "code": "torch_import_failed",
                "message": f"PyTorch 运行时加载失败：{type(exc).__name__}: {exc}",
            }
        )
        return payload

    torch_cuda = str(getattr(torch.version, "cuda", "") or "")
    payload.update(
        {
            "torch_version": str(getattr(torch, "__version__", "")),
            "torch_cuda_version": torch_cuda,
        }
    )
    if not torch_cuda:
        payload.update(
            {
                "code": "cpu_torch",
                "message": "当前打包/安装的是 CPU 版 PyTorch，无法使用 NVIDIA GPU 加速。",
            }
        )
        return payload

    cuda_available = False
    cuda_error = ""
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001 - user-facing diagnostic
        cuda_error = f"{type(exc).__name__}: {exc}"
    payload["cuda_available"] = cuda_available
    if cuda_error:
        payload["cuda_error"] = cuda_error

    smi = payload.get("nvidia_smi") if isinstance(payload.get("nvidia_smi"), dict) else {}
    driver_cuda = str(smi.get("cuda_version") or "")
    driver_version = _version_tuple(driver_cuda)
    torch_version = _version_tuple(torch_cuda)
    if not cuda_available and driver_version and torch_version and driver_version < torch_version:
        payload.update(
            {
                "code": "driver_too_old",
                "message": (
                    f"NVIDIA 驱动最高支持 CUDA {driver_cuda}，但当前打包的 PyTorch "
                    f"需要 CUDA {torch_cuda}。请更新 NVIDIA 显卡驱动后重启应用。"
                ),
            }
        )
        return payload

    if not cuda_available:
        if smi.get("available"):
            message = (
                f"检测到 NVIDIA 驱动，但 PyTorch CUDA {torch_cuda} 初始化失败。"
                "请更新 NVIDIA 显卡驱动，或确认当前显卡/驱动支持该 CUDA runtime。"
            )
        else:
            message = (
                "未检测到可用 NVIDIA GPU 或 NVIDIA 驱动。"
                f"当前打包的 PyTorch 需要 CUDA {torch_cuda}。"
            )
        if cuda_error:
            message = f"{message} 原始错误：{cuda_error}"
        payload.update({"code": "cuda_unavailable", "message": message})
        return payload

    devices = []
    try:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": f"{props.major}.{props.minor}",
                    "total_memory_mb": round(props.total_memory / 1024 / 1024, 1),
                }
            )
    except Exception as exc:  # noqa: BLE001 - optional diagnostic
        payload["device_query_error"] = f"{type(exc).__name__}: {exc}"
    payload.update(
        {
            "status": "ok",
            "ok": True,
            "code": "ok",
            "message": f"CUDA 可用，PyTorch CUDA runtime {torch_cuda}。",
            "devices": devices,
        }
    )
    return payload


def _write_or_print_json(payload: dict, *, output_env: str) -> None:
    payload_text = json.dumps(payload, ensure_ascii=False)
    output_path = os.getenv(output_env, "").strip()
    if output_path:
        Path(output_path).expanduser().write_text(payload_text, encoding="utf-8")
    else:
        print(payload_text, flush=True)


if _CUDA_PROBE_CHILD:
    _write_or_print_json(_cuda_probe_payload(), output_env="JAV_TRANS_CUDA_PROBE_OUTPUT")
    raise SystemExit(0)


if _SMOKE_IMPORTS:
    try:
        import torchcodec
        from torchcodec._core import core_library_path, ffmpeg_major_version

        _write_or_print_json(
            {
                "status": "ok",
                "ok": True,
                "torchcodec_version": getattr(torchcodec, "__version__", ""),
                "torchcodec_ffmpeg_major_version": ffmpeg_major_version,
                "torchcodec_core_library_path": str(core_library_path),
            },
            output_env="JAV_TRANS_SMOKE_OUTPUT",
        )
        raise SystemExit(0)
    except Exception as exc:  # noqa: BLE001 - packaging smoke must report all import failures
        _write_or_print_json(
            {
                "status": "error",
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            output_env="JAV_TRANS_SMOKE_OUTPUT",
        )
        raise SystemExit(1)


atexit.register(_cleanup_temp)


_server_error: list[BaseException] = []


def _run_server() -> None:
    try:
        import uvicorn
        from web.app import create_app

        # Assignment, not setdefault: EVENTS_PORT already started from whatever
        # the env asked for, and if that one was taken the app has to hear about
        # the port we actually resolved.
        os.environ["JAV_TRANS_EVENTS_PORT"] = str(EVENTS_PORT)
        uvicorn.run(
            create_app(),
            host="127.0.0.1",
            port=PORT,
            workers=1,
            reload=False,
            log_level="warning",
        )
    except BaseException as exc:  # noqa: BLE001 - a daemon thread swallows this
        # Without this the thread died quietly and the window still opened, onto
        # a port nothing was listening on: a broken .venv looked like a blank app
        # rather than like a missing dependency.
        _server_error.append(exc)
        traceback.print_exc()


def _wait_for_server(port: int, timeout_s: float = 60.0) -> bool:
    """Block until the server accepts a connection, or it is clearly not coming.

    Replaces a flat 1.5s sleep, which was a race on a cold start and, worse, hid
    the case where the server thread had already crashed.
    """
    import socket

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _server_error:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.15)
    return False


def _signal_app_ready() -> None:
    """Tell the launcher exe that the app is really up.

    It hides its console window on this signal, so it is only ever sent once the
    server answers and the window exists - anything earlier would hide the very
    output that explains a failed start.
    """
    path = os.getenv("JAV_TRANS_READY_FILE", "").strip()
    if not path:
        return
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass


_shutdown = threading.Event()

if __name__ == "__main__":
    print(f"jav-trans: http://127.0.0.1:{PORT}", flush=True)
    t = threading.Thread(target=_run_server, daemon=True)
    t.start()
    if not _wait_for_server(PORT):
        print("\n服务没能启动，界面无法打开。", flush=True)
        if _server_error:
            print(f"原因: {type(_server_error[0]).__name__}: {_server_error[0]}", flush=True)
        else:
            print(f"端口 {PORT} 在 60 秒内没有响应。", flush=True)
        print("依赖可能不完整，可以运行 jav-trans.exe --doctor 自检并修复。", flush=True)
        raise SystemExit(1)

    try:
        import webview
        from webview.dom import DOMEventHandler

        def _bind(window):
            # pywebview calls this once the window exists, which is the moment
            # the launcher exe is waiting for to hide its console.
            _signal_app_ready()

            def _on_drop(e):
                files = e.get("dataTransfer", {}).get("files", [])
                paths = [f.get("pywebviewFullPath") for f in files if f.get("pywebviewFullPath")]
                if paths:
                    window.evaluate_js(f"window.__pywebviewDrop({json.dumps(paths)})")

            def _on_dragover(e):
                pass

            window.dom.document.events.dragover += DOMEventHandler(_on_dragover, True, False)
            window.dom.document.events.drop += DOMEventHandler(_on_drop, True, False)

        win = webview.create_window(
            "jav-trans",
            f"http://127.0.0.1:{PORT}",
            width=1280,
            height=820,
            resizable=True,
        )
        webview.start(
            _bind,
            win,
            icon=_webview_icon_arg(),
        )
    except ImportError:
        import webbrowser

        webbrowser.open(f"http://127.0.0.1:{PORT}")
        _signal_app_ready()
        _shutdown.wait()
