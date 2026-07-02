import atexit
import json
import multiprocessing
import os
import shutil
import sys
import threading
import time
from pathlib import Path

multiprocessing.freeze_support()

_FROZEN = bool(getattr(sys, "frozen", False))
_RESOURCE_ROOT = (
    Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)).resolve()
    if _FROZEN
    else Path(__file__).resolve().parent
)
_ROOT = Path(sys.executable).resolve().parent if _FROZEN else _RESOURCE_ROOT

os.environ.setdefault("JAVTRANS_RUNTIME_ROOT", str(_ROOT))
os.environ.setdefault("JAVTRANS_RESOURCE_ROOT", str(_RESOURCE_ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HOME", str(_ROOT / "models"))
os.environ.setdefault("HF_HUB_CACHE", str(_ROOT / "tmp" / "cache" / "hf" / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(_ROOT / "tmp" / "cache" / "hf" / "xet"))
os.environ.setdefault("TORCH_HOME", str(_ROOT / "tmp" / "cache" / "torch"))

for _SRC in (_RESOURCE_ROOT / "src", _ROOT / "src"):
    if _SRC.exists() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

_BIN_DIR = _RESOURCE_ROOT / "bin"
if _BIN_DIR.exists():
    current_path = os.environ.get("PATH", "")
    bin_text = str(_BIN_DIR)
    if bin_text not in current_path.split(os.pathsep):
        os.environ["PATH"] = bin_text + (os.pathsep + current_path if current_path else "")

_APP_ICON_PATH = _RESOURCE_ROOT / "src" / "assets" / "images" / "icon.ico"


def _webview_icon_arg() -> str | None:
    return str(_APP_ICON_PATH) if _APP_ICON_PATH.exists() else None

PORT = int(os.getenv("JAVTRANS_PORT", "17321"))
EVENTS_PORT = int(os.getenv("JAVTRANS_EVENTS_PORT", "17322"))

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


atexit.register(_cleanup_temp)


def _run_server() -> None:
    import uvicorn
    from web.app import create_app

    os.environ.setdefault("JAVTRANS_EVENTS_PORT", str(EVENTS_PORT))
    uvicorn.run(
        create_app(),
        host="127.0.0.1",
        port=PORT,
        workers=1,
        reload=False,
        log_level="warning",
    )


_shutdown = threading.Event()

if __name__ == "__main__":
    t = threading.Thread(target=_run_server, daemon=True)
    t.start()
    time.sleep(1.5)

    try:
        import webview
        from webview.dom import DOMEventHandler

        def _bind(window):
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
            "JAVTrans",
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
        _shutdown.wait()
