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
os.environ.setdefault("HF_HUB_CACHE", str(_ROOT / "temp" / "hf-cache" / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(_ROOT / "temp" / "hf-cache" / "xet"))
os.environ.setdefault("TORCH_HOME", str(_ROOT / "temp" / "torch"))

for _SRC in (_RESOURCE_ROOT / "src", _ROOT / "src"):
    if _SRC.exists() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

_BIN_DIR = _RESOURCE_ROOT / "bin"
if _BIN_DIR.exists():
    current_path = os.environ.get("PATH", "")
    bin_text = str(_BIN_DIR)
    if bin_text not in current_path.split(os.pathsep):
        os.environ["PATH"] = bin_text + (os.pathsep + current_path if current_path else "")

_ICON_PATH = _RESOURCE_ROOT / "icon.ico"

PORT = int(os.getenv("JAVTRANS_PORT", "17321"))
EVENTS_PORT = int(os.getenv("JAVTRANS_EVENTS_PORT", "17322"))

# Dirs to keep across runs (model caches, reusable state)
_KEEP_DIRS = {
    "temp/hf-cache",
    "temp/audio-separator",
    "temp/web",          # keeps jobs.json + uploads; sub-cleanup below
}

# Globs inside temp/ that are always safe to remove (one-time job artifacts)
_CLEAN_GLOBS = [
    "jobs_*",
    "chunk_*",
    "recovery_*",
    "pytest_*",
    "smoke_api_*",
]


def _cleanup_temp() -> None:
    temp = _ROOT / "temp"
    if not temp.exists():
        return

    # 1. One-time job/chunk dirs at temp root
    for pattern in _CLEAN_GLOBS:
        for p in temp.glob(pattern):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                p.unlink(missing_ok=True)

    # 2. Inside temp/web/jobs/<id>/ — remove audio/ subdirs (large chunks),
    #    keep everything else (checkpoints, aligned_segments.json)
    jobs_root = temp / "web" / "jobs"
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
            icon=str(_ICON_PATH) if _ICON_PATH.exists() else None,
        )
    except ImportError:
        import webbrowser

        webbrowser.open(f"http://127.0.0.1:{PORT}")
        _shutdown.wait()
