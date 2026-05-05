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

_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

PORT = int(os.getenv("JAVTRANS_PORT", "17321"))
EVENTS_PORT = int(os.getenv("JAVTRANS_EVENTS_PORT", "17322"))

_ROOT = Path(__file__).parent

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
        webview.start(_bind, win)
    except ImportError:
        import webbrowser

        webbrowser.open(f"http://127.0.0.1:{PORT}")
        _shutdown.wait()
