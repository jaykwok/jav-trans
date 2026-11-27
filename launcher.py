from __future__ import annotations

import atexit
import json
import logging
import multiprocessing
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

_FROZEN = bool(getattr(sys, "frozen", False))
_RESOURCE_ROOT = (
    Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)).resolve()
    if _FROZEN
    else Path(__file__).resolve().parent
)
_ROOT = Path(sys.executable).resolve().parent if _FROZEN else _RESOURCE_ROOT
_APP_ICON_PATH = _RESOURCE_ROOT / "src" / "assets" / "images" / "icon.ico"

_log = logging.getLogger("jav_sub.launcher")

PORT: int = 2233
EVENTS_PORT: int = 2234
_FFMPEG_SHARED_DIRECTORIES: tuple[str, ...] = ()

# Globs inside tmp/ that are always safe to remove (one-time run artifacts).
# Coverage note: atexit cleanup is best-effort; it sweeps these top-level
# tmp/ entries plus tmp/jobs/<id>/audio/ for non-failed jobs (failed jobs keep
# their extracted WAV so users can reproduce or retry without re-extracting).
# Persistent caches (cache/), logs (logs/), committed outputs (outputs/), and
# private settings (config/) live outside tmp/ and are never touched here.
_CLEAN_GLOBS = [
    "chunks",            # tmp/chunks/<id> — intermediate per-chunk audio segments
    "asr_timeouts",      # tmp/asr_timeouts/ — ASR timeout diagnostic logs
    "pytest",            # tmp/pytest/ — pytest tmp_path artifacts
    "recovery_*",        # tmp/recovery_* — recovery state if present
    "smoke_api_*",       # tmp/smoke_api_* — smoke-test artifacts if present
]

# Which windows are running out of this workspace. The sweep below deletes
# tmp/chunks and every job's audio/, and a second window is not a second
# workspace: ports step forward on collision, so two instances share one tmp/
# and the first one closed used to delete the chunks the other was still
# transcribing. A file per instance, held open for the life of the process, is
# what makes "someone else is still running" a fact rather than an assumption.
_INSTANCE_DIR_NAME = "launcher"
_instance_slot = None
_cleanup_registered = False

# A slot only proves nobody else was there at the instant of the scan, and the
# sweep runs after the scan has let go. Without a gate a second window could
# publish its slot in between - passing unseen - and start writing chunks that
# the first window's sweep then deleted. Publishing a slot and scanning-then-
# sweeping each take this one OS-held lock, which leaves only two orderings, both
# safe: the newcomer is already visible to the scan (veto), or it arrives after
# the sweep finished (and had nothing there to lose).
_GATE_NAME = "workspace.gate"
_GATE_CLAIM_WAIT_S = 20.0
_GATE_SWEEP_WAIT_S = 5.0
_GATE_POLL_S = 0.05

_server_error: list[BaseException] = []
_server_failed = threading.Event()
_shutdown = threading.Event()
_update_restart_requested = threading.Event()
_uvicorn_server = None
_active_window = None


def _ensure_src_on_path() -> None:
    for src_dir in (_RESOURCE_ROOT / "src", _ROOT / "src"):
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))


def _configure_runtime_environment() -> tuple[str, ...]:
    """Configure runtime env vars, sys.path, PATH, and FFmpeg shared DLL dirs."""
    global _FFMPEG_SHARED_DIRECTORIES

    os.environ.setdefault("JAV_SUB_RUNTIME_ROOT", str(_ROOT))
    os.environ.setdefault("JAV_SUB_RESOURCE_ROOT", str(_RESOURCE_ROOT))
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("HF_HOME", str(_ROOT / "models"))
    os.environ.setdefault("HF_HUB_CACHE", str(_ROOT / "models" / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(_ROOT / "models" / "xet"))
    os.environ.setdefault("TORCH_HOME", str(_ROOT / "cache" / "torch"))

    _ensure_src_on_path()

    bin_dir = _RESOURCE_ROOT / "bin"
    if bin_dir.exists():
        current_path = os.environ.get("PATH", "")
        bin_text = str(bin_dir)
        if bin_text not in current_path.split(os.pathsep):
            os.environ["PATH"] = bin_text + (os.pathsep + current_path if current_path else "")

    from utils.ffmpeg_runtime import configure_ffmpeg_shared_runtime

    _FFMPEG_SHARED_DIRECTORIES = configure_ffmpeg_shared_runtime()
    return _FFMPEG_SHARED_DIRECTORIES


def _has_console() -> bool:
    return sys.stdout is not None and sys.stderr is not None


def _show_error_dialog(title: str, message: str) -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes

        # MB_OK (0x0) | MB_ICONERROR (0x10)
        ctypes.windll.user32.MessageBoxW(0, str(message), str(title), 0x10)
        return True
    except Exception:  # noqa: BLE001 - best-effort GUI error notice
        _log.debug("Could not display Windows error dialog", exc_info=True)
        return False


def _report_fatal_startup_error(summary: str, details: list[str] | None = None) -> None:
    lines = [line for line in (summary, *(details or [])) if line]
    full_text = "\n".join(lines)
    _log.error("%s", full_text)
    if _has_console():
        for line in lines:
            print(line, flush=True)
    if _FROZEN or not _has_console():
        _show_error_dialog("jav-sub 启动失败", full_text)


def _webview_icon_arg() -> str | None:
    return str(_APP_ICON_PATH) if _APP_ICON_PATH.exists() else None


def _port_accepts_connection(port: int, timeout_s: float = 0.05) -> bool:
    """Return True if another process already accepts TCP connections on 127.0.0.1:<port>.

    On Windows, binding 127.0.0.1:<port> can succeed even when another service
    already listens on 0.0.0.0:<port>. A short-timeout connect_ex probe detects
    that wildcard listener before attempting bind().
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout_s)
        try:
            return probe.connect_ex(("127.0.0.1", port)) == 0
        except OSError:
            return False


def _bind_free_socket(
    preferred: int,
    attempts: int = 20,
    *,
    host: str = "127.0.0.1",
    exclude: set[int] | None = None,
) -> tuple[socket.socket, int]:
    """Bind and return an exclusive listening socket starting at `preferred`.

    Probes `connect_ex` first to skip ports already served by a `0.0.0.0`
    listener on Windows, sets `SO_EXCLUSIVEADDRUSE` on Windows (never
    `SO_REUSEADDR`) to prevent double-binding, and keeps the socket bound so
    uvicorn can receive it directly via `sockets=[sock]` without a TOCTOU race.
    """
    skipped = exclude or set()
    last_error: OSError | None = None
    for offset in range(max(1, attempts)):
        candidate = preferred + offset
        if candidate > 65535:
            break
        if candidate in skipped:
            continue
        if _port_accepts_connection(candidate):
            continue
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            sock.bind((host, candidate))
            sock.set_inheritable(False)
            return sock, candidate
        except OSError as exc:
            last_error = exc
            sock.close()
    raise OSError(
        f"No free TCP port available starting at {preferred} (tried {max(1, attempts)} ports)"
    ) from last_error


def _first_free_port(
    preferred: int,
    attempts: int = 20,
    *,
    exclude: set[int] | None = None,
) -> int:
    """`preferred` if it is free, else the next port that is.

    The console runs on a fixed default, and a default is a number someone
    else also picked: another app holding it used to surface as a traceback in
    a daemon thread while the window opened anyway - pointing at whatever was
    already listening there. Walking forward is what Jupyter does with 8888.
    """
    try:
        sock, port = _bind_free_socket(preferred, attempts=attempts, exclude=exclude)
    except OSError:
        return preferred
    sock.close()
    return port


def _resolve_events_port(
    web_port: int,
    preferred_events_port: int,
    *,
    finder=None,
) -> int:
    """Resolve an events port distinct from `web_port` without exceeding 65535."""
    locate = finder if finder is not None else _first_free_port
    candidate = locate(preferred_events_port)
    if candidate == web_port:
        fallback = web_port + 1 if web_port < 65535 else 2234
        candidate = locate(fallback)
    return candidate


def _instance_dir(root: Path) -> Path:
    return root / "tmp" / _INSTANCE_DIR_NAME


def _try_lock(handle) -> bool:
    """Take the OS lock on an open handle without waiting for it."""
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


def _acquire_gate(root: Path, wait_s: float, note: str = ""):
    """The workspace gate, or None if it stays held for longer than `wait_s`.

    The lock lives in the OS, so a window that crashes while holding it releases
    it on the way out rather than freezing every future window.
    """
    try:
        directory = _instance_dir(root)
        directory.mkdir(parents=True, exist_ok=True)
        handle = open(directory / _GATE_NAME, "a+", encoding="utf-8")
    except OSError:
        return None
    deadline = time.monotonic() + max(0.0, wait_s)
    announced = False
    while True:
        if _try_lock(handle):
            return handle
        if time.monotonic() >= deadline:
            handle.close()
            return None
        if note and not announced:
            announced = True
            if _has_console():
                print(note, flush=True)
        time.sleep(_GATE_POLL_S)


def _release_gate(handle) -> None:
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


def _claim_instance_slot(root: Path):
    """Publish this window's slot under the gate, or None if it cannot be.

    None is not "start anyway without cleaning up on the way out". A window
    writes chunk audio into the same tmp/ everyone sweeps, so an unregistered
    window is invisible to the next scan and its work is a deletion target -
    which is the whole failure this slot exists to prevent. Whoever cannot
    register does not start.
    """
    gate = _acquire_gate(
        root,
        _GATE_CLAIM_WAIT_S,
        note="[launcher] 另一个窗口正在清理临时文件，等待它完成…",
    )
    if gate is None:
        _log.warning(
            "Timed out waiting %ss for workspace gate during instance registration",
            int(_GATE_CLAIM_WAIT_S),
        )
        if _has_console():
            print(
                f"[launcher] 等待另一个窗口清理超过 {int(_GATE_CLAIM_WAIT_S)} 秒，放弃登记。",
                flush=True,
            )
        return None
    try:
        return _publish_instance_slot(root)
    finally:
        _release_gate(gate)


def _publish_instance_slot(root: Path):
    """Write and hold this window's slot file. Callers must hold the gate."""
    try:
        directory = _instance_dir(root)
        directory.mkdir(parents=True, exist_ok=True)
        handle = open(
            directory / f"instance-{os.getpid()}.lock", "w", encoding="utf-8"
        )
    except OSError:
        return None
    try:
        handle.write(json.dumps({"pid": os.getpid(), "started": time.time()}))
        handle.flush()
        if os.name != "nt":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def _slot_is_held(path: Path) -> bool:
    """Is a live process still holding this slot?

    Two mechanisms because the platforms answer differently. On Windows an open
    file cannot be unlinked, so a failed delete *is* the liveness proof, and a
    successful one clears a slot left behind by a crash. Elsewhere the delete
    would always succeed, so the lock has to be asked directly.
    """
    if os.name == "nt":
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except Exception:  # noqa: BLE001 - treat un-unlinkable slot conservatively as held
            return True
        return False
    try:
        import fcntl

        with open(path, "r+", encoding="utf-8") as probe:
            try:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
    except FileNotFoundError:
        return False
    except OSError:
        # Unreadable is not "nobody owns it": refuse to sweep on a maybe.
        return True
    path.unlink(missing_ok=True)
    return False


def _other_instances_running(root: Path, *, own_slot=None) -> bool:
    own_name = Path(getattr(own_slot, "name", "")).name
    try:
        slots = sorted(_instance_dir(root).glob("instance-*.lock"))
    except OSError:
        return True
    for slot in slots:
        if slot.name == own_name:
            continue
        if _slot_is_held(slot):
            return True
    return False


def _failed_job_dir_names(root: Path) -> set[str]:
    """Sanitized tmp/jobs/<id> directory names belonging to failed Web jobs."""
    jobs_file = root / "cache" / "web" / "jobs.json"
    if not jobs_file.is_file():
        return set()
    try:
        payload = json.loads(jobs_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    if not isinstance(payload, list):
        return set()
    _ensure_src_on_path()
    from pipeline.ids import sanitize_job_id

    failed: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "failed":
            continue
        raw_id = str(item.get("id") or "").strip()
        if raw_id:
            failed.add(sanitize_job_id(raw_id))
    return failed


def _sweep_temp(root: Path) -> None:
    tmp = root / "tmp"
    if not tmp.exists():
        return

    # 1. One-time job/chunk dirs at tmp root
    for pattern in _CLEAN_GLOBS:
        for p in tmp.glob(pattern):
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.is_file():
                    p.unlink(missing_ok=True)
            except (Exception, SystemExit) as exc:
                _log.warning("Temp sweep skipped %s: %r", p, exc)

    # 2. Inside tmp/jobs/<id>/ - remove audio/ subdirs for non-failed jobs
    #    (failed jobs preserve their extracted WAV for troubleshooting/repro),
    #    keeping checkpoints and aligned_segments.json intact.
    failed_job_dirs = _failed_job_dir_names(root)
    jobs_root = tmp / "jobs"
    if jobs_root.exists():
        for job_dir in jobs_root.iterdir():
            if not job_dir.is_dir():
                continue
            if job_dir.name in failed_job_dirs:
                continue
            audio_dir = job_dir / "audio"
            if audio_dir.exists():
                try:
                    shutil.rmtree(audio_dir, ignore_errors=True)
                except (Exception, SystemExit) as exc:
                    _log.warning("Temp sweep skipped %s: %r", audio_dir, exc)


def _cleanup_temp() -> None:
    global _instance_slot
    slot = _instance_slot
    try:
        # No slot means we never established who owns this workspace, and a
        # sweep on a guess is what this guard exists to stop.
        if slot is None:
            return
        # Scan and sweep are one step, not two: a window registering right now
        # is invisible to the scan, so if the gate cannot be taken the sweep is
        # skipped outright rather than run against a list already out of date.
        gate = _acquire_gate(_ROOT, _GATE_SWEEP_WAIT_S)
        if gate is None:
            return
        try:
            if not _other_instances_running(_ROOT, own_slot=slot):
                _sweep_temp(_ROOT)
        finally:
            _release_gate(gate)
    except Exception as exc:  # noqa: BLE001 - best-effort atexit cleanup
        _log.warning("Exit temp cleanup interrupted: %r", exc)
    finally:
        _instance_slot = None
        if slot is not None:
            try:
                slot.close()
                Path(slot.name).unlink(missing_ok=True)
            except OSError as exc:
                _log.warning("Could not remove launcher instance slot: %r", exc)


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
    _ensure_src_on_path()
    from utils.subprocess_tools import no_window_subprocess_kwargs

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
    elif _has_console():
        print(payload_text, flush=True)


def _run_smoke_imports() -> int:
    """Verify TorchCodec and FFmpeg shared DLL loading inside the runtime."""
    try:
        import torchcodec
        from torchcodec._core import (
            core_library_path,
            ffmpeg_major_version,
            get_ffmpeg_library_versions,
        )

        if ffmpeg_major_version is None:
            raise RuntimeError("torchcodec did not detect a usable FFmpeg major version")
        core_path = Path(str(core_library_path)) if core_library_path else None
        if core_path is None or not core_path.is_file():
            raise RuntimeError(
                f"torchcodec core library path does not exist: {core_library_path!r}"
            )
        ffmpeg_versions = get_ffmpeg_library_versions()
        if not isinstance(ffmpeg_versions, dict) or not ffmpeg_versions:
            raise RuntimeError("torchcodec failed to load FFmpeg shared libraries")

        _write_or_print_json(
            {
                "status": "ok",
                "ok": True,
                "torchcodec_version": getattr(torchcodec, "__version__", ""),
                "torchcodec_ffmpeg_major_version": ffmpeg_major_version,
                "torchcodec_core_library_path": str(core_path),
                "torchcodec_ffmpeg_library_versions": ffmpeg_versions,
            },
            output_env="JAV_SUB_SMOKE_OUTPUT",
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - packaging smoke must report all import failures
        _write_or_print_json(
            {
                "status": "error",
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            output_env="JAV_SUB_SMOKE_OUTPUT",
        )
        return 1


def _request_update_exit() -> None:
    """Ask the running desktop window and uvicorn server to exit cleanly for self-update."""
    if _update_restart_requested.is_set():
        return
    _update_restart_requested.set()
    _log.info("Update staged; initiating graceful shutdown to apply update")

    def _finish_shutdown() -> None:
        # Allow the HTTP response confirming the staged update to flush first.
        time.sleep(0.35)
        _shutdown.set()
        window = _active_window
        if window is not None:
            try:
                window.destroy()
            except Exception:
                _log.debug("Could not destroy pywebview window during update exit", exc_info=True)
        server = _uvicorn_server
        if server is not None:
            server.should_exit = True

    threading.Thread(target=_finish_shutdown, name="update-shutdown", daemon=True).start()


def _run_server(server_sock: socket.socket | None = None) -> None:
    global _uvicorn_server
    try:
        import uvicorn
        from core.updater import register_update_exit_callback
        from web.app import create_app

        register_update_exit_callback(_request_update_exit)
        # Assignment, not setdefault: EVENTS_PORT already started from whatever
        # the env asked for, and if that one was taken the app has to hear about
        # the port we actually resolved.
        os.environ["JAV_SUB_EVENTS_PORT"] = str(EVENTS_PORT)
        _log.info("Starting uvicorn server on 127.0.0.1:%s (events_port=%s)", PORT, EVENTS_PORT)
        config = uvicorn.Config(
            create_app(),
            host="127.0.0.1",
            port=PORT,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        _uvicorn_server = server
        if server_sock is not None:
            server.run(sockets=[server_sock])
        else:
            server.run()
    except BaseException as exc:  # noqa: BLE001 - a daemon thread swallows this
        # Without this the thread died quietly and the window still opened, onto
        # a port nothing was listening on: a broken .venv looked like a blank app
        # rather than like a missing dependency.
        _server_error.append(exc)
        _log.exception("Web server thread crashed")
        if _has_console():
            traceback.print_exc()
        _server_failed.set()
        _shutdown.set()
    finally:
        if server_sock is not None:
            try:
                server_sock.close()
            except OSError:
                pass


def _wait_for_server(
    port: int,
    timeout_s: float = 60.0,
    *,
    server_thread: threading.Thread | None = None,
) -> bool:
    """Block until our uvicorn server marks itself started, or fails early.

    Checking `uvicorn.Server.started` guarantees we wait for both FastAPI
    lifespan startup and socket accept on our own server instance rather than
    accidentally connecting to an unrelated process on the same port.
    """
    del port  # Kept in signature for callers/logging clarity.
    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        if _server_error or _server_failed.is_set():
            return False
        server = _uvicorn_server
        if server is not None and bool(getattr(server, "started", False)):
            return True
        if server_thread is not None and not server_thread.is_alive():
            return False
        time.sleep(0.05)
    server = _uvicorn_server
    return bool(server is not None and getattr(server, "started", False)) and not _server_error


def _stop_server_gracefully(
    server_thread: threading.Thread | None,
    *,
    timeout_s: float = 10.0,
) -> None:
    """Signal uvicorn to exit cleanly and wait for FastAPI lifespan + store flush."""
    _shutdown.set()
    server = _uvicorn_server
    if server is not None:
        server.should_exit = True
    if server_thread is not None and server_thread.is_alive():
        server_thread.join(timeout=timeout_s)
        if server_thread.is_alive():
            _log.warning(
                "Uvicorn server thread did not exit within %.1fs during shutdown",
                timeout_s,
            )


def _signal_app_ready() -> None:
    """Tell the launcher exe that the app is really up.

    It hides its console window on this signal, so it is only ever sent once the
    server answers and the window exists - anything earlier would hide the very
    output that explains a failed start.
    """
    path = os.getenv("JAV_SUB_READY_FILE", "").strip()
    if not path:
        return
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(os.getpid()), encoding="utf-8")
    except OSError as exc:
        _log.warning("Could not write app-ready signal %s: %s", path, exc)


def _watch_server_after_ready(window) -> None:
    """Surface a post-startup uvicorn crash inside the open desktop window."""
    if not _server_failed.wait():
        return
    if _shutdown.is_set() and not _server_error:
        return
    _ensure_src_on_path()
    from core.diagnostics import current_process_log_hint, format_exception_chain

    reason = format_exception_chain(_server_error[0]) if _server_error else "未知原因"
    _log.error("Local web server exited while desktop window was open: %s", reason)
    message = (
        f"本地后台服务已异常退出：{reason}\n"
        f"详细堆栈请查看程序目录下的 {current_process_log_hint()}。"
    )
    try:
        window.evaluate_js(
            "if (window.__javSubServerCrash) "
            f"window.__javSubServerCrash({json.dumps(message, ensure_ascii=False)}); "
            f"else alert({json.dumps(message, ensure_ascii=False)});"
        )
    except Exception:
        _log.debug("Could not push server-crash notice to pywebview window", exc_info=True)


def main(argv: list[str] | None = None) -> int:
    global PORT, EVENTS_PORT, _instance_slot, _cleanup_registered, _uvicorn_server, _active_window

    args = list(sys.argv[1:] if argv is None else argv)

    _server_error.clear()
    _server_failed.clear()
    _shutdown.clear()
    _update_restart_requested.clear()
    _uvicorn_server = None
    _active_window = None

    _configure_runtime_environment()

    if "--cuda-probe-child" in args:
        _write_or_print_json(_cuda_probe_payload(), output_env="JAV_SUB_CUDA_PROBE_OUTPUT")
        return 0

    if "--smoke-imports" in args:
        return _run_smoke_imports()

    from core.diagnostics import (
        configure_process_logging,
        current_process_log_hint,
        format_exception_chain,
    )
    from core.typed_config import env_int

    app_log_path = configure_process_logging("app.log")
    if app_log_path is not None:
        _log.info("Launcher started pid=%s root=%s log=%s", os.getpid(), _ROOT, app_log_path)

    _instance_slot = _claim_instance_slot(_ROOT)
    if not _cleanup_registered:
        atexit.register(_cleanup_temp)
        _cleanup_registered = True

    if _instance_slot is None:
        # Starting anyway is the failure, not the fallback: this window would
        # write chunk audio into a tmp/ it has no registered claim on, where the
        # window that is sweeping right now cannot see it and will delete it.
        _report_fatal_startup_error(
            "没能登记这个工作目录的使用权，界面未启动。",
            [
                "另一个窗口可能正在清理临时文件；请等它结束后重试。若反复出现，检查 tmp/launcher/ 是否可写。",
                f"详细日志请查看 {current_process_log_hint()}。",
            ],
        )
        return 1

    preferred_port = env_int("JAV_SUB_PORT", 2233, minimum=1, maximum=65535)
    preferred_events_port = env_int("JAV_SUB_EVENTS_PORT", 2234, minimum=1, maximum=65535)
    try:
        server_sock, PORT = _bind_free_socket(preferred_port)
    except OSError as exc:
        _report_fatal_startup_error(
            f"无法绑定本地服务端口（从 {preferred_port} 起）：{format_exception_chain(exc)}",
            [f"详细日志请查看 {current_process_log_hint()}。"],
        )
        return 1
    EVENTS_PORT = _resolve_events_port(PORT, preferred_events_port)

    if _has_console():
        print(f"jav-sub: http://127.0.0.1:{PORT}", flush=True)

    server_thread = threading.Thread(
        target=_run_server,
        args=(server_sock,),
        name="uvicorn-server",
        daemon=True,
    )
    server_thread.start()

    try:
        if not _wait_for_server(PORT, server_thread=server_thread):
            details: list[str] = []
            if _server_error:
                reason = format_exception_chain(_server_error[0])
                details.append(f"原因: {reason}")
            else:
                details.append(f"端口 {PORT} 在 60 秒内没有响应。")
            details.append(f"详细日志请查看 {current_process_log_hint()}。")
            details.append("依赖可能不完整，可以运行 jav-sub.exe --doctor 自检并修复。")
            _report_fatal_startup_error("服务没能启动，界面无法打开。", details)
            return 1

        try:
            import webview
            from webview.dom import DOMEventHandler

            def _bind(window):
                # pywebview calls this once the window exists, which is the moment
                # the launcher exe is waiting for to hide its console.
                _signal_app_ready()
                threading.Thread(
                    target=_watch_server_after_ready,
                    args=(window,),
                    name="server-health-watch",
                    daemon=True,
                ).start()

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
                "jav-sub",
                f"http://127.0.0.1:{PORT}",
                width=1280,
                height=820,
                resizable=True,
                # The layout is a fixed 344px sidebar + fluid job panel with no
                # responsive breakpoint; below this the job panel gets crushed
                # (titles collapse to 1-2 characters, action buttons run off-screen).
                min_size=(900, 600),
            )
            _active_window = win
            webview.start(
                _bind,
                win,
                icon=_webview_icon_arg(),
            )
        except ImportError:
            import webbrowser

            _log.info("pywebview not installed; opening default browser on port %s", PORT)
            webbrowser.open(f"http://127.0.0.1:{PORT}")
            _signal_app_ready()
            _shutdown.wait()
        except KeyboardInterrupt:
            _log.info("Launcher interrupted by user")
            return 130
        except Exception as exc:  # noqa: BLE001 - report unexpected desktop window failure
            _log.exception("Desktop window failed unexpectedly")
            _report_fatal_startup_error(
                f"界面窗口异常退出：{format_exception_chain(exc)}",
                [f"详细日志请查看 {current_process_log_hint()}。"],
            )
            return 1
    finally:
        _stop_server_gracefully(server_thread)

    if _update_restart_requested.is_set():
        from core.updater import UPDATE_EXIT_CODE

        return UPDATE_EXIT_CODE

    if _server_error:
        reason = format_exception_chain(_server_error[0])
        _report_fatal_startup_error(
            f"后台服务在运行期间异常退出：{reason}",
            [f"详细日志请查看 {current_process_log_hint()}。"],
        )
        return 1

    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001 - last-resort guard for windowless frozen builds
        _log.exception("Launcher crashed during early startup")
        _report_fatal_startup_error(
            f"启动器发生未捕获异常：{type(exc).__name__}: {exc}",
            [f"详细日志请查看 logs/app-{os.getpid()}.log。"],
        )
        raise SystemExit(1) from exc
