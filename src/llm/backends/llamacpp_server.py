"""Managed llama.cpp server backend.

Runs GGUF models (quantized Qwen3 and other JA->ZH fine-tunes) through an
official ``llama-server`` binary instead of in-process Python bindings:
llama-cpp-python publishes no CUDA wheels for Python 3.14 on Windows, while
llama.cpp ships prebuilt CUDA server binaries with every release. The server
speaks the OpenAI chat-completions protocol, so this backend is a thin process
manager plus an OpenAI client pointed at 127.0.0.1.

VRAM handoff: the ASR stage worker keeps the 1.7B model resident between jobs.
On an 8GB card it cannot coexist with a 6GB GGUF, so the server spawn first
shuts the global ASR worker down (it reloads lazily on the next ASR stage) and
the pipeline stops this server again once a job's translation stage finishes.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from core import gpu_admission, resources
from core.typed_config import env_int
from llm.async_transport import close_quietly, run_cancellable_request
from llm.backends.base import ManagedTranslationBackend
from llm.errors import BackendLeaseInvalidatedError, TranslationCancelledError, ResponseTruncatedError
from utils.model_paths import PROJECT_ROOT
from utils.subprocess_tools import (
    child_state,
    close_kill_on_close_job,
    no_window_subprocess_kwargs,
)

_SERVER_EXE_NAMES = ("llama-server.exe", "llama-server")
_LOCAL_KIND = "local_server"
# How often a task waiting out someone else's model load re-checks: short
# enough that 取消 feels immediate, long enough not to spin.
_STARTUP_WAIT_POLL_S = 0.2
# Same, for the thread resolving (and possibly downloading) the GGUF.
_MODEL_RESOLVE_POLL_S = 0.1
_NO_CLEANUP = {
    "stuck": False,
    "pid": None,
    "attempts": 0,
    "last_error": "",
    "since": 0.0,
    "generation": 0,
    "process_state": "gone",
    "handle_open": False,
    "blocks_gpu": False,
}


def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def server_limits() -> dict[str, int]:
    return {
        "ctx": env_int("LLAMACPP_CTX_SIZE", 1024, minimum=1024, maximum=131072),
        "gpu_layers": env_int("LLAMACPP_N_GPU_LAYERS", 999, minimum=0, maximum=999),
        "parallel": env_int("LLAMACPP_PARALLEL", 8, minimum=1, maximum=16),
        "startup_s": env_int("LLAMACPP_STARTUP_TIMEOUT_S", 300, minimum=10, maximum=3600),
    }


def resolve_server_executable() -> str:
    """Locate llama-server: explicit setting first, then PATH.

    Raises with an actionable message instead of guessing -- a wrong binary
    (CPU build, stale version) would fail later in a much less readable way.
    """
    explicit = _env("LLAMACPP_SERVER_PATH")
    if explicit:
        path = Path(explicit)
        if path.is_dir():
            for name in _SERVER_EXE_NAMES:
                candidate = path / name
                if candidate.is_file():
                    return str(candidate)
        elif path.is_file():
            return str(path)
        raise RuntimeError(
            f"LLAMACPP_SERVER_PATH 指向的 llama-server 不存在：{explicit}。"
            "请填写 llama-server.exe 的完整路径或其所在目录。"
        )
    for name in _SERVER_EXE_NAMES:
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError(
        "未找到 llama-server。安装方式任选其一："
        "`winget install -e --id ggml.llamacpp`（Vulkan 构建，装完即在 PATH 上），"
        "或从 https://github.com/ggml-org/llama.cpp/releases 下载 "
        "llama-*-bin-win-cuda-*.zip（N 卡上更快）解压后，在翻译设置里填写 llama-server.exe 路径。"
    )


def resolve_gguf_model_path(
    *, download_enabled: bool = True, cancel_event=None
) -> str:
    """Explicit local GGUF path wins; otherwise download the configured repo
    file into the project Hugging Face cache (honours HF_HOME and proxy env)."""
    explicit = _env("LLAMACPP_GGUF_PATH")
    if explicit:
        if Path(explicit).is_file():
            return explicit
        raise RuntimeError(f"LLAMACPP_GGUF_PATH 指向的 GGUF 文件不存在：{explicit}")
    repo = _env("LLAMACPP_MODEL_REPO")
    filename = _env("LLAMACPP_MODEL_FILE")
    if not repo or not filename:
        raise RuntimeError(
            "未配置 GGUF 模型：请在翻译设置里选择一个 GGUF 预设，"
            "或填写本地 GGUF 文件路径（LLAMACPP_GGUF_PATH）。"
        )
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    cached = try_to_load_from_cache(repo_id=repo, filename=filename)
    if isinstance(cached, str) and Path(cached).is_file():
        return cached
    if not download_enabled:
        raise RuntimeError(
            f"本地未找到 {repo}/{filename}，且未启用自动下载。"
        )
    from utils import hf_progress

    return hf_hub_download(
        repo_id=repo,
        filename=filename,
        # The 4.6GB one. Without the event the transfer has no way of hearing
        # that the task asking for it has been cancelled.
        tqdm_class=hf_progress.tqdm_class(cancel_event),
    )


def _torch_library_dir() -> Path | None:
    """``torch/lib`` without importing torch (which costs seconds and VRAM)."""
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    for location in getattr(spec, "submodule_search_locations", None) or []:
        candidate = Path(location) / "lib"
        if candidate.is_dir():
            return candidate
    return None


def cuda_library_dirs() -> tuple[str, ...]:
    """Directories to put on the server's PATH so ``ggml-cuda.dll`` can load.

    The CUDA llama.cpp zip links cuBLAS dynamically but ships no CUDA runtime;
    its own release publishes that as a separate ~400MB `cudart-*` download.
    Without `cublas64_*.dll` on the search path the CUDA backend fails to
    register *silently* and the server runs on the CPU: measured 2026-08-04 on
    Hy-MT2-1.8B-Q8_0, 90 tok/s prompt eval and ~4 tok/s generation, roughly 40x
    off, with nothing in the log saying why.

    torch already installs the matching CUDA libraries into the venv this app
    runs from, so point the server at those instead of asking for a second
    download of the same DLLs.
    """
    lib_dir = _torch_library_dir()
    if lib_dir is None or not any(lib_dir.glob("cublas64_*.dll")):
        return ()
    return (str(lib_dir),)


def server_environment() -> dict[str, str]:
    env = dict(os.environ)
    extra = cuda_library_dirs()
    if extra:
        env["PATH"] = os.pathsep.join([*extra, env.get("PATH", "")])
    return env


_ACCELERATOR_PREFIXES = ("cuda", "vulkan", "rocm", "hip", "sycl", "metal", "opencl")


def probe_compute_devices(exe: str, env: dict[str, str] | None = None) -> list[str]:
    """Accelerators the server can actually see, via its own ``--list-devices``.

    Cheap (no model load) and the only honest answer: whether ``-ngl`` does
    anything depends on which ggml backend DLLs loaded, not on what the binary
    was named or built for.
    """
    try:
        result = subprocess.run(
            [exe, "--list-devices"],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            **no_window_subprocess_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return []
    devices: list[str] = []
    for line in (result.stdout or "").splitlines():
        entry = line.strip()
        if entry.split(":", 1)[0].lower().startswith(_ACCELERATOR_PREFIXES):
            devices.append(entry)
    return devices


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wrap_response_format(schema: dict | None) -> dict | None:
    """Put a bare JSON schema into the shape llama-server expects.

    The engine hands profiles' `schema` through unwrapped; the OpenAI transport
    wraps it the same way. An already-wrapped value (it has a "type") is passed
    through so a caller can ask for plain `json_object`.
    """
    if not schema:
        return None
    if "type" in schema and "properties" not in schema:
        return schema
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "subtitle_translations",
            "strict": True,
            "schema": schema,
        },
    }


def _release_asr_worker_vram() -> None:
    """Free the resident ASR GPU worker before loading the GGUF.

    The worker respawns lazily on the next ASR stage; skipping this on an 8GB
    card turns the server spawn into an instant CUDA OOM. Which is exactly why
    this is not best-effort: if the ASR child could not be stopped, its VRAM is
    still held, and starting a multi-GB server next to it is the failure this
    is meant to prevent. Refuse instead, with the same message and the same
    manual recovery entry points as any other blocked cleanup.
    """
    from pipeline import gpu_worker

    try:
        stopped = gpu_worker.shutdown_global_worker()
    except Exception as exc:
        raise RuntimeError(
            f"无法确认 ASR GPU 子进程已退出，拒绝加载本地翻译模型：{exc!r}"
        ) from exc
    if not stopped or gpu_worker.asr_gpu_blocked():
        status = gpu_worker.asr_gpu_cleanup_status()
        raise RuntimeError(
            "ASR GPU 子进程仍未退出，显存未释放，拒绝启动本地翻译服务器"
            f"（pid={status.get('pid')}，已重试 {status.get('attempts')} 次，"
            f"原因：{status.get('last_error') or '未知'}）。"
            "可在任务页点击「重试清理」/「重新检查」，或查看诊断信息。"
        )
    # The ASR worker is not the only thing that can be holding the card - an
    # abandoned server from an earlier backend instance is another - so the
    # question is asked of every owner, not just of this subsystem's own.
    if resources.gpu_blocked():
        blockers = [
            record for record in resources.pending() if record["blocked"]
        ]
        detail = "；".join(
            f"{record['kind']} pid={record['pid']} 进程={record['process_state']}"
            for record in blockers
        )
        raise RuntimeError(
            "仍有未确认退出的 GPU 子进程，显存未释放，拒绝启动本地翻译服务器"
            f"（{detail or '未知'}）。"
            "可在任务页点击「重试清理」/「重新检查」，或查看诊断信息。"
        )


if os.name == "nt":
    import ctypes

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JobObjectExtendedLimitInformation = 9

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    def _bind_kill_on_close_job(proc: subprocess.Popen):
        """Tie the server to this process: if the web app dies hard, the OS
        reaps the server, so no orphan keeps holding 6GB of VRAM."""
        try:
            kernel32 = ctypes.windll.kernel32
            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return None
            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(
                job,
                _JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            ):
                kernel32.CloseHandle(job)
                return None
            if not kernel32.AssignProcessToJobObject(job, int(proc._handle)):
                kernel32.CloseHandle(job)
                return None
            return job
        except Exception:
            return None

else:  # pragma: no cover - project targets Windows

    def _bind_kill_on_close_job(proc: subprocess.Popen):
        return None


class LlamaCppServerBackend(ManagedTranslationBackend):
    """OpenAI-compatible chat against a managed local llama-server process."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # "A start is in progress" is state, not a held lock: a task that
        # arrives during one waits here, where its cancel event can end the
        # wait, instead of blocking on `_lock` where nothing can.
        self._startup_done = threading.Condition(self._lock)
        self._starting = False
        self._proc: subprocess.Popen | None = None
        self._job_handle = None
        self._port: int | None = None
        self._model_path = ""
        self._log_path: Path | None = None
        # Set when a request was abandoned while the server was still working on
        # it. HTTP abort != generation stopped, and the difference is VRAM.
        # Guarded by its own short lock, never by `_lock`: status is read from
        # the web event loop, and `_lock` is held across model loads.
        self._status_lock = threading.Lock()
        self._draining_since: float | None = None
        # A teardown that did not confirm. Same contract as the ASR worker:
        # the process is still ours, a replacement must not start, and the state
        # is visible and retryable rather than being swallowed by a bare
        # `except TimeoutExpired: pass`.
        # Identity of the servers this instance owns. The id comes from the
        # registry so that two instances created one after the other cannot
        # claim the same one: with a fixed id and a generation restarting at 0,
        # instance B's first server had exactly the identity instance A's first
        # server had, and A's queued cleanup terminated B's process.
        self._resource_token = resources.new_resource_id("llamacpp-server")
        # Bumped for every server this instance starts. A cleanup request names
        # the generation it was raised for, so a late one cannot act on the
        # server that replaced it.
        self._server_generation = 0
        # Set when a request was abandoned mid-generation: this generation stops
        # taking new tasks and is closed once the ones holding it finish.
        self._retiring = False
        # One-way: set when the registry takes this instance out of service.
        # Being removed from the registry only stops the *next* lookup - a caller
        # already holding this object would still reach `_ensure_server()` and
        # start a server to replace the one being closed.
        self._invalidated = False

    # -- lifecycle -----------------------------------------------------------

    def _server_log_path(self) -> Path:
        log_dir = PROJECT_ROOT / "tmp" / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "llamacpp_server.log"

    def _build_command(self, exe: str, model_path: str, port: int) -> list[str]:
        limits = server_limits()
        ctx, ngl, parallel = limits["ctx"], limits["gpu_layers"], limits["parallel"]
        # -c is the total context; llama-server splits it across -np slots, so
        # the env var means "context per slot". Flash attention stays on the
        # server's own default ("auto" on current builds).
        command = [
            exe,
            "-m", model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(ctx * parallel),
            "-np", str(parallel),
            "-ngl", str(ngl),
            "--no-webui",
        ]
        return command

    def _health_ok(self, port: int) -> bool:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2.0
            ) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _log_tail(self, max_chars: int = 800) -> str:
        if self._log_path is None:
            return ""
        try:
            text = self._log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[-max_chars:]

    def _acquire_gpu_permit(self, cancel_event) -> gpu_admission.GpuPermit:
        """Wait for the card, and let 取消 end the wait.

        The wait itself is the point: another job's ASR stage can run for
        minutes, and the answer to "when can I load the model" is "when it
        finishes, or when you cancel". What it must never be is "stop their
        worker", which is what asking for the device by tearing the ASR worker
        down amounted to.
        """
        try:
            return gpu_admission.acquire(
                gpu_admission.LOCAL_LLM,
                cancel=cancel_event,
                description="llama-server",
            )
        except gpu_admission.GpuAdmissionCancelled as exc:
            self._raise_if_cancelled(cancel_event)
            raise TranslationCancelledError("任务已取消") from exc

    def _server_ready_locked(self) -> bool:
        """A server that is running *and* answering. Callers hold `_lock`.

        `_port` is set only once the health check passes, so this cannot report
        a model that is still loading as usable - which matters now that a
        second task consults it instead of queueing behind the load.
        """
        return self._proc is not None and self._proc.poll() is None and bool(self._port)

    def _claim_startup(self, cancel_event) -> bool:
        """True if this thread is the one that must start the server.

        False means someone else already has it running. The waiting is done
        here, on a condition, rather than on `_lock`: the lock is held across
        the parts of a start that must not interleave, and a thread blocked on
        a lock cannot be cancelled - the same shape of bug as waiting for the
        ASR module lock, one layer up.
        """
        while True:
            self._raise_if_cancelled(cancel_event)
            with self._lock:
                # Re-checked every pass: a reset can land while we wait, and an
                # instance out of service starts nothing and hands out nothing.
                self._raise_if_invalidated()
                if self._server_ready_locked():
                    return False
                if not self._starting:
                    self._starting = True
                    return True
                # Releases `_lock` while it waits, and the starter notifies on
                # its way out - success or failure, so a failed start hands the
                # attempt to the next waiter rather than stranding it.
                self._startup_done.wait(_STARTUP_WAIT_POLL_S)

    def _ensure_server(self, cancel_event=None) -> None:
        # Before the fast path: an instance taken out of service must not start a
        # server, and must not hand out the one it still has.
        self._raise_if_invalidated()
        if not self._claim_startup(cancel_event):
            return
        try:
            self._start_server(cancel_event)
        finally:
            with self._lock:
                self._starting = False
                self._startup_done.notify_all()

    def _require_clean_slate_locked(self) -> None:
        """Refuse to start while a previous server is still unaccounted for.

        Doubles as the re-check: a process that has since exited on its own
        confirms here and clears the state. Only a server that is still
        unaccounted for blocks the start - loading a second multi-GB model next
        to one that still holds its VRAM is the failure this exists to prevent.
        """
        if self._teardown_locked():
            return
        status = self.cleanup_status()
        raise RuntimeError(
            "上一个 llama-server 未确认退出，拒绝启动新的本地翻译服务器"
            f"（pid={status.get('pid')}，已重试 {status.get('attempts')} 次，"
            f"原因：{status.get('last_error') or '未知'}）。"
            "可在任务页点击「重试清理」/「重新检查」，或查看诊断信息。"
        )

    def _resolve_model_path(self, cancel_event) -> str:
        """Locate the GGUF, cancellably - which on a first run means a download.

        Two mechanisms, because neither is sufficient alone. The progress bar
        carries the cancel into the transfer itself, but huggingface_hub owns
        the thread the bytes arrive on and what it does with an exception from a
        progress bar is its business, not a contract. So the *wait* is made
        cancellable here regardless: the resolution runs on its own thread and
        this one stops waiting when asked. A download that keeps going anyway
        lands in the cache and the next attempt starts from it.

        A new thread inherits no run: `core.events`'s job_id is thread-local, and
        hf_progress's module-level fallback is shared, so without this the
        download's progress would arrive unattributed - or, if another job's ASR
        model was downloading at the same time, under *that* job's id, which
        passes every check the receiving end makes.
        """
        from utils import hf_progress

        identity = hf_progress.current_identity()
        resolved: list[str] = []
        failure: list[BaseException] = []
        finished = threading.Event()

        def resolve() -> None:
            hf_progress.adopt_identity(identity)
            try:
                resolved.append(resolve_gguf_model_path(cancel_event=cancel_event))
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                failure.append(exc)
            finally:
                finished.set()

        threading.Thread(
            target=resolve, name="llamacpp-model-resolve", daemon=True
        ).start()
        while not finished.wait(_MODEL_RESOLVE_POLL_S):
            self._raise_if_cancelled(cancel_event)
        self._raise_if_cancelled(cancel_event)
        if failure:
            raise failure[0]
        return resolved[0]

    def _start_server(self, cancel_event) -> None:
        # Before any waiting: a server we could not confirm dead is still
        # holding the card, so queueing for a permit behind it would be waiting
        # for a thing that is never coming. It is our own process either way -
        # this stops it, which is not the same as asking for the device by
        # stopping the *ASR* worker below.
        with self._lock:
            self._raise_if_invalidated()
            self._require_clean_slate_locked()

        # Preparation, holding neither the instance lock nor the card. Resolving
        # the GGUF is a 4.6GB download on a first run: under `_lock` it blocked
        # every sibling task on a *lock*, where a cancel event cannot reach them
        # (they never got as far as the condition they were supposed to wait
        # on); holding the permit across it would keep an ASR stage off a card
        # that nothing has loaded yet.
        self._raise_if_cancelled(cancel_event)
        exe = resolve_server_executable()
        model_path = self._resolve_model_path(cancel_event)
        env = server_environment()

        # Exclusive use of the card, taken *before* the instance lock and before
        # the ASR worker is stopped. Stopping it used to be how this path asked
        # for the device - `_release_asr_worker_vram()` blocked inside
        # `shutdown_global_worker()` on the ASR module lock, held for the whole
        # of another job's transcription, where no cancel event could reach it.
        permit = self._acquire_gpu_permit(cancel_event)
        handed_over = False
        proc: subprocess.Popen | None = None
        generation = 0
        port = 0
        try:
            # Asks the driver what it can see, so it belongs after the permit -
            # and outside the lock, where its 120s timeout cannot strand anyone.
            devices = probe_compute_devices(exe, env)
            with self._lock:
                self._raise_if_invalidated()
                self._raise_if_cancelled(cancel_event)
                # Again, holding the permit this time: the waits above are long
                # enough for a server to have appeared and died in them.
                self._require_clean_slate_locked()

                port = _pick_free_port()
                _release_asr_worker_vram()

                self._log_path = self._server_log_path()
                command = self._build_command(exe, model_path, port)
                if not devices and server_limits()["gpu_layers"] > 0:
                    # Not fatal - a CPU-only build is a legitimate choice - but
                    # it is a 40x slowdown that otherwise shows up only as
                    # "翻译很慢".
                    print(
                        "[WARN] llama-server 看不到任何 GPU 设备，将以 CPU 运行（会慢几十倍）。"
                        "CUDA 版还需要 cuBLAS 运行库：确认 llama-server.exe 是 CUDA 构建，"
                        "或解压对应 release 的 cudart-llama-bin-win-cuda-*.zip 到同一目录。",
                        flush=True,
                    )
                with open(self._log_path, "ab") as log_file:
                    log_file.write(
                        (
                            "\n=== llama-server start: "
                            + " ".join(command)
                            + "\n=== devices: "
                            + ("; ".join(devices) or "(none - CPU only)")
                            + "\n"
                        ).encode("utf-8")
                    )
                    self._proc = subprocess.Popen(
                        command,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        cwd=str(PROJECT_ROOT),
                        env=env,
                        **no_window_subprocess_kwargs(),
                    )
                self._server_generation += 1
                self._retiring = False
                self._job_handle = _bind_kill_on_close_job(self._proc)
                resources.register(
                    resource_id=self._resource_id(),
                    generation=self._server_generation,
                    kind=_LOCAL_KIND,
                    description="llama-server",
                    process=self._proc,
                    job_handle=self._job_handle,
                    stopper=self._stop_server,
                    holds_gpu=True,
                )
                # Handed over, never released: exclusive use becomes a resident
                # claim on the identity just registered, in one lock section.
                # Registering and then releasing are two steps with a gap
                # between them, and an ASR admission check taken before the
                # first could be acted on after the second - which is a
                # transcription starting beside a resident model.
                gpu_admission.handover(
                    permit,
                    resource_id=self._resource_id(),
                    generation=self._server_generation,
                )
                handed_over = True
                # Invalidation may have landed after the startup entrance check
                # but before registration. Whichever side wins must publish stop
                # intent, so a timed-out close never makes this live owner
                # invisible to GPU admission or to the cleanup watchdog.
                if self.is_invalidated():
                    resources.schedule_stop(self._resource_id(), self._server_generation)
                proc = self._proc
                generation = self._server_generation
        finally:
            # Only the paths that never got as far as registering: nothing
            # is resident, so the card goes back. After the handover there
            # is nothing to release - the claim ends with its record, whose
            # lifetime the registry (and its watchdog) already owns.
            if not handed_over:
                gpu_admission.release(permit)

        self._await_ready(proc, generation, port, cancel_event)

    def _await_ready(self, proc, generation: int, port: int, cancel_event) -> None:
        """Wait for the model to load, without holding the instance lock.

        A load is minutes on a large model. Holding `_lock` across it made 关闭
        and every sibling task wait behind it; the process this loop is watching
        is named by argument, so nothing here needs the lock except the short
        writes that publish the result.
        """
        timeout_s = server_limits()["startup_s"]
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._raise_if_cancelled(cancel_event)
            with self._lock:
                current = self._proc
                current_generation = self._server_generation
            if current is not proc or current_generation != generation:
                # Closed or replaced while it was loading. Reporting that as a
                # startup failure of *this* server would name the wrong process.
                raise RuntimeError(
                    "本地翻译服务器在启动过程中被关闭或替换；请重新运行该任务。"
                )
            exit_code = proc.poll()
            if exit_code is not None:
                tail = self._log_tail()
                self._teardown_if_current(proc)
                raise RuntimeError(
                    f"llama-server 启动即退出（exit={exit_code}），"
                    f"日志尾部：{tail}"
                )
            if self._health_ok(port):
                with self._lock:
                    if self._proc is not proc:
                        raise RuntimeError(
                            "本地翻译服务器在启动过程中被关闭或替换；请重新运行该任务。"
                        )
                    self._port = port
                return
            time.sleep(0.5)
        tail = self._log_tail()
        self._teardown_if_current(proc)
        raise RuntimeError(
            f"llama-server {timeout_s}s 内未就绪（模型加载超时），日志尾部：{tail}"
        )

    def _teardown_if_current(self, proc) -> None:
        """Stop the server this start produced, and only that one."""
        with self._lock:
            if self._proc is proc:
                self._teardown_locked()

    def _make_async_client(self, port: int):
        # Built per request, on that request's loop - see the note in
        # openai_compat._make_async_client. Here it matters for a second reason:
        # the process is shared between concurrent jobs, so cancelling one must
        # not close a client the others are using.
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="local-llamacpp",
            timeout=600.0,
            max_retries=0,
        )

    def _resource_id(self) -> str:
        return self._resource_token

    def _stop_server(self, proc, job_handle) -> resources.StopOutcome:
        """The blocking part of stopping one server. Called by the registry.

        Everything here is about the process object it was handed. The old
        version cleared `_proc` first and swallowed the wait timeout, so a
        server that ignored kill() looked exactly like one that exited: the
        instance was dropped from the registry, `draining` was cleared, and the
        next translation loaded a second multi-GB model beside the first.
        """
        detail = ""
        if proc is not None and child_state(proc) != "gone":
            try:
                proc.kill()
            except Exception as exc:
                detail = f"kill() failed: {exc!r}"
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                detail = detail or "process still alive 10s after kill()"
            except Exception as exc:
                detail = detail or f"wait() failed: {exc!r}"
        # Closing the job object is the last escalation available: it is
        # kill-on-close, so the whole tree goes with it - including any child
        # llama-server spawned that kill() never reached.
        if job_handle and os.name == "nt":
            if close_kill_on_close_job(job_handle):
                job_handle = None
            else:
                # A failed close released nothing; dropping the integer would
                # leak it, so it stays on this record.
                detail = detail or "job object handle could not be closed"
        elif job_handle:  # pragma: no cover - project targets Windows
            job_handle = None
        if proc is not None and child_state(proc) != "gone":
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        state = child_state(proc) if proc is not None else "gone"
        if state != "gone":
            detail = detail or f"llama-server is {state} after kill()"
            print(f"[WARN] llama-server 未确认退出：{detail}", flush=True)
        with self._status_lock:
            self._job_handle = job_handle
            if state == "gone" and self._proc is proc:
                self._proc = None
        return resources.StopOutcome(
            process_state=state, job_handle=job_handle, detail=detail
        )

    def _teardown_locked(self) -> bool:
        """Stop the current server. True *only* when it is confirmed gone.

        The process reference is what makes a retry possible, so it is kept
        until the process is confirmed gone.
        """
        proc = self._proc
        generation = self._server_generation
        # The port is dropped either way: whatever this process is doing,
        # nothing may be sent to it again.
        self._port = None
        if proc is None:
            self._retiring = False
            return True
        if not resources.is_registered(self._resource_id(), generation):
            # A process adopted outside `_ensure_server` (tests, and any future
            # path) still has to be stoppable through the one entry point.
            resources.register(
                resource_id=self._resource_id(),
                generation=generation,
                kind=_LOCAL_KIND,
                description="llama-server",
                process=proc,
                job_handle=self._job_handle,
                stopper=self._stop_server,
                holds_gpu=True,
            )
        result = resources.request_stop(self._resource_id(), generation)
        if result.stopped:
            self._retiring = False
        return result.stopped

    def cleanup_status(self) -> dict:
        """Is a server we tried to stop still unaccounted for?

        Read from the web event loop, so `_lock` is off limits here - it is held
        across model loads. The registry read takes only its own short lock.
        """
        records = [
            record
            for record in resources.pending(_LOCAL_KIND)
            if record["resource_id"] == self._resource_id()
        ]
        if not records:
            return dict(_NO_CLEANUP)
        record = records[0]
        return {
            "stuck": True,
            "pid": record["pid"],
            "attempts": record["attempts"],
            "last_error": record["last_error"],
            "since": record["since"],
            "generation": record["generation"],
            "process_state": record["process_state"],
            "handle_open": record["handle_open"],
            # Two different claims. A process that may still be resident holds
            # the card; a handle nobody could close does not, and blocking every
            # GPU stage on it would be a made-up cost.
            "blocks_gpu": record["process_state"] != "gone",
        }

    def close(self) -> bool:
        """Stop the server for good. Returns whether it is confirmed stopped.

        This is the "the last user left / the user changed the setting" path -
        it always targets whatever this instance currently owns. 重试清理 is a
        different operation and goes through the registry.
        """
        with self._lock:
            stopped = self._teardown_locked()
        if stopped:
            # Only now is the unknown resolved: the process that might still
            # have been generating is gone, so nothing of that request outlives
            # it. A teardown that did not confirm leaves the flag standing.
            with self._status_lock:
                self._draining_since = None
        return stopped

    def invalidate(self) -> None:
        """Take this instance out of service permanently.

        Called by the registry when the instance stops being the one in use -
        settings reset, or its last holder let go. There is no way back: a task
        that was already holding the object must fail rather than start new work
        on it, and "new work" includes starting a server.
        """
        with self._status_lock:
            self._invalidated = True
        resources.schedule_stop(self._resource_id(), self._server_generation)

    def is_invalidated(self) -> bool:
        with self._status_lock:
            return self._invalidated

    def _raise_if_invalidated(self) -> None:
        if self.is_invalidated():
            raise BackendLeaseInvalidatedError(
                "本地翻译服务实例已被停用（翻译设置已变更，或该实例已被关闭），"
                "不会为它启动新的服务器；请重新运行该任务。"
            )

    def generation_retiring(self) -> bool:
        """Is the running server's generation on its way out?

        Set when a request was abandoned mid-generation. The server keeps
        serving the tasks that already hold it, and takes no new ones: that is
        what stops an endless stream of arriving tasks from extending an unknown
        forever, without killing a task that is still translating.
        """
        with self._status_lock:
            return bool(self._retiring and self._proc is not None)

    # -- request -------------------------------------------------------------

    def _request_completion(self, request: dict, cancel_event):
        """One local completion, abortable while the model is still thinking.

        A local request is silent from the first byte to the last: no streaming,
        one blocking read, up to the 600s client timeout. Without this, 取消 on a
        local job was unreachable for as long as the model kept generating.

        Aborting the HTTP request is *not* proof that llama-server stopped
        generating - it only proves this process stopped waiting. The server is
        shared with whatever other job holds a lease on it, so killing it to
        cancel one job is not on the table; instead the backend is marked as
        draining, and the page says so rather than implying the card is free.
        """
        port = self._port

        async def operation():
            client = self._make_async_client(port)
            try:
                result = client.chat.completions.create(**request)
                # Tests replace the client with a plain synchronous double; the
                # transport itself is async.
                if inspect.isawaitable(result):
                    result = await result
                return result
            finally:
                await close_quietly(client)

        try:
            response = run_cancellable_request(operation, cancel_event=cancel_event)
        except BaseException:
            with self._status_lock:
                if self._proc is not None:
                    self._draining_since = time.time()
                    # This generation is no longer trustworthy: it may still be
                    # generating for a request nobody is reading. Existing users
                    # keep it, new tasks get a fresh one once they let go.
                    self._retiring = True
            raise
        # Deliberately *not* cleared here. The server runs several slots in
        # parallel, so another request finishing says nothing about the one that
        # was abandoned - there is no id to correlate them by. The unknown only
        # ends when the process itself is confirmed gone.
        return response

    def drain_status(self) -> dict:
        """A snapshot, taken without ever waiting on the lifecycle lock.

        `self._lock` is held across model load and health checks - up to the
        300s startup timeout. The page polls this every 5 seconds on the event
        loop, so taking that lock here would stall the whole web app, cancel
        endpoint included, for as long as a load takes.
        """
        with self._status_lock:
            since = self._draining_since
            proc = self._proc
        try:
            running = proc is not None and proc.poll() is None
        except Exception:
            running = proc is not None
        if since is not None and not running:
            # The process is gone, so nothing of that request survives it.
            with self._status_lock:
                if self._draining_since == since:
                    self._draining_since = None
            since = None
        return {
            "draining": bool(since is not None and running),
            "since": float(since or 0.0),
        }

    # -- TranslationBackend --------------------------------------------------

    def name(self) -> str:
        return "llamacpp"

    def cache_identity(self) -> str:
        explicit = _env("LLAMACPP_GGUF_PATH")
        if explicit:
            return f"llamacpp:{Path(explicit).name}"
        repo = _env("LLAMACPP_MODEL_REPO")
        filename = _env("LLAMACPP_MODEL_FILE")
        return f"llamacpp:{repo}/{filename}"

    def supports_json_schema(self) -> bool:
        # llama-server compiles a JSON schema into a GBNF grammar and constrains
        # decoding to it. Measured on Hy-MT2-1.8B-Q8_0 (2026-08-04): free-form,
        # it echoed the prompt back and produced no parseable JSON in 3/3
        # attempts; grammar-constrained, 3/3 answered every id, and each request
        # took ~17s instead of ~45-70s because it stops rambling.
        return True

    def chat_completion(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        response_format: dict | None = None,
        reasoning_effort: str | None = None,
        expected_count: int = 0,
        cancel_event=None,
        on_progress: Callable[[dict], None] | None = None,
        on_usage: Callable[[dict], None] | None = None,
        sampling_parameters: dict | None = None,
    ) -> str:
        del reasoning_effort
        self._raise_if_cancelled(cancel_event)
        # Checked at the moment of use, not only by whoever resolved this object:
        # a reset can land between a caller's lease check and this call.
        self._raise_if_invalidated()
        self._ensure_server(cancel_event)
        self._raise_if_cancelled(cancel_event)
        self._emit_progress(on_progress, {"phase": "translating", "expected": expected_count})
        request: dict = {
            "model": self.cache_identity(),
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        # These are llama.cpp extensions, not OpenAI SDK keyword arguments.
        # The SDK merges extra_body into the JSON sent to the server.
        extra_body = {
            key: sampling_parameters[key]
            for key in ("top_k", "repeat_penalty")
            if sampling_parameters is not None and key in sampling_parameters
        }
        if extra_body:
            request["extra_body"] = extra_body
        wrapped = _wrap_response_format(response_format)
        if wrapped is not None:
            request["response_format"] = wrapped
        response = self._request_completion(request, cancel_event)
        self._raise_if_cancelled(cancel_event)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._emit_usage(
                on_usage,
                {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
            )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("llama-server 返回了空的 choices")
        if getattr(choices[0], "finish_reason", None) == "length":
            raise ResponseTruncatedError("llama-server exhausted the output token budget", limit=max_tokens)
        return str(choices[0].message.content or "")
