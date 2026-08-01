"""Managed llama.cpp server backend.

Runs GGUF models (SakuraLLM/GalTransl, quantized Qwen3, ...) through an
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

from llm.backends.base import BaseTranslationBackend
from utils.model_paths import PROJECT_ROOT
from utils.subprocess_tools import no_window_subprocess_kwargs

_SERVER_EXE_NAMES = ("llama-server.exe", "llama-server")


def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(float(_env(name, str(default)) or default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


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
        "未找到 llama-server。安装方式任选其一：`winget install llama.cpp`，"
        "或从 https://github.com/ggml-org/llama.cpp/releases 下载 "
        "llama-*-bin-win-cuda-*.zip 解压后，在翻译设置里填写 llama-server.exe 路径。"
    )


def resolve_gguf_model_path(*, download_enabled: bool = True) -> str:
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
            "未配置 GGUF 模型：请在翻译设置里选择一个 Sakura/GGUF 预设，"
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
    return hf_hub_download(repo_id=repo, filename=filename)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _release_asr_worker_vram() -> None:
    """Best-effort: free the resident ASR GPU worker before loading the GGUF.

    The worker respawns lazily on the next ASR stage; skipping this on an 8GB
    card turns the server spawn into an instant CUDA OOM.
    """
    try:
        from pipeline import gpu_worker

        gpu_worker.shutdown_global_worker()
    except Exception:
        pass


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


class LlamaCppServerBackend(BaseTranslationBackend):
    """OpenAI-compatible chat against a managed local llama-server process."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._job_handle = None
        self._port: int | None = None
        self._client = None
        self._model_path = ""
        self._log_path: Path | None = None

    # -- lifecycle -----------------------------------------------------------

    def _server_log_path(self) -> Path:
        log_dir = PROJECT_ROOT / "tmp" / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "llamacpp_server.log"

    def _build_command(self, exe: str, model_path: str, port: int) -> list[str]:
        ctx = _env_int("LLAMACPP_CTX_SIZE", 8192, 1024, 131072)
        ngl = _env_int("LLAMACPP_N_GPU_LAYERS", 999, 0, 999)
        parallel = _env_int("LLAMACPP_PARALLEL", 4, 1, 16)
        # -c is the total context; llama-server splits it across -np slots, so
        # the env var means "context per slot". Flash attention stays on the
        # server's own default ("auto" on current builds).
        return [
            exe,
            "-m", model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(ctx * parallel),
            "-np", str(parallel),
            "-ngl", str(ngl),
            "--no-webui",
        ]

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

    def _ensure_server(self, cancel_event=None) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None and self._port:
                return
            self._teardown_locked()

            exe = resolve_server_executable()
            model_path = resolve_gguf_model_path()
            port = _pick_free_port()
            _release_asr_worker_vram()

            self._log_path = self._server_log_path()
            command = self._build_command(exe, model_path, port)
            with open(self._log_path, "ab") as log_file:
                log_file.write(
                    ("\n=== llama-server start: " + " ".join(command) + "\n").encode("utf-8")
                )
                self._proc = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT),
                    **no_window_subprocess_kwargs(),
                )
            self._job_handle = _bind_kill_on_close_job(self._proc)
            self._port = port

            timeout_s = _env_int("LLAMACPP_STARTUP_TIMEOUT_S", 300, 10, 3600)
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                self._raise_if_cancelled(cancel_event)
                exit_code = self._proc.poll()
                if exit_code is not None:
                    tail = self._log_tail()
                    self._teardown_locked()
                    raise RuntimeError(
                        f"llama-server 启动即退出（exit={exit_code}），"
                        f"日志尾部：{tail}"
                    )
                if self._health_ok(port):
                    self._client = self._make_client(port)
                    return
                time.sleep(0.5)
            tail = self._log_tail()
            self._teardown_locked()
            raise RuntimeError(
                f"llama-server {timeout_s}s 内未就绪（模型加载超时），日志尾部：{tail}"
            )

    def _make_client(self, port: int):
        from openai import OpenAI

        return OpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="local-llamacpp",
            timeout=600.0,
            max_retries=0,
        )

    def _teardown_locked(self) -> None:
        proc, self._proc = self._proc, None
        self._client = None
        self._port = None
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if self._job_handle is not None and os.name == "nt":
            import ctypes

            try:
                ctypes.windll.kernel32.CloseHandle(self._job_handle)
            except Exception:
                pass
            self._job_handle = None

    def close(self) -> None:
        with self._lock:
            self._teardown_locked()

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

    def chat_completion(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        response_format: dict | None = None,
        stream: bool = True,
        reasoning_effort: str | None = None,
        api_format: str | None = None,
        expected_count: int = 0,
        cancel_event=None,
        on_progress: Callable[[dict], None] | None = None,
        on_usage: Callable[[dict], None] | None = None,
    ) -> str:
        del response_format, stream, reasoning_effort, api_format
        self._raise_if_cancelled(cancel_event)
        self._ensure_server(cancel_event)
        self._raise_if_cancelled(cancel_event)
        self._emit_progress(on_progress, {"phase": "translating", "expected": expected_count})
        response = self._client.chat.completions.create(
            model=self.cache_identity(),
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
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
        return str(choices[0].message.content or "")
