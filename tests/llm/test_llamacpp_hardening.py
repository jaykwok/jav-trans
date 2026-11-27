"""Regression tests for P0-4, P0-6, P1-1, P1-4, P1-5, P1-8 and P2 local backend hardening."""

from __future__ import annotations

import ast
import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from core import resources
from llm import engine as engine_module
from llm import request_config
from llm import settings as llm_settings
from llm import translator
from llm.backends import llamacpp_env, llamacpp_runtime, llamacpp_server, llamacpp_vram
from llm.errors import RetryableTranslationFormatError, TranslationCancelledError
from llm.profiles import get_profile


class _AliveProc:
    pid = 43210

    def __init__(self) -> None:
        self.killed = False

    def poll(self):
        return 0 if self.killed else None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if self.killed:
            return 0
        raise subprocess.TimeoutExpired(cmd="llama-server", timeout=timeout or 0)


# ---------------------------------------------------------------------------
# P0-4: Streaming progress & no fake pre-request translating event
# ---------------------------------------------------------------------------


def test_llamacpp_chat_completion_streams_progress_without_fake_pre_event(monkeypatch):
    backend = llamacpp_server.LlamaCppServerBackend()
    proc = _AliveProc()
    backend._proc = proc
    backend._port = 18080
    monkeypatch.setattr(backend, "_ensure_server", lambda cancel_event=None: None)

    async def _chunk_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content='{"translations":[{"id":0,"text":"你好"},'),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content='{"id":1,"text":"世界"}]}'),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        yield SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=150),
        )

    captured_request: dict = {}

    class _FakeCompletions:
        async def create(self, **kwargs):
            captured_request.update(kwargs)
            return _chunk_stream()

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions()),
        close=lambda: None,
    )
    monkeypatch.setattr(backend, "_make_async_client", lambda port: fake_client)

    events: list[dict] = []
    usages: list[dict] = []
    text = backend.chat_completion(
        [{"role": "user", "content": "hi"}],
        expected_count=2,
        on_progress=events.append,
        on_usage=usages.append,
    )

    assert json.loads(text)["translations"][1]["text"] == "世界"
    assert captured_request["stream"] is True
    assert captured_request["stream_options"] == {"include_usage": True}
    # No fake pre-request translating event without "translated"
    assert all(
        "translated" in ev
        for ev in events
        if ev.get("phase") == "translating"
    )
    translating_events = [ev for ev in events if ev.get("phase") == "translating"]
    assert translating_events
    assert translating_events[0]["translated"] >= 1
    assert events[-1] == {"phase": "done", "translated": 2, "expected": 2}
    assert usages == [
        {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
        }
    ]


def test_llamacpp_chat_completion_non_streaming_stub_emits_done(monkeypatch):
    backend = llamacpp_server.LlamaCppServerBackend()
    backend._proc = _AliveProc()
    backend._port = 18081
    monkeypatch.setattr(backend, "_ensure_server", lambda cancel_event=None: None)

    stub_reply = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"translations":[{"id":0,"text":"测试"}]}'),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kw: stub_reply)),
        close=lambda: None,
    )
    monkeypatch.setattr(backend, "_make_async_client", lambda port: fake_client)

    events: list[dict] = []
    result = backend.chat_completion(
        [{"role": "user", "content": "hi"}],
        expected_count=1,
        on_progress=events.append,
    )

    assert "测试" in result
    assert events == [{"phase": "done", "translated": 1, "expected": 1}]


# ---------------------------------------------------------------------------
# P0-6: Hard CUDA check (no silent CPU fallback) & managed CUDA precedence
# ---------------------------------------------------------------------------


def test_start_server_refuses_silent_cpu_fallback_when_ngl_positive(monkeypatch):
    backend = llamacpp_server.LlamaCppServerBackend()
    monkeypatch.setenv("LLAMACPP_N_GPU_LAYERS", "999")
    monkeypatch.setattr(backend, "_resolve_server_executable", lambda _c: "llama-server.exe")
    monkeypatch.setattr(backend, "_resolve_model_path", lambda _c: "model.gguf")
    monkeypatch.setattr(backend, "_acquire_gpu_permit", lambda _c: object())
    released: list[object] = []
    monkeypatch.setattr(
        llamacpp_server.gpu_admission, "release", lambda permit: released.append(permit)
    )
    monkeypatch.setattr(llamacpp_server, "probe_compute_devices", lambda _exe, _env: [])
    popen_called = False

    def _forbid_popen(*_a, **_kw):
        nonlocal popen_called
        popen_called = True
        raise AssertionError("Popen must not run without CUDA devices when ngl > 0")

    monkeypatch.setattr(llamacpp_server.subprocess, "Popen", _forbid_popen)

    with pytest.raises(RuntimeError, match="LLAMACPP_N_GPU_LAYERS=0"):
        backend._start_server(cancel_event=None)

    assert not popen_called
    assert len(released) == 1


def test_start_server_allows_cpu_only_when_ngl_explicitly_zero(monkeypatch, tmp_path):
    backend = llamacpp_server.LlamaCppServerBackend()
    monkeypatch.setenv("LLAMACPP_N_GPU_LAYERS", "0")
    monkeypatch.setattr(backend, "_resolve_server_executable", lambda _c: "llama-server.exe")
    monkeypatch.setattr(backend, "_resolve_model_path", lambda _c: str(tmp_path / "m.gguf"))
    monkeypatch.setattr(backend, "_acquire_gpu_permit", lambda _c: object())
    monkeypatch.setattr(llamacpp_server.gpu_admission, "handover", lambda *_a, **_kw: None)
    monkeypatch.setattr(llamacpp_server.gpu_admission, "release", lambda *_a, **_kw: None)
    monkeypatch.setattr(llamacpp_server, "_release_asr_worker_vram", lambda: None)
    monkeypatch.setattr(llamacpp_server, "probe_compute_devices", lambda _exe, _env: [])
    monkeypatch.setattr(llamacpp_server, "_pick_free_port", lambda: 18999)
    monkeypatch.setattr(backend, "_server_log_path", lambda: tmp_path / "server.log")
    monkeypatch.setattr(llamacpp_server, "_bind_kill_on_close_job", lambda _proc: None)
    monkeypatch.setattr(resources, "register", lambda **_kw: None)

    started_cmd: list[str] = []
    fake_proc = _AliveProc()

    def _fake_popen(cmd, **_kw):
        started_cmd.extend(cmd)
        return fake_proc

    monkeypatch.setattr(llamacpp_server.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(backend, "_await_ready", lambda *_a, **_kw: None)

    backend._start_server(cancel_event=None)

    assert "-ngl" in started_cmd
    assert started_cmd[started_cmd.index("-ngl") + 1] == "0"


def test_resolve_server_executable_prefers_managed_over_system_path(monkeypatch, tmp_path):
    monkeypatch.delenv("LLAMACPP_SERVER_PATH", raising=False)
    managed_exe = tmp_path / "managed" / "llama-server.exe"
    managed_exe.parent.mkdir(parents=True)
    managed_exe.write_bytes(b"managed")
    system_exe = tmp_path / "system" / "llama-server.exe"
    system_exe.parent.mkdir(parents=True)
    system_exe.write_bytes(b"system")

    monkeypatch.setattr(llamacpp_runtime, "managed_server_executable", lambda: managed_exe)
    monkeypatch.setattr(llamacpp_server.shutil, "which", lambda _name: str(system_exe))

    assert llamacpp_server.resolve_server_executable(download_enabled=False) == str(managed_exe)


def test_resolve_server_executable_auto_downloads_before_falling_back_to_path(monkeypatch, tmp_path):
    monkeypatch.delenv("LLAMACPP_SERVER_PATH", raising=False)
    downloaded_exe = tmp_path / "downloaded" / "llama-server.exe"
    monkeypatch.setattr(llamacpp_runtime, "managed_server_executable", lambda: None)
    monkeypatch.setattr(llamacpp_runtime, "auto_download_allowed", lambda: True)
    monkeypatch.setattr(
        llamacpp_runtime,
        "ensure_managed_llama_server",
        lambda cancel_event=None: str(downloaded_exe),
    )
    monkeypatch.setattr(
        llamacpp_server.shutil, "which", lambda _name: "C:/system/cpu-llama-server.exe"
    )

    assert llamacpp_server.resolve_server_executable(download_enabled=True) == str(downloaded_exe)


# ---------------------------------------------------------------------------
# P1-1: Explicit proxy bypass for localhost health check & async client
# ---------------------------------------------------------------------------


def test_health_ok_bypasses_system_proxy(monkeypatch):
    backend = llamacpp_server.LlamaCppServerBackend()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")

    seen_handlers: list[object] = []

    class _FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    class _FakeOpener:
        def open(self, url, timeout=2.0):
            assert url == "http://127.0.0.1:19001/health"
            assert timeout == 2.0
            return _FakeResp()

    def _fake_build_opener(*handlers):
        seen_handlers.extend(handlers)
        return _FakeOpener()

    monkeypatch.setattr(llamacpp_server.urllib.request, "build_opener", _fake_build_opener)

    assert backend._health_ok(19001) is True
    assert len(seen_handlers) == 1
    assert isinstance(seen_handlers[0], llamacpp_server.urllib.request.ProxyHandler)
    assert seen_handlers[0].proxies == {}


def test_make_async_client_disables_trust_env(monkeypatch):
    backend = llamacpp_server.LlamaCppServerBackend()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    client = backend._make_async_client(19002)
    try:
        assert client._client._trust_env is False
    finally:
        import asyncio

        asyncio.run(client.close())


# ---------------------------------------------------------------------------
# P1-4: Plan alignment only when pending batches exist
# ---------------------------------------------------------------------------


def test_run_batched_skips_prepare_backend_when_all_cached(tmp_path):
    profile = get_profile("json")
    segments = [{"text": "おはよう", "start": 0.0, "end": 1.0}]
    cache_path = str(tmp_path / "cache.jsonl")
    calls = {"prepare": 0, "chat": 0}

    def _chat(_messages, **_kw):
        calls["chat"] += 1
        return json.dumps({"translations": [{"id": 0, "text": "早上好"}]}, ensure_ascii=False)

    def _prepare(_cancel):
        calls["prepare"] += 1
        return llamacpp_vram.LlamaAdaptivePlan(
            ctx_per_slot=4096,
            parallel=3,
            kv_unified=True,
            focus_radius=6,
            focus_context_chars=5000,
            total_vram_mb=16384,
            free_vram_mb=15000,
            budget_mib=13000,
            weights_mib=4800.0,
            kv_cache_mib=800.0,
            estimated_vram_mib=6100.0,
            adaptive=True,
        )

    common = dict(
        profile=profile,
        backend_name="llamacpp",
        chat=_chat,
        backoff_sleep=lambda *_a, **_kw: None,
        crash_probe=lambda: 0,
        batch_size=1,
        max_workers=1,
        api_retries=1,
        batch_repair_retries=1,
        batch_max_requests=5,
        prefix_warmup=False,
        extra_glossary="",
        full_context="",
        full_source_payload="",
        use_full_json_prefix=False,
        cache_path=cache_path,
        cache_lock=threading.Lock(),
        target_lang="简体中文",
        glossary="",
        character_reference="",
        prompt_version=profile.cache_signature(),
        model_identity="llamacpp:test",
        compact_system_prompt=True,
        reasoning_effort="",
        prepare_backend=_prepare,
        max_workers_cap=4,
    )

    first, timings1, _ = engine_module.run_batched(segments, **common)
    assert first == ["早上好"]
    assert calls == {"prepare": 1, "chat": 1}
    assert timings1[-1]["max_workers"] == 3

    # Second run: 100% cache hit -> prepare_backend must NOT be invoked!
    second, _timings2, _ = engine_module.run_batched(segments, **common)
    assert second == ["早上好"]
    assert calls == {"prepare": 1, "chat": 1}


# ---------------------------------------------------------------------------
# P1-5: HTTP 4xx/5xx vs connection error, stale proc guard, _await_ready cancel
# ---------------------------------------------------------------------------


def test_http_503_and_context_limit_are_retryable_without_retiring_server(monkeypatch):
    backend = llamacpp_server.LlamaCppServerBackend()
    proc = _AliveProc()
    backend._proc = proc
    backend._port = 18090

    req = httpx.Request("POST", "http://127.0.0.1:18090/v1/chat/completions")

    for status, body in (
        (503, "server busy"),
        (500, "internal error"),
        (400, "request exceeds the available context size"),
    ):
        resp = httpx.Response(status, request=req)
        err = httpx.HTTPStatusError(body, request=req, response=resp)

        def _make_err_client(error=err):
            class _ErrCompletions:
                async def create(self, **_kw):
                    raise error

            return SimpleNamespace(
                chat=SimpleNamespace(completions=_ErrCompletions()),
                close=lambda: None,
            )

        monkeypatch.setattr(
            backend, "_make_async_client", lambda _p, _factory=_make_err_client: _factory()
        )

        with pytest.raises(RetryableTranslationFormatError):
            backend._request_completion({"model": "m"}, cancel_event=None)

        assert backend.generation_retiring() is False
        assert backend.drain_status()["draining"] is False


def test_http_404_raises_runtime_error_without_retiring_server(monkeypatch):
    backend = llamacpp_server.LlamaCppServerBackend()
    backend._proc = _AliveProc()
    backend._port = 18091

    req = httpx.Request("POST", "http://127.0.0.1:18091/v1/chat/completions")
    err = httpx.HTTPStatusError("not found", request=req, response=httpx.Response(404, request=req))

    class _ErrCompletions:
        async def create(self, **_kw):
            raise err

    monkeypatch.setattr(
        backend,
        "_make_async_client",
        lambda _p: SimpleNamespace(
            chat=SimpleNamespace(completions=_ErrCompletions()),
            close=lambda: None,
        ),
    )

    with pytest.raises(RuntimeError, match="HTTP 404"):
        backend._request_completion({"model": "m"}, cancel_event=None)

    assert backend.generation_retiring() is False
    assert backend.drain_status()["draining"] is False


def test_connection_drop_marks_draining_only_if_proc_still_current(monkeypatch):
    backend = llamacpp_server.LlamaCppServerBackend()
    old_proc = _AliveProc()
    new_proc = _AliveProc()
    backend._proc = old_proc
    backend._port = 18092

    class _SwapAndFailCompletions:
        async def create(self, **_kw):
            # Simulate server replaced while this request was in flight
            backend._proc = new_proc
            raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(
        backend,
        "_make_async_client",
        lambda _p: SimpleNamespace(
            chat=SimpleNamespace(completions=_SwapAndFailCompletions()),
            close=lambda: None,
        ),
    )

    with pytest.raises(RuntimeError, match="请求失败"):
        backend._request_completion({"model": "m"}, cancel_event=None)

    # Stale old_proc must NOT retire or drain new_proc
    assert backend.generation_retiring() is False
    assert backend.drain_status()["draining"] is False


def test_await_ready_cleans_up_unready_process_on_cancel(monkeypatch):
    backend = llamacpp_server.LlamaCppServerBackend()
    proc = _AliveProc()
    backend._proc = proc
    backend._server_generation = 1
    cancel = threading.Event()
    torn_down: list[object] = []

    monkeypatch.setattr(backend, "_health_ok", lambda _p: False)
    monkeypatch.setattr(
        llamacpp_server,
        "_wait_briefly",
        lambda ev, _s: cancel.set(),
    )
    monkeypatch.setattr(
        backend, "_teardown_if_current", lambda p: torn_down.append(p)
    )

    with pytest.raises(TranslationCancelledError):
        backend._await_ready(proc, generation=1, port=18093, cancel_event=cancel)

    assert torn_down == [proc]


# ---------------------------------------------------------------------------
# P1-8 & P2: env stripping, no circular import, VRAM device selection, snapshot,
# runtime cancellable Condition & atomic os.replace & config.json copy
# ---------------------------------------------------------------------------


def test_server_environment_strips_sensitive_credentials(monkeypatch):
    monkeypatch.setenv("API_KEY", "sk-secret-api")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token")
    monkeypatch.setenv("CUSTOM_SECRET", "top-secret")
    monkeypatch.setenv("DB_PASSWORD", "pw123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    env = llamacpp_env.server_environment()
    for forbidden in ("API_KEY", "OPENAI_API_KEY", "HF_TOKEN", "CUSTOM_SECRET", "DB_PASSWORD"):
        assert forbidden not in env
    assert env.get("CUDA_VISIBLE_DEVICES") == "0"


def test_llamacpp_runtime_has_no_llamacpp_server_import():
    runtime_path = Path(llamacpp_runtime.__file__).resolve()
    tree = ast.parse(runtime_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "llamacpp_server" not in alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert "llamacpp_server" not in mod
            for alias in node.names:
                assert alias.name != "llamacpp_server"


def test_vram_selects_first_cuda_device_and_respects_cuda_visible_devices(monkeypatch):
    devices = [
        "CUDA0: NVIDIA GeForce RTX 4060 (8188 MiB, 7200 MiB free)",
        "CUDA1: NVIDIA RTX 4090 (24564 MiB, 23000 MiB free)",
    ]
    assert llamacpp_vram.parse_device_vram_mb(devices) == (8188, 7200)

    seen_cmd: list[list[str]] = []

    def _fake_run(cmd, **_kw):
        seen_cmd.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="12288, 10000\n")

    monkeypatch.setattr(llamacpp_vram.subprocess, "run", _fake_run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert llamacpp_vram.query_nvidia_smi_vram_mb() == (12288, 10000)
    assert "-i" in seen_cmd[-1]
    assert seen_cmd[-1][seen_cmd[-1].index("-i") + 1] == "1"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert llamacpp_vram.query_nvidia_smi_vram_mb() is None

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert llamacpp_vram.query_nvidia_smi_vram_mb() is None


def test_request_config_snapshots_llamacpp_model_keys(monkeypatch):
    for key in ("LLAMACPP_GGUF_PATH", "LLAMACPP_MODEL_REPO", "LLAMACPP_MODEL_FILE"):
        assert key in request_config.SNAPSHOT_KEYS

    monkeypatch.setenv("LLAMACPP_GGUF_PATH", "")
    monkeypatch.setenv("LLAMACPP_MODEL_REPO", "repo/a")
    monkeypatch.setenv("LLAMACPP_MODEL_FILE", "model-a.gguf")
    snap = request_config.capture()
    backend = llamacpp_server.LlamaCppServerBackend()

    with request_config.bound(snap):
        monkeypatch.setenv("LLAMACPP_MODEL_REPO", "repo/b")
        monkeypatch.setenv("LLAMACPP_MODEL_FILE", "model-b.gguf")
        assert backend.cache_identity() == "llamacpp:repo/a/model-a.gguf"


def test_ensure_managed_llama_server_waiter_is_cancellable(monkeypatch):
    gate_started = threading.Event()
    gate_release = threading.Event()
    waiter_cancel = threading.Event()

    monkeypatch.setattr(llamacpp_runtime, "managed_server_executable", lambda: None)
    monkeypatch.setattr(llamacpp_runtime, "detect_cuda_major", lambda: 13)

    def _slow_resolve(**_kw):
        gate_started.set()
        gate_release.wait(timeout=5.0)
        raise RuntimeError("stop-holder")

    monkeypatch.setattr(llamacpp_runtime, "resolve_latest_release_asset", _slow_resolve)

    holder_errors: list[BaseException] = []

    def _holder():
        try:
            llamacpp_runtime.ensure_managed_llama_server()
        except BaseException as exc:
            holder_errors.append(exc)

    t_holder = threading.Thread(target=_holder)
    t_holder.start()
    assert gate_started.wait(timeout=2.0)

    def _cancel_soon():
        time.sleep(0.05)
        waiter_cancel.set()

    t_cancel = threading.Thread(target=_cancel_soon)
    t_cancel.start()
    t0 = time.monotonic()
    with pytest.raises(TranslationCancelledError):
        llamacpp_runtime.ensure_managed_llama_server(cancel_event=waiter_cancel)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.5

    gate_release.set()
    t_cancel.join(timeout=2.0)
    t_holder.join(timeout=2.0)
    assert holder_errors and "stop-holder" in str(holder_errors[0])


def test_runtime_missing_cuda_message_references_config_json(monkeypatch, tmp_path):
    monkeypatch.setattr(llamacpp_runtime, "managed_server_executable", lambda: None)
    monkeypatch.setattr(llamacpp_runtime, "managed_runtime_dir", lambda: tmp_path / "llama-cpp")
    monkeypatch.setattr(llamacpp_runtime, "detect_cuda_major", lambda: 13)
    monkeypatch.setattr(
        llamacpp_runtime,
        "resolve_latest_release_asset",
        lambda **_kw: llamacpp_runtime.FALLBACK_CUDA13_RELEASE,
    )
    with pytest.raises(RuntimeError) as exc_info:
        llamacpp_runtime.ensure_managed_llama_server(
            download_fn=lambda *_a, **_kw: (_ for _ in ()).throw(OSError("network down")),
        )
    msg = str(exc_info.value)
    assert "config/config.json" in msg
    assert ".env" not in msg


def test_translator_has_no_dead_plaintext_api_key_or_think_regex():
    assert not hasattr(translator, "API_KEY")
    assert not hasattr(translator, "OPENAI_COMPATIBILITY_BASE_URL")
    assert not hasattr(translator, "_THINK_BLOCK_RE")
    assert not hasattr(translator, "_warn_about_inert_context")
    assert not hasattr(llm_settings, "API_KEY")
    assert not hasattr(llm_settings, "OPENAI_COMPATIBILITY_BASE_URL")
