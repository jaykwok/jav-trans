from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
import pytest

import web as _web_package

_SRC_WEB = Path(__file__).resolve().parents[2] / "src" / "web"
if str(_SRC_WEB) not in _web_package.__path__:
    _web_package.__path__.append(str(_SRC_WEB))

from web.app import create_app
from web import pipeline_manager as pm
from web.routes import config as config_routes
from web.routes import files as files_routes


async def _drain_queue(queue: asyncio.Queue) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        queue.task_done()


async def _reset_pm_state() -> None:
    async with pm._state_lock:
        pm._jobs.clear()
        pm._cancel_events.clear()
    await _drain_queue(pm.gpu_queue)
    await _drain_queue(pm.trans_queue)


def test_jobs_api_crud(tmp_path, monkeypatch):
    asyncio.run(_test_jobs_api_crud(tmp_path, monkeypatch))


def test_app_exposes_icon_assets(tmp_path, monkeypatch):
    asyncio.run(_test_app_exposes_icon_assets(tmp_path, monkeypatch))


def test_config_lists_recommended_asr_backend_first(monkeypatch):
    asyncio.run(_test_config_lists_recommended_asr_backend_first(monkeypatch))


def test_model_requirements_for_17b_include_boundary(tmp_path, monkeypatch):
    asyncio.run(_test_model_requirements_for_17b_include_boundary(tmp_path, monkeypatch))


def test_model_requirements_dedupe_default_asr_and_boundary(tmp_path, monkeypatch):
    asyncio.run(_test_model_requirements_dedupe_default_asr_and_boundary(tmp_path, monkeypatch))


def test_model_requirements_marks_disabled_boundary_download(tmp_path, monkeypatch):
    asyncio.run(_test_model_requirements_marks_disabled_boundary_download(tmp_path, monkeypatch))


def test_model_requirements_includes_cuda_driver_warning(tmp_path, monkeypatch):
    asyncio.run(_test_model_requirements_includes_cuda_driver_warning(tmp_path, monkeypatch))


def test_cuda_environment_status_reads_frozen_probe_file(tmp_path, monkeypatch):
    config_routes._cuda_environment_status.cache_clear()
    monkeypatch.setattr(config_routes, "is_frozen", lambda: True)
    monkeypatch.setattr(config_routes, "PROJECT_ROOT", tmp_path)

    def fake_run(command, *, env, **kwargs):
        Path(env["JAV_TRANS_CUDA_PROBE_OUTPUT"]).write_text(
            '{"status":"ok","ok":true,"code":"ok","message":"file result"}',
            encoding="utf-8",
        )
        return config_routes.subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(config_routes.subprocess, "run", fake_run)

    try:
        payload = config_routes._cuda_environment_status()
    finally:
        config_routes._cuda_environment_status.cache_clear()

    assert payload["ok"] is True
    assert payload["code"] == "ok"
    assert payload["message"] == "file result"
    assert not list((tmp_path / "tmp").glob("cuda-probe-*.json"))


def test_windows_picker_subprocesses_hide_console(monkeypatch):
    calls: list[dict] = []

    monkeypatch.setattr(
        files_routes,
        "no_window_subprocess_kwargs",
        lambda: {"creationflags": 0x08000000},
    )

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return files_routes.subprocess.CompletedProcess(
            command,
            0,
            stdout="D:\\Videos\\sample.mp4\nD:\\Videos\\sample.mkv\n",
            stderr="",
        )

    monkeypatch.setattr(files_routes.subprocess, "run", fake_run)

    assert files_routes._pick_files_windows() == [
        "D:\\Videos\\sample.mp4",
        "D:\\Videos\\sample.mkv",
    ]
    assert files_routes._pick_folder_windows() == "D:\\Videos\\sample.mp4\nD:\\Videos\\sample.mkv"

    assert len(calls) == 2
    for call in calls:
        assert call["command"][0] == "powershell"
        assert call["creationflags"] == 0x08000000
        assert call["check"] is False
        assert call["capture_output"] is True


def test_settings_proxy_updates_runtime_env(monkeypatch):
    asyncio.run(_test_settings_proxy_updates_runtime_env(monkeypatch))


def test_proxy_test_reports_unconfigured_when_no_proxy(monkeypatch):
    asyncio.run(_test_proxy_test_reports_unconfigured(monkeypatch))


def test_proxy_test_ok_when_reachable(monkeypatch):
    asyncio.run(_test_proxy_test_ok_when_reachable(monkeypatch))


def test_settings_llm_api_format_updates_runtime_env(monkeypatch):
    asyncio.run(_test_settings_llm_api_format_updates_runtime_env(monkeypatch))


def test_settings_translation_fields_update_runtime_env(monkeypatch):
    asyncio.run(_test_settings_translation_fields_update_runtime_env(monkeypatch))


def test_settings_ignores_removed_asr_context(monkeypatch):
    asyncio.run(_test_settings_ignores_removed_asr_context(monkeypatch))


def test_settings_env_file_quotes_multiline_values(tmp_path, monkeypatch):
    asyncio.run(_test_settings_env_file_quotes_multiline_values(tmp_path, monkeypatch))


def test_settings_creates_env_file_on_first_save(tmp_path, monkeypatch):
    asyncio.run(_test_settings_creates_env_file_on_first_save(tmp_path, monkeypatch))


def test_models_api_falls_back_to_v1_models(monkeypatch):
    asyncio.run(_test_models_api_falls_back_to_v1_models(monkeypatch))


def test_jobs_snapshot_saved_translation_settings(tmp_path, monkeypatch):
    asyncio.run(_test_jobs_snapshot_saved_translation_settings(tmp_path, monkeypatch))


def test_jobs_api_retry_cancelled_job(tmp_path, monkeypatch):
    asyncio.run(_test_jobs_api_retry_cancelled_job(tmp_path, monkeypatch))


def test_public_error_message_prefers_exception_detail():
    class DetailError(RuntimeError):
        detail = "GPU 显存不足：请切换到 0.6B"

        def __str__(self) -> str:
            return "oom: internal wrapper"

    assert pm._public_error_message(DetailError()) == "GPU 显存不足：请切换到 0.6B"


def test_jobs_api_rejects_invalid_job_spec(tmp_path, monkeypatch):
    asyncio.run(_test_jobs_api_rejects_invalid_job_spec(tmp_path, monkeypatch))


def test_open_routes_are_limited_to_job_paths(tmp_path, monkeypatch):
    asyncio.run(_test_open_routes_are_limited_to_job_paths(tmp_path, monkeypatch))


async def _test_app_exposes_icon_assets(tmp_path, monkeypatch):
    image_dir = tmp_path / "src" / "assets" / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "icon.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIEND\xaeB`\x82"
    )
    (image_dir / "icon.ico").write_bytes(b"\x00\x00\x01\x00")

    import web.app as web_app

    monkeypatch.setattr(web_app, "resource_root", lambda: tmp_path)
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        icon = await client.get("/assets/images/icon.png")
        app_icon = await client.get("/assets/images/icon.ico")

    assert icon.status_code == 200
    assert app_icon.status_code == 200
    assert icon.headers["content-type"] == "image/png"
    index_html = (_SRC_WEB / "static" / "index.html").read_text(encoding="utf-8")
    assert "/assets/images/icon.png" in index_html
    assert 'href="/icon.png"' not in index_html
    assert 'src="/icon.png"' not in index_html
    assert "favicon.ico" not in index_html


async def _test_config_lists_recommended_asr_backend_first(monkeypatch):
    monkeypatch.delenv("ASR_BACKEND", raising=False)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["backends"][0] == config_routes.RECOMMENDED_ASR_BACKEND
    assert payload["defaults"]["asr_backend"] == config_routes.RECOMMENDED_ASR_BACKEND
    assert payload["recommended_asr_backend"] == config_routes.RECOMMENDED_ASR_BACKEND
    assert payload["engine_defaults"]["asr_backend"] == config_routes.RECOMMENDED_ASR_BACKEND
    assert "translation_batch_size" not in payload["defaults"]
    assert payload["defaults"]["translation_max_workers"] == 16
    assert "show_speaker" not in payload["defaults"]
    assert set(payload["backends"]) == set(config_routes.BACKENDS)


def _isolate_model_requirement_env(
    tmp_path,
    monkeypatch,
    *,
    boundary_no_download: str = "0",
    cuda_status: dict | None = None,
) -> None:
    models_root = tmp_path / "models"
    resource_root = tmp_path / "resource"
    monkeypatch.setattr(config_routes.model_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_routes.model_paths, "MODELS_ROOT", models_root)
    monkeypatch.setattr(config_routes.model_paths, "RESOURCE_ROOT", resource_root)
    monkeypatch.setattr(config_routes.model_paths, "BUNDLED_MODELS_ROOT", resource_root / "models")
    monkeypatch.setattr(config_routes.model_paths, "is_frozen", lambda: False)
    env_values = {
        "ASR_MODEL_ID": "",
        "ASR_MODEL_PATH": "",
        "SPEECH_BOUNDARY_JA_PTM": config_routes.RECOMMENDED_ASR_BACKEND,
        "SPEECH_BOUNDARY_JA_MODEL_PATH": "models/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
        "SPEECH_BOUNDARY_JA_NO_DOWNLOAD": boundary_no_download,
    }
    for key, value in env_values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        config_routes,
        "_cuda_environment_status",
        lambda: cuda_status
        or {
            "status": "ok",
            "ok": True,
            "code": "ok",
            "message": "CUDA 可用。",
            "torch_cuda_version": "12.8",
        },
    )


async def _test_model_requirements_for_17b_include_boundary(tmp_path, monkeypatch):
    _isolate_model_requirement_env(tmp_path, monkeypatch)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/api/model-requirements",
            params={"asr_backend": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["missing_count"] == 1
    assert payload["needs_download"] is True
    assert payload["download_disabled"] is False
    assert payload["checkpoint_missing_count"] == 0
    assert len(payload["required_checkpoints"]) == 5
    assert all(item["present"] for item in payload["required_checkpoints"])
    by_role = {
        tuple(item["roles"]): item["repo_id"]
        for item in payload["required_models"]
    }
    assert by_role[("asr", "boundary_feature")] == "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"


async def _test_model_requirements_dedupe_default_asr_and_boundary(tmp_path, monkeypatch):
    _isolate_model_requirement_env(tmp_path, monkeypatch)
    local_model = tmp_path / "models" / "jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
    local_model.mkdir(parents=True)
    (local_model / "config.json").write_text("{}", encoding="utf-8")
    (local_model / "model.safetensors").write_bytes(b"weights")

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/api/model-requirements",
            params={"asr_backend": config_routes.RECOMMENDED_ASR_BACKEND},
        )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["required_models"]) == 1
    assert payload["missing_count"] == 0
    assert payload["needs_download"] is False
    merged = next(
        item
        for item in payload["required_models"]
        if item["repo_id"] == config_routes.RECOMMENDED_ASR_BACKEND
    )
    assert set(merged["roles"]) == {"asr", "boundary_feature"}
    assert merged["present"] is True


async def _test_model_requirements_marks_disabled_boundary_download(tmp_path, monkeypatch):
    _isolate_model_requirement_env(tmp_path, monkeypatch, boundary_no_download="1")

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/api/model-requirements",
            params={"asr_backend": "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf"},
        )

    assert response.status_code == 200
    payload = response.json()
    boundary = next(
        item for item in payload["required_models"] if item["roles"] == ["boundary_feature"]
    )
    assert boundary["download_enabled"] is False
    assert payload["needs_download"] is True
    assert payload["download_disabled"] is True
    assert payload["checkpoint_missing_count"] == 0
    assert payload["pipeline_ready"] is False


async def _test_model_requirements_includes_cuda_driver_warning(tmp_path, monkeypatch):
    _isolate_model_requirement_env(
        tmp_path,
        monkeypatch,
        cuda_status={
            "status": "error",
            "ok": False,
            "code": "driver_too_old",
            "message": "NVIDIA 驱动最高支持 CUDA 12.1，但当前打包的 PyTorch 需要 CUDA 12.8。",
            "torch_cuda_version": "12.8",
            "nvidia_smi": {
                "cuda_version": "12.1",
                "driver_version": "531.79",
            },
        },
    )
    local_model = tmp_path / "models" / "jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
    local_model.mkdir(parents=True)
    (local_model / "config.json").write_text("{}", encoding="utf-8")
    (local_model / "model.safetensors").write_bytes(b"weights")

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/api/model-requirements",
            params={"asr_backend": config_routes.RECOMMENDED_ASR_BACKEND},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["missing_count"] == 0
    assert payload["gpu_ready"] is False
    assert payload["pipeline_ready"] is False
    assert payload["cuda"]["code"] == "driver_too_old"
    assert "CUDA 12.8" in payload["cuda"]["message"]


async def _test_settings_proxy_updates_runtime_env(monkeypatch):
    for key in (
        "PROXY_PROTOCOL",
        "PROXY_HOST",
        "PROXY_PORT",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    monkeypatch.setattr(config_routes, "_update_env_file", lambda _changes: None)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/settings",
            json={
                "proxy_protocol": "http",
                "proxy_host": "127.0.0.1",
                "proxy_port": 7890,
            },
        )
        assert response.status_code == 200
        assert os.environ["PROXY_PROTOCOL"] == "http"
        assert os.environ["PROXY_HOST"] == "127.0.0.1"
        assert os.environ["PROXY_PORT"] == "7890"
        assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7890"
        assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890"
        assert os.environ["ALL_PROXY"] == "http://127.0.0.1:7890"
        assert "HF_ENDPOINT" not in os.environ

        current = await client.get("/api/settings")
        assert current.status_code == 200
        payload = current.json()
        assert payload["proxy_protocol"] == "http"
        assert payload["proxy_host"] == "127.0.0.1"
        assert payload["proxy_port"] == 7890

        cleared = await client.post(
            "/api/settings",
            json={"proxy_host": "", "proxy_port": None},
        )

    assert cleared.status_code == 200
    assert os.environ["PROXY_HOST"] == ""
    assert os.environ["PROXY_PORT"] == ""
    assert "HTTP_PROXY" not in os.environ
    assert "HTTPS_PROXY" not in os.environ
    assert "ALL_PROXY" not in os.environ


async def _test_proxy_test_reports_unconfigured(monkeypatch):
    for key in ("PROXY_HOST", "PROXY_PORT", "PROXY_PROTOCOL"):
        monkeypatch.delenv(key, raising=False)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/proxy-test")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["proxy_url"] == ""
    assert "未启用" in payload["error"]


class _FakeProxyResponse:
    status_code = 200


class _FakeProxyClient:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        self.requested_url = url
        return _FakeProxyResponse()


async def _test_proxy_test_ok_when_reachable(monkeypatch):
    monkeypatch.setenv("PROXY_PROTOCOL", "http")
    monkeypatch.setenv("PROXY_HOST", "127.0.0.1")
    monkeypatch.setenv("PROXY_PORT", "7890")
    # Capture the real client BEFORE patching so the ASGI test client below is
    # unaffected; the route uses config_routes.httpx.AsyncClient (patched).
    original_async_client = httpx.AsyncClient
    monkeypatch.setattr(config_routes.httpx, "AsyncClient", _FakeProxyClient)

    transport = httpx.ASGITransport(app=create_app())
    async with original_async_client(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/proxy-test")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["status_code"] == 200
    assert payload["proxy_url"] == "http://127.0.0.1:7890"
    assert isinstance(payload["elapsed_ms"], int)


async def _test_settings_llm_api_format_updates_runtime_env(monkeypatch):
    monkeypatch.delenv("LLM_API_FORMAT", raising=False)
    monkeypatch.setattr(config_routes, "_read_env_entry", lambda key: (key == "LLM_API_FORMAT", "chat"))
    monkeypatch.setattr(config_routes, "_update_env_file", lambda _changes: None)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/settings",
            json={"llm_api_format": "responses"},
        )
        assert response.status_code == 200
        assert os.environ["LLM_API_FORMAT"] == "responses"

        settings = await client.get("/api/settings")
        assert settings.status_code == 200
        assert settings.json()["llm_api_format"] == "responses"

        invalid = await client.post(
            "/api/settings",
            json={"llm_api_format": "legacy"},
        )

    assert invalid.status_code == 422


async def _test_settings_translation_fields_update_runtime_env(monkeypatch):
    for key in ("TRANSLATION_GLOSSARY", "TARGET_LANG", "LLM_REASONING_EFFORT"):
        monkeypatch.delenv(key, raising=False)
    stale_env = {
        "TRANSLATION_GLOSSARY": "old glossary",
        "TARGET_LANG": "简体中文",
        "LLM_REASONING_EFFORT": "xhigh",
    }
    monkeypatch.setattr(
        config_routes,
        "_read_env_entry",
        lambda key: (key in stale_env, stale_env.get(key, "")),
    )
    monkeypatch.setattr(config_routes, "_update_env_file", lambda _changes: None)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/settings",
            json={
                "translation_glossary": "",
                "target_lang": "繁體中文",
                "llm_reasoning_effort": "medium",
            },
        )
        assert response.status_code == 200

        settings = await client.get("/api/settings")

    assert settings.status_code == 200
    payload = settings.json()
    assert payload["translation_glossary"] == ""
    assert payload["target_lang"] == "繁體中文"
    assert payload["llm_reasoning_effort"] == "medium"


async def _test_settings_ignores_removed_asr_context(monkeypatch):
    monkeypatch.delenv("ASR_CONTEXT", raising=False)
    monkeypatch.setattr(config_routes, "_update_env_file", lambda _changes: None)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/settings",
            json={"asr_context": "小那海"},
        )

    assert response.status_code == 200
    assert "ASR_CONTEXT" not in os.environ


async def _test_settings_env_file_quotes_multiline_values(tmp_path, monkeypatch):
    from dotenv import dotenv_values

    env_path = tmp_path / ".env"
    env_path.write_text("API_KEY=old\nTARGET_LANG=简体中文\n", encoding="utf-8")
    monkeypatch.setattr(config_routes, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("API_KEY", raising=False)
    malicious_value = "safe\nOPENAI_COMPATIBILITY_BASE_URL=https://evil.example"

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/settings",
            json={"api_key": malicious_value},
        )

    assert response.status_code == 200
    values = dotenv_values(env_path)
    assert values["API_KEY"] == malicious_value
    assert values.get("OPENAI_COMPATIBILITY_BASE_URL") is None
    assert os.environ["API_KEY"] == malicious_value


async def _test_settings_creates_env_file_on_first_save(tmp_path, monkeypatch):
    from dotenv import dotenv_values

    env_path = tmp_path / ".env"
    assert not env_path.exists()
    monkeypatch.setattr(config_routes, "PROJECT_ROOT", tmp_path)
    for key in (
        "API_KEY",
        "OPENAI_COMPATIBILITY_BASE_URL",
        "LLM_MODEL_NAME",
        "TARGET_LANG",
        "LLM_API_FORMAT",
        "LLM_REASONING_EFFORT",
    ):
        monkeypatch.delenv(key, raising=False)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/settings",
            json={
                "api_key": "first-run-key",
                "base_url": "https://api.example.test/v1",
                "model": "translator-test",
                "target_lang": "繁體中文",
                "llm_api_format": "responses",
                "llm_reasoning_effort": "medium",
            },
        )
        settings = await client.get("/api/settings")

    assert response.status_code == 200
    assert env_path.exists()
    text = env_path.read_text(encoding="utf-8")
    assert "# Defaults live in src/core/config.py" in text
    assert "# ASR_STAGE_WORKER_VRAM_BUDGET_MB=auto" in text
    assert "# ASR_STAGE_WORKER_VRAM_RATIO=0.95" in text
    assert "# ASR_STAGE_WORKER_MODE=subprocess" not in text
    assert "# ASR_BATCH_SIZE_BY_REPO=" in text
    values = dotenv_values(env_path)
    assert values["API_KEY"] == "first-run-key"
    assert values["OPENAI_COMPATIBILITY_BASE_URL"] == "https://api.example.test/v1"
    assert values["LLM_MODEL_NAME"] == "translator-test"
    assert values["TARGET_LANG"] == "繁體中文"
    assert values["LLM_API_FORMAT"] == "responses"
    assert values["LLM_REASONING_EFFORT"] == "medium"
    assert "ASR_BACKEND" not in values
    assert "ASR_BATCH_SIZE_BY_REPO" not in values
    assert settings.status_code == 200
    payload = settings.json()
    assert payload["api_key_set"] is True
    assert payload["base_url"] == "https://api.example.test/v1"
    assert payload["model"] == "translator-test"
    assert payload["target_lang"] == "繁體中文"


async def _test_models_api_falls_back_to_v1_models(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_COMPATIBILITY_BASE_URL", "https://api.example.test")

    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if str(request.url) == "https://api.example.test/models":
            return httpx.Response(
                200,
                text="<html></html>",
                headers={"content-type": "text/html"},
            )
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "gpt-5.4-mini"},
                    {"id": "gpt-5.5"},
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    original_async_client = httpx.AsyncClient

    def mock_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(config_routes.httpx, "AsyncClient", mock_async_client)

    app_transport = httpx.ASGITransport(app=create_app())
    async with original_async_client(
        transport=app_transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/models")

    assert response.status_code == 200
    assert response.json() == {"models": ["gpt-5.4-mini", "gpt-5.5"]}
    assert requests == [
        "https://api.example.test/models",
        "https://api.example.test/v1/models",
    ]


async def _test_jobs_api_crud(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    original_create_job = pm.create_job

    async def fake_create_job(spec):
        jobs = await original_create_job(spec)
        assert not pm.gpu_queue.empty()
        return jobs

    monkeypatch.setattr(pm, "create_job", fake_create_job)

    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/jobs",
                json={"video_paths": ["sample.mp4"], "skip_translation": True},
            )
            assert response.status_code == 201
            payload = response.json()
            assert payload["ids"]
            job_id = payload["ids"][0]

            response = await client.get("/api/jobs")
            assert response.status_code == 200
            jobs = response.json()
            assert jobs
            assert jobs[0]["status"] in {
                "queued",
                "asr",
                "translating",
                "done",
                "failed",
            }

            response = await client.get(f"/api/jobs/{job_id}")
            assert response.status_code == 200
            assert response.json()["id"] == job_id

            response = await client.delete(f"/api/jobs/{job_id}")
            assert response.status_code == 200
            assert response.json() == {"ok": True}
    finally:
        await _reset_pm_state()


async def _test_jobs_api_rejects_invalid_job_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()

    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            invalid_payloads = [
                {"video_paths": []},
                {"video_paths": ["sample.mp4"], "vad_threshold": 1.5},
                {"video_paths": ["sample.mp4"], "translation_max_workers": 0},
                {"video_paths": ["sample.mp4"], "translation_max_workers": 65},
                {"video_paths": ["sample.mp4"], "translation_max_workers": 99},
            ]
            for payload in invalid_payloads:
                response = await client.post("/api/jobs", json=payload)
                assert response.status_code == 422

        assert await pm.list_jobs() == []
    finally:
        await _reset_pm_state()


async def _test_jobs_snapshot_saved_translation_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    monkeypatch.setenv("ASR_CONTEXT", "小那海")
    monkeypatch.setenv("TRANSLATION_GLOSSARY", "ねこ-猫")
    monkeypatch.setenv("TARGET_LANG", "繁體中文")
    monkeypatch.setenv("LLM_API_FORMAT", "responses")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "medium")
    await _reset_pm_state()

    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            created = await client.post(
                "/api/jobs",
                json={"video_paths": ["sample.mp4"]},
            )
            assert created.status_code == 201
            job_id = created.json()["ids"][0]

            response = await client.get(f"/api/jobs/{job_id}")
            assert response.status_code == 200
            spec = response.json()["spec"]

        assert "asr_context" not in spec
        assert spec["translation_glossary"] == "ねこ-猫"
        assert spec["target_lang"] == "繁體中文"
        assert spec["llm_api_format"] == "responses"
        assert spec["llm_reasoning_effort"] == "medium"
    finally:
        await _reset_pm_state()


async def _test_jobs_api_retry_cancelled_job(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()

    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            created = await client.post(
                "/api/jobs",
                json={"video_paths": ["sample.mp4"]},
            )
            assert created.status_code == 201
            job_id = created.json()["ids"][0]

            cancelled = await client.delete(f"/api/jobs/{job_id}")
            assert cancelled.status_code == 200

            retried = await client.post(f"/api/jobs/{job_id}/retry")
            assert retried.status_code == 200
            payload = retried.json()
            assert payload["id"] == job_id
            assert payload["status"] == "queued"

            missing = await client.post("/api/jobs/missing/retry")
            assert missing.status_code == 404

            conflict = await client.post(f"/api/jobs/{job_id}/retry")
            assert conflict.status_code == 409
    finally:
        await _reset_pm_state()


async def _test_open_routes_are_limited_to_job_paths(tmp_path, monkeypatch):
    from web.routes import files as files_routes

    monkeypatch.setattr(pm, "_jobs_path", tmp_path / "jobs.json")
    await _reset_pm_state()
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")
    other_video = tmp_path / "other.mp4"
    other_video.write_bytes(b"other")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    artifact_path = output_dir / "sample.srt"
    artifact_path.write_text("1\n", encoding="utf-8")
    unrelated_path = tmp_path / "secret.txt"
    unrelated_path.write_text("secret", encoding="utf-8")
    opened: list[tuple[str, str]] = []
    app = create_app()

    class DummyPopen:
        def __init__(self, args):
            opened.append(("popen", str(args[-1])))

    if os.name == "nt":
        monkeypatch.setattr(
            files_routes.os,
            "startfile",
            lambda path: opened.append(("startfile", str(path))),
            raising=False,
        )
    else:
        monkeypatch.setattr(files_routes.subprocess, "Popen", DummyPopen)

    try:
        jobs = await pm.create_job(
            pm.JobSpec(
                video_paths=[str(video_path)],
                output_dir=str(output_dir),
            )
        )
        job = jobs[0]
        async with pm._state_lock:
            job.status = "done"
            job.artifacts = ["sample.srt"]
            pm._jobs[job.id] = job

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            allowed_video = await client.post(
                "/api/open-video",
                params={"job_id": job.id, "path": str(video_path)},
            )
            blocked_video = await client.post(
                "/api/open-video",
                params={"job_id": job.id, "path": str(other_video)},
            )
            allowed_folder = await client.post(
                "/api/open-folder",
                params={"job_id": job.id, "path": "sample.srt"},
            )
            blocked_folder = await client.post(
                "/api/open-folder",
                params={"job_id": job.id, "path": str(unrelated_path)},
            )

        assert allowed_video.status_code == 200
        assert blocked_video.status_code == 403
        assert allowed_folder.status_code == 200
        assert blocked_folder.status_code == 403
        open_kind = "startfile" if os.name == "nt" else "popen"
        assert (open_kind, str(video_path.resolve())) in opened
        assert (open_kind, str(output_dir.resolve())) in opened
    finally:
        await _reset_pm_state()
