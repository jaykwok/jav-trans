from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, get_args

import httpx
from fastapi import APIRouter, HTTPException, Query

from asr.backends.qwen import active_qwen_asr_model_id
from core.config import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SETTINGS,
    REASONING_EFFORTS,
    load_config,
    read_config_file,
    sync_network_proxy_environment,
    update_config_file,
)
from core.diagnostics import format_exception_chain, redact_url_credentials
from utils import model_paths
from utils.model_paths import PROJECT_ROOT
from utils.runtime_paths import is_frozen
from utils.subprocess_tools import no_window_subprocess_kwargs
from web.models import (
    JobSpec,
    SettingsRead,
    SettingsUpdate,
    normalize_llm_reasoning_effort as _normalize_llm_reasoning_effort,
)


router = APIRouter()

SUBTITLE_MODES = list(get_args(JobSpec.model_fields["subtitle_mode"].annotation))
DEFAULT_JOB_DEFAULTS = {
    name: field.default
    for name, field in JobSpec.model_fields.items()
    if not field.is_required()
}
def _config_file_path() -> Path:
    return PROJECT_ROOT / "config" / "config.json"


def _setting(key: str) -> str:
    return os.getenv(key, DEFAULT_SETTINGS.get(key, ""))


def _read_config_entry(key: str) -> tuple[bool, str]:
    values = read_config_file(_config_file_path())
    if key in values:
        return True, values[key] or ""
    return False, ""


def _read_config_value(key: str) -> str:
    found, value = _read_config_entry(key)
    return value if found else ""


def _runtime_or_config_value(key: str) -> str:
    if key in os.environ:
        return os.environ.get(key, "").strip()
    return _read_config_value(key)


def _runtime_or_config_or_setting(key: str, fallback: str = "") -> str:
    if key in os.environ:
        return os.environ.get(key, "").strip()
    found, value = _read_config_entry(key)
    if found:
        return value
    return _setting(key) or fallback


def _update_config_file(updates: dict[str, str]) -> None:
    try:
        update_config_file(updates, path=_config_file_path())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _mask_key(k: str) -> str:
    if not k:
        return ""
    if len(k) <= 8:
        return "***"
    return k[:4] + "****..." + k[-4:]


def _short_model_name(repo_id: str) -> str:
    return repo_id.strip("/").rsplit("/", 1)[-1] or repo_id


def _model_requirement(
    *,
    repo_id: str,
    explicit_path: str = "",
    download_enabled: bool = True,
) -> dict[str, Any]:
    status = model_paths.model_spec_status(
        explicit_path or None,
        repo_id,
        download=download_enabled,
    )
    return {
        "repo_id": repo_id,
        "short_name": _short_model_name(repo_id),
        "local_path": status["path"],
        "checked_paths": status["checked_paths"],
        "present": bool(status["present"]),
        "download_enabled": bool(download_enabled),
    }


def _cuda_probe_command() -> list[str]:
    if is_frozen():
        return [sys.executable, "--cuda-probe-child"]
    return [sys.executable, str(PROJECT_ROOT / "launcher.py"), "--cuda-probe-child"]


def _parse_probe_json(stdout: str) -> dict[str, Any] | None:
    text = (stdout or "").strip()
    if not text:
        return None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


_CUDA_ENV_CACHE_LOCK = threading.Lock()
_CUDA_ENV_CACHE: dict[str, Any] | None = None


def _probe_cuda_environment_status() -> dict[str, Any]:
    from core.diagnostics import format_exit_code

    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    probe_output_path: Path | None = None
    if is_frozen():
        try:
            probe_dir = PROJECT_ROOT / "tmp"
            probe_dir.mkdir(parents=True, exist_ok=True)
            fd, path_text = tempfile.mkstemp(
                prefix="cuda-probe-",
                suffix=".json",
                dir=probe_dir,
            )
        except OSError:
            fd, path_text = tempfile.mkstemp(prefix="jav-sub-cuda-probe-", suffix=".json")
        os.close(fd)
        probe_output_path = Path(path_text)
        env["JAV_SUB_CUDA_PROBE_OUTPUT"] = str(probe_output_path)
    try:
        completed = subprocess.run(
            _cuda_probe_command(),
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            **no_window_subprocess_kwargs(),
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic endpoint
        if probe_output_path is not None:
            probe_output_path.unlink(missing_ok=True)
        return {
            "status": "error",
            "ok": False,
            "code": "probe_failed",
            "message": f"CUDA 环境检测无法启动：{type(exc).__name__}: {exc}",
        }
    payload = _parse_probe_json(completed.stdout)
    if payload is None and probe_output_path is not None and probe_output_path.exists():
        try:
            payload = _parse_probe_json(probe_output_path.read_text(encoding="utf-8"))
        except OSError:
            payload = None
    if probe_output_path is not None:
        probe_output_path.unlink(missing_ok=True)
    if payload is None:
        detail = (completed.stderr or completed.stdout or "").strip()
        exit_text = f"（退出码 {format_exit_code(completed.returncode)}）"
        return {
            "status": "error",
            "ok": False,
            "code": "probe_invalid_output",
            "message": f"CUDA 环境检测没有返回有效结果{exit_text}。{detail}",
        }
    payload.setdefault("ok", payload.get("status") == "ok")
    payload.setdefault("status", "ok" if payload.get("ok") else "error")
    if completed.returncode != 0 and payload.get("ok"):
        payload["status"] = "warning"
        payload["ok"] = False
    if completed.stderr.strip():
        payload["stderr"] = completed.stderr.strip()
    return payload


def _cuda_environment_status() -> dict[str, Any]:
    global _CUDA_ENV_CACHE
    with _CUDA_ENV_CACHE_LOCK:
        if _CUDA_ENV_CACHE is not None:
            return dict(_CUDA_ENV_CACHE)
    result = _probe_cuda_environment_status()
    # Only cache definitive results; transient probe failures should be retried
    # on the next refresh instead of staying pinned for the whole process life.
    if result.get("code") not in {"probe_failed", "probe_invalid_output"}:
        with _CUDA_ENV_CACHE_LOCK:
            _CUDA_ENV_CACHE = dict(result)
    return result


def _clear_cuda_environment_cache() -> None:
    global _CUDA_ENV_CACHE
    with _CUDA_ENV_CACHE_LOCK:
        _CUDA_ENV_CACHE = None


def _proxy_settings_from_runtime() -> tuple[str, str, int | None]:
    protocol = _runtime_or_config_or_setting("PROXY_PROTOCOL", "http").lower()
    if protocol not in {"http", "https", "socks5"}:
        protocol = "http"
    host = _runtime_or_config_or_setting("PROXY_HOST", "")
    port_text = _runtime_or_config_or_setting("PROXY_PORT", "")
    try:
        port = int(port_text) if str(port_text).strip() else None
    except (TypeError, ValueError):
        port = None
    if port is not None and not (1 <= port <= 65535):
        port = None
    return protocol, host, port


def _proxy_settings_changes(update: SettingsUpdate) -> dict[str, str] | None:
    proxy_fields_present = any(
        value is not None
        for value in (update.proxy_protocol, update.proxy_host, update.proxy_port)
    )
    if not proxy_fields_present:
        return None

    current_protocol, current_host, current_port = _proxy_settings_from_runtime()
    protocol = (update.proxy_protocol or current_protocol or "http").strip().lower()
    if protocol not in {"http", "https", "socks5"}:
        raise HTTPException(status_code=422, detail="proxy_protocol must be http, https, or socks5")
    host = (
        update.proxy_host.strip()
        if update.proxy_host is not None
        else current_host.strip()
    )
    port = update.proxy_port if update.proxy_port is not None else current_port
    if not host or port is None:
        return {
            "PROXY_PROTOCOL": protocol,
            "PROXY_HOST": "",
            "PROXY_PORT": "",
        }

    return {
        "PROXY_PROTOCOL": protocol,
        "PROXY_HOST": host,
        "PROXY_PORT": str(port),
    }


def _clear_saved_hf_mirror_if_present(changes: dict[str, str]) -> None:
    endpoint = _runtime_or_config_or_setting("HF_ENDPOINT", "").strip().rstrip("/")
    if endpoint == "https://hf-mirror.com":
        changes["HF_ENDPOINT"] = ""


def _strip_llm_endpoint_path(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    lower = normalized.lower()
    for suffix in ("/chat/completions", "/responses", "/completions"):
        if lower.endswith(suffix):
            return normalized[: -len(suffix)].rstrip("/")
    return normalized


def _model_endpoint_candidates(base_url: str) -> list[str]:
    normalized = _strip_llm_endpoint_path(base_url)
    if not normalized:
        return []
    if normalized.lower().endswith("/models"):
        return [normalized]

    candidates = [f"{normalized}/models"]
    if not normalized.lower().endswith("/v1"):
        candidates.append(f"{normalized}/v1/models")

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _model_id_from_item(item: dict) -> str | None:
    for field in ("id", "slug", "model", "name"):
        value = item.get(field)
        if value:
            return str(value)
    return None


def _extract_model_ids(payload: Any) -> list[str]:
    """Model ids from either OpenAI's `data[]` or a `models[]` variant.

    bigmodel.cn's `/api/v1/models` (the Codex-CLI-compatible gateway) answers
    200 with `{"models": [{"slug": "glm-5.3", ...}]}` instead of OpenAI's
    `{"data": [{"id": ...}]}` - same list-of-models intent, different key
    names, confirmed against the vendor with a live key on 2026-09-01.
    """
    if not isinstance(payload, dict):
        return []
    for key in ("data", "models"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        ids = [
            _model_id_from_item(item) for item in items if isinstance(item, dict)
        ]
        ids = [model_id for model_id in ids if model_id]
        if ids:
            return sorted(ids)
    return []


_CUDA_PROBE_LOCK = threading.Lock()


async def _cuda_environment_status_off_the_loop() -> dict[str, Any]:
    """The probe, run where a 20-second child process cannot stop the server.

    `_cuda_environment_status` spawns a torch-importing child and waits up to
    20s for it. Called straight from an async route, that wait happens *on the
    event loop*: every other request - the progress stream, 取消 - waits behind
    it, because a coroutine only yields where it awaits. The result is cached,
    so this was a first-load-only stall, which is exactly when the page is
    trying to render.

    The lock makes it single-flight: `lru_cache` does not deduplicate concurrent
    callers, and two page loads racing would otherwise spawn two probes. The
    second waits in a worker thread and gets the cached answer.
    """
    return await asyncio.get_running_loop().run_in_executor(
        None, _cuda_environment_status_blocking
    )


def _cuda_environment_status_blocking() -> dict[str, Any]:
    with _CUDA_PROBE_LOCK:
        return _cuda_environment_status()


@router.get("/config")
async def get_config() -> dict[str, Any]:
    load_config()
    return {
        "subtitle_modes": SUBTITLE_MODES,
        "defaults": {
            "subtitle_mode": DEFAULT_JOB_DEFAULTS["subtitle_mode"],
            "skip_translation": DEFAULT_JOB_DEFAULTS["skip_translation"],
            "translation_max_workers": DEFAULT_JOB_DEFAULTS["translation_max_workers"],
        },
    }


@router.get("/model-requirements")
async def get_model_requirements(
    include_cuda: bool = Query(default=True),
) -> dict[str, Any]:
    """Model readiness, plus the CUDA probe unless the caller opts out.

    The readiness notice re-checks itself on a timer while a model is still
    missing (a first run downloads it while the notice is on screen). Those polls
    pass include_cuda=0: the probe spawns a torch-importing child, which is not
    something to repeat every few seconds next to a running job.
    """
    load_config()
    # One ASR model serves transcription, the encoder features, and the
    # alignment head; the optional aligner checkpoint rides on
    # ASR_ALIGNMENT_HEAD_PATH and never blocks a run, so readiness is one model
    # row plus the CUDA probe.
    requirements = [
        _model_requirement(
            repo_id=active_qwen_asr_model_id(),
            explicit_path=_runtime_or_config_or_setting("ASR_MODEL_PATH"),
        ),
    ]
    missing = [item for item in requirements if not item["present"]]
    cuda_status = (
        await _cuda_environment_status_off_the_loop() if include_cuda else None
    )
    gpu_ready = bool(cuda_status.get("ok")) if cuda_status else None
    return {
        "required_models": requirements,
        "cuda": cuda_status,
        "missing_count": len(missing),
        "needs_download": any(item["download_enabled"] for item in missing),
        "download_disabled": any(not item["download_enabled"] for item in missing),
        "all_present": not missing,
        "gpu_ready": gpu_ready,
        "pipeline_ready": (not missing and gpu_ready) if cuda_status else None,
    }


@router.get("/settings", response_model=SettingsRead)
async def get_settings() -> SettingsRead:
    api_key = _runtime_or_config_value("API_KEY")
    base_url = _runtime_or_config_or_setting("OPENAI_COMPATIBILITY_BASE_URL")
    model = _runtime_or_config_or_setting("LLM_MODEL_NAME")
    proxy_protocol, proxy_host, proxy_port = _proxy_settings_from_runtime()
    translation_glossary = _runtime_or_config_or_setting("TRANSLATION_GLOSSARY")
    llm_reasoning_effort = _normalize_llm_reasoning_effort(
        _runtime_or_config_or_setting("LLM_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
    )
    target_lang = _runtime_or_config_or_setting("TARGET_LANG", "简体中文")
    translation_backend = _runtime_or_config_or_setting("TRANSLATION_BACKEND", "llamacpp")
    if translation_backend not in {"openai", "llamacpp"}:
        translation_backend = "llamacpp"
    disable_update_check = (
        _runtime_or_config_or_setting("JAV_SUB_DISABLE_UPDATE_CHECK", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )

    return SettingsRead(
        api_key_set=bool(api_key),
        api_key_preview=_mask_key(api_key),
        base_url=base_url,
        model=model,
        proxy_protocol=proxy_protocol,
        proxy_host=proxy_host,
        proxy_port=proxy_port,
        translation_glossary=translation_glossary,
        llm_reasoning_effort=llm_reasoning_effort,
        target_lang=target_lang,
        translation_backend=translation_backend,
        disable_update_check=disable_update_check,
    )


@router.post("/settings")
async def post_settings(update: SettingsUpdate) -> dict:
    changes: dict[str, str] = {}
    reset_llamacpp_translation_backend = False
    if update.api_key is not None:
        changes["API_KEY"] = update.api_key
    if update.base_url is not None:
        changes["OPENAI_COMPATIBILITY_BASE_URL"] = update.base_url
    if update.model is not None:
        changes["LLM_MODEL_NAME"] = update.model
    proxy_changes = _proxy_settings_changes(update)
    if proxy_changes is not None:
        changes.update(proxy_changes)
    if update.translation_glossary is not None:
        changes["TRANSLATION_GLOSSARY"] = update.translation_glossary
    if update.llm_reasoning_effort is not None:
        if update.llm_reasoning_effort not in REASONING_EFFORTS:
            raise HTTPException(
                status_code=422,
                detail=f"llm_reasoning_effort must be one of: {', '.join(REASONING_EFFORTS)}",
            )
        changes["LLM_REASONING_EFFORT"] = update.llm_reasoning_effort
    if update.target_lang is not None:
        changes["TARGET_LANG"] = update.target_lang
    if update.translation_backend is not None:
        if update.translation_backend not in {"openai", "llamacpp"}:
            raise HTTPException(
                status_code=422,
                detail="translation_backend must be one of: openai, llamacpp",
            )
        previous_backend = _runtime_or_config_or_setting("TRANSLATION_BACKEND", "llamacpp")
        changes["TRANSLATION_BACKEND"] = update.translation_backend
        reset_llamacpp_translation_backend = (
            previous_backend != update.translation_backend
            and update.translation_backend != "llamacpp"
        )
    if update.disable_update_check is not None:
        changes["JAV_SUB_DISABLE_UPDATE_CHECK"] = "1" if update.disable_update_check else "0"
    if changes:
        _clear_saved_hf_mirror_if_present(changes)
        try:
            _update_config_file(changes)
        except OSError as exc:
            raise HTTPException(status_code=503, detail="保存设置失败，请检查程序目录的写入权限和磁盘空间。") from exc
        # Publish only after persistence succeeds. A failed save must leave the
        # running app using the same settings the next launch will read.
        os.environ.update(changes)
        if changes.get("HF_ENDPOINT") == "":
            os.environ.pop("HF_ENDPOINT", None)
        if proxy_changes is not None:
            sync_network_proxy_environment(clear_existing=True)
    if reset_llamacpp_translation_backend:
        from llm.backends import reset_backend

        # Releasing a model may wait for an active generate() call. Keep that
        # wait off the FastAPI event loop so SSE and cancellation stay alive.
        await asyncio.to_thread(reset_backend, "llamacpp")
    result: dict[str, object] = {"ok": True}
    if update.translation_backend == "llamacpp":
        from llm.backends import llamacpp_runtime

        result["llama_runtime_prefetch"] = (
            llamacpp_runtime.start_managed_llama_server_prefetch()
        )
    return result


@router.post("/proxy-test")
async def test_proxy_connection() -> dict:
    """Verify the configured network proxy can actually reach HuggingFace.

    Use the configured proxy explicitly so NO_PROXY cannot turn this check
    into a direct request. Any HTTP response (even 4xx) means the transport
    works -- only a connection-level failure (timeout / refused / bad auth) means the proxy is
    broken. Used by the Web「测试连接」button so a wrong port fails loud instead
    of silently hanging model downloads.
    """
    from core.config import network_proxy_url_from_env

    proxy_url = network_proxy_url_from_env()
    if not proxy_url:
        return {"ok": False, "proxy_url": "", "error": "未启用代理，或地址/端口为空"}
    safe_proxy_url = redact_url_credentials(proxy_url)
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(
            proxy=proxy_url, timeout=8.0, follow_redirects=True
        ) as client:
            resp = await client.get("https://huggingface.co/")
    except ImportError:
        return {
            "ok": False,
            "proxy_url": safe_proxy_url,
            "error": "代理依赖不完整，请更新运行环境（SOCKS5 需要 httpx[socks] / socksio）",
        }
    except (httpx.HTTPError, ValueError) as exc:
        safe_detail = redact_url_credentials(format_exception_chain(exc))
        logging.getLogger(__name__).warning(
            "Proxy test failed for %s: %s", safe_proxy_url, safe_detail
        )
        return {
            "ok": False,
            "proxy_url": safe_proxy_url,
            "error": f"经代理连接失败：{safe_detail}",
        }
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return {
        "ok": True,
        "proxy_url": safe_proxy_url,
        "status_code": resp.status_code,
        "elapsed_ms": elapsed_ms,
    }


@router.get("/models")
async def get_models() -> dict[str, list[str]]:
    api_key = _runtime_or_config_value("API_KEY")
    base_url = _runtime_or_config_value("OPENAI_COMPATIBILITY_BASE_URL")
    if not api_key or not base_url:
        missing = "API Key" if not api_key else "API Base URL"
        raise HTTPException(
            status_code=400,
            detail=f"缺少「{missing}」：请在「翻译设置」中填写并保存后，再点「获取」。",
        )

    errors: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            for url in _model_endpoint_candidates(base_url):
                try:
                    response = await client.get(
                        url,
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                except httpx.HTTPError as exc:
                    errors.append(f"{url}: {format_exception_chain(exc)}")
                    continue

                content_type = response.headers.get("content-type", "")
                if response.status_code >= 400:
                    errors.append(
                        f"{url}: HTTP {response.status_code}"
                        + (f" ({content_type})" if content_type else "")
                    )
                    continue

                try:
                    payload = response.json()
                except json.JSONDecodeError:
                    errors.append(
                        f"{url}: HTTP {response.status_code} returned non-JSON"
                        + (f" ({content_type})" if content_type else "")
                    )
                    continue

                models = _extract_model_ids(payload)
                if models:
                    return {"models": models}
                errors.append(f"{url}: JSON response did not contain data[].id")
    except ImportError as exc:
        logging.getLogger(__name__).warning("HTTP client missing dependency in /api/models: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="代理依赖不完整，请更新运行环境（SOCKS5 需要 httpx[socks] / socksio）",
        ) from exc

    detail = (
        "获取模型列表失败：请确认「API Base URL」填的是服务商的接口地址"
        "（例如 OpenRouter 的 https://openrouter.ai/api/v1，或 DeepSeek 的 "
        "https://api.deepseek.com），以及该 Key 是否可用。"
    )
    if errors:
        detail += "\n尝试过的地址：" + "; ".join(errors)
    raise HTTPException(status_code=502, detail=detail)
