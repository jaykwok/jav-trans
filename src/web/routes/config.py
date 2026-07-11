from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, get_args

import httpx
from fastapi import APIRouter, HTTPException, Query

from asr.backends.qwen import (
    DEFAULT_CUT_EDGE_REFINER_CHECKPOINT_BY_REPO,
    DEFAULT_OUTER_EDGE_REFINER_CHECKPOINT_BY_REPO,
    DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO,
    DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO,
    DEFAULT_SPEECH_BOUNDARY_SCORER_CHECKPOINT_BY_REPO,
    checkpoint_path_for_repo_env,
    qwen_asr_default_model_path,
    qwen_asr_repo_id,
)
from core.config import (
    DEFAULT_SETTINGS,
    apply_network_proxy_environment,
    load_config,
)
from utils import model_paths
from utils.model_paths import PROJECT_ROOT
from utils.runtime_paths import is_frozen
from utils.subprocess_tools import no_window_subprocess_kwargs
from web.models import (
    JobSpec,
    SettingsRead,
    SettingsUpdate,
    normalize_llm_api_format as _normalize_llm_api_format,
    normalize_llm_reasoning_effort as _normalize_llm_reasoning_effort,
)


router = APIRouter()

BACKENDS = list(get_args(JobSpec.model_fields["asr_backend"].annotation))
RECOMMENDED_ASR_BACKEND = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
SUBTITLE_MODES = list(get_args(JobSpec.model_fields["subtitle_mode"].annotation))
DEFAULT_JOB_DEFAULTS = {
    name: field.default
    for name, field in JobSpec.model_fields.items()
    if not field.is_required()
}
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MODEL_ROLE_LABELS = {
    "asr": "ASR",
    "boundary_feature": "Boundary",
}
_CHECKPOINT_SPECS = (
    (
        "speech_island_scorer",
        "SpeechIslandScorer",
        "SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO",
        DEFAULT_SPEECH_BOUNDARY_SCORER_CHECKPOINT_BY_REPO,
    ),
    (
        "outer_edge_refiner",
        "Outer Edge Refiner",
        "OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO",
        DEFAULT_OUTER_EDGE_REFINER_CHECKPOINT_BY_REPO,
    ),
    (
        "semantic_split_model",
        "Semantic Split Model",
        "SEMANTIC_SPLIT_MODEL_PATH_BY_REPO",
        DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO,
    ),
    (
        "cut_edge_refiner",
        "Cut Edge Refiner",
        "CUT_EDGE_REFINER_MODEL_PATH_BY_REPO",
        DEFAULT_CUT_EDGE_REFINER_CHECKPOINT_BY_REPO,
    ),
    (
        "pre_asr_cueqc",
        "Pre-ASR CueQC",
        "PRE_ASR_CUEQC_MODEL_PATH_BY_REPO",
        DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO,
    ),
)


def _format_env_line(key: str, value: str) -> str:
    if not _ENV_KEY_RE.fullmatch(key):
        raise ValueError(f"Invalid .env key: {key!r}")
    return f"{key}={json.dumps(str(value), ensure_ascii=False)}\n"


def _setting(key: str) -> str:
    return os.getenv(key, DEFAULT_SETTINGS.get(key, ""))


def _read_env_entry(key: str) -> tuple[bool, str]:
    from dotenv import dotenv_values
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return False, ""
    values = dotenv_values(str(env_path))
    if key in values:
        return True, values[key] or ""
    return False, ""


def _read_env_value(key: str) -> str:
    found, value = _read_env_entry(key)
    return value if found else ""


def _runtime_or_env_value(key: str) -> str:
    if key in os.environ:
        return os.environ.get(key, "").strip()
    return _read_env_value(key)


def _runtime_or_env_or_setting(key: str, fallback: str = "") -> str:
    if key in os.environ:
        return os.environ.get(key, "").strip()
    found, value = _read_env_entry(key)
    if found:
        return value
    return _setting(key) or fallback


_ENV_FILE_LOCK = threading.Lock()


def _initial_env_template_lines() -> list[str]:
    return [
        "# jav-trans local overrides.\n",
        "# Defaults live in src/core/config.py. Keep these examples commented unless\n",
        "# you want to override the built-in runtime defaults on this machine.\n",
        "\n",
        "# --- ASR / VRAM tuning examples ---\n",
        "# ASR_BACKEND=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf\n",
        "# ASR_BACKEND=jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf\n",
        "# ASR_BATCH_SIZE=auto\n",
        "# ASR_BATCH_SIZE_BY_REPO=jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf=12,jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf=4\n",
        "# ASR_STAGE_WORKER_VRAM_BUDGET_MB=auto\n",
        "# ASR_STAGE_WORKER_VRAM_RATIO=0.95\n",
        "# GPU_BATCH_PROFILE_ENABLED=1\n",
        "# GPU_BATCH_PROFILE_GROWTH_THRESHOLD=0.80\n",
        "# SEMANTIC_SPLIT_INFERENCE_BATCH_SIZE=auto\n",
        "# ASR_STAGE_WORKER_OOM_RETRY_LIMIT=6\n",
        "# SPEECH_BOUNDARY_JA_WINDOW_S=20\n",
        "# SPEECH_BOUNDARY_JA_OVERLAP_S=4\n",
        "# PRE_ASR_CUEQC_ENABLED=1\n",
        "# PRE_ASR_CUEQC_DROP_THRESHOLD=0.625\n",
        "\n",
        "# --- Model/cache examples ---\n",
        "# HF_HOME=./models\n",
        "# TORCH_HOME=./tmp/cache/torch\n",
        "# PROXY_PROTOCOL=http\n",
        "# PROXY_HOST=127.0.0.1\n",
        "# PROXY_PORT=7890\n",
        "# RUN_LOG_ENABLED=1\n",
        "# RUN_LOG_DIR=./tmp/log\n",
        "\n",
        "# --- Saved Web settings ---\n",
    ]


def _update_env_file(updates: dict[str, str]) -> None:
    env_path = PROJECT_ROOT / ".env"
    # Serialize concurrent settings saves so they don't clobber each other's
    # keys, and write atomically (tmp + replace) so a crash mid-write cannot
    # leave a truncated .env.
    with _ENV_FILE_LOCK:
        lines = (
            env_path.read_text(encoding="utf-8").splitlines(keepends=True)
            if env_path.exists()
            else _initial_env_template_lines()
        )
        pending = dict(updates)
        new_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if "=" in stripped and not stripped.startswith("#"):
                k = stripped.split("=", 1)[0].strip()
                if k in pending:
                    new_lines.append(_format_env_line(k, pending.pop(k)))
                    continue
            new_lines.append(line if line.endswith("\n") else line + "\n")
        for k, v in pending.items():
            new_lines.append(_format_env_line(k, v))
        tmp_path = env_path.parent / f"{env_path.name}.tmp"
        tmp_path.write_text("".join(new_lines), encoding="utf-8")
        os.replace(tmp_path, env_path)


def _mask_key(k: str) -> str:
    if not k:
        return ""
    if len(k) <= 8:
        return "***"
    return k[:4] + "****..." + k[-4:]


def _ordered_backends(backends: list[str]) -> list[str]:
    return sorted(
        backends,
        key=lambda item: (item != RECOMMENDED_ASR_BACKEND, BACKENDS.index(item)),
    )


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _short_model_name(repo_id: str) -> str:
    return repo_id.strip("/").rsplit("/", 1)[-1] or repo_id


def _model_requirement(
    *,
    role: str,
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
        "roles": [role],
        "role_labels": [_MODEL_ROLE_LABELS.get(role, role)],
        "repo_id": repo_id,
        "short_name": _short_model_name(repo_id),
        "local_path": status["path"],
        "checked_paths": status["checked_paths"],
        "present": bool(status["present"]),
        "download_enabled": bool(download_enabled),
    }


def _merge_model_requirements(requirements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for requirement in requirements:
        key = (str(requirement["repo_id"]), str(requirement["local_path"]))
        existing = by_key.get(key)
        if existing is None:
            clone = dict(requirement)
            clone["roles"] = list(requirement.get("roles") or [])
            clone["role_labels"] = list(requirement.get("role_labels") or [])
            clone["checked_paths"] = list(requirement.get("checked_paths") or [])
            by_key[key] = clone
            merged.append(clone)
            continue

        for role in requirement.get("roles") or []:
            if role not in existing["roles"]:
                existing["roles"].append(role)
        for label in requirement.get("role_labels") or []:
            if label not in existing["role_labels"]:
                existing["role_labels"].append(label)
        for path in requirement.get("checked_paths") or []:
            if path not in existing["checked_paths"]:
                existing["checked_paths"].append(path)
        existing["present"] = bool(existing["present"] or requirement.get("present"))
        existing["download_enabled"] = bool(
            existing["download_enabled"] or requirement.get("download_enabled")
        )
    return merged


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


@lru_cache(maxsize=1)
def _cuda_environment_status() -> dict[str, Any]:
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
            fd, path_text = tempfile.mkstemp(prefix="jav-trans-cuda-probe-", suffix=".json")
        os.close(fd)
        probe_output_path = Path(path_text)
        env["JAV_TRANS_CUDA_PROBE_OUTPUT"] = str(probe_output_path)
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
        return {
            "status": "error",
            "ok": False,
            "code": "probe_invalid_output",
            "message": f"CUDA 环境检测没有返回有效结果。{detail}",
        }
    payload.setdefault("ok", payload.get("status") == "ok")
    payload.setdefault("status", "ok" if payload.get("ok") else "error")
    if completed.returncode != 0 and payload.get("ok"):
        payload["status"] = "warning"
        payload["ok"] = False
    if completed.stderr.strip():
        payload["stderr"] = completed.stderr.strip()
    return payload


def _checkpoint_requirements(repo_id: str) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    for role, label, mapping_env, default_mapping in _CHECKPOINT_SPECS:
        path = ""
        error = ""
        try:
            path = checkpoint_path_for_repo_env(
                repo_id=repo_id,
                mapping_env=mapping_env,
                default_mapping=default_mapping,
                required=False,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            error = str(exc)
        requirements.append(
            {
                "roles": [role],
                "role_labels": [label],
                "repo_id": repo_id,
                "short_name": label,
                "local_path": path,
                "present": bool(path),
                "download_enabled": False,
                "mapping_env": mapping_env,
                "error": error,
            }
        )
    return requirements


def _proxy_settings_from_runtime() -> tuple[str, str, int | None]:
    protocol = _runtime_or_env_or_setting("PROXY_PROTOCOL", "http").lower()
    if protocol not in {"http", "https", "socks5"}:
        protocol = "http"
    host = _runtime_or_env_or_setting("PROXY_HOST", "")
    port_text = _runtime_or_env_or_setting("PROXY_PORT", "")
    try:
        port = int(port_text) if str(port_text).strip() else None
    except (TypeError, ValueError):
        port = None
    if port is not None and not (1 <= port <= 65535):
        port = None
    return protocol, host, port


def _sync_proxy_settings(update: SettingsUpdate) -> dict[str, str] | None:
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
        os.environ["PROXY_PROTOCOL"] = protocol
        os.environ["PROXY_HOST"] = ""
        os.environ["PROXY_PORT"] = ""
        apply_network_proxy_environment("", clear_existing=True)
        return {
            "PROXY_PROTOCOL": protocol,
            "PROXY_HOST": "",
            "PROXY_PORT": "",
        }

    os.environ["PROXY_PROTOCOL"] = protocol
    os.environ["PROXY_HOST"] = host
    os.environ["PROXY_PORT"] = str(port)
    proxy_url = f"{protocol}://{host}:{port}"
    apply_network_proxy_environment(proxy_url, clear_existing=True)
    return {
        "PROXY_PROTOCOL": protocol,
        "PROXY_HOST": host,
        "PROXY_PORT": str(port),
    }


def _clear_saved_hf_mirror_if_present(changes: dict[str, str]) -> None:
    endpoint = _runtime_or_env_or_setting("HF_ENDPOINT", "").strip().rstrip("/")
    if endpoint == "https://hf-mirror.com":
        changes["HF_ENDPOINT"] = ""
        os.environ.pop("HF_ENDPOINT", None)


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


def _extract_model_ids(payload: Any) -> list[str]:
    data = payload.get("data", []) if isinstance(payload, dict) else []
    return sorted(
        str(item["id"])
        for item in data
        if isinstance(item, dict) and item.get("id")
    )


@router.get("/config")
async def get_config() -> dict[str, Any]:
    load_config()
    return {
        "backends": _ordered_backends(BACKENDS),
        "subtitle_modes": SUBTITLE_MODES,
        "engine_defaults": {
            "asr_backend": DEFAULT_SETTINGS.get("ASR_BACKEND", ""),
        },
        "recommended_asr_backend": RECOMMENDED_ASR_BACKEND,
        "defaults": {
            "asr_backend": DEFAULT_JOB_DEFAULTS["asr_backend"],
            "subtitle_mode": DEFAULT_JOB_DEFAULTS["subtitle_mode"],
            "skip_translation": DEFAULT_JOB_DEFAULTS["skip_translation"],
            "translation_max_workers": DEFAULT_JOB_DEFAULTS["translation_max_workers"],
        },
    }


@router.get("/model-requirements")
async def get_model_requirements(
    asr_backend: str | None = Query(default=None),
) -> dict[str, Any]:
    load_config()
    backend = (asr_backend or DEFAULT_JOB_DEFAULTS["asr_backend"]).strip()
    try:
        selected_asr_repo = qwen_asr_repo_id(backend)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    asr_model_id = _runtime_or_env_or_setting("ASR_MODEL_ID").strip() or selected_asr_repo
    asr_model_path = _runtime_or_env_or_setting("ASR_MODEL_PATH")
    boundary_model_id = selected_asr_repo
    boundary_model_path = _runtime_or_env_or_setting("SPEECH_BOUNDARY_JA_MODEL_PATH", "")
    if not boundary_model_path:
        boundary_model_path = qwen_asr_default_model_path(selected_asr_repo)
    boundary_download_enabled = not _truthy(
        _runtime_or_env_or_setting("SPEECH_BOUNDARY_JA_NO_DOWNLOAD", "0")
    )
    requirements = [
        _model_requirement(
            role="asr",
            repo_id=asr_model_id,
            explicit_path=asr_model_path,
        ),
        _model_requirement(
            role="boundary_feature",
            repo_id=boundary_model_id,
            explicit_path=boundary_model_path,
            download_enabled=boundary_download_enabled,
        ),
    ]
    requirements = _merge_model_requirements(requirements)
    missing = [item for item in requirements if not item["present"]]
    checkpoint_requirements = _checkpoint_requirements(selected_asr_repo)
    missing_checkpoints = [
        item for item in checkpoint_requirements if not item["present"]
    ]
    cuda_status = _cuda_environment_status()
    return {
        "asr_backend": backend,
        "required_models": requirements,
        "required_checkpoints": checkpoint_requirements,
        "cuda": cuda_status,
        "missing_count": len(missing),
        "checkpoint_missing_count": len(missing_checkpoints),
        "needs_download": any(item["download_enabled"] for item in missing),
        "download_disabled": any(not item["download_enabled"] for item in missing),
        "all_present": not missing,
        "gpu_ready": bool(cuda_status.get("ok")),
        "pipeline_ready": not missing and not missing_checkpoints and bool(cuda_status.get("ok")),
    }


@router.get("/settings", response_model=SettingsRead)
async def get_settings() -> SettingsRead:
    api_key = _runtime_or_env_value("API_KEY")
    base_url = _runtime_or_env_or_setting("OPENAI_COMPATIBILITY_BASE_URL")
    model = _runtime_or_env_or_setting("LLM_MODEL_NAME")
    proxy_protocol, proxy_host, proxy_port = _proxy_settings_from_runtime()
    translation_glossary = _runtime_or_env_or_setting("TRANSLATION_GLOSSARY")
    llm_api_format = _runtime_or_env_or_setting("LLM_API_FORMAT", "chat")
    llm_reasoning_effort = _normalize_llm_reasoning_effort(
        _runtime_or_env_or_setting("LLM_REASONING_EFFORT", "xhigh")
    )
    target_lang = _runtime_or_env_or_setting("TARGET_LANG", "简体中文")
    return SettingsRead(
        api_key_set=bool(api_key),
        api_key_preview=_mask_key(api_key),
        base_url=base_url,
        model=model,
        proxy_protocol=proxy_protocol,
        proxy_host=proxy_host,
        proxy_port=proxy_port,
        translation_glossary=translation_glossary,
        llm_api_format=_normalize_llm_api_format(llm_api_format),
        llm_reasoning_effort=llm_reasoning_effort,
        target_lang=target_lang,
    )


@router.post("/settings")
async def post_settings(update: SettingsUpdate) -> dict:
    changes: dict[str, str] = {}
    if update.api_key is not None:
        changes["API_KEY"] = update.api_key
        os.environ["API_KEY"] = update.api_key
    if update.base_url is not None:
        changes["OPENAI_COMPATIBILITY_BASE_URL"] = update.base_url
        os.environ["OPENAI_COMPATIBILITY_BASE_URL"] = update.base_url
    if update.model is not None:
        changes["LLM_MODEL_NAME"] = update.model
        os.environ["LLM_MODEL_NAME"] = update.model
    proxy_changes = _sync_proxy_settings(update)
    if proxy_changes is not None:
        changes.update(proxy_changes)
    if update.translation_glossary is not None:
        changes["TRANSLATION_GLOSSARY"] = update.translation_glossary
        os.environ["TRANSLATION_GLOSSARY"] = update.translation_glossary
    if update.llm_api_format is not None:
        changes["LLM_API_FORMAT"] = update.llm_api_format
        os.environ["LLM_API_FORMAT"] = update.llm_api_format
    if update.llm_reasoning_effort is not None:
        if update.llm_reasoning_effort not in {"medium", "xhigh"}:
            raise HTTPException(
                status_code=422,
                detail="llm_reasoning_effort must be one of: medium, xhigh",
            )
        changes["LLM_REASONING_EFFORT"] = update.llm_reasoning_effort
        os.environ["LLM_REASONING_EFFORT"] = update.llm_reasoning_effort
    if update.target_lang is not None:
        changes["TARGET_LANG"] = update.target_lang
        os.environ["TARGET_LANG"] = update.target_lang
    if changes:
        _clear_saved_hf_mirror_if_present(changes)
        _update_env_file(changes)
    return {"ok": True}


@router.post("/proxy-test")
async def test_proxy_connection() -> dict:
    """Verify the configured network proxy can actually reach HuggingFace.

    httpx trusts HTTP(S)_PROXY/ALL_PROXY env by default; the proxy-settings
    sync writes those into os.environ, so a plain client.get() exercises the
    proxy. Any HTTP response (even 4xx) means the proxy transport works -- only
    a connection-level failure (timeout / refused / bad auth) means the proxy is
    broken. Used by the Web「测试连接」button so a wrong port fails loud instead
    of silently hanging model downloads.
    """
    from core.config import network_proxy_url_from_env

    proxy_url = network_proxy_url_from_env()
    if not proxy_url:
        return {"ok": False, "proxy_url": "", "error": "未启用代理，或地址/端口为空"}
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.get("https://huggingface.co/")
    except httpx.HTTPError as exc:
        return {"ok": False, "proxy_url": proxy_url, "error": f"经代理连接失败：{exc}"}
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return {
        "ok": True,
        "proxy_url": proxy_url,
        "status_code": resp.status_code,
        "elapsed_ms": elapsed_ms,
    }


@router.get("/models")
async def get_models() -> dict[str, list[str]]:
    api_key = _runtime_or_env_value("API_KEY")
    base_url = _runtime_or_env_value("OPENAI_COMPATIBILITY_BASE_URL")
    if not api_key or not base_url:
        raise HTTPException(
            status_code=400,
            detail="API_KEY and OPENAI_COMPATIBILITY_BASE_URL are required",
        )

    errors: list[str] = []
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        for url in _model_endpoint_candidates(base_url):
            try:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            except httpx.HTTPError as exc:
                errors.append(f"{url}: {exc}")
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

    detail = "Failed to fetch models"
    if errors:
        detail += ": " + "; ".join(errors)
    raise HTTPException(status_code=502, detail=detail)
