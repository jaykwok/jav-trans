from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, get_args

import httpx
from fastapi import APIRouter, HTTPException, Query

from asr.backends.qwen import qwen_asr_repo_id
from boundary.ja.backend import (
    DEFAULT_MODEL_PATH as DEFAULT_BOUNDARY_JA_MODEL_PATH,
    DEFAULT_PTM as DEFAULT_BOUNDARY_JA_PTM,
)
from core.config import DEFAULT_SETTINGS, load_config
from utils import model_paths
from utils.model_paths import PROJECT_ROOT, normalize_hf_endpoint
from web.models import JobSpec, SettingsRead, SettingsUpdate


router = APIRouter()

BACKENDS = list(get_args(JobSpec.model_fields["asr_backend"].annotation))
RECOMMENDED_ASR_BACKEND = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame"
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


def _update_env_file(updates: dict[str, str]) -> None:
    env_path = PROJECT_ROOT / ".env"
    lines = (
        env_path.read_text(encoding="utf-8").splitlines(keepends=True)
        if env_path.exists()
        else []
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
    env_path.write_text("".join(new_lines), encoding="utf-8")


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


def _sync_hf_endpoint(value: str) -> str:
    try:
        return normalize_hf_endpoint(value) or ""
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _normalize_llm_reasoning_effort(value: str) -> str:
    normalized = (value or "xhigh").strip().lower()
    return normalized if normalized in {"medium", "xhigh"} else "xhigh"


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
    boundary_model_id = (
        _runtime_or_env_or_setting("SPEECH_BOUNDARY_JA_PTM", DEFAULT_BOUNDARY_JA_PTM).strip()
        or DEFAULT_BOUNDARY_JA_PTM
    )
    boundary_model_path = _runtime_or_env_or_setting(
        "SPEECH_BOUNDARY_JA_MODEL_PATH",
        DEFAULT_BOUNDARY_JA_MODEL_PATH,
    )
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
    return {
        "asr_backend": backend,
        "required_models": requirements,
        "missing_count": len(missing),
        "needs_download": any(item["download_enabled"] for item in missing),
        "download_disabled": any(not item["download_enabled"] for item in missing),
        "all_present": not missing,
    }


@router.get("/settings", response_model=SettingsRead)
async def get_settings() -> SettingsRead:
    api_key = _runtime_or_env_value("API_KEY")
    base_url = _runtime_or_env_or_setting("OPENAI_COMPATIBILITY_BASE_URL")
    model = _runtime_or_env_or_setting("LLM_MODEL_NAME")
    hf_endpoint = _runtime_or_env_value("HF_ENDPOINT")
    asr_context = _runtime_or_env_or_setting("ASR_CONTEXT")
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
        hf_endpoint=hf_endpoint,
        asr_context=asr_context,
        translation_glossary=translation_glossary,
        llm_api_format=llm_api_format if llm_api_format in {"chat", "responses"} else "chat",
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
    if update.hf_endpoint is not None:
        changes["HF_ENDPOINT"] = _sync_hf_endpoint(update.hf_endpoint)
    if update.asr_context is not None:
        changes["ASR_CONTEXT"] = update.asr_context
        os.environ["ASR_CONTEXT"] = update.asr_context
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
        _update_env_file(changes)
    return {"ok": True}


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
