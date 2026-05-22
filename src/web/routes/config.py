from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, get_args

import httpx
from fastapi import APIRouter, HTTPException

from core.config import DEFAULT_SETTINGS, load_config
from utils.model_paths import PROJECT_ROOT, normalize_hf_endpoint
from web.models import JobSpec, SettingsRead, SettingsUpdate


router = APIRouter()

BACKENDS = list(get_args(JobSpec.model_fields["asr_backend"].annotation))
RECOMMENDED_ASR_BACKEND = "whisper-ja-anime-v0.3"
SUBTITLE_MODES = list(get_args(JobSpec.model_fields["subtitle_mode"].annotation))
DEFAULT_JOB_DEFAULTS = {
    name: field.default
    for name, field in JobSpec.model_fields.items()
    if not field.is_required()
}
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
            "multi_cue_split": DEFAULT_JOB_DEFAULTS["multi_cue_split"],
            "vad_threshold": DEFAULT_JOB_DEFAULTS["vad_threshold"],
            "translation_max_workers": DEFAULT_JOB_DEFAULTS["translation_max_workers"],
        },
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
