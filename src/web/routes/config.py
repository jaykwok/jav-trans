from __future__ import annotations

import os
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
DEFAULT_JOB_SPEC = JobSpec(video_paths=[])


def _setting(key: str) -> str:
    return os.getenv(key, DEFAULT_SETTINGS.get(key, ""))


def _read_env_entry(key: str) -> tuple[bool, str]:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return False, ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        if k.strip() == key:
            return True, v.strip()
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
                new_lines.append(f"{k}={pending.pop(k)}\n")
                continue
        new_lines.append(line if line.endswith("\n") else line + "\n")
    for k, v in pending.items():
        new_lines.append(f"{k}={v}\n")
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


@router.get("/config")
async def get_config() -> dict[str, Any]:
    load_config()
    return {
        "backends": _ordered_backends(BACKENDS),
        "subtitle_modes": SUBTITLE_MODES,
        "defaults": {
            "asr_backend": DEFAULT_JOB_SPEC.asr_backend,
            "subtitle_mode": DEFAULT_JOB_SPEC.subtitle_mode,
            "skip_translation": DEFAULT_JOB_SPEC.skip_translation,
            "show_gender": DEFAULT_JOB_SPEC.show_gender,
            "multi_cue_split": DEFAULT_JOB_SPEC.multi_cue_split,
            "asr_recovery": DEFAULT_JOB_SPEC.asr_recovery,
            "vad_threshold": DEFAULT_JOB_SPEC.vad_threshold,
            "translation_batch_size": DEFAULT_JOB_SPEC.translation_batch_size,
            "translation_max_workers": DEFAULT_JOB_SPEC.translation_max_workers,
        },
    }


@router.get("/settings", response_model=SettingsRead)
async def get_settings() -> SettingsRead:
    api_key = _runtime_or_env_value("API_KEY")
    base_url = _runtime_or_env_or_setting("OPENAI_COMPATIBILITY_BASE_URL")
    model = _runtime_or_env_or_setting("LLM_MODEL_NAME")
    hf_endpoint = _runtime_or_env_value("HF_ENDPOINT")
    translation_glossary = _runtime_or_env_or_setting("TRANSLATION_GLOSSARY")
    llm_api_format = _runtime_or_env_or_setting("LLM_API_FORMAT", "chat")
    llm_reasoning_effort = _runtime_or_env_or_setting("LLM_REASONING_EFFORT", "max")
    target_lang = _runtime_or_env_or_setting("TARGET_LANG", "简体中文")
    return SettingsRead(
        api_key_set=bool(api_key),
        api_key_preview=_mask_key(api_key),
        base_url=base_url,
        model=model,
        hf_endpoint=hf_endpoint,
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
    if update.translation_glossary is not None:
        changes["TRANSLATION_GLOSSARY"] = update.translation_glossary
        os.environ["TRANSLATION_GLOSSARY"] = update.translation_glossary
    if update.llm_api_format is not None:
        changes["LLM_API_FORMAT"] = update.llm_api_format
        os.environ["LLM_API_FORMAT"] = update.llm_api_format
    if update.llm_reasoning_effort is not None:
        if update.llm_reasoning_effort not in {"low", "medium", "max"}:
            raise HTTPException(
                status_code=422,
                detail="llm_reasoning_effort must be one of: low, medium, max",
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
    base_url = _runtime_or_env_value("OPENAI_COMPATIBILITY_BASE_URL").rstrip("/")
    if not api_key or not base_url:
        raise HTTPException(
            status_code=400,
            detail="API_KEY and OPENAI_COMPATIBILITY_BASE_URL are required",
        )

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        data = payload.get("data", [])
        models = sorted(
            str(item["id"])
            for item in data
            if isinstance(item, dict) and item.get("id")
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Invalid models response: {exc}") from exc
    return {"models": models}
