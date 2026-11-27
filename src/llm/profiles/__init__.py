"""Profile registry and backend-scoped selection.

``TRANSLATION_PROMPT_PROFILE`` pins a profile explicitly; the default ``auto``
switches on when the *selected backend's own* model config matches a registered
model family, and otherwise selects the unified ``json`` batch contract.

Both API models (``openai``) and the local Hy-MT3-7B GGUF model (``llamacpp``)
now share the single ``json`` contract (with smaller VRAM-bounded windowed
batches on ``llamacpp``). The registry and backend-scoped detection seam stay
intact so future model families can plug in without touching the engine.
"""

from __future__ import annotations

import os

from llm import request_config
from llm.profiles.base import ProfileContext, TranslationProfile
from llm.profiles.json_v3 import JsonProfile

__all__ = [
    "ProfileContext",
    "TranslationProfile",
    "register_profile",
    "get_profile",
    "list_profiles",
    "select_profile",
]

_REGISTRY: dict[str, TranslationProfile] = {}
# Auto-detection table: profile id -> lowercase substrings that identify the
# model family in the selected backend's model config.
_MATCH_TOKENS: dict[str, tuple[str, ...]] = {}
# Aliases accepted in TRANSLATION_PROMPT_PROFILE.
_PIN_ALIASES: dict[str, str] = {
    "off": "json",
    "none": "json",
}

_DEFAULT_PROFILE_ID = "json"


def register_profile(
    profile: TranslationProfile,
    *,
    match_tokens: tuple[str, ...] = (),
    replace: bool = False,
) -> None:
    if not profile.id:
        raise ValueError("profile.id must be non-empty")
    if profile.id in _REGISTRY and not replace:
        raise ValueError(f"translation profile already registered: {profile.id}")
    _REGISTRY[profile.id] = profile
    if match_tokens:
        _MATCH_TOKENS[profile.id] = tuple(token.lower() for token in match_tokens)
    elif replace:
        _MATCH_TOKENS.pop(profile.id, None)


def get_profile(profile_id: str) -> TranslationProfile:
    try:
        return _REGISTRY[profile_id]
    except KeyError:
        raise KeyError(f"unknown translation profile: {profile_id}") from None


def list_profiles() -> list[str]:
    return sorted(_REGISTRY)


def _env(name: str) -> str:
    return str(os.getenv(name, "") or "").strip()


def _detection_haystack(backend: str, *, model_name: str | None = None) -> str:
    if backend == "llamacpp":
        return " ".join(
            (
                str(request_config.getenv("LLAMACPP_GGUF_PATH", "") or "").strip(),
                str(request_config.getenv("LLAMACPP_MODEL_FILE", "") or "").strip(),
                str(request_config.getenv("LLAMACPP_MODEL_REPO", "") or "").strip(),
            )
        ).lower()
    if backend == "openai":
        # Self-hosted fine-tunes behind an OpenAI-compatible server (vLLM /
        # llama-server started by hand) are detected via the model name.
        return (_env("LLM_MODEL_NAME") if model_name is None else model_name).lower()
    return ""


def select_profile(
    *,
    backend: str | None = None,
    model_name: str | None = None,
) -> TranslationProfile:
    """Which prompt contract this run speaks.

    ``backend`` and ``model_name`` let a caller that has already settled those -
    a task holds a backend lease and a frozen endpoint snapshot - select from
    what it settled rather than from whatever the settings panel says now.
    """
    pinned = _env("TRANSLATION_PROMPT_PROFILE").lower() or "auto"
    if pinned != "auto":
        profile_id = _PIN_ALIASES.get(pinned, pinned)
        if profile_id in _REGISTRY:
            return _REGISTRY[profile_id]
        return _REGISTRY[_DEFAULT_PROFILE_ID]

    backend = (backend or _env("TRANSLATION_BACKEND")).lower() or "llamacpp"
    haystack = _detection_haystack(backend, model_name=model_name)
    if haystack:
        for profile_id, tokens in _MATCH_TOKENS.items():
            if any(token in haystack for token in tokens):
                return _REGISTRY[profile_id]
    return _REGISTRY[_DEFAULT_PROFILE_ID]


register_profile(JsonProfile())
