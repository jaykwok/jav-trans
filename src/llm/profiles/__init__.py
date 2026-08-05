"""Profile registry and backend-scoped selection.

``TRANSLATION_PROMPT_PROFILE`` pins a profile explicitly; the default ``auto``
switches on exactly when the *selected backend's own* model config clearly names
a known model family.

Scoping by backend matters (learned the hard way, back when the built-in
``LLAMACPP_MODEL_FILE`` default was a line-oriented model): reading a GGUF file
name while the openai backend is selected would hijack every default
installation into that model's contract. Detection reads config strings only --
it must never load a backend or spawn a server.

Two contracts ship, split by where the model runs rather than by vendor:

* ``json`` - the batch contract, for API models. It carries the layers that
  make a whole film cohere: up to 64 cues per request, the full-transcript
  prefix, the character table and the glossary.
* ``hymt2`` - one cue per request, bare template, for the local llama.cpp
  default. Not a downgrade by taste: Hy-MT2 measured 6/300 untranslated on the
  bare template against 152/300 on the batch contract, and every context layer
  added made it worse. The full-transcript prefix is also unavailable locally
  regardless, since it does not fit an 8GB card's context budget.

The removed Sakura/GalTransl profile is not coming back, and ``hymt2`` does not
reinstate its defect: that profile wrote ``""`` for a line it could not match
and returned successfully, which is why an untranslated cue could reach the
screen without failing the job. ``hymt2`` raises on an empty reply instead.
"""

from __future__ import annotations

import os

from llm.profiles.base import ProfileContext, TranslationProfile
from llm.profiles.hymt2 import HyMt2Profile
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


def _detection_haystack(backend: str) -> str:
    if backend == "llamacpp":
        return " ".join(
            (
                _env("LLAMACPP_GGUF_PATH"),
                _env("LLAMACPP_MODEL_FILE"),
                _env("LLAMACPP_MODEL_REPO"),
            )
        ).lower()
    if backend == "openai":
        # Self-hosted fine-tunes behind an OpenAI-compatible server (vLLM /
        # llama-server started by hand) are detected via the model name.
        return _env("LLM_MODEL_NAME").lower()
    # The transformers local backend cannot load GGUF releases, and custom
    # backends must opt in via TRANSLATION_PROMPT_PROFILE.
    return ""


def select_profile() -> TranslationProfile:
    pinned = _env("TRANSLATION_PROMPT_PROFILE").lower() or "auto"
    if pinned != "auto":
        profile_id = _PIN_ALIASES.get(pinned, pinned)
        if profile_id in _REGISTRY:
            return _REGISTRY[profile_id]
        return _REGISTRY[_DEFAULT_PROFILE_ID]

    backend = _env("TRANSLATION_BACKEND").lower() or "openai"
    haystack = _detection_haystack(backend)
    if haystack:
        for profile_id, tokens in _MATCH_TOKENS.items():
            if any(token in haystack for token in tokens):
                return _REGISTRY[profile_id]
    return _REGISTRY[_DEFAULT_PROFILE_ID]


register_profile(JsonProfile())
# Detection reads config strings only - it must never load a backend. The
# tokens cover the GGUF file name, the repo id, and a self-hosted model name
# served over an OpenAI-compatible endpoint.
register_profile(
    HyMt2Profile(),
    match_tokens=("hy-mt2", "hymt2", "hunyuan-mt", "hunyuan_mt"),
)
