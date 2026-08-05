"""Check the translation configuration before a run can waste time on it.

A forgotten API key used to surface only when the first batch fired - after ASR
had already spent ten minutes on the video, and phrased in the openai SDK's
words. These checks run twice: once when the job is queued or retried (so the
user hears about it before ASR starts) and once in the OpenAI client factory
(which covers a setting cleared while the queue drains, and is the only place
that knows the backend in use is really the openai one).

Missing settings are only checked for the OpenAI-compatible backend: ``llamacpp``
raises its own actionable Chinese messages while resolving a model path, and
duplicating those conditions would let the two copies drift apart. The one thing
checked for ``llamacpp`` is a retired model still pinned in `.env`, because that
config produces garbage output rather than an error.
"""

from __future__ import annotations

import os

from core.stage_errors import MISSING_API_KEY, MISSING_BASE_URL, MISSING_MODEL

# v1.0 shipped a Sakura/GalTransl GGUF as the llamacpp default, and its
# line-oriented prompt contract was removed on 2026-08-04. Such a model cannot
# answer the JSON contract, so a `.env` left over from that release would
# translate a whole video into unparseable replies.
RETIRED_GGUF_TOKENS = ("sakura", "galtransl")
RETIRED_GGUF_MODEL = (
    "配置里的本地 GGUF 模型（Sakura / GalTransl 系）已不再支持："
    "它只认自己的行式模板，无法按当前的 JSON 格式回答。"
    "请在「翻译设置」里清空「自定义 GGUF 文件路径」以改用内置模型，"
    "或换用能按 JSON 回答的 GGUF。"
)


def _missing(name: str) -> bool:
    return not os.getenv(name, "").strip()


def _retired_gguf_model() -> bool:
    haystack = " ".join(
        os.getenv(name, "")
        for name in ("LLAMACPP_GGUF_PATH", "LLAMACPP_MODEL_FILE", "LLAMACPP_MODEL_REPO")
    ).lower()
    return any(token in haystack for token in RETIRED_GGUF_TOKENS)


def translation_config_problems(backend: str | None = None) -> list[str]:
    """Actionable messages for everything the translation stage still needs."""
    from llm.backends import selected_backend_name

    selected = selected_backend_name(backend)
    if selected == "llamacpp":
        return [RETIRED_GGUF_MODEL] if _retired_gguf_model() else []
    if selected != "openai":
        return []

    problems: list[str] = []
    if _missing("API_KEY"):
        problems.append(MISSING_API_KEY)
    if _missing("OPENAI_COMPATIBILITY_BASE_URL"):
        problems.append(MISSING_BASE_URL)
    if _missing("LLM_MODEL_NAME"):
        problems.append(MISSING_MODEL)
    return problems


def require_translation_config(backend: str | None = None) -> None:
    """Raise one RuntimeError listing every missing translation setting."""
    problems = translation_config_problems(backend)
    if problems:
        raise RuntimeError("\n".join(problems))


__all__ = ["require_translation_config", "translation_config_problems"]
