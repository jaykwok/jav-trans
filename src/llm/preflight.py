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


def translation_budget_warnings() -> list[str]:
    """Where the `max_tokens` fallback would silently clamp every batch.

    Warnings, never errors: the numbers involved are all legal, and a run that
    is merely sending less room than it asked for should not be blocked from
    starting. But it should not be silent either. The ceiling is a fallback
    nobody tuned, so it shrinking a computed budget is an accident rather than
    a decision - the request goes out short, and if the reply is then cut off
    the truncation escalation cannot raise it back past the same line.

    Measured against a batch of *empty* cues, so what it compares is the part of
    the budget that comes from configuration alone (per-item structure plus the
    reasoning allowance). Whatever the source text adds is on top, which makes
    this a floor: if it already exceeds the ceiling, every real batch does. The
    converse does not hold, which is why the per-request warning in
    `translator._max_tokens_budget` exists as well - this one is early, that one
    is exact.

    Compared against the budget the endpoint would actually get, not against the
    configured fallback: once an endpoint has named a ceiling, the fallback is
    not what binds, and warning about it would be describing a number nothing
    uses.
    """
    from llm import profiles as profiles_module
    from llm import settings as llm_settings
    from llm import translator as translator_module

    batch_size = max(1, int(llm_settings.TRANSLATION_BATCH_SIZE))
    floor = profiles_module.select_profile().response_token_budget(
        [{"text": ""} for _ in range(batch_size)],
        reasoning_effort=llm_settings.LLM_REASONING_EFFORT,
    )
    if floor is None:
        return []
    floor = int(floor)
    # The pure one: the warning-emitting variant would print here at startup and
    # then mark the endpoint as already-warned, so the run-time clamp - the one
    # that sees the real source text - would never say anything.
    effective = translator_module._plain_max_tokens_budget(floor)
    if effective >= floor:
        return []
    return [
        f"翻译预算下限 {floor} tokens 超过本端点实际可用的 {effective}，"
        "每个批次都会被压到这个上限，且回复被切断后无法再向上重试。"
        f"（batch={batch_size}，推理档={llm_settings.LLM_REASONING_EFFORT}，"
        f"推理配额={llm_settings.TRANSLATION_REASONING_TOKEN_ALLOWANCE}，"
        f"TRANSLATION_MAX_TOKENS={int(translator_module.TRANSLATION_MAX_TOKENS)}）"
        "调高 TRANSLATION_MAX_TOKENS，或调小批次/推理配额。"
    ]


__all__ = [
    "require_translation_config",
    "translation_budget_warnings",
    "translation_config_problems",
]
