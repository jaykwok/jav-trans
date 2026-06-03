from __future__ import annotations

import os


QWEN_ASR_06B_REPO_ID = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame"
QWEN_ASR_17B_REPO_ID = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame"

DEFAULT_QWEN_ASR_BACKEND = QWEN_ASR_06B_REPO_ID
QWEN_ASR_BACKEND_REPOS: dict[str, str] = {
    QWEN_ASR_06B_REPO_ID: QWEN_ASR_06B_REPO_ID,
    QWEN_ASR_17B_REPO_ID: QWEN_ASR_17B_REPO_ID,
}


def current_qwen_asr_backend(default: str = DEFAULT_QWEN_ASR_BACKEND) -> str:
    return (os.getenv("ASR_BACKEND", default) or default).strip() or default


def qwen_asr_repo_id(backend: str | None = None) -> str:
    normalized = (backend or current_qwen_asr_backend()).strip()
    try:
        return QWEN_ASR_BACKEND_REPOS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Qwen ASR backend: {normalized!r}. "
            f"Choose one of: {', '.join(sorted(QWEN_ASR_BACKEND_REPOS))}"
        ) from exc


def active_qwen_asr_model_id() -> str:
    override = os.getenv("ASR_MODEL_ID", "").strip()
    return override or qwen_asr_repo_id(current_qwen_asr_backend())


def active_qwen_asr_model_path() -> str:
    return os.getenv("ASR_MODEL_PATH", "").strip()
