from __future__ import annotations

import os


QWEN_ASR_06B_REPO_ID = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame"
QWEN_ASR_17B_REPO_ID = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame"

DEFAULT_QWEN_ASR_BACKEND = QWEN_ASR_06B_REPO_ID
QWEN_ASR_BACKEND_REPOS: dict[str, str] = {
    QWEN_ASR_06B_REPO_ID: QWEN_ASR_06B_REPO_ID,
    QWEN_ASR_17B_REPO_ID: QWEN_ASR_17B_REPO_ID,
}
DEFAULT_QWEN_ASR_BATCH_SIZE_BY_REPO: dict[str, int] = {
    QWEN_ASR_06B_REPO_ID: 48,
    QWEN_ASR_17B_REPO_ID: 12,
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


def qwen_asr_batch_size_by_repo() -> dict[str, int]:
    raw = os.getenv("ASR_BATCH_SIZE_BY_REPO", "").strip()
    if not raw:
        return dict(DEFAULT_QWEN_ASR_BATCH_SIZE_BY_REPO)
    mapping = dict(DEFAULT_QWEN_ASR_BATCH_SIZE_BY_REPO)
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "Invalid ASR_BATCH_SIZE_BY_REPO entry "
                f"{item!r}; expected '<repo_id>=<positive_int>'"
            )
        repo_id, value = item.rsplit("=", 1)
        repo_id = repo_id.strip()
        if repo_id not in QWEN_ASR_BACKEND_REPOS:
            raise ValueError(
                f"Invalid ASR_BATCH_SIZE_BY_REPO repo {repo_id!r}; "
                f"expected one of {sorted(QWEN_ASR_BACKEND_REPOS)}"
            )
        mapping[repo_id] = max(1, int(value.strip()))
    return mapping


def qwen_asr_default_batch_size(backend: str | None = None) -> int:
    repo_id = qwen_asr_repo_id(backend)
    return qwen_asr_batch_size_by_repo()[repo_id]
