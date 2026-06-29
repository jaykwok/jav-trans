from __future__ import annotations

import os
from pathlib import Path

from utils.runtime_paths import runtime_path

QWEN_ASR_06B_REPO_ID = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame"
QWEN_ASR_17B_REPO_ID = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame"
QWEN_ASR_REPO_ID = QWEN_ASR_17B_REPO_ID

DEFAULT_QWEN_ASR_BACKEND = QWEN_ASR_REPO_ID
QWEN_ASR_BACKEND_REPOS: dict[str, str] = {
    QWEN_ASR_06B_REPO_ID: QWEN_ASR_06B_REPO_ID,
    QWEN_ASR_REPO_ID: QWEN_ASR_REPO_ID,
}
DEFAULT_QWEN_ASR_BATCH_SIZE_BY_REPO: dict[str, int] = {
    QWEN_ASR_06B_REPO_ID: 64,
    QWEN_ASR_REPO_ID: 32,
}
DEFAULT_BOUNDARY_REFINER_CHECKPOINT_BY_REPO: dict[str, str] = {
    QWEN_ASR_17B_REPO_ID: (
        "src/boundary/checkpoints/"
        "boundary_edge_refiner_v8_safe_tight.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt"
    ),
    QWEN_ASR_06B_REPO_ID: (
        "src/boundary/checkpoints/"
        "boundary_edge_refiner_v8_safe_tight.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame.pt"
    ),
}
DEFAULT_CUEQC_CHECKPOINT_BY_REPO: dict[str, str] = {
    QWEN_ASR_17B_REPO_ID: (
        "src/asr/checkpoints/"
        "cueqc_mamba_v4_binary.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt"
    ),
    QWEN_ASR_06B_REPO_ID: (
        "src/asr/checkpoints/"
        "cueqc_mamba_v4_binary.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame.pt"
    ),
}
DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO: dict[str, str] = {
    QWEN_ASR_17B_REPO_ID: (
        "src/asr/checkpoints/"
        "cueqc_pre_asr_mamba_v10_binary.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt"
    ),
    QWEN_ASR_06B_REPO_ID: (
        "src/asr/checkpoints/"
        "cueqc_pre_asr_mamba_v10_binary.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame.pt"
    ),
}
DEFAULT_SPEECH_BOUNDARY_SCORER_CHECKPOINT_BY_REPO: dict[str, str] = {
    QWEN_ASR_17B_REPO_ID: (
        "src/boundary/ja/checkpoints/"
        "speech_boundary_ja_frame_boundary_scorer_v7.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt"
    ),
    QWEN_ASR_06B_REPO_ID: (
        "src/boundary/ja/checkpoints/"
        "speech_boundary_ja_frame_boundary_scorer_v7.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame.pt"
    ),
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


def qwen_asr_repo_tag(repo_id: str | None = None) -> str:
    normalized = (repo_id or DEFAULT_QWEN_ASR_BACKEND).strip()
    if not normalized:
        normalized = DEFAULT_QWEN_ASR_BACKEND
    if normalized in QWEN_ASR_BACKEND_REPOS:
        normalized = qwen_asr_repo_id(normalized)
    return normalized.replace("/", "-")


def qwen_asr_default_model_path(repo_id: str | None = None) -> str:
    return f"models/{qwen_asr_repo_tag(repo_id)}"


def repo_path_mapping(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "Invalid repo path mapping entry "
                f"{item!r}; expected '<repo_id>=<path>'"
            )
        repo_id, path = item.split("=", 1)
        repo_id = repo_id.strip()
        path = path.strip()
        if not repo_id or not path:
            raise ValueError(
                "Invalid repo path mapping entry "
                f"{item!r}; repo id and path are required"
            )
        mapping[repo_id] = path
    return mapping


def repo_path_mapping_env(mapping: dict[str, str]) -> str:
    return ",".join(f"{repo_id}={path}" for repo_id, path in mapping.items())


def checkpoint_path_for_repo_env(
    *,
    repo_id: str | None,
    mapping_env: str,
    default_mapping: dict[str, str] | None = None,
    required: bool = True,
) -> str:
    selected_repo = qwen_asr_repo_id((repo_id or current_qwen_asr_backend()).strip())
    raw = os.getenv(mapping_env, "").strip()
    if raw.lower() in {"auto", "default"}:
        mapping = dict(default_mapping or {})
    elif raw:
        try:
            mapping = repo_path_mapping(raw)
        except ValueError as exc:
            raise RuntimeError(f"{mapping_env} is malformed: {exc}") from exc
    else:
        mapping = dict(default_mapping or {})
    if not mapping:
        if not required:
            return ""
        raise RuntimeError(
            f"{mapping_env} is required. Set an explicit repo-id mapping like "
            f"{selected_repo}=path/to/checkpoint.pt"
        )
    path = mapping.get(selected_repo, "").strip()
    if not path:
        if not required:
            return ""
        raise RuntimeError(
            f"{mapping_env} has no checkpoint for ASR repo {selected_repo!r}. "
            "Add a '<repo_id>=<checkpoint_path>' entry for the selected ASR backend."
        )
    checkpoint_path = runtime_path(Path(path).expanduser())
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{mapping_env} checkpoint for ASR repo {selected_repo!r} does not exist: "
            f"{path} (resolved: {checkpoint_path})"
        )
    return str(checkpoint_path)


def validate_checkpoint_repo_id(
    actual_repo_id: str | None,
    expected_repo_id: str | None,
    *,
    checkpoint_kind: str,
    metadata_key: str,
) -> str:
    expected = qwen_asr_repo_id((expected_repo_id or current_qwen_asr_backend()).strip())
    actual = str(actual_repo_id or "").strip()
    if not actual:
        raise ValueError(
            f"{checkpoint_kind} checkpoint missing {metadata_key}; "
            "repo-id binding cannot be verified"
        )
    if actual in QWEN_ASR_BACKEND_REPOS:
        actual = qwen_asr_repo_id(actual)
    if actual != expected:
        raise ValueError(
            f"{checkpoint_kind} checkpoint {metadata_key}={actual!r} does not match "
            f"selected repo {expected!r}"
        )
    return actual


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
