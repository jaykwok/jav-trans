from __future__ import annotations

import os
from pathlib import Path

from utils.runtime_paths import resource_path, runtime_path

QWEN_ASR_06B_REPO_ID = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf"
QWEN_ASR_17B_REPO_ID = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
QWEN_ASR_REPO_ID = QWEN_ASR_17B_REPO_ID

DEFAULT_QWEN_ASR_BACKEND = QWEN_ASR_REPO_ID
QWEN_ASR_BACKEND_REPOS: dict[str, str] = {
    QWEN_ASR_06B_REPO_ID: QWEN_ASR_06B_REPO_ID,
    QWEN_ASR_17B_REPO_ID: QWEN_ASR_17B_REPO_ID,
}
SMALL_MODEL_CHECKPOINT_ROOT = Path("src/checkpoints")


def repo_checkpoint_path(repo_id: str, model_name: str, version: str) -> str:
    repo_tag = repo_id.replace("/", "-")
    filename = f"{model_name}_{version}.{repo_tag}.pt"
    return (SMALL_MODEL_CHECKPOINT_ROOT / repo_tag / filename).as_posix()


DEFAULT_QWEN_ASR_BATCH_SIZE_BY_REPO: dict[str, int] = {
    QWEN_ASR_06B_REPO_ID: 12,
    QWEN_ASR_17B_REPO_ID: 4,
}
DEFAULT_QWEN_ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO: dict[str, int] = {
    QWEN_ASR_06B_REPO_ID: 4096,
    QWEN_ASR_17B_REPO_ID: 6144,
}
DEFAULT_OUTER_EDGE_REFINER_CHECKPOINT_BY_REPO: dict[str, str] = {
    repo_id: repo_checkpoint_path(repo_id, "outer_edge_refiner", "v1")
    for repo_id in QWEN_ASR_BACKEND_REPOS
}
DEFAULT_SEMANTIC_SPLIT_CHECKPOINT_BY_REPO: dict[str, str] = {
    repo_id: repo_checkpoint_path(repo_id, "semantic_split_model", "v2")
    for repo_id in QWEN_ASR_BACKEND_REPOS
}
DEFAULT_CUT_EDGE_REFINER_CHECKPOINT_BY_REPO: dict[str, str] = {
    repo_id: repo_checkpoint_path(repo_id, "cut_edge_refiner", "v1")
    for repo_id in QWEN_ASR_BACKEND_REPOS
}
DEFAULT_PRE_ASR_CUEQC_CHECKPOINT_BY_REPO: dict[str, str] = {
    repo_id: repo_checkpoint_path(repo_id, "pre_asr_cueqc", "v12")
    for repo_id in QWEN_ASR_BACKEND_REPOS
}
# The v8 registry contains only native speech-only checkpoints. Incompatible
# v7 speech/split weights are not converted at load time.
DEFAULT_SPEECH_BOUNDARY_SCORER_CHECKPOINT_BY_REPO: dict[str, str] = {
    repo_id: repo_checkpoint_path(repo_id, "speech_island_scorer", "v8")
    for repo_id in QWEN_ASR_BACKEND_REPOS
}
# Learned boundary-proposal candidate source. Both active Split v2 chains are
# bound to a repo-specific proposer; bootstrap candidates are not accepted.
DEFAULT_SPEECH_BOUNDARY_PROPOSAL_CHECKPOINT_BY_REPO: dict[str, str] = {
    repo_id: repo_checkpoint_path(repo_id, "boundary_proposal_scorer", "v1")
    for repo_id in QWEN_ASR_BACKEND_REPOS
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
    expanded_path = Path(path).expanduser()
    candidate_paths = [runtime_path(expanded_path)]
    if not expanded_path.is_absolute():
        resource_candidate = resource_path(expanded_path)
        if resource_candidate != candidate_paths[0]:
            candidate_paths.append(resource_candidate)
    for checkpoint_path in candidate_paths:
        if checkpoint_path.exists() and checkpoint_path.is_file():
            return str(checkpoint_path)
    checked = ", ".join(str(candidate) for candidate in candidate_paths)
    if not checked:
        checked = str(runtime_path(expanded_path))
    raise FileNotFoundError(
        f"{mapping_env} checkpoint for ASR repo {selected_repo!r} does not exist: "
        f"{path} (checked: {checked})"
    )


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


def qwen_asr_min_physical_vram_mb_by_repo() -> dict[str, int]:
    raw = os.getenv("ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO", "").strip()
    if not raw:
        return dict(DEFAULT_QWEN_ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO)
    mapping = dict(DEFAULT_QWEN_ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO)
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            raise ValueError(
                "Invalid ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO entry "
                f"{item!r}; expected '<repo_id>=<positive_int>'"
            )
        repo_id, value = item.rsplit("=", 1)
        repo_id = repo_id.strip()
        if repo_id not in QWEN_ASR_BACKEND_REPOS:
            raise ValueError(
                f"Invalid ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO repo {repo_id!r}; "
                f"expected one of {sorted(QWEN_ASR_BACKEND_REPOS)}"
            )
        mapping[repo_id] = max(1, int(value.strip()))
    return mapping


def qwen_asr_min_physical_vram_mb(backend: str | None = None) -> int:
    repo_id = qwen_asr_repo_id(backend)
    return qwen_asr_min_physical_vram_mb_by_repo()[repo_id]
