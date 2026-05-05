import json
import os
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = PROJECT_ROOT / "models"
TEMP_ROOT = PROJECT_ROOT / "temp"
HF_RUNTIME_CACHE_ROOT = TEMP_ROOT / "hf-cache"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"

os.environ.setdefault("HF_HOME", "./models")
os.environ.setdefault("HF_HUB_CACHE", "./temp/hf-cache/hub")
os.environ.setdefault("HF_XET_CACHE", "./temp/hf-cache/xet")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def normalize_hf_endpoint(value: str | None = None) -> str | None:
    endpoint = (os.getenv("HF_ENDPOINT") if value is None else value) or ""
    endpoint = endpoint.strip().rstrip("/")
    if not endpoint:
        os.environ.pop("HF_ENDPOINT", None)
        return None

    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "HF_ENDPOINT must be empty or a full URL starting with http:// or https://"
        )

    os.environ["HF_ENDPOINT"] = endpoint
    return endpoint


def _project_relative(path: str | Path) -> str:
    candidate = Path(path)
    try:
        if candidate.is_absolute():
            return candidate.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (OSError, ValueError):
        return str(path).replace("\\", "/")
    return str(path).replace("\\", "/")


def _project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def model_dir_name(repo_id: str) -> str:
    return repo_id.strip("/").replace("/", "-")


def canonical_model_dir(repo_id: str) -> Path:
    return MODELS_ROOT / model_dir_name(repo_id)


def _iter_local_model_candidates(repo_id: str):
    yield canonical_model_dir(repo_id)


def _indexed_safetensors_complete(path: Path) -> bool:
    index_path = path / "model.safetensors.index.json"
    if not index_path.exists():
        return True
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False

    shard_names = {
        str(value)
        for value in weight_map.values()
        if isinstance(value, str) and value.strip()
    }
    if not shard_names:
        return False

    for shard_name in shard_names:
        shard_path = path / shard_name
        try:
            if not shard_path.is_file() or shard_path.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


def _path_has_model_files(path: Path) -> bool:
    if path.is_file():
        return True
    if path.is_dir():
        if not any(path.iterdir()):
            return False
        return _indexed_safetensors_complete(path)
    return False


def _download_snapshot(
    repo_id: str,
    target_dir: Path,
    *,
    revision: str | None = None,
    allow_patterns: str | list[str] | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Model download requires huggingface_hub. Install huggingface_hub>=0.25."
        ) from exc

    from utils import hf_progress

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[models] downloading {repo_id} -> {_project_relative(target_dir)}", flush=True)
    progress_kwargs = hf_progress.snapshot_download_kwargs(snapshot_download)
    fallback_token = None
    if not progress_kwargs:
        fallback_token = hf_progress.fallback_start(repo_id)
    try:
        endpoint = normalize_hf_endpoint() or DEFAULT_HF_ENDPOINT
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=target_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            endpoint=endpoint,
            **progress_kwargs,
        )
    except Exception as exc:
        if fallback_token is not None:
            hf_progress.fallback_error(fallback_token, exc)
        raise
    else:
        if fallback_token is not None:
            hf_progress.fallback_done(fallback_token)
    return str(target_dir)


def resolve_model_spec(
    explicit_path: str | None,
    repo_id: str,
    *,
    download: bool = False,
    revision: str | None = None,
    allow_patterns: str | list[str] | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    if explicit_path:
        candidate = _project_path(explicit_path).resolve()
        if _path_has_model_files(candidate):
            return str(candidate)
        if download:
            return _download_snapshot(
                repo_id,
                candidate,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )

    canonical = canonical_model_dir(repo_id).resolve()
    for candidate in _iter_local_model_candidates(repo_id):
        resolved = candidate.expanduser().resolve()
        if _path_has_model_files(resolved):
            return str(resolved)

    if download:
        return _download_snapshot(
            repo_id,
            canonical,
            revision=revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    return repo_id


WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "").strip()

