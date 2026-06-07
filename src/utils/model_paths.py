import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from utils.runtime_paths import is_frozen, resource_root, runtime_root

PROJECT_ROOT = runtime_root()
RESOURCE_ROOT = resource_root()
MODELS_ROOT = PROJECT_ROOT / "models"
BUNDLED_MODELS_ROOT = RESOURCE_ROOT / "models"
TMP_ROOT = PROJECT_ROOT / "tmp"
HF_RUNTIME_CACHE_ROOT = TMP_ROOT / "cache" / "hf"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_INFERENCE_IGNORE_PATTERNS = [
    "optimizer.pt",
    "**/optimizer.pt",
    "optimizer.bin",
    "**/optimizer.bin",
    "scheduler.pt",
    "**/scheduler.pt",
    "scaler.pt",
    "**/scaler.pt",
    "rng_state*.pth",
    "**/rng_state*.pth",
    "trainer_state.json",
    "**/trainer_state.json",
    "training_args.bin",
    "**/training_args.bin",
]
MODEL_CONFIG_FILENAMES = ("config.json",)
MODEL_WEIGHT_PATTERNS = (
    "model*.safetensors",
    "*.safetensors",
    "pytorch_model*.bin",
    "model*.bin",
    "tf_model*.h5",
    "flax_model*.msgpack",
)
SHARDED_SAFETENSORS_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")

if is_frozen():
    os.environ.setdefault("HF_HOME", str(MODELS_ROOT))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_RUNTIME_CACHE_ROOT / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(HF_RUNTIME_CACHE_ROOT / "xet"))
else:
    os.environ.setdefault("HF_HOME", "./models")
    os.environ.setdefault("HF_HUB_CACHE", "./tmp/cache/hf/hub")
    os.environ.setdefault("HF_XET_CACHE", "./tmp/cache/hf/xet")
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
            resolved = candidate.resolve()
            try:
                return resolved.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                return resolved.relative_to(RESOURCE_ROOT).as_posix()
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
    if not is_frozen():
        return
    bundled = BUNDLED_MODELS_ROOT / model_dir_name(repo_id)
    try:
        if bundled.resolve() != canonical_model_dir(repo_id).resolve():
            yield bundled
    except OSError:
        yield bundled


def _indexed_safetensors_complete(path: Path) -> bool:
    index_path = path / "model.safetensors.index.json"
    if not index_path.exists():
        return False
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


def _has_required_model_config(path: Path) -> bool:
    return any((path / filename).is_file() for filename in MODEL_CONFIG_FILENAMES)


def _has_positive_weight_file(path: Path) -> bool:
    for pattern in MODEL_WEIGHT_PATTERNS:
        for candidate in path.glob(pattern):
            try:
                if candidate.is_file() and candidate.stat().st_size > 0:
                    return True
            except OSError:
                continue
    return False


def _has_sharded_safetensors_file(path: Path) -> bool:
    try:
        return any(
            candidate.is_file() and SHARDED_SAFETENSORS_RE.fullmatch(candidate.name)
            for candidate in path.iterdir()
        )
    except OSError:
        return False


def _path_has_model_files(path: Path) -> bool:
    if path.is_file():
        try:
            return path.stat().st_size > 0
        except OSError:
            return False
    if path.is_dir():
        if not any(path.iterdir()):
            return False
        if not _has_required_model_config(path):
            return False
        if (path / "model.safetensors.index.json").exists():
            return _indexed_safetensors_complete(path)
        if _has_sharded_safetensors_file(path):
            return False
        return _has_positive_weight_file(path)
    return False


def model_spec_status(explicit_path: str | None, repo_id: str, *, download: bool = True) -> dict:
    target_path = canonical_model_dir(repo_id).resolve()
    checked_paths: list[str] = []

    def add_checked(path: Path) -> None:
        value = str(path)
        if value not in checked_paths:
            checked_paths.append(value)

    if explicit_path:
        candidate = _project_path(explicit_path).resolve()
        target_path = candidate
        add_checked(candidate)
        if _path_has_model_files(candidate):
            return {
                "repo_id": repo_id,
                "present": True,
                "path": str(candidate),
                "checked_paths": checked_paths,
            }
        if download:
            return {
                "repo_id": repo_id,
                "present": False,
                "path": str(candidate),
                "checked_paths": checked_paths,
            }

    for candidate in _iter_local_model_candidates(repo_id):
        resolved = candidate.expanduser().resolve()
        add_checked(resolved)
        if _path_has_model_files(resolved):
            return {
                "repo_id": repo_id,
                "present": True,
                "path": str(resolved),
                "checked_paths": checked_paths,
            }

    return {
        "repo_id": repo_id,
        "present": False,
        "path": str(target_path),
        "checked_paths": checked_paths,
    }


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

    effective_ignore_patterns = _merge_ignore_patterns(ignore_patterns)
    existed_before = target_dir.exists()
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
            ignore_patterns=effective_ignore_patterns,
            endpoint=endpoint,
            **progress_kwargs,
        )
    except BaseException as exc:
        if not existed_before:
            try:
                target_dir.rmdir()
            except OSError:
                pass
        if fallback_token is not None:
            hf_progress.fallback_error(fallback_token, exc)
        raise
    else:
        if fallback_token is not None:
            hf_progress.fallback_done(fallback_token)
    if not _path_has_model_files(target_dir):
        raise RuntimeError(
            f"Model download for {repo_id} did not produce a complete local model at "
            f"{_project_relative(target_dir)}. The partial files were kept; retrying "
            "the run will continue the download."
        )
    return str(target_dir)


def _merge_ignore_patterns(ignore_patterns: str | list[str] | None) -> list[str]:
    patterns = list(DEFAULT_INFERENCE_IGNORE_PATTERNS)
    if ignore_patterns is None:
        return patterns
    extra_patterns = [ignore_patterns] if isinstance(ignore_patterns, str) else ignore_patterns
    for pattern in extra_patterns:
        if pattern not in patterns:
            patterns.append(pattern)
    return patterns


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

