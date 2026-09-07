import fnmatch
import hashlib
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
# Downloaded weights belong with the other models, not in the throwaway
# tree: `tmp/` is documented as safe to delete, and a 5GB GGUF landing
# there means "clear the cache" silently costs a re-download.
HF_RUNTIME_CACHE_ROOT = MODELS_ROOT
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

def default_hf_token_path() -> Path:
    """Where ``hf auth login`` actually put the token.

    huggingface_hub derives ``HF_TOKEN_PATH`` from ``HF_HOME``, and this app
    redirects ``HF_HOME`` so weights land inside the app directory - which
    silently logs the user out: the token sits at the *default* HF_HOME and is
    never read. Every download then goes out anonymous, which is slower and is
    a flat 401 on any gated repo, with nothing saying why. Pinning the token
    path back keeps `hf auth login` meaning what it says without copying a
    secret into the project tree.
    """
    cache_root = os.getenv("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache_root) / "huggingface" / "token"


if is_frozen():
    os.environ.setdefault("HF_HOME", str(MODELS_ROOT))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_RUNTIME_CACHE_ROOT / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(HF_RUNTIME_CACHE_ROOT / "xet"))
else:
    os.environ.setdefault("HF_HOME", "./models")
    os.environ.setdefault("HF_HUB_CACHE", "./models/hub")
    os.environ.setdefault("HF_XET_CACHE", "./models/xet")
os.environ.setdefault("HF_TOKEN_PATH", str(default_hf_token_path()))
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


MODEL_RECEIPT = "model-revision.json"
MODEL_RECEIPT_SCHEMA = "model_revision_v1"


def require_commit_sha(revision: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("模型固定版本必须是完整的 40 位 commit SHA")
    return revision.lower()


def revision_model_dir(repo_id: str, revision: str) -> Path:
    return MODELS_ROOT / ".pinned" / model_dir_name(repo_id) / require_commit_sha(revision)


def inference_model_files(root: Path, *, include_receipt: bool = False):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.name == MODEL_RECEIPT and not include_receipt:
            continue
        if any(
            fnmatch.fnmatch(relative.as_posix(), pattern) or fnmatch.fnmatch(path.name, pattern)
            for pattern in DEFAULT_INFERENCE_IGNORE_PATTERNS
        ):
            continue
        yield path


def _model_file_records(root: Path) -> dict:
    records = {}
    for path in inference_model_files(root):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        records[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
        }
    return records


def record_model_revision(root: Path, repo_id: str, revision: str) -> dict:
    if not _path_has_model_files(root):
        raise ValueError("固定版本模型缺少配置或完整权重")
    payload = {
        "schema": MODEL_RECEIPT_SCHEMA, "repo_id": repo_id,
        "revision": require_commit_sha(revision), "files": _model_file_records(root),
    }
    with (root / MODEL_RECEIPT).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def verify_model_revision(root: Path, repo_id: str, revision: str) -> dict:
    revision = require_commit_sha(revision)
    try:
        payload = json.loads((root / MODEL_RECEIPT).read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("模型目录没有可验证的固定版本记录") from exc
    if not isinstance(payload, dict) or (
        payload.get("schema") != MODEL_RECEIPT_SCHEMA
        or payload.get("repo_id") != repo_id or payload.get("revision") != revision
        or not _path_has_model_files(root)
    ):
        raise ValueError("模型目录与固定版本不符")
    if payload.get("files") != _model_file_records(root):
        raise ValueError("模型文件已改变，与固定版本的哈希记录不符")
    return payload


def _resolve_pinned_model(
    explicit_path: str | None, repo_id: str, revision: str, *, download: bool,
    allow_patterns=None, ignore_patterns=None,
) -> str:
    revision = require_commit_sha(revision)
    target = _project_path(explicit_path).resolve() if explicit_path else revision_model_dir(repo_id, revision).resolve()
    if target.exists():
        verify_model_revision(target, repo_id, revision)
        return str(target)
    if not download:
        raise FileNotFoundError("固定版本模型尚未准备好")
    # A mutable canonical cache never proves which revision it contains.
    # Keep each SHA separate; only a completed, verified directory is reusable.
    from filelock import FileLock

    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target.with_name(target.name + ".lock")), timeout=60):
        if target.exists():
            verify_model_revision(target, repo_id, revision)
            return str(target)
        staging = target.with_name(target.name + ".partial")
        _download_snapshot(
            repo_id, staging, revision=revision,
            allow_patterns=allow_patterns, ignore_patterns=ignore_patterns,
        )
        record_model_revision(staging, repo_id, revision)
        staging.replace(target)
    return str(target)


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
    canonical = canonical_model_dir(repo_id).resolve()
    target_path = canonical
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
        if download and candidate != canonical:
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
    try:
        endpoint = normalize_hf_endpoint() or DEFAULT_HF_ENDPOINT
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=target_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=effective_ignore_patterns,
            endpoint=endpoint,
            tqdm_class=hf_progress.tqdm_class(),
        )
    except BaseException:
        if not existed_before:
            try:
                target_dir.rmdir()
            except OSError:
                pass
        raise
    if not _path_has_model_files(target_dir):
        raise RuntimeError(
            f"模型 {repo_id} 没有下载完整（已下载的部分保留在 "
            f"{_project_relative(target_dir)}）。请检查网络后重试，重试会接着下载；"
            "网络太慢可以先在「网络代理」中配置代理。"
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
    if revision is not None:
        return _resolve_pinned_model(
            explicit_path, repo_id, revision, download=download,
            allow_patterns=allow_patterns, ignore_patterns=ignore_patterns,
        )
    if explicit_path:
        candidate = _project_path(explicit_path).resolve()
        if _path_has_model_files(candidate):
            return str(candidate)
        canonical = canonical_model_dir(repo_id).resolve()
        if download and candidate != canonical:
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
