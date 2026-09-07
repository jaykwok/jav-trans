"""What a release actually contains, hashed, so a later build can be compared.

The version number on an archive says which commit it was built from. It does
not say which model weights were downloaded that afternoon, which llama-server
binary happened to be on PATH, or whether the lock file had been touched - and
those are exactly the things that differ between two builds of one commit, and
exactly what a "it worked on the release but not on mine" report needs answered.

Every payload file is hashed, including configuration and Python code. The
manifest records the source commit, dependency lock and verified model receipts,
so a pin describes the files actually bundled rather than only a download option.

    uv run python packaging/record_release_manifest.py --dist dist/jav-trans
    uv run python packaging/record_release_manifest.py --verify dist/release-assets/manifest.json

Paths are recorded relative to the roots they were found under: a manifest is
something to attach to a release, and it has no business carrying the layout of
whoever's machine built it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from utils.model_paths import model_dir_name, verify_model_revision  # noqa: E402

MANIFEST_SCHEMA = "release_manifest_v2"

# Suffix -> category. Order does not matter; a file belongs to one category.
CATEGORY_BY_SUFFIX = {
    ".safetensors": "weights",
    ".pt": "weights",
    ".pth": "weights",
    ".bin": "weights",
    ".gguf": "weights",
    ".onnx": "weights",
    ".npz": "weights",
    ".jinja": "template",
    ".j2": "template",
    ".tmpl": "template",
    ".exe": "binary",
    ".dll": "binary",
    ".so": "binary",
    ".dylib": "binary",
}

# Repo files that decide what was built, hashed from the source tree rather than
# from the payload because that is where they live.
REPO_FILES = (
    "uv.lock",
    "pyproject.toml",
    "packaging/model-pins.json",
)

_READ_CHUNK = 1024 * 1024


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_READ_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def category_of(path: Path) -> str:
    return CATEGORY_BY_SUFFIX.get(path.suffix.lower(), "payload")


def _git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def git_state() -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        # A release built from an edited tree is not the commit it names, and
        # this is the only place that can still notice.
        "dirty": bool(status),
    }


def model_pins() -> dict:
    try:
        payload = json.loads((ROOT / "packaging" / "model-pins.json").read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    models = payload.get("models") if isinstance(payload, dict) else None
    return models if isinstance(models, dict) else {}


def scan(dist_dir: Path) -> tuple[list[dict], dict]:
    entries: list[dict] = []
    for path in sorted(dist_dir.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        category = category_of(path)
        entries.append(
            {
                "path": path.relative_to(dist_dir).as_posix(),
                "root": "dist",
                "category": category,
                "bytes": size,
                "sha256": sha256_of(path),
            }
        )
    return entries, {"count": 0, "bytes": 0}


def repo_entries() -> list[dict]:
    entries = []
    for name in REPO_FILES:
        path = ROOT / name
        if not path.is_file():
            continue
        entries.append(
            {
                "path": name,
                "root": "repo",
                "category": "lock",
                "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
            }
        )
    return entries


def verify_bundled_models(dist_dir: Path, pins: dict) -> list[dict]:
    if not pins:
        raise ValueError("发行构建缺少模型 pin")
    verified = []
    for repo, revision in pins.items():
        candidates = [
            dist_dir / "models" / model_dir_name(repo),
            dist_dir / "_internal" / "models" / model_dir_name(repo),
        ]
        found = [path for path in candidates if path.is_dir()]
        if len(found) != 1:
            raise ValueError(f"发行包缺少唯一的固定模型载荷：{repo}")
        verify_model_revision(found[0], repo, revision)
        verified.append({
            "repo_id": repo, "revision": revision,
            "path": found[0].relative_to(dist_dir).as_posix(),
        })
    return verified


def build_manifest(dist_dir: Path, *, require_models: bool = False) -> dict:
    pins = model_pins()
    verified_models = verify_bundled_models(dist_dir, pins) if require_models else []
    entries, unhashed = scan(dist_dir)
    entries.extend(repo_entries())
    return {
        "schema": MANIFEST_SCHEMA,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "model_pins": pins,
        "models_verified": require_models,
        "model_payloads": verified_models,
        "dist_root": dist_dir.name,
        "files": sorted(entries, key=lambda item: (item["root"], item["path"])),
        "unhashed": unhashed,
    }


def _resolve(entry: dict, dist_dir: Path) -> Path:
    if entry.get("root") not in {"repo", "dist"}:
        raise ValueError("清单包含未知文件根目录")
    root = (ROOT if entry["root"] == "repo" else dist_dir).resolve()
    path = (root / entry["path"]).resolve()
    path.relative_to(root)
    return path


def verify(manifest_path: Path, dist_dir: Path) -> int:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != MANIFEST_SCHEMA:
        print(f"[packaging] {manifest_path} 不是发行清单", file=sys.stderr)
        return 1
    if payload.get("models_verified"):
        try:
            verify_bundled_models(dist_dir, payload.get("model_pins", {}))
        except (OSError, ValueError) as exc:
            print(f"[packaging] 模型载荷验证失败：{exc}", file=sys.stderr)
            return 1
    missing: list[str] = []
    changed: list[str] = []
    for entry in payload.get("files") or []:
        path = _resolve(entry, dist_dir)
        if not path.is_file():
            missing.append(entry["path"])
            continue
        if sha256_of(path) != entry["sha256"]:
            changed.append(entry["path"])

    recorded = {
        (entry.get("root"), entry["path"]) for entry in payload.get("files") or []
    }
    extra = [
        path.relative_to(dist_dir).as_posix()
        for path in sorted(dist_dir.rglob("*"))
        if path.is_file()
        and ("dist", path.relative_to(dist_dir).as_posix()) not in recorded
    ]

    for label, items in (("缺失", missing), ("内容不同", changed), ("多出", extra)):
        for item in items:
            print(f"[packaging] {label}: {item}")
    if missing or changed or extra:
        print(
            f"[packaging] 清单不匹配：缺失 {len(missing)}、不同 {len(changed)}、"
            f"多出 {len(extra)}",
            file=sys.stderr,
        )
        return 1
    print(f"[packaging] 清单一致：{len(payload.get('files') or [])} 个文件")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", default="dist/jav-trans")
    parser.add_argument("--output", default="dist/release-assets/release-manifest.json")
    parser.add_argument("--skip-models", action="store_true", help="record an explicitly model-free build")
    parser.add_argument(
        "--verify",
        default="",
        help="check an existing manifest against --dist instead of writing one",
    )
    args = parser.parse_args(argv)

    dist_dir = Path(args.dist)
    if not dist_dir.is_absolute():
        dist_dir = ROOT / dist_dir
    if not dist_dir.is_dir():
        print(f"[packaging] 没有找到构建产物目录：{dist_dir}", file=sys.stderr)
        return 1

    if args.verify:
        manifest_path = Path(args.verify)
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        return verify(manifest_path, dist_dir)

    try:
        manifest = build_manifest(dist_dir, require_models=not args.skip_models)
    except (OSError, ValueError) as exc:
        print(f"[packaging] 发行载荷验证失败：{exc}", file=sys.stderr)
        return 1
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    from pipeline.artifact_store import write_json_atomic

    write_json_atomic(output, manifest, project_root=ROOT)
    hashed = len(manifest["files"])
    print(f"[packaging] 已记录 {hashed} 个文件的哈希 -> {output}")
    if manifest["git"]["dirty"]:
        print("[packaging] WARNING: 工作区有未提交改动，这个包不等于它标注的提交")
    if not manifest["model_pins"]:
        print("[packaging] WARNING: 没有模型版本固定记录，构建不可复现")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
