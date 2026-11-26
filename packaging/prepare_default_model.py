"""Fetch the models a release ships with, at the revision the release pins.

A release that says "the 1.7B ASR model" and downloads whatever the branch
points at today is not reproducible: two builds of the same commit can carry
different weights, and nothing in either build says which. The alignment head
has been pinned by sha since it moved into `.env` defaults; this is the other
half of the same rule.

The pins live in `packaging/model-pins.json`, next to this file and in git, so
the pin is part of the commit that produced the build. Refresh it deliberately:

    uv run python packaging/prepare_default_model.py --pin

which asks the Hub what the branch currently points at and writes it down. That
is the one network call that decides what a release contains, so it is a
separate, explicit step rather than something a build does on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.model_paths import (  # noqa: E402
    require_commit_sha, resolve_model_spec, revision_model_dir, verify_model_revision,
)

PINS_PATH = Path(__file__).resolve().parent / "model-pins.json"
PINS_SCHEMA = "release_model_pins_v1"

REQUIRED_MODEL_SPECS = [
    {
        "label": "bundled default 1.7B ASR / SpeechBoundary model",
        "repo_id": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    },
]


def read_pins() -> dict[str, str]:
    try:
        payload = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict) or payload.get("schema") != PINS_SCHEMA:
        return {}
    models = payload.get("models")
    if not isinstance(models, dict):
        return {}
    return {str(repo): require_commit_sha(str(sha)) for repo, sha in models.items() if str(sha).strip()}


def write_pins(pins: dict[str, str]) -> None:
    PINS_PATH.write_text(
        json.dumps(
            {"schema": PINS_SCHEMA, "models": dict(sorted(pins.items()))},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def resolve_current_revision(repo_id: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi(token=False).model_info(repo_id, timeout=20)
    sha = str(getattr(info, "sha", "") or "")
    if not sha:
        raise SystemExit(f"[packaging] {repo_id} 没有返回 commit sha，无法固定版本")
    return require_commit_sha(sha)


def verified_release_models() -> list[tuple[Path, str, str]]:
    """The exact verified inputs consumed by the PyInstaller specification."""
    pins = read_pins()
    result = []
    for spec in REQUIRED_MODEL_SPECS:
        repo = spec["repo_id"]
        revision = pins.get(repo)
        if revision is None:
            raise ValueError("缺少模型 pin；请先运行 prepare_default_model.py --pin")
        path = revision_model_dir(repo, revision).resolve()
        verify_model_revision(path, repo, revision)
        result.append((path, repo, spec["label"]))
    return result


def pin_models() -> int:
    pins = read_pins()
    for spec in REQUIRED_MODEL_SPECS:
        repo_id = spec["repo_id"]
        sha = resolve_current_revision(repo_id)
        previous = pins.get(repo_id)
        pins[repo_id] = sha
        if previous and previous != sha:
            print(f"[packaging] {repo_id}: {previous} -> {sha}")
        else:
            print(f"[packaging] {repo_id}: {sha}")
    write_pins(pins)
    print(f"[packaging] 已写入 {PINS_PATH.relative_to(ROOT).as_posix()}")
    return 0


def prepare() -> int:
    pins = read_pins()
    for spec in REQUIRED_MODEL_SPECS:
        repo_id = spec["repo_id"]
        revision = pins.get(repo_id)
        if not revision:
            print(
                f"[packaging] {repo_id} 没有固定版本。"
                "请先运行 `uv run python packaging/prepare_default_model.py --pin`。",
                file=sys.stderr,
            )
            return 1
        revision = require_commit_sha(revision)
        model_path = Path(
            resolve_model_spec(
                None,
                repo_id,
                download=True,
                revision=revision,
                allow_patterns=spec.get("allow_patterns"),
            )
        )
        if not model_path.exists():
            print(
                f"[packaging] {spec['label']} missing after download: {model_path}",
                file=sys.stderr,
            )
            return 1
        verify_model_revision(model_path, repo_id, revision)
        print(f"[packaging] {spec['label']} ready @ {revision}: {model_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pin",
        action="store_true",
        help="ask the Hub what each repo currently points at and record it",
    )
    args = parser.parse_args(argv)
    if args.pin:
        return pin_models()
    return prepare()


if __name__ == "__main__":
    raise SystemExit(main())
