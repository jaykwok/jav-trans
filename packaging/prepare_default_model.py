from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.model_paths import resolve_model_spec


REQUIRED_MODEL_SPECS = [
    {
        "label": "bundled 0.6B ASR / SpeechBoundary model",
        "repo_id": "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame",
    },
    {
        "label": "bundled 1.7B ASR model",
        "repo_id": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame",
    },
    {
        "label": "bundled forced aligner model",
        "repo_id": "Qwen/Qwen3-ForcedAligner-0.6B",
    },
]


def main() -> int:
    for spec in REQUIRED_MODEL_SPECS:
        model_path = Path(
            resolve_model_spec(
                None,
                spec["repo_id"],
                download=True,
                revision=spec.get("revision"),
                allow_patterns=spec.get("allow_patterns"),
            )
        )
        if not model_path.exists():
            print(
                f"[packaging] {spec['label']} missing after download: {model_path}",
                file=sys.stderr,
            )
            return 1
        print(f"[packaging] {spec['label']} ready: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
