from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.model_paths import resolve_model_spec


DEFAULT_MODEL_REPO = "efwkjn/whisper-ja-anime-v0.3"


def main() -> int:
    model_path = Path(resolve_model_spec(None, DEFAULT_MODEL_REPO, download=True))
    if not model_path.exists():
        print(f"[packaging] default model missing after download: {model_path}", file=sys.stderr)
        return 1
    print(f"[packaging] default ASR model ready: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
