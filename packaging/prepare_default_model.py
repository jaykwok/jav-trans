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
        "label": "default ASR model",
        "repo_id": "efwkjn/whisper-ja-anime-v0.3",
    },
    {
        "label": "default WhisperSeg VAD model",
        "repo_id": "TransWithAI/Whisper-Vad-EncDec-ASMR-onnx",
        "revision": "6ac29e2cbf2f4f8e9b639861766a8639dd666e9c",
        "allow_patterns": ["model.onnx", "model_metadata.json"],
    },
    {
        "label": "WhisperSeg feature extractor model",
        "repo_id": "openai/whisper-base",
        "allow_patterns": [
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "normalizer.json",
        ],
    },
    {
        "label": "default forced aligner model",
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
