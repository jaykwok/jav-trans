import inspect
from pathlib import Path

from asr import transcribe
from asr.qwen_native import prepare_transcription_inputs


ROOT = Path(__file__).resolve().parents[1]


def test_asr_context_chunk_helper_removed():
    assert not hasattr(transcribe, "_build_ASR_CONTEXT_for_chunk")


def test_prepare_transcription_inputs_has_no_context_parameter():
    assert "contexts" not in inspect.signature(prepare_transcription_inputs).parameters


def test_removed_asr_context_surface_stays_absent():
    checked_paths = [
        ROOT / ".env",
        ROOT / ".env.example",
        ROOT / "README.md",
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "tools").rglob("*.py")),
        *sorted((ROOT / "src" / "web" / "static").rglob("*.js")),
        ROOT / "src" / "web" / "static" / "index.html",
    ]
    retired_tokens = (
        "ASR_CONTEXT",
        "asr_context",
        "--asr-context",
        "contexts=",
        "accepts_contexts",
        "_build_ASR_CONTEXT",
        "Context hint",
        "context_len",
    )
    for path in checked_paths:
        if not path.exists() or path.is_dir():
            continue
        text = path.read_text(encoding="utf-8")
        for token in retired_tokens:
            assert token not in text, f"{token} remains in {path.relative_to(ROOT)}"
