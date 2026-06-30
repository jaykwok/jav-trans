from __future__ import annotations

from types import SimpleNamespace

from asr import local_backend
from helpers import ASR_17B_BACKEND


def test_qwen_text_result_includes_generation_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("ASR_BACKEND", ASR_17B_BACKEND)
    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"not used")
    monkeypatch.setattr(local_backend, "_get_wav_duration", lambda _path: 1.25)

    backend = local_backend.LocalAsrBackend("cpu")
    result, log = backend._build_text_result(
        str(audio_path),
        SimpleNamespace(language="Japanese", text="テスト"),
        "Japanese",
    )

    assert result["text"] == "テスト"
    assert result["asr_generation"]["backend"] == ASR_17B_BACKEND
    assert result["asr_generation"]["configured_max_new_tokens"] == local_backend.ASR_MAX_NEW_TOKENS
    assert result["asr_generation"]["model_max_target_positions"] is None
    assert result["asr_generation"]["policy"] == "qwen_from_pretrained_limit"
    assert result["asr_generation"]["error_kind"] is None
    assert "ASR 输出模式: text_only" in log


def test_qwen_generation_metadata_records_timeout_kind(monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", ASR_17B_BACKEND)
    metadata = local_backend._qwen_generation_metadata(
        error_kind="timeout",
        error_detail="worker timeout",
        worker_mode="subprocess",
    )

    assert metadata["backend"] == ASR_17B_BACKEND
    assert metadata["worker_mode"] == "subprocess"
    assert metadata["error_kind"] == "timeout"
    assert metadata["error_detail"] == "worker timeout"


def test_qwen_generation_safety_normalizes_inner_generation_config(monkeypatch):
    generation_config = SimpleNamespace(
        temperature=0.0,
        pad_token_id=None,
        eos_token_id=[151645, 151643],
        repetition_penalty=1.0,
    )
    model = SimpleNamespace(
        model=SimpleNamespace(
            thinker=SimpleNamespace(generation_config=generation_config)
        )
    )
    monkeypatch.setattr(local_backend, "ASR_REPETITION_PENALTY", 1.05)

    local_backend._apply_generation_safety(model)

    assert generation_config.temperature is None
    assert generation_config.pad_token_id == 151645
    assert generation_config.repetition_penalty == 1.05


def test_qwen_generation_safety_handles_direct_generation_config(monkeypatch):
    generation_config = SimpleNamespace(
        temperature=0.0,
        pad_token_id=None,
        eos_token_id=151645,
        repetition_penalty=1.0,
    )
    model = SimpleNamespace(generation_config=generation_config)
    monkeypatch.setattr(local_backend, "ASR_REPETITION_PENALTY", 1.0)

    local_backend._apply_generation_safety(model)

    assert generation_config.temperature is None
    assert generation_config.pad_token_id == 151645
    assert generation_config.repetition_penalty == 1.0


def test_qwen_generation_safety_clears_default_temperature_with_nested_eos(monkeypatch):
    generation_config = SimpleNamespace(
        temperature=1.0,
        do_sample=False,
        pad_token_id=None,
        eos_token_id=None,
        repetition_penalty=1.0,
    )
    thinker_config = SimpleNamespace(
        pad_token_id=None,
        eos_token_id=[151645, 151643],
    )
    model = SimpleNamespace(
        model=SimpleNamespace(
            thinker=SimpleNamespace(
                generation_config=generation_config,
                config=thinker_config,
            )
        )
    )
    monkeypatch.setattr(local_backend, "ASR_REPETITION_PENALTY", 1.0)

    local_backend._apply_generation_safety(model)

    assert generation_config.temperature is None
    assert generation_config.pad_token_id == 151645
    assert thinker_config.pad_token_id == 151645
