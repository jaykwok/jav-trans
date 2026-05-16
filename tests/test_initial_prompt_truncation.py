from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from whisper import model_backend, transcribe


def test_long_prompt_is_char_truncated(monkeypatch):
    monkeypatch.setenv("ASR_INITIAL_PROMPT_MAX_CHARS", "24")

    prompt = transcribe._truncate_initial_prompt(
        "first words should be removed latest context stays"
    )

    assert len(prompt) <= 20
    assert prompt == "latest context stays"


def test_prompt_ids_token_cap(monkeypatch):
    monkeypatch.setattr(model_backend, "_ASR_INITIAL_PROMPT_MAX_TOKENS", 180)
    prompt_ids = torch.arange(300).reshape(1, 300)

    capped, original_count, kept_count = model_backend._cap_initial_prompt_ids(prompt_ids)

    assert original_count == 300
    assert kept_count == 180
    assert int(capped.numel()) == 180
    assert torch.equal(capped, prompt_ids[..., -180:])


def test_overflow_retry(monkeypatch, tmp_path):
    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"not used")
    generate_calls: list[dict] = []

    class FakeTensor:
        def to(self, *_args, **_kwargs):
            return self

    class FakeProcessor:
        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(input_features=FakeTensor())

        def get_decoder_prompt_ids(self, **_kwargs):
            return [(1, 2)]

        def get_prompt_ids(self, *_args, **_kwargs):
            return torch.arange(8).reshape(1, 8)

        def batch_decode(self, *_args, **_kwargs):
            return ["正常テキスト"]

    class FakeModel:
        config = SimpleNamespace(max_target_positions=None)

        def generate(self, input_features, do_sample=False, **kwargs):
            del input_features, do_sample
            generate_calls.append(dict(kwargs))
            if len(generate_calls) == 1:
                raise RuntimeError("decoder_input_ids exceeds max_target_positions")
            return torch.tensor([[1, 2, 3]])

    backend = model_backend.WhisperModelBackend(
        preset_name="anime-whisper",
        repo_id="repo",
        model_path="model",
        generation_kwargs={"max_new_tokens": 12},
        backend_label="FakeWhisper",
    )
    backend._processor = FakeProcessor()
    backend._model = FakeModel()

    monkeypatch.setattr(backend, "load", lambda on_stage=None: None)
    monkeypatch.setattr(
        model_backend,
        "load_audio_16k_mono",
        lambda _path: (np.zeros(16000, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(
        "whisper.local_backend._get_wav_duration",
        lambda _path: 1.0,
    )
    monkeypatch.setattr(
        "whisper.local_backend._clean_master_text",
        lambda text: text,
    )

    result = backend.transcribe_texts(
        [str(audio_path)],
        initial_prompts=["previous context"],
    )

    assert result[0]["text"] == "正常テキスト"
    assert len(generate_calls) == 2
    assert "prompt_ids" in generate_calls[0]
    assert "prompt_ids" not in generate_calls[1]


def test_dynamic_budget_prevents_whisper_decoder_overflow(monkeypatch, tmp_path):
    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"not used")
    generate_calls: list[dict] = []

    class FakeTensor:
        def to(self, *_args, **_kwargs):
            return self

    class FakeProcessor:
        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(input_features=FakeTensor())

        def get_decoder_prompt_ids(self, **_kwargs):
            return [(1, 2), (2, 3)]

        def get_prompt_ids(self, *_args, **_kwargs):
            return torch.arange(18).reshape(1, 18)

        def batch_decode(self, *_args, **_kwargs):
            return ["予算内テキスト"]

    class FakeModel:
        config = SimpleNamespace(max_target_positions=20)

        def generate(self, input_features, do_sample=False, **kwargs):
            del input_features, do_sample
            generate_calls.append(dict(kwargs))
            prompt_len = int(kwargs.get("prompt_ids").numel()) if kwargs.get("prompt_ids") is not None else 0
            decoder_len = 1 + len(kwargs.get("forced_decoder_ids") or []) + prompt_len
            assert decoder_len + int(kwargs["max_new_tokens"]) <= 20
            return torch.tensor([[1, 2, 3]])

    backend = model_backend.WhisperModelBackend(
        preset_name="anime-whisper",
        repo_id="repo",
        model_path="model",
        generation_kwargs={"max_new_tokens": 18},
        backend_label="FakeWhisper",
    )
    backend._processor = FakeProcessor()
    backend._model = FakeModel()

    monkeypatch.setattr(backend, "load", lambda on_stage=None: None)
    monkeypatch.setenv("ASR_INITIAL_PROMPT_MAX_TOKENS", "18")
    monkeypatch.setenv("ASR_MIN_EFFECTIVE_NEW_TOKENS", "6")
    monkeypatch.setattr(
        model_backend,
        "load_audio_16k_mono",
        lambda _path: (np.zeros(16000, dtype=np.float32), 16000),
    )
    monkeypatch.setattr("whisper.local_backend._get_wav_duration", lambda _path: 1.0)
    monkeypatch.setattr("whisper.local_backend._clean_master_text", lambda text: text)

    result = backend.transcribe_texts(
        [str(audio_path)],
        initial_prompts=["previous context"],
    )[0]

    assert result["text"] == "予算内テキスト"
    assert len(generate_calls) == 1
    assert generate_calls[0]["max_new_tokens"] == 6
    assert int(generate_calls[0]["prompt_ids"].numel()) == 11
    assert result["asr_generation"]["budget"]["clipped_max_new_tokens"] is True
    assert result["asr_generation"]["budget"]["clipped_prompt_tokens"] is True
