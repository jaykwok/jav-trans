from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from whisper import model_backend


class FakeTensor:
    def to(self, *_args, **_kwargs):
        return self


class FakeProcessor:
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _token: 2)

    def __init__(self, decoded_text: str = "品質信号テスト"):
        self.decoded_text = decoded_text

    def __call__(self, *_args, **_kwargs):
        return SimpleNamespace(input_features=FakeTensor())

    def get_decoder_prompt_ids(self, **_kwargs):
        return [(1, 2)]

    def get_prompt_ids(self, *_args, **_kwargs):
        return torch.arange(8).reshape(1, 8)

    def batch_decode(self, *_args, **_kwargs):
        return [self.decoded_text]


class FakeOutput:
    def __init__(self, *, token: int = 1, logprob: float = -0.25, scores: bool = True):
        self.sequences = torch.tensor([[token]])
        self.beam_indices = None
        self._logprob = logprob
        if scores:
            logits = torch.full((1, 4), -4.0)
            logits[0, token] = 4.0
            self.scores = (logits,)


class FakeModel:
    config = SimpleNamespace(max_target_positions=None)
    generation_config = SimpleNamespace(no_speech_token_id=2)

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def generate(self, input_features, do_sample=False, **kwargs):
        del input_features
        self.calls.append({"do_sample": do_sample, **kwargs})
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output

    def compute_transition_scores(
        self,
        sequences,
        scores,
        beam_indices=None,
        normalize_logits=True,
    ):
        del sequences, scores, beam_indices, normalize_logits
        return torch.tensor([[-0.25]])


def _backend_with(model, processor):
    backend = model_backend.WhisperModelBackend(
        preset_name="anime-whisper",
        repo_id="repo",
        model_path="model",
        generation_kwargs={"max_new_tokens": 12},
        backend_label="FakeWhisper",
    )
    backend._processor = processor
    backend._model = model
    return backend


def _patch_runtime(monkeypatch, backend):
    monkeypatch.setattr(backend, "load", lambda on_stage=None: None)
    monkeypatch.setattr(
        model_backend,
        "load_audio_16k_mono",
        lambda _path: (np.zeros(16000, dtype=np.float32), 16000),
    )
    monkeypatch.setattr("whisper.local_backend._get_wav_duration", lambda _path: 1.0)
    monkeypatch.setattr("whisper.local_backend._clean_master_text", lambda text: text)


def test_signals_extracted_on_success(monkeypatch, tmp_path):
    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"not used")
    model = FakeModel([FakeOutput()])
    backend = _backend_with(model, FakeProcessor())
    _patch_runtime(monkeypatch, backend)

    result = backend.transcribe_texts([str(audio_path)])[0]

    assert isinstance(result["avg_logprob"], float)
    assert isinstance(result["no_speech_prob"], float)
    assert isinstance(result["compression_ratio"], float)
    assert model.calls[0]["return_dict_in_generate"] is True
    assert model.calls[0]["output_scores"] is True


def test_signals_none_on_missing_scores(monkeypatch, tmp_path):
    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"not used")
    model = FakeModel([torch.tensor([[1, 2, 3]])])
    backend = _backend_with(model, FakeProcessor())
    _patch_runtime(monkeypatch, backend)

    result = backend.transcribe_texts([str(audio_path)])[0]

    assert result["avg_logprob"] is None
    assert result["no_speech_prob"] is None
    assert result["compression_ratio"] is None


def test_overflow_retry_signals_from_retry_output(monkeypatch, tmp_path):
    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"not used")
    model = FakeModel(
        [
            RuntimeError("decoder_input_ids exceeds max_target_positions"),
            FakeOutput(token=3),
        ]
    )
    backend = _backend_with(model, FakeProcessor("リトライ後テキスト"))
    _patch_runtime(monkeypatch, backend)

    result = backend.transcribe_texts(
        [str(audio_path)],
        initial_prompts=["previous context"],
    )[0]

    assert result["text"] == "リトライ後テキスト"
    assert result["avg_logprob"] == -0.25
    assert isinstance(result["no_speech_prob"], float)
    assert isinstance(result["compression_ratio"], float)
    assert len(model.calls) == 2
    assert "prompt_ids" in model.calls[0]
    assert "prompt_ids" not in model.calls[1]
