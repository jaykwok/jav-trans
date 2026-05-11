from types import SimpleNamespace

from audio.loading import load_audio_16k_mono
from whisper import model_backend
from whisper.backends.base import BaseAsrBackend
from whisper.model_backend import create_whisper_model_backend


def test_whisper_model_backend_anime_preset_protocol():
    backend = create_whisper_model_backend("anime-whisper", "cpu")

    assert isinstance(backend, BaseAsrBackend), "WhisperModelBackend does not conform to BaseAsrBackend protocol"
    assert backend.accepts_contexts is False, "accepts_contexts should be False"
    assert backend.is_subprocess is False, "is_subprocess should be False"
    assert backend.timestamp_mode == "forced", f"timestamp_mode should be 'forced', got {backend.timestamp_mode}"
    assert backend.preset_name == "anime-whisper"
    assert backend.repo_id == "litagin/anime-whisper"
    assert backend.generation_kwargs["num_beams"] == 1
    assert backend.generation_kwargs["no_repeat_ngram_size"] == 0


def test_whisper_preset_names_are_public_backend_names():
    assert set(model_backend.WHISPER_PRESETS) == {
        "anime-whisper",
        "whisper-ja-anime-v0.3",
        "whisper-ja-1.5b",
    }


def test_generation_config_normalized_for_deterministic_asr():
    model = SimpleNamespace(
        generation_config=SimpleNamespace(
            temperature=0.0,
            pad_token_id=None,
            eos_token_id=151645,
        )
    )

    model_backend._normalize_generation_config_for_deterministic_asr(model)

    assert model.generation_config.temperature is None
    assert model.generation_config.pad_token_id == 151645


def test_generation_config_keeps_default_temperature():
    model = SimpleNamespace(
        generation_config=SimpleNamespace(
            temperature=1.0,
            pad_token_id=42,
            eos_token_id=151645,
        )
    )

    model_backend._normalize_generation_config_for_deterministic_asr(model)

    assert model.generation_config.temperature == 1.0
    assert model.generation_config.pad_token_id == 42


def test_whisper_audio_loader_uses_soundfile_path(monkeypatch):
    import numpy as np

    def fail_if_librosa_is_used(*_args, **_kwargs):
        raise AssertionError("librosa.load should not be used")

    monkeypatch.setattr(
        "librosa.load",
        fail_if_librosa_is_used,
        raising=False,
    )
    monkeypatch.setattr(
        "soundfile.read",
        lambda *_args, **_kwargs: (np.zeros(1600, dtype=np.float32), 16000),
    )

    audio, sample_rate = load_audio_16k_mono("sample.wav")

    assert sample_rate == 16000
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.shape[0] == 1600

if __name__ == "__main__":
    test_whisper_model_backend_anime_preset_protocol()

