from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.omni import speech_to_text_transport as stt


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.headers = {"X-Generation-Id": "generation-1"}

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_openrouter_stt_payload_keeps_diarization_in_provider_adapter(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"audio")
    transport = stt.OpenRouterSpeechToTextTransport(
        model="x-ai/grok-stt-1.0",
        api_key="secret",
        base_url="https://openrouter.ai/api/v1/chat/completions",
        timeout_s=30,
    )

    payload = transport.request_payload(
        audio_path=audio,
        language="ja",
        diarize=True,
        filler_words=False,
        vad_threshold=0.5,
    )

    assert transport.endpoint == "https://openrouter.ai/api/v1/audio/transcriptions"
    assert payload["model"] == "x-ai/grok-stt-1.0"
    assert payload["timestamp_granularities"] == ["word"]
    assert payload["provider"]["options"]["x-ai"] == {
        "diarize": True,
        "filler_words": False,
        "vad_threshold": 0.5,
    }
    assert payload["input_audio"]["format"] == "mp3"


def test_openrouter_stt_normalizes_speaker_words(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"audio")
    transport = stt.OpenRouterSpeechToTextTransport(
        model="x-ai/grok-stt-1.0",
        api_key="secret",
        base_url="https://openrouter.ai/api/v1",
        timeout_s=30,
    )
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "text": "はい、そうです",
                "duration": 1.2,
                "words": [
                    {
                        "text": "はい",
                        "start": 0.1,
                        "end": 0.4,
                        "speaker": 0,
                        "confidence": 0.9,
                    },
                    {
                        "word": "そうです",
                        "start": 0.6,
                        "end": 1.1,
                        "speaker": 1,
                    },
                ],
                "usage": {"cost": 0.001},
            }
        )

    monkeypatch.setattr(stt, "urlopen", fake_urlopen)
    result = transport.transcribe(audio_path=audio, diarize=True)

    assert captured["timeout"] == 30
    assert captured["request"].get_header("Authorization") == "Bearer secret"
    assert [word["speaker"] for word in result.parsed["words"]] == [0, 1]
    assert result.parsed["diagnostics"]["speaker_count"] == 2
    assert result.response_headers["x-generation-id"] == "generation-1"


def test_openrouter_stt_rejects_silent_diarization_drop(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"audio")
    transport = stt.OpenRouterSpeechToTextTransport(
        model="x-ai/grok-stt-1.0",
        api_key="secret",
        base_url="https://openrouter.ai/api/v1",
        timeout_s=30,
    )

    monkeypatch.setattr(
        stt,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(
            {
                "text": "はい",
                "words": [{"text": "はい", "start": 0.1, "end": 0.4}],
            }
        ),
    )

    with pytest.raises(RuntimeError, match="have no speaker"):
        transport.transcribe(audio_path=audio, diarize=True)


def test_stt_factory_reads_named_omni_profile(tmp_path: Path) -> None:
    env_file = tmp_path / "openrouter"
    env_file.write_text(
        "OMNI_API_KEY=secret\n"
        "OMNI_BASE_URL=https://openrouter.ai/api/v1\n"
        "OMNI_MODEL=google/gemini-3.6-flash\n",
        encoding="utf-8",
    )

    transport = stt.create_speech_to_text_transport(
        profile="openrouter",
        env_file=env_file,
        model_override="x-ai/grok-stt-1.0",
    )

    assert isinstance(transport, stt.OpenRouterSpeechToTextTransport)
    assert transport.model == "x-ai/grok-stt-1.0"
    assert transport.max_concurrency == 10
