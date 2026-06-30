from pipeline import audio as pipeline_audio
import pytest


def _extract_filter_arg(command: list[str]) -> str:
    assert "-af" in command
    return command[command.index("-af") + 1]


def test_extract_audio_uses_dynaudnorm_by_default(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, check, timeout):
        calls.append(command)
        assert check is True
        assert timeout == 30.0

    monkeypatch.delenv("AUDIO_DYNAUDNORM", raising=False)
    monkeypatch.setenv("AUDIO_EXTRACT_TIMEOUT_S", "30")
    monkeypatch.setattr(pipeline_audio.subprocess, "run", fake_run)

    pipeline_audio.extract_audio("input.mp4", str(tmp_path / "out.wav"))

    assert calls
    filter_chain = _extract_filter_arg(calls[0])
    assert "agate=threshold=0.01" in filter_chain
    assert "dynaudnorm=f=250:g=15" in filter_chain


def test_extract_audio_can_disable_dynaudnorm(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, check, timeout):
        calls.append(command)
        assert check is True
        assert timeout == 30.0

    monkeypatch.setenv("AUDIO_DYNAUDNORM", "0")
    monkeypatch.setenv("AUDIO_EXTRACT_TIMEOUT_S", "30")
    monkeypatch.setattr(pipeline_audio.subprocess, "run", fake_run)

    pipeline_audio.extract_audio("input.mp4", str(tmp_path / "out.wav"))

    assert calls
    filter_chain = _extract_filter_arg(calls[0])
    assert "dynaudnorm" not in filter_chain


def test_extract_audio_timeout_raises_clear_error(monkeypatch, tmp_path):
    def fake_run(command, check, timeout):
        raise pipeline_audio.subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setenv("AUDIO_EXTRACT_TIMEOUT_S", "30")
    monkeypatch.setattr(pipeline_audio.subprocess, "run", fake_run)

    with pytest.raises(TimeoutError, match="ffmpeg audio extraction timed out"):
        pipeline_audio.extract_audio("input.mp4", str(tmp_path / "out.wav"))
