from pipeline import audio as pipeline_audio
def _extract_filter_arg(command: list[str]) -> str:
    assert "-af" in command
    return command[command.index("-af") + 1]


def test_extract_audio_uses_dynaudnorm_by_default(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, check):
        calls.append(command)
        assert check is True

    monkeypatch.delenv("AUDIO_DYNAUDNORM", raising=False)
    monkeypatch.setattr(pipeline_audio.subprocess, "run", fake_run)

    pipeline_audio.extract_audio("input.mp4", str(tmp_path / "out.wav"))

    assert calls
    filter_chain = _extract_filter_arg(calls[0])
    assert "agate=threshold=0.01" in filter_chain
    assert "dynaudnorm=f=250:g=15" in filter_chain


def test_extract_audio_can_disable_dynaudnorm(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, check):
        calls.append(command)
        assert check is True

    monkeypatch.setenv("AUDIO_DYNAUDNORM", "0")
    monkeypatch.setattr(pipeline_audio.subprocess, "run", fake_run)

    pipeline_audio.extract_audio("input.mp4", str(tmp_path / "out.wav"))

    assert calls
    filter_chain = _extract_filter_arg(calls[0])
    assert "dynaudnorm" not in filter_chain

