from whisper import transcribe


def test_asr_context_is_read_at_runtime(monkeypatch):
    monkeypatch.setenv("ASR_CONTEXT", "XXX")
    assert "XXX" in transcribe._build_ASR_CONTEXT_for_chunk({"start": 0})

    monkeypatch.setenv("ASR_CONTEXT", "")
    assert "XXX" not in transcribe._build_ASR_CONTEXT_for_chunk({"start": 0})


def test_asr_head_context_is_read_at_runtime(monkeypatch):
    monkeypatch.setenv("ASR_HEAD_CONTEXT", "HEAD")
    monkeypatch.setattr(transcribe, "_ASR_HEAD_CONTEXT_MAX_START_S", 1.0)
    assert "HEAD" in transcribe._build_ASR_CONTEXT_for_chunk({"start": 0})

    monkeypatch.setenv("ASR_HEAD_CONTEXT", "")
    assert "HEAD" not in transcribe._build_ASR_CONTEXT_for_chunk({"start": 0})
