from asr import transcribe


def test_asr_context_is_read_at_runtime(monkeypatch):
    monkeypatch.setenv("ASR_CONTEXT", "XXX")
    assert "XXX" in transcribe._build_ASR_CONTEXT_for_chunk({"start": 0})

    monkeypatch.setenv("ASR_CONTEXT", "")
    assert "XXX" not in transcribe._build_ASR_CONTEXT_for_chunk({"start": 0})


def test_asr_context_is_not_limited_to_video_head(monkeypatch):
    monkeypatch.setenv("ASR_CONTEXT", "小那海あや")

    assert transcribe._build_ASR_CONTEXT_for_chunk({"start": 0}) == "小那海あや"
    assert transcribe._build_ASR_CONTEXT_for_chunk({"start": 120}) == "小那海あや"
