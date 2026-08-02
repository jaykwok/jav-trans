from types import SimpleNamespace

from asr import chunking, local_backend, pipeline, result_cache, transcribe


def test_chunk_settings_refresh_between_calls(monkeypatch, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setenv("ASR_CHUNK_ROOT", str(first))
    monkeypatch.setenv("KEEP_ASR_CHUNKS", "0")
    assert chunking.current_asr_chunk_root() == first.resolve()
    assert pipeline.current_asr_chunk_root() == first.resolve()
    assert not chunking.keep_asr_chunks()

    monkeypatch.setenv("ASR_CHUNK_ROOT", str(second))
    monkeypatch.setenv("KEEP_ASR_CHUNKS", "1")
    assert chunking.current_asr_chunk_root() == second.resolve()
    assert pipeline.current_asr_chunk_root() == second.resolve()
    assert chunking.keep_asr_chunks()


def test_asr_language_refreshes_between_calls(monkeypatch):
    """It was a module-level constant read once at import.

    That made it the only ASR setting a persistent worker could not pick up
    between jobs, and the last standing excuse for reloading `asr.pipeline` to
    refresh the environment - a reload that forked the module's function objects
    away from the submodules it shares them with.
    """
    monkeypatch.setenv("ASR_LANGUAGE", "Japanese")
    assert pipeline._asr_language_for_chunking() == "Japanese"

    monkeypatch.setenv("ASR_LANGUAGE", "English")
    assert pipeline._asr_language_for_chunking() == "English"

    monkeypatch.setenv("ASR_LANGUAGE", "   ")
    assert pipeline._asr_language_for_chunking() == "Japanese"


def test_transcribe_settings_refresh_between_calls(monkeypatch):
    monkeypatch.setenv("ASR_RESULT_CACHE_ENABLED", "0")
    monkeypatch.setenv("ASR_INVALID_SEGMENT_DURATION", "0.2")
    monkeypatch.setenv("ASR_MIN_REPAIRED_SEGMENT_DURATION", "0.8")
    assert not result_cache.result_cache_enabled()
    assert transcribe._asr_invalid_segment_duration_s() == 0.2
    assert transcribe._asr_min_repaired_segment_duration_s() == 0.8

    monkeypatch.setenv("ASR_RESULT_CACHE_ENABLED", "1")
    monkeypatch.setenv("ASR_INVALID_SEGMENT_DURATION", "0.05")
    monkeypatch.setenv("ASR_MIN_REPAIRED_SEGMENT_DURATION", "0.5")
    assert result_cache.result_cache_enabled()
    assert transcribe._asr_invalid_segment_duration_s() == 0.05
    assert transcribe._asr_min_repaired_segment_duration_s() == 0.5


def test_generation_penalty_refreshes_between_models(monkeypatch):
    def model():
        return SimpleNamespace(
            generation_config=SimpleNamespace(
                do_sample=False,
                temperature=None,
                pad_token_id=1,
                eos_token_id=1,
                repetition_penalty=1.0,
            )
        )

    first = model()
    second = model()
    monkeypatch.setenv("ASR_REPETITION_PENALTY", "1.05")
    local_backend._apply_generation_safety(first)
    monkeypatch.setenv("ASR_REPETITION_PENALTY", "1.2")
    local_backend._apply_generation_safety(second)
    assert first.generation_config.repetition_penalty == 1.05
    assert second.generation_config.repetition_penalty == 1.2
