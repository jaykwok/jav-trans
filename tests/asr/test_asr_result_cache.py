from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

from helpers import ASR_17B_BACKEND

from asr import result_cache
from asr.result_cache import _cacheable_text_results


def _write_wav(path: Path, *, seconds: float = 0.05, rate: int = 16000, value: int = 0) -> None:
    frames = int(seconds * rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(struct.pack(f"<{frames}h", *([value] * frames)))


def _setup_cache_env(monkeypatch, tmp_path: Path) -> Path:
    cache_root = tmp_path / "asr_cache"
    monkeypatch.setenv("ASR_RESULT_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("ASR_BACKEND", ASR_17B_BACKEND)
    monkeypatch.delenv("ASR_RESULT_CACHE_ENABLED", raising=False)
    return cache_root


def _text_result(text: str = "テスト") -> dict:
    return {
        "text": text,
        "raw_text": text,
        "duration": 0.05,
        "language": "Japanese",
        "normalized_path": "somewhere/else.wav",
        "log": ["ASR 加载生成上限: 128"],
    }


def test_store_lookup_roundtrip_strips_path(monkeypatch, tmp_path):
    _setup_cache_env(monkeypatch, tmp_path)
    wav = tmp_path / "chunk.wav"
    _write_wav(wav, value=7)

    result_cache.store(wav, _text_result())
    cached = result_cache.lookup(wav)

    assert cached is not None
    assert cached["text"] == "テスト"
    assert "normalized_path" not in cached


def test_lookup_hits_for_same_pcm_at_different_path(monkeypatch, tmp_path):
    _setup_cache_env(monkeypatch, tmp_path)
    original = tmp_path / "job_a" / "chunk_0001.wav"
    rerun = tmp_path / "job_b" / "chunk_0042.wav"
    _write_wav(original, value=7)
    _write_wav(rerun, value=7)

    result_cache.store(original, _text_result())

    assert result_cache.lookup(rerun) is not None


def test_lookup_misses_for_different_pcm(monkeypatch, tmp_path):
    _setup_cache_env(monkeypatch, tmp_path)
    first = tmp_path / "a.wav"
    other = tmp_path / "b.wav"
    _write_wav(first, value=7)
    _write_wav(other, value=8)

    result_cache.store(first, _text_result())

    assert result_cache.lookup(other) is None


def test_key_changes_with_model_id_override(monkeypatch, tmp_path):
    _setup_cache_env(monkeypatch, tmp_path)
    wav = tmp_path / "chunk.wav"
    _write_wav(wav, value=7)

    result_cache.store(wav, _text_result())
    assert result_cache.lookup(wav) is not None

    monkeypatch.setenv("ASR_MODEL_ID", f"{ASR_17B_BACKEND}-local-tune")
    assert result_cache.lookup(wav) is None


def test_key_changes_with_generation_inputs(monkeypatch, tmp_path):
    _setup_cache_env(monkeypatch, tmp_path)
    wav = tmp_path / "chunk.wav"
    _write_wav(wav, value=7)

    result_cache.store(wav, _text_result())
    monkeypatch.setenv("ASR_MAX_NEW_TOKENS", "256")

    assert result_cache.lookup(wav) is None


def test_key_ignores_unrelated_env(monkeypatch, tmp_path):
    _setup_cache_env(monkeypatch, tmp_path)
    wav = tmp_path / "chunk.wav"
    _write_wav(wav, value=7)

    result_cache.store(wav, _text_result())
    monkeypatch.setenv("ASR_CONTEXT", "actor-b")
    monkeypatch.setenv("ASR_CHUNK_ROOT", str(tmp_path / "elsewhere"))

    assert result_cache.lookup(wav) is not None


def test_timeout_and_quarantined_results_never_stored(monkeypatch, tmp_path):
    cache_root = _setup_cache_env(monkeypatch, tmp_path)
    wav = tmp_path / "chunk.wav"
    _write_wav(wav, value=7)

    timed_out = _text_result("")
    timed_out["log"] = ["TIMEOUT: skipped after 300s"]
    result_cache.store(wav, timed_out)

    quarantined = _text_result("")
    quarantined["asr_generation"] = {"policy": "quarantined_result"}
    result_cache.store(wav, quarantined)

    assert result_cache.lookup(wav) is None
    assert not list(cache_root.rglob("*.json")) or all(
        path.name == "signature.json" for path in cache_root.rglob("*.json")
    )


def test_corrupt_entry_is_a_miss(monkeypatch, tmp_path):
    _setup_cache_env(monkeypatch, tmp_path)
    wav = tmp_path / "chunk.wav"
    _write_wav(wav, value=7)

    result_cache.store(wav, _text_result())
    entries = [
        path
        for path in Path(result_cache.result_cache_root()).rglob("*.json")
        if path.name != "signature.json"
    ]
    assert entries
    entries[0].write_text("{not json", encoding="utf-8")

    assert result_cache.lookup(wav) is None


def test_disabled_env_means_no_reads_or_writes(monkeypatch, tmp_path):
    cache_root = _setup_cache_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ASR_RESULT_CACHE_ENABLED", "0")
    wav = tmp_path / "chunk.wav"
    _write_wav(wav, value=7)

    result_cache.store(wav, _text_result())

    assert not cache_root.exists()
    assert result_cache.lookup(wav) is None


def test_restore_text_result_rehydrates_for_current_chunk(monkeypatch, tmp_path):
    _setup_cache_env(monkeypatch, tmp_path)
    wav = tmp_path / "chunk.wav"
    _write_wav(wav, seconds=0.1, value=7)

    result_cache.store(wav, _text_result())
    cached = result_cache.lookup(wav)
    restored = result_cache.restore_text_result(
        {"index": 3, "path": str(wav), "start": 12.0, "end": 12.1},
        cached,
    )

    assert restored["normalized_path"] == str(wav.resolve())
    assert restored["duration"] == 0.1
    assert any("ASR result cache hit" in entry for entry in restored["log"])


def test_cacheable_results_exclude_quarantined_and_timeout():
    text_results = {
        0: {"text": "ok", "log": [], "asr_generation": {"policy": "ok"}},
        1: {
            "text": "",
            "log": ["QUARANTINED: kind=timeout, respawn_count=3"],
            "asr_generation": {"policy": "quarantined_result"},
        },
        2: {"text": "ok2", "log": ["TIMEOUT: 180s"], "asr_generation": {}},
    }
    filtered = _cacheable_text_results(text_results)
    # Quarantined (1) and timed-out (2) are excluded; only the clean result (0)
    # persists, so quarantined chunks get re-transcribed on resume instead of
    # being silently restored as empty completed results.
    assert set(filtered.keys()) == {0}


class _CountingTextBackend:
    request_batch_size = 2

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def transcribe_texts(self, audio_paths, on_stage=None):
        self.calls.append(list(audio_paths))
        return [
            {"text": "テスト", "raw_text": "テスト", "duration": 0.05, "language": "Japanese"}
            for _ in audio_paths
        ]


def test_transcribe_loop_resumes_entirely_from_cache(monkeypatch, tmp_path):
    _setup_cache_env(monkeypatch, tmp_path)
    from asr import transcribe

    chunks = []
    for index in range(3):
        wav = tmp_path / f"chunk_{index}.wav"
        _write_wav(wav, value=index + 1)
        chunks.append(
            {
                "index": index,
                "start": float(index),
                "end": float(index) + 1.0,
                "path": str(wav),
                "source_audio_path": str(tmp_path / "src.wav"),
            }
        )

    first_backend = _CountingTextBackend()
    results, _ = transcribe._transcribe_asr_chunks_text_only(
        first_backend, chunks, "ASR 文本转写"
    )
    assert len(results) == 3
    assert first_backend.calls

    second_backend = _CountingTextBackend()
    resumed, _ = transcribe._transcribe_asr_chunks_text_only(
        second_backend, chunks, "ASR 文本转写"
    )
    assert len(resumed) == 3
    assert second_backend.calls == []
    assert all(
        any("ASR result cache hit" in entry for entry in result.get("log", []))
        for result in resumed
    )


def _setup_head_env(monkeypatch, tmp_path: Path) -> Path:
    head_path = tmp_path / "ctc_aligner.pt"
    head_path.write_bytes(b"head-weights-v1")
    monkeypatch.setenv("ASR_ALIGNMENT_HEAD_PATH", str(head_path))
    return head_path


def _aligned_result(mode: str = "ctc_forced_alignment") -> dict:
    return {
        "words": [{"start": 0.1, "end": 0.4, "word": "テスト"}],
        "text": "テスト",
        "raw_text": "テスト",
        "alignment_mode": mode,
        "duration": 0.05,
        "language": "Japanese",
    }


def test_finalize_cache_roundtrip_and_text_guard(monkeypatch, tmp_path):
    _setup_cache_env(monkeypatch, tmp_path)
    _setup_head_env(monkeypatch, tmp_path)
    wav = tmp_path / "chunk.wav"
    _write_wav(wav, value=7)

    result_cache.finalize_store(
        wav, text="テスト", result=_aligned_result(), log=["decoded", "aligned"]
    )
    cached = result_cache.finalize_lookup(wav, text="テスト")

    assert cached is not None
    restored_result, restored_log = cached
    assert restored_result["alignment_mode"] == "ctc_forced_alignment"
    assert restored_result["words"]
    assert any("finalize cache hit" in entry for entry in restored_log)
    # Different source text means different words: must miss.
    assert result_cache.finalize_lookup(wav, text="別のテキスト") is None


def test_finalize_cache_inert_without_head(monkeypatch, tmp_path):
    cache_root = _setup_cache_env(monkeypatch, tmp_path)
    monkeypatch.delenv("ASR_ALIGNMENT_HEAD_PATH", raising=False)
    wav = tmp_path / "chunk.wav"
    _write_wav(wav, value=7)

    result_cache.finalize_store(
        wav, text="テスト", result=_aligned_result(), log=[]
    )

    assert result_cache.finalize_lookup(wav, text="テスト") is None
    assert not cache_root.exists()


def test_finalize_cache_stores_only_real_alignments(monkeypatch, tmp_path):
    _setup_cache_env(monkeypatch, tmp_path)
    _setup_head_env(monkeypatch, tmp_path)
    wav = tmp_path / "chunk.wav"
    _write_wav(wav, value=7)

    result_cache.finalize_store(
        wav,
        text="テスト",
        result=_aligned_result(mode="boundary_proportional"),
        log=["Subtitle timing: alignment declined, using proportional"],
    )

    assert result_cache.finalize_lookup(wav, text="テスト") is None


def test_finalize_cache_key_changes_with_head_digest(monkeypatch, tmp_path):
    _setup_cache_env(monkeypatch, tmp_path)
    head_path = _setup_head_env(monkeypatch, tmp_path)
    wav = tmp_path / "chunk.wav"
    _write_wav(wav, value=7)

    result_cache.finalize_store(wav, text="テスト", result=_aligned_result(), log=[])
    assert result_cache.finalize_lookup(wav, text="テスト") is not None

    head_path.write_bytes(b"head-weights-v2-retrained")

    assert result_cache.finalize_lookup(wav, text="テスト") is None


def test_align_results_second_pass_served_from_finalize_cache(monkeypatch, tmp_path):
    _setup_cache_env(monkeypatch, tmp_path)
    _setup_head_env(monkeypatch, tmp_path)
    from asr import transcribe
    from tests.asr.test_alignment_head_wiring import _LifecycleBackend

    wav = tmp_path / "chunk_0000.wav"
    _write_wav(wav, value=7)
    text_result = {
        "text": "テスト",
        "raw_text": "テスト",
        "duration": 0.05,
        "language": "Japanese",
        "normalized_path": str(wav),
        "log": [],
    }

    first_backend = _LifecycleBackend(
        model=object(), finalize_mode="ctc_forced_alignment"
    )
    first_results, _ = transcribe._align_TRANSCRIPTION_results(
        first_backend, [dict(text_result)]
    )
    assert first_results[0][0]["alignment_mode"] == "ctc_forced_alignment"

    second_backend = _LifecycleBackend(model=None, finalize_mode="ctc_forced_alignment")
    second_results, _ = transcribe._align_TRANSCRIPTION_results(
        second_backend, [dict(text_result)]
    )

    assert second_results[0][0]["alignment_mode"] == "ctc_forced_alignment"
    assert any(
        "finalize cache hit" in entry for entry in second_results[0][1]
    )
    # No model load, no finalize call: the whole pass came from the cache.
    assert "load" not in second_backend.events
    assert not [e for e in second_backend.events if not isinstance(e, str)]
