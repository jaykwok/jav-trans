from __future__ import annotations

from pathlib import Path

import pytest

from asr import transcribe


class _SplitBackend:
    is_subprocess = False
    accepts_contexts = True
    request_batch_size = 1
    align_batch_size = 1

    def __init__(self):
        self.unload_forced_aligner_calls = 0

    def unload_forced_aligner(self, on_stage=None):
        del on_stage
        self.unload_forced_aligner_calls += 1


def _chunk(tmp_path: Path) -> tuple[dict, Path]:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    chunk_audio = tmp_path / "chunk.wav"
    chunk_audio.write_bytes(b"chunk")
    return (
        {
            "index": 0,
            "start": 10.0,
            "end": 20.0,
            "path": str(chunk_audio),
            "source_audio_path": str(source),
        },
        source,
    )


def _sentinel_result() -> dict:
    return {
        "words": [{"start": 0.0, "end": 0.0, "word": "長いテキスト"}],
        "text": "これは十分に長いテキストです",
        "raw_text": "これは十分に長いテキストです",
        "alignment_mode": "forced_aligner",
        "duration": 10.0,
        "language": "Japanese",
    }


def test_sentinel_island_split_default_off(monkeypatch, tmp_path):
    chunk, source = _chunk(tmp_path)

    monkeypatch.delenv("ALIGNMENT_SENTINEL_ISLAND_SPLIT", raising=False)

    backend = _SplitBackend()
    words, log = transcribe._split_alignment_sentinel_with_speech_islands(
        backend,
        str(source),
        chunk,
        _sentinel_result(),
    )

    assert words == []
    assert log == []


def test_sentinel_island_split_success_offsets_words(monkeypatch, tmp_path):
    chunk, source = _chunk(tmp_path)
    extracted_spans: list[list[tuple[float, float]]] = []
    prepared_chunks: list[dict] = []

    monkeypatch.setenv("ALIGNMENT_SENTINEL_ISLAND_SPLIT", "1")
    monkeypatch.setenv("ASR_CHUNK_PACK_FRAME_HOP_S", "0.02")
    monkeypatch.setattr(
        transcribe,
        "_build_alignment_sentinel_island_plan",
        lambda *_args, **_kwargs: {
            "source_path": str(source),
            "chunk_start": 10.0,
            "duration": 10.0,
            "island_spans": [(10.88, 12.12), (14.88, 16.12)],
            "log": [],
        },
    )

    def fake_extract(_source_path, spans):
        extracted_spans.append(list(spans))
        split_dir = tmp_path / "split"
        split_dir.mkdir(exist_ok=True)
        infos = [
            {"index": idx, "start": start, "end": end, "path": str(tmp_path / f"{idx}.wav")}
            for idx, (start, end) in enumerate(spans)
        ]
        prepared_chunks.extend(infos)
        return split_dir, infos

    def fake_prepare(_backend, chunks, _label, on_stage=None):
        del on_stage
        assert chunks == prepared_chunks
        return [
            (
                {
                    "words": [{"start": 0.10, "end": 0.40, "word": "これ"}],
                    "text": "これ",
                    "alignment_mode": "forced_aligner",
                },
                ["Alignment 模式: forced_aligner"],
            ),
            (
                {
                    "words": [{"start": 0.20, "end": 0.50, "word": "です"}],
                    "text": "です",
                    "alignment_mode": "forced_aligner",
                },
                ["Alignment 模式: forced_aligner"],
            ),
        ], {"total_s": 0.0}

    monkeypatch.setattr(transcribe, "_extract_wav_chunks", fake_extract)
    monkeypatch.setattr(transcribe, "_prepare_asr_chunk_results", fake_prepare)
    monkeypatch.setattr(transcribe, "_delete_path_for_cleanup", lambda _path: None)

    backend = _SplitBackend()
    words, log = transcribe._split_alignment_sentinel_with_speech_islands(
        backend,
        str(source),
        chunk,
        _sentinel_result(),
    )

    assert len(extracted_spans) == 1
    flat_spans = [value for span in extracted_spans[0] for value in span]
    assert flat_spans == pytest.approx([10.88, 12.12, 14.88, 16.12])
    assert [word["word"] for word in words] == ["これ", "です"]
    flat_word_times = [value for word in words for value in (word["start"], word["end"])]
    assert flat_word_times == pytest.approx([0.98, 1.28, 5.08, 5.38])
    assert any("Alignment speech-island split 成功" in line for line in log)
    assert not any("Alignment 哨兵触发" in line for line in log)
    assert backend.unload_forced_aligner_calls >= 2


def test_finalize_uses_island_split_before_vad_fallback(monkeypatch, tmp_path):
    chunk, source = _chunk(tmp_path)
    result = _sentinel_result()
    split_words = [{"start": 1.0, "end": 1.4, "word": "修正"}]

    monkeypatch.setattr(
        transcribe,
        "_split_alignment_sentinel_with_speech_islands",
        lambda *args, **kwargs: (split_words, ["Alignment speech-island split 成功: words=1"]),
    )
    monkeypatch.setattr(
        transcribe,
        "_build_timestamp_fallback",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fallback should not run")),
    )

    words, log = transcribe._finalize_aligned_chunk_without_asr_retry(
        chunk,
        result,
        ["Alignment 模式: forced_aligner"],
        backend=_SplitBackend(),
        source_audio_path=str(source),
    )

    assert words == split_words
    assert any("Alignment speech-island split 成功" in line for line in log)
    assert not any("Alignment 哨兵触发" in line for line in log)


def test_finalize_keeps_old_fallback_when_island_split_fails(monkeypatch, tmp_path):
    chunk, source = _chunk(tmp_path)
    result = _sentinel_result()

    monkeypatch.setattr(
        transcribe,
        "_split_alignment_sentinel_with_speech_islands",
        lambda *args, **kwargs: ([], ["Alignment speech-island split 失败: words=0"]),
    )
    monkeypatch.setattr(
        transcribe,
        "_build_timestamp_fallback",
        lambda *args, **kwargs: (
            [{"start": 0.0, "end": 10.0, "word": "fallback"}],
            "aligner_vad_fallback",
            {"speech_span_count": 2, "vad_error": ""},
        ),
    )

    words, log = transcribe._finalize_aligned_chunk_without_asr_retry(
        chunk,
        result,
        ["Alignment 模式: forced_aligner"],
        backend=_SplitBackend(),
        source_audio_path=str(source),
    )

    assert words == [{"start": 0.0, "end": 10.0, "word": "fallback"}]
    assert any("Alignment speech-island split 失败" in line for line in log)
    assert any("Alignment 哨兵触发" in line for line in log)
    assert any("Alignment 回退: 使用 VAD 约束比例时间戳" in line for line in log)


def test_batch_sentinel_island_split_uses_single_prepare_call(monkeypatch, tmp_path):
    first, source = _chunk(tmp_path)
    second = dict(first)
    second["index"] = 1
    second["start"] = 30.0
    second["end"] = 40.0
    second_audio = tmp_path / "chunk2.wav"
    second_audio.write_bytes(b"chunk2")
    second["path"] = str(second_audio)
    chunks = [first, second]
    extracted_spans: list[list[tuple[float, float]]] = []
    prepare_calls: list[list[dict]] = []

    monkeypatch.setenv("ALIGNMENT_SENTINEL_ISLAND_SPLIT", "1")
    monkeypatch.setenv("ASR_CHUNK_PACK_FRAME_HOP_S", "0.02")
    monkeypatch.setattr(
        transcribe,
        "_build_alignment_sentinel_island_plan",
        lambda _source_audio_path, chunk, *_args, **_kwargs: {
            "source_path": str(source),
            "chunk_start": float(chunk["start"]),
            "duration": float(chunk["end"]) - float(chunk["start"]),
            "island_spans": [
                (float(chunk["start"]) + 0.88, float(chunk["start"]) + 2.12),
                (float(chunk["start"]) + 4.88, float(chunk["start"]) + 6.12),
            ],
            "log": [],
        },
    )

    def fake_extract(_source_path, spans):
        extracted_spans.append(list(spans))
        split_dir = tmp_path / f"split_{len(extracted_spans)}"
        split_dir.mkdir(exist_ok=True)
        infos = [
            {"index": idx, "start": start, "end": end, "path": str(split_dir / f"{idx}.wav")}
            for idx, (start, end) in enumerate(spans)
        ]
        return split_dir, infos

    def fake_prepare(_backend, split_chunks, _label, on_stage=None):
        del on_stage
        prepare_calls.append([dict(chunk) for chunk in split_chunks])
        return [
            (
                {
                    "words": [{"start": 0.10, "end": 0.40, "word": f"w{idx}"}],
                    "text": f"w{idx}",
                    "alignment_mode": "forced_aligner",
                },
                ["Alignment 模式: forced_aligner"],
            )
            for idx, _chunk in enumerate(split_chunks)
        ], {"total_s": 0.0}

    monkeypatch.setattr(transcribe, "_extract_wav_chunks", fake_extract)
    monkeypatch.setattr(transcribe, "_prepare_asr_chunk_results", fake_prepare)
    monkeypatch.setattr(transcribe, "_delete_path_for_cleanup", lambda _path: None)

    backend = _SplitBackend()
    words_by_index, logs_by_index, attempted = (
        transcribe._split_alignment_sentinels_with_speech_islands_batch(
            backend,
            str(source),
            chunks,
            [(_sentinel_result(), []), (_sentinel_result(), [])],
        )
    )

    assert attempted is True
    assert len(extracted_spans) == 2
    assert len(prepare_calls) == 1
    assert len(prepare_calls[0]) == 4
    assert set(words_by_index) == {0, 1}
    assert [word["word"] for word in words_by_index[0]] == ["w0", "w1"]
    assert [word["word"] for word in words_by_index[1]] == ["w2", "w3"]
    assert logs_by_index[0][-1].startswith("Alignment speech-island split 成功")
    assert logs_by_index[1][-1].startswith("Alignment speech-island split 成功")
    assert backend.unload_forced_aligner_calls >= 2


def test_batch_sentinel_island_split_reports_skipped_candidates(monkeypatch, tmp_path):
    chunk, source = _chunk(tmp_path)

    monkeypatch.setenv("ALIGNMENT_SENTINEL_ISLAND_SPLIT", "1")
    monkeypatch.setattr(
        transcribe,
        "_build_alignment_sentinel_island_plan",
        lambda *_args, **_kwargs: {
            "source_path": str(source),
            "chunk_start": 10.0,
            "duration": 10.0,
            "island_spans": [],
            "log": ["Alignment speech-island split 跳过: island_count<=1"],
        },
    )
    monkeypatch.setattr(
        transcribe,
        "_prepare_asr_chunk_results",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not prepare")),
    )

    words_by_index, logs_by_index, attempted = (
        transcribe._split_alignment_sentinels_with_speech_islands_batch(
            _SplitBackend(),
            str(source),
            [chunk],
            [(_sentinel_result(), [])],
        )
    )

    assert attempted is True
    assert words_by_index == {}
    assert any("island_count<=1" in line for line in logs_by_index[0])
