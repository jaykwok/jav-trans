from __future__ import annotations

import importlib
import sys
import wave
from pathlib import Path

from audio.chunk_packer import PackedChunk
from boundary.base import SegmentationResult, SpeechSegment

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _write_wav(path: Path, seconds: float, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * int(sample_rate * seconds))


def _segments() -> list[SpeechSegment]:
    return [
        SpeechSegment(start=index * 1.0, end=index * 1.0 + 0.8)
        for index in range(10)
    ]


class _StubSpeechBoundaryBackend:
    name = "stub"

    def segment(
        self,
        audio_path: str,
        *,
        target_sr: int = 16000,
        threshold_override: float | None = None,
    ) -> SegmentationResult:
        del audio_path, target_sr, threshold_override
        segments = _segments()
        return SegmentationResult(
            segments=segments,
            groups=[[segment] for segment in segments],
            method=self.name,
            audio_duration_sec=10.0,
            parameters={"backend": self.name},
            processing_time_sec=0.0,
        )

    def signature(self) -> dict:
        return {"backend": self.name}


class _EmptyAllowedSpeechBoundaryBackend:
    name = "empty_allowed"

    def segment(
        self,
        audio_path: str,
        *,
        target_sr: int = 16000,
        threshold_override: float | None = None,
    ) -> SegmentationResult:
        del audio_path, target_sr, threshold_override
        return SegmentationResult(
            segments=[],
            groups=[],
            method=self.name,
            audio_duration_sec=10.0,
            parameters={"backend": self.name, "allow_empty": True},
            processing_time_sec=0.0,
        )

    def signature(self) -> dict:
        return {"backend": self.name, "allow_empty": True}


class _EmptyDisallowedSpeechBoundaryBackend:
    name = "empty_disallowed"

    def segment(
        self,
        audio_path: str,
        *,
        target_sr: int = 16000,
        threshold_override: float | None = None,
    ) -> SegmentationResult:
        del audio_path, target_sr, threshold_override
        return SegmentationResult(
            segments=[],
            groups=[],
            method=self.name,
            audio_duration_sec=10.0,
            parameters={"backend": self.name},
            processing_time_sec=0.0,
        )

    def signature(self) -> dict:
        return {"backend": self.name}


class _RecordingBackend:
    is_subprocess = False
    accepts_contexts = True
    timestamp_mode = "forced"
    request_batch_size = 1
    align_batch_size = 1

    def __init__(self) -> None:
        self.audio_paths: list[str] = []
        self.finalized_texts: list[str] = []

    def load(self, on_stage=None) -> None:
        return None

    def close(self) -> None:
        return None

    def unload_model(self, on_stage=None) -> None:
        return None

    def unload_forced_aligner(self, on_stage=None) -> None:
        return None

    def transcribe_texts(self, audio_paths, contexts=None, on_stage=None):
        del contexts, on_stage
        self.audio_paths.extend(audio_paths)
        results = []
        for index, path in enumerate(audio_paths):
            text = "怪しい怪しい怪しい怪しい怪しい" if index == 0 else "東京"
            results.append(
                {
                    "text": text,
                    "raw_text": text,
                    "avg_logprob": -0.3,
                    "no_speech_prob": 0.1,
                    "compression_ratio": 1.2,
                    "duration": 0.5,
                    "language": "Japanese",
                    "normalized_path": str(Path(path).resolve()),
                    "log": ["fake"],
                }
            )
        return results

    def finalize_text_results(self, text_results, on_stage=None):
        del on_stage
        self.finalized_texts.extend(result["text"] for result in text_results)
        return [
            (
                {
                    "words": [{"start": 0.0, "end": 0.5, "word": result["text"]}],
                    "text": result["text"],
                    "raw_text": result["raw_text"],
                    "alignment_mode": "fake",
                    "duration": result["duration"],
                    "language": result["language"],
                },
                ["Alignment 模式: fake"],
            )
            for result in text_results
        ]


class _LowLogprobBackend(_RecordingBackend):
    def transcribe_texts(self, audio_paths, contexts=None, on_stage=None):
        del contexts, on_stage
        self.audio_paths.extend(audio_paths)
        return [
            {
                "text": "低信頼テキスト",
                "raw_text": "低信頼テキスト",
                "avg_logprob": -1.5,
                "no_speech_prob": 0.1,
                "compression_ratio": 1.2,
                "duration": 0.5,
                "language": "Japanese",
                "normalized_path": str(Path(path).resolve()),
                "log": ["fake"],
            }
            for path in audio_paths
        ]


def _reload_pipeline(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("BOUNDARY_FEATURE_FRAME_HOP_S", "0.02")
    monkeypatch.setenv("BOUNDARY_PLANNER_TARGET_PADDING_S", "0")
    monkeypatch.setenv("BOUNDARY_PLANNER_TARGET_CHUNK_S", "9.0")
    monkeypatch.setenv("BOUNDARY_PLANNER_MAX_CHUNK_S", "30.0")
    monkeypatch.setenv("ASR_CHUNK_ROOT", str(tmp_path / "chunks"))
    monkeypatch.setenv("ASR_CHECKPOINT_ENABLED", "0")

    from asr import pipeline as asr

    return importlib.reload(asr)


def _run_transcription(monkeypatch, tmp_path: Path):
    asr = _reload_pipeline(monkeypatch, tmp_path)
    source = tmp_path / "source_boundary.wav"
    _write_wav(source, seconds=12.0)

    import boundary

    monkeypatch.setattr(boundary, "get_boundary_backend", lambda: _StubSpeechBoundaryBackend())
    backend = _RecordingBackend()
    monkeypatch.setattr(asr, "_resolve_asr_backend", lambda _device: backend)

    segments, log, details = asr._transcribe_and_align_local(str(source), "cpu")
    return backend, segments, log, details


def _run_transcription_with_backend(
    monkeypatch,
    tmp_path: Path,
    *,
    backend,
):
    asr = _reload_pipeline(monkeypatch, tmp_path)
    source = tmp_path / "source_adaptive_boundary.wav"
    _write_wav(source, seconds=12.0)

    import boundary

    monkeypatch.setattr(boundary, "get_boundary_backend", lambda: _StubSpeechBoundaryBackend())
    monkeypatch.setattr(asr, "_resolve_asr_backend", lambda _device: backend)

    segments, log, details = asr._transcribe_and_align_local(str(source), "cpu")
    return backend, segments, log, details


def test_boundary_planner_packs_speech_segments_before_transcribe(monkeypatch, tmp_path):
    backend, _segments, log, details = _run_transcription(monkeypatch, tmp_path)

    assert len(backend.audio_paths) < 10
    assert details["chunk_count"] == len(backend.audio_paths)
    assert any("[chunk] idx=0" in entry and "speech_segment_count=10" in entry for entry in log)


def test_empty_allowed_boundary_does_not_fallback_to_full_audio(monkeypatch, tmp_path):
    asr = _reload_pipeline(monkeypatch, tmp_path)
    source = tmp_path / "source_empty_allowed.wav"
    _write_wav(source, seconds=12.0)

    import boundary

    monkeypatch.setattr(boundary, "get_boundary_backend", lambda: _EmptyAllowedSpeechBoundaryBackend())
    backend = _RecordingBackend()
    monkeypatch.setattr(asr, "_resolve_asr_backend", lambda _device: backend)

    segments, log, details = asr._transcribe_and_align_local(str(source), "cpu")

    assert backend.audio_paths == []
    assert segments == []
    assert details["chunk_count"] == 0
    assert any("切分完成：共 0 个处理块" in entry for entry in log)


def test_empty_disallowed_boundary_skips_asr_without_full_audio_fallback(monkeypatch, tmp_path):
    asr = _reload_pipeline(monkeypatch, tmp_path)
    source = tmp_path / "source_empty_disallowed.wav"
    _write_wav(source, seconds=12.0)

    import boundary

    monkeypatch.setattr(boundary, "get_boundary_backend", lambda: _EmptyDisallowedSpeechBoundaryBackend())
    backend = _RecordingBackend()
    monkeypatch.setattr(asr, "_resolve_asr_backend", lambda _device: backend)

    segments, log, details = asr._transcribe_and_align_local(str(source), "cpu")

    assert segments == []
    assert backend.audio_paths == []
    assert backend.finalized_texts == []
    assert details["chunk_count"] == 0
    assert details["transcript_chunks"] == []
    assert details["asr_qc"]["dropped_uncertain_count"] == 0
    assert any("切分完成：共 0 个处理块" in entry for entry in log)


def test_packed_chunk_metadata_uses_source_span_index_after_short_chunk_drop():
    from asr import pipeline as asr

    skipped = PackedChunk(
        start=0.0,
        end=0.1,
        speech_segments=[SpeechSegment(0.0, 0.1, 0.1)],
        duration=0.1,
        left_padding_s=0.0,
        right_padding_s=0.0,
        split_reason="skipped",
        boundary_score=0.1,
        boundary_reason="wrong",
        boundary_source="wrong",
    )
    kept = PackedChunk(
        start=1.0,
        end=2.0,
        speech_segments=[
            SpeechSegment(1.0, 1.3, 0.8),
            SpeechSegment(1.6, 2.0, 0.9),
        ],
        duration=1.0,
        left_padding_s=0.1,
        right_padding_s=0.2,
        split_reason="tail",
        parent_chunk_id=7,
        island_id=1,
        island_count=2,
        internal_gap_count=1,
        internal_gap_max_s=0.3,
        boundary_score=0.87,
        boundary_reason="speaker_change",
        boundary_source="cut",
        boundary_decision_merge=False,
        boundary_merge_prob=0.13,
        boundary_split_prob=0.87,
        boundary_decision_source="frame_sequence_refiner",
    )
    chunk_infos = [
        {
            "index": 0,
            "source_span_index": 1,
            "start": kept.start,
            "end": kept.end,
        }
    ]
    log: list[str] = []

    asr._annotate_packed_chunks(chunk_infos, [skipped, kept], log)

    assert chunk_infos[0]["speech_segment_count"] == 2
    assert chunk_infos[0]["boundary_parent_chunk_id"] == 7
    assert chunk_infos[0]["boundary_reason"] == "speaker_change"
    assert chunk_infos[0]["boundary_source"] == "cut"
    assert chunk_infos[0]["boundary_score"] == 0.87
    assert chunk_infos[0]["boundary_decision_merge"] is False
    assert chunk_infos[0]["boundary_merge_prob"] == 0.13
    assert chunk_infos[0]["boundary_split_prob"] == 0.87
    assert chunk_infos[0]["boundary_decision_source"] == "frame_sequence_refiner"
    assert any("speech_segment_count=2" in entry and "source=cut" in entry for entry in log)


def test_alignment_fallback_count_deduplicates_chunk_log_markers():
    from asr import pipeline as asr

    log = [
        "chunk 1: Alignment 回退: 使用 VAD 约束比例时间戳",
        "chunk 1: Alignment VAD 回退语音区间: 2",
        "chunk 2: Alignment 降级后仍异常: 改用等比分配时间戳",
        "chunk 2: Alignment VAD 回退异常: fallback_vad failed",
    ]

    assert asr._alignment_fallback_count_from_log(log) == 2


def test_adaptive_precision_drops_low_logprob_before_alignment(monkeypatch, tmp_path):
    monkeypatch.setenv("ASR_QC_ADAPTIVE_BASE_LOGPROB", "-1.0")
    monkeypatch.setenv("ASR_QC_DROP_UNCERTAIN", "1")
    backend, segments, log, details = _run_transcription_with_backend(
        monkeypatch,
        tmp_path,
        backend=_LowLogprobBackend(),
    )

    assert segments == []
    assert backend.finalized_texts == []
    assert details["asr_qc"]["dropped_uncertain_count"] == len(backend.audio_paths)
    assert details["stage_timings"]["asr_adaptive_dropped_chunks"] == len(backend.audio_paths)
    assert any(entry.startswith("ASR Adaptive Precision: dropped_uncertain=") for entry in log)
    assert all(chunk["text"] == "" for chunk in details["transcript_chunks"])
