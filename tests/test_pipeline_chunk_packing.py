from __future__ import annotations

import importlib
import sys
import wave
from pathlib import Path

from audio.chunk_packer import PackedChunk
from boundary.base import SegmentationResult, SpeechSegment
from boundary.refiner import BoundaryDecision

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
        self.finalized_payloads: list[dict] = []

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
        self.finalized_payloads.extend(dict(result) for result in text_results)
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


class _FakeSequenceRefiner:
    feature_names = ("gap_s",)
    feature_schema_hash = "test-feature-schema"

    def decide_sequence(self, features: list[list[float]]) -> list[BoundaryDecision]:
        return [
            BoundaryDecision(
                source="frame_sequence_refiner",
                start_refine_delta_s=0.0,
                end_refine_delta_s=0.0,
            )
            for _ in features
        ]

    def signature(self) -> dict:
        return {"schema": "boundary_refiner_v5", "type": "fake_sequence_refiner"}


class _FakeSequenceFeatureProvider:
    def features_for_boundary(
        self,
        *,
        left_start_s: float,
        left_end_s: float,
        right_start_s: float,
        right_end_s: float,
    ) -> list[float]:
        del left_start_s, right_end_s
        return [right_start_s - left_end_s]

    def validate_for_checkpoint(self, feature_names, feature_schema_hash) -> None:
        del feature_names, feature_schema_hash

    def signature(self) -> dict:
        return {"schema": "speech_boundary_ja_sequence_feature_frames_v1", "type": "fake"}


def _reload_pipeline(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("BOUNDARY_FEATURE_FRAME_HOP_S", "0.02")
    monkeypatch.setenv("BOUNDARY_PLANNER_TARGET_CHUNK_S", "9.0")
    monkeypatch.setenv("BOUNDARY_PLANNER_MAX_CORE_CHUNK_S", "30.0")
    monkeypatch.setenv("ASR_CHUNK_ROOT", str(tmp_path / "chunks"))
    monkeypatch.setenv("ASR_CHECKPOINT_ENABLED", "0")
    monkeypatch.setenv("CUEQC_MODEL_PATH", "")

    from asr import pipeline as asr

    asr = importlib.reload(asr)
    monkeypatch.setattr(
        asr,
        "load_frame_sequence_refiner_checkpoint",
        lambda *_args, **_kwargs: _FakeSequenceRefiner(),
    )
    monkeypatch.setattr(
        asr,
        "_required_sequence_feature_provider_from_result",
        lambda *_args, **_kwargs: _FakeSequenceFeatureProvider(),
    )
    return asr


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


def test_boundary_planner_emits_one_asr_chunk_per_speech_island(monkeypatch, tmp_path):
    backend, _segments, log, details = _run_transcription(monkeypatch, tmp_path)

    assert len(backend.audio_paths) == 10
    assert details["chunk_count"] == len(backend.audio_paths)
    assert details["boundary_signature"]["backend"] == "stub"
    assert details["boundary_signature"]["boundary_pipeline"]["version"] == 5
    assert all(
        "source_boundary.wav" not in str(Path(path).name) for path in backend.audio_paths
    )
    chunk_log_entries = [entry for entry in log if entry.startswith("[chunk] idx=")]
    assert len(chunk_log_entries) == 10
    assert all("speech_segment_count=1" in entry for entry in chunk_log_entries)
    assert any("[chunk] idx=0" in entry and "speech_segment_count=1" in entry for entry in log)


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
    assert any("切分完成：共 0 个处理块" in entry for entry in log)


def test_packed_chunk_metadata_uses_source_span_index_after_short_chunk_drop():
    from asr import pipeline as asr

    skipped = PackedChunk(
        start=0.0,
        end=0.1,
        speech_segments=[SpeechSegment(0.0, 0.1, 0.1)],
        duration=0.1,
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
        split_reason="tail",
        parent_chunk_id=7,
        island_id=1,
        island_count=2,
        internal_gap_count=1,
        internal_gap_max_s=0.3,
        boundary_score=0.87,
        boundary_reason="utterance_switch",
        boundary_source="cut",
        boundary_start_refine_delta_s=0.04,
        boundary_end_refine_delta_s=-0.03,
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
    assert chunk_infos[0]["boundary_reason"] == "utterance_switch"
    assert chunk_infos[0]["boundary_source"] == "cut"
    assert chunk_infos[0]["boundary_score"] == 0.87
    assert chunk_infos[0]["boundary_start_refine_delta_s"] == 0.04
    assert chunk_infos[0]["boundary_end_refine_delta_s"] == -0.03
    assert chunk_infos[0]["boundary_decision_source"] == "frame_sequence_refiner"
    assert any("speech_segment_count=2" in entry and "source=cut" in entry for entry in log)


def test_alignment_fallback_window_metadata_uses_speech_core():
    from asr import pipeline as asr

    chunk = {
        "start": 10.0,
        "end": 20.0,
    }
    text_result = {
        "text": "テスト",
        "raw_text": "テスト",
        "duration": 10.0,
        "language": "Japanese",
        "normalized_path": "missing.wav",
        "log": [],
    }

    annotated = asr._with_alignment_fallback_window(chunk, text_result)

    assert annotated["alignment_fallback_start_s"] == 0.0
    assert annotated["alignment_fallback_end_s"] == 10.0
    assert annotated["alignment_fallback_source"] == "chunk"


def test_alignment_fallback_count_deduplicates_chunk_log_markers():
    from asr import pipeline as asr

    log = [
        "chunk 0: Alignment 回退窗口: speech_core",
        "chunk 1: Alignment 回退: 使用 VAD 约束比例时间戳",
        "chunk 1: Alignment VAD 回退语音区间: 2",
        "chunk 1: Alignment 回退窗口: speech_core",
        "chunk 2: Alignment 降级后仍异常: 改用等比分配时间戳",
        "chunk 2: Alignment VAD 回退异常: fallback_vad failed",
        "chunk 3: Alignment 哨兵触发: 时间轴异常，不重新调用 ASR，改用 VAD/比例回退",
        "chunk 3: Alignment 回退窗口: speech_core",
    ]

    assert asr._alignment_fallback_count_from_log(log) == 3


def test_alignment_outcome_metadata_is_written_to_chunks_and_segments():
    from asr import pipeline as asr

    chunk = {"index": 4, "start": 10.0, "end": 12.0}
    text_result = {
        "text": "こんにちは",
        "raw_text": "こんにちは",
        "duration": 2.0,
        "language": "Japanese",
    }
    chunk_words = [
        {"start": 0.1, "end": 0.4, "word": "こん"},
        {"start": 0.4, "end": 0.8, "word": "にちは"},
    ]
    outcome = asr._alignment_outcome_for_chunk(
        chunk=chunk,
        chunk_result={**text_result, "alignment_mode": "forced_aligner"},
        chunk_words=chunk_words,
        chunk_log=["Alignment 词数: 2", "Alignment 模式: forced_aligner"],
    )
    transcript = asr._build_transcript_chunks(
        [chunk],
        [text_result],
        {4: outcome},
    )
    segments = asr._annotate_segments_with_alignment_outcomes(
        [
            {
                "start": 10.1,
                "end": 10.8,
                "text": "こんにちは",
                "source_chunk_index": 4,
                "words": [
                    {
                        "start": 10.1,
                        "end": 10.4,
                        "word": "こん",
                        "source_chunk_index": 4,
                    }
                ],
            }
        ],
        {4: outcome},
    )

    assert outcome["alignment_quality"] == "forced"
    assert outcome["forced_success"] is True
    assert transcript[0]["alignment_quality"] == "forced"
    assert transcript[0]["fallback_subtype"] == "none"
    assert segments[0]["alignment_quality"] == "forced"
    assert segments[0]["source_chunk_indices"] == [4]


def test_low_logprob_chunks_continue_without_legacy_adaptive_review(monkeypatch, tmp_path):
    backend, segments, log, details = _run_transcription_with_backend(
        monkeypatch,
        tmp_path,
        backend=_LowLogprobBackend(),
    )

    assert segments
    assert backend.finalized_texts
    assert "asr_qc" not in details
    assert "asr_adaptive_review_chunks" not in details["stage_timings"]
    assert not any(entry.startswith("ASR Adaptive Precision:") for entry in log)
    assert all(chunk["text"] for chunk in details["transcript_chunks"])


def test_cueqc_shadow_records_without_skipping_alignment(monkeypatch, tmp_path):
    backend, segments, log, details = _run_transcription(monkeypatch, tmp_path)

    assert backend.finalized_payloads
    assert len(backend.finalized_payloads) == len(backend.audio_paths)
    assert details["cueqc_shadow"]["shadow_only"] is True
    assert details["cueqc_shadow"]["candidate_count"] == len(backend.audio_paths)
    assert details["cueqc_shadow"]["counts"]["display_hint"] == {"keep": len(backend.audio_paths)}
    assert details["transcript_chunks"]
    assert all("cueqc_shadow" in chunk for chunk in details["transcript_chunks"])
    assert segments
    assert all(segment.get("cueqc_shadow") for segment in segments)
    assert any(entry.startswith("CueQC: candidates=") for entry in log)


def test_cueqc_runtime_fallback_summary_is_logged_and_reported(monkeypatch, tmp_path):
    asr = _reload_pipeline(monkeypatch, tmp_path)

    class CaptureBackend:
        def capture_asr_internals(self, chunks):
            assert len(chunks) == 2
            return [
                {"ok": False, "error": "capture timeout"},
                {"ok": True},
            ]

    class FakeRefiner:
        def decide(self, candidates, *, asr_internals):
            del asr_internals
            return [
                {
                    "schema": "cueqc_shadow_v1",
                    "model_version": "cueqc_mamba_v3_fusion",
                    "decision_version": "cueqc_display_binary_v1",
                    "mode": "fallback_keep",
                    "display_hint": "keep",
                    "cluster_id": "runtime-test",
                    "confidence": 1.0,
                    "display_prob_keep": 1.0,
                    "display_prob_drop": 0.0,
                    "fallback_stage": "capture",
                    "fallback_detail": "capture timeout",
                    "reasons": ["cueqc_capture_error", "cueqc_capture_error:capture timeout"],
                },
                {
                    "schema": "cueqc_shadow_v1",
                    "model_version": "cueqc_mamba_v3_fusion",
                    "decision_version": "cueqc_display_binary_v1",
                    "mode": "cueqc_mamba_v3_fusion",
                    "display_hint": "drop",
                    "cluster_id": "runtime-test",
                    "confidence": 0.95,
                    "display_prob_keep": 0.05,
                    "display_prob_drop": 0.95,
                    "drop_threshold": 0.85,
                    "threshold_profile": {"mode": "base"},
                    "reasons": ["cueqc_mamba_v3:drop:p_drop=0.950:threshold=0.850"],
                },
            ]

    candidates = [
        {"chunk_index": 0, "start": 0.0, "end": 1.0, "duration_s": 1.0, "text": "あ"},
        {"chunk_index": 1, "start": 1.0, "end": 2.0, "duration_s": 1.0, "text": "い"},
    ]
    log: list[str] = []

    decisions = asr._apply_cueqc_v3_model(
        refiner=FakeRefiner(),
        candidates=candidates,
        backend=CaptureBackend(),
        audio_path=str(tmp_path / "source.wav"),
        log=log,
    )
    report = asr._merge_cueqc_v3_decisions(
        {"counts": {"display_hint": {"keep": 2}}, "decisions": []},
        candidates,
        decisions,
    )

    assert any("capture fallback candidates=1/2" in entry for entry in log)
    assert any("fallback=1" in entry and "cueqc_capture_error" in entry for entry in log)
    assert report["counts"]["display_hint"] == {"keep": 1, "drop": 1}
    assert report["counts"]["fallback_stage"] == {"capture": 1}
    assert report["counts"]["fallback_reason"] == {"cueqc_capture_error": 1}
    assert report["fallback_summary"]["details"] == {"capture timeout": 1}
