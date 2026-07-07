from __future__ import annotations

import importlib
import json
import sys
import wave
from pathlib import Path

import numpy as np

from audio.chunk_packer import PackedChunk
from boundary.base import SegmentationResult, SpeechSegment
from boundary.sequence_features import (
    CHUNK_POOLED_PTM_SCHEMA,
    DEFAULT_CHUNK_POOLED_PTM_BINS,
    chunk_pooled_ptm_feature_names,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asr.pre_asr_cueqc import PRE_ASR_CUEQC_PTM_BINS, PRE_ASR_CUEQC_PTM_DIM


def _pre_asr_ptm_feature_names() -> list[str]:
    return chunk_pooled_ptm_feature_names(
        ptm_dim=PRE_ASR_CUEQC_PTM_DIM,
        bins=PRE_ASR_CUEQC_PTM_BINS,
    )


def _pre_asr_ptm_values() -> list[float]:
    return [0.0] * len(_pre_asr_ptm_feature_names())


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
            parameters={
                "backend": self.name,
                "frame_scores": [0.9] * 500,
                "candidate_frame_scores": [0.1] * 500,
                "frame_hop_s": 0.02,
                "sequence_feature_frames": {"schema": "test"},
            },
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
    request_batch_size = 1

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

    def transcribe_texts(self, audio_paths, on_stage=None):
        del on_stage
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
                ["Subtitle timing mode: fake"],
            )
            for result in text_results
        ]


class _LowLogprobBackend(_RecordingBackend):
    def transcribe_texts(self, audio_paths, on_stage=None):
        del on_stage
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


class _FakeBoundaryModel:
    def __init__(self, schema: str) -> None:
        self.schema = schema

    def signature(self) -> dict:
        return {"schema": self.schema, "type": "fake"}


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

    def chunk_pooled_ptm_feature_names(
        self,
        *,
        bins: int = PRE_ASR_CUEQC_PTM_BINS,
    ) -> list[str]:
        return chunk_pooled_ptm_feature_names(
            ptm_dim=PRE_ASR_CUEQC_PTM_DIM,
            bins=bins,
        )

    def chunk_pooled_ptm_features(
        self,
        *,
        start_s: float,
        end_s: float,
        bins: int = PRE_ASR_CUEQC_PTM_BINS,
    ) -> list[float]:
        del start_s, end_s
        return [0.0] * len(self.chunk_pooled_ptm_feature_names(bins=bins))


def _reload_pipeline(monkeypatch, tmp_path: Path, *, enable_cueqc: bool = False):
    asr_backend = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
    outer_checkpoint = tmp_path / "outer.pt"
    split_checkpoint = tmp_path / "split.pt"
    cut_checkpoint = tmp_path / "cut.pt"
    for path in (outer_checkpoint, split_checkpoint, cut_checkpoint):
        path.write_bytes(b"checkpoint")
    monkeypatch.setenv("ASR_BACKEND", asr_backend)
    monkeypatch.setenv(
        "OUTER_EDGE_REFINER_MODEL_PATH_BY_REPO",
        f"{asr_backend}={outer_checkpoint}",
    )
    monkeypatch.setenv(
        "SEMANTIC_SPLIT_MODEL_PATH_BY_REPO",
        f"{asr_backend}={split_checkpoint}",
    )
    monkeypatch.setenv(
        "CUT_EDGE_REFINER_MODEL_PATH_BY_REPO",
        f"{asr_backend}={cut_checkpoint}",
    )
    monkeypatch.setenv("BOUNDARY_FEATURE_FRAME_HOP_S", "0.02")
    monkeypatch.setenv("ASR_CHUNK_ROOT", str(tmp_path / "chunks"))
    monkeypatch.setenv("ASR_CHECKPOINT_ENABLED", "0")
    if enable_cueqc:
        monkeypatch.setenv("CUEQC_SHADOW_ENABLED", "1")
    else:
        monkeypatch.delenv("CUEQC_SHADOW_ENABLED", raising=False)
    monkeypatch.delenv("CUEQC_MODEL_PATH_BY_REPO", raising=False)

    from asr import pipeline as asr

    asr = importlib.reload(asr)
    monkeypatch.setattr(asr, "load_outer_edge_refiner", lambda *_args, **_kwargs: _FakeBoundaryModel("outer_edge_refiner_v1"))
    monkeypatch.setattr(asr, "load_semantic_split_verifier", lambda *_args, **_kwargs: _FakeBoundaryModel("semantic_split_verifier_v1"))
    monkeypatch.setattr(asr, "load_cut_edge_refiner", lambda *_args, **_kwargs: _FakeBoundaryModel("cut_edge_refiner_v1"))
    monkeypatch.setattr(
        asr,
        "_required_sequence_feature_provider_from_result",
        lambda *_args, **_kwargs: _FakeSequenceFeatureProvider(),
    )
    monkeypatch.setattr(
        asr,
        "build_semantic_boundary_chunks",
        lambda segments, **_kwargs: [
            PackedChunk(
                start=segment.start,
                end=segment.end,
                source_abs_start=segment.start,
                source_abs_end=segment.end,
                speech_segments=[segment],
                duration=segment.end - segment.start,
                split_reason="semantic_boundary",
                boundary_source="speech_island",
                pre_asr_ptm_pooling_schema=CHUNK_POOLED_PTM_SCHEMA,
                pre_asr_ptm_pooling_bins=PRE_ASR_CUEQC_PTM_BINS,
                pre_asr_ptm_pooling_dim=len(_pre_asr_ptm_feature_names()),
                pre_asr_ptm_pooled_features=_pre_asr_ptm_values(),
            )
            for segment in segments
        ],
    )
    return asr


def _run_transcription(monkeypatch, tmp_path: Path, *, enable_cueqc: bool = False):
    asr = _reload_pipeline(monkeypatch, tmp_path, enable_cueqc=enable_cueqc)
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
    assert details["boundary_signature"]["boundary_pipeline"]["version"] == 9
    assert all(
        "source_boundary.wav" not in str(Path(path).name) for path in backend.audio_paths
    )
    chunk_log_entries = [entry for entry in log if entry.startswith("[chunk] idx=")]
    assert len(chunk_log_entries) == 10
    assert all("speech_segment_count=1" in entry for entry in chunk_log_entries)
    assert any("[chunk] idx=0" in entry and "speech_segment_count=1" in entry for entry in log)


def test_pipeline_persists_pre_asr_candidates_before_transcription(monkeypatch, tmp_path):
    backend, _segments, _log, details = _run_transcription(monkeypatch, tmp_path)

    candidates = details["pre_asr_candidates"]
    assert len(candidates) == len(backend.audio_paths)
    assert candidates[0]["sample_id"] == "preasr-source_boundary-chunk00000"
    assert candidates[0]["video_id"] == "source_boundary"
    assert candidates[0]["feature_names"]
    assert "text" not in " ".join(candidates[0]["feature_names"]).lower()


def test_pipeline_exports_pre_asr_candidates_only_when_requested(monkeypatch, tmp_path):
    export_path = tmp_path / "pre_asr_candidates.jsonl"
    monkeypatch.setenv("PRE_ASR_CUEQC_EXPORT_CANDIDATES_PATH", str(export_path))
    monkeypatch.setenv("PRE_ASR_CUEQC_EXPORT_CANDIDATES_APPEND", "0")

    backend, _segments, log, _details = _run_transcription(monkeypatch, tmp_path)

    rows = [
        json.loads(line)
        for line in export_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == len(backend.audio_paths)
    assert rows[0]["sample_id"] == "preasr-source_boundary-chunk00000"
    assert rows[0]["feature_names"]
    assert any("Pre-ASR CueQC: exported candidates" in entry for entry in log)


def test_pre_asr_candidates_accept_numpy_pooled_features(monkeypatch, tmp_path):
    asr = _reload_pipeline(monkeypatch, tmp_path)
    source = tmp_path / "source_boundary.wav"
    values = np.asarray(_pre_asr_ptm_values(), dtype=np.float32)
    span = PackedChunk(
        start=0.0,
        end=0.8,
        source_abs_start=0.0,
        source_abs_end=0.8,
        speech_segments=[SpeechSegment(0.0, 0.8, 0.9)],
        duration=0.8,
        split_reason="semantic_boundary",
        boundary_source="speech_island",
        pre_asr_ptm_pooling_schema=CHUNK_POOLED_PTM_SCHEMA,
        pre_asr_ptm_pooling_bins=PRE_ASR_CUEQC_PTM_BINS,
        pre_asr_ptm_pooling_dim=len(values),
        pre_asr_ptm_pooled_features=values,
    )

    candidates = asr._pre_asr_candidates_for_spans(str(source), [span])

    assert isinstance(candidates[0]["pre_asr_ptm_pooled_features"], list)
    assert candidates[0]["ptm_pooling_available"] is True
    json.dumps(candidates, ensure_ascii=False)


def test_pre_asr_candidate_export_sanitizes_numpy_values(monkeypatch, tmp_path):
    import asr.pipeline as asr

    asr = importlib.reload(asr)
    export_path = tmp_path / "pre_asr_candidates.jsonl"
    monkeypatch.setenv("PRE_ASR_CUEQC_EXPORT_CANDIDATES_PATH", str(export_path))
    monkeypatch.setenv("PRE_ASR_CUEQC_EXPORT_CANDIDATES_APPEND", "0")
    log: list[str] = []

    asr._write_pre_asr_candidates_if_requested(
        [{"sample_id": "numpy", "scores": np.asarray([0.25, 0.75], dtype=np.float32)}],
        log=log,
    )

    row = json.loads(export_path.read_text(encoding="utf-8").strip())
    assert row["scores"] == [0.25, 0.75]
    assert any("Pre-ASR CueQC: exported candidates" in entry for entry in log)


def test_json_payload_summarizes_large_numpy_arrays(monkeypatch, tmp_path):
    asr = _reload_pipeline(monkeypatch, tmp_path)
    payload = asr._json_payload(
        {
            "small": np.asarray([1.0, 2.0], dtype=np.float32),
            "large": np.ones((65, 65), dtype=np.float32),
        }
    )

    assert payload["small"] == [1.0, 2.0]
    assert payload["large"]["array_type"] == "ndarray"
    assert payload["large"]["dtype"] == "float32"
    assert payload["large"]["shape"] == [65, 65]
    assert len(payload["large"]["sha256"]) == 64


def test_pipeline_export_candidates_append_disabled_only_overwrites_once(monkeypatch, tmp_path):
    import asr.pipeline as asr

    asr = importlib.reload(asr)
    export_path = tmp_path / "pre_asr_candidates.jsonl"
    monkeypatch.setenv("PRE_ASR_CUEQC_EXPORT_CANDIDATES_PATH", str(export_path))
    monkeypatch.setenv("PRE_ASR_CUEQC_EXPORT_CANDIDATES_APPEND", "0")

    first_log: list[str] = []
    second_log: list[str] = []
    asr._write_pre_asr_candidates_if_requested(
        [{"sample_id": "first", "chunk_index": 0}],
        log=first_log,
    )
    asr._write_pre_asr_candidates_if_requested(
        [{"sample_id": "second", "chunk_index": 1}],
        log=second_log,
    )

    rows = [
        json.loads(line)
        for line in export_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["sample_id"] for row in rows] == ["first", "second"]
    assert any("mode=w" in entry for entry in first_log)
    assert any("mode=a" in entry for entry in second_log)


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
        boundary_decision_source="edge_sequence_refiner_v8",
        subtitle_min_duration_s=20.0 / 24.0,
        below_subtitle_min_duration=False,
        micro_chunk_candidate=True,
        micro_resolve_action="merge_micro_into_left",
        micro_resolve_reason="left_split_weaker",
        left_split_score=0.31,
        right_split_score=0.82,
        primary_cut_candidates=[
            {
                "kind": "primary",
                "time_s": 1.5,
                "frame": 75,
                "score": 0.82,
                "prominence": 0.2,
                "speech_valley": 0.7,
                "strength": 1.72,
            }
        ],
        weak_cut_candidates=[
            {
                "kind": "weak",
                "time_s": 1.8,
                "frame": 90,
                "score": 0.3,
                "prominence": 0.12,
                "speech_valley": 0.5,
                "strength": 0.92,
            }
        ],
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
    assert chunk_infos[0]["boundary_decision_source"] == "edge_sequence_refiner_v8"
    assert chunk_infos[0]["subtitle_min_duration_s"] == 20.0 / 24.0
    assert chunk_infos[0]["micro_chunk_candidate"] is True
    assert chunk_infos[0]["micro_resolve_action"] == "merge_micro_into_left"
    assert chunk_infos[0]["right_split_score"] == 0.82
    assert chunk_infos[0]["primary_cut_candidates"][0]["time_s"] == 1.5
    assert chunk_infos[0]["weak_cut_candidates"][0]["time_s"] == 1.8
    assert any("speech_segment_count=2" in entry and "source=cut" in entry for entry in log)


def test_cut_candidates_for_segment_filters_to_segment_window():
    from asr import pipeline as asr

    chunks_by_index = {
        0: {
            "weak_cut_candidates": [
                {"kind": "weak", "time_s": 1.0, "frame": 50, "strength": 0.5},
                {"kind": "weak", "time_s": 3.0, "frame": 150, "strength": 0.9},
            ]
        }
    }
    segment = {
        "start": 2.0,
        "end": 4.0,
        "words": [{"source_chunk_index": 0, "start": 2.1, "end": 2.2, "word": "あ"}],
    }

    candidates = asr._cut_candidates_for_segment(
        segment,
        chunks_by_index,
        key="weak_cut_candidates",
    )

    assert [candidate["time_s"] for candidate in candidates] == [3.0]


def test_alignment_window_metadata_uses_chunk_span():
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

    annotated = asr._with_alignment_window(chunk, text_result)

    assert annotated["alignment_window_start_s"] == 0.0
    assert annotated["alignment_window_end_s"] == 10.0
    assert annotated["alignment_window_source"] == "chunk"
    assert "alignment_fallback_start_s" not in annotated


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
        chunk_result={**text_result, "alignment_mode": "boundary_proportional"},
        chunk_words=chunk_words,
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

    assert outcome["alignment_quality"] == "boundary"
    assert outcome["alignment_mode"] == "boundary_proportional"
    assert transcript[0]["alignment_quality"] == "boundary"
    assert transcript[0]["alignment_issue_subtype"] == "none"
    assert "fallback_subtype" not in transcript[0]
    assert segments[0]["alignment_quality"] == "boundary"
    assert segments[0]["alignment_issue_subtype"] == "none"
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


def test_asr_after_cueqc_env_is_ignored_by_main_pipeline(monkeypatch, tmp_path):
    backend, segments, log, details = _run_transcription(
        monkeypatch,
        tmp_path,
        enable_cueqc=True,
    )

    assert backend.finalized_payloads
    assert len(backend.finalized_payloads) == len(backend.audio_paths)
    assert "cueqc_shadow" not in details
    assert details["transcript_chunks"]
    assert all("cueqc_shadow" not in chunk for chunk in details["transcript_chunks"])
    assert segments
    assert all("cueqc_shadow" not in segment for segment in segments)
    assert not any(entry.startswith("CueQC:") for entry in log)
