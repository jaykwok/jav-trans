from __future__ import annotations

import importlib
import sys
import wave
from pathlib import Path

import pytest

from audio.chunk_packer import PackedChunk
from boundary.base import SegmentationResult, SpeechSegment
from boundary.refiner import BoundaryDecision
from boundary.sequence_features import (
    CHUNK_POOLED_PTM_SCHEMA,
    DEFAULT_CHUNK_POOLED_PTM_BINS,
    chunk_pooled_ptm_feature_names,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _boundary_config() -> dict:
    return {
        "feature_frame_hop_s": 0.02,
        "boundary_refiner_model_path": "",
        "boundary_planner_sequence_batch_size": 256,
        "drop_enabled": False,
        "drop_min_duration_s": 0.20,
        "drop_rms_dbfs": -40.0,
    }


def _write_wav(path: Path, seconds: float = 2.0, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * int(sample_rate * seconds))


def _set_boundary_refiner_mapping(monkeypatch, tmp_path: Path) -> None:
    asr_backend = "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf"
    checkpoint = tmp_path / "boundary_edge_refiner_v8_safe_tight.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame-hf.pt"
    checkpoint.write_bytes(b"v8")
    monkeypatch.setenv("ASR_BACKEND", asr_backend)
    monkeypatch.setenv(
        "BOUNDARY_REFINER_MODEL_PATH_BY_REPO",
        f"{asr_backend}={checkpoint}",
    )


def test_boundary_cache_key_ignores_asr_prompt_budget(monkeypatch, tmp_path):
    from boundary import cache as boundary_cache

    monkeypatch.setenv("BOUNDARY_CACHE_DIR", str(tmp_path / "boundary-cache"))
    audio = tmp_path / "sample.cf3671a5.wav"
    _write_wav(audio)

    lookup_a = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_boundary_ja", "threshold": 0.35},
        boundary_config=_boundary_config(),
    )
    monkeypatch.setenv("ASR_INITIAL_PROMPT_MAX_CHARS", "160")
    monkeypatch.setenv("ASR_MAX_NEW_TOKENS", "256")
    lookup_b = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_boundary_ja", "threshold": 0.35},
        boundary_config=_boundary_config(),
    )

    assert lookup_a["digest"] == lookup_b["digest"]
    assert lookup_a["path"] == lookup_b["path"]


def test_boundary_cache_signature_ignores_removed_chunk_export_env(monkeypatch, tmp_path):
    from boundary import cache as boundary_cache

    monkeypatch.setenv("BOUNDARY_CACHE_DIR", str(tmp_path / "boundary-cache"))
    audio = tmp_path / "sample.cf3671a5.wav"
    _write_wav(audio)

    lookup_a = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_boundary_ja", "threshold": 0.35},
        boundary_config=_boundary_config(),
    )
    monkeypatch.setenv("SEGMENT_MIN_SPEECH", "9.0")
    monkeypatch.setenv("SEGMENT_MAX_CHUNK", "9.0")
    lookup_b = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_boundary_ja", "threshold": 0.35},
        boundary_config=_boundary_config(),
    )
    monkeypatch.setenv("ASR_CHUNK_MIN_DURATION_S", "0.4")
    lookup_c = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_boundary_ja", "threshold": 0.35},
        boundary_config=_boundary_config(),
    )

    assert lookup_a["digest"] == lookup_b["digest"]
    assert lookup_a["digest"] == lookup_c["digest"]


def test_boundary_cache_key_changes_with_boundary_config(monkeypatch, tmp_path):
    from boundary import cache as boundary_cache

    monkeypatch.setenv("BOUNDARY_CACHE_DIR", str(tmp_path / "boundary-cache"))
    audio = tmp_path / "sample.cf3671a5.wav"
    _write_wav(audio)

    lookup_a = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_boundary_ja", "threshold": 0.35},
        boundary_config=_boundary_config(),
    )
    cfg = _boundary_config()
    cfg["boundary_planner_sequence_batch_size"] = 128
    lookup_b = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_boundary_ja", "threshold": 0.35},
        boundary_config=cfg,
    )

    assert lookup_a["digest"] != lookup_b["digest"]
    assert lookup_a["path"] != lookup_b["path"]


def test_boundary_cache_key_ignores_retired_split_dense_decoder_env(monkeypatch, tmp_path):
    from boundary import cache as boundary_cache

    monkeypatch.setenv("BOUNDARY_CACHE_DIR", str(tmp_path / "boundary-cache"))
    audio = tmp_path / "sample.cf3671a5.wav"
    _write_wav(audio)

    lookup_a = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_boundary_ja", "decoder": "topographic_split_micro_resolver_v5"},
        boundary_config=_boundary_config(),
    )
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_SPLIT_MIN_PRIMARY_SCORE", "0.37")
    lookup_b = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_boundary_ja", "decoder": "topographic_split_micro_resolver_v5"},
        boundary_config=_boundary_config(),
    )
    monkeypatch.setenv("SPEECH_BOUNDARY_JA_DENSE_CUT_GAP_S", "0.09")
    lookup_c = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_boundary_ja", "decoder": "topographic_split_micro_resolver_v5"},
        boundary_config=_boundary_config(),
    )

    assert lookup_a["digest"] == lookup_b["digest"] == lookup_c["digest"]
    assert lookup_a["path"] == lookup_b["path"] == lookup_c["path"]


def test_boundary_cache_dir_does_not_change_digest(monkeypatch, tmp_path):
    from boundary import cache as boundary_cache

    audio = tmp_path / "sample.cf3671a5.wav"
    _write_wav(audio)
    monkeypatch.setenv("BOUNDARY_CACHE_DIR", str(tmp_path / "a"))
    lookup_a = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_boundary_ja"},
        boundary_config=_boundary_config(),
    )
    monkeypatch.setenv("BOUNDARY_CACHE_DIR", str(tmp_path / "b"))
    lookup_b = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_boundary_ja"},
        boundary_config=_boundary_config(),
    )

    assert lookup_a["digest"] == lookup_b["digest"]
    assert lookup_a["path"] != lookup_b["path"]


def test_boundary_cache_round_trips_packed_chunks(monkeypatch, tmp_path):
    from boundary import cache as boundary_cache

    monkeypatch.setenv("BOUNDARY_CACHE_DIR", str(tmp_path / "boundary-cache"))
    audio = tmp_path / "sample.cf3671a5.wav"
    _write_wav(audio)
    signature = {"backend": "stub_vad", "threshold": 0.35}
    config = _boundary_config()
    chunks = [
        PackedChunk(
            start=0.0,
            end=2.4,
            duration=2.4,
            split_reason="edge_refiner:utterance_switch",
            parent_chunk_id=3,
            island_id=1,
            island_count=2,
            core_start=0.2,
            core_end=0.4,
            internal_gap_count=1,
            internal_gap_max_s=0.5,
            boundary_score=0.87,
            boundary_reason="utterance_switch",
            boundary_source="split_boundary",
            boundary_start_refine_delta_s=0.01,
            boundary_end_refine_delta_s=-0.02,
            boundary_decision_source="edge_sequence_refiner_v8",
            subtitle_min_duration_s=20.0 / 24.0,
            below_subtitle_min_duration=True,
            micro_chunk_candidate=True,
            micro_resolve_action="preserve_micro_candidate",
            micro_resolve_reason="balanced_split_evidence",
            left_split_score=0.7,
            right_split_score=0.72,
            primary_cut_candidates=[
                {
                    "kind": "primary",
                    "time_s": 1.2,
                    "frame": 60,
                    "score": 0.8,
                    "prominence": 0.2,
                    "speech_valley": 0.7,
                    "strength": 1.7,
                }
            ],
            weak_cut_candidates=[
                {
                    "kind": "weak",
                    "time_s": 1.7,
                    "frame": 85,
                    "score": 0.4,
                    "prominence": 0.1,
                    "speech_valley": 0.6,
                    "strength": 1.1,
                }
            ],
            pre_asr_ptm_pooling_schema=CHUNK_POOLED_PTM_SCHEMA,
            pre_asr_ptm_pooling_bins=DEFAULT_CHUNK_POOLED_PTM_BINS,
            pre_asr_ptm_pooling_dim=2,
            pre_asr_ptm_pooled_features=[0.1, 0.2],
            speech_segments=[
                SpeechSegment(
                    0.2,
                    0.4,
                    0.9,
                    weak_cut_candidates=[
                        {
                            "kind": "weak",
                            "time_s": 0.3,
                            "frame": 15,
                            "score": 0.3,
                            "prominence": 0.1,
                            "speech_valley": 0.6,
                            "strength": 1.0,
                        }
                    ],
                )
            ],
        )
    ]

    boundary_cache.save_processing_spans(
        str(audio),
        boundary_signature=signature,
        boundary_config=config,
        processing_spans=chunks,
        runtime_boundary_signature={"backend": "stub_vad", "boundary_pipeline": {"version": 1}},
        speech_segments=chunks[0].speech_segments,
        speech_groups=[chunks[0].speech_segments],
    )
    loaded = boundary_cache.load_processing_spans(
        str(audio),
        boundary_signature=signature,
        boundary_config=config,
    )

    assert loaded is not None
    loaded_chunks, runtime_signature, event = loaded
    assert event["status"] == "hit"
    assert runtime_signature["backend"] == "stub_vad"
    assert isinstance(loaded_chunks[0], PackedChunk)
    assert loaded_chunks[0].start == 0.0
    assert loaded_chunks[0].split_reason == "edge_refiner:utterance_switch"
    assert loaded_chunks[0].parent_chunk_id == 3
    assert loaded_chunks[0].island_id == 1
    assert loaded_chunks[0].island_count == 2
    assert loaded_chunks[0].core_start == 0.2
    assert loaded_chunks[0].core_end == 0.4
    assert loaded_chunks[0].internal_gap_count == 1
    assert loaded_chunks[0].internal_gap_max_s == 0.5
    assert loaded_chunks[0].boundary_score == 0.87
    assert loaded_chunks[0].boundary_reason == "utterance_switch"
    assert loaded_chunks[0].boundary_source == "split_boundary"
    assert loaded_chunks[0].boundary_start_refine_delta_s == 0.01
    assert loaded_chunks[0].boundary_end_refine_delta_s == -0.02
    assert loaded_chunks[0].boundary_decision_source == "edge_sequence_refiner_v8"
    assert loaded_chunks[0].subtitle_min_duration_s == pytest.approx(20.0 / 24.0)
    assert loaded_chunks[0].below_subtitle_min_duration is True
    assert loaded_chunks[0].micro_chunk_candidate is True
    assert loaded_chunks[0].micro_resolve_action == "preserve_micro_candidate"
    assert loaded_chunks[0].left_split_score == pytest.approx(0.7)
    assert loaded_chunks[0].right_split_score == pytest.approx(0.72)
    assert loaded_chunks[0].primary_cut_candidates[0]["time_s"] == pytest.approx(1.2)
    assert loaded_chunks[0].weak_cut_candidates[0]["time_s"] == pytest.approx(1.7)
    assert loaded_chunks[0].pre_asr_ptm_pooling_schema == CHUNK_POOLED_PTM_SCHEMA
    assert loaded_chunks[0].pre_asr_ptm_pooling_bins == DEFAULT_CHUNK_POOLED_PTM_BINS
    assert loaded_chunks[0].pre_asr_ptm_pooling_dim == 2
    assert loaded_chunks[0].pre_asr_ptm_pooled_features == pytest.approx([0.1, 0.2])
    assert loaded_chunks[0].speech_segments[0].score == 0.9
    assert loaded_chunks[0].speech_segments[0].weak_cut_candidates[0]["time_s"] == pytest.approx(0.3)


class _CountingSpeechBoundaryBackend:
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def signature(self) -> dict:
        return {"backend": self.name, "threshold": 0.35}

    def segment(
        self,
        audio_path: str,
        *,
        target_sr: int = 16000,
        threshold_override: float | None = None,
    ) -> SegmentationResult:
        del audio_path, target_sr, threshold_override
        self.calls += 1
        segments = [
            SpeechSegment(0.2, 0.6, 0.8),
            SpeechSegment(1.2, 1.6, 0.7),
        ]
        return SegmentationResult(
            segments=segments,
            groups=[[segment] for segment in segments],
            method=self.name,
            audio_duration_sec=2.0,
            parameters={"backend": self.name, "threshold": 0.35},
            processing_time_sec=0.0,
        )


class _ScoreExportingSpeechBoundaryBackend(_CountingSpeechBoundaryBackend):
    def segment(
        self,
        audio_path: str,
        *,
        target_sr: int = 16000,
        threshold_override: float | None = None,
    ) -> SegmentationResult:
        del audio_path, target_sr, threshold_override
        self.calls += 1
        segments = [SpeechSegment(0.0, 5.0, 0.9), SpeechSegment(6.0, 12.0, 0.8)]
        return SegmentationResult(
            segments=segments,
            groups=[[segment] for segment in segments],
            method=self.name,
            audio_duration_sec=12.0,
            parameters={
                "backend": self.name,
                "threshold": 0.10,
                "frame_scores": [0.9] * 12,
                "split_boundary_frame_scores": [0.05] * 5 + [0.98] * 2 + [0.05] * 5,
                "frame_hop_s": 1.0,
            },
            processing_time_sec=0.0,
        )


class _FakeSequenceRefiner:
    feature_names = ("gap_s",)
    feature_schema_hash = "test-feature-schema"

    def decide_sequence(self, features: list[list[float]]) -> list[BoundaryDecision]:
        return [
            BoundaryDecision(
                source="edge_sequence_refiner_v8",
                start_refine_delta_s=0.0,
                end_refine_delta_s=0.0,
            )
            for _ in features
        ]

    def signature(self) -> dict:
        return {"schema": "boundary_edge_refiner_v8_safe_tight", "type": "fake_sequence_refiner"}


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
        bins: int = DEFAULT_CHUNK_POOLED_PTM_BINS,
    ) -> list[str]:
        return chunk_pooled_ptm_feature_names(ptm_dim=64, bins=bins)

    def chunk_pooled_ptm_features(
        self,
        *,
        start_s: float,
        end_s: float,
        bins: int = DEFAULT_CHUNK_POOLED_PTM_BINS,
    ) -> list[float]:
        del start_s, end_s
        return [0.0] * len(self.chunk_pooled_ptm_feature_names(bins=bins))


def _patch_fake_refiner(monkeypatch, asr) -> None:
    monkeypatch.setattr(
        asr,
        "_boundary_refiner_runtime_adapter",
        lambda _path: "edge_sequence_v2",
    )
    monkeypatch.setattr(
        asr,
        "load_edge_sequence_refiner_v8_checkpoint",
        lambda *_args, **_kwargs: _FakeSequenceRefiner(),
    )
    monkeypatch.setattr(
        asr,
        "_required_sequence_feature_provider_from_result",
        lambda *_args, **_kwargs: _FakeSequenceFeatureProvider(),
    )


def test_pipeline_uses_boundary_scores_but_does_not_cache_score_arrays(monkeypatch, tmp_path):
    _set_boundary_refiner_mapping(monkeypatch, tmp_path)
    monkeypatch.setenv("BOUNDARY_CACHE_DIR", str(tmp_path / "boundary-cache"))
    monkeypatch.setenv("BOUNDARY_FEATURE_FRAME_HOP_S", "1.0")

    from asr import pipeline as asr

    asr = importlib.reload(asr)
    _patch_fake_refiner(monkeypatch, asr)
    audio = tmp_path / "sample.cf3671a5.wav"
    _write_wav(audio, seconds=12.0)

    backend = _ScoreExportingSpeechBoundaryBackend()
    import boundary

    monkeypatch.setattr(boundary, "get_boundary_backend", lambda: backend)

    spans = asr._build_processing_spans(str(audio))

    assert backend.calls == 1
    assert len(spans) == 2
    assert all(isinstance(span, PackedChunk) for span in spans)
    assert asr._LAST_BOUNDARY_SIGNATURE["boundary_pipeline"]["score_frame_hop_s"] == 1.0
    assert (
        asr._LAST_BOUNDARY_SIGNATURE["boundary_pipeline"]["sequence_boundary_refiner"]["schema"]
        == "boundary_edge_refiner_v8_safe_tight"
    )
    assert asr._LAST_BOUNDARY_SIGNATURE["boundary_pipeline"]["feature_sources"] == {
        "speech_scores": True,
        "split_boundary_scores": True,
    }
    cached_files = list((tmp_path / "boundary-cache").glob("*.json"))
    assert len(cached_files) == 1
    payload = cached_files[0].read_text(encoding="utf-8")
    assert "frame_scores" not in payload
    assert "split_boundary_frame_scores" not in payload


def test_pipeline_uses_boundary_cache_for_prompt_budget_change(monkeypatch, tmp_path):
    _set_boundary_refiner_mapping(monkeypatch, tmp_path)
    monkeypatch.setenv("BOUNDARY_CACHE_DIR", str(tmp_path / "boundary-cache"))
    monkeypatch.setenv("BOUNDARY_FEATURE_FRAME_HOP_S", "0.02")

    from asr import pipeline as asr

    asr = importlib.reload(asr)
    _patch_fake_refiner(monkeypatch, asr)
    audio = tmp_path / "sample.cf3671a5.wav"
    _write_wav(audio)

    backend = _CountingSpeechBoundaryBackend()
    import boundary

    monkeypatch.setattr(boundary, "get_boundary_backend", lambda: backend)

    first = asr._build_processing_spans(str(audio))
    monkeypatch.setenv("ASR_INITIAL_PROMPT_MAX_CHARS", "160")
    second = asr._build_processing_spans(str(audio))

    assert backend.calls == 1
    assert len(first) == len(second) == 2
    assert isinstance(second[0], PackedChunk)
    assert second[0].speech_segments[0].start == 0.2
