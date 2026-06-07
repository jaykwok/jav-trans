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


def _boundary_config() -> dict:
    return {
        "feature_frame_hop_s": 0.02,
        "boundary_refiner_enabled": False,
        "boundary_refiner_model_path": "",
        "boundary_refiner_backbone": "transformers.Mamba2Model",
        "boundary_refiner_threshold": 0.5,
        "boundary_planner_max_core_chunk_s": 5.0,
        "boundary_planner_max_padded_chunk_s": 9.0,
        "boundary_planner_target_chunk_s": 3.0,
        "boundary_planner_min_chunk_s": 0.4,
        "boundary_planner_start_weight": 1.5,
        "boundary_planner_target_padding_s": 2.0,
        "boundary_planner_max_splits_per_segment": 16,
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
    monkeypatch.setenv("ASR_INITIAL_PROMPT_MAX_TOKENS", "64")
    monkeypatch.setenv("ASR_MIN_EFFECTIVE_NEW_TOKENS", "96")
    lookup_b = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_boundary_ja", "threshold": 0.35},
        boundary_config=_boundary_config(),
    )

    assert lookup_a["digest"] == lookup_b["digest"]
    assert lookup_a["path"] == lookup_b["path"]


def test_boundary_cache_signature_uses_current_chunk_export_env_only(monkeypatch, tmp_path):
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
    assert lookup_a["digest"] != lookup_c["digest"]


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
    cfg["boundary_refiner_enabled"] = True
    cfg["boundary_planner_target_chunk_s"] = 8.0
    lookup_b = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_boundary_ja", "threshold": 0.35},
        boundary_config=cfg,
    )

    assert lookup_a["digest"] != lookup_b["digest"]
    assert lookup_a["path"] != lookup_b["path"]


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
            left_padding_s=0.2,
            right_padding_s=2.0,
            split_reason="boundary_refiner:utterance_switch",
            parent_chunk_id=3,
            island_id=1,
            island_count=2,
            core_start=0.2,
            core_end=0.4,
            internal_gap_count=1,
            internal_gap_max_s=0.5,
            boundary_score=0.87,
            boundary_reason="utterance_switch",
            boundary_source="cut",
            boundary_decision_merge=False,
            boundary_merge_prob=0.13,
            boundary_split_prob=0.87,
            boundary_refine_delta_s=None,
            boundary_decision_source="frame_sequence_refiner",
            speech_segments=[SpeechSegment(0.2, 0.4, 0.9)],
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
    assert loaded_chunks[0].right_padding_s == 2.0
    assert loaded_chunks[0].split_reason == "boundary_refiner:utterance_switch"
    assert loaded_chunks[0].parent_chunk_id == 3
    assert loaded_chunks[0].island_id == 1
    assert loaded_chunks[0].island_count == 2
    assert loaded_chunks[0].core_start == 0.2
    assert loaded_chunks[0].core_end == 0.4
    assert loaded_chunks[0].internal_gap_count == 1
    assert loaded_chunks[0].internal_gap_max_s == 0.5
    assert loaded_chunks[0].boundary_score == 0.87
    assert loaded_chunks[0].boundary_reason == "utterance_switch"
    assert loaded_chunks[0].boundary_source == "cut"
    assert loaded_chunks[0].boundary_decision_merge is False
    assert loaded_chunks[0].boundary_merge_prob == 0.13
    assert loaded_chunks[0].boundary_split_prob == 0.87
    assert loaded_chunks[0].boundary_decision_source == "frame_sequence_refiner"
    assert loaded_chunks[0].speech_segments[0].score == 0.9


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


class _CutScoredSpeechBoundaryBackend(_CountingSpeechBoundaryBackend):
    def segment(
        self,
        audio_path: str,
        *,
        target_sr: int = 16000,
        threshold_override: float | None = None,
    ) -> SegmentationResult:
        del audio_path, target_sr, threshold_override
        self.calls += 1
        segments = [SpeechSegment(0.0, 12.0, 0.9)]
        return SegmentationResult(
            segments=segments,
            groups=[segments],
            method=self.name,
            audio_duration_sec=12.0,
            parameters={
                "backend": self.name,
                "threshold": 0.10,
                "frame_scores": [0.9] * 12,
                "cut_frame_scores": [0.05] * 5 + [0.98] * 2 + [0.05] * 5,
                "frame_hop_s": 1.0,
            },
            processing_time_sec=0.0,
        )


def test_pipeline_uses_boundary_scores_but_does_not_cache_score_arrays(monkeypatch, tmp_path):
    monkeypatch.setenv("BOUNDARY_CACHE_DIR", str(tmp_path / "boundary-cache"))
    monkeypatch.setenv("BOUNDARY_FEATURE_FRAME_HOP_S", "1.0")
    monkeypatch.setenv("BOUNDARY_REFINER_ENABLED", "1")
    monkeypatch.setenv("BOUNDARY_PLANNER_MAX_CORE_CHUNK_S", "30.0")
    monkeypatch.setenv("BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S", "30.0")
    monkeypatch.setenv("BOUNDARY_PLANNER_TARGET_CHUNK_S", "5.0")
    monkeypatch.setenv("BOUNDARY_PLANNER_MIN_CHUNK_S", "3.0")
    monkeypatch.setenv("BOUNDARY_REFINER_MODEL_PATH", "")

    from asr import pipeline as asr

    asr = importlib.reload(asr)
    audio = tmp_path / "sample.cf3671a5.wav"
    _write_wav(audio, seconds=12.0)

    backend = _CutScoredSpeechBoundaryBackend()
    import boundary

    monkeypatch.setattr(boundary, "get_boundary_backend", lambda: backend)

    spans = asr._build_processing_spans(str(audio))

    assert backend.calls == 1
    assert len(spans) == 2
    assert all(isinstance(span, PackedChunk) for span in spans)
    assert {span.boundary_source for span in spans} == {"cut"}
    assert {span.boundary_score for span in spans} == {0.98}
    assert asr._LAST_BOUNDARY_SIGNATURE["boundary_pipeline"]["score_frame_hop_s"] == 1.0
    assert asr._LAST_BOUNDARY_SIGNATURE["boundary_pipeline"]["boundary_refiner"]["type"] == (
        "heuristic_boundary_refiner"
    )
    cached_files = list((tmp_path / "boundary-cache").glob("*.json"))
    assert len(cached_files) == 1
    payload = cached_files[0].read_text(encoding="utf-8")
    assert "frame_scores" not in payload
    assert "cut_frame_scores" not in payload


def test_pipeline_uses_boundary_cache_for_prompt_budget_change(monkeypatch, tmp_path):
    monkeypatch.setenv("BOUNDARY_CACHE_DIR", str(tmp_path / "boundary-cache"))
    monkeypatch.setenv("BOUNDARY_FEATURE_FRAME_HOP_S", "0.02")
    monkeypatch.setenv("BOUNDARY_REFINER_MODEL_PATH", "")

    from asr import pipeline as asr

    asr = importlib.reload(asr)
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
