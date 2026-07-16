from __future__ import annotations

import wave
from pathlib import Path

import pytest

from audio.chunk_packer import PackedChunk
from boundary.base import SpeechSegment
from boundary.sequence_features import CHUNK_POOLED_PTM_SCHEMA, DEFAULT_CHUNK_POOLED_PTM_BINS


def _boundary_config() -> dict:
    return {
        "feature_frame_hop_s": 0.02,
        "outer_edge_refiner_model_path": "outer.pt",
        "semantic_split_model_path": "split.pt",
        "cut_edge_refiner_model_path": "cut.pt",
    }


def _write_wav(path: Path, seconds: float = 2.0, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * int(sample_rate * seconds))


def test_boundary_cache_key_ignores_asr_generation_budget(monkeypatch, tmp_path) -> None:
    from boundary import cache as boundary_cache

    assert boundary_cache.BOUNDARY_CACHE_VERSION == 20
    monkeypatch.setenv("BOUNDARY_CACHE_DIR", str(tmp_path / "boundary-cache"))
    audio = tmp_path / "sample.wav"
    _write_wav(audio)
    lookup_a = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_island_v8", "threshold": 0.15},
        boundary_config=_boundary_config(),
    )
    monkeypatch.setenv("ASR_MAX_NEW_TOKENS", "256")
    lookup_b = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_island_v8", "threshold": 0.15},
        boundary_config=_boundary_config(),
    )

    assert lookup_a["digest"] == lookup_b["digest"]


def test_boundary_cache_key_changes_with_any_boundary_model(tmp_path, monkeypatch) -> None:
    from boundary import cache as boundary_cache

    monkeypatch.setenv("BOUNDARY_CACHE_DIR", str(tmp_path / "boundary-cache"))
    audio = tmp_path / "sample.wav"
    _write_wav(audio)
    original = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_island_v8"},
        boundary_config=_boundary_config(),
    )
    changed_config = _boundary_config()
    changed_config["semantic_split_model_path"] = "split-v2.pt"
    changed = boundary_cache.build_cache_lookup(
        str(audio),
        boundary_signature={"backend": "speech_island_v8"},
        boundary_config=changed_config,
    )

    assert original["digest"] != changed["digest"]


def test_delete_boundary_cache_variants_for_audio_key(tmp_path, monkeypatch) -> None:
    from boundary import cache as boundary_cache

    root = tmp_path / "boundary-cache"
    root.mkdir()
    monkeypatch.setenv("BOUNDARY_CACHE_DIR", str(root))
    matching = [
        root / "abcdef12.first.json",
        root / "abcdef12.second.json",
        root / "abcdef12.second.json.123.tmp",
    ]
    unrelated = root / "12345678.first.json"
    for path in [*matching, unrelated]:
        path.write_text("{}", encoding="utf-8")

    assert boundary_cache.delete_for_audio_cache_key("ABCDEF12") == 3
    assert all(not path.exists() for path in matching)
    assert unrelated.exists()
    assert boundary_cache.delete_for_audio_cache_key("../boundary-cache") == 0


def test_boundary_cache_round_trips_shared_absolute_cut_metadata(monkeypatch, tmp_path) -> None:
    from boundary import cache as boundary_cache

    monkeypatch.setenv("BOUNDARY_CACHE_DIR", str(tmp_path / "boundary-cache"))
    audio = tmp_path / "sample.wav"
    _write_wav(audio)
    chunk = PackedChunk(
        start=1.0,
        end=2.5,
        source_abs_start=1.0,
        source_abs_end=2.5,
        speech_segments=[SpeechSegment(1.0, 2.5, 0.9)],
        duration=1.5,
        split_reason="semantic_split",
        boundary_source="shared_absolute_cut",
        primary_cut_candidates=[
            {
                "kind": "primary",
                "time_s": 2.5,
                "frame": 125,
                "proposal_time_s": 2.46,
                "p_cut": 0.97,
            }
        ],
        pre_asr_ptm_pooling_schema=CHUNK_POOLED_PTM_SCHEMA,
        pre_asr_ptm_pooling_bins=DEFAULT_CHUNK_POOLED_PTM_BINS,
        pre_asr_ptm_pooling_dim=2,
        pre_asr_ptm_pooled_features=[0.1, 0.2],
    )
    signature = {"backend": "speech_island_v8"}
    runtime_signature = {
        "backend": "speech_island_v8",
        "boundary_pipeline": {"version": 9},
    }
    boundary_cache.save_processing_spans(
        str(audio),
        boundary_signature=signature,
        boundary_config=_boundary_config(),
        processing_spans=[chunk],
        runtime_boundary_signature=runtime_signature,
        speech_segments=chunk.speech_segments,
        speech_groups=[chunk.speech_segments],
    )

    loaded = boundary_cache.load_processing_spans(
        str(audio),
        boundary_signature=signature,
        boundary_config=_boundary_config(),
    )

    assert loaded is not None
    chunks, loaded_signature, event = loaded
    assert event["status"] == "hit"
    assert loaded_signature["boundary_pipeline"]["version"] == 9
    assert chunks[0].source_abs_start == pytest.approx(1.0)
    assert chunks[0].source_abs_end == pytest.approx(2.5)
    assert chunks[0].boundary_source == "shared_absolute_cut"
    assert chunks[0].primary_cut_candidates[0]["time_s"] == pytest.approx(2.5)
    assert chunks[0].pre_asr_ptm_pooled_features == pytest.approx([0.1, 0.2])


def test_boundary_cache_v20_round_trips_paired_inner_edge_metadata(
    monkeypatch, tmp_path
) -> None:
    from boundary import cache as boundary_cache

    monkeypatch.setenv("BOUNDARY_CACHE_DIR", str(tmp_path / "boundary-cache"))
    audio = tmp_path / "sample.wav"
    _write_wav(audio)
    chunk = PackedChunk(
        start=1.2,
        end=2.7,
        duration=1.5,
        speech_segments=[SpeechSegment(1.2, 2.7, 0.9)],
        split_reason="acoustic_split_v3",
        acoustic_start=1.2,
        acoustic_end=2.7,
        acoustic_duration=1.5,
        display_start=1.25,
        display_end=2.65,
        display_duration=1.4,
        boundary_pipeline_version=10,
        semantic_event_ids=["event-001"],
        semantic_event_probabilities=[
            {"p_cut": 0.96, "p_continue": 0.03, "p_unsure": 0.01}
        ],
        paired_inner_edges={
            "event_ids": ["event-001"],
            "left_speech_end": 2.7,
            "right_speech_start": 3.0,
            "action": "safe",
        },
        removed_gap_spans=[{"start": 2.7, "end": 3.0, "duration": 0.3}],
        removed_gap_duration_s=0.3,
    )
    signature = {"backend": "speech_island_v8"}
    boundary_cache.save_processing_spans(
        str(audio),
        boundary_signature=signature,
        boundary_config=_boundary_config(),
        processing_spans=[chunk],
        runtime_boundary_signature={"boundary_pipeline": {"version": 10}},
    )

    loaded = boundary_cache.load_processing_spans(
        str(audio),
        boundary_signature=signature,
        boundary_config=_boundary_config(),
    )
    assert loaded is not None
    restored = loaded[0][0]
    assert restored.boundary_pipeline_version == 10
    assert restored.semantic_event_ids == ["event-001"]
    assert restored.paired_inner_edges["action"] == "safe"
    assert restored.removed_gap_duration_s == pytest.approx(0.3)
    assert restored.display_start == pytest.approx(1.25)
