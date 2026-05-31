from __future__ import annotations

import importlib
import sys
import wave
from pathlib import Path

from audio.chunk_packer import PackedChunk
from vad.base import SegmentationResult, SpeechSegment

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _chunk_config() -> dict:
    return {
        "packing_enabled": True,
        "pack_frame_hop_s": 1.0 / 29.97,
        "pack_window_frames": 899,
        "pack_reserve_frames": 45,
        "pack_target_padding_frames": 60,
        "pack_gap_merge_frames": 45,
        "pre_asr_island_split_enabled": False,
        "pre_asr_island_split_min_core_frames": 420,
        "pre_asr_island_split_min_gap_frames": 18,
        "pre_asr_island_split_min_island_frames": 3,
        "pre_asr_island_split_max_children": 8,
        "pre_asr_valley_split_enabled": False,
        "pre_asr_valley_split_min_core_frames": 420,
        "pre_asr_valley_split_target_core_frames": 270,
        "pre_asr_valley_split_min_valley_frames": 6,
        "pre_asr_valley_split_min_child_frames": 45,
        "pre_asr_valley_split_max_children": 8,
        "pre_asr_valley_split_threshold": 0.20,
    }


def _write_wav(path: Path, seconds: float = 2.0, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * int(sample_rate * seconds))


def test_vad_chunk_cache_key_ignores_asr_prompt_budget(monkeypatch, tmp_path):
    from whisper import vad_chunk_cache

    monkeypatch.setenv("VAD_CHUNK_CACHE_DIR", str(tmp_path / "vad-cache"))
    audio = tmp_path / "sample.cf3671a5.wav"
    _write_wav(audio)

    lookup_a = vad_chunk_cache.build_cache_lookup(
        str(audio),
        vad_signature={"backend": "whisperseg_v1", "threshold": 0.35},
        chunk_config=_chunk_config(),
    )
    monkeypatch.setenv("ASR_INITIAL_PROMPT_MAX_CHARS", "160")
    monkeypatch.setenv("ASR_INITIAL_PROMPT_MAX_TOKENS", "64")
    monkeypatch.setenv("ASR_MIN_EFFECTIVE_NEW_TOKENS", "96")
    lookup_b = vad_chunk_cache.build_cache_lookup(
        str(audio),
        vad_signature={"backend": "whisperseg_v1", "threshold": 0.35},
        chunk_config=_chunk_config(),
    )

    assert lookup_a["digest"] == lookup_b["digest"]
    assert lookup_a["path"] == lookup_b["path"]


def test_vad_chunk_cache_key_changes_with_vad_threshold(monkeypatch, tmp_path):
    from whisper import vad_chunk_cache

    monkeypatch.setenv("VAD_CHUNK_CACHE_DIR", str(tmp_path / "vad-cache"))
    audio = tmp_path / "sample.cf3671a5.wav"
    _write_wav(audio)

    lookup_a = vad_chunk_cache.build_cache_lookup(
        str(audio),
        vad_signature={"backend": "whisperseg_v1", "threshold": 0.35},
        chunk_config=_chunk_config(),
    )
    lookup_b = vad_chunk_cache.build_cache_lookup(
        str(audio),
        vad_signature={"backend": "whisperseg_v1", "threshold": 0.45},
        chunk_config=_chunk_config(),
    )

    assert lookup_a["digest"] != lookup_b["digest"]
    assert lookup_a["path"] != lookup_b["path"]


def test_vad_chunk_cache_round_trips_packed_chunks(monkeypatch, tmp_path):
    from whisper import vad_chunk_cache

    monkeypatch.setenv("VAD_CHUNK_CACHE_DIR", str(tmp_path / "vad-cache"))
    audio = tmp_path / "sample.cf3671a5.wav"
    _write_wav(audio)
    signature = {"backend": "stub_vad", "threshold": 0.35}
    config = _chunk_config()
    chunks = [
        PackedChunk(
            start=0.0,
            end=2.4,
            duration=2.4,
            left_padding_s=0.2,
            right_padding_s=2.0,
            split_reason="tail",
            parent_chunk_id=3,
            island_id=1,
            island_count=2,
            core_start=0.2,
            core_end=0.4,
            internal_gap_count=0,
            internal_gap_max_s=0.0,
            split_policy="r15_pre_asr_island_v1",
            valley_split_count=1,
            valley_score_min=0.05,
            vad_segments=[SpeechSegment(0.2, 0.4, 0.9)],
        )
    ]

    vad_chunk_cache.save_processing_spans(
        str(audio),
        vad_signature=signature,
        chunk_config=config,
        processing_spans=chunks,
        runtime_vad_signature={"backend": "stub_vad", "chunk_packing": {"enabled": True}},
        vad_segments=chunks[0].vad_segments,
        vad_groups=[chunks[0].vad_segments],
    )
    loaded = vad_chunk_cache.load_processing_spans(
        str(audio),
        vad_signature=signature,
        chunk_config=config,
    )

    assert loaded is not None
    loaded_chunks, runtime_signature, event = loaded
    assert event["status"] == "hit"
    assert runtime_signature["backend"] == "stub_vad"
    assert isinstance(loaded_chunks[0], PackedChunk)
    assert loaded_chunks[0].start == 0.0
    assert loaded_chunks[0].right_padding_s == 2.0
    assert loaded_chunks[0].split_reason == "tail"
    assert loaded_chunks[0].parent_chunk_id == 3
    assert loaded_chunks[0].island_id == 1
    assert loaded_chunks[0].island_count == 2
    assert loaded_chunks[0].core_start == 0.2
    assert loaded_chunks[0].core_end == 0.4
    assert loaded_chunks[0].split_policy == "r15_pre_asr_island_v1"
    assert loaded_chunks[0].valley_split_count == 1
    assert loaded_chunks[0].valley_score_min == 0.05
    assert loaded_chunks[0].vad_segments[0].score == 0.9


def test_vad_chunk_cache_key_changes_with_pre_asr_island_split(monkeypatch, tmp_path):
    from whisper import vad_chunk_cache

    monkeypatch.setenv("VAD_CHUNK_CACHE_DIR", str(tmp_path / "vad-cache"))
    audio = tmp_path / "sample.cf3671a5.wav"
    _write_wav(audio)

    lookup_a = vad_chunk_cache.build_cache_lookup(
        str(audio),
        vad_signature={"backend": "whisperseg_v1", "threshold": 0.35},
        chunk_config=_chunk_config(),
    )
    cfg = _chunk_config()
    cfg["pre_asr_island_split_enabled"] = True
    lookup_b = vad_chunk_cache.build_cache_lookup(
        str(audio),
        vad_signature={"backend": "whisperseg_v1", "threshold": 0.35},
        chunk_config=cfg,
    )

    assert lookup_a["digest"] != lookup_b["digest"]


def test_vad_chunk_cache_key_changes_with_pre_asr_valley_split(monkeypatch, tmp_path):
    from whisper import vad_chunk_cache

    monkeypatch.setenv("VAD_CHUNK_CACHE_DIR", str(tmp_path / "vad-cache"))
    audio = tmp_path / "sample.cf3671a5.wav"
    _write_wav(audio)

    lookup_a = vad_chunk_cache.build_cache_lookup(
        str(audio),
        vad_signature={"backend": "fusionvad_ja", "threshold": 0.10},
        chunk_config=_chunk_config(),
    )
    cfg = _chunk_config()
    cfg["pre_asr_valley_split_enabled"] = True
    lookup_b = vad_chunk_cache.build_cache_lookup(
        str(audio),
        vad_signature={"backend": "fusionvad_ja", "threshold": 0.10},
        chunk_config=cfg,
    )

    assert lookup_a["digest"] != lookup_b["digest"]


class _CountingVadBackend:
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


class _ScoredVadBackend(_CountingVadBackend):
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
                "frame_scores": [0.9] * 5 + [0.05] * 2 + [0.9] * 5,
            },
            processing_time_sec=0.0,
        )


def test_pipeline_uses_frame_scores_for_valley_split_but_does_not_cache_scores(monkeypatch, tmp_path):
    monkeypatch.setenv("VAD_CHUNK_CACHE_DIR", str(tmp_path / "vad-cache"))
    monkeypatch.setenv("ASR_CHUNK_PACKING_ENABLED", "1")
    monkeypatch.setenv("ASR_CHUNK_PACK_FRAME_HOP_S", "1.0")
    monkeypatch.setenv("ASR_CHUNK_PACK_WINDOW_FRAMES", "30")
    monkeypatch.setenv("ASR_CHUNK_PACK_RESERVE_FRAMES", "2")
    monkeypatch.setenv("ASR_CHUNK_PACK_TARGET_PADDING_FRAMES", "1")
    monkeypatch.setenv("ASR_CHUNK_PACK_GAP_MERGE_FRAMES", "0")
    monkeypatch.setenv("ASR_PRE_ASR_VALLEY_SPLIT_ENABLED", "1")
    monkeypatch.setenv("ASR_PRE_ASR_VALLEY_SPLIT_MIN_CORE_FRAMES", "8")
    monkeypatch.setenv("ASR_PRE_ASR_VALLEY_SPLIT_TARGET_CORE_FRAMES", "5")
    monkeypatch.setenv("ASR_PRE_ASR_VALLEY_SPLIT_MIN_VALLEY_FRAMES", "2")
    monkeypatch.setenv("ASR_PRE_ASR_VALLEY_SPLIT_MIN_CHILD_FRAMES", "3")
    monkeypatch.setenv("ASR_PRE_ASR_VALLEY_SPLIT_THRESHOLD", "0.2")

    from whisper import pipeline as asr
    asr = importlib.reload(asr)
    audio = tmp_path / "sample.cf3671a5.wav"
    _write_wav(audio, seconds=12.0)

    backend = _ScoredVadBackend()
    import vad

    monkeypatch.setattr(vad, "get_vad_backend", lambda: backend)

    spans = asr._build_processing_spans(str(audio))

    assert backend.calls == 1
    assert len(spans) == 2
    assert all(isinstance(span, PackedChunk) for span in spans)
    assert {span.split_policy for span in spans} == {"r16_pre_asr_valley_v1"}
    cached_files = list((tmp_path / "vad-cache").glob("*.json"))
    assert len(cached_files) == 1
    payload = cached_files[0].read_text(encoding="utf-8")
    assert "frame_scores" not in payload


def test_pipeline_uses_vad_chunk_cache_for_prompt_budget_change(monkeypatch, tmp_path):
    monkeypatch.setenv("VAD_CHUNK_CACHE_DIR", str(tmp_path / "vad-cache"))
    monkeypatch.setenv("ASR_CHUNK_PACKING_ENABLED", "1")
    monkeypatch.setenv("ASR_CHUNK_PACK_FRAME_HOP_S", str(1.0 / 29.97))
    monkeypatch.setenv("ASR_CHUNK_PACK_WINDOW_FRAMES", "899")
    monkeypatch.setenv("ASR_CHUNK_PACK_RESERVE_FRAMES", "45")
    monkeypatch.setenv("ASR_CHUNK_PACK_TARGET_PADDING_FRAMES", "0")
    monkeypatch.setenv("ASR_CHUNK_PACK_GAP_MERGE_FRAMES", "45")

    from whisper import pipeline as asr
    asr = importlib.reload(asr)
    audio = tmp_path / "sample.cf3671a5.wav"
    _write_wav(audio)

    backend = _CountingVadBackend()
    import vad

    monkeypatch.setattr(vad, "get_vad_backend", lambda: backend)

    first = asr._build_processing_spans(str(audio))
    monkeypatch.setenv("ASR_INITIAL_PROMPT_MAX_CHARS", "160")
    second = asr._build_processing_spans(str(audio))

    assert backend.calls == 1
    assert len(first) == len(second) == 1
    assert isinstance(second[0], PackedChunk)
    assert second[0].vad_segments[0].start == 0.2
