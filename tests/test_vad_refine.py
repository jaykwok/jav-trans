from __future__ import annotations

import wave
from pathlib import Path

from audio.vad_refine import refine_chunks_via_vad
from vad.base import SegmentationResult, SpeechSegment


def _write_wav(path: Path, duration_s: float, sample_rate: int = 16000) -> None:
    n_frames = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_frames)


class _SplitVadBackend:
    name = "split_test"

    def segment(
        self,
        audio_path: str,
        *,
        target_sr: int = 16000,
        threshold_override: float | None = None,
    ) -> SegmentationResult:
        del threshold_override
        segs = [SpeechSegment(start=0.0, end=0.8), SpeechSegment(start=1.0, end=2.0)]
        return SegmentationResult(
            segments=segs,
            groups=[[s] for s in segs],
            method=self.name,
            audio_duration_sec=2.0,
            parameters={},
            processing_time_sec=0.0,
        )

    def signature(self) -> dict:
        return {"backend": self.name}


class _TimeoutVadBackend:
    name = "timeout_test"

    def segment(
        self,
        audio_path: str,
        *,
        target_sr: int = 16000,
        threshold_override: float | None = None,
    ) -> SegmentationResult:
        del threshold_override
        import time

        time.sleep(10)
        raise RuntimeError("should not reach here")

    def signature(self) -> dict:
        return {"backend": self.name}


def test_vad_split_returns_sub_chunks(tmp_path):
    wav_path = tmp_path / "chunk_0000.wav"
    _write_wav(wav_path, duration_s=2.0)

    chunk = {
        "index": 0,
        "start": 5.0,
        "end": 7.0,
        "path": str(wav_path),
        "source_audio_path": "audio.wav",
    }
    result = refine_chunks_via_vad([chunk], vad_backend=_SplitVadBackend())

    assert len(result) >= 2
    for sub in result:
        assert sub["_vad_parent_index"] == 0
        assert Path(sub["path"]).exists()
        assert sub["start"] >= 5.0
        assert sub["end"] <= 7.0 + 0.01


def test_vad_timeout_returns_original_chunk(tmp_path):
    wav_path = tmp_path / "chunk_0000.wav"
    _write_wav(wav_path, duration_s=1.0)

    chunk = {
        "index": 3,
        "start": 10.0,
        "end": 11.0,
        "path": str(wav_path),
        "source_audio_path": "audio.wav",
    }
    result = refine_chunks_via_vad(
        [chunk],
        vad_backend=_TimeoutVadBackend(),
        timeout_per_chunk_s=0.001,
    )

    assert len(result) == 1
    assert result[0]["index"] == 3
    assert result[0]["path"] == str(wav_path)
    assert result[0]["_vad_parent_index"] == 3


def test_vad_refine_passes_threshold_override(tmp_path):
    class RecordingVadBackend(_SplitVadBackend):
        def __init__(self) -> None:
            self.seen_thresholds: list[float | None] = []

        def segment(
            self,
            audio_path: str,
            *,
            target_sr: int = 16000,
            threshold_override: float | None = None,
        ) -> SegmentationResult:
            self.seen_thresholds.append(threshold_override)
            return super().segment(
                audio_path,
                target_sr=target_sr,
                threshold_override=threshold_override,
            )

    wav_path = tmp_path / "chunk_0000.wav"
    _write_wav(wav_path, duration_s=2.0)
    backend = RecordingVadBackend()

    refine_chunks_via_vad(
        [
            {
                "index": 0,
                "start": 5.0,
                "end": 7.0,
                "path": str(wav_path),
                "source_audio_path": "audio.wav",
            }
        ],
        vad_backend=backend,
        threshold_override=0.42,
    )

    assert backend.seen_thresholds == [0.42]
