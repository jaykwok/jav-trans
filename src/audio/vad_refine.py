from __future__ import annotations

import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

from vad.base import VadBackend


def refine_chunks_via_vad(
    chunks: list[dict],
    *,
    vad_backend: VadBackend,
    threshold_override: float = 0.25,
    min_silence_ms: int = 200,
    min_speech_ms: int = 250,
    timeout_per_chunk_s: float = 30.0,
) -> list[dict]:
    """VAD-subdivide QC-failed chunks for re-transcription.

    Each returned dict preserves the original keys plus a '_vad_parent_index'
    int so the caller can group sub-chunks back to the original position and
    merge text results.

    On VAD returning <=1 segment or timeout/exception the original chunk is
    returned unchanged (with '_vad_parent_index' added).  threshold_override,
    min_silence_ms, min_speech_ms are API parameters; the vad_backend uses its
    own configured thresholds internally.
    """
    result: list[dict] = []
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        for chunk in chunks:
            original_index = int(chunk.get("index", 0))
            try:
                future = executor.submit(vad_backend.segment, chunk["path"])
                seg_result = future.result(timeout=timeout_per_chunk_s)
            except FuturesTimeoutError:
                executor.shutdown(wait=False, cancel_futures=True)
                executor = ThreadPoolExecutor(max_workers=1)
                result.append({**chunk, "_vad_parent_index": original_index})
                continue
            except Exception:
                result.append({**chunk, "_vad_parent_index": original_index})
                continue

            segments = seg_result.segments
            if len(segments) <= 1:
                result.append({**chunk, "_vad_parent_index": original_index})
                continue

            sub_chunks = _slice_wav_at_segments(chunk, segments)
            if sub_chunks:
                result.extend(sub_chunks)
            else:
                result.append({**chunk, "_vad_parent_index": original_index})
    finally:
        executor.shutdown(wait=False)
    return result


def _slice_wav_at_segments(chunk: dict, segments) -> list[dict]:
    """Slice a chunk WAV into sub-WAVs at VAD speech segment boundaries."""
    chunk_path = chunk["path"]
    chunk_start = float(chunk.get("start", 0.0))
    chunk_dir = Path(chunk_path).parent
    original_index = int(chunk.get("index", 0))
    sub_chunks: list[dict] = []

    try:
        with wave.open(chunk_path, "rb") as wav:
            params = wav.getparams()
            frame_rate = wav.getframerate()
            total_frames = wav.getnframes()
    except Exception:
        return []

    _MIN_SUB_CHUNK_FRAMES = int(0.5 * frame_rate)

    for sub_idx, seg in enumerate(segments):
        start_frame = max(0, int(seg.start * frame_rate))
        end_frame = min(total_frames, int(seg.end * frame_rate))
        if end_frame - start_frame < _MIN_SUB_CHUNK_FRAMES:
            continue

        sub_path = chunk_dir / (
            f"refine_{Path(chunk_path).stem}_{sub_idx:04d}_{uuid.uuid4().hex[:6]}.wav"
        )
        try:
            with wave.open(chunk_path, "rb") as wav:
                wav.setpos(start_frame)
                frames = wav.readframes(end_frame - start_frame)
            with wave.open(str(sub_path), "wb") as out:
                out.setparams(params)
                out.writeframes(frames)
        except Exception:
            continue

        sub_chunks.append(
            {
                "index": original_index * 100000 + sub_idx,
                "start": chunk_start + seg.start,
                "end": chunk_start + seg.end,
                "path": str(sub_path),
                "source_audio_path": chunk.get("source_audio_path", ""),
                "_vad_parent_index": original_index,
            }
        )

    return sub_chunks
