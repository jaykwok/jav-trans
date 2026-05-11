import os
import re
import uuid
import wave
from pathlib import Path
from typing import Callable


_SEGMENT_MIN_SILENCE_S = float(os.getenv("SEGMENT_MIN_SILENCE", "0.35"))
_SEGMENT_MIN_CHUNK_S = float(os.getenv("SEGMENT_MIN_CHUNK", "1.2"))
_SEGMENT_MAX_CHUNK_S = float(os.getenv("SEGMENT_MAX_CHUNK", "18.0"))
_SEGMENT_TARGET_CHUNK_S = min(
    _SEGMENT_MAX_CHUNK_S,
    float(os.getenv("SEGMENT_TARGET_CHUNK", str(min(60.0, _SEGMENT_MAX_CHUNK_S)))),
)
_SEGMENT_MIN_SPEECH_S = float(os.getenv("SEGMENT_MIN_SPEECH", "0.25"))
_SEGMENT_PAD_S = float(os.getenv("SEGMENT_PAD", "0.15"))
_VAD_MERGE_SHORT_MAX_S = float(os.getenv("VAD_MERGE_SHORT_MAX_S", "0.8"))
_VAD_MERGE_GAP_MAX_S = float(os.getenv("VAD_MERGE_GAP_MAX_S", "0.3"))
_SEGMENT_SILENCE_DB = os.getenv("SEGMENT_SILENCE_DB", "-32dB").strip()
_KEEP_ASR_CHUNKS = os.getenv("KEEP_ASR_CHUNKS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_ASR_CHUNK_ROOT = Path(
    os.getenv("ASR_CHUNK_ROOT", Path("temp") / "chunks")
).resolve()
_LAST_VAD_SIGNATURE: dict = {}


def _get_wav_duration(audio_path: str) -> float:
    with wave.open(audio_path, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
    return frames / rate if rate else 0.0


def _build_processing_spans(audio_path: str) -> list[tuple[float, float]]:
    global _LAST_VAD_SIGNATURE
    from vad import get_vad_backend

    vad = get_vad_backend()
    result = vad.segment(audio_path)
    _LAST_VAD_SIGNATURE = result.parameters
    spans = [(g[0].start, g[-1].end) for g in result.groups]
    return spans or [(0.0, result.audio_duration_sec)]


def _extract_wav_chunks(
    audio_path: str,
    spans: list[tuple[float, float]],
    on_stage: Callable[[str], None] | None = None,
) -> tuple[Path, list[dict]]:
    root = _ASR_CHUNK_ROOT
    root.mkdir(parents=True, exist_ok=True)
    source_audio_path = str(Path(audio_path).resolve())
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(audio_path).stem)
    chunk_dir = root / f"{safe_prefix}_{uuid.uuid4().hex[:8]}"
    chunk_dir.mkdir(parents=True, exist_ok=False)

    chunk_infos: list[dict] = []
    with wave.open(audio_path, "rb") as reader:
        params = reader.getparams()
        frame_rate = reader.getframerate()
        total_frames = reader.getnframes()
        min_chunk_frames = max(1, int(_SEGMENT_MIN_SPEECH_S * frame_rate))
        valid_spans: list[tuple[int, int, float, float]] = []

        for start, end in spans:
            start_frame = max(0, int(start * frame_rate))
            end_frame = min(total_frames, int(end * frame_rate))
            if end_frame - start_frame < min_chunk_frames:
                continue
            valid_spans.append(
                (
                    start_frame,
                    end_frame,
                    start_frame / frame_rate,
                    end_frame / frame_rate,
                )
            )

        total_chunks = len(valid_spans)

        for idx, (start_frame, end_frame, start_time, end_time) in enumerate(
            valid_spans, 1
        ):
            if on_stage:
                on_stage(f"音频切块 {idx}/{total_chunks}...")

            reader.setpos(start_frame)
            frames = reader.readframes(end_frame - start_frame)

            chunk_path = chunk_dir / f"chunk_{idx - 1:04d}.wav"
            with wave.open(str(chunk_path), "wb") as writer:
                writer.setparams(params)
                writer.writeframes(frames)

            chunk_infos.append(
                {
                    "index": idx - 1,
                    "start": start_time,
                    "end": end_time,
                    "path": str(chunk_path),
                    "source_audio_path": source_audio_path,
                }
            )

    return chunk_dir, chunk_infos


def _chunk_duration(chunk: dict) -> float:
    return max(0.0, float(chunk.get("end", 0.0)) - float(chunk.get("start", 0.0)))


def _chunk_original_boundaries(chunk: dict) -> list[dict]:
    existing = chunk.get("merged_from")
    if isinstance(existing, list) and existing:
        boundaries = []
        for item in existing:
            if not isinstance(item, dict):
                continue
            try:
                boundaries.append(
                    {
                        "index": int(item.get("index", len(boundaries))),
                        "start": float(item["start"]),
                        "end": float(item["end"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        if boundaries:
            return boundaries

    return [
        {
            "index": int(chunk.get("index", 0)),
            "start": float(chunk.get("start", 0.0)),
            "end": float(chunk.get("end", 0.0)),
        }
    ]


def _can_merge_short_vad_chunks(left: dict, right: dict) -> bool:
    left_duration = _chunk_duration(left)
    right_duration = _chunk_duration(right)
    if left_duration >= _VAD_MERGE_SHORT_MAX_S or right_duration >= _VAD_MERGE_SHORT_MAX_S:
        return False

    gap = float(right.get("start", 0.0)) - float(left.get("end", 0.0))
    if gap < 0.0 or gap >= _VAD_MERGE_GAP_MAX_S:
        return False

    merged_duration = float(right.get("end", 0.0)) - float(left.get("start", 0.0))
    return 0.0 < merged_duration <= _SEGMENT_MAX_CHUNK_S


def _write_merged_vad_chunk(
    chunk_dir: Path,
    left: dict,
    right: dict,
    merge_index: int,
) -> dict | None:
    source_audio_path = str(left.get("source_audio_path") or "")
    if not source_audio_path or source_audio_path != str(right.get("source_audio_path") or ""):
        return None

    source_path = Path(source_audio_path)
    if not source_path.exists():
        return None

    start_time = float(left.get("start", 0.0))
    end_time = float(right.get("end", start_time))
    if end_time <= start_time:
        return None

    merged_path = chunk_dir / (
        f"chunk_{int(left.get('index', 0)):04d}_merge_{int(right.get('index', 0)):04d}_{merge_index:03d}.wav"
    )
    try:
        with wave.open(str(source_path), "rb") as reader:
            params = reader.getparams()
            frame_rate = reader.getframerate()
            total_frames = reader.getnframes()
            start_frame = max(0, int(start_time * frame_rate))
            end_frame = min(total_frames, int(end_time * frame_rate))
            if end_frame <= start_frame:
                return None
            reader.setpos(start_frame)
            frames = reader.readframes(end_frame - start_frame)

        with wave.open(str(merged_path), "wb") as writer:
            writer.setparams(params)
            writer.writeframes(frames)
    except Exception:
        return None

    merged = dict(left)
    merged.update(
        {
            "start": start_time,
            "end": end_time,
            "path": str(merged_path),
            "source_audio_path": source_audio_path,
            "merged_from": _chunk_original_boundaries(left)
            + _chunk_original_boundaries(right),
            "vad_merge": "short_adjacent",
        }
    )
    return merged


def _merge_short_vad_chunks(
    chunk_dir: Path,
    chunks: list[dict],
    on_stage: Callable[[str], None] | None = None,
) -> list[dict]:
    if (
        len(chunks) < 2
        or _VAD_MERGE_SHORT_MAX_S <= 0.0
        or _VAD_MERGE_GAP_MAX_S <= 0.0
    ):
        return chunks

    merged_chunks: list[dict] = []
    merge_count = 0
    idx = 0
    while idx < len(chunks):
        current = chunks[idx]
        idx += 1
        while idx < len(chunks) and _can_merge_short_vad_chunks(current, chunks[idx]):
            merged = _write_merged_vad_chunk(
                chunk_dir,
                current,
                chunks[idx],
                merge_count,
            )
            if merged is None:
                break
            current = merged
            merge_count += 1
            idx += 1
        merged_chunks.append(current)

    if merge_count and on_stage:
        on_stage(f"VAD 微短段预合并 {len(chunks)} -> {len(merged_chunks)} 个处理块")
    return merged_chunks
