import os
import re
import uuid
import wave
from pathlib import Path
from typing import Callable


_KEEP_ASR_CHUNKS = os.getenv("KEEP_ASR_CHUNKS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_ASR_CHUNK_ROOT = Path(
    os.getenv("ASR_CHUNK_ROOT", Path("tmp") / "chunks")
).resolve()
_LAST_BOUNDARY_SIGNATURE: dict = {}


def _get_wav_duration(audio_path: str) -> float:
    with wave.open(audio_path, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
    return frames / rate if rate else 0.0


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
        valid_spans: list[tuple[int, int, int, float, float]] = []

        for span_index, (start, end) in enumerate(spans):
            start_frame = min(total_frames, max(0, int(start * frame_rate)))
            end_frame = min(total_frames, max(0, int(end * frame_rate)))
            if end_frame <= start_frame:
                continue
            valid_spans.append(
                (
                    span_index,
                    start_frame,
                    end_frame,
                    start_frame / frame_rate,
                    end_frame / frame_rate,
                )
            )

        total_chunks = len(valid_spans)

        for idx, (span_index, start_frame, end_frame, start_time, end_time) in enumerate(
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
                    "source_span_index": span_index,
                    "start": start_time,
                    "end": end_time,
                    "path": str(chunk_path),
                    "source_audio_path": source_audio_path,
                }
            )

    return chunk_dir, chunk_infos


def _chunk_duration(chunk: dict) -> float:
    return max(0.0, float(chunk.get("end", 0.0)) - float(chunk.get("start", 0.0)))
