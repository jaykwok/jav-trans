import os
import re
import uuid
import wave
from pathlib import Path
from typing import Callable


def current_asr_chunk_root() -> Path:
    return Path(os.getenv("ASR_CHUNK_ROOT", Path("tmp") / "chunks")).resolve()


def keep_asr_chunks() -> bool:
    return os.getenv("KEEP_ASR_CHUNKS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
    root = current_asr_chunk_root()
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


# --- where to cut ---------------------------------------------------------
#
# Everything above turns a list of spans into files on disk. What follows picks
# the spans, from the blank runs the CTC alignment head reports.
#
# The value is a persisted wire constant that predates this module: it appears
# in job payloads written by `asr.pipeline`, so the string stays even though the
# pre-gate it was named after is gone.
CHUNK_CUT_SCHEMA = "blank_run_pregate_v1"


def _clamp_spans(
    spans: list[tuple[float, float]], total_s: float
) -> list[tuple[float, float]]:
    clamped: list[tuple[float, float]] = []
    for begin, end in spans:
        begin = max(0.0, min(float(total_s), float(begin)))
        end = max(begin, min(float(total_s), float(end)))
        if end > begin:
            clamped.append((begin, end))
    return sorted(clamped)


def cut_at_pauses(
    blank_spans: list[tuple[float, float]],
    total_s: float,
    *,
    max_s: float = 30.0,
    min_s: float = 2.0,
) -> list[tuple[float, float]]:
    """Contiguous chunks covering ALL of the audio, cut inside pauses.

    This is the second reading of the blank runs and the one that survived. The
    first read them as "what not to decode", and a 2026-07-31 measurement
    falsified that on this domain: the head separates lexically dense speech from
    vocalisation-dense audio rather than speech from silence, so skipping the
    blank stretches lost real lines embedded in moaning. That reading now lives
    only in `tools/align/pregate_reference.py`, as the baseline it is measured
    against.

    Choosing *where to cut* has no such failure mode, because nothing is
    dropped. The output tiles `[0, total_s]` exactly - adjacent chunks share an
    edge and the last one ends at `total_s` - so a mistake here can only put a
    boundary in a slightly worse place, never lose a word. That makes it safe to
    use the very signal that was too weak to gate with.

    Cuts land at the middle of a pause rather than its edge, which is where a
    boundary does least damage to the words on either side.

    **The latest legal pause wins, so chunks run as long as `max_s` allows.**
    This used to aim for a separate `target_s` of 20s, and that cost real
    transcription quality on 2026-08-02: `max_s` is the encoder's own window
    (`asr.pipeline._FEATURE_CHUNK_S`) and the processor pads anything shorter up
    to it, so a 20s chunk buys nothing back for the third of the context it
    gives away. Measured on one 2h09m film, 384 chunks at 20s against 258 at
    30s: `成人になるために` came back as `政治になるために`, a chunk of pure
    vocalisation hallucinated `カレロンか`, and postgate's `repeated_unit` rate
    went 10.5% -> 14.6%. Aiming at the ceiling instead keeps the old 30s decode
    window and still lands every boundary in a pause, which fixed-length cutting
    never did.
    """
    total_s = max(0.0, float(total_s))
    if total_s <= 0.0:
        return []
    if max_s <= 0.0:
        raise ValueError("max_s must be > 0")
    if min_s > max_s:
        raise ValueError("min_s must be <= max_s")
    if total_s <= max_s:
        return [(0.0, total_s)]

    candidates = [
        (begin + end) / 2.0
        for begin, end in _clamp_spans(blank_spans, total_s)
        if 0.0 < (begin + end) / 2.0 < total_s
    ]

    chunks: list[tuple[float, float]] = []
    cursor = 0.0
    while total_s - cursor > max_s:
        window = [
            point
            for point in candidates
            if cursor + min_s <= point <= cursor + max_s
        ]
        # The last pause in the window, so the chunk is as long as `max_s`
        # permits. A hard cut at `max_s` is the fallback when the window holds no
        # pause at all - that is the fixed-length behaviour, and it is what every
        # chunk got before the head was configured.
        cut = max(window) if window else cursor + max_s
        chunks.append((cursor, cut))
        cursor = cut
    chunks.append((cursor, total_s))

    if len(chunks) > 1 and chunks[-1][1] - chunks[-1][0] < min_s:
        # A sliver at the end is not worth its own decode call, and merging it
        # backwards keeps the tiling exact.
        tail = chunks.pop()
        chunks[-1] = (chunks[-1][0], tail[1])
    return chunks
