"""Transcribe full videos with Grok STT through the shared STT adapter.

The tool owns resumable media chunking, cost preflight, concurrency and output
assembly. Provider authentication and response normalization stay in
``speech_to_text_transport``. Extracted chunks preserve source PTS gaps so the
returned local word times remain on the video timeline.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

from tools.omni.audio_teacher_batch import (
    iter_completed_audio_teacher_items,
    resolve_worker_count,
)
from tools.omni.speech_to_text_transport import (
    SpeechToTextTransport,
    create_speech_to_text_transport,
)


SCHEMA = "grok_fullfilm_stt_v1"
DEFAULT_MODEL = "x-ai/grok-stt-1.0"
DEFAULT_PRICE_PER_HOUR_USD = 0.10
DEFAULT_CHUNK_S = 300.0
DEFAULT_OVERLAP_S = 5.0
DEFAULT_MAX_COST_USD = 10.0
TIMELINE_FILTER = "aresample=16000:async=1000:first_pts=0"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    os.replace(temporary, path)


def parse_video_spec(value: str) -> dict[str, str]:
    """Parse ``FILM_ID=PATH`` or derive the id from a plain path."""
    raw = str(value or "").strip()
    if not raw:
        raise argparse.ArgumentTypeError("video specification cannot be empty")
    if "=" in raw:
        film_id, path_text = raw.split("=", 1)
        film_id = film_id.strip()
        path = Path(path_text.strip()).expanduser()
        if not film_id or not path_text.strip():
            raise argparse.ArgumentTypeError("use FILM_ID=PATH")
    else:
        path = Path(raw).expanduser()
        film_id = path.stem
    if not film_id:
        raise argparse.ArgumentTypeError("film id cannot be empty")
    return {"film_id": film_id, "source": str(path)}


def probe_duration_s(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    duration = float(completed.stdout.strip())
    if duration <= 0.0:
        raise ValueError(f"media duration must be positive: {path}")
    return duration


def build_manifest(
    videos: Iterable[dict[str, str]],
    *,
    output_dir: Path,
    model: str,
    chunk_s: float,
    overlap_s: float,
    price_per_hour_usd: float,
    max_cost_usd: float,
) -> dict[str, Any]:
    if chunk_s <= 0.0:
        raise ValueError("chunk_s must be positive")
    if overlap_s < 0.0 or overlap_s >= chunk_s / 2.0:
        raise ValueError("overlap_s must be non-negative and less than half chunk_s")
    if price_per_hour_usd < 0.0 or max_cost_usd <= 0.0:
        raise ValueError("price and budget must be positive")

    film_rows: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    requested_audio_s = 0.0
    seen_ids: set[str] = set()
    for video in videos:
        film_id = str(video["film_id"])
        source = Path(video["source"]).resolve()
        if film_id in seen_ids:
            raise ValueError(f"duplicate film id: {film_id}")
        seen_ids.add(film_id)
        if not source.is_file():
            raise FileNotFoundError(source)
        duration_s = probe_duration_s(source)
        chunk_count = int(math.ceil(duration_s / chunk_s))
        film_rows.append(
            {
                "film_id": film_id,
                "source": str(source),
                "duration_s": round(duration_s, 6),
                "chunk_count": chunk_count,
                "base_media_cost_usd": round(
                    duration_s / 3600.0 * price_per_hour_usd,
                    9,
                ),
            }
        )
        for index in range(chunk_count):
            nominal_start = index * chunk_s
            nominal_end = min(duration_s, (index + 1) * chunk_s)
            request_start = max(0.0, nominal_start - overlap_s)
            request_end = min(duration_s, nominal_end + overlap_s)
            request_duration = request_end - request_start
            requested_audio_s += request_duration
            chunk_id = f"{film_id}-{index:04d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "film_id": film_id,
                    "chunk_index": index,
                    "source": str(source),
                    "audio": str(output_dir / "audio" / f"{chunk_id}.mp3"),
                    "nominal_start_s": round(nominal_start, 6),
                    "nominal_end_s": round(nominal_end, 6),
                    "request_start_s": round(request_start, 6),
                    "request_end_s": round(request_end, 6),
                    "request_duration_s": round(request_duration, 6),
                    "estimated_cost_usd": round(
                        request_duration / 3600.0 * price_per_hour_usd,
                        9,
                    ),
                }
            )

    if not film_rows:
        raise ValueError("at least one video is required")
    estimated_cost = requested_audio_s / 3600.0 * price_per_hour_usd
    if estimated_cost > max_cost_usd + 1e-12:
        raise RuntimeError(
            f"preflight refused: estimated ${estimated_cost:.6f} exceeds "
            f"${max_cost_usd:.2f} budget"
        )
    return {
        "schema": SCHEMA,
        "model": model,
        "diarize": True,
        "timeline_filter": TIMELINE_FILTER,
        "speaker_cut_rule": "different speaker and non-overlapping adjacent words",
        "chunk_s": chunk_s,
        "overlap_s": overlap_s,
        "price_per_hour_usd": price_per_hour_usd,
        "max_cost_usd": max_cost_usd,
        "requested_audio_hours": round(requested_audio_s / 3600.0, 9),
        "estimated_cost_usd": round(estimated_cost, 9),
        "films": film_rows,
        "chunks": chunks,
    }


def write_or_validate_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(
                f"existing manifest differs from this run; use a new output directory: {path}"
            )
        return
    _atomic_json(path, manifest)


def build_chunk_command(chunk: dict[str, Any], output: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(chunk["request_start_s"]),
        "-t",
        str(chunk["request_duration_s"]),
        "-i",
        str(chunk["source"]),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        TIMELINE_FILTER,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "32k",
        "-y",
        str(output),
    ]


def prepare_chunk(chunk: dict[str, Any]) -> None:
    output = Path(chunk["audio"])
    if output.is_file() and output.stat().st_size > 0:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.mp3")
    subprocess.run(build_chunk_command(chunk, temporary), check=True)
    os.replace(temporary, output)


def prepare_audio(manifest: dict[str, Any], workers: int) -> None:
    chunks = list(manifest["chunks"])
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(prepare_chunk, chunk): chunk for chunk in chunks}
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(chunks):
                print(f"audio prepared {completed}/{len(chunks)}", flush=True)


def _call_chunk(
    chunk: dict[str, Any],
    *,
    transport: SpeechToTextTransport,
    model: str,
    language: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = transport.transcribe(
        audio_path=Path(chunk["audio"]),
        language=language,
        diarize=True,
        filler_words=False,
        vad_threshold=0.5,
    )
    return {
        "schema": SCHEMA,
        "chunk": chunk,
        "model": model,
        "transport": transport.transport_name,
        "generation_id": result.response_headers.get("x-generation-id", ""),
        "latency_s": round(time.perf_counter() - started, 6),
        "parsed": result.parsed,
        "response": result.raw,
    }


def transcribe(
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    transport: SpeechToTextTransport,
    workers: int,
    attempts: int,
    language: str,
) -> None:
    responses = output_dir / "responses"
    responses.mkdir(parents=True, exist_ok=True)
    tasks = [
        chunk
        for chunk in manifest["chunks"]
        if not (responses / f"{chunk['chunk_id']}.json").is_file()
    ]
    print(f"provider calls pending {len(tasks)}/{len(manifest['chunks'])}", flush=True)
    if not tasks:
        return
    worker_count = resolve_worker_count(
        requested=workers,
        provider_limit=transport.max_concurrency,
        item_count=len(tasks),
    )

    def run_one(chunk: dict[str, Any]) -> None:
        output = responses / f"{chunk['chunk_id']}.json"
        last_error: Exception | None = None
        for attempt in range(1, max(1, attempts) + 1):
            try:
                record = _call_chunk(
                    chunk,
                    transport=transport,
                    model=str(manifest["model"]),
                    language=language,
                )
                record["attempts"] = attempt
                _atomic_json(output, record)
                return
            except Exception as error:  # noqa: BLE001
                last_error = error
                if attempt < attempts:
                    time.sleep(float(attempt))
        raise RuntimeError(f"{chunk['chunk_id']}: {last_error}")

    completed = len(manifest["chunks"]) - len(tasks)
    for item in iter_completed_audio_teacher_items(
        items=tasks,
        worker=run_one,
        max_workers=max(1, worker_count),
    ):
        completed += 1
        print(
            f"provider completed {completed}/{len(manifest['chunks'])} "
            f"chunk={item.item['chunk_id']}",
            flush=True,
        )


def normalized_words(record: dict[str, Any]) -> list[dict[str, Any]]:
    chunk = record["chunk"]
    request_start = float(chunk["request_start_s"])
    nominal_start = float(chunk["nominal_start_s"])
    nominal_end = float(chunk["nominal_end_s"])
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(record["parsed"].get("words") or []):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        try:
            local_start = float(raw["start_s"])
            local_end = float(raw["end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        absolute_start = request_start + local_start
        absolute_end = request_start + max(local_start, local_end)
        midpoint = (absolute_start + absolute_end) / 2.0
        if midpoint < nominal_start or midpoint >= nominal_end:
            continue
        speaker = raw.get("speaker")
        result.append(
            {
                "film_id": chunk["film_id"],
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "word_index": index,
                "text": text,
                "start_s": round(absolute_start, 6),
                "end_s": round(absolute_end, 6),
                "speaker": speaker,
                "speaker_id": f"{chunk['chunk_id']}:S{speaker}",
                "confidence": raw.get("confidence"),
            }
        )
    return result


def speaker_change_rows(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for previous, current in zip(words, words[1:]):
        if previous["film_id"] != current["film_id"]:
            continue
        if previous["chunk_id"] != current["chunk_id"]:
            continue
        if previous["speaker"] == current["speaker"]:
            continue
        overlap_s = max(0.0, float(previous["end_s"]) - float(current["start_s"]))
        accepted = overlap_s == 0.0
        rows.append(
            {
                "film_id": current["film_id"],
                "chunk_id": current["chunk_id"],
                "previous_speaker": previous["speaker"],
                "next_speaker": current["speaker"],
                "previous_text": previous["text"],
                "next_text": current["text"],
                "previous_end_s": previous["end_s"],
                "next_start_s": current["start_s"],
                "overlap_s": round(overlap_s, 6),
                "accepted": accepted,
                "cut_s": round(
                    (float(previous["end_s"]) + float(current["start_s"])) / 2.0,
                    6,
                )
                if accepted
                else None,
            }
        )
    return rows


def compile_outputs(manifest: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    response_dir = output_dir / "responses"
    words: list[dict[str, Any]] = []
    actual_cost = 0.0
    for chunk in manifest["chunks"]:
        path = response_dir / f"{chunk['chunk_id']}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("chunk") != chunk or record.get("model") != manifest["model"]:
            raise RuntimeError(f"response does not match manifest: {path}")
        words.extend(normalized_words(record))
        try:
            actual_cost += float(record["parsed"].get("usage", {}).get("cost") or 0.0)
        except (TypeError, ValueError):
            pass
    words.sort(
        key=lambda word: (
            str(word["film_id"]),
            float(word["start_s"]),
            float(word["end_s"]),
        )
    )
    cuts = speaker_change_rows(words)
    accepted = [row for row in cuts if row["accepted"]]
    _write_jsonl(output_dir / "grok.words.jsonl", words)
    _write_jsonl(output_dir / "grok.speaker_cuts.jsonl", cuts)
    by_film: dict[str, list[dict[str, Any]]] = {}
    for word in words:
        by_film.setdefault(str(word["film_id"]), []).append(word)
    summary = {
        "schema": SCHEMA,
        "model": manifest["model"],
        "diarize": True,
        "word_count": len(words),
        "speaker_change_count": len(cuts),
        "accepted_nonoverlap_speaker_cuts": len(accepted),
        "rejected_overlapping_speaker_changes": len(cuts) - len(accepted),
        "estimated_cost_usd": manifest["estimated_cost_usd"],
        "provider_actual_cost_usd": round(actual_cost, 9),
        "films": {
            film_id: {
                "word_count": len(film_words),
                "speaker_change_count": sum(
                    row["film_id"] == film_id for row in cuts
                ),
                "accepted_nonoverlap_speaker_cuts": sum(
                    row["film_id"] == film_id for row in accepted
                ),
            }
            for film_id, film_words in by_film.items()
        },
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        action="append",
        required=True,
        type=parse_video_spec,
        help="Repeatable FILM_ID=PATH; a plain PATH uses its stem as FILM_ID.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", default="openrouter")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path.home() / ".config" / "omni" / "openrouter",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider-slug", default="xai")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--chunk-s", type=float, default=DEFAULT_CHUNK_S)
    parser.add_argument("--overlap-s", type=float, default=DEFAULT_OVERLAP_S)
    parser.add_argument(
        "--price-per-hour-usd",
        type=float,
        default=DEFAULT_PRICE_PER_HOUR_USD,
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=DEFAULT_MAX_COST_USD,
    )
    parser.add_argument("--audio-workers", type=int, default=2)
    parser.add_argument("--api-workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        args.video,
        output_dir=output_dir,
        model=args.model,
        chunk_s=args.chunk_s,
        overlap_s=args.overlap_s,
        price_per_hour_usd=args.price_per_hour_usd,
        max_cost_usd=args.max_cost_usd,
    )
    write_or_validate_manifest(output_dir / "manifest.json", manifest)
    print(
        f"preflight films={len(manifest['films'])} chunks={len(manifest['chunks'])} "
        f"audio={manifest['requested_audio_hours']:.3f}h "
        f"max_cost=${manifest['estimated_cost_usd']:.4f}",
        flush=True,
    )
    prepare_audio(manifest, args.audio_workers)
    if args.prepare_only:
        return
    transport = create_speech_to_text_transport(
        profile=args.profile,
        env_file=args.env_file,
        model_override=args.model,
        timeout_s=args.timeout_s,
        provider_option_slug=args.provider_slug,
    )
    transcribe(
        manifest,
        output_dir=output_dir,
        transport=transport,
        workers=args.api_workers,
        attempts=args.attempts,
        language=args.language,
    )
    summary = compile_outputs(manifest, output_dir=output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
