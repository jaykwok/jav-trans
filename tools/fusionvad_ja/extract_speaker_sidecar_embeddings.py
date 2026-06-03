#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.loading import load_audio_16k_mono
from tools.fusionvad_ja.probe_speaker_sidecar import build_adjacent_speaker_change_rows


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def project_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_blocks(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    blocks = payload.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError(f"missing blocks list in {path}")
    return [dict(block) for block in blocks if isinstance(block, dict)]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _text(block: dict[str, Any]) -> str:
    return str(block.get("ja_text") or block.get("text") or block.get("ja") or "").strip()


def _normalize_embedding(values: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 0 or not math.isfinite(norm):
        arr = np.zeros_like(arr, dtype=np.float32)
        if arr.size:
            arr[0] = 1.0
    else:
        arr = arr / norm
    return [round(float(v), 8) for v in arr.tolist()]


def _slice_audio(audio: np.ndarray, start: float, end: float, sample_rate: int) -> np.ndarray:
    start_i = max(0, min(len(audio), int(round(start * sample_rate))))
    end_i = max(start_i, min(len(audio), int(round(end * sample_rate))))
    return np.asarray(audio[start_i:end_i], dtype=np.float32)


def energy_mfcc_embedding(clip: np.ndarray, sample_rate: int) -> np.ndarray:
    if clip.size == 0:
        return np.zeros(8, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(clip, dtype=np.float32))))
    abs_clip = np.abs(clip)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(clip).astype(np.float32))))) if clip.size > 1 else 0.0
    peak = float(np.max(abs_clip)) if clip.size else 0.0
    duration = clip.size / max(1, sample_rate)
    try:
        import librosa

        mfcc = librosa.feature.mfcc(y=clip.astype(np.float32), sr=sample_rate, n_mfcc=6)
        mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)
    except Exception:
        mfcc_mean = np.zeros(6, dtype=np.float32)
    return np.concatenate(
        [
            np.asarray([rms, zcr, peak, duration], dtype=np.float32),
            mfcc_mean[:4],
        ],
        axis=0,
    )


def build_embedding_rows(
    blocks: list[dict[str, Any]],
    audio: np.ndarray,
    *,
    sample_rate: int,
    backend: str,
    min_duration_s: float,
    max_segments: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {"short": 0, "empty_audio": 0, "backend_unsupported": 0}
    if backend not in {"energy_mfcc", "modelscope_eres2netv2"}:
        raise ValueError(f"unsupported backend: {backend}")
    if backend == "modelscope_eres2netv2":
        skipped["backend_unsupported"] = len(blocks)
        raise RuntimeError(
            "modelscope_eres2netv2 extraction needs the 3D-Speaker speakerlab package. "
            "Use --backend energy_mfcc for smoke, or provide external embeddings to "
            "probe_speaker_sidecar.py until speakerlab is installed."
        )

    for index, block in enumerate(blocks):
        if max_segments > 0 and len(rows) >= max_segments:
            break
        start = _float(block.get("start"))
        end = max(start, _float(block.get("end"), start))
        duration = end - start
        if duration < min_duration_s:
            skipped["short"] += 1
            continue
        clip = _slice_audio(audio, start, end, sample_rate)
        if clip.size == 0:
            skipped["empty_audio"] += 1
            continue
        embedding = energy_mfcc_embedding(clip, sample_rate)
        row = {
            "segment_id": str(block.get("cue_id", index)),
            "cue_id": block.get("cue_id", index),
            "index": index,
            "start": round(start, 6),
            "end": round(end, 6),
            "duration_s": round(duration, 6),
            "text": _text(block)[:160],
            "backend": backend,
            "embedding": _normalize_embedding(embedding),
        }
        source_ids = block.get("source_segment_ids")
        if isinstance(source_ids, list):
            row["source_segment_ids"] = source_ids
        rows.append(row)
    return rows, skipped


def build_summary(
    *,
    bilingual_path: Path,
    audio_path: Path,
    output_dir: Path,
    backend: str,
    min_duration_s: float,
    speaker_threshold: float,
    max_segments: int,
) -> dict[str, Any]:
    blocks = load_blocks(bilingual_path)
    audio, sample_rate = load_audio_16k_mono(str(audio_path))
    rows, skipped = build_embedding_rows(
        blocks,
        audio,
        sample_rate=sample_rate,
        backend=backend,
        min_duration_s=min_duration_s,
        max_segments=max_segments,
    )
    pairs = build_adjacent_speaker_change_rows(rows, threshold=speaker_threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "speaker_embeddings.jsonl"
    pairs_path = output_dir / "speaker_pairs.jsonl"
    write_jsonl(embeddings_path, rows)
    write_jsonl(pairs_path, pairs)
    scores = sorted(float(row["speaker_change_score"]) for row in pairs)
    summary = {
        "source_bilingual": project_rel(bilingual_path),
        "source_audio": project_rel(audio_path),
        "backend": backend,
        "sample_rate": sample_rate,
        "block_count": len(blocks),
        "embedding_count": len(rows),
        "pair_count": len(pairs),
        "speaker_change_count": sum(1 for row in pairs if row.get("speaker_change")),
        "speaker_threshold": speaker_threshold,
        "score_min": round(scores[0], 6) if scores else 0.0,
        "score_p50": round(scores[len(scores) // 2], 6) if scores else 0.0,
        "score_max": round(scores[-1], 6) if scores else 0.0,
        "skipped": skipped,
        "embeddings_path": project_rel(embeddings_path),
        "pairs_path": project_rel(pairs_path),
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    return summary


def build_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Speaker Sidecar Embedding Probe",
            "",
            f"- source: `{summary['source_bilingual']}`",
            f"- audio: `{summary['source_audio']}`",
            f"- backend: `{summary['backend']}`",
            f"- embedding_count: {summary['embedding_count']} / {summary['block_count']}",
            f"- pair_count: {summary['pair_count']}",
            f"- speaker_change_count: {summary['speaker_change_count']}",
            f"- score p50/max: {summary['score_p50']} / {summary['score_max']}",
            f"- skipped: `{summary['skipped']}`",
            f"- embeddings: `{summary['embeddings_path']}`",
            f"- pairs: `{summary['pairs_path']}`",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract offline speaker-sidecar embeddings from subtitle cues and "
            "emit adjacent speaker-change pairs. This does not alter runtime VAD."
        )
    )
    parser.add_argument("--bilingual", required=True, help="bilingual.json path")
    parser.add_argument("--audio", required=True, help="16k/any-rate mono/stereo audio path")
    parser.add_argument(
        "--output-dir",
        default="agents/temp/fusionvad-ja/speaker-sidecar-embeddings",
    )
    parser.add_argument(
        "--backend",
        choices=("energy_mfcc", "modelscope_eres2netv2"),
        default="energy_mfcc",
    )
    parser.add_argument("--min-duration-s", type=float, default=0.45)
    parser.add_argument("--speaker-threshold", type=float, default=0.35)
    parser.add_argument("--max-segments", type=int, default=0)
    args = parser.parse_args(argv)

    summary = build_summary(
        bilingual_path=project_path(args.bilingual),
        audio_path=project_path(args.audio),
        output_dir=project_path(args.output_dir),
        backend=args.backend,
        min_duration_s=float(args.min_duration_s),
        speaker_threshold=float(args.speaker_threshold),
        max_segments=int(args.max_segments),
    )
    print(
        "summary={summary} embeddings={embeddings} pairs={pairs} changes={changes}".format(
            summary=project_rel(project_path(args.output_dir) / "summary.json"),
            embeddings=summary["embedding_count"],
            pairs=summary["pair_count"],
            changes=summary["speaker_change_count"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
