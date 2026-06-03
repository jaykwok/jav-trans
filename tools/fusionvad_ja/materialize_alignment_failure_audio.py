#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.loading import load_audio_16k_mono  # noqa: E402


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "review_type",
        "failure_bucket",
        "sample_id",
        "audio",
        "duration_s",
        "source_audio_path",
        "source_start_s",
        "source_end_s",
        "chunk_start_s",
        "chunk_end_s",
        "display_text",
        "manual_label",
        "manual_text",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_stem(value: Any) -> str:
    raw = str(value or "sample")
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in raw)
    return clean.strip("_") or "sample"


def row_float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def resolve_source_audio(row: Mapping[str, Any]) -> Path:
    value = str(row.get("source_audio_path") or row.get("audio") or "")
    if not value:
        raise ValueError("missing source_audio_path")
    return project_path(value)


def source_span(row: Mapping[str, Any], *, pad_s: float, audio_duration_s: float) -> tuple[float, float, float, float]:
    chunk_start = max(0.0, row_float(row, "start"))
    chunk_end = max(chunk_start, row_float(row, "end"))
    if chunk_end <= chunk_start:
        chunk_end = chunk_start + max(0.02, row_float(row, "duration_s"))
    clip_start = max(0.0, chunk_start - pad_s)
    clip_end = min(audio_duration_s, max(clip_start + 0.02, chunk_end + pad_s))
    return clip_start, clip_end, chunk_start, chunk_end


def slice_audio(audio: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray:
    start_sample = max(0, min(len(audio), int(round(start_s * sample_rate))))
    end_sample = max(0, min(len(audio), int(round(end_s * sample_rate))))
    if end_sample <= start_sample:
        return np.zeros(max(1, int(round(0.02 * sample_rate))), dtype=np.float32)
    return np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)


def materialize_rows(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    pad_s: float,
    skip_missing_audio: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[Path, tuple[np.ndarray, int]] = {}
    out_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        sample_id = safe_stem(row.get("sample_id") or f"failure-{index:06d}")
        try:
            source_audio = resolve_source_audio(row)
            if not source_audio.exists():
                raise FileNotFoundError(str(source_audio))
            if source_audio not in cache:
                cache[source_audio] = load_audio_16k_mono(str(source_audio))
            audio, sample_rate = cache[source_audio]
            audio_duration_s = len(audio) / sample_rate if sample_rate > 0 else 0.0
            clip_start, clip_end, chunk_start, chunk_end = source_span(
                row,
                pad_s=pad_s,
                audio_duration_s=audio_duration_s,
            )
            clip = slice_audio(audio, sample_rate, clip_start, clip_end)
            output_audio = audio_dir / f"{sample_id}.wav"
            sf.write(str(output_audio), clip, sample_rate)
            clip_duration = len(clip) / sample_rate if sample_rate > 0 else 0.0
            out_rows.append(
                {
                    **row,
                    "audio": project_rel(output_audio),
                    "duration_s": round(clip_duration, 3),
                    "sample_rate": sample_rate,
                    "source_audio_path": project_rel(source_audio),
                    "source_start_s": round(clip_start, 3),
                    "source_end_s": round(clip_end, 3),
                    "source_duration_s": round(audio_duration_s, 3),
                    "chunk_start_s": round(max(0.0, chunk_start - clip_start), 3),
                    "chunk_end_s": round(min(clip_duration, chunk_end - clip_start), 3),
                    "materialized_from_start_s": round(chunk_start, 3),
                    "materialized_from_end_s": round(chunk_end, 3),
                    "materialized_pad_s": round(pad_s, 3),
                }
            )
        except Exception as exc:
            error = {
                "sample_id": sample_id,
                "source_audio_path": str(row.get("source_audio_path") or row.get("audio") or ""),
                "error": str(exc),
            }
            errors.append(error)
            if not skip_missing_audio:
                raise
    return out_rows, errors


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(project_path(args.manifest))
    if args.limit is not None:
        rows = rows[: args.limit]
    materialized, errors = materialize_rows(
        rows,
        output_dir=output_dir,
        pad_s=args.pad_s,
        skip_missing_audio=args.skip_missing_audio,
    )
    manifest_path = output_dir / "alignment_failure_audio_manifest.jsonl"
    csv_path = output_dir / "alignment_failure_audio_manifest.csv"
    errors_path = output_dir / "alignment_failure_audio_errors.json"
    summary_path = output_dir / "alignment_failure_audio_summary.json"
    write_jsonl(manifest_path, materialized)
    write_csv(csv_path, materialized)
    errors_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "source_manifest": project_rel(project_path(args.manifest)),
        "input_rows": len(rows),
        "materialized_rows": len(materialized),
        "errors": len(errors),
        "pad_s": args.pad_s,
        "manifest_jsonl": project_rel(manifest_path),
        "manifest_csv": project_rel(csv_path),
        "errors_json": project_rel(errors_path),
        "review_type_counts": dict(sorted(Counter(str(row.get("review_type") or "") for row in materialized).items())),
        "failure_bucket_counts": dict(sorted(Counter(str(row.get("failure_bucket") or "") for row in materialized).items())),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize WAV clips for alignment failure review manifests."
    )
    parser.add_argument("--manifest", required=True, help="alignment_failure_manifest.jsonl")
    parser.add_argument(
        "--output-dir",
        default="agents/audits/alignment-failure-audio",
    )
    parser.add_argument("--pad-s", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-missing-audio", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.pad_s < 0.0:
        parser.error("--pad-s must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(f"manifest={summary['manifest_jsonl']}")
    print(f"csv={summary['manifest_csv']}")
    print(f"rows={summary['materialized_rows']} errors={summary['errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
