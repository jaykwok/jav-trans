#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import soundfile as sf

from boundary.ja import sample_hf_audio_16k_mono, stable_hf_audio_id


def load_dataset_stream(*, name: str, split: str, decode_audio: bool = True):
    try:
        from datasets import Audio, Features, Value, load_dataset
    except ImportError as exc:
        raise SystemExit("datasets is required for HF audio materialization: uv pip install datasets") from exc
    features = Features(
        {
            "ogg": Value("binary"),
            "txt": Value("string"),
            "__key__": Value("string"),
            "__url__": Value("string"),
        }
    )
    if name == "litagin/Galgame_Speech_ASR_16kHz":
        return load_dataset(name, split=split, streaming=True, features=features)
    dataset = load_dataset(name, split=split, streaming=True)
    if not decode_audio:
        try:
            dataset_features = dataset.features or {}
            if "audio" in dataset_features and isinstance(dataset_features["audio"], Audio):
                dataset = dataset.cast_column("audio", Audio(decode=False))
        except Exception:
            pass
    return dataset


def materialize_from_dataset(
    *,
    dataset,
    args: argparse.Namespace,
    output_dir: Path,
    audio_dir: Path,
    decode_mode: str,
    existing_rows_by_input: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    existing_rows_by_input = existing_rows_by_input or {}
    if args.shuffle_buffer_size > 0:
        dataset = dataset.shuffle(seed=args.shuffle_seed, buffer_size=args.shuffle_buffer_size)
    for index, example in enumerate(dataset):
        if index < args.start_index:
            continue
        if index >= args.start_index + args.limit:
            break
        input_id = f"{args.dataset}:{args.split}:{index}"
        try:
            existing_row = existing_rows_by_input.get(input_id)
            if args.reuse_existing and existing_row and Path(str(existing_row.get("audio") or "")).exists():
                manifest_rows.append(dict(existing_row))
                continue
            audio, sample_rate = sample_hf_audio_16k_mono(example)
            audio_id = str(
                example.get("__key__")
                or example.get("id")
                or stable_hf_audio_id(dataset_name=args.dataset, split=args.split, index=index)
            )
            text = str(
                example.get("txt")
                or example.get("text")
                or example.get("transcription")
                or example.get("transcript")
                or example.get("sentence")
                or ""
            )
            output_audio_path = audio_dir / f"{Path(audio_id).stem[:80]}.wav"
            if not output_audio_path.exists() or not args.reuse_existing:
                sf.write(str(output_audio_path), audio, sample_rate)
            manifest_rows.append(
                {
                    "input": input_id,
                    "audio_id": audio_id,
                    "audio": str(output_audio_path),
                    "duration_s": len(audio) / sample_rate if sample_rate else 0.0,
                    "sample_rate": sample_rate,
                    "source": args.dataset,
                    "text": text,
                    "decode_mode": decode_mode,
                }
            )
        except Exception as exc:
            manifest_rows.append(
                {
                    "input": input_id,
                    "source": args.dataset,
                    "decode_mode": decode_mode,
                    "error": str(exc),
                }
            )
    return manifest_rows


def materialize(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    existing_rows_by_input = load_existing_rows(output_dir / "hf_audio_manifest.json") if args.reuse_existing else {}
    try:
        dataset = load_dataset_stream(name=args.dataset, split=args.split, decode_audio=True)
        manifest_rows = materialize_from_dataset(
            dataset=dataset,
            args=args,
            output_dir=output_dir,
            audio_dir=audio_dir,
            decode_mode="torchcodec",
            existing_rows_by_input=existing_rows_by_input,
        )
    except Exception as exc:
        manifest_rows = [
            {
                "input": f"{args.dataset}:{args.split}:load",
                "source": args.dataset,
                "decode_mode": "torchcodec",
                "error": str(exc),
            }
        ]
    valid = [row for row in manifest_rows if "error" not in row]
    if not valid and args.fallback_decode_false:
        dataset = load_dataset_stream(name=args.dataset, split=args.split, decode_audio=False)
        manifest_rows = materialize_from_dataset(
            dataset=dataset,
            args=args,
            output_dir=output_dir,
            audio_dir=audio_dir,
            decode_mode="decode_false",
            existing_rows_by_input=existing_rows_by_input,
        )
    manifest_path = output_dir / "hf_audio_manifest.json"
    summary_path = output_dir / "hf_audio_summary.json"
    manifest_path.write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    valid = [row for row in manifest_rows if "error" not in row]
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "start_index": args.start_index,
        "requested_limit": args.limit,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "shuffle_seed": args.shuffle_seed if args.shuffle_buffer_size > 0 else None,
        "rows": len(manifest_rows),
        "valid_rows": len(valid),
        "error_rows": len(manifest_rows) - len(valid),
        "decode_mode_counts": {
            str(mode): sum(1 for row in manifest_rows if row.get("decode_mode") == mode)
            for mode in sorted({row.get("decode_mode") for row in manifest_rows})
        },
        "duration_s_total": sum(float(row.get("duration_s") or 0.0) for row in valid),
        "manifest": str(manifest_path),
        "audio_dir": str(audio_dir),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")


def load_existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(rows, list):
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("error"):
            continue
        input_id = str(row.get("input") or "")
        if input_id and row.get("audio"):
            indexed[input_id] = dict(row)
    return indexed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize HF audio samples to local wav files for SpeechBoundary-JA research.")
    parser.add_argument("--dataset", default="litagin/Galgame_Speech_ASR_16kHz")
    parser.add_argument("--split", default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--shuffle-buffer-size", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=13)
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing wav files in output-dir/audio.")
    parser.add_argument("--no-fallback-decode-false", dest="fallback_decode_false", action="store_false")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "speech-boundary-ja" / "hf-audio"))
    parser.set_defaults(fallback_decode_false=True)
    args = parser.parse_args(argv)
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.shuffle_buffer_size < 0:
        parser.error("--shuffle-buffer-size must be non-negative")
    return args


if __name__ == "__main__":
    materialize(parse_args())
