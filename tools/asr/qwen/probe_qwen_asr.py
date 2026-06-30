#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.model_paths import resolve_model_spec


@dataclass(frozen=True)
class TextDistance:
    reference_chars: int
    hypothesis_chars: int
    distance: int
    cer: float


def normalize_asr_eval_text(text: str) -> str:
    return re.sub(r"[\s　、。！？!?,.・「」『』（）()\[\]【】~〜～…]+", "", text or "")


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_char in enumerate(reference, start=1):
        current = [row_index]
        for col_index, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[col_index] + 1,
                    current[col_index - 1] + 1,
                    previous[col_index - 1] + (0 if ref_char == hyp_char else 1),
                )
            )
        previous = current
    return previous[-1]


def text_distance(reference: str, hypothesis: str) -> TextDistance:
    normalized_ref = normalize_asr_eval_text(reference)
    normalized_hyp = normalize_asr_eval_text(hypothesis)
    distance = levenshtein_distance(normalized_ref, normalized_hyp)
    ref_chars = len(normalized_ref)
    return TextDistance(
        reference_chars=ref_chars,
        hypothesis_chars=len(normalized_hyp),
        distance=distance,
        cer=distance / max(ref_chars, 1),
    )


def load_manifest_rows(paths: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_text in paths:
        payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"manifest must be a JSON list: {path_text}")
        for row in payload:
            if isinstance(row, Mapping):
                rows.append(dict(row))
    return rows


def select_rows(rows: list[dict[str, Any]], *, limit: int | None) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        audio = row.get("audio")
        text = str(row.get("text") or "").strip()
        if audio and text and Path(str(audio)).exists():
            selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def cuda_snapshot() -> dict[str, Any]:
    try:
        import torch

        return {
            "cuda_available": bool(torch.cuda.is_available()),
            "allocated": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
            "reserved": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0,
            "max_allocated": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        }
    except Exception as exc:
        return {"error": str(exc)}


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = select_rows(load_manifest_rows(args.manifest), limit=args.limit)
    if not rows:
        raise ValueError("no rows with existing audio and text were selected")

    os.environ["ASR_MODEL_ID"] = args.model_id
    if args.model_path:
        os.environ["ASR_MODEL_PATH"] = args.model_path
    os.environ["ASR_DTYPE"] = args.dtype
    os.environ["ASR_ATTENTION"] = args.attention
    os.environ["ASR_BATCH_SIZE"] = str(args.batch_size)
    os.environ["ASR_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    os.environ["TRANSCRIPTION_TIMEOUT_S"] = str(args.timeout_s)
    os.environ["ASR_LANGUAGE"] = args.language
    os.environ["ASR_FORCE_LANGUAGE"] = "1"

    model_spec = resolve_model_spec(args.model_path or None, args.model_id, download=not args.no_download)
    from asr.local_backend import LocalAsrBackend

    backend = LocalAsrBackend(args.device)
    events: list[str] = []
    start_load = time.perf_counter()
    backend.load(on_stage=events.append)
    load_s = time.perf_counter() - start_load

    outputs_path = output_dir / "asr_probe_outputs.jsonl"
    distances: list[TextDistance] = []
    durations: list[float] = []
    start_infer = time.perf_counter()
    try:
        with outputs_path.open("w", encoding="utf-8") as handle:
            for batch_start in range(0, len(rows), args.batch_size):
                batch = rows[batch_start : batch_start + args.batch_size]
                batch_paths = [str(row["audio"]) for row in batch]
                batch_contexts = [args.context] * len(batch_paths)
                batch_time = time.perf_counter()
                results = backend.transcribe_texts(batch_paths, contexts=batch_contexts, on_stage=events.append)
                elapsed = time.perf_counter() - batch_time
                for row, result in zip(batch, results, strict=True):
                    predicted = str(result.get("text") or result.get("raw_text") or "")
                    reference = str(row.get("text") or "")
                    distance = text_distance(reference, predicted)
                    distances.append(distance)
                    duration_s = float(row.get("duration_s") or 0.0)
                    durations.append(duration_s)
                    payload = {
                        "audio_id": row.get("audio_id"),
                        "audio": row.get("audio"),
                        "duration_s": duration_s,
                        "reference": reference,
                        "prediction": predicted,
                        "distance": asdict(distance),
                        "elapsed_batch_s": elapsed,
                        "asr_generation": result.get("asr_generation"),
                        "log": result.get("log", []),
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                    print(
                        f"probed {len(distances)}/{len(rows)} audio_id={payload['audio_id']} "
                        f"cer={distance.cer:.4f}",
                        flush=True,
                    )
    finally:
        backend.close()

    total_distance = sum(item.distance for item in distances)
    total_ref_chars = sum(item.reference_chars for item in distances)
    total_duration_s = sum(durations)
    infer_s = time.perf_counter() - start_infer
    summary = {
        "model_id": args.model_id,
        "model_spec": model_spec,
        "device": args.device,
        "dtype": args.dtype,
        "attention": args.attention,
        "batch_size": args.batch_size,
        "rows": len(rows),
        "load_s": load_s,
        "infer_s": infer_s,
        "audio_duration_s": total_duration_s,
        "realtime_factor": infer_s / total_duration_s if total_duration_s > 0 else 0.0,
        "cer": total_distance / max(total_ref_chars, 1),
        "distance": total_distance,
        "reference_chars": total_ref_chars,
        "outputs": str(outputs_path),
        "events": events,
        "cuda": cuda_snapshot(),
    }
    summary_path = output_dir / "asr_probe_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"outputs={outputs_path}")
    print(f"summary={summary_path}")
    print(f"cer={summary['cer']:.4f} rtf={summary['realtime_factor']:.4f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe local Qwen3-ASR direct transcription on manifest audio/text.")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--model-id", default="jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--language", default="Japanese")
    parser.add_argument("--context", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
