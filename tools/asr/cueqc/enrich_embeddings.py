#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_TEXT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_AUDIO_EMBEDDING_MODEL = "rinna/japanese-hubert-base"


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


def _device_arg(value: str) -> str:
    value = value.strip()
    if value and value != "auto":
        return value
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def _round_vector(values: Iterable[float], digits: int) -> list[float]:
    return [round(float(value), digits) for value in values if math.isfinite(float(value))]


def _batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _candidate_text(row: Mapping[str, Any], *, prefix: str) -> str:
    text = str(row.get("text") or row.get("raw_text") or row.get("text_preview") or "")
    text = " ".join(text.replace("\r", " ").replace("\n", " ").split()).strip()
    return f"{prefix}{text}" if prefix and text else text


def _load_text_embedder(model_name: str, *, device: str):
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=device)

        def encode(texts: list[str], *, batch_size: int) -> list[list[float]]:
            vectors = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vectors.tolist()

        return encode, "sentence_transformers"
    except Exception:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        def encode(texts: list[str], *, batch_size: int) -> list[list[float]]:
            out: list[list[float]] = []
            with torch.inference_mode():
                for batch in _batched(texts, batch_size):
                    encoded = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    hidden = model(**encoded).last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                    pooled = F.normalize(pooled, p=2, dim=1)
                    out.extend(pooled.detach().cpu().tolist())
            return out

        return encode, "transformers_mean_pool"


def enrich_text_embeddings(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    text_prefix: str,
    vector_digits: int,
) -> dict[str, Any]:
    encode, backend = _load_text_embedder(model_name, device=device)
    texts = [_candidate_text(row, prefix=text_prefix) for row in rows]
    vectors = encode(texts, batch_size=batch_size)
    for row, vector in zip(rows, vectors):
        embeddings = dict(row.get("embeddings") or {})
        embeddings["text"] = {
            "model": model_name,
            "backend": backend,
            "dim": len(vector),
            "normalized": True,
            "vector": _round_vector(vector, vector_digits),
        }
        row["embeddings"] = embeddings
    return {
        "enabled": True,
        "model": model_name,
        "backend": backend,
        "device": device,
        "count": len(vectors),
        "dim": len(vectors[0]) if vectors else 0,
    }


def _audio_path(row: Mapping[str, Any]) -> str:
    audio = row.get("audio")
    if isinstance(audio, Mapping):
        value = str(audio.get("path") or "")
        if value:
            return value
    return str(row.get("source_audio_path") or "")


def _read_audio(
    path: str,
    *,
    target_sample_rate: int,
    max_seconds: float,
) -> list[float] | None:
    if not path:
        return None
    audio_path = Path(path)
    if not audio_path.exists() or not audio_path.is_file():
        return None
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    tensor = torch.from_numpy(waveform).mean(dim=1)
    if max_seconds > 0:
        tensor = tensor[: int(max_seconds * sample_rate)]
    if sample_rate != target_sample_rate:
        tensor = AF.resample(tensor, sample_rate, target_sample_rate)
    return tensor.detach().cpu().tolist()


def _load_audio_embedder(model_name: str, *, device: str, sample_rate: int):
    import torch
    import torch.nn.functional as F
    from transformers import AutoFeatureExtractor, AutoModel

    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def encode(waves: list[list[float]], *, batch_size: int) -> list[list[float]]:
        out: list[list[float]] = []
        with torch.inference_mode():
            for batch in _batched(waves, batch_size):
                inputs = extractor(
                    batch,
                    sampling_rate=sample_rate,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                hidden = model(**inputs).last_hidden_state
                if "attention_mask" in inputs:
                    mask = inputs["attention_mask"]
                    feature_lengths = model._get_feat_extract_output_lengths(mask.sum(-1))
                    frame_mask = torch.zeros(
                        hidden.shape[:2],
                        dtype=hidden.dtype,
                        device=hidden.device,
                    )
                    for index, length in enumerate(feature_lengths.tolist()):
                        frame_mask[index, : int(length)] = 1.0
                    pooled = (hidden * frame_mask.unsqueeze(-1)).sum(dim=1) / frame_mask.sum(
                        dim=1,
                    ).clamp(min=1.0).unsqueeze(-1)
                else:
                    pooled = hidden.mean(dim=1)
                pooled = F.normalize(pooled, p=2, dim=1)
                out.extend(pooled.detach().cpu().tolist())
        return out

    return encode, "transformers_hidden_mean_pool"


def enrich_audio_embeddings(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    sample_rate: int,
    max_seconds: float,
    vector_digits: int,
) -> dict[str, Any]:
    encode, backend = _load_audio_embedder(model_name, device=device, sample_rate=sample_rate)
    indexed_waves: list[tuple[int, list[float]]] = []
    missing = 0
    for index, row in enumerate(rows):
        wave = _read_audio(
            _audio_path(row),
            target_sample_rate=sample_rate,
            max_seconds=max_seconds,
        )
        if wave is None or not wave:
            missing += 1
            continue
        indexed_waves.append((index, wave))
    vectors = encode([wave for _index, wave in indexed_waves], batch_size=batch_size)
    for (row_index, _wave), vector in zip(indexed_waves, vectors):
        row = rows[row_index]
        embeddings = dict(row.get("embeddings") or {})
        embeddings["audio"] = {
            "model": model_name,
            "backend": backend,
            "dim": len(vector),
            "sample_rate": sample_rate,
            "max_seconds": max_seconds,
            "normalized": True,
            "vector": _round_vector(vector, vector_digits),
        }
        row["embeddings"] = embeddings
    return {
        "enabled": True,
        "model": model_name,
        "backend": backend,
        "device": device,
        "count": len(vectors),
        "missing_audio": missing,
        "dim": len(vectors[0]) if vectors else 0,
        "sample_rate": sample_rate,
        "max_seconds": max_seconds,
    }


def run(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.input))
    if args.max_items is not None:
        rows = rows[: args.max_items]
    device = _device_arg(args.device)
    meta: dict[str, Any] = {
        "schema": "cueqc_embedding_enrichment_v1",
        "input": args.input,
        "output": args.output,
        "device": device,
        "candidate_count": len(rows),
    }
    if not args.skip_text:
        meta["text"] = enrich_text_embeddings(
            rows,
            model_name=args.text_model,
            device=device,
            batch_size=args.text_batch_size,
            text_prefix=args.text_prefix,
            vector_digits=args.vector_digits,
        )
    else:
        meta["text"] = {"enabled": False}
    if not args.skip_audio:
        meta["audio"] = enrich_audio_embeddings(
            rows,
            model_name=args.audio_model,
            device=device,
            batch_size=args.audio_batch_size,
            sample_rate=args.audio_sample_rate,
            max_seconds=args.audio_max_seconds,
            vector_digits=args.vector_digits,
        )
    else:
        meta["audio"] = {"enabled": False}
    for row in rows:
        row["embedding_meta"] = {
            "schema": "cueqc_candidate_embedding_meta_v1",
            "text_model": "" if args.skip_text else args.text_model,
            "audio_model": "" if args.skip_audio else args.audio_model,
            "device": device,
        }
    count = write_jsonl(Path(args.output), rows)
    meta["written"] = count
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(f"embeddings={args.output}")
    print(f"count={count}")
    print(f"text_dim={(meta.get('text') or {}).get('dim', 0)} audio_dim={(meta.get('audio') or {}).get('dim', 0)}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add high-dimensional text/audio embeddings to CueQC candidates.")
    parser.add_argument("--input", required=True, help="cueqc_candidates.jsonl")
    parser.add_argument("--output", required=True, help="cueqc_candidates.embedded.jsonl")
    parser.add_argument("--summary", default="", help="optional embedding summary JSON")
    parser.add_argument("--device", default=os.getenv("CUEQC_EMBED_DEVICE", "auto"))
    parser.add_argument("--text-model", default=os.getenv("CUEQC_TEXT_EMBED_MODEL", DEFAULT_TEXT_EMBEDDING_MODEL))
    parser.add_argument("--audio-model", default=os.getenv("CUEQC_AUDIO_EMBED_MODEL", DEFAULT_AUDIO_EMBEDDING_MODEL))
    parser.add_argument("--text-prefix", default=os.getenv("CUEQC_TEXT_EMBED_PREFIX", ""))
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--audio-batch-size", type=int, default=4)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--audio-max-seconds", type=float, default=8.0)
    parser.add_argument("--vector-digits", type=int, default=6)
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args(argv)
    if args.text_batch_size <= 0:
        parser.error("--text-batch-size must be positive")
    if args.audio_batch_size <= 0:
        parser.error("--audio-batch-size must be positive")
    if args.audio_sample_rate <= 0:
        parser.error("--audio-sample-rate must be positive")
    if args.audio_max_seconds < 0:
        parser.error("--audio-max-seconds must be non-negative")
    if args.vector_digits < 0:
        parser.error("--vector-digits must be non-negative")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
