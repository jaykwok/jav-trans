#!/usr/bin/env python3
"""Extract CueQC Mamba v3-Fusion features from ASR internals.

Reads either labeled ``cueqc_train.jsonl`` rows or unlabeled
``cueqc_candidate_v1`` rows, resolves each sample's audio from the baseline wav
tree, and runs the teacher-forced
``AsrInternalsCapturer`` to capture, per candidate:

* ``asr_frames``  — Qwen3-ASR encoder hidden states [T, D_asr]
* ``token_trace`` — per-token features             [L, K_tok]
* ``decoder_stats`` — chunk-level aggregates         [K_dec]
* ``structured`` — metadata vector                   [S]

Raw (variable-length) features are stored — NOT padded, NOT normalized.
Padding happens in the training collate_fn; normalization is computed on the
train split only inside the training script (to avoid test leakage).

Label convention (v3): ``drop = 0``, ``keep = 1`` (inverted from v2).
Unlabeled candidate rows are encoded as ``-1`` and are intended for prediction /
pseudo-label export, not direct training.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

import numpy as np

# Torchcodec DLL stub (see cueqc_model.py for rationale).
if "torchcodec" not in sys.modules:
    for _mod_name in ("torchcodec", "torchcodec.decoders"):
        _stub = types.ModuleType(_mod_name)
        _stub.__spec__ = ModuleSpec(_mod_name, loader=None)
        _stub.__path__ = []
        sys.modules[_mod_name] = _stub

# Blank HF_ENDPOINT breaks huggingface_hub; pop it so the default is used.
if not (os.environ.get("HF_ENDPOINT") or "").strip():
    os.environ.pop("HF_ENDPOINT", None)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.asr_internals import AsrInternalsCapturer  # noqa: E402
from asr.cueqc_features import (  # noqa: E402
    DECODER_FEATURE_NAMES,
    STRUCTURED_FEATURE_NAMES,
    TOKEN_FEATURE_NAMES,
    K_DEC,
    K_TOK,
    S_DIM,
    build_decoder_stats,
    build_structured_features,
    build_token_trace,
)

SAMPLE_RATE = 16000
# v3 label convention: 0 = drop, 1 = keep (inverted from v2's keep=0/drop=1).
DISPLAY_LABEL_MAP = {"drop": 0, "keep": 1}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _label_from_row(row: dict[str, Any]) -> int:
    targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    label = targets.get("display_label")
    if label is not None:
        try:
            parsed = int(label)
        except (TypeError, ValueError):
            parsed = -1
        return parsed if parsed in {0, 1} else -1
    display = targets.get("display_decision") or targets.get("display_hint")
    return DISPLAY_LABEL_MAP.get(str(display or "").strip(), -1)


def _video_id_from_sample(sample_id: str) -> str:
    # "cueqc-867HTTM-0045-chunk01431" -> "867HTTM-0045"
    m = re.match(r"cueqc-(.+)-chunk\d+$", sample_id)
    return m.group(1) if m else sample_id


def _find_wav(audio_root: Path, video_id: str) -> Path | None:
    pattern = f"{video_id}*.wav"
    for jobs_dir in audio_root.rglob(f"{video_id}_b5"):
        for wav in jobs_dir.rglob(pattern):
            return wav
    for wav in audio_root.rglob(pattern):
        return wav
    return None


def _chunk_window(row: dict[str, Any]) -> tuple[float, float]:
    start = row.get("start")
    end = row.get("end")
    if start is not None and end is not None:
        return float(start), float(end)
    align = (row.get("features") or {}).get("alignment_diagnostics") or {}
    end_v = align.get("fallback_window_end_s")
    return 0.0, float(end_v) if end_v is not None else 1.0


def _resolve_audio_id(sample_id: str, video_id: str) -> str:
    """audio_id for group-splitting = the video id (one movie = one group)."""
    return video_id


def _row_to_candidate(row: dict[str, Any], start_s: float, end_s: float) -> dict[str, Any]:
    """Flatten a training row into the candidate shape ``build_structured_features`` expects."""
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    candidate = {
        "start": start_s,
        "end": end_s,
        "duration_s": end_s - start_s,
        "text_features": features.get("text_features") if isinstance(features.get("text_features"), dict) else {},
        "adjacency": features.get("adjacency") if isinstance(features.get("adjacency"), dict) else {},
        "boundary": features.get("boundary") if isinstance(features.get("boundary"), dict) else {},
        "asr_signals": features.get("asr_signals") if isinstance(features.get("asr_signals"), dict) else {},
    }
    return candidate


def extract(args: argparse.Namespace) -> int:
    import torch

    source_path = Path(args.train or args.input)
    source_mode = "train" if args.train else "candidate"
    rows = _read_jsonl(source_path)
    audio_root = Path(args.audio_root)
    n_rows = len(rows) if args.max_samples is None else min(len(rows), args.max_samples)
    print(
        f"rows={len(rows)} processing={n_rows} source_mode={source_mode} "
        f"audio_root={audio_root} device={args.device}"
    )

    from asr.local_backend import (
        active_qwen_asr_model_id,
        active_qwen_asr_model_path,
        resolve_model_spec,
    )

    model_spec = args.model_spec or resolve_model_spec(
        active_qwen_asr_model_path() or None,
        active_qwen_asr_model_id() or "jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame",
    )
    print(f"model_spec={model_spec}")
    capturer = AsrInternalsCapturer(model_spec=model_spec, device=args.device)

    wav_cache: dict[str, Path | None] = {}

    samples: list[dict[str, Any]] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []
    skipped = 0

    for idx in range(n_rows):
        row = rows[idx]
        sample_id = str(row.get("sample_id") or "")
        video_id = _video_id_from_sample(sample_id)
        if video_id not in wav_cache:
            wav_cache[video_id] = _find_wav(audio_root, video_id)
        wav_path = wav_cache[video_id]
        if wav_path is None:
            skipped += 1
            continue

        text = str(row.get("text") or row.get("raw_text") or "")
        start_s, end_s = _chunk_window(row)

        try:
            internals = capturer.extract(wav_path, text, start_s=start_s, end_s=end_s)
        except Exception as exc:  # noqa: BLE001 - offline tool keeps going on bad chunks
            print(f"WARN extract {sample_id}: {exc}")
            skipped += 1
            continue

        asr_frames = internals["asr_frames"]
        if asr_frames.shape[0] == 0:
            # v3 data constraint: drop samples with no asr frames.
            skipped += 1
            continue

        token_trace = build_token_trace(
            token_logprobs=internals["token_logprobs"],
            token_entropies=internals["token_entropies"],
            token_margins=internals["token_top1_top2_margins"],
            decoded_tokens=internals["decoded_tokens"],
            has_timestamps=bool(internals.get("has_timestamps", False)),
        )
        decoder_stats = build_decoder_stats(
            token_trace=token_trace,
            text=text,
            duration_s=max(end_s - start_s, 1e-6),
            has_timestamps=bool(internals.get("has_timestamps", False)),
        )
        # asr_confidence from teacher-forced logprobs (real value, no longer None).
        asr_confidence = float(np.mean(internals["token_logprobs"])) if internals["token_logprobs"].size else None
        candidate = _row_to_candidate(row, start_s, end_s)
        structured = build_structured_features(
            candidate,
            n_tokens=internals["token_ids"].shape[0],
            asr_confidence=asr_confidence,
        )

        targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
        display = str(targets.get("display_decision") or targets.get("display_hint") or "")
        label = _label_from_row(row)

        samples.append({
            "sample_id": sample_id,
            "audio_id": _resolve_audio_id(sample_id, video_id),
            "asr_frames": torch.from_numpy(np.ascontiguousarray(asr_frames, dtype=np.float32)),
            "token_trace": torch.from_numpy(np.ascontiguousarray(token_trace, dtype=np.float32)),
            "decoder_stats": torch.from_numpy(np.ascontiguousarray(decoder_stats, dtype=np.float32)),
            "structured": torch.from_numpy(np.ascontiguousarray(structured, dtype=np.float32)),
        })
        labels.append(label)
        meta.append({
            "sample_id": sample_id,
            "cluster_id": row.get("cluster_id", ""),
            "audio_id": _resolve_audio_id(sample_id, video_id),
            "start": start_s,
            "end": end_s,
            "video_id": video_id,
            "chunk_index": row.get("chunk_index"),
            "text": text[:80],
            "display_hint": display,
            "source_schema": row.get("schema", ""),
        })

        if (idx + 1) % 25 == 0:
            print(f"  processed {idx + 1}/{n_rows} (kept={len(samples)} skipped={skipped})")

    capturer.close()

    n = len(samples)
    asr_dim = int(samples[0]["asr_frames"].shape[1]) if n else 0
    print(f"extracted={n} skipped={skipped} asr_dim={asr_dim}")
    print(f"labels_keep={labels.count(1)} labels_drop={labels.count(0)} labels_unlabeled={labels.count(-1)}")

    bundle = {
        "schema": "cueqc_mamba_v3_fusion_features",
        "version": 3,
        "samples": samples,
        "labels": torch.tensor(labels, dtype=torch.long),
        "meta": meta,
        "feature_config": {
            "source_path": str(source_path),
            "source_mode": source_mode,
            "asr_dim": asr_dim,
            "token_dim": K_TOK,
            "decoder_dim": K_DEC,
            "structured_dim": S_DIM,
            "feature_source": "asr_internal",
            "uses_bge": False,
            "text_embedding": "none",
            "token_feature_names": list(TOKEN_FEATURE_NAMES),
            "decoder_feature_names": list(DECODER_FEATURE_NAMES),
            "structured_feature_names": list(STRUCTURED_FEATURE_NAMES),
        },
        "label_config": {"drop": 0, "keep": 1},
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    print(f"saved={out} n={n} asr_dim={asr_dim} token_dim={K_TOK} decoder_dim={K_DEC} structured_dim={S_DIM}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract CueQC v3-Fusion features from ASR internals.")
    p.add_argument("--train", default="", help="labeled cueqc_train.jsonl")
    p.add_argument("--input", default="", help="unlabeled cueqc_candidate_v1 JSONL for prediction/self-training")
    p.add_argument("--audio-root", required=True, help="baseline root containing jobs/<VIDEO>_b5/audio/*.wav")
    p.add_argument("--output", required=True, help="output .pt path")
    p.add_argument("--model-spec", default="", help="Qwen3-ASR repo id (default env ASR_MODEL_SPEC)")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args(argv)
    if bool(args.train) == bool(args.input):
        p.error("exactly one of --train or --input is required")
    if args.input and args.max_samples is not None and args.max_samples <= 0:
        p.error("--max-samples must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    return extract(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
