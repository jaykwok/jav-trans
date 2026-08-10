#!/usr/bin/env python3
"""Realign frozen real-JAV lines with two CTC heads for a blind boundary A/B.

The transcript and acoustic window are held fixed.  This deliberately avoids
running each head's blank segmentation independently: different regions would
mix segmentation and ASR differences into what is meant to be a boundary-only
human comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.alignment import AlignmentHead, normalize_text  # noqa: E402
from asr.encoder_features import qwen3_asr_audio_output_lengths  # noqa: E402
from audio.loading import load_audio_16k_mono  # noqa: E402
from tools.audits.audit_nav import audit_generated_at  # noqa: E402
from utils.gpu_safety import apply_vram_safety_cap  # noqa: E402


SCHEMA = "ctc_alignment_ab_jav_prediction_v1"
SUMMARY_SCHEMA = "ctc_alignment_ab_jav_prediction_summary_v1"
SAMPLE_RATE = 16000
CONTEXT_S = 1.0


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_context(
    *, line_start_s: float, line_duration_s: float, audio_duration_s: float
) -> tuple[float, float]:
    start = max(0.0, float(line_start_s) - CONTEXT_S)
    end = min(
        float(audio_duration_s),
        float(line_start_s) + float(line_duration_s) + CONTEXT_S,
    )
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from asr.backends.qwen import active_qwen_asr_model_id, active_qwen_asr_model_path
    from utils.model_paths import resolve_model_spec

    answer_rows = _rows(Path(args.answers))
    manifest_by_id = {str(row["row_id"]): row for row in _rows(Path(args.manifest))}
    frozen: list[dict[str, Any]] = []
    for row in answer_rows:
        row_id = str(row.get("row_id") or "")
        source = manifest_by_id.get(row_id)
        text = str(row.get("text") or "")
        if not source or not normalize_text(text):
            continue
        frozen.append({**row, "audio": str(source["audio"])})
    if args.limit and len(frozen) > args.limit:
        rng = np.random.default_rng(args.seed)
        positions = sorted(rng.choice(len(frozen), size=args.limit, replace=False))
        frozen = [frozen[int(position)] for position in positions]
    if not frozen:
        raise SystemExit("no frozen JAV lines are eligible")

    apply_vram_safety_cap(0.95)
    model_spec = resolve_model_spec(
        active_qwen_asr_model_path() or None, active_qwen_asr_model_id(), download=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_spec)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_spec, dtype=dtype, device_map=str(device)
    )
    model.eval()
    head_a = AlignmentHead.load(args.checkpoint_a, device=device)
    head_b = AlignmentHead.load(args.checkpoint_b, device=device)

    def encode(clip: np.ndarray) -> np.ndarray:
        inputs = processor.apply_transcription_request(audio=[clip], language=None)
        moved = {
            key: (
                value.to(device=device, dtype=dtype)
                if key == "input_features"
                else value.to(device=device)
            )
            if torch.is_tensor(value)
            else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            features = model.get_audio_features(
                input_features=moved["input_features"],
                input_features_mask=moved["input_features_mask"],
            ).pooler_output
        frames = int(
            qwen3_asr_audio_output_lengths(moved["input_features_mask"].sum(dim=1))[0]
        )
        return features[:frames].detach().float().cpu().numpy()

    audio_cache: dict[str, np.ndarray] = {}
    output_rows: list[dict[str, Any]] = []
    skipped = {"load": 0, "rate": 0, "unalignable": 0, "invalid_extent": 0}
    for index, row in enumerate(frozen):
        audio_path = Path(str(row["audio"]))
        if not audio_path.is_absolute():
            audio_path = PROJECT_ROOT / audio_path
        key = str(audio_path.resolve())
        if key not in audio_cache:
            try:
                audio, rate = load_audio_16k_mono(key)
            except Exception:  # noqa: BLE001
                skipped["load"] += 1
                continue
            if rate != SAMPLE_RATE:
                skipped["rate"] += 1
                continue
            audio_cache[key] = np.asarray(audio, dtype=np.float32)
        audio = audio_cache[key]
        total_s = len(audio) / SAMPLE_RATE
        clip_start, clip_end = fixed_context(
            line_start_s=float(row["line_start_s"]),
            line_duration_s=float(row["line_duration_s"]),
            audio_duration_s=total_s,
        )
        clip = np.ascontiguousarray(
            audio[int(clip_start * SAMPLE_RATE) : int(clip_end * SAMPLE_RATE)]
        )
        features = encode(clip)
        aligned_a = head_a.align_extent(features, str(row["text"]))
        aligned_b = head_b.align_extent(features, str(row["text"]))
        if aligned_a is None or aligned_b is None:
            skipped["unalignable"] += 1
            continue
        start_a, end_a = clip_start + aligned_a[1], clip_start + aligned_a[2]
        start_b, end_b = clip_start + aligned_b[1], clip_start + aligned_b[2]
        if not (
            0.0 <= start_a < end_a <= total_s + 1e-6
            and 0.0 <= start_b < end_b <= total_s + 1e-6
        ):
            skipped["invalid_extent"] += 1
            continue
        output_rows.append(
            {
                "schema": SCHEMA,
                "row_id": str(row["row_id"]),
                "line_id": str(row.get("line_id") or row["row_id"]),
                "source_id": str(row.get("source_id") or ""),
                "source_partition": str(row.get("source_partition") or ""),
                "audio": key,
                "audio_duration_s": round(total_s, 6),
                "text": str(row["text"]),
                "frozen_clip_start_s": round(clip_start, 6),
                "frozen_clip_end_s": round(clip_end, 6),
                "model_a_start_s": round(start_a, 6),
                "model_a_end_s": round(end_a, 6),
                "model_b_start_s": round(start_b, 6),
                "model_b_end_s": round(end_b, 6),
                "start_delta_ms": round(abs(start_a - start_b) * 1000.0, 3),
                "end_delta_ms": round(abs(end_a - end_b) * 1000.0, 3),
            }
        )
        print(f"[{index + 1}/{len(frozen)}] paired={len(output_rows)}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    checkpoint_a = Path(args.checkpoint_a).resolve()
    checkpoint_b = Path(args.checkpoint_b).resolve()
    summary = {
        "schema": SUMMARY_SCHEMA,
        "generated_at": audit_generated_at(),
        "requested_rows": len(frozen),
        "paired_rows": len(output_rows),
        "videos": len({row["source_id"] for row in output_rows}),
        "skipped": skipped,
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_a_sha256": _sha256(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "checkpoint_b_sha256": _sha256(checkpoint_b),
        "fixed_context_s": CONTEXT_S,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
