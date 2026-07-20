#!/usr/bin/env python3
"""Build zero-overlap fixed-context bases for an independent Scorer v10 smoke."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from audio.loading import load_audio_16k_mono

SAMPLE_RATE = 16000


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build(*, cores: Path, negatives: Path, output_dir: Path, base_count: int) -> dict:
    core_rows = _rows(cores)
    negative_rows = _rows(negatives)
    if base_count <= 0 or len(core_rows) < base_count * 2:
        raise ValueError("independent smoke has insufficient unused cores")
    required = {"breathing", "music", "noise", "non_speech"}
    available = {str(row.get("eval_type")) for row in negative_rows}
    if not required.issubset(available):
        raise ValueError(f"independent smoke lacks partial types: {sorted(required - available)}")
    # Prefer long pairs so the smoke covers >8 s and includes >20 s when inventory permits.
    core_rows.sort(key=lambda row: (float(row.get("duration_s") or 0.0), str(row["audio_id"])), reverse=True)
    selected = core_rows[: base_count * 2]
    audio_dir = output_dir / "base_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    bases = []
    used = set()
    for index in range(base_count):
        pair = selected[index * 2 : index * 2 + 2]
        chunks = []
        for row in pair:
            audio, rate = load_audio_16k_mono(str(row["audio"]))
            if rate != SAMPLE_RATE:
                raise ValueError("independent core is not 16 kHz")
            chunks.append(np.asarray(audio, dtype=np.float32))
            used.add(str(row["audio_id"]))
        gap = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
        clean = np.concatenate((chunks[0], gap, chunks[1]))
        first_end = len(chunks[0])
        second_start = first_end + len(gap)
        source_id = f"scorer-independent-eval-base-{index:03d}"
        path = audio_dir / f"{source_id}.wav"
        sf.write(path, clean, SAMPLE_RATE, subtype="PCM_16")
        bases.append({
            "schema": "speech_scorer_v10_independent_eval_base_v1",
            "source_id": source_id,
            "audio": str(path),
            "partition": "eval",
            "row_role": "speech",
            "core_ids": [str(row["audio_id"]) for row in pair],
            "sample_count": len(clean),
            "duration_s": len(clean) / SAMPLE_RATE,
            "additive_overlay": None,
            "canonical_spans": [
                {"start_sample": 0, "end_sample": first_end, "label": "speech", "core_id": str(pair[0]["audio_id"])},
                {"start_sample": first_end, "end_sample": second_start, "label": "background", "background_id": "synthetic_silence_gap"},
                {"start_sample": second_start, "end_sample": len(clean), "label": "speech", "core_id": str(pair[1]["audio_id"])},
            ],
        })
    manifest = output_dir / "base_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bases), encoding="utf-8")
    negative_manifest = output_dir / "negative_manifest.jsonl"
    negative_manifest.write_text("".join(json.dumps({**row, "source_partition": "eval", "background_type": row["eval_type"]}, ensure_ascii=False) + "\n" for row in negative_rows), encoding="utf-8")
    summary = {
        "schema": "speech_scorer_v10_independent_eval_smoke_build_v1",
        "diagnostic_only": True,
        "base_count": len(bases),
        "unique_core_count": len(used),
        "max_core_use_count": 1,
        "duration_min_s": min(row["duration_s"] for row in bases),
        "duration_max_s": max(row["duration_s"] for row in bases),
        "partial_overlay_types": sorted(required),
        "missing_formal_types": ["kissing", "moaning"],
        "base_manifest": str(manifest),
        "negative_manifest": str(negative_manifest),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", required=True)
    parser.add_argument("--negatives", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-count", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(build(cores=Path(args.cores), negatives=Path(args.negatives), output_dir=Path(args.output_dir), base_count=args.base_count), ensure_ascii=False))
