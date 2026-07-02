#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PTM_REPO_ID = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    source = np.load(args.features)
    ptm = np.asarray(source["ptm"], dtype=np.float32)
    mfcc = np.asarray(source["mfcc"], dtype=np.float32)
    frame_hop_s = float(source["frame_hop_s"][0])
    total = min(ptm.shape[0], mfcc.shape[0])
    labels = np.zeros(total, dtype=np.int8)
    weights = np.zeros(total, dtype=np.float32)
    omni_rows = [
        row
        for row in _read_jsonl(Path(args.omni_labels))
        if str(row.get("audio_id") or row.get("video_id")) == args.audio_id
    ]
    for target_label in ("definite_drop", "definite_keep"):
        for row in omni_rows:
            if row.get("label") != target_label:
                continue
            start = max(0, int(np.floor(float(row["start"]) / frame_hop_s)))
            end = min(total, int(np.ceil(float(row["end"]) / frame_hop_s)))
            if target_label == "definite_keep":
                pad = int(round(args.keep_padding_s / frame_hop_s))
                start = max(0, start - pad)
                end = min(total, end + pad)
                labels[start:end] = 1
            else:
                labels[start:end] = 0
            weights[start:end] = float(row.get("omni_confidence") or 1.0)

    output = Path(args.output_dir)
    feature_dir = output / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    window_frames = int(round(args.window_s / frame_hop_s))
    real_records: list[dict] = []
    real_manifest: list[dict] = []
    for window_index, start in enumerate(range(0, total, window_frames)):
        end = min(total, start + window_frames)
        active = weights[start:end] > 0.0
        if float(active.mean()) < args.min_coverage:
            continue
        audio_id = f"{args.audio_id}-omni-window-{window_index:04d}"
        feature_path = feature_dir / f"{audio_id}.npz"
        np.savez_compressed(
            feature_path,
            ptm=ptm[start:end, : args.ptm_dim],
            mfcc=mfcc[start:end],
        )
        duration_s = (end - start) * frame_hop_s
        record_index = len(real_records)
        real_records.append(
            {
                "audio_id": audio_id,
                "source": "omni_real_speech_island_hard_negative_v1",
                "duration_s": duration_s,
                "text": "",
                "teacher_segments": {},
                "frame_hop_s": frame_hop_s,
                "speech_frames": labels[start:end].tolist(),
                "label_quality": "supervised",
                "frame_weights": weights[start:end].tolist(),
                "boundary_metadata": {
                    "source_audio_id": args.audio_id,
                    "source_start_s": start * frame_hop_s,
                    "source_end_s": end * frame_hop_s,
                    "omni_label_schema": "pre_asr_omni_label_v1",
                },
            }
        )
        real_manifest.append(
            {
                "audio_id": audio_id,
                "audio_path": "",
                "duration_s": duration_s,
                "feature_coverage_ratio": float(active.mean()),
                "feature_path": str(feature_path),
                "frame_count": end - start,
                "frame_hop_s": frame_hop_s,
                "label_index": record_index,
                "label_quality": "supervised",
                "mfcc_dim": mfcc.shape[1],
                "ptm": PTM_REPO_ID,
                "ptm_dim": args.ptm_dim,
                "source": "omni_real_speech_island_hard_negative_v1",
                "speech_frame_count": int(labels[start:end].sum()),
            }
        )

    base_labels = _read_jsonl(Path(args.base_labels))
    base_manifest = _read_jsonl(Path(args.base_manifest))
    label_offset = len(base_labels)
    combined_labels = base_labels + real_records
    combined_manifest = list(base_manifest)
    for _repeat in range(args.real_repeat):
        for row in real_manifest:
            combined_manifest.append(
                {**row, "label_index": label_offset + int(row["label_index"])}
            )
    labels_path = output / "speech_island_labels.jsonl"
    manifest_path = output / "feature_manifest.jsonl"
    _write_jsonl(labels_path, combined_labels)
    _write_jsonl(manifest_path, combined_manifest)
    summary = {
        "schema": "speech_island_real_omni_dataset_v1",
        "audio_id": args.audio_id,
        "omni_rows": len(omni_rows),
        "real_windows": len(real_records),
        "real_repeat": args.real_repeat,
        "combined_manifest_rows": len(combined_manifest),
        "weighted_coverage": float((weights > 0.0).mean()),
        "speech_ratio_on_labeled_frames": float(
            labels[weights > 0.0].mean()
        ),
        "labels": str(labels_path),
        "feature_manifest": str(manifest_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--omni-labels", required=True)
    parser.add_argument("--audio-id", required=True)
    parser.add_argument("--base-labels", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ptm-dim", type=int, default=128)
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--keep-padding-s", type=float, default=0.08)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--real-repeat", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
