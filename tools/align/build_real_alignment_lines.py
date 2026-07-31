#!/usr/bin/env python3
"""Produce the subtitle lines the new chain would emit on real JAV audio.

This runs the architecture end to end on real windows - encoder over
everything, blank runs to pick where words are, decode only those regions,
forced-align the text, and hand the spans to the same
`build_aligned_word_timestamps` the production path uses. The output is
therefore what a viewer would actually see, timing included, which is what an
audit of alignment accuracy has to be run against.

Two things this deliberately does not do. It does not use the composites, whose
speech is clean galgame; every line here is real JAV audio. And it does not use
any of the retired boundary chain to choose regions - the blank runs are the
only segmentation, so a bad region is the new design's own fault.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.alignment import AlignmentHead, blank_runs, normalize_text  # noqa: E402
from asr.subtitle_timing import build_aligned_word_timestamps  # noqa: E402
from audio.loading import load_audio_16k_mono  # noqa: E402
from utils.gpu_safety import apply_vram_safety_cap  # noqa: E402
from asr.encoder_features import qwen3_asr_audio_output_lengths  # noqa: E402

SAMPLE_RATE = 16000
FEATURE_CHUNK_S = 30.0
# Below this the transcript is a repeated cycle, not speech (Phase 0: 0.107 for
# a runaway decode against 0.475 for real speech).
RUNAWAY_UNIQUE_RATIO = 0.25
LINES_SCHEMA = "real_alignment_line_v1"


def speech_regions(
    runs: list[tuple[float, float]],
    total_s: float,
    *,
    min_speech_s: float,
    merge_gap_s: float,
) -> list[tuple[float, float]]:
    regions: list[tuple[float, float]] = []
    cursor = 0.0
    for begin, end in runs:
        if begin > cursor:
            regions.append((cursor, begin))
        cursor = max(cursor, end)
    if cursor < total_s:
        regions.append((cursor, total_s))
    merged: list[list[float]] = []
    for begin, end in regions:
        if merged and begin - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = end
        else:
            merged.append([begin, end])
    return [(b, e) for b, e in merged if e - b >= min_speech_s]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--windows", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--min-blank-s", type=float, default=0.6)
    parser.add_argument("--min-speech-s", type=float, default=1.0)
    parser.add_argument("--merge-gap-s", type=float, default=0.4)
    parser.add_argument("--pad-s", type=float, default=0.15)
    parser.add_argument("--max-region-s", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from asr.backends.qwen import active_qwen_asr_model_id, active_qwen_asr_model_path
    from utils.model_paths import resolve_model_spec

    apply_vram_safety_cap(0.95)
    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    rng = np.random.default_rng(args.seed)
    by_video: dict[str, list[dict]] = {}
    for row in rows:
        by_video.setdefault(str(row["source_id"]), []).append(row)
    videos = sorted(by_video)
    rng.shuffle(videos)
    picked = [by_video[video][0] for video in videos[: args.windows]]

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
    head = AlignmentHead.load(args.checkpoint, device=device)

    def _move(clip: np.ndarray) -> dict:
        inputs = processor.apply_transcription_request(audio=[clip], language=None)
        return {
            key: (
                value.to(device=device, dtype=dtype)
                if key == "input_features"
                else value.to(device=device)
            )
            if torch.is_tensor(value)
            else value
            for key, value in inputs.items()
        }

    def encode(clip: np.ndarray) -> np.ndarray:
        moved = _move(clip)
        with torch.inference_mode():
            features = model.get_audio_features(
                input_features=moved["input_features"],
                input_features_mask=moved["input_features_mask"],
            ).pooler_output
        frames = int(
            qwen3_asr_audio_output_lengths(moved["input_features_mask"].sum(dim=1))[0]
        )
        return features[:frames].detach().float().cpu().numpy()

    def transcribe(clip: np.ndarray) -> str:
        moved = _move(clip)
        with torch.inference_mode():
            generated = model.generate(
                **moved, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        suffix = generated[:, moved["input_ids"].shape[1] :]
        decoded = processor.batch_decode(
            suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # Without `parse_output` the decode still carries the prompt template
        # ("language Japanese<asr_text>"), ~25 characters that were never
        # spoken; the aligner would place them in the audio and both the score
        # and every timestamp after them would be wrong.
        parsed = processor.parse_output(decoded)
        if isinstance(parsed, dict):
            parsed = [parsed]
        return str(parsed[0].get("transcription") or "")

    lines: list[dict] = []
    stats = {"regions": 0, "runaway": 0, "empty": 0, "unalignable": 0}
    for order, row in enumerate(picked):
        audio, rate = load_audio_16k_mono(str(row["audio"]))
        if rate != SAMPLE_RATE:
            continue
        audio = np.asarray(audio, dtype=np.float32)
        total_s = len(audio) / SAMPLE_RATE

        runs: list[tuple[float, float]] = []
        width = int(FEATURE_CHUNK_S * SAMPLE_RATE)
        for offset in range(0, len(audio), width):
            clip = np.ascontiguousarray(audio[offset : offset + width])
            if len(clip) < SAMPLE_RATE // 2:
                continue
            log_probs = head.log_probs(encode(clip))
            base = offset / SAMPLE_RATE
            runs.extend(
                (base + b, base + e)
                for b, e in blank_runs(
                    log_probs, upsample=head.upsample, min_seconds=args.min_blank_s
                )
            )

        for index, (raw_begin, raw_end) in enumerate(
            speech_regions(
                runs,
                total_s,
                min_speech_s=args.min_speech_s,
                merge_gap_s=args.merge_gap_s,
            )
        ):
            begin = max(0.0, raw_begin - args.pad_s)
            end = min(total_s, min(raw_end + args.pad_s, begin + args.max_region_s))
            clip = np.ascontiguousarray(
                audio[int(begin * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
            )
            if len(clip) < SAMPLE_RATE // 4:
                continue
            stats["regions"] += 1
            text = transcribe(clip)
            normalized = normalize_text(text)
            if not normalized:
                stats["empty"] += 1
                continue
            if len(set(normalized)) / len(normalized) < RUNAWAY_UNIQUE_RATIO:
                stats["runaway"] += 1
                continue
            spans = head.align(encode(clip), text)
            if not spans:
                stats["unalignable"] += 1
                continue
            words, mode, meta = build_aligned_word_timestamps(
                text, spans, 0.0, end - begin
            )
            if mode != "ctc_forced_alignment" or not words:
                stats["unalignable"] += 1
                continue
            # Region-relative times become window-relative, which is what the
            # audit will cut against.
            line_start = begin + min(word["start"] for word in words)
            line_end = begin + max(word["end"] for word in words)
            lines.append(
                {
                    "schema": LINES_SCHEMA,
                    "line_id": f"{row['sample_id']}#r{index:03d}",
                    "sample_id": row["sample_id"],
                    "source_id": row["source_id"],
                    "source_partition": row["source_partition"],
                    "audio": str(row["audio"]),
                    "window_duration_s": round(total_s, 3),
                    "region_start_s": round(begin, 3),
                    "region_end_s": round(end, 3),
                    "line_start_s": round(line_start, 3),
                    "line_end_s": round(line_end, 3),
                    "line_duration_s": round(line_end - line_start, 3),
                    "characters": len(normalized),
                    "chars_per_s": round(
                        len(normalized) / max(1e-6, line_end - line_start), 3
                    ),
                    "alignment_score": meta.get("alignment_score"),
                    "boundary_trimmed": meta.get("boundary_trimmed"),
                    "text": text,
                }
            )
        print(
            f"[{order + 1}/{len(picked)}] {row['sample_id'][:34]:34s} "
            f"lines={len(lines)}",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")

    scores = [l["alignment_score"] for l in lines if l["alignment_score"] is not None]
    summary = {
        "schema": "real_alignment_lines_summary_v1",
        "windows": len(picked),
        "videos": len({l["source_id"] for l in lines}),
        "lines": len(lines),
        **stats,
        "runaway_rate": round(stats["runaway"] / stats["regions"], 4)
        if stats["regions"]
        else None,
        "median_line_duration_s": round(
            statistics.median([l["line_duration_s"] for l in lines]), 3
        )
        if lines
        else None,
        "median_chars_per_s": round(
            statistics.median([l["chars_per_s"] for l in lines]), 3
        )
        if lines
        else None,
        "median_alignment_score": round(statistics.median(scores), 4)
        if scores
        else None,
        "trimmed_lines": sum(1 for l in lines if l.get("boundary_trimmed")),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
