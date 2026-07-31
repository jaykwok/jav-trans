#!/usr/bin/env python3
"""Measure the alignment head against geometry it never saw during training.

The composite corpus was built by placing known galgame clips at known sample
offsets inside real JAV audio, so every composite carries exact geometric truth
for free. The cores used here are excluded from the feature cache
(`build_alignment_features.py --holdout-composites`), so nothing below is
measured on a clip the head has memorised.

**What this can and cannot establish.** The manifest records where each clip was
*placed*, not where speech begins inside it - a source clip with 300 ms of
leading silence puts its first character 300 ms after `start_s`, and no error
has occurred. So the absolute onset error is not identifiable from geometry
alone, and this tool does not pretend otherwise. What geometry does give,
without any human labelling:

  * **containment** - every character of a core must land inside that core's
    window. Padding can only push characters inward, so a character outside is
    unambiguously an alignment error. Necessary, not sufficient.
  * **context shift** - align each core standalone, then inside the composite,
    and subtract the known placement offset. The two should agree exactly; what
    they differ by is alignment error under added noise and neighbouring speech,
    with the padding term cancelling out because it is identical in both. This
    is the closest thing to a true error measure available for free, and it is
    the number to read first.
  * **blank behaviour** - the negative unit span and the inter-unit gaps are
    both filled with real JAV audio drawn from `definite_drop` chunks, which a
    human audited as containing no words. All of it should read as blank. This
    is the gate reading of the same tensor, and note what it tests: a head
    trained only on clean galgame speech, asked about real JAV audio it has
    never seen. A weak result here is a domain-transfer finding, not an
    alignment finding, and the two are reported separately below.

The plan's kill gate (median <= 150 ms, p90 <= 400 ms) is defined on a human-
checked set. This tool decides whether that human time is worth spending.
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

from asr.alignment import (  # noqa: E402
    ALIGNMENT_MODEL_SCHEMA,
    BLANK_INDEX,
    ENCODER_FPS,
    AlignmentVocab,
    align_text,
    build_head,
    speech_extent,
    frame_to_seconds,
    normalize_text,
)
from audio.loading import load_audio_16k_mono  # noqa: E402
from utils.gpu_safety import apply_vram_safety_cap  # noqa: E402
from asr.encoder_features import EncoderFeatureConfig, Qwen3AsrEncoder  # noqa: E402

SAMPLE_RATE = 16000


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve(path_text: str) -> Path:
    path = Path(str(path_text).replace("\\", "/"))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction * 100.0))


def _describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "median_ms": round(statistics.median(values) * 1000.0, 1),
        "p90_ms": round(_percentile(values, 0.90) * 1000.0, 1),
        "p99_ms": round(_percentile(values, 0.99) * 1000.0, 1),
        "mean_ms": round(statistics.fmean(values) * 1000.0, 1),
        "max_ms": round(max(values) * 1000.0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--composites", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    import torch

    apply_vram_safety_cap(0.95)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if str(payload.get("schema")) != ALIGNMENT_MODEL_SCHEMA:
        raise SystemExit(f"not an alignment checkpoint: {payload.get('schema')!r}")
    vocab = AlignmentVocab.from_payload(payload["vocab"])
    upsample = int(payload["upsample"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = build_head(
        vocab_size=vocab.size,
        input_dim=int(payload.get("input_dim", 2048)),
        hidden_dim=int(payload["hidden_dim"]),
        upsample=upsample,
        blocks=int(payload["blocks"]),
        dropout=0.0,
    )
    head.load_state_dict(payload["state_dict"])
    head.to(device).eval()

    rows = _read_jsonl(Path(args.composites))
    rng = np.random.default_rng(args.seed)
    if args.limit and args.limit < len(rows):
        picked = rng.choice(len(rows), size=args.limit, replace=False)
        rows = [rows[i] for i in sorted(picked)]

    extractor = Qwen3AsrEncoder(
        EncoderFeatureConfig(model_path=args.model_path or "", device="cuda")
    )

    def posteriors(audios: list[np.ndarray]):
        features = extractor.encode_batch(audios, sample_rate=SAMPLE_RATE)
        outputs = []
        for feature in features:
            tensor = torch.from_numpy(feature).unsqueeze(0).to(device)
            with torch.inference_mode():
                outputs.append(head(tensor)[0].float().cpu())
        return outputs

    contained = 0
    characters = 0
    before_core = 0
    after_core = 0
    shifts: list[float] = []
    char_shifts: list[float] = []
    first_char_shifts: list[float] = []
    worst_char_shifts: list[float] = []
    start_offsets: list[float] = []
    end_offsets: list[float] = []
    edged_start_offsets: list[float] = []
    edged_end_offsets: list[float] = []
    negative_blank_rates: list[float] = []
    gap_blank_rates: list[float] = []
    core_blank_rates: list[float] = []
    failures: list[dict] = []
    details: list[dict] = []
    cores_evaluated = 0
    composites_evaluated = 0

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        loaded: list[tuple[dict, np.ndarray]] = []
        for row in batch:
            try:
                audio, rate = load_audio_16k_mono(str(_resolve(row["audio"])))
            except Exception as error:  # noqa: BLE001
                failures.append({"sample_id": row.get("sample_id"), "error": str(error)})
                continue
            if rate != SAMPLE_RATE:
                failures.append({"sample_id": row.get("sample_id"), "error": "rate"})
                continue
            loaded.append((row, np.asarray(audio, dtype=np.float32)))
        if not loaded:
            continue

        composite_posteriors = posteriors([audio for _, audio in loaded])

        for (row, audio), log_probs in zip(loaded, composite_posteriors):
            cores = sorted(
                row.get("core_spans") or (), key=lambda c: float(c["start_s"])
            )
            texts = [normalize_text(str(core.get("text") or "")) for core in cores]
            if not cores or not all(texts):
                continue
            joined = "".join(texts)
            try:
                spans = align_text(log_probs, joined, vocab, upsample=upsample)
            except (ValueError, RuntimeError) as error:
                failures.append(
                    {"sample_id": row.get("sample_id"), "error": str(error)}
                )
                continue
            composites_evaluated += 1

            # Standalone alignment of the same clips: the padding term is
            # identical in both, so the difference isolates alignment error.
            standalone_audios: list[np.ndarray] = []
            standalone_cores: list[int] = []
            for index, core in enumerate(cores):
                source = _resolve(str(core.get("source_audio") or ""))
                if not source.exists():
                    continue
                try:
                    clip, rate = load_audio_16k_mono(str(source))
                except Exception:  # noqa: BLE001
                    continue
                if rate != SAMPLE_RATE:
                    continue
                standalone_audios.append(np.asarray(clip, dtype=np.float32))
                standalone_cores.append(index)
            standalone_posteriors = (
                posteriors(standalone_audios) if standalone_audios else []
            )
            standalone_starts: dict[int, list[float]] = {}
            for core_index, clip_log_probs in zip(
                standalone_cores, standalone_posteriors
            ):
                try:
                    clip_spans = align_text(
                        clip_log_probs, texts[core_index], vocab, upsample=upsample
                    )
                except (ValueError, RuntimeError):
                    continue
                standalone_starts[core_index] = [s.start_s for s in clip_spans]

            cursor = 0
            for index, core in enumerate(cores):
                text = texts[index]
                core_spans = spans[cursor : cursor + len(text)]
                cursor += len(text)
                if not core_spans:
                    continue
                cores_evaluated += 1
                core_start = float(core["start_s"])
                core_end = float(core["end_s"])
                for span in core_spans:
                    characters += 1
                    if span.start_s < core_start:
                        before_core += 1
                    elif span.end_s > core_end:
                        after_core += 1
                    else:
                        contained += 1
                start_offsets.append(core_spans[0].start_s - core_start)
                end_offsets.append(core_end - core_spans[-1].end_s)
                # Same two numbers after walking each edge outward through the
                # head's own blank frames. Both offsets are one-sided (positive
                # = predicted extent sits inside the true core), so a drop here
                # is the correction recovering real speech - while a swing to
                # NEGATIVE would mean it overshot into neighbouring audio, which
                # is the failure this has to be watched for.
                edged = speech_extent(log_probs, core_spans, upsample=upsample)
                edged_start = core_spans[0].start_s if edged is None else edged[0]
                edged_end = core_spans[-1].end_s if edged is None else edged[1]
                edged_start_offsets.append(edged_start - core_start)
                edged_end_offsets.append(core_end - edged_end)
                # Compare every character, not just the first. A single boundary
                # character pulled into neighbouring audio would otherwise
                # condemn a core whose remaining twenty characters are exact -
                # and which reading is right is the difference between "1% of
                # cores are unusable" and "one character at the seam moves".
                shift = None
                reference = standalone_starts.get(index)
                if reference and len(reference) == len(core_spans):
                    per_char = [
                        abs((span.start_s - core_start) - clip_start)
                        for span, clip_start in zip(core_spans, reference)
                    ]
                    shift = statistics.median(per_char)
                    shifts.append(shift)
                    first_char_shifts.append(per_char[0])
                    worst_char_shifts.append(max(per_char))
                    char_shifts.extend(per_char)
                details.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "core_index": index,
                        "core_count": len(cores),
                        "core_duration_s": round(core_end - core_start, 4),
                        "characters": len(text),
                        "chars_per_s": round(len(text) / max(1e-6, core_end - core_start), 3),
                        "shift_ms": round(shift * 1000.0, 1) if shift is not None else None,
                        "first_char_shift_ms": round(first_char_shifts[-1] * 1000.0, 1)
                        if shift is not None
                        else None,
                        "worst_char_shift_ms": round(worst_char_shifts[-1] * 1000.0, 1)
                        if shift is not None
                        else None,
                        "start_offset_ms": round(
                            (core_spans[0].start_s - core_start) * 1000.0, 1
                        ),
                        "end_offset_ms": round(
                            (core_end - core_spans[-1].end_s) * 1000.0, 1
                        ),
                        "start_offset_edged_ms": round(
                            edged_start_offsets[-1] * 1000.0, 1
                        ),
                        "end_offset_edged_ms": round(edged_end_offsets[-1] * 1000.0, 1),
                        "outside_characters": sum(
                            1
                            for s in core_spans
                            if s.start_s < core_start or s.end_s > core_end
                        ),
                        "mean_score": round(
                            sum(s.score for s in core_spans) / len(core_spans), 4
                        ),
                    }
                )

            predicted = log_probs.argmax(dim=-1).cpu().numpy()
            total_frames = int(predicted.shape[0])

            def blank_rate(begin_s: float, finish_s: float) -> float | None:
                first = int(round(begin_s * upsample * ENCODER_FPS))
                last = int(round(finish_s * upsample * ENCODER_FPS))
                first = max(0, min(total_frames, first))
                last = max(0, min(total_frames, last))
                if last - first < 1:
                    return None
                window = predicted[first:last]
                return float((window == BLANK_INDEX).mean())

            negative = row.get("negative_unit_span") or {}
            if negative:
                rate = blank_rate(
                    float(negative.get("start_s", 0.0)),
                    float(negative.get("end_s", 0.0)),
                )
                if rate is not None:
                    negative_blank_rates.append(rate)
            # A dict of sample offsets for the two flanks, not a list of spans.
            gaps = row.get("inter_unit_gaps") or {}
            for side in ("left", "right"):
                begin = gaps.get(f"{side}_start_sample")
                finish = gaps.get(f"{side}_end_sample")
                if begin is None or finish is None:
                    continue
                rate = blank_rate(
                    float(begin) / SAMPLE_RATE, float(finish) / SAMPLE_RATE
                )
                if rate is not None:
                    gap_blank_rates.append(rate)
            for core in cores:
                rate = blank_rate(float(core["start_s"]), float(core["end_s"]))
                if rate is not None:
                    core_blank_rates.append(rate)

    frame_ms = frame_to_seconds(1, upsample=upsample) * 1000.0
    result = {
        "schema": "asr_alignment_geometry_eval_v1",
        "checkpoint": str(args.checkpoint),
        "upsample": upsample,
        "frame_resolution_ms": round(frame_ms, 2),
        "composites_evaluated": composites_evaluated,
        "cores_evaluated": cores_evaluated,
        "characters": characters,
        "containment": {
            "rate": round(contained / characters, 5) if characters else 0.0,
            "before_core_start": before_core,
            "after_core_end": after_core,
        },
        # The headline number: alignment error under noise and context, with the
        # unknown clip padding cancelled out. Per core, over all its characters.
        "context_shift": _describe(shifts),
        # Every character pooled, which is the unit subtitle timing consumes.
        "context_shift_per_character": _describe(char_shifts),
        # The seam characters specifically, where a neighbour can pull a symbol
        # across the boundary without disturbing the rest of the core.
        "context_shift_first_character": _describe(first_char_shifts),
        "context_shift_worst_character": _describe(worst_char_shifts),
        # One-sided. A negative value is an error; a positive one is padding,
        # alignment lag, or both, and cannot be separated without human checks.
        "core_start_offset": _describe([v for v in start_offsets]),
        "core_end_offset": _describe([v for v in end_offsets]),
        "core_start_offset_edged": _describe([v for v in edged_start_offsets]),
        "core_end_offset_edged": _describe([v for v in edged_end_offsets]),
        "core_start_offset_edged_negative_fraction": round(
            sum(1 for v in edged_start_offsets if v < -1e-9) / len(edged_start_offsets),
            5,
        )
        if edged_start_offsets
        else 0.0,
        "core_start_offset_negative_fraction": round(
            sum(1 for v in start_offsets if v < -1e-9) / len(start_offsets), 5
        )
        if start_offsets
        else 0.0,
        "blank_rate": {
            "negative_unit_span_real_jav": round(
                statistics.fmean(negative_blank_rates), 4
            )
            if negative_blank_rates
            else None,
            "inter_unit_gaps_silence": round(statistics.fmean(gap_blank_rates), 4)
            if gap_blank_rates
            else None,
            "core_spans_speech": round(statistics.fmean(core_blank_rates), 4)
            if core_blank_rates
            else None,
        },
        "failures": len(failures),
        "failure_sample": failures[:5],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Per-core rows, so a bad tail can be diagnosed rather than only counted.
    with output.with_suffix(".details.jsonl").open("w", encoding="utf-8") as handle:
        for record in details:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
