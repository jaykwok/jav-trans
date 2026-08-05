#!/usr/bin/env python3
"""Sweep the Viterbi blank bias against composite geometry.

**What this is for.** The 07-31 geometry pass measured the head's spans sitting
INSIDE the true core at both edges - median 230.8 ms at the head, 371.7 ms at the
tail. That is the signature of CTC's peaky posterior, not of a lag: forced
alignment returns the frames the model is most confident about, which for the
first and last character is the middle of the sound rather than its edge.

`speech_extent` already corrects the two OUTER edges by walking through blank.
This sweeps the other lever, which acts on every character: subtracting a
constant from the blank column before the search makes staying in blank slightly
more expensive, so each character widens onto the frames it actually occupies.
It costs nothing at runtime and needs no retraining.

**What makes the result readable.** Two things that were not available before:

  * `--leading-silence` joins the per-core measurement from
    `measure_core_leading_silence.py`, which removes the term that made absolute
    onset error unidentifiable. Without it this tool reports `core_start_offset`
    as before and says so; the number is then a bound, not an error.
  * `context_shift` is reported per bias too, because it is the one metric that
    must NOT improve: it cancels the inset by construction, so a bias that moves
    it is moving something other than the peak, and that is a warning rather
    than a win.

**The failure mode being watched for.** A large bias does not merely widen
characters, it makes the path leave blank early and enter the neighbouring
sound. `overshoot_share` counts cores whose corrected start lands BEFORE the
true onset by more than one output frame - the point past which this stops
buying accuracy and starts stealing audio from the previous line.
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
    AlignmentVocab,
    align_text,
    build_head,
    frame_to_seconds,
    normalize_text,
    speech_extent,
)
from audio.loading import load_audio_16k_mono  # noqa: E402
from utils.gpu_safety import apply_vram_safety_cap  # noqa: E402
from asr.encoder_features import EncoderFeatureConfig, Qwen3AsrEncoder  # noqa: E402

SAMPLE_RATE = 16000
SCHEMA = "asr_alignment_blank_bias_sweep_v1"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve(path_text: str) -> Path:
    path = Path(str(path_text).replace("\\", "/"))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64) * 1000.0
    return {
        "count": len(values),
        "median_ms": round(float(np.median(array)), 1),
        "mean_ms": round(float(array.mean()), 1),
        "p05_ms": round(float(np.percentile(array, 5)), 1),
        "p90_ms": round(float(np.percentile(array, 90)), 1),
        "p99_ms": round(float(np.percentile(array, 99)), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--composites", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--leading-silence",
        default="",
        help="details.jsonl from measure_core_leading_silence.py. Without it "
        "the onset numbers are bounds that include unknown clip padding.",
    )
    parser.add_argument(
        "--bias",
        action="append",
        type=float,
        default=None,
        help="repeatable. 0.0 must be included to have a reference arm.",
    )
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    biases = sorted(set(args.bias or [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]))
    if 0.0 not in biases:
        raise SystemExit("--bias must include 0.0 as the reference arm")

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

    silence: dict[tuple[str, int], float] = {}
    if args.leading_silence:
        for row in _read_jsonl(Path(args.leading_silence)):
            value = row.get("leading_silence_ms")
            if value is None:
                continue
            silence[(str(row["sample_id"]), int(row["core_index"]))] = (
                float(value) / 1000.0
            )

    rows = _read_jsonl(Path(args.composites))
    rng = np.random.default_rng(args.seed)
    if args.limit and args.limit < len(rows):
        picked = rng.choice(len(rows), size=args.limit, replace=False)
        rows = [rows[i] for i in sorted(picked)]

    extractor = Qwen3AsrEncoder(
        EncoderFeatureConfig(model_path=args.model_path or "", device="cuda")
    )
    frame_s = frame_to_seconds(1, upsample=upsample)

    # Per bias, per core. Paired by construction: every arm sees exactly the same
    # cores and the same posteriors, so the only thing that varies is the bias.
    onset_error: dict[float, list[float]] = {b: [] for b in biases}
    start_offset: dict[float, list[float]] = {b: [] for b in biases}
    end_offset: dict[float, list[float]] = {b: [] for b in biases}
    edged_start: dict[float, list[float]] = {b: [] for b in biases}
    edged_end: dict[float, list[float]] = {b: [] for b in biases}
    shifts: dict[float, list[float]] = {b: [] for b in biases}
    overshoot: dict[float, int] = {b: 0 for b in biases}
    scored: dict[float, list[float]] = {b: [] for b in biases}
    cores_seen = 0
    failures = 0

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        loaded: list[tuple[dict, np.ndarray]] = []
        for row in batch:
            try:
                audio, rate = load_audio_16k_mono(str(_resolve(row["audio"])))
            except Exception:  # noqa: BLE001
                failures += 1
                continue
            if rate != SAMPLE_RATE:
                failures += 1
                continue
            loaded.append((row, np.asarray(audio, dtype=np.float32)))
        if not loaded:
            continue

        features = extractor.encode_batch(
            [audio for _, audio in loaded], sample_rate=SAMPLE_RATE
        )
        for (row, _audio), feature in zip(loaded, features):
            cores = sorted(
                row.get("core_spans") or (), key=lambda c: float(c["start_s"])
            )
            texts = [normalize_text(str(core.get("text") or "")) for core in cores]
            if not cores or not all(texts):
                continue
            with torch.inference_mode():
                log_probs = head(
                    torch.from_numpy(feature).unsqueeze(0).to(device)
                )[0].float().cpu()
            joined = "".join(texts)

            per_bias_spans = {}
            for bias in biases:
                try:
                    per_bias_spans[bias] = align_text(
                        log_probs, joined, vocab, upsample=upsample, blank_bias=bias
                    )
                except (ValueError, RuntimeError):
                    per_bias_spans = {}
                    failures += 1
                    break
            if not per_bias_spans:
                continue

            reference = per_bias_spans[0.0]
            cursor = 0
            for index, core in enumerate(cores):
                text = texts[index]
                window = slice(cursor, cursor + len(text))
                cursor += len(text)
                if not reference[window]:
                    continue
                cores_seen += 1
                core_start = float(core["start_s"])
                core_end = float(core["end_s"])
                pad = silence.get((str(row.get("sample_id")), index))
                truth = core_start + pad if pad is not None else None
                base = [s.start_s for s in reference[window]]

                for bias in biases:
                    spans = per_bias_spans[bias][window]
                    if not spans:
                        continue
                    start_offset[bias].append(spans[0].start_s - core_start)
                    end_offset[bias].append(core_end - spans[-1].end_s)
                    edged = speech_extent(log_probs, spans, upsample=upsample)
                    edged_start_s = spans[0].start_s if edged is None else edged[0]
                    edged_end_s = spans[-1].end_s if edged is None else edged[1]
                    edged_start[bias].append(edged_start_s - core_start)
                    edged_end[bias].append(core_end - edged_end_s)
                    scored[bias].append(
                        sum(s.score for s in spans) / len(spans)
                    )
                    # Against the reference arm's own alignment, so this reads
                    # "how far did the bias move each character", not accuracy.
                    shifts[bias].append(
                        statistics.median(
                            [abs(s.start_s - b) for s, b in zip(spans, base)]
                        )
                    )
                    if truth is not None:
                        onset_error[bias].append(spans[0].start_s - truth)
                        if edged_start_s < truth - frame_s:
                            overshoot[bias] += 1

    arms = []
    for bias in biases:
        arm = {
            "blank_bias": bias,
            "cores": len(start_offset[bias]),
            # One-sided: positive means the span sits inside the true core.
            "core_start_offset": _describe(start_offset[bias]),
            "core_end_offset": _describe(end_offset[bias]),
            "core_start_offset_edged": _describe(edged_start[bias]),
            "core_end_offset_edged": _describe(edged_end[bias]),
            # Movement relative to bias 0, not an accuracy measure.
            "movement_vs_reference": _describe(shifts[bias]),
            "mean_char_score": (
                round(statistics.fmean(scored[bias]), 4) if scored[bias] else None
            ),
        }
        if silence:
            arm["onset_error_vs_measured_speech"] = _describe(onset_error[bias])
            arm["overshoot_cores"] = overshoot[bias]
            arm["overshoot_share"] = (
                round(overshoot[bias] / len(onset_error[bias]), 4)
                if onset_error[bias]
                else None
            )
        arms.append(arm)

    result = {
        "schema": SCHEMA,
        "checkpoint": str(args.checkpoint),
        "upsample": upsample,
        "frame_resolution_ms": round(frame_s * 1000.0, 2),
        "cores_evaluated": cores_seen,
        "failures": failures,
        "leading_silence_joined": bool(silence),
        # Stated in the artifact because it changes what the onset numbers mean:
        # without the join they include each clip's unknown leading silence and
        # are bounds, not errors.
        "onset_is_identifiable": bool(silence),
        "arms": arms,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
