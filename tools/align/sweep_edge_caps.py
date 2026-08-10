#!/usr/bin/env python3
"""Size `ONSET_BACKOFF_MAX_S` and `CODA_EXTEND_MAX_S` against measured speech.

**Both caps were originally sized the same wrong way.** `speech_extent` walks a
segment's first and last character outward through frames the head itself calls
blank, capped at these two constants. On 2026-07-31 they were set to 0.30 / 0.40
to cover measured "insets" of 230.8 ms and 371.7 ms, both computed against the
core's PLACEMENT window. But a core is a clean galgame clip placed whole: the
window is where the clip sits, not where its speech starts and stops, so the
clip's own edge silence was counted as inset the head had to walk back. Measured
(`measure_core_leading_silence.py`), that silence is a median 90.0 ms at the head
and **274.8 ms at the tail** - roughly three quarters of the tail figure was
never alignment error at all.

**One GPU pass, both edges, no cross product.** The two walks are independent
loops on opposite ends of the same posterior, so a start under cap A and an end
under cap B need no joint evaluation. Each core contributes one row per onset cap
and one per coda cap, all from identical tensors - so differences between caps
carry no sampling noise whatsoever.

**The two edges need different metrics, because their costs differ.**

At the onset one direction is nearly free: starting early only shows the line
sooner, and `_normalize_subtitle_timeline` still enforces non-overlap. So the
onset arm looks for a knee - the largest cap before the audibly-early tail
(>200 ms, the blind-audit threshold) starts climbing. That sweep chose 0.20 on
2026-08-10.

At the coda neither direction is free. Too short and the cue's `acoustic_end` is
early, so timeline polish clamps display to `acoustic_end + 0.5 s` and the line
can vanish on the last syllable. Too long and it lingers into silence, eats the
gap to the next cue, and inflates `previous_word_end_s` so the layout DP believes
the piece holds more content than it does. A knee needs a free axis; with none,
the coda arm is bracketed instead:

  * **floor** from `share_before` - ends that sit inside the speech. Per the
    detector's bias (see `edge_silence_s`), "still ends early" can only be
    understated, so this direction is conservative and trustworthy.
  * **ceiling** from `share_past_core_end`, which needs no detector at all:
    `core_end_s - core_start_s` equals the source clip's duration exactly, so a
    predicted end past `core_end_s` has provably left the clip and walked into
    the surrounding JAV drop audio. That is a hard containment error - and it
    doubles as a domain-transfer reading, since the walk is supposed to
    self-limit on the first non-blank frame of that audio.

**Limits.** Both edge detectors understate speech, so every "reaches outside the
speech" share is an upper bound; the bias is a property of the clip and identical
across arms, which is what keeps the arm-to-arm comparison readable. Composites
wrap each core in dropped audio, so the walk's self-limit almost never fires -
this corpus is the worst case for over-walking, though a real film measured on
2026-08-10 was not far off it (72.2% of line starts still had >=0.70 s of
residual silence after both walks). And the corpus is clean galgame, not the real
domain.
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
    CODA_EXTEND_MAX_S,
    ENCODER_FRAME_S,
    ONSET_BACKOFF_MAX_S,
    AlignmentVocab,
    align_text,
    build_head,
    normalize_text,
    speech_extent,
)
from audio.loading import load_audio_16k_mono  # noqa: E402
from utils.gpu_safety import apply_vram_safety_cap  # noqa: E402

SAMPLE_RATE = 16000
SCHEMA = "asr_edge_cap_sweep_v2"
AUDIBLE_EARLY_S = 0.200


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve(path_text: str) -> Path:
    path = Path(str(path_text).replace("\\", "/"))
    return path if path.is_absolute() else PROJECT_ROOT / path


def quantized_cap_s(cap_s: float, *, upsample: int) -> float:
    """The cap the walk can actually reach, in seconds.

    `speech_extent` converts the cap to a whole number of frames by truncation,
    so 0.30 s and 0.29 s are the same cap and reporting the requested value alone
    would invent distinctions the code cannot make.
    """
    frame_s = ENCODER_FRAME_S / float(upsample)
    return max(0, int(max(0.0, cap_s) / frame_s)) * frame_s


def summarize(errors: list[float]) -> dict:
    """Distribution of boundary error in ms.

    Sign means the same thing at both edges - negative = the prediction sits
    before the measured boundary, positive = after - but which one is expensive
    flips. At the onset, negative is the cheap direction (the line shows early);
    at the coda, negative is the costly one (the line ends inside the speech).
    """
    if not errors:
        return {"count": 0}
    array = np.asarray(errors, dtype=np.float64)

    def pct(fraction: float) -> float:
        return round(float(np.percentile(array, fraction * 100.0)) * 1000.0, 1)

    return {
        "count": len(errors),
        "median_ms": round(statistics.median(errors) * 1000.0, 1),
        "mean_ms": round(statistics.fmean(errors) * 1000.0, 1),
        "p05_ms": pct(0.05),
        "p25_ms": pct(0.25),
        "p75_ms": pct(0.75),
        "p95_ms": pct(0.95),
        "share_before": round(float((array < 0.0).mean()), 4),
        "share_before_over_200ms": round(float((array < -AUDIBLE_EARLY_S).mean()), 4),
        "share_after": round(float((array > 0.0).mean()), 4),
        "share_after_over_100ms": round(float((array > 0.100).mean()), 4),
    }


def load_head(checkpoint: Path, model_path: str):
    import torch

    apply_vram_safety_cap(0.95)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
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

    from asr.encoder_features import EncoderFeatureConfig, Qwen3AsrEncoder

    extractor = Qwen3AsrEncoder(
        EncoderFeatureConfig(model_path=model_path or "", device="cuda")
    )

    def posteriors(audios: list[np.ndarray]):
        features = extractor.encode_batch(audios, sample_rate=SAMPLE_RATE)
        outputs = []
        for feature in features:
            tensor = torch.from_numpy(feature).unsqueeze(0).to(device)
            with torch.inference_mode():
                outputs.append(head(tensor)[0].float().cpu())
        return outputs

    return posteriors, vocab, upsample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--composites", required=True)
    parser.add_argument(
        "--edge-silence",
        required=True,
        help="`.details.jsonl` from measure_core_leading_silence.py (schema v2, "
        "carrying both edges), from the same manifest/seed/limit so rows join "
        "without an index",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--onset-caps", default="0.00,0.05,0.10,0.15,0.20,0.30")
    parser.add_argument("--coda-caps", default="0.00,0.10,0.15,0.20,0.25,0.30,0.40")
    args = parser.parse_args()

    onset_caps = [float(p) for p in str(args.onset_caps).split(",") if p.strip()]
    coda_caps = [float(p) for p in str(args.coda_caps).split(",") if p.strip()]
    if not onset_caps or not coda_caps:
        raise SystemExit("both edges need at least one cap")

    edges: dict[tuple[str, int], tuple[float, float, float]] = {}
    for row in read_jsonl(resolve(args.edge_silence)):
        leading = row.get("leading_silence_ms")
        trailing = row.get("trailing_silence_ms")
        if leading is None or trailing is None or row.get("core_end_s") is None:
            continue
        edges[(str(row.get("sample_id")), int(row["core_index"]))] = (
            float(leading) / 1000.0,
            float(trailing) / 1000.0,
            float(row["core_end_s"]),
        )
    if not edges:
        raise SystemExit(
            "edge-silence file carries no usable cores - schema v1 has no tail; "
            "re-run measure_core_leading_silence.py"
        )

    posteriors, vocab, upsample = load_head(resolve(args.checkpoint), args.model_path)

    rows = read_jsonl(resolve(args.composites))
    rng = np.random.default_rng(args.seed)
    if args.limit and args.limit < len(rows):
        picked = rng.choice(len(rows), size=args.limit, replace=False)
        rows = [rows[i] for i in sorted(picked)]

    start_errors: dict[float, list[float]] = {c: [] for c in onset_caps}
    end_errors: dict[float, list[float]] = {c: [] for c in coda_caps}
    past_core_end: dict[float, list[float]] = {c: [] for c in coda_caps}
    raw_start: list[float] = []
    raw_end: list[float] = []
    cores_evaluated = 0
    cores_unjoined = 0
    failures = 0

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        loaded: list[tuple[dict, np.ndarray]] = []
        for row in batch:
            try:
                audio, rate = load_audio_16k_mono(str(resolve(row["audio"])))
            except Exception:  # noqa: BLE001
                failures += 1
                continue
            if rate != SAMPLE_RATE:
                failures += 1
                continue
            loaded.append((row, np.asarray(audio, dtype=np.float32)))
        if not loaded:
            continue

        for (row, _audio), log_probs in zip(
            loaded, posteriors([audio for _, audio in loaded])
        ):
            cores = sorted(
                row.get("core_spans") or (), key=lambda c: float(c["start_s"])
            )
            texts = [normalize_text(str(core.get("text") or "")) for core in cores]
            if not cores or not all(texts):
                continue
            try:
                spans = align_text(log_probs, "".join(texts), vocab, upsample=upsample)
            except (ValueError, RuntimeError):
                failures += 1
                continue

            cursor = 0
            for index, core in enumerate(cores):
                text = texts[index]
                core_spans = spans[cursor : cursor + len(text)]
                cursor += len(text)
                if not core_spans:
                    continue
                joined = edges.get((str(row.get("sample_id")), index))
                if joined is None:
                    cores_unjoined += 1
                    continue
                leading, trailing, core_end = joined
                cores_evaluated += 1

                true_start = float(core["start_s"]) + leading
                true_end = core_end - trailing
                raw_start.append(core_spans[0].start_s - true_start)
                raw_end.append(core_spans[-1].end_s - true_end)

                # Independent walks on opposite ends, so each edge is swept
                # against its own cap with the other held at production. No
                # cross product is needed and none is implied.
                for cap in onset_caps:
                    edged = speech_extent(
                        log_probs,
                        core_spans,
                        upsample=upsample,
                        backoff_max_s=cap,
                        extend_max_s=CODA_EXTEND_MAX_S,
                    )
                    value = core_spans[0].start_s if edged is None else edged[0]
                    start_errors[cap].append(value - true_start)
                for cap in coda_caps:
                    edged = speech_extent(
                        log_probs,
                        core_spans,
                        upsample=upsample,
                        backoff_max_s=ONSET_BACKOFF_MAX_S,
                        extend_max_s=cap,
                    )
                    value = core_spans[-1].end_s if edged is None else edged[1]
                    end_errors[cap].append(value - true_end)
                    past_core_end[cap].append(value - core_end)

    def coda_arm(cap: float) -> dict:
        overshoot = np.asarray(past_core_end[cap], dtype=np.float64)
        outside = overshoot[overshoot > 0.0]
        return {
            "extend_max_s": cap,
            "reachable_cap_ms": round(
                quantized_cap_s(cap, upsample=upsample) * 1000.0, 1
            ),
            # Exact and detector-free: past core_end_s the walk has left the clip.
            "share_past_core_end": round(float((overshoot > 0.0).mean()), 4)
            if overshoot.size
            else None,
            "overshoot_ms_p90": round(float(np.percentile(outside, 90)) * 1000.0, 1)
            if outside.size
            else 0.0,
            **summarize(end_errors[cap]),
        }

    summary = {
        "schema": SCHEMA,
        "checkpoint": str(args.checkpoint),
        "composites": str(args.composites),
        "edge_silence": str(args.edge_silence),
        "cores_evaluated": cores_evaluated,
        "cores_unjoined": cores_unjoined,
        "failures": failures,
        "upsample": upsample,
        "frame_ms": round(ENCODER_FRAME_S / float(upsample) * 1000.0, 2),
        "production": {
            "onset_backoff_max_s": ONSET_BACKOFF_MAX_S,
            "coda_extend_max_s": CODA_EXTEND_MAX_S,
        },
        "uncorrected_start": summarize(raw_start),
        "uncorrected_end": summarize(raw_end),
        "onset_caps": [
            {
                "backoff_max_s": cap,
                "reachable_cap_ms": round(
                    quantized_cap_s(cap, upsample=upsample) * 1000.0, 1
                ),
                **summarize(start_errors[cap]),
            }
            for cap in onset_caps
        ],
        "coda_caps": [coda_arm(cap) for cap in coda_caps],
    }

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
