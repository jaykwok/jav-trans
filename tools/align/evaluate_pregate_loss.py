#!/usr/bin/env python3
"""What the blank-run pre-gate would throw away, measured against human labels.

The gate's mistakes are not symmetric. Audio it skips is never transcribed, so
speech lost here is lost for good; audio it keeps in error costs decode time and
is filtered downstream. A single accuracy number would average those two
together and hide exactly the one that matters, so this tool reports them apart:

  * **irreversible loss** - seconds labelled `speech` that fall outside every
    kept region.

**Read that first number as a loose upper bound, not as loss.** The relabel
corpus answers "is there a word in this span", so a span is `speech` if it
contains one - and its spans are long (median 7.47 s, p90 17.1 s on the test
partition). A gate that correctly decodes the two seconds of talking inside a
seventeen-second span is scored here as having lost fifteen. The number is still
worth computing, because it is monotone in the gate's aggressiveness and a
config that looks bad here cannot look good under tighter truth; but a config
cannot be *cleared* by it. Measuring loss in the only unconfounded currency -
decoding what the gate threw away and seeing whether words come out - is what
`tools/align/measure_pregate_dropped_audio.py` does.
  * **decode fraction** - seconds kept over seconds of audio. This is the saving
    being bought, and it is only worth reading once the loss is acceptable.
  * **wasted decode** - kept seconds labelled `non_vocal` or
    `non_semantic_vocal`. Reversible, reported for completeness, never traded
    against the first number.

`unsure` spans are reported on their own and counted in neither direction. They
are the spans a teacher could not decide, and folding them into either side
would let an unresolved labelling question masquerade as a gate result.

The ground truth is the relabelled corpus (`drop_span_words_v1`), which is
sparse: labelled spans do not cover the whole window. Every rate below is
therefore over labelled seconds, not over wall-clock seconds, and windows are
drawn from a held-out partition so the gate is not read on audio the alignment
head was fitted near.

One encoder pass serves every configuration in the sweep. The gate is pure
geometry over the same per-frame posteriors, so comparing settings costs
nothing extra and the choice can be made from measurement rather than assumed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.alignment import (  # noqa: E402
    BLANK_INDEX,
    ENCODER_FPS,
    AlignmentHead,
    blank_runs,
)
from asr.pregate import (  # noqa: E402
    PREGATE_SCHEMA,
    PreGateConfig,
    covered_seconds,
    duration,
    speech_regions,
)
from audio.loading import load_audio_16k_mono  # noqa: E402
from utils.gpu_safety import apply_vram_safety_cap  # noqa: E402
from asr.encoder_features import qwen3_asr_audio_output_lengths  # noqa: E402

RESULT_SCHEMA = "pregate_loss_eval_v1"
SAMPLE_RATE = 16000
FEATURE_CHUNK_S = 30.0
SPEECH_LABEL = "speech"
UNSURE_LABEL = "unsure"
WASTE_LABELS = ("non_vocal", "non_semantic_vocal")

# The sweep. `default` is what `PreGateConfig` ships; the rest move one axis at a
# time so a change in the loss can be attributed to a parameter rather than to a
# combination. `keep_everything` is the control - it must show zero loss, and if
# it does not the harness is wrong rather than the gate.
SWEEP: dict[str, dict[str, float]] = {
    "keep_everything": {"min_blank_s": 1e6},
    "default": {},
    "blank_0.4": {"min_blank_s": 0.4},
    "blank_0.8": {"min_blank_s": 0.8},
    "blank_1.2": {"min_blank_s": 1.2},
    "pad_0.0": {"pad_s": 0.0},
    "pad_0.3": {"pad_s": 0.3},
    "min_speech_0.0": {"min_speech_s": 0.0},
    "min_speech_2.0": {"min_speech_s": 2.0},
    "merge_0.0": {"merge_gap_s": 0.0},
    "merge_1.0": {"merge_gap_s": 1.0},
}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _spans_by_label(row: dict) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for span in row.get("spans") or ():
        label = str(span.get("type") or "")
        try:
            begin, end = float(span["start_s"]), float(span["end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > begin:
            grouped[label].append((begin, end))
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="")
    parser.add_argument("--partition", default="test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from asr.backends.qwen import active_qwen_asr_model_id, active_qwen_asr_model_path
    from utils.model_paths import resolve_model_spec

    apply_vram_safety_cap(0.95)

    rows = _read_jsonl(Path(args.dataset))
    if args.split:
        partitions = {
            str(entry.get("example_id")): str(entry.get("partition") or "")
            for entry in _read_jsonl(Path(args.split))
        }
        rows = [
            row
            for row in rows
            if partitions.get(str(row.get("example_id"))) == args.partition
        ]
    if not rows:
        raise SystemExit("no rows in the requested partition")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows))
    if args.limit > 0:
        order = order[: args.limit]
    picked = [rows[int(index)] for index in order]

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

    def posteriors(clip: np.ndarray):
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
        return head.log_probs(features[:frames].detach().float().cpu().numpy())

    configs = {
        name: PreGateConfig(**overrides) for name, overrides in SWEEP.items()
    }
    totals: dict[str, dict[str, float]] = {
        name: defaultdict(float) for name in configs
    }
    labelled_totals: dict[str, float] = defaultdict(float)
    blank_frames: dict[str, int] = defaultdict(int)
    label_frames: dict[str, int] = defaultdict(int)
    audio_s = 0.0
    windows = 0
    failures: list[dict] = []

    for row in picked:
        path = str(row.get("audio") or "")
        try:
            audio, rate = load_audio_16k_mono(path)
        except Exception as error:  # noqa: BLE001
            failures.append({"example_id": row.get("example_id"), "error": str(error)})
            continue
        if rate != SAMPLE_RATE:
            failures.append({"example_id": row.get("example_id"), "error": "rate"})
            continue
        clip = np.asarray(audio, dtype=np.float32)
        total_s = len(clip) / SAMPLE_RATE
        if total_s <= 0.0:
            continue

        # Concatenate posteriors before deriving runs, rather than deriving runs
        # per chunk and shifting them. A pause that straddles a chunk seam would
        # otherwise arrive as two shorter runs and could fail `min_blank_s`,
        # which would invent speech regions exactly at the seams.
        pieces = []
        width = int(FEATURE_CHUNK_S * SAMPLE_RATE)
        for offset in range(0, len(clip), width):
            piece = np.ascontiguousarray(clip[offset : offset + width])
            if len(piece) < SAMPLE_RATE // 2:
                continue
            pieces.append(posteriors(piece))
        if not pieces:
            continue
        log_probs = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]

        grouped = _spans_by_label(row)
        for label, spans in grouped.items():
            labelled_totals[label] += duration(spans)
        audio_s += total_s
        windows += 1

        # Raw blank rate per label, before any of the gate's geometry touches
        # it. If speech reads as blank at the frame level then no arrangement of
        # `min_blank_s` and padding can save the gate, and the finding is about
        # domain transfer of the head rather than about these parameters.
        predicted = log_probs.argmax(dim=-1).detach().cpu().numpy()
        frame_s = 1.0 / (ENCODER_FPS * head.upsample)
        for label, spans in grouped.items():
            for begin, end in spans:
                first = max(0, min(len(predicted), int(round(begin / frame_s))))
                last = max(first, min(len(predicted), int(round(end / frame_s))))
                if last <= first:
                    continue
                window_frames = predicted[first:last]
                blank_frames[label] += int((window_frames == BLANK_INDEX).sum())
                label_frames[label] += int(last - first)

        for name, config in configs.items():
            runs = blank_runs(
                log_probs, upsample=head.upsample, min_seconds=config.min_blank_s
            )
            regions = speech_regions(runs, total_s, config)
            bucket = totals[name]
            bucket["decoded_s"] += duration(regions)
            for label, spans in grouped.items():
                bucket[f"{label}_kept_s"] += covered_seconds(spans, regions)

    report: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "pregate_schema": PREGATE_SCHEMA,
        "checkpoint": args.checkpoint,
        "partition": args.partition if args.split else "all",
        "windows": windows,
        "audio_hours": round(audio_s / 3600.0, 4),
        "labelled_seconds": {
            label: round(value, 2) for label, value in sorted(labelled_totals.items())
        },
        "failures": failures[:20],
        # The reading that decides whether the gate is fixable by tuning.
        "blank_rate_by_label": {
            label: round(blank_frames[label] / count, 4)
            for label, count in sorted(label_frames.items())
            if count > 0
        },
        "configs": {},
    }

    speech_total = labelled_totals.get(SPEECH_LABEL, 0.0)
    waste_total = sum(labelled_totals.get(label, 0.0) for label in WASTE_LABELS)
    for name, config in configs.items():
        bucket = totals[name]
        speech_kept = bucket.get(f"{SPEECH_LABEL}_kept_s", 0.0)
        lost = max(0.0, speech_total - speech_kept)
        waste_kept = sum(bucket.get(f"{label}_kept_s", 0.0) for label in WASTE_LABELS)
        unsure_kept = bucket.get(f"{UNSURE_LABEL}_kept_s", 0.0)
        report["configs"][name] = {
            "config": {
                "min_blank_s": config.min_blank_s,
                "min_speech_s": config.min_speech_s,
                "merge_gap_s": config.merge_gap_s,
                "pad_s": config.pad_s,
                "max_region_s": config.max_region_s,
            },
            # Named for what it is. Span labels mark "contains a word" over
            # stretches that are mostly not speech, so this over-counts by an
            # unknown factor and only bounds the loss from above.
            "speech_span_seconds_outside_kept_regions": round(lost, 2),
            "irreversible_speech_loss_s": round(lost, 2),
            "irreversible_speech_loss_min": round(lost / 60.0, 3),
            "irreversible_speech_loss_rate": round(lost / speech_total, 5)
            if speech_total > 0
            else None,
            "speech_recall": round(speech_kept / speech_total, 5)
            if speech_total > 0
            else None,
            "decode_fraction": round(bucket.get("decoded_s", 0.0) / audio_s, 5)
            if audio_s > 0
            else None,
            "wasted_decode_s": round(waste_kept, 2),
            "wasted_decode_rate": round(waste_kept / waste_total, 5)
            if waste_total > 0
            else None,
            "unsure_kept_s": round(unsure_kept, 2),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    header = (
        f"{'config':18s} {'跨度损失上界(min)':>16s} {'占比':>8s} "
        f"{'解码占比':>9s} {'白解码率':>9s}"
    )
    print()
    print(f"前置闸 · {windows} 窗口 / {report['audio_hours']}h / {args.partition} 分区")
    print(
        "注意：左侧两列是**上界**，不是漏词率。标注是「这段里有没有词」，"
        "跨度中位 7.5s，正确只解码其中说话的两秒也会被算成丢掉其余部分。"
    )
    print()
    print("逐帧 blank 率（未经闸门几何）：", json.dumps(report["blank_rate_by_label"]))
    print()
    print(header)
    for name, entry in report["configs"].items():
        loss_rate = entry["irreversible_speech_loss_rate"]
        print(
            f"{name:18s} {entry['irreversible_speech_loss_min']:>10.3f} "
            f"{(f'{loss_rate:.2%}' if loss_rate is not None else '--'):>8s} "
            f"{(f'{entry['decode_fraction']:.1%}' if entry['decode_fraction'] is not None else '--'):>9s} "
            f"{(f'{entry['wasted_decode_rate']:.1%}' if entry['wasted_decode_rate'] is not None else '--'):>9s}"
        )
    print()


if __name__ == "__main__":
    main()
