#!/usr/bin/env python3
"""Acceptance A/B for alignment heads, on a real film, one encoder pass.

The question a retrain has to answer is not "did val loss drop" - it is whether
the **free-running** posterior separates moaning from speech better than the
shipped head does, without becoming deaf to quiet speech. Those are two failures
in opposite directions and a single number hides one of them, so all four
acceptance criteria are reported side by side per head:

  1. `auc_blank`        vocalisation vs lexical dialogue. Baseline 0.9077; must rise.
  2. `dialogue_median`  blank share on unambiguous speech. Baseline 0.8429;
                        must NOT rise - a head that calls everything blank scores
                        a perfect AUC and is useless.
  3. `dialogue_at_1000` lexical cues the head calls 100% blank. Zero tolerance:
                        2026-08-11 found `おち○ちん、` and `うーん。` at exactly
                        1.0000, which is the posterior being deaf to breathy
                        speech rather than confident about silence.
  4. `auc_density`      the text-only control, `-chars_per_second`. Identical for
                        every head; it is here because an acoustic AUC only means
                        something above what the shipped text rule already knows.

**The encoder runs once for all heads.** Hidden states are what the encoder
produces and the head is a thin classifier on top, so re-encoding per head would
spend minutes of GPU to compute the same tensor and would also let the two heads
see different numerics. Same tensor, different classifiers, is the only way the
comparison is about the heads.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

SCHEMA = "alignment_head_acceptance_v1"


def percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return float(ordered[index])

    return {
        "n": len(ordered),
        "p10": round(at(0.10), 4),
        "p25": round(at(0.25), 4),
        "median": round(float(statistics.median(ordered)), 4),
        "p75": round(at(0.75), 4),
        "p90": round(at(0.90), 4),
        "mean": round(float(statistics.fmean(ordered)), 4),
    }


def auc(positive: list[float], negative: list[float]) -> float:
    """P(a random positive scores above a random negative), ties at 0.5."""
    if not positive or not negative:
        return float("nan")
    merged = sorted(
        [(value, 1) for value in positive] + [(value, 0) for value in negative]
    )
    index = 0
    rank_sum_positive = 0.0
    while index < len(merged):
        stop = index
        while stop + 1 < len(merged) and merged[stop + 1][0] == merged[index][0]:
            stop += 1
        average_rank = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            if merged[position][1] == 1:
                rank_sum_positive += average_rank
        index = stop + 1
    n_pos, n_neg = len(positive), len(negative)
    return (rank_sum_positive - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def encode_film(audio_path: str, *, context_frames: int, batch_size: int = 4):
    """Encoder hidden states for the whole film, in the production window plan.

    Copied from `asr.pipeline._blank_runs_for_audio` and stopped one step early:
    it returns the hidden states instead of the head's log-probs, so every head
    under test reads the same tensor.
    """
    import numpy as np
    import torch

    from asr.alignment import plan_head_windows
    from asr.pipeline import (
        _FEATURE_CHUNK_S,
        _asr_language_for_chunking,
        _load_asr_model_for_features,
    )
    from asr.encoder_features import qwen3_asr_audio_output_lengths
    from asr.qwen_native import move_processor_inputs, prepare_transcription_inputs
    from audio.loading import load_audio_16k_mono

    model, processor = _load_asr_model_for_features()
    audio, rate = load_audio_16k_mono(audio_path)
    if rate != 16000:
        raise SystemExit(f"unexpected sample rate {rate}")
    clip = np.asarray(audio, dtype=np.float32)
    duration_s = len(clip) / 16000.0

    width = int(_FEATURE_CHUNK_S * 16000)
    plan = plan_head_windows(
        len(clip), window_samples=width, context_frames=context_frames
    )
    windows = [np.ascontiguousarray(clip[start:end]) for start, end, _ in plan]
    bases = [base for _, _, base in plan]

    hidden_states: list[np.ndarray] = []
    returned_frames: list[int] = []
    started = time.time()
    for start in range(0, len(windows), batch_size):
        group = windows[start : start + batch_size]
        inputs = prepare_transcription_inputs(
            processor, audio=group, language=_asr_language_for_chunking()
        )
        moved = move_processor_inputs(inputs, device=model.device, dtype=model.dtype)
        with torch.inference_mode():
            features = model.get_audio_features(
                input_features=moved["input_features"],
                input_features_mask=moved["input_features_mask"],
            ).pooler_output
        lengths = [
            int(value)
            for value in qwen3_asr_audio_output_lengths(
                moved["input_features_mask"].sum(dim=1)
            )
        ]
        block = features.detach().float().cpu().numpy()
        offset = 0
        for length in lengths:
            # fp16: the head normalises its input first thing, and this is the
            # same precision the training cache stores.
            hidden_states.append(block[offset : offset + length].astype(np.float16))
            returned_frames.append(int(length))
            offset += length
        print(
            f"  encoded {min(start + batch_size, len(windows))}/{len(windows)} windows",
            flush=True,
        )
    print(f"  encoder pass {time.time() - started:.1f}s", flush=True)
    return hidden_states, returned_frames, bases, duration_s


def head_readings(head, hidden_states, returned_frames, bases):
    """`(is_blank, frame_posteriors)` over the whole film, one pass per head.

    The two readings come out of the same forward call because they have to
    describe the same audio: the point of the comparison is whether the frame
    classes separate what blank could not, and a second pass would let the two
    disagree for reasons that have nothing to do with the question.

    `frame_posteriors` is None for a v1 head, which is what makes the old and new
    readings reportable side by side - the shipped head simply has no column in
    the new table rather than being excluded from the run.
    """
    import numpy as np
    import torch

    from asr.alignment import BLANK_INDEX, overlap_save_slices

    pieces = [
        head.log_probs_with_frames(np.asarray(state, dtype=np.float32))
        for state in hidden_states
    ]
    keep = overlap_save_slices(
        list(zip(bases[: len(pieces)], returned_frames)),
        context_frames=head.context_frames,
    )

    def stitch(index: int):
        kept = [
            piece[index][begin * head.upsample : finish * head.upsample]
            for piece, (begin, finish) in zip(pieces, keep)
            if finish > begin
        ]
        return torch.cat(kept, dim=0) if len(kept) > 1 else kept[0]

    log_probs = stitch(0)
    predicted = log_probs.argmax(dim=-1).detach().cpu().numpy()
    is_blank = (predicted == BLANK_INDEX).astype(np.float64)
    if pieces[0][1] is None:
        return is_blank, None
    return is_blank, stitch(1).exp().detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bilingual", required=True, help="unfiltered *.bilingual.json")
    parser.add_argument("--audio", required=True)
    parser.add_argument(
        "--head",
        action="append",
        required=True,
        help="repeatable, `label=path/to/checkpoint.pt`",
    )
    parser.add_argument("--min-run", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()


    from core.config import load_config

    load_config()

    from asr.alignment import AlignmentHead, frame_to_seconds
    from subtitles.vocalisation import (
        _carries_lexical_content,
        _strip_decoration,
        block_text,
        drop_vocalisation_runs,
        is_non_semantic_vocalisation,
    )

    heads: list[tuple[str, object]] = []
    for spec in args.head:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--head wants `label=path`, got {spec!r}")
        heads.append((label, AlignmentHead.load(str(PROJECT_ROOT / path))))
        print(f"loaded head {label}: {path}")

    context = {head.context_frames for _, head in heads}
    upsamples = {head.upsample for _, head in heads}
    if len(context) != 1 or len(upsamples) != 1:
        raise SystemExit(
            "heads disagree on context frames or upsample; one encoder pass "
            "cannot serve both and the comparison would not be like for like"
        )

    blocks = json.loads(Path(args.bilingual).read_text(encoding="utf-8"))["blocks"]
    kept, diagnostics = drop_vocalisation_runs(blocks, min_run=args.min_run)
    kept_ids = {id(block) for block in kept}
    groups: dict[str, list[int]] = {
        "vocalisation_dropped": [],
        "vocalisation_isolated": [],
        "dialogue_lexical": [],
        "kana_only_kept": [],
    }
    for index, block in enumerate(blocks):
        text = block_text(block)
        flagged = is_non_semantic_vocalisation(text)
        if flagged and id(block) not in kept_ids:
            groups["vocalisation_dropped"].append(index)
        elif flagged:
            groups["vocalisation_isolated"].append(index)
        elif _carries_lexical_content(_strip_decoration(text)):
            groups["dialogue_lexical"].append(index)
        else:
            groups["kana_only_kept"].append(index)
    print(f"cues: {len(blocks)}  filter: {diagnostics}")
    for name, members in groups.items():
        print(f"  {name}: {len(members)}")

    hidden_states, returned_frames, bases, duration_s = encode_film(
        args.audio, context_frames=next(iter(context))
    )
    frame_s = frame_to_seconds(1, upsample=next(iter(upsamples)))

    def spans(is_blank, posteriors) -> dict[str, list[dict]]:
        from asr.alignment import span_class_shares

        out: dict[str, list[dict]] = {}
        for name, members in groups.items():
            rows = []
            for index in members:
                block = blocks[index]
                start = block.get("acoustic_start")
                end = block.get("acoustic_end")
                if start is None or end is None:
                    start, end = block.get("start"), block.get("end")
                if start is None or end is None or float(end) <= float(start):
                    continue
                first = int(float(start) / frame_s)
                last = max(first + 1, int(round(float(end) / frame_s)))
                window = is_blank[first : min(last, len(is_blank))]
                if window.size == 0:
                    continue
                text = block_text(block)
                row = {
                    "index": index,
                    "text": text,
                    "blank_share": float(window.mean()),
                    "duration_s": float(end) - float(start),
                    "chars_per_second": len(text)
                    / max(1e-6, float(end) - float(start)),
                }
                if posteriors is not None:
                    shares = span_class_shares(
                        posteriors,
                        float(start),
                        float(end),
                        upsample=next(iter(upsamples)),
                    )
                    row.update(
                        {
                            "silence_share": shares["silence"],
                            "vocalisation_share": shares["vocalisation"],
                            "speech_share": shares["speech"],
                            "speech_max_run_s": shares["speech_max_run_s"],
                        }
                    )
                rows.append(row)
            out[name] = rows
        return out

    report = {
        "schema": SCHEMA,
        "audio": args.audio,
        "bilingual": args.bilingual,
        "audio_duration_s": round(duration_s, 3),
        "frame_ms": round(frame_s * 1000, 4),
        "filter_diagnostics": diagnostics,
        "heads": {},
    }

    # Per-cue shares for every head, keyed by cue index. The group medians say
    # the scale moved; only the paired numbers say whether a *particular* cue got
    # worse. `dialogue_median` rising by 3pp reads very differently if the same
    # cues moved a little than if a handful collapsed to total silence.
    paired: dict[int, dict[str, object]] = {}

    control_done = False
    for label, head in heads:
        print(f"\n=== {label} ===", flush=True)
        is_blank, posteriors = head_readings(
            head, hidden_states, returned_frames, bases
        )
        measured = spans(is_blank, posteriors)
        for name, rows in measured.items():
            for row in rows:
                entry = paired.setdefault(
                    row["index"],
                    {
                        "group": name,
                        "text": row["text"],
                        "duration_s": round(row["duration_s"], 4),
                    },
                )
                entry[label] = round(row["blank_share"], 6)
                if "speech_share" in row:
                    entry[f"{label}:speech"] = round(row["speech_share"], 6)
                    entry[f"{label}:vocalisation"] = round(
                        row["vocalisation_share"], 6
                    )
                    entry[f"{label}:silence"] = round(row["silence_share"], 6)
                    # Per cue, not only as a group percentile: the joint verdict
                    # needs it on the individual cue to tell "one second of
                    # speech then five of moaning" from "six seconds of
                    # moaning", which the shares alone cannot do.
                    entry[f"{label}:speech_run"] = round(row["speech_max_run_s"], 6)
        positive = [row["blank_share"] for row in measured["vocalisation_dropped"]]
        negative = [row["blank_share"] for row in measured["dialogue_lexical"]]
        at_one = [
            row for row in measured["dialogue_lexical"] if row["blank_share"] >= 0.99999
        ]
        entry = {
            "overall_blank_share": round(float(is_blank.mean()), 6),
            "auc_blank": round(auc(positive, negative), 4),
            "dialogue_median": percentiles(negative).get("median"),
            "dialogue_at_1000": len(at_one),
            "dialogue_at_1000_examples": [row["text"] for row in at_one[:12]],
            "groups": {
                name: percentiles([row["blank_share"] for row in rows])
                for name, rows in measured.items()
            },
            "threshold": [
                {
                    "cut": cut,
                    "recall": round(
                        sum(1 for v in positive if v >= cut) / max(len(positive), 1), 4
                    ),
                    "false_drop_share": round(
                        sum(1 for v in negative if v >= cut) / max(len(negative), 1), 4
                    ),
                }
                for cut in (0.90, 0.93, 0.95, 0.97, 0.98)
            ],
        }

        # The new reading, reported beside the old one rather than instead of it.
        # A v1 head has no frame classes and simply carries no `frame` block, so
        # one table holds both generations and the comparison stays paired on the
        # same cues.
        entry["frame_head_available"] = posteriors is not None
        if posteriors is not None:
            # Scored as "how much of this span is NOT speech", which is the same
            # question `blank_share` was asked - so the two AUCs are comparable
            # and any difference is the class system, not the statistic.
            non_speech_positive = [
                1.0 - row["speech_share"] for row in measured["vocalisation_dropped"]
            ]
            non_speech_negative = [
                1.0 - row["speech_share"] for row in measured["dialogue_lexical"]
            ]
            # The like-for-like counterpart of `dialogue_at_1000`, which counts
            # cues whose every frame ARGMAXES to blank - a rate, and one that
            # really can be exactly 1. `speech_share` is a mean posterior and a
            # softmax never makes it exactly 0, so testing it against zero would
            # report a triumphant 0 for reasons that have nothing to do with the
            # head hearing speech. `speech_max_run_s == 0` is the rate version:
            # not one frame of this cue cleared the speech threshold.
            deaf = [
                row
                for row in measured["dialogue_lexical"]
                if row["speech_max_run_s"] <= 0.0
            ]
            entry["frame"] = {
                "auc_non_speech": round(
                    auc(non_speech_positive, non_speech_negative), 4
                ),
                "auc_vocalisation": round(
                    auc(
                        [
                            row["vocalisation_share"]
                            for row in measured["vocalisation_dropped"]
                        ],
                        [
                            row["vocalisation_share"]
                            for row in measured["dialogue_lexical"]
                        ],
                    ),
                    4,
                ),
                "dialogue_speech_median": percentiles(
                    [row["speech_share"] for row in measured["dialogue_lexical"]]
                ).get("median"),
                # The v2 failure in its new spelling: a cue of real words in
                # which not one frame votes speech.
                "dialogue_with_zero_speech": len(deaf),
                "dialogue_with_zero_speech_examples": [row["text"] for row in deaf[:12]],
                # How near the floor the quietest dialogue gets. v2's failure was
                # a saturation - 87 cues at exactly 1.0000 blank - so the shape
                # of this tail is the thing to compare, not only its count.
                "dialogue_speech_share_floor": round(
                    min(
                        (row["speech_share"] for row in measured["dialogue_lexical"]),
                        default=0.0,
                    ),
                    6,
                ),
                "dialogue_below_5pc_speech": sum(
                    1
                    for row in measured["dialogue_lexical"]
                    if row["speech_share"] < 0.05
                ),
                "groups": {
                    name: {
                        "n": len(rows),
                        "silence": percentiles(
                            [row["silence_share"] for row in rows]
                        ).get("median"),
                        "vocalisation": percentiles(
                            [row["vocalisation_share"] for row in rows]
                        ).get("median"),
                        "speech": percentiles(
                            [row["speech_share"] for row in rows]
                        ).get("median"),
                        "speech_max_run_s": percentiles(
                            [row["speech_max_run_s"] for row in rows]
                        ).get("median"),
                    }
                    for name, rows in measured.items()
                },
                "threshold": [
                    {
                        "cut": cut,
                        "recall": round(
                            sum(1 for v in non_speech_positive if v >= cut)
                            / max(len(non_speech_positive), 1),
                            4,
                        ),
                        "false_drop_share": round(
                            sum(1 for v in non_speech_negative if v >= cut)
                            / max(len(non_speech_negative), 1),
                            4,
                        ),
                    }
                    for cut in (0.70, 0.80, 0.90, 0.95, 0.98)
                ],
            }
        if not control_done:
            report["auc_density_control"] = round(
                auc(
                    [-r["chars_per_second"] for r in measured["vocalisation_dropped"]],
                    [-r["chars_per_second"] for r in measured["dialogue_lexical"]],
                ),
                4,
            )
            control_done = True
        report["heads"][label] = entry
        print(
            f"  auc_blank={entry['auc_blank']:.4f}  "
            f"dialogue_median={entry['dialogue_median']:.4f}  "
            f"dialogue_at_1000={entry['dialogue_at_1000']}  "
            f"overall_blank={entry['overall_blank_share']:.4f}"
        )

    report["cues"] = [
        {"index": index, **values} for index, values in sorted(paired.items())
    ]

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== acceptance ===")
    print(f"text-only control AUC(-chars/s) = {report['auc_density_control']:.4f}")
    print(
        f"{'head':>22}  {'AUC(blank)':>10}  {'dialogue med':>12}  "
        f"{'blank=1.000':>11}  {'voc med':>8}"
    )
    for label, entry in report["heads"].items():
        print(
            f"{label:>22}  {entry['auc_blank']:>10.4f}  {entry['dialogue_median']:>12.4f}  "
            f"{entry['dialogue_at_1000']:>11}  "
            f"{entry['groups']['vocalisation_dropped']['median']:>8.4f}"
        )

    framed = {
        label: entry["frame"]
        for label, entry in report["heads"].items()
        if entry.get("frame")
    }
    if framed:
        print("\n=== frame-class reading (v2 heads only) ===")
        print(
            f"{'head':>22}  {'AUC(1-sp)':>10}  {'AUC(voc)':>9}  "
            f"{'dlg speech':>11}  {'speech=0':>9}"
        )
        for label, frame in framed.items():
            print(
                f"{label:>22}  {frame['auc_non_speech']:>10.4f}  "
                f"{frame['auc_vocalisation']:>9.4f}  "
                f"{frame['dialogue_speech_median']:>11.4f}  "
                f"{frame['dialogue_with_zero_speech']:>9}"
            )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
