#!/usr/bin/env python3
"""Decode what the pre-gate threw away, and see whether words come out.

`evaluate_pregate_loss.py` can only bound the loss from above, because the
relabel corpus marks spans that *contain* a word rather than the speech itself,
and its spans run to seventeen seconds. This tool removes that confound by
asking the decoder directly: take the audio the gate skipped, transcribe it, and
count the words that were about to be lost.

The obvious trap is that a decoder handed wordless audio does not return
nothing - it hallucinates. So text coming out of a dropped region is not by
itself evidence the gate was wrong, and three independent readings are kept
apart rather than merged:

  * **post-gate flags** (`asr.postgate`) catch the degenerate hallucinations,
    the loops and impossible rates.
  * **forced-alignment score** catches the fluent ones. Invented text does not
    fit the acoustics, and this is the signal the old design never had.
  * **label corroboration** - whether the dropped region overlaps a span a human
    audited as containing a word. Coarse, but it is independent of both of the
    above and of the model that produced them.

A matched sample of KEPT regions is decoded alongside, for two reasons: it is
the reference distribution the alignment score has to be read against, and it
is the control that says the decode path itself is working. Without it a low
score in the dropped set would be uninterpretable.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.alignment import AlignmentHead, blank_runs, normalize_text  # noqa: E402
from asr.cue_features import build_candidate  # noqa: E402
from asr.postgate import PostGateConfig, review  # noqa: E402
from tools.align.pregate_reference import PreGateConfig, speech_regions  # noqa: E402
from audio.loading import load_audio_16k_mono  # noqa: E402
from utils.gpu_safety import apply_vram_safety_cap  # noqa: E402
from asr.encoder_features import qwen3_asr_audio_output_lengths  # noqa: E402

RESULT_SCHEMA = "pregate_dropped_audio_probe_v1"
SAMPLE_RATE = 16000
FEATURE_CHUNK_S = 30.0


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _complement(
    regions: list[tuple[float, float]], total_s: float
) -> list[tuple[float, float]]:
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for begin, end in regions:
        if begin > cursor:
            gaps.append((cursor, begin))
        cursor = max(cursor, end)
    if cursor < total_s:
        gaps.append((cursor, total_s))
    return gaps


def _pieces(
    spans: list[tuple[float, float]], *, min_s: float, max_s: float
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for begin, end in spans:
        width = end - begin
        if width < min_s:
            continue
        count = max(1, int(np.ceil(width / max_s)))
        step = width / count
        out.extend(
            (begin + index * step, begin + (index + 1) * step) for index in range(count)
        )
    return out


def _overlaps(span: tuple[float, float], spans: list[tuple[float, float]]) -> float:
    begin, end = span
    return sum(
        max(0.0, min(end, other_end) - max(begin, other_begin))
        for other_begin, other_end in spans
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="")
    parser.add_argument("--partition", default="test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--windows", type=int, default=40)
    parser.add_argument("--min-piece-s", type=float, default=1.0)
    parser.add_argument("--max-piece-s", type=float, default=20.0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--kept-sample", type=int, default=120)
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
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows))[: max(1, args.windows)]
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
    config = PreGateConfig()
    post_config = PostGateConfig()

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

    def features_of(clip: np.ndarray):
        moved = _move(clip)
        with torch.inference_mode():
            audio_features = model.get_audio_features(
                input_features=moved["input_features"],
                input_features_mask=moved["input_features_mask"],
            ).pooler_output
        frames = int(
            qwen3_asr_audio_output_lengths(moved["input_features_mask"].sum(dim=1))[0]
        )
        return audio_features[:frames].detach().float().cpu().numpy()

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
        parsed = processor.parse_output(decoded)
        if isinstance(parsed, dict):
            parsed = [parsed]
        return str(parsed[0].get("transcription") or "")

    records: list[dict[str, Any]] = []
    dropped_s = 0.0
    kept_s = 0.0
    audio_s = 0.0
    kept_budget = max(0, args.kept_sample)

    for row in picked:
        try:
            audio, rate = load_audio_16k_mono(str(row.get("audio") or ""))
        except Exception:  # noqa: BLE001
            continue
        if rate != SAMPLE_RATE:
            continue
        clip = np.asarray(audio, dtype=np.float32)
        total_s = len(clip) / SAMPLE_RATE
        if total_s <= 0:
            continue
        audio_s += total_s

        posteriors = []
        width = int(FEATURE_CHUNK_S * SAMPLE_RATE)
        for offset in range(0, len(clip), width):
            piece = np.ascontiguousarray(clip[offset : offset + width])
            if len(piece) < SAMPLE_RATE // 2:
                continue
            posteriors.append(head.log_probs(features_of(piece)))
        if not posteriors:
            continue
        log_probs = torch.cat(posteriors, dim=0) if len(posteriors) > 1 else posteriors[0]

        runs = blank_runs(
            log_probs, upsample=head.upsample, min_seconds=config.min_blank_s
        )
        kept = speech_regions(runs, total_s, config)
        dropped = _complement(kept, total_s)
        kept_s += sum(end - begin for begin, end in kept)
        dropped_s += sum(end - begin for begin, end in dropped)

        truth = [
            (float(span["start_s"]), float(span["end_s"]))
            for span in row.get("spans") or ()
            if str(span.get("type")) == "speech"
        ]

        probes = [
            ("dropped", span)
            for span in _pieces(
                dropped, min_s=args.min_piece_s, max_s=args.max_piece_s
            )
        ]
        if kept_budget > 0:
            kept_pieces = _pieces(kept, min_s=args.min_piece_s, max_s=args.max_piece_s)
            take = kept_pieces[: min(len(kept_pieces), kept_budget)]
            kept_budget -= len(take)
            probes.extend(("kept", span) for span in take)

        for arm, (begin, end) in probes:
            first = int(begin * SAMPLE_RATE)
            last = min(len(clip), int(end * SAMPLE_RATE))
            piece = np.ascontiguousarray(clip[first:last])
            if len(piece) < SAMPLE_RATE // 2:
                continue
            text = transcribe(piece).strip()
            score = None
            characters = len(normalize_text(text))
            if characters:
                aligned = head.align_extent(features_of(piece), text)
                if aligned:
                    spans = aligned[0]
                    score = sum(span.score for span in spans) / len(spans)
            duration_s = (last - first) / SAMPLE_RATE
            candidate = build_candidate(
                chunk={"index": 0, "start": begin, "end": end},
                text_result={"text": text, "raw_text": text},
                position=0,
                chunks=[{"index": 0, "start": begin, "end": end}],
                text_results=[{"text": text, "raw_text": text}],
                audio_id=str(row.get("example_id") or ""),
            )
            verdict = review(candidate, alignment_score=score, config=post_config)
            # Not all text in dropped audio is a loss worth counting. This
            # domain's ASR emits a great deal of genuine non-semantic
            # vocalisation - はぁ, んっ, ちゅっ - and the relabel work already
            # decided those are not words. What must not be lost is lexical
            # content, so it is counted separately rather than folded in.
            text_signals = candidate.get("text_features") or {}
            lexical = bool(text_signals.get("has_stable_vocabulary")) or bool(
                text_signals.get("has_kanji")
            )
            records.append(
                {
                    "lexical": lexical,
                    "has_kanji": bool(text_signals.get("has_kanji")),
                    "kana_ratio": text_signals.get("kana_ratio"),
                    "example_id": row.get("example_id"),
                    "arm": arm,
                    "start_s": round(begin, 3),
                    "end_s": round(end, 3),
                    "duration_s": round(duration_s, 3),
                    "text": text,
                    "characters": characters,
                    "chars_per_s": round(characters / duration_s, 3)
                    if duration_s > 0
                    else 0.0,
                    "alignment_score": round(score, 4) if score is not None else None,
                    "postgate_flags": verdict["flags"],
                    "labelled_speech_overlap_s": round(
                        _overlaps((begin, end), truth), 3
                    ),
                }
            )

    def _summarise(arm: str) -> dict[str, Any]:
        subset = [record for record in records if record["arm"] == arm]
        scored = [
            record["alignment_score"]
            for record in subset
            if record["alignment_score"] is not None
        ]
        with_text = [record for record in subset if record["characters"] > 0]
        clean = [record for record in with_text if not record["postgate_flags"]]
        lexical = [record for record in clean if record["lexical"]]
        return {
            "lexical_pieces": len(lexical),
            "lexical_rate": round(len(lexical) / len(subset), 4) if subset else None,
            "lexical_seconds": round(
                sum(record["duration_s"] for record in lexical), 2
            ),
            "pieces": len(subset),
            "seconds": round(sum(record["duration_s"] for record in subset), 2),
            "with_text": len(with_text),
            "with_text_rate": round(len(with_text) / len(subset), 4) if subset else None,
            "clean_text": len(clean),
            "clean_text_rate": round(len(clean) / len(subset), 4) if subset else None,
            "clean_text_seconds": round(
                sum(record["duration_s"] for record in clean), 2
            ),
            "clean_and_corroborated": sum(
                1 for record in clean if record["labelled_speech_overlap_s"] > 0.0
            ),
            "alignment_score": {
                "count": len(scored),
                "median": round(statistics.median(scored), 4) if scored else None,
                "p10": round(sorted(scored)[int(0.10 * len(scored))], 4)
                if scored
                else None,
                "p90": round(sorted(scored)[min(len(scored) - 1, int(0.90 * len(scored)))], 4)
                if scored
                else None,
            },
            "flag_counts": dict(
                sorted(
                    (
                        (flag, sum(1 for r in subset if flag in r["postgate_flags"]))
                        for flag in {f for r in subset for f in r["postgate_flags"]}
                    ),
                    key=lambda item: -item[1],
                )
            ),
        }

    report = {
        "schema": RESULT_SCHEMA,
        "checkpoint": args.checkpoint,
        "partition": args.partition if args.split else "all",
        "windows": len(picked),
        "audio_seconds": round(audio_s, 2),
        "gate": {
            "kept_seconds": round(kept_s, 2),
            "dropped_seconds": round(dropped_s, 2),
            "decode_fraction": round(kept_s / audio_s, 5) if audio_s else None,
        },
        "arms": {arm: _summarise(arm) for arm in ("kept", "dropped")},
    }
    dropped_arm = report["arms"]["dropped"]
    if dropped_arm["pieces"]:
        # The headline: of the audio the gate threw away, how much produced text
        # that survives every check. Scaled back to the whole dropped pool.
        report["estimated_lost_speech_seconds"] = round(
            dropped_s * (dropped_arm["clean_text_seconds"] / max(1e-9, dropped_arm["seconds"])),
            2,
        )
        report["estimated_lost_lexical_seconds"] = round(
            dropped_s * (dropped_arm["lexical_seconds"] / max(1e-9, dropped_arm["seconds"])),
            2,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with output.with_suffix(".records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print()


if __name__ == "__main__":
    main()
