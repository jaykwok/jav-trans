#!/usr/bin/env python3
"""Pair real JAV audio with its own ASR text, as CTC training data.

The alignment head was trained only on clean galgame speech, and a 2026-07-31
measurement showed what that costs: its blank reading separates lexically dense
speech from vocalisation-dense audio rather than speech from silence. Inside
human-labelled speech it calls 92.2% of frames blank, against 99.66% in audio
with no voice at all - 7.4 points of headroom, because はぁ / んっ / ちゅっ were
never in its vocabulary as *characters*. In this domain words sit inside that
vocalisation, so a gate built on the reading dropped real lines.

The fix is to show the head the domain. The pairing is free, exactly as it was
for galgame: the ASR transcribes the audio, and its output is the target.

**Clips are cut on a fixed grid, not on the head's pauses.** Selecting training
audio with the very reading that is being repaired would feed back its own blind
spot - stretches it already calls blank would never appear as training targets,
which is precisely the region that needs to stop being blank. A fixed grid is
unbiased with respect to the thing being fixed.

**Runaway decodes are filtered out with the post-gate.** A repetition loop
(`んっ、んんっ…` twelve times) is text the audio does not support, and training
on it would teach the head to align loops - so `asr.postgate` flags are applied
here and flagged clips are dropped rather than paired.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import wave

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from asr.alignment import ENCODER_FPS, normalize_text  # noqa: E402
from asr.cue_features import build_candidate  # noqa: E402
from asr.postgate import PostGateConfig, review  # noqa: E402
from audio.loading import load_audio_16k_mono  # noqa: E402
from utils.gpu_safety import apply_vram_safety_cap  # noqa: E402

MANIFEST_SCHEMA = "real_alignment_clip_manifest_v1"
SAMPLE_RATE = 16000
# Only the real-audio provenance in the relabel corpus. The same file also holds
# 4096 synthetic hardmix windows - galgame speech composited into JAV noise -
# and training on those would reinforce exactly the domain gap being closed.
REAL_PROVENANCE = "real_omni_joint"
# `example_id` looks like `omni-joint-boundary-preasr-v1:294-fhd-...-w00`, and on
# Windows a colon in a path opens an NTFS alternate data stream instead of a
# file: every clip silently became a stream hanging off a zero-byte file named
# after the dataset. Anything that walks the directory - or copies it - loses
# the audio. Keep this whitelist rather than stripping known-bad characters.
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(value: str) -> str:
    return _UNSAFE_NAME.sub("_", value).strip("_") or "clip"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_wav(path: Path, clip: np.ndarray) -> None:
    samples = np.clip(np.asarray(clip, dtype=np.float32), -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes((samples * 32767.0).astype("<i2").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="")
    parser.add_argument(
        "--partitions",
        default="train,val",
        help="comma-separated; `test` is excluded by default so the pre-gate "
        "measurement stays honest after retraining",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--provenance",
        default=REAL_PROVENANCE,
        help="empty string keeps every provenance; the default keeps only real "
        "audio, because the corpus is 83%% synthetic hardmix",
    )
    parser.add_argument("--windows", type=int, default=400)
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--min-clip-seconds", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from asr.backends.qwen import active_qwen_asr_model_id, active_qwen_asr_model_path
    from utils.model_paths import resolve_model_spec

    apply_vram_safety_cap(0.95)

    rows = _read_jsonl(Path(args.dataset))
    wanted = {part.strip() for part in args.partitions.split(",") if part.strip()}
    if args.split:
        partitions = {
            str(entry.get("example_id")): str(entry.get("partition") or "")
            for entry in _read_jsonl(Path(args.split))
        }
        rows = [
            row
            for row in rows
            if partitions.get(str(row.get("example_id"))) in wanted
        ]
    if args.provenance:
        rows = [row for row in rows if str(row.get("provenance")) == args.provenance]
    if not rows:
        raise SystemExit("no rows in the requested partitions/provenance")
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
    post_config = PostGateConfig()

    def transcribe(clips: list[np.ndarray]) -> list[str]:
        """One `generate` per batch rather than per clip.

        Unbatched, the per-call prompt and launch overhead dominated: 75 s of
        audio took ~40 s to decode, an RTF of 0.53 against the 0.123 measured
        for the decoder itself. A whole window goes in one call instead.
        """
        if not clips:
            return []
        inputs = processor.apply_transcription_request(audio=clips, language=None)
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
        texts = [str(entry.get("transcription") or "") for entry in parsed]
        if len(texts) != len(clips):
            raise SystemExit(
                f"decoder returned {len(texts)} transcripts for {len(clips)} clips; "
                "pairing text with the wrong audio would poison the training set"
            )
        return texts

    output_dir = Path(args.output_dir)
    clip_dir = output_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    skipped: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    audio_s = 0.0

    for index, row in enumerate(picked):
        source = str(row.get("audio") or "")
        try:
            audio, rate = load_audio_16k_mono(source)
        except Exception:  # noqa: BLE001
            skipped["audio_unreadable"] += 1
            continue
        if rate != SAMPLE_RATE:
            skipped["unexpected_sample_rate"] += 1
            continue
        clip = np.asarray(audio, dtype=np.float32)
        video_id = str(row.get("video_id") or "")
        example_id = str(row.get("example_id") or f"window{index:05d}")
        width = int(args.clip_seconds * SAMPLE_RATE)

        pieces: list[tuple[int, np.ndarray]] = []
        for offset in range(0, len(clip), width):
            piece = np.ascontiguousarray(clip[offset : offset + width])
            if len(piece) / SAMPLE_RATE < args.min_clip_seconds:
                skipped["clip_too_short"] += 1
                continue
            pieces.append((offset, piece))

        for (offset, piece), raw_text in zip(
            pieces, transcribe([piece for _, piece in pieces])
        ):
            seconds = len(piece) / SAMPLE_RATE
            text = raw_text.strip()
            normalized = normalize_text(text)
            if not normalized:
                skipped["empty_text"] += 1
                continue
            # CTC needs at least one frame per character, and the extractor
            # applies the same rule later; rejecting here keeps the manifest
            # honest about how many clips it actually contributes.
            if len(normalized) > seconds * ENCODER_FPS:
                skipped["text_denser_than_frame_rate"] += 1
                continue
            candidate = build_candidate(
                chunk={"index": 0, "start": 0.0, "end": seconds},
                text_result={"text": text, "raw_text": text},
                position=0,
                chunks=[{"index": 0, "start": 0.0, "end": seconds}],
                text_results=[{"text": text, "raw_text": text}],
                audio_id=example_id,
                video_id=video_id,
            )
            verdict = review(candidate, config=post_config)
            if verdict["flags"]:
                for flag in verdict["flags"]:
                    flag_counts[flag] += 1
                skipped["postgate_flagged"] += 1
                continue

            audio_id = f"{example_id}@{offset // width:03d}"
            clip_path = clip_dir / f"{_safe_name(audio_id)}.wav"
            _write_wav(clip_path, piece)
            manifest.append(
                {
                    "schema": MANIFEST_SCHEMA,
                    "audio_id": audio_id,
                    "audio": str(clip_path),
                    "text": normalized,
                    "duration_s": round(seconds, 4),
                    "video_id": video_id,
                    "source_audio": source,
                    "source_offset_s": round(offset / SAMPLE_RATE, 3),
                }
            )
            audio_s += seconds

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in manifest:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    characters = Counter()
    for entry in manifest:
        characters.update(entry["text"])
    summary = {
        "schema": "real_alignment_manifest_summary_v1",
        "provenance": args.provenance,
        "windows_requested": len(picked),
        "clips": len(manifest),
        "audio_hours": round(audio_s / 3600.0, 4),
        "videos": len({entry["video_id"] for entry in manifest}),
        "distinct_characters": len(characters),
        "total_characters": sum(characters.values()),
        "chars_per_second": round(sum(characters.values()) / audio_s, 3)
        if audio_s > 0
        else None,
        "skipped": dict(skipped),
        "postgate_flags": dict(flag_counts),
        "manifest": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
