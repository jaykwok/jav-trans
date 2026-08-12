#!/usr/bin/env python3
"""What exactly did the strip remove, and is any of it real speech?

The stripped target is only safe in one direction. A missed moan stays in the
target and costs a little supervision; a stripped *word* trains the head to call
real speech blank, which is the failure this whole line of work exists to remove.
So the removed fragments get audited rather than sampled by eye.

Three checks, cheapest first:

1. **Kanji.** `_carries_lexical_content` treats any kanji as lexical, so a
   removed fragment containing one is a contradiction in the classifier, not a
   judgement call. The expected count is zero, and a non-zero count is a bug.
2. **Frequency.** Removing the same fragment ten thousand times is a decision
   about a handful of distinct strings. Listing them by count puts nearly the
   whole mass under human review for the price of reading 40 lines.
3. **Third party.** The local ASR transcribes clips whose removed span was
   longest. If it reads back words where the script says moaning, the strip is
   deleting speech.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import unicodedata
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))



from tools.align.build_vocalisation_stripped_manifest import split_parts  # noqa: E402
from subtitles.vocalisation import is_non_semantic_vocalisation  # noqa: E402

SCHEMA = "stripped_fragment_audit_v1"

KANJI = "CJK UNIFIED"


def has_kanji(text: str) -> bool:
    return any(unicodedata.name(ch, "").startswith(KANJI) for ch in text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--listen-n", type=int, default=60)
    parser.add_argument("--asr-batch", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (PROJECT_ROOT / args.manifest)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    print(f"rows: {len(rows)}")

    removed = collections.Counter()
    removed_kanji: list[tuple[str, str]] = []
    per_row_removed: list[tuple[int, dict, str]] = []
    for row in rows:
        script = str(row.get("script_text") or "")
        gone = [
            fragment
            for fragment, is_separator in split_parts(script)
            if not is_separator and is_non_semantic_vocalisation(fragment)
        ]
        for fragment in gone:
            removed[fragment] += 1
            if has_kanji(fragment):
                removed_kanji.append((fragment, script))
        if gone:
            per_row_removed.append((sum(len(f) for f in gone), row, "".join(gone)))

    print(f"\ndistinct removed fragments: {len(removed)}")
    print(f"total removals: {sum(removed.values())}")
    print(f"kanji-bearing removals: {len(removed_kanji)}")
    for fragment, script in removed_kanji[:20]:
        print(f"  BUG removed={fragment!r} from {script[:50]!r}")

    total = sum(removed.values())
    print("\n=== most-removed fragments ===")
    covered = 0
    for fragment, count in removed.most_common(40):
        covered += count
        print(f"  {count:>6}  {covered / total:>6.1%}  {fragment}")

    report = {
        "schema": SCHEMA,
        "rows": len(rows),
        "distinct_removed_fragments": len(removed),
        "total_removals": total,
        "kanji_bearing_removals": len(removed_kanji),
        "top_40_coverage": round(covered / max(total, 1), 4),
        "top_fragments": [
            {"fragment": fragment, "count": count}
            for fragment, count in removed.most_common(60)
        ],
    }

    # The listening test: the clips where the strip removed the most, decoded by
    # a third party that has never seen the script.
    if args.listen_n > 0:
        import numpy as np

        from asr.backends.registry import _resolve_asr_backend
        from audio.loading import load_audio_16k_mono
        from core.config import load_config

        load_config()
        per_row_removed.sort(key=lambda item: -item[0])
        picked = per_row_removed[: args.listen_n]
        staging = PROJECT_ROOT / "tmp" / "strip-audit"
        staging.mkdir(parents=True, exist_ok=True)
        backend = _resolve_asr_backend("cuda")

        heard_rows = []
        for start in range(0, len(picked), args.asr_batch):
            group = picked[start : start + args.asr_batch]
            wav_paths = []
            for offset, (_size, row, _gone) in enumerate(group):
                audio = PROJECT_ROOT / str(row["audio"]).replace("\\", "/")
                clip, rate = load_audio_16k_mono(str(audio))
                clip = np.asarray(clip, dtype=np.float32)
                # Named by position, not by `audio_id`. These corpora share a
                # 40-character prefix across every clip, so truncating the id
                # wrote the whole batch to one file and the ASR read the last
                # clip back sixteen times - which looked like sixteen agreeing
                # transcripts rather than one file.
                wav_path = staging / f"clip-{start + offset:05d}.wav"
                with wave.open(str(wav_path), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(int(rate))
                    handle.writeframes(
                        (np.clip(clip, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                    )
                wav_paths.append(str(wav_path))
            results = backend.transcribe_texts(wav_paths)
            for (_size, row, gone), result in zip(group, results):
                heard_rows.append(
                    {
                        "audio_id": row["audio_id"],
                        "script": row["script_text"],
                        "target": row["text"],
                        "removed": gone,
                        "local_asr": str((result or {}).get("text") or ""),
                    }
                )
            for path in wav_paths:
                Path(path).unlink(missing_ok=True)
            print(f"  transcribed {len(heard_rows)}/{len(picked)}", flush=True)

        # If the strip deleted speech, the local ASR should report content that
        # the *target* does not have. Measure that as characters heard but not
        # in the target, ignoring anything the target already covers.
        leaked = 0
        for record in heard_rows:
            target_chars = collections.Counter(record["target"])
            extra = collections.Counter(record["local_asr"]) - target_chars
            record["chars_beyond_target"] = sum(extra.values())
            record["kanji_beyond_target"] = sum(
                count for ch, count in extra.items() if has_kanji(ch)
            )
            if record["kanji_beyond_target"] > 0:
                leaked += 1
        report["listened"] = len(heard_rows)
        report["clips_with_kanji_beyond_target"] = leaked
        report["records"] = heard_rows
        print(f"\n=== listening test on {len(heard_rows)} heaviest strips ===")
        print(
            f"clips where the local ASR heard kanji the target lacks: "
            f"{leaked} ({leaked / max(len(heard_rows), 1):.1%})"
        )
        worst = sorted(heard_rows, key=lambda r: -r["kanji_beyond_target"])[:10]
        for record in worst:
            print(f"  removed={record['removed'][:30]}")
            print(f"  target ={record['target'][:44]}")
            print(f"  heard  ={record['local_asr'][:44]}")

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
