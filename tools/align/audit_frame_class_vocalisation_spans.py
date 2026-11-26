#!/usr/bin/env python3
"""Audit ③ of the L1 frame labels: is anything labelled `vocalisation` a word?

L1 is the only source of new information in the three-class labels, and the only
one that can be wrong in the expensive direction. Labelling a moan as speech
costs a little supervision; labelling *speech* as vocalisation teaches the head
that a word is a moan, and the filter downstream then deletes it. The checks
already run are both readings of the same script the labels came from - kanji
count, block frequency - so neither can catch a script that was wrong, or an
alignment that put a semantic block's audio under a vocalisation label.

A third party settles it. The local ASR has never seen these scripts, and it is
asked to decode exactly the spans the labels call vocalisation. Where it reports
lexical content, the label and an independent decode disagree, and those spans
are the ones to listen to.

The longest spans are chosen rather than a uniform sample, deliberately: a long
span is where a mislabel costs the most supervision, it is the easiest for an
ear to judge, and an alignment that dragged a semantic block into a vocalisation
one produces a long span rather than a short one. That biases the sample toward
finding problems, which is the right direction for a safety check.

Also writes the blind manifest for the listening page, since the ASR is a proxy
for the ear and the gate is stated in the ear's terms.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys
import unicodedata
import wave

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

SCHEMA = "frame_class_vocalisation_span_audit_v1"
MANIFEST_SCHEMA = "frame_class_span_audit_manifest_v1"


def has_lexical(text: str) -> bool:
    """Kanji, latin or digits - the same test the filter settles cues with."""
    for char in str(text or ""):
        if char.isdigit() or (char.isascii() and char.isalpha()):
            return True
        if unicodedata.name(char, "").startswith("CJK UNIFIED"):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="frame_class_label_report.json")
    parser.add_argument("--cache", default="anime-nsfw")
    parser.add_argument("--listen-n", type=int, default=60)
    parser.add_argument("--asr-batch", type=int, default=8)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    report = json.loads((PROJECT_ROOT / args.report).read_text(encoding="utf-8"))
    spans = report["by_cache"][args.cache]["longest_vocalisation_spans"][
        : args.listen_n
    ]
    if not spans:
        raise SystemExit("the report carries no vocalisation spans")
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from asr.backends.registry import _resolve_asr_backend
    from audio.loading import load_audio_16k_mono
    from core.config import load_config

    load_config()
    staging = PROJECT_ROOT / "tmp" / "frame-class-span-audit"
    staging.mkdir(parents=True, exist_ok=True)
    backend = _resolve_asr_backend("cuda")

    records: list[dict] = []
    for start in range(0, len(spans), args.asr_batch):
        group = spans[start : start + args.asr_batch]
        paths = []
        for offset, span in enumerate(group):
            audio_path = PROJECT_ROOT / str(span["audio"]).replace("\\", "/")
            clip, rate = load_audio_16k_mono(str(audio_path))
            clip = np.asarray(clip, dtype=np.float32)
            first = max(0, int(round(float(span["start_s"]) * rate)))
            last = min(clip.size, int(round(float(span["end_s"]) * rate)))
            cut = clip[first:last]
            # Named by position. These corpora share a long id prefix, and
            # truncating the id once wrote a whole batch to one file - which
            # read back as sixteen agreeing transcripts of the last clip.
            wav_path = staging / f"span-{start + offset:05d}.wav"
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(int(rate))
                handle.writeframes(
                    (np.clip(cut, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                )
            paths.append(str(wav_path))
        results = backend.transcribe_texts(paths)
        for span, result in zip(group, results):
            heard = str((result or {}).get("text") or "")
            records.append(
                {
                    **span,
                    "local_asr": heard,
                    "asr_heard_lexical": bool(has_lexical(heard)),
                }
            )
        for path in paths:
            Path(path).unlink(missing_ok=True)
        print(f"  decoded {len(records)}/{len(spans)}", flush=True)

    disagreements = [r for r in records if r["asr_heard_lexical"]]
    summary = {
        "schema": SCHEMA,
        "report": args.report,
        "cache": args.cache,
        "spans_audited": len(records),
        "seconds_audited": round(sum(float(r["seconds"]) for r in records), 1),
        "asr_heard_lexical": len(disagreements),
        "asr_heard_lexical_share": round(len(disagreements) / max(len(records), 1), 4),
        "records": records,
    }
    (out_dir / "span_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            handle.write(
                json.dumps(
                    {
                        "schema": MANIFEST_SCHEMA,
                        "row_id": f"span-{index:03d}",
                        "audio": str(
                            PROJECT_ROOT / str(record["audio"]).replace("\\", "/")
                        ),
                        "start_s": float(record["start_s"]),
                        "end_s": float(record["end_s"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    key = out_dir / "answer_key.jsonl"
    with key.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            handle.write(
                json.dumps({"row_id": f"span-{index:03d}", **record}, ensure_ascii=False)
                + "\n"
            )

    print(f"\nspans audited: {len(records)} ({summary['seconds_audited']}s)")
    print(
        f"local ASR heard lexical content in: {len(disagreements)} "
        f"({summary['asr_heard_lexical_share']:.1%})"
    )
    for record in disagreements[:20]:
        print(f"  script={record['text'][:26]!r}  heard={record['local_asr'][:40]!r}")
    frequent = collections.Counter(r["text"] for r in records)
    print("\nmost frequent span texts in the sample:")
    for text, count in frequent.most_common(10):
        print(f"  {count:>3}  {text[:44]}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
