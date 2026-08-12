#!/usr/bin/env bash
# Retrain the CTC alignment head on the v2 corpora.
#
# Usage: train.sh <output-dir> [extra flags...]
#
# What changed against the shipped head, and why each change is here:
#
#   * **Three domains instead of one.** The shipped head saw galgame full clips
#     only. It now also sees 61h of anime SFW speech (Grok-timed) and 47h of
#     anime NSFW clips whose targets have the vocalisation stripped out.
#   * **Blank-only rows.** 8,765 clips whose script is nothing but vocalisation,
#     trained as explicit CTC blank. Each blank cache carries the SAME
#     `--cache-domain` as text from the same corpus, because the per-domain cap
#     computes `0.30 * nonempty / 0.70` and a domain with no text rows keeps zero
#     blank rows - every one of them would be dropped without a word.
#   * **`--select-domain anime-nsfw`.** The pooled val loss is dominated by the
#     largest cache, which is the SFW speech - the arm least like the real
#     domain. NSFW mixed clips are the closest proxy for a JAV soundtrack.
#
# Frame teachers are declared per cache: the two Grok archives cover
# `galgame-teacher` and `anime-sfw` and nothing else. The recovered and
# blank-only caches have no word timings by nature and carry CTC text loss only.
#
# `--acoustic-targets` and `--max-empty-train-fraction` are deliberately NOT
# fixed here - the first run set both at once and the acceptance pass could not
# say which of them moved the operating point. They are passed per arm.
set -euo pipefail

cd /d/Projects/jav-trans
export PYTHONIOENCODING=utf-8

FEATURES=datasets/train/align-features-v2
GALGAME=datasets/train/galgame-grok-ctc-teacher-20k-v1
SFW=agents/temp/20260811_140000_anime-text-corpus/sfw_full
OUT=$1
shift

uv run python tools/align/train_ctc_aligner.py \
  --cache-dir "$FEATURES/galgame-teacher"         --cache-domain galgame \
  --cache-dir "$FEATURES/galgame-recovered"       --cache-domain galgame \
  --cache-dir "$FEATURES/galgame-vocal-blank"     --cache-domain galgame \
  --cache-dir "$FEATURES/anime-sfw"               --cache-domain anime-sfw \
  --cache-dir "$FEATURES/anime-sfw-vocal-blank"   --cache-domain anime-sfw \
  --cache-dir "$FEATURES/anime-nsfw"              --cache-domain anime-nsfw \
  --cache-dir "$FEATURES/anime-nsfw-vocal-blank"  --cache-domain anime-nsfw \
  --include-empty-targets \
  --select-domain anime-nsfw \
  --frame-teacher-results  "$GALGAME/teacher/results.jsonl" \
  --frame-teacher-manifest "$GALGAME/compiled/ctc_manifest.jsonl" \
  --frame-teacher-cache    galgame-teacher \
  --frame-teacher-results  "$SFW/results.jsonl" \
  --frame-teacher-manifest "$SFW/ctc_manifest.jsonl" \
  --frame-teacher-cache    anime-sfw \
  --frame-loss-weight 0.05 \
  --epochs 12 \
  --batch-size 32 \
  --output-dir "$OUT" \
  "$@"
