#!/usr/bin/env bash
# Train the v3 alignment head: CTC plus a three-class frame output.
#
# Usage: train_ctc_aligner_v3.sh <arm: S|F> <frame-class-loss-weight> <output-dir> \
#                                <frame-class-labels.npz> [extra flags...]
#
# What v3 changes against v2, and what it deliberately does not:
#
#   * **A frame classifier beside the CTC classifier.** One extra Linear(512, 3)
#     over the same trunk - ~1.5k parameters, one encoder pass, one checkpoint.
#     It exists because v2 had a single non-word class and moaning had to share
#     it with silence, so every dose of "moaning is blank" also taught the head
#     that breathy speech is blank. Three ablations falsified the dose; no
#     threshold reaches a class that does not exist.
#   * **The corrected vocalisation lexicon.** Both arms override the cached
#     target text, because the cache was built when `いい`/`いいっ`/`はいっ` were
#     classified as moaning and stripped - 370 fragments across the two stripped
#     corpora, none of which is a moan. The features are a function of the audio,
#     so only the text column is stale.
#   * **Nothing about the encoder, the geometry, or the vocabulary policy.**
#
# The two arms differ in exactly one thing: what CTC is asked to emit.
#
#   S  stripped targets, as v2 had them. Moaning stays blank on the CTC side and
#      the frame head carries the three-way distinction. The chunker's pause
#      reading is unchanged, so nothing downstream has to move with it.
#   F  full-script targets. Moaning is aligned as text, so blank goes back to
#      meaning silence - cleaner in principle, but the chunker then has to read
#      the speech posterior instead of blank runs or it stops cutting inside
#      moaning passages.
#
# v2's own configuration is reproduced otherwise: acoustic-only vocabulary,
# blank rows included under a 0.30 per-domain cap, checkpoint selected on
# anime-nsfw val loss, 12 epochs at batch 32. Roughly 40 minutes per arm on a
# 4060 Ti.
set -euo pipefail

cd /d/Projects/jav-trans
export PYTHONIOENCODING=utf-8

ARM=$1
LAMBDA=$2
OUT=$3
LABELS=$4
shift 4

FEATURES=datasets/train/align-features-v2
STRIPPED=agents/temp/20260903_093536_p0-stripped-targets
FULL="$STRIPPED/full-script-targets"

case "$ARM" in
  S)
    # The P0-corrected strip. Both manifest views per cache, because the cache
    # holds the rows that became blank as well as the rows that kept text.
    OVERRIDES=(
      --text-override-manifest "$STRIPPED/nsfw/stripped_text_manifest.jsonl"
      --text-override-cache    anime-nsfw
      --text-override-manifest "$STRIPPED/nsfw/stripped_blank_manifest.jsonl"
      --text-override-cache    anime-nsfw
      --text-override-manifest "$STRIPPED/galgame-recovered/stripped_text_manifest.jsonl"
      --text-override-cache    galgame-recovered
      --text-override-manifest "$STRIPPED/galgame-recovered/stripped_blank_manifest.jsonl"
      --text-override-cache    galgame-recovered
    )
    ;;
  F)
    # Full script everywhere the moans were removed - including the blank-only
    # caches, whose whole target was deleted rather than trimmed. Leaving those
    # empty would keep the largest single source of "vocalisation is blank" in
    # place and make the arm test nothing.
    OVERRIDES=(
      --text-override-manifest "$FULL/anime-nsfw.jsonl"
      --text-override-cache    anime-nsfw
      --text-override-manifest "$FULL/galgame-recovered.jsonl"
      --text-override-cache    galgame-recovered
      --text-override-manifest "$FULL/vocal-blank.jsonl"
      --text-override-cache    galgame-vocal-blank
      --text-override-manifest "$FULL/vocal-blank.jsonl"
      --text-override-cache    anime-sfw-vocal-blank
      --text-override-manifest "$FULL/vocal-blank.jsonl"
      --text-override-cache    anime-nsfw-vocal-blank
    )
    ;;
  *)
    echo "arm must be S (stripped targets) or F (full-script targets)" >&2
    exit 2
    ;;
esac

uv run python tools/align/train_ctc_aligner.py \
  --cache-dir "$FEATURES/galgame-teacher"         --cache-domain galgame \
  --cache-dir "$FEATURES/galgame-recovered"       --cache-domain galgame \
  --cache-dir "$FEATURES/galgame-vocal-blank"     --cache-domain galgame \
  --cache-dir "$FEATURES/anime-sfw"               --cache-domain anime-sfw \
  --cache-dir "$FEATURES/anime-sfw-vocal-blank"   --cache-domain anime-sfw \
  --cache-dir "$FEATURES/anime-nsfw"              --cache-domain anime-nsfw \
  --cache-dir "$FEATURES/anime-nsfw-vocal-blank"  --cache-domain anime-nsfw \
  "${OVERRIDES[@]}" \
  --include-empty-targets \
  --max-empty-train-fraction 0.30 \
  --acoustic-targets \
  --select-domain anime-nsfw \
  --frame-class-labels "$LABELS" \
  --frame-class-loss-weight "$LAMBDA" \
  --epochs 12 \
  --batch-size 32 \
  --output-dir "$OUT" \
  "$@"
