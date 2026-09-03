#!/usr/bin/env bash
# Build the full three-class frame label set over the v2 feature caches.
#
# Usage: build_frame_class_labels_v1.sh <output-dir> [extra flags...]
#
# The label sources and why each cache is where it is:
#
#   L1  anime-nsfw       - the only source of new information. The full script,
#                          moans included, is force-aligned with the punctuated
#                          general head and each punctuation block is put to the
#                          shared vocalisation lexicon. Carries the quality gate.
#   L4  galgame-recovered - same construction over text a local decode produced,
#                          so it is the least trustworthy of the two.
#   L2  *-vocal-blank    - the script already says the whole clip is vocalisation,
#                          so energy alone splits loud from quiet and nothing can
#                          be mislabelled as speech.
#   L3  galgame-teacher, anime-sfw - Grok word islands are speech; quiet gaps far
#                          from a word are silence; the voiced-wordless middle
#                          stays ignored, because on SFW anime it is breath, SFX
#                          and BGM and energy is not a moan detector.
#
# **Both manifest views per teacher cache.** The crop and full views of one
# teacher run share `source_id` but not `audio_id`, and the audio path is looked
# up by `audio_id`. Naming only the crop manifest leaves every full row without
# audio and therefore without a single silence label - measured, that was 9
# silence frames where both views give 7,271, and the run reported success.
set -euo pipefail

cd /d/Projects/jav-trans
export PYTHONIOENCODING=utf-8

GALGAME=datasets/train/galgame-grok-ctc-teacher-20k-v1
SFW=agents/temp/20260811_140000_anime-text-corpus/sfw_full
STRIPPED=agents/temp/20260903_093536_p0-stripped-targets
OUT=$1
shift

uv run python tools/align/build_frame_class_labels.py \
  --out-dir "$OUT" \
  --l1-manifest       "$STRIPPED/nsfw/stripped_text_manifest.jsonl" \
  --l1-blank-manifest "$STRIPPED/nsfw/stripped_blank_manifest.jsonl" \
  --l4-manifest       "$STRIPPED/galgame-recovered/stripped_text_manifest.jsonl" \
  --l4-blank-manifest "$STRIPPED/galgame-recovered/stripped_blank_manifest.jsonl" \
  --teacher-results  "$GALGAME/teacher/results.jsonl" \
  --teacher-manifest "$GALGAME/compiled/ctc_manifest.jsonl" \
  --teacher-cache    galgame-teacher \
  --teacher-source-manifest "$GALGAME/compiled/ctc_manifest.jsonl,$GALGAME/compiled/full_manifest.jsonl" \
  --teacher-results  "$SFW/results.jsonl" \
  --teacher-manifest "$SFW/ctc_manifest.jsonl" \
  --teacher-cache    anime-sfw \
  --teacher-source-manifest "$SFW/ctc_manifest.jsonl,$SFW/full_manifest.jsonl" \
  "$@"
