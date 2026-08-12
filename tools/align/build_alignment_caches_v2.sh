#!/usr/bin/env bash
# Build every v2 alignment feature cache, one after another.
#
# Sequential on purpose: they all want the same GPU, and the encoder forward is
# not the bottleneck (RTF 0.00069) - running them in parallel would only make
# the audio decode contend for the same disk.
#
# `--allow-empty-text` everywhere, including the text caches: two of them carry a
# handful of rows whose script turned out to be pure vocalisation once stripped,
# and dropping those silently would move rows out of supervision without saying so.
set -euo pipefail

cd /d/Projects/jav-trans
export PYTHONIOENCODING=utf-8

MANIFESTS=${MANIFESTS:-agents/temp/20260811_160000_align-feature-caches/manifests}
OUT=datasets/train/align-features-v2
HOLDOUT=datasets/train/galgame-grok-ctc-teacher-20k-v1/rebuild/heldout_sources.jsonl

build() {
  local name=$1 group_field=$2 holdout=$3
  echo ""
  echo "=============================================================="
  echo "  $name  (group field: $group_field)"
  echo "=============================================================="
  local extra=()
  if [ "$holdout" = "yes" ]; then extra=(--holdout-composites "$HOLDOUT"); fi
  uv run python tools/align/build_alignment_features.py \
    --manifest "$MANIFESTS/$name.jsonl" \
    --output-dir "$OUT/$name" \
    --group-field "$group_field" \
    --allow-empty-text \
    --batch-size 16 \
    "${extra[@]}"
}

build galgame-teacher        group        yes
build galgame-recovered      group        yes
build galgame-vocal-blank    group        yes
build anime-sfw              group        no
build anime-sfw-vocal-blank  group        no
build anime-nsfw             source_group no
build anime-nsfw-vocal-blank group        no

echo ""
echo "=== all caches built ==="
for d in "$OUT"/*/; do
  printf '%-30s %s\n' "$(basename "$d")" "$(python -c "import json,sys;s=json.load(open(sys.argv[1],encoding='utf-8'));print(f\"{s['clips_written']} rows, {s['audio_hours']}h, {s['empty_target_clips']} blank, val={s['split']['val_rows']}\")" "$d/summary.json" 2>/dev/null || echo MISSING)"
done
