# Subtitle QC

Local subtitle review tools for comparing ASR transcripts, translation output, and subtitle timing across VAD variants.

The default workflow does not cut video. It generates HTML reports that seek into the original video so a reviewer can quickly inspect likely hallucinations, transcription mistakes, timing splits, and translation issues.

## Usage

Generate all reports for one video stem:

```bash
uv run --no-sync python subtitle_qc/generate.py --video video/MKMP-577.mp4
```

Outputs are written under:

```text
subtitle_qc/output/<video-stem>/
```

Generated files:

- `japanese_transcript_compare.html`: Japanese transcript comparison across VAD modes.
- `japanese_wordline_report.html`: cue-by-cue Japanese report with available word timestamps.
- `translation_compare.html`: Japanese and Chinese translation comparison across VAD modes.
- `review_index.html`: prioritized human review page with one seekable video player, local labels, and JSON export.
- `review_items.json`: machine-readable review candidates and labels.
- `summary.json`: input paths and high-level counts.

`review_index.html` stores labels in browser `localStorage` and can export them as
`<video-stem>.subtitle_qc_labels.json`. The built-in issue tags are focused on
human supervision: ASR hallucination, transcription error, translation error,
timing error, VAD boundary error, missing speech, and extra speech.

Compare historical Japanese subtitle files for the same video:

```bash
uv run --no-sync python subtitle_qc/generate.py \
  --video video/MKMP-577.mp4 \
  --history-subtitles \
  video/MKMP-577/MKMP-577.whisperseg_adaptive.srt \
  video/MKMP-577/MKMP-577.fusion_lite.srt \
  video/MKMP-577/MKMP-577.whisperseg_adaptive.bilingual.json
```

Use `--history-only` when you only want to compare historical subtitle files and
do not want to regenerate the VAD/translation comparison reports.

Historical comparison accepts `.srt` and `.json`. SRT input keeps Japanese lines
when kana is present and treats the remaining lines as translation/context.
JSON input supports common cue arrays such as `blocks`, `segments`, `cues`,
`subtitles`, and fields like `start`, `end`, `ja_text`, `text`, and `zh_text`.
It writes:

- `history_japanese_compare.html`: aligned Japanese subtitle differences.
- `history_review_items.json`: machine-readable likely-different cue list.

Compare Japanese transcripts from different ASR backends:

```bash
uv run --no-sync python subtitle_qc/compare_asr_backends.py \
  --video video/MKMP-577.mp4 \
  --input anime=video/MKMP-577/MKMP-577.whisperseg_adaptive.srt \
  --input qwen=video/MKMP-577/MKMP-577.fusion_lite.srt \
  --input whisper15=video/MKMP-577/MKMP-577.fusion_lite_boost.srt \
  --base anime
```

The ASR backend comparison accepts `.srt` and `.json`, aligns cues by time
overlap plus Japanese text similarity, and writes:

- `asr_backend_japanese_compare.html`: sentence-by-sentence Japanese transcript comparison.
- `asr_backend_review_items.json`: likely-different ASR cue list.
- `asr_backend_summary.json`: input counts and output paths.

Run a VAD matrix with a selected ASR backend:

```bash
uv run --no-sync python subtitle_qc/compare_vad.py --asr-backend qwen3-asr-1.7b
```

By default this auto-discovers videos that have references under
`video/reference/`, runs four VAD backends, and writes Japanese-only SRTs to
`video/<video-stem>/<video-stem>.<asr-label>_<vad-label>.srt`. The default VAD set is:

- `whisperseg_adaptive=whisperseg-adaptive`
- `fusion_lite=fusion_lite`
- `fusion_lite_boost=fusion_lite_boost`
- `fusion_lite_sigmoid=fusion_lite_sigmoid`

Outputs and aggregate evaluation files are kept under:

```text
agents/temp/subtitle_qc/<task-name>/
```

Use one or more `--video` arguments to limit the run:

```bash
uv run --no-sync python subtitle_qc/compare_vad.py \
  --asr-backend whisper-ja-anime-v0.3 \
  --video MKMP-577
```

Use repeatable `--vad` arguments to compare a custom VAD subset. Each value can
be a backend name or `label=backend`:

```bash
uv run --no-sync python subtitle_qc/compare_vad.py \
  --asr-backend qwen3-asr-1.7b \
  --vad adaptive=whisperseg-adaptive \
  --vad boost=fusion_lite_boost
```

The script skips existing output SRTs unless `--force` is passed. It can still
read older flat outputs such as `video/<video-stem>.<label>.srt`; pass
`--flat-video-output` only if you need to keep writing that older layout. To
score already generated subtitles without running ASR again:

```bash
uv run --no-sync python subtitle_qc/compare_vad.py \
  --asr-backend qwen3-asr-1.7b \
  --task-name qwen-reference-vad-compare \
  --evaluate-only
```

When it is going to run ASR, the script checks that WhisperSeg ONNX is using
CUDA before starting, because CPU fallback can make full-video VAD comparisons
take hours. Pass `--allow-whisperseg-cpu` only for small smoke tests.

Evaluate generated Japanese subtitles against an external Japanese reference:

```bash
uv run --no-sync python subtitle_qc/evaluate_reference.py --video video/MKMP-577.mp4
```

Reference subtitles live under:

```text
video/reference/
```

Treat downloaded references as weak external references unless their source is
known. SubtitleCat files are user-uploaded or machine-translated assets and are
not assumed to be official production subtitles. By default the evaluator
auto-discovers `video/reference/**/<video-stem>*` and compares it with known VAD
outputs such as `video/<video-stem>/<video-stem>.fusion_lite.srt` and
`video/<video-stem>/<video-stem>.whisperseg_adaptive.srt`. It reuses the SRT/JSON cue
parsers, timestamp helpers, Japanese line splitting, and text-similarity logic
from `generate.py`. ASS reference input is also supported for downloaded anime
subtitle files, but non-Japanese ASS files should not be placed in
`video/reference/`.

You can pass explicit candidates:

```bash
uv run --no-sync python subtitle_qc/evaluate_reference.py \
  --video video/MKMP-577.mp4 \
  --reference video/reference/weak/subtitlecat/MKMP-577.subtitlecat.ja.srt \
  --candidate adaptive=video/MKMP-577/MKMP-577.whisperseg_adaptive.srt \
  --candidate boost=video/MKMP-577/MKMP-577.fusion_lite_boost.srt
```

The reference evaluator aligns by each reference cue's time window and joins all
candidate Japanese text inside that window before scoring. This is more useful
for external subtitles than one-to-one cue matching because downloaded
references often merge several spoken lines into one long cue. It writes:

- `reference_eval_summary.md`: compact ranking table.
- `reference_eval_metrics.csv`: per-candidate aggregate metrics.
- `reference_eval_worst_cues.csv`: lowest-similarity reference windows for manual review.
- `reference_eval.json`: full machine-readable metrics and worst cue payload.

Optional clip generation:

```bash
uv run --no-sync python subtitle_qc/generate.py --video video/MKMP-577.mp4 --make-clips
uv run --no-sync python subtitle_qc/generate.py --video video/MKMP-577.mp4 --make-compilation
```

Clip generation is intentionally opt-in because it can be slow and can consume substantial disk space.
