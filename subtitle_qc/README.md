# Subtitle QC

Local subtitle review tools for comparing ASR transcripts, translation output, and subtitle timing across VAD variants.

The default workflow does not cut video. It generates HTML reports that seek into the original video so a reviewer can quickly inspect likely hallucinations, transcription mistakes, timing splits, and translation issues.

## Usage

Generate all reports for one video stem:

```bash
.venv/bin/python subtitle_qc/generate.py --video video/MKMP-577.mp4
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
.venv/bin/python subtitle_qc/generate.py \
  --video video/MKMP-577.mp4 \
  --history-subtitles \
  video/MKMP-577.whisperseg_adaptive.srt \
  video/MKMP-577.fusion_lite.srt \
  video/MKMP-577.whisperseg_adaptive.bilingual.json
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
.venv/bin/python subtitle_qc/compare_asr_backends.py \
  --video video/MKMP-577.mp4 \
  --input anime=video/MKMP-577.whisperseg_adaptive.srt \
  --input qwen=video/MKMP-577.fusion_lite.srt \
  --input whisper15=video/MKMP-577.fusion_lite_boost.srt \
  --base anime
```

The ASR backend comparison accepts `.srt` and `.json`, aligns cues by time
overlap plus Japanese text similarity, and writes:

- `asr_backend_japanese_compare.html`: sentence-by-sentence Japanese transcript comparison.
- `asr_backend_review_items.json`: likely-different ASR cue list.
- `asr_backend_summary.json`: input counts and output paths.

Optional clip generation:

```bash
.venv/bin/python subtitle_qc/generate.py --video video/MKMP-577.mp4 --make-clips
.venv/bin/python subtitle_qc/generate.py --video video/MKMP-577.mp4 --make-compilation
```

Clip generation is intentionally opt-in because it can be slow and can consume substantial disk space.
