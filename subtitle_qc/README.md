# Subtitle QC / 字幕质检

`subtitle_qc` 是本地字幕人工质检工具，用来比较不同 VAD、ASR 后端、历史字幕和外部参考字幕。默认不会切视频，而是生成可直接 seek 原视频的 HTML 报告，方便快速检查 ASR 幻觉、转写错误、时间轴切分、翻译问题、漏语音和多余语音。

单视频报告默认写到对应视频目录：

```text
video/<video-stem>/subtitle_qc/
```

批量 VAD 矩阵仍把运行日志、临时中间件和聚合评测放在 `agents/temp/subtitle_qc/<task-name>/`，最终 SRT/JSON 会按视频归档到 `video/<video-stem>/`。

## 单视频报告

为一个视频生成全部字幕质检报告：

```bash
uv run --no-sync python subtitle_qc/generate.py --video video/MKMP-577.mp4
```

生成文件：

- `japanese_transcript_compare.html`：不同 VAD 模式的日文转写对比。
- `japanese_wordline_report.html`：逐 cue 日文报告，包含可用的词级时间戳。
- `translation_compare.html`：不同 VAD 模式的日文/中文翻译对比。
- `review_index.html`：人工 review 入口，内置原视频播放器、标签和 JSON 导出。
- `review_items.json`：机器可读的 review 候选项和标签。
- `summary.json`：输入路径和高层计数。

`review_index.html` 会把标签保存在浏览器 `localStorage`，并可导出为 `<video-stem>.subtitle_qc_labels.json`。内置问题标签包括 ASR 幻觉、转写错误、翻译错误、时间轴错误、VAD 边界错误、漏语音和多余语音。

## 历史字幕对比

比较同一视频的历史日文字幕文件：

```bash
uv run --no-sync python subtitle_qc/generate.py \
  --video video/MKMP-577.mp4 \
  --history-subtitles \
  video/MKMP-577/MKMP-577.whisperseg_adaptive.srt \
  video/MKMP-577/MKMP-577.fusion_lite.srt \
  video/MKMP-577/MKMP-577.whisperseg_adaptive.bilingual.json
```

只比较历史字幕、不重新生成 VAD/翻译对比报告时，加 `--history-only`。

历史对比支持 `.srt` 和 `.json`。SRT 会保留含假名的日文行，其余行作为翻译/上下文；JSON 支持常见 cue 数组字段，如 `blocks`、`segments`、`cues`、`subtitles`，以及 `start`、`end`、`ja_text`、`text`、`zh_text` 等字段。

输出文件：

- `history_japanese_compare.html`：对齐后的日文字幕差异。
- `history_review_items.json`：机器可读的疑似差异 cue 列表。

## ASR 后端对比

比较不同 ASR 后端生成的日文转写：

```bash
uv run --no-sync python subtitle_qc/compare_asr_backends.py \
  --video video/MKMP-577.mp4 \
  --input anime=video/MKMP-577/MKMP-577.whisperseg_adaptive.srt \
  --input qwen=video/MKMP-577/MKMP-577.fusion_lite.srt \
  --input whisper15=video/MKMP-577/MKMP-577.fusion_lite_boost.srt \
  --base anime
```

ASR 后端对比接受 `.srt` 和 `.json`，按时间重叠和日文文本相似度对齐 cue，输出到 `video/<video-stem>/subtitle_qc/`：

- `asr_backend_japanese_compare.html`：逐句日文转写对比。
- `asr_backend_review_items.json`：疑似不同的 ASR cue 列表。
- `asr_backend_summary.json`：输入计数和输出路径。

## VAD 矩阵

使用指定 ASR 后端跑 VAD 矩阵：

```bash
uv run --no-sync python subtitle_qc/compare_vad.py --asr-backend qwen3-asr-1.7b
```

默认会发现 `video/reference/` 下有参考字幕的视频，跑四个 VAD 后端，并把日文 SRT 写到：

```text
video/<video-stem>/<video-stem>.<asr-label>_<vad-label>.srt
```

默认 VAD 集合：

- `whisperseg_adaptive=whisperseg-adaptive`
- `fusion_lite=fusion_lite`
- `fusion_lite_boost=fusion_lite_boost`
- `fusion_lite_sigmoid=fusion_lite_sigmoid`

运行日志、质量报告副本和聚合评测文件保存在：

```text
agents/temp/subtitle_qc/<task-name>/
```

限制到一个或多个视频：

```bash
uv run --no-sync python subtitle_qc/compare_vad.py \
  --asr-backend whisper-ja-anime-v0.3 \
  --video MKMP-577
```

自定义 VAD 子集，参数可写后端名或 `label=backend`：

```bash
uv run --no-sync python subtitle_qc/compare_vad.py \
  --asr-backend qwen3-asr-1.7b \
  --vad adaptive=whisperseg-adaptive \
  --vad boost=fusion_lite_boost
```

脚本默认跳过已存在的输出 SRT；传 `--force` 可强制重跑。旧的平铺输出 `video/<video-stem>.<label>.srt` 仍可读取；只有确实需要继续写旧布局时才传 `--flat-video-output`。

只评测已有字幕、不重新跑 ASR：

```bash
uv run --no-sync python subtitle_qc/compare_vad.py \
  --asr-backend qwen3-asr-1.7b \
  --task-name qwen-reference-vad-compare \
  --evaluate-only
```

脚本在跑 ASR 前会检查 WhisperSeg ONNX 是否使用 CUDA，避免 CPU fallback 跑完整视频耗时数小时。只有小型 smoke test 才建议传 `--allow-whisperseg-cpu`。

## 外部参考字幕评测

用外部日文参考字幕评测候选字幕：

```bash
uv run --no-sync python subtitle_qc/evaluate_reference.py --video video/MKMP-577.mp4
```

参考字幕放在：

```text
video/reference/
```

下载参考字幕默认视为弱参考，除非来源明确可靠。SubtitleCat 文件是用户上传或机器翻译资产，不应当默认视为官方制作字幕。评测器会自动发现 `video/reference/**/<video-stem>*`，并与 `video/<video-stem>/` 下的候选 SRT/JSON 对比。ASS 参考输入也支持，但非日文 ASS 不应放进 `video/reference/`。

显式指定参考和候选：

```bash
uv run --no-sync python subtitle_qc/evaluate_reference.py \
  --video video/MKMP-577.mp4 \
  --reference video/reference/weak/subtitlecat/MKMP-577.subtitlecat.ja.srt \
  --candidate adaptive=video/MKMP-577/MKMP-577.whisperseg_adaptive.srt \
  --candidate boost=video/MKMP-577/MKMP-577.fusion_lite_boost.srt
```

评测按每条参考 cue 的时间窗口聚合候选日文文本再打分，比一对一 cue 匹配更适合外部字幕，因为下载字幕经常把多句对白合成一个长 cue。输出文件：

- `reference_eval_summary.md`：紧凑排名表。
- `reference_eval_metrics.csv`：每个候选的聚合指标。
- `reference_eval_worst_cues.csv`：相似度最低的参考窗口，供人工复核。
- `reference_eval.json`：完整机器可读指标和 worst cue payload。

## 可选切片

切出每个 review item 的短视频片段：

```bash
uv run --no-sync python subtitle_qc/generate.py --video video/MKMP-577.mp4 --make-clips
uv run --no-sync python subtitle_qc/generate.py --video video/MKMP-577.mp4 --make-compilation
```

切片默认关闭，因为它会变慢并占用较多磁盘空间。
