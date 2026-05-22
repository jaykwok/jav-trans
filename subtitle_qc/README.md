# Subtitle QC / 字幕质检

`subtitle_qc` 是本地字幕人工质检工具，用来比较不同 VAD、ASR 后端、历史字幕和外部参考字幕。默认不会切视频，而是生成可直接 seek 原视频的 HTML 报告，方便快速检查 ASR 幻觉、转写错误、时间轴切分、翻译问题、漏语音和多余语音。

单视频报告默认写到对应视频目录：

```text
video/<video-stem>/subtitle_qc/
```

批量 VAD 矩阵只把运行日志和中间件放在 `agents/temp/subtitle_qc/<task-name>/`。最终 SRT/JSON、质量报告和 reference eval 都按视频归档到 `video/<video-stem>/`；跨视频 summary 写到 `video/subtitle_qc/<task-name>/`。

## 单视频报告

为一个视频生成全部字幕质检报告：

```bash
uv run --no-sync python subtitle_qc/generate.py --video video/<video-stem>.mp4
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
  --video video/<video-stem>.mp4 \
  --history-subtitles \
  video/<video-stem>/<video-stem>.whisperseg_adaptive.srt \
  video/<video-stem>/<video-stem>.fusion_lite.srt \
  video/<video-stem>/<video-stem>.whisperseg_adaptive.bilingual.json
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
  --video video/<video-stem>.mp4 \
  --input anime=video/<video-stem>/<video-stem>.whisperseg_adaptive.srt \
  --input qwen=video/<video-stem>/<video-stem>.fusion_lite.srt \
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

默认会发现 `video/reference/` 下有参考字幕的视频，跑两个 VAD 后端，并把日文 SRT 写到：

```text
video/<video-stem>/<video-stem>.<asr-label>_<vad-label>.srt
```

默认 VAD 集合：

- `whisperseg_adaptive=whisperseg-adaptive`
- `fusion_lite=fusion_lite`

运行日志和中间件保存在：

```text
agents/temp/subtitle_qc/<task-name>/
```

每个视频的质量报告和 reference eval 保存在：

```text
video/<video-stem>/subtitle_qc/<task-name>/
```

跨视频聚合 summary 保存在：

```text
video/subtitle_qc/<task-name>/
```

限制到一个或多个视频：

```bash
uv run --no-sync python subtitle_qc/compare_vad.py \
  --asr-backend whisper-ja-anime-v0.3 \
  --video <video-stem>
```

自定义 VAD 子集，参数可写后端名或 `label=backend`：

```bash
uv run --no-sync python subtitle_qc/compare_vad.py \
  --asr-backend qwen3-asr-1.7b \
  --vad adaptive=whisperseg-adaptive \
  --vad lite=fusion_lite
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

## VAD 参数调参

对同一批视频跑一组 VAD 参数 trial，并用全片内容审阅指标排序：

```bash
uv run --no-sync python subtitle_qc/tune_vad.py \
  --asr-backend whisper-ja-1.5b \
  --video REAL-988 \
  --task-name real988_vad_tuning
```

默认 trial 会覆盖 `whisperseg-adaptive` 和 `fusion_lite` 的小网格。也可以显式指定 trial，格式是 `label:backend[:KEY=VALUE,...]`：

```bash
uv run --no-sync python subtitle_qc/tune_vad.py \
  --asr-backend whisper-ja-1.5b \
  --video REAL-988 \
  --task-name real988_custom_tuning \
  --trial lite_base:fusion_lite \
  --trial lite_gate08:fusion_lite:FUSION_VAD_MIN_GATE_OVERLAP_RATIO=0.08 \
  --trial adaptive_t38:whisperseg-adaptive:WHISPERSEG_THRESHOLD=0.38
```

脚本会复用 `compare_vad.py` 的归档逻辑，输出候选字幕到：

```text
video/<video-stem>/<video-stem>.<trial-label>.srt
```

调参汇总写到：

```text
video/subtitle_qc/<task-name>/tuning_summary.md
video/subtitle_qc/<task-name>/tuning_summary.csv
video/subtitle_qc/<task-name>/tuning_summary.json
```

每个视频还会生成全片内容审阅：

```text
video/<video-stem>/subtitle_qc/<task-name>/content_review/
```

调参同样默认检查 WhisperSeg ONNX CUDA，避免意外 CPU fallback。只有确认接受慢速 CPU 跑法时才传 `--allow-whisperseg-cpu`。

## 全片内容审阅

对同一视频目录下的候选字幕做全片内容审阅，不抽样：

```bash
uv run --no-sync python subtitle_qc/review_content.py --video <video-stem>
```

默认自动发现两种当前 VAD 候选：

- `whisperseg_adaptive`
- `fusion_lite`

自动发现使用正向 allowlist：文件后缀必须精确匹配上面两种 label，或以 `_<label>` 结尾，例如 `<video-stem>.whisper_ja_1_5b_fusion_lite.srt`。未知后缀会直接报错，避免旧实验产物被误归类。

同一 label 同时存在 `.srt` 和 `.bilingual.json` 时，默认优先读取 `.srt`；传 `--all-candidates` 仍只包含 allowlist 能识别的字幕文件。调参 trial 或任意历史字幕请显式指定候选：

```bash
uv run --no-sync python subtitle_qc/review_content.py \
  --video <video-stem> \
  --candidate lite=video/<video-stem>/<video-stem>.fusion_lite.srt \
  --candidate adaptive=video/<video-stem>/<video-stem>.whisperseg_adaptive.srt
```

输出目录默认是 `video/<video-stem>/subtitle_qc/content_review/`，包括：

- `content_review_summary.md`：启发式汇总和候选排序。
- `content_review_metrics.csv`：全片 cue 数、时长、gap、字符数和问题计数。
- `content_review_issues.csv`：疑似噪声、重复、断句、过长/过快等问题 cue。
- `content_review_full_text.md`：每个候选的全片逐句文本。
- `content_review_window_compare.md`：按时间窗口展开的候选对照。
- `content_review.json`：机器可读结果。

## 外部参考字幕评测

用外部日文参考字幕评测候选字幕：

```bash
uv run --no-sync python subtitle_qc/evaluate_reference.py --video video/<video-stem>.mp4
```

参考字幕放在：

```text
video/reference/
```

下载参考字幕默认视为弱参考，除非来源明确可靠。SubtitleCat 文件是用户上传或机器翻译资产，不应当默认视为官方制作字幕。评测器会自动发现 `video/reference/**/<video-stem>*`，并与 `video/<video-stem>/` 下的候选 SRT/JSON 对比。ASS 参考输入也支持，但非日文 ASS 不应放进 `video/reference/`。

显式指定参考和候选：

```bash
uv run --no-sync python subtitle_qc/evaluate_reference.py \
  --video video/<video-stem>.mp4 \
  --reference video/reference/weak/subtitlecat/<video-stem>.subtitlecat.ja.srt \
  --candidate adaptive=video/<video-stem>/<video-stem>.whisperseg_adaptive.srt \
  --candidate lite=video/<video-stem>/<video-stem>.fusion_lite.srt
```

评测按每条参考 cue 的时间窗口聚合候选日文文本再打分，比一对一 cue 匹配更适合外部字幕，因为下载字幕经常把多句对白合成一个长 cue。输出文件：

- `reference_eval_summary.md`：紧凑排名表。
- `reference_eval_metrics.csv`：每个候选的聚合指标。
- `reference_eval_worst_cues.csv`：相似度最低的参考窗口，供人工复核。
- `reference_eval.json`：完整机器可读指标和 worst cue payload。

## 可选切片

切出每个 review item 的短视频片段：

```bash
uv run --no-sync python subtitle_qc/generate.py --video video/<video-stem>.mp4 --make-clips
uv run --no-sync python subtitle_qc/generate.py --video video/<video-stem>.mp4 --make-compilation
```

切片默认关闭，因为它会变慢并占用较多磁盘空间。
