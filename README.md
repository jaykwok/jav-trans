# JAVTrans

JAVTrans 是一个本地JAV字幕生成工具，面向 Windows + NVIDIA 显卡，也可在 WSL2 / Linux 下源码运行。它把视频处理成日文字幕、中文字幕或中日双语字幕，并把音频准备、SpeechBoundary-JA、Boundary Refiner、可选 Pre-ASR CueQC、Qwen ASR、字幕时间轴、LLM 翻译和质量报告串成一条本地优先的流水线。

项目目标：本地完成视频、音频、边界切分、ASR 和字幕时间轴重计算；LLM 只负责翻译、术语一致和口吻连贯，不负责脑补剧情或修正 ASR 误听。

致谢：[WhisperJAV](https://github.com/a63n/WhisperJAV) 为本项目早期路线提供了重要参考。

---

## 项目背景

本项目的边界系统不是传统 VAD。目标不是单纯判断“有没有人声”，而是生成适合字幕和 ASR 的 speech-core chunk：尽量接近一句台词一个 chunk，避免把 BGM、环境声、贴连短句或长独白粗暴混成同一种情况。

当前设计把职责拆开：

- SpeechBoundary-JA Scorer 只做帧级 `speech/split`。
- Boundary Refiner 只修 chunk 两端。
- Pre-ASR CueQC 只做 ASR 前 `keep/drop` 路由。
- 字幕 layout 只处理显示规则，不反向修改 ASR chunk 语义。

这样做是为了避免一个模型同时承担“找语音、切句、删噪声、修边界、做字幕排版”。设计演进、实验记录、失败路线和更新记录都放在 [HISTORY.md](HISTORY.md)。

---

## 快速开始

### Release 版

GitHub Releases 只用于发布 source code、release notes 和版本说明，不上传大型 Windows `.7z` 运行包。Windows 运行包由维护者按需本地打包成单个 `.7z`，通过网盘或其他外部分发渠道提供；拿到包后解压运行：

```text
JAVTrans.exe
```

Windows 打包版默认应内置 ASR 模型、`ffmpeg` / `ffprobe`，以及 repo-id registry 使用的 SpeechBoundary-JA / Boundary Refiner / CueQC checkpoint。

Release 版不内置 Microsoft Edge WebView2 Runtime。大多数 Windows 10/11 已自带；如果无法打开窗口，请安装 [Microsoft Edge WebView2 Evergreen Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/)。

### 源码运行

推荐环境：

- Windows 10/11 或 WSL2 / Linux。
- NVIDIA 独立显卡和较新的驱动。
- Python 3.13+。
- FFmpeg，并确保命令行能直接执行 `ffmpeg`。
- Git。

安装：

```powershell
git clone https://github.com/jaykwok/jav-trans.git
cd jav-trans

uv venv
uv pip install --upgrade pip
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，填写翻译配置：

```env
API_KEY=你的翻译_API_KEY
OPENAI_COMPATIBILITY_BASE_URL=https://api.deepseek.com
LLM_MODEL_NAME=deepseek-v4-flash
LLM_API_FORMAT=chat
LLM_REASONING_EFFORT=xhigh
TARGET_LANG=简体中文
```

国内网络下载 Hugging Face 模型较慢时可设置：

```env
HF_ENDPOINT=https://hf-mirror.com
```

启动网页控制台：

```powershell
$env:PYTHONIOENCODING="utf-8"
uv run --no-sync python launcher.py
```

默认地址为 `http://127.0.0.1:17321`。

Linux / WSL2 下如果只启动浏览器版 Web 服务，也可以直接运行：

```bash
PYTHONIOENCODING=utf-8 PYTHONPATH=src uv run python -m uvicorn web.app:create_app --factory --host 127.0.0.1 --port 17321
```

Web 提交是否使用 CUDA 取决于后端服务进程是否能看到 GPU，而不是浏览器本身。完整 SpeechBoundary-JA / ASR smoke 应确认日志中出现 `cuda_available=True`、`device=cuda:0` 或 `actual_device=cuda`。

---

## 使用流程

1. 打开网页控制台。
2. 选择视频文件。
3. 选择字幕模式、ASR 后端和翻译设置。
4. 提交任务。
5. 在输出目录查看 SRT、质量报告和日志。

勾选“不翻译（仅日文字幕）”时，流水线仍会执行边界规划、可选 Pre-ASR CueQC、ASR 和 Boundary chunk 字幕时间轴生成，但跳过 LLM 翻译，最终输出 `<视频名>.ja.srt`。这是验证本地边界 / ASR / 字幕时间轴链路的推荐 smoke 模式。

---

## 完整工作流

```text
视频输入
  -> 任务上下文 / 配置解析
  -> 音频抽取与标准化
  -> SpeechBoundary-JA feature extraction
     - Qwen ASR repo 对应的 frozen PTM/encoder frame features
     - MFCC / timing numeric features
  -> SpeechBoundary-JA Scorer v5
     - dense frame speech_prob
     - dense frame split_boundary_prob
  -> Scorer decoder
     - speech hysteresis 生成 coarse speech islands
     - topographic split peak / acoustic valley 生成 primary_cut_candidates
     - weak_cut_candidates 作为字幕布局和审计时间锚点
     - micro chunk resolver 合并过短且证据较弱的 split
  -> Boundary Refiner v6
     - 只修 start/end edge delta
     - 不新增 chunk、不合并 chunk、不做 drop
  -> chunk packing / boundary-cache
  -> 可选 Pre-ASR CueQC v6
     - keep_for_asr / drop_before_asr
     - drop 的 chunk 不导出 wav、不进入 ASR
  -> ASR wav chunk export
  -> Qwen ASR text transcription
  -> Boundary chunk subtitle timing
     - ASR 文本只负责显示
     - 时间轴以 Boundary chunk 为准
  -> 字幕 layout
     - 20-frame 最小显示时间
     - 2-frame 最小间隔
     - 7s 最大显示 soft guard
     - 长 cue 先按 ASR 文本断句，再吸附 weak cut，没有 weak cut 才比例估算
  -> 可选 LLM 翻译
  -> SRT / bilingual JSON / quality report / logs
```

关键约束：

- ASR chunk 切分只使用 scorer 的 primary acoustic cut candidates。
- weak cut candidates 有明确时间点，但不直接强切 ASR chunk；它们用于字幕 layout、审计和后续训练。
- `20 / video_fps` 是字幕最短显示和 micro chunk 风险线，不是 runtime duration-only drop 阈值。
- 7 秒是字幕显示 soft guard，不是 ASR chunk 上限。
- Runtime 不使用具体词黑名单或时长启发式删除短促人声；是否进入 ASR 由 Pre-ASR CueQC 模型标签决定。

---

## 模型架构

### Qwen ASR backend

默认模型族使用 Qwen3 ASR 日语 Anime / Galgame checkpoint：

| 用途 | Repo id |
| --- | --- |
| 默认 ASR | `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame` |
| 低配置 ASR | `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame` |

同一个 ASR repo id 也决定 SpeechBoundary-JA frozen feature 的来源，以及三个本地 checkpoint 的 registry key。checkpoint 文件名只是人工可读 tag，真正归属以 metadata 中的 repo id / schema / feature hash 为准。

### SpeechBoundary-JA Scorer v5

| 项 | 内容 |
| --- | --- |
| Schema | `speech_boundary_ja_mamba2_frame_boundary_scorer_v5` |
| Model type | `mamba2_frame_boundary_scorer` |
| Backbone | `BoundarySequenceClassifier` + `transformers.Mamba2Model` wrapper |
| Input | Qwen PTM/encoder frame features + MFCC / timing frame features |
| Output dim | `2` |
| Output heads | `speech_prob`, `split_boundary_prob` |
| Decoder contract | `topographic_split_micro_resolver_v3` |

`speech_prob` 负责找 speech frames；`split_boundary_prob` 负责找应切开的 acoustic boundary。Scorer 不做 keep/drop，也不承担字幕显示时长规则。

Decoder 会输出两类 acoustic cut：

- `primary_cut_candidates`：高可信切点，用于 ASR chunk split。
- `weak_cut_candidates`：弱证据切点，包含 `time_s/frame/score/prominence/speech_valley/strength`，透传到 chunk metadata、boundary cache、ASR chunk metadata 和字幕 layout。

### Boundary Refiner v6

| 项 | 内容 |
| --- | --- |
| Schema | `boundary_edge_refiner_v6` |
| Runtime adapter | `edge_sequence_v1` |
| Backbone | `BoundarySequenceClassifier` + `transformers.Mamba2Model` wrapper |
| Input | scorer 产出的 island edge 上下文特征；包含 left/right/gap 的 PTM/MFCC/timing 统计 |
| Output dim | `2` |
| Output heads | `start_delta_s`, `end_delta_s` |
| Delta clamp | checkpoint metadata 中的 `boundary_delta_max_s` |

Boundary Refiner 只修 chunk 两端。它不学习中间切点、不新增 chunk、不合并 chunk、不做删除路由，也不学习 ASR padding/context budget。

### Pre-ASR CueQC v6

| 项 | 内容 |
| --- | --- |
| Schema | `cueqc_pre_asr_mamba_v6_binary` |
| Feature schema | `pre_asr_cueqc_features_v2` |
| Runtime position | Boundary Refiner 后、wav chunk export 前 |
| Architecture | `Linear(input_dim, hidden_size) -> GELU -> Linear(hidden_size, 2)` |
| Output | `keep_for_asr`, `drop_before_asr` |
| Decision | `p_drop >= PRE_ASR_CUEQC_DROP_THRESHOLD` 时 drop；当前 cold-start 推荐阈值 `0.999` |

Pre-ASR CueQC 只看 ASR 前数值特征：duration、speech segment count、internal gap、refiner delta、scorer speech/split 分布、邻接 gap、micro chunk evidence 等。它禁止使用 ASR text、raw text、token trace、decoder stats、ASR confidence 和 subtitle timing。

当前 1.7B repo 已有 conservative cold-start checkpoint，但默认仍关闭；开启前应先跑 no-translate workflow smoke / audit，确认 false-drop 风险可接受。

### ASR-after CueQC shadow

ASR-after CueQC v4 只保留为显式 opt-in 的 shadow / hard-negative mining 工具，不参与默认 keep/drop。默认链路的删除路由应前置到 Pre-ASR CueQC。

---

## 默认配置

常用配置见 [.env.example](.env.example)。通常只需要修改：

- `API_KEY`
- `OPENAI_COMPATIBILITY_BASE_URL`
- `LLM_MODEL_NAME`
- `HF_ENDPOINT`
- `ASR_BACKEND`
- `ASR_BATCH_SIZE_BY_REPO`

推荐 batch 档位：

| 显存档 | `ASR_BATCH_SIZE_BY_REPO` |
| --- | --- |
| 6GB 默认 / 分发 | `1.7B=32, 0.6B=64` |
| 8GB 本机实验 | `0.6B=128, 1.7B=64` |

如果后台还有其他 CUDA 进程或出现 OOM，先退回更小 batch。

推理需要 ASR / SpeechBoundary-JA frozen feature Hugging Face 模型，以及与当前 repo id 匹配的本地 checkpoint。源码运行时如果本地没有 Hugging Face 模型，会按需下载到 `models/`。registry 缺失、覆盖映射未命中当前 repo id、文件不存在、schema 不匹配或 metadata 不匹配都会 fail-fast。

训练时生成的 CUDA feature cache、synthetic WAV、sequence JSONL、tensor cache 和 `datasets/train/...` 产物都不是运行依赖，不随源码或 Windows release 打包。

---

## 字幕与文本策略

- ASR 文本会做 Unicode NFKC、空白归一、换行折叠和展示安全处理。
- `ASR_CONTEXT` / `ASR_HEAD_CONTEXT` 只作为 Qwen ASR 提示词，不作为字幕后处理删除规则。
- 字幕时间轴来自 Boundary chunk；ASR 输出文本只负责显示，不驱动默认切分。
- LLM 翻译前会先固定 cue plan，翻译不会重排时间轴。

---

## 输出与缓存

- `video/<视频名>/`：正式字幕、质量报告和人工质检报告。
- `models/`：Hugging Face 模型缓存。
- `tmp/jobs/<job_id>/`：Web / pipeline 单次任务临时目录；`JOB_TEMP_DIR` 默认是 `./tmp/jobs`。
- `tmp/chunks/`：ASR wav chunk 和 crash-resume checkpoint 的一次性运行目录。
- `tmp/cache/boundary/`：SpeechBoundary-JA frame score 到 Boundary Refiner 输出的 boundary-cache。
- `tmp/cache/torch/`、`tmp/cache/hf/`：torch / Hugging Face 运行缓存。
- `tmp/log/`：启用运行日志后的任务日志。
- `datasets/`：本地训练、验证、测试数据归档，默认 ignored；不参与普通推理和 release 打包。
- `agents/temp/`：研究脚本、smoke、临时日志和中间产物。
- `agents/audits/`：可长期复查的本地审计页，默认 ignored，不随 `git push` 发布。

本地审计页服务：

```powershell
.\tools\audits\serve_audits.ps1
```

Linux / WSL2：

```bash
tools/audits/serve_audits.sh
```

审计服务支持音频 Range seek 和导航页删除 API。直接打开 HTML 可以浏览页面，但删除按钮不能真正移动本地审计目录。

成功运行后默认删除一次性 job 临时目录；保留可复用缓存，例如 `models/`、`tmp/cache/boundary/` 和 Web 状态。

---

## 常见问题

### 模型下载慢

设置：

```env
HF_ENDPOINT=https://hf-mirror.com
```

或提前把模型下载到 `models/` 对应目录。

### CUDA 没有被使用

确认日志中出现：

```text
actual_device=cuda
model_param_device=cuda:*
```

受限 sandbox、错误的 PyTorch wheel、驱动问题或从非 GPU 环境启动 Web 服务都可能导致 CPU fallback。

### 显存不足

默认配置按 6GB 级显存目标设置：

```env
ASR_BATCH_SIZE_BY_REPO=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=32,jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=64
```

8GB 本机可尝试：

```env
ASR_BATCH_SIZE_BY_REPO=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=64,jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=128
```

如果仍然 OOM，优先降低：

```env
ASR_BATCH_SIZE_BY_REPO=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=8,jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=24
```

### 长任务怎么排查

启用运行日志后，日志会写入 `tmp/log/` 或任务输出目录。反馈问题时请保留 `.run.log`、质量报告和对应 SRT。

---

## 开发

主要代码位置：

- `src/main.py`：主流程编排。
- `src/core/`：配置和任务上下文。
- `src/pipeline/`：音频、缓存、输出、质量报告和阶段日志。
- `src/asr/`：ASR、Boundary 字幕时间轴分配、Pre-ASR CueQC / ASR-after shadow CueQC 和转写流程。
- `src/boundary/`：Boundary Refiner checkpoint loader、edge-sequence Mamba2 adapter、core planner 和 boundary-cache。
- `src/boundary/ja/`：SpeechBoundary-JA scorer、PTM/MFCC feature cache schema、训练数据 manifest 和 frame-score 训练工具。
- `src/llm/`：翻译 prompt、cache、glossary、API patch 和 translator。
- `src/subtitles/`：SRT writer、字幕选项和字幕 QC。
- `src/web/`：FastAPI 接口和静态前端。
- `tools/`：训练、字幕审计、workflow smoke 和发布辅助脚本。

常用测试：

```bash
PYTHONIOENCODING=utf-8 uv run pytest tests/test_config.py tests/web/test_jobs_api.py tests/test_asr_backend_dispatch.py
PYTHONIOENCODING=utf-8 uv run pytest tests/test_boundary_cache.py tests/test_boundary_planner.py tests/test_chunk_packer.py tests/test_pipeline_chunk_config_runtime.py
PYTHONIOENCODING=utf-8 uv run pytest tests/test_translation_cache.py tests/test_translator_prompt.py tests/test_quality_report_output.py
```

维护者本地打包 Windows 运行包：

```powershell
uv pip install pyinstaller torchcodec
.\packaging\build_windows.ps1 -Clean
```

构建产物位于 `dist\JAVTrans\JAVTrans.exe`。如需分发，可再压缩为 `.7z` 并上传到网盘；GitHub Releases 只发布源码和版本说明。打包细节见 `packaging/README.md`。

---

## 工具索引

所有 Python 工具都从项目根目录执行，并使用当前 `.venv`：

```powershell
$env:PYTHONIOENCODING='utf-8'
uv run python -m <module> --help
```

常用入口：

- `tools.workflows.run_full_workflow`：命令行完整工作流 smoke。
- `tools.web.smoke.start_server` / `submit_job` / `poll_job` / `summarize_job`：Web 服务 smoke 和任务汇总。
- `tools.audits.audit_nav`、`tools.audits.serve_static`、`tools.audits.serve_audits.ps1`、`tools.audits.serve_audits.sh`：维护和启动本地审计导航页。
- `tools.audits.generate_cueqc_cluster_audit_html`：生成音频审计页，支持 chunk/context 播放、筛选排序和字幕对照。
- `tools.audits.generate_cueqc_cluster_broadcast_html`：生成独立簇级 keep/drop 广播标注页；混簇/跳过只记录 abstain。
- `tools.asr.cueqc.export_pre_asr_v6_audit_candidates`：从 current workflow `.timings.json` 导出 Pre-ASR CueQC v6 审计候选。

命令行完整工作流 smoke：

```powershell
uv run python -m tools.workflows.run_full_workflow --video video/<your-video>.mp4 --task-name 20260617_191654_cli-smoke --label smoke
```

Web smoke：

```powershell
uv run python -m tools.web.smoke.start_server --run-dir agents/temp/20260617_191654_web-smoke
uv run python -m tools.web.smoke.submit_job --video-path video/<your-video>.mp4 --output-dir video --run-dir agents/temp/20260617_191654_web-smoke
uv run python -m tools.web.smoke.poll_job --job-id-file agents/temp/20260617_191654_web-smoke/job_id.txt --run-dir agents/temp/20260617_191654_web-smoke --interval-seconds 300
uv run python -m tools.web.smoke.summarize_job --job-id <job_id> --run-dir agents/temp/20260617_191654_web-smoke
```

训练、诊断、实验记录和动态计划不在 README 展开；见 [HISTORY.md](HISTORY.md)。

---

## 更新记录

更新记录、实验路线、踩坑笔记和后续计划见 [HISTORY.md](HISTORY.md)。
