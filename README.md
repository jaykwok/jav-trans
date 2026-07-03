# jav-trans

jav-trans 是一个本地JAV字幕生成工具，面向 Windows + NVIDIA 显卡，也可在 WSL2 / Linux 下源码运行。它把视频处理成日文字幕、中文字幕或中日双语字幕，并把音频准备、语音岛检测、语义切分、可选 Pre-ASR CueQC、Qwen ASR、字幕时间轴、LLM 翻译和质量报告串成一条本地优先的流水线。

项目目标：本地完成视频、音频、边界切分、ASR 和字幕时间轴重计算；LLM 只负责翻译、术语一致和口吻连贯，不负责脑补剧情或修正 ASR 误听。

致谢：[WhisperJAV](https://github.com/a63n/WhisperJAV) 为本项目早期路线提供了重要参考。

---

## 项目背景

本项目的边界系统不是传统 VAD。目标不是单纯判断“有没有人声”，而是生成适合字幕和 ASR 的 speech-core chunk：尽量接近一句台词一个 chunk，避免把 BGM、环境声、贴连短句或长独白粗暴混成同一种情况。

当前设计把职责拆开：

- SpeechIslandScorer 只检测 speech island。
- Outer Edge Refiner 只修整条 island 的外边界。
- Semantic Split Model 只判断候选点 `cut/continue/unsure`。
- Cut Edge Refiner 只把确认的 cut 吸附到一个共享绝对时间戳。
- Pre-ASR CueQC 只做 ASR 前 `keep/drop` 路由。
- 字幕 layout 只处理显示规则，不反向修改 ASR chunk 语义。

这样做是为了避免一个模型同时承担“找语音、切句、删噪声、修边界、做字幕排版”。设计演进、实验记录、失败路线和更新记录都放在 [HISTORY.md](HISTORY.md)。

---

## 快速开始

### Release 版

GitHub Releases 只用于发布 source code、release notes 和版本说明，不上传大型 Windows `.7z` 运行包。Windows 运行包由维护者按需本地打包成单个 `.7z`，通过网盘或其他外部分发渠道提供；拿到包后解压运行：

```text
jav-trans.exe
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
uv pip install "transformers @ git+https://github.com/huggingface/transformers.git"
uv pip install -r requirements.txt
```

启动网页控制台：

```powershell
$env:PYTHONIOENCODING="utf-8"
uv run --no-sync python launcher.py
```

默认地址为 `http://127.0.0.1:17321`。首次运行可以没有 `.env`；打开页面后在“翻译 API”面板填写 API Key、Base URL、模型和目标语言，保存或提交任务时会自动写入项目根目录 `.env`。新建的 `.env` 只启用实际保存的本机值，ASR batch、后端、显存预算等研究项会以注释示例形式写入。国内网络下载 Hugging Face 模型较慢时，可在“识别设置”里填写代理协议、地址和端口。

Linux / WSL2 下如果只启动浏览器版 Web 服务，也可以直接运行：

```bash
PYTHONIOENCODING=utf-8 PYTHONPATH=src uv run python -m uvicorn web.app:create_app --factory --host 127.0.0.1 --port 17321
```

Web 提交是否使用 CUDA 取决于后端服务进程是否能看到 GPU，而不是浏览器本身。完整 SpeechBoundary-JA / ASR smoke 应确认日志中出现 `cuda_available=True`、`device=cuda:0` 或 `actual_device=cuda`。
Windows 打包版会自带 CUDA 版 PyTorch runtime，但仍需要用户本机 NVIDIA 驱动支持对应 CUDA runtime；Web 会在模型要求检查中提示驱动过旧或 CUDA 初始化失败。

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
  -> Shared Qwen feature extraction
     - Qwen ASR repo 对应的 frozen PTM/encoder frame features
     - MFCC / timing numeric features
  -> SpeechIslandScorer v8
     - 仅输出 dense speech_prob
     - speech hysteresis 生成高召回 speech islands
     - acoustic valley 只作为非绑定候选
  -> Outer Edge Refiner v1
     - 只修整条 speech island 的 start/end
  -> Semantic Split Verifier v1
     - 对 core 内候选判断 cut / continue / unsure
  -> Cut Edge Refiner v1
     - 只精修已确认候选
     - 相邻 chunk 共用一个 source absolute cut timestamp
  -> chunk packing / boundary-cache
  -> 可选 Pre-ASR CueQC v11
     - keep_for_asr / drop_before_asr
     - local PTM 表征 + 前后差分 + Mamba 时序上下文
     - drop 的 chunk 不导出 wav、不进入 ASR
  -> ASR wav chunk export
  -> Qwen ASR text transcription
  -> Boundary chunk subtitle timing
     - ASR 文本负责字幕文本
     - acoustic timeline 来自 source absolute boundary
  -> Subtitle Layout v2
     - acoustic/display 双时间轴
     - 20-frame 最小显示时间（固定 `24000/1001` 基准）
     - 2-frame 最小间隔（固定 `24000/1001` 基准）
     - 7s 最大显示 soft guard
     - 长 cue 先按 ASR 文本断句，再吸附 weak cut，没有 weak cut 才比例估算
  -> 可选 LLM 翻译
  -> SRT / bilingual JSON / quality report / logs
```

关键约束：

- SpeechIslandScorer 不做句内结构决策；声学候选只有经过 Semantic Split Model 接受后才会切。
- 内部 cut 是一个共享绝对时间戳，不允许左右 chunk 各自修边。
- `20 / (24000/1001)` 是字幕最短显示和 micro chunk 风险线，不是 runtime duration-only drop 阈值。
- 7 秒是字幕显示 soft guard，不是 ASR chunk 上限。
- Runtime 不使用具体词黑名单或时长启发式删除短促人声；是否进入 ASR 由 Pre-ASR CueQC 模型标签决定。

---

## 模型架构

### Qwen ASR backend

默认模型族使用 Qwen3 ASR 日语 Anime / Galgame checkpoint：

| 用途 | Repo id |
| --- | --- |
| 默认高质量 ASR | `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf` |
| 低配 / 更快 ASR | `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf` |

同一个 ASR repo id 决定 frozen PTM feature 和五个前置模型的 checkpoint registry key。`0.6B` 与 `1.7B` 均提供各自独立训练的五模型 checkpoint，不能跨 repo 复用或转换权重。

生产 checkpoint 均为自包含文件，不从 `agents/temp` 加载：

| 模型 | 正式目录与文件名 |
| --- | --- |
| SpeechIslandScorer | `src/boundary/ja/checkpoints/speech_island_scorer_v8.<repo-tag>.pt` |
| Outer Edge Refiner | `src/boundary/checkpoints/outer_edge_refiner_v1.<repo-tag>.pt` |
| Semantic Split Model | `src/boundary/checkpoints/semantic_split_model_v1.<repo-tag>.pt` |
| Cut Edge Refiner | `src/boundary/checkpoints/cut_edge_refiner_v1.<repo-tag>.pt` |
| Pre-ASR CueQC | `src/asr/checkpoints/pre_asr_cueqc_v11.<repo-tag>.pt` |

每个文件的 `metadata.artifact` 记录生产文件名、模型角色、流水线序号、训练 run、promotion 时间和 repo 绑定。Web 页面会显示五模型的独立阶段进度，并在切换 `0.6B` / `1.7B` 时按 repo id 自动选择同系列 checkpoint。

### SpeechIslandScorer v8

| 项 | 内容 |
| --- | --- |
| Schema | `speech_boundary_ja_mamba2_speech_island_scorer_v8` |
| Model type | `mamba2_speech_island_scorer` |
| Input | 128 维 Qwen PTM frame features + MFCC |
| Output | `speech_prob` |
| Decoder | `speech_hysteresis_islands_v1` |

Scorer 只负责高召回找 speech island。acoustic valley 可以生成候选区域，但不会直接产生最终切分。

### Semantic boundary models

| 模型 | Schema / runtime adapter | 职责 |
| --- | --- | --- |
| Outer Edge Refiner | `outer_edge_refiner_v1` / `speech_island_outer_edges_v1` | 只修整条 island 的 start/end |
| Semantic Split | `semantic_split_verifier_v1` / `candidate_cut_continue_unsure_v1` | 判断 `cut/continue/unsure` |
| Cut Edge Refiner | `cut_edge_refiner_v1` / `shared_absolute_cut_v1` | 精修已确认 cut 的共享绝对时间戳 |

短 core（`<=6s`）需要 `p_cut>=0.90`，其余候选需要 `p_cut>=0.75`，切分后单侧至少 `1.2s`。Outer Refiner 不处理内部边界，Split Model 不移动时间轴，Cut Refiner 不决定是否切。

### Pre-ASR CueQC v11

| 项 | 内容 |
| --- | --- |
| Schema | `cueqc_pre_asr_semantic_chunk_v11_binary` |
| Model arch | `cueqc_pre_asr_semantic_chunk_v11` |
| Feature schema | `pre_asr_cueqc_features_v8` |
| Runtime adapter | `pre_asr_semantic_chunk_sequence_v3` |
| Architecture | repo-specific: `1.7B` 使用 local + neighbor differences + gated Mamba；`0.6B` 使用验证更稳的 local-only |
| Output | `keep_for_asr`, `drop_before_asr` |
| Decision | `p_drop >= 0.95` 时 drop；低置信默认 keep |

Cue 按原始时序送入模型。`1.7B` 训练使用 sequence window，但只对平衡采样的 anchor 计算 loss；显式差分帮助 Mamba 建模邻接变化。`0.6B` 的独立 holdout 显示时序残差会降低 keep recall，因此正式 checkpoint 将 `temporal_residual_scale` 设为 `0`，保留 local branch。模型禁止使用 ASR text、token trace、decoder stats、ASR confidence 和 subtitle timing。

当前 `1.7B` checkpoint 已进入默认 registry，6GB 默认配置会启用 Pre-ASR CueQC。held-out operating point 在阈值 `0.95` 下为 drop precision `98.13%`、drop recall `92.92%`、semantic keep recall `95.06%`。

`0.6B` 五模型也已进入默认 registry，独立验证结果如下：

| 模型 | 选定 operating point / held-out 指标 |
| --- | --- |
| SpeechIslandScorer | threshold `0.15`：precision `79.79%`、recall `98.25%` |
| Outer Edge Refiner | start/end MAE `14.32/12.88ms` |
| Semantic Split | `p_cut>=0.75`：cut precision `94.63%`、cut recall `56.59%`、continue false-cut `0.16%` |
| Cut Edge Refiner | MAE `49.07ms`、p90 `138.47ms` |
| Pre-ASR CueQC | threshold `0.95`：drop precision `98.35%`、drop recall `86.23%`、keep recall `94.87%`、false-drop `4/78` |

### 离线 ASR-after CueQC v4

ASR-after CueQC v4 不属于默认 workflow，也不在主线 runtime 中运行。它只作为离线审计 / hard-negative mining 工具使用，输入来自已生成产物：

```powershell
$env:PYTHONIOENCODING="utf-8"
uv run --no-sync python tools/asr/cueqc/export_candidates.py `
  --aligned path/to/video.aligned_segments.json `
  --transcript path/to/video.transcript.json `
  --output agents/temp/YYYYMMDD_HHMMSS_cueqc-offline/candidates.jsonl
```

如果 `transcript.json` 与 `aligned_segments.json` 同目录同 stem，可省略 `--transcript`。默认运行不会读取 ASR-after CueQC checkpoint、不会捕获 ASR decoder stats，也不会把 shadow decision 写回字幕结果。

---

## 默认配置

默认配置内置在 `src/core/config.py`，首次保存 Web 设置时会自动生成 `.env`。`.env` 只用于本机私密值和显式覆盖，不复制默认配置。通常只需要在 Web “翻译 API”面板填写：

- `API_KEY`
- `OPENAI_COMPATIBILITY_BASE_URL`
- `LLM_MODEL_NAME`
- 代理协议 / 地址 / 端口（可选，用于模型下载和 HTTP 请求）

ASR 6GB 默认值已经内置。默认使用 `1.7B` 高质量模型；需要切到 `0.6B` 低配/更快档，或覆盖 batch / 显存预算时，再通过“参数调优”里的环境变量覆盖，或手动编辑首次保存后生成的 `.env`。

6GB 默认配置：

```env
ASR_BACKEND=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf
ASR_BATCH_SIZE=auto
ASR_BATCH_SIZE_BY_REPO=jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf=12,jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf=4
ASR_STAGE_WORKER_VRAM_BUDGET_MB=5600
ASR_STAGE_WORKER_OOM_RETRY_LIMIT=3
SPEECH_BOUNDARY_JA_WINDOW_S=20
SPEECH_BOUNDARY_JA_OVERLAP_S=4
SEMANTIC_SPLIT_INFERENCE_BATCH_SIZE=128
PRE_ASR_CUEQC_ENABLED=1
```

ASR stage 固定由统一 GPU worker 持有 CUDA：Boundary/PTM feature extraction、Pre-ASR CueQC、ASR 和对齐都在同一个 GPU owner 进程里顺序执行，Web / 调度主进程只做任务编排、缓存索引和输出写入。OOM、CUDA 状态异常或超过 `ASR_STAGE_WORKER_VRAM_BUDGET_MB` 时会杀掉 worker，不会把 Web 主进程一起带崩。

`ASR_STAGE_WORKER_VRAM_BUDGET_MB=5600` 是 6GB 卡的软 OOM 线。即使 PyTorch 没抛 `OutOfMemoryError`，只要 worker 侧 peak reserved/allocated 超过预算，就按 OOM 处理并按 `ASR_STAGE_WORKER_OOM_RETRY_LIMIT` 重启 worker、降低 batch 后重跑，避免 Windows 进入共享显存后严重变慢。默认会从内置 batch 逐步降到 `ASR_BATCH_SIZE=1`；如果 batch=1 仍 OOM，任务会停止，Web 任务卡会提示切换到 `0.6B` 低显存档。

推理需要 ASR / SpeechBoundary-JA frozen feature Hugging Face 模型，以及与当前 repo id 匹配的本地 checkpoint。源码运行时如果本地没有 Hugging Face 模型，会按需下载到 `models/`。registry 缺失、覆盖映射未命中当前 repo id、文件不存在、schema 不匹配或 metadata 不匹配都会 fail-fast。

训练时生成的 CUDA feature cache、synthetic WAV、sequence JSONL、tensor cache 和 `datasets/train/...` 产物都不是运行依赖，不随源码或 Windows release 打包。

---

## 字幕与文本策略

- ASR 文本会做 Unicode NFKC、空白归一、换行折叠和展示安全处理。
- Qwen3-ASR runtime 始终使用 Transformers 官方 `apply_transcription_request(audio=..., language=...)` 路径，不提供演员名 / 人名 context 提示分支。
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
- `tmp/log/<job_id>/`：默认启用的本地诊断目录；包含 `.run.log` 和持久化 `.timings.json`。
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

审计导航会显示每个审计产物的生成时间，优先使用 summary 时间，其次使用目录名前缀，便于区分多轮审计页。审计服务支持音频 Range seek 和导航页删除 API。直接打开 HTML 可以浏览页面，但删除按钮不能真正移动本地审计目录。

成功运行后默认删除一次性 job 临时目录；保留可复用缓存，例如 `models/`、`tmp/cache/boundary/` 和 Web 状态。

---

## 常见问题

### 模型下载慢

在 Web “识别设置”里填写代理，例如：

```env
PROXY_PROTOCOL=http
PROXY_HOST=127.0.0.1
PROXY_PORT=7890
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

默认配置按 6GB 级显存目标设置，优先使用 1.7B 高质量档：

```env
ASR_BACKEND=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf
ASR_BATCH_SIZE_BY_REPO=jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf=12,jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf=4
ASR_STAGE_WORKER_VRAM_BUDGET_MB=5600
SPEECH_BOUNDARY_JA_WINDOW_S=20
SPEECH_BOUNDARY_JA_OVERLAP_S=4
```

如果仍然 OOM，先降低当前模型 batch：

```env
ASR_BATCH_SIZE=2
```

需要低配/更快档时切到 0.6B：

```env
ASR_BACKEND=jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf
ASR_BATCH_SIZE=auto
```

### 长任务怎么排查

运行日志默认写入 `tmp/log/<job_id>/`。`.run.log` 便于查错，`.timings.json` 记录音频准备、Boundary/Pre-ASR/ASR、翻译、写出等阶段耗时和显存快照；Web 完成任务后也会把这两个文件列在“其他文件”里。反馈问题时请保留 `.run.log`、`.timings.json`、质量报告和对应 SRT。

---

## 开发

主要代码位置：

- `src/main.py`：主流程编排。
- `src/core/`：配置和任务上下文。
- `src/pipeline/`：音频、缓存、输出、质量报告和阶段日志。
- `src/asr/`：ASR、Boundary 字幕时间轴分配、Pre-ASR CueQC 和转写流程；ASR-after CueQC v4 仅保留为离线审计工具入口。
- `src/boundary/`：Boundary Refiner checkpoint loader、edge-sequence Mamba2 adapter、core planner 和 boundary-cache。
- `src/boundary/ja/`：SpeechBoundary-JA scorer、PTM/MFCC feature cache schema、训练数据 manifest 和 frame-score 训练工具。
- `src/llm/`：翻译 prompt、cache、glossary、API patch 和 translator。
- `src/subtitles/`：SRT writer、字幕选项和字幕 QC。
- `src/web/`：FastAPI 接口和静态前端。
- `tools/`：训练、字幕审计、workflow smoke 和发布辅助脚本。

常用测试：

```bash
PYTHONIOENCODING=utf-8 uv run pytest tests/test_config.py tests/web/test_jobs_api.py tests/test_asr_backend_dispatch.py
PYTHONIOENCODING=utf-8 uv run pytest tests/test_boundary_cache.py tests/test_semantic_boundary_runtime.py tests/test_chunk_packer.py tests/test_pipeline_chunk_config_runtime.py
PYTHONIOENCODING=utf-8 uv run pytest tests/test_translation_cache.py tests/test_translator_prompt.py tests/test_quality_report_output.py
```

维护者本地打包 Windows 运行包：

```powershell
uv pip install pyinstaller torchcodec
.\packaging\build_windows.ps1 -Clean
```

构建产物位于 `dist\jav-trans\jav-trans.exe`。如需分发，可再压缩为 `.7z` 并上传到网盘；GitHub Releases 只发布源码和版本说明。打包细节见 `packaging/README.md`。

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
- `tools.audits.generate_cueqc_prediction_audit_html`：根据 `cueqc_predictions.jsonl` 采样生成 CueQC 预测 false-drop 审计页，支持混采/高置信策略与字幕对照。
- `tools.audits.generate_subtitle_ab_compare_audit_html`：生成整片旧/新字幕 A/B 对比审计页，用于评估边界或时间轴改动效果。
- `tools.asr.convert_qwen3_asr_to_hf`：把 legacy 非 `-hf` Qwen3-ASR fine-tune safetensors 权重迁移到 Transformers-native `-hf` layout（`thinker.audio_tower.* -> model.audio_tower.*`、`thinker.audio_tower.proj{1,2}.* -> model.multi_modal_projector.linear_{1,2}.*`、`thinker.model.* -> model.language_model.*`，并复用 `Qwen/Qwen3-ASR-*-hf` 模板文件）。
- `tools.asr.cueqc.export_semantic_boundary_candidates`：导出五段式 boundary runtime 的最终 semantic chunks。
- `tools.asr.cueqc.label_semantic_pre_asr_with_omni`：为 Pre-ASR v11 生成 `definite_drop/definite_keep/ambiguous_ignore` 弱标签；上传前统一转为 16k mono 32kbps MP3，长音频使用保留语义中心的窗口。

Qwen3-ASR `-hf` 转换示例：

```powershell
$env:PYTHONIOENCODING='utf-8'
uv run python -m tools.asr.convert_qwen3_asr_to_hf `
  --source-model-dir path/to/legacy-non-hf-qwen3-asr `
  --output-dir agents/temp/YYYYMMDD_HHMMSS_qwen3-asr-17b-ja-hf `
  --template-repo Qwen/Qwen3-ASR-1.7B-hf `
  --max-shard-size 2GB
```

该工具只转换模型仓库格式，不改变训练权重语义。项目 runtime 与 SFT 工具均使用支持 Qwen3-ASR 的原生 Transformers `-hf` 路径，不再依赖 `qwen-asr` 包。

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
