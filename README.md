# JAVTrans

JAVTrans 是一个本地字幕生成工具，面向 Windows + NVIDIA 显卡，也可在 WSL2 / Linux 下源码运行。它把视频处理成日文字幕、中文字幕或中日双语字幕，并把音频准备、speech-island 边界规划、ASR、强制对齐、字幕时间轴归一化、LLM 翻译和质量报告串成一条本地优先的流水线。

项目目标：本地完成视频、音频、边界切分、ASR 和时间轴重计算；LLM 只负责翻译、术语一致和口吻连贯，不负责脑补剧情或修正 ASR 误听。

致谢：[WhisperJAV](https://github.com/a63n/WhisperJAV) 为本项目早期路线提供了重要参考。实验历史、路线取舍、调试记录和参考来源见 [HISTORY.md](HISTORY.md)。

---

## 当前状态

- 默认 ASR：`jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`。
- 可选 ASR：`jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`。
- 默认边界系统：SpeechBoundary-JA，backend key 为 `speech_boundary_ja`。它不是传统 VAD，而是 `Qwen PTM + MFCC/energy frame scores -> boundary candidates -> Boundary Refiner -> constrained planner -> ASR chunks` 的 speech-island boundary pipeline。
- 默认 Boundary Refiner：随源码提供 learned `transformers.Mamba2Model` 小 checkpoint，文件为 `src/boundary/checkpoints/boundary_refiner.pt`，用于边界候选的 merge / split 决策和左右 ASR context budget。
- 默认显存目标：单阶段峰值适配 6GB 级 NVIDIA 显卡。更小显存可手动降低 `.env` 中的 ASR / aligner batch size。
- 当前字幕策略：边界层尽量把 ASR chunk 切成接近一句台词；字幕层优先用可靠词级 `word.start` 锚定 cue start，并负责 2-frame gap、显示时长、短 cue 合并和读速控制。

---

## 快速开始

### Release 版

GitHub Releases 只用于发布 source code、release notes 和版本说明，不上传大型 Windows `.7z` 运行包。Windows 运行包由维护者按需本地打包成单个 `.7z`，通过网盘或其他外部分发渠道提供；拿到包后解压运行：

```text
JAVTrans.exe
```

Windows 打包版默认内置 0.6B ASR、1.7B ASR、forced aligner、`ffmpeg` / `ffprobe` 和小型 Boundary Refiner checkpoint。

Release 版不内置 Microsoft Edge WebView2 Runtime。大多数 Windows 10/11 已自带；如果无法打开窗口，请安装 [Microsoft Edge WebView2 Evergreen Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/)。

### 源码运行

推荐环境：

- Windows 10/11 或 WSL2 / Linux。
- NVIDIA 独立显卡和较新的驱动。
- Python 3.13。
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
LLM_MODEL_NAME=deepseek-v4-pro
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

Web 提交是否使用 CUDA 取决于后端服务进程是否能看到 GPU，而不是浏览器本身。完整 SpeechBoundary-JA / ASR / ForcedAligner smoke 应确认日志中出现 `cuda_available=True`、`device=cuda:0` 或 `actual_device=cuda`。

---

## 使用流程

1. 打开网页控制台。
2. 选择视频文件。
3. 选择字幕模式、ASR 后端和翻译设置。
4. 提交任务。
5. 在输出目录查看 SRT、质量报告和日志。

勾选“不翻译（仅日文字幕）”时，流水线仍会执行边界规划、ASR、ASR QC、forced alignment 和字幕时间轴归一化，但跳过 LLM 翻译，最终输出 `<视频名>.ja.srt`。这是验证本地边界 / ASR / 对齐链路的推荐 smoke 模式。

主流水线：

```text
视频
-> 音频准备
-> SpeechBoundary-JA bootstrap frame scores
-> boundary candidate extraction
-> Boundary Refiner scoring
-> constrained boundary planner
-> ASR
-> ASR QC
-> forced alignment
-> display_text / align_text 预处理
-> cue plan 时间轴归一化
-> LLM 翻译
-> SRT / quality report
```

LLM 翻译前会先固定 cue plan。SRT writer 只写入已经归一化的时间轴，不再隐式改变时间轴。

翻译缓存分三层：

- `translation_cache.jsonl`：本地 batch cache，用于完全相同 cue / timing / prompt 的精确复跑与 crash resume。
- `translation_cache.memory.jsonl`：本地 translation memory，按日文文本、目标语言、词汇表、人物参考、prompt 语义版本和模型族复用译文。
- Provider prompt cache：通过稳定的 system prompt / 全片 JSON 前缀降低 API 成本；它不等同于本地翻译缓存。

---

## 默认模型

| 用途 | 默认来源 | 本地缓存 / 文件 |
| --- | --- | --- |
| 默认 ASR | `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame` | `models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame` |
| 可选 ASR | `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame` | `models/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame` |
| SpeechBoundary-JA frozen feature | `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame` | `models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame` |
| Forced aligner | `Qwen/Qwen3-ForcedAligner-0.6B` | `models/Qwen-Qwen3-ForcedAligner-0.6B` |
| Boundary Refiner | learned `transformers.Mamba2Model` v2 checkpoint | `src/boundary/checkpoints/boundary_refiner.pt` |

常用配置见 [.env.example](.env.example)。通常只需要修改 API key、翻译模型、`HF_ENDPOINT`、ASR backend 和 batch size。

推理需要源码内置的 `src/boundary/checkpoints/boundary_refiner.pt`，以及 ASR / SpeechBoundary-JA frozen feature / forced aligner Hugging Face 模型。Windows 打包版默认内置 0.6B ASR、1.7B ASR 和 forced aligner；源码运行时如果本地没有模型，仍会按需下载到 `models/`。Boundary Refiner 训练时生成的 CUDA feature cache、synthetic WAV、sequence JSONL 和 `datasets/train/...` 产物都不是运行依赖，不随源码或 Windows release 打包。

---

## 字幕与文本策略

- 系统维护 `display_text` 和 `align_text` 两份文本。
- `display_text` 用于最终字幕显示，只做 Unicode NFKC、空白归一、换行折叠和展示安全处理。
- `align_text` 只给 forced aligner 使用，可删除标点、emoji、音乐符号和不可发音装饰符。
- `ASR_CONTEXT` / `ASR_HEAD_CONTEXT` 只作为 Qwen ASR 提示词，不再作为字幕后处理删除规则。
- 不使用具体词黑名单，不直接删除 `ん`、`あ`、喘息、呻吟、拟声、低信息短句或常见台词。
- 重复循环、低置信、fallback 和 `asr_review_uncertain` 只作为 QC / 诊断 / 审计信号；默认不会直接清空最终字幕。
- forced aligner 失败时不伪造精确时间轴，会保留可诊断 fallback 标签。

---

## 输出与缓存

- `video/<视频名>/`：正式字幕、质量报告和人工质检报告。
- `models/`：Hugging Face 模型缓存。
- `tmp/jobs/<job_id>/`：Web / pipeline 单次任务临时目录；`JOB_TEMP_DIR` 默认是 `./tmp/jobs`。
- `tmp/chunks/`：ASR wav chunk 和 crash-resume checkpoint 的一次性运行目录。
- `tmp/cache/boundary/`：SpeechBoundary-JA frame score 到 Boundary Planner 输出的 boundary-cache v2。
- `tmp/cache/torch/`、`tmp/cache/hf/`：torch / Hugging Face 运行缓存。
- `tmp/log/`：启用运行日志后的任务日志。
- `datasets/`：本地训练、验证、测试数据归档，默认 ignored；不参与普通推理和 release 打包。
- `agents/temp/`：研究脚本、smoke、临时日志和中间产物。
- `agents/audits/`：可长期复查的本地审计页，统一从 `agents/audits/index.html` 进入。

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

默认配置按 6GB 级显存目标设置。如果仍然 OOM，优先降低：

```env
ASR_BATCH_SIZE_BY_REPO=jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=24,jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=8
ALIGNER_BATCH_SIZE=24
ALIGN_LONG_CHUNK_BATCH_SIZE=24
```

### 长任务怎么排查

启用运行日志后，日志会写入 `tmp/log/` 或任务输出目录。反馈问题时请保留 `.run.log`、质量报告和对应 SRT。

---

## 开发

主要代码位置：

- `src/main.py`：主流程编排。
- `src/core/`：配置和任务上下文。
- `src/pipeline/`：音频、缓存、输出、质量报告和阶段日志。
- `src/asr/`：ASR、forced alignment、prealign 和转写流程。
- `src/boundary/`：Boundary features、candidate extraction、Boundary Refiner 接口、sequence backbone、constrained planner 和 boundary-cache v2。
- `src/boundary/ja/`：SpeechBoundary-JA bootstrap scorer、PTM/MFCC feature cache schema、训练数据 manifest 和 frame-score 训练工具。
- `src/llm/`：翻译 prompt、cache、glossary、API patch 和 translator。
- `src/subtitles/`：SRT writer、字幕选项和字幕 QC。
- `src/web/`：FastAPI 接口和静态前端。
- `tools/`：训练、诊断、字幕审计和发布辅助脚本。

常用测试：

```bash
PYTHONIOENCODING=utf-8 uv run pytest tests/test_config.py tests/web/test_jobs_api.py tests/test_asr_backend_dispatch.py
PYTHONIOENCODING=utf-8 uv run pytest tests/test_boundary_cache.py tests/test_boundary_candidates.py tests/test_boundary_planner.py tests/test_chunk_packer.py tests/test_pipeline_chunk_config_runtime.py
PYTHONIOENCODING=utf-8 uv run pytest tests/test_translation_cache.py tests/test_translator_prompt.py tests/test_quality_report_output.py
```

维护者本地打包 Windows 运行包：

```powershell
uv pip install pyinstaller torchcodec
.\packaging\build_windows.ps1 -Clean
```

构建产物位于 `dist\JAVTrans\JAVTrans.exe`。如需分发，可再压缩为 `.7z` 并上传到网盘；GitHub Releases 只发布源码和版本说明。打包细节见 `packaging/README.md`。
