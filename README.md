# JAVTrans

Windows 本地视频字幕生成工具。JAVTrans 以 Web 控制台为主入口，把长视频转成中文字幕或中日双语 SRT：音频提取、WhisperSeg VAD、ASR、强制对齐、F0 性别检测、OpenAI-compatible 翻译、质量报告和断点续传都在本机流水线内完成。

项目按 Windows + NVIDIA GPU 的单机工作流设计，主入口是 `run_web.py`。Web 任务队列通过 `JobSpec -> JobContext` 把每个任务的 ASR、字幕、输出、翻译并发和临时文件选项显式传给后端，避免用全局 `.env` 热覆盖任务参数。旧的根目录 `run.py` 薄包装入口已移除。

## 使用说明

### 1. 安装运行环境

完整视频流水线按 Windows + NVIDIA GPU 使用场景设计。CPU 可以跑部分测试和轻量逻辑，但 ASR、对齐和 VAD 推理建议使用 NVIDIA 显卡。

需要先安装这些基础环境：

| 环境 | 安装链接 | 说明 |
| --- | --- | --- |
| NVIDIA Driver | [NVIDIA 官方驱动下载](https://www.nvidia.com/Download/index.aspx) | 建议使用较新的 Game Ready 或 Studio Driver |
| Python | [Python Windows releases](https://www.python.org/downloads/windows/) | 推荐 Python 3.10 到 3.13，当前开发环境使用 Python 3.13 |
| FFmpeg | [FFmpeg Download](https://ffmpeg.org/download.html) | `ffmpeg` 和 `ffprobe` 必须能在 PowerShell 里直接访问 |
| Git | [Git for Windows](https://git-scm.com/downloads/win) | 用于 clone 仓库；下载 zip 的用户可以跳过 |
| PyTorch | [PyTorch Get Started](https://pytorch.org/get-started/locally/) | 进入页面选择 Windows、Pip、Python、CUDA，然后复制官方安装命令 |
| WebView2 Runtime | [Microsoft Edge WebView2](https://developer.microsoft.com/en-us/microsoft-edge/webview2/) | 可选；安装后 `pywebview` 会打开桌面窗口，否则用浏览器打开 |

安装后建议确认命令可用：

```powershell
ffmpeg -version
ffprobe -version
py --version
```

### 2. 获取项目并创建虚拟环境

```powershell
git clone <your-repo-url> jav-trans
cd jav-trans
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
```

如果本机没有 Python 3.13，可以把 `py -3.13` 换成已安装的版本，例如 `py -3.11`。

### 3. 安装 Python 依赖

先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择与你显卡驱动匹配的 CUDA wheel，并在当前虚拟环境里执行官方给出的命令。随后安装项目依赖：

```powershell
.venv/Scripts/python -m pip install -r requirements.txt
```

确认 PyTorch 能看到 CUDA：

```powershell
.venv/Scripts/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

输出 `True` 才表示完整 GPU 流水线可用。

### 4. 配置翻译服务

复制示例配置：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，至少填入：

```env
API_KEY=your_key_here
OPENAI_COMPATIBILITY_BASE_URL=https://api.deepseek.com
LLM_MODEL_NAME=deepseek-v4-pro
```

`.env` 只放跨任务持久配置。视频路径、输出目录、ASR 后端、字幕模式、batch、worker、保留临时文件等任务级参数都从 Web 页面传入。

`HF_ENDPOINT` 可设为 HuggingFace 镜像，例如 `https://hf-mirror.com`。必须包含 `http://` 或 `https://`，否则模型下载前会被拒绝。

### 5. 启动 Web 控制台

```powershell
.venv/Scripts/python run_web.py
```

默认地址是 `http://127.0.0.1:17321`。如果安装了 `pywebview` 和 WebView2，会打开本机桌面窗口；否则会用默认浏览器打开。端口可用 `JAVTRANS_PORT` 覆盖：

```powershell
$env:JAVTRANS_PORT = "18080"
.venv/Scripts/python run_web.py
```

### 6. 提交视频任务

1. 在页面选择视频文件或文件夹。
2. 选择 ASR 后端，推荐模型 `whisper-ja-anime-v0.3` 会排在第一位。
3. 设置输出模式：中文字幕或中日双语。
4. 按需要设置人名提示、翻译 batch、worker、输出目录、术语表、HF 镜像等参数。
5. 提交任务后，Web 通过 SSE 实时显示音频准备、ASR、对齐、F0、翻译、写入阶段。
6. 完成后从任务卡下载 SRT，或打开输出文件夹。

### 7. 临时文件和任务清理

成功任务在“保留临时文件”关闭时会直接删除当前 job 的一次性临时目录和翻译缓存，不会移动到 `agents/rm`。

Web 页面里的“清空已完成”会删除已完成、失败、已取消的任务记录，并删除对应的 `temp/web/jobs/<job_id>` 目录。关闭 Web 窗口时只清理一次性运行残留和 job audio 子目录；`models/`、`temp/hf-cache` 和 Web 任务状态会保留。

`agents/rm` 只用于工程维护时的手动归档，不属于应用运行时清理路径。

## Features

- Web 控制台：选择/拖入视频、配置 ASR 后端、翻译参数、输出目录、保留临时文件、重试断点续传。
- 多 ASR 后端：`anime-whisper`、`whisper-ja-anime-v0.3`、`whisper-ja-1.5b`、`qwen3-asr-1.7b`。
- 推荐 ASR：Web 后端列表会把 `whisper-ja-anime-v0.3` 排在第一位。
- 时间轴：ASR 文本后卸载 ASR 模型，再加载 Qwen3 Forced Aligner 做强制对齐，降低 8GB VRAM 压力。
- 翻译：OpenAI-compatible API，支持模型列表动态获取、batch、并发 worker、reasoning effort、术语表。
- 质量控制：ASR QC、字幕 QC、可选 quality report、可选 QC hard fail。
- 缓存与续跑：ASR checkpoint、`aligned_segments.json`、translation JSONL cache。
- 临时文件策略：成功任务默认直接清理一次性临时文件，保留模型和可复用运行缓存。

## Backend Debugging

Web 是主工作流。旧的 `src/main.py --input ...` CLI 已移除；后端调试通过测试、诊断脚本或直接调用 `run_asr_alignment_f0()` / `run_translation_and_write()` 完成。

```powershell
.venv/Scripts/python -m pytest tests/test_model_paths.py tests/web/test_jobs_api.py -q
.venv/Scripts/python scripts/asr_backends_compare.py video.mp4 --output-dir temp/reports --log
```

常规使用不要把视频路径、输出目录、ASR 后端、字幕模式或 batch/worker 写入 `.env`，这些任务级参数由 Web 表单传递。

## Architecture

```text
run_web.py
  ├─ starts FastAPI app from src/web/app.py
  ├─ opens pywebview window or system browser
  └─ registers runtime temp cleanup on process exit

src/web/
  ├─ routes/files.py       file/folder picking and output helpers
  ├─ routes/config.py      settings, model list, ASR backend ordering
  ├─ routes/jobs.py        job CRUD and retry/cancel
  ├─ routes/events.py      SSE stream
  ├─ pipeline_manager.py   GPU queue + translation queue
  └─ static/               Web UI assets

src/main.py
  ├─ run_asr_alignment_f0()
  ├─ run_translation_and_write()
  └─ two-stage backend pipeline API

Pipeline:
video
  -> ffmpeg audio extraction
  -> WhisperSeg/ffmpeg VAD
  -> selected ASR backend
  -> ASR model unload
  -> Qwen3 Forced Aligner
  -> F0 gender detection and subtitle merge
  -> OpenAI-compatible translation
  -> SRT + optional quality report
```

GPU-heavy work is serialized through the Web GPU queue. Translation runs through a separate queue and can use multiple API workers. The ASR model is unloaded before forced alignment to keep VRAM usage predictable on 8GB GPUs.

## ASR Backends

| Backend | Model | Timing mode | Notes |
| --- | --- | --- | --- |
| `whisper-ja-anime-v0.3` | `efwkjn/whisper-ja-anime-v0.3` | forced | Recommended in Web UI |
| `anime-whisper` | `litagin/anime-whisper` | forced | Conservative default config |
| `whisper-ja-1.5b` | `efwkjn/whisper-ja-1.5B` | forced | Larger Whisper-family option |
| `qwen3-asr-1.7b` | `Qwen/Qwen3-ASR-1.7B` | forced/native/hybrid infrastructure | Qwen ASR path |

Forced alignment uses `Qwen/Qwen3-ForcedAligner-0.6B` unless overridden.

## Models and Caches

First use downloads HuggingFace repos into canonical directories under `models/`:

```text
models/
├── litagin-anime-whisper/
├── efwkjn-whisper-ja-anime-v0.3/
├── efwkjn-whisper-ja-1.5B/
├── Qwen-Qwen3-ASR-1.7B/
├── Qwen-Qwen3-ForcedAligner-0.6B/
└── TransWithAI-Whisper-Vad-EncDec-ASMR-onnx/
```

The directory name is the HuggingFace repo id with `/` replaced by `-`. Runtime cache is separate:

- `temp/hf-cache`: HuggingFace hub/xet cache
- `temp/web`: Web job state and resumable artifacts
- `temp/jobs`, `temp/chunks`, `temp/recovery`: one-time job artifacts, cleaned after success unless the task keeps temp files

`HF_ENDPOINT` may be set in Web settings or `.env` to a mirror such as `https://hf-mirror.com`. It must include `http://` or `https://`; invalid values are rejected before model download.

## Configuration

Shared defaults live in `src/core/config.py`. `.env` is now for cross-task persistent settings that Web settings also reads/writes:

| Key | Purpose |
| --- | --- |
| `API_KEY` | Translation API key |
| `OPENAI_COMPATIBILITY_BASE_URL` | OpenAI-compatible endpoint |
| `LLM_MODEL_NAME` | Translation model name |
| `LLM_REASONING_EFFORT` | Reasoning effort: `low`, `medium`, or `max` |
| `TARGET_LANG` | Target translation language |
| `HF_ENDPOINT` | Optional HuggingFace mirror URL |
| `TRANSLATION_GLOSSARY` | `source→target, source→target` glossary |

Task-level settings come from the Web form and are carried through `JobSpec -> JobContext`: input files, output directory, ASR backend, ASR context, subtitle mode, skip translation, show gender, recovery, VAD threshold, translation batch size, translation workers, quality report, temp retention, and resume job id.

The Web advanced textarea can still pass expert-only environment-style overrides into one job, but those should not be committed to `.env.example` unless they are meant to be persistent defaults.

## Output

Default output goes next to the input video. The Web form can override the output directory per task.

| File | Description |
| --- | --- |
| `{video}.srt` | Final Chinese or bilingual subtitle |
| `{video}.ja.srt` | Japanese preview subtitle when translation is skipped |
| `{video}.quality_report.json` | Optional quality report |
| `temp/web/jobs/<job_id>/*.aligned_segments.json` | ASR/alignment resume cache |
| `temp/web/jobs/<job_id>/translation_cache.jsonl` | Translation resume cache |
| `temp/web/jobs.json` | Web job list and statuses |
| `log/*.run.log` | Optional persistent run logs from diagnostic runs or advanced Web overrides |

## Cleanup Behavior

The application directly deletes runtime temp files; it does not move them to `agents/rm`.

When a task finishes successfully and "保留临时文件" is off, backend cleanup removes current job audio, translation cache, matching ASR checkpoints, and empty runtime temp directories.

On Web process exit, `run_web.py` removes one-time runtime directories such as:

- `temp/jobs`
- `temp/chunks`
- `temp/recovery`
- `temp/jobs_*`
- `temp/chunk_*`
- `temp/recovery_*`
- `temp/pytest_*`
- `temp/smoke_api_*`
- `temp/web/jobs/<id>/audio`

Reusable caches are preserved:

- `temp/hf-cache`
- `temp/web` non-audio job state and resume artifacts
- `models/`

## Utility Scripts

The repository keeps only general-purpose backend diagnostics:

```powershell
.venv/Scripts/python scripts/asr_backends_compare.py video.mp4 --output-dir temp/reports --log
.venv/Scripts/python scripts/benchmark_asr.py audio.wav --backend whisper-ja-anime-v0.3 --json-out temp/asr_benchmark.json
```

Ad-hoc local benchmark scripts and the old root `run.py` wrapper have been removed from the working tree.

## Packaging

PyInstaller onedir packaging is available for the Web app:

```powershell
.venv/Scripts/pip install pyinstaller pywebview
.venv/Scripts/python -m PyInstaller packaging/javtrans-web.spec
```

The packaged app still downloads models into `models/` on first use. Windows needs WebView2 Runtime, usually available with Microsoft Edge.

## Tests

```powershell
.venv/Scripts/python -m pytest -q
```

The test configuration uses `temp/pytest_verify` as pytest basetemp. Keep `temp/.gitkeep` so the parent directory exists in a fresh clone.

## Repository Hygiene

Tracked source should stay small. The following are intentionally ignored:

- `.env`, `.env.*` except `.env.example`, `.venv/`, `.claude/`
- `models/*` except `models/README.md`
- `video/`
- `temp/*` except `temp/.gitkeep`
- `agents/`
- `log/`, `reports/`, `build/`, `dist/`
- Python bytecode and pytest caches

Before uploading to Git, check that no private API keys, local videos, downloaded models, or agent scratch files are staged.

Suggested GitHub repository description:

```text
Windows 本地视频字幕生成工具，支持 ASR 对齐、F0 性别检测和中日字幕翻译。
```

## Vendored Code

WhisperSeg VAD code is adapted from WhisperJAV and kept under:

- `src/vad/whisperseg/whisperseg_core.py`
- `src/vad/whisperseg/postprocess.py`
- `src/vad/whisperseg/LICENSE.WhisperJAV`

Follow the original license terms for that vendored code and the separate licenses/terms of all external models and APIs.
