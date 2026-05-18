# JAVTrans

Windows 本地视频字幕生成工具。一键将视频转化为中日双语/中文字幕。
项目提供直观的网页控制台，集成音频提取、语音识别（ASR）、强制对齐、音高性别检测、翻译前 ASR 噪声过滤及大语言模型（LLM）翻译流水线，核心视频与音频处理均在本地显卡进行。

**💖 致谢与来源提示：**
本项目在核心设计思路和部分代码实现（特别是 VAD 分段处理部分）上，参考并借鉴了优秀的开源项目 **[WhisperJAV](https://github.com/a63n/WhisperJAV)**。特此对 WhisperJAV 作者及其贡献表示衷心感谢！

---

## 🚀 快速开始

本项目主要为 **Windows 搭配 NVIDIA 独立显卡**的用户设计，以确保生成速度和质量。

### 方式 A：下载 Release 版

如果你只是想直接使用，优先下载 GitHub Releases 中的 Windows 压缩包。解压后运行：

```text
JAVTrans.exe
```

Release 版已包含 Python 运行环境、FFmpeg、默认 ASR 模型 `efwkjn/whisper-ja-anime-v0.3`，以及默认流程需要的 WhisperSeg VAD、`openai/whisper-base` 特征提取器和 Qwen forced aligner。首次使用仍需要在页面的“翻译设置”中填写 API Key / Base URL / 模型名。其他 ASR 模型会在需要时下载到 exe 同目录的 `models/`。
如果任务启用了运行日志，日志会写到 exe 同目录的 `temp/log/`，反馈问题时可以一并提交这个目录下的 `.run.log` 文件。

注意：Release 版不内置 Microsoft Edge WebView2 Runtime。大多数 Windows 10/11 已自带；如果无法打开窗口，请到 [Microsoft Edge WebView2 官方下载页](https://developer.microsoft.com/en-us/microsoft-edge/webview2/) 安装 Evergreen Runtime。联网环境可选 Evergreen Bootstrapper；离线环境可下载 Evergreen Standalone Installer 的 x64 版本。

### 方式 B：源码运行

### 1. 准备环境

开始之前，请确保您的电脑已准备好以下基础环境：

- **NVIDIA 显卡驱动**：请确保驱动为较新版本。
- **Python**：推荐安装 [Python 3.10 - 3.13](https://www.python.org/downloads/windows/)（安装时请务必勾选 "Add python.exe to PATH"）。
- **FFmpeg**：视频处理必需工具。请 [下载 FFmpeg](https://ffmpeg.org/download.html) 并配置到系统环境变量，确保在命令行中输入 `ffmpeg` 不会报错。
- **Git**（可选）：用于下载本项目，如果没有 Git，您也可以直接在 GitHub 页面点击 "Download ZIP" 下载后解压。

### 2. 下载并安装

打开命令行终端（如 PowerShell），逐行输入以下命令：

```powershell
# 1. 下载项目（如果下载了 ZIP 文件，请解压后 cd 进入项目文件夹）
git clone https://github.com/jaykwok/jav-trans.git
cd jav-trans

# 2. 创建并激活独立的 Python 虚拟环境
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip

# 3. 安装带有 CUDA (显卡) 支持的 PyTorch
# 注意：以下命令适用于常见的 CUDA 12.1，如果您的环境不同，请去 https://pytorch.org/get-started/locally/ 获取对应命令
.venv\Scripts\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 安装项目的其他依赖
.venv\Scripts\python -m pip install -r requirements.txt
```

### 3. 配置翻译服务

本项目使用各大平台的 AI 接口来进行高质量字幕翻译。请在项目目录下找到 `.env.example` 文件，将其复制一份并重命名为 `.env`。

用记事本打开 `.env` 文件，填入你的翻译 API 信息，例如：

```env
API_KEY=你的翻译_API_KEY
OPENAI_COMPATIBILITY_BASE_URL=https://api.deepseek.com
LLM_MODEL_NAME=deepseek-chat
LLM_API_FORMAT=chat
LLM_REASONING_EFFORT=xhigh
TARGET_LANG=简体中文
```

`LLM_API_FORMAT` 默认为 `chat`，走 OpenAI Chat Completions 兼容格式；需要使用 OpenAI Responses API 兼容格式时改为 `responses`。
`.env` 主要保存跨任务持久配置，例如 API、模型名、HF 镜像、术语表和演员名提示。视频路径、输出目录、ASR 后端、字幕模式、batch/worker、是否保留临时文件等任务级参数由网页表单传入，不建议写进 `.env`。

*(可选)* 如果国内网络下载 AI 模型速度较慢，可以在 `.env` 中加入这行来加速下载：
```env
HF_ENDPOINT=https://hf-mirror.com
```

### 4. 运行工具

所有准备工作完成后，在项目目录下运行以下命令启动工具：

```powershell
.venv\Scripts\python run_web.py
```
启动后，程序会自动打开网页（默认地址为 `http://127.0.0.1:17321`）。
在页面上，您只需：
1. **选择视频文件**
2. **确认翻译和输出设置**
3. **点击提交**，即可坐和放宽，等待字幕生成完毕！

---

## 🛠️ 主要功能特点

- **小白友好的网页界面**：所有的模型选择、字幕格式、并发设置都可以通过网页轻松配置。
- **断点续传与多层缓存**：支持音频缓存、ASR checkpoint、`aligned_segments.json` 复用、翻译 cache，以及独立的 VAD/chunk 边界缓存；只改 ASR prompt/token 参数时可复用 VAD 切分结果。
- **懂二次元的识别模型**：支持 `anime-whisper`、`whisper-ja-anime-v0.3`、`whisper-ja-1.5b`、`qwen3-asr-1.7b`。引擎默认与 Web 推荐首选均为 `whisper-ja-anime-v0.3`。
- **WhisperSeg VAD + 长 chunk 流程**：默认使用 WhisperSeg，阈值 `0.35`；开启 VAD chunk packing，将相邻语音段打包成更适合 Whisper/forced alignment 的长 chunk。
- **自适应低幻觉 ASR 策略**：ASR QC 默认且唯一使用 adaptive precision。高 `no_speech_prob`、高压缩率、异常字符密度、重复循环、上下文泄漏、乱码和生成异常会硬丢弃；低风险真实对白的低 `avg_logprob` 会自适应放宽，并写入 quality report 审计。
- **ASR generation budget 防溢出**：Whisper 系列会根据 decoder 窗口、forced decoder ids、prompt tokens 动态裁剪 prompt 和 `max_new_tokens`，质量报告会统计 overflow/error/timeout/quarantine；生成失败不再通过温度重试或 recovery 补写内容。
- **翻译前噪声过滤**：在提交给 LLM 前过滤空字幕、纯引号片段、纯英文幻觉 token 和纯特殊符号片段，减少无效翻译请求。
- **智能性别区分**：forced alignment 后执行词级 F0 性别检测，并根据 gender turn 重新切分字幕，让对话翻译更加稳定。
- **高自由度翻译**：支持接入任何兼容 OpenAI 接口的大语言模型，甚至可以设置特定词汇的“术语表”。
- **质量报告与运行日志**：可在网页中开启运行日志和质量报告，便于复测 ASR generation 计数、字幕质量 warning 和性能耗时。

## ⚙️ 当前默认流程

当前主流水线为：

```text
视频 -> 音频准备 -> WhisperSeg VAD -> VAD chunk packing -> ASR -> Adaptive Precision QC -> Forced Alignment
-> 词级 F0 性别检测 -> gender turn 重切段 -> 翻译前 ASR 噪声过滤
-> LLM 翻译 -> SRT / quality report
```

当前 ASR 以“少但准”为默认目标：不确定的内容宁可不出现在字幕里，也不把疑似幻觉交给后续对齐和翻译。旧的 ASR recovery、温度 fallback、prompt overflow retry 已从后端移除；保留的 timestamp/alignment fallback 只用于给已确认文本补时间轴，不会改写或新增 ASR 文本。

常用缓存位置：

- `temp/vad-cache/`：VAD/chunk 边界缓存，只绑定音频指纹、VAD 参数和 chunk/drop/merge 参数，不绑定 ASR prompt/token 参数。
- 当前任务临时目录：音频缓存、ASR checkpoint、`aligned_segments.json`、翻译 cache、质量报告和运行日志。
- `models/`：HuggingFace 模型缓存。

## 💡 给开发者的说明

如果您是开发者并希望深入了解或修改项目：
- 核心后端逻辑位于 `src/main.py`、`src/core/`、`src/pipeline/`、`src/whisper/`、`src/llm/`、`src/audio/` 等子模块。
- Web 接口和页面由 `src/web/` 提供，采用 FastAPI。
- 默认配置集中在 `src/core/config.py`，本地 `.env` 只覆盖跨任务持久配置；Web 任务级参数通过 `JobSpec -> JobContext` 显式传入后端。
- 本项目引入的部分第三方代码（例如 `src/vad/whisperseg`）保留了其原始的许可证，请遵循相应协议。

### 构建 Windows Release

在已安装依赖的 `.venv` 环境下运行：

```powershell
.\packaging\build_windows.ps1 -Clean
```

构建产物位于 `dist\JAVTrans\JAVTrans.exe`。打包细节见 `packaging/README.md`。
由于包内包含 PyTorch/CUDA 运行库、默认 ASR 模型和默认流程辅助模型，发布目录会达到数 GB；上传 GitHub Release 时通常需要分卷压缩或改用外部大文件分发。
