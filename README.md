# JAVTrans

Windows 本地视频字幕生成工具。一键将视频转化为中日双语/中文字幕。
项目提供直观的网页控制台，集成音频提取、语音识别（ASR）、强制对齐、音高性别检测及大语言模型（LLM）翻译流水线，核心视频与音频处理均在本地显卡进行。

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

Release 版已包含 Python 运行环境、FFmpeg和默认 ASR 模型 `efwkjn/whisper-ja-anime-v0.3`。首次使用仍需要在页面的“翻译设置”中填写 API Key / Base URL / 模型名。其他 ASR 模型和辅助模型会在需要时下载到 exe 同目录的 `models/`。

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
```

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
- **断点续传与缓存**：支持任务意外中断后的恢复，长视频处理不再担惊受怕。
- **懂二次元的识别模型**：内置了多种针对日语/动漫优化的语音识别模型（推荐默认首选的 `whisper-ja-anime-v0.3`）。
- **智能性别区分**：能根据声音特征分辨男女角色，让对话翻译更加生动准确。
- **高自由度翻译**：支持接入任何兼容 OpenAI 接口的大语言模型，甚至可以设置特定词汇的“术语表”。

## 💡 给开发者的说明

如果您是开发者并希望深入了解或修改项目：
- 核心后端逻辑主要位于 `src/main.py` 和 `src/core/` 目录。
- Web 接口和页面由 `src/web/` 提供，采用 FastAPI。
- 本项目引入的部分第三方代码（例如 `src/vad/whisperseg`）保留了其原始的许可证，请遵循相应协议。

### 构建 Windows Release

在已安装依赖的 `.venv` 环境下运行：

```powershell
.\packaging\build_windows.ps1 -Clean
```

构建产物位于 `dist\JAVTrans\JAVTrans.exe`。打包细节见 `packaging/README.md`。
由于包内包含 PyTorch/CUDA 运行库和默认 ASR 模型，发布目录会达到数 GB；上传 GitHub Release 时通常需要分卷压缩或改用外部大文件分发。
