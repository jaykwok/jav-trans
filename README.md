# jav-trans

jav-trans 是一个面向 Windows 和 NVIDIA 显卡的本地字幕生成工具。选择视频后，它会自动完成日语语音识别、字幕时间轴、可选翻译和质量检查，输出日文、译文或中日双语 SRT。

视频、音频、语音识别和时间轴始终在本机处理。只有选择 API 翻译时，字幕文本才会发送到你配置的服务；仅日文和本地翻译模式不会上传媒体或字幕。

![网页控制台主界面](docs/images/ui-web-console.png)

## 主要功能

- 本地网页控制台：选择视频、调整设置、查看进度并打开结果。
- 日语识别与精细时间轴：使用 Qwen3-ASR 1.7B 和 CTC 对齐生成字幕。
- 三种输出：仅日文、仅译文、中日双语。
- 两种翻译方式：OpenAI 兼容 API，或本地 Hy-MT2-7B Q4。
- 质量报告：标出时间轴、字幕布局和翻译中的可疑项，方便按时间码复查。
- 断点与缓存：失败后可以复用已有识别结果，不必总是从头开始。

## 运行要求

- Windows 10/11。
- NVIDIA 独立显卡；8GB 及以上显存推荐，当前 ASR 模型要求至少 6144MiB 物理显存。
- Windows 发布包已包含 FFmpeg Shared，无需另行下载或安装。首次启动需要联网安装 Python 依赖；ASR 模型约 3.9GB，本地翻译模型约 4.6GB。
- 将发布包解压到普通可写目录，并预留至少 15GB 空间。

正式 ASR 不会静默回退到 CPU。CUDA 不可用、驱动过旧或显存不足时，任务会明确报错。

## 快速开始

1. 从 [Releases](https://github.com/jaykwok/jav-trans/releases) 下载并解压 Windows 发布包。
2. 双击 `jav-trans.exe`。首次启动会自动安装依赖；中断后重新运行即可继续。
3. 在网页控制台中选择一个或多个视频。
4. 选择字幕模式和翻译方式，按需填写 API Key 或本地模型设置。
5. 点击“开始任务”，完成后从任务卡打开 SRT 和可选的质量报告。

发布包自带 FFmpeg Shared。普通用户无需单独配置 FFmpeg；只有从源码运行时才需要自行安装。

首次使用建议勾选“不翻译（仅日文字幕）”跑一个短视频，以确认 CUDA、模型下载和输出目录都正常。

## 选择处理模式

| 模式 | 是否发送字幕文本 | 适合场景 |
| --- | --- | --- |
| 仅日文 | 否 | 最快验证环境，或只需要日文 SRT |
| 本地翻译 | 否 | 隐私优先、零 API 成本的中文草稿 |
| API 翻译 | 是 | 需要术语表、全片上下文和更稳定的成品质量 |

API 模式默认使用 OpenRouter。通常只需填写 API Key；如果改用其他服务，Base URL 与模型名必须配套：OpenRouter 使用 `厂商/模型` 形式，DeepSeek 官方 API 使用裸模型名。

本地翻译固定使用 `Hy-MT2-7B-Q4_K_M.gguf`，首次使用时自动下载。先安装 llama.cpp：

```powershell
winget install -e --id ggml.llamacpp
```

然后在网页控制台选择“本地 Hy-MT2”。本地后端逐句翻译，不使用 API 模式的术语表、角色参考和全片上下文；它更适合作为隐私优先的草稿方案。NVIDIA 用户也可以下载 llama.cpp CUDA 版本，并在设置中填写 `llama-server.exe` 路径。

## 输出与隐私

- SRT 和质量报告写入所选输出目录；仅日文模式会生成 `<视频名>.ja.srt`。
- `models/` 保存已下载模型，`tmp/` 保存任务状态、缓存和日志。
- 成功任务会清理一次性临时文件，但保留模型与跨任务 ASR 缓存。
- API 翻译只发送字幕文本，不发送视频或音频。具体数据处理规则仍取决于你选择的 API 服务。
- 质量报告用于辅助人工复查，不保证每条识别或翻译都完全正确。

删除运行中的任务会先请求取消；任务进入“已取消”后再次删除，才会清理该任务的临时目录。

## 常见问题

### 程序窗口无法打开

在程序目录打开 PowerShell，运行：

```powershell
.\jav-trans.exe --doctor
```

需要保留启动日志时使用 `--keep-console`；需要重装运行环境时使用 `--reinstall`。

### 下载速度很慢

在网页控制台的“识别设置”中填写代理，或启动时传入代理：

```powershell
.\jav-trans.exe --proxy http://127.0.0.1:7890
```

### CUDA 或显存报错

先关闭其他占用显卡的程序并更新 NVIDIA 驱动。保持默认的 `ASR_BATCH_SIZE=auto`，程序会在 OOM 后自动降低批大小；当前没有更小的 ASR 模型可切换。

### API 翻译失败

确认 API Key、Base URL 和模型名属于同一服务。OpenRouter 的模型名通常形如 `deepseek/deepseek-v4-flash`；DeepSeek 官方地址 `https://api.deepseek.com` 使用 `deepseek-v4-flash`。其他兼容服务请使用其文档给出的地址和模型名。

### 如何反馈长任务问题

日志位于 `tmp/log/<job_id>/`。反馈时请附上 `.run.log`、对应 SRT 和质量报告，并先移除 API Key、视频路径等隐私信息。

## 从源码运行

源码运行还需要 Git、uv、符合 `pyproject.toml` 约束的 Python，以及 FFmpeg Shared。TorchCodec 依赖 FFmpeg 共享 DLL；Windows 上应确保 Shared 版位于 `PATH`，且不要让静态版排在它前面。

```powershell
winget install --id Gyan.FFmpeg.Shared --exact

git clone https://github.com/jaykwok/jav-trans.git
cd jav-trans

uv venv
uv sync

$env:PYTHONIOENCODING="utf-8"
uv run --no-sync python launcher.py
```

浏览器默认打开 `http://127.0.0.1:2233`。端口被占用时会自动选择下一个可用端口，并在启动日志中显示实际地址。请使用 `uv` 安装项目依赖，不要用 `pip install` 逐个安装，以免误装 CPU 版 PyTorch。

## 更多文档

- [历史与实验记录](docs/HISTORY.md)
- [模型目录说明](models/README.md)
- [打包说明](packaging/README.md)

致谢：[WhisperJAV](https://github.com/meizhong986/WhisperJAV) 为本项目早期路线提供了重要参考。
