# JAVTrans

JAVTrans 是一个面向 Windows + NVIDIA 显卡的本地字幕生成工具。它把视频处理成中文字幕或中日双语字幕，提供网页控制台，并把音频准备、VAD 分段、ASR、强制对齐、F0 性别检测、字幕时间轴归一化、LLM 翻译和质量报告串成一条流水线。

项目目标很明确：本地完成视频/音频/ASR 相关重计算，LLM 只负责翻译，不承担 ASR 误听修复、画面脑补或剧情改写。

本项目在核心设计思路和部分代码实现，尤其是 VAD 分段处理上，参考并借鉴了 [WhisperJAV](https://github.com/a63n/WhisperJAV)。感谢 WhisperJAV 作者及其贡献。

---

## 快速开始

### 方式 A：使用 Release 版

如果只是直接使用，优先下载 GitHub Releases 中的 Windows 压缩包。解压后运行：

```text
JAVTrans.exe
```

Release 版已包含 Python 运行环境、FFmpeg、默认 ASR 模型 `efwkjn/whisper-ja-anime-v0.3`，以及默认流程需要的 whisperseg-adaptive VAD、`openai/whisper-base` 特征提取器和 Qwen forced aligner。首次使用仍需要在页面的“翻译设置”中填写 API Key、Base URL 和模型名。其他 ASR 模型会在需要时下载到 exe 同目录的 `models/`。

Release 版不内置 Microsoft Edge WebView2 Runtime。大多数 Windows 10/11 已自带；如果无法打开窗口，请安装 [Microsoft Edge WebView2 Evergreen Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/)。

任务启用运行日志后，日志会写入 exe 同目录的 `temp/log/`。反馈问题时可以一并提交对应 `.run.log`。

### 方式 B：源码运行

源码运行适合开发、调参和验证。推荐环境：

- Windows 10/11
- NVIDIA 独立显卡和较新的驱动
- Python 3.13（推荐）
- FFmpeg，并确保命令行能直接执行 `ffmpeg`
- Git，可选

安装步骤：

```powershell
git clone https://github.com/jaykwok/jav-trans.git
cd jav-trans

uv venv
uv pip install --upgrade pip

# 推荐 Python 3.13 搭配 PyTorch Stable 2.7.0+ cu128（CUDA 12.8）。
# 其他 CUDA 环境请以 PyTorch 官网命令为准。
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

uv pip install -r requirements.txt
```

Linux / WSL 源码运行时，默认启用的 TEN VAD fallback 还需要系统运行库 `libc++.so.1` 和 `libc++abi.so.1`。Ubuntu / WSL 可安装：

```bash
sudo apt update
sudo apt install -y libc++1 libc++abi1
```

Linux / WSL 的虚拟环境仍由 `uv` 管理在当前项目下。安装完成后可验证 TEN VAD：

```bash
uv run --no-sync python -c "import ctypes.util; print(ctypes.util.find_library('c++')); from ten_vad import TenVad; print('TEN VAD ok')"
```

复制 `.env.example` 为 `.env`，填写翻译服务配置：

```env
API_KEY=你的翻译_API_KEY
OPENAI_COMPATIBILITY_BASE_URL=https://api.deepseek.com
LLM_MODEL_NAME=deepseek-chat
LLM_API_FORMAT=chat
LLM_REASONING_EFFORT=xhigh
TARGET_LANG=简体中文
```

术语表使用 `原文-译文` 格式，支持逗号或换行分隔。原文和译文两侧空格会被自动清理：

```env
TRANSLATION_GLOSSARY=ちんぽ-肉棒, チンポ-肉棒
```

国内网络下载模型较慢时可设置：

```env
HF_ENDPOINT=https://hf-mirror.com
```

启动网页控制台：

```powershell
uv run --no-sync python run_web.py
```

默认地址为 `http://127.0.0.1:17321`。页面中选择视频、确认翻译和输出设置，然后提交任务即可。

---

## 功能概览

- 网页控制台：文件选择、ASR 后端、字幕模式、翻译设置、并发 worker、质量报告和临时文件保留都可以在页面里操作。
- 多层缓存：支持音频缓存、ASR checkpoint、`aligned_segments.json`、翻译 cache，以及独立的 VAD/chunk 边界缓存。
- 默认 ASR：引擎默认和 Web 推荐均为 `whisper-ja-anime-v0.3`。
- 支持 ASR 后端：`anime-whisper`、`qwen3-asr-1.7b`、`whisper-ja-1.5b`、`whisper-ja-anime-v0.3`。
- 默认 VAD：`whisperseg-adaptive`，基础阈值 `0.35`，会根据整段 speech ratio 自适应调一次阈值。
- 实验 VAD：`fusion_lite`、`fusion_lite_boost`、`fusion_lite_sigmoid`，用于对比 whisperseg-adaptive 与简单特征融合策略。
- Adaptive Precision ASR QC：硬拒绝明显幻觉，低风险真实对白可自适应放宽低 `avg_logprob`。
- Forced alignment + F0：ASR 后进行词级强制对齐，再做 F0 性别检测和 gender turn 重切段。
- 翻译前 cue plan：LLM 翻译前先固定字幕时间轴和 cue 数量，SRT writer 不再改变时间轴。
- LLM 翻译：支持 OpenAI-compatible Chat Completions 和 Responses API，保留 reasoning effort、API 格式、目标语言、术语表、worker 数这些可手动配置项。
- 质量报告：默认输出 Markdown，同时保留 JSON sidecar 方便自动化评测。

---

## 当前默认流程

主流水线：

```text
视频 -> 音频准备 -> whisperseg-adaptive VAD -> VAD chunk packing -> ASR -> Adaptive Precision QC
-> Forced Alignment -> 词级 F0 性别检测 -> gender turn 重切段
-> 翻译前 ASR 噪声过滤 -> 翻译前 cue plan 时间轴归一化
-> LLM 逐 cue 翻译 -> SRT / quality report
```

当前 ASR 以“少但准”为默认目标。不确定内容宁可不出现在字幕里，也不把疑似幻觉交给后续对齐和翻译。旧的 ASR recovery、temperature fallback、prompt overflow retry 已移除；主 VAD 空结果会直接跳过 ASR；timestamp/alignment fallback 只用于给已确认文本补时间轴，不改写或新增 ASR 文本。

### ASR 与 VAD

默认 VAD 是 `whisperseg-adaptive`。公开可选的 VAD 路线只保留 `whisperseg-adaptive` 和实验 `fusion_lite*` 系列。Silero 只作为 fusion-lite 系列内部 speech prior，不作为独立主 VAD 暴露。

主 VAD 初始化或推理失败会直接抛错并进入 Web 日志。主 VAD 返回空结果是合法“无语音”结果，不会 fallback 成整段音频转写。

`ASR_LONG_CHUNK_PROFILE=on` 时强制开启 VAD chunk packing 与 post-alignment F0：

```env
ASR_CHUNK_PACKING_ENABLED=1
F0_GENDER_POST_ALIGNMENT=1
```

Whisper 系列 generation budget 会根据 decoder 窗口、forced decoder ids、prompt ids 和 `WHISPER_MAX_NEW_TOKENS` 动态裁剪；Qwen 不套 Whisper 448 decoder 窗口。

### Fusion-lite VAD

`fusion_lite` 受 FusionVAD 的简单特征融合思路启发，但不引入 pyannote，也不训练模型。它以 whisperseg 作为候选主信号，Silero 只提供辅助 speech prior，再叠加 RMS、spectral flux 和 duration。

线性基线公式：

```text
speech_score =
  0.45 * whisperseg_score
+ 0.25 * silero_overlap_ratio
+ 0.15 * rms_score
+ 0.10 * spectral_flux_score
+ 0.05 * duration_score
```

仅当 `speech_score < 0.45` 且 `silero_overlap_ratio < 0.05` 时丢弃候选。`fusion_lite_boost` 和 `fusion_lite_sigmoid` 是固定配方后缀后端，不新增模式配置，便于后续确定保留项后删除其他实验模式。

### 字幕时间轴

LLM 翻译前必须先生成稳定 cue plan。流程会通过 `ffprobe` 读取真实 `avg_frame_rate` / `r_frame_rate`，失败时按 `30000/1001`，即 29.97fps 兜底。

cue plan 负责：

- 基于 forced alignment 词级时间轴排序。
- 合并双语短句。
- 软拆长字幕。
- 裁剪或合并 overlap。
- 固定保留 2 帧字幕 gap。

默认字幕约束：

```env
SUBTITLE_SOFT_MAX_S=5.5
MAX_SUBTITLE_DURATION=6.5
ASR_MERGE_HARD_MAX_DURATION=9.0
```

相邻短块合并按帧数判断，而不是硬编码秒数。普通短块合并默认允许 `gap <= 6 frames` 且合并后 `duration <= 120 frames`。跨 F0 gender guard 时，只允许边界日文文本重叠的极短尾巴：后一 cue `<= 20.5 frames`、gap `<= 2.5 frames`，合并后 `gender=None`。speaker guard 仍然是硬边界。

最终写入 SRT、`bilingual.json` 和 quality report 的都是同一份已归一化 cue。

### 翻译策略

当前翻译 prompt version：`v2.7`。

LLM 只负责逐 cue 翻译、遵守术语表和人名罗马音规则。全片上下文只用于翻译连贯、指代判断、口吻一致和术语一致，不授权根据上下文修正 ASR 误听、同音词、上下文漂移、术语漂移或被切断半句。

LLM 输入中仍会保留 `[M]` / `[F]` 声学标签，帮助判断语气和对话切换；可见性别标签由本地规则在输出前移除，最终 SRT 不输出 `[M]` / `[F]`。

2026 年检索到的字幕翻译质量实践与当前路线一致：全片或多行上下文、术语表或术语记忆、结构化输出、窄范围后编辑能带来主要收益。表达强度不是 temperature，而是 prompt 层面的风格约束：保留粗俗程度、调情/命令/羞耻语气、情绪强弱和短促呻吟，不净化成书面弱表达。

当前固定内部采样：

```text
temperature=0.2
top_p=0.9
```

它们用于降低随机性和术语漂移，不作为“表达强度”控制项。

翻译默认保留：

- fixed full-JSON prefix
- prefix warmup
- 全片 glossary 预抽取
- 翻译后长度异常 repair

repair pass 只处理译文长度异常，不做 ASR/剧情修复。

翻译 batch 大小不作为前端/API/env 配置项。后端按 cue 数和 worker 数自动计算：

```text
batch = min(cue_count, clamp(25 + (25 - 10) * worker_count * 3, 25, 200))
```

主流程仍优先使用 full-JSON prefix，而不是退回纯滑窗。

---

## 配置边界

`.env` 只保存跨任务持久配置和默认偏好，例如：

- `API_KEY`
- `OPENAI_COMPATIBILITY_BASE_URL`
- `LLM_MODEL_NAME`
- `LLM_API_FORMAT`
- `LLM_REASONING_EFFORT`
- `TARGET_LANG`
- `TRANSLATION_GLOSSARY`
- `HF_ENDPOINT`
- `ASR_CONTEXT`
- `ASR_BACKEND`
- `ASR_VAD_BACKEND`
- `ASR_VAD_ADAPTIVE`
- `ASR_LONG_CHUNK_PROFILE`
- `ASR_CHUNK_PACK*`
- `WHISPERSEG_*`
- `SILERO_VAD_*`
- `FUSION_VAD_*`
- `VAD_CHUNK_CACHE_*`
- `ASR_QC_ADAPTIVE_*`
- `F0_GENDER_*`

视频路径、输出目录、字幕模式、worker、是否保留临时文件等任务级参数由 `JobSpec -> JobContext` 显式传入后端，不再依赖全局 `.env` 热覆盖。翻译 batch 大小由后端运行时公式自动计算，不是配置项。

Web 设置行为：

- 演员名/人名提示 `ASR_CONTEXT` 是持久设置，打开页面时从 `/api/settings` 恢复。
- 用户手动清空后提交会清空持久值。
- 前端不提供单独“保存设置”按钮，提交任务即保存当前表单配置。
- `OPENAI_COMPATIBILITY_BASE_URL` 是 OpenAI-compatible API 配置名，保留不改。

---

## 路径与缓存

- `models/`：HuggingFace 模型缓存。首次运行把 repo 下载到 `models/<namespace>-<repo>/`。
- `temp/vad-cache/`：VAD/chunk 边界缓存，只绑定音频指纹、VAD 参数和 chunk/drop/merge 参数，不绑定 ASR prompt/token 参数。
- `temp/jobs/`：Web 任务临时目录，包含音频缓存、ASR checkpoint、`aligned_segments.json`、翻译 cache 等。
- `temp/log/`：高级项启用 `RUN_LOG_ENABLED=1` 后写入运行日志。
- `video/<视频名>/`：对应视频的字幕、质量报告、历史对比报告和人工质检报告目录。质量报告以 `.md` 为主产物，同时保留 `.json` sidecar；Web 勾选质量报告时默认写到这里。
- `subtitle_qc/`：独立字幕人工质检工具，默认把单视频 HTML/JSON 报告写到 `video/<视频名>/subtitle_qc/`。批量 VAD 矩阵的每视频质量报告和 reference eval 也写到 `video/<视频名>/subtitle_qc/<任务名>/`，跨视频 summary 写到 `video/subtitle_qc/<任务名>/`；运行日志和中间件仍在 `agents/temp/subtitle_qc/`。具体命令见 `subtitle_qc/README.md`。

成功运行后默认删除一次性 job 临时目录；保留下次可复用的运行缓存，例如 `models/`、`temp/vad-cache/` 和 `temp/web` 状态。Web“保留临时文件”仅用于调试当前任务。

---

## 开发说明

后端调试入口：

- 运行测试。
- 使用诊断脚本。
- 直接调用 `run_asr_alignment_f0()` / `run_translation_and_write()`。

主要代码位置：

- `src/main.py`：主流程编排。
- `src/core/`：配置和任务上下文。
- `src/pipeline/`：音频、缓存、输出、质量报告和阶段日志。
- `src/whisper/`：ASR 后端和转写流程。
- `src/vad/`：VAD 后端。
- `src/llm/`：翻译 prompt、cache、glossary、API patch 和 translator。
- `src/subtitles/`：SRT writer、字幕选项和字幕 QC。
- `src/web/`：FastAPI 接口和静态前端。

开发环境约定：

- Windows 生产目标：RTX 4060 Ti 8GB，模型串行分时加载，阶段结束后卸载并清 CUDA cache。
- 开发和测试时统一使用当前工作目录下由 `uv venv` 创建的 `.venv`。
- Python 命令写成 `uv run --no-sync python ...`；pip 命令写成 `uv pip ...`，避免区分 Windows 和 Linux 虚拟环境路径。
- 临时运行产物放在 `temp/` 或 `agents/temp/`，正式报告放在对应的 `video/<视频名>/` 目录。
- 删除或归档本地文件时移动到 `agents/rm/`。

构建 Windows Release：

```powershell
.\packaging\build_windows.ps1 -Clean
```

构建产物位于 `dist\JAVTrans\JAVTrans.exe`。打包细节见 `packaging/README.md`。由于包内包含 PyTorch/CUDA 运行库、默认 ASR 模型和默认流程辅助模型，发布目录会达到数 GB；上传 GitHub Release 时通常需要分卷压缩或改用外部大文件分发。

本项目引入的部分第三方代码，例如 `src/vad/whisperseg`，保留其原始许可证，请遵循相应协议。

---

## 当前 Backlog

暂无

---

## 任务历史与验证记录

编号规则：

- `Rxx`：近期任务，按完成时间顺序编号。
- `Vxx`：关键验证记录，绑定一个或多个近期任务。
- `Hxx`：历史归档任务。

### 近期任务

| 编号 | 内容 | 验收 |
|------|------|------|
| R01 | 全量审计修复：任务级 env 覆盖、aligned cache scope、ASR/字幕/quality 参数运行时化、翻译 cancel_event 透传 | 基线 315 passed, 5 skipped；完成后逐步增至 334+ passed |
| R02 | 第二轮后端审计：ASR/aligned cache signature、`.env.example` 默认、SubtitleOptions、Web retry/cancel、stream timeout、Protocol 补齐 | 343 passed, 5 skipped |
| R03 | ASR generation budget + ONNX CUDA runtime + VAD/chunk cache | 359 passed, 5 skipped；匿名样片 A anime-whisper 全量中日双语 649.54s，whisperseg CUDA VAD/切块 9.32s，ASR generation overflow/error 为 0 |
| R04 | 删除 ASR recovery / temperature fallback / prompt overflow retry，并清理前端旧 ASR Recovery 控件；早期固定阈值 precision 方案后续被 adaptive-only 替换 | 后端全量 365 passed, 5 skipped；前端/Web 定向 13 passed |
| R05 | adaptive precision ASR 默认化：保留硬幻觉拒绝，低风险低 `avg_logprob` 对白自适应放宽 | 定向回归 68 passed；匿名 5min smoke：adaptive drops 2，overflow/error/timeout/quarantine 为 0 |
| R06 | 默认 ASR 切为 `whisper-ja-anime-v0.3`；新增 `video/test` 通用测试集评测工具；删除 strict/normal ASR 精度模式，只保留 adaptive precision | 全量 373 passed, 5 skipped |
| R07 | 本地 `.env` 适配当前默认流程并按同类参数归类注释；文档同步 `.env` 边界 | dotenv 解析通过；关键 adaptive/default ASR 配置齐全；旧 strict 配置不存在 |
| R08 | 新增 Silero / hybrid VAD 实验并完成取舍：hard/soft gate 过度依赖 Silero，后续从当前代码与公开配置中移除 | 匿名样片 B 前 5 分钟历史 smoke：hybrid hard 漏太多，hybrid soft 改善但仍不作为保留路线 |
| R09 | 新增 `fusion_lite` VAD 实验后端，只保留 `whisperseg-adaptive` 与 `fusion_lite` 两条公开路线；字幕默认软目标/硬上限收紧为 5.5s/6.5s | 匿名样片 B 前 5 分钟：whisperseg 14 字幕/11 drops；fusion_lite 15 字幕/7 drops；匿名样片 C/D 全片 VAD 对比 generation overflow/error 均为 0；字幕定向 23 passed，全量 383 passed, 5 skipped |
| R10 | 全量审计修复：whisperseg 空结果除零、旧 chunking 整段 fallback、timestamp fallback 参数运行时化、alignment fallback 统计、字幕 writer/Web/pipeline 过时路径清理 | 定向回归通过，后续以全量 pytest 基线更新 |
| R11 | Fusion-lite 后缀实验 + 匿名样片 A 对比 + 帧率驱动 SRT overlap 归一化 | 匿名样片 A 四模式对比完成；`fusion_lite_boost` 最接近 whisperseg-adaptive；新增逐句 HTML 报告；字幕/fps/主流程定向 73 passed |
| R12 | 匿名样片 C 四模式双语对比 + frame-based 短尾 cue 合并 | 匿名样片 C 四模式全流程双语输出完成；新增 frame-based overlapping tail merge，修复 `受け` / `受けて` 这类极短 gap 被 F0 gender 抖动切成两条的问题；字幕定向 50 passed |

### 关键验证记录

#### V01 · ASR generation budget / VAD cache（R03）

- ONNX CUDA smoke 通过：whisperseg `model.onnx` 可创建 `CUDAExecutionProvider` session，provider 为 `['CUDAExecutionProvider', 'CPUExecutionProvider']`。
- 匿名样片 A 复测：ASR+Alignment 266.00s，输出 578 条字幕。
- 对比 R02：总耗时 729.36s -> 649.40s；ASR+Alignment 430.61s -> 266.00s。
- 逐句字幕对比报告：`video/<video-stem>/<video-stem>.subtitle_compare.html`。
- VAD/chunk cache smoke：修改 ASR prompt 上限后 aligned cache miss、ASR 重跑，但 VAD chunk cache hit；静音分析与切块 2.34s -> 0.01s。

#### V02 · 删除 ASR recovery 和 fallback 重写路径（R04）

- 后端已删除 ASR recovery、temperature fallback、prompt overflow retry；生成失败或不确定时不再重写补救。
- timestamp/alignment fallback 仅用于时间轴，不新增 ASR 文本。
- 后端 `JobSpec` / `JobContext` / `/api/config` 不再暴露 `asr_recovery`。
- 验证：compileall 通过；precision/QC/cache/ASR 定向 66 passed；全量 365 passed, 5 skipped。

#### V03 · adaptive precision 默认化（R05-R06）

- 默认 ASR 精度策略更新为 adaptive precision。
- adaptive 阈值写入 ASR checkpoint / aligned cache signature，`ASR_QC_ADAPTIVE_*` 变化会触发重算。
- 匿名 5 分钟 smoke：`whisper-ja-anime-v0.3`，ASR+Alignment 16.96s，输出 82 段，`asr_dropped_uncertain_count=2`，generation overflow/error/timeout/quarantine 均为 0。
- Engine 默认 ASR 改为 `whisper-ja-anime-v0.3`，与 Web 推荐默认一致。
- 新增通用测试集评测工具：`tests/testset_quality_eval.py`。
- 全量 pytest 基线：373 passed, 5 skipped。

#### V04 · VAD 方案取舍和 fusion-lite（R08-R09）

- Silero / `hybrid_precision` 曾作为低幻觉 VAD 方案验证；hard gate 漏掉大量真实对白，soft gate 有改善但仍过度依赖 Silero。
- 当前保留 VAD 路线：默认 `whisperseg-adaptive`，以及实验 `fusion_lite*`。
- `fusion_lite` 使用可解释公式融合 whisperseg 分数、Silero 重叠、RMS、spectral flux 和时长分数。
- 匿名样片 B 前 5 分钟：whisperseg 48 VAD segments / 129.22s speech / 14 字幕 / 11 drops；fusion_lite 23 VAD segments / 89.34s speech / 15 字幕 / 7 drops；generation overflow/error/timeout/quarantine 均为 0。
- 匿名样片 C / D 全片三模式历史对比显示 fusion_lite 输出接近 whisperseg，但 drops 少于 whisperseg。
- 逐句报告：`video/<video-a>/<video-a>_<video-b>.full_vad_modes_line_compare.html`，同一报告也归档到 `video/<video-b>/`。

#### V05 · fusion-lite 后缀实验和帧率驱动 SRT 归一化（R11）

- 新增 `fusion_lite_boost` 和 `fusion_lite_sigmoid` 后缀后端，不新增 `FUSION_VAD_SCORING_MODE`。
- 匿名样片 A 使用 `whisper-ja-anime-v0.3`、跳过翻译进行 VAD/ASR 对比。
- CUDA ONNX 在 sandbox 内失败，原因是 GPU 被操作系统/sandbox 阻断；外部执行确认 RTX 4060 Ti、Torch CUDA 和 ONNXRuntime CUDA provider 可用。
- 匿名样片 A 汇总：`whisperseg-adaptive` 700 SRT / 210 ASR drops；`fusion_lite` 683 SRT / 167 drops；`fusion_lite_boost` 688 SRT / 185 drops；`fusion_lite_sigmoid` 677 SRT / 148 drops。
- SRT overlap 处理重构：新增 `probe_video_fps()`，SRT writer 在写出前排序、软拆、合并/裁剪重叠，并强制保留 2 帧 gap。
- quality report 新增 subtitle overlap 统计。
- 验证：字幕/fps/主流程定向 73 passed。

#### V06 · 四模式双语对比和短尾合并（R12）

- 匿名样片 C 四模式完整双语对比已输出到 `video/<video-stem>/<video-stem>.*.srt`，逐句 HTML 报告为 `video/<video-stem>/<video-stem>.vad_bilingual_compare.html`。
- 四模式 ASR generation error / overflow / timeout 均为 0，最终 SRT 未包含可见 `[M]` / `[F]` 性别标签。
- 匿名样片 C 汇总：`whisperseg_adaptive` 230 SRT / 96 ASR drops；`fusion_lite` 231 SRT / 63 drops；`fusion_lite_boost` 230 SRT / 84 drops；`fusion_lite_sigmoid` 230 SRT / 50 drops。
- cue plan 短尾合并改为 frame-based 规则。
- 匿名样片 C 离线验证：`00:02:56,839 --> 00:02:59,160` 合并为 `アルマリスト 室で イラックス ステイマンを受けて`。
- 验证：字幕定向 50 passed。

### 历史任务摘要

| 编号 | 大致内容 | 验收 / 备注 |
|------|----------|-------------|
| H01 | ASR Recovery 接入 VAD 二次细分，改善异常 ASR 文本块的重跑路径 | 历史功能，R04 已从后端移除 |
| H02 | 建立 F0 词级时间轴与 multi-cue gender 切分 | 已完成 |
| H03 | Web 控制台、Stage 事件 JSON 化、重试断点续传和 cancel event 透传 | 已完成 |
| H04 | HF 镜像开关、Web 配置项扩展 | 已完成 |
| H05 | 后端稳定性、CLI 瘦身、全局 env 并发污染治理 | 完成后 179 passed |
| H06 | `transformers` 兼容性回滚，保留四个稳定 ASR 后端 | 依赖固定回 `transformers==4.57.6` |
| H07 | GitHub 发布前文档/配置/入口收口；翻译上下文和 cache key 收口 | 完成后定向 32 passed |
| H08 | Windows Release exe 打包配置 | 已完成 |
| H09 | Web 表单记忆与右键粘贴体验 | 已完成 |
| H10 | OpenAI Responses 翻译格式兼容 | 已完成 |
| H11 | F0 后 gender turn 字幕重切段 | F0 定向 15 passed；ASR job/cache 定向 7 passed |
| H12 | 翻译重试与请求清理；Micu+Grok Responses 特例移入 `src/llm/patch.py`；翻译前 ASR 噪声过滤扩展到纯英文幻觉 token | 已完成 |
| H13 | 后端稳定性收口：`JobSpec` 边界、finished job 删除锁顺序、run logger 泄漏、translation cache 损坏容忍 | 58 passed |
| H14 | 后端大文件拆分：`src/main.py` helper 迁入 `src/pipeline/` 多个子模块 | 宽后端回归 93 passed |
| H15 | 前端 `app.js` 拆分为 ES Module，并修复日志刷新导致粘贴菜单关闭的问题 | 语法检查和手动验证通过 |
| H16 | 翻译 reasoning effort 收口为 `medium` / `xhigh`，Responses 不做兼容降级映射 | 定向 36 passed |
| H17 | 翻译 fixed-prefix 批处理、并发诊断、术语/人名规则、局部 repair pass | 229 passed |
| H18 | Web 演员名持久化、提交自动保存设置、移除手动保存按钮 | Web 定向 10 passed + JS check |
| H19 | 翻译前 ASR 噪声过滤扩展到纯特殊符号段 | 定向 56 passed |
| H20 | 拆分 `src/whisper/pipeline.py`：后端 registry、checkpoint 等职责外移 | 241 passed |
| H21 | 拆分 `src/llm/translator.py`：translation cache 和 prompt 构建外移 | 241 passed |
| H22 | 压缩 `src/main.py`：stage log、output writer 等职责外移 | 241 passed |
| H23 | ASR 滑动上下文注入：`initial_prompts`、gender/gap 重置 | 241 passed |
| H24 | VAD 微短段预合并：短 speech chunk 物理拼接并保留 `merged_from` 元数据 | 241 passed |
| H25 | 字幕软切分点：长段优先按中文标点/日文助词词边界拆分 | 241 passed |
| H26 | Repair Pass 增强：长度错配强制纳入 repair 候选 | 237 passed |
| H27 | ASR 质量信号：`avg_logprob`、`no_speech_prob`、`compression_ratio`；历史 temperature fallback 已在 R04 移除 | 277 passed |
| H28 | whisperseg 默认阈值 0.35；`SpeechSegment.score`；negative offset env；adaptive VAD | 299 passed |
| H29 | VAD chunk packing + 词时间戳后置 F0 gender split | 253 passed |
| H30 | VAD chunk packing 默认开启；ASR overflow initial prompt 双层截断 | 256 passed |
| H31 | None 段 gender carry-over | 262 passed |
| H32 | soft split 扩展 None 长段，`gender=None` 且长段强制 hard word split | 302 passed |
| H33 | 短段丢弃 gate：duration + RMS AND 双条件，env opt-in | 312 passed |
| H34 | F0 carry-over 默认放宽；修复 `nan_ratio_threshold` 透传问题 | 312 passed |
| H35 | `F0_GENDER_NONE_TOLERANCE` 2 -> 3；post-split 第二次 carry-over pass | 315 passed |

### 历史验证基线

匿名样片四后端 skip-translation 对比：

| 后端 | 状态 | ASR 转写 | Wall time | 字幕数 |
|------|------|----------|-----------|--------|
| `anime-whisper` | ok | 48.52s | 336.88s | 150 |
| `qwen3-asr-1.7b` | ok | 251.71s | 578.15s | 164 |
| `whisper-ja-1.5b` | ok | 170.74s | 464.33s | 165 |
| `whisper-ja-anime-v0.3` | ok | 41.89s | 229.46s | 151 |

默认全量翻译 anime-whisper + bilingual：`pipeline_total=575.30s`，字幕块数 150，产物 `video/<video-stem>/<video-stem>.srt`。

匿名样片历史基准：

- H27 前后基线：491.5s，493 ASR chunks，字幕 365 段，F/M/None=117/124/124，Mixed=13。
- R03 当前基线：总耗时 649.54s；whisperseg CUDA VAD/切块 9.32s；ASR+Alignment 266.00s；输出 578 条字幕；ASR generation overflow/error 为 0。
