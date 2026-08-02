# jav-trans

jav-trans 是一个面向 Windows + NVIDIA 显卡的本地 JAV 字幕生成工具。它把视频处理成日文字幕、中文字幕或中日双语字幕，并把音频准备、切分、Qwen ASR、CTC 强制对齐、字幕时间轴、LLM 翻译和质量报告串成一条本地优先的流水线。

项目目标：本地完成视频、音频、切分、ASR 和字幕时间轴重计算；LLM 只负责翻译、术语一致和口吻连贯，不负责脑补剧情或修正 ASR 误听。

**翻译可以完全本地运行**：除 OpenAI 兼容 API 外，内置 llama.cpp 后端托管 GGUF 量化模型（预设 galgame 特调的 Sakura-GalTransl，7B Q6_K 适配 8G 显存），以及进程内 Transformers 后端。选本地后端时整条流水线不出网。详见下方「翻译后端支持」。

致谢：[WhisperJAV](https://github.com/a63n/WhisperJAV) 为本项目早期路线提供了重要参考。

---

## 界面预览

![网页控制台主界面](docs/images/ui-web-console.png)

任务提交、翻译后端选择（API / 本地 GGUF / 本地 Transformers）、实时阶段进度、显存与耗时监控、质量报告都在本地网页控制台完成。更多截图放在 `docs/images/`。

---

## 设计原则

本项目在 ASR 之前**不做任何丢弃式判断**。音频只被切开，不被筛选：切点由 ASR encoder 上的一个 CTC 对齐头给出的 blank 游程决定，输出精确铺满整个文件，每一秒都会进入解码器。

这条原则来自代价不对称：**丢错不可逆，留错可过滤**。判断放在有文本之后。

当前职责划分：

- **切分**只决定边界落在哪里。切点由对齐头的停顿给出；没有配对齐头时退化为定长切分。切错只是边界变差，永远丢不了词。
- **ASR 解码**输出日文文本。
- **CTC 强制对齐**把文本逐字对回音频，产出真实字级时间戳与对齐分数。
- **后置闸**（`src/asr/postgate.py`）只对已有文本**打标不删除**：失控重复、不可能的语速、与邻块重复等。标记随字幕下发，由下游决定是否过滤。
- **字幕 layout** 只处理显示规则，不反向修改 ASR chunk 语义。

设计演进、实验记录和失败路线见 [docs/HISTORY.md](docs/HISTORY.md)。

---

## 快速开始

推荐环境：

- Windows 10/11。
- NVIDIA 独立显卡和较新的驱动。
- Python 3.14+（与 `pyproject.toml` 的运行时约束一致）。
- FFmpeg Shared（TorchCodec 需要 FFmpeg 共享 DLL），并确保命令行能直接执行 `ffmpeg`。
- Git。

Windows 请安装 Shared 版。不要同时保留会优先占用 `PATH` 的 `Gyan.FFmpeg`
静态版：

```powershell
winget uninstall --id Gyan.FFmpeg --exact
winget install --id Gyan.FFmpeg.Shared --exact

# 关闭并重新打开终端后验证：应指向 full_build-shared\bin
where.exe ffmpeg
Get-ChildItem (Split-Path (Get-Command ffmpeg).Source) -Filter "avcodec-*.dll"
```

如果 `where.exe ffmpeg` 仍列出多个版本，请确保
`Gyan.FFmpeg.Shared\...\full_build-shared\bin` 排在 `PATH` 最前，随后重启
jav-trans。仅有 `ffmpeg.exe` 还不够；其目录必须同时存在
`avcodec-*.dll`、`avformat-*.dll` 和 `avutil-*.dll`。

项目安装：

```powershell
git clone https://github.com/jaykwok/jav-trans.git
cd jav-trans

uv venv
uv sync
```

Qwen3-ASR 原生支持要求 `transformers>=5.13.0`（由 `uv sync` 按 `pyproject.toml` 安装）。`pyproject.toml` 同时把 `torch` 钉到 `pytorch-cu130` 索引，请勿改用 `pip install` 逐个装依赖——那样会从 PyPI 取到 CPU 版 torch。

启动网页控制台：

```powershell
$env:PYTHONIOENCODING="utf-8"
uv run --no-sync python launcher.py
```

默认地址为 `http://127.0.0.1:17321`。首次运行可以没有 `.env`；打开页面后在“翻译设置”面板填写 API Key、Base URL、模型和目标语言，保存或开始任务时会自动写入项目根目录 `.env`。新建的 `.env` 只启用实际保存的本机值，ASR batch、显存预算等运行参数会以注释示例形式写入。国内网络下载 Hugging Face 模型较慢时，可在“识别设置”里填写代理协议、地址和端口。

**翻译后端支持**：通过 `TRANSLATION_BACKEND` 选择三种后端。`openai`——OpenAI 兼容 API（支持 Chat 与 Responses）。`llamacpp`——**本地翻译推荐**：程序托管官方 llama-server 运行 GGUF 量化模型，预设为 galgame 特调的 Sakura-GalTransl 系列（7B Q6_K 约 6.3GB，官方 8G 显存档；另有 6G 档 IQ4_XS 与 14B 档），需先 `winget install llama.cpp` 或从 GitHub Releases 下载 CUDA 包并在设置里填路径；Sakura 系模型会自动切换到其官方行式翻译模板（术语表 + 历史上文），其许可为 CC-BY-NC-SA 4.0 禁止商用；翻译开始时临时释放 ASR 显存、切回 ASR 时自动重载。`local`——进程内 Transformers（bf16，不支持 GGUF；量化需求请用 `llamacpp`）。详细配置和扩展指南见 [翻译后端架构文档](docs/translation-backend-architecture.md)。

Web 提交是否使用 CUDA 取决于后端服务进程是否能看到 GPU，而不是浏览器本身。完整 ASR smoke 应确认日志中出现 `cuda_available=True`、`device=cuda:0` 或 `actual_device=cuda`。
Web 会在模型要求检查中提示驱动过旧或 CUDA 初始化失败。

---

## 使用流程

1. 打开网页控制台。
2. 选择视频文件。
3. 选择字幕模式和翻译设置。
4. 选中的视频会立即进入右侧“待开始”列表；确认后点击“开始任务”。
5. 在输出目录查看 SRT、质量报告和日志。

从右侧任务列表删除已结束任务时，会清理任务临时目录；跨任务的 ASR 结果缓存（`tmp/asr_cache/`）按设计保留，不随任务删除。运行中的任务第一次删除只执行取消，进入“已取消”后再次删除才会清理临时目录。

勾选“不翻译（仅日文字幕）”时，流水线仍会执行切分、ASR 和字幕时间轴生成，但跳过 LLM 翻译，最终输出 `<视频名>.ja.srt`。这是验证本地切分 / ASR / 字幕时间轴链路的推荐 smoke 模式。

---

## 完整工作流

```text
视频输入
  -> 任务上下文 / 配置解析
  -> 音频抽取与标准化
  -> 切分（asr.chunking.cut_at_pauses）
     - ASR encoder 前向 -> CTC 对齐头 -> 每帧 blank 后验
     - 连续 blank 游程即停顿，切点落在停顿中央
     - 输出精确铺满 [0, 总时长]，相邻块共边，不丢任何音频
     - 未配置 ASR_ALIGNMENT_HEAD_PATH 时退化为定长切分
  -> ASR wav chunk export
  -> Qwen ASR text transcription
  -> CTC 强制对齐（asr.alignment）
     - 复用同一个 encoder 再跑一次前向（RTF 0.00069，约为一次解码的 0.56%）
     - 逐字时间戳 + 对齐分数
     - 起点/终点按对齐头自己判为 blank 的帧向外走，修正 CTC 尖峰造成的跨度内缩
  -> 后置闸（asr.postgate）
     - 失控重复 / 语速不可能 / 与邻块重复 等只打标，不删除
     - 标记随 chunk 下发到 segment，由下游决定是否过滤
  -> Subtitle Layout v2
     - acoustic/display 双时间轴
     - 20-frame 最小显示时间（固定 `24000/1001` 基准）
     - 2-frame 最小间隔（固定 `24000/1001` 基准）
     - 7s 最大显示 soft guard
  -> 可选 LLM 翻译
  -> SRT / bilingual JSON / quality report / logs
```

关键约束：

- **ASR 之前不做丢弃式判断。** 切分只决定边界，输出铺满整个文件。
- 内部 cut 是一个共享绝对时间戳，不允许左右 chunk 各自修边。
- `20 / (24000/1001)` 是字幕最短显示和 micro chunk 风险线，不是 runtime duration-only drop 阈值。
- 7 秒是字幕显示 soft guard，不是 ASR chunk 上限。
- Runtime 不使用具体词黑名单或时长启发式删除短促人声。
- 字幕行的起止**不取单个首字或末字**，用稳健分位数。
- 超长 cue 的**拆分点取对齐头量出的词间静音**（≥0.12s 才算间隙，≥0.60s 视为完整停顿，切点落在静音中央），文本切点取该静音之后那个词的真实起点。只有 `ctc_forced_alignment` 的词参与；没有真实停顿可用时退回按字数比例切分。
- 后置闸的 `min_alignment_score` 保持关闭，因为该阈值尚未标定。
- 对齐头**默认启用**：`ASR_ALIGNMENT_HEAD_PATH` 默认指向 HF 上的 `ctc_aligner.pt`。置空可回退到定长切分 + 比例时间轴。checkpoint 缺失、下载失败或损坏都只 warn 一次并降级，不会让转写失败。
- allocated/reserved/shared VRAM 只写运行诊断，不参与功能判断；显式 CUDA 请求不可用时直接报错，不回退 CPU。

---

## 模型架构

ASR 只有一个 repo：`jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf`，不提供更小的低显存档。

除 ASR 外**只有一个可训练组件**：CTC 对齐头。它是 ASR encoder 上的一个薄头，encoder 全程冻结。

```text
ASR encoder 输出 2048 维 @13fps（76.9 ms/帧）
  -> LayerNorm
  -> Conv1d(2048->512, 上采样 x2)   # 38.46 ms/帧
  -> 若干层小 encoder
  -> Linear(-> 字表 + blank)
```

- **上采样 x2 不是可选项**：CTC 每个输出 token 至少要一帧，而日语语速在 13fps 下每 mora 仅 1.6~2.2 帧。
- **CTC 目标是「字」不是 kana**，因此不需要 g2p 依赖。
- **训练数据是** `(音频, ASR 文本)` 配对：ASR 转写自己的音频，输出即目标。
- **同一份输出有两个读法**：与文本对齐得到时间轴，blank 游程用来选切点。两个读法都不丢音频。

`forced_align` 在 `src/asr/alignment.py` 内自己实现，不依赖 `torchaudio`（本项目 Python 3.14，torch 所在索引上没有匹配的 torchaudio wheel）。正确性由穷举所有合法 CTC 路径的参照实现验证（`tests/asr/test_asr_alignment_head.py`）。

对齐头跟 ASR 权重一起发在 HF 上，不进本仓库：它是 encoder 专属的，换 encoder 即作废。`ASR_ALIGNMENT_HEAD_PATH` 接受两种写法：

```text
hf:<repo>@<commit sha>#<文件名>   # 默认；首次运行下载进 HF 缓存，之后离线
./path/to/ctc_aligner.pt          # 本地文件，覆盖默认值
```

默认值**钉死 commit sha 而不是 `main`**，换头是显式改配置的动作。

首次运行下载到 **`models/ctc_aligner.pt`**（14.7MB，与 ASR 权重同目录，走项目代理设置），之后完全离线；打包版直接内置在同一位置，不下载。同名的 `models/ctc_aligner.pt.revision` 记录这份文件对应哪个 sha，改了默认 sha 会重新下载。`ctc_aligner.pt` 的字节内容参与 ASR finalize 缓存签名，换头会自动让旧的对齐结果失效。

---

## 默认配置

默认配置内置在 `src/core/config.py`，首次保存 Web 设置时会自动生成 `.env`。`.env` 只用于本机私密值和显式覆盖，不复制默认配置。通常只需要在 Web “翻译 API”面板填写：

- `API_KEY`
- `OPENAI_COMPATIBILITY_BASE_URL`
- `LLM_MODEL_NAME`
- 代理协议 / 地址 / 端口（可选，用于模型下载和 HTTP 请求）

ASR 显存自适应默认值已经内置。batch 或显存预算可通过“参数调优”里的环境变量覆盖，或手动编辑首次保存后生成的 `.env`。

默认配置：

```env
ASR_BACKEND=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf
ASR_BATCH_SIZE=auto
ASR_BATCH_SIZE_BY_REPO=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf=4
ASR_STAGE_WORKER_VRAM_BUDGET_MB=auto
ASR_STAGE_WORKER_VRAM_RATIO=0.95
ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf=6144
ASR_STAGE_WORKER_RAM_RATIO=0.95
ASR_STAGE_WORKER_HEARTBEAT_S=10
ASR_STAGE_WORKER_OOM_RETRY_LIMIT=6
GPU_BATCH_PROFILE_ENABLED=1
GPU_BATCH_PROFILE_GROWTH_THRESHOLD=0.80
ASR_CHUNK_TARGET_S=20.0
ASR_CHUNK_MAX_S=30.0
ASR_CHUNK_MIN_S=2.0
ASR_CHUNK_MIN_PAUSE_S=0.6
```

ASR stage 固定由统一 GPU worker 持有 CUDA：切分用的 encoder 前向、ASR 解码和 CTC 对齐都在同一个 GPU owner 进程里顺序执行，Web / 调度主进程只做任务编排、缓存索引和输出写入。OOM、CUDA 状态异常或超过 `ASR_STAGE_WORKER_VRAM_BUDGET_MB` 时会杀掉 worker，不会把 Web 主进程一起带崩。

`ASR_STAGE_WORKER_VRAM_BUDGET_MB=auto` 按物理 dedicated VRAM × `0.95` 计算软 OOM 线；RTX 4060 Ti `8188MiB` 的 cap 约为 `7779MiB`。**推荐 8GB 及以上**：1.7B 权重约 3.4GB 常驻，默认配置（batch 8 / 20 秒块）在 8GB 卡上峰值约 `5692MiB`。`ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO` 的 `6144MiB` 是**硬下限**而不是推荐值：低于它直接拒绝启动，等于它则需要 auto batch 降到 4~6 才能跑，余量很小。检查在模型加载前完成；shared VRAM 不计入可用预算，任何正的基线增量都立即视为 soft OOM，显式放大的 worker budget 和 CPU fallback 都不能绕过。监控不可用会直接停止。物理 RAM 使用按 `total-available` 计算，超过 `total × ASR_STAGE_WORKER_RAM_RATIO`（默认 `0.95`）同样停止。

GPU worker 默认每 10 秒输出一次当前阶段、总耗时和静默时长心跳。字幕 cue plan 会单独记录 timeline normalize、两轮 anchor-aware DP、polish 和 finalize 进度。

对齐后的 segment 会按内容签名缓存。签名包含 ASR backend、字幕选项与参与结果的运行配置；只改输出路径一类不影响内容的设置不会让缓存失效。

每个 ASR chunk 的转写结果另有一层跨任务的内容寻址缓存（`tmp/asr_cache/<模型签名>/<音频sha256>.json`）：键为 chunk 音频 PCM 内容加模型与解码参数，与路径、chunk 序号和任务无关。同一部片重跑、崩溃续跑、甚至不同任务里字节相同的 chunk 都直接命中，整段跳过 encoder+decoder。超时与隔离结果不会入缓存；`ASR_RESULT_CACHE_ENABLED=0` 可关闭，`ASR_RESULT_CACHE_ROOT` 改位置。

`ASR_BATCH_SIZE=auto` 以 5600MB 下的 repo 默认表为基线，按显存预算比例放缩初始 batch。ASR text batch 发生 GPU OOM 时会重启 worker、降低 batch 并从结果缓存续跑。切分阶段逐个编码固定 30 秒窗口，没有可降的 batch，在那里 OOM 会直接停止而不是假装重试。RAM OOM 同样直接停止，不伪装成可由 GPU batch 修复的问题。

auto batch 会在 `tmp/cache/gpu_batch_profiles.json` 按 GPU、显存预算、模型、精度、attention 实现和 chunk 时长跨任务学习。v3 profile 记录已验证安全 batch 与 OOM 不安全上界：阶段 peak allocated 低于预算 `80%` 时，在两者之间二分探测；尚无 OOM 上界时则向当前阶段上限折半推进，OOM 后本次任务仍先减半恢复，同时把安全值压到不安全值以下。chunk 时长是 profile 身份的一部分，因此不同显存和不同 chunk 几何各自学各自的上限，不需要手动配置。当前只覆盖 ASR chunk batch；显式数字 batch 不参与 profile 学习。

推理只需要 ASR Hugging Face 模型本身；同一份权重会被加载两处，切分阶段短暂加载一次取 encoder 特征（算完即卸，不与解码模型同驻），解码阶段再加载一次并一直用到对齐 pass 结束。**权重按需加载**：需要它的阶段自己加载，所以整片命中缓存的续跑完全不会加载模型。源码运行时如果本地没有模型，会按需下载到 `models/`。对齐头（14.7MB）默认从同一个 HF repo 按固定 commit sha 取，落在 `models/ctc_aligner.pt`；置空则**不报错**，切分退化为定长、字幕时间轴退化为按字数比例摊开。

阶段耗时表的各行严格加总为总计。ASR 在独立的 GPU owner 进程里运行，进程启动、环境传递与结果回传单列为「ASR Worker 启动与传输」；未归入任何具名阶段的剩余时间落在「其他」。

训练产物（CUDA feature cache、synthetic WAV、sequence JSONL、tensor cache、`datasets/train/...`）都不是运行依赖，不随源码分发。

---

## 字幕与文本策略

- ASR 文本会做 Unicode NFKC、空白归一、换行折叠和展示安全处理。
- Qwen3-ASR runtime 始终使用 Transformers 官方 `apply_transcription_request(audio=..., language=...)` 路径，不提供演员名 / 人名 context 提示分支。
- 字幕时间轴来自 CTC 强制对齐的逐字时间戳；对齐头未配置时退化为按字数比例摊开。
- LLM 翻译前会先固定 cue plan，翻译不会重排时间轴。
- 最终中文输出遵循 Netflix Chinese (Simplified) TTSG：每行 ≤16 全角单位、最多 2 行（下宽金字塔）、时长 5/6s–7s、2 帧最小间隔、语音结束后出点约 +0.5s；不用逗号句号（句中停顿为单个空格）、省略号为单个 U+2026、全角？！且不连用、半角数字、无斜体。标点归一化与折行在 `src/subtitles/zh_style.py` 的写盘层完成，翻译缓存保留 LLM 原文；质量报告以 `spec_*` 指标核查全部硬指标。`SRT_LINE_MAX_CHARS` 默认 16。

---

## 输出与缓存

- `video/<视频名>/`：正式字幕、质量报告和人工质检报告。
- `models/`：Hugging Face 模型缓存。
- `tmp/jobs/<job_id>/`：Web / pipeline 单次任务临时目录；`JOB_TEMP_DIR` 默认是 `./tmp/jobs`。
- `tmp/chunks/`：ASR wav chunk 的一次性运行目录。
- `tmp/asr_cache/`：跨任务 ASR 结果缓存；按内容寻址，任务删除时保留。
- `tmp/cache/torch/`、`tmp/cache/hf/`：torch / Hugging Face 运行缓存。
- `tmp/log/<job_id>/`：默认启用的本地诊断目录；包含 `.run.log` 和持久化 `.timings.json`。
- `datasets/`：本地训练、验证、测试数据归档，默认 ignored；不进入 GitHub 源码仓库。
- `agents/temp/`：研究脚本、smoke、临时日志和中间产物。
- `agents/audits/`：可长期复查的本地审计页，默认 ignored，不随 `git push` 发布。

本地审计页服务：

```powershell
.\tools\audits\serve_audits.ps1
```

审计导航会显示每个审计产物的生成时间，优先使用 summary 时间，其次使用目录名前缀，便于区分多轮审计页。审计服务支持音频 Range seek 和导航页删除 API。直接打开 HTML 可以浏览页面，但删除按钮不能真正移动本地审计目录。

成功运行后默认删除一次性 job 临时目录；保留可复用缓存，例如 `models/`、`tmp/cache/` 和 Web 状态。

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

受限 sandbox、错误的 PyTorch wheel、驱动问题或从非 GPU 环境启动 Web 服务都会使 CUDA 启动检查失败；正式工作流不会转到 CPU 继续执行。

### 显存不足

默认配置已按 **8GB 级显存**目标设置（见上文「默认配置」），硬下限是 `6144MiB`。

先让 auto batch 自己收敛：`ASR_BATCH_SIZE=auto` 会按 GPU、显存预算、模型和 chunk 时长学习安全 batch，OOM 一次就记下上界并二分回退，profile 跨任务保存在 `tmp/cache/gpu_batch_profiles.json`。8GB 和 16GB 卡各自学各自的，不共用。

只有在 auto 反复 OOM 时才手动钉值——显式数字会关掉学习：

```env
ASR_BATCH_SIZE=2
```

只有 1.7B 一档，没有更小的模型可切换。降 batch 之后仍然 OOM 就只能关掉其他占用 GPU 的程序；本程序不会改用 CPU 继续推理。

### 长任务怎么排查

运行日志默认写入 `tmp/log/<job_id>/`。`.run.log` 便于查错，`.timings.json` 记录音频准备、切分、ASR、字幕时间轴、翻译、写出等阶段耗时和显存快照；Web 完成任务后也会把这两个文件列在“其他文件”里。反馈问题时请保留 `.run.log`、`.timings.json`、质量报告和对应 SRT。

---

## 开发

主要代码位置：

- `src/main.py`：主流程编排。
- `src/core/`：配置和任务上下文。
- `src/pipeline/`：音频、缓存、输出、质量报告和阶段日志。
- `src/asr/`：ASR 转写、切分（`chunking.py`）、cue 特征（`cue_features.py`）、CTC 对齐头（`alignment.py`）、字幕时间轴（`subtitle_timing.py`）与后置闸（`postgate.py`）。
- `src/llm/`：翻译侧三层——`backends/`（transport）、`profiles/`（各模型家族 prompt 合同）、`engine.py`（唯一编排循环），`translator.py` 是门面；另有 prompt、cache、glossary、修复批与术语预抽取。
- `src/subtitles/`：SRT writer、字幕选项和字幕 QC。
- `src/web/`：FastAPI 接口和静态前端。
- `tools/`：对齐头训练、审计页、离线 Teacher、ASR SFT 和 workflow smoke 工具。

常用测试：

```powershell
$env:PYTHONIOENCODING='utf-8'
uv run pytest tests/asr           # 转写、切分、对齐头、后置闸
uv run pytest tests/pipeline      # 编排、GPU worker、batch profile、产物写出
uv run pytest tests/llm           # 翻译 backend/profile/engine 与缓存
uv run pytest tests/subtitles tests/web tests/tools
uv run pytest                     # 全量
```

---

## 工具索引

所有 Python 工具都从项目根目录执行，并使用当前 `.venv`：

```powershell
$env:PYTHONIOENCODING='utf-8'
uv run python -m <module> --help
```

常用入口：

- `tools.workflows.run_full_workflow`：命令行完整工作流 smoke。
- `tools.workflows.promote_torch_checkpoint`：晋升生产 checkpoint。
- `tools.web.smoke.start_server` / `submit_job` / `poll_job` / `summarize_job`：Web 服务 smoke 和任务汇总。
- `tools.align.*`：CTC 对齐头训练链——`build_alignment_features`（encoder 特征抽取）、`build_real_alignment_manifest` / `build_real_alignment_lines`（真实数据 manifest）、`train_ctc_aligner`（训练）、`evaluate_alignment_geometry` / `evaluate_pregate_loss` / `measure_pregate_dropped_audio`（几何与切分评估）。
- `tools.audits.audit_nav` / `serve_audits.ps1`：审计页导航与 Windows 本地服务。
- `tools.audits.review_page_core` / `audit_prompt` / `binary_clip_audit`：人工审计页共享 Core（`MM:SS.mmm` 区间显示与播放器、状态、完成度、保存 API）与可复用提示配置；任务特有布局与 verdict 组合由 Adapter 提供。设计合同见 [Human Audit Page Core](docs/audits/20260723_human-audit-page-core-v1.md)。
- `tools.audits.select_alignment_onset_audit` / `generate_alignment_onset_audit_html` / `evaluate_alignment_onset_audit`：对齐头起止点人工审计的抽样、页面生成与统计。
- `tools.audits.generate_subtitle_ab_compare_audit_html`：两版字幕的 A/B 对照审计页。
- `tools.datasets.label_drop_spans_words` / `apply_drop_span_relabels` / `cut_long_drop_span_clips`：drop-span 逐词 Teacher 标注、复核回写与长片段切分。
- `tools.audits.build_word_definition_calibration` / `evaluate_word_teacher_calibration`：逐词 Teacher 的「什么算词」校准集与一致性评估。
- `tools.omni.run_audio_teacher` / `audio_teacher_batch`：离线音频 Teacher Core；统一处理 `--prompt/--prompt-file`、`--folder/--file/--manifest`、provider-safe 并发、进度、续跑和主线程串行化落盘。
- `tools.omni.audio_teacher_transport`：Qwen、OpenRouter、Google AI Studio 三个 provider Adapter 的唯一分派入口；`--env-file qwen|openrouter|gemini` 只接受这三个已知 profile（`~/.config/omni/` 下的隔离配置），请求与响应协议互不冒充。
- `tools.omni.gemini_native` / `inspect_gemini_quota`：Google AI Studio 原生 Interactions 音频 Adapter；内联音频、结构化输出、多 Key 轮换与保守的本地滚动配额账本（多 Key 只增加配额轮换槽，不增加并发；`inspect_gemini_quota` 不发请求，只显示脱敏状态）。
- `tools.omni.timestamp_contract`：Teacher 时间坐标的严格 `MM:SS.mmm` wire schema、格式化、解析和 source-bound 校验；不提供数字秒兼容或时间猜测。
- `tools.sft.*`：Qwen3-ASR SFT 自训链路——数据集准备、云端训练资产与训练脚本；线上 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf` 即该链路的发布产物。
- `tools.asr.convert_qwen3_asr_to_hf`：把 Qwen3-ASR 权重转换为 HF 布局。

命令行完整工作流 smoke：

```powershell
uv run python -m tools.workflows.run_full_workflow --video video/<your-video>.mp4 --task-name 20260617_191654_cli-smoke --label smoke
```

训练细节、实验记录和路线演变不在 README 展开；见 [docs/HISTORY.md](docs/HISTORY.md)。
