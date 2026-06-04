# JAVTrans

JAVTrans 是一个本地字幕生成工具，面向 Windows + NVIDIA 显卡，也可在 WSL2 / Linux 下源码运行。它把视频处理成中文字幕或中日双语字幕，并把音频准备、speech-island 边界规划、ASR、强制对齐、字幕时间轴归一化、LLM 翻译和质量报告串成一条本地优先的流水线。

项目目标：本地完成视频、音频、边界切分、ASR 和时间轴重计算；LLM 只负责翻译、术语一致和口吻连贯，不负责脑补剧情或修正 ASR 误听。

致谢：[WhisperJAV](https://github.com/a63n/WhisperJAV) 为本项目早期路线提供了重要参考。本项目后续扩展了 SpeechBoundary-JA（由早期 FusionVAD-JA 实验演进）、Qwen3-ASR 目标域 SFT、pre-align、forced-alignment fallback 和字幕时间轴 polish。

实验历史、路线取舍、调试记录和参考来源见 [HISTORY.md](HISTORY.md)。

---

## 当前状态

- 默认 ASR：`jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`。ASR backend key 就是 Hugging Face repo ID；首次下载会写入合法本地目录 `models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame`。
- 可选 ASR：`jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`，本地缓存目录为 `models/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame`。
- 默认边界系统：SpeechBoundary-JA，backend key 为 `speech_boundary_ja`。它已不是严格意义的 VAD；它负责 Qwen PTM + MFCC / energy bootstrap frame scoring，随后由 `src/boundary/` 做 candidate extraction、Boundary Refiner scoring、constrained planning 和 boundary-cache v1。JA 目标域 bootstrap scorer 位于 `src/boundary/ja/`，不保留旧 `fusionvad_ja` / `src/vad/` alias。
- SpeechBoundary-JA 特征层：`jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`，同时作为轻量 ASR probe 和 frozen PTM feature。当前 schema 统一使用 `ptm` / `ptm_dim` / `ptm_proj` 命名，不再沿用早期 `whisper_*` 字段。
- 当前边界主线：speech-island 边界优先，目标是尽量切成“一句台词一个 ASR chunk”，减少多句台词、长 gap、BGM/噪声和换人片段被揉成一坨。验收优先看 `start` p90/p95、fallback chunk duration、long/gap-crossing chunk、`start_weighted_speech_recall`、ASR empty / hallucination 是否恶化；旧 `speech_duration_recall` 只作历史对比。recall guardrail 暂按 `>=0.93`，必要时可为更准边界牺牲一部分整段 recall。
- `chunk growth` 不再作为主要否决 gate，只作为成本和极端爆炸观察指标。90 分钟片子有几百个 ASR chunk 是正常现象；如果边界更准、fallback 更短、ASR 失败不明显恶化，可以接受更多 chunk。
- Boundary Refiner 当前使用 deterministic bootstrap 策略，默认开启；learned refiner checkpoint schema 已固定为 `boundary_refiner_v1`。backbone 入口只保留 `transformers.Mamba2Model`，直接对应 Hugging Face Transformers 的纯 PyTorch Mamba2 实现；不再暴露 `mamba2`、`torch_mamba2`、BiGRU、TCN 或 Transformer fallback，也不把 Linux-only `mamba-ssm`、Triton 或自定义 CUDA kernel 作为默认依赖。
- 字幕层后置修正已进入代码：ASR QC 对 kana-only 语气词/呻吟重复默认保留并标记 review，cue-stage planner 和 speaker sidecar 作为离线审计/校准工具推进。ERes2NetV2 sidecar 已跑通，当前只作为 cue-stage review、penalty 和后续 ASR/QC hard-negative 来源，不替代 SpeechBoundary-JA。
- speaker sidecar 路线：首选 ERes2NetV2 / 3D-Speaker，CAM++ 只作 baseline。它只在 pre-align / cue 阶段对相邻 speech island 提 speaker embedding，辅助判断是否跨 speaker，避免把男女/多人对话合并成一条 cue。当前已提供 `energy_mfcc` smoke extractor 验证 JSONL 链路，并接入可选 `modelscope_eres2netv2` backend；`energy_mfcc` 不是可靠 speaker 模型，真实约束需要安装 ModelScope 可选依赖并下载 ERes2NetV2/3D-Speaker embedding 模型后重跑。
- 非默认路线和失败实验只保留在 [HISTORY.md](HISTORY.md)，不进入当前 Web / CLI 主线。

---

## 快速开始

### Release 版

如果只是使用，优先下载 GitHub Releases 中的 Windows 压缩包。解压后运行：

```text
JAVTrans.exe
```

Release 版是否内置模型以对应 release notes 为准。当前 main 分支的默认模型会按 Hugging Face repo 下载到 exe 或项目同级的 `models/`。首次使用需要在页面的“翻译设置”中填写 API Key、Base URL 和模型名。

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
LLM_MODEL_NAME=deepseek-chat
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

Web 提交是否使用 CUDA 取决于后端服务进程是否能看到 GPU，而不是浏览器本身。受限 sandbox 内启动 Web 时可能看不到 CUDA；跑完整 SpeechBoundary-JA / ASR / ForcedAligner smoke 时需确认运行日志包含 `cuda_available=True`、`device=cuda:0` 和边界 scorer `actual_device=cuda`。

---

## 使用流程

1. 打开网页控制台。
2. 选择视频文件。
3. 选择字幕模式、ASR 后端和翻译设置。
4. 提交任务。
5. 在输出目录查看 SRT、质量报告和日志。

勾选“不翻译（仅日文字幕）”时，流水线仍会执行边界规划、ASR、ASR QC、forced alignment 和字幕时间轴归一化，但跳过 LLM 翻译，最终输出 `<视频名>.ja.srt`。这也是验证本地边界/ASR/对齐链路的推荐 smoke 模式。

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

---

## 默认模型

| 用途 | 默认来源 | 本地缓存 / 文件 |
|------|----------|-----------------|
| 默认 ASR | `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame` | `models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame` |
| 可选 ASR | `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame` | `models/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame` |
| SpeechBoundary-JA frozen feature | `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame` | `models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame` |
| Forced aligner | `Qwen/Qwen3-ForcedAligner-0.6B` | `models/Qwen-Qwen3-ForcedAligner-0.6B` |

常用配置：

```env
ASR_BACKEND=jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame
ASR_BOUNDARY_BACKEND=speech_boundary_ja
ASR_MODEL_ID=
ALIGNER_MODEL_ID=Qwen/Qwen3-ForcedAligner-0.6B
ASR_DTYPE=bfloat16
ASR_CHUNK_MIN_DURATION_S=0.25
ASR_CONTEXT_RESET_GAP_S=0.5
```

Boundary Refiner 配置：

```env
BOUNDARY_CACHE_ENABLED=1
BOUNDARY_CACHE_DIR=./temp/boundary-cache
BOUNDARY_FEATURE_FRAME_HOP_S=0.02
BOUNDARY_REFINER_ENABLED=1
BOUNDARY_REFINER_MODEL_PATH=src/boundary/checkpoints/boundary_refiner.pt
BOUNDARY_REFINER_BACKBONE=transformers.Mamba2Model
BOUNDARY_REFINER_THRESHOLD=0.5
BOUNDARY_FRAME_SEQUENCE_LEFT_CONTEXT_S=0.60
BOUNDARY_FRAME_SEQUENCE_RIGHT_CONTEXT_S=0.60
BOUNDARY_FRAME_SEQUENCE_MAX_PTM_DIMS=64
BOUNDARY_FRAME_SEQUENCE_INCLUDE_MFCC=1
BOUNDARY_PLANNER_MAX_CHUNK_S=30.0
BOUNDARY_PLANNER_TARGET_CHUNK_S=9.0
BOUNDARY_PLANNER_MIN_CHUNK_S=0.4
BOUNDARY_PLANNER_START_WEIGHT=1.5
BOUNDARY_PLANNER_TARGET_PADDING_S=2.0
BOUNDARY_PLANNER_MAX_SPLITS_PER_SEGMENT=16
BOUNDARY_PLANNER_SEQUENCE_BATCH_SIZE=256
```

Timing 分两层处理：

- Boundary / ASR chunk planning 使用秒级上下文参数，例如 `BOUNDARY_PLANNER_TARGET_CHUNK_S=9.0` 和 `BOUNDARY_PLANNER_MAX_CHUNK_S=30.0`。`BOUNDARY_FEATURE_FRAME_HOP_S` 只是 frame-score 网格的 fallback，默认 `0.02s`，不代表视频帧率。
- 字幕显示 / timing polish 使用真实视频 FPS。流程会探测源视频帧率并计算 `frame_duration_s=1/fps`，再按 Netflix-style 规则保留 `2 frames` gap、压缩前一条 end、限制最短/最长显示时长。24fps、29.97fps、60fps 的两帧 gap 会自然换算成不同秒数。

当前 canonical `BOUNDARY_REFINER_MODEL_PATH` 固定为 `src/boundary/checkpoints/boundary_refiner.pt`。如果该文件不存在，运行时会使用 deterministic bootstrap refiner；如果文件存在，则加载 learned Boundary Refiner。训练好的 learned refiner 使用 `boundary_refiner_v1` checkpoint schema；checkpoint 内容 SHA1、模型路径、`transformers.Mamba2Model` backbone、candidate extractor version 和 planner config 都会进入 boundary-cache signature，更新权重会自动触发 cache miss。

Frame/window sequence refiner 是当前主线训练入口。`src/boundary/sequence_features.py` 是唯一 schema authority：训练和 runtime 都从这里读取 feature config、feature dim、feature names 和 `feature_schema_hash`，禁止在训练脚本、runtime 或 planner 中手写维度常量。`runtime_adapter=frame_sequence_v1` 的 checkpoint 必须带 `feature_schema=frame_sequence_features_v1`、`feature_schema_hash` 和完整 `feature_signature`；运行时会用 SpeechBoundary-JA 导出的 PTM/MFCC frame windows 重新计算 feature names/hash，不匹配就直接报错，不做旧 checkpoint 兼容或静默回退。

Feature cache 的长期规则：大体积 frame/window 特征不写入 boundary-cache JSON。boundary-cache 只保存 ASR-facing packed spans、决策诊断和 signature；训练或诊断需要持久化密集特征时，使用对应数据集目录下的 `.npz` / `.pt` sidecar，并把 `feature_schema_hash` / checkpoint SHA1 / planner config 写进 signature，保证公式或模型变化会触发 cache miss。

SpeechBoundary-JA feature cache schema 是断兼容的新 schema：feature `.npz` 使用 `ptm` + `mfcc`，manifest 使用 `ptm_dim` + `mfcc_dim`。旧实验缓存不会被迁移，重新生成即可。

`models/`、`datasets/`、`agents/temp/`、`agents/audits/` 是本地运行资产，不应提交训练数据或大模型权重。例外是小型 Boundary Refiner 头：如果 `src/boundary/checkpoints/boundary_refiner.pt` 体积可控并成为默认质量路径，可以随源码提交，版本号和训练说明记录在 README / HISTORY；不要恢复旧 `src/vad` checkpoint 路径。若后续 checkpoint 变大或需要多版本分发，再改为 GitHub Release 或 Hugging Face artifact。

---

## SpeechBoundary-JA 推进路线

命名决策：用户可见名称使用 `SpeechBoundary-JA`，代码/backend key 使用 `speech_boundary_ja`，JA 域实现位于 `src/boundary/ja/` 和 `tools/boundary/ja/`。不继续使用 `FusionVAD-JA` 的原因是当前系统已经不是传统 speech/non-speech VAD，而是 speech-island boundary proposal、Boundary Refiner、constrained planner、ASR/QC feedback 的组合；`fusion` 只是早期实现来源，`VAD` 会误导后续维护和参数设计。

当前主线先做 supervised Boundary Refiner，再考虑 RL / DPO：

```text
clip-level clean speech islands
-> synthetic multi-island timeline
-> dense PTM / MFCC / energy / speech_prob / cut_prob features
-> boundary candidates and exact split / merge / refine labels
-> tools/boundary/build_refiner_gap_dataset.py
-> tools/boundary/build_refiner_frame_sequence_dataset.py
-> tools/boundary/train_refiner.py
-> src/boundary/checkpoints/boundary_refiner.pt
-> BOUNDARY_REFINER_MODEL_PATH
```

后续推进顺序：

1. 命名收口已完成：`src/vad/fusionvad_ja/` 和 `tools/vad/fusionvad_ja/` 已断兼容迁到 `src/boundary/ja/` 与 `tools/boundary/ja/`；旧 key、旧路径、旧 cache 不做 alias 或迁移。
2. 数据重建：清除旧 gap-only / BiLSTM / endpoint-head 训练格式，使用 Galgame 与 anime clean speech islands 重新生成 sequence dataset。每条样本包含多 island、touching speech、short/long gap、real negative gap、BGM/noise、轻量 overlap 和 source/speaker switch。过渡期 gap dataset 只允许作为 loader smoke 或 warm-up，不作为最终 Seq2Seq 目标。
3. 模型升级：Boundary Refiner backbone 只保留 `transformers.Mamba2Model`。输入从 per-gap summary 升级为连续窗口序列，特征包含 Qwen PTM、MFCC、energy、speech probability、cut probability、candidate metadata；输出逐候选 split / merge / refine score 和可选 boundary offset。当前 frame/window sequence dataset、trainer、`FrameSequenceBoundaryRefiner`、runtime feature provider、schema/hash 校验和 PackedChunk 决策诊断已接通；当 checkpoint metadata 标记 `runtime_adapter=frame_sequence_v1` 时，主 pipeline 会让 SpeechBoundary-JA 导出低维 PTM/MFCC frame windows并交给 sequence refiner 做 gap 决策。
4. Planner 接入：`pack_speech_segments()` 只 materialize planner 输出，不再承担固定 gap 合并策略。当前 planner 已能读取 refiner decision 并写入 PackedChunk 诊断字段；sequence refiner 已改成 bounded batch 打分，并接入轻量 DP / Viterbi-style constrained planner。当前 DP v2 是第一版可解释 baseline：cost 主要由模型 split/merge score、chunk duration、hard max chunk 和长 gap penalty 组成；它能避免退回局部 greedy，但还不是最终字幕目标函数。`BOUNDARY_PLANNER_SEQUENCE_BATCH_SIZE` 默认 `256`，用于避免 pure PyTorch Mamba2 naive path 在全片候选一次性推理时过慢。该变更把 planner signature 升到 `constrained_sequence_dp_planner_v2`，旧 boundary-cache 需要重建。
5. GPU 闭环：先跑 synthetic exact truth，再跑匿名样片 A。主 gate 是 start boundary、fallback chunk duration、long/gap-crossing chunk、ASR empty / hallucination、forced/partial 比例；chunk 数只作为成本指标，不作为主要否决项。匿名样片 A frame-sequence greedy 与 DP v2 都已跑通：DP v2 把 fallback duration p90/max 从 `14.07/26.88s` 降到 `12.81/20.72s`，safe ratio 从 `0.314` 提到 `0.418`，但 fallback/sentinel chunk 数从 `322` 增到 `366`，ASR repeat-loop / nonlexical 问题仍存在。因此 DP v2 证明了“全局规划能压粗时间轴”，但 learned checkpoint 和 planner cost 暂不固化为默认质量路径。
6. ASR feedback：supervised 稳定后再引入 preliminary ASR 文本、token confidence、local CER、aligner sentinel、fallback duration 和 QC reject 作为 dense reward，做 RL / DPO。Unified Joint Model 保留在 backlog，等 SpeechBoundary-JA 能稳定产出 pseudo boundary labels 后再评估。

下一轮 planner cost 校准方向：把当前简单 cost 拆成可审计项，包括 calibrated model NLL、fallback-safe duration、long/gap-crossing penalty、start-boundary priority、字幕最小显示时长、2-frame gap、CPS/readability 和 ASR/QC feedback。Netflix timing 规则支持 `start/in-time` 优先、`end/out-time` 后置压缩和 2-frame gap；SubER / OptiSub 类字幕评测也说明字幕质量要同时看 timing、segmentation、显示时长和阅读速度。REBORN 的 RL boundary segmentation 可作为后续 ASR feedback / DPO 参考，但当前先用确定性 DP 做可解释 baseline。

训练数据构造要注意：旧 cutpoint 数据集主要提供 split supervision（贴连换人、短 gap 换人、可删除长 gap），不能单独当作完整 merge/split 训练集。正式训练前需要混入 clean speech-island 原料构造 same-utterance merge-positive，例如：

- `litagin/Galgame_Speech_ASR_16kHz`：已验证可作为精确 speech-island 原料。
- `joujiboi/japanese-anime-speech-v2`：约 29 万条、约 450 小时 anime / visual novel audio-text pairs，split 为 `sfw` / `nsfw`，适合作为额外 clean speech-island 原料源。JAV 目标域训练可以提高 `nsfw` 权重，但仍需保留 galgame / sfw 防止过拟合单一来源。

正式训练建议先生成 class balance summary，确认既有 merge-positive、split-negative、touching-speech、long-gap、speaker/source switch 和 low-value non-speech。单类 dataset 只允许用于 loader smoke，不进入主训练。

评估脚本按当前 diagnostics schema 断兼容运行：`fallback_type != none` 才算 alignment fallback，`fallback_subtype` 只作为 reason 原样统计，`sentinel_lines` 非空才计 sentinel fallback；缺少 `fallback_type` 的旧 diagnostics 会直接报错。

---

## 输出与缓存

- `video/<视频名>/`：正式字幕、质量报告和人工质检报告。
- `models/`：Hugging Face 模型缓存。
- `temp/boundary-cache/`：SpeechBoundary-JA frame score 到 Boundary Planner 输出的 boundary-cache v1。
- `temp/jobs/`：Web 任务临时目录。
- `temp/log/`：启用运行日志后的任务日志。
- `datasets/`：本地训练、验证、测试数据归档，默认 ignored。
- `agents/temp/`：研究脚本、smoke、临时日志和中间产物。
- `agents/audits/`：可长期复查的本地审计页。统一从 `agents/audits/index.html` 进入；审计产物直接放在该目录下，不再套 `speech-boundary-ja/` 子目录；导航页始终指向最新审计且不使用自动跳转。旧审计产物清理时移入 `agents/rm/`。

成功运行后默认删除一次性 job 临时目录；保留可复用缓存，例如 `models/`、`temp/boundary-cache/` 和 Web 状态。

---

## 字幕与文本策略

- 系统维护 `display_text` 和 `align_text` 两份文本。
- `display_text` 用于最终字幕显示，只做 Unicode NFKC、空白归一、换行折叠和展示安全处理。
- `align_text` 只给 forced aligner 使用，可删除标点、emoji、音乐符号和不可发音装饰符。
- 不使用具体词黑名单，不直接删除 `ん`、`あ`、喘息、呻吟、拟声、低信息短句或常见台词。
- 人工审计标签区分 hard drop 与低信息人声：`删除/无字幕价值` 只用于纯噪声、BGM、静音、机械声等硬丢弃片段；`低信息人声/呻吟` 用于呻吟、喘息、笑声、叹息、短促拟声等目标域可转写但字幕价值低的发声，可与 `整条文本可用`、`时间轴准确` 同时选择。
- cue merge 审计支持左右侧向标签：`上句/下句可用`、`上句/下句文本错`、`上句/下句无字幕价值`、`上句/下句低信息`。当上下两条一好一坏时，优先用侧向标签表达，不再只靠备注。
- 重复循环、低置信、文本/音频比例异常、align-text-empty、fallback、ASR dropped uncertain 只作为 QC / 诊断 / 样本池信号。kana-only 非词汇重复默认 `preserve_with_review`，只有长文本循环、异常字符密度、声学信号异常或 drop 策略显式开启时才进入删除路线。
- forced aligner 失败时不伪造精确时间轴，会保留可诊断 fallback 标签。

---

## 常见问题

### 模型下载慢

设置：

```env
HF_ENDPOINT=https://hf-mirror.com
```

或提前把模型下载到 `models/` 对应目录。

### CUDA 没有被使用

Codex sandbox 或部分受限环境可能隔离 GPU。跑全片边界规划 / ASR / ForcedAligner、Torch CUDA、feature cache 或训练时，需要确认日志中出现：

```text
actual_device=cuda
model_param_device=cuda:*
```

SpeechBoundary-JA 的 feature cache、训练、逐帧概率导出和全片 workflow 都按“能 CUDA 就提权 CUDA”处理；不要为了省事让 CPU 跑大规模评测，否则会把等待时间和中间产物放大，且容易误判进度。

### 长任务怎么排查

启用运行日志后，日志会写入 `temp/log/` 或任务输出目录。反馈问题时请保留 `.run.log`、质量报告和对应 SRT。

---

## 开发

主要代码位置：

- `src/main.py`：主流程编排。
- `src/core/`：配置和任务上下文。
- `src/pipeline/`：音频、缓存、输出、质量报告和阶段日志。
- `src/asr/`：ASR、forced alignment、prealign 和转写流程。
- `src/boundary/`：Boundary features、candidate extraction、Boundary Refiner 接口、Windows-friendly sequence backbone、constrained planner、boundary-cache v1 和 `boundary.get_boundary_backend()`。
- `src/boundary/ja/`：SpeechBoundary-JA bootstrap scorer、PTM/MFCC feature cache schema、训练数据 manifest 和 frame-score 训练工具。
- `tools/asr/qwen/`：Qwen3-ASR SFT 数据、云端训练脚本和 probe。
- `tools/asr/diagnostics/`：ASR / forced-alignment 诊断、fallback 度量和失败样本导出。
- `tools/boundary/`：Boundary Refiner 数据集构造、训练和 source manifest 工具。
- `tools/boundary/ja/`：SpeechBoundary-JA 数据、训练、评测、边界分析和全工作流工具。
- `tools/subtitles/`：字幕 postprocess、cue planner、speaker sidecar 和审计校准工具。
- `tools/audits/`：人工审计 manifest、素材切片和 HTML 审计页工具。
- `src/llm/`：翻译 prompt、cache、glossary、API patch 和 translator。
- `src/subtitles/`：SRT writer、字幕选项和字幕 QC。
- `src/web/`：FastAPI 接口和静态前端。

常用测试：

```bash
PYTHONIOENCODING=utf-8 uv run pytest tests/test_config.py tests/web/test_jobs_api.py tests/test_asr_backend_dispatch.py
PYTHONIOENCODING=utf-8 uv run pytest tests/test_boundary_cache.py tests/test_boundary_candidates.py tests/test_boundary_planner.py tests/test_chunk_packer.py tests/test_pipeline_chunk_config_runtime.py
PYTHONIOENCODING=utf-8 uv run pytest tests/test_asr_alignment_diagnostics.py tests/test_alignment_quality.py
```

构建 Windows Release：

```powershell
.\packaging\build_windows.ps1 -Clean
```

构建产物位于 `dist\JAVTrans\JAVTrans.exe`。打包细节见 `packaging/README.md`。
