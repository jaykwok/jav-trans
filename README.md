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
- JAV 短字幕默认策略：speech core 目标约 `3s`，硬上限 `5s`；ASR 输入可以保留 padded context，硬上限 `9s`。forced aligner 失败时 fallback 只使用 speech core 窗口，不把 padding 计入字幕时间轴。
- `chunk growth` 不再作为主要否决 gate，只作为成本和极端爆炸观察指标。90 分钟片子有几百个 ASR chunk 是正常现象；如果边界更准、fallback 更短、ASR 失败不明显恶化，可以接受更多 chunk。
- 最新 NAMH-055 短 core 闭环：`target=3s / max_core=5s / max_padded=9s` 后，fallback core p90/max 为 `3.07/5.00s`，unsafe fallback 为 `0`；新瓶颈转为日文 cue 过密。字幕层已统一让日文-only / 双语都走 cue merge、timing polish、final post-polish merge 和 2-frame gap；日文-only cue replay 已从 `2166` 降到 `2002` blocks，per-minute subtitle count 从 `24.01` 降到 `22.19`，无 overlap。后续继续优化 cue-density planner / low-info vocal，而不是回头把 ASR chunk 放粗。
- 当前 ASR 低信息/重复/幻觉路线：不先加词黑名单、不直接改模型，也不把呻吟、喘息、短促 kana 一律当幻觉删除。先用 diagnostics 生成归因审计页，把样本分成重复/QC reject、空文本/非词、sentinel fallback、低信息人声、QC warn 和 forced 正常对照，再由人工多选标注 `真实低信息人声`、`ASR 幻觉/错听`、`非语音噪声/BGM`、`多人/重叠语音`、`轻声/弱人声`、`上下文切太短`、`chunk 混入多句/噪声`、`文本可用`、`时间轴准确` 等原因。NAMH-055 当前审计页为 `agents/audits/asr-attribution-namh055-audio/index.html`，共 84 条、每类 14 条；后续依据人工分布决定 cue planner、ASR QC 或训练 hard-negative，而不是先拍脑袋调阈值。
- Boundary Refiner 当前使用 deterministic bootstrap 策略，默认开启；learned refiner checkpoint schema 已固定为 `boundary_refiner_v1`。backbone 入口只保留 `transformers.Mamba2Model`，直接对应 Hugging Face Transformers 的纯 PyTorch Mamba2 实现；不再暴露 `mamba2`、`torch_mamba2`、BiGRU、TCN 或 Transformer fallback，也不把 Linux-only `mamba-ssm`、Triton 或自定义 CUDA kernel 作为默认依赖。
- 字幕层后置修正已进入代码：ASR QC 对 kana-only 语气词/呻吟重复默认保留并标记 review；cue-stage planner 只保留 timing/readability/fallback 风险、2-frame gap、显示时长和短 cue 合并这些字幕观感核心。运行时人声聚类、声纹 sidecar、Web speaker 显示和低能量 pre-ASR drop 已从 active tree 移除；`speaker_proxy` / `speaker_turn` 只保留为 Boundary Refiner synthetic data 的训练元数据。
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
- NVIDIA 独立显卡和较新的驱动。默认 batch 配置按单阶段峰值适配 6GB 级显存；更小显存也可运行，但速度会明显变慢，并需要在 `.env` 中调低 `ASR_BATCH_SIZE_BY_REPO`、`ALIGNER_BATCH_SIZE` 和 `ALIGN_LONG_CHUNK_BATCH_SIZE`。
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

翻译阶段的缓存分三层：

- 本地 batch cache：`translation_cache.jsonl`，用于完全相同 cue/timing/prompt 的精确复跑与 crash resume。
- 本地 translation memory：`translation_cache.memory.jsonl`，按日文文本、目标语言、词汇表、人物参考、prompt 语义版本和模型族复用译文；时间轴或 batch 变化时仍可命中。
- Provider prompt cache：通过稳定的 system prompt / 全片 JSON 前缀和可选 warmup 降低 API 成本；它不等同于本地翻译缓存。

短片 / 审计片段的单请求翻译也会走同一套本地缓存。自动提取的全片术语表会按全片日文文本 hash 存储，避免多个任务共享同一路径时串用旧术语表。

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
ASR_BATCH_SIZE=auto
ASR_BATCH_SIZE_BY_REPO=jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=48,jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=12
ALIGNER_BATCH_SIZE=48
ALIGN_LONG_CHUNK_BATCH_SIZE=48
ASR_CHUNK_MIN_DURATION_S=0.25
ASR_CONTEXT_RESET_GAP_S=0.5
```

当前批量默认按 RTX 4060 Ti 8GB、NAMH-055 短 chunk benchmark 的 6GB 级显存目标设置。`ASR_BATCH_SIZE=auto` 会按 `ASR_BACKEND` 查 `ASR_BATCH_SIZE_BY_REPO`：0.6B 使用 `48`，1.7B 使用 `12`。0.6B batch `48` 峰值约 `5139 MiB`，0.6B batch `64` 峰值约 `6200 MiB` 并超过目标；1.7B batch `16` 峰值约 `6119 MiB`（当时空闲底噪约 `1.1GB`），因此默认留到 `12`；forced aligner batch `48` 全量对齐峰值约 `4349 MiB`。如果本机显存更小或同时有其他 GPU 进程，请先调低 `ASR_BATCH_SIZE_BY_REPO`，再降低 forced aligner batch。

Boundary Refiner 配置：

```env
BOUNDARY_CACHE_ENABLED=1
BOUNDARY_CACHE_DIR=./tmp/cache/boundary
BOUNDARY_FEATURE_FRAME_HOP_S=0.02
BOUNDARY_REFINER_ENABLED=1
BOUNDARY_REFINER_MODEL_PATH=src/boundary/checkpoints/boundary_refiner.pt
BOUNDARY_REFINER_BACKBONE=transformers.Mamba2Model
BOUNDARY_REFINER_DEVICE=auto
BOUNDARY_REFINER_THRESHOLD=0.5
BOUNDARY_FRAME_SEQUENCE_LEFT_CONTEXT_S=0.60
BOUNDARY_FRAME_SEQUENCE_RIGHT_CONTEXT_S=0.60
BOUNDARY_FRAME_SEQUENCE_MAX_PTM_DIMS=64
BOUNDARY_FRAME_SEQUENCE_INCLUDE_MFCC=1
BOUNDARY_PLANNER_MAX_CORE_CHUNK_S=5.0
BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S=9.0
BOUNDARY_PLANNER_TARGET_CHUNK_S=3.0
BOUNDARY_PLANNER_MIN_CHUNK_S=0.4
BOUNDARY_PLANNER_START_WEIGHT=1.5
BOUNDARY_PLANNER_TARGET_PADDING_S=2.0
BOUNDARY_PLANNER_MAX_SPLITS_PER_SEGMENT=16
BOUNDARY_PLANNER_SEQUENCE_BATCH_SIZE=256
BOUNDARY_DP_CHUNK_BASE_COST=0.04
BOUNDARY_DP_OVER_TARGET_WEIGHT=0.30
BOUNDARY_DP_FAR_OVER_TARGET_WEIGHT=1.50
BOUNDARY_DP_UNDER_MIN_WEIGHT=0.20
BOUNDARY_DP_LONG_GAP_WEIGHT=0.35
BOUNDARY_DP_SPLIT_MERGE_WEIGHT=0.35
```

Timing 分两层处理：

- Boundary planner 使用秒级 speech-core 参数：`BOUNDARY_PLANNER_TARGET_CHUNK_S=3.0`、`BOUNDARY_PLANNER_MAX_CORE_CHUNK_S=5.0`。这个 core 是 fallback / 字幕时间轴窗口，目标是更接近一句台词，避免 8-9s 多句字幕。
- ASR 输入使用单独的 padded window：`BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S=9.0`。padding 只服务识别上下文，不作为 forced-aligner 失败后的 fallback 字幕窗口。
- `BOUNDARY_FEATURE_FRAME_HOP_S` 只是 frame-score 网格的 fallback，默认 `0.02s`，不代表视频帧率。
- 字幕显示 / timing polish 使用真实视频 FPS。流程会探测源视频帧率并计算 `frame_duration_s=1/fps`，再按 Netflix-style 规则保留 `2 frames` gap、压缩前一条 end、限制最短/最长显示时长。24fps、29.97fps、60fps 的两帧 gap 会自然换算成不同秒数。
- `SUBTITLE_MERGE_ADJACENT` 是字幕层总开关，控制日文-only 和双语输出的相邻短 cue 合并；`SUBTITLE_DENSE_CUE_MERGE_*` 只在该总开关开启时生效。cue merge 不会让 ASR chunk 变长，只影响最终 cue plan。当前 writer 会在 timing polish 后再跑一次受控短 cue 合并，避免“第一遍按 raw gap 不能合、polish 压到 2-frame gap 后仍残留”的微 cue。

本策略来自两类依据：公开检索没有找到可靠 JAV 单句字幕时长统计；本地 `fallback-window-risk-audit-video` 40 条 `8.75-9.10s` fallback-window 人工审计中，`needs_split` 为 `35/40`。因此 `9s` 不再作为 subtitle/fallback core 默认上限，只保留为 ASR padded context 上限。Netflix timing guidelines 强调 in-time 贴近音频起点、字幕间保留 2-frame gap；Netflix Japanese guide 还限制日语读速到每秒最多 4 个字符，OOONA/ATA/Karamitroglou 等通用字幕建议也更支持短 cue 或拆分长音频。

当前 `3s / 5s / 9s` 不是手写魔法数，而是从本地 Galgame 100k clean speech-island 时长分布推导：

```text
source p5 / p50 / p80 / p90 = 1.05 / 4.50 / 7.61 / 9.56s
target_domain_speedup = 1.5  # Galgame 台词相对 JAV 目标域更慢，先按 1.5x 保守折算

target_core = round_0.1(clamp(p50 / speedup, 2.0, 3.5)) = 3.0s
max_core = floor_0.5(clamp(p80 / speedup, target_core + 1.0, 5.5)) = 5.0s
target_padding = round_0.1(clamp((p90 - max_core) / 2, 1.0, 2.0)) = 2.0s
max_padded = floor_0.5(min(p90, max_core + 2 * target_padding, 9.0)) = 9.0s
min_chunk = round_0.05(clamp(p5 / speedup * 0.60, 0.25, 0.50)) = 0.4s
```

复算命令：

```bash
PYTHONIOENCODING=utf-8 UV_CACHE_DIR=agents/temp/uv-cache \
  uv run --no-sync python tools/boundary/recommend_timing_params.py \
  datasets/train/boundary-sources/galgame-asr-100k-ogg/summary.json --env
```

当前 canonical `BOUNDARY_REFINER_MODEL_PATH` 固定为 `src/boundary/checkpoints/boundary_refiner.pt`。如果该文件不存在，运行时会使用 deterministic bootstrap refiner；如果文件存在，则加载 learned Boundary Refiner。训练好的 learned refiner 使用 `boundary_refiner_v1` checkpoint schema；checkpoint 内容 SHA1、模型路径、`transformers.Mamba2Model` backbone、refiner device、candidate extractor version 和 planner config 都会进入 boundary-cache signature，更新权重或设备策略会自动触发 cache miss。`BOUNDARY_REFINER_DEVICE=auto` 会在 CUDA 可见时把这个小模型放到 GPU；如果它落到 CPU，Transformers Mamba2 pure PyTorch naive path 会明显拖慢全片 planner sweep。

Frame/window sequence refiner 是当前主线训练入口。`src/boundary/sequence_features.py` 是唯一 schema authority：训练和 runtime 都从这里读取 feature config、feature dim、feature names 和 `feature_schema_hash`，禁止在训练脚本、runtime 或 planner 中手写维度常量。`runtime_adapter=frame_sequence_v1` 的 checkpoint 必须带 `feature_schema=frame_sequence_features_v1`、`feature_schema_hash` 和完整 `feature_signature`；运行时会用 SpeechBoundary-JA 导出的 PTM/MFCC frame windows 重新计算 feature names/hash，不匹配就直接报错，不做旧 checkpoint 兼容或静默回退。

Feature cache 的长期规则：大体积 frame/window 特征不写入 boundary-cache JSON。boundary-cache 只保存 ASR-facing packed spans、决策诊断和 signature；训练或诊断需要持久化密集特征时，使用对应数据集目录下的 `.npz` / `.pt` sidecar，并把 `feature_schema_hash` / checkpoint SHA1 / planner config 写进 signature，保证公式或模型变化会触发 cache miss。

Boundary cache 只作为加速和诊断缓存，不作为最终质量结论。若某个实验只能用旧 cache 做近似复算，允许直接重跑 SpeechBoundary-JA / Boundary Refiner / planner；涉及 ASR empty、重复循环、forced-aligner fallback、字幕观感或上线判断时，以完整 workflow 重跑结果为准。DP planner / 风险映射本身是确定性计算，不因为跑在 CPU 上就变成近似。aligned-segments cache 当前签名版本为 `5`，会记录完整 `SubtitleOptions.signature()`；subtitle merge/polish/读速参数变化会自动 miss cache，避免复用旧 cue plan。

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
2. 数据重建：清除旧 gap-only / BiLSTM / endpoint-head 训练格式，使用 Galgame 与 anime clean speech islands 重新生成 sequence dataset。每条样本包含多 island、touching speech、short/long gap、real negative gap、BGM/noise、轻量 overlap 和 source/utterance switch。过渡期 gap dataset 只允许作为 loader smoke 或 warm-up，不作为最终 Seq2Seq 目标。
3. 模型升级：Boundary Refiner backbone 只保留 `transformers.Mamba2Model`。输入从 per-gap summary 升级为连续窗口序列，特征包含 Qwen PTM、MFCC、energy、speech probability、cut probability、candidate metadata；输出逐候选 split / merge / refine score 和可选 boundary offset。当前 frame/window sequence dataset、trainer、`FrameSequenceBoundaryRefiner`、runtime feature provider、schema/hash 校验和 PackedChunk 决策诊断已接通；当 checkpoint metadata 标记 `runtime_adapter=frame_sequence_v1` 时，主 pipeline 会让 SpeechBoundary-JA 导出低维 PTM/MFCC frame windows并交给 sequence refiner 做 gap 决策。
4. Planner 接入：`pack_speech_segments()` 只 materialize planner 输出，不再承担固定 gap 合并策略。当前 planner 已能读取 refiner decision 并写入 PackedChunk 诊断字段；sequence refiner 已改成 bounded batch 打分，并接入轻量 DP / Viterbi-style constrained planner。当前 DP v2 是第一版可解释 baseline：cost 主要由模型 split/merge score、speech-core duration、hard core max 和长 gap penalty 组成；它能避免退回局部 greedy，但还不是最终字幕目标函数。`BOUNDARY_PLANNER_SEQUENCE_BATCH_SIZE` 默认 `256`，用于避免 pure PyTorch Mamba2 naive path 在全片候选一次性推理时过慢。Candidate extractor v2 新增 overlong single-island soft candidate：当没有 hard cut / valley 时，在 target 附近用 soft cut score 或 speech-score valley 找切点，避免 20-30s 单 island fallback 粗时间轴。该变更会触发旧 boundary-cache 重建。
5. GPU 闭环：先跑 synthetic exact truth，再跑匿名样片 A / NAMH-055。主 gate 是 start boundary、fallback core duration、long/gap-crossing chunk、ASR empty / hallucination 和字幕观感；chunk 数只作为成本指标，不作为主要否决项。fallback / sentinel 数只作为观察项，因为当前 Qwen3-ForcedAligner 没做 JAV / galgame 目标域 finetune，可能和 Qwen ASR SFT 输出风格不完全匹配。匿名样片 A frame-sequence greedy、DP v2、soft-candidate DP v2 和 fallback-window 修正都已跑通：soft-candidate v2 消除了 `20s+` fallback 粗 chunk，fallback-window 修正把 forced-aligner 失败后的插值窗口从 padded chunk 收窄到 speech core。NAMH-055 已按 target `3s`、max core `5s`、max padded `9s` 完整闭环，fallback core 已安全；当前优先级转到 cue-density planner、低信息人声/重复语气词审计和 ASR/QC feedback。
6. ASR feedback：supervised 稳定后再引入 preliminary ASR 文本、token confidence、local CER、aligner sentinel、fallback duration 和 QC reject 作为 dense reward，做 RL / DPO。Unified Joint Model 保留在 backlog，等 SpeechBoundary-JA 能稳定产出 pseudo boundary labels 后再评估。

下一轮 planner / refiner 校准方向：边界层不再优先放宽 core。真实 boundary-only DP sweep 显示：单纯把 target/cost 调紧只能小幅降低 p90；补上 overlong speech island soft candidate 后，完整 ASR/QC/aligner 闭环确认最坏 padded fallback 已从 `20.72s` 降到 `13.10s`；再把 fallback 插值窗口从 padded chunk 收窄到 speech core 后，实际 fallback 时间轴主要受 planner core 约束。NAMH-055 短 core 闭环已把 fallback core p90/max 压到 `3.07/5.00s` 且 unsafe fallback 为 `0`，所以当前要把字幕质量目标从“粗 fallback 时间轴”转到 cue 层：calibrated model NLL、字幕最小显示时长、2-frame gap、CPS/readability、短 cue 合并、low-info vocal 标记和 ASR/QC feedback。Netflix timing 规则支持 `start/in-time` 优先、`end/out-time` 后置压缩和 2-frame gap；SubER / OptiSub 类字幕评测也说明字幕质量要同时看 timing、segmentation、显示时长和阅读速度。REBORN 的 RL boundary segmentation 可作为后续 ASR feedback / DPO 参考，但当前先保留确定性 DP 和可审计 cue planner baseline。

Backlog：等 SpeechBoundary-JA 能稳定产出 pseudo boundary labels 后，研究不依赖 forced aligner 的直接字幕边界 / timeline model。该路线把“输出字幕文本和时间轴边界”作为主任务，forced aligner 只作为审计或 teacher 信号之一，不再把 fallback 数量当作上线 gate。

训练数据构造要注意：旧 cutpoint 数据集主要提供 split supervision（贴连换人、短 gap 换人、可删除长 gap），不能单独当作完整 merge/split 训练集。正式训练前需要混入 clean speech-island 原料构造 same-utterance merge-positive，例如：

- `litagin/Galgame_Speech_ASR_16kHz`：已验证可作为精确 speech-island 原料。
- `joujiboi/japanese-anime-speech-v2`：约 29 万条、约 450 小时 anime / visual novel audio-text pairs，split 为 `sfw` / `nsfw`，适合作为额外 clean speech-island 原料源。JAV 目标域训练可以提高 `nsfw` 权重，但仍需保留 galgame / sfw 防止过拟合单一来源。

当前本地可复用 HF 源数据：

| 来源 | 本地路径 | 条数 | 备注 |
| --- | --- | ---: | --- |
| `litagin/Galgame_Speech_ASR_16kHz` | `datasets/train/boundary-sources/galgame-asr-100k-ogg/` | 100000 | 原始 OGG + TXT + manifest，约 `3.0G` 文件系统占用；后续 Boundary Refiner 数据构造优先复用它，不再现场流式下载。 |
| `joujiboi/japanese-anime-speech-v2` nsfw | `datasets/train/fusionvad-ja/v1-boundary-sources/japanese-anime-speech-v2-nsfw-512/` | 512 | 已 materialize 为 WAV，用作 anime / NSFW clean speech island seed。 |
| `joujiboi/japanese-anime-speech-v2` sfw | `datasets/train/fusionvad-ja/v1-boundary-sources/japanese-anime-speech-v2-sfw-256/` | 256 | 已 materialize 为 WAV，用作泛化 seed。 |

这些目录是本地训练资产，默认不提交。`datasets/train/fusionvad-ja/v1-23-boundary-refiner/` 里还有约 `31G` 的旧 synthetic / feature cache，可复现实验但不是源数据；后续正式重建 sequence dataset 时应优先从上表源池重新生成。

正式训练建议先生成 class balance summary，确认既有 merge-positive、split-negative、touching-speech、long-gap、source/utterance switch 和 low-value non-speech。单类 dataset 只允许用于 loader smoke，不进入主训练。

评估脚本按当前 diagnostics schema 断兼容运行：`fallback_type != none` 才算 alignment fallback，`fallback_subtype` 只作为 reason 原样统计，`sentinel_lines` 非空才计 sentinel fallback；缺少 `fallback_type` 的旧 diagnostics 会直接报错。

---

## 输出与缓存

- `video/<视频名>/`：正式字幕、质量报告和人工质检报告。
- `models/`：Hugging Face 模型缓存。
- `tmp/jobs/<job_id>/`：Web / pipeline 单次任务临时目录；`JOB_TEMP_DIR` 默认就是 `./tmp/jobs`，翻译 batch cache 和 translation memory 默认写在对应 job 目录下。
- `tmp/chunks/`：ASR wav chunk 和 crash-resume checkpoint 的一次性运行目录。
- `tmp/cache/boundary/`：SpeechBoundary-JA frame score 到 Boundary Planner 输出的 boundary-cache v1。
- `tmp/cache/torch/`、`tmp/cache/hf/`：torch / Hugging Face 运行缓存，不和 `models/` 顶层模型目录混放。
- `tmp/log/`：启用运行日志后的任务日志。
- `datasets/`：本地训练、验证、测试数据归档，默认 ignored。
- `agents/temp/`：研究脚本、smoke、临时日志和中间产物。
- `agents/audits/`：可长期复查的本地审计页。统一从 `agents/audits/index.html` 进入；审计产物直接放在该目录下，不再套 `speech-boundary-ja/` 子目录；导航页始终指向最新审计且不使用自动跳转。用 live-server 审计时从项目根目录启动 `live-server --middleware=tools/audits/live_server_audit_middleware.js`，导航页删除按钮会调用 `POST /__audit_api__/delete-audit`，并把审计目录移动到 `agents/rm/audit-deletions/`；不再保留独立 Python 审计服务。fallback-window 和 ASR 归因审计生成器都支持 `--media-mode audio|video`，NSFW 场景可生成音频版审计页，播放时仍会同步显示完整日语字幕。

成功运行后默认删除一次性 job 临时目录；保留可复用缓存，例如 `models/`、`tmp/cache/boundary/` 和 Web 状态。旧版 `temp/` 已断兼容废弃，active code 不再写入该目录；如果本地还有旧运行数据，可以手动归档或删除。

---

## 字幕与文本策略

- 系统维护 `display_text` 和 `align_text` 两份文本。
- `display_text` 用于最终字幕显示，只做 Unicode NFKC、空白归一、换行折叠和展示安全处理。
- `align_text` 只给 forced aligner 使用，可删除标点、emoji、音乐符号和不可发音装饰符。
- `ASR_CONTEXT` / `ASR_HEAD_CONTEXT` 只作为 Qwen ASR 提示词，不再作为字幕后处理删除规则。演员名、片名、术语或自我介绍可能在片中真实说出，最终字幕不做“像提示词就删”的硬过滤。
- 不使用具体词黑名单，不直接删除 `ん`、`あ`、喘息、呻吟、拟声、低信息短句或常见台词。
- 人工审计标签区分 hard drop、低信息人声和时间轴问题：`删除/无字幕价值` 只用于纯噪声、BGM、静音、机械声等硬丢弃片段；`低信息人声/呻吟` 用于呻吟、喘息、笑声、叹息、短促拟声等目标域可转写但字幕价值低的发声，可与 `整条文本可用`、`时间轴准确` 同时选择。时间轴不准时优先细分为 `开头偏早/偏晚`、`结尾偏早/偏晚`、`时间窗过长/过短`、`跨无声/噪声`，不要只写备注。
- cue merge 审计支持左右侧向标签：`上句/下句可用`、`上句/下句文本错`、`上句/下句无字幕价值`、`上句/下句低信息`。当上下两条一好一坏时，优先用侧向标签表达，不再只靠备注。
- 重复循环、低置信、文本/音频比例异常、align-text-empty、fallback、`asr_review_uncertain` 只作为 QC / 诊断 / 样本池信号。ASR QC 不再提供删除不确定文本的开关；kana-only 非词汇重复默认 `preserve_with_review`，最终字幕不会因为这些诊断被清空。审计页里的“重复建议需复核”也只是人工标签，不会自动把 suggested text 写回最终字幕。
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

短 chunk 默认批量以“单阶段显存峰值适配 6GB 级显卡”为目标，而不是榨满整张 8GB 卡。当前实测默认是 0.6B ASR `48`、1.7B ASR `12`、forced aligner `48`；0.6B ASR `64` 虽能跑一部分 chunk，但持续采样会越过 6GB，不作为默认。

DP sweep 的性能要分阶段看：SpeechBoundary-JA PTM 和 learned Boundary Refiner 可在 CUDA 上跑，但 planner profile 循环、风险映射和 JSONL 写入是 CPU-bound。后半段 CPU 跑不代表近似；它仍是基于真实 boundary/refiner 输出的确定性规划。看到后半段 CPU 占用高不一定代表没启用 CUDA；以 summary 中的 `refiner_signature.actual_device=cuda:0` 和边界 scorer 日志为准。

### 长任务怎么排查

启用运行日志后，日志会写入 `tmp/log/` 或任务输出目录。反馈问题时请保留 `.run.log`、质量报告和对应 SRT。

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
- `tools/subtitles/`：字幕 postprocess、cue planner 和审计校准工具。
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
