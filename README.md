# JAVTrans

JAVTrans 是一个本地字幕生成工具，面向 Windows + NVIDIA 显卡，也可在 WSL2 / Linux 下源码运行。它把视频处理成中文字幕或中日双语字幕，并把音频准备、VAD 分段、ASR、强制对齐、字幕时间轴归一化、LLM 翻译和质量报告串成一条本地优先的流水线。

项目目标：本地完成视频、音频、VAD、ASR 和时间轴重计算；LLM 只负责翻译、术语一致和口吻连贯，不负责脑补剧情或修正 ASR 误听。

致谢：[WhisperJAV](https://github.com/a63n/WhisperJAV) 为本项目早期路线提供了重要参考。本项目后续扩展了 FusionVAD-JA、Qwen3-ASR 目标域 SFT、pre-align、forced-alignment fallback 和字幕时间轴 polish。

实验历史、路线取舍、调试记录和参考来源见 [HISTORY.md](HISTORY.md)。

---

## 当前状态

- 默认 ASR：`jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`。ASR backend key 就是 Hugging Face repo ID；首次下载会写入合法本地目录 `models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame`。
- 默认 VAD：`fusionvad_ja`，使用随仓库提交的 FusionVAD-JA v1.19b split-cut endpoint refiner 和 v1.21 drop-gap imitation head。
- 可选 ASR：`jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`，本地缓存目录为 `models/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame`。
- FusionVAD-JA 特征层：`jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`，同时作为轻量 ASR probe 和 VAD frozen feature。
- 当前 VAD 主线：speech-island 边界优先。目标不是把所有可疑人声都合成高召回 proposal，而是尽量切成“一句台词一个 chunk”，减少多句台词、长 gap、BGM/噪声和换人片段被揉成一坨。验收优先看 `start` p90/p95、fallback chunk duration、long/gap-crossing chunk、`start_weighted_speech_recall`、ASR empty / hallucination 是否恶化；旧 `speech_duration_recall` 保留作历史对比。recall guardrail 暂按 `>=0.93`，必要时可为更准边界牺牲一部分整段 recall。
- `chunk growth` 不再作为主要否决 gate，只作为成本和极端爆炸观察指标。90 分钟片子有几百个 ASR chunk 是正常现象；如果边界更准、fallback 更短、ASR 失败不明显恶化，可以接受更多 chunk。
- 默认 head 已从单一 `cut` 目标改为 `cut_drop` / `cut_point` 双目标：前者表示 silence / noise / BGM 等可删除 gap，后者表示贴连台词或换人边界，只能切分不能删除音频。
- 实验性 pre-ASR boundary packing / R19 reward planner 仍默认关闭。v1.19b 先作为 split-cut 默认 head；v1.23 residual cut split 已把匿名样片 A 的 fallback p90 从 `28.47s` 降到 `12.91s`，证明“二次切 residual long child”方向有效；但 chunk `236 -> 862`、字幕 `1018 -> 2764`，短字幕和 repeat-loop reject 增多，因此暂不替换默认。
- v1.23 后置修正已进入代码：字幕层新增 dense short cue merge，ASR QC 对 kana-only 语气词/呻吟重复默认保留并标记 review，speaker sidecar 新增离线 adjacent embedding probe。真实 v1.23 replay 显示当前 dense merge 对匿名样片 A 没有触发；离线 cue-stage planner 宽松扫描可把 `short_segment_ratio` 从 `0.205` 降到 `0.124` 且不引入 overlap，接入 alignment diagnostics 后仍可降到 `0.134`。ERes2NetV2 sidecar 已跑通，标准化 sweep 中 th85 后为 `0.144`、th95 后为 `0.137`。人工审计显示原始 th95-extra 和 th95-constrained extra 都有较高问题率，th95-constrained 不应直接推广为默认；当前安全候选回到 `th85` 基线，并把 speaker-change / 读速 / fallback 风险作为 cue-stage review、penalty 和后续 ASR/QC hard-negative 来源。当前最新审计是 `th85-high-risk v3-side-labels` 40 条，输出 `manual_cue_planner_th85_high_risk_labels_v3_side_labels.jsonl`，用于按左右侧向标签确认 th85 基线自身是否安全。
- speaker sidecar 路线：首选 ERes2NetV2 / 3D-Speaker，CAM++ 只作 baseline。它不替代 VAD，只在 pre-align / cue 阶段对相邻 speech island 提 speaker embedding，辅助判断是否跨 speaker，避免把男女/多人对话合并成一条 cue。当前已提供 `energy_mfcc` smoke extractor 验证 JSONL 链路，并接入可选 `modelscope_eres2netv2` backend；`energy_mfcc` 不是可靠 speaker 模型，真实约束需要安装 ModelScope 可选依赖并下载 ERes2NetV2/3D-Speaker embedding 模型后重跑。
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
uv run --no-sync python launcher.py
```

默认地址为 `http://127.0.0.1:17321`。

Linux / WSL2 下如果只启动浏览器版 Web 服务，也可以直接运行：

```bash
PYTHONPATH=src .venv/bin/python -m uvicorn web.app:create_app --factory --host 127.0.0.1 --port 17321
```

Web 提交是否使用 CUDA 取决于后端服务进程是否能看到 GPU，而不是浏览器本身。受限 sandbox 内启动 Web 时可能看不到 CUDA；跑完整 VAD/ASR/ForcedAligner smoke 时需确认运行日志包含 `cuda_available=True`、`device=cuda:0` 和 FusionVAD-JA `actual_device=cuda`。

---

## 使用流程

1. 打开网页控制台。
2. 选择视频文件。
3. 选择字幕模式、ASR 后端和翻译设置。
4. 提交任务。
5. 在输出目录查看 SRT、质量报告和日志。

勾选“不翻译（仅日文字幕）”时，流水线仍会执行 VAD、ASR、ASR QC、forced alignment 和字幕时间轴归一化，但跳过 LLM 翻译，最终输出 `<视频名>.ja.srt`。这也是验证本地 VAD/ASR/对齐链路的推荐 smoke 模式。

主流水线：

```text
视频
-> 音频准备
-> FusionVAD-JA frame mask
-> frame-based chunk packing / speech-island packing
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
| FusionVAD-JA frozen feature | `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame` | `models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame` |
| FusionVAD-JA v1.19b head | 随仓库提交 | `src/vad/fusionvad_ja/checkpoints/fusionvad_ja_v1_19b_splitcut_touch4096_endpoint_refiner.pt` |
| FusionVAD-JA v1.21 drop-gap head | 随仓库提交 | `src/vad/fusionvad_ja/checkpoints/fusionvad_ja_v1_21_dropgap_imitation_head.pt` |
| Forced aligner | `Qwen/Qwen3-ForcedAligner-0.6B` | `models/Qwen-Qwen3-ForcedAligner-0.6B` |

常用配置：

```env
ASR_BACKEND=jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame
ASR_VAD_BACKEND=fusionvad_ja
ASR_MODEL_ID=
ALIGNER_MODEL_ID=Qwen/Qwen3-ForcedAligner-0.6B
ASR_DTYPE=bfloat16
```

实验性长 chunk 切分开关：

```env
ASR_PRE_ASR_RISK_SPLIT_ENABLED=1
ASR_PRE_ASR_RISK_SPLIT_THRESHOLD=1.0
ASR_PRE_ASR_RISK_SPLIT_CONTINUOUS_THRESHOLD=2.0
ASR_PRE_ASR_RISK_SPLIT_MIN_GAP_FRAMES=6
```

这一路线用于复现实验性 pre-ASR boundary packing。R18 gap-first 在匿名样片 A 的 GPU 闭环中没有显著改善 sentinel / unsafe fallback，因此不建议作为日常默认开关；开启后会改变 VAD chunk cache key。

当前 FusionVAD-JA 研究路线：

```text
v1.19b: split-cut endpoint refiner, default checkpoint
v1.20: boundary-first supervised refiner
       start/end error、fallback chunk duration、gap crossing 优先，frame recall 作为 guardrail
       first-pass 256-step 已跑通但不达默认替换门槛，当前仍保留 v1.19b 默认
v1.21: reward planner -> offline teacher -> imitation learning
       学 keep / split / drop-gap，不直接把 planner 上线
       multitask split/drop-gap imitation 已验证会退化成常数策略
       drop_gap-only offline packer 当前最强离线点为 512-th080：
       recall 0.9333，start p90 7.86s -> 4.54s，long chunk 3126 -> 1763
       匿名样片 A GPU 闭环未过 gate：forced 提升但 vad_coarse/unsafe fallback 仍偏高
       仍暂不替换默认，下一步需要更强 candidate policy / boundary objective
v1.22: supervised cutpoint/boundary dataset + head
       用 Galgame exact-island 合成贴连、短 gap、多 island、随机 source 边界
       先训练 cut_point / cut_drop / start / end；RL 只作为 v1.23 候选切点 planner 微调
       4096 first-pass 已完成；从 v1.19b 初始化能保住 recall 并学到 cut 信号
       匿名样片 A GPU 闭环未过 gate：cut-as-boundary 后仍有大量 28.47s single-island fallback
       当前结论：cut 不能直接删 speech；下一步要补连续长 island 内部句级切点目标
v1.23: optional constrained RL over candidate cuts
       只在 VAD valley / cut_drop / cut_point / endpoint 候选上决策，避免逐帧任意切
       residual cut split GPU 闭环已完成：fallback p90 明显下降，但短字幕/QC reject 增多
       下一步不是直接默认开启，而是补 subtitle timing polish、ASR repeat-loop 处理和 speaker sidecar probe
```

`models/`、`datasets/`、`agents/temp/`、`agents/audits/` 是本地运行资产，不应提交训练数据或大模型权重。仓库只提交小型 FusionVAD-JA head。

---

## 输出与缓存

- `video/<视频名>/`：正式字幕、质量报告和人工质检报告。
- `models/`：Hugging Face 模型缓存。
- `temp/vad-cache/`：VAD/chunk 边界缓存。
- `temp/jobs/`：Web 任务临时目录。
- `temp/log/`：启用运行日志后的任务日志。
- `datasets/`：本地训练、验证、测试数据归档，默认 ignored。
- `agents/temp/`：研究脚本、smoke、临时日志和中间产物。
- `agents/audits/`：可长期复查的本地审计页。统一从 `agents/audits/index.html` 进入；审计产物直接放在该目录下，不再套 `fusionvad-ja/` 子目录；导航页始终指向最新审计且不使用自动跳转。旧审计产物清理时移入 `agents/rm/`。

成功运行后默认删除一次性 job 临时目录；保留可复用缓存，例如 `models/`、`temp/vad-cache/` 和 Web 状态。

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

Codex sandbox 或部分受限环境可能隔离 GPU。跑全片 VAD/ASR/ForcedAligner、Torch CUDA、feature cache 或训练时，需要确认日志中出现：

```text
actual_device=cuda
model_param_device=cuda:*
```

FusionVAD-JA 的 feature cache、训练、逐帧概率导出和全片 workflow 都按“能 CUDA 就提权 CUDA”处理；不要为了省事让 CPU 跑大规模评测，否则会把等待时间和中间产物放大，且容易误判进度。

### 长任务怎么排查

启用运行日志后，日志会写入 `temp/log/` 或任务输出目录。反馈问题时请保留 `.run.log`、质量报告和对应 SRT。

---

## 开发

主要代码位置：

- `src/main.py`：主流程编排。
- `src/core/`：配置和任务上下文。
- `src/pipeline/`：音频、缓存、输出、质量报告和阶段日志。
- `src/asr/`：ASR、forced alignment、prealign 和转写流程。
- `src/vad/`：VAD 后端。
- `src/vad/fusionvad_ja/`：FusionVAD-JA 实现。
- `tools/asr/qwen/`：Qwen3-ASR SFT 数据、云端训练脚本和 probe。
- `tools/asr/diagnostics/`：ASR / forced-alignment 诊断、fallback 度量和失败样本导出。
- `tools/vad/fusionvad_ja/`：FusionVAD-JA 数据、训练、评测、边界分析和全工作流工具。
- `tools/subtitles/`：字幕 postprocess、cue planner、speaker sidecar 和审计校准工具。
- `tools/audits/`：人工审计 manifest、素材切片和 HTML 审计页工具。
- `src/llm/`：翻译 prompt、cache、glossary、API patch 和 translator。
- `src/subtitles/`：SRT writer、字幕选项和字幕 QC。
- `src/web/`：FastAPI 接口和静态前端。

常用测试：

```bash
.venv/bin/python -m pytest tests/test_config.py tests/web/test_jobs_api.py tests/test_asr_backend_dispatch.py
.venv/bin/python -m pytest tests/test_chunk_packer.py tests/test_vad_chunk_cache.py tests/test_pipeline_chunk_config_runtime.py
.venv/bin/python -m pytest tests/test_asr_alignment_diagnostics.py tests/test_alignment_quality.py
```

构建 Windows Release：

```powershell
.\packaging\build_windows.ps1 -Clean
```

构建产物位于 `dist\JAVTrans\JAVTrans.exe`。打包细节见 `packaging/README.md`。
