# JAVTrans

JAVTrans 是一个面向 Windows + NVIDIA 显卡的本地字幕生成工具。它把视频处理成中文字幕或中日双语字幕，提供网页控制台，并把音频准备、VAD / speech-island 分段、ASR、强制对齐、字幕时间轴归一化、LLM 翻译和质量报告串成一条流水线。

项目目标很明确：本地完成视频/音频/ASR 相关重计算，LLM 只负责翻译，不承担 ASR 误听修复、画面脑补或剧情改写。

致谢：[WhisperJAV](https://github.com/a63n/WhisperJAV) 为本项目早期路线提供了重要参考，尤其是面向 JAV 场景的本地字幕流水线。本项目后续在此基础上扩展了 FusionVAD-JA 研究线、Qwen3-ASR 目标域 SFT、pre-align 和字幕时间轴 polish 等优化。感谢 WhisperJAV 作者及其贡献。

---

## 快速开始

### 方式 A：使用 Release 版

如果只是直接使用，优先下载 GitHub Releases 中的 Windows 压缩包。解压后运行：

```text
JAVTrans.exe
```

Release 版已包含 Python 运行环境、FFmpeg、默认 ASR 模型 `efwkjn/whisper-ja-anime-v0.3`，以及默认流程需要的 fusion_lite VAD（WhisperSeg 候选 + Silero gate）、`openai/whisper-base` 特征提取器和 Qwen forced aligner。首次使用仍需要在页面的“翻译设置”中填写 API Key、Base URL 和模型名。其他 ASR 模型会在需要时下载到 exe 同目录的 `models/`。

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
uv run --no-sync python launcher.py
```

默认地址为 `http://127.0.0.1:17321`。页面中选择视频、确认翻译和输出设置，然后提交任务即可。

---

## 功能概览

- 网页控制台：文件选择、ASR 后端、字幕模式、翻译设置、并发 worker、质量报告和临时文件保留都可以在页面里操作。
- 多层缓存：支持音频缓存、ASR checkpoint、`aligned_segments.json`、翻译 cache，以及独立的 VAD/chunk 边界缓存。
- 默认 ASR：引擎默认和 Web 推荐均为 `whisper-ja-anime-v0.3`。
- 支持 ASR 后端：`anime-whisper`、`qwen3-asr-1.7b`、`whisper-ja-1.5b`、`whisper-ja-anime-v0.3`。
- 默认 VAD：`fusion_lite`，以 WhisperSeg 候选为主，Silero overlap、RMS、spectral flux 和时长分数作为辅助。
- 可选 VAD：`whisperseg-adaptive`，用于和默认 `fusion_lite` 对比。
- Adaptive Precision ASR QC：硬拒绝明显幻觉，低风险真实对白可自适应放宽低 `avg_logprob`。
- Forced alignment：ASR 后进行词级强制对齐，失败时保留可诊断 fallback 质量标签。
- F0 / gender：保留为历史可选实验，不再作为当前主线分段或翻译标签策略；后续优先使用 speech-island 边界和 speaker sidecar。
- 翻译前 cue plan：LLM 翻译前先固定字幕时间轴和 cue 数量，SRT writer 不再改变时间轴。
- LLM 翻译：支持 OpenAI-compatible Chat Completions 和 Responses API，保留 reasoning effort、API 格式、目标语言、术语表、worker 数这些可手动配置项。
- 质量报告：默认输出 Markdown，同时保留 JSON sidecar 方便自动化评测。

---

## 当前默认流程

主流水线：

```text
视频 -> 音频准备 -> fusion_lite VAD -> VAD chunk packing / speech-island packing
-> ASR -> Adaptive Precision QC -> Forced Alignment
-> 翻译前空/纯符号段过滤 -> 翻译前 cue plan 时间轴归一化
-> LLM 逐 cue 翻译 -> SRT / quality report
```

当前 ASR 以高召回、可诊断为默认目标。旧的 ASR recovery、temperature fallback、prompt overflow retry 已移除；主 VAD 空结果会直接跳过 ASR；timestamp/alignment fallback 只用于给已确认文本补时间轴，不改写或新增 ASR 文本。Adaptive Precision QC 默认只记录低置信、重复、异常密度等风险信号；只有显式设置 `ASR_QC_DROP_UNCERTAIN=1` 时才会清空高风险 ASR 文本。

### ASR 与 VAD

默认 VAD 是 `fusion_lite`。公开可选的 VAD 后端只保留 `fusion_lite` 和 `whisperseg-adaptive`。Silero 只作为 fusion-lite 系列内部 speech prior，不作为独立主 VAD 暴露。

主 VAD 初始化或推理失败会直接抛错并进入 Web 日志。主 VAD 返回空结果是合法“无语音”结果，不会 fallback 成整段音频转写。

`ASR_LONG_CHUNK_PROFILE=on` 时强制开启 VAD chunk packing；F0/gender 已降级为历史可选实验，不再是当前推荐的长 chunk 处理主线：

```env
ASR_CHUNK_PACKING_ENABLED=1
ASR_CHUNK_PACK_WINDOW_FRAMES=899
ASR_CHUNK_PACK_RESERVE_FRAMES=45
ASR_CHUNK_PACK_TARGET_PADDING_FRAMES=60
ASR_CHUNK_PACK_GAP_MERGE_FRAMES=45
```

VAD chunk packing 已改为帧驱动动态 padding：每个视频任务启动时通过 `ffprobe` 读取真实 FPS，并把 `ASR_CHUNK_PACK_FRAME_HOP_S=1/fps` 注入本次 ASR stage；失败时才按 `30000/1001` 兜底。帧数参数保持固定：Qwen forced-aligner 的 30s feature window 近似为 `899` 帧，预留 `45` 帧安全余量，目标 padding 和可合并 gap 均为 `60/45` 帧。实际左右 padding 会受 chunk 剩余容量和相邻 gap 约束；超长连续 speech 会按 `window - reserve - 2 * target_padding` 的 core 长度切分，并在人工切点两侧保留帧数推导的上下文。

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

仅当 `speech_score < 0.45` 且 `silero_overlap_ratio < 0.05` 时丢弃候选。`fusion_lite_boost` 已从公开模式和代码路径移除。

### FusionVAD-JA 研究计划

FusionVAD-JA 是训练型 VAD 研究线，用于复现 FusionVAD 论文的“PTM 特征 + MFCC + simple addition fusion”思路，并面向日语/JAV/galgame 近域数据做适配。研究代码在 `src/vad/fusionvad_ja/`，CLI 在 `tools/fusionvad_ja/`，临时 smoke / 运行日志输出写入 `agents/temp/fusionvad-ja/`；人工审计入口和可长期复查的审计页统一写入 `agents/audits/fusionvad-ja/`；下载数据、feature cache 和 checkpoint 归档到 ignored `datasets/`。

当前定位：

- 默认 VAD 已切到 `fusionvad_ja`，当前 operating point 为 FusionVAD-JA v1.16 endpoint refiner：`jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame` frozen audio feature + MFCC/energy addition BiLSTM，多头输出 speech/start/end/cut，默认 `speech_threshold=0.020`、`cut_threshold=0.960`、pad `0.2s`。本地缓存目录按 Hugging Face repo 规则为 `models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame`；base Qwen3-ASR-0.6B 不再作为默认特征源，如需做 ablation 可显式改 `FUSIONVAD_JA_PTM=qwen3-asr-0.6b-base` 或 `FUSIONVAD_JA_MODEL_PATH`。
- VAD 目标从“直接喂 ASR 的高召回大块 proposal”调整为两级结构：第一级 FusionVAD-JA 输出高召回 frame-level speech mask，不漏对白、呻吟、喘息、短促人声；第二级 speech-island / endpoint packer 把 mask 切成更接近一句台词的 ASR chunk，避免多句话、长 gap、噪音或多人交替被揉成一坨。
- 新增 fallback-safe boundary gate：forced aligner 成功率不是唯一目标；如果 forced alignment 失败并回退到 `vad_coarse`，该 chunk 的粗时间轴也必须足够短，默认用 `<=8s` 作为可审计/可接受上限。否则即使 ASR 文本正确，字幕也会出现几十秒级长 cue 或 fallback 片段，后置 timing polish 只能压缩 cue end，不能修复 chunk 边界本身。
- 8 分钟附近这类“没明显说话却出现台词”的瑕疵，优先按 non-speech/gap 诱发 ASR hallucination 与 chunk 粒度过粗处理，不再单纯归因到字幕 end 偏长。字幕 timing polish 可以压缩显示 end，但不能阻止 ASR 在多送音频上生成文本。
- F0 gender 标签不再作为当前主线切分或翻译提示。原因是当 VAD chunk 本身过大、包含多 speech island 或男女交替时，F0/gender 会被混合 chunk 稀释，反而给 cue 合并、翻译 prompt 和 speaker 判断引入噪声。相关代码和历史测试暂保留为 legacy/ablation，后续清理另开任务。
- CAM++ / 3D-Speaker / WeSpeaker 的定位改为二阶段 speaker sidecar：只在 speech-island 边界足够细之后，用于判断相邻 island 是否跨 speaker、辅助“不要跨 speaker 合并”。它们不直接替代 VAD，也不做 speech/non-speech hard gate。pyannote 仅作为强 baseline 参考，不进默认依赖。
- Qwen3-ASR-1.7B full SFT 是目标域 ASR 主线，默认 Hugging Face 来源为 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`，本地缓存目录为 `models/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame`；Qwen3-ASR-0.6B full SFT 默认来源为 `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`，既可作为轻量 ASR probe，也作为 FusionVAD-JA frozen feature extractor，避免用户同时下载 base 0.6B 和 fine-tuned 0.6B。
- Forced aligner 暂不 finetune。当前优先级是文本预处理、fallback 质量标签、失败样本池和同口径 held-out 复测。
- Galgame ASR 数据多数已经按语音裁切，因此可把原 clip 当作 speech island，在前后和中间拼接随机长度静音、白噪声、hum、MUSAN/DNS/BGM 或本地 hard-negative，构造精确 `actual_speech_segments`。这条 synthetic timeline 不再只是早期 VAD smoke，而是下一轮 VAD / boundary refiner / aligner bench 的公共数据底座。

当前数据和约束：

- `litagin/Galgame_Speech_ASR_16kHz` 是核心近域 ASR / VAD 弱监督来源；AVA-Speech / VoxConverse 只作为强时间标注 seed；MUSAN / DNS / 本地视频 / 合成 gap 作为 negative 和增强素材。
- 标签 JSONL 保持 `audio_id`、`source`、`duration_s`、`text`、`teacher_segments`、`frame_hop_s`、`speech_frames`、`label_quality`；`teacher_conflict` 只审计，不默认进训练。
- 公开文档、测试 fixture、commit message 和可跟踪报告一律使用匿名样片名，不写真实视频 stem 或含真实 stem 的 `agents/temp/` / `agents/audits/` 路径。

当前路线结论：

- 匿名样片 A 已用当前 v1.9 文本/后处理规则、同一 FusionVAD-JA operating point 复测 base / 200k / full v5 checkpoint-15500。旧 v1.8 对照包含历史黑名单和 direct drop，不再作为主参考。
- 当前规则下三组都处理同一 `337` 个 VAD chunks。base 输出 `806` 段、`829` cues、`8085` 字；200k 输出 `794` 段、`843` cues、`13846` 字；checkpoint-15500 输出 `802` 段、`870` cues、`15203` 字。
- 结论是 full SFT 方向仍然成立：segments 数接近，但 200k / 15500 的目标域文本覆盖明显高于 base；同时 forced aligner fallback 仍高，分别约 base `51.0%`、200k `49.3%`、15500 `50.4%`，说明当前主要瓶颈已经转向 alignment / fallback / QC，而不是 ASR 是否能输出。
- checkpoint-21000 已拉到本地并用 v1.11 long-gap VAD 跑匿名样片 A 闭环；它不是纯 ASR checkpoint 对比，因为 VAD 口径从 v1.5 切到 v1.11。未做长段保护时，该组合切出 `89` 个更长的 VAD chunks，输出 `262` 段、`269` cues，ASR+align `285.6s`；但诊断中 forced 仅 `14/89`（`15.7%`），fallback `38/89`（`42.7%`），failure candidates `76/89`（`85.4%`），主要 bucket 是 `empty_text_for_chunk` `35` 和 `vad_coarse_alignment` `31`。结论：v1.11 提升 real-heldout 召回后，长 chunk / 空输出 / coarse fallback 成为新的下游瓶颈；下一步要在 v1.11 内先修 chunk packing / pre-align / fallback，而不是直接把这组结果解读为 21000 ASR 变差。
- Qwen3-ASR-0.6B full ASR-only SFT 已上传到 `https://huggingface.co/jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`，本地按 `models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame` 缓存；Galgame 16 clip direct probe 显示 CER 从 base `0.2348` 降到 full `0.1288`，RTF 约 `0.232`。基于该模型的 full29239 feature cache 已生成到 `datasets/*/fusionvad-ja/v1-12/qwen3-asr-0.6b-full29239/`，并成为 v1.13+ FusionVAD-JA 默认特征源。Qwen3-ASR-1.7B full SFT 已上传到 `https://huggingface.co/jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`，运行默认 `ASR_MODEL_ID` 指向该 repo。

R15 / R16 工作流修正：

1. 离线分析匿名样片 A 的 VAD cache + alignment diagnostics，不跑 ASR，按时间 overlap 标出多 speech island + 长 gap、连续 speech 无明显 gap、疑似多人/男女交替、ASR empty 和 `vad_coarse_after_sentinel` 的长 chunk 成因。
2. 新增 env-gated pre-ASR speech-island splitter，默认关闭；只处理高风险 chunk，不全局缩短。R15 第一版优先使用明确 internal gap；R16 第一版已接入 per-frame FusionVAD score 的低分 valley 离线分析和 opt-in split 参数。
3. 训练 / 微调 boundary-aware head：继续守住 frame-level recall，同时增加 internal gap penalty、boundary loss 和 exact-island synthetic 评测。目标是“一句或一个 speech island 一条 ASR chunk”，不是单纯提高 frame F1；若 rule-based valley split 覆盖率不足或 chunk 增幅过高，应转向训练 boundary refiner。
4. 重新评估 F0/gender：当前不再依赖 F0 标签做主线切分；speaker sidecar 只在 speech-island 变细后进入 probe，用来避免跨 speaker 合并。
5. 对比三组闭环：v1.13 baseline、v1.13 + rule-based speech-island splitter、v1.13 + trained boundary refiner。指标必须同时看 chunk 数增幅、ASR empty、repeat/hallucination proxy、forced/partial/fallback、最终 SRT 观感和审计页可播放片段。
6. fallback-safe boundary metric 成为 R16/R17 的验收门槛：统计 coarse fallback chunk 数、unsafe fallback 数、fallback_safe_ratio、sentinel fallback 数、fallback duration p90/max、internal gap 和 speech-island 数。v1.13/v1.14 的样片 A 复测已证明“forced 数略变”不足以判断字幕可用性。
7. v1.15 明确改为 endpoint/boundary refiner，而不是“更会判断有没有人声”的单头 VAD：输入沿用 full29239 Qwen3-ASR-0.6B frozen feature + MFCC/energy，输出 `speech`、`start`、`end`、`cut` 四路 logits。`speech` 继续服务 recall，`start/end` 学 speech island 边界，`cut` 学长 gap / 内部非语音的可切点；允许 end 略长，但禁止 fallback chunk 长到 20-30s。

synthetic timeline / boundary refiner 计划：

- 由来：人工复听确认 `litagin/Galgame_Speech_ASR_16kHz` 多数 clip 本身已经裁成语音片段；在 clip 前后拼接随机空白或噪声后，前置 gap 长度就是 speech start，`start + clip_duration` 就是 speech end。多段拼接时每个 speech island 都有可计算真值。
- 适用范围：该数据天然适合训练和评测 frame-level VAD、speech boundary refiner、VAD-constrained alignment fallback，以及 forced aligner 的 start/end 鲁棒性 bench。它只提供片段级边界，不天然提供字/词级时间轴，所以不能直接等价为 Qwen forced aligner finetune 数据。
- 数据形态：`tools/fusionvad_ja/build_galgame_synthetic_timeline.py` 负责生成 16k mono WAV、标准 VAD label JSONL、兼容训练的 `manifest.json`、详细 `synthetic_timeline_details.jsonl`，以及公共 `boundary_manifest.jsonl`。`boundary_manifest.jsonl` 同时保留 `actual_speech_segments` 和带训练 pad 的 `speech_segments`，后续 bench 默认用 actual span 做边界误差真值；`transition_regions` 记录 crossfade 模糊区，`augmentation` 记录背景、overlap、gain、filter 和 codec 处理。
- v5 long-gap 口径：脚本默认使用 `5-30ms` equal-power crossfade、随机 gain `-3~3dB`、轻量 lowpass/bandpass 概率 `0.25`、codec 模拟概率 `0.05`、overlap speech 概率 `0.12`、背景混合概率 `0.5`；speech 最长裁到 `8s`，speech 最短保留到 `0.05s`，middle gap 默认 `1-6s`，首尾 gap 默认 `0.5-4s`，real negative gap 默认概率 `0.75`。无 negative manifest 时仍会退回 silence / white_noise / hum / fade_noise；旧硬拼接行为仅用于显式传 `--crossfade-ms-min 0 --crossfade-ms-max 0 --gain-db-min 0 --gain-db-max 0 --filter-prob 0 --codec-prob 0 --overlap-speech-prob 0 --background-mix-prob 0 --speech-label-pad-s 0` 的单测。
- 当前执行：v5 long-gap split 已生成到 ignored `datasets/*/fusionvad-ja/v1-11/`，train/val/test 分别 `256/64/64` 条，skipped 均为 `0`。speech frame ratio 为 `0.574/0.551/0.568`，总时长 p50 约 `17.04/16.95/17.67s`、p90 约 `22.70/21.95/22.45s`、max 约 `26.98/24.56/24.98s`。train/val/test 的 real negative gap 为 `554/146/149`，background mix 为 `118/34/34`，overlap speech 为 `26/12/7`，filter 为 `69/11/14`，codec 为 `15/7/1`，每个 split 均已写出 `boundary_manifest.jsonl`。
- 当前 benchmark：v1.13 使用 full29239 0.6B frozen feature，并把 synthetic v5 的训练标签改为 exact speech-island。训练混合 v1-mini strong/negative `302` 条 + exact-island synthetic `256` 条，共 `558` 条，checkpoint 位于 `datasets/train/fusionvad-ja/v1-13/qwen3-asr-0.6b-full29239/exact-island-bilstm-fromscratch-v1mini-galgame-synthv5-558-batch16-lr2e-4-steps1024/`。在 exact-island test64 上，threshold `0.20` + pad `0.2s` 为 recall `0.9505`、precision `0.7360`、extra audio ratio `1.2915`；但该点漏语音过多。当前默认取 threshold `0.10` + pad `0.2s`：boundary speech-duration recall `0.9935`、missed speech `1.82s`、extra audio ratio `1.6012`，start/end p50 误差约 `0.628s/2.002s`。对照 full29239 long-gap 旧头 threshold `0.02` 为 recall `0.9958`、missed `1.17s`、extra ratio `1.6225`、start/end p50 `1.202s/2.068s`。结论：v1.13 的主要收益是 start 边界更接近真实 speech island；end 偏长不是同等 blocker，因为最终字幕 cue timing polish 会为 2-frame gap / linger 规则压缩前一条 cue 的 end。下一步真实 held-out / 匿名样片 A downstream 复测重点看 start、漏语音、长 chunk、ASR empty 和 forced-aligner fallback。
- v1.14 boundary-aware 候选：在 v1.13 checkpoint 上 fine-tune `256` steps，新增 opt-in `boundary_loss_weight=0.25` 和 `gap_loss_weight=0.10`，训练仍是同一 full29239 feature + `558` 条混合数据，checkpoint 位于 `datasets/train/fusionvad-ja/v1-14/qwen3-asr-0.6b-full29239/boundary-aware-ft-v1-13-558-batch16-lr5e-5-steps256/`。synthetic exact-island test64 sweep 显示 `threshold=0.005` + pad `0.2s` 为 speech recall `0.9992`、missed `0.49s`、extra audio ratio `1.5401`、start/end p50 `1.152s/1.282s`；`threshold=0.02` 为 recall `0.9913`、missed `5.27s`、extra ratio `1.3325`、start/end p50 `0.741s/0.637s`。真实 v1.6 held-out 只作为参考信号（人工边界不绝对精确）：`threshold=0.005` 为 recall `0.9660`、missed `13.18s`、extra `1.4368`；`threshold=0.003` 为 recall `0.9961`、missed `1.50s`、extra `1.5950`；`threshold<=0.002` 接近 all-positive，边界收益基本消失。匿名样片 A 用 `threshold=0.003` 跑同一 Qwen3-ASR-1.7B full29239 + ForcedAligner downstream：chunks `222`、segments `986`、cues `1061`、ASR+Alignment `1093.3s`、fallback chunks `115`（`51.8%`）、`vad_coarse_after_sentinel=103`、forced `101`。对比 v1.13 的 chunks `227`、segments `967`、cues `1020`、ASR+Alignment `1079.4s`、fallback `114`、`vad_coarse_after_sentinel=104`、forced `106`，v1.14 没有带来 downstream fallback 改善。结论：boundary-aware loss 保留为训练方向，但当前 v1.14 不替换 v1.13 默认。
- 真实 held-out 复测：同一 v1.6 real-heldout `79` 条上，v1.5 posw2 + threshold `0.00015` + pad `0.2s` 为 recall `0.9556`、missed speech `17.24s`、extra audio ratio `1.3765`；v1.11 long-gap + threshold `0.02` + pad `0.2s` 为 recall `0.9809`、missed speech `7.42s`、extra audio ratio `1.5021`；v1.13 full29239 exact-island + threshold `0.10` + pad `0.2s` 为 recall `0.9994`、missed speech `0.22s`、extra audio ratio `1.6255`。结论：v1.13 更符合“宁可多送、不漏人声”的 proposal 目标，但 almost-all-positive 倾向更强，必须依赖 ASR / aligner / 后处理去消化多送音频。
- 匿名样片 A downstream：v1.13 默认全流程已跑同一 Qwen3-ASR-1.7B full checkpoint-29239 和 Qwen3-ForcedAligner。对比 v1.11 framepack baseline：chunks `240→227`、segments `965→967`、cues `1026→1020`、ASR+Alignment `1026.1s→1079.4s`，fallback chunks `137→114`，`vad_coarse_after_sentinel 122→104`，forced `101→106`，nonlexical `4→2`。方向有改善，但不是 Phase 1d staged island repair 的替代品；Phase 1d 仍明显更强（fallback `48`、`vad_coarse_after_sentinel=42`），只是耗时更高（约 `1642s`）。当前结论：v1.13 可作为默认 FusionVAD-JA feature/head，alignment repair 继续作为独立 opt-in 杠杆。
- downstream caveat：v1.11 默认 `merge_gap=0` 仍可能输出少量超长 chunk，导致 Qwen ASR 空输出和 forced aligner sentinel 增多。当前已把 downstream chunk packing 改成“固定帧数 + 任务级真实 FPS”：`window_frames=899`、`reserve_frames=45`、`target_padding_frames=60`、`gap_merge_frames=45`，每个视频用 `1/fps` 换算这些帧对应的秒数，只有 FPS 探测失败才回退 29.97。在匿名样片 A 旧 raw VAD segments 上离线重算，processing spans 从旧 split28 的 `255` 降到 `240`，最长 `28.50s`，平均 `25.05s`，最大左右 padding 均约 `2.002s`，split reason 为 `overlong=216`、`capacity=13`、`gap=11`。后续 v1.11 评估必须同时报告 `transcript_chunks`、chunk duration 分布、ASR empty count、`nonlexical_text`、`drop_or_review`、`vad_coarse` fallback、`fallback_subtype` 和 SRT cues，不能只看 VAD recall。
- R14 Phase 1a GPU 结论：`ASR_CHUNK_PACK_MAX_CORE_FRAMES=419` 在 synthetic64 上跑通 64/64 且全程 CUDA，但**未通过验收**。chunks `137→148`（+8%，可控），forced `77→84`，fallback `60→64`，`vad_coarse_after_sentinel` `25→28`，cue-level `vad_coarse` p90 max-boundary error `4.51s→4.01s` 仍为 `FAIL_PHASE_1_2`。这说明“只在长 gap 处拆超长 chunk”只能改善一部分粗时间轴误差，不能解决 sentinel。
- R14 Phase 1b 结论：`nonlexical` / `align_text_empty` 已显式分流，纯省略号/符号保留 display_text 并走粗时间轴，但不再计入真正 `vad_coarse` fallback。清洗后 synthetic64 为 `forced=84`、`nonlexical=32`、true fallback `vad_coarse=28`、`drop_or_review=4`；gate 仍 `FAIL_PHASE_1_2`，剩余瓶颈集中在 `vad_coarse_after_sentinel` 非空文本块。
- R14 Phase 1c GPU 结论：`ALIGNMENT_SENTINEL_ISLAND_SPLIT=1` 只对 `vad_coarse_after_sentinel` 的非空文本 chunk 做 aligner-local speech-island splitting，方向有效但初版偏慢。synthetic64 中 chunks `148`、segments `180→182`，forced `84→85`，fallback `28→11`，`vad_coarse_after_sentinel=11`，gate 为 `PASS_RECLASSIFICATION_CLEANUP`；匿名样片 A 中 chunks `240→267`、segments `965→1061`、forced `101→154`、fallback `137→48`、`vad_coarse_after_sentinel 122→42`，审计对比页已生成到 `agents/audits/fusionvad-ja/alignment-compare-sample-a-qwen29239-island-split/index.html`，稳定入口 `agents/audits/fusionvad-ja/latest-audit.html` 指向该页。初版运行时间约 `1053.5s→3150.4s`，因为每个失败 chunk 串行卸载/重载 ASR 与 aligner；正式默认仍关闭。
- R14 Phase 1d GPU 结论：staged batch island retry 已把 Phase 1c 的 per-chunk 重载改成“收集全部 sentinel chunk -> 一次性物化 island clips -> 批量 ASR -> 卸载 -> 批量 forced align -> merge 回原 chunk”。synthetic64 64/64 跑通且指标与 Phase 1c 对齐：chunks `148`、segments `182`、forced `85`、fallback `11`、`vad_coarse_after_sentinel=11`，gate `PASS_RECLASSIFICATION_CLEANUP`。匿名样片 A 长片复测也保持质量不回退：chunks `267`、segments `1061`、forced `154`、fallback `48`、`vad_coarse_after_sentinel=42`，ASR+Alignment `1613.1s`、总计 `1641.7s`。这比旧 per-chunk Phase 1c 的 `3150.4s` 明显快，同时仍比 baseline `1053.5s` 慢；结论是 staged batch 适合作为 opt-in alignment repair，不宜默认开启。
- Qwen 后端日志噪音已定位：`temperature` warning 来自 Transformers 在 greedy / deterministic generation 下忽略 sampling-only 参数；`pad_token_id` warning 来自 Qwen ASR 底层 `thinker.generate()` 未显式配置 pad token，Transformers 每次回退到 `eos_token_id=151645` 并打印。修复方向不是压日志，而是在加载 Qwen ASR / ForcedAligner 后归一化底层 `generation_config`：清除非默认 `temperature`，并在缺失时设置 `pad_token_id=eos_token_id`；不改变 greedy 解码语义。
- 模型路线：先继续当前 `Qwen3-ASR-0.6B full SFT frozen feature + MFCC addition BiLSTM` 高召回线。2026-05-31 Grok 检索结论是当前复现方向仍和近期工作一致：FusionVAD 2025 的 addition fusion 仍是最贴近本项目的轻量路线；Interspeech 2024 transformer VAD 证明 wav2vec2/XLS-R 等预训练表征适合 VAD；但这些并不直接解决目标域 long gap / 多 island 边界，关键仍是 synthetic exact-island 数据与 downstream fallback 闭环。并行 frozen SSL baseline 仍列为二期候选，优先 `reazon-research/japanese-hubert-base-k2`，其次 `rinna/japanese-hubert-base` / `rinna/japanese-wav2vec2-base`。
- 评测顺序：先用 synthetic timeline 测 start/end 误差、recall、extra audio ratio 和 inference cost；再回到 v1.6 真实 held-out 与匿名样片 A 同口径测 downstream ASR / alignment fallback。只有 synthetic 和真实 held-out 都有收益，才考虑替换 FusionVAD-JA feature extractor 或训练 boundary refiner。
- Forced aligner 路线：Qwen3-ForcedAligner-0.6B 仍是主线。官方模型卡确认其支持日语、最长约 5 分钟、词/字级 timestamp，并与 Qwen3-ASR 配套；但目前没有找到公开 forced-aligner finetune recipe。MFA Japanese 更适合规范文本和词典化发音，不作为当前主线。

v1.9 ASR / forced alignment 文本策略：

- `display_text` 是最终字幕显示文本，只做展示安全处理：Unicode NFKC、换行归一为空格、连续空白折叠和首尾 trim。不得在 `display_text` 上压缩重复假名、重复短语、拟声或低信息短文本，因为这些在目标域里可能是字幕语义。
- `align_text` 是 forced aligner 专用文本，可以删除标点、emoji / 装饰符、音乐符号和明显不可发音标记，也可以压缩极端重复假名、长音符和重复短语；这些操作必须记录 flags，并保留从 `align_text` 字符到 `display_text` 覆盖范围的映射。
- 不使用按具体字样维护的黑名单，不直接删除 `ん`、`あ`、喘息/呻吟拟声、常见台词、历史工具签名或纯英文长词。ASR 后处理已删除噪声词表、灰区词表、假名/呻吟特例 direct drop、工具签名 direct drop、AnimeWhisper 后置括号/重复清洗、最终字幕文本重复压缩和翻译前纯英文幻觉 direct drop；当前只因空文本、纯标点/纯符号和上下文泄漏这类明确非字幕内容而删除。
- 翻译 prompt 的源文序列化同样不再压缩重复发声，也不使用固定拟声词映射表；重复循环只作为 QC / 诊断信号，译文是否概括交给 LLM 在上下文中判断。
- speaker sidecar / diarization 不再把假名-only 文本当作 BGM 跳过；只跳过空文本或纯符号/纯标点这类没有语言/数字信号的片段，避免把目标域可字幕化人声排除在 speaker embedding 之外。当前它只作为 speech-island 之后的辅助聚类信号，不替代 VAD。
- 重复循环、低置信、文本/音频比例异常、align-text-empty、forced-aligner fallback、ASR dropped uncertain 和人工 hard-negative 结果默认只作为 QC / 诊断 / 样本池信号；`ASR_QC_DROP_UNCERTAIN=0` 是默认值，是否删除交给后续可解释 QC 策略，不再用词表兜底。
- forced aligner 失败时不伪造精确时间轴。诊断导出已使用 `forced`、`partial`、`nonlexical`、`vad_coarse`、`proportional`、`drop_or_review` 六类质量标签，并单独记录 `fallback_type=none|vad_coarse|proportional|unknown` 与更细的 `fallback_subtype`；subtype 用于区分 `asr_empty_text`、`align_text_empty`、`nonlexical_text`、`text_without_output_segment`、`vad_coarse_after_sentinel`、`proportional_after_align_error`、`word_timing_low_coverage` 等原因。失败样本进入 VAD / ASR / aligner 后处理样本池。
- 失败样本池闭环分三步：`diagnose_asr_alignment.py` 生成 `failure_candidates.jsonl`，`export_alignment_failure_manifest.py` 转成人工审计 manifest，`materialize_alignment_failure_audio.py` 再按 `source_audio_path + start/end` 切出 WAV 片段，避免依赖中间 chunk 文件路径。
- 实现口径：`src/whisper/prealign.py` 负责 `raw_text -> display_text -> align_text` 和 char-span mapping；`src/whisper/local_backend.py` 只把 `align_text` 送入 forced aligner，拿到词级时间后再映射回 `display_text`。

下一步：

1. 保持 v1.16 endpoint refiner 作为默认 FusionVAD-JA feature/head；不恢复 base 0.6B 默认。
2. R16/R17 下一步继续从 rule-based valley / 全局 max-core 切短，转向 endpoint / boundary refiner：训练一个能把长连续 positive island 切成自然 speech island 的边界层；验收看 fallback_safe_ratio、unsafe fallback 数、fallback duration p90/max、synthetic truth start/end p50/p90、sample A fallback 是否跨大段无声、ASR empty 和人工字幕观感。
3. R14 Phase 1d staged batch island retry 仍作为 opt-in repair 和质量上限参考；只有 pre-ASR splitter 无法覆盖时才启用。
4. 用 v1.16 输出继续扩展失败样本池：优先收集 `vad_coarse_after_sentinel`、ASR empty、long low-information、repeat/hallucination proxy 和人工 hard-negative。
5. 用 synthetic timeline v5 `boundary_manifest.jsonl` 继续作为 VAD / boundary refiner / frozen SSL baseline / forced-aligner bench 的共同输入。

参考来源：Whisper hallucination on non-speech `https://arxiv.org/abs/2501.11378`，Calm-Whisper `https://www.isca-archive.org/interspeech_2025/wang25b_interspeech.pdf`，Dynamic Speech Endpoint Detection `https://arxiv.org/abs/2210.14252`，Semantic VAD `https://arxiv.org/abs/2305.12450`，WhisperX `https://arxiv.org/html/2303.00747v2` / `https://github.com/m-bain/whisperX`，stable-ts `https://github.com/jianfch/stable-ts`，FusionVAD arXiv `https://arxiv.org/abs/2506.01365`，Interspeech 2024 transformer VAD `https://www.isca-archive.org/interspeech_2024/karan24_interspeech.pdf`，Qwen3-ASR `https://github.com/QwenLM/Qwen3-ASR`，Qwen3-ASR finetuning `https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning`，Qwen3-ASR-0.6B base `https://huggingface.co/Qwen/Qwen3-ASR-0.6B`，Qwen3-ASR-1.7B base `https://huggingface.co/Qwen/Qwen3-ASR-1.7B`，本项目 0.6B full SFT `https://huggingface.co/jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`，本项目 1.7B full SFT `https://huggingface.co/jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`，Qwen3-ForcedAligner-0.6B `https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B`，pyannote speaker diarization `https://huggingface.co/pyannote/speaker-diarization-3.1`，3D-Speaker `https://github.com/modelscope/3D-Speaker`，WeSpeaker `https://github.com/wenet-e2e/wespeaker`，Reazon Japanese HuBERT `https://huggingface.co/reazon-research/japanese-hubert-base-k2`，rinna Japanese HuBERT `https://huggingface.co/rinna/japanese-hubert-base`，rinna Japanese wav2vec2 `https://huggingface.co/rinna/japanese-wav2vec2-base`，Hugging Face VAD model index `https://huggingface.co/models?pipeline_tag=voice-activity-detection`。

### 字幕时间轴

LLM 翻译前必须先生成稳定 cue plan。流程会通过 `ffprobe` 读取真实 `avg_frame_rate` / `r_frame_rate`，失败时按 `30000/1001`，即 29.97fps 兜底。

cue plan 负责：

- 基于 forced alignment 词级时间轴排序。
- 合并双语短句。
- 软拆长字幕。
- 裁剪或合并 overlap。
- 固定保留 2 帧字幕 gap。
- 在最终 cue 层做 timing polish：短尴尬 gap 收敛到 2 帧，真实停顿保留，字幕可在不撞下一 cue 的前提下 linger。

默认字幕约束：

```env
SUBTITLE_SOFT_MAX_S=5.5
MAX_SUBTITLE_DURATION=6.5
ASR_MERGE_HARD_MAX_DURATION=9.0
SUBTITLE_TIMING_POLISH_ENABLED=1
SUBTITLE_SHORT_GAP_COLLAPSE_S=0.5
SUBTITLE_LINGER_S=0.45
```

字幕 timing polish 只作用于最终 cue plan，不改变 VAD、ASR 或 forced aligner 原始输出。当前参考 Netflix timed-text 常见原则采用真实视频 FPS：最小 gap 为 2 帧；若相邻 cue 间隔短于 `SUBTITLE_SHORT_GAP_COLLAPSE_S`，则把上一 cue 延到 `next_start - 2 frames`，避免 0.1-0.4s 的突兀闪断；若间隔不短于该阈值，则最多延长 `SUBTITLE_LINGER_S`，但保留至少 `SUBTITLE_SHORT_GAP_COLLAPSE_S` 的真实停顿。单条字幕同样可 linger，但仍受 `MAX_SUBTITLE_DURATION` 约束。关闭 `SUBTITLE_TIMING_POLISH_ENABLED=0` 可回到旧的 alignment end 行为。

相邻短块合并按帧数判断，而不是硬编码秒数。普通短块合并默认允许 `gap <= 6 frames` 且合并后 `duration <= 120 frames`。历史 F0 gender guard 只保留为 legacy/ablation，不再作为当前主线的字幕切分依据；speaker guard 仍然是硬边界，但应建立在 speech-island 足够细的前提下。

最终写入 SRT、`bilingual.json` 和 quality report 的都是同一份已归一化 cue。

### 翻译策略

当前翻译 prompt version：`v2.7`。

LLM 只负责逐 cue 翻译、遵守术语表和人名罗马音规则。全片上下文只用于翻译连贯、指代判断、口吻一致和术语一致，不授权根据上下文修正 ASR 误听、同音词、上下文漂移、术语漂移或被切断半句。

LLM 输入默认不再依赖 `[M]` / `[F]` 声学标签。早期 F0/gender 标签曾用于辅助口吻和对话切换，但在大 chunk、多 speech island 或男女交替场景下噪声较大；后续若需要 speaker 信息，应优先来自 speech-island 之后的 speaker sidecar，并且最终 SRT 不输出可见 speaker/gender 标签，除非用户显式开启。

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
- `F0_GENDER_*`（legacy/ablation，当前主线不依赖）

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

常见坑 / 执行权限：

- Codex 默认 sandbox 可能隔离 GPU 或驱动能力。需要跑全片 VAD/ASR/ForcedAligner、ONNXRuntime CUDA、Torch CUDA、feature cache、训练、`nvidia-smi` 或任何“必须确认 GPU 占用”的命令时，应提权执行；否则可能看不到 CUDA、回落 CPU，或者日志里出现 `CUDA failure 35` / `CPUExecutionProvider`。验收时必须看实际日志：Torch 路径看 `actual_device=cuda` / `model_param_device=cuda:*`，ONNX 路径看 `CUDAExecutionProvider`，必要时配合 `nvidia-smi`。
- 联网默认受限。Hugging Face / ModelScope 下载、`uv pip` / `npm install` / `curl` / `git fetch` / Grok 或外部 API 探测等网络操作，若出现 DNS、连接、证书、超时或 403/429 之外的网络层错误，优先按“需要提权或代理环境”处理，不要把它误判成代码逻辑失败。
- 长跑命令不要在一次性 exec 里直接用 `cmd > log 2>&1 &` 后立即退出 shell。当前工具的 shell 退出后，后台子进程可能被收走，表现为空日志、无产物、无 Python traceback。本地全片 workflow / 训练 / 大规模评测要么让 exec session 前台持有进程并把 stdout/stderr 重定向到日志，要么用脚本在同一个 shell 内启动后台进程、循环 `tail` 日志并 `wait` 子进程。日志统一放 `agents/temp/.../*.run.log`。
- 发现 CUDA 任务异常慢、GPU 占用为 0、日志没有 provider/device 信息、或后台长跑日志一直为空时，先停下确认执行权限和进程托管方式，再重跑；不要让 CPU fallback 跑完整片。

构建 Windows Release：

```powershell
.\packaging\build_windows.ps1 -Clean
```

构建产物位于 `dist\JAVTrans\JAVTrans.exe`。打包细节见 `packaging/README.md`。由于包内包含 PyTorch/CUDA 运行库、默认 ASR 模型和默认流程辅助模型，发布目录会达到数 GB；上传 GitHub Release 时通常需要分卷压缩或改用外部大文件分发。

本项目引入的部分第三方代码，例如 `src/vad/whisperseg`，保留其原始许可证，请遵循相应协议。

---

## 当前 Backlog

- **R15/R16 主线：Pre-ASR speech-island / boundary-aware chunking**。用当前 FusionVAD-JA frame mask / raw segments 做离线分析和 env-gated splitter，只切高风险长 chunk / 多 island chunk。R16 rule-based probability valley 已证明能覆盖 sentinel 风险但 chunk 增幅过高，下一步优先训练 boundary-aware head / refiner，同时守住 recall、internal gap、start/end 误差、ASR empty、hallucination proxy 和 forced-aligner fallback。
- 从 Hugging Face / ModelScope 单独下载并模块化评测 CAM++ / 3D-Speaker / WeSpeaker speaker embedding/聚类能力，确认是否可作为 speech-island 之后的 speaker sidecar；不作为默认 VAD，也不作为 speech/non-speech hard gate。
- 评测 `efwkjn/cohere-asr-ja-v0.1`，确认其与当前 ASR 流程及 `transformers` 版本约束的兼容性，再决定是否纳入候选后端。
- 增加本地/厂商翻译 API 适配层，允许在现有 OpenAI-compatible 翻译之外接入专用翻译服务，例如腾讯 `hy-mt2`。
- **[→R14]** forced-alignment fallback 层专项已完成阶段性收口：全局/半全局长 gap 切短未通过 GPU gate；sentinel-only speech-island repair 有效，staged batch 版可作为 opt-in repair，但耗时仍高于 baseline。当前主线前移到 R15/R16 pre-ASR speech-island packing；R14 细节见任务历史折叠记录。
- ASR chunk packing 固定帧数参数暂定为 `899/45/60/45`，每个新视频任务必须重新读取真实 FPS 后换算帧时长；60fps 等高帧率视频会自然得到更短秒级窗口。该策略源自字幕 cue plan 的按帧 gap 口径和 Netflix timed text 的 2-frame gap 思路，但 ASR packing 服务的是 30s 模型窗口保护，不直接复用字幕最终 2-frame gap。
- 增加 frozen SSL boundary baseline（R14 Phase 2 候选）：优先 `reazon-research/japanese-hubert-base-k2`，其次 `rinna/japanese-hubert-base` / `rinna/japanese-wav2vec2-base`，和当前 Qwen3-ASR-0.6B frozen feature 线比较；该模型若接入 CTC aligner 则与 VAD frozen feature 复用同一份下载。
- F0/gender 切分列入 legacy cleanup：当前不再作为主线切分、cue 合并或翻译标签来源；先保留代码和回归以免破坏历史路径，后续在 speech-island/speaker sidecar 稳定后再决定是否删除环境项、UI 文案和测试。
- 轻量 VAD 路线列入二期 backlog：FunASR FSMN / Silero / TEN 只作为 teacher、baseline、hard-negative miner 或最终蒸馏目标，不在当前阶段替换 Qwen3-ASR-0.6B frozen feature。若未来蒸馏，验收指标必须同时守住 recall、missed speech seconds、extra audio ratio、downstream ASR 空输出和 forced-aligner fallback。
- 扩展失败样本池：当前 ASR/alignment 诊断已按 `failure_bucket` 导出候选，可用 `tools/fusionvad_ja/export_alignment_failure_manifest.py` 转成人工审计 manifest，再用 `tools/fusionvad_ja/materialize_alignment_failure_audio.py` 切出审计 WAV；审计页和稳定入口统一写入 ignored `agents/audits/fusionvad-ja/`，下一步把人工确认 hard-negative、ASR 空输出和 held-out 复测失败样本合并成可训练/可审计数据包。
- 等 Qwen3-ASR-1.7B / 0.6B full SFT checkpoint 稳定后，用同一批 held-out 复测漏对白、多送音频、空输出、hallucination、低置信和 forced-aligner fallback。
- 二期 probe `joujiboi/Galgame-VisualNovel-Reupload` 的 streaming parquet 字段、样本质量、去重、下载速度和 license 边界；只作为 Qwen3-ASR / FusionVAD-JA 候选数据源，不进入第一轮默认数据混合。

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
| R11 | Fusion-lite 后缀实验 + 匿名样片 A 对比 + 帧率驱动 SRT overlap 归一化 | 匿名样片 A 四模式对比完成；`fusion_lite_boost` 在当时历史四模式里最接近 whisperseg-adaptive；当前已移除，不再作为公开模式；新增逐句 HTML 报告；字幕/fps/主流程定向 73 passed |
| R12 | 匿名样片 C 四模式双语对比 + frame-based 短尾 cue 合并 | 匿名样片 C 四模式全流程双语输出完成；新增 frame-based overlapping tail merge，修复 `受け` / `受けて` 这类极短 gap 被 F0 gender 抖动切成两条的问题；字幕定向 50 passed |
| R13 | 全量代码审计修复：Web 设置 `.env` 安全写入、本地打开接口绑定 job 授权、前端动态渲染转义、完成态 artifact 过滤已清理临时文件、pytest 配置清理 | Web 定向 25 passed；全量 440 passed, 5 skipped；前端 JS `node --check` 通过 |
| R14 | **forced-alignment fallback 层专项（measure-first）**：全量代码/训练审计确认瓶颈已转移到 alignment fallback（`vad_coarse_alignment` ~55%，更好 ASR 无效）。**Phase 0**：`tools/fusionvad_ja/measure_fallback_timing_error.py` 在 synthetic timeline 64 条 test clip 上量化 forced/vad_coarse/proportional 的 per-cue start/end 误差，确认 vad_coarse p90 比 forced 差 >250ms，进入 Phase 1/2。**Phase 1a**：`ASR_CHUNK_PACK_MAX_CORE_FRAMES=419` 按长 gap 定向切短，chunks `137→148`（+8%）但 sentinel 不降，GPU gate 未过。**Phase 1b**：显式区分 `nonlexical` / `align_text_empty`，不可强对齐的纯符号/省略号保留 display_text 并走粗时间轴，但不再污染真正 `vad_coarse` fallback；`vad_coarse_after_sentinel` 仍为 28 个，gate 仍 `FAIL_PHASE_1_2`。**Phase 1c**：已实现 `ALIGNMENT_SENTINEL_ISLAND_SPLIT=1` 的 opt-in 局部修复，只对 sentinel 非空文本 chunk 按 speech islands 局部重跑 ASR+align；GPU 闭环显示 synthetic64 `vad_coarse_after_sentinel=11`，匿名样片 A `122→42`。**Phase 1d**：已把初版 per-chunk 重载改为 staged batch island retry，继续保持 ASR/aligner 分阶段加载；synthetic64 与匿名样片 A 指标均与 Phase 1c 对齐，长片总时长约 `1642s`，比旧 per-chunk `3150s` 明显快但仍慢于 baseline，暂作为 opt-in repair。**前端快速修复（已完成）**：worker 上限 `max` 改 `8`，SSE 健康时停轮询，文件路径入队去重逻辑抽取；JS `node --check` 通过。 | Phase 0/1a/1b/1d 产物分别在 `agents/temp/fusionvad-ja/fallback-timing-error*`、`diagnostics-fallback-timing-error-*`；Phase 1c 审计页在 `agents/audits/fusionvad-ja/alignment-compare-sample-a-qwen29239-island-split/`；当前定向回归 alignment 16 passed，subtitle 38 passed |
| R15/R16/R17 | Pre-ASR speech-island / boundary-aware chunking 路线修正：样片 A 双语字幕主观效果已可用，但仍有非语音/gap 诱发的语气词或短句幻觉；当前结论是 VAD 不应直接输出大块给 ASR，而应先输出高召回 frame mask，再由 speech-island / endpoint packer 切成更接近一句台词的 ASR chunk。F0/gender 标签从当前主线降级为 legacy/ablation；CAM++/3D-Speaker/WeSpeaker 仅作为 speech-island 后的 speaker sidecar。 | R15 gap-based splitter + 离线成因分析已实现；R16 frame-score 导出、valley split opt-in 参数和离线分析工具已实现。样片 A CUDA frame-score 导出成功，但 rule-based valley split 即使保守配置仍约 `227→444` chunks（`1.956x`），暂不进 GPU 闭环。新增 boundary/gap loss 并训练 v1.14 256-step 候选；synthetic test64 th0.005 有正向信号，但真实 held-out / 样片 A downstream gate 未过。新增 fallback-safe boundary metric 后，v1.13/v1.14 在样片 A 上 `fallback_safe_ratio=0.0`。R17/v1.15 已新增 endpoint refiner 多头训练入口，下一步用 synthetic exact-island 训练正式候选并跑 downstream gate |

### 关键验证记录

<details>
<summary>展开 V01-V08 早期验证记录</summary>

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
- 当前保留 VAD 路线：默认 `fusion_lite`，以及对照 `whisperseg-adaptive`。
- `fusion_lite` 使用可解释公式融合 whisperseg 分数、Silero 重叠、RMS、spectral flux 和时长分数。
- 匿名样片 B 前 5 分钟：whisperseg 48 VAD segments / 129.22s speech / 14 字幕 / 11 drops；fusion_lite 23 VAD segments / 89.34s speech / 15 字幕 / 7 drops；generation overflow/error/timeout/quarantine 均为 0。
- 匿名样片 C / D 全片三模式历史对比显示 fusion_lite 输出接近 whisperseg，但 drops 少于 whisperseg。
- 逐句报告：`video/<video-a>/<video-a>_<video-b>.full_vad_modes_line_compare.html`，同一报告也归档到 `video/<video-b>/`。

#### V05 · fusion-lite 后缀实验和帧率驱动 SRT 归一化（R11）

- 历史新增 `fusion_lite_boost` 后缀后端，不新增 `FUSION_VAD_SCORING_MODE`；当前这些后缀后端均已移除。
- 匿名样片 A 使用 `whisper-ja-anime-v0.3`、跳过翻译进行 VAD/ASR 对比。
- CUDA ONNX 在 sandbox 内失败，原因是 GPU 被操作系统/sandbox 阻断；外部执行确认 RTX 4060 Ti、Torch CUDA 和 ONNXRuntime CUDA provider 可用。
- 匿名样片 A 历史汇总曾用于后缀实验取舍；当前公开后端已收敛为 `whisperseg-adaptive` 与 `fusion_lite`。
- SRT overlap 处理重构：新增 `probe_video_fps()`，SRT writer 在写出前排序、软拆、合并/裁剪重叠，并强制保留 2 帧 gap。
- quality report 新增 subtitle overlap 统计。
- 验证：字幕/fps/主流程定向 73 passed。

#### V06 · 四模式双语对比和短尾合并（R12）

- 匿名样片 C 历史四模式完整双语对比已输出到 `video/<video-stem>/<video-stem>.*.srt`，逐句 HTML 报告为 `video/<video-stem>/<video-stem>.vad_bilingual_compare.html`；当前公开模式已收敛为三模式。
- 四模式 ASR generation error / overflow / timeout 均为 0，最终 SRT 未包含可见 `[M]` / `[F]` 性别标签。
- 匿名样片 C 历史汇总曾用于后缀实验取舍；当前公开后端已收敛为 `whisperseg-adaptive` 与 `fusion_lite`。
- cue plan 短尾合并改为 frame-based 规则。
- 匿名样片 C 离线验证：`00:02:56,839 --> 00:02:59,160` 合并为 `アルマリスト 室で イラックス ステイマンを受けて`。
- 验证：字幕定向 50 passed。

#### V07 · 全量代码审计和 Web 安全收口（R13）

- `/api/settings` 写 `.env` 时对值做 dotenv-safe quoting，并校验 key 名，避免换行值注入额外配置项。
- `/api/open-video` 与 `/api/open-folder` 改为必须携带 `job_id`，只允许打开该 job 的视频路径或已登记 artifact，避免任意本地路径被页面请求触发打开。
- Web 任务列表、文件 chip、模型选项等动态文本统一转义或使用 `textContent`，并对进度百分比和下载 URL 做边界处理。
- Web 完成态 artifact 列表只保留真实存在的文件，避免 `keep_temp_files=false` 清理后继续暴露已删除的临时 JSON。
- 验证：`tests/web/test_jobs_api.py tests/web/test_cancel_resume.py` 25 passed；全量 pytest 440 passed, 5 skipped；Torch CUDA NVML 初始化 warning 仍为环境 warning。

#### V08 · 前端快速修复（R14 快速修复部分，已完成）

审计发现的三处前端独立问题已直接修复，不依赖 Phase 0/1/2 进展：

- `src/web/static/index.html` `t-translation-max-workers` `max` 从 `32` 改为 `8`，消除用户提交 9–32 被后端 422 拒的静默失败。
- `src/web/static/js/jobsApi.js::startJobPolling` 增加 SSE 健康判断（`state.sse?.readyState === 1`），SSE 正常时跳过 3s 轮询，仅在 SSE 断连时作为 fallback，消除高频冗余请求。
- `src/web/static/js/files.js` 4 处重复的路径入队逻辑抽为 `addPathsToState(paths)`（`pickFiles`/`pickFolder`/`__pywebviewDrop`/`drop` handler），行为等价纯重构。
- 验证：全部 14 个前端 JS 文件 `node --check` 通过。

</details>

<details>
<summary>展开 R14 forced-alignment fallback 详细验证记录（V09-V12）</summary>

#### V09 · R14 Phase 0 量化 + 根因定位（gate=进 Phase 1/2）

GPU 真实流水线在 synthetic v5 long-gap test64 上跑通 64/64（FusionVAD-JA `actual_device=cuda`，失败 0），用新工具 `tools/fusionvad_ja/measure_fallback_timing_error.py` 量化 forced vs vad_coarse 的 per-cue 边界误差，结论：

- **gate = 进 Phase 1/2**：vad_coarse 比 forced p90 max-boundary 差 `~2158ms`，远超 250ms 门槛。fallback 确实显著伤时间轴，方向成立。
- **度量口径 caveat（重要）**：forced 自身 p90 也有 `~2347ms`，但**不是 forced 对齐质量差**。深挖确认：cue 覆盖≥80% island 子集 p90=2347ms ≈ 切碎子集 2505ms（切碎不是主因），真正来源是 synthetic v5 真值边界本身有 ~2s 级模糊（crossfade + speech-label-pad + transition_regions）叠加 VAD 0.2s pad，与 boundary 评测 start/end p90 `2.45s/1.88s` 量级吻合。**故成功判据改为「vad_coarse 向 forced 收敛」，不追绝对 ms**；真实样片 A 旁证（无 synthetic 边界模糊）作为更干净复核。
- **根因链锁定**：vad_coarse 的 `fallback_subtype` 全部为 `vad_coarse_after_sentinel`（51/51），即 ASR 有文本、forced aligner 吐 sentinel（对齐失败）→ 降级。vad_coarse 的 pred cue 时长 p50 `5.5s`/p90 `11.1s`/max `19.4s`，远超 8s 真值 island；forced 仅 p50 `4.1s`/max `9.8s`。即 **high-recall VAD 把多 speech island + 中间长 gap 打包成 11–19s 超长 chunk → Qwen aligner(NAR) 在长 chunk + 大段非语音上 sentinel**。
- **杠杆重排（数据驱动）**：(1) `max_new_tokens` **不适用**——Qwen3-ForcedAligner 是 NAR（`logits.argmax`，无 `generate`/`max_new_tokens`），ASR 文本也未被截；该官方杠杆只对 ASR transcribe 有效且已设。(2) **首选：chunk packing 切短**——收紧 max-chunk 时长 / 按长 gap 强制切分，别把多 island 合进 19s 超长 chunk，直击 sentinel；对应原 Backlog "v1.11 内部消融扫 merge_gap/max chunk"。(3) **CTC aligner 备援**（Phase 2，对长 chunk/噪声更鲁棒）。(4) chunk overlap 降级为次要（边界词截断非主因）。
- 验证：`tests/test_fallback_timing_error.py tests/test_asr_alignment_diagnostics.py tests/test_alignment_quality.py` 9 passed。产物：`agents/temp/fusionvad-ja/fallback-timing-error/summary.md`、GPU 流水线 `agents/temp/fusionvad-ja/fallback-timing-error-phase0-qwen29239-synth64-gpu/`。

#### V10 · R14 Phase 1a chunk packing 长 gap 定向切分（未通过 GPU 闭环）

根因（V09）= high-recall VAD 把多 island + 长 gap 合并成超长 chunk → aligner sentinel。按"最小幅度、只拆跨 gap 超长 chunk"策略实现：

- 新增 env-gated `ASR_CHUNK_PACK_MAX_CORE_FRAMES`（默认 `0`=OFF，不改正式默认）：core 超软上限时在下一个 island gap 处切，`split_reason="soft_cap"`；单 island / 单长 VAD 段（无内部 gap）不动，由现有 overlong 兜底。改动文件：`src/audio/chunk_packer.py`（`FramePackingConfig.max_core_frames` + `max_core_s` + 主循环）、`src/whisper/pipeline.py`（cfg 读取 + 调用）、`src/core/config.py`（默认 0）、`src/whisper/vad_chunk_cache.py`（纳入 cache signature）。
- **离线重算**（64 case 的 vad-cache 原始 VAD 段，frame_hop=1/29.97）：>14s 超长 chunk 中 21/23 是多段累积（可切，最大内部 gap 0.84–1.5s），仅 2 个单长段。`cap=14s`（419 帧）把 core>14 chunk 从 `13→3`（剩 3 为无 gap 单长段），chunk 数 `137→148`（**+8%**，远低于 +20–30% 上限），core_p90 `13.94→10.83s`，soft_cap 切 11 点；15s/16s 增幅更小（+7%/+4%）。实验起点定 **419 帧**。
- 单测：`tests/test_chunk_packer.py` +2（soft_cap 在 gap 切 / 单段不切 / `max_core=0` 回归）。回归：chunk_packer + pipeline_chunk_packing + vad_chunk_cache + asr_stage_env_scope + aligned_segments_cache 共 30 passed。
- **GPU 闭环结论**：开 `ASR_CHUNK_PACK_MAX_CORE_FRAMES=419` 重跑 synthetic64，64/64 完成且 FusionVAD-JA `actual_device=cuda`。chunks `137→148`（+8%）、forced `77→84`，但 fallback `60→64`、`vad_coarse_after_sentinel` `25→28`，cue-level `vad_coarse` p90 max-boundary error 仅 `4.51s→4.01s`，gate 仍 `FAIL_PHASE_1_2`。结论：最小幅度长 gap 切分不足以解决 sentinel；sample A GPU 旁证暂缓，避免把失败 lever 带到真实样片长跑。
- 新增诊断对比：`tools/fusionvad_ja/compare_alignment_diagnostics.py` 现在输出 `fallback_subtype_counts` 和相对第一组的 delta；Phase 1a 对比产物在 `agents/temp/fusionvad-ja/diagnostics-fallback-timing-error-phase1a-compare/`。

#### V11 · R14 Phase 1b subtype 分流与非词 fallback 显式化

Phase 1a 后不继续盲目加大 chunk packing 幅度，先按 subtype 拆解失败：

- 新增 `tools/fusionvad_ja/analyze_alignment_failure_subtypes.py`：输入 `diagnostics.jsonl`，输出 subtype 路线、时长/文本长度分布、典型样本和建议动作。Phase 1a 分析显示 `nonlexical_text=32`、`vad_coarse_after_sentinel=28`、`align_text_empty=4`、`long_low_information_text=1`。其中 nonlexical 多数是纯省略号/符号，应该走显式非词时间策略；sentinel 才是 aligner 鲁棒性问题，p50/p90/max 时长约 `7.34/11.42/12.90s`。
- `src/whisper/local_backend.py` 的 Qwen finalize 阶段增加显式分流：`align_text` 为空且 display compact 长度为 0 时，跳过 forced aligner，记录 `Alignment 策略: nonlexical_fallback`，使用粗时间轴但 `alignment_mode="nonlexical"`；`align_text` 为空但 display 仍有语言/数字信号时，记录 `align_text_empty_fallback` 并用 `alignment_mode="align_text_empty"`。这不会删除 ASR 文本，只避免把不可对齐文本污染为 forced aligner 失败。
- 统计口径同步收口：`src/whisper/alignment_quality.py` 把 `nonlexical` / `align_text_empty` 纳入正常可诊断模式，显式模式即使旧日志里含 `VAD 回退语音区间` 也不再计为 `fallback_type=vad_coarse`；`tools/fusionvad_ja/diagnose_asr_alignment.py` 的 reason 层同样不再把这两类显式模式计入 `alignment_fallback`。`local_backend.py` 后续日志改为 `Alignment 非词粗时间轴语音区间` / `Alignment align_text 为空粗时间轴语音区间`，避免新产物继续污染 fallback ratio。
- **GPU 闭环**：沿用 Phase 1a 的 `ASR_CHUNK_PACK_MAX_CORE_FRAMES=419`，synthetic64 全部完成。清洗后诊断为 chunks `148`、output segments `180`、`forced=84`、`nonlexical=32`、真正 fallback `vad_coarse=28`（`18.9%`）、`drop_or_review=4`；对比第一组，true fallback `60→28` 是统计口径变干净，不代表 sentinel 消失。`vad_coarse_after_sentinel` 仍 `25→28`，说明真问题未解。
- **Timing gate 仍失败**：Phase 1b cue-level forced p90 max-boundary `2338.7ms`，`vad_coarse` p90 `4006.6ms`，delta `1667.9ms`，仍 `FAIL_PHASE_1_2`。nonlexical 显式化解决“诊断混淆”，不解决 `Qwen3-ForcedAligner` 对非空文本长块/噪声块 sentinel。
- 产物：GPU 流水线 `agents/temp/fusionvad-ja/fallback-timing-error-phase1b-nonlexical-maxcore419-qwen29239-synth64-gpu/`；计时 `agents/temp/fusionvad-ja/fallback-timing-error-phase1b-nonlexical-maxcore419/`；诊断 `agents/temp/fusionvad-ja/diagnostics-fallback-timing-error-phase1b-nonlexical-maxcore419/`；三阶段对比 `agents/temp/fusionvad-ja/diagnostics-fallback-timing-error-phase1b-compare/`。
- 验证：`tests/test_empty_segments_alignment.py tests/test_alignment_quality.py tests/test_asr_alignment_diagnostics.py tests/test_alignment_failure_subtype_analysis.py tests/test_alignment_diagnostics_compare.py tests/test_fallback_timing_error.py tests/test_chunk_packer.py tests/test_pipeline_chunk_packing.py tests/test_vad_chunk_cache.py tests/test_asr_stage_env_scope.py tests/test_aligned_segments_cache.py` 共 48 passed（6 个 NVML warning 为本机环境 warning）。
- 下一步建议：Phase 1c 只针对 `vad_coarse_after_sentinel` 非空文本块做 aligner-local speech-island splitting；若仍无效，再进 Phase 2 CTC/secondary aligner。`nonlexical_text` 走显式粗时间轴并保留 display_text，`align_text_empty` 进入 prealign/review 小修，不应再作为主要 fallback 瓶颈。

#### V12 · R14 Phase 1c sentinel-only speech-island split（GPU 闭环完成，默认关闭）

Phase 1a 的全局/半全局 chunk 缩短没有让 sentinel 下降，Phase 1b 又确认真正剩余瓶颈是 `vad_coarse_after_sentinel`：ASR 有非空文本，但 forced aligner 在含长 gap / 噪声 / 多 speech island 的 chunk 上吐 sentinel。社区和论文常见处理不是把所有 ASR chunk 继续切短，而是在 alignment 层对失败块做 VAD/speech-island 局部切分，或退到 CTC/Viterbi aligner；这更符合当前 high-recall VAD 的取舍。

- 实现：新增 `ALIGNMENT_SENTINEL_ISLAND_SPLIT=1` opt-in 开关，默认 `0`，不改变正式默认。只在 forced aligner sentinel 且文本非空时触发；`nonlexical` / `align_text_empty` / 空文本不触发。局部 VAD 使用 `detect_speech_spans(chunk["path"])`，按 `ASR_CHUNK_PACK_FRAME_HOP_S` 把 `ALIGNMENT_SENTINEL_ISLAND_PAD_FRAMES`、`ALIGNMENT_SENTINEL_ISLAND_MERGE_GAP_FRAMES` 换成秒，默认各 `6` 帧；`ALIGNMENT_SENTINEL_ISLAND_MIN_S=0.25`，`ALIGNMENT_SENTINEL_ISLAND_MAX_SPLITS=8`。
- 流程：正常主 pipeline 仍保持“ASR 文本模型阶段加载 -> 卸载 -> forced aligner 阶段加载 -> 卸载”的显存边界；Phase 1c 只在最终 fallback 阶段对失败 chunk 串行重载局部 ASR+align。局部 split 前后显式 `unload_forced_aligner()`，避免 ASR 与 aligner 常驻共存。若局部 split 成功，不再写 `Alignment 哨兵触发` / fallback 日志；若失败，保留原有 VAD/比例回退路径和诊断口径。
- 代码：`src/whisper/transcribe.py` 增加 `_split_alignment_sentinel_with_speech_islands()` 与 frame-based island pad/merge helper；`src/whisper/pipeline.py` 在 `_finalize_aligned_chunk_without_asr_retry()` 传入 backend/source audio，并把 `Alignment speech-island...` 日志纳入 asr_log。新增 `tests/test_alignment_sentinel_island_split.py` 覆盖默认关闭、局部 split 成功 offset、成功优先于 fallback、失败保留旧 fallback。
- 验证：`.venv/bin/python -m pytest tests/test_alignment_sentinel_island_split.py tests/test_asr_alignment_diagnostics.py tests/test_alignment_quality.py tests/test_fallback_timing_error.py` 13 passed；`.venv/bin/python -m py_compile src/whisper/transcribe.py src/whisper/pipeline.py tools/fusionvad_ja/diagnose_asr_alignment.py` 通过。
- GPU 闭环：synthetic64 baseline 为 chunks `148`、segments `180`、forced `84`、fallback `28`、`vad_coarse_after_sentinel=28`；开启 Phase 1c 后 segments `182`、forced `85`、fallback `11`、`vad_coarse_after_sentinel=11`，gate 为 `PASS_RECLASSIFICATION_CLEANUP`。匿名样片 A baseline 为 chunks `240`、segments `965`、forced `101`、fallback `137`、`vad_coarse_after_sentinel=122`、runtime `1053.5s`；开启 Phase 1c 后 chunks `267`、segments `1061`、forced `154`、fallback `48`、`vad_coarse_after_sentinel=42`、runtime `3150.4s`。结论：策略有效，但当前 per-chunk 模型卸载/重载过慢，正式默认保持关闭。
- 旁证审计：新增 `tools/fusionvad_ja/generate_alignment_compare_review_html.py`，把 baseline 与 opt-in 输出做逐片段视频审阅页，产物 `agents/audits/fusionvad-ja/alignment-compare-sample-a-qwen29239-island-split/index.html`；`agents/audits/fusionvad-ja/index.html` / `latest-audit.html` 指向最新审计页。
- Phase 1d 代码：`src/whisper/transcribe.py` 新增 `_split_alignment_sentinels_with_speech_islands_batch()`，复用现有整批 `text_only -> unload -> forced align` helper；`src/whisper/pipeline.py` 在主 align 完成后收集所有 sentinel candidates，批量执行 island retry，再把成功 words merge 回原 chunk。这样不让 ASR 与 aligner 常驻共存，继续保持 8GB 显存友好的 staged pipeline，但避免每个失败 chunk 都重复加载模型。单 chunk 旧函数保留给局部 retry / 单测兜底。
- Phase 1d synthetic64 GPU 闭环：staged batch 版 64/64 完成，诊断结果与 Phase 1c 对齐：chunks `148`、segments `182`、forced `85`、fallback `11`、`vad_coarse_after_sentinel=11`、gate `PASS_RECLASSIFICATION_CLEANUP`。这说明 batch 化无质量回退；但 synthetic64 是 64 个短 WAV 逐个完整 workflow，因此仍会重复加载/卸载模型，不适合衡量吞吐收益。
- Phase 1d 匿名样片 A GPU 闭环：长片 staged batch 版完成，质量指标与 Phase 1c per-chunk 初版对齐：chunks `267`、segments `1061`、forced `154`、fallback `48`、`vad_coarse_after_sentinel=42`、ASR dropped uncertain `0`、align-text-empty `10`。耗时从 baseline `1053.5s` 增到 `1641.7s`，但相比旧 per-chunk Phase 1c `3150.4s` 明显收敛；结论是 batch 版适合作为可选 repair，不应默认开启。
- Qwen 后端 warning 处理：Grok 检索临时返回空流，改用 Transformers / Qwen 包源码定位。`temperature` warning 来自 deterministic/greedy generate 下 sampling-only `temperature=0.0` 被忽略；`pad_token_id` warning 来自底层 Qwen `thinker.generate()` 未配置 pad token，Transformers 每次回退到 `eos_token_id=151645`。`src/whisper/local_backend.py` 已在加载 Qwen ASR / ForcedAligner 后归一化底层 `generation_config`：清除非默认 `temperature`，缺失 `pad_token_id` 时设为第一个 `eos_token_id`；这不改变 greedy 解码，只减少无效 warning。
- 工具修复：`tools/fusionvad_ja/diagnose_asr_alignment.py` 支持单个 `--case-label` 广播到多个 aligned JSON，避免 synthetic64 诊断汇总因统一 label 报错；多个 label 时仍按文件数严格匹配。
- 下一步：先审阅 Phase 1d 匿名样片 A 的对比页和字幕观感；若 fallback 余量仍不可接受，再进入 Phase 2：CTC/secondary aligner（例如日语 HuBERT/wav2vec/CTC forced alignment）或专门 boundary refiner。Phase 1d 暂保持 opt-in，不默认开启。

</details>

#### V13 · 字幕 cue timing polish（Netflix-style 2-frame gap + linger）

Phase 1c 把 fallback 降下来后，审计发现字幕切换观感仍偏突兀：部分 cue 虽然时间轴更准确，但相邻字幕之间出现 0.1-0.4s 的短空隙，视觉上像闪断。参考 Netflix timed-text 常见规则后，当前策略是只在最终 cue plan 层做 polish，不回写 VAD/ASR/forced aligner 原始结果：

- 使用每个视频真实 FPS，失败才回退 `30000/1001`。最小字幕 gap 固定为 2 帧。
- `SUBTITLE_SHORT_GAP_COLLAPSE_S=0.5`：若相邻 cue 的 gap 短于该阈值，把上一 cue 延到 `next_start - 2 frames`，消除短闪断。
- `SUBTITLE_LINGER_S=0.45`：若 gap 不短于 0.5s，上一 cue 最多 linger 0.45s，但必须保留至少 0.5s 的真实停顿。
- 单条 cue 也允许 linger，但仍受 `MAX_SUBTITLE_DURATION` 约束；`SUBTITLE_TIMING_POLISH_ENABLED=0` 可关闭并回到旧 alignment end 行为。
- 验证：`tests/test_subtitle_options.py tests/test_subtitle_quality_pass.py` 38 passed；`src/subtitles/options.py src/subtitles/writer.py` 语法检查通过。

#### V14 · FusionVAD-JA v1.13 full29239 exact-island 默认切换

用户明确选择优先复用 0.6B full SFT 模型资产，降低分发和下载门槛；因此研究默认不再保留 base Qwen3-ASR-0.6B 专用回退开关。通用 `FUSIONVAD_JA_MODEL_PATH` 仍可用于 ablation，但默认特征源直接切到 `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`，本地缓存目录为 `models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame`。

- Grok 检索：FusionVAD 2025 的 MFCC + frozen PTM addition fusion 仍是最贴近本项目的轻量复现路线；Interspeech 2024 transformer VAD 和 HF VAD 模型列表说明预训练语音表征仍是主流方向，但没有替代当前“synthetic exact-island + downstream fallback 闭环”的更直接解法。
- 训练：用 full29239 0.6B feature cache 重新训练 exact-island head，混合 v1-mini strong/negative `302` 条 + synthetic v5 exact-island `256` 条，共 `558` 条；命令日志 `agents/temp/fusionvad-ja/train-full29239-exact-island.run.log`。checkpoint：`datasets/train/fusionvad-ja/v1-13/qwen3-asr-0.6b-full29239/exact-island-bilstm-fromscratch-v1mini-galgame-synthv5-558-batch16-lr2e-4-steps1024/fusionvad_ja_addition_bilstm.pt`。
- 评测：exact-island test64 上 threshold `0.20` + pad `0.2s` 为 recall `0.9505`、precision `0.7360`、extra audio ratio `1.2915`，漏语音过多；当前默认取 threshold `0.10` + pad `0.2s`。boundary bench：v1.13 th0.10 recall `0.9935`、missed `1.82s`、extra ratio `1.6012`、start/end p50 `0.628s/2.002s`；full29239 long-gap 旧头 th0.02 为 recall `0.9958`、missed `1.17s`、extra ratio `1.6225`、start/end p50 `1.202s/2.068s`。结论是 start 边界改善；end 偏长可由最终 cue timing polish 压缩，不作为当前默认切换 blocker。真实 held-out / 匿名样片 A downstream 验证应重点看漏语音、长 chunk、ASR empty 和 forced-aligner fallback。
- 真实闭环：v1.6 real-heldout recall `0.9994`、missed `0.22s`、extra ratio `1.6255`，符合高召回 proposal 目标。匿名样片 A 对比 v1.11 baseline：chunks `240→227`、segments `965→967`、fallback chunks `137→114`、`vad_coarse_after_sentinel 122→104`、ASR+Alignment `1026s→1079s`；改善有限，不能替代 Phase 1d staged island repair（fallback `48`、`vad_coarse_after_sentinel=42`）。
- 代码默认：`src/vad/fusionvad_ja/backend.py` 后端名改为 `fusionvad_ja_v1_13_full29239_exact_island`，默认 checkpoint / model path / threshold 同步到 v1.13；`tools/fusionvad_ja/run_full_workflow.py` 和 `export_fusionvad_operating_point.py` 同步默认值。
- 验证：`.venv/bin/python -m pytest tests/test_fusion_lite_vad_backend.py tests/test_fusionvad_ja_dataset.py -q` 90 passed；`src/vad/fusionvad_ja/backend.py tools/fusionvad_ja/run_full_workflow.py tools/fusionvad_ja/export_fusionvad_operating_point.py` 语法检查通过。

#### V15 · R15/R16 speech-island 边界主线与 F0/gender 降级

样片 A 的 full SFT 双语字幕主观效果已经可用，但人工审阅发现仍有少量语气词/短句幻觉，典型表现是无明显说话、gap 或低信息声音上出现字幕。这类问题更像 non-speech/gap 诱发 ASR hallucination 与 chunk 粒度过粗，而不是单纯 subtitle end 偏长。当前工作流因此调整为：

- FusionVAD-JA 第一级继续输出高召回 frame-level speech mask，保证不漏对白、呻吟、喘息和短促人声。
- 第二级新增 pre-ASR speech-island / endpoint packer，把 mask 切成更接近一句台词或一个 speech island 的 ASR chunk，避免多句话、多 island、长 gap、噪音或多人交替揉在同一 chunk。
- F0/gender 标签从当前主线降级为 legacy/ablation。它在大 chunk、多人交替或混合 island 场景下不稳定，继续作为切分和翻译提示会放大错误；后续如果需要说话人信息，优先在 speech-island 之后接 CAM++ / 3D-Speaker / WeSpeaker sidecar。
- R14 Phase 1d staged batch island retry 保留为 opt-in repair 和上限参考，但不再作为默认解法；主线前移到 ASR 之前，减少 ASR empty、hallucination proxy 和 forced-aligner sentinel 的输入风险。

下一步验收：离线 chunk 成因分析不跑 ASR；env-gated splitter 只切高风险 chunk；boundary-aware head 必须同时报告 recall、missed speech、internal gap、start/end 误差、chunk 数增幅、ASR empty、repeat/hallucination proxy、forced/partial/fallback 和最终 SRT 审计观感。

#### V16 · R15 第一版实现：pre-ASR gap-based island splitter + 离线成因分析

已实现 R15 的后端第一版，默认关闭，不改变正式默认行为：

- `src/audio/chunk_packer.py` 新增 `pre_asr_island_split_enabled` 及 frame-based 阈值：`min_core_frames`、`min_gap_frames`、`min_island_frames`、`max_children`。策略只处理已经 packed 后的高风险 multi-island chunk；单 island / 无内部 gap 的长 speech 不切，避免重复 R14 Phase 1a 的全局缩短失败路线。子 chunk 记录 `parent_chunk_id`、`island_id`、`island_count`、`internal_gap_count`、`internal_gap_max_s` 和 `split_policy="r15_pre_asr_island_v1"`。
- `src/whisper/pipeline.py` 接入 env：`ASR_PRE_ASR_ISLAND_SPLIT_ENABLED`、`ASR_PRE_ASR_ISLAND_SPLIT_MIN_CORE_FRAMES`、`ASR_PRE_ASR_ISLAND_SPLIT_MIN_GAP_FRAMES`、`ASR_PRE_ASR_ISLAND_SPLIT_MIN_ISLAND_FRAMES`、`ASR_PRE_ASR_ISLAND_SPLIT_MAX_CHILDREN`，并把策略写入 runtime VAD signature。
- `src/whisper/vad_chunk_cache.py` 把新 env 和 packed chunk metadata 纳入 cache signature / payload；改参数会正确触发 VAD/chunk cache miss。
- `tools/fusionvad_ja/analyze_pre_asr_island_chunks.py` 新增离线分析工具：读取 VAD cache 的 packed chunks 和可选 `diagnostics.jsonl`，按时间 overlap 匹配 diagnostics，不依赖 chunk id，输出 `summary.json`、`chunk_analysis.jsonl`、`summary.md`。

离线跑匿名样片 A v1.13 exact-island 现有 VAD cache + diagnostics，产物：

- `agents/temp/fusionvad-ja/pre-asr-island-analysis-sample-a-v1-13/summary.md`
- 输入 VAD cache：`agents/temp/fusionvad-ja/full-workflow-qwen29239-sample-a-v1-13-exact-island/vad-cache/2bb8f0e7.fb3cf35efd15ca7e.json`
- 输入 diagnostics：`agents/temp/fusionvad-ja/diagnostics-full-workflow-qwen29239-sample-a-v1-13-exact-island/diagnostics.jsonl`

关键发现：

- chunks `227`，multi-island chunks 只有 `19`，`multi_island_long_gap` 只有 `3`。
- `vad_coarse_after_sentinel` 为 `104`，但多数落在 `continuous_speech_no_internal_gap + long_chunk`；duration/core p50/p90 均接近 `28.467s/24.467s`。
- 结论：R15 gap-based splitter 是必要 guardrail，但对当前 v1.13 样片 A 不是主杠杆；真正瓶颈是 FusionVAD-JA 输出的长连续 positive island 过多。下一步必须进入 R16：导出/利用 per-frame probability valley，或训练 boundary-aware head / boundary refiner，把“单个 28s 长 speech island”切成更接近一句话的 island。

验证：

- `.venv/bin/python -m pytest tests/test_chunk_packer.py tests/test_vad_chunk_cache.py tests/test_pipeline_chunk_config_runtime.py tests/test_pre_asr_island_analysis.py -q` -> 24 passed。
- `.venv/bin/python -m pytest tests/test_pipeline_chunk_packing.py tests/test_asr_alignment_diagnostics.py tests/test_alignment_quality.py tests/test_alignment_diagnostics_compare.py -q` -> 18 passed。
- `.venv/bin/python -m py_compile src/audio/chunk_packer.py src/whisper/pipeline.py src/whisper/vad_chunk_cache.py src/core/config.py tools/fusionvad_ja/analyze_pre_asr_island_chunks.py` 通过。

#### V17 · R16 第一版实现：frame-score valley split 离线接口

R15 发现样片 A 的主要残余失败不是多 island + 明确 gap，而是长连续 positive island。因此 R16 第一版先做可量化、默认关闭的 valley split 接口，不直接改正式默认：

- `src/vad/fusionvad_ja/backend.py` 新增 `FUSIONVAD_JA_EXPORT_FRAME_SCORES=1` opt-in：只在诊断/离线分析时把 per-frame probabilities 写入 `SegmentationResult.parameters["frame_scores"]`，默认不写大数组进 cache。
- `src/audio/chunk_packer.py` 新增 R16 参数：`pre_asr_valley_split_enabled`、`min_core_frames`、`target_core_frames`、`min_valley_frames`、`min_child_frames`、`max_children`、`threshold`。策略只在有 frame scores 且长连续段内存在持续低分 valley 时切，默认关闭；R15 gap splitter 仍作为 guardrail。
- `src/whisper/pipeline.py` / `src/core/config.py` / `.env.example` 接入 `ASR_PRE_ASR_VALLEY_SPLIT_*` 与 cache signature；`src/whisper/vad_chunk_cache.py` 升到 v5，并 round-trip `valley_split_count` / `valley_score_min`。
- 新增 `tools/fusionvad_ja/export_frame_scores.py`：对指定音频导出 FusionVAD-JA frame scores，用于不跑 ASR 的 R16 离线分析。
- 新增 `tools/fusionvad_ja/analyze_valley_splits.py`：读取 VAD cache + frame scores + 可选 diagnostics，离线模拟 valley split，输出 `summary.json`、`valley_split_plan.jsonl`、`summary.md`，重点看 chunk 增幅、`vad_coarse_after_sentinel` 风险 chunk 覆盖率和 child duration 分布。

本轮样片 A 离线实测：

- `tools/fusionvad_ja/export_frame_scores.py` 已用提权 CUDA 跑完整匿名样片 A，`runtime_device.actual_device=cuda`，导出 `269833` 帧，片长 `5396.64s`，产物在 `agents/temp/fusionvad-ja/r16-frame-scores-sample-a-v1-13/frame_scores.json`。
- 默认 R16 valley 参数在同一 VAD cache 上可把 `vad_coarse_after_sentinel` 风险 chunk `104/104` 全覆盖，但代价是 chunks `227→687`（`3.026x`），过度切分。
- 更保守参数（例如 `min_core_frames=900/1200/1500`、`min_valley_frames=60/90/120`、`min_child_frames=300/360/450`、`threshold=0.05/0.03/0.02`、`max_children=3`）仍为 chunks `227→444`（`1.956x`），risk split `103/104`。这说明当前 FusionVAD probability valley 在长连续 positive island 内过于普遍，rule-based valley split 能覆盖风险，但不能以可接受 chunk 增幅进入主流程。
- 结论：R16 rule-based valley split 暂不进入 GPU 小闭环或默认 pipeline；后续应训练 boundary-aware head / boundary refiner，让模型显式学习 internal gap / endpoint，而不是继续靠固定 probability threshold 找 valley。

后续验收口径：

- 若新 boundary-aware/refiner 输出能在离线阶段把 chunk 增幅控制在约 `1.2x` 以内，同时显著降低长连续 island / `vad_coarse_after_sentinel` 风险，再开 `ASR_PRE_ASR_VALLEY_SPLIT_ENABLED=1` 或新 refiner flag 做 synthetic64 + 样片 A GPU 闭环。v1.14 256-step 候选已完成 gate：synthetic exact-island 有正向信号，但真实 held-out 的高召回阈值会接近 all-positive；匿名样片 A downstream `threshold=0.003` 没有降低 fallback（`114→115`）或 sentinel（`104→103`），因此不替换 v1.13 默认。
- GPU 闭环指标继续看 ASR empty、repeat/hallucination proxy、forced/partial/fallback、最终 SRT 观感和可播放审计片段。
- 验证：`.venv/bin/python -m pytest tests/test_chunk_packer.py tests/test_vad_chunk_cache.py tests/test_pipeline_chunk_config_runtime.py tests/test_pre_asr_island_analysis.py tests/test_valley_split_analysis.py -q` -> 30 passed；新增/改动脚本语法检查通过；R16 frame-score 导出日志在 `agents/temp/logs/r16-export-frame-scores-sample-a-v1-13.run.log`。

#### V18 · v1.14 boundary-aware gate 结论

v1.14 是“在 v1.13 exact-island head 上继续 fine-tune boundary/gap loss”的小步实验，目标是减少 internal gap 和粗长 end，而不是直接换默认：

- 训练配置：`boundary_loss_weight=0.25`、`gap_loss_weight=0.10`、`256` steps、同一 full29239 frozen feature + MFCC addition BiLSTM，输出 `datasets/train/fusionvad-ja/v1-14/qwen3-asr-0.6b-full29239/boundary-aware-ft-v1-13-558-batch16-lr5e-5-steps256/fusionvad_ja_addition_bilstm.pt`。
- synthetic exact-island test64：`threshold=0.005` 的 recall `0.9992`、missed `0.49s`、extra `1.5401`、start/end p50 `1.152s/1.282s`；说明 boundary/gap loss 能降低部分 extra 和 end 偏长。
- v1.6 real-heldout 参考集：该集是真实本地视频人工审计，不是拼接合成，且边界标签可能不如 synthetic 精确；只用作“是否明显漏人声”的参考。v1.14 `threshold=0.003` 为 recall `0.9961`、missed `1.50s`、extra `1.5950`，`threshold=0.005` 漏到 `13.18s`，`threshold<=0.002` 又接近 all-positive。
- 匿名样片 A downstream：同一 Qwen3-ASR-1.7B full29239 + Qwen3-ForcedAligner，v1.14 `threshold=0.003` 输出 chunks `222`、segments `986`、cues `1061`、ASR+Alignment `1093.3s`、fallback chunks `115`、`vad_coarse_after_sentinel=103`、forced `101`。v1.13 baseline 是 chunks `227`、segments `967`、cues `1020`、ASR+Alignment `1079.4s`、fallback `114`、`vad_coarse_after_sentinel=104`、forced `106`。
- 结论：v1.14 不过 downstream gate，不替换 v1.13 默认。下一步不要继续只调 threshold 或小步 fine-tune；应回到 R16/R15 的根因：训练更明确的 endpoint/boundary refiner，或在 ASR 前引入可控的 speech-island packer，并用 sample A 的 fallback 审计页验证字幕观感。

#### V19 · fallback-safe boundary gate 与 v1.13/v1.14 复测

用户侧人工观看样片 A 双语字幕后确认：主观效果已经可用，但少数片段仍会在非语音/gap 上出现语气词或短句幻觉。这里的关键指标不能只看 `forced` 比例，因为 forced aligner 失败后如果回退到 20-30s 的 `vad_coarse` chunk，字幕时间轴仍然不可用。2026-05-31 Grok 检索后，相关方向和本项目匹配为：WhisperX 的 VAD-first segmentation + forced alignment、stable-ts 的 silence suppression/regroup、Semantic VAD / Dynamic Speech Endpoint Detection 的 endpoint 建模。项目内结论是新增 fallback-safe boundary gate：粗 fallback chunk 默认必须 `<=8s`，否则视为 unsafe。

新增工具：

- `tools/fusionvad_ja/measure_fallback_safe_boundaries.py`：读取 VAD chunk cache + `diagnostics.jsonl`，输出 `summary.json`、`chunk_metrics.jsonl`、`unsafe_fallback_chunks.jsonl`、`summary.md`。
- 指标：`fallback_chunk_count`、`fallback_unsafe_count`、`fallback_safe_ratio`、`sentinel_fallback_count`、fallback duration/core duration/internal gap/speech-island 统计和最长 unsafe fallback 列表。
- 单测：`tests/test_fallback_safe_boundary_metrics.py` 覆盖长粗 fallback 判 unsafe、短粗 fallback 判 safe、forced chunk 不计入 fallback。

样片 A 同口径复测：

- v1.13 baseline：`chunks=227`、`fallback_chunk_count=104`、`fallback_unsafe_count=104`、`fallback_safe_ratio=0.0`、`sentinel_fallback_count=104`、fallback duration p50/p90/max 均为 `28.466667s`。产物：`agents/temp/fusionvad-ja/fallback-safe-boundary-metrics-sample-a-v1-13/`。
- v1.14 `threshold=0.003`：`chunks=222`、`fallback_chunk_count=104`、`fallback_unsafe_count=104`、`fallback_safe_ratio=0.0`、`sentinel_fallback_count=103`、fallback duration p50/p90/max 均为 `28.466667s`。产物：`agents/temp/fusionvad-ja/fallback-safe-boundary-metrics-sample-a-v1-14-th0p003/`。
- 解释：v1.14 的 forced/sentinel 数没有本质改善，且 fallback-safe gate 完全未过；这些 unsafe fallback 多数在 cache metadata 中呈现为单个长 positive island（v1.14 fallback `speech_island_count p90=1`、`internal_gap_max p90=0`），说明仅靠 gap splitter 或阈值扫描无法解决。真正要做的是内部 endpoint / speech-island boundary prediction。

下一步：

- v1.15 不继续只调 threshold；训练或接入 endpoint/boundary refiner，目标是把长连续 positive island 切成更像一句台词的 ASR chunk。
- 验收顺序：先 synthetic exact-island / fallback-safe boundary metric，再 v1.6 real-heldout recall，最后匿名样片 A downstream + 可播放审计页。
- 成功标准不只看 `forced↑`，还必须看 `unsafe fallback↓`、fallback duration p90/max 降低、ASR empty 不上升、start recall 不明显下降、最终字幕观感变自然。

#### V20 · v1.15 endpoint/boundary refiner 训练入口

v1.15 的目标已经从“更会判断有没有人声”改成“在高召回前提下给出可用边界和切点”：不再把多个 speech island / 长 gap / BGM / 噪声 / overlap 粗暴合成 20-30s 大 chunk，除非中间 gap 很短且 endpoint/cut score 不支持切。end 可以略长，因为最终字幕 timing polish 会压前一条 cue 的 end；但 fallback chunk 过长不可接受。

本轮实现：

- `src/vad/fusionvad_ja/model.py` 新增 `AdditionFusionEndpointBiLSTM`：沿用 PTM feature + MFCC addition fusion + 2 层 BiLSTM 思路，输出四路 logits：`speech`、`start`、`end`、`cut`。
- `src/vad/fusionvad_ja/train.py` 新增 `EndpointRefinerTrainConfig`、`endpoint_targets_from_record`、`build_endpoint_feature_windows`、`train_endpoint_refiner_classifier`。训练 loss 由 `speech BCE + start/end boundary BCE + internal gap loss + cut BCE` 组成；start/end/cut 目标优先从 `teacher_segments["supervised"]` 派生，缺失时回退 `speech_frames`。v1.15-b 后，start/end/cut 辅助头支持独立 `pos_weight`，避免边界/切点正样本比例过低时塌成全负。
- `tools/fusionvad_ja/train_endpoint_refiner.py` 新增 CLI，接口沿用 `--labels` / `--feature-manifest`，可直接吃现有 full29239 feature cache。
- `tools/fusionvad_ja/export_endpoint_refiner_predictions.py` 新增导出 CLI：输出 `speech_frames`、`start_frames`、`end_frames`、`cut_frames` 和四路 probability summary；`tools/fusionvad_ja/benchmark_boundary_predictions.py` 追加 `cut_gap_coverage_ratio` / `cut_supported_ratio`，用于检查 cut 头是否真的覆盖 synthetic long gap。
- `tools/fusionvad_ja/measure_fallback_safe_boundaries.py` 已扩展：除 fallback duration p50/p90/max 外，支持 `--boundary-manifest` 统计 synthetic truth start/end p50/p90，支持 `--measure-audio-silence` 统计 sample A fallback chunk 是否跨大段静音。

已执行 smoke：

- endpoint refiner CPU smoke：`datasets/train/fusionvad-ja/v1-mini/mixed-train/labels.jsonl` + `datasets/train/fusionvad-ja/v1-mini/feature-cache-16/feature_manifest.json`，`max_steps=2`，产物 `agents/temp/fusionvad-ja/v1-15-endpoint-refiner-smoke/fusionvad_ja_endpoint_refiner.pt`，参数量 `12276`（小配置，仅验证入口，不代表正式模型）。
- endpoint refiner GPU 训练候选：full29239 feature cache，synthetic exact-island train256 + v1-mini strong/negative `302` 条，共 `558` 条；`batch_size=16`、`lr=2e-4`、`steps=256`、`positive_loss_weight=2.0`、`boundary/internal_gap/cut_loss_weight=0.5`、`boundary_radius_frames=1`、`cut_min_gap_s=0.5`。训练在 RTX 4060 Ti 提权 CUDA 下完成，loss `2.2774→1.0909`，frame_accuracy `0.3988→0.7991`，参数量 `1,230,884`，checkpoint `datasets/train/fusionvad-ja/v1-15/qwen3-asr-0.6b-full29239/endpoint-refiner-synthv5-v1mini-558-batch16-lr2e-4-steps256/fusionvad_ja_endpoint_refiner.pt`。
- sample A v1.13 fallback-safe v2：`fallback_safe_ratio=0.0`，`fallback_unsafe_count=104`，fallback duration p50/p90/max 仍为 `28.466667s`；开启音频静音 proxy 后，`fallback_long_silence_count=10`、fallback longest silence p90 `1.0s`、max `4.2s`，产物 `agents/temp/fusionvad-ja/fallback-safe-boundary-metrics-sample-a-v1-13-v2/`。
- sample A v1.14 th0.003 fallback-safe v2：`fallback_safe_ratio=0.0`，`fallback_unsafe_count=104`，fallback duration p50/p90/max 仍为 `28.466667s`；`fallback_long_silence_count=10`、fallback longest silence p90 `1.0s`、max `4.0s`，产物 `agents/temp/fusionvad-ja/fallback-safe-boundary-metrics-sample-a-v1-14-th0p003-v2/`。

synthetic64 gate 结果：

- v1.15 `speech_threshold=0.5` 不合格：speech recall `0.7739`，missed speech `137.19s`，extra audio ratio `0.8548`，start/end p50 `0.276s/0.349s`；这是 precision 型 operating point，不符合高召回要求。
- v1.15 speech 阈值扫描显示：`threshold=0.02` 可保 recall `0.9969`、missed `1.9s`，但 extra audio ratio `1.623`、start/end p50 `1.886s/1.678s`，不优于 v1.13 `threshold=0.10` 的既有 gate（recall `0.9935`、missed `1.82s`、extra `1.6012`、start/end p50 `0.628s/2.002s`）。
- v1.15 cut 头暂不可直接切 speech：`cut_threshold=0.10` 可覆盖 gap（`cut_gap_coverage_ratio=0.984`）但 recall 只有 `0.5265`；`cut_threshold=0.14` 保 recall 约 `0.9908`，但 cut 覆盖只有 `0.159` 且 supported ratio 低，说明 cut logits 尚未学成可靠切点。
- 结论：v1.15 训练/导出/评测入口可用，但这个 558-row/256-step checkpoint **未过 synthetic gate**，不跑 v1.6 real-heldout 或样片 A downstream，不替换 v1.13 默认。产物：`agents/temp/fusionvad-ja/v1-15-endpoint-refiner-test64-probabilities/`、`agents/temp/fusionvad-ja/v1-15-endpoint-refiner-threshold-sweep/`、`agents/temp/fusionvad-ja/v1-15-endpoint-refiner-threshold-sweep-cut-applied/`。

v1.15-b 结果：

- 根因量化：训练混合数据中 start/end 正样本只有约 `0.4-0.7%`，cut 在 v1-mini 部分约 `3%`，无 `pos_weight` 的 BCE 会自然学成全负。v1.15-b 增加 `--start-positive-loss-weight` / `--end-positive-loss-weight` / `--cut-positive-loss-weight`，默认仍为 `1.0`，不改变旧 checkpoint 解释。
- v1.15-b 训练：仍用 full29239 feature cache，synthetic exact-island train256 + v1-mini strong/negative `302` 条，共 `558` 条；`batch_size=16`、`lr=2e-4`、`steps=512`、`positive_loss_weight=2.0`、`boundary_loss_weight=1.0`、`internal_gap_loss_weight=0.5`、`cut_loss_weight=0.75`、`start/end pos_weight=120`、`cut pos_weight=8`。RTX 4060 Ti 提权 CUDA 完成，loss `4.4001→2.5554`，frame_accuracy `0.3784→0.8757`，checkpoint `datasets/train/fusionvad-ja/v1-15/qwen3-asr-0.6b-full29239/endpoint-refiner-synthv5-v1mini-558-batch16-lr2e-4-steps512-posaux120-cut8/fusionvad_ja_endpoint_refiner.pt`。
- synthetic64 推荐候选点：`speech_threshold=0.010` + `cut_threshold=0.94` + `apply_cut_to_speech`，recall `0.9970`、missed speech `1.80s`、extra audio ratio `1.5147`、predicted segments `186`、cut-gap coverage `0.762`、cut-supported ratio `0.839`、start/end p50 `0.968s/0.926s`。对比 v1.13 `threshold=0.10`：recall `0.9935`、missed `1.82s`、extra `1.6012`、predicted segments `74`、start/end p50 `0.628s/2.002s`。结论：synthetic 上 v1.15-b 以更多、更碎的 speech-island 换来更低 extra 和更短 end fallback 风险。
- v1.6 real-heldout 参考复测：同一真实本地人工审计 `79` 条（非合成，边界不绝对精确）上，v1.15-b `speech=0.010/cut=0.94` + pad `0.2s` 为 recall `0.9978`、missed speech `0.84s`、extra audio ratio `1.6253`。对比 v1.13 的 recall `0.9994`、missed `0.22s`、extra `1.6255`，v1.15-b 守住高召回但没有在真实 held-out 上降低 extra。
- 结论：v1.15-b **可进入小样片 A downstream opt-in 对比**，但还不足以替换 v1.13 默认。进入 downstream 时必须重点看 chunks 增幅、ASR empty、forced/fallback、unsafe fallback duration、字幕观感；如果 chunk 数和 ASR 成本过高，v1.15-b 只能作为 cut/endpoint teacher 或下一轮训练样本筛选器。

下一步执行顺序：

1. 优先评估 v1.16 endpoint refiner；只有 synthetic64 + v1.6 参考都过线后，再接匿名样片 A downstream 小闭环。
2. 若 sample A 的 unsafe fallback 和幻觉 proxy 有实质改善，再扩大 failure-driven 数据构造；若未改善，v1.16 仍作为 endpoint/cut teacher 和默认高召回边界候选继续迭代。
3. 默认 FusionVAD-JA 已切到 v1.16 endpoint refiner；v1.13 full29239 exact-island 仅保留为历史 baseline / ablation。

v1.16 结果（4096 条 Galgame multi-island synthetic）：

- 数据动机：用户确认 Galgame 原始训练 clip 基本是已裁切语音，适合人工构造“多条语音 + 长 gap/白噪/真实 negative/BGM/overlap”的精确时间轴数据；目标不是单纯提高 speech recall，而是让 VAD/refiner 学会一句话 / speech island 边界，降低 fallback chunk 过长和多 island 合并。
- 训练数据：`datasets/train/fusionvad-ja/v1-16/galgame-synthetic-timeline-v6-boundary4096-train/`，`4096` records，skipped `0`，总时长 `105208.753s`，每条 `3` 或 `4` 个 speech island（`3:3566`、`4:530`）。增强包括 `1-6s` gap、silence/white_noise/hum/real_negative/fade_noise、随机 gain、band/lowpass、`5-30ms` equal-power crossfade、BGM/background mix、overlap speech 和轻量 codec aug。exact-island labels 输出 `datasets/train/fusionvad-ja/v1-16/galgame-synthetic-timeline-v6-boundary4096-exact-island-train/labels.jsonl`，speech frame ratio `0.5621`。
- 特征缓存：冻结特征使用 `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`（本地缓存 `models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame`），输出 `datasets/train/fusionvad-ja/v1-16/qwen3-asr-0.6b-full29239/galgame-synthetic-timeline-v6-boundary4096-feature-cache/`，CUDA bfloat16，cached `4096`、errors `0`、skipped `0`，约 `19G`。
- 训练实现修复：首次训练在本机 RAM 下被 OOM kill，根因是 `train_endpoint_refiner_classifier()` eager 加载全部 `.npz` feature。已改为 endpoint/refiner lazy feature window：训练集只保存 feature manifest 引用，batch 内按需加载当前 `.npz`，避免 4096 条 Qwen 0.6B feature 常驻内存；`tests/test_fusionvad_ja_dataset.py` 通过 `92 passed`。
- 训练配置：`batch_size=16`、`lr=2e-4`、`steps=2048`、`positive_loss_weight=2.0`、`boundary_loss_weight=1.0`、`internal_gap_loss_weight=0.5`、`cut_loss_weight=0.75`、`start/end pos_weight=120`、`cut pos_weight=8`，混合 v1-mini strong/negative 作为补充。checkpoint `datasets/train/fusionvad-ja/v1-16/qwen3-asr-0.6b-full29239/endpoint-refiner-boundary4096-v1mini-batch16-lr2e-4-steps2048-posaux120-cut8/fusionvad_ja_endpoint_refiner.pt`，trainable params `1,889,252`，final loss `1.6412`，frame_accuracy `0.9305`。
- synthetic64 gate：从 `agents/temp/fusionvad-ja/v1-16-endpoint-refiner-synth64-probabilities/predictions.jsonl` 做阈值 sweep。推荐候选 `speech_threshold=0.020` + `cut_threshold=0.960` + `apply_cut_to_speech`，recall `0.9998`、missed speech `0.14s`、extra audio ratio `1.3425`、predicted segments `212`、cut-gap coverage `0.968`、cut-supported ratio `0.923`、start/end p50 `0.399s/0.601s`。相比 v1.15-b 的 missed `1.80s`、extra `1.5147`、start/end p50 `0.968s/0.926s` 明显改善。
- v1.6 real-heldout 参考：同一真实本地人工审计 `79` 条，`speech=0.020/cut=0.960` + pad `0.2s` 为 recall `0.9927`、missed speech `2.82s`、missed segments `4`、extra audio ratio `1.5919`。对比 v1.11 的 recall `0.9809`、missed `7.42s` 更稳，但 extra 仍偏高；由于 v1.6 边界人工标注不够精确，只作为“是否明显漏人声”的参考。
- 当前结论：v1.16 已过 synthetic boundary gate，并已作为默认 FusionVAD-JA operating point；下一步重点看 sample A 的 chunk 增幅、ASR empty、forced/fallback、unsafe fallback duration、无声/gap 幻觉字幕是否下降。


### 历史任务摘要

<details>
<summary>展开历史任务摘要（H01-H35）</summary>

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

</details>

### 历史验证基线

<details>
<summary>展开历史验证基线</summary>

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

</details>

### FusionVAD-JA 研究归档

<details>
<summary>展开 v1-mini 至 v1.8 详细实验记录和旧计划</summary>

原始 FusionVAD-JA 研究计划与实验记录：

FusionVAD-JA 是训练型 VAD 研究线，用于复现 FusionVAD 论文的“PTM 特征 + MFCC + 简单 addition fusion”思路，并面向日语/JAV/galgame 近域数据做适配。研究代码在 `src/vad/fusionvad_ja/`，临时 smoke 输出写入 `agents/temp/fusionvad-ja/`。下载后的数据、feature cache 和 checkpoint 按 split 归档到 `datasets/train/fusionvad-ja/`、`datasets/val/fusionvad-ja/`、`datasets/test/fusionvad-ja/`；`datasets/` 整体不进入 Git 跟踪。本折叠区记录早期实验语境；当前默认 VAD 与模型来源以 README 前文“当前定位”为准。

首轮数据混合：

- `litagin/Galgame_Speech_ASR_16kHz`：核心近域弱正样本，无 VAD 时间戳，作为 `teacher_agree` / weak-positive 使用。
- AVA-Speech：电影语音活动标注，作为 supervised seed。
- VoxConverse：带 speaker/timestamp 的 diarization 数据，作为多说话人 speech span seed。
- MUSAN、DNS Challenge、本地视频和合成负样本：提供音乐、噪声、非语音 negative。
- ReazonSpeech、JSUT、JVS：暂列可选日语正样本，不进入首轮默认混合。

标签 schema 使用 JSONL，字段包含 `audio_id`、`source`、`duration_s`、`text`、`teacher_segments`、`frame_hop_s`、`speech_frames` 和 `label_quality`。默认可训练质量为 `supervised`、`teacher_agree`、`negative`；`teacher_conflict` 保留审计但默认不进训练。

训练 v1 计划：

- 冻结 ja whisper 1.5B / `whisper-ja-anime` 生态的 encoder 作为默认 PTM feature extractor；`whisper-large-v3` 作为后续 ablation，不影响现有 ASR。
- 同步提取 16k mono MFCC，默认 `n_mfcc=40`，约 20ms hop，并与 VAD frame 对齐。
- 先离线缓存 frozen encoder feature，再训练轻量头，避免训练时反复跑大模型。
- addition fusion：`whisper_feat -> 256`，`mfcc -> 256`，两路 projection 后相加。
- 分类器使用 2 层 BiLSTM + 轻量 frame head，目标可训练参数小于 2M。

正式训练启动口径：

- v0 只用于链路 smoke，不能作为正式模型结论；Galgame weak-positive + synthetic negative 的 val/test 太容易，会产生虚高 F1。
- v1-mini 可以开始训练的前置条件：AVA/VoxConverse 至少一个 supervised seed split 能同时产出 label JSONL 和本地 16k mono 音频；MUSAN/DNS/本地素材至少一个 real-negative split 可用；train/val/test 按音频 ID 去重且互斥；feature cache `errors=0` 或错误样本有明确跳过报告。
- v1-mini 训练后只报告研究指标，不替换默认 VAD；必须同时给出 supervised val/test 的 frame-level precision、recall、F1、speech positive ratio 和 predicted positive ratio。
- 泛化判断至少需要三类 held-out：AVA/VoxConverse supervised、JAV/galgame 近域样本、MUSAN/DNS/本地非语音负样本；同一来源内不得用相邻切片同时进入 train 和 val/test。
- 晋级到默认 VAD 候选前，必须在同一 held-out 上与 `fusion_lite`、`whisperseg-adaptive` 做对比，并确认长静音、音乐、多人重叠、呻吟/短促日语对白不会明显退化。
- 本机已安装 FFmpeg shared libraries，`torchcodec` import 与本地 WAV decode 已验证；PATH 中的 `ffmpeg` 命令仍优先命中 static build，但不影响 torchcodec 动态链接。FusionVAD-JA 数据物化默认优先使用 Hugging Face `Audio` / torchcodec decode；仅当默认 decode 加载或采样失败时 fallback 到 `Audio(decode=False)` bytes 路径。

v1 执行顺序：

1. 先用小样本探测 AVA/VoxConverse schema，确认 timestamp 与音频字段可读取。
2. 将可用 supervised 音频物化到 `datasets/train|val|test/fusionvad-ja/v1-supervised/`，同时写入 label JSONL、audio manifest 和 split summary。
3. 将 MUSAN/DNS/本地 negative 物化到 `datasets/train|val|test/fusionvad-ja/v1-negative/`；没有真实 negative 时只允许继续做 smoke，不标记为正式训练。
4. 构建 v1 feature cache，默认 PTM 为 `whisper-ja-1.5b`，并保留 `whisper-large-v3` ablation hook。
5. 训练 addition-fusion BiLSTM v1-mini checkpoint，评估 supervised val/test 与近域 smoke split；checkpoint 只归档到 `datasets/train/fusionvad-ja/`。
6. 跑 `fusion_lite` / `whisperseg-adaptive` 同集 baseline 后，再决定是否扩大数据或改 teacher/pseudo-label 策略。

当前研究实现拆分：

- `src/vad/fusionvad_ja/dataset.py`：审计、伪标签 schema、frame label、supervised / weak-positive / negative record 构造。
- `src/vad/fusionvad_ja/manifest.py`：把 label JSONL 与音频 manifest 对齐，过滤不可训练质量和缺失音频。
- `src/vad/fusionvad_ja/features.py`：读取 16k mono 音频，提取 frozen Whisper encoder feature 与 MFCC，并写 `.npz` feature cache。cache key 绑定音频路径、大小、mtime 和 `FeatureConfig`。
- `src/vad/fusionvad_ja/model.py`：`AdditionFusionBiLSTM`，默认 `1280 -> 256` 的 Whisper projection、`40 -> 256` 的 MFCC projection，addition 后进入 2 层 BiLSTM 和 frame head。
- `src/vad/fusionvad_ja/train.py`：保留 tiny waveform smoke，同时新增 cached feature 训练；训练顺序按 seed 确定性 shuffle，checkpoint 记录 `window_order`、输入维度和可训练参数量。
- `tools/fusionvad_ja/evaluate_addition_bilstm.py`：读取 cached feature split 和 checkpoint，输出 frame-level loss、accuracy、precision、recall、F1、正帧比例和预测正帧比例。
- `tools/fusionvad_ja/calibrate_addition_threshold.py`：在 validation split 上扫阈值，按 F1 / precision / recall 约束选择研究阈值。
- `tools/fusionvad_ja/evaluate_vad_baselines.py`：在同一 label/manifest 上评估现有 `fusion_lite` 与 `whisperseg-adaptive`，用于和训练型头对比。
- `tools/fusionvad_ja/materialize_hf_audio.py`：物化 Hugging Face 音频；默认使用 torchcodec decode，再写本地 16k mono WAV；失败时 fallback 到 `Audio(decode=False)` bytes 路径。
- `tools/fusionvad_ja/slice_labeled_audio.py`：把长 supervised 音频按 frame label 切成短 clip，并同步裁剪 `teacher_segments`。
- `tools/fusionvad_ja/build_local_video_audit_candidates.py`：从本地视频确定性抽取 16k mono WAV 短片段，生成 manual audit candidates / manifest，用于真实近域 held-out 人工标注。

当前可见结论：

- FusionVAD-JA 当前作为 high-recall proposal generator 使用，重点是不漏对白、呻吟、喘息、短促人声；precision 和 hard-negative 过滤留给后续 ASR/aligner 失败样本闭环。
- `ASR_VAD_BACKEND=fusionvad_ja` 已成为当前默认；旧 `fusion_lite` / `whisperseg-adaptive` 保留为显式 baseline。
- Qwen3-ASR-1.7B full SFT 仍是目标域 ASR 主线；Qwen3-ASR-0.6B full SFT 作为 FusionVAD-JA frozen feature extractor 和后续轻量 ASR probe。
- Forced aligner 先不 finetune。当前优先级是把 ASR 文本拆成 `display_text` / `align_text` 两层、记录 fallback 质量标签、汇总真实失败样本池。
- README / 测试 fixture / 可跟踪报告一律使用匿名样片名，不写真实视频 stem 或含真实 stem 的 `agents/temp/` 路径。

v1.9 ASR / forced alignment 文本策略：

- `display_text` 是最终字幕显示文本，只做展示安全处理：Unicode NFKC、换行归一为空格、连续空白折叠和首尾 trim。不得在 `display_text` 上压缩重复假名、重复短语、拟声或低信息短文本，因为这些在目标域里可能是字幕语义。
- `align_text` 是 forced aligner 专用文本，可以删除标点、emoji / 装饰符、音乐符号和明显不可发音标记，也可以压缩极端重复假名、长音符和重复短语；这些操作必须记录 flags，并保留从 `align_text` 字符到 `display_text` 覆盖范围的映射。
- 不使用按具体字样维护的黑名单，不直接删除 `ん`、`あ`、喘息/呻吟拟声、常见台词、历史工具签名或纯英文长词。ASR 后处理已删除噪声词表、灰区词表、假名/呻吟特例 direct drop、工具签名 direct drop、AnimeWhisper 后置括号/重复清洗、最终字幕文本重复压缩和翻译前纯英文幻觉 direct drop；当前只因空文本、纯标点/纯符号和上下文泄漏这类明确非字幕内容而删除。
- 翻译 prompt 的源文序列化同样不再压缩重复发声，也不使用固定拟声词映射表；重复循环只作为 QC / 诊断信号，译文是否概括交给 LLM 在上下文中判断。
- speaker diarization 不再把假名-only 文本当作 BGM 跳过；只跳过空文本或纯符号/纯标点这类没有语言/数字信号的片段，避免把目标域可字幕化人声排除在 speaker embedding 之外。
- 重复循环、低置信、文本/音频比例异常、align-text-empty、forced-aligner fallback、ASR dropped uncertain 和人工 hard-negative 结果默认只作为 QC / 诊断 / 样本池信号；`ASR_QC_DROP_UNCERTAIN=0` 是默认值，是否删除交给后续可解释 QC 策略，不再用词表兜底。
- forced aligner 失败时不伪造精确时间轴。后续输出质量标签分为 `forced`、`partial`、`vad_coarse`、`proportional`、`drop_or_review`；失败样本进入 VAD / ASR / aligner 后处理样本池。
- 实现口径：`src/whisper/prealign.py` 负责 `raw_text -> display_text -> align_text` 和 char-span mapping；`src/whisper/local_backend.py` 只把 `align_text` 送入 forced aligner，拿到词级时间后再映射回 `display_text`。

<details>
<summary>历史实验记录 v1-mini 至 v1.6</summary>

v1-mini full 结果：

- split 仍然是 VoxConverse supervised speech spans + MUSAN music/noise real-negative，train/val/test 分别为 302/252/98 条，其中 train 为 151 supervised + 151 negative，val 为 126 + 126，test 为 49 + 49；feature cache 使用 `whisper-ja-1.5b` CUDA half precision，`frame_hop_s=0.02`、`n_mfcc=40`，三组 cache 均 `errors=0`、`skipped=0`。
- 训练命令使用 cached features，不回传 frozen encoder：`max_steps=2048`、`batch_size=8`、`learning_rate=0.001`、`device=cuda`；checkpoint 写入 `datasets/train/fusionvad-ja/v1-mini/addition-bilstm-2048-batch8/fusionvad_ja_addition_bilstm.pt`。
- 训练指标：loss `0.0640`、frame accuracy `0.9770`、positive ratio `0.4731`，可训练参数 `1,942,145`。
- validation 阈值 0.5：frame accuracy `0.9878`、precision `0.9811`、recall `0.9934`、F1 `0.9872`、positive ratio `0.4753`、predicted positive ratio `0.4812`。
- validation 阈值扫描最佳为 `0.6`：F1 `0.9875`、precision `0.9836`、recall `0.9915`、predicted positive ratio `0.4791`。
- test 阈值 0.5：frame accuracy `0.9901`、precision `0.9834`、recall `0.9960`、F1 `0.9897`、positive ratio `0.4754`、predicted positive ratio `0.4816`。
- test 阈值 0.6：frame accuracy `0.9907`、precision `0.9855`、recall `0.9952`、F1 `0.9903`、predicted positive ratio `0.4801`。
- 同集 baseline：`whisperseg-adaptive` 98/98 成功，WhisperSeg 主模型走 `CUDAExecutionProvider`，test F1 `0.9163`、precision `0.8479`、recall `0.9966`、predicted positive ratio `0.5588`；`fusion_lite` 98/98 成功，WhisperSeg 主信号走 `CUDAExecutionProvider`，test F1 `0.9246`、precision `0.8624`、recall `0.9966`、predicted positive ratio `0.5494`。
- 结论：addition-fusion BiLSTM 在这个 v1-mini split 上明显提高 precision，并保持接近 baseline 的 recall；但该 split 仍主要覆盖 VoxConverse 英语/多人 speech span 与 MUSAN 音乐负样本，不代表 JAV/galgame 泛化结论，也不能替换默认 VAD。下一步应加入 Galgame / 本地 JAV near-domain teacher labels、DNS/noise 负样本，并按来源隔离扩大 held-out。

v1.1 Galgame weak-positive 混合结果：

- 数据准备：`litagin/Galgame_Speech_ASR_16kHz` 使用 Hugging Face streaming shuffle，`shuffle_buffer_size=4096`、`shuffle_seed=20260524`；物化 train/val/test 为 512/128/128 条本地 16k mono WAV，全部经 torchcodec decode，错误 0，总时长分别约 `2734.0s`、`623.0s`、`673.2s`。Galgame 只有文本和整段音频，没有 VAD 时间戳，因此本轮用 `trim_head_s=0.04`、`trim_tail_s=0.04` 生成 weak-positive `teacher_agree` 标签，speech ratio 约 `0.984-0.985`。
- 训练集：v1-mini train 302 条加 Galgame train 512 条，得到 814 条 mixed train，其中 `supervised=151`、`negative=151`、`teacher_agree=512`，总时长 `8702.9s`；feature cache 使用 `whisper-ja-1.5b` CUDA half precision，train/val/test cache 分别为 814/128/128 条，`errors=0`、`skipped=0`。
- 训练命令使用 cached features，不回传 frozen encoder：`max_steps=2048`、`batch_size=16`、`learning_rate=0.001`、`device=cuda`；checkpoint 写入 `datasets/train/fusionvad-ja/v1-1/addition-bilstm-2048-batch16/fusionvad_ja_addition_bilstm.pt`。
- 训练指标：loss `0.0534`、frame accuracy `0.9806`、positive ratio `0.6342`，可训练参数 `1,942,145`。batch size 16 在 RTX 4060 Ti 8GB 上可跑；Codex sandbox 中 CUDA 可能不可见，feature cache / training 长跑需提权执行。
- 阈值校准仍使用 v1-mini supervised+negative validation，不使用 Galgame weak-positive validation 选阈值；最佳阈值为 `0.25`，validation F1 `0.9773`、precision `0.9723`、recall `0.9823`、positive ratio `0.4753`、predicted positive ratio `0.4802`。
- v1-mini supervised+negative test：阈值 `0.25` 时 F1 `0.9842`、precision `0.9818`、recall `0.9867`、predicted positive ratio `0.4778`；阈值 `0.5` 时 F1 `0.9770`、precision `0.9868`、recall `0.9675`、predicted positive ratio `0.4662`。相比 v1-mini-only checkpoint 的 test F1 `0.9903` 有回退，但仍明显高于同集 `fusion_lite` F1 `0.9246` 和 `whisperseg-adaptive` F1 `0.9163`。
- Galgame weak-positive held-out test 只作为近域 recall smoke，不是真实 VAD 泛化指标：v1.1 阈值 `0.25` 时 F1 `0.9137`、precision `0.9845`、recall `0.8524`、predicted positive ratio `0.8527`；阈值 `0.5` 时 F1 `0.8458`、recall `0.7380`。同一弱标签下 `whisperseg-adaptive` F1 `0.9617`、recall `0.9396`，`fusion_lite` F1 `0.9540`、recall `0.9244`，二者的 WhisperSeg 主路径日志均确认 `CUDAExecutionProvider`。
- 结论：直接混入整段 Galgame weak-positive 会把训练集正帧比例推高，但模型仍会主动切掉 Galgame clip 内部低语音置信区域；这对真实 VAD 可能是合理行为，却会被 weak-positive 标签当作 false negative。下一步不应简单扩大 full-clip weak-positive，而应引入 teacher segment pseudo-label、teacher conflict 审计、DNS/本地非语音负样本，以及少量人工/强监督的 JAV/galgame held-out，再重新比较 v1.1 与默认 `fusion_lite`。

v1.2 teacher-student 伪标签计划：

- 目标：把 Galgame / JAV 近域音频从 full-clip weak-positive 改成 teacher segment pseudo-label，用作 domain adaptation，而不是把现有 VAD teacher 当成最终评价标准。最终是否超过 `fusion_lite` / `whisperseg-adaptive` 必须依赖独立强标注或人工审计 held-out。
- Teacher：首轮使用 `whisperseg-adaptive` 与 `fusion_lite`。二者不是训练目标本身，只作为噪声标注器；默认只采纳重叠一致的高置信 speech span。`teacher_conflict` 和边界不确定区域写入审计文件，默认不进训练。
- 标签策略：新增可选 `frame_weights` / ignore mask。`speech_frames=1, frame_weights>0` 表示高置信 speech；`speech_frames=0, frame_weights>0` 表示高置信 non-speech；`frame_weights=0` 表示 ignore，不参与 BCE。老 JSONL 无 `frame_weights` 时按全 1 兼容。
- 默认伪标签规则：两个 teacher 都判 speech 的帧为 positive，两个 teacher 都不判 speech 且形成足够长 gap 的区域为 negative，任一 teacher 单独判 speech、positive 边界附近、短间隙都设为 ignore。`teacher_agree` 训练权重低于 supervised / real-negative，首轮建议 `0.3-0.6`。
- 训练策略：优先从 v1-mini checkpoint 低学习率 fine-tune，而不是从头混入 Galgame；v1-mini supervised+negative validation 继续作为阈值校准集，Galgame weak split 只作为 recall smoke。
- 验收顺序：
  1. 实现 `tools/fusionvad_ja/build_teacher_pseudo_labels.py`，输入 materialized manifest，输出 pseudo-label JSONL、summary、per-clip teacher audit 和 conflict audit。
  2. 扩展 `LabelRecord` 与 cached-feature 训练，使 `frame_weights=0` 的帧不参与 loss / accuracy，teacher pseudo-label 可低权重训练。
  3. 用 Galgame train 小样本先跑 64 条 teacher pseudo-label smoke，确认 `teacher_agree`、`teacher_conflict`、ignored frame ratio、positive/negative/weighted frame ratio 可解释。
  4. 构建 v1.2 Galgame pseudo-label feature cache，使用 v1-mini checkpoint fine-tune，并在 v1-mini val/test、Galgame pseudo-label smoke、现有 baseline 上对比。
  5. 再制作少量人工/强审计 JAV/galgame held-out，否则不宣称真实泛化超过 teacher 或默认 VAD。

v1.2 首轮执行记录：

- 已扩展标签 schema：`frame_weights` 为可选字段，旧 label JSONL 不带该字段时按全 1 兼容；`frame_weights=0` 的帧在 addition-fusion train/eval 和 baseline eval 中会被忽略。
- 已新增 `tools/fusionvad_ja/build_teacher_pseudo_labels.py`：输入 materialized manifest，运行 `whisperseg-adaptive` / `fusion_lite`，输出 `teacher_pseudo_labels.jsonl`、`teacher_pseudo_manifest.json`、`teacher_pseudo_audit.jsonl`、`teacher_conflicts.jsonl`、error report 和 summary。
- Galgame train 64 条 smoke：输入 `datasets/train/fusionvad-ja/v1-galgame/galgame-materialized-512/hf_audio_manifest.json`，输出到 `datasets/train/fusionvad-ja/v1-2/teacher-pseudo-galgame-train64/`；WhisperSeg 主路径日志确认 `CUDAExecutionProvider`，records `64`、errors `0`、`teacher_agree=58`、`teacher_conflict=6`。
- 64 条 smoke 的 frame 统计：frames `14208`、active frame ratio `0.8590`、ignored frame ratio `0.1410`、conflict frame ratio `0.0516`、weighted speech frame ratio `0.8427`、weighted negative frame ratio `0.0163`。这说明新策略不再把整段 Galgame 硬标为 speech，而是把 teacher 分歧和边界留给 ignore。
- training manifest dry-run：64 条 pseudo labels 中 58 条 `teacher_agree` 进入训练候选，6 条 `teacher_conflict` 默认过滤，skipped `0`；summary 写入 `datasets/train/fusionvad-ja/v1-2/teacher-pseudo-galgame-train64/training-manifest/training_manifest_summary.json`。
- Galgame train 512 条 teacher pseudo-label 全量生成完成：输出到 `datasets/train/fusionvad-ja/v1-2/teacher-pseudo-galgame-train512/`，日志 `agents/temp/fusionvad-ja-v1-2-teacher-pseudo-galgame-train512.run.log`；WhisperSeg / `fusion_lite` 主路径日志确认 `onnx_provider=CUDAExecutionProvider`，records `512`、errors `0`、`teacher_agree=477`、`teacher_conflict=35`。
- 512 条 frame 统计：frames `136944`、active frame ratio `0.9117`、ignored frame ratio `0.0883`、conflict frame ratio `0.0316`、weighted speech frame ratio `0.8709`、weighted negative frame ratio `0.0408`。`teacher_conflict` 保留到 audit/conflict JSONL，但默认不进训练。
- train512 training manifest 和 feature cache 完成：477 条 `teacher_agree` 可训练样本，skipped `0`；`whisper-ja-1.5b` + MFCC CUDA half precision feature cache `cached=477`、errors `0`，输出 `datasets/train/fusionvad-ja/v1-2/teacher-pseudo-galgame-train512/feature-cache-full/`。
- 已给 cached-feature 训练补 `--init-checkpoint`，并允许 `train_addition_bilstm.py` 重复传入多组 `--labels` / `--feature-manifest`。v1.2 fine-tune 使用 v1-mini checkpoint 初始化，训练混合 v1-mini 强监督/真实负样本 302 条 + Galgame teacher pseudo 477 条，共 779 条 cached examples。
- v1.2 fine-tune 命令口径：`max_steps=1024`、`batch_size=16`、`learning_rate=0.0002`、`device=cuda`、init checkpoint `datasets/train/fusionvad-ja/v1-mini/addition-bilstm-2048-batch8/fusionvad_ja_addition_bilstm.pt`；checkpoint 写入 `datasets/train/fusionvad-ja/v1-2/addition-bilstm-ft-v1mini-mixed779-batch16-lr2e-4-steps1024/fusionvad_ja_addition_bilstm.pt`。
- v1.2 训练指标：loss `0.0302`、frame accuracy `0.9891`、positive ratio `0.5565`、可训练参数 `1,942,145`。Torch CUDA probe 写入 `agents/temp/fusionvad-ja-v1-2-torch-cuda-probe.run.log`，提权环境确认 `torch_cuda_available=True`、设备 `NVIDIA GeForce RTX 4060 Ti`。
- 阈值校准继续使用 v1-mini supervised+negative validation，CUDA 运行，summary `datasets/val/fusionvad-ja/v1-2/addition-bilstm-ft-v1mini-mixed779-batch16-lr2e-4-steps1024-threshold-calibration-cuda/threshold_calibration.json`；最佳阈值 `0.5`，validation F1 `0.9853`、precision `0.9803`、recall `0.9904`、positive ratio `0.4753`、predicted positive ratio `0.4802`。
- v1-mini held-out test 回归：阈值 `0.5` 时 F1 `0.9909`、precision `0.9892`、recall `0.9926`、positive ratio `0.4754`、predicted positive ratio `0.4771`。相对 v1-mini 2048-step checkpoint 的 test F1 `0.9903`、precision `0.9855`、recall `0.9952`，v1.2 主要换来更高 precision，recall 小幅下降。
- Galgame weak-positive held-out 仍只作为近域 smoke，不是真实 VAD 指标：阈值 `0.5` 时 F1 `0.9383`、precision `0.9865`、recall `0.8946`、predicted positive ratio `0.8930`；阈值 `0.25` 时 F1 `0.9530`、precision `0.9851`、recall `0.9230`、predicted positive ratio `0.9228`。该结果说明 v1.2 比 v1.1 更愿意保留 Galgame speech-like 区域，但仍会切掉一部分 full-clip weak 标签中的低置信区域。
- 当前 teacher 组合仍高度相关：`fusion_lite` 以 WhisperSeg 候选为主，和 `whisperseg-adaptive` 不是完全独立 teacher；v1.2 可以作为第一版 domain adaptation，但不能证明已经超过 teacher。下一步应加入更独立的 Silero/TEN/人工审计 held-out，或抽样人工修正 Galgame/JAV 边界后再做真实泛化结论。
- 本轮验收：`.venv/bin/python -m pytest tests/test_fusionvad_ja_dataset.py tests/test_fusion_lite_vad_backend.py tests/test_vad_ab1_ab2.py tests/test_vad_ab3_ab4.py -q` 结果 `82 passed`；`git diff --check` 通过。

v1.3 方向与首轮审计准备：

- Grok / 外部依据：FusionVAD 论文与 ISCA 版本确认“MFCC + PTM feature + simple addition fusion + BiLSTM”是合理主线，addition 通常比 cross-attention 更稳且更省；论文还指出 MFCC 与 PTM 错误类型互补。Teacher-student VAD 论文显示 pseudo-label 可提升 noisy / real-world VAD 泛化，但最终判断必须依赖独立 held-out。TEN VAD / Silero 官方项目均是轻量独立 VAD，可作为 teacher diversity 来源；FireRedVAD 也值得后续评估，但首轮不引入新依赖。
- 方向取舍：暂不继续盲目扩大 `whisperseg-adaptive + fusion_lite` 同源 teacher 数据；优先做两件事：一是引入 research-only 独立 teacher 对比，二是从现有 Galgame pseudo-label 里抽小规模人工审计候选，先建立能判断“是否超过 teacher”的近域 held-out。
- 已新增 research-only backend resolver：`src/vad/fusionvad_ja/research_backends.py` 支持 `whisperseg-adaptive`、`fusion_lite`、`silero`、`ten_vad`、`ten_silero`，仅供 `tools/fusionvad_ja/` 研究脚本使用；`src/vad/__init__.py` 仍只公开 `fusion_lite` 和 `whisperseg-adaptive`，默认 VAD / Web / `.env` 不变。
- 已新增 `tools/fusionvad_ja/select_audit_candidates.py`：读取 `teacher_pseudo_audit.jsonl` 和 labels，按 `teacher_conflict_high`、`text_but_low_active`、`ignored_ratio_high`、`negative_gap_high`、`clean_teacher_agree`、`long_clip` 六类抽样，输出人工审计候选 JSONL/CSV。
- Galgame train512 审计候选已生成：`datasets/train/fusionvad-ja/v1-3/audit-candidates-galgame-train512/`，共 `72` 条，每类 `12` 条；其中 `teacher_conflict=35`、`teacher_agree=37`。这批候选适合优先人工标注 speech boundary，用作近域 validation/test 或 teacher 校准集。
- 已新增 `tools/fusionvad_ja/generate_manual_audit_html.py`：把候选 JSONL/CSV 生成 standalone HTML 标注页，可听音频、标 speech segments、用 teacher union/intersection 初始化、浏览器本地缓存进度，并导出/保存 `manual_labels.jsonl`。不接入主 Web，不修改 `src/web/static/`。
- Galgame v1.3 人工标注页已生成：`datasets/val/fusionvad-ja/v1-3/manual-audit-galgame/manual_audit.html`，同目录 `audio/` 已复制 72 条候选音频，便于浏览器本地播放。页面 UI 为中文；打开 HTML 后标注，最终保存/下载 `manual_labels.jsonl`；后续工具会把其中 `speech_segments` 转成 frame labels。人工口径暂定为：Galgame 片段通常已经按对白裁切，若整段基本都是可辨识词句/音节的日语对白，优先标为全段 speech；背景音乐、底噪或环境声垫在对白下面时仍按对白区间标为 speech，不必扣掉；纯呻吟、喘息、笑声、哭声、尖叫等非语言人声标为 non-speech，或在不确定/不纳入训练时填 `skip_reason=moan_only` / `human_nonverbal` 跳过；纯 BGM / 无对白样本可填 `skip_reason=pure_bgm` / `no_dialogue`；对白夹杂非语言人声时只圈对白片段。
- 已新增 `tools/fusionvad_ja/convert_manual_audit_labels.py`，把浏览器导出的 `manual_labels.jsonl` 转成标准 `labels.jsonl` + `manifest.json`。本轮 72 条人工审计全部已审，转换后 `supervised=36`、`negative=36`、skipped `0`，总时长 `560.14s`，人工 speech frame ratio `0.6973`；输出在 `datasets/val/fusionvad-ja/v1-3/manual-audit-galgame/strong-labels/`。
- 人工 held-out feature cache 已完成：`whisper-ja-1.5b` CUDA half precision，cached `72`、errors `0`、skipped `0`，输出 `datasets/val/fusionvad-ja/v1-3/manual-audit-galgame/feature-cache-strong/`。普通 sandbox 后台 job 可能让 Torch/ORT 初始化异常落 CPU；feature cache 使用前台 CUDA 跑通。
- 人工 held-out 首轮对比：v1-mini threshold 0.5 F1 `0.6516`、precision `0.9209`、recall `0.5041`；v1.2 threshold 0.5 F1 `0.8262`、precision `0.8923`、recall `0.7693`；v1.2 在该人工集上阈值扫描最佳 threshold `0.15`，F1 `0.8471`、precision `0.8307`、recall `0.8643`。提权 CUDA baseline 输出 `datasets/val/fusionvad-ja/v1-3/manual-audit-galgame/baseline-vads-cuda/`：`fusion_lite` F1 `0.8862`、precision `0.9236`、recall `0.8518`；`whisperseg-adaptive` F1 `0.8548`、precision `0.8219`、recall `0.8906`；`ten_vad` F1 `0.7802`、precision `0.7085`、recall `0.8681`；`silero` F1 `0.6163`、precision `0.9805`、recall `0.4494`。结论：v1.2 已明显优于 v1-mini，但还没有超过当前 `fusion_lite`；v1.3 训练应继续提高 Galgame/JAV 近域 recall，同时控制 TEN 带来的 false positive。
- `whisperseg-adaptive + silero` 32 条 smoke：输出 `datasets/train/fusionvad-ja/v1-3/teacher-pseudo-galgame-train32-whisperseg-silero/`，errors `0`、`teacher_agree=27`、`teacher_conflict=5`、active frame ratio `0.6008`、ignored frame ratio `0.3992`、conflict frame ratio `0.3485`。Silero 与 WhisperSeg 分歧很大，首轮不适合 strict 2/2 agreement 直接扩大训练，更适合作为 hard audit / uncertain sampler。
- `ten_vad` direct 8 条 smoke：输出 `datasets/train/fusionvad-ja/v1-3/teacher-pseudo-galgame-train8-ten-vad/`，errors `0`、`teacher_agree=8`、active frame ratio `0.8687`。TEN 本机可用，可作为低成本独立 teacher。
- `whisperseg-adaptive + ten_vad` 32 条 smoke：输出 `datasets/train/fusionvad-ja/v1-3/teacher-pseudo-galgame-train32-whisperseg-ten/`，WhisperSeg 主路径日志确认 `CUDAExecutionProvider`，errors `0`、`teacher_agree=32`、conflict frame ratio `0.0943`、active frame ratio `0.8490`、weighted speech frame ratio `0.8331`、weighted negative frame ratio `0.0159`。相比 Silero，TEN 更适合作为 v1.3 strict-agreement teacher 候选。
- v1.3 建议执行顺序：先人工审计 40-80 条候选，形成小型 Galgame/JAV 强标注 held-out；再用 `whisperseg-adaptive + ten_vad` 生成 train512/train1k pseudo-label，保留 Silero 分歧样本为 ignore 或审计集；最后在 v1-mini strong test、人工近域 held-out、MUSAN/DNS/local negative 三类集合上同时比较 v1.2/v1.3、`fusion_lite`、`whisperseg-adaptive`、TEN/Silero baseline。
- v1.3 `whisperseg-adaptive + ten_vad` train512 pseudo-label 已完成：输出 `datasets/train/fusionvad-ja/v1-3/teacher-pseudo-galgame-train512-whisperseg-ten/`，records `512`、errors `0`、`teacher_agree=507`、`teacher_conflict=5`；frames `136944`、active frame ratio `0.7977`、ignored frame ratio `0.2023`、conflict frame ratio `0.1392`、weighted speech frame ratio `0.7889`、weighted negative frame ratio `0.0088`。注意 `teacher_conflicts.jsonl` 记录局部分歧帧，行数会大于最终 `teacher_conflict` 样本数。
- v1.3 training manifest 和 feature cache 已完成：507 条 `teacher_agree` 可训练样本，skipped `0`；`whisper-ja-1.5b` + MFCC CUDA half precision feature cache `cached=507`、errors `0`，输出 `datasets/train/fusionvad-ja/v1-3/teacher-pseudo-galgame-train512-whisperseg-ten/feature-cache-full/`。
- v1.3 fine-tune 使用 v1-mini checkpoint 初始化，训练混合 v1-mini 强监督/真实负样本 302 条 + Galgame `whisperseg-adaptive + ten_vad` strict-agreement pseudo 507 条，共 809 条 cached examples。命令口径：`max_steps=1024`、`batch_size=16`、`learning_rate=0.0002`、`device=cuda`、init checkpoint `datasets/train/fusionvad-ja/v1-mini/addition-bilstm-2048-batch8/fusionvad_ja_addition_bilstm.pt`；checkpoint 写入 `datasets/train/fusionvad-ja/v1-3/addition-bilstm-ft-v1mini-whisperseg-ten809-batch16-lr2e-4-steps1024/fusionvad_ja_addition_bilstm.pt`。
- v1.3 训练指标：loss `0.0208`、frame accuracy `0.9926`、positive ratio `0.5524`、可训练参数 `1,942,145`。`nvidia-smi` 训练中确认 Torch 进程走 GPU，显存约 `2.3GB`。
- v1.3 人工 Galgame held-out 结果不达预期：threshold `0.5` 时 F1 `0.7926`、precision `0.7914`、recall `0.7938`、predicted positive ratio `0.6994`；人工集阈值扫描最佳 threshold `0.05`，F1 `0.8306`、precision `0.7567`、recall `0.9205`。这低于 v1.2 人工最佳 F1 `0.8471`，也低于 `fusion_lite` baseline F1 `0.8862`。
- v1.3 v1-mini strong test 回归通过：threshold `0.5` 时 F1 `0.9912`、precision `0.9916`、recall `0.9908`、positive ratio `0.4754`、predicted positive ratio `0.4751`，略高于 v1.2 test F1 `0.9909`。v1-mini validation threshold `0.5` 为 F1 `0.9871`、precision `0.9884`、recall `0.9858`。
- v1.3 Galgame weak-positive held-out 仍只作为近域 smoke，不是真实 VAD 指标：threshold `0.5` 时 F1 `0.9282`、precision `0.9885`、recall `0.8749`；threshold `0.25` 时 F1 `0.9444`、precision `0.9849`、recall `0.9071`；threshold `0.05` 时 F1 `0.9646`、precision `0.9844`、recall `0.9456`。低阈值能保留更多 Galgame weak speech-like 区域，但人工强标注显示 false positive 风险同步上升。
- v1.3 结论：`whisperseg-adaptive + ten_vad` strict agreement 不是当前可推广的提升路线。TEN 提高了近域 recall 倾向，但伪标签负帧过少、冲突帧比例更高，训练后在人工 held-out 上 precision 明显下降。该 checkpoint 只作为研究产物保留，不替换默认 VAD，也不作为当前最佳 FusionVAD-JA 候选。
- 下一步计划：优先把人工 held-out 扩到至少 200-300 条，并覆盖 Galgame/JAV 对白、纯呻吟/喘息、BGM、人声非语言和真实静音；训练侧回到高 precision teacher 或人工强标注混合，考虑 `fusion_lite` / `whisperseg` 作为主 teacher，TEN/Silero 只用于 hard-negative/uncertain sampler；同时补充 MUSAN/DNS/local video negative 与 human-nonverbal negative，减少“全段都是 speech”的伪标签偏置；所有候选继续在 v1-mini strong test、人工近域 held-out、Galgame weak smoke 三类集合上并排比较。
- 本轮验收：`.venv/bin/python -m pytest tests/test_fusionvad_ja_dataset.py -q` 结果 `55 passed`；`.venv/bin/python -m pytest tests/test_fusion_lite_vad_backend.py tests/test_vad_ab1_ab2.py tests/test_vad_ab3_ab4.py -q` 结果 `31 passed`；`git diff --check` 通过。

v1.4 Qwen3-ASR / high-recall VAD 计划：

- 目标修正：如果主 ASR 计划转向 finetune `Qwen3-ASR-1.7B`，VAD 不再优先追最高 frame F1 或最高 precision，而应定位为 high-recall proposal generator。工程目标是“不漏对白”，允许多送一部分非语音给 ASR，再由 ASR / forced aligner / 字幕文本约束做后处理。
- 本轮执行决策：本地后端 ASR 暂时继续使用 `Qwen3-ASR-1.7B`，不把 `Qwen3-ASR-0.6B` 注册成主 ASR 后端；`Qwen3-ASR-0.6B` 只作为 FusionVAD-JA 的 frozen PTM feature extractor 和低成本 ASR probe。1.7B 的 finetune 放到租用 GPU 服务器，本机只负责数据整理、评测、feature cache、轻量 VAD 头训练和对比。
- 评估指标同步调整：除 precision / recall / F1 外，必须新增 `missed_speech_seconds`、`missed_speech_segments`、`extra_audio_ratio`、segment padding 后的 recall，以及 downstream ASR CER/WER。候选阈值选择先以 recall 约束为主，例如人工 near-domain held-out recall 目标先设 `>=0.95`，再限制 extra audio ratio。
- Qwen3-ASR 外部依据：官方 README / model card 显示 `Qwen3-ASR-1.7B` 和 `Qwen3-ASR-0.6B` 均支持日语，并提供 transformers / vLLM 推理、forced aligner 和 JSONL audio-text fine-tuning；技术报告称 1.7B 是开源 SOTA 取向，0.6B 是 accuracy-efficiency trade-off，0.6B 在高并发下有很高吞吐。该定位与“0.6B 轻量特征提取 + 1.7B 精修 ASR”的分工一致。
- Qwen3-ASR-0.6B 决策：值得作为下一轮重点 probe，但先不 finetune。先跑两条实验线：一是 direct ASR，对比 0.6B / 1.7B / 当前 ASR 在 Galgame/JAV 样本上的 CER、漏字、幻觉、速度和显存；二是 frozen feature VAD，把当前 `whisper-ja-1.5b` feature extractor 替换为 Qwen3-ASR-0.6B 的 frozen audio features，保持 MFCC + addition-fusion BiLSTM 思路不变，比较 recall、missed speech 和 extra audio ratio。
- Qwen3-ASR feature cache 口径：`tools/fusionvad_ja/build_feature_cache.py --ptm qwen3-asr-0.6b` 读取 `Qwen/Qwen3-ASR-0.6B` 或本地 `models/Qwen-Qwen3-ASR-0.6B`，调用 `model.thinker.get_audio_features()` 只取 audio tower 输出，不跑文本生成；Qwen audio token 帧率低于 20ms MFCC，cache 阶段线性上采样到 MFCC 帧数后继续写入兼容的 `whisper` feature key。本地已下载 `models/Qwen-Qwen3-ASR-0.6B`，目录约 `1.8G`，`config.json` 显示 0.6B audio feature `output_dim=1024`。
- Qwen3-ASR VAD 头部参数：0.6B audio feature 维度为 `1024`，可沿用默认 `fusion_dim=256`、`hidden_dim=192` 并保持可训练参数小于 2M；若后续切到 1.7B audio feature 等更大维度，再改用 `fusion_dim=160`、`hidden_dim=160`。由于输入特征分布不同，不建议从 `whisper-ja-1.5b` feature checkpoint 直接 `--init-checkpoint` fine-tune，Qwen feature 线首轮从头训练更清晰。
- Qwen3-ASR-0.6B finetune 触发条件：只有当 0.6B direct ASR 已接近 1.7B 且速度/显存优势明显，或 0.6B frozen feature VAD 明显优于 Whisper 特征，或 1.7B 训练/推理成本不可接受时，再考虑 0.6B LoRA / full SFT。否则 0.6B 优先作为轻量 frozen feature extractor 和低成本 ASR baseline。
- Qwen3-ASR-1.7B 显存判断：RTX 4060 Ti 8GB 对 1.7B 推理大概率可做小 batch / bf16 或 fp16，官方示例也允许通过较小 `max_inference_batch_size` 避免 OOM；但 8GB 不适合直接跑官方 full SFT 口径。原因是 full fine-tuning 除模型权重外还需要梯度、Adam optimizer state 和音频/文本 activation，显存远高于推理。官方 finetuning 示例默认 `batch_size=32`、`grad_acc=4`，未提供 LoRA / QLoRA 参数，这个默认口径不能直接套到 8GB 单卡。
- 1.7B 可尝试路线：先只做 inference / eval；若要本机训练，优先自建 LoRA / QLoRA 或 decoder-only LoRA 方案，micro-batch 从 `1` 开始，短音频切片，开启 gradient checkpointing / FlashAttention 2，必要时冻结 audio encoder 或只训练文本侧/少量 adapter。即便如此，8GB 仍属高风险实验，必须先用 10-50 条 Galgame 小样本 smoke 测峰值显存，再决定是否扩大；full SFT 建议放到更大显存机器或云 GPU。
- v1.4 执行顺序：先实现 Qwen3-ASR direct-ASR eval JSONL 和人工 held-out CER 计算；再实现 Qwen3-ASR-0.6B frozen feature cache adapter；随后用同一 v1-mini strong test、人工 Galgame held-out、Galgame weak smoke 比较 `whisper-ja-1.5b feature` 与 `qwen3-asr-0.6b feature`；阈值选择按 recall 优先，目标先看人工 near-domain recall `>=0.95` 时的 `missed_speech_seconds` 和 `extra_audio_ratio`；最后再决定是否做 1.7B LoRA smoke。
- v1.4 已新增 research-only 工具：`tools/fusionvad_ja/export_qwen_asr_sft.py` 可把 Galgame manifest 导出成 Qwen SFT JSONL 包；`tools/fusionvad_ja/probe_qwen_asr.py` 可用本地 `Qwen3-ASR-1.7B` 跑 direct ASR probe，并输出 CER/RTF；`tools/fusionvad_ja/export_addition_predictions.py` 可从 addition-BiLSTM checkpoint 导出 frame prediction / probability JSONL；`tools/fusionvad_ja/vad_recall_metrics.py` 可从 frame prediction JSONL 计算 high-recall 指标与 padding trade-off；`tools/fusionvad_ja/build_galgame_synthetic_timeline.py` 可把已裁切 Galgame speech clip 与确定长度 silence / white-noise / hum gap 拼成精确时间轴 supervised VAD 样本。以上均不接入默认 VAD / Web。
- v1.4 Qwen feature v1-mini-only 首轮：`Qwen3-ASR-0.6B` feature cache 已完成 v1-mini train/val/test/manual Galgame，cached 分别 `302/252/98/72`、errors `0`，feature dim `1024`、MFCC dim `40`。从头训练 `addition-bilstm-v1mini-2048-batch16`，可训练参数 `1,876,609`；v1-mini validation threshold `0.5` F1 `0.9861`、precision `0.9817`、recall `0.9905`，test F1 `0.9869`、precision `0.9785`、recall `0.9954`。但人工 Galgame threshold `0.5` F1 只有 `0.4444`、precision `0.9839`、recall `0.2870`，说明该头对 Galgame 近域 speech 概率极保守，不能只靠常规阈值迁移。
- Galgame synthetic timeline 方案：由于 `litagin/Galgame_Speech_ASR_16kHz` 多数音频本身已按语音裁切，可以把 Galgame clip 当作 speech island，再拼接确定长度的 silence / low-noise / hum island 形成强时间轴 supervised 样本。v1.4 已生成 train/val/test `256/64/64` 条，speech frame ratio `0.8635/0.8584/0.8672`，training manifest skipped `0`；Qwen feature cache cached `256/64/64`、errors `0`。这类样本适合教模型边界和 gap，不等同真实 JAV 泛化集。
- Qwen synthetic fine-tune：用 Qwen v1-mini checkpoint 初始化，混合 v1-mini strong/negative `302` 条 + Galgame synthetic timeline `256` 条，共 `558` 条，`max_steps=1024`、`batch_size=16`、`learning_rate=0.0002`，checkpoint `datasets/train/fusionvad-ja/v1-4/qwen3-asr-0.6b/addition-bilstm-ft-v1mini-galgame-synth558-batch16-lr2e-4-steps1024/`。训练 loss `0.0499`、frame accuracy `0.9818`、positive ratio `0.5688`。v1-mini regression 基本保留：validation threshold `0.5` F1 `0.9880`、precision `0.9879`、recall `0.9882`；test F1 `0.9865`、precision `0.9784`、recall `0.9948`。
- v1.4 synthetic fine-tune 近域结果：synthetic Galgame threshold `0.5` 仍偏保守，validation F1 `0.7508`、precision `0.9891`、recall `0.6050`，test F1 `0.7489`、precision `0.9942`、recall `0.6006`；人工 Galgame threshold `0.5` F1 `0.6918`、precision `0.8654`、recall `0.5761`，较 v1-mini-only 的 recall `0.2870` 明显改善但还不是可用默认阈值。人工 held-out high-recall 口径下，threshold `0.00005` + `0.2s` padding 可达 recall `0.9553`、missed speech `17.48s`、extra audio ratio `1.3811`；threshold `0.00002` + `0.2s` padding 可达 recall `0.9666`、missed speech `13.06s`、extra audio ratio `1.4001`。结论：该 checkpoint 可以作为 high-recall proposal generator 研究候选，但阈值极低且 extra audio 明显，不替换默认 VAD。
- v1.4 下一步：synthetic timeline 应增加更长/更多 gap、MUSAN/DNS/local video negative、human-nonverbal negative，并把人工 Galgame held-out 扩到 `200-300` 条；训练侧可尝试对 Galgame synthetic 样本过采样或 focal/positive-recall loss，但必须同时守住 v1-mini strong test 和人工 non-speech precision。Qwen 0.6B frozen feature 线暂不 finetune 0.6B 本体；1.7B 仍作为本地 ASR 推理/未来云端 SFT 目标。
- v1.5 synthetic timeline v2 实现：`tools/fusionvad_ja/build_galgame_synthetic_timeline.py` 增加 `--negative-manifest`、`--negative-gap-prob`、`--background-manifest`、`--background-mix-prob`、SNR 配置和 `--speech-label-pad-s`。默认不开启这些新选项，旧 v1.4 行为不变。v2 用 MUSAN 真实 negative gap 替代一部分 synthetic gap，并按 5-20dB 随机混背景；`speech_label_pad_s=0.08` 只扩训练标签，不改音频本体，用于高 recall 边界训练。
- v1.5 数据：Galgame synthetic timeline v2 train/val/test 分别为 `256/64/64` 条，`min_speech_s=0.05` 以保留极短近域声音，skipped 均为 `0`。speech frame ratio 为 `0.7966/0.7970/0.8037`，比 v1.4 的约 `0.86` 更接近真实有 gap 的时间轴；real MUSAN negative gap 数为 `533/143/124`，background mix 数为 `139/30/39`。Qwen3-ASR-0.6B feature cache 使用 CUDA bfloat16、batch size 16，cached `256/64/64`、errors `0`。
- v1.5 训练：继续从 Qwen v1-mini-only checkpoint 初始化，训练混合 v1-mini strong/negative `302` 条 + Galgame synthetic v2 `256` 条，共 `558` 条，`max_steps=1024`、`batch_size=16`、`learning_rate=0.0002`。新增 `train_addition_bilstm.py --positive-loss-weight`，默认 `1.0`；本轮训练 BCE baseline 和 `positive_loss_weight=2.0` 两个变体。BCE checkpoint `datasets/train/fusionvad-ja/v1-5/qwen3-asr-0.6b/addition-bilstm-ft-v1mini-galgame-synthv2-bce558-batch16-lr2e-4-steps1024/`，train loss `0.0656`、frame accuracy `0.9750`。posw2 checkpoint `datasets/train/fusionvad-ja/v1-5/qwen3-asr-0.6b/addition-bilstm-ft-v1mini-galgame-synthv2-posw2-558-batch16-lr2e-4-steps1024/`，train loss `0.0963`、frame accuracy `0.9703`。
- v1.5 评估：BCE threshold `0.5` 在 v1-mini test F1 `0.9906`、recall `0.9933`，但 manual Galgame recall 只有 `0.5474`。posw2 threshold `0.5` 在 v1-mini test F1 `0.9892`、recall `0.9960`，manual Galgame F1 `0.7572`、precision `0.8251`、recall `0.6996`，说明正类加权更符合高 recall 目标。v2 synthetic test threshold `0.5` 仍偏保守，posw2 recall `0.4190`；低阈值是当前必要的 proposal 模式。
- v1.5 推荐 operating points：posw2 + threshold `0.001` + pad `0.2s` 在人工 Galgame 上 precision `0.7310`、recall `0.9501`、F1 `0.8263`、missed speech `19.52s`、extra audio ratio `1.2997`。更激进的 threshold `0.0001` + pad `0.2s` 达到 recall `0.9838`、missed speech `6.32s`、extra audio ratio `1.3809`。对比 v1.4 threshold `0.00005` + pad `0.2s` 的 recall `0.9553`、extra audio ratio `1.3811`，v1.5 posw2 在几乎相同 extra audio ratio 下把人工 Galgame recall 提升到 `0.9838`。
- v1.5 结论：当前最佳 FusionVAD-JA 候选是 Qwen3-ASR-0.6B frozen feature + MFCC addition BiLSTM posw2 checkpoint，使用低阈值 + padding 作为高 recall proposal generator；仍不替换默认 `fusion_lite`，也不接入 Web。下一步应补本地 JAV/manual negative、human-nonverbal negative、BGM/长静音 held-out，并用 Qwen3-ASR-1.7B downstream ASR 评估“多切一点”是否真的提升字幕召回且不显著增加幻觉。
- v1.6 real-heldout 启动：已从 `video/` 顶层 10 个本地视频各抽 8 条、每条 8s 的 16k mono WAV，避开片头/片尾 60s，共 80 条，输出 `datasets/val/fusionvad-ja/v1-6/real-heldout-local-video-audit-80/`。候选文件 `audit_candidates.jsonl`，manifest `manifest.json`，中文人工标注页面 `manual_audit.html`，浏览器导出文件名固定为 `manual_labels.jsonl`。本轮先不训练，先把真实本地 held-out 做出来。
- v1.6 标注口径：可转写日语对白、呻吟声/喘息声如果希望 Qwen3-ASR-1.7B 后续尝试识别或保留上下文，就标为 speech；纯 BGM、机械/环境噪声、静音、无法作为 ASR 输入的非语音人声标为 non-speech；不确定样本可先跳过或标注备注，避免污染强标签。
- v1.6 人工审计工具更新：`tools/fusionvad_ja/generate_manual_audit_html.py` 的主流程改为四类快速标注优先：全段语音、非语音、开头到当前点、当前点到末尾；多段表格仍保留给少数中间有长空洞或多段对白的样本。网页里只设起点会默认生成 `start -> duration`，只设终点会默认生成 `0 -> end`；`tools/fusionvad_ja/convert_manual_audit_labels.py` 也支持单侧边界 JSONL，保证导出和转换语义一致。
- v1.6 人工标注已完成：浏览器导出 `manual_labels.jsonl` 为 `79` 条，候选原始数为 `80` 条；本轮直接用已审 `79` 条继续评估。转强标签输出 `datasets/val/fusionvad-ja/v1-6/real-heldout-local-video-audit-80/strong-labels/`，records `79`、skipped `0`、`supervised=55`、`negative=24`，总时长 `632.0s`，人工 speech frame ratio `0.6138`。
- v1.6 Qwen3-ASR-0.6B feature cache 已完成：输出 `datasets/val/fusionvad-ja/v1-6/qwen3-asr-0.6b/real-heldout-local-video-audit-80-feature-cache/`，配置为 `device=cuda`、`dtype=bfloat16`、batch size `16`，cached `79`、errors `0`、skipped `0`。
- v1.6 现有 VAD baseline：输出 `datasets/val/fusionvad-ja/v1-6/real-heldout-local-video-audit-80/baseline-vads-cuda/`，`fusion_lite` 79/79 成功，F1 `0.7969`、precision `0.8534`、recall `0.7475`、predicted positive ratio `0.5377`；`whisperseg-adaptive` 79/79 成功，F1 `0.7697`、precision `0.7404`、recall `0.8015`、predicted positive ratio `0.6645`。WhisperSeg 日志确认 `CUDAExecutionProvider`。
- v1.6 v1.5 posw2 raw frame 对比：threshold `0` 等价于全帧 speech，precision `0.6141`、recall `1.0000`、F1 `0.7609`、predicted positive ratio `1.0000`，只能作为“不做 VAD 裁剪”的上限对照；threshold `0.0001` raw F1 `0.8046`、precision `0.7023`、recall `0.9418`；threshold `0.001` raw F1 `0.7975`、precision `0.7808`、recall `0.8150`。
- v1.6 high-recall padding 对比：统一 `pad=0.2s` 后，threshold `0.0001` precision `0.6825`、recall `0.9631`、F1 `0.7988`、missed speech `14.32s`、extra audio ratio `1.4112`；threshold `0.00015` precision `0.6941`、recall `0.9551`、F1 `0.8039`、missed speech `17.40s`、extra audio ratio `1.3761`；threshold `0.0002` precision `0.7033`、recall `0.9482`、F1 `0.8076`、extra audio ratio `1.3484`；threshold `0.001` recall 只有 `0.8729`。threshold `0` + pad 仍是全帧 speech，extra audio ratio `1.6283`。
- v1.6 结论：当前本地 real-heldout 推荐 operating point 暂定为 v1.5 posw2 + threshold `0.00015` + pad `0.2s`，它刚好满足 recall `>=0.95`，extra audio ratio `1.3761` 接近 manual Galgame 目标上沿。threshold `0.0001` 更稳召回但额外音频偏多；threshold `0.0002` 额外音频更低但已经略低于 recall 目标。下一步可以进入 Qwen3-ASR-1.7B downstream ASR 对比，重点验证额外音频是否导致幻觉；如果 false positive 主要来自 BGM/呻吟/非语音人声，下一轮优先补 hard negative / human-nonverbal negative，而不是继续扩大 Galgame full-speech 数据。
- v1.6 AnimeWhisper 下游 ASR proxy 对比已完成：新增 `tools/fusionvad_ja/evaluate_vad_asr_downstream.py`，在同一 79 条 real-heldout 强标签上比较 `whisperseg-adaptive`、`fusion_lite`、FusionVAD-JA v1.5 posw2 operating point，并把 VAD 切片送入 `anime-whisper`。输出目录为 `datasets/val/fusionvad-ja/v1-6/real-heldout-local-video-audit-80/downstream-asr-anime-whisper-vad-compare/`，总表为 `downstream_asr_summary.json`。该集没有人工参考转写，因此结果只能作为“漏语音/额外音频/幻觉倾向”代理指标，不是 CER/WER。
- v1.6 下游结果：`fusion_lite` recall `0.7474`、precision `0.8534`、F1 `0.7969`、missed speech `98.00s`、extra audio ratio `0.8758`、ASR chunks `80`、predicted audio `339.76s`、negative-record text chars `41`、no-overlap text chars `66`；`whisperseg-adaptive` recall `0.8014`、precision `0.7403`、F1 `0.7696`、missed speech `77.06s`、extra audio ratio `1.0824`、chunks `112`、predicted audio `419.91s`、negative chars `75`、no-overlap chars `107`；`fusionvad` recall `0.9551`、precision `0.6941`、F1 `0.8039`、missed speech `17.40s`、extra audio ratio `1.3762`、chunks `134`、predicted audio `533.86s`、negative chars `182`、no-overlap chars `201`。
- v1.6 下游结论：FusionVAD-JA 达成高 recall 目标，漏语音秒数相比 `fusion_lite` 明显下降，但会把更多 negative / no-overlap 音频交给 ASR；AnimeWhisper 对所有 VAD 的切片几乎都会产出非空文本，negative/no-overlap 中常见 `ん`、喘息/亲吻拟声等短文本，说明“多切一点”确实增加 hallucination proxy。下一轮优先方向是补 human-nonverbal、BGM、真实 hard negative 训练和/或 ASR 后处理过滤，而不是直接把当前 FusionVAD-JA 晋级为默认 VAD。
- v1.6 策略修正：Galgame 训练集本身包含大量呻吟、喘息、亲吻声等字幕转写，因此 FusionVAD-JA 的正类不应等同传统 speech benchmark 的“清晰词句”。当前研究口径改为：凡是希望送入 ASR 并可能生成字幕的人声、拟声、短促发声都可标为 speech；negative 只指纯 BGM、静音、环境/机械声和无字幕价值残留。所以上述 `negative/no-overlap text chars` 不能自动视为坏样本，必须人工区分“目标域可接受非语言人声”和“真实幻觉”。
- v1.6 hard-negative/目标非语言人声审计：新增 `tools/fusionvad_ja/select_asr_hard_negative_candidates.py`，从 downstream ASR JSONL 中挑出人工 negative 但 ASR 有有效文本、或与人工 speech 无重叠但 ASR 有有效文本的 chunk。已对 `fusionvad` 输出生成 18 条审计候选，输出 `datasets/val/fusionvad-ja/v1-6/real-heldout-local-video-audit-80/asr-hard-negative-audit-fusionvad/`；其中 `manual_negative_asr_text=14`、`no_overlap_asr_text=4`，中文审计页为 `manual_audit.html`。人工标注口径：真实可字幕化呻吟/喘息/拟声标为全段语音；纯 BGM/噪声/静音中的 ASR 文本标为非语音，后续作为 Qwen3-ASR hard-negative 空输出或幻觉过滤样本。
- v1.6 hard-negative 审计流程：人工审计 `datasets/val/fusionvad-ja/v1-6/real-heldout-local-video-audit-80/asr-hard-negative-audit-fusionvad/manual_audit.html` 导出的 `manual_labels.jsonl`；审计完成后先用 `tools/fusionvad_ja/convert_manual_audit_labels.py` 转成强标签并统计 speech/negative 占比；再把标为 speech 的样本作为“目标非语言人声/短促发声”保留给 VAD 正类或后续人工转写，把标为 negative 的样本作为 VAD hard-negative 和 Qwen3-ASR 空输出/幻觉过滤样本。
- v1.6 hard-negative 审计已回收并转换：`manual_labels.jsonl` 已保存并转成 `strong-labels/labels.jsonl` 与 `strong-labels/manifest.json`，records `18`、skipped `0`、`supervised=14`、`negative=4`，manual speech ratio `0.9056`。这说明首批 ASR hard-negative 候选多数其实是目标域可字幕化人声/拟声，不应当一概压成 VAD negative；后续更适合把 14 条 speech 作为目标非语言人声正例保留，把 4 条 negative 作为真实 hard-negative。
- 审计页说明：`Teacher 并集` / `Teacher 交集` 只用于 teacher pseudo-label 审计时快速用 teacher segment 初始化人工边界；ASR hard-negative 审计没有 teacher segments，因此这两个按钮没有意义。`tools/fusionvad_ja/generate_manual_audit_html.py` 已改为仅在候选实际包含 teacher segments 时显示这两个按钮，并把页面标注口径改成“可字幕化人声/拟声/短促发声标为语音，纯 BGM/静音/噪声标为非语音”。
- v1.6 ASR SFT 候选导出：新增 `tools/fusionvad_ja/export_manual_audit_asr_sft_candidates.py`，专门处理人工审计强标签，不信任 ASR 候选文本为正例真值。已从 `strong-labels/manifest.json` 导出到 `asr-sft-candidates/`：`v1-6-fusionvad-audit_empty_hard_negative.jsonl` 含 4 条 `text=""` 空输出 hard-negative，可用于 Qwen3-ASR 幻觉抑制 smoke；`v1-6-fusionvad-audit_speech_review.jsonl/csv` 含 14 条目标非语言人声 review 候选，保留 `candidate_asr_text` 但 `text=""`，必须人工确认转写后才能进 ASR 正样本 SFT。该包位于 `datasets/val/` 下，只作为 held-out 审计产物和后续数据设计依据，直接拿来训练会污染 v1.6 real-heldout。
- 下一轮计划：基于审计结果启动 v1.7。若 18 条里多数是目标域可字幕化人声，保持当前 v1.5 posw2 `threshold=0.00015 + pad=0.2s` 高 recall operating point，并优先推进 ASR finetune / post-filter；若纯 BGM/噪声/静音占比高，则先扩充本地 hard-negative、重新训练 FusionVAD-JA 头并在 v1.6 real-heldout 上复测 recall、missed speech seconds、extra audio ratio 和 AnimeWhisper/Qwen3-ASR downstream proxy。ASR SFT 正样本不能直接使用 AnimeWhisper 原始文本当真值，除非人工在 notes 或后续转写文件中确认。

</details>

- v1.7 本机 VAD operating point 固化：新增 `tools/fusionvad_ja/export_fusionvad_operating_point.py`，把 `export_addition_predictions.py` 与 `vad_recall_metrics.py` 串成一键复现入口；该工具原默认 operating point 为 `fusionvad-ja-v1.5-posw2`、threshold `0.00015`、pad `0.2s`，v1.11 后已同步为 `fusionvad-ja-v1.11-synthv5-longgap-posw2`、threshold `0.02`、pad `0.2s`。`vad_recall_metrics.py` 同步抽出可复用 `compute_recall_metrics()`，并把 `missed_speech_segments` 修正为连续 false-negative frame run 数，避免误读成“有漏帧样本数”。该工具只导出研究评测，不训练模型、不替换默认 VAD、不接入 Web。
- v1.7 real-heldout 复现输出：在 `datasets/val/fusionvad-ja/v1-6/real-heldout-local-video-audit-80/fusionvad-operating-point-v1-5-posw2/` 已生成 `operating_point_summary.json`、`high_recall_metrics.json` 和 `frame-predictions/predictions.jsonl`。同一 79 条 v1.6 real-heldout 上 raw recall `0.9210`；`pad=0.2s` 后 recall `0.9556`、precision `0.6942`、F1 `0.8042`、missed speech `17.24s`、missed speech segments `17`、extra audio ratio `1.3765`。该结果与此前 v1.6 high-recall 结论同口径，后续可作为 Qwen3-ASR finetune 前后的固定 VAD 对照。
- v1.7 当前下一步：不急于重训 VAD。先保持 v1.5 posw2 高召回 operating point，继续收集真实 hard-negative / human-nonverbal / BGM / 长静音样本，并推进 Qwen3-ASR SFT 数据准备与小样本 smoke；等 Qwen3-ASR-1.7B 云端 finetune 后，再用 finetuned ASR 的真实失败样本反哺下一版 FusionVAD-JA。
- Qwen3-ASR SFT 数据准备入口：新增 `tools/fusionvad_ja/prepare_qwen_asr_sft_dataset.py`，用于本机 smoke 和云端 full 复现同一数据生成流程。输入支持 Hugging Face streaming 直读 `litagin/Galgame_Speech_ASR_16kHz` / `litagin/Galgame_Speech_SER_16kHz`，也支持本地已物化 `hf_audio_manifest.json`；输出 `qwen-sft/train|val|test.jsonl` 只含 `audio` 与 `text`，`text` 采用官方 finetuning 口径 `language Japanese<asr_text>...`，额外来源、duration、emotion `cls` 等字段写到 `manifest/*.manifest.jsonl`。脚本支持 `--mode smoke/full`、固定 `--shuffle-seed`、`--hf-cache-dir`、`--hf-endpoint`、`--revision`、`--hard-negative-jsonl` 和 `HF_XET_HIGH_PERFORMANCE=1`。
- Qwen3-ASR SFT 本机 smoke：未联网，使用本地 `datasets/train/fusionvad-ja/v1-galgame/galgame-materialized-512/hf_audio_manifest.json` 生成 `datasets/train/qwen3-asr-ja-galgame/v1-smoke-local-asr10/`，split counts 为 train `6`、val `2`、test `2`，确认真实本地 WAV、日语 transcript 规范化和 Qwen SFT JSONL 格式可用。SER 全量与公开 ASR 全量仍留给云端直接下载生成，避免本地上传几十 GB 数据。
- Qwen3-ASR SFT 云端建议命令：云服务器先设置 `HF_HOME` 到数据盘，海外节点优先直连官方 HF；国内节点可按平台实测加 `HF_ENDPOINT`。云端默认没有 `Qwen/Qwen3-ASR-1.7B`，所以必须先下载模型，再生成 SFT 数据。新增 `tools/fusionvad_ja/prepare_qwen_asr_cloud_assets.sh`，默认把模型下载到 `models/Qwen-Qwen3-ASR-1.7B`，把首轮数据包生成到 `datasets/train/qwen3-asr-ja-galgame/v1-pilot-asr200k/`；默认只取 ASR `200000` 条 train + `1000/1000` val/test，并以原始 OGG 落盘，避免把 5353.9h 数据解码成约 617GB WAV。若要 SER，显式设置 `SFT_INCLUDE_SER=1`；若要无限 full，显式设置 `SFT_ASR_TRAIN_LIMIT=0` 并准备 TB 级数据盘。可通过 `HF_CACHE_DIR`、`HF_ENDPOINT_VALUE`、`QWEN_MODEL_DIR`、`QWEN_MODEL_REVISION`、`HF_MAX_WORKERS`、`SFT_OUTPUT_ROOT`、`SFT_REVISION`、`SFT_HF_AUDIO_FORMAT`、`SFT_*_LIMIT` 覆盖。
- Qwen3-ASR SFT 云端训练入口：新增 `tools/fusionvad_ja/run_qwen_asr_sft_cloud.sh`，默认读取上一步生成的 `qwen-sft/train.jsonl` 和 `qwen-sft/val.jsonl`，优先使用本地 `models/Qwen-Qwen3-ASR-1.7B`，否则回退官方 HF model id；官方 `qwen3_asr_sft.py` 会下载到 `agents/temp/qwen3_asr_sft.py`。推荐云端顺序为：先跑 `tools/fusionvad_ja/prepare_qwen_asr_cloud_assets.sh` 准备模型和数据，再用 `INSTALL_QWEN_ASR_DEPS=1 INSTALL_FLASH_ATTN=1 DRY_RUN=1 tools/fusionvad_ja/run_qwen_asr_sft_cloud.sh` 安装训练依赖并检查命令，最后正式执行训练。launcher 默认 `BATCH_SIZE=1`、`GRAD_ACC=32` 是保守防 OOM 起点，24GB/48GB/80GB 卡可逐步提高 `BATCH_SIZE`，多卡设置 `NPROC_PER_NODE`；输出 checkpoint 写入 `datasets/train/qwen3-asr-ja-galgame/v1-full-qwen3-asr-1.7b-sft/`，日志写入 `agents/temp/qwen3-asr-sft-cloud.run.log`。
- Qwen3-ASR 可选数据源 backlog：`joujiboi/Galgame-VisualNovel-Reupload` 作为二期 probe 候选，理由是它声称提供约 `705万` 行 Galgame / VisualNovel audio + Japanese text，Parquet 形态可能比 OOPPEENN 原始 `479GB` 压缩包更易 streaming。风险是 reupload 页面显示 license 不清晰 / No License，且必须先抽样确认字段、文本质量、去重、下载速度和版权边界；因此不进入第一轮 `litagin ASR + SER` 默认 full SFT 混合。
- v1.8 研究分支当前代码状态：`feat/fusionvad-ja-research` 已新增 `fusionvad_ja` VAD 后端，默认使用 v1.5 posw2 checkpoint、Qwen3-ASR-0.6B frozen feature、threshold `0.00015`、pad `0.2s`，并把研究分支默认 `ASR_VAD_BACKEND` 临时切到 `fusionvad_ja` 方便整链路测试。该默认值变更只代表当前研究分支实验口径，不等价于 main / Web 正式默认 VAD 切换；合入 main 前应重新决策默认值。
- v1.8 新增本地复现工具：`tools/fusionvad_ja/run_full_workflow.py` 用于固定 FusionVAD-JA high-recall VAD + Qwen3-ASR + forced aligner 跑整链路，并归档 SRT、aligned segments、transcript、QC 和 run log；`tools/fusionvad_ja/diagnose_asr_alignment.py` 用于离线诊断每个 VAD chunk 的 ASR 文本、`align_text`、forced-aligner fallback、QC/drop 原因和失败候选；`tools/fusionvad_ja/sweep_addition_thresholds.py` 支持按视频帧数换算 padding，例如默认 `6` 帧、`30000/1001` fps，避免只用固定秒数描述字幕切割边界。
- v1.8 aligner 前处理已落地：新增 `src/whisper/prealign.py`，维护 `display_text` 与 `align_text` 两份文本。`display_text` 保留给字幕显示，`align_text` 给 forced aligner 使用，会做 NFKC、空白/标点/装饰符号清理、长重复假名压缩、重复短语压缩；`src/whisper/local_backend.py` 已改为使用该模块，并在 forced aligner 后把时间戳映射回原显示文本。目标是先降低 fallback 和异常时间轴，不急于 finetune forced aligner。
- v1.8 匿名样片 A 历史 checkpoint 对照：统一使用 FusionVAD-JA high-recall operating point，对比原版 Qwen3-ASR-1.7B、200k Galgame SFT、full v5 checkpoint-5000、full v5 checkpoint-6000。base 输出 `453` 段、`488` 条 SRT cue、`5860` 字、ASR+align `854.2s`、fallback `0.191`；200k 输出 `383` 段、`417` cue、`6545` 字、`1100.7s`、fallback `0.254`；5000 输出 `381` 段、`422` cue、`7454` 字、`904.6s`、fallback `0.242`；6000 输出 `380` 段、`412` cue、`7633` 字、`1041.7s`、fallback `0.274`。这组结果包含旧 ASR 后处理黑名单、direct drop 和旧 alignment 诊断口径，只保留为历史记录，不再作为当前主参考。
- v1.8 ASR/alignment 历史诊断结果：四组匿名样片 A 共 `1348` 个 chunks，失败候选 `1000`，fallback chunks `414`（`30.7%`），ASR dropped uncertain `421`，align-text-empty `18`。分组看，base fallback `27.6%`、dropped `101`、segments `453`；200k fallback `31.5%`、dropped `105`、align_empty `5`、segments `383`；5000 fallback `30.3%`、dropped `102`、align_empty `7`、segments `381`；6000 fallback `33.5%`、dropped `113`、align_empty `6`、segments `380`。额外用 qwen5000 跑匿名样片 B 时 fallback `38.2%`、dropped `110`、segments `794`。该结果已被 v1.9 当前规则复测取代，仅用于追溯规则变化影响。
- v1.8 当前结论：FusionVAD-JA 暂时继续保持 high-recall proposal generator 定位，不再本轮追求 precision 或立即重训 VAD；forced aligner 也暂不 finetune，因为现有 Galgame ASR 数据没有字符/词级时间轴真值，直接训练 aligner 风险高。下一步优先做三件事：给输出增加 alignment 质量标签 `forced|partial|vad_coarse|proportional|drop_or_review`；把长 chunk、低信息人声、重复循环、align-text-empty 和 dropped uncertain 汇入失败样本池；等 Qwen3-ASR-1.7B full SFT checkpoint 稳定后，用同一批 held-out 统计漏对白、多送音频、空输出、hallucination、低置信和 fallback，再决定是否训练下一版 VAD / 后处理 / aligner。
- v1.8 产物路径：为避免 README 和提交历史暴露真实片名，公开记录只使用“匿名样片 A / 匿名样片 B”这类别名，不再写入真实视频 stem 或含真实 stem 的 `agents/temp/` 路径；本地原始诊断产物仍保留在 `agents/temp/fusionvad-ja/` 下，仅用于个人复查。后续所有报告、测试 fixture、文档示例、commit message 和可跟踪文件都应使用匿名样片名。
- v1.9 ASR 后处理清理：全量审计后删除词表驱动的 ASR direct drop，包括 `ASR_NOISE_WORDS`、噪声词表、灰区词表、假名/呻吟短句特例、历史工具签名特例和对应白名单；同时取消 AnimeWhisper 后置括号/重复清洗、最终字幕文本重复压缩、翻译 prompt 源文重复压缩、固定拟声词映射表和翻译前纯英文幻觉 direct drop。Adaptive Precision 的清空文本行为改为 `ASR_QC_DROP_UNCERTAIN=1` opt-in，默认只诊断；speaker diarization 也不再把假名-only 文本当成 BGM 跳过。`はぁ`、`うん`、`気持ち`、`好き`、重复短促发声等目标域文本不再因为具体字样、假名集合、重复形态、低置信或英文字符形态被直接删除/改写。空文本、纯标点/纯符号和上下文泄漏仍会过滤；重复循环、低置信、异常密度等保留为 QC/诊断信号。
- v1.9 alignment 诊断收敛：新增可复用 `alignment_quality` 分类口径，离线诊断 JSONL 每个 chunk 显式输出 `alignment_quality` 和 `fallback_type`，summary 聚合质量标签与 fallback 类型计数。`forced` 只表示 forced aligner 正常产出；`partial` 表示 forced 对齐存在哨兵、异常或时间轴重叠风险；`vad_coarse` / `proportional` 是可解释粗时间轴；`drop_or_review` 表示文本为空、`align_text` 为空、ASR dropped uncertain 或无输出片段等需要审计。该标签默认只用于闭环比较和失败样本池，不直接删除 ASR 文本。
- v1.9+ alignment fallback 细分：`classify_alignment_quality()` 和 `diagnose_asr_alignment.py` 现在额外输出 `fallback_subtype` 与 `fallback_subtype_counts`。subtype 不改变 `alignment_quality` 大类，只用于归因，例如 `asr_empty_text`、`asr_dropped_uncertain`、`align_text_empty`、`text_without_output_segment`、`vad_coarse_after_sentinel`、`proportional_after_align_error`、`word_timing_zero_heavy`、`word_timing_low_coverage`。
- v1.9 失败样本池导出口径：`tools/fusionvad_ja/diagnose_asr_alignment.py` 的 `failure_candidates.jsonl` 现在使用统一 `failure_candidate` 布尔字段，而不是只看旧 `failure_reasons`；只要 `alignment_quality != forced` 或存在 QC/ASR/alignment 失败原因就会进入候选。每条候选额外写入 `failure_bucket`，当前 buckets 包括 `asr_dropped_uncertain`、`align_text_empty`、`empty_text_for_chunk`、`text_without_output_segment`、`partial_alignment`、`vad_coarse_alignment`、`proportional_alignment`、`unknown_alignment_fallback`、`long_low_information_text`、`abnormal_char_density`、`asr_qc_reject`、`asr_qc_warn` 和 `diagnostic_warning`。后续 held-out 复测可直接按 bucket 观察 full SFT 是否减少 fallback / review 样本。
- v1.9 checkpoint 对比汇总：新增 `tools/fusionvad_ja/compare_alignment_diagnostics.py`，输入多个 `diagnose_asr_alignment.py` 输出目录，生成 `checkpoint_comparison.json`、`checkpoint_comparison_rows.jsonl` 和 `checkpoint_comparison.md`。该工具只消费诊断产物，不跑模型；本地拿到 full SFT checkpoint 后，先跑同一 held-out，再用它比较不同 checkpoint 的 forced 比例、fallback 比例、failure candidate 比例和 failure bucket 分布。
- v1.9 失败 manifest 导出：新增 `tools/fusionvad_ja/export_alignment_failure_manifest.py`，把 `failure_candidates.jsonl` 转成 `alignment_failure_manifest.jsonl/csv`。导出行保留匿名 case label、源音频路径、aligned JSON 引用、chunk 起止、`alignment_quality`、`fallback_type`、`failure_bucket`、候选显示文本和空白 `manual_label/manual_text/notes` 字段；当前不复制/切分音频，避免误判 chunk 文件位置，后续审计页面或切片脚本再按 `source_audio_path + start/end` 生成音频片段。
- v1.9 失败音频物化：新增 `tools/fusionvad_ja/materialize_alignment_failure_audio.py`，消费 `alignment_failure_manifest.jsonl`，按 `source_audio_path + start/end` 可选 padding 切出 16k mono WAV，并导出 `alignment_failure_audio_manifest.jsonl/csv`、错误列表和 summary。该工具用于本地人工复听、hard-negative 汇总和后续 checkpoint 闭环对比，不改变字幕输出。
- v1.9 当前规则匿名样片 A 复测：在删除词表黑名单和 direct drop 后，重新用同一 FusionVAD-JA operating point 跑 base、200k 和 full v5 checkpoint-15500。三组均为 `337` chunks 且 `ASR_QC_DROP_UNCERTAIN=0`。base 输出 `806` 段、`829` cues、`8085` 字、ASR+align `889.3s`、fallback `172/337`（`51.0%`）、failure candidates `200`、QC reject `36`；200k 输出 `794` 段、`843` cues、`13846` 字、`1047.0s`、fallback `166/337`（`49.3%`）、failure candidates `177`、QC reject `18`；15500 输出 `802` 段、`870` cues、`15203` 字、`879.5s`、fallback `170/337`（`50.4%`）、failure candidates `177`、QC reject `18`。
- v1.9 当前规则结论：full SFT 方向仍成立，但判断口径从“segments 是否更多”改为“同一 high-recall chunks 下目标域文本覆盖、重复循环和 alignment fallback”。base 在当前规则下也会保留大量短促人声，因此 segments 数不再能说明召回；200k / 15500 的字符数和近域拟声覆盖明显高于 base。15500 当前文本覆盖最高、速度也接近 base，但 align-text-empty 从 200k 的 `5` 增到 `10`，forced aligner fallback 仍约半数 chunks，下一步应优先做 alignment/fallback/QC 归因，而不是用旧黑名单重新压低输出。
- v1.10 synthetic timeline v4 作为短 gap 过渡版：它证明 crossfade、背景混合、overlap speech 和 `boundary_manifest.jsonl` bench 能工作，但 speech frame ratio 约 `0.83-0.84`，gap 太短，不适合作为后续当前基线。相关 ignored 生成物已移入 `agents/rm/`，只保留历史结论。
- v1.11 synthetic timeline v5 long-gap 已成为默认生成口径：`tools/fusionvad_ja/build_galgame_synthetic_timeline.py` 默认启用长 gap、`speech_label_pad_s=0.08`、real negative gap 概率 `0.75` 和背景混合概率 `0.5`。v5 train/val/test 已生成到 ignored `datasets/*/fusionvad-ja/v1-11/`，分别 `256/64/64` 条，skipped `0`，speech frame ratio `0.574/0.551/0.568`，总时长 p50 约 `17s`、p90 约 `22s`，更接近真实视频里“短语音 + 长非语音”的压力测试。
- v1.11 Qwen3-ASR-0.6B feature cache、训练和边界 benchmark：v5 train/val/test feature cache 均完成，使用 CUDA bfloat16；训练混合 v1-mini strong/negative `302` 条 + v5 long-gap synthetic `256` 条，共 `558` 条，checkpoint 位于 `datasets/train/fusionvad-ja/v1-11/qwen3-asr-0.6b/addition-bilstm-ft-v1mini-galgame-synthv5-longgap-posw2-558-batch16-lr2e-4-steps1024/`。val sweep 选 threshold `0.02`；test padded recall `0.9934`、missed speech `4.18s`、extra audio ratio `1.3240`，boundary recall `0.9940`、missed speech `3.65s`、extra audio ratio `1.3741`、overlap speech recall `0.9877`。研究分支后端默认已切到该 v1.11 long-gap operating point，仍不代表 main 正式默认。
- v1.11 real-heldout 复测：在 `datasets/val/fusionvad-ja/v1-6/real-heldout-local-video-audit-80/fusionvad-operating-point-v1-11-synthv5-longgap/` 已生成 operating point 输出。相比 v1.5 posw2 的 recall `0.9556`、missed speech `17.24s`、extra audio ratio `1.3765`，v1.11 为 recall `0.9809`、missed speech `7.42s`、extra audio ratio `1.5021`。当前策略接受这个方向，但必须用匿名样片 downstream ASR/alignment 继续验证多送音频的真实代价。
- v1.11 downstream 初测：checkpoint-21000 已从云端同步到本地模型目录，匿名样片 A 使用 v1.11 long-gap VAD + threshold `0.02` + pad `0.2s` + Qwen3-ASR-1.7B full SFT checkpoint-21000 跑通。该口径切出 `89` 个 chunks，输出 `262` 段、`269` cues，pipeline `312.4s`，ASR+align `285.6s`，质量报告提示 `repetition_ratio=0.074`、`asr_generation_error_count=35`。诊断汇总：forced `14/89`、fallback `38/89`、failure candidates `76/89`、`drop_or_review=46`，主要 buckets 为 `empty_text_for_chunk=35`、`vad_coarse_alignment=31`、`align_text_empty=7`。结论：v1.11 real-heldout 召回收益成立，但当前默认 chunk 边界/合并策略会把长非语音或复杂片段送给 ASR/aligner，所以下一步应先扫 v1.11 `merge_gap` / chunk packing / max chunk 时长，再评估是否需要新一轮 VAD 训练。
- v1.11 Qwen3-ASR-0.6B cloud full SFT：为评估“原版 0.6B frozen feature”与“Galgame full SFT 后 0.6B frozen feature”是否会改善 FusionVAD-JA，已在 RTX 5090 32GB 云端启动 full ASR-only SFT。稳定配置为 `batch_size=8`、`grad_acc=16`、effective batch `128`、`lr=2e-5`、`map_num_proc=16`、`num_workers=8`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；`batch_size=16`、`grad_acc=8` 短测在 step `36` OOM，显存形态为真实峰值不足而非碎片主因。当前训练已从 `checkpoint-1500` 恢复继续跑，后续本机拿到 checkpoint 后只做同口径 held-out / downstream 复测，不把 0.6B SFT 直接视为 VAD 改进。
- Sources：Qwen3-ASR GitHub `https://github.com/QwenLM/Qwen3-ASR`，Qwen3-ASR finetuning `https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning`，Qwen3-ASR technical report `https://arxiv.org/abs/2601.21337`，Qwen3-ASR-0.6B model card `https://huggingface.co/Qwen/Qwen3-ASR-0.6B`，Qwen3-ASR-1.7B model card `https://huggingface.co/Qwen/Qwen3-ASR-1.7B`。
- Sources：FusionVAD arXiv `https://arxiv.org/abs/2506.01365`，ISCA Archive `https://www.isca-archive.org/interspeech_2025/tripathi25_interspeech.html`，teacher-student VAD `https://dl.acm.org/doi/abs/10.1109/TASLP.2021.3073596`，TEN VAD `https://github.com/ten-framework/ten-vad`，Silero VAD `https://github.com/snakers4/silero-vad`，FireRedVAD `https://github.com/FireRedTeam/FireRedVAD`。

ONNXRuntime CUDA 诊断：

- 当前 `.venv` 中 `torch 2.12.0+cu130`、`torch.cuda.is_available=True`、`onnxruntime-gpu 1.27.0.dev20260511001`，可用 providers 包含 `CUDAExecutionProvider`。
- `onnxruntime-gpu` 预发行包依赖的 CUDA 13 / cuDNN 9 shared libraries 位于 `.venv/lib/python3.14/site-packages/nvidia/cu13/lib` 与 `.venv/lib/python3.14/site-packages/nvidia/cudnn/lib`。`src/vad/whisperseg/whisperseg_core.py` 会优先从 `ORT_CUDA_PRELOAD_DIRS` 读取 preload 目录；未设置时自动尝试上述 `.venv` 目录。
- Codex sandbox 内可能仍报 `CUDA failure 35` 并回落 CPU；跑 WhisperSeg / fusion_lite 大样本对比时应使用非 sandbox / 提权执行，日志中必须看到 `onnx_provider=CUDAExecutionProvider` 或 `providers=['CUDAExecutionProvider', 'CPUExecutionProvider']`。
- `fusion_lite` 的 Silero gate 当前仍可能先尝试 Silero ONNX 并打印一次 `libcudnn.so.9` 警告；本轮同集 baseline 没有失败，WhisperSeg 主信号已确认走 CUDA。

v1-mini-balanced-32 历史结果：

- VoxConverse 使用 Hugging Face `Audio` / torchcodec 物化，train/val/test 分别取 dev index `0-11`、`12-15`、`16-19`，音频和父音频 ID 均互斥；torchcodec decode rows 分别为 12/4/4，error 0。
- VoxConverse supervised 20s clip：train 151、val 126、test 49，skipped 0、errors 0；speech ratio 均值约 0.95，只能提供 supervised speech span，不能单独覆盖 non-speech。
- MUSAN 已下载到 `datasets/raw/musan/` 并解出 `music=660`、`noise=930`、`speech=426` 个 wav。下载包因断点续传尾部有 trailing garbage，但有效 `musan/` 目录已解出，当前 v1-mini 只使用 `music` / `noise` real-negative。
- MUSAN negative 20s clip：train 151、val 126、test 49，skipped 0、errors 0，speech ratio 0。
- v1-mini-balanced-32 从每个 split 中固定取 16 条 VoxConverse supervised + 16 条 MUSAN negative；feature cache 全部使用 `whisper-ja-1.5b` CUDA half precision，`whisper_dim=1280`、`mfcc_dim=40`、cached 32、errors 0、skipped 0。
- GPU 诊断：`WhisperEncoderFeatureExtractor(ptm='whisper-ja-1.5b', device='cuda')` 的参数位于 `cuda:0`，加载后 CUDA allocated 约 3.09GB；20s clip encoder feature 提取约 0.8s/条，首条约 1.7-2.1s。`build_feature_cache.py` 已打印实际 model path、device、dtype、CUDA memory 和逐条进度，避免误判为 CPU 路径。
- 256-step 训练过短，模型非常保守：val F1 0.0566、precision 0.9763、recall 0.0291；test F1 0.0442、precision 1.0、recall 0.0226。
- 2048-step addition-fusion BiLSTM：trainable params 1,942,145，train loss 0.1212、frame accuracy 0.9501、positive ratio 0.4883。Held-out val `frame_accuracy=0.9513`、`precision=0.9855`、`recall=0.9132`、`F1=0.9480`、`positive_ratio=0.4859`、`predicted_positive_ratio=0.4503`；held-out test `frame_accuracy=0.9885`、`precision=0.9817`、`recall=0.9948`、`F1=0.9882`、`positive_ratio=0.4837`、`predicted_positive_ratio=0.4901`。
- 该结果是首个 real-negative supervised mini 结果，但数据仍小，且 positive 主要来自 VoxConverse、negative 主要来自 MUSAN music；不能视为 JAV/galgame 泛化结论。下一步需要扩大 balanced split，并加入 Galgame / 本地 JAV near-domain teacher labels 与 `fusion_lite`、`whisperseg-adaptive` 同集 baseline。

v1-probe 结果：

- AVA-Speech HF 源 `nccratliri/vad-human-ava-speech` 小样本可生成 supervised timestamp label，但该数据集只含 VAD 标注字段，不含原始音频；未另行取得 AVA 音频前不能直接进入 feature cache。
- VoxConverse HF 源 `diarizers-community/voxconverse` 小样本可用 `Audio(decode=False)` 物化 WAV：3 条 dev 音频、总时长 1116.65s、error 0；label/audio manifest 对齐得到 3 条 supervised example，AVA 3 条因缺音频被跳过。
- VoxConverse 3 条样本按 20s 非重叠切窗得到 56 条 supervised clip、总时长 1105.48s、error 0；但全部 `speech_ratio > 0.78`，平均 `0.9701`，只能作为 supervised-positive smoke，不能替代真实 negative。
- v1-mini-smoke 已完成“VoxConverse supervised-positive + synthetic negative”闭环：56 条 VoxConverse supervised clip + 56 条 synthetic negative，feature cache 112、errors 0、skipped 0；256-step addition-fusion BiLSTM smoke `loss=0.3769`、`frame_accuracy=0.8328`、`positive_ratio=0.4901`，可训练参数 1,942,145；同集 sanity eval `frame_accuracy=0.9859`、`precision=0.9719`、`recall=0.9997`、`F1=0.9856`。该结果只验证新数据/切窗/训练/eval 闭环，不是泛化指标。

MUSAN 本地下载建议：

```bash
mkdir -p datasets/raw/musan
curl -L https://openslr.magicdatatech.com/resources/17/musan.tar.gz -o datasets/raw/musan/musan.tar.gz
tar -xzf datasets/raw/musan/musan.tar.gz -C datasets/raw/musan
```

当前 smoke 结果：

- 真实 ja whisper 1.5B feature cache 已验证：`whisper_dim=1280`、`mfcc_dim=40`、`frame_hop_s=0.02`。
- 2 正 + 2 负真实 feature cache：cached 4、errors 0、skipped 0。
- 4 正 + 8 负真实 feature cache：cached 12、errors 0、skipped 0；48-step addition-fusion BiLSTM smoke 写出 checkpoint，`positive_ratio=0.2071`，`trainable_parameters=1,942,145`。
- v0 训练批次：物化 Galgame 64 条，选 32 条短 weak-positive + 64 条 synthetic negative，真实 feature cache 96 条、errors 0、skipped 0；256-step addition-fusion BiLSTM smoke `loss=0.3196`、`frame_accuracy=0.8574`、`positive_ratio=0.3503`。
- v0 validation / test 归档：各 8 条 Galgame weak-positive + 16 条 synthetic negative，feature cache 各 24 条、errors 0、skipped 0；使用 v0 checkpoint 评估时 val/test `frame_accuracy=1.0`、`F1=1.0`，该结果主要说明链路可执行和 split 可归档，不能视为正式泛化精度。
- addition-fusion BiLSTM 默认结构可训练参数为 1,942,145，低于 2M；早期 smoke checkpoint 写入 `agents/temp/fusionvad-ja/`，正式研究 checkpoint 写入 ignored `datasets/train/fusionvad-ja/`。
- 该结果仅证明训练链路和 shape 对齐可用，不代表正式 VAD 精度结论。

参考来源：

- FusionVAD paper: <https://arxiv.org/abs/2506.01365>
- Hugging Face paper page: <https://huggingface.co/papers/2506.01365>
- AVA-Speech VAD: <https://huggingface.co/datasets/nccratliri/vad-human-ava-speech>
- VoxConverse: <https://huggingface.co/datasets/diarizers-community/voxconverse>
- MUSAN: <https://www.openslr.org/17/>
- DNS Challenge: <https://github.com/microsoft/DNS-Challenge>

</details>
