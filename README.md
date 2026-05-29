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
- Forced alignment + F0：ASR 后进行词级强制对齐，再做 F0 性别检测和 gender turn 重切段。
- 翻译前 cue plan：LLM 翻译前先固定字幕时间轴和 cue 数量，SRT writer 不再改变时间轴。
- LLM 翻译：支持 OpenAI-compatible Chat Completions 和 Responses API，保留 reasoning effort、API 格式、目标语言、术语表、worker 数这些可手动配置项。
- 质量报告：默认输出 Markdown，同时保留 JSON sidecar 方便自动化评测。

---

## 当前默认流程

主流水线：

```text
视频 -> 音频准备 -> fusion_lite VAD -> VAD chunk packing -> ASR -> Adaptive Precision QC
-> Forced Alignment -> 词级 F0 性别检测 -> gender turn 重切段
-> 翻译前空/纯符号段过滤 -> 翻译前 cue plan 时间轴归一化
-> LLM 逐 cue 翻译 -> SRT / quality report
```

当前 ASR 以高召回、可诊断为默认目标。旧的 ASR recovery、temperature fallback、prompt overflow retry 已移除；主 VAD 空结果会直接跳过 ASR；timestamp/alignment fallback 只用于给已确认文本补时间轴，不改写或新增 ASR 文本。Adaptive Precision QC 默认只记录低置信、重复、异常密度等风险信号；只有显式设置 `ASR_QC_DROP_UNCERTAIN=1` 时才会清空高风险 ASR 文本。

### ASR 与 VAD

默认 VAD 是 `fusion_lite`。公开可选的 VAD 后端只保留 `fusion_lite` 和 `whisperseg-adaptive`。Silero 只作为 fusion-lite 系列内部 speech prior，不作为独立主 VAD 暴露。

主 VAD 初始化或推理失败会直接抛错并进入 Web 日志。主 VAD 返回空结果是合法“无语音”结果，不会 fallback 成整段音频转写。

`ASR_LONG_CHUNK_PROFILE=on` 时强制开启 VAD chunk packing 与 post-alignment F0：

```env
ASR_CHUNK_PACKING_ENABLED=1
ASR_CHUNK_PACK_WINDOW_FRAMES=899
ASR_CHUNK_PACK_RESERVE_FRAMES=45
ASR_CHUNK_PACK_TARGET_PADDING_FRAMES=60
ASR_CHUNK_PACK_GAP_MERGE_FRAMES=45
F0_GENDER_POST_ALIGNMENT=1
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

- 当前研究分支允许临时把 `ASR_VAD_BACKEND` 切到 `fusionvad_ja` 做整链路实验；这不等价于 main / Web 正式默认 VAD 切换，合入 main 前必须重新决策默认值。
- 当前研究分支 operating point 已切到 FusionVAD-JA v1.11 long-gap：Qwen3-ASR-0.6B frozen audio feature + MFCC addition BiLSTM，threshold `0.02`，pad `0.2s`，定位仍是 high-recall proposal generator。v1.5 posw2 只保留为历史对照。
- VAD 目标是不漏对白、呻吟、喘息、短促人声；precision、hard-negative 过滤和幻觉控制先交给 ASR / aligner / 后处理失败样本闭环。
- Qwen3-ASR-1.7B full SFT 是目标域 ASR 主线；Qwen3-ASR-0.6B 暂时作为 FusionVAD-JA frozen feature extractor 和后续轻量 ASR probe。
- Forced aligner 暂不 finetune。当前优先级是文本预处理、fallback 质量标签、失败样本池和同口径 held-out 复测。
- Galgame ASR 数据多数已经按语音裁切，因此可把原 clip 当作 speech island，在前后和中间拼接随机长度静音、白噪声、hum、MUSAN/DNS/BGM 或本地 hard-negative，构造精确 `actual_speech_segments`。这条 synthetic timeline 不再只是早期 VAD smoke，而是下一轮 VAD / boundary refiner / aligner bench 的公共数据底座。

当前数据和约束：

- `litagin/Galgame_Speech_ASR_16kHz` 是核心近域 ASR / VAD 弱监督来源；AVA-Speech / VoxConverse 只作为强时间标注 seed；MUSAN / DNS / 本地视频 / 合成 gap 作为 negative 和增强素材。
- 标签 JSONL 保持 `audio_id`、`source`、`duration_s`、`text`、`teacher_segments`、`frame_hop_s`、`speech_frames`、`label_quality`；`teacher_conflict` 只审计，不默认进训练。
- 公开文档、测试 fixture、commit message 和可跟踪报告一律使用匿名样片名，不写真实视频 stem 或含真实 stem 的 `agents/temp/` / `agents/audits/` 路径。

当前同口径 ASR 对比结论：

- 匿名样片 A 已用当前 v1.9 文本/后处理规则、同一 FusionVAD-JA operating point 复测 base / 200k / full v5 checkpoint-15500。旧 v1.8 对照包含历史黑名单和 direct drop，不再作为主参考。
- 当前规则下三组都处理同一 `337` 个 VAD chunks。base 输出 `806` 段、`829` cues、`8085` 字；200k 输出 `794` 段、`843` cues、`13846` 字；checkpoint-15500 输出 `802` 段、`870` cues、`15203` 字。
- 结论是 full SFT 方向仍然成立：segments 数接近，但 200k / 15500 的目标域文本覆盖明显高于 base；同时 forced aligner fallback 仍高，分别约 base `51.0%`、200k `49.3%`、15500 `50.4%`，说明当前主要瓶颈已经转向 alignment / fallback / QC，而不是 ASR 是否能输出。
- checkpoint-21000 已拉到本地并用 v1.11 long-gap VAD 跑匿名样片 A 闭环；它不是纯 ASR checkpoint 对比，因为 VAD 口径从 v1.5 切到 v1.11。未做长段保护时，该组合切出 `89` 个更长的 VAD chunks，输出 `262` 段、`269` cues，ASR+align `285.6s`；但诊断中 forced 仅 `14/89`（`15.7%`），fallback `38/89`（`42.7%`），failure candidates `76/89`（`85.4%`），主要 bucket 是 `empty_text_for_chunk` `35` 和 `vad_coarse_alignment` `31`。结论：v1.11 提升 real-heldout 召回后，长 chunk / 空输出 / coarse fallback 成为新的下游瓶颈；下一步要在 v1.11 内先修 chunk packing / pre-align / fallback，而不是直接把这组结果解读为 21000 ASR 变差。
- Qwen3-ASR-0.6B full ASR-only SFT 已在云端 RTX 5090 32GB 上启动，用途是后续比较原版 0.6B 与 Galgame full SFT 后 0.6B 作为 FusionVAD-JA frozen feature extractor 的差异。当前稳定参数为 `batch_size=8`、`grad_acc=16`、effective batch `128`、`lr=2e-5`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，从 `checkpoint-1500` 继续训练；截至 2026-05-29 18:59 CST 约 `1568/29239` steps。RTX 5090 上 `batch_size=16`、`grad_acc=8` 已短测失败：step `36` OOM，尝试分配 `3.89GiB` 时仅剩 `2.34GiB`，PyTorch reserved-unallocated 仅 `206MiB`，结论是真实峰值显存不足，不是主要碎片问题；后续提速方向应是 length-aware / frame-budget batching，而不是固定 `bs16`。

synthetic timeline / boundary refiner 计划：

- 由来：人工复听确认 `litagin/Galgame_Speech_ASR_16kHz` 多数 clip 本身已经裁成语音片段；在 clip 前后拼接随机空白或噪声后，前置 gap 长度就是 speech start，`start + clip_duration` 就是 speech end。多段拼接时每个 speech island 都有可计算真值。
- 适用范围：该数据天然适合训练和评测 frame-level VAD、speech boundary refiner、VAD-constrained alignment fallback，以及 forced aligner 的 start/end 鲁棒性 bench。它只提供片段级边界，不天然提供字/词级时间轴，所以不能直接等价为 Qwen forced aligner finetune 数据。
- 数据形态：`tools/fusionvad_ja/build_galgame_synthetic_timeline.py` 负责生成 16k mono WAV、标准 VAD label JSONL、兼容训练的 `manifest.json`、详细 `synthetic_timeline_details.jsonl`，以及公共 `boundary_manifest.jsonl`。`boundary_manifest.jsonl` 同时保留 `actual_speech_segments` 和带训练 pad 的 `speech_segments`，后续 bench 默认用 actual span 做边界误差真值；`transition_regions` 记录 crossfade 模糊区，`augmentation` 记录背景、overlap、gain、filter 和 codec 处理。
- v5 long-gap 口径：脚本默认使用 `5-30ms` equal-power crossfade、随机 gain `-3~3dB`、轻量 lowpass/bandpass 概率 `0.25`、codec 模拟概率 `0.05`、overlap speech 概率 `0.12`、背景混合概率 `0.5`；speech 最长裁到 `8s`，speech 最短保留到 `0.05s`，middle gap 默认 `1-6s`，首尾 gap 默认 `0.5-4s`，real negative gap 默认概率 `0.75`。无 negative manifest 时仍会退回 silence / white_noise / hum / fade_noise；旧硬拼接行为仅用于显式传 `--crossfade-ms-min 0 --crossfade-ms-max 0 --gain-db-min 0 --gain-db-max 0 --filter-prob 0 --codec-prob 0 --overlap-speech-prob 0 --background-mix-prob 0 --speech-label-pad-s 0` 的单测。
- 当前执行：v5 long-gap split 已生成到 ignored `datasets/*/fusionvad-ja/v1-11/`，train/val/test 分别 `256/64/64` 条，skipped 均为 `0`。speech frame ratio 为 `0.574/0.551/0.568`，总时长 p50 约 `17.04/16.95/17.67s`、p90 约 `22.70/21.95/22.45s`、max 约 `26.98/24.56/24.98s`。train/val/test 的 real negative gap 为 `554/146/149`，background mix 为 `118/34/34`，overlap speech 为 `26/12/7`，filter 为 `69/11/14`，codec 为 `15/7/1`，每个 split 均已写出 `boundary_manifest.jsonl`。
- 当前 benchmark：v1.11 long-gap 头在 v5 test 上 threshold `0.02` + pad `0.2s` raw recall `0.9910`、precision `0.8124`、F1 `0.8929`；padded recall `0.9934`、missed speech `4.18s`、extra audio ratio `1.3240`。`boundary_manifest.jsonl` 边界评测 test split speech-duration recall `0.9940`、missed speech `3.65s`、extra audio ratio `1.3741`、overlap speech recall `0.9877`；start/end p50 误差约 `0.675s/0.814s`，p90 约 `2.45s/1.88s`。结论：v1.11 比短 gap 版本更适合真实长静音 / BGM / hard-negative 场景，仍作为 high-recall proposal generator，不作为精确切轴最终模型。
- 真实 held-out 复测：同一 v1.6 real-heldout `79` 条上，v1.5 posw2 + threshold `0.00015` + pad `0.2s` 为 recall `0.9556`、missed speech `17.24s`、extra audio ratio `1.3765`；v1.11 long-gap + threshold `0.02` + pad `0.2s` 为 recall `0.9809`、missed speech `7.42s`、extra audio ratio `1.5021`。结论：v1.11 明显更贴近“宁可多送、不漏人声”的策略，但会多送约 `38.9s` 额外音频，必须继续用 downstream ASR / alignment fallback 判断这部分代价是否可接受。
- downstream caveat：v1.11 默认 `merge_gap=0` 仍可能输出少量超长 chunk，导致 Qwen ASR 空输出和 forced aligner sentinel 增多。当前已把 downstream chunk packing 改成“固定帧数 + 任务级真实 FPS”：`window_frames=899`、`reserve_frames=45`、`target_padding_frames=60`、`gap_merge_frames=45`，每个视频用 `1/fps` 换算这些帧对应的秒数，只有 FPS 探测失败才回退 29.97。在匿名样片 A 旧 raw VAD segments 上离线重算，processing spans 从旧 split28 的 `255` 降到 `240`，最长 `28.50s`，平均 `25.05s`，最大左右 padding 均约 `2.002s`，split reason 为 `overlong=216`、`capacity=13`、`gap=11`。后续 v1.11 评估必须同时报告 `transcript_chunks`、chunk duration 分布、ASR empty count、`nonlexical_text`、`drop_or_review`、`vad_coarse` fallback、`fallback_subtype` 和 SRT cues，不能只看 VAD recall。
- 模型路线：先继续当前 `Qwen3-ASR-0.6B frozen feature + MFCC addition BiLSTM` 高召回线；并行增加一个 frozen SSL baseline，优先 probe `reazon-research/japanese-hubert-base-k2`，其次 `rinna/japanese-hubert-base`，再看 `rinna/japanese-wav2vec2-base`。Grok 检索显示这些是更贴近日语语音表征的候选；旧 XLSR large 日语 ASR fine-tune 不作为主线。
- 评测顺序：先用 synthetic timeline 测 start/end 误差、recall、extra audio ratio 和 inference cost；再回到 v1.6 真实 held-out 与匿名样片 A 同口径测 downstream ASR / alignment fallback。只有 synthetic 和真实 held-out 都有收益，才考虑替换 FusionVAD-JA feature extractor 或训练 boundary refiner。
- Forced aligner 路线：Qwen3-ForcedAligner-0.6B 仍是主线。官方模型卡确认其支持日语、最长约 5 分钟、词/字级 timestamp，并与 Qwen3-ASR 配套；但目前没有找到公开 forced-aligner finetune recipe。MFA Japanese 更适合规范文本和词典化发音，不作为当前主线。

v1.9 ASR / forced alignment 文本策略：

- `display_text` 是最终字幕显示文本，只做展示安全处理：Unicode NFKC、换行归一为空格、连续空白折叠和首尾 trim。不得在 `display_text` 上压缩重复假名、重复短语、拟声或低信息短文本，因为这些在目标域里可能是字幕语义。
- `align_text` 是 forced aligner 专用文本，可以删除标点、emoji / 装饰符、音乐符号和明显不可发音标记，也可以压缩极端重复假名、长音符和重复短语；这些操作必须记录 flags，并保留从 `align_text` 字符到 `display_text` 覆盖范围的映射。
- 不使用按具体字样维护的黑名单，不直接删除 `ん`、`あ`、喘息/呻吟拟声、常见台词、历史工具签名或纯英文长词。ASR 后处理已删除噪声词表、灰区词表、假名/呻吟特例 direct drop、工具签名 direct drop、AnimeWhisper 后置括号/重复清洗、最终字幕文本重复压缩和翻译前纯英文幻觉 direct drop；当前只因空文本、纯标点/纯符号和上下文泄漏这类明确非字幕内容而删除。
- 翻译 prompt 的源文序列化同样不再压缩重复发声，也不使用固定拟声词映射表；重复循环只作为 QC / 诊断信号，译文是否概括交给 LLM 在上下文中判断。
- speaker diarization 不再把假名-only 文本当作 BGM 跳过；只跳过空文本或纯符号/纯标点这类没有语言/数字信号的片段，避免把目标域可字幕化人声排除在 speaker embedding 之外。
- 重复循环、低置信、文本/音频比例异常、align-text-empty、forced-aligner fallback、ASR dropped uncertain 和人工 hard-negative 结果默认只作为 QC / 诊断 / 样本池信号；`ASR_QC_DROP_UNCERTAIN=0` 是默认值，是否删除交给后续可解释 QC 策略，不再用词表兜底。
- forced aligner 失败时不伪造精确时间轴。诊断导出已使用 `forced`、`partial`、`vad_coarse`、`proportional`、`drop_or_review` 五类质量标签，并单独记录 `fallback_type=none|vad_coarse|proportional|unknown` 与更细的 `fallback_subtype`；subtype 用于区分 `asr_empty_text`、`align_text_empty`、`text_without_output_segment`、`vad_coarse_after_sentinel`、`proportional_after_align_error`、`word_timing_low_coverage` 等原因。失败样本进入 VAD / ASR / aligner 后处理样本池。
- 失败样本池闭环分三步：`diagnose_asr_alignment.py` 生成 `failure_candidates.jsonl`，`export_alignment_failure_manifest.py` 转成人工审计 manifest，`materialize_alignment_failure_audio.py` 再按 `source_audio_path + start/end` 切出 WAV 片段，避免依赖中间 chunk 文件路径。
- 实现口径：`src/whisper/prealign.py` 负责 `raw_text -> display_text -> align_text` 和 char-span mapping；`src/whisper/local_backend.py` 只把 `align_text` 送入 forced aligner，拿到词级时间后再映射回 `display_text`。

下一步：

1. 保持当前 FusionVAD-JA high-recall operating point，不急着追 precision 或替换正式默认 VAD。
2. 用 synthetic timeline v5 long-gap 的 `boundary_manifest.jsonl`，作为 VAD / HuBERT-wav2vec baseline / forced-aligner bench 的共同输入。
3. 等 Qwen3-ASR-1.7B / 0.6B full SFT 后续 checkpoint 稳定后，用同一批 held-out 统计漏对白、多送音频、空输出、hallucination、低置信和 forced-aligner fallback。
4. 把长 chunk、低信息人声、重复循环、align-text-empty、ASR dropped uncertain 和人工 hard-negative 汇入失败样本池。
5. 再决定下一版工作重心是 VAD hard-negative、ASR 后处理、forced aligner fallback、boundary refiner，还是补充少量时间轴真值。

参考来源：FusionVAD arXiv `https://arxiv.org/abs/2506.01365`，Qwen3-ASR `https://github.com/QwenLM/Qwen3-ASR`，Qwen3-ASR finetuning `https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning`，Qwen3-ASR-0.6B `https://huggingface.co/Qwen/Qwen3-ASR-0.6B`，Qwen3-ASR-1.7B `https://huggingface.co/Qwen/Qwen3-ASR-1.7B`，Qwen3-ForcedAligner-0.6B `https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B`，Reazon Japanese HuBERT `https://huggingface.co/reazon-research/japanese-hubert-base-k2`，rinna Japanese HuBERT `https://huggingface.co/rinna/japanese-hubert-base`，rinna Japanese wav2vec2 `https://huggingface.co/rinna/japanese-wav2vec2-base`。

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

- 从 Hugging Face 单独下载并模块化评测 `cam++` speaker embedding/聚类能力，确认是否可作为 Whisper/anime 工作流的 speaker sidecar。
- 评测 `efwkjn/cohere-asr-ja-v0.1`，确认其与当前 ASR 流程及 `transformers` 版本约束的兼容性，再决定是否纳入候选后端。
- 增加本地/厂商翻译 API 适配层，允许在现有 OpenAI-compatible 翻译之外接入专用翻译服务，例如腾讯 `hy-mt2`。
- 当前规则下 base / 200k / full checkpoint-15500 的 v1.5 VAD 本地闭环复测已完成；checkpoint-21000 + v1.11 long-gap VAD 也已跑通，但属于不同 VAD 口径。下一步做 v1.11 内部消融：固定 checkpoint-21000，扫 `merge_gap`、chunk packing 和 max chunk 时长，目标是在保持 high recall 的同时降低空输出、`drop_or_review` 和 `vad_coarse` fallback。
- synthetic timeline v5 long-gap 的 `boundary_manifest.jsonl` benchmark 已接入；v1.11 long-gap FusionVAD-JA 头已训练并作为研究分支当前候选。v1.6 real-heldout 对比显示 v1.11 召回更高但 extra audio ratio 也更高；匿名样片 A downstream 初测显示 v1.11 会产生更少但更长的 VAD chunks，需继续做 chunk 边界/打包策略，而不是只调 ASR。
- ASR chunk packing 固定帧数参数暂定为 `899/45/60/45`，每个新视频任务必须重新读取真实 FPS 后换算帧时长；60fps 等高帧率视频会自然得到更短秒级窗口。该策略源自字幕 cue plan 的按帧 gap 口径和 Netflix timed text 的 2-frame gap 思路，但 ASR packing 服务的是 30s 模型窗口保护，不直接复用字幕最终 2-frame gap。
- 继续优化 pre-align / fallback：保持 `display_text` 与 `align_text` 分离，不恢复具体字样黑名单；`diagnose_asr_alignment.py` 已新增 `fallback_subtype`，下一步按 subtype 统计 forced-aligner 失败主因，再决定是调文本规范化、chunk packing、fallback 分段，还是补人工审计样本。
- 增加 frozen SSL boundary baseline：优先 `reazon-research/japanese-hubert-base-k2`，其次 `rinna/japanese-hubert-base` / `rinna/japanese-wav2vec2-base`，和当前 Qwen3-ASR-0.6B frozen feature 线比较 recall、start/end error、extra audio ratio、速度和显存。
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

### FusionVAD-JA 研究归档

<details>
<summary>展开 v1-mini 至 v1.8 详细实验记录和旧计划</summary>

原始 FusionVAD-JA 研究计划与实验记录：

FusionVAD-JA 是训练型 VAD 研究线，用于复现 FusionVAD 论文的“PTM 特征 + MFCC + 简单 addition fusion”思路，并面向日语/JAV/galgame 近域数据做适配。研究代码在 `src/vad/fusionvad_ja/`，临时 smoke 输出写入 `agents/temp/fusionvad-ja/`。下载后的数据、feature cache 和 checkpoint 按 split 归档到 `datasets/train/fusionvad-ja/`、`datasets/val/fusionvad-ja/`、`datasets/test/fusionvad-ja/`；`datasets/` 整体不进入 Git 跟踪。当前研究分支允许临时把默认 VAD 切到 `fusionvad_ja` 做整链路实验，但合入 main / Web 前必须重新决策默认值。

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

- FusionVAD-JA 在本研究分支暂时作为 high-recall proposal generator 使用，重点是不漏对白、呻吟、喘息、短促人声；precision 和 hard-negative 过滤留给后续 ASR/aligner 失败样本闭环。
- 研究分支可临时把 `ASR_VAD_BACKEND` 切到 `fusionvad_ja` 做整链路实验；这不等价于 main / Web 正式默认 VAD 切换。
- Qwen3-ASR-1.7B full SFT 仍是目标域 ASR 主线；Qwen3-ASR-0.6B 暂时作为 FusionVAD-JA frozen feature extractor 和后续轻量 ASR probe。
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
