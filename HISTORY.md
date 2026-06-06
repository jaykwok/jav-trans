# JAVTrans History

本文件记录实验过程、idea 来源、调试坑、失败路线、指标和参考来源。README 只保留新用户使用说明、当前工作流和当前状态。

公开记录统一使用匿名样片名，不写真实视频 stem。

---

## 当前结论

- 2026-06-07 README 分发化精简：README 删除 NAMH-055 / 匿名样片 A 指标、Boundary Refiner 训练路线、Galgame 100k 参数公式、失败实验、审计标签细节和历史路径记录，只保留新用户需要的安装、使用、默认模型、当前工作流、目录结构、常见问题和开发入口。实验依据、路线取舍、指标和调试过程继续只写 HISTORY，避免 README 变成实验日志。
- 2026-06-04 命名决策并执行断兼容迁移：当前系统不再适合继续叫 `FusionVAD-JA`。它已经不是严格 speech/non-speech VAD，而是 `Qwen PTM + MFCC/energy bootstrap frame scores -> boundary candidate extraction -> Boundary Refiner scoring -> constrained planner -> ASR chunks` 的 speech-island boundary system。新概念名定为 **SpeechBoundary-JA**，backend key 为 `speech_boundary_ja`；active package 已从 `src/vad/fusionvad_ja/` / `tools/vad/fusionvad_ja/` 迁到 `src/boundary/ja/` / `tools/boundary/ja/`，不保留 `fusionvad_ja` alias、旧 cache 或旧路径兼容。
- 2026-06-04 破坏式重构：旧 `ASR_PRE_ASR_*` / R15-R23 规则 packer、旧 `ASR_CHUNK_PACK_*` 配置和旧 `vad-cache` 语义已从 active code、测试和 Web 可见配置中移除。当前主线是 `SpeechBoundary-JA frame probabilities -> boundary candidate extraction -> Boundary Refiner scoring -> constrained boundary planner -> ASR chunks`；cache 断兼容升级为 `boundary-cache v1`，signature 包含 SpeechBoundary-JA、Qwen feature/PTM、Boundary Refiner、candidate extractor、planner config 和 `BOUNDARY_FEATURE_FRAME_HOP_S`。
- 2026-06-04 SpeechBoundary-JA schema 断兼容迁移：feature cache / manifest / checkpoint / model layer 统一改为 `ptm`、`ptm_dim`、`ptm_proj`，不再沿用早期 `whisper_*` 命名；仓库内 v1.17、v1.19b、v1.21 小 checkpoint 已迁移 state_dict key。旧 feature cache 和旧实验 checkpoint 不做兼容迁移，重新生成。
- 2026-06-04 配置面清理：旧 `VAD_MIN_OFF` / `VAD_PAD` / `SEGMENT_*` 已从 active defaults、`.env.example`、boundary-cache signature 和 Web advanced 透传中移除；当前只保留语义明确的 `ASR_CHUNK_MIN_DURATION_S`（导出 wav chunk 最小时长）和 `ASR_CONTEXT_RESET_GAP_S`（滑动 ASR 文本上下文重置 gap）。
- 2026-06-04 审计发现并修复 chunk metadata 错位：`_extract_wav_chunks` 过滤过短 span 后，`_annotate_packed_chunks` 过去按输出位置 zip `PackedChunk`，会把 `vad_seg_count` / boundary reason/source/score 错贴到后一个 chunk；现在 chunk info 记录 `source_span_index`，annotation 按原始 span index 对齐，并补回归测试。
- 2026-06-04 timing 语义修正：Boundary / ASR chunk planning 采用秒级上下文参数，`BOUNDARY_FEATURE_FRAME_HOP_S` 只表示 VAD/frame-score 网格 fallback，默认 `0.02s`；字幕显示 / timing polish 单独按真实视频 FPS 计算 `frame_duration_s` 和 Netflix-style `2-frame gap`。主流程不再把视频 FPS 注入 Boundary 配置，aligned-cache 签名改为 v3，并在 `subtitle` 签名中记录 `video_fps` / `frame_gap_s`。
- 2026-06-04 BiLSTM 路线断兼容删除：旧 `AdditionFusion*BiLSTM`、v1.17/v1.19b/v1.21 endpoint / imitation checkpoint、addition / endpoint / imitation 训练导出 CLI、drop-gap imitation offline packer 和旧大测试文件已移出 active tree 到 `agents/rm/bilstm-removal/`。当前 `src/boundary/ja/` 模块只做 Qwen frozen feature + MFCC / energy bootstrap frame scoring；边界决策主线只走 `src/boundary/` 的 Boundary Refiner / Mamba2。
- 新主线模块结构：`src/boundary/features.py` 组装帧级特征，`candidates.py` 提取 gap midpoint / cut peak / low-score valley 候选，`refiner.py` 提供 BoundaryRefiner interface 和 bootstrap refiner，`backbones.py` 放 Windows-friendly Mamba2 research wrapper，`planner.py` 做 constrained planning，`cache.py` 负责 boundary-cache v1。`src/boundary/ja/` 放 JA 目标域 bootstrap scorer。`src/audio/chunk_packer.py` 只保留把 planner 输出 materialize 成 `PackedChunk` 的 ASR-facing 职责。
- 2026-06-04 Boundary Refiner backbone 入口收束到唯一实现路径 `transformers.Mamba2Model`：learned checkpoint schema 固定为 `boundary_refiner_v1`，runtime / CLI / cache signature / checkpoint payload 都使用该值。它直接对应 Hugging Face Transformers 的纯 PyTorch Mamba2 wrapper；`mamba2`、`torch_mamba2`、BiGRU、TCN、Transformer fallback 不再作为可选入口，避免训练和分发形成多套模型协议。
- 2026-06-04 Sequence feature schema authority 落地：`src/boundary/sequence_features.py` 统一提供 `frame_sequence_features_v1` 的 default config、feature dim、feature names、`feature_schema_hash` 和 train/runtime validation。`runtime_adapter=frame_sequence_v1` checkpoint 必须带 `feature_schema`、`feature_schema_hash` 和 `feature_signature`；主 pipeline 会用 SpeechBoundary-JA 导出的 PTM/MFCC frame windows 重新计算 names/hash，不匹配直接 fail-fast，不做旧 checkpoint alias 或静默回退。`PackedChunk` / boundary-cache / ASR chunk metadata 已写入 `boundary_decision_merge`、`boundary_merge_prob`、`boundary_split_prob`、`boundary_refine_delta_s` 和 `boundary_decision_source`，供后续 QC、forced alignment 和审计页追踪决策依据。
- 2026-06-07 runtime speaker / sidecar 断兼容清理：运行时人声聚类、声纹 sidecar、Web speaker 显示、speaker-aware cue merge、cue planner speaker score、speaker sidecar 工具/测试和低能量 pre-ASR drop 已从 active tree 移除或移到 `agents/rm/speaker-runtime-removal/`。字幕层只保留 Netflix-style timing/readability/fallback 风险、2-frame gap、显示时长和短 cue 合并这些观感核心；`speaker_proxy` / `speaker_turn` 只作为 Boundary Refiner synthetic data 里的训练元数据，不进入 runtime 决策或用户配置。
- 2026-06-07 翻译缓存分层：Grok / 官方文档检索确认 provider prompt cache 依赖“稳定长前缀 + 变量放后”的 exact-prefix 机制，适合降低 API token/latency，但不解决本地复跑 cache miss。当前本地翻译改为 `translation_cache.jsonl` 精确 batch cache + `translation_cache.memory.jsonl` 文本级 translation memory：一级仍绑定 cue timing / batch / prompt signature 用于 crash resume；二级绑定 normalized JA text、target language、normalized glossary / auto glossary、character reference、prompt version 和 model family，允许同一任务 Boundary / cue timing 调整后复用译文。低信息或单字符循环文本不写入 memory，避免目标域呻吟/短促发声被过度复用。timings 中区分 `translation_cache_hit`、`translation_memory_hit`、`translation_memory_hit_count` 和 provider prompt-cache usage，避免把三种缓存混为一谈。参考来源：OpenAI Prompt Caching、Claude Prompt Caching、DelTA 多级翻译记忆、2024 BUCC 字幕自适应翻译 fuzzy-match 研究。
- 2026-06-07 翻译链路审计修复：短片 / 审计片段过去会走 `single_request_full_context`，但该 path 没有接 `cache_path`，因此不会命中本地 batch cache / translation memory，也不会写入二级 memory；现在 single-request 与 batched path 共用缓存语义。另修 prompt user message 硬编码“中文字幕”的问题，改为使用 `TARGET_LANG`；自动全片 glossary cache 从固定 `translation_global_glossary.json` 改为 `translation_global_glossary.<source_hash>.json`，避免共享 translation cache 路径时串用上一部片的自动术语表。
- 2026-06-07 项目级运行目录断兼容从 `temp/` 改为 `tmp/`：`JOB_TEMP_DIR=./tmp/jobs`，ASR chunk/checkpoint 为 `./tmp/chunks`，boundary cache 为 `./tmp/cache/boundary`，torch / HF 运行缓存为 `./tmp/cache/torch` 与 `./tmp/cache/hf`。`agents/temp/` 继续只用于研究脚本和审计中间产物，不纳入用户运行缓存结构。旧 `temp/` 不做 alias 或迁移逻辑，避免长期维护两套 runtime root。
- 2026-06-06 Galgame 100k 本地源池落地：先用 streaming duration scan 读取 `litagin/Galgame_Speech_ASR_16kHz` 前 100000 条，只解析 OGG header，不保存音频，确认分布为 mean/p50/p75/p90/p95/p99/max `5.136/4.504/6.937/9.560/11.353/15.187/28.981s`，`>=5s/9s/12s/15s/20s` 比例 `0.441/0.123/0.039/0.011/0.001`，与 dataset card 全量平均约 `5.145s` 一致。随后按用户要求把这 100000 条保留在本地，输出 `datasets/train/boundary-sources/galgame-asr-100k-ogg/`，保存原始压缩 OGG + TXT + `manifest.jsonl`，不转 WAV。校验：manifest `100000` 行、OGG `100000` 个、TXT `100000` 个、errors `0`，summary 记录实际 OGG 字节 `2.313 GiB`，文件系统占用约 `3.0G`（10 万小文件有块占用）。这批数据作为后续 Boundary Refiner / synthetic speech-island dataset 的本地源池，避免每轮构造都重新流式下载 HF。基于该分布新增可复算公式与工具 `tools/boundary/recommend_timing_params.py`：`target_core=round_0.1(clamp(p50/speedup,2.0,3.5))`、`max_core=floor_0.5(clamp(p80/speedup,target+1.0,5.5))`、`target_padding=round_0.1(clamp((p90-max_core)/2,1.0,2.0))`、`max_padded=floor_0.5(min(p90,max_core+2*padding,9.0))`、`min_chunk=round_0.05(clamp(p5/speedup*0.60,0.25,0.50))`。当前 `target_domain_speedup=1.5` 推导出 `target/core/padded/min/padding = 3.0/5.0/9.0/0.4/2.0s`，与现行默认一致；后续换成 anime 混合源或重新采样时直接换 summary 复算。
- 2026-06-05 fallback 时间轴窗口修正：soft-candidate DP v2 已把 20s+ 粗 chunk 消掉，但剩余长 fallback 多数是 `core_duration_s≈8-9s` 被 2s 左右 ASR padding 显示成 `12-13s`。主流程现在继续给 ASR 输入 padded chunk 保留识别上下文，但在 forced aligner 失败、走比例/VAD fallback 时间戳时，只在 speech core 窗口插值；`alignment_fallback_start_s/end_s/source` 会随 text_result 进入 `LocalAsrBackend.finalize_text_results()`，retry / sentinel fallback 分支也使用同一窗口。完整 GPU 闭环实测确认：padded fallback p50/p90/max 仍是 `8.73/12.60/13.10s`，但实际 fallback 时间轴窗口 p50/p90/max 已降到 `6.10/8.76/9.10s`，safe ratio `0.732`，`>10s` fallback 消失。
- 2026-06-05 JAV 短字幕 core 策略收紧：公开检索未找到可靠 JAV 单句字幕时长统计；可用参考是 Netflix timing / Japanese 读速、OOONA、ATA、Karamitroglou 等通用字幕规范，它们都更支持短 cue、贴近 in-time 和长音频拆分。本地 `agents/audits/fallback-window-risk-audit-video/manual_fallback_window_risk_labels.jsonl` 40 条 `8.75-9.10s` fallback-window 人工审计中，`needs_split=35/40`、`multiple_islands=11/40`、`timing_end_early=11/40`。因此 `9s` 不再作为 subtitle/fallback speech-core 默认上限，只保留为 ASR padded context 上限；新默认是 `BOUNDARY_PLANNER_TARGET_CHUNK_S=3.0`、`BOUNDARY_PLANNER_MAX_CORE_CHUNK_S=5.0`、`BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S=9.0`。旧 `BOUNDARY_PLANNER_MAX_CHUNK_S` 断兼容删除，不保留 alias。
- 2026-06-05 NAMH-055 短 chunk 批量与显存监控：基于 `namh055-shortcore-bs4` 已生成的 2459 个 ASR chunks 做阶段 benchmark。0.6B ASR-only 256 chunk：batch `32` 用时 `32.25s`、峰值 `4272 MiB`；batch `48` 用时 `29.62s`、峰值 `5139 MiB`；batch `64` 曾未监控跑通 `29.50s`，但本次 0.2s 持续 `nvidia-smi` 采样在 `6200 MiB` 超过 6GB 目标并被中断；batch `128` 之前已因 `CUDA driver error: device not ready` 失败。1.7B ASR：batch `48` 峰值 `7178 MiB`，batch `24` 峰值 `6327 MiB`，batch `16` 峰值 `6119 MiB`（当时空闲底噪约 `1.1GB`）；用户决定 1.7B 默认保守用 `12`。aligner-only 全量 2459 chunk：batch `16` 用时 `83.72s`、峰值 `3259 MiB`；batch `32` 用时 `80.78s`、峰值 `3864 MiB`；batch `48` 用时 `75.71s`、峰值 `4349 MiB`，均无错误。结论：当前 `ASR_BATCH_SIZE=auto`，通过 `ASR_BATCH_SIZE_BY_REPO` 配置表解析为 0.6B -> `48`、1.7B -> `12`；aligner 默认 `48/48`。
- 2026-06-06 ASR context 后处理断兼容删除：NAMH-055 审计发现 `ASR_CONTEXT=小那海あや` 时，旧 fragment-level prompt/context leak 规则会把真实自我介绍 `小那海あやです` 从最终 cue 中删掉，只剩 `よろしくお願いします`；但词级 `words` 已有对应时间轴。结论是该规则在 JAV / Galgame 目标域误伤成本高、价值低。主流程已删除 `context leak` QC reason、相似度阈值、fragment 删除函数和 Web/env advanced 前缀；`ASR_CONTEXT` / `ASR_HEAD_CONTEXT` 只作为 ASR 提示词，不再驱动最终字幕删除或 QC reject。字幕层继续依赖空文本、纯标点、低信息/重复、信号质量、fallback 和人工审计等更直接的信号。
- 2026-06-06 F0/gender 与最终文本突变清理：旧 F0/gender route、`run_asr_alignment_f0` 历史命名、Web `show_gender` 可见项和相关测试已从 active tree 移除。ASR QC `reject_count` / `review_uncertain`、重复循环 `suggested_text` 和低信息 profile 均保持 review-only；审计页标签从“采用去重建议”改为“重复建议需复核”，点击不会再自动写 `manual_text`。字幕 merge 不再去重 overlap，保留目标域真实重复语气词、呻吟和短促发声。
- Windows 分发约束：默认路线不依赖 Linux-only `mamba-ssm`、Triton 或自定义 CUDA kernel。`src/boundary/backbones.py` 只提供 Hugging Face Transformers `Mamba2Model` 的纯 PyTorch wrapper 作为研究 backbone。
- Unified Joint Model 路线进入 backlog，不进入当前重构：未来可考虑在 Qwen3-ASR decoder 中加入 `<boundary>` / `<dramatic_pause>` / `<sentence_end>` 等 token，做 joint segmentation + transcription 或 Samba-ASR 类长上下文模型。但这需要重新准备 boundary supervision、ASR SFT/RL/DPO 和云端 GPU 训练，当前维护成本和训练成本高，不作为默认路线。
- active backend key 已改为 `speech_boundary_ja`，旧 `fusionvad_ja` key / package / path 不做兼容。
- 默认 ASR backend key 已切到 Hugging Face repo ID 本身：`jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`；可选 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`。短 key 不再进入主线，避免 Web/API/cache/download 出现两套命名。
- SpeechBoundary-JA frozen feature 默认使用 `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`，不再默认下载 base 0.6B。
- SpeechBoundary-JA 默认运行依赖不再包含仓库内小 checkpoint；旧 checkpoint 只作为历史产物保存在本地回收区。
- 破坏式维护重构：`src/whisper/` 改为 `src/asr/`，`tools/fusionvad_ja/` 按职责拆到 `tools/asr/`、`tools/vad/`、`tools/subtitles/`、`tools/audits/`；Whisper/WhisperSeg/TEN/Silero/FusionLite 当前主线代码和测试移入 `agents/rm/obsolete-mainline-cleanup-20260603/`。
- 当前主线不再是 high-recall proposal VAD，也不再是固定 gap packer，而是 SpeechBoundary-JA / Boundary Refiner 驱动的 speech-island boundary pipeline：ASR 前 chunk 要尽量接近一句台词，避免长连续 chunk、内部 gap、多 speech island 和非语音多送诱发 ASR 空输出、非语音幻觉和 forced aligner sentinel。
- 边界优先级：`start` 略高于 `end`，但两者都要进 gate；允许为了切准 speech island 牺牲少量 frame recall，但不能漏掉完整台词 island。
- 下一步应把 v1.20-v1.23 的经验收敛到 learned Boundary Refiner：显式优化 start/end error、fallback chunk duration、gap crossing、单 chunk 台词数和 ASR/aligner QC reward；recall 继续作为 guardrail，而不是唯一主目标。
- 现行 `tools/` 已按职责重构：`tools/asr/qwen/` 放 Qwen SFT，`tools/asr/diagnostics/` 放 ASR/alignment 诊断，`tools/boundary/` 放 Boundary Refiner 数据和训练，`tools/boundary/ja/` 放 SpeechBoundary-JA 训练评测，`tools/subtitles/` 放字幕 postprocess / cue planner / 审计校准，`tools/audits/` 放审计页与人工审计工具。旧历史段落里的 `tools/vad/fusionvad_ja/...` 路径保留为当时记录。

### SpeechBoundary-JA 下一步计划

1. 断兼容改名已完成：`src/vad/fusionvad_ja/` -> `src/boundary/ja/`，`tools/vad/fusionvad_ja/` -> `tools/boundary/ja/`，配置前缀 `FUSIONVAD_JA_` -> `SPEECH_BOUNDARY_JA_`，backend key `fusionvad_ja` -> `speech_boundary_ja`。旧 key / 旧 path / 旧 cache 不做 alias。
2. 数据格式升级：删除 gap-only / BiLSTM / endpoint-head 训练格式，把 Galgame 与 `joujiboi/japanese-anime-speech-v2` clean speech islands 重新生成 sequence dataset。每条样本包含多 island、touching speech、short/long gap、real negative gap、BGM/noise、轻量 overlap、source/utterance switch。
3. Learned refiner：只保留 `transformers.Mamba2Model` backbone。输入升级为连续窗口序列：Qwen PTM、MFCC、energy、speech_prob、cut_prob、candidate metadata。输出 split / merge / refine score 和可选 boundary offset。
4. Planner 接入：`pack_speech_segments()` 只 materialize planner 输出。planner 负责 start/end 权重、fallback-safe duration、gap-crossing penalty、最小/最大 chunk 约束和 ASR-facing span 输出。
5. 验收：synthetic exact truth 与匿名样片 A 双闭环。主 gate 是 start p90/p95、fallback chunk duration、long/gap-crossing chunk、ASR empty / hallucination、forced/partial 比例；chunk 数只作为成本指标。
6. 后续强化：supervised 稳定后再加入 preliminary ASR text、token confidence、local CER、aligner sentinel、fallback duration 和 QC reject 做 dense reward / DPO / RL。Unified Joint Model 继续放 backlog，等 SpeechBoundary-JA 能产出稳定 pseudo boundary labels 后再评估。

---

## 设计来源

### FusionVAD 复现路线

最初目标是复现 FusionVAD 的轻量结构，而不是直接把 WhisperSeg / FSMN / Silero 作为最终 VAD。核心思路：

```text
frozen PTM audio feature
+ MFCC / energy
-> addition fusion
-> 2-layer BiLSTM
-> lightweight heads
```

早期设想使用 Whisper-large-v3 encoder 冻结特征，后续为了体积、速度和分发体验，改为 Qwen3-ASR-0.6B full SFT 作为 frozen feature。这样用户后续只需要下载 fine-tuned 0.6B，不必同时保留 base 0.6B。

### Galgame 数据集的启发

人工复听后确认 `litagin/Galgame_Speech_ASR_16kHz` 多数 clip 本身已经按语音裁切。于是可以把原 clip 当作精确 speech island：

```text
random gap + speech clip + random gap + speech clip + ...
```

前置 gap 长度就是 speech start，`start + clip_duration` 就是 speech end。这个性质把 Galgame ASR 数据从弱监督正样本升级成了 synthetic timeline / boundary refiner 的核心数据底座。

### 目标域标注口径

Galgame / JAV 目标域里，喘息、呻吟、亲吻声、短促拟声可能本身就是字幕内容。因此当前 speech 定义不是传统 benchmark 的“清晰词句”，而是：

- 可字幕化对白、人声、喘息、呻吟、短促拟声：speech。
- 纯 BGM、静音、机械声、环境噪声、无字幕价值残留：non-speech。

这也是为什么当前 operating point 仍偏高召回：后端 ASR 和后处理可以过滤一部分多送音频，但漏掉真实目标域人声更难补救。

---

## 数据源与角色

- `litagin/Galgame_Speech_ASR_16kHz`：核心近域 ASR / VAD 来源，适合构造 synthetic speech island。
- `litagin/Galgame_Speech_SER_16kHz`：早期作为候选，后续放弃进入默认 full SFT；避免重复或衍生风险。
- `litagin/VisualNovel_Dataset_Metadata`：元数据候选，只作数据理解和去重参考。
- AVA-Speech：电影 speech activity 标注，首轮 supervised seed。
- VoxConverse：speaker/timestamp diarization 数据，多说话人 speech span seed。
- MUSAN / DNS Challenge：音乐、噪声、非语音负样本和增强素材。
- 本地视频 hard-negative：真实 BGM、静音、非语音人声、ASR 幻觉样本来源。
- `joujiboi/Galgame-VisualNovel-Reupload` 等视觉小说数据集：二期 backlog；进入训练前必须审计 license、字段、文本质量、去重和下载速度。

标签 schema 使用 JSONL：

```json
{
  "audio_id": "...",
  "source": "...",
  "duration_s": 0.0,
  "text": "...",
  "teacher_segments": [],
  "frame_hop_s": 0.02,
  "speech_frames": [],
  "label_quality": "supervised | teacher_agree | teacher_conflict | negative"
}
```

---

## Qwen3-ASR SFT 路线

### 1.7B full SFT

目标：让 ASR 能覆盖目标域中的对白、喘息、呻吟和短促拟声，降低“VAD 送进去了但 ASR 不认”的问题。

云端训练结论：

- 数据：`litagin/Galgame_Speech_ASR_16kHz` full ASR-only。
- 初始学习率：`2e-5`。
- effective batch：`128`。
- RTX 5090 32GB 曾用于 0.6B full SFT；RTX PRO 6000 96GB 用于 1.7B full SFT。
- 最终模型上传到 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`。

### 0.6B full SFT

目标：

- 替代 ja-whisper-anime 做更快的日语 ASR probe。
- 作为 FusionVAD-JA frozen feature extractor。
- 降低分发时的模型数量和空间成本。

结果：

- Galgame 16 clip direct probe 中，CER 从 base `0.2348` 降到 full `0.1288`。
- RTF 约 `0.232`。
- 最终模型上传到 `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`。
- 该模型成为 v1.13+ FusionVAD-JA 默认 frozen feature。

### 云端训练坑

- 数据集几十 GB，用户本地上传到云服务器太慢；更合理方式是在云端脚本直连 Hugging Face 下载并生成训练集。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 可缓解显存碎片，但不能弥补真实峰值不足。
- 5090 32GB 上 0.6B `batch_size=16`、`grad_acc=8` 曾在 step 36 OOM；稳定配置为 `batch_size=8`、`grad_acc=16`、effective batch `128`。
- 大 batch feature cache 在 WSL2 8GB RAM 下可能被系统 kill，没有 Python traceback；需要查内存和系统日志，不能只看显存。

---

## FusionVAD-JA 版本记录

### v0 / v1-mini

目的：验证数据、feature cache、addition-fusion BiLSTM 训练链路能跑通。

- 真实 ja whisper 1.5B feature cache 验证：`whisper_dim=1280`、`mfcc_dim=40`、`frame_hop_s=0.02`。
- 早期 addition-fusion BiLSTM 可训练参数约 `1.94M`。
- v1-mini 使用 VoxConverse supervised-positive + MUSAN / synthetic negative，只能证明训练闭环可行，不能代表目标域泛化。

### v1.5

目的：引入 Galgame synthetic timeline v2、MUSAN negative gap、背景混合和 positive loss weight。

结果：

- posw2 + threshold `0.001` + pad `0.2s` 在人工 Galgame 上 precision `0.7310`、recall `0.9501`、F1 `0.8263`。
- threshold `0.0001` + pad `0.2s` 达到 recall `0.9838`、extra audio ratio `1.3809`。
- 结论：低阈值 + padding 是当前 high-recall proposal 模式的必要选择。

### v1.6 real-heldout

目的：从本地视频抽真实 held-out，人工标注 VAD 片段，验证 Galgame synthetic 是否泛化。

数据：

- 10 个本地视频各抽 8 条、每条 8s。
- 候选 80 条，人工导出 79 条。
- 强标签：`supervised=55`、`negative=24`，总时长 `632.0s`，speech frame ratio `0.6138`。

基线：

- `fusion_lite`：F1 `0.7969`、precision `0.8534`、recall `0.7475`。
- `whisperseg-adaptive`：F1 `0.7697`、precision `0.7404`、recall `0.8015`。
- FusionVAD-JA v1.5 posw2 threshold `0.00015` + pad `0.2s`：recall `0.9551`、precision `0.6941`、F1 `0.8039`、extra audio ratio `1.3761`。

结论：FusionVAD-JA 达成高召回目标，但会把更多 negative / no-overlap 音频送给 ASR；后续必须用 ASR / alignment downstream 验证多送代价。

### v1.8 / v1.9 ASR 与 alignment 清理

问题暴露：

- 旧规则里存在具体词黑名单、假名/呻吟短句 direct drop、工具签名特例、AnimeWhisper 后置括号/重复清洗、翻译前重复压缩等非泛化策略。
- 这些规则会误伤目标域真实文本，例如 `はぁ`、`うん`、喘息、呻吟和短促发声。

处理：

- 删除词表驱动 direct drop。
- ASR QC 高风险文本改为 review-only，不再提供 `ASR_QC_DROP_UNCERTAIN` 删除开关，最终字幕不因该诊断被清空。
- 建立 `display_text` / `align_text` 双文本策略。
- alignment 诊断增加 `forced`、`partial`、`nonlexical`、`vad_coarse`、`proportional`、`drop_or_review`。
- 失败样本池统一用 `failure_candidate` 和 `failure_bucket`。

匿名样片 A 当前规则复测：

- base：`806` segments、`829` cues、`8085` chars、fallback `172/337`。
- 200k SFT：`794` segments、`843` cues、`13846` chars、fallback `166/337`。
- full checkpoint-15500：`802` segments、`870` cues、`15203` chars、fallback `170/337`。

结论：full SFT 方向成立，但主要瓶颈已经转向 alignment / fallback / QC，而不是 ASR 是否能输出文本。

### v1.10 / v1.11 synthetic timeline

v4 证明 crossfade、背景混合、overlap speech 和 `boundary_manifest.jsonl` bench 可用，但 gap 太短，speech frame ratio 约 `0.83-0.84`，不适合作为长期基线。

v5 long-gap 成为默认生成口径：

- train/val/test：`256/64/64` 条。
- speech frame ratio：`0.574/0.551/0.568`。
- 总时长 p50 约 `17s`，p90 约 `22s`。
- 默认启用长 gap、`speech_label_pad_s=0.08`、real negative gap 概率 `0.75`、背景混合概率 `0.5`。
- 支持 5-30ms equal-power crossfade、随机 gain、轻量 filter、低概率 codec、overlap speech。

v1.11 训练结果：

- 混合 v1-mini strong/negative `302` 条 + synthetic v5 `256` 条。
- val sweep 选 threshold `0.02`。
- test padded recall `0.9934`、missed speech `4.18s`、extra audio ratio `1.3240`。
- real-heldout recall 从 v1.5 的 `0.9556` 提升到 `0.9809`，missed speech 从 `17.24s` 降到 `7.42s`，extra audio ratio 升到 `1.5021`。

下游问题：

- 匿名样片 A 使用 v1.11 + Qwen3-ASR-1.7B full SFT checkpoint-21000，未做长段保护时只切出 `89` 个更长 VAD chunks。
- forced 仅 `14/89`，fallback `38/89`，failure candidates `76/89`。
- 主要 bucket 是 `empty_text_for_chunk` 和 `vad_coarse_alignment`。
- 结论：召回收益成立，但 chunk 边界/合并策略成为新瓶颈。

### v1.13

变化：切到 Qwen3-ASR-0.6B full SFT frozen feature，并把 synthetic v5 标签改为 exact speech-island。

synthetic exact-island test64：

- threshold `0.10` + pad `0.2s`：recall `0.9935`、missed `1.82s`、extra audio ratio `1.6012`。
- start/end p50 约 `0.628s/2.002s`。
- 主要收益是 start 边界更接近真实 speech island。

匿名样片 A downstream：

- 对比 v1.11 framepack baseline，chunks `240 -> 227`。
- fallback chunks `137 -> 114`。
- `vad_coarse_after_sentinel 122 -> 104`。
- forced `101 -> 106`。

结论：方向改善，但还不能替代后续 alignment repair。

### v1.14

变化：在 v1.13 上做 boundary-aware fine-tune，加入 `boundary_loss_weight=0.25` 和 `gap_loss_weight=0.10`。

结果：

- synthetic 上有信号。
- 真实 held-out 和匿名样片 A downstream 未过 gate。
- 匿名样片 A：chunks `222`、segments `986`、fallback chunks `115`、`vad_coarse_after_sentinel=103`、forced `101`。

结论：boundary-aware loss 方向保留，但 v1.14 不替换默认。

### v1.15

变化：明确改成 endpoint / boundary refiner，不再只是单头 speech VAD。

输出头：

- `speech`
- `start`
- `end`
- `cut`

训练目标：

- `speech` 继续服务 recall。
- `start/end` 学 speech island 边界。
- `cut` 学长 gap / 内部非语音可切点。
- 允许 end 偏长一点，但禁止 fallback chunk 长到 20-30s。

558-row checkpoint 未过 gate，但证明四头训练入口可行。

### v1.16

变化：扩大到 4096 条 Galgame multi-island synthetic。

结论：

- synthetic boundary gate 明显改善。
- 说明 synthetic exact-island 数据规模对 boundary refiner 有直接收益。

### v1.17

变化：使用 32768 条 synthetic 训练 endpoint refiner，并提交小型 checkpoint 到仓库。

当前默认：

```env
FUSIONVAD_JA_CHECKPOINT=src/vad/fusionvad_ja/checkpoints/fusionvad_ja_v1_17_endpoint_refiner.pt
FUSIONVAD_JA_MODEL_PATH=models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame
FUSIONVAD_JA_THRESHOLD=0.020
FUSIONVAD_JA_CUT_THRESHOLD=0.960
FUSIONVAD_JA_PAD_S=0.2
```

结论：

- v1.17 是默认 head 升级，能改善 synthetic boundary gate。
- 但匿名样片 A downstream 的 forced-aligner sentinel / unsafe fallback 未被根治。
- 后续重点不是继续提高 frame recall，而是 pre-ASR speech-island / boundary packing。

---

## Forced Alignment 与 Chunk Packing 记录

### R14 Phase 0

目标：确认 fallback 是否真的影响时间轴。

发现：

- `vad_coarse` 比 `forced` 时间轴差约 `2.16s p90`，超过门槛。
- forced 自身 p90 也约 `2.3s`，但主要来自 synthetic 真值边界模糊、crossfade/pad/transition 和 VAD pad，不代表 aligner 必然差。
- 根因链收敛为：high-recall VAD 把多 island + 长 gap 合成超长 chunk，Qwen forced aligner 在长 chunk + 大段非语音上吐 sentinel。

### R14 Phase 1a

尝试：`ASR_CHUNK_PACK_MAX_CORE_FRAMES=419`，只在长 gap 处拆超长 chunk。

结果：

- chunks `137 -> 148`，增长 `+8%`。
- forced `77 -> 84`。
- fallback `60 -> 64`。
- `vad_coarse_after_sentinel 25 -> 28`。
- 未过 gate。

结论：只在长 gap 处拆超长 chunk只能改善一部分粗时间轴误差，不能解决 sentinel。

### R14 Phase 1b

处理：`nonlexical` / `align_text_empty` 显式分流。

结果：

- 纯省略号/符号保留 display_text 并走粗时间轴。
- 不再计入真正 `vad_coarse` fallback。
- 剩余瓶颈集中在 `vad_coarse_after_sentinel` 非空文本块。

### R14 Phase 1c

尝试：`ALIGNMENT_SENTINEL_ISLAND_SPLIT=1`，只对 `vad_coarse_after_sentinel` 的非空文本 chunk 做 aligner-local speech-island splitting。

synthetic64：

- fallback `28 -> 11`。
- `vad_coarse_after_sentinel=11`。
- gate `PASS_RECLASSIFICATION_CLEANUP`。

匿名样片 A：

- chunks `240 -> 267`。
- forced `101 -> 154`。
- fallback `137 -> 48`。
- `vad_coarse_after_sentinel 122 -> 42`。

问题：

- 初版每个失败 chunk 串行卸载/重载 ASR 与 aligner。
- 耗时约 `1053.5s -> 3150.4s`。

### R14 Phase 1d

优化：staged batch island retry。

流程：

```text
收集全部 sentinel chunk
-> 一次性物化 island clips
-> 批量 ASR
-> 卸载 ASR
-> 批量 forced align
-> merge 回原 chunk
```

结果：

- synthetic64 指标与 Phase 1c 对齐。
- 匿名样片 A：forced `154`、fallback `48`、`vad_coarse_after_sentinel=42`。
- ASR+Alignment `1613.1s`、总计 `1641.7s`。

结论：适合作为 opt-in repair / 质量上限参考，不宜默认开启。

### R15 / R16

路线改为 pre-ASR speech-island / boundary-aware chunking：

- 不回到 FSMN。
- 不默认引入 pyannote。
- CAM++ / 3D-Speaker / WeSpeaker 只作为 speaker sidecar，在 speech-island 足够细后辅助判断相邻 island 是否跨 speaker。
- rule-based valley split 覆盖风险高，chunk 增幅过大，不进默认。
- 验收指标不只看 forced 数，还看 ASR empty、unsafe fallback、fallback duration、SRT 观感。

### R17

尝试：使用 v1.17 endpoint refiner 的 `cut` score 做 opt-in pre-ASR cut split。

离线样片 A：

- threshold `0.96`：chunks `241 -> 267`，增长 `1.11x`。
- threshold `0.95`：chunks `241 -> 287`，增长 `1.19x`。
- threshold `0.94`：chunks `241 -> 313`，增长 `1.30x`。

GPU 闭环 threshold `0.95`：

- chunks `241 -> 258`，增长 `+7.1%`。
- forced `105 -> 123`。
- `vad_coarse_after_sentinel 114 -> 113`。
- unsafe fallback `114 -> 109`。
- fallback safe ratio `0.0 -> 0.035`。
- fallback duration p90 仍为 `28.47s`。

结论：

- cut score 能改善少量 forced alignment。
- 它没有解决长连续 chunk 的粗 fallback。
- 不默认启用，不继续只扫 cut threshold。
- 下一步应转向更强的 pre-ASR boundary packing / endpoint refiner。

### R18

动机：R17 证明局部 cut score 有信号，但不能解决 sentinel / unsafe fallback。根因更像全局 packing 问题：长 chunk、多个 speech island、内部 gap、连续长人声、cut/valley 信号和 fallback 风险需要一起决策。

参考思路：

- WhisperX 的 VAD Cut & Merge 证明长音频 ASR 前需要显式切分和合并策略。
- Semantic VAD / endpoint detection 说明 endpoint 应作为独立目标，而不是只做 speech/non-speech。
- streaming ASR endpoint detection 的辅助 SAD / endpoint loss 思路适合迁移为 start/end/cut/risk 多头。

已落地的最小版本：

- 新增 env-gated `ASR_PRE_ASR_RISK_SPLIT_*`，默认关闭，不改变正式默认。
- 新增 `r18_pre_asr_risk_v1` policy：先计算 chunk fallback 风险，再选择边界。
- 风险因子：long core、unsafe duration、multi island、internal gap。
- 切点优先级：明确 internal gap -> endpoint cut score -> low VAD valley。
- 输出 metadata：`risk_split_count`、`risk_score`、`risk_reasons`，并进入 transcript chunk annotation 和 VAD chunk cache。
- `run_full_workflow.py` 已透传 R18 env，避免 GPU 闭环时参数只存在于父进程。

当前状态：

- 单测覆盖默认关闭、多 island 长 chunk 切分、连续长 chunk 使用 cut score、cache key 变化、cache round-trip、ASR stage env 透传和 full workflow env 透传。
- 这仍是 rule / cost packer 的第一版，不是最终模型。GPU 小闭环已经证明 R18 gap-first 没有显著解决 `vad_coarse_after_sentinel`、unsafe fallback 和 fallback p90。

实施验收记录：

- 代码范围：`src/audio/chunk_packer.py`、`src/whisper/pipeline.py`、`src/whisper/vad_chunk_cache.py`、`src/core/config.py`、`.env.example`、`tools/vad/fusionvad_ja/run_full_workflow.py`。
- 测试范围：`tests/test_chunk_packer.py`、`tests/test_vad_chunk_cache.py`、`tests/test_asr_stage_env_scope.py`、`tests/test_pipeline_chunk_config_runtime.py`、`tests/test_run_full_workflow_env.py`。
- 验收命令：`.venv/bin/python -m pytest tests/test_chunk_packer.py tests/test_vad_chunk_cache.py tests/test_run_full_workflow_env.py tests/test_asr_stage_env_scope.py tests/test_pipeline_chunk_config_runtime.py -q`。
- 结果：`44 passed`，仅有 Codex sandbox 内 NVML 初始化 warning；不影响 packing / cache / env 透传结论。
- README 决策：不更新。R18 仍是 opt-in 实验路线，默认关闭，不改变新用户安装、默认工作流或分发说明。

离线复算：

- 工具：`tools/vad/fusionvad_ja/analyze_r18_risk_splits.py`。
- 输入：匿名样片 A v1.17 endpoint-refiner 的 VAD cache、diagnostics、R17 frame/cut score。
- 输出：`agents/temp/fusionvad-ja/r18-risk-split-offline-sample-a*/summary.json`、`summary.md`、`risk_split_plan.jsonl`、`simulated_chunks.jsonl`。
- 口径限制：离线复算只重打 cached VAD segments，并用 core overlap 把旧 diagnostics 映射到新 chunk；不跑 ASR / forced aligner，因此只能评估 chunk 分布和风险覆盖，不能替代 GPU 闭环。

| 参数 | chunks | 增幅 | sentinel 风险旧 chunk 被拆 | duration p50/p90 | 结论 |
|------|--------|------|-----------------------------|------------------|------|
| 默认 R18 `risk=1.0,gap=6` | `241 -> 372` | `1.544x` | `56/114` | `16.34/28.47s` | 覆盖较多，但 chunk +54%，过激，不适合作为默认 GPU 闭环起点。 |
| 保守 `risk=2.0,gap=6/12/18` | `241 -> 268-269` | `1.112-1.116x` | `11/114` | `27.00/28.47s` | 成本可控，但只覆盖少数 sentinel 风险。 |
| 更保守 `risk=2.5,gap=6/12/18` | `241 -> 253-258` | `1.050-1.071x` | `2-6/114` | `27.29/28.47s` | 太保守，对粗 fallback 基本无杠杆。 |

结论：

- 只靠 R18 的 rule/cost packing 不能十拿九稳解决 p90 粗 fallback；一旦控制 chunk 增幅，能处理的主要是少量多 island / internal gap chunk。
- 剩余瓶颈主要落在连续长 island / overlong chunk：旧 cache 中大量 chunk 本身已被 hard-cap overlong 切到接近 `28.47s`，即使重新 packing，fallback duration p90 仍不动。
- 下一步不建议直接用默认 R18 跑 GPU；更合理路线是继续 endpoint/boundary refiner，让模型在连续长 island 内提供更可靠 cut/valley 信号，或把 R18 改成更明确的 fallback-risk objective 后再做小闭环。
- 当前 R18 保持 opt-in、默认关闭。

Netflix / WhisperX 复核后的策略修正：

- Netflix timed text 规则强调 cue 需要贴近对白起点、保持可读时长、字幕间保留最小 gap；通用字幕 event 通常不应长期接近 `20-30s`。日语规则还有更严格的行长和读速限制。
- WhisperX 类长音频 ASR 路线允许 ASR chunk 接近 `30s`，但那依赖后续 word-level forced alignment。当前项目在 forced aligner sentinel 时只能退回 chunk 粗时间轴，所以 ASR chunk 过长会直接污染 fallback 字幕。
- 因此 R18 不应简单照搬“30s ASR chunk 合法”或“7s 字幕 cue 最大”任何一边，而应先把明显多句、多 island、内部 gap 的 chunk 拆开；连续长语音则保守处理。
- 已新增 `ASR_PRE_ASR_RISK_SPLIT_CONTINUOUS_THRESHOLD=2.0`：`risk=1.0,gap=6` 时仍会积极切明确内部 gap，但没有内部 gap 的连续长 island 需要更高风险分才允许使用 endpoint cut / VAD valley 切。
- 该改动保持 R18 默认关闭，但改变 R18 opt-in 行为和 VAD chunk cache signature。

gap-first 离线复算：

| 参数 | chunks | 增幅 | sentinel 风险旧 chunk 被拆 | duration p50/p90 | 结论 |
|------|--------|------|-----------------------------|------------------|------|
| `risk=1.0,continuous=2.0,gap=6` | `241 -> 269` | `1.116x` | `11/114` | `27.00/28.47s` | 新默认实验档，主要只切明确 gap；成本可控。 |
| `risk=1.0,continuous=1.5,gap=6` | `241 -> 372` | `1.544x` | `56/114` | `16.34/28.47s` | 连续长 island 也被 cut score 大量切，回到旧激进行为。 |
| `risk=1.0,continuous=2.5,gap=6` | `241 -> 258` | `1.071x` | `6/114` | `27.29/28.47s` | 太保守，收益更小。 |

GPU 闭环：

- 命令脚本：`agents/temp/run_r18_gapfirst_sample_a_gpu.sh`。
- 工作流输出：`agents/temp/fusionvad-ja/full-workflow-qwen29239-sample-a-v1-17-r18-gapfirst/`。
- 诊断输出：`agents/temp/fusionvad-ja/diagnostics-sample-a-v1-17-r18-gapfirst/`。
- fallback-safe 指标：`agents/temp/fusionvad-ja/fallback-safe-boundary-metrics-sample-a-v1-17-r18-gapfirst/`。
- 对比表：`agents/temp/fusionvad-ja/r18-gapfirst-gpu-compare/summary.md`。
- VAD 日志确认使用 CUDA：`requested_device=cuda actual_device=cuda`。
- 全片匿名样片 A 耗时 `1116.54s`，ASR chunks `250`，输出 segments `899`。

| 版本 | chunks | forced | `vad_coarse_after_sentinel` | unsafe fallback | fallback p50/p90/max | ASR empty warn | QC reject |
|------|--------|--------|-----------------------------|-----------------|----------------------|----------------|-----------|
| baseline v1.17 | `241` | `105` | `114` | `114` | `28.47 / 28.47 / 28.47s` | `6` | `16` |
| R17 cut th0.95 | `258` | `123` | `113` | `109` | `27.41 / 28.47 / 28.47s` | `5` | `17` |
| R18 gap-first | `250` | `109` | `117` | `114` | `28.47 / 28.47 / 28.47s` | `8` | `16` |

结论：

- `risk=1.0,continuous=2.0,gap=6` 的 chunk 增幅可控，但真实 GPU 闭环没有过 gate：forced 只从 baseline `105 -> 109`，sentinel `114 -> 117`，unsafe fallback 不变，fallback p90 不变。
- R18 规则没有触达核心瓶颈。多数粗 fallback 仍是接近 hard-cap 的长连续 chunk / overlong chunk；只切明确 gap 的收益太小，而无差别切连续长 island 又依赖当前还不够可靠的 cut/valley 信号。
- R18 保持 opt-in、默认关闭。不建议继续围绕 `risk/gap/continuous` 做大规模扫参；下一步应训练更强的 endpoint / boundary refiner，或设计 fallback-risk objective，让模型直接学习“哪里适合切成一句台词”。

R18 后续 cut-signal 离线审计：

- 工具：`tools/vad/fusionvad_ja/analyze_fallback_cut_signal.py`。
- 输入：全量 `chunk_metrics.jsonl`，不是 `unsafe_fallback_chunks.jsonl` top-N；后者只保留最长 20 条审计样本。
- 目的：确认现有 v1.17 的 `speech/cut` 概率在 unsafe fallback chunk 内是否已经包含足够切点。如果已有信号足够，说明 packer 阈值还有空间；如果信号不够，说明必须改训练目标。
- 输出：
  - `agents/temp/fusionvad-ja/fallback-cut-signal-sample-a-v1-17-baseline-full/`
  - `agents/temp/fusionvad-ja/fallback-cut-signal-sample-a-v1-17-r17-full/`
  - `agents/temp/fusionvad-ja/fallback-cut-signal-sample-a-v1-17-r18-full/`

| 版本 | unsafe rows | 有 cut/valley 候选 | 可贪心拆到 9s 子 chunk | greedy 后 max-child p90 |
|------|-------------|--------------------|-------------------------|-------------------------|
| baseline v1.17 | `114` | `69` | `7` | `24.47s` |
| R17 cut th0.95 | `109` | `55` | `9` | `24.47s` |
| R18 gap-first | `114` | `67` | `8` | `24.47s` |

结论：现有 v1.17 cut/valley 信号只能覆盖约 `6-8%` 的 unsafe fallback，且 p90 子 chunk 时长不动。继续扫 R17/R18 规则阈值不是主杠杆；下一步应训练新的 boundary/cut head，使目标直接服务 fallback-safe “一句台词边界”，尤其是连续长 island 内的可切点。

### R19 / v1.18 训练目标调整

动机：R18 后的全量 cut-signal 审计证明，现有 v1.17 cut head 在 unsafe fallback chunk 内缺少足够可用切点。继续调 packer 只是在不存在的信号上扫阈值。

最小实现：

- 新增 `EndpointRefinerTrainConfig.cut_boundary_radius_frames`，默认 `0`，不改变 v1.17 旧行为。
- `endpoint_targets_from_record()` 仍保留原逻辑：长 gap（`gap >= cut_min_gap_s`）整段标为 cut。
- 当 `gap < cut_min_gap_s` 且 `cut_boundary_radius_frames > 0` 时，把相邻 speech island 的 `previous.end` / `current.start` 附近若干帧也标为 cut 正样本。
- CLI 新增 `tools/vad/fusionvad_ja/train_endpoint_refiner.py --cut-boundary-radius-frames`。

目的：

- v1.18 训练不再只让 cut 学“大段静音 gap”，还要让 cut/head 学“相邻台词边界附近可以切”。
- 这直接服务 fallback-safe chunk packing：即使 VAD high-recall 把连续人声或短 gap 合成一坨，也希望 cut head 能提供更密集、可解释的句子边界候选。
- 默认仍关闭，直到新 checkpoint 通过 synthetic boundary gate 与匿名样片 A GPU downstream gate。

验收：

- `tests/test_fusionvad_ja_dataset.py` 覆盖短 gap boundary cut target 与 CLI 参数校验。
- smoke：`agents/temp/fusionvad-ja/v1-18-cut-boundary-radius-smoke/`，1 step 训练成功，checkpoint config 写入 `cut_boundary_radius_frames=1`。

训练与 test64 阈值扫描：

- 训练脚本：`agents/temp/run_v1_18_cutboundary2_train.sh`。
- checkpoint：`datasets/train/fusionvad-ja/v1-18/qwen3-asr-0.6b-full29239/endpoint-refiner-boundary32768-cutboundary2-batch16-lr2e-4-steps2048-posaux120-cut8-nogap/fusionvad_ja_endpoint_refiner.pt`。
- 训练集：32768 条 Galgame synthetic exact-island / long-gap 样本，使用 Qwen3-ASR-0.6B full SFT frozen feature。
- 训练参数：batch 16，lr `2e-4`，2048 steps，`cut_boundary_radius_frames=2`，`positive_aux_weight=120`，`internal_gap_loss_weight=0`。
- 训练结果：final loss `0.8426`，frame accuracy `0.9378`，trainable params `1889252`。
- 预测导出：
  - v1.18：`agents/temp/fusionvad-ja/v1-18-cutboundary2-test64-predictions-th002-cut05/`
  - v1.17 对照：`agents/temp/fusionvad-ja/v1-17-test64-predictions-th002-cut05/`
- 阈值扫描：
  - v1.18 no-cut：`agents/temp/fusionvad-ja/v1-18-cutboundary2-threshold-sweep/`
  - v1.18 cut-applied：`agents/temp/fusionvad-ja/v1-18-cutboundary2-threshold-sweep-cut-applied/`

关键对比：

| 版本 / operating point | recall | missed speech | extra audio ratio | segments | start p50 | end p50 | cut gap coverage |
|------------------------|--------|---------------|-------------------|----------|-----------|---------|------------------|
| v1.17 `speech=0.020,cut=0.960,cut-applied` | `0.999992` | `0.005s` | `1.2545` | `192` | `0.305s` | `0.298s` | `0.984` |
| v1.18 no-cut best recall-safe `speech=0.030` | `1.000000` | `0.000s` | `1.3674` | `190` | `0.552s` | `0.776s` | `1.000` |
| v1.18 cut-applied best recall-safe `speech=0.030,cut=0.960` | `1.000000` | `0.000s` | `1.3329` | `193` | `0.484s` | `0.632s` | `0.905` |
| v1.18 cut-applied `recall>=0.999` best | `0.999623` | `0.229s` | `1.3152` | `195` | `0.411s` | `0.572s` | `0.937` |

结论：

- v1.18 的 cut-boundary 目标确实让 cut 更积极，但在 synthetic boundary gate 上没有超过 v1.17。
- 即使放宽到 `recall>=0.999`，v1.18 仍比 v1.17 多送音频、边界更粗。
- 不替换默认 head，不进入匿名样片 A GPU downstream 闭环。
- 失败原因更像训练目标仍不够直接：短 gap boundary 正样本会增加 cut 密度，但没有明确约束“fallback-safe 子 chunk 时长 / 一句台词边界 / 避免长连续 island 粗 fallback”。
- 下一步应设计 v1.19：显式 fallback-risk / max-child-duration / sentence-island objective，或做一个 post-VAD boundary proposal 模型，而不是继续在 v1.18 上扫阈值。

补充 fallback-safe synthetic gate：

- `tools/vad/fusionvad_ja/benchmark_boundary_predictions.py` 新增预测段级指标：
  - `fallback_target_duration_s`，默认 `8.0s`。
  - `fallback_gap_overlap_s`，默认 `0.5s`。
  - `long_predicted_segment_count / ratio`。
  - `predicted_gap_crossing_segment_count / ratio`。
  - `predicted_segment_duration` p50/p90/p95/max。
  - `predicted_gap_overlap` p50/p90/p95/max。
- 目的：模拟 forced aligner sentinel 时的最坏情况。如果某个 VAD/packing operating point fallback 后会生成过长 cue 或跨大段 truth gap，即使 frame recall 高也不能算过 gate。
- 新 sweep 产物：
  - v1.17：`agents/temp/fusionvad-ja/v1-17-endpoint-refiner-threshold-sweep-cut-applied-fallbacksafe/`
  - v1.18 no-cut：`agents/temp/fusionvad-ja/v1-18-cutboundary2-threshold-sweep-fallbacksafe/`
  - v1.18 cut-applied：`agents/temp/fusionvad-ja/v1-18-cutboundary2-threshold-sweep-cut-applied-fallbacksafe/`

fallback-safe 对比（`recall>=0.9999`）：

| 版本 / operating point | recall | extra | segments | long segments | gap crossing | pred dur p90/max | start/end p50 |
|------------------------|--------|-------|----------|---------------|--------------|------------------|---------------|
| v1.17 `speech=0.040,cut=0.960,cut-applied` | `0.999992` | `1.2063` | `171` | `29` | `13` | `8.620/12.225s` | `0.300/0.281s` |
| v1.18 no-cut `speech=0.030` | `1.000000` | `1.3674` | `190` | `40` | `67` | `9.222/17.940s` | `0.552/0.776s` |
| v1.18 cut-applied `speech=0.030,cut=0.960` | `1.000000` | `1.3329` | `193` | `37` | `48` | `8.996/17.940s` | `0.484/0.632s` |

结论：

- v1.18 的问题不是单纯阈值；在 fallback-safe 指标下也明显劣于 v1.17。
- v1.17 的 `speech=0.04,cut=0.96` synthetic gate 甚至优于早期 `speech=0.02,cut=0.96`，但是否替换默认 operating point 需要真实样片 GPU 闭环验证，不能只凭 synthetic gate。
- v1.19 训练目标应直接减少 `long_predicted_segment_count` 和 `predicted_gap_crossing_segment_count`，同时守住 near-1 recall。

匿名样片 A GPU 验证 `v1.17 speech=0.04,cut=0.96`：

- 脚本：`agents/temp/run_v1_17_th04_sample_a_gpu.sh`。
- workflow：`agents/temp/fusionvad-ja/full-workflow-qwen29239-sample-a-v1-17-th04-cut096/`。
- diagnostics：`agents/temp/fusionvad-ja/diagnostics-sample-a-v1-17-th04-cut096/`。
- fallback-safe metrics：`agents/temp/fusionvad-ja/fallback-safe-boundary-metrics-sample-a-v1-17-th04-cut096/`。
- 日志确认 VAD 使用 CUDA：`requested_device=cuda actual_device=cuda`。
- 全片耗时 `1056.7s`，ASR chunks `241`，字幕 segments `884`。

| 版本 | chunks | forced | partial | nonlexical | `vad_coarse_after_sentinel` | unsafe fallback | fallback p50/p90/max | ASR empty warn | QC reject |
|------|--------|--------|---------|------------|-----------------------------|-----------------|----------------------|----------------|-----------|
| baseline v1.17 `0.02/0.96` | `241` | `105` | `0` | `6` | `114` | `114` | `28.47 / 28.47 / 28.47s` | `6` | `16` |
| v1.17 `0.04/0.96` | `241` | `108` | `1` | `6` | `113` | `113` | `28.47 / 28.47 / 28.47s` | `6` | `13` |
| R17 cut th0.95 | `258` | `123` | `0` | `5` | `113` | `109` | `27.41 / 28.47 / 28.47s` | `5` | `17` |
| R18 gap-first | `250` | `109` | `0` | `8` | `117` | `114` | `28.47 / 28.47 / 28.47s` | `8` | `16` |

结论：

- `0.04/0.96` 在 synthetic gate 上更漂亮，但真实样片只带来很小变化：forced `105 -> 108`、sentinel/unsafe `114 -> 113`、fallback p90 不变。
- 不替换默认 operating point；继续保持 v1.17 默认，并把 `0.04/0.96` 记录为 synthetic 优但真实闭环收益不足的负例。
- 这进一步确认主瓶颈不是全局 speech threshold，而是 long overlong chunk / continuous island 内缺少可靠句边界。

---

## ASR / Alignment 文本策略

当前策略来自 v1.8 / v1.9 的清理。

原则：

- `display_text` 是最终字幕显示文本，只做展示安全处理。
- `align_text` 是 forced aligner 专用文本，可删除标点、emoji、装饰符、音乐符号和不可发音标记。
- 不使用具体字样黑名单。
- 不直接删除目标域常见短促发声、喘息、呻吟、拟声和低信息短句。
- 重复循环、低置信、文本/音频比例异常、align-text-empty、forced-aligner fallback、`asr_review_uncertain` 默认只作为 QC / 诊断 / 样本池信号，不再触发最终字幕文本删除。
- forced aligner 失败时不伪造精确时间轴，保留 fallback quality label。

失败样本池闭环：

```text
diagnose_asr_alignment.py
-> failure_candidates.jsonl
-> export_alignment_failure_manifest.py
-> materialize_alignment_failure_audio.py
-> 人工审计 / hard-negative / 下轮 VAD 或 ASR 数据
```

---

## 字幕时间轴

时间轴策略来自 Netflix / 字幕行业实践的简化适配：

- 每个任务用 `ffprobe` 读取真实 FPS。
- 失败时按 `30000/1001` 兜底。
- cue plan 在 LLM 翻译前固定。
- 最小字幕 gap 为 2 帧。
- 短尴尬 gap 可折叠为 2 帧。
- 真实停顿保留。
- 前一条 cue 可适度 linger，但必须受最大时长约束。

关键结论：

- ASR 输出文本时，start 边界比 end 更重要。
- end 偏长可以在 cue timing polish 中压缩，尤其是两条字幕相邻时可以压缩前者 end 来保留 gap。
- 但如果 VAD / chunk 本身跨了大段无声，后置 timing polish 无法修复 ASR 幻觉或 forced aligner coarse fallback。

### R19 · Reward-shaped speech-island segmentation

用户提出：能否用强化学习做 speech-island 划分，把“时间轴太粗”和“多个 speech island 中间夹 gap / 白噪声 / BGM / 空白却被合成一段”作为强惩罚；切点越接近 speech start 越加分，end 也加分但权重低于 start。

检索与判断：

- 方向成立。已有类似“用 RL 学 speech boundary”的研究，例如 REBORN（Reinforcement-Learned Boundary Segmentation with Iterative Training for Unsupervised ASR，NeurIPS 2024）用 RL 优化语音边界，使无监督 ASR 的 phoneme perplexity 更好。
- 但 REBORN 的 reward 服务于无监督 ASR，不直接服务本项目的 fallback-safe subtitle boundary。我们当前目标更具体：forced aligner 失败时，fallback chunk 不能是 20-30s 的粗时间轴，也不能跨大段 truth gap。
- 因为 Galgame synthetic exact-island 已提供精确 speech-island 真值，第一版不应直接上 REINFORCE/PPO。直接 RL 会引入稀疏 reward、训练不稳定、reward hacking 和 GPU 闭环成本高的问题。

决策：

- v1.19 先做 **reward-shaped structured segmentation**，借用 RL 的 reward 设计，但用确定性 DP / beam search 或离线 cost planner 选择 cut。
- 先离线验证 reward 是否抓住当前瓶颈；只有 synthetic fallback-safe gate 和匿名样片 A GPU 小闭环都证明有效，才考虑接入主 pipeline 或进一步训练 boundary/refiner。

v1.19 reward 初稿：

- 强惩罚：预测段跨 `>=0.5s` truth gap。
- 强惩罚：fallback 子段超过 `8-9s`。
- 中惩罚：chunk 数暴涨、切得太碎、子段短于最小可读/可识别时长。
- 奖励：切点靠近真实 speech start/end；start 权重大于 end。
- 保护项：不漏掉完整台词 island。frame recall 不再是主优化目标，允许为切准 start/end 和减少长 fallback chunk 牺牲少量 recall。

实施顺序：

1. 新增离线 planner / evaluator：读取 `boundary_manifest.jsonl` + endpoint prediction probabilities，在候选 cut 上用 reward/cost 选切分，输出 fallback-safe 指标。
2. 在 test64 上对比 v1.17 baseline / R17 / R18 / v1.18，先判断 reward 是否能显著降低 `long_predicted_segment_count` 和 `predicted_gap_crossing_segment_count`。
3. 若离线成立，再把 planner 的候选生成逻辑迁移到 `chunk_packer.py` 的 opt-in R19 开关；默认仍关闭。
4. 若真实样片仍缺可用 cut 信号，再训练 boundary/refiner：目标不再只是 speech mask，而是直接优化 fallback-safe 子段、truth-gap crossing、单句台词 chunk 和 start-biased boundary。

首轮离线实现：

- 新增 `tools/vad/fusionvad_ja/plan_reward_boundary_segments.py`。
- 输入：synthetic `boundary_manifest.jsonl` + endpoint prediction probabilities。
- 输出：`summary.json`、`plan_details.jsonl`。
- 支持三种 candidate source：
  - `probability`：当前模型的 cut / endpoint / valley 概率。
  - `oracle`：synthetic truth gap，仅用于上限分析。
  - `hybrid`：两者合并，用于确认 reward 方向。
- 关键修正：R19 不能只选“切点”。对大 gap / cut / valley，应支持删除一个 cut zone；但 endpoint/start/end 只能做切点，不能删除音频，否则会误切 speech。这个约束来自单测暴露的问题。

test64 结果（v1.17 predictions，baseline 为 `speech=0.02`、pad `0.2s`、merge gap `0.15s`）：

| 方案 | recall | missed speech | segments | long segments | gap crossing | dur p90/max | extra |
|------|--------|---------------|----------|---------------|--------------|-------------|-------|
| baseline | `1.000000` | `0.000s` | `190` | `31` | `27` | `8.826/17.140s` | `1.2809` |
| R19 probability v2 | `0.999234` | `0.465s` | `223` | `21` | `28` | `7.996/9.760s` | `1.2709` |
| R19 oracle truth-cost | `1.000000` | `0.000s` | `195` | `29` | `22` | `8.544/12.285s` | `1.2595` |
| R19 hybrid truth-cost | `0.999506` | `0.300s` | `223` | `21` | `22` | 见产物 | `1.2587` |

产物：

- `agents/temp/fusionvad-ja/r19-reward-boundary-plan-v1-17-probability-test64-v2/`
- `agents/temp/fusionvad-ja/r19-reward-boundary-plan-v1-17-oracle-test64-v2/`
- `agents/temp/fusionvad-ja/r19-reward-boundary-plan-v1-17-hybrid-truthcost-test64/`

结论：

- reward-shaped planner 的目标函数方向成立：可以明显压低 long predicted segment，并在有 truth gap/cut zone 信号时降低 gap crossing。
- 但当前 v1.17 概率候选不能稳健上线：它能把 long segment `31 -> 21`，但 gap crossing 没降，且会带来少量 missed speech。
- 因此下一步不是把 R19 planner 直接接主 pipeline，而是用它生成/评估 v1.19 训练目标：让 boundary/refiner 学到“可删除的 gap/cut zone”和“只可切不可删的 endpoint”，并显式优化 fallback-safe metrics。

### R19 数据升级：speaker-random synthetic timeline

用户提出：除了在 speech island 中间拼接 gap / 白噪声 / BGM，还应把不同人的 Galgame 语音随机串联在一起，训练模型识别“换人/换声线的 speech boundary”。如果有性别、角色或声优标注，优先让相邻 island 来自不同性别/角色/声优。

检索与判断：

- speaker change detection 领域已有类似做法：把不同说话人的短语音拼成 synthetic conversation，用来训练 speaker-change boundary。这个和 Galgame exact-island 构造天然匹配。
- 但本项目不应把目标写成“识别男女/角色”，否则会回到已降级的 F0/gender 路线；目标应是 speaker-turn / utterance boundary，即“可切，不一定可删”。
- `VisualNovel_Dataset_Metadata` 可能提供角色/声优元数据，但 `Galgame_Speech_ASR_16kHz` 当前本地 materialized manifest 不一定能直接映射到角色。因此 v1.19 第一版先用 `speaker_proxy_id` 占位：默认从 manifest 字段读取；没有字段时退化为 audio/hash 级 proxy。后续可接 CAM++ / 3D-Speaker / WeSpeaker 聚类填充 proxy。

设计：

- `cut_drop_zone`：中间是 silence / white noise / hum / BGM / real negative gap，可删除。
- `cut_point`：相邻 speech island 几乎无 gap 或短 gap，但换 speaker proxy / source audio，只能切分，不能删除音频。
- synthetic timeline 内部 speech island 应支持随机采样，而不是只按 manifest 顺序取连续样本。
- 每条输出显式记录：
  - `speaker_proxy_ids`
  - `speaker_turn_boundaries`
  - `cut_point_segments`
  - `cut_drop_zones`
  - `source_audio_ids`

执行策略：

1. 先给 `build_galgame_synthetic_timeline.py` 增加 opt-in 随机 speech island 采样和 speaker proxy 元数据。
2. 小样本 smoke 确认 manifest / boundary_manifest 能记录 speaker turn 和 gap zone。
3. 后续再把这些 targets 接到 v1.19 训练：增加 `cut_point` / `cut_drop_zone` 双头或在现有 cut head 上拆 target。

实施进展：

- `build_galgame_synthetic_timeline.py` 已新增 opt-in `--randomize-speech-order`、`--speaker-proxy-mode`、`--speaker-proxy-retry-count`、`--cut-point-max-gap-s`、`--cut-drop-min-gap-s`。
- 输出 manifest / boundary manifest / labels 均记录 `speaker_proxy_ids`、`speaker_turn_boundaries`、`cut_point_segments`、`cut_drop_zones`。其中 labels 通过 `boundary_metadata` 保存这些训练目标，避免只在旁路 manifest 中可见。
- `endpoint_targets_from_record()` 已读取 `boundary_metadata`：`cut_drop_zones` 标整段可删除 gap，`cut_point_segments` 标短半径切点；暂时复用现有 `cut` head，不马上拆模型结构，减少 v1.17 checkpoint 兼容风险。
- 小样本 smoke 已覆盖随机 speaker boundary：`test_build_galgame_synthetic_timeline_records_speaker_random_boundaries`；目标读取覆盖：`test_endpoint_targets_use_explicit_cut_metadata`。
- 设计坑：不能把 speaker turn 当作可删除区域。gap/noise/BGM/silence 是 `cut_drop_zone`，可从 fallback chunk 中删；换人/换 source 的连续 utterance 是 `cut_point`，只能切分，不能删音频。

连续 speech-island 修正：

- 第一版 v8 4096 数据虽然 `--gap-min-s 0`，但连续/短 gap 样本比例太低：8192 个内部边界里 `gap <= 0.12s` 只有 244 个，实际训练会偏向长 gap 删除。
- 新增显式分布控制：
  - `--touch-gap-prob`：内部 speech island 之间完全 0-sample 贴连。
  - `--short-gap-prob` + `--short-gap-max-s`：内部 speech island 之间采样 0 到短 gap 上限。
- 重新生成 `galgame-synthetic-timeline-v8-speaker-random-touch4096-train`：4096 条、8192 个内部边界；`touch=2109`、`short=2104`、`regular=3979`；`cut_point_segments=4213`、`cut_drop_zones=3965`、`ambiguous_gap=14`。这版才真正覆盖“连续多 speech island 拼接，中间没有 gap 或只有极短 gap”的训练目标。
- `build_exact_island_labels.py` 已保留 boundary metadata；否则 exact-island 转换会丢掉 `cut_point_segments` / `cut_drop_zones`，导致 v1.19 训练实际没有学到新目标。

v1.19 touch4096 smoke：

- 生成 feature cache：`datasets/train/fusionvad-ja/v1-19/qwen3-asr-0.6b-full29239/galgame-synthetic-timeline-v8-speaker-random-touch4096-feature-cache/feature_manifest.json`，CUDA + bf16，4096/4096 cached，0 errors。
- 训练：`endpoint-refiner-touch4096-batch16-lr2e-4-steps1024-posaux120-cut8`，1024 steps，loss `1.5212`，frame accuracy `0.9133`，trainable params `1,889,252`。
- test64 直接 speech mask（speech=0.02, cut=0.5）不达标：recall `0.9974`，missed `1.48s`，extra ratio `1.4919`，long segments `41`，gap-crossing `80`。相比 v1.17 baseline（recall `1.0`、extra `1.2809`、long `31`、gap-crossing `27`），speech mask 明显更宽，不能默认替换。
- 但 cut signal 对 R19 planner 有用：probability planner 把 long `41 -> 7`，gap-crossing `80 -> 71`，extra `1.4919 -> 1.4129`，但 recall 掉到 `0.9648`；hybrid truth-cost 把 gap-crossing 降到 `67`，recall `0.9707`。结论：贴连/短 gap 数据方向成立，但 `cut` 和 `speech` 仍相互污染，复用单一 cut head 不够稳定。
- 下一步 v1.19b：拆 `cut_drop` / `cut_point` 双目标或至少在 loss 上分权；同时加 recall guard / speech mask regularization，避免为了学 cut 牺牲 frame recall 和扩大 extra audio。

v1.19b split-cut 默认候选：

- 按“直接替换默认，不保留旧 4-head 兼容”的方向重构 endpoint refiner：输出从 `speech/start/end/cut` 改为 `speech/start/end/cut_drop/cut_point`。
- `cut_drop` 表示 silence / white noise / hum / BGM / real negative gap，可从 fallback chunk 中删除；`cut_point` 表示贴连台词、短 gap 或换 speaker/source 的 utterance boundary，只能切分不能删除音频。
- 训练：`datasets/train/fusionvad-ja/v1-19/qwen3-asr-0.6b-full29239/endpoint-refiner-splitcut-touch4096-batch16-lr2e-4-steps1024-posaux120-cut16/`，CUDA，1024 steps，batch 16，lr `2e-4`，trainable params `1,889,349`，loss `1.6056`，frame accuracy `0.9138`。
- 导出：`agents/temp/fusionvad-ja/v1-19b-splitcut-touch4096-step1024-predictions-th002-cut05/`，speech F1 `0.9282`，precision `0.8687`，recall `0.9963`；`cut_drop` F1 `0.6049` / recall `0.9679`，`cut_point` F1 `0.0519` / recall `0.2229`。
- Synthetic boundary benchmark at speech threshold `0.02`：`agents/temp/fusionvad-ja/v1-19b-splitcut-touch4096-step1024-boundary-benchmark/`，speech-duration recall `0.99794`，missed speech `121.59s`，extra audio ratio `1.2199`，predicted segments `12560`，long segments `4216`，gap-crossing segments `3639`，p50/p90 predicted segment duration `4.12s/14.96s`。
- Threshold sweep：`agents/temp/fusionvad-ja/v1-19b-threshold-sweep/threshold_sweep_summary.json`。`speech=0.02` 和 `speech=0.10` 都会让 1 秒纯静音末尾触发短段，默认不采用；`speech=0.20` 保留 synthetic recall `0.98937`，extra audio ratio 降到 `1.08859`，gap-crossing 降到 `1409`，作为 v1.19b 默认 operating point。后续用真实 held-out 再决定是否向 `0.15` 或 `0.10` 回调。
- R19 planner 仍不默认开启：`agents/temp/fusionvad-ja/r19-reward-boundary-plan-v1-19b-step1024-probability/` 把 segments `12560 -> 19558`、long `4216 -> 1252`、gap-cross `3639 -> 3233`，但 recall 降到 `0.943963`，且切分数量暴涨；按新的 boundary-first 主线，它是有用的离线 teacher / 上限分析，不是可直接上线的默认策略。
- 默认 checkpoint 已切到 `src/vad/fusionvad_ja/checkpoints/fusionvad_ja_v1_19b_splitcut_touch4096_endpoint_refiner.pt`，默认阈值 `speech=0.20`、`cut_drop/cut_point=0.50`；旧 v1.17 checkpoint 暂留在目录中作为历史产物，不作为默认。

### 主线切换：boundary-first VAD

用户明确调整目标：不一定非要保持高召回主线。当前 VAD 主目标改为 speech-island 边界切准，尽量“一句台词一个 chunk”。`start` 边界比 `end` 略重要，但两者都重要。`end` 偏长可以被字幕 timing polish 适度压缩；`start` 偏晚会直接漏掉台词开头，`start` 偏早则更容易把静音/BGM/噪声送进 ASR 诱发幻觉。

新的验收优先级：

1. start/end boundary error，start 权重略高。
2. fallback chunk duration p50/p90/max，禁止 20-30s 粗 fallback。
3. predicted gap crossing 与 gap overlap。
4. 单 chunk 台词数 / speech island 数，目标是一句台词一个 chunk。
5. ASR empty / hallucination proxy。
6. frame recall 只作为 guardrail：不能漏完整台词 island，但允许为边界质量牺牲少量帧级 recall。

v1.20 训练方向：

- 数据继续使用 Galgame exact-island synthetic timeline，但提高连续/短 gap、多 speaker/source 拼接、BGM/噪声贯穿、真实 negative gap 的比例。
- loss 从 “speech BCE + boundary/cut auxiliary” 改成 boundary-first：start/end loss 加权，start 权重大于 end；internal gap / cut_drop loss 继续强约束；cut_point 独立优化贴连 speech island；speech mask 作为 guardrail 而非唯一主目标。
- evaluation 不再用 `recall>=0.999` 做 hard gate，改用 boundary score：`start_p50/p90`、`end_p50/p90`、`long_predicted_segment_count`、`predicted_gap_crossing_segment_count`、`predicted_segment_duration p90/max` 和 chunk 数增幅。

### v1.20-v1.22 执行路线：先监督，再 imitation，最后候选切点 RL

用户提出：因为 synthetic timeline 已经可以随机拼接不同 Galgame speech island、gap、BGM、白噪声、短 gap 和贴连边界，是否可以直接用强化学习训练 speech-island 划分。

结论：

- RL 现在比早期更合理，因为我们已有可控环境、明确 reward 和 exact-island 真值。
- 但第一步不应直接上 REINFORCE/PPO。当前数据有精确 start/end、cut_drop、cut_point 标签，监督学习的样本效率和稳定性更高，能先把明显可学的边界打牢。
- 真 RL 只适合后置到候选切点层：动作空间限制为 keep / split / drop-gap，并只在 VAD valley、cut_drop、cut_point、start/end 这类候选上决策。禁止逐帧任意动作，否则容易 reward hacking、segment explosion 或漏完整台词。

执行顺序：

1. **v1.20 boundary-first supervised refiner**
   - 训练入口支持独立 `start_loss_weight` / `end_loss_weight`，默认 `start > end`。
   - `speech_loss_weight` 降为 guardrail；`internal_gap`、`cut_drop`、`cut_point` 提权。
   - metrics 记录 component loss 和 boundary-first 权重，方便横向比较。
2. **v1.21 reward planner teacher / imitation**
   - 用 R19 planner 在 offline synthetic 上生成 keep / split / drop-gap teacher。
   - 训练模型模仿 planner 决策，而不是把 planner 直接作为 runtime 默认。
   - gate 重点看 fallback chunk duration、gap crossing、单 chunk 台词数、complete island miss。
3. **v1.22 optional RL fine-tune**
   - 只在 v1.21 已稳定后尝试。
   - reward：start 接近真值权重大于 end；跨大 gap、20-30s fallback chunk、segment explosion、漏完整 island 强惩罚；ASR empty / hallucination proxy 可作为下游奖励。
   - RL 成功条件不是 synthetic reward 变高，而是匿名样片 A / held-out 的字幕观感和 fallback-safe metrics 同步改善。

本轮代码落地：

- `EndpointRefinerTrainConfig` 新增 `start_loss_weight` / `end_loss_weight`。
- `tools/vad/fusionvad_ja/train_endpoint_refiner.py` 默认改成 boundary-first：`speech=0.5`、`start=2.0`、`end=1.5`、`internal_gap=1.0`、`cut_drop=1.0`、`cut_point=1.0`、legacy `boundary_loss=0.0`。
- `train_metrics.json` 新增 `mean_component_losses` 与 `boundary_first` 权重记录。

v1.20 first-pass 执行：

- CPU smoke：`agents/temp/fusionvad-ja/v1-20-boundary-first-smoke-cpu/`，4 steps 跑通真实 4096 labels + Qwen3-ASR-0.6B full feature cache，`train_metrics.json` 正确写入 `boundary_first` 和 `mean_component_losses`。
- GPU first-pass：`datasets/train/fusionvad-ja/v1-20/qwen3-asr-0.6b-full29239/endpoint-refiner-boundary-first-touch4096-batch8-lr2e-4-steps256/`，CUDA，batch 8，256 steps，loss `8.3672 -> 5.6199`，显存约 `2.3GB`，checkpoint-step-128 / 256 和 final checkpoint 均保存。
- `speech_threshold=0.20` 导出：`agents/temp/fusionvad-ja/v1-20-boundary-first-touch4096-step256-predictions-th020-cut05/`，speech F1 `0.8874`，precision `0.8486`，recall `0.9299`；cut_drop F1 `0.4445` / recall `0.9567`；cut_point F1 `0.0000`。
- Boundary benchmark at `speech=0.20`：`agents/temp/fusionvad-ja/v1-20-boundary-first-touch4096-step256-boundary-benchmark/`，speech-duration recall `0.9652`，missed speech `2056.50s`，extra ratio `1.1903`，start p50 `1.350s`，end p50 `1.203s`，long segments `3554`，gap-crossing `2486`。
- `speech_threshold=0.10` 对照：speech recall 提升到 `0.9750`；boundary benchmark recall `0.9891`，但 extra ratio `1.3083`，start/end p50 都约 `2.41s`，chunk 明显变粗。

结论：

- v1.20 训练链路和新 loss 记录已跑通，但 256-step first-pass **不能替换默认 v1.19b**。
- 失败不是单纯 operating point 问题。降低 speech threshold 能补 recall，但会扩大 extra audio 和 start/end error；cut_point 仍未学出，说明贴连/换人边界需要更强 supervision 或 teacher。
- 下一步不应靠调低阈值，而应：
 1. 训练更久（回到 1024+ steps）并适度提高 `cut_point_positive_loss_weight` / `cut_point_loss_weight`。
 2. 单独 sweep cut_point threshold，看是否 target/nontarget 分布有可用分界；若没有，进入 v1.21 planner teacher / imitation。
 3. 用 boundary-first gate 选 checkpoint，不用 frame recall 单指标选模型。

cut_point 强化与 v1.21 teacher 启动：

- 先修正一个评测坑：`export_endpoint_refiner_predictions.py` 原先只读取 checkpoint 的 `boundary_radius_frames` / `cut_min_gap_s`，没有读取 `cut_boundary_radius_frames`，导致 cut_point target 用默认半径 `0` 评估，和训练半径不一致。已修正并加单测。
- v1.20 cutpoint64 训练：`datasets/train/fusionvad-ja/v1-20/qwen3-asr-0.6b-full29239/endpoint-refiner-boundary-first-cutpoint64-touch4096-batch8-lr2e-4-steps1024/`，CUDA，batch 8，1024 steps，lr `2e-4`，`cut_point_loss_weight=3.0`，`cut_point_positive_loss_weight=64`，`cut_boundary_radius_frames=2`。
- 训练曲线：loss `10.4787 -> 5.9876`，frame accuracy `0.6385 -> 0.8168`；显存约 `2.3-3.1GB`。
- 导出 at `speech=0.20`：`agents/temp/fusionvad-ja/v1-20-cutpoint64-touch4096-step1024-predictions-th020-cut05/`，speech F1 `0.9395` / precision `0.9761` / recall `0.9055`；cut_drop F1 `0.4812` / recall `0.9682`；cut_point F1 `0.1073` / recall `0.6057`。
- Boundary at `speech=0.20`：`agents/temp/fusionvad-ja/v1-20-cutpoint64-touch4096-step1024-th020-boundary-benchmark/`，recall `0.9450`，extra ratio `1.0069`，start/end p50 约 `0.36s`，long segments `2795`，gap-crossing `561`。
- 导出 at `speech=0.10`：speech recall `0.9352`；boundary recall `0.9622`，extra ratio `1.0480`，long segments `3126`，gap-crossing `924`。这说明 cut_point 强化明显提升边界精度，但 speech mask recall 还不够，不能直接替换默认 v1.19b。
- v1.21 planner teacher：
  - probability planner：`agents/temp/fusionvad-ja/v1-21-teacher-plan-cutpoint64-probability-th010/`，segments `9979 -> 23056`，long `3126 -> 614`，gap-crossing `924 -> 814`，recall `0.9125`。
  - hybrid truth-cost teacher：`agents/temp/fusionvad-ja/v1-21-teacher-plan-cutpoint64-hybrid-truthcost-th010/`，segments `9979 -> 22924`，long `3126 -> 614`，gap-crossing `924 -> 651`，recall `0.9109`。
  - 结论：planner 作为 runtime 仍会过切且掉 recall；但作为 teacher 可以提供“高价值 split/drop-gap”训练信号。
- 新增 `tools/vad/fusionvad_ja/export_boundary_imitation_targets.py`，把 planner `plan_details.jsonl` 转成 v1.21 imitation targets：`split_frames`、`drop_gap_frames`、`split_points`、`drop_gap_zones`。
- v1.21 imitation targets：`agents/temp/fusionvad-ja/v1-21-imitation-targets-cutpoint64-hybrid-truthcost-th010/`，4096 rows，`split_point=14534`，`drop_gap=3612`，split positive frame ratio `0.01654`，drop-gap positive frame ratio `0.03691`。
- 下一步：训练 imitation head / policy head 时不能照单全收 planner。应把 teacher 当候选监督，加入 recall guard 和 segment-count penalty，优先学习“减少 long/gap-crossing 但不漏完整 island”的子集。

v1.21 imitation head 执行记录：

- 新增 `AdditionFusionImitationBiLSTM`，输出 `split` / `drop_gap` 两个 logits；新增 `tools/vad/fusionvad_ja/train_imitation_head.py` 和 `tools/vad/fusionvad_ja/export_imitation_head_predictions.py`。
- 先跑 multitask plain BCE 1024 steps：`datasets/train/fusionvad-ja/v1-21/qwen3-asr-0.6b-full29239/imitation-head-cutpoint64-hybrid-truthcost-batch8-lr2e-4-steps1024/`。结果退化为常数策略：split best F1 `0.0330`，drop_gap best F1 `0.0774`，target/non-target p50 几乎相同。
- 改成 positive-window sampling 后，multitask 仍不能学出可用 split/drop_gap：`imitation-head-poswin128-cutpoint64-hybrid-truthcost-batch8-lr2e-4-steps1024/`，drop_gap target/non-target p50 仍几乎一致。
- 发现关键评测/训练坑：v1.21 imitation targets 按真实视频帧率 `29.97fps` 生成，`frame_hop_s=0.0333667`；Qwen feature cache 是 `frame_hop_s=0.02`。早期训练和导出都直接 `min(feature_frames, target_frames)` 截断，导致 target 时间轴错贴到 feature 前半段。已新增 `resize_binary_frames()`，训练与导出统一把 binary targets 重采样到 feature frame count，并加单测覆盖。
- balanced frame loss + 重采样 target 的 multitask 512 steps：`imitation-head-poswin128-balanced-resizedtarget-cutpoint64-hybrid-truthcost-batch8-lr2e-4-steps512/`。修复后有弱分离但仍不可直接用：split best F1 `0.0371`，drop_gap best F1 `0.0910`。
- drop_gap-only 512 steps：`imitation-head-dropgaponly-poswin128-balanced-resizedtarget-batch8-lr2e-4-steps512/`。这是第一版真正有用的候选：drop_gap best F1 `0.2366`，precision `0.1767`，recall `0.3581`，target/non-target p50 `0.8686 / 0.5078`。
- drop_gap-only 2048 steps：`imitation-head-dropgaponly-poswin128-balanced-resizedtarget-batch8-lr2e-4-steps2048/`。训练窗口 F1 提升到 `0.7268`，全量分离度更强但更保守：drop_gap best F1 `0.1870`，precision `0.1309`，recall `0.3269`，target/non-target p50 `0.7986 / 0.2506`。
- 结论：v1.21 不应把 split/drop_gap 放进同一个 imitation head 直接模仿 planner。split teacher 太稀疏且容易与全局先验混淆；drop_gap-only 可以作为“可删除内部 gap scorer”进入 offline packer 消融。512 版更偏 F1，2048 版更偏高分离/高置信，二者都暂不替换默认 VAD。
- offline packer 实现：新增 `tools/vad/fusionvad_ja/apply_drop_gap_packer.py`，输入 baseline `speech_frames` 和 drop_gap 逐帧概率，只在长父段内部删除高置信 drop_gap run；不进入主 pipeline，不改默认 VAD。实现坑：不能把 baseline segment 重建成新 frame mask，否则无应用区间时也会误删极短片段；已改为只在原始 `speech_frames` 上把实际应用的 drop_gap 区间置零。
- CUDA 导出：按“能 CUDA 就提权 CUDA”重跑 512/2048 逐帧概率，输出 `agents/temp/fusionvad-ja/v1-21-dropgaponly-step512-probabilities-cuda/` 和 `agents/temp/fusionvad-ja/v1-21-dropgaponly-step2048-probabilities-cuda/`，均 4096 rows。CPU 版本已移入 `agents/rm/fusionvad-ja-cpu-dropgap-probabilities-20260602/`。
- offline packer 消融（boundary benchmark 参数同 baseline：pad `0.2s`、merge gap `0.15s`、fallback target `8s`）：

| run | recall | missed_s | extra_ratio | pred_segments | long | gap_cross | dur_p90 | applied | removed_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | `0.9622` | `2230.7` | `1.0480` | `9979` | `3126` | `924` | `12.94s` | `0` | `0.0` |
| 512-th085 | `0.9444` | `3280.3` | `1.0295` | `12324` | `2182` | `877` | `9.27s` | `2345` | `2028.4` |
| 512-th090 | `0.9556` | `2622.7` | `1.0410` | `11018` | `2725` | `891` | `11.10s` | `1039` | `829.2` |
| 2048-th090 | `0.9593` | `2403.1` | `1.0448` | `10509` | `2904` | `900` | `11.84s` | `530` | `397.4` |
| 2048-th095 | `0.9619` | `2252.2` | `1.0476` | `10067` | `3094` | `920` | `12.72s` | `88` | `58.8` |

- 口径修正：用户明确当前目标不是整段 frame recall 最大化，而是 `start` 边界和 speech-island 分块准确。只要 recall `>=0.93`，可以牺牲一部分整段 recall 来换更短、更可 fallback 的 chunk。按这个口径补扫 512-th080/th082/th084 和 2048-th085/th088：

| run | recall | missed_s | extra_ratio | long | gap_cross | dur_p90 | start_p90 | start_p95 | end_p90 | applied | removed_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | `0.9622` | `2230.7` | `1.0480` | `3126` | `924` | `12.94s` | `7.857s` | `9.174s` | `7.808s` | `0` | `0.0` |
| 512-th080 | `0.9333` | `3939.0` | `1.0181` | `1763` | `874` | `8.44s` | `4.540s` | `6.396s` | `4.694s` | `3434` | `3136.6` |
| 512-th082 | `0.9378` | `3671.3` | `1.0227` | `1940` | `870` | `8.72s` | `4.853s` | `6.764s` | `4.928s` | `2994` | `2688.0` |
| 512-th084 | `0.9422` | `3411.1` | `1.0272` | `2097` | `875` | `9.02s` | `5.108s` | `7.197s` | `5.215s` | `2562` | `2249.8` |
| 2048-th085 | `0.9563` | `2578.6` | `1.0417` | `2707` | `890` | `11.04s` | `6.528s` | `8.321s` | `6.476s` | `994` | `766.1` |

- 新结论：在边界优先口径下，`512-th080` 是当前最强离线候选：recall 仍有 `0.9333`，但 start p90 `7.857s -> 4.540s`，start p95 `9.174s -> 6.396s`，long chunk `3126 -> 1763`，dur p90 `12.94s -> 8.44s`。这比 2048 系列更符合“粗时间轴先切准”的目标。仍不直接替换默认；下一步应该用 512-th080 做匿名样片 A GPU 闭环，审计 ASR 空输出、字幕观感和 fallback 是否真的改善。

v1.21 512-th080 匿名样片 A GPU 闭环：

- 执行脚本：`agents/temp/run_v1_21_dropgap512_th080_sample_a_gpu.sh`，CUDA 提权运行。首次中断后通过 `temp/asr_checkpoint_48102ec6a4.json` 恢复 `300/410` ASR chunk，续跑完成。
- 运行产物：
  - workflow：`agents/temp/fusionvad-ja/full-workflow-anon-a-v1-21-dropgap512-th080/`
  - diagnostics：`agents/temp/fusionvad-ja/diagnostics-anon-a-v1-21-dropgap512-th080/`
  - fallback-safe metrics：`agents/temp/fusionvad-ja/fallback-safe-boundary-metrics-anon-a-v1-21-dropgap512-th080/`
- runtime：ASR+alignment `366.80s`，总计 `368.44s`；输出日文字幕 `874` segments / `963` blocks。
- alignment diagnostics：chunks `410`，forced `222`，partial `1`，nonlexical `9`，drop_or_review `19`，vad_coarse `159`；fallback chunks `173/410`，其中 `vad_coarse_after_sentinel=159`，ASR QC reject `19`，align-text-empty `9`。
- fallback-safe：coarse fallback chunks `160`，unsafe fallback chunks `115`，fallback safe ratio `0.281`，fallback duration p50/p90/max `13.06 / 25.71 / 28.47s`，fallback crossing long silence `12`。
- 结论：512-th080 在离线 synthetic 指标上明显改善 start / long chunk，但真实样片 A 下没有过 fallback-safe gate。forced 数提升，但粗 fallback 仍集中在 20-30s 长 chunk，说明 drop-gap imitation 只处理了部分内部 gap，不能解决连续长 speech island / overlong chunk。v1.21 继续保持非默认；下一步需要更强 candidate policy 或直接训练 boundary objective，而不是把 512-th080 打开为默认。

v1.22 / v1.23 计划修正：

- Grok 查询失败后用内置搜索补查 endpointing、subtitle segmentation、RL speech boundary。结论：VAD / endpoint / subtitle cutpoint 不是同一个任务；只看 speech/silence 不足以判断“短暂停顿”和“真的该切字幕”。Netflix timing 规则也支持“in-time 尽量贴 speech start，out-time 可适当延后/压缩”的思路。
- RL 不适合现在直接逐帧训练整套 VAD。逐帧 action 容易 reward hacking、segment explosion、漏完整台词。更稳路线是：先做 supervised cutpoint head，再把 RL 限制在候选切点 planner 上。
- v1.22 目标：构造更干净的 exact-island cutpoint 数据集。用 Galgame clip 随机拼接多条 speech island，覆盖：
  - touch gap：无 gap 贴连，边界只能是 `cut_point`；
  - short gap：0-0.35s 短停顿，仍作为 `cut_point`；
  - regular gap：>=0.60s gap/noise/BGM，作为 `cut_drop`；
  - 随机 source / speaker proxy 顺序，避免模型只记住数据集原顺序；
  - BGM / noise / crossfade / gain / filter / codec / overlap 轻量增强。
- v1.22 首个实现：新增 `tools/vad/fusionvad_ja/build_v1_22_cutpoint_dataset.py`，它是 `build_galgame_synthetic_timeline.py` 的稳定 preset wrapper。底层仍输出 `labels.jsonl`、`manifest.json`、`boundary_manifest.jsonl` 和 `boundary_metadata`，因此可以直接复用现有 feature cache、endpoint-refiner 训练和 boundary benchmark。
- v1.22 smoke：`agents/temp/fusionvad-ja/v1-22-cutpoint-dataset-smoke16/`，16 records，`cut_point=52`，`cut_drop=11`，gap policy `regular=12 / short=33 / touch=19`。说明 wrapper 能稳定构造贴连、短 gap 和可删除 gap 三类监督。
- 单测：`test_build_v1_22_cutpoint_dataset_wrapper_records_cutpoint_and_drop_zones` 覆盖 wrapper 输出 summary、`boundary_manifest.jsonl`、`LabelRecord.boundary_metadata`；相关 synthetic timeline 回归 4 passed。
- 长 chunk 审计页已生成：`agents/audits/fusionvad-ja/long-fallback-r21-dropgap512-th080/index.html`。内容为 R21 dropgap512-th080 的 20 条 unsafe long fallback chunk，使用匿名样片 A 原视频 + 日文 VTT overlay + chunk ASR 文本 / 重叠字幕 / 指标，便于人工判断长 chunk 是真实长台词、噪声幻觉、还是多 speech island 被合并。
- v1.22 正式 4096 数据集已生成：`datasets/train/fusionvad-ja/v1-22/galgame-cutpoint-supervised-4096/`。统计：`records=4096`，`duration_s_total=161520.35`，`cut_point_segment_count=12256`，`cut_drop_zone_count=4092`，`speaker_turn_boundary_count=16384`，gap policy `regular=4128 / short=7432 / touch=4824`。构造特点：每条样本随机串联 5 条 Galgame speech island，覆盖贴连、短 gap、regular 可删 gap、随机 source/speaker proxy、背景混合、crossfade、filter、codec 和 overlap。
- v1.22 CUDA feature cache 已完成：`datasets/train/fusionvad-ja/v1-22/qwen3-asr-0.6b-full29239/galgame-cutpoint-supervised-4096-feature-cache/`。Qwen3-ASR-0.6B full SFT frozen feature，`device=cuda`，`dtype=bfloat16`，`cached=4096`，`errors=0`，产物约 `33G`。日志：`agents/temp/v1-22-feature-cache-4096-bs64-cuda.run.log`，确认 `param_device=cuda:0` / `param_dtype=torch.bfloat16`。
- v1.22 supervised cutpoint head first-pass 已训练：`datasets/train/fusionvad-ja/v1-22/qwen3-asr-0.6b-full29239/endpoint-refiner-cutpoint-supervised4096-batch8-lr2e-4-steps1024-boundaryfirst/`。训练参数：batch 8，lr `2e-4`，1024 steps，trainable params `1,889,349`，boundary-first 权重 `speech=0.35 / start=3.0 / end=2.0 / internal_gap=1.5 / cut_drop=2.0 / cut_point=3.0`。loss `13.6723 -> 9.5304`，final frame_accuracy `0.5269`，positive_ratio `0.8958`。
- v1.22 first-pass 导出 at `speech_threshold=0.20 / cut=0.50`：`agents/temp/fusionvad-ja/v1-22-cutpoint-supervised4096-step1024-predictions-th020-cut050/`。frame metrics：speech F1 `0.7946`，precision `0.9527`，recall `0.6814`；cut_drop F1 `0.2402` / recall `0.9244`；cut_point F1 `0.1135` / recall `0.5602`。注意：本次为分析写入了 `--include-probabilities`，目录约 `965M`；后续除非要阈值重算，不应默认写概率全量 JSONL。
- v1.22 first-pass boundary benchmark：`th020` synthetic speech-duration recall `0.6853`，missed speech `45106.10s`，extra ratio `0.7245`，start p50/p90 `1.449s/4.842s`，end p50/p90 `1.638s/4.794s`，predicted segment duration p90/max `7.08s/35.90s`，long `2007`，gap-crossing `435`。低阈值扫也救不回 recall：

| speech th | recall | missed_s | extra_ratio | start p90 | end p90 | dur p90 | long | gap_cross |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.05` | `0.8144` | `26596.9` | `0.8857` | `7.589s` | `7.133s` | `10.72s` | `5102` | `1436` |
| `0.08` | `0.7673` | `33353.6` | `0.8272` | `6.138s` | `5.658s` | `8.96s` | `3985` | `886` |
| `0.10` | `0.7475` | `36186.2` | `0.8025` | `5.611s` | `5.360s` | `8.50s` | `3548` | `713` |
| `0.15` | `0.7118` | `41313.1` | `0.7574` | `5.062s` | `4.942s` | `7.58s` | `2648` | `533` |
| `0.20` | `0.6853` | `45106.1` | `0.7245` | `4.842s` | `4.794s` | `7.08s` | `2007` | `435` |

- v1.22 first-pass 结论：数据构造方向成立，cut_drop/cut_point 信号能学到一部分，但当前 loss 把 speech mask 压坏，最高低阈值 recall 也只有 `0.8144`，远低于 `>=0.93` guardrail。不要直接把同一训练目标放大到 32768；下一版应先修正目标：提高 speech guardrail / 两阶段训练（先保住 speech mask，再训 boundary/cut）/ 或从 v1.19b 稳定 head 初始化，只训练新增 cutpoint/boundary 头。
- recall 口径修正：旧 `speech_duration_recall` 是线性时长重叠，覆盖 speech island 后半段和覆盖前半段会拿到同样分数；这不符合当前“start 更重要”的字幕边界目标。`tools/vad/fusionvad_ja/benchmark_boundary_predictions.py` 新增 `start_weighted_speech_recall`，对每个 truth speech island 按 `w(x)=(1-x)^gamma` 积分，`x=0` 为 island 起点，默认 `gamma=2.0`。单测覆盖：只覆盖前半段得分 `0.875`，只覆盖后半段得分 `0.125`，线性 recall 都是 `0.5`。
- 用新口径重算 v1.22 4096 first-pass，`speech_threshold=0.05/0.08/0.10/0.15/0.20` 的 `start_weighted_speech_recall` 分别为 `0.8204 / 0.7747 / 0.7555 / 0.7207 / 0.6945`。它只比线性 duration recall 高 `0.006-0.009`，说明这版模型并非单纯“漏开头被旧指标误伤”，而是总体 speech 覆盖不足；不改变“不直接放大到 32768”的结论。
- v1.22b speechguard：复用 4096 feature cache，把 `speech_loss_weight` 拉高到 `2.0`，boundary/cut 降为辅助，CUDA 训练 1024 steps：`datasets/train/fusionvad-ja/v1-22/qwen3-asr-0.6b-full29239/endpoint-refiner-cutpoint-supervised4096-batch8-lr2e-4-steps1024-speechguard/`。frame speech recall 回到 `0.9841`，但 cut/start/end 头在 `0.5` 阈值下全为 0；boundary benchmark recall `0.9848` / start-weighted `0.9789`，但 start p50/p90 `10.30s/28.66s`、duration p90/max `38.78s/46.02s`、long `5061`，说明单纯保 speech 会退化为大段保守 mask。
- v1.22c init-v1.19b：训练工具新增 `--init-checkpoint`，从默认 v1.19b head 初始化，低学习率 `5e-5` CUDA 微调 512 steps：`datasets/train/fusionvad-ja/v1-22/qwen3-asr-0.6b-full29239/endpoint-refiner-cutpoint-supervised4096-initv119b-batch8-lr5e-5-steps512/`。frame metrics at `speech=0.20/cut=0.50`：speech F1 `0.9514`，precision `0.9134`，recall `0.9927`；cut_drop F1 `0.3029` / recall `0.9272`；cut_point F1 `0.1293` / recall `0.3780`。说明从稳定 head 初始化能同时保 speech 和保留 cut 信号。
- v1.22c raw speech boundary：recall `0.9930` / start-weighted `0.9887`，但 start p50/p90 `14.39s/29.66s`，duration p90/max `40.40s/46.20s`，long `4557`。根因是 raw speech mask 会把多个 island 合成超长段，cut 信号没有参与 split。
- v1.22c apply-cut-to-speech：把 cut 直接从 speech 中删除，start p50/p90 改善到 `1.12s/4.30s`，duration p90 `6.96s`，但 speech recall 下降到 `0.7328` / start-weighted `0.7243`。结论：cut 不能作为删除 speech 的 hard gate。
- v1.22c cut-split offline：`benchmark_boundary_predictions.py` 新增 `--cut-split-mode split`，用 cut run 的中点拆分 predicted speech segment，不删除 speech。结果：recall `0.9930` / start-weighted `0.9887` 保持不变，start p50/p90 `1.19s/3.51s`，duration p90/max `6.98s/22.28s`。但 naive cut-split 把 segment count `5435 -> 49008`，gap-crossing `3005 -> 6027`，说明 cut 信号有价值但不能无约束全切。
- v1.22 当前结论：正确方向不是“再把 speech loss 拉高”或“直接 apply cut 删除音频”，而是 **cut-as-boundary constrained planner**：只对超过目标时长/跨大 gap 的高风险长段切，限制最小子段时长、最大目标时长、候选 cut run 数和 segment count 增幅；cut_drop 可优先切 regular gap，cut_point 只能切分贴连/换人边界。通过 synthetic gate 后再接匿名样片 A GPU 闭环。
- v1.22c 匿名样片 A GPU 闭环已完成。脚本：`agents/temp/run_v1_22c_cut_boundary_sample_a_gpu.sh`；日志：`agents/temp/v1-22c-cutboundary-anon-a-gpu.run.log`；workflow：`agents/temp/fusionvad-ja/full-workflow-anon-a-v1-22c-cutboundary/`；diagnostics：`agents/temp/fusionvad-ja/diagnostics-anon-a-v1-22c-cutboundary/`；fallback-safe metrics：`agents/temp/fusionvad-ja/fallback-safe-boundary-metrics-anon-a-v1-22c-cutboundary/`。本次配置：v1.22c head，`speech=0.20`，`cut=0.50`，`--no-fusionvad-apply-cut-to-speech`，`ASR_PRE_ASR_CUT_SPLIT_ENABLED=1`，R18 risk split 作为二阶段 safety net。
- v1.22c GPU 结果未过 gate：`chunks=236`，`segments=1018`，`forced=108`，`vad_coarse=108`，`fallback=120`，`vad_coarse_after_sentinel=108`，`unsafe=108`，fallback duration p90/max 仍是 `28.47s/28.47s`，总耗时 `1176.5s`。对比：R18 gap-first `chunks=250 / forced=109 / unsafe=114 / unsafe p90=28.47s`；R21 dropgap512-th080 `chunks=410 / forced=222 / unsafe=115 / unsafe p90=26.91s`。v1.22c 没有 chunk 数爆炸，但大量 fallback 仍是 28.47s single-island long chunk，说明当前 cut signal 对“连续长语音内部切点”不足。
- v1.22c 结论修正：cut-as-boundary 思路在 synthetic/offline 上成立，但真实样片 A 的瓶颈已经从“多 island + gap 被合并”转向“单个 VAD-positive 连续长 island 缺可靠内部切点”。下一步不应把 v1.22c 直接替换默认，也不应继续只扫 `cut_threshold`；应改为训练/规划更强的连续长 island boundary objective，例如句级 endpoint/cutpoint teacher、ASR/aligner sentinel 反向 hard negative、候选切点 constrained RL 或更明确的 pause/energy/phoneme evidence。
- 审计页刷新 bug：`agents/audits/fusionvad-ja/latest-audit.html` 原先使用 `meta refresh content=0` 自动跳转到最新审计页，live-server 打开后会反复刷新，影响人工审计。已移除自动跳转，改为静态入口链接；`rg` 未再发现 `http-equiv="refresh"` / `location.reload` / `window.location` 类自动刷新逻辑。
- v1.23 才考虑 constrained RL：动作空间只允许在候选点上做 keep / split / drop-gap；reward 以 start 准、fallback chunk <= 8-9s、不跨长 gap、不漏完整 island、segment count 可控、ASR empty / hallucination / aligner sentinel 下降为准。

v1.23 residual cut split 离线归因：

- 目的：确认 v1.22c 的 28.47s unsafe fallback 是“模型没有切点信号”，还是“切点信号存在但 packing 策略没用上”。
- 先提权 CUDA 导出匿名样片 A 的 v1.22c 逐帧分数：`agents/temp/fusionvad-ja/v1-23-anon-a-v1-22c-frame-scores.json`，`frame_count=269833`，`frame_hop=0.02s`，日志 `agents/temp/v1-23-export-frame-scores-anon-a-cuda.run.log`。
- 用 `tools/vad/fusionvad_ja/analyze_fallback_cut_signal.py` 分析 `fallback-safe-boundary-metrics-anon-a-v1-22c-cutboundary/unsafe_fallback_chunks.jsonl`：
  - 20/20 unsafe 都是 `vad_coarse_after_sentinel`。
  - 20/20 都是 `speech_island_count=1`、`internal_gap_count=0`、`split_reason=pre_asr_cut_split`。
  - 20/20 都有 cut 候选；按 `target_child_s=9.0`，17/20 用离线贪心可切到目标内。
  - 真实 `_plan_cut_split_frames_for_segment` 复算：20/20 都能找到 split frame，16/20 子段 max <= 9s。说明瓶颈不是 cut head 完全无信号。
- 更细根因：v1.22c 先在超长连续段上跑 `pre_asr_cut_split`，但 `max_children=8` 被父段消耗后，仍留下 24.47s residual child；随后 R18 risk splitter 因为 single-island continuous risk score 只有 `1.5`，低于 `continuous_threshold=2.0`，没有继续切这些 residual child。
- 代码补丁：`src/audio/chunk_packer.py` 在 risk split 阶段保留 chunk 的 `split_policy`，并给 “`r17_pre_asr_cut_v1` 产生后仍超过 target 的 single residual child” 增加 `residual_cut_child` 风险理由。单测 `test_pre_asr_risk_split_revisits_long_residual_cut_child` 覆盖：普通 long continuous chunk 仍受 `continuous_threshold=2.0` 保护，但 residual cut child 可继续切。
- 新增工具：`tools/vad/fusionvad_ja/analyze_residual_cut_split.py`，直接从已有 `processing_spans` 模拟 residual risk split；它比 `analyze_r18_risk_splits.py` 更贴近 v1.22c 的真实失败链路，因为后者会从 raw VAD segments 重新 pack。
- residual 模拟结果：

| config | target split | chunk growth | target max child p90 | target max child max | 结论 |
|---|---:|---:|---:|---:|---|
| target 14s / max children 4 | `20/20` | `2.347x` | `17.853s` | `17.909s` | 能切掉 unsafe，但 ASR 调用明显增加 |
| target 9s / max children 8 | `20/20` | `3.653x` | `14.210s` | `17.442s` | 更接近 fallback-safe，但 chunk 爆炸风险更高 |

- 当前结论：v1.22c 失败的下一层根因是 **cut 信号没有二次用于 residual long child**，不是完全无可切点。直接把 residual cut split 打开为默认会让全片 chunk 数约 `2.35-3.65x`，不适合默认。下一步应做 v1.23 受限策略：只对真实 `vad_coarse_after_sentinel` 高风险形态或接近 hard cap 的 residual child 应用，限制每个父 chunk 的新增 child 数、目标时长和全片 chunk growth，再跑匿名样片 A GPU 闭环。
- 2026-06-03 路线修正：`chunk growth` 不再作为主要否决 gate，只保留为成本和极端爆炸观察指标。用户确认 90 分钟片子几百个 ASR chunk 是正常的；当前质量主 gate 改为 `start` 边界优先、`end` 边界次优先、fallback chunk 不能粗到 `20-30s`、ASR empty / hallucination 不能明显恶化、字幕观感不能变差。按 Netflix Timed Text Timing Guidelines，subtitle in-time 应尽量贴近第一帧音频，out-time 可在无冲突时略延后，并保留 2-frame gap；这支持“start 比 end 更关键，end 可由 subtitle polish 压缩”的工程取舍。
- v1.23 执行口径：先跑 residual cut split 闭环，不再因为 chunk 数增加 `2.35-3.65x` 直接否决；只有出现极端爆炸、ASR empty/hallucination 明显恶化或字幕观感变差时才回退。验收重点是 unsafe fallback p90/max、`vad_coarse_after_sentinel`、start 边界、长 continuous residual 是否被切成更接近一句台词的 island。
- speaker sidecar 路线：CAM++ 不替代 VAD，只作为 speaker-change 辅助；优先升级为 ERes2NetV2 / 3D-Speaker。流程是 FusionVAD / cutpoint 先给 speech island，再对相邻 island 提 speaker embedding，计算 cosine / speaker-change score；speaker-change 高时增强 cut、避免跨人合并，speaker 相似且 gap 极短时允许合并。它只影响 pre-align / cue-stage packing，不负责 speech/non-speech。
- 参考依据：Netflix Timed Text Timing Guidelines <https://partnerhelp.netflixstudios.com/hc/en-us/articles/360051554394-Timed-Text-Style-Guide-Subtitle-Timing-Guidelines>；Two-pass Endpoint Detection <https://arxiv.org/abs/2401.08916>；Joint Segmenting and Decoding for Long-Form ASR <https://arxiv.org/abs/2204.10749>；Phoenix-VAD <https://arxiv.org/abs/2509.20410>；3D-Speaker <https://github.com/modelscope/3D-Speaker>；ERes2NetV2 <https://www.isca-archive.org/interspeech_2024/chen24l_interspeech.pdf>；CAM++ <https://arxiv.org/abs/2303.00332>。
- v1.23 residual cut split 匿名样片 A GPU 闭环完成。脚本：`agents/temp/run_v1_23_residual_cut_sample_a_gpu.sh`；日志：`agents/temp/v1-23-residual-cut-anon-a-gpu.run.log`；workflow：`agents/temp/fusionvad-ja/full-workflow-anon-a-v1-23-residual-cut-split/`；diagnostics：`agents/temp/fusionvad-ja/diagnostics-anon-a-v1-23-residual-cut-split/`；fallback-safe：`agents/temp/fusionvad-ja/fallback-safe-boundary-metrics-anon-a-v1-23-residual-cut-split/`。配置：v1.22c head，`speech=0.20`，`cut=0.50`，cut split first stage，risk split second stage，target core `270` frames，max children `8`，不 apply cut to speech。
- v1.23 GPU 结果：chunks `862`，segments `2764`，blocks `2694`，ASR+alignment `1836.86s`，total `1867.00s`。alignment quality：forced `462`，vad_coarse `336`，drop_or_review `54`，nonlexical `10`；fallback chunks `364/862`，其中 `vad_coarse_after_sentinel=336`、ASR QC repeat-loop reject `54`、align-text-empty `10`。quality report 告警：`kana_only_ratio=0.311`、`short_segment_ratio=0.647`、`per_min_subtitle_count=30.0`，说明切细后字幕过短/过密和重复语气词问题变突出。
- v1.23 fallback-safe：coarse fallback `337`，unsafe `263`，safe ratio `0.220`，fallback duration p50/p90/max `10.82 / 12.91 / 28.47s`，unsafe p50/p90/max `11.22 / 12.95 / 28.47s`，long-silence crossing `15`。对比 v1.22c：chunks `236`，forced `108`，unsafe `108`，fallback p50/p90/max `28.47 / 28.47 / 28.47s`。对比 v1.21 dropgap512-th080：chunks `410`，forced `222`，unsafe `115`，fallback p50/p90/max `13.06 / 25.71 / 28.47s`。结论：v1.23 实质性打掉 28s 粗 fallback p90，但不是免费收益。
- v1.23 当前判断：方向有效，但不直接默认。下一步应保持 boundary-first 目标，同时补三件事：(1) subtitle timing polish，把过密短 cue 合理合并/压 end、保证 2-frame gap；(2) ASR repeat-loop / 低信息输出后处理，避免切短后幻觉和语气词循环放大；(3) ERes2NetV2 speaker sidecar offline probe，用 speaker-change score 辅助 cue-stage 合并/切分，优先解决多人/男女对话 cue 是否跨 speaker 的判断。

v1.23 后置修正 first-pass：

- 背景：v1.23 residual cut split 把粗 fallback p90 打下来，但短字幕密度、重复语气词和跨 speaker cue 合并成为新的瓶颈。用户确认 chunk growth 不是硬 gate，质量 gate 应转向 start 边界、fallback duration、ASR empty / hallucination 和字幕观感。
- 检索依据：ASR hallucination 文献指出非语音/模糊人声会诱发重复循环幻觉，但目标域中的 disfluency、喘息、呻吟、短促 kana 也可能是真实内容，不能用“低信息文本”本身作为删除依据；Netflix timing guideline 支持 in-time/start 优先、out-time/end 可后处理压缩并保留 2-frame gap；ERes2NetV2 / 3D-Speaker / CAM++ 更适合作为 speaker embedding sidecar，不替代 VAD。
- `src/subtitles/options.py` / `src/subtitles/writer.py`：新增 `SUBTITLE_DENSE_CUE_MERGE_*` 和 `_merge_dense_short_cues`。它只合并短、近、文本量小、同 speaker 或未知 speaker 的 micro cues；不移动下一条 start，后续仍由 normalize/polish 保证 2-frame gap。默认阈值保守：4 frames gap、24 frames 单 cue、90 frames 合并后总长、12 text units。
- `src/whisper/qc.py`：新增 `vocalization_repetition` profile。kana-only、短 unit、低字符密度、时长不长的重复语气词/呻吟从 `repeat_ngram_loop reject` 改为 `repeated_nonlexical_vocalization warn` + `preserve_with_review`；lexical phrase loop、高密度文本和 signal reject 仍会 reject。没有引入具体词黑名单。
- `tools/subtitles/probe_speaker_sidecar.py`：新增离线 adjacent speaker-change probe。输入预计算的 ERes2NetV2 / 3D-Speaker / CAM++ embedding JSONL，输出相邻 segment 的 cosine、`speaker_change_score=1-cosine` 和阈值判断。先保证 sidecar 指标链路，不把真实模型下载和依赖接进默认 pipeline。
- 测试：`.venv/bin/python -m py_compile src/subtitles/options.py src/subtitles/writer.py src/whisper/qc.py tools/subtitles/probe_speaker_sidecar.py`；`.venv/bin/python -m pytest tests/test_subtitle_options.py tests/test_subtitle_quality_pass.py tests/test_asr_qc_signals.py tests/test_qc_backend_context.py tests/test_speaker_sidecar_probe.py`，结果 `68 passed`。
- 下一步：用 v1.23 产物离线重放 subtitle writer，比较 dense cue merge 前后 `per_min_subtitle_count`、`short_segment_ratio` 和字幕观感；用 diagnostics 统计 `repeated_nonlexical_vocalization` 与旧 `repeat_ngram_loop` 差异；接 ERes2NetV2 / 3D-Speaker extractor 对匿名样片 A islands 做 sidecar probe。
- 2026-06-03 v1.23 subtitle postprocess replay 已补工具和实测：`tools/subtitles/replay_subtitle_postprocess.py` 读取既有 `bilingual.json` / `aligned_segments.json` / `timings.json`，分别以 dense cue merge OFF/ON 重放 `prepare_srt_blocks()`，输出 `before_blocks.json`、`after_blocks.json`、`summary.json`、`summary.md`。测试：`.venv/bin/python -m pytest tests/test_replay_subtitle_postprocess.py tests/test_subtitle_quality_pass.py tests/test_asr_qc_signals.py tests/test_speaker_sidecar_probe.py`，结果 `53 passed`。
- 匿名样片 A replay 产物：`agents/temp/fusionvad-ja/subtitle-postprocess-replay-v1-23-anon-a/summary.md`。结果：blocks `1543 -> 1543`，dense merges `0 -> 0`，short segment ratio `0.205 -> 0.205`，per-min subtitle count `17.11 -> 17.11`，nonlexical repetition count `17 -> 17`。结论：当前 dense short cue merge 太保守，对 v1.23 真实输出没有实际缓解；后续不应盲目放宽阈值，而应做 cue-stage planner：结合 start/end、最小 2-frame gap、读速、speaker-change、非词汇重复和 fallback 质量做局部 merge/polish。
- repeat-loop 策略复核：Galgame SFT 后大量喘息、呻吟、短 kana 重复是真实目标域内容的可能性很高，因此不能把 `repeat_ngram_loop` 一刀切成幻觉。当前 `repeated_nonlexical_vocalization` 只把短、低密度、非词汇 profile 标为 `preserve_with_review`；夹杂语义词、异常字符密度或长串 lexical loop 仍保留 reject / warning。这符合“先保留可审计内容，再用后处理和人工样本反哺”的路线。
- RL 位置再次收敛：不做逐帧 RL、不直接用 RL 改 VAD layers。下一版 RL 只做 constrained candidate-cut policy，动作限定为 `keep` / `split` / `drop-gap`，候选点来自 VAD valley、cut_drop、cut_point、endpoint、speaker-change 或 subtitle cue 风险点。reward 以 start 准、fallback chunk 不粗、不过长跨 gap、不明显增加 ASR empty / hallucination、字幕观感可接受为主；synthetic reward 不能单独作为上线 gate。
- 2026-06-03 cue-stage planner 离线诊断补充：新增 `tools/subtitles/analyze_subtitle_cue_merge_candidates.py`，只做诊断和模拟，不改正式 writer 默认。它读取 v1.23 `bilingual.json`，解释相邻 cue 为什么没有被 dense merge，并用更宽的候选规则模拟局部合并，输出 `before_blocks.json`、`planner_blocks.json`、`planner_actions.json`、`summary.json`、`summary.md`。测试：`.venv/bin/python -m pytest tests/test_subtitle_cue_merge_candidates.py tests/test_replay_subtitle_postprocess.py tests/test_subtitle_quality_pass.py tests/test_subtitle_options.py`，结果 `42 passed`。
- 匿名样片 A blocker 分布：相邻 pair `1542`，dense merge 基本被 `text_units_too_large=1540`、`single_duration_too_long=1494`、`combined_duration_too_long=1301`、`gap_too_large=459`、`sentence_boundary=161` 挡住。这说明 v1.23 的短字幕密度不是 micro-cue 小规则能解决，而是 cue-stage 需要在更长但仍可读的范围内做规划。
- 离线扫描：保守参数 `min_score=0.72/max_gap=0.45s/max_combined=4.8s/max_text_units=34` 合并 `0`。`wide1` (`0.55/0.8s/6.5s/48`) 合并 `70`，blocks `1543 -> 1475`，short ratio `0.205 -> 0.161`，per-min `17.11 -> 16.35`，overlap `0`。`wide2` (`0.45/1.2s/6.5s/56`) 合并 `166`，blocks `1543 -> 1386`，short ratio `0.205 -> 0.124`，per-min `17.11 -> 15.37`，kana-only `0.093 -> 0.072`，overlap `0`。wide2 仍不满足 `QC_MAX_PER_MIN=8`，但证明“cue-stage planner”有实际杠杆；下一步应引入 speaker sidecar / 读速 / fallback 风险和人工审计，而不是直接把 wide2 作为默认。
- 2026-06-03 cue planner 约束接入：`analyze_subtitle_cue_merge_candidates.py` 增加 `--diagnostics`、`--speaker-pairs`、`--speaker-change-policy`、`--fallback-risk-policy`。speaker sidecar 可按 cue id / index / source segment / chunk id 匹配相邻 pair，`speaker_change` 默认 block；alignment diagnostics 通过 `source_chunk_index` 关联 cue，fallback / sentinel / ASR reject 默认 penalize，也可 block。测试扩大到 `44 passed`。
- 匿名样片 A + diagnostics 结果：`wide2` + fallback risk penalize 合并 `144`，blocks `1543 -> 1407`，short ratio `0.205 -> 0.134`，per-min `17.11 -> 15.60`，overlap `0`；fallback risk block 合并 `141`，short ratio `0.140`。约束统计显示 `fallback_risk_pair=713`、`fallback_risk_boundary=592`，但仍保留足够合并空间。结论：fallback 风险不应一刀切禁止 cue merge，适合作为 penalty / review signal；真正需要下一步补的是 ERes2NetV2/3D-Speaker speaker-pair 实测，防止把换人对话误合并。
- 2026-06-03 speaker sidecar extractor first-pass：新增 `tools/subtitles/extract_speaker_sidecar_embeddings.py`。它从 `bilingual.json` + 原始音频切 cue，输出 `speaker_embeddings.jsonl` 与 adjacent `speaker_pairs.jsonl`，并可直接喂给 cue planner。当前 backend：`energy_mfcc` 仅用于 smoke / schema 验证；`modelscope_eres2netv2` 已预留但会明确提示需要 3D-Speaker `speakerlab` 包，不静默假跑。测试：`.venv/bin/python -m pytest tests/test_speaker_sidecar_embeddings.py tests/test_speaker_sidecar_probe.py tests/test_subtitle_cue_merge_candidates.py`，结果 `6 passed`。
- 匿名样片 A energy_mfcc smoke：`2545/2694` cues 生成 embedding，`2544` pairs，`speaker_change_count=14`，接入 cue planner 后只阻断 `4` 个候选，合并数仍为 `144`，short ratio `0.134`。结论：链路完整，但 energy/MFCC 不是可靠换人模型；它只能证明 sidecar schema 与 planner 约束能跑通。下一步若要真正判断男女/多人 cue，应安装 3D-Speaker / speakerlab 并用 `iic/speech_eres2netv2_sv_zh-cn_16k-common` 或 `iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common` 重跑。
- 2026-06-03 speaker sidecar extractor second-pass：`modelscope_eres2netv2` 从占位报错改为真实 ModelScope speaker-verification pipeline backend，默认 model id `iic/speech_eres2netv2_sv_zh-cn_16k-common`，支持 `--device`、`--batch-size`、`--model-id`，并在缺依赖 / 缺模型 / 网络或 CUDA 问题时明确失败，不回退到 `energy_mfcc`。本地探测发现 `modelscope` 已装但缺 `addict` 等可选依赖，真实匿名样片 A 还需提权安装依赖和下载模型后再跑。测试：`.venv/bin/python -m pytest tests/test_speaker_sidecar_embeddings.py tests/test_speaker_sidecar_probe.py tests/test_subtitle_cue_merge_candidates.py`，结果 `7 passed`。
- 2026-06-03 ERes2NetV2 真实 sidecar probe：用 `uv pip install addict sortedcontainers simplejson datasets oss2` 补齐 ModelScope 可选依赖，模型下载到项目内 `models/modelscope/iic/speech_eres2netv2_sv_zh-cn_16k-common`（约 `71M`，ignored）。`--backend modelscope_eres2netv2 --device gpu --batch-size 32` 跑匿名样片 A v1.23：`embedding_count=2545/2694`，`pair_count=2544`，embedding dim `192`，skip `short=126 / empty_audio=23`。原始阈值 `0.35` 过敏感：score p50 `0.6826`，th75 `919` changes，th85 `423`，th95 `111`。cue planner wide2 + diagnostics + fallback penalize 对比：ERes2NetV2 th65 block `85` merges / short ratio `0.1647`；th75 block `100` / `0.1567`；th85 block `130` / `0.1429`；th85 penalize `123` / `0.1444`。结论：真实 speaker sidecar 有实用约束力，但分数需校准；当前更适合作为 cue-stage penalty/review 信号或高阈值 block，不应直接默认硬 block。
- 2026-06-03 speaker sidecar cue-planner sweep 工具：新增 `tools/subtitles/sweep_speaker_sidecar_cue_planner.py`，输入 `speaker_embeddings.jsonl` 后自动生成多阈值 adjacent speaker pairs，并调用 cue planner 扫 `block/penalize` policy，输出 `sweep_summary.json/md`。匿名样片 A 标准化 sweep：产物 `agents/temp/fusionvad-ja/speaker-sidecar-cue-planner-sweep-v1-23-anon-a-eres2netv2/`；th75 changes `919`、merges `104`、short ratio `0.1577`；th85 changes `423`、merges `127`、short ratio `0.1441`；th95 changes `111`、merges `139`、short ratio `0.1367`；block 与 penalize 在本轮最终 merge 数相同，但 blocker 分布不同。该结果支持“speaker sidecar 是 cue-stage review/penalty/高阈值 block 信号，不直接作为低阈值硬规则”。测试：`.venv/bin/python -m pytest tests/test_speaker_sidecar_cue_planner_sweep.py tests/test_speaker_sidecar_embeddings.py tests/test_speaker_sidecar_probe.py tests/test_subtitle_cue_merge_candidates.py`，结果 `8 passed`。
- 2026-06-03 cue planner reading-density gate：`analyze_subtitle_cue_merge_candidates.py` / `sweep_speaker_sidecar_cue_planner.py` 新增可选 `--max-reading-units-per-s`，默认 `0` 关闭。它阻止合并后 `text_units / duration` 过高的 cue，记录 `reading_density_too_high` blocker，只用于离线诊断。匿名样片 A ERes2NetV2 sweep 对比：无 reading gate 时 th85/th95 penalize 分别 `127/139` merges、short ratio `0.1441/0.1367`；reading12 变为 `49/57` merges、`0.1879/0.1834`；reading16 为 `74/84`、`0.1783/0.1738`；reading20 为 `86/98`、`0.1729/0.1693`。结论：读速 gate 是有用的保护/审计信号，但会明显砍掉合并空间，当前不应作为主优化目标或默认开启。
- 2026-06-03 cue planner 字幕对照导出：`analyze_subtitle_cue_merge_candidates.py` 现在随 `before_blocks.json` / `planner_blocks.json` 同步导出 `before.bilingual.srt/vtt` 与 `planner.bilingual.srt/vtt`，并把路径写入 `summary.outputs`。标准 ERes2NetV2 sweep 已刷新并包含字幕文件：`agents/temp/fusionvad-ja/speaker-sidecar-cue-planner-sweep-v1-23-anon-a-eres2netv2/`；当前默认口径为 `min_score=0.45/max_gap=1.2s/max_combined=6.5s/max_text_units=56`，本轮 th75/th85/th95 merges 为 `95/121/137`，short ratio 为 `0.1604/0.1478/0.1379`。另生成 reading16 诊断 sweep：`agents/temp/fusionvad-ja/speaker-sidecar-cue-planner-sweep-v1-23-anon-a-eres2netv2-reading16/`，th85/th95 merges `77/85`，short ratio `0.1777/0.1739`。结论不变：读速 gate 是保护/审计信号，标准候选仍优先看 th85/th95 的实际字幕观感。
- 2026-06-03 cue planner merge review 清单：新增 `tools/subtitles/export_cue_planner_merge_review.py`，把 `planner_actions.json` + `before_blocks.json` 导出为风险优先的 `merge_review_items.jsonl/csv` 和 `summary.md`。字段包含合并前后文本、时间、score、gap、combined duration/text units、speaker-change score、fallback/cross-chunk/读速风险标签。匿名样片 A 产物：th85 `agents/temp/fusionvad-ja/cue-planner-merge-review-v1-23-anon-a-eres2netv2-th85/`，121 items，风险标签 reading_density_high `53`、near_speaker_threshold `20`、fallback_risk/crosses_chunk `7`；th95 `agents/temp/fusionvad-ja/cue-planner-merge-review-v1-23-anon-a-eres2netv2-th95/`，137 items，reading_density_high `60`、near_speaker_threshold `15`、high_speaker_score `14`、fallback_risk/crosses_chunk `9`。结论：后续人工审计不必通读全片，可先看 high-priority merge items 判断 th85/th95 是否合并跨 speaker、fallback 粗时间轴或高读速片段。
- 2026-06-03 th85 vs th95 merge review 只读对比：th95 比 th85 多 `18` 个合并，extra 风险标签为 near_speaker_threshold `15`、reading_density_high `9`、loose_gap `3`、fallback_risk/crosses_chunk `2`。这说明 th95 的新增收益主要来自放过更接近 speaker-change 阈值的相邻 cue，而不是大量低风险短 cue。下一步人工审计应优先看 th95-extra 18 条；若跨 speaker / 高读速观感可接受，可把 th95 作为 cue-stage 候选，否则退回 th85。
- 2026-06-03 审计页导航统一：新增 `tools/audits/audit_nav.py`，后续审计页统一刷新 `agents/audits/index.html` 和 `agents/audits/latest-audit.html`；`latest-audit.html` 只提供静态链接，不再自动跳转。审计产物直接放在 `agents/audits/` 下，不再套 `fusionvad-ja/` 子目录。以后从 `agents/audits/index.html` 进入最新审计。早期 th95-extra 审计页后续因标签语义拆分作废；当前活跃审计页见后续 v3-side-labels 记录。
- 2026-06-03 cue planner 人工审计校准闭环：新增 `tools/subtitles/calibrate_cue_planner_from_manual_audit.py`，把审计 JSONL + source manifest 合并成可复用统计，输出 label/risk tag/speaker score/planner score/reading density 分桶与参数建议。用户标注的 th95-extra 18 条结果：`keep_text=5`、`needs_realign=7`、`bad_asr=4`、`drop_non_speech=1`、`needs_split=1`，整体 problem rate `0.722`。风险结论：`loose_gap` 3/3 问题、`high_speaker_score` 2/2 问题；`near_speaker_threshold` 15 条里 5 条 keep，不能硬 block；`reading_density_high` 9 条里 4 条 keep，只适合作 review/protection 信号。`bad_asr/drop_non_speech` 应回流 ASR QC / hard-negative，不应全算作 merge policy 失败。产物：`agents/temp/fusionvad-ja/cue-planner-manual-calibration-th95-extra-20260603/summary.md`。
- 2026-06-03 th95-constrained 实验：`analyze_subtitle_cue_merge_candidates.py` 增加可选 `--speaker-score-penalty-threshold` / `--speaker-score-penalty` / `--speaker-score-block-threshold`，默认全部关闭，不改正式默认。按人工校准建议重放匿名样片 A：`speaker_threshold=0.95`、`max_gap_s=0.5`、`speaker_score_penalty_threshold=0.85`、`speaker_score_penalty=0.12`、fallback risk 继续 penalize。结果：blocks `1543 -> 1429`，merges `119`，short ratio `0.2048 -> 0.1421`，per-min `17.11 -> 15.84`，kana-only `0.0927 -> 0.0791`，overlap `0`。相比 th85 baseline 的 121 merges，th95-constrained 更保守但仍有不同合并集合。
- 2026-06-03 cue planner 差集与二次审计：新增 `tools/subtitles/compare_cue_planner_merge_reviews.py` 和 `tools/subtitles/export_cue_planner_audio_audit_manifest.py`。th95-constrained vs th85：candidate `119`、baseline `121`、extra `19`、dropped `21`；extra 风险标签 reading_density_high `12`、near_speaker_threshold `4`、fallback_risk/crosses_chunk `2`、long_combined_duration `1`。已生成新审计页 `agents/audits/cue-planner-th95-constrained-extra-audio-audit/index.html`，导航 `agents/audits/index.html` 已指向该页。下一步由人工审计这 19 条，判断 th95-constrained 是否真的优于 th85，不能仅凭 short ratio 默认上线。
- 2026-06-03 th95-constrained extra 人工审计完成（旧标签口径，后续作废）：用户标注 `manual_cue_planner_th95_constrained_extra_labels.jsonl` 共 `19/19` 条。校准产物曾为 `agents/temp/cue-planner-th95-constrained-extra-calibration/summary.md`，后续因 `drop_non_speech` 语义混淆已删除，不作为当前决策依据。旧多选标签统计：`bad_asr=13`、`needs_split=8`、`timing_accurate=7`、`needs_realign=7`、`drop_non_speech=6`、`keep_text=3`；按旧问题优先归类后 overall problem rate `0.947`。这批结果只保留为“旧标签过严”的历史证据。
- 2026-06-03 th85 high-risk 基线审计准备：按“先确认安全基线”的新计划，从 `agents/temp/fusionvad-ja/cue-planner-merge-review-v1-23-anon-a-eres2netv2-th85/merge_review_items.jsonl` 取风险优先前 `40` 条，生成 `agents/temp/fusionvad-ja/cue-planner-th85-high-risk-audit-manifest/cue_planner_audio_audit_manifest.jsonl`，再从匿名样片 A v1.23 源音频切片到 `agents/audits/cue-planner-th85-high-risk-audio-audit/`，`pad_s=1.2`，`materialized_rows=40`、`errors=0`。风险组成包括 `reading_density_high` 12 条、`near_speaker_threshold,reading_density_high` 9 条、`near_speaker_threshold` 8 条，以及 fallback/cross-chunk/loose-gap/long-duration 组合。审计页：`agents/audits/cue-planner-th85-high-risk-audio-audit/index.html`；导航 `agents/audits/index.html` 已指向该最新页。目标：先评估 th85 baseline 自身问题率，再决定 v1.24 cue-stage planner 是否应继续保守、引入 reading density 保护，或只把 speaker sidecar 当 review/penalty 信号。
- 2026-06-03 审计标签语义修正：Grok 检索到 ASR/语料标注通常把 laughter / breath / sigh / groan / grunt 等 nonverbal vocalization 与纯噪声、BGM、静音区分处理；Galgame 目标域里的呻吟、喘息、短促拟声可能可转写且时间轴有效，不能和 hard drop 共用 `drop_non_speech`。审计页因此把旧 `非语音/无字幕` 改成 `删除/无字幕价值`（纯噪声、BGM、静音、机械声等硬丢弃），新增 `low_info_vocal` / `低信息人声/呻吟`，允许它和 `文本可用`、`时间轴准确` 多选共存；快捷键 `4` 对应该新标签。校准脚本新增 `low_info_keep` / `low_info_review` bucket，并兼容旧数据中 `drop_non_speech + keep/timing` 的混选，避免把目标域真实低信息人声误计为 ASR/QC hard problem。th95-constrained 19 条旧统计因此应视为“旧标签过严”的历史口径，后续以新标签重新审计 th85-high-risk 40 条。
- 2026-06-03 th95-constrained 旧标签重算：曾用新校准口径重跑旧 `manual_cue_planner_th95_constrained_extra_labels.jsonl`，reviewed `19/19`，overall problem rate 从旧口径 `0.947` 降到 `0.842`；bucket 为 `asr_qc=15`、`merge_timing=1`、`keep=1`、`low_info_keep=2`。这验证旧 `drop_non_speech` 确实混入了“低信息但文本/时间轴可用”的语义，但 reading-density 和 near-speaker-threshold 风险仍偏高，th95-constrained 仍不应直接默认。旧重算产物和旧人工 JSONL 后续已删除，不作为当前决策依据。
- 2026-06-03 旧审计结果作废删除：由于 `drop_non_speech` 旧标签混淆了 hard drop 与低信息人声，原 th95-extra / th95-constrained 人工 JSONL 和对应校准产物已删除，需要按新标签重新审计。活跃 `agents/audits/` 只保留 `cue-planner-th85-high-risk-audio-audit/`；该页重生为 `dataset_id=cue-planner-th85-high-risk-audio-audit-v3-side-labels`，输出文件名改为 `manual_cue_planner_th85_high_risk_labels_v3_side_labels.jsonl`，避免浏览器 localStorage 和旧文件名继续复用旧标签。
- 2026-06-03 v3 侧向审计标签：为解决“上/下两条 cue 一好一坏”的长期标注问题，审计页新增 `left_*` / `right_*` 标签：`上句/下句可用`、`上句/下句文本错`、`上句/下句无字幕价值`、`上句/下句低信息`。人工导出继续保留 `manual_label` 兼容字段，同时新增 `manual_labels` 多选数组；校准脚本新增 `side_mixed` bucket 和左右侧 label counts，用于把“需要拆分但只有一侧坏”的样本单独统计，不再误判为整条字幕失败。当前最需要审计的是 th85 high-risk 基线 40 条，对应 `agents/audits/cue-planner-th85-high-risk-audio-audit/index.html`。

---

## 2026-06-04 · Boundary Refiner 训练入口落地

- 主线从固定 `gap <= N` 规则收敛为 `candidate extraction -> Boundary Refiner -> constrained planner`。backbone 入口只保留实际实现名 `transformers.Mamba2Model`，对应 Hugging Face Transformers 纯 PyTorch Mamba2；不再暴露 `mamba2`、`torch_mamba2`、BiGRU、TCN 等同义或 fallback 入口。
- 新增 `tools/boundary/build_refiner_gap_dataset.py`：读取 FusionVAD label JSONL + feature manifest，按 runtime `RefinerInput` / `DEFAULT_REFINER_FEATURES` 构造 supervised gap samples。feature manifest 断兼容要求 `ptm_dim`，不再读取旧 `whisper_dim`。输出 `boundary_refiner_gap_dataset_v1` JSONL 和 class balance summary。
- 新增 `tools/boundary/train_refiner.py`：训练 gap-level `BoundarySequenceClassifier(transformers.Mamba2Model)`，保存标准 `boundary_refiner_v1` checkpoint，并立即通过 `load_boundary_refiner()` 做 loader smoke。当前第一版是 gap-level classifier，目的是打通 schema / cache / runtime loader；后续再扩为相邻 gap 序列、dense PTM/MFCC window、preliminary ASR signal 或 RL/DPO。
- 单测：`tests/test_boundary_refiner_training.py` 覆盖 dataset builder、缺失 `ptm_dim` 报错、checkpoint round trip 和 loader smoke。相关回归 `tests/test_boundary_refiner_training.py tests/test_boundary_refiner.py tests/test_boundary_planner.py tests/test_boundary_cache.py tests/test_pipeline_chunk_config_runtime.py`：`28 passed`。
- v1.22 smoke：64 条 feature rows 生成 `323` 个 gap samples，其中 `merge_positive=64`、`split_negative=259`。产物：`agents/temp/boundary-refiner/v1-smoke/gaps.jsonl` 和 summary。
- 训练 smoke：CPU 与提权 CUDA 各跑 3 steps，均能保存并加载 checkpoint。CUDA 产物：`agents/temp/boundary-refiner/v1-smoke/train-cuda/boundary_refiner.pt`，metrics：`agents/temp/boundary-refiner/v1-smoke/train-cuda/metrics.json`。受限 sandbox 内 `torch.cuda.is_available=False` 且 NVML 初始化失败；提权后 `.venv` 中 `torch 2.12.0+cu130` 可见 RTX 4060 Ti，确认训练需要提权 CUDA。
- Transformers Mamba2 会提示 `fast path is not available ... Falling back to the naive implementation`。这是当前 Windows-friendly / pure PyTorch 默认路径，符合分发目标；不把 Linux-only `mamba-ssm`、Triton 或自定义 CUDA kernel 放进默认依赖。
- 数据限制：v1.22 cutpoint 数据主要提供 split supervision（贴连换人、短 gap 换人、可删除长 gap），不能单独训练完整 merge/split policy。正式训练前需要混入 clean speech-island 原料构造 same-utterance merge-positive。
- 新数据源候选：Grok/HF 页面确认 `joujiboi/japanese-anime-speech-v2` 约 292,637 audio-text pairs、约 450 小时 anime / visual novel speech，平均 SFW clip 约 5.3s，GPL。它适合作为额外 clean speech-island 原料源，与 `litagin/Galgame_Speech_ASR_16kHz` 一起合成多 island、touching、short gap、long gap、speaker/source switch 的 Boundary Refiner 训练集；进入正式训练前需本地审计 license、字段、下载规模和文本质量。
- 数据源落地：`tools/vad/fusionvad_ja/materialize_hf_audio.py` 已支持 `txt`、`text`、`transcription`、`transcript`、`sentence` 文本字段。`joujiboi/japanese-anime-speech-v2` 实际 split 为 `sfw` / `nsfw`；按目标域需要保留 NSFW 并提高权重。已 materialize `nsfw=512`、`sfw=256`，`litagin/Galgame_Speech_ASR_16kHz` 复用现有 `galgame-materialized-512`。
- 旧生成数据清理：历史 generated datasets / feature caches / train-v0 旧产物已移入 `agents/rm/generated-boundary-datasets-20260604/`，源数据和负样本素材保留。后续如需释放空间再人工确认清理 `agents/rm/`。
- v1.23 mixed source manifest：新增 `tools/boundary/build_weighted_source_manifest.py`，按 `anime_nsfw=45`、`galgame=35`、`anime_sfw=20` 采样 `20000` 行，实际 group counts 为 `9000/7000/4000`。目的不是排除 NSFW，而是让 JAV 目标域更贴近，同时保留 galgame 和 SFW 泛化。
- v1.23 mixed synthetic timeline：输出 `datasets/train/fusionvad-ja/v1-23-boundary-refiner/mixed-nsfw45-galgame35-sfw20-boundary4096/`。4096 records，总时长 `129092.42s`，speech frame ratio `0.8566`，speaker turn boundaries `16384`，cut point segments `11444`，cut drop zones `4882`。内部 gap policy：regular `4940`、short `7389`、touch `4055`；gap mode：real_negative `12319`、fade_noise `2112`、hum `2006`、silence `2061`、white_noise `2023`；background mix `1663`、overlap mix `326`。
- feature cache 取舍：最初压缩 `.npz` 写入时 write 约 `6s/batch`，明显拖慢 CUDA；用户确认磁盘空间充足后切到 `--no-compress`。无压缩 cache 输出 `datasets/train/fusionvad-ja/v1-23-boundary-refiner/qwen3-asr-0.6b-full29239/mixed-nsfw45-galgame35-sfw20-boundary4096-feature-cache-nocompress/`，4096/4096 cached，errors/skipped `0`，大小约 `26G`；write 降到约 `0.18-0.20s/batch`，瓶颈回到 PTM 前向。
- v1.23 gap dataset：`synthetic_merge_positives_per_record=1` 生成 `20711` gap samples，`merge_positive=4096`、`split_negative=16615`。label reasons：`merge_synthetic_intra_island=4096`、`split_speaker_change=6750`、`split_overlap=4984`、`split_gap_zone=4777`、`split_long_gap=104`。feature dims：`ptm_dim=1024`、`mfcc_dim=40`。
- v1.23 learned Boundary Refiner v0：CUDA 训练 `300` steps，`batch_size=512`、`lr=5e-4`、`weight_decay=0.01`、hidden `128`、layers `2`、state `32`，产物 `datasets/train/fusionvad-ja/v1-23-boundary-refiner/qwen3-asr-0.6b-full29239/boundary-refiner-mamba2-mixed4096-v0/boundary_refiner.pt`。val：accuracy `0.99565`，merge precision `0.97837`，merge recall `1.0`，merge F1 `0.98906`，FP `9`、FN `0`。注意：这个指标主要验证 schema / 数据闭环 / first supervised signal 成立，不能替代匿名样片 GPU downstream 验收。
- v1.23 Mamba2 v0 downstream 诊断口径修正：`measure_fallback_safe_boundaries.py` 不再用旧 `vad_coarse_after_sentinel` / bucket / quality 规则库反推 fallback，也不保留 dataclass/helper 别名层；直接以当前 diagnostics 原字段为准：`fallback_type != none` 才算 alignment fallback，`fallback_subtype` 原值只做 reason，`sentinel_lines` 非空才计 sentinel fallback，缺失 `fallback_type` 的旧 diagnostics 直接报错。修正后匿名样片 A v0：chunks `1098`，alignment fallback `387`，unsafe `113`，safe ratio `0.708`，fallback duration p50/p90/max `5.54 / 10.80 / 23.09s`，unsafe p50/p90/max `10.25 / 12.76 / 23.09s`；reason counts 为 `proportional_after_sentinel=373`、`asr_qc_reject=14`。
- 2026-06-04 固定 learned refiner artifact 策略：canonical path 改为 `src/boundary/checkpoints/boundary_refiner.pt`。该文件不存在时默认走 deterministic bootstrap refiner；文件存在时加载 `boundary_refiner_v1` learned checkpoint，并把路径、SHA1、backbone 和 planner config 纳入 boundary-cache signature。后续如果 checkpoint 体积可控且作为默认质量路径，可随 GitHub 源码提交；版本号和训练数据说明记录在 README / HISTORY，不恢复旧 `src/vad` checkpoint 路径。
- Seq2Seq 过渡入口：`tools/boundary/build_refiner_gap_dataset.py` 新增 `--output-sequence-jsonl`，可把同一音频/feature row 的 gap samples 聚合成 `boundary_refiner_sequence_dataset_v1`，包含 `sequence_features`、`sequence_labels`、`sequence_reasons` 和 gap indexes。`tools/boundary/train_refiner.py` 改为 padded sequence training：gap row 自动升为长度 1，sequence row 按 mask 计算 BCE loss 和指标。当前仍是候选级 sequence 过渡，不是最终 dense PTM/MFCC frame Seq2Seq；下一步继续把输入扩展到连续窗口特征和候选 offset/refine label。
- Frame/window sequence 数据层：新增 `tools/boundary/build_refiner_frame_sequence_dataset.py`，从 feature cache `.npz` 读取 `ptm/mfcc`，按相邻 speech island gap 构造候选序列。每个 step 使用 left/gap/right 窗口的 PTM/MFCC mean/std 统计，输出 `boundary_refiner_frame_sequence_dataset_v1` 的 `sequence_features` / `sequence_labels`，可直接被 `train_refiner.py` 训练。训练和 runtime 均使用 `src/boundary/sequence_features.py` 校验 feature names/hash，避免手写维度常量。
- Runtime adapter 接入：新增 `FrameSequenceBoundaryRefiner` / `load_frame_sequence_refiner_checkpoint()`，能加载同一个 `boundary_refiner_v1` checkpoint，并对外部传入的 candidate/window `sequence_features` 输出逐 step `BoundaryDecision`。随后接入 `FrameSequenceFeatureProvider` 和 planner `sequence_refiner` 参数；当 checkpoint metadata 标记 `runtime_adapter=frame_sequence_v1` 时，pipeline 会临时要求 SpeechBoundary-JA 导出低维 PTM/MFCC frame windows，构造 left/gap/right sequence features，并优先用 sequence refiner 决策相邻 speech island 是否合并。若 checkpoint 的 feature schema/hash 与 runtime 不一致，直接报错。下一步是训练正式 checkpoint 后做匿名样片 GPU 闭环，验证 fallback duration、start boundary、ASR empty/hallucination 是否改善。
- 长期维护评估：采纳 `get_default_config()` / `get_feature_dim()` / `validate_sequence_features()` 和 checkpoint `feature_schema_hash`，因为它们能把 feature schema 变成单一事实源，减少断兼容重构后的隐性硬编码。拒绝把大体积 `sequence_feature_frames` 塞进 boundary-cache JSON；runtime 只临时导出并用于 planning，训练数据继续使用 `.npz` / `.pt` sidecar。当前 PackedChunk 已写入 `boundary_decision_merge`、`boundary_merge_prob`、`boundary_split_prob`、`boundary_refine_delta_s`、`boundary_decision_source`，供 ASR QC、forced alignment 和审计追踪。下一步不是继续增加 helper/alias，而是把当前 per-gap sequence 调用升级成整段候选一次性批量打分，再接轻量 DP / Viterbi constrained planner，发挥 Mamba2 长上下文优势。
- v1.23 frame-sequence Mamba2 v1 训练完成：dataset `boundary-refiner-frame-sequence-v1` 共 `4096` 条 sequence、`20711` 个 sequence items，`feature_dim=630`，`feature_schema_hash=eb441dce527ffc4d75bcdd82f6aeb5df1e6ec9ba`。CUDA checkpoint：`datasets/train/fusionvad-ja/v1-23-boundary-refiner/qwen3-asr-0.6b-full29239/boundary-refiner-frame-sequence-mamba2-v1/boundary_refiner.pt`，体积约 `2.2MB`。validation：precision `1.0`、recall `0.99756`、F1 `0.99878`、FP `0`、FN `1`。这是 synthetic validation，不等同 downstream 质量。
- 匿名样片 A greedy frame-sequence GPU 闭环完成：脚本 `agents/temp/run_v1_23_frame_sequence_refiner_sample_a_gpu.sh`，输出 `agents/temp/speech-boundary-ja/full-workflow-anon-a-v1-23-frame-sequence-refiner-mamba2-v1/`，使用 Qwen3-ASR-1.7B full SFT + Qwen3-ForcedAligner。结果：chunks `823`，segments `2154`，blocks `1939`，ASR+alignment `1830.7s`。diagnostics：forced `416`、partial `2`、proportional `297`、nonlexical `62`、drop_or_review `46`；alignment fallback `322`，其中 sentinel `322`。fallback-safe：fallback duration p50/p90/p95/max `10.41 / 14.07 / 16.39 / 26.88s`，unsafe fallback `221`，safe ratio `0.314`，fallback speech-island p90 `2`，long silence crossing `9`。质量报告仍告警 short segment ratio `0.399`、per-minute subtitle count `21.6`。结论：learned frame-sequence refiner 已真实接入（`learned_sequence_split=235`），但 greedy 用法仍不足以作为默认质量路径。
- v1.23 planner 升级：`SequenceBoundaryRefiner.decide_sequence()` 已从 per-gap 调用改为一次性批量打分；planner 接入轻量 DP / Viterbi-style 全局规划，在 split/merge score、target duration、max chunk、start weight 和长 gap 代价之间做分段决策。DP 可为了 fallback-safe target 切开高 merge score gap，但会把诊断写成 `source=boundary_planner` / `reason=planner_dp`，避免误报成模型硬 split。planner signature 升级为 `constrained_sequence_dp_planner_v2`，pipeline signature 升为 `boundary_pipeline.version=2`，旧 boundary-cache 需要重建。
- v1.23 DP v2 离线 boundary inspection：新增 `tools/boundary/inspect_boundary_packing.py`，只跑 SpeechBoundary-JA + Boundary Refiner + planner，不跑 ASR/aligner，用于快速检查 packed chunk 分布。匿名样片 A prepared wav 重算结果：chunks `974`（greedy `823`），duration p50/p90/p95/max `8.57 / 12.64 / 12.99 / 21.14s`，core p50/p90/p95/max `5.85 / 8.87 / 9.14 / 18.89s`，speech-island count p90/p95 `1/1`、max `2`，internal gap max p95 `0`、max `0.14s`。split reasons：`valley_candidate=468`、`learned_sequence_split=384`、`cut_candidate=118`、`planner_dp=3`。结论：DP v2 离线分布明显更贴近“一句台词一个 chunk”，值得跑完整 GPU 闭环；成本是 chunk 数约 `+18%`，符合 chunk growth 只作成本指标的路线。
- 性能坑与修复：Hugging Face `transformers.Mamba2Model` pure PyTorch naive path 能在 Windows-friendly 路线上运行，但全片候选一次性打分会有明显 planner 耗时。已新增 `BOUNDARY_PLANNER_SEQUENCE_BATCH_SIZE`（默认 `256`）做 bounded sequence batching，并纳入 planner/cache signature。batched inspection 复测：chunks `973`，duration p50/p90/p95/max `8.60 / 12.64 / 12.99 / 21.14s`，core p50/p90/p95/max `5.88 / 8.87 / 9.14 / 18.89s`，speech-island count p90/p95 `1/1`、max `2`，split reasons `valley_candidate=468`、`learned_sequence_split=381`、`cut_candidate=118`、`planner_dp=5`。分布与未分批基本一致，下一步跑完整 GPU 闭环；如仍慢，再加 batch overlap、candidate pruning 或缓存 refiner logits。
- v1.23 DP v2 匿名样片 A GPU 闭环完成：脚本 `agents/temp/run_v1_23_frame_sequence_refiner_dp_v2_sample_a_gpu.sh`，输出 `agents/temp/speech-boundary-ja/full-workflow-anon-a-v1-23-frame-sequence-refiner-dp-v2/`，diagnostics `agents/temp/speech-boundary-ja/diagnostics-anon-a-v1-23-frame-sequence-refiner-dp-v2/`，fallback-safe `agents/temp/speech-boundary-ja/fallback-safe-boundary-metrics-anon-a-v1-23-frame-sequence-refiner-dp-v2/`。结果：chunks `973`，segments `2242`，blocks `2018`，ASR+alignment `1902.1s`，total `1932.2s`。diagnostics：forced `488`、partial `1`、proportional `342`、nonlexical `97`、drop_or_review `45`；alignment fallback `366`，sentinel `366`。对比 greedy frame-sequence：forced `416 -> 488`，fallback `322 -> 366`，nonlexical `62 -> 97`，segments `2154 -> 2242`，ASR+alignment `1830.7s -> 1902.1s`。fallback-safe：fallback duration p50/p90/max `10.41 / 14.07 / 26.88s -> 9.32 / 12.81 / 20.72s`，unsafe `221 -> 213`，safe ratio `0.314 -> 0.418`，long silence crossing `9 -> 10`。结论：DP v2 证明“全局规划能压粗 fallback 时间轴”，尤其消除了多 island 拼成 `26.88s` 的最坏样式；但 sentinel/fallback 数仍上升，repeat-loop / nonlexical 仍多，说明当前 cost 还不是最终字幕目标函数，不能直接固化为默认质量路径。
- DP cost 检索与路线判断：Grok 复核 Netflix Timed Text Timing Guidelines、SubER、OptiSub、REBORN、DPDP / segmental speech segmentation 后，当前 DP 框架方向成立，但代价函数要从简单 `duration + merge_score + gap` 升级为可审计分项。字幕侧依据：Netflix timing 强调 in-time 贴近第一帧音频、字幕间 2-frame gap、最小显示时长，out-time 可在不冲突时延后或由 polish 压缩；SubER / OptiSub 说明自动字幕质量同时包含 timing、segmentation、duration、CPS/readability。ASR 侧依据：REBORN 用 lower perplexity 作为 boundary reward 证明下游 ASR feedback 可反哺 segmentation，但本项目有 synthetic exact-island 和 forced-aligner/QC 诊断，第一阶段仍应先用 deterministic DP 做可解释 baseline，第二阶段再接 local CER / token confidence / sentinel / fallback duration 做 RL 或 DPO。
- 下一轮 DP cost 计划：把 cost 显式拆成 `model_nll`（校准 `merge_prob/split_prob`，替代线性 `1-score`）、`duration/readability`（target/min/max、最小显示时长、CPS 可选）、`gap_crossing`（长 silence/BGM/noise/real-negative gap 高惩罚）、`start_boundary`（start 错误权重大于 end，end 可由 cue polish 压缩）、`fallback_safety`（>8s fallback、20-30s 粗时间轴强惩罚）和 `asr_feedback`（ASR empty、repeat-loop、aligner sentinel、QC reject）。实现上先离线重算权重 sweep，不直接改默认；通过后再跑匿名样片 A GPU 闭环，并用审计页抽查长 unsafe fallback。
- 验证：`tests/test_boundary_refiner.py tests/test_boundary_refiner_training.py` 17 passed；聚焦回归 `tests/test_config.py tests/web/test_jobs_api.py tests/test_asr_backend_dispatch.py tests/test_boundary_cache.py tests/test_boundary_candidates.py tests/test_boundary_planner.py tests/test_boundary_refiner.py tests/test_boundary_refiner_training.py tests/test_chunk_packer.py tests/test_pipeline_chunk_config_runtime.py tests/test_pipeline_chunk_packing.py tests/test_run_full_workflow_env.py tests/test_speech_boundary_refine.py tests/test_boundary_ja_current.py` 93 passed；`git diff --check` passed。
- 2026-06-05 DP cost 参数化与真实重跑 sweep：按用户要求先提交上一版保存点（commit `ebf91d5 Refactor speech boundary sequence planner`，未 push），随后把 `BoundaryPlannerConfig` 的 DP cost 拆成 env 可控项：`BOUNDARY_DP_CHUNK_BASE_COST`、`BOUNDARY_DP_OVER_TARGET_WEIGHT`、`BOUNDARY_DP_FAR_OVER_TARGET_WEIGHT`、`BOUNDARY_DP_UNDER_MIN_WEIGHT`、`BOUNDARY_DP_LONG_GAP_WEIGHT`、`BOUNDARY_DP_SPLIT_MERGE_WEIGHT`，纳入 pipeline / cache signature / `.env.example` / README。新增 `tools/boundary/sweep_dp_costs.py`，旧版先尝试用既有 boundary-cache 做近似 fallback-safe 模拟，后因用户确认允许重跑，改为真实 boundary-only sweep：同一 prepared wav 只跑一次 SpeechBoundary-JA + frame-sequence features，再对多组 DP profile 真实调用 `pack_speech_segments()`，不重跑 ASR/aligner。产物：`agents/temp/speech-boundary-ja/dp-cost-real-sweep-v1/summary.md`。
- 真实 sweep 结果：SpeechBoundary-JA segment time `24.48s`，speech segments/groups `394/334`。baseline DP v2：chunks `973`，duration p50/p90/max `8.60/12.64/21.14s`，映射上一轮 fallback-risk p50/p90/max `9.52/12.81/21.14s`，>20s chunks `2`。`duration_tight_8s` / `gap_strict_8s` / `fallback_safe_8s` / `start_priority_8s` 都收敛到 chunks `1034`，duration p50/p90/max `8.24/11.86/21.14s`，映射 fallback-risk p50/p90/max `9.33/11.98/21.14s`，>20s chunks 仍 `2`。结论：单纯调 DP cost 能略降 p90，但无法消除 20s+ 长 chunk；下一步优先补 overlong speech island 的候选切点 / dense boundary labels，而不是继续盲调 cost。
- refiner 设备坑：第一次真实 sweep 发现 SpeechBoundary-JA PTM 已在 CUDA，但 `load_frame_sequence_refiner_checkpoint()` 默认把 learned Boundary Refiner 留在 CPU，Transformers Mamba2 pure PyTorch naive path 导致后半段长时间 CPU 184%。已新增 `BOUNDARY_REFINER_DEVICE=auto`，loader 会在 CUDA 可见时把小 refiner 放到 GPU，并把 requested/actual device 写入 refiner signature；pipeline、研究脚本、cache signature 和 `.env.example` 已同步。重跑确认 `refiner_signature.actual_device=cuda:0`。
- 质量口径修正：fallback / sentinel 数量只作为观察项，不再作为硬 gate。原因是当前 Qwen3-ForcedAligner 未做 JAV / galgame 目标域 finetune，和 Qwen ASR SFT 输出风格可能不完全匹配；当前主 gate 应是 start boundary、fallback-risk duration、20-30s 粗 chunk、ASR empty / hallucination / repeat-loop 和字幕观感。Backlog 新增“直接字幕边界 / timeline model”：等 SpeechBoundary-JA 能稳定产出 pseudo boundary labels 后，研究不依赖 forced aligner 的字幕文本 + 时间轴边界模型，forced aligner 只作为审计或 teacher 信号之一。
- 2026-06-05 overlong single-island soft candidate：candidate extractor 升到 v2，新增 `soft_cut` / `soft_valley`。当 over-target 单 speech island 没有 hard cut / valley 时，在 target 附近搜索 soft cut score 或 speech-score valley，避免连续 speech island 因缺少候选切点残留 20s+ 粗 chunk。真实 boundary-only sweep v2（不跑 ASR/aligner，但重跑 SpeechBoundary-JA + frame-sequence refiner + DP planner）结果：baseline chunks `1044`，duration p50/p90/max `8.12/12.34/14.40s`，`>20s=0`；8s profiles chunks `1144`，duration p50/p90/max `7.66/11.40/15.36s`，`>20s=0`。对比 v1 的 `>20s=2`，soft candidate 解决了那两个单 island 长 chunk。产物：`agents/temp/speech-boundary-ja/dp-cost-real-sweep-v2-soft-candidate/summary.md`。
- DP sweep 性能口径：后半段 CPU 高占用不是“近似”。SpeechBoundary-JA PTM 和 learned Boundary Refiner 可用 CUDA，summary 已确认 `refiner_signature.actual_device=cuda:0`，segment time `27.29s`；但每个 profile 的 `pack_speech_segments()`、风险区间映射和 JSONL 写入是 CPU-bound，单 profile 约 `210s`。因此 planner 后半段可以接受 CPU 跑；只有特征提取、训练、ASR/aligner 等模型阶段误落 CPU 才需要停掉重跑。
- soft candidate 完整 GPU 闭环：脚本 `agents/temp/run_v1_23_frame_sequence_refiner_dp_v2_soft_candidate_sample_a_gpu.sh`，输出 `agents/temp/speech-boundary-ja/full-workflow-anon-a-v1-23-frame-sequence-refiner-dp-v2-soft-candidate/`，diagnostics `agents/temp/speech-boundary-ja/diagnostics-anon-a-v1-23-frame-sequence-refiner-dp-v2-soft-candidate/`，fallback-safe `agents/temp/speech-boundary-ja/fallback-safe-boundary-metrics-anon-a-v1-23-frame-sequence-refiner-dp-v2-soft-candidate/`。完整链路结果：chunks `1044`，segments `2440`，blocks `2160`，ASR+alignment `1924.2s`，total `1955.3s`；diagnostics：forced `537`、partial `1`、proportional `361`、nonlexical `102`、drop_or_review `43`；alignment fallback `385`，sentinel `385`。对比旧 DP v2：chunks `973 -> 1044`，forced `488 -> 537`，fallback `366 -> 385`，nonlexical `97 -> 102`，segments `2242 -> 2440`。fallback-safe：fallback duration p50/p90/max `9.32/12.81/20.72s -> 8.73/12.60/13.10s`，unsafe `213 -> 211`，safe ratio `0.418 -> 0.452`，long silence crossing `10 -> 11`。结论：soft candidate 真实消除了最坏 `20s+` fallback 粗 chunk，可作为 overlong safety baseline；但它没有解决 ASR repeat-loop / nonlexical 和 forced-aligner sentinel，下一步要转向 dense boundary labels、ASR/QC feedback 和审计页观感确认。
- fallback-window 完整 GPU 闭环：脚本 `agents/temp/run_v1_23_frame_sequence_refiner_dp_v2_soft_candidate_fallback_window_sample_a_gpu.sh`，输出 `agents/temp/speech-boundary-ja/full-workflow-anon-a-v1-23-frame-sequence-refiner-dp-v2-soft-candidate-fallback-window/`，diagnostics `agents/temp/speech-boundary-ja/diagnostics-anon-a-v1-23-frame-sequence-refiner-dp-v2-soft-candidate-fallback-window/`，fallback-safe `agents/temp/speech-boundary-ja/fallback-safe-boundary-metrics-anon-a-v1-23-frame-sequence-refiner-dp-v2-soft-candidate-fallback-window/`。结果：chunks `1044`，segments `2388`，blocks `2125`，ASR+alignment `1914.2s`，total `1944.8s`；diagnostics：forced `537`、partial `1`、proportional `361`、nonlexical `102`、drop_or_review `43`；alignment fallback `385`，sentinel `385`。fallback-safe 新口径：`fallback_duration_s` 使用 speech core window，p50/p90/p95/max `6.10/8.76/8.87/9.10s`；`fallback_padded_chunk_duration_s` 保留原 ASR chunk 口径，p50/p90/p95/max `8.73/12.60/12.81/13.10s`；unsafe `103`，safe ratio `0.732`，long-silence crossing `8`，speech-island count p90/p95 `1/1`。结论：本轮没有减少 forced-aligner sentinel 数，但解决了“ASR padding 被误计入 fallback 时间轴”的粗时间轴问题；下一步继续沿原路线做 dense boundary labels / ASR-QC feedback 和审计页观感确认，不切到 speaker sidecar。
- fallback-window 风险审计页：`tools/audits/generate_long_fallback_chunk_audit_html.py` 改为 fallback-window aware，主播放区间使用 `fallback_window_start/end`，同时展示 padded chunk 与 speech core 对照，人工标签改为多选。已基于最新 unsafe rows 生成 `agents/audits/fallback-window-risk-audit/index.html`，40 条样本，完整日语字幕 VTT `agents/audits/fallback-window-risk-audit/full.ja.vtt`。随后为 NSFW 视频不便打开的审计场景，生成器新增 `--media-mode audio|video`；audio 模式用 ffmpeg 从源视频抽取 `audit_audio.m4a`，页面使用 `<audio>` 播放并在下方同步显示完整日语字幕当前 cue。已生成 `agents/audits/fallback-window-risk-audit-audio/index.html`，导航 `agents/audits/index.html` latest 已指向音频版。目的：人工确认剩余 `8-9s` fallback 是否确实需要继续切分、是否只是低信息人声/重复循环、以及 soft-candidate 新切点是否影响观感。
- live-server 审计导航断兼容重构：删除按钮不再依赖独立 Python HTTP 服务，只保留 live-server middleware 写文件入口。新增 `tools/audits/live_server_audit_middleware.js`，从项目根目录用 `live-server --middleware=tools/audits/live_server_audit_middleware.js` 启动后，导航页调用 `POST /__audit_api__/delete-audit`，middleware 再执行 `uv run python tools/audits/audit_nav.py delete --href ...`，删除仍统一移动到 `agents/rm/audit-deletions/` 并重建 `agents/audits/index.html` / `latest-audit.html`。该方案不新增 serve 依赖，适配用户从项目根目录启动 live-server 的审计习惯。
- 审计目录清理：按“只保留最新适配审计”的要求，`agents/audits/` 当前只保留 `fallback-window-risk-audit-video/`、`index.html`、`latest-audit.html`。旧的 cue-planner、audio fallback-window、plain fallback-window、soft-candidate 审计目录均通过 `audit_nav.py delete` 移动到 `agents/rm/audit-deletions/`，没有硬删除。
- fallback-window 审计标签补齐：原页面只有 `时间轴准确` 正向标签，无法区分时间轴偏前、偏后或窗口粗细。已把标签 schema 收口为通用 `timing_accurate`，并新增 `needs_realign`、`timing_start_early`、`timing_start_late`、`timing_end_early`、`timing_end_late`、`timing_window_too_long`、`timing_window_too_short`、`timing_crosses_gap_noise`。后续人工审计优先多选这些结构化 timing 标签，少依赖备注；旧 fallback-window 审计结果若使用旧 `timing_ok`，需要重审或映射后再统计。
- NAMH-055 `.env` 正常加载短 core 完整闭环：输出 `agents/temp/speech-boundary-ja/namh055-env-current-seconds/`，fallback-safe `agents/temp/speech-boundary-ja/fallback-safe-boundary-metrics-namh055-env-current-seconds/summary.json`。配置为 `BOUNDARY_PLANNER_TARGET_CHUNK_S=3.0`、`BOUNDARY_PLANNER_MAX_CORE_CHUNK_S=5.0`、`BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S=9.0`，全片 chunks `2459`、aligned segments `3794`、原 ja-only SRT blocks `3233`、ASR+alignment `371.0s`、total `406.2s`。fallback core p50/p90/p95/max `1.20/3.07/3.86/5.00s`，unsafe fallback `0`、safe ratio `1.0`。结论：边界/fallback 粗时间轴在 NAMH-055 上已达安全口径，当前主要问题转为日文 cue density 和低信息人声/重复语气词，而不是继续把 ASR chunk 放粗。
- cue-density runtime 修复：发现 `prepare_srt_blocks()` 过去不是单次收敛。第一次 prepare 先按 raw alignment gap 合并，再 timing polish 把部分短 gap 压到 2-frame gap；这些新可合并的 micro cues 只有手动第二遍 prepare 才会合并。现在 `_prepare_subtitle_blocks()` 在 timing polish + final no-overlap normalize 后，再执行一次受 `SUBTITLE_MERGE_ADJACENT` 控制的 bounded short-cue merge，并立刻做最终 normalize。`prepare_srt_blocks()` 同时断开旧的 `mode == bilingual` 隐式开关，日文-only / 双语统一由 `SubtitleOptions.merge_adjacent` 控制；dense cue merge 也受同一个总开关约束。
- NAMH-055 ja-only cue replay 结果：工具 `tools/subtitles/replay_subtitle_postprocess.py` 新增 `--source aligned` / `--mode srt|bilingual` 后，用最新 aligned segments 重放。final merge 前：blocks `2166`、per-minute `24.01`、short_segment_ratio `0.1316`、duration p50/p90/max `1.447/3.533/5.65s`。final merge 后单次 prepare 已直接得到 blocks `2002`、per-minute `22.19`、short_segment_ratio `0.1169`、duration p50/p90/max `1.733/3.533/5.65s`、overlap `0`、nonlexical repetition count `14`。产物：`agents/temp/speech-boundary-ja/namh055-env-current-seconds-ja-cue-replay-final-merge/summary.json`。
- cue planner analyzer 断兼容扩展：`tools/subtitles/analyze_subtitle_cue_merge_candidates.py` 新增 `--aligned`、`--source blocks|aligned`、`--mode srt|bilingual`，可直接分析 ja-only aligned segments，并按模式导出 `before/planner.{ja,bilingual}.srt/.vtt`。在 NAMH-055 final-merge 后复算，额外 planner merges 为 `0`、blocks `2002 -> 2002`；这说明当前最大收益来自 runtime final merge，而不是 analyzer 的临时 planner score。后续 cue-density 路线应继续研究 low-info vocal 合并/标记、speaker-aware merge guard 和更强 cue planner，不要把这次收益误归因给 `min_score=0.62` 的 planner heuristic。
- 验证：`tests/test_subtitle_quality_pass.py tests/test_subtitle_options.py tests/test_replay_subtitle_postprocess.py tests/test_subtitle_cue_merge_candidates.py` 52 passed。
- ASR 低信息/重复/幻觉归因审计入口落地：新增 `tools/audits/generate_asr_attribution_audit_html.py`，从 alignment diagnostics + aligned segments + 完整日语 SRT 离线抽样，不重跑 ASR/aligner。采样桶固定为 `repeat_or_qc_reject`、`nonlexical_empty`、`sentinel_fallback`、`low_info_vocal`、`asr_qc_warn`、`forced_control`，用于区分真实低信息人声、ASR 幻觉/错听、非语音噪声/BGM、多人/重叠、轻声弱人声、边界上下文过短/过长、文本可用和时间轴准确等原因。NAMH-055 当前 diagnostics 全量统计：chunks `2459`，alignment quality 为 forced `1200`、proportional `861`、nonlexical `378`、drop_or_review `20`；ASR QC ok/warn/reject 为 `2102/337/20`；fallback subtype 为 none `1200`、proportional_after_sentinel `861`、nonlexical_text `378`、asr_qc_reject `20`；low-information 分布为 not_low_information `1132`、short_kana `462`、short_nonlexical `385`、empty `378`、repeated_nonlexical `98`、long_sparse `4`。已生成音频审计页 `agents/audits/asr-attribution-namh055-audio/index.html`，共 `84` 条、每桶 `14` 条，完整日语 VTT `agents/audits/asr-attribution-namh055-audio/full.ja.vtt`，人工导出文件名 `manual_asr_attribution_labels.jsonl`；导航 latest 已指向该页。该步骤是评估闭环：先看人工分布，再决定是否改 ASR QC、cue planner、speaker sidecar、hard-negative 训练或 0.6B/1.7B 稳定性对比。
- 验证：`tests/test_asr_attribution_audit.py tests/test_audit_nav.py tests/test_long_fallback_audit_media_mode.py` 7 passed；`python -m py_compile tools/audits/generate_asr_attribution_audit_html.py` passed。

---

## 已降级路线

- `fusion_lite`：保留为 baseline / fallback 思路，不再是当前默认 VAD。
- FSMN / Silero / TEN：保留为 teacher、baseline、hard-negative miner 或未来小模型蒸馏候选，不作为当前主切分路线。
- F0 / gender：不再作为主线切分或翻译提示。原因是大 chunk 混合多 speech island 或男女交替时，F0/gender 会被稀释并引入噪声。
- pyannote：强 baseline 可参考，但官方预训练 diarization 模型通常需要 HF token / 条款接受，不进默认依赖。
- forced aligner finetune：暂不做。当前没有公开可复用的 Qwen3-ForcedAligner finetune recipe，也没有字/词级时间轴真值。

---

## 常见坑

- Codex sandbox 可能隔离 GPU；全片 VAD/ASR/ForcedAligner、ONNXRuntime CUDA、Torch CUDA、feature cache 或训练需要提权，并确认 `actual_device=cuda` / `model_param_device=cuda:*` / `CUDAExecutionProvider`。
- Torch CUDA 也可能在 sandbox 内显示 `cuda_available=False`，但提权后同一个 `.venv` 可正常看到 GPU。2026-06-04 Boundary Refiner smoke 即为例：sandbox 报 NVML 初始化失败，提权后 RTX 4060 Ti 可见。
- SpeechBoundary-JA 的 feature cache、训练、逐帧概率导出和全片 workflow 不再用 CPU 跑大规模任务；能 CUDA 就提权 CUDA。2026-06-02 曾用 CPU 导出早期 drop_gap 逐帧概率，虽然跑完但效率差且产物已移入 `agents/rm/fusionvad-ja-cpu-dropgap-probabilities-20260602/`，后续重跑走 CUDA。
- 联网默认受限；Hugging Face / ModelScope 下载、`uv pip`、`npm install`、`curl`、`git fetch`、外部搜索或 API 探测遇到网络错误时，先按“需要提权或代理环境”处理。
- 长跑命令不要静默后台化后直接退出 shell；全片 workflow / 训练 / 大规模评测要么前台持有进程，要么在同一 shell 内循环 tail 日志并 `wait`。
- WSL2 8GB RAM 下，大 batch feature cache 可能因主机内存被 kill 而没有 Python traceback。
- Qwen 后端曾频繁输出 temperature / pad token warning；根因是 greedy generation 下 sampling-only 参数被忽略，以及底层 generation_config 缺 pad token。修复方向是在加载后归一化 generation_config，不改变 greedy 解码语义。

---

## 参考来源

- WhisperJAV: <https://github.com/a63n/WhisperJAV>
- FusionVAD: <https://arxiv.org/abs/2506.01365>
- Whisper hallucination on non-speech: <https://arxiv.org/abs/2501.11378>
- Dynamic Speech Endpoint Detection: <https://arxiv.org/abs/2210.14252>
- Semantic VAD: <https://arxiv.org/abs/2305.12450>
- WhisperX: <https://github.com/m-bain/whisperX>
- stable-ts: <https://github.com/jianfch/stable-ts>
- Qwen3-ASR: <https://github.com/QwenLM/Qwen3-ASR>
- Qwen3-ASR finetuning: <https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning>
- Qwen3-ASR-0.6B: <https://huggingface.co/Qwen/Qwen3-ASR-0.6B>
- Qwen3-ASR-1.7B: <https://huggingface.co/Qwen/Qwen3-ASR-1.7B>
- Qwen3-ForcedAligner-0.6B: <https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B>
- 本项目 Qwen3-ASR-0.6B SFT: <https://huggingface.co/jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame>
- 本项目 Qwen3-ASR-1.7B SFT: <https://huggingface.co/jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame>
- AVA-Speech VAD: <https://huggingface.co/datasets/nccratliri/vad-human-ava-speech>
- VoxConverse: <https://huggingface.co/datasets/diarizers-community/voxconverse>
- MUSAN: <https://www.openslr.org/17/>
- DNS Challenge: <https://github.com/microsoft/DNS-Challenge>
- pyannote speaker diarization: <https://huggingface.co/pyannote/speaker-diarization-3.1>
- 3D-Speaker: <https://github.com/modelscope/3D-Speaker>
- WeSpeaker / CAM++: <https://github.com/wenet-e2e/wespeaker>
- Reazon Japanese HuBERT: <https://huggingface.co/reazon-research/japanese-hubert-base-k2>
- rinna Japanese HuBERT: <https://huggingface.co/rinna/japanese-hubert-base>
- rinna Japanese wav2vec2: <https://huggingface.co/rinna/japanese-wav2vec2-base>
- NonverbalTTS: <https://arxiv.org/abs/2507.13155>
- Rochester non-word transcription notes: <https://www.cs.rochester.edu/research/speech/nonwords.html>
- Switchboard transcription guidelines: <https://isip.piconepress.com/projects/switchboard/doc/transcription_guidelines/transcription_guidelines.pdf>
