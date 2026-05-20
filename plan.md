# JAVTrans 工程计划

> **执行前必读**：重新读取本文件、`.env` 和涉及源文件。Python 统一使用 `.venv/bin/python`，pip 统一使用 `.venv/bin/pip`。项目运行临时/cache 文件放 `./temp/`；需要归档删除时移动到 `agents/rm/`。

**工作目录**：项目根目录

---

## 1. 当前架构与运行约定

### 1.1 主流程

- Windows 本机生产目标：NVIDIA RTX 4060 Ti 8GB，串行分时加载模型，阶段结束后卸载并清 CUDA cache。
- 主流水线：视频 -> 音频准备 -> WhisperSeg VAD/chunk packing -> ASR -> Forced Alignment 词级时间轴 -> F0 性别检测 -> F0 后 gender turn 重切段 -> 翻译前 ASR 噪声过滤 -> 翻译前 cue plan 时间轴归一化 -> LLM 逐 cue 翻译 -> SRT/quality report。
- 普通入口：`.venv/bin/python run_web.py`。旧 `src/main.py --input ...` CLI 已移除。
- 后端调试入口：测试、诊断脚本，或直接调用 `run_asr_alignment_f0()` / `run_translation_and_write()`。

### 1.2 ASR / VAD / 对齐

- Engine 默认 ASR：`ASR_BACKEND=whisper-ja-anime-v0.3`。
- Web 推荐默认 ASR：`whisper-ja-anime-v0.3`，`/api/config` 同时暴露 `engine_defaults.asr_backend` 与 `recommended_asr_backend`。
- 支持 ASR 后端：`anime-whisper`、`qwen3-asr-1.7b`、`whisper-ja-1.5b`、`whisper-ja-anime-v0.3`。
- 默认 VAD：`ASR_VAD_BACKEND=whisperseg-adaptive`，`ASR_VAD_ADAPTIVE=1`，`WHISPERSEG_THRESHOLD=0.35`。当前保留的用户可选 VAD 路线是 `whisperseg-adaptive` 和实验 `fusion_lite` / `fusion_lite_boost` / `fusion_lite_sigmoid`；旧 `whisperseg` 名称不再作为公开兼容别名，但可作为 fusion 内部 primary；Silero 只作为 fusion-lite 系列内部 speech prior，不作为独立主 VAD 暴露；ffmpeg silencedetect VAD fallback 已移除。
- `fusion_lite` 受 FusionVAD 特征融合思想启发，但不引入 pyannote 或训练流程；公式为 `speech_score = 0.45 * whisperseg_score + 0.25 * silero_overlap_ratio + 0.15 * rms_score + 0.10 * spectral_flux_score + 0.05 * duration_score`，仅当 `speech_score < 0.45` 且 `silero_overlap_ratio < 0.05` 时丢弃候选。权重理由：WhisperSeg 作为候选主信号占最大权重，Silero 只提供 speech prior 而不一票否决，RMS/spectral flux/duration 补充传统声学特征。
- 主 VAD 初始化或推理失败时直接抛错并进入 Web 日志；主 VAD 返回空结果时直接跳过 ASR，不再整段音频 fallback，也不先转写再丢弃。
- WhisperSeg 空结果是合法“无语音”结果，不能因空 groups 统计除零而升级为 VAD 异常；旧 chunking helper 也不得再把空 VAD 回退成整段音频。
- `ASR_LONG_CHUNK_PROFILE=on` 时强制开启 VAD chunk packing 与 post-alignment F0：`ASR_CHUNK_PACKING_ENABLED=1`、`F0_GENDER_POST_ALIGNMENT=1`。
- Whisper generation budget 由共享层按 `max_target_positions`、forced decoder ids、prompt ids 和 `WHISPER_MAX_NEW_TOKENS` 动态裁剪；Qwen 不套 Whisper 448 decoder 窗口。
- ASR 精度策略固定为 adaptive precision，不再提供 `ASR_PRECISION_MODE` 模式开关。高 `no_speech_prob`、高压缩率、异常字符密度、重复循环、上下文泄漏、乱码和生成异常在 alignment 前硬丢弃；低风险真实对白可按自适应 `avg_logprob` 阈值保留，所有判定进入 quality report 审计。
- ASR recovery、temperature fallback、prompt overflow retry 已移除；文本生成失败或不确定时不做“补救式重写”。timestamp/alignment fallback 只允许补时间轴，不允许改写或新增 ASR 文本。
- ASR checkpoint / `aligned_segments.json` cache 均校验结构化 signature；ASR context、语言、生成参数、VAD/chunk/F0/timeline 关键输入变化时不得误复用旧 cache。
- VAD/chunk cache 单独缓存 VAD 边界与 chunk packing 结果，不缓存 chunk wav；signature 覆盖 audio fingerprint、VAD 参数和 chunk/drop/merge 参数，不包含 ASR prompt/token/generation 参数。

### 1.3 F0 / 字幕策略

- 字幕约束：`MAX_SUBTITLE_DURATION=6.5`，`SUBTITLE_SOFT_MAX_S=5.5`，`ASR_MERGE_HARD_MAX_DURATION=9.0`。7s 参考 Netflix Timed Text 单条字幕行业上限；BBC/眼动研究支持根据阅读速度弹性处理，因此 5.5s 作为软拆分目标，6.5s 作为保守硬上限，避免短文本长时间挂屏。
- LLM 翻译前必须先生成稳定 cue plan：`ffprobe` 读取真实 `avg_frame_rate`/`r_frame_rate`，失败时按 `30000/1001`（29.97fps）兜底；基于 forced alignment 词级时间轴完成排序、双语短句合并、软拆、overlap 裁剪/合并，字幕间隔固定为 2 帧，禁止保留 overlap。LLM 只翻译该 cue plan；SRT writer 只负责换行和格式化，不得再改变 cue 时间轴或 cue 数量。规范化后的同一份 blocks 必须写入 SRT、`bilingual.json` 和 quality report。
- 相邻短块合并受标点、speaker guard 和 gender guard 限制；短 gap / 短尾判断使用视频 fps 换算帧数而不是硬编码秒数。普通短块合并默认允许 `gap <= 6 frames` 且合并后 `duration <= 120 frames`；若 F0 gender guard 冲突，仅当后一 cue 是 `<= 20.5 frames` 的极短尾巴、gap `<= 2.5 frames`、且日文边界存在文本重叠时才允许合并并把合并后 `gender` 置为 `None`。speaker guard 仍为硬边界。
- `SubtitleOptions` 是字幕策略的任务级配置入口；timeline、reading、gap、merge、权重等参数不得依赖 import-time 全局常量。
- F0 None carry-over 默认开启：`F0_GENDER_NONE_TOLERANCE=3`，`F0_GENDER_CARRYOVER_MAX_GAP_S=15.0`，`F0_GENDER_CARRYOVER_MAX_SEGMENT_S=12.0`。
- `gender=None` 且时长超过软切分阈值的长段必须能被 hard word split 拆开，避免 None 长字幕穿透。

### 1.4 翻译策略

- 默认翻译配置为 OpenAI-compatible LLM 服务；翻译请求使用流式输出 + 结构化 JSON 输出。
- Web 任务默认 `translation_batch_size=200`、`translation_max_workers=4`。
- 翻译批处理采用 fixed full-JSON prefix + `requested_ids` 策略：全片 cue plan JSON 作为稳定前缀，本地计算每个 batch 的全局 cue id 区间，LLM 只翻译指定 id。
- 前缀预热默认开启；超过 `TRANSLATION_FULL_JSON_PREFIX_MAX_CHARS` 时回退全片摘要上下文。
- Grok 搜索结论支持保留 full/multi-line context、术语表、结构化 JSON 输出和窄范围 post-edit；当前默认继续保留 prefix warmup、全片 glossary 预抽取和 translation repair pass，但 repair 只处理译文长度异常，不扩展成 ASR/剧情修复。
- 翻译进度日志包含并发诊断事件：`batch_start`、`batch_first_token`、`batch_finish`，记录 wall-clock ts、worker thread、requested ids、耗时、cache hit/miss token。
- 翻译 reasoning effort 只保留两档：`medium` / `xhigh`。Chat Completions、标准 Responses、Micu+Grok Responses patch 均直接透传，不把 `xhigh` 映射为 `high`。
- Micu/Grok Responses streaming read timeout 使用 `TRANSLATION_STREAM_READ_TIMEOUT_S`，必须有限且可配置，保证取消/backoff 可中断。
- 当前翻译 prompt version：`v2.6`。

### 1.5 文本与质量规则

- 翻译前 ASR 噪声过滤本地剔除空白/引号类噪声、纯英文幻觉 token、纯特殊符号段；含日文/CJK/字母或数字的短语义段保留。
- 翻译风格：性器官优先统一为“肉棒”“小穴”，不固定“菊花”。
- 人名默认按日语读音罗马音化；人物参考只用于识别字幕文本中已经明确出现的人名，LLM 不得根据参考名推测、补全或替换源文。
- LLM 不再承担 ASR 误听、同音词、上下文漂移、术语漂移或被切断半句修复；没有画面信息时这些推断容易改错源文。翻译后 repair pass 仅保留译文长度异常这类纯译文质量候选。
- quality report 需要暴露 ASR generation error、overflow、timeout、quarantine、empty speech text、adaptive precision dropped uncertain items、alignment fallback count/ratio 等风险信号；默认写入 `reports/`，主产物为 Markdown，同时保留 JSON sidecar 供测试脚本读取。

---

## 2. 配置边界

### 2.1 `.env` 只保留跨任务持久配置和默认偏好

- `API_KEY`
- `OPENAI_COMPATIBILITY_BASE_URL`
- `LLM_MODEL_NAME`
- `LLM_API_FORMAT`
- `LLM_REASONING_EFFORT`
- `TARGET_LANG`
- `HF_ENDPOINT`
- `TRANSLATION_GLOSSARY`
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

`.env` 按翻译服务、翻译偏好、模型下载、ASR 默认后端、VAD/chunk/cache、adaptive ASR QC、F0/gender 和质量报告分组注释。旧 `ASR_PRECISION_MODE`、`ASR_DROP_UNCERTAIN_ENABLED`、`ASR_QC_STRICT_*` 不再使用。

视频路径、输出目录、字幕模式、batch/worker、是否保留临时文件等任务级参数由 `JobSpec -> JobContext` 显式传入后端，不再依赖全局 `.env` 热覆盖；Web 表单可按任务覆盖 `.env` 中的默认 ASR 偏好。

### 2.2 路径与缓存

- `HF_HOME` 默认 `./models`；首次运行把 HuggingFace repo 下载到 `models/<namespace>-<repo>/`。
- `HF_HUB_CACHE`、`HF_XET_CACHE`、`TORCH_HOME` 默认在 `./temp/` 下。
- Web 任务运行日志：高级项启用 `RUN_LOG_ENABLED=1` 后默认写入 `temp/log/`。
- 成功运行后默认删除一次性 job 临时目录；保留下次可复用的运行缓存，例如 `temp/hf-cache`、`temp/vad-cache`、`temp/web` 状态和 `models/`。
- Web“保留临时文件”仅用于调试，保留当前任务临时目录；不再通过全局 `KEEP_TEMP_FILES` 控制任务行为。
- 所有项目配置、README、agent 本地说明使用项目相对路径，不写本机绝对路径。

### 2.3 Web 设置行为

- Web 演员名/人名提示（`ASR_CONTEXT`）是持久设置：打开页面时从 `/api/settings` 恢复，提交任务时自动保存。
- 用户手动清空后提交会清空持久值。
- 前端不提供单独“保存设置”按钮，提交任务即保存当前表单配置。
- `OPENAI_COMPATIBILITY_BASE_URL` 是 OpenAI-compatible API 配置名，保留不改。

---

## 3. 当前 Backlog

| 优先级 | 项目 | 验收标准 |
|--------|------|----------|
| P2 | Windows 生产环境 default-on 验证 | RTX 4060 Ti 8GB 下确认 CUDA/ONNXRuntime provider、模型串行加载、cache 命中/失效、输出目录和临时目录清理行为；至少完成一个代表视频的 Web 全流程 smoke。 |

---

## 4. 最近完成基线

| Task | 内容 | 验收 |
|------|------|------|
| T-AJ | 全量审计修复：任务级 env 覆盖、aligned cache scope、ASR/字幕/quality 参数运行时化、翻译 cancel_event 透传 | 基线 315 passed, 5 skipped；完成后逐步增至 334+ passed |
| T-AK | 第二轮后端审计：ASR/aligned cache signature、`.env.example` 默认、SubtitleOptions、Web retry/cancel、stream timeout、Protocol 补齐 | `343 passed, 5 skipped` |
| T-AL | ASR generation budget + ONNX CUDA runtime + VAD/chunk cache | `359 passed, 5 skipped`；SORA-575 anime-whisper 全量中日双语 649.54s，WhisperSeg CUDA VAD/切块 9.32s，ASR generation overflow/error 为 0 |
| T-AM | 删除 ASR recovery / temperature fallback / prompt overflow retry，并清理前端旧 ASR Recovery 控件；早期固定阈值 precision 方案后续被 adaptive-only 替换 | 后端全量 `365 passed, 5 skipped`；前端/Web 定向 `13 passed` |
| T-AN | adaptive precision ASR 默认化：保留硬幻觉拒绝，低风险低 `avg_logprob` 对白自适应放宽 | 定向回归 `68 passed`；Oni Chichi BDRIP 5min smoke：adaptive drops 2，overflow/error/timeout/quarantine 为 0 |
| T-AO | 默认 ASR 切为 `whisper-ja-anime-v0.3`；新增 `video/test` 通用测试集评测工具；删除 strict/normal ASR 精度模式，只保留 adaptive precision | 全量 `.venv/bin/python -m pytest -q` 为 `373 passed, 5 skipped` |
| T-AP | 本地 `.env` 适配当前默认流程并按同类参数归类注释；README/plan 同步 `.env` 边界 | dotenv 解析通过；关键 adaptive/default ASR 配置齐全；旧 strict 配置不存在 |
| T-AQ | 新增 Silero / hybrid VAD 实验并完成取舍：hard/soft gate 过度依赖 Silero，后续从当前代码与公开配置中移除 | NAMH-055 前 5 分钟历史 smoke：hybrid hard 漏太多，hybrid soft 改善但仍不作为保留路线 |
| T-AR | 新增 `fusion_lite` VAD 实验后端，只保留 `whisperseg-adaptive` 与 `fusion_lite` 两条公开路线；字幕默认软目标/硬上限收紧为 5.5s/6.5s | NAMH-055 前 5 分钟：WhisperSeg 14 字幕/11 drops；fusion_lite 15 字幕/7 drops；HAME-052/NMSL-036 全片 VAD 对比 generation overflow/error 均为 0；字幕定向 23 passed，全量 383 passed, 5 skipped |
| T-AS | 全量审计修复：WhisperSeg 空结果除零、旧 chunking 整段 fallback、timestamp fallback 参数运行时化、alignment fallback 统计、字幕 writer/Web/pipeline 过时路径清理 | 定向回归通过，后续以全量 pytest 基线更新 |
| T-AT | Fusion-lite 后缀实验 + SORA-575 对比 + 帧率驱动 SRT overlap 归一化 | SORA-575 四模式对比完成；`fusion_lite_boost` 最接近 whisperseg-adaptive；新增逐句 HTML 报告；字幕/fps/主流程定向 `73 passed` |
| T-AU | HAME-052 四模式双语对比 + frame-based 短尾 cue 合并 | HAME-052 四模式全流程双语输出完成；新增 frame-based overlapping tail merge，修复 `受け` / `受けて` 这类极短 gap 被 F0 gender 抖动切成两条的问题；字幕定向 `50 passed` |

### T-AL 关键验证记录

- ONNX CUDA smoke 通过：WhisperSeg `model.onnx` 可创建 `CUDAExecutionProvider` session，provider 为 `['CUDAExecutionProvider', 'CPUExecutionProvider']`。
- SORA-575 复测：ASR+Alignment 266.00s，输出 578 条字幕。
- 对比 T-AK：总耗时 729.36s -> 649.40s；ASR+Alignment 430.61s -> 266.00s。
- 逐句字幕对比报告：`reports/SORA-575.subtitle_compare.html`。
- TorchCodec/libavutil 噪声已修复：timestamp fallback 音频读取改用 `soundfile` 路径 `load_audio_16k_mono()`。
- TEN VAD 增加 Linux `libc++.so.1` 预检，缺失时返回简短 `vad_error`，避免析构异常刷屏；Windows 下不做该 Linux 依赖检查。
- VAD/chunk cache smoke：修改 ASR prompt 上限后 aligned cache miss、ASR 重跑，但 VAD chunk cache hit；静音分析与切块 2.34s -> 0.01s。
- VAD/chunk cache 日志：`agents/temp/tal-vad-cache-smoke-v3.run.log`；汇总：`agents/temp/tal-vad-cache-smoke/summary.json`。

### T-AM 关键验证记录

- 早期固定阈值 precision 方案曾作为默认策略；后续 T-AN/T-AO 已替换为 adaptive-only。
- 后端已删除 ASR recovery、temperature fallback、prompt overflow retry；生成失败或不确定时不再重写补救。timestamp/alignment fallback 仅用于时间轴，不新增 ASR 文本。
- 后端 `JobSpec` / `JobContext` / `/api/config` 不再暴露 `asr_recovery`；前端已移除 `ASR Recovery` 开关、preset 字段、配置回填和提交 payload。
- 验证：compileall 通过；precision/QC/cache/ASR 定向 `66 passed`；全量 `.venv/bin/python -m pytest -q` 为 `365 passed, 5 skipped`。
- 前端验证：`node --check` 覆盖 `settings.js` / `files.js` / `presets.js` / `main.js`；Web/API 与 ASR env 定向 `13 passed`。

### T-AN 关键验证记录

- 默认 ASR 精度策略更新为 adaptive precision：硬拒绝高 `no_speech_prob`、高压缩率、异常字符密度、重复循环、上下文泄漏、乱码和生成异常；仅对低风险真实对白放宽低 `avg_logprob`。
- adaptive 阈值写入 ASR checkpoint / aligned cache signature，`ASR_QC_ADAPTIVE_*` 变化会触发重算。
- Oni Chichi BDRIP 旧固定阈值 drops 离线重判：24 条 -> adaptive reject 8 条，预计恢复 16 条；579 字/7.84s 重复幻觉仍由 `abnormal_char_density` 硬拒绝。
- 5 分钟 BDRIP smoke：`whisper-ja-anime-v0.3`，ASR+Alignment 16.96s，输出 82 段，`asr_dropped_uncertain_count=2`，generation overflow/error/timeout/quarantine 均为 0。
- 验证：compileall 通过；QC/cache/ASR/testset 定向 `68 passed`；`git diff --check` 通过。

### T-AO 关键验证记录

- Engine 默认 ASR 改为 `whisper-ja-anime-v0.3`，与 Web 推荐默认一致；`src/core/config.py`、`src/core/job_context.py`、`src/whisper/backends/registry.py`、README 和 Web/API 测试已同步。
- 新增通用测试集评测工具：`tests/testset_quality_eval.py`，支持 `video/test/index.json` 中任意视频 + `.ass`/`.srt` 参考字幕，输出转写覆盖、无参考重叠、翻译 CER/F1 和 ASR drop 代理指标。
- `video/test` 测试集已去重为一集一份参考字幕，优先保留 BDRIP；重复和无配套字幕文件移动到 `agents/rm/`。
- 删除 ASR 精度模式开关：不再读取 `ASR_PRECISION_MODE`，不再保留 strict/normal 分支；`ASR_QC_STRICT_*` 和旧通用 `ASR_QC_*_THRESHOLD` 配置已从默认配置、`.env.example`、checkpoint signature 和测试中移除。
- 当前唯一 ASR 丢弃策略为 adaptive precision：`ASR_QC_ADAPTIVE_*` 控制硬拒绝和自适应 `avg_logprob` 阈值；丢弃项写入 `asr_dropped_uncertain_items`，pipeline 计时使用 `asr_adaptive_dropped_chunks`。
- 验证：定向回归 `65 passed`；compileall 通过；`git diff --check` 通过；全量 `.venv/bin/python -m pytest -q` 为 `373 passed, 5 skipped`。

### T-AP 关键验证记录

- `.env` 已按翻译服务、翻译偏好、模型下载、ASR 默认后端、VAD/chunk/cache、adaptive ASR QC、F0/gender 和质量报告分组并补充注释。
- `.env` 保留本地真实 API/模型/术语表/ASR_CONTEXT 等值，同时显式补齐 `ASR_BACKEND=whisper-ja-anime-v0.3`、WhisperSeg、VAD chunk cache 和 `ASR_QC_ADAPTIVE_*` 默认项。
- `.env` 不再包含 `ASR_PRECISION_MODE`、`ASR_DROP_UNCERTAIN_ENABLED`、`ASR_QC_STRICT_*` 等过时配置。
- 验证：`dotenv_values(".env")` 解析通过；关键配置齐全；旧 strict 配置不存在。

### T-AQ / T-AR 关键验证记录

- Silero / `hybrid_precision` 曾作为低幻觉 VAD 方案验证；hard gate 漏掉大量真实对白，soft gate 有改善但仍过度依赖 Silero，因此当前代码和公开配置不再暴露 `silero` / `hybrid_precision` 主 VAD 模式。
- 当前保留 VAD 路线：默认 `whisperseg-adaptive`，以及实验 `fusion_lite`；`whisperseg` 不再作为公开兼容别名。ffmpeg silencedetect VAD fallback 已移除；WhisperSeg/Silero 初始化或推理失败直接报错，主 VAD 空结果直接跳过 ASR。
- `fusion_lite` 受 FusionVAD 论文的“MFCC/手工声学特征 + PTM 特征简单融合”思想启发，但本项目不引入 pyannote、不训练模型；用可解释公式融合 WhisperSeg 分数、Silero 重叠、RMS、spectral flux 和时长分数。
- `fusion_lite` 公式：`speech_score = 0.45 * whisperseg_score + 0.25 * silero_overlap_ratio + 0.15 * rms_score + 0.10 * spectral_flux_score + 0.05 * duration_score`；丢弃条件：`speech_score < 0.45 and silero_overlap_ratio < 0.05`。这样 Silero 只提供辅助证据，不再一票否决 WhisperSeg 高置信候选。
- `SILERO_VAD_*`、`FUSION_VAD_*`、`ASR_VAD_PRIMARY`、`ASR_VAD_GATE` 已纳入 ASR stage advanced env 和 VAD/chunk cache signature；Silero 的 `allow_empty` 语义保留给 fusion gate，表示 gate 可以给出空 speech prior，但推理失败会直接报错。
- NAMH-055 前 5 分钟：WhisperSeg 48 VAD segments / 129.22s speech / 14 字幕 / 11 drops；fusion_lite 23 VAD segments / 89.34s speech / 15 字幕 / 7 drops；generation overflow/error/timeout/quarantine 均为 0。
- HAME-052 / NMSL-036 全片三模式历史对比显示 fusion_lite 输出接近 WhisperSeg，但 drops 少于 WhisperSeg；逐句报告：`reports/HAME-052_NMSL-036.full_vad_modes_line_compare.html`。
- 字幕时长策略更新：Grok 检索 Netflix Timed Text、BBC 字幕指南和眼动研究后，将默认 `MAX_SUBTITLE_DURATION` 从 8.0 先收紧到 7.0，随后按观看体验要求进一步收紧到 6.5；`SUBTITLE_SOFT_MAX_S` 从 6.0 收紧到 5.5；长字幕后续应继续做词时间轴驱动的强制重分段。
- 证据：`agents/temp/t-ar-fusion-lite-namh055/summary.json`、`agents/temp/t-ar-fusion-lite-namh055-asr/summary.json`、`agents/temp/t-as-full-vad-compare/summary.json`、`reports/NAMH-055.5min.vad_modes_fusion_lite_line_compare.html`。

### T-AT 关键验证记录

- 按“简单配置、方便删除实验模式”的原则新增 `fusion_lite_boost` 和 `fusion_lite_sigmoid` 后缀后端，不新增 `FUSION_VAD_SCORING_MODE` 这类模式参数；`fusion_lite` 保留为线性基线。
- 修复 fusion 内部 `ASR_VAD_PRIMARY=whisperseg` 加载路径：公开 registry 仍不接受裸 `whisperseg`，但 fusion primary 可直接加载 `WhisperSegVadBackend`。
- SORA-575 使用 `whisper-ja-anime-v0.3`、跳过翻译进行 VAD/ASR 对比。CUDA ONNX 在 sandbox 内失败，原因是 GPU 被操作系统/sandbox 阻断；外部执行确认 RTX 4060 Ti、Torch CUDA 和 ONNXRuntime CUDA provider 可用。
- SORA-575 汇总：`whisperseg-adaptive` 700 SRT / 210 ASR drops；`fusion_lite` 683 SRT / 167 drops；`fusion_lite_boost` 688 SRT / 185 drops；`fusion_lite_sigmoid` 677 SRT / 148 drops。逐句对齐以 `whisperseg-adaptive` 为基准，`fusion_lite_boost` 最接近：SAME 564、MISSING 41、平均相似度 0.9560。
- 逐句报告产物默认写入 `reports/`；历史 SORA-575 报告包含 `sora575_sentence_report.html`、`sora575_sentence_report.md`、`sora575_sentence_diffs_only.md`、`sora575_sentence_report.csv`、`sora575_sentence_report_summary.json`。
- SRT overlap 处理重构：新增 `probe_video_fps()`，优先读 `avg_frame_rate`，再读 `r_frame_rate`；读不到 fps 时按 `30000/1001`。`SubtitleOptions` 持有 `video_fps`，SRT writer 在写出前排序、软拆、合并/裁剪重叠，并强制保留 2 帧 gap。旧 `SUBTITLE_GAP_PADDING` 已移除，不再保留兼容配置。
- quality report 新增 `subtitle_overlap_count`、`subtitle_overlap_total_s`、`subtitle_overlap_max_s` 和前 5 个 overlap examples；正常写出后这些值应为 0。
- 验证：`.venv/bin/python -m pytest tests/test_subtitle_options.py tests/test_subtitle_quality_pass.py tests/test_subtitle_qc.py tests/test_srt_wrap.py tests/test_video_fps_probe.py tests/test_skip_translation.py tests/test_job_tempdir.py tests/test_aligned_segments_cache.py tests/web/test_cancel_resume.py -q` -> `73 passed`。

### T-AU 关键验证记录

- HAME-052 四模式完整双语对比已输出到 `video/HAME-052.*.srt`，逐句 HTML 报告为 `video/HAME-052.vad_bilingual_compare.html`；四模式 ASR generation error / overflow / timeout 均为 0，最终 SRT 未包含可见 `[M]` / `[F]` 性别标签。
- HAME-052 汇总：`whisperseg_adaptive` 230 SRT / 96 ASR drops；`fusion_lite` 231 SRT / 63 drops；`fusion_lite_boost` 230 SRT / 84 drops；`fusion_lite_sigmoid` 230 SRT / 50 drops。逐句报告以 `whisperseg_adaptive` 为基准，`fusion_lite` 平均相似度最高。
- cue plan 短尾合并改为 frame-based 规则：普通短块合并 `gap <= 6 frames`、合并后 `duration <= 120 frames`；跨 F0 gender guard 只允许日文边界存在文本重叠的极短尾巴，要求 `gap <= 2.5 frames` 且尾块 `<= 20.5 frames`，合并后 gender 置为 `None`。speaker guard 继续硬阻断。
- HAME-052 离线验证：`00:02:56,839 --> 00:02:59,160` 合并为 `アルマリスト 室で イラックス ステイマンを受けて`，避免 `受け` / `受けて` 被拆成两条造成语义断裂。
- 验证：`.venv/bin/python -m pytest tests/test_subtitle_quality_pass.py tests/test_subtitle_options.py tests/test_srt_wrap.py tests/test_subtitle_qc.py -q` -> `50 passed`。

---

## 5. 历史记录

本节只保留已完成任务的大致内容和关键验收，避免把已落地 Step 细节继续放在主计划里。

### 5.1 已完成任务摘要

| Task | 大致内容 | 验收 / 备注 |
|------|----------|-------------|
| T-A | ASR Recovery 接入 VAD 二次细分，改善异常 ASR 文本块的重跑路径。 | 历史功能，T-AM 已从后端移除 |
| T-B | 建立 F0 词级时间轴与 multi-cue gender 切分。 | 已完成 |
| T-C ~ T-E | Web 控制台、Stage 事件 JSON 化、重试断点续传和 cancel event 透传。 | 已完成 |
| T-F ~ T-G | HF 镜像开关、Web 配置项扩展。 | 已完成 |
| T-H ~ T-J | 后端稳定性、CLI 瘦身、全局 env 并发污染治理。 | T-J 后 179 passed |
| T-K | `transformers` 兼容性回滚，保留四个稳定 ASR 后端。 | 依赖固定回 `transformers==4.57.6` |
| T-L ~ T-M | GitHub 发布前文档/配置/入口收口；翻译上下文和 cache key 收口。 | T-M 后定向 32 passed |
| T-N | Windows Release exe 打包配置。 | 已完成 |
| T-O | Web 表单记忆与右键粘贴体验。 | 已完成 |
| T-P | OpenAI Responses 翻译格式兼容。 | 已完成 |
| T-Q | F0 后 gender turn 字幕重切段。 | F0 定向 15 passed；ASR job/cache 定向 7 passed |
| T-R | 翻译重试与请求清理；Micu+Grok Responses 特例移入 `src/llm/patch.py`；翻译前 ASR 噪声过滤扩展到纯英文幻觉 token。 | 已完成 |
| T-S | 后端稳定性收口：`JobSpec` 边界、finished job 删除锁顺序、run logger 泄漏、translation cache 损坏容忍。 | 58 passed |
| T-U | 后端大文件拆分：`src/main.py` helper 迁入 `src/pipeline/` 多个子模块。 | 宽后端回归 93 passed |
| T-V | 前端 `app.js` 拆分为 ES Module，并修复日志刷新导致粘贴菜单关闭的问题。 | 语法检查和手动验证通过 |
| T-W | 翻译 reasoning effort 收口为 `medium` / `xhigh`，Responses 不做兼容降级映射。 | 定向 36 passed |
| T-X | 翻译 fixed-prefix 批处理、并发诊断、术语/人名规则、局部 repair pass。 | 229 passed |
| T-Y | Web 演员名持久化、提交自动保存设置、移除手动保存按钮。 | Web 定向 10 passed + JS check |
| T-Z | 翻译前 ASR 噪声过滤扩展到纯特殊符号段。 | 定向 56 passed |
| B1 | 拆分 `src/whisper/pipeline.py`：后端 registry、checkpoint 等职责外移。 | 241 passed |
| B2 | 拆分 `src/llm/translator.py`：translation cache 和 prompt 构建外移。 | 241 passed |
| B3 | 压缩 `src/main.py`：stage log、output writer 等职责外移。 | 241 passed |
| B4 | ASR 滑动上下文注入：`initial_prompts`、gender/gap 重置。 | 241 passed |
| B5 | VAD 微短段预合并：短 speech chunk 物理拼接并保留 `merged_from` 元数据。 | 241 passed |
| B6 | 字幕软切分点：长段优先按中文标点/日文助词词边界拆分。 | 241 passed |
| B7 | Repair Pass 增强：长度错配强制纳入 repair 候选。 | 237 passed |
| T-AA | ASR 质量信号：`avg_logprob`、`no_speech_prob`、`compression_ratio`；历史 temperature fallback 已在 T-AM 移除。 | 277 passed |
| T-AB | WhisperSeg 默认阈值 0.35；`SpeechSegment.score`；negative offset env；adaptive VAD。 | 299 passed |
| T-AC | VAD chunk packing + 词时间戳后置 F0 gender split。 | 253 passed |
| T-AD | T-AC 默认开启；ASR overflow initial prompt 双层截断。 | 256 passed |
| T-AE | None 段 gender carry-over。 | 262 passed |
| T-AF | soft split 扩展 None 长段，`gender=None` 且长段强制 hard word split。 | 302 passed |
| T-AG | 短段丢弃 gate：duration + RMS AND 双条件，env opt-in。 | 312 passed |
| T-AH | F0 carry-over 默认放宽；修复 `nan_ratio_threshold` 透传问题。 | 312 passed |
| T-AI | `F0_GENDER_NONE_TOLERANCE` 2 -> 3；post-split 第二次 carry-over pass。 | 315 passed |
| T-AJ | 全量审计修复：任务级 env 覆盖、aligned cache scope、运行时参数化、翻译 cancel_event 透传。 | 基线 315 passed, 5 skipped；完成后增至 334+ passed |
| T-AK | 第二轮后端审计：ASR/aligned cache signature、`.env.example` 默认、SubtitleOptions、Web retry/cancel、stream timeout、Protocol 补齐。 | 343 passed, 5 skipped |
| T-AL | ASR generation budget、ONNX CUDA runtime、VAD/chunk cache、ASR generation QC 暴露。 | 359 passed, 5 skipped |

### 5.2 历史回归里程碑

- T-S：58 passed。
- T-U：宽后端回归 93 passed。
- T-X：229 passed。
- T-Z：定向 56 passed。
- T-AC：253 passed。
- T-AI：315 passed。
- T-AK：343 passed, 5 skipped。
- T-AL：359 passed, 5 skipped。

---

## 6. 历史验证基线

### HAME-052 四后端 skip-translation 对比

| 后端 | 状态 | ASR 转写 | Wall time | 字幕数 |
|------|------|----------|-----------|--------|
| `anime-whisper` | ok | 48.52s | 336.88s | 150 |
| `qwen3-asr-1.7b` | ok | 251.71s | 578.15s | 164 |
| `whisper-ja-1.5b` | ok | 170.74s | 464.33s | 165 |
| `whisper-ja-anime-v0.3` | ok | 41.89s | 229.46s | 151 |

默认全量翻译（anime-whisper + bilingual）：`pipeline_total=575.30s`，字幕块数 150，产物 `video/HAME-052.srt`。

### SORA-575 历史基准

- T-AA 前后基线：491.5s，493 ASR chunks，字幕 365 段，F/M/None=117/124/124（34% None），Mixed=13。
- T-AL 当前基线：总耗时 649.54s；WhisperSeg CUDA VAD/切块 9.32s；ASR+Alignment 266.00s；输出 578 条字幕；ASR generation overflow/error 为 0。
