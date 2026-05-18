# JAVTrans 工程计划

> **执行前必读**：重新读取本文件、`.env` 和涉及源文件。Python 统一使用 `.venv/bin/python`，pip 统一使用 `.venv/bin/pip`。项目运行临时/cache 文件放 `./temp/`；需要归档删除时移动到 `agents/rm/`。

**工作目录**：项目根目录

---

## 1. 当前架构与运行约定

### 1.1 主流程

- Windows 本机生产目标：NVIDIA RTX 4060 Ti 8GB，串行分时加载模型，阶段结束后卸载并清 CUDA cache。
- 主流水线：视频 -> 音频准备 -> WhisperSeg VAD/chunk packing -> ASR -> Forced Alignment 词级时间轴 -> F0 性别检测 -> F0 后 gender turn 重切段 -> 翻译前 ASR 噪声过滤 -> LLM 翻译 -> SRT/quality report。
- 普通入口：`.venv/bin/python run_web.py`。旧 `src/main.py --input ...` CLI 已移除。
- 后端调试入口：测试、诊断脚本，或直接调用 `run_asr_alignment_f0()` / `run_translation_and_write()`。

### 1.2 ASR / VAD / 对齐

- Engine 默认 ASR：`ASR_BACKEND=anime-whisper`。
- Web 推荐默认 ASR：`whisper-ja-anime-v0.3`，`/api/config` 同时暴露 `engine_defaults.asr_backend` 与 `recommended_asr_backend`。
- 支持 ASR 后端：`anime-whisper`、`qwen3-asr-1.7b`、`whisper-ja-1.5b`、`whisper-ja-anime-v0.3`。
- 默认 VAD：`ASR_VAD_BACKEND=whisperseg`，`WHISPERSEG_THRESHOLD=0.35`。
- `ASR_LONG_CHUNK_PROFILE=on` 时强制开启 VAD chunk packing 与 post-alignment F0：`ASR_CHUNK_PACKING_ENABLED=1`、`F0_GENDER_POST_ALIGNMENT=1`。
- Whisper generation budget 由共享层按 `max_target_positions`、forced decoder ids、prompt ids 和 `WHISPER_MAX_NEW_TOKENS` 动态裁剪；Qwen 不套 Whisper 448 decoder 窗口。
- 默认 ASR 精度策略：`ASR_PRECISION_MODE=strict`。低置信、疑似重复幻觉、上下文泄漏、乱码和生成异常的文本在 alignment 前直接丢弃，进入 quality report 审计，不进入 F0、翻译和最终字幕。
- ASR recovery、temperature fallback、prompt overflow retry 已移除；文本生成失败或不确定时不做“补救式重写”。timestamp/alignment fallback 只允许补时间轴，不允许改写或新增 ASR 文本。
- ASR checkpoint / `aligned_segments.json` cache 均校验结构化 signature；ASR context、语言、生成参数、VAD/chunk/F0/timeline 关键输入变化时不得误复用旧 cache。
- VAD/chunk cache 单独缓存 VAD 边界与 chunk packing 结果，不缓存 chunk wav；signature 覆盖 audio fingerprint、VAD 参数和 chunk/drop/merge 参数，不包含 ASR prompt/token/generation 参数。

### 1.3 F0 / 字幕策略

- 字幕约束：`MAX_SUBTITLE_DURATION=8.0`，`ASR_MERGE_HARD_MAX_DURATION=9.0`。
- 相邻短块合并受标点、speaker guard 和 gender guard 限制。
- `SubtitleOptions` 是字幕策略的任务级配置入口；timeline、reading、gap、merge、权重等参数不得依赖 import-time 全局常量。
- F0 None carry-over 默认开启：`F0_GENDER_NONE_TOLERANCE=3`，`F0_GENDER_CARRYOVER_MAX_GAP_S=15.0`，`F0_GENDER_CARRYOVER_MAX_SEGMENT_S=12.0`。
- `gender=None` 且时长超过软切分阈值的长段必须能被 hard word split 拆开，避免 None 长字幕穿透。

### 1.4 翻译策略

- 默认翻译配置为 OpenAI-compatible LLM 服务；翻译请求使用流式输出 + 结构化 JSON 输出。
- Web 任务默认 `translation_batch_size=200`、`translation_max_workers=4`。
- 翻译批处理采用 fixed full-JSON prefix + `requested_ids` 策略：全片字幕 JSON 作为稳定前缀，本地计算每个 batch 的全局 id 区间，LLM 只翻译指定 id。
- 前缀预热默认开启；超过 `TRANSLATION_FULL_JSON_PREFIX_MAX_CHARS` 时回退全片摘要上下文。
- 翻译进度日志包含并发诊断事件：`batch_start`、`batch_first_token`、`batch_finish`，记录 wall-clock ts、worker thread、requested ids、耗时、cache hit/miss token。
- 翻译 reasoning effort 只保留两档：`medium` / `xhigh`。Chat Completions、标准 Responses、Micu+Grok Responses patch 均直接透传，不把 `xhigh` 映射为 `high`。
- Micu/Grok Responses streaming read timeout 使用 `TRANSLATION_STREAM_READ_TIMEOUT_S`，必须有限且可配置，保证取消/backoff 可中断。
- 当前翻译 prompt version：`v2.5`。

### 1.5 文本与质量规则

- 翻译前 ASR 噪声过滤本地剔除空白/引号类噪声、纯英文幻觉 token、纯特殊符号段；含日文/CJK/字母或数字的短语义段保留。
- 翻译风格：性器官优先统一为“肉棒”“小穴”，不固定“菊花”。
- 人名默认按日语读音罗马音化；ASR 同音纠错必须保守，不能把不同汉字姓氏或不同读音称呼强行合并。
- 翻译后默认执行轻量 repair pass：代码侧选择高风险 id，repair prompt 只使用抽象原因类别和相邻上下文，不把片内错例硬编码进静态 prompt。
- quality report 需要暴露 ASR generation error、overflow、timeout、quarantine、empty speech text、strict precision dropped uncertain items 等风险信号。

---

## 2. 配置边界

### 2.1 `.env` 只保留跨任务持久配置

- `API_KEY`
- `OPENAI_COMPATIBILITY_BASE_URL`
- `LLM_MODEL_NAME`
- `LLM_API_FORMAT`
- `LLM_REASONING_EFFORT`
- `TARGET_LANG`
- `HF_ENDPOINT`
- `TRANSLATION_GLOSSARY`
- `ASR_CONTEXT`

视频路径、输出目录、ASR 后端、字幕模式、batch/worker、是否保留临时文件等任务级参数由 `JobSpec -> JobContext` 显式传入后端，不再依赖全局 `.env` 热覆盖。

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
| T-AM | strict precision ASR 默认化并删除 ASR recovery / temperature fallback / prompt overflow retry | `365 passed, 5 skipped` |

### T-AL 关键验证记录

- ONNX CUDA smoke 通过：WhisperSeg `model.onnx` 可创建 `CUDAExecutionProvider` session，provider 为 `['CUDAExecutionProvider', 'CPUExecutionProvider']`。
- SORA-575 复测：ASR+Alignment 266.00s，输出 578 条字幕。
- 对比 T-AK：总耗时 729.36s -> 649.40s；ASR+Alignment 430.61s -> 266.00s。
- 逐句字幕对比报告：`reports/SORA-575.subtitle_compare.html`。
- TorchCodec/libavutil 噪声已修复：timestamp fallback 音频读取改用 `soundfile` 路径 `load_audio_16k_mono()`。
- TEN VAD 增加 `libc++.so.1` 预检，缺失时返回简短 `vad_error`，避免析构异常刷屏。
- VAD/chunk cache smoke：修改 ASR prompt 上限后 aligned cache miss、ASR 重跑，但 VAD chunk cache hit；静音分析与切块 2.34s -> 0.01s。
- VAD/chunk cache 日志：`agents/temp/tal-vad-cache-smoke-v3.run.log`；汇总：`agents/temp/tal-vad-cache-smoke/summary.json`。

### T-AM 关键验证记录

- strict precision ASR 成为默认策略：`ASR_PRECISION_MODE=strict`，可疑/低置信文本在 alignment 前清空并写入 quality report。
- 后端已删除 ASR recovery、temperature fallback、prompt overflow retry；生成失败或不确定时不再重写补救。timestamp/alignment fallback 仅用于时间轴，不新增 ASR 文本。
- 后端 `JobSpec` / `JobContext` / `/api/config` 不再暴露 `asr_recovery`；旧前端字段即使提交也由后端忽略。
- 验证：compileall 通过；strict/QC/cache/ASR 定向 `66 passed`；全量 `.venv/bin/python -m pytest -q` 为 `365 passed, 5 skipped`。

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
