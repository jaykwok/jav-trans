# JAVTrans 工程计划

> **For agentic workers:** 执行前重新读取本文件、`.env` 和涉及源文件。Python 统一使用 `.venv/Scripts/python`，pip 统一使用 `.venv/Scripts/pip`。项目运行临时/cache 文件放 `./temp/`；需要归档删除时移动到 `agents/rm/`。普通回复不要使用 cc-connect relay；session 输出会自动回灌。

**工作目录：** 项目根目录

---

## 1. 当前锁定架构

- Windows 本机生产目标：NVIDIA RTX 4060 Ti 8GB，串行分时加载模型，阶段结束后卸载并清 CUDA cache。
- 主流水线：视频 → 音频准备 → WhisperSeg VAD → ASR → Forced Alignment 词级时间轴 → F0 性别检测 → F0 后 gender turn 重切段 → 翻译前 ASR 噪声过滤 → LLM 翻译 → SRT/quality report。
- 默认 ASR 配置仍为 `ASR_BACKEND=anime-whisper`；Web UI 推荐排序把 `whisper-ja-anime-v0.3` 放在第一位。
- 支持后端：`anime-whisper`、`qwen3-asr-1.7b`、`whisper-ja-1.5b`、`whisper-ja-anime-v0.3`。
- 默认 VAD：`ASR_VAD_BACKEND=whisperseg`，`WHISPERSEG_THRESHOLD=0.35`。
- 默认翻译配置示例为 OpenAI-compatible LLM 服务；翻译请求使用流式输出 + 结构化 JSON 输出，Web 任务默认 `translation_batch_size=200`，`translation_max_workers=4`。
- 翻译批处理采用 fixed full-JSON prefix + `requested_ids` 策略：全片字幕 JSON 作为稳定前缀，本地计算每个 batch 的全局 id 区间，LLM 只翻译指定 id；`batch_size` 表示每次翻译编号区间长度。前缀预热默认开启，超过 `TRANSLATION_FULL_JSON_PREFIX_MAX_CHARS` 时回退全片摘要上下文。
- 翻译进度日志包含并发诊断事件：`batch_start`、`batch_first_token`、`batch_finish`，记录 wall-clock ts、worker thread、requested ids、耗时、cache hit/miss token，便于判断 API 请求是否真实并行。
- 翻译前 ASR 噪声过滤在本地剔除空白/引号类噪声、纯英文幻觉 token、纯特殊符号段（如 `◆◆◆`、`♪♪♪`、`！？！？`）；含日文/CJK/字母或数字的短语义段（如 `えっ！？`、`もう1回`）保留。
- 翻译 reasoning effort 只保留两档：`medium` / `xhigh`。Chat Completions、标准 Responses、Micu+Grok Responses patch 均直接透传 `medium` 或 `xhigh`，不再把 `xhigh` 映射为 `high`。
- 翻译风格：性器官优先统一为“肉棒”“小穴”，不固定“菊花”；人名默认按日语读音罗马音化，ASR 同音纠错必须保守，不能把不同汉字姓氏或不同读音的称呼强行合并。
- 翻译后默认执行轻量 repair pass：代码侧选择高风险 id，repair prompt 只使用抽象原因类别和相邻上下文，不把片内错例硬编码进静态 prompt。
- 字幕约束：`MAX_SUBTITLE_DURATION=8.0`，`ASR_MERGE_HARD_MAX_DURATION=9.0`，相邻短块合并受标点、speaker guard 和 gender guard 限制。
- 默认 ASR recovery：`ASR_RECOVERY_ENABLED=0`。异常 ASR 文本块排查时才手动打开；男女混句边界由 F0 后 gender turn 重切段处理。
- 断点续传：ASR checkpoint、`aligned_segments.json`、translation cache、translation artifact snapshot。
- 主入口：`.venv/Scripts/python run_web.py`。后端调试入口是测试、诊断脚本，或直接调用 `run_asr_alignment_f0()` / `run_translation_and_write()`。
- Web 任务参数通过 `JobSpec -> JobContext` 显式传入后端，不再依赖全局 `.env` 热覆盖 ASR、字幕、输出目录、batch/worker、临时文件保留等任务级配置。
- `.env` 只保留跨任务持久配置：`API_KEY`、`OPENAI_COMPATIBILITY_BASE_URL`、`LLM_MODEL_NAME`、`LLM_API_FORMAT`、`LLM_REASONING_EFFORT`、`TARGET_LANG`、`HF_ENDPOINT`、`TRANSLATION_GLOSSARY`、`ASR_CONTEXT`。

---

## 2. 当前行为约定

- 普通使用入口是 `.venv/Scripts/python run_web.py`；旧 `src/main.py --input ...` CLI 已移除。
- 需要观察后端长跑进度或收集用户反馈日志时，通过 Web 高级项启用 `RUN_LOG_ENABLED=1`；默认写入 `temp/log/`。
- `HF_HOME` 默认 `./models`；首次运行把 HuggingFace repo 下载到 `models/<namespace>-<repo>/`。
- `HF_HUB_CACHE`、`HF_XET_CACHE`、`TORCH_HOME` 默认在 `./temp/` 下；ASR recovery 输出默认在 `temp/recovery`。
- `HF_ENDPOINT` 必须为空或完整 URL，例如 `https://hf-mirror.com`。
- 成功运行后默认直接删除一次性 job 临时目录；保留下次可复用的运行缓存，例如 `temp/hf-cache`、`temp/web` 状态和 `models/`。
- Web“保留临时文件”选项仅用于调试，保留当前任务临时目录；不再通过全局 `KEEP_TEMP_FILES` 控制任务行为。
- Web 演员名 / 人名提示（`ASR_CONTEXT`）是持久设置：打开页面时从 `/api/settings` 恢复，提交任务时自动保存；用户手动清空后提交会清空持久值。前端不再提供单独“保存设置”按钮，提交任务即保存当前表单配置。
- 所有项目配置、README、agent 本地说明应使用项目相对路径，不写本机绝对路径。
- `OPENAI_COMPATIBILITY_BASE_URL` 是 OpenAI-compatible API 配置名，保留不改。
- 翻译 cache key 由 prompt version、目标语言、术语表、人物参考、模型名和 batch source 共同决定；当前 prompt version 为 `v2.5`，用于隔离 fixed-prefix、repair 和人名策略变更后的旧缓存。

---

## 3. 已完成任务摘要

| Task | 内容 | 结果 |
|------|------|------|
| T-S ✅ | 后端稳定性收口（JobSpec 边界、删除锁顺序、run logger 泄漏、cache 损坏容忍） | 58 passed |
| T-U ✅ | 后端大文件拆分：`src/main.py` helper 迁入 `src/pipeline/`（8 个子模块） | 宽回归 93 passed |
| T-V ✅ | 前端 `app.js` 拆分为 14 个 ES Module；修复日志刷新导致粘贴菜单被关闭的 bug | 语法全通，手动验证 |
| T-W ✅ | 翻译 reasoning effort 收口为 `medium` / `xhigh` 两档；Responses 不做兼容降级映射 | 36 passed |
| T-X ✅ | 翻译 fixed-prefix 批处理、并发诊断、术语/人名规则、局部 repair pass | 229 passed |
| T-Y ✅ | Web 演员名持久化、提交自动保存设置、移除手动保存按钮 | 定向 10 passed + JS check |
| T-Z ✅ | 翻译前 ASR 噪声过滤扩展到纯特殊符号段 | 定向 56 passed |
| B1 ✅ | 拆分 `src/whisper/pipeline.py` → `backends/registry.py` + `checkpoint.py` + 3 新模块 | 241 passed |
| B2 ✅ | 拆分 `src/llm/translator.py` → `cache.py` + `prompt.py` | 241 passed |
| B3 ✅ | 压缩 `src/main.py`（-228 行）→ `stage_log.py` + `output_writer.py` | 241 passed |
| B4 ✅ | ASR 滑动上下文注入（`initial_prompts`，gender/gap 重置） | 241 passed |
| B5 ✅ | VAD 微短段预合并（`_merge_short_vad_chunks`，`merged_from` 元数据） | 241 passed |
| B6 ✅ | 字幕软切分点（`soft_split_long_segments`，6s 阈值，标点/助词词边界） | 241 passed |
| B7 ✅ | Repair Pass 长度错配强制候选（ratio [0.25, 4.0]，reason `length_mismatch`） | 237 passed |

<details>
<summary>T-S 到 T-Z 详细记录</summary>

**T-S 完成内容：**
- `JobSpec` 增加输入边界，`/api/config` 避免 `video_paths` 必填校验污染默认配置读取。
- finished job 删除流程改为锁内更新状态、锁外递归删除 temp。
- `run_translation_and_write()` 三路径统一关闭 run logger 并清理 `events._thread_local.run_logger`。
- translation cache JSON/JSONL 损坏时打印 warning 但不中断任务。

**T-U 完成内容（`src/pipeline/` 子模块）：**
- `audio.py`：filter chain、视频 hash、audio cache key、`extract_audio()`、时长 probe。
- `cleanup.py`：translation cache 清理、ASR checkpoint 清理、job temp 清理。
- `gender_split.py`：F0 None 过滤、gender turn 重切段、ASR noise 过滤。
- `artifacts.py`：`AsrArtifacts`、snapshot 路径、序列化、atomic 写、加载恢复。
- `ids.py`：`sanitize_job_id()`。
- `output.py`：输出目录解析、bilingual 模式解析。
- `aligned_cache.py`：`aligned_segments.json` cache 读取与 key 校验。
- `quality.py`：quality report、术语表解析、quality segment 转换。

当前后端大文件排行（供 B1–B3 参考）：
- `src/llm/translator.py`：2270 行。
- `src/whisper/pipeline.py`：2259 行。
- `src/main.py`：1487 行。
- `src/whisper/local_backend.py`：1209 行。
- `src/web/pipeline_manager.py`：531 行。

**T-V 完成内容（折叠）：**
- `src/web/static/app.js` 拆为 `src/web/static/js/` 下 14 个 ES Module，入口改为 `<script type="module" src="js/main.js">`。
- 旧 `app.js` 归档到 `agents/rm/app.js.bak`。
- 修复日志刷新滚动事件导致右键粘贴菜单关闭的问题。

**T-W 完成内容：**
- `src/llm/translator.py`：默认 `LLM_REASONING_EFFORT=xhigh`；Chat 和 Responses 均只归一化到 `medium` / `xhigh`。
- `src/llm/patch.py`：Micu+Grok Responses 特例不再把 `xhigh` 映射成 `high`，直接发送 `{"effort":"xhigh"}`。
- `src/core/job_context.py`、`src/web/pipeline_manager.py`、`src/web/routes/config.py`、`src/web/models.py`：Web 任务和 settings API 只接受/快照 `medium` / `xhigh`。
- `src/web/static/index.html`、`src/web/static/js/settings.js`：前端推理强度下拉框只显示 `medium` / `xhigh`，默认 `xhigh`。
- `.env.example` 与默认配置同步为 `LLM_REASONING_EFFORT=xhigh`；真实 `.env` 是本地私密文件，不提交。
- 验证：`py_compile`、`node --check src/web/static/js/settings.js`、定向 pytest 36 passed。

**T-X 完成内容：**
- `src/llm/translator.py`：批量翻译改为 full JSON stable prefix + `requested_ids`；本地按 batch 区间计算需要翻译的全局 id，模型只返回指定 id。`TRANSLATION_PREFIX_WARMUP=1` 默认预热前缀，`TRANSLATION_FULL_JSON_PREFIX_MAX_CHARS` 控制 full-prefix 上限，超限回退 summary。
- 翻译并发诊断写入 progress JSONL：`batch_start`、`batch_first_token`、`batch_finish`，附带 `started_ts`、`first_token_ts`、`finished_ts`、worker thread、requested ids、request count、missing ids、cache hit/miss token。
- 翻译风格规则收口：男性器官统一“肉棒”，女性器官统一“小穴”，去掉“菊花”固定译法；人名按日语读音罗马音化，ASR 同音纠错只在明显同一称呼时进行，禁止把不同汉字姓氏或不同读音强行合并。
- 翻译后 repair pass 默认开启：只修复代码侧选中的高风险 id；候选覆盖女性器官术语漂移、明显 ASR 同音/上下文漂移、半句断裂。repair prompt 只暴露抽象 reason（如 `asr_homophone_or_context_drift`），不在静态 prompt 中堆片内错例。
- `PROMPT_VERSION=v2.5`，确保 fixed-prefix、repair、人名策略调整后不复用旧缓存。
- NAMH-055 验证：repair 修复“小穴/阴道/芒果/香肠/半句断裂”等问题，术语残留清零。
- NMSL-036 验证：576 条，fixed-prefix 并发 3 batch；人名从过度合并 `高松` 修正为保守罗马音化；repair 修复 #85/#446/#515/#576 等 ASR 同音导致的上下文漂移。最终验证：`.venv/Scripts/python -m pytest` 229 passed。

**T-Y 完成内容：**
- `src/web/models.py`、`src/web/routes/config.py`：`SettingsRead/SettingsUpdate` 支持 `asr_context`，`/api/settings` 读写 `ASR_CONTEXT` 并同步运行时环境和 `.env`。
- `src/web/pipeline_manager.py`：创建任务时如果 spec 未显式提供演员名，则从持久 `ASR_CONTEXT` 快照到任务 spec，保证队列中的任务不受后续修改污染。
- `src/web/static/js/settings.js`：加载 settings 时回填 `r-asr-context`；提交任务前自动保存演员名、HF 镜像、翻译设置和 API 连接配置。
- `src/web/static/js/formMemory.js`：`r-asr-context` 不再进入 localStorage 表单记忆，避免和持久 settings 双源冲突。
- `src/web/static/index.html`：移除“保存设置”按钮；用户提交任务即保存配置，手动清空演员名并提交会清空持久值。
- 验证：`tests/web/test_jobs_api.py` 10 passed；`node --check src/web/static/js/settings.js` / `files.js` / `formMemory.js` 通过。

**T-Z 完成内容：**
- `src/pipeline/gender_split.py`：翻译前 ASR 噪声过滤新增纯特殊符号判断。去空白后若整段没有 Unicode 字母或数字，则视为无语言信息噪声，过滤 `◆◆◆`、`♪♪♪`、`！？！？` 等 ASR 特殊符号段。
- 过滤规则保持保守：`えっ！？`、`もう1回`、`ラブ` 等含日文/字母/数字的语义段保留，不交给 LLM 判断。
- SORA-575 `whisper-ja-1.5b` 离线验证：应用新规则后，`aligned_segments.json` 中 947 段会在翻译前删除 2 条 `◆◆◆`。
- 验证：`tests/test_f0_filter.py` 9 passed；`tests/test_e2e_task_s.py` + `tests/test_e2e_crash_resume.py` 7 passed；翻译/cache/progress 定向 40 passed。

</details>

## 4. 当前待办 / Backlog

> **Backlog 已清空。** B1–B7 全部完成，全量回归 241 passed。

<details>
<summary>B1–B7 已完成记录</summary>

### B1 ✅：拆分 `src/whisper/pipeline.py`

完成：新增 `src/whisper/backends/registry.py`（后端选择/dispatch）和 `src/whisper/checkpoint.py`（checkpoint 路径/读写），pipeline.py 通过 import 保持所有原名可访问。241 tests passed。

### B2 ✅：拆分 `src/llm/translator.py`

完成：新增 `src/llm/cache.py`（translation cache 读写、cache key）和 `src/llm/prompt.py`（系统提示、payload 构建、PROMPT_VERSION）；translator.py 通过 `prompt_module` / `translation_cache` 代理保持原名可访问。241 tests passed。

### B3 ✅：压缩 `src/main.py` 主编排

完成：新增 `src/pipeline/stage_log.py`（stage event / run log helper）和 `src/pipeline/output_writer.py`（JSON/timings/SRT 写出路径）；main.py 净减 228 行。241 tests passed。

### B4 ✅：ASR 滑动上下文注入

完成：`WhisperModelBackend.transcribe_texts()` 增加 `initial_prompts` 参数；`pipeline.py` 维护滑动窗口（`ASR_SLIDING_CONTEXT_SEGS=2`），间隔 > 0.5s 或 gender turn 切换时重置。241 tests passed。

### B5 ✅：VAD 微短段预合并

完成：`pipeline.py` 新增 `_merge_short_vad_chunks()`，相邻双段均 < `VAD_MERGE_SHORT_MAX_S=0.8s` 且间隔 < `VAD_MERGE_GAP_MAX_S=0.3s` 时物理拼接；保留 `merged_from` 元数据供对齐回溯。241 tests passed。

### B6 ✅：字幕软切分点

完成：`src/subtitles/writer.py` 新增 `soft_split_long_segments()`，对 end-start > `SUBTITLE_SOFT_MAX_S`（默认 6.0s）且有 words 的 segment，优先在中文句末标点（。！？）或日文助词（は/が/を/に 等）词边界处拆分；`src/main.py` 在 segments→srt_blocks 转换前调用。fallback 模式（无 words）不受影响。241 tests passed。

### B7 ✅：Repair Pass 增强——长度错配强制候选

完成：`_select_translation_repair_ids()` 增加长度比率检测，len(zh)/max(len(ja),1) 超出 [TRANSLATION_REPAIR_LENGTH_RATIO_MIN=0.25, TRANSLATION_REPAIR_LENGTH_RATIO_MAX=4.0] 窗口时强制入候选，reason=`length_mismatch`，优先于低置信度候选。237 tests passed。

</details>

---

### 暂缓（低优先级，待验证后再决定）

- **温度回退/压缩率监控**：当前 `do_sample=False` + beams 模式收益有限，待实际幻觉统计后再决定。
- **自适应 VAD 阈值**：当前 0.35 表现稳定，复杂度与收益比待评估。

---

## 5. 折叠历史基线

<details>
<summary>历史完成项 T-A 到 T-R</summary>

- **T-A** ✅ ASR Recovery → VAD 二次细分（`src/audio/vad_refine.py`）。
- **T-B** ✅ F0 词级时间轴 + multi-cue gender 切分。
- **T-C / T-D / T-E** ✅ Web 控制台、Stage 事件 JSON 化、重试断点续传和 cancel event 透传。
- **T-F / T-G** ✅ HF 镜像开关、Web 配置项扩展。
- **T-H / T-I / T-J** ✅ 后端稳定性、CLI 瘦身、全局 env 并发污染根治。
- **T-K** ↩️ transformers 兼容性回滚，保留四个稳定 ASR 后端，依赖固定回 `transformers==4.57.6`。
- **T-L / T-M** ✅ GitHub 发布前文档、配置、入口、翻译上下文与 cache key 收口。
- **T-N** ✅ Windows Release exe 打包配置。
- **T-O** ✅ Web 表单记忆与右键粘贴。
- **T-P** ✅ OpenAI Responses 翻译格式兼容。
- **T-Q** ✅ F0 后 gender turn 字幕重切段。
- **T-R** ✅ 翻译重试与请求清理；Micu+Grok Responses 特例移入 `src/llm/patch.py`；翻译前 ASR 噪声过滤扩展到纯英文幻觉 token。

</details>

<details>
<summary>HAME-052 历史验证基线</summary>

四后端 skip-translation 对比：

| 后端 | 状态 | ASR 转写 | Wall time | 字幕数 |
|------|------|----------|-----------|--------|
| `anime-whisper` | ok | 48.52s | 336.88s | 150 |
| `qwen3-asr-1.7b` | ok | 251.71s | 578.15s | 164 |
| `whisper-ja-1.5b` | ok | 170.74s | 464.33s | 165 |
| `whisper-ja-anime-v0.3` | ok | 41.89s | 229.46s | 151 |

默认全量翻译（anime-whisper + bilingual）：`pipeline_total=575.30s`，字幕块数 150，产物 `video/HAME-052.srt`。

</details>

<details>
<summary>历史回归记录</summary>

- T-B 完成后：176 tests passed。
- T-C 完成后：178 tests passed。
- T-H 完成后：180 tests passed。
- T-I 完成后：179 tests passed。
- T-J 完成后：179 tests passed。
- T-L 完成后：配置 / 模型路径 / Web jobs API 定向回归 16 tests passed。
- T-M 完成后：翻译上下文 / 缓存 key / 清理 / Web jobs 定向回归 32 tests passed。
- T-Q 完成后：F0 定向 15 passed；ASR job/cache 定向 7 passed。
- T-S 完成后：后端稳定性收口定向 58 tests passed。
- T-U 完成后：后端拆分定向 38 passed；宽后端回归 93 passed。
- T-X 完成后：翻译 fixed-prefix / repair / 人名策略定向通过；全量 229 passed。
- T-Y 完成后：Web settings/jobs 定向 10 passed；前端相关 JS check 通过。
- T-Z 完成后：ASR 噪声过滤、E2E、翻译/cache/progress 定向 56 passed。

</details>
