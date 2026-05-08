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
- 翻译 reasoning effort 只保留两档：`medium` / `xhigh`。Chat Completions、标准 Responses、Micu+Grok Responses patch 均直接透传 `medium` 或 `xhigh`，不再把 `xhigh` 映射为 `high`。
- 字幕约束：`MAX_SUBTITLE_DURATION=8.0`，`ASR_MERGE_HARD_MAX_DURATION=9.0`，相邻短块合并受标点、speaker guard 和 gender guard 限制。
- 默认 ASR recovery：`ASR_RECOVERY_ENABLED=0`。异常 ASR 文本块排查时才手动打开；男女混句边界由 F0 后 gender turn 重切段处理。
- 断点续传：ASR checkpoint、`aligned_segments.json`、translation cache、translation artifact snapshot。
- 主入口：`.venv/Scripts/python run_web.py`。后端调试入口是测试、诊断脚本，或直接调用 `run_asr_alignment_f0()` / `run_translation_and_write()`。
- Web 任务参数通过 `JobSpec -> JobContext` 显式传入后端，不再依赖全局 `.env` 热覆盖 ASR、字幕、输出目录、batch/worker、临时文件保留等任务级配置。
- `.env` 只保留跨任务持久配置：`API_KEY`、`OPENAI_COMPATIBILITY_BASE_URL`、`LLM_MODEL_NAME`、`LLM_API_FORMAT`、`LLM_REASONING_EFFORT`、`TARGET_LANG`、`HF_ENDPOINT`、`TRANSLATION_GLOSSARY`。

---

## 2. 当前行为约定

- 普通使用入口是 `.venv/Scripts/python run_web.py`；旧 `src/main.py --input ...` CLI 已移除。
- 需要观察后端长跑进度或收集用户反馈日志时，通过 Web 高级项启用 `RUN_LOG_ENABLED=1`；默认写入 `temp/log/`。
- `HF_HOME` 默认 `./models`；首次运行把 HuggingFace repo 下载到 `models/<namespace>-<repo>/`。
- `HF_HUB_CACHE`、`HF_XET_CACHE`、`TORCH_HOME` 默认在 `./temp/` 下；ASR recovery 输出默认在 `temp/recovery`。
- `HF_ENDPOINT` 必须为空或完整 URL，例如 `https://hf-mirror.com`。
- 成功运行后默认直接删除一次性 job 临时目录；保留下次可复用的运行缓存，例如 `temp/hf-cache`、`temp/web` 状态和 `models/`。
- Web“保留临时文件”选项仅用于调试，保留当前任务临时目录；不再通过全局 `KEEP_TEMP_FILES` 控制任务行为。
- 所有项目配置、README、agent 本地说明应使用项目相对路径，不写本机绝对路径。
- `OPENAI_COMPATIBILITY_BASE_URL` 是 OpenAI-compatible API 配置名，保留不改。

---

## 3. 已完成任务摘要

| Task | 内容 | 结果 |
|------|------|------|
| T-S ✅ | 后端稳定性收口（JobSpec 边界、删除锁顺序、run logger 泄漏、cache 损坏容忍） | 58 passed |
| T-U ✅ | 后端大文件拆分：`src/main.py` helper 迁入 `src/pipeline/`（8 个子模块） | 宽回归 93 passed |
| T-V ✅ | 前端 `app.js` 拆分为 14 个 ES Module；修复日志刷新导致粘贴菜单被关闭的 bug | 语法全通，手动验证 |
| T-W ✅ | 翻译 reasoning effort 收口为 `medium` / `xhigh` 两档；Responses 不做兼容降级映射 | 36 passed |

<details>
<summary>T-S 到 T-W 详细记录</summary>

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
- `src/whisper/pipeline.py`：2259 行。
- `src/llm/translator.py`：1626 行。
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

</details>

## 4. 当前待办 / Backlog

### B1：继续拆分 `src/whisper/pipeline.py`

现状：当前最大后端文件，约 2259 行。

建议拆分方向：
- backend registry / dispatch：后端列表、label、选择逻辑。
- ASR checkpoint：读写、路径、恢复策略。
- chunk/VAD orchestration：切块、VAD backend 调度、chunk 元数据。
- transcribe/alignment orchestration：ASR 文本转写、forced alignment、stage details 汇总。
- recovery：ffmpeg VAD 二次细分与重转写路径。

边界：保持公开入口不变，先拆纯 helper，再拆阶段编排，避免一次性重排主流程。

### B2：继续拆分 `src/llm/translator.py`

现状：约 1626 行，仍是第二大后端文件。

建议拆分方向：
- cache：JSON/JSONL cache 读写、损坏 warning、cache key。
- prompt：系统提示、用户 payload、术语表和人物参考。
- client/chat：Chat Completions 请求和流式解析。
- client/responses：Responses API 请求和流式解析。
- batching/retry：batch 切分、缺失 id 重试、进度事件。

边界：当前翻译 API 行为稳定，拆分时优先保留 `translate_segments()` 对外签名。

### B3：压缩 `src/main.py` 主编排

现状：已从约 1621 行降到约 1487 行，但仍包含 ASR stage event、timing、JSON 输出、翻译写出三段重复结构。

建议拆分方向：
- stage event / run log helper。
- JSON/timings payload writer。
- ASR artifact build。
- translation write-output paths 的重复写 JSON/SRT/timings 分支。

边界：`run_asr_alignment_f0()` / `run_translation_and_write()` 暂时保留为后端公开入口。

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

</details>
