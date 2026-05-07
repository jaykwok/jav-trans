# JAVTrans 工程计划

> **For agentic workers:** 执行前重新读取本文件、`.env` 和涉及源文件。Python 统一使用 `.venv/Scripts/python`，pip 统一使用 `.venv/Scripts/pip`。项目运行临时/cache 文件放 `./temp/`；需要归档删除时移动到 `agents/rm/`。普通回复不要使用 cc-connect relay；session 输出会自动回灌。

**工作目录：** 项目根目录

---

## 1. 当前锁定架构

- Windows 本机生产目标：NVIDIA RTX 4060 Ti 8GB，串行分时加载模型，阶段结束后卸载并清 CUDA cache。
- 主流水线：视频 → 音频准备 → WhisperSeg VAD → ASR → Forced Alignment 词级时间轴 → F0 性别检测 → F0 后 gender turn 重切段 → DeepSeek 翻译 → SRT/quality report。
- 默认 ASR 配置仍为 `ASR_BACKEND=anime-whisper`；Web UI 推荐排序把 `whisper-ja-anime-v0.3` 放在第一位。
- 支持后端：
  - `anime-whisper`
  - `qwen3-asr-1.7b`
  - `whisper-ja-1.5b`
  - `whisper-ja-anime-v0.3`
- 默认 VAD：`ASR_VAD_BACKEND=whisperseg`，`WHISPERSEG_THRESHOLD=0.35`。
- 默认翻译：`deepseek-v4-pro`，Reasoning + JSON Mode + Streaming；Web 任务默认 `translation_batch_size=100`，`translation_max_workers=8`。
- 字幕约束：`MAX_SUBTITLE_DURATION=8.0`，`ASR_MERGE_HARD_MAX_DURATION=9.0`，相邻短块合并受标点、speaker guard 和 gender guard 限制。
- 默认 ASR recovery：`ASR_RECOVERY_ENABLED=0`。只在手动排查异常 ASR 文本块时打开；recovery 通过 ffmpeg VAD 二次细分 + 同 ASR backend 重转写实现（已替换旧 audio-separator），不用于解决男女混句或字幕边界问题。
- 断点续传：ASR checkpoint、`aligned_segments.json`、translation cache。
- 主入口：`.venv/Scripts/python run_web.py`。根目录旧 `run.py` 已移除。
- 后端调试入口：测试、诊断脚本，或直接调用 `run_asr_alignment_f0()` / `run_translation_and_write()`；旧 `src/main.py --input ...` CLI 已移除。
- Web 任务参数通过 `JobSpec -> JobContext` 显式传入后端，不再依赖全局 `.env` 热覆盖 ASR、字幕、输出目录、batch/worker、临时文件保留等任务级配置。
- `.env` 只保留跨任务持久配置：`API_KEY`、`OPENAI_COMPATIBILITY_BASE_URL`、`LLM_MODEL_NAME`、`LLM_API_FORMAT`、`LLM_REASONING_EFFORT`、`TARGET_LANG`、`HF_ENDPOINT`、`TRANSLATION_GLOSSARY`。

---

## 2. 当前行为约定

- 普通使用入口是 `.venv/Scripts/python run_web.py`；旧 `src/main.py --input ...` CLI 已移除。
- 需要观察后端长跑进度或收集用户反馈日志时，通过 Web 高级项启用 `RUN_LOG_ENABLED=1`；默认写入 `temp/log/`，也可用 `RUN_LOG_DIR=./temp/log` 显式指定。
- `scripts/asr_backends_compare.py` 只有带 `--log` / `--log-dir` 才写持久日志。
- `HF_HOME` 默认 `./models`；首次运行把 HuggingFace repo 下载到 `models/<namespace>-<repo>/`，例如 `litagin/anime-whisper` → `models/litagin-anime-whisper/`。
- 不再兼容旧 HF cache 作为模型读取兜底；新下载目标固定为 `./models` 下的准确模型代号目录。
- `HF_HUB_CACHE`、`HF_XET_CACHE`、`TORCH_HOME` 默认在 `./temp/` 下；ASR recovery 输出默认在 `temp/recovery`，避免 `models/` 顶层出现 cache 或非模型代号目录。
- `HF_ENDPOINT` 必须为空或完整 URL，例如 `https://hf-mirror.com`；`hf-mirror.com` 这类缺协议值会在模型下载前被拒绝。
- 成功运行后默认直接删除一次性 job 临时目录；保留下次可复用的运行缓存，例如 `temp/hf-cache`、`temp/web` 状态和 `models/`。
- Web“保留临时文件”选项仅用于调试，保留当前任务临时目录；不再通过全局 `KEEP_TEMP_FILES` 控制任务行为。
- 所有项目配置、README、agent 本地说明应使用项目相对路径，不写本机绝对路径。
- `OPENAI_COMPATIBILITY_BASE_URL` 是 OpenAI-compatible API 配置名，保留不改。
- `F0_THRESHOLD_HZ` 只控制男女声分类阈值；长句内男女混说话人的字幕边界由 `MULTI_CUE_SPLIT_ENABLED=1` 的 F0 后 gender turn 重切段处理。

---

## 3. 已完成基线

- **T-A** ✅ ASR Recovery → VAD 二次细分（`src/audio/vad_refine.py`）
- **T-B** ✅ F0 词级时间轴 + multi-cue gender 切分
- **T-D** ✅ 后端 Stage 事件 JSON 化（`src/core/events.py`）
- **T-C** ✅ Web 控制台全量实现（FastAPI + SSE + PyWebView）；含 Web UI 前端改进（sticky 提交、重试、取消修复等）
- **T-E** ✅ Web UI 完善：重试断点续传、cancel_event 透传
- **T-F** ✅ HF 镜像开关（`hf_endpoint` 前后端同步）
- **T-G** ✅ 前端配置项扩展（输出目录、并行 worker、保留临时、推理强度、目标语言、术语表）
- **T-H** ✅ 后端稳定性：deque 限界、I/O 防抖、SSE 心跳、VAD timeout、jobs TTL（180 tests）
- **T-I** ✅ 代码库清理：main.py CLI 瘦身、BACKENDS 单一来源、advanced env textarea、cache 改 JSONL（179 tests）
- **T-J** ✅ B3 全局 env 并发污染根治：JobContext dataclass、pipeline_manager 删热补丁、main.py/translator.py 显式传参（179 tests）
- **T-K** ↩️ transformers 兼容性回滚：因 `qwen-asr` 与 `transformers>=5.x` 冲突，移除实验性第五 ASR 后端，依赖固定回 `transformers==4.57.6`
- **T-L** ✅ GitHub 发布前收口：移除旧 `run.py`、归档本地 ad-hoc scripts、README 重写、`.env.example` 缩为持久设置模板、`.gitignore` 补 `.env.*` 保护（配置相关 16 tests）
- **T-M** ✅ 发布前二次收口：翻译 prompt 显式接收任务级 `asr_context`，translation cache key 纳入人物参考；清理 cohere 本地模型/cache、pycache、孤儿 Web job 临时目录；README 使用说明改为运行环境安装优先（32 tests）
- **T-O** ✅ Web 表单易用性：所有可填写控件自动记忆上次填写值；文本输入控件支持右键粘贴菜单，API Key 不进入本地明文记忆。
- **T-P** ✅ 翻译 API 格式兼容：翻译侧支持 `LLM_API_FORMAT=chat|responses`，默认 Chat Completions；Responses API 使用 `/responses`、流式 output_text delta、JSON object text format。
- **T-Q** ✅ F0 后 gender turn 重切段：F0 标注 `words[].gender` 后按 `M/F` 转换积极拆分字幕段，避免男女同处长句时不换行（F0 相关 15 tests + ASR cache/job 7 tests）。

基线：子进程隔离、WhisperSeg VAD、翻译并行、断点续传、SRT 折行、quality report 等主功能均已落地；README、`.env.example`、`.gitignore` 已按 Web-first 架构收口，cohere 源码/模型/cache 已退出工作流。

---

## 4. HAME-052 验证基线

### 四后端 skip-translation 对比

| 后端 | 状态 | ASR 转写 | Wall time | 字幕数 |
|------|------|----------|-----------|--------|
| `anime-whisper` | ok | 48.52s | 336.88s | 150 |
| `qwen3-asr-1.7b` | ok | 251.71s | 578.15s | 164 |
| `whisper-ja-1.5b` | ok | 170.74s | 464.33s | 165 |
| `whisper-ja-anime-v0.3` | ok | 41.89s | 229.46s | 151 |

### 默认全量翻译（anime-whisper + bilingual）

pipeline_total=575.30s，字幕块数 150，产物 `video/HAME-052.srt`。

---

## 5. 当前 Task

### T-Q：F0 后 gender turn 字幕重切段 ✅

**完成内容**：
- `src/main.py` 在 `detect_gender_f0_word_level()` 成功标注词级 gender 后、`F0_FILTER_NONE_SEGMENTS` 前执行 `_split_segments_on_f0_gender_turns()`。
- 重切段策略为积极拆分：相邻有效词级 gender 从 `M` 到 `F` 或从 `F` 到 `M` 变化时立即拆段，不要求静音 gap；`None` 词不触发拆分并并入当前片段。
- 重切后的子段从对应 `words` 重建 `start` / `end` / `text` / `words` / `gender` / `source_chunk_index`，并记录 `asr_details["f0_gender_split"]`。
- 运行日志新增 `f0_gender_split segments_before=... segments_after=... split_count=...`，便于确认是否生效。
- ASR recovery 继续默认关闭，仅作为异常 ASR 文本救火工具；男女混句边界问题由 F0 后重切段解决。

**验收**：
```powershell
.venv/Scripts/python -m py_compile src/main.py src/audio/f0_gender.py
.venv/Scripts/python -m pytest tests/test_word_level_f0.py tests/test_gender_split.py tests/test_f0_filter.py -q --tb=short -p no:cacheprovider
.venv/Scripts/python -m pytest tests/test_job_tempdir.py tests/test_aligned_segments_cache.py -q --tb=short -p no:cacheprovider
```

结果：F0 定向 15 passed；ASR job/cache 定向 7 passed。

### T-P：OpenAI Responses 翻译格式兼容 ✅

**完成内容**：
- 翻译侧新增 `LLM_API_FORMAT`，可选 `chat` / `responses`，默认 `chat`。
- `chat` 保持原 `/chat/completions` 兼容路径、JSON Mode、streaming 和 DeepSeek thinking extra_body。
- `responses` 使用 OpenAI Responses API 兼容路径：`client.responses.create(stream=True, input=..., text={"format":{"type":"json_object"}})`，并解析 `response.output_text.delta`、reasoning delta、completed / incomplete / failed 事件。
- Web 设置 API (`/api/settings`) 支持读写 `llm_api_format`，Pydantic 限定为 `chat|responses`。
- 当前 `/api/models` 已确认可远程获取模型列表：配置 `https://api.deepseek.com` 返回 `deepseek-v4-flash`、`deepseek-v4-pro`。

**验收**：
```powershell
.venv/Scripts/python -m py_compile src/llm/translator.py src/core/config.py src/web/models.py src/web/routes/config.py
.venv/Scripts/python -m pytest tests/test_translation_progress.py tests/test_config.py tests/web/test_jobs_api.py -q --tb=short -p no:cacheprovider
```

### T-O：Web 表单记忆与右键粘贴 ✅

**完成内容**：
- Web 前端所有可填写控件（任务参数、翻译设置、开关、下拉、textarea）通过 `localStorage` 自动记忆上次填写值。
- 文本输入控件（input / textarea，包括 API Key 输入框）支持右键弹出“粘贴”菜单并写入当前光标位置。
- `s-apikey` 仅支持右键粘贴，不写入本地表单记忆，避免 API Key 明文持久化到浏览器存储。
- 后端 `/api/config`、`/api/settings` 加载完成后再次应用表单记忆，保证动态下拉框和后端持久设置不会覆盖用户上次前端选择。

**验收**：
```powershell
.venv/Scripts/python -m py_compile run_web.py src/web/app.py src/web/routes/config.py src/web/routes/jobs.py
```

### T-N：Windows Release exe 打包 ✅

**完成内容**：
- 新增 PyInstaller onedir 发布配置 `packaging/javtrans-web.spec`，产物名为 `JAVTrans.exe`。
- 新增 `packaging/build_windows.ps1` 和 `packaging/prepare_default_model.py`，构建前确认默认 ASR 模型 `efwkjn/whisper-ja-anime-v0.3` 以及默认流程辅助模型已下载。
- Release 包内置 Python 运行环境、当前 `.venv` 依赖、`ffmpeg.exe`、`ffprobe.exe`、ffmpeg 同目录 DLL、`icon.ico`、`icon.png`、Web 静态资源、默认 ASR 模型、WhisperSeg VAD、`openai/whisper-base` 特征提取器和 Qwen forced aligner。
- frozen 运行时把 `.env`、`temp/`、用户后续下载的 `models/` 定位到 exe 同目录；包内默认模型作为只读 fallback。
- exe 图标通过 PyInstaller `icon.ico` 设置；pywebview 窗口图标通过 `webview.start(icon=...)` 设置；Web 页面 favicon / header / drop-zone 使用根目录图标资产。
- README 增加 Release 使用方式和构建命令；`packaging/README.md` 记录打包边界。

**边界**：
- 不内置 Microsoft Edge WebView2 Runtime，普通用户缺失时需安装 Microsoft 官方运行时。
- 只内置默认 ASR 模型 `efwkjn/whisper-ja-anime-v0.3` 和默认流程辅助模型；其他 ASR 模型仍按需下载到 exe 同目录 `models/`。

**验收**：
```powershell
.venv/Scripts/python -m py_compile run_web.py src/utils/runtime_paths.py src/utils/model_paths.py src/core/config.py src/web/app.py packaging/prepare_default_model.py
.venv/Scripts/python -m pytest tests/test_model_paths.py tests/test_config.py tests/web/test_jobs_api.py -q --tb=short -p no:cacheprovider
.venv/Scripts/python -m PyInstaller --noconfirm --clean packaging/javtrans-web.spec
```

---

### T-M：发布前二次收口 ✅

**完成内容**：
- 翻译侧 `translate_segments()` 增加显式 `character_reference`，Web 任务通过 `ctx.asr_context` 传入翻译 prompt，不再只读导入时的全局 `ASR_CONTEXT`。
- translation cache key 纳入 `character_reference`，避免同一字幕在不同人名提示下错误命中旧缓存。
- 本地清理 cohere 残留模型/cache、Python bytecode、pytest 临时目录、孤儿 `temp/web/jobs/*`。
- README 使用说明前置，并把运行环境安装、官方下载链接、虚拟环境、依赖安装、Web 启动、任务提交和清理行为串成主流程。
- README / tests 移除旧 `temp/audio-separator` 运行缓存描述；ASR recovery 当前为 ffmpeg VAD 二次细分 + 同 ASR backend 重转写。

**验收**：
```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; .venv/Scripts/python -m pytest tests/test_batch_translation.py tests/test_translation_cache.py tests/test_translation_cache_key_versioned.py tests/test_translation_progress.py tests/test_auto_cleanup.py tests/web/test_jobs_api.py tests/web/test_cancel_resume.py -q --tb=short -p no:cacheprovider
```

结果：32 passed。

---

### T-L：GitHub 发布前文档 / 配置 / 入口收口 ✅（已归档）

**完成内容**：
- 根目录旧 `run.py` 已移除，当前主入口为 `.venv/Scripts/python run_web.py`。
- `src/main.py --input ...` CLI 已移除；后端 smoke/debug 通过测试、诊断脚本或直接调用 `run_asr_alignment_f0()` / `run_translation_and_write()`。
- `scripts/` 仅保留通用诊断脚本：`asr_backends_compare.py`、`benchmark_asr.py`。
- `.env.example` 只保留 Web 设置页会持久化的跨任务配置，不再放视频路径、输出目录、ASR 后端、字幕模式、batch/worker、临时文件保留等任务级参数。
- 真实 `.env` 已收敛为本地持久翻译服务配置；不得提交到 Git。
- `.gitignore` 忽略 `.env.*`，但保留 `.env.example`。
- README 已按 Web-first 架构重写，并加入 GitHub repository description 建议；使用说明以运行环境安装开头，包含官方安装链接。

**验收**：
```powershell
.venv/Scripts/python -m pytest tests/test_config.py tests/test_model_paths.py tests/web/test_jobs_api.py -q --tb=short -p no:cacheprovider
```

结果：16 passed。

---

### T-K：transformers 兼容性回滚 ↩️

**回滚原因**：`qwen-asr` 当前依赖 `transformers==4.57.6`，实验性第五 ASR 后端需要 `transformers>=5.x`。为避免破坏现有 qwen3 后端，回到四后端架构。

**当前状态**：
- 删除实验性第五 ASR 后端源码。
- 删除对应测试。
- `src/whisper/pipeline.py` 只注册四个稳定 ASR 后端。
- `src/web/models.py` 只暴露四个稳定 ASR 后端。
- `requirements.txt` 固定 `transformers==4.57.6`。

**验收**：
```powershell
.venv/Scripts/python -m pip install transformers==4.57.6
.venv/Scripts/python -m pytest tests/test_asr_backend_dispatch.py tests/web/test_jobs_api.py -q --tb=short -p no:cacheprovider
```

---

## 6. Backlog（暂不实施）

暂无待实施项。

---

## 7. 回归验证记录

全量命令：
```powershell
.venv/Scripts/python -m pytest -q --basetemp=temp/pytest_hame052_plan -p no:cacheprovider
```

历史关键记录：
- T-B 完成后：176 tests passed
- T-C 完成后（C7）：178 tests passed（含 `tests/web/test_jobs_api.py` + `test_pipeline_overlap.py`）
- T-H 完成后：180 tests passed
- T-I 完成后：179 tests passed（I1 删除 CLI batch 代码，对应测试随之移除）
- T-J 完成后：179 tests passed
- T-K 回滚后：实验性第五 ASR 后端代码/测试/注册删除，transformers 固定回 4.57.6
- T-L 完成后：配置 / 模型路径 / Web jobs API 定向回归 16 tests passed
- T-M 完成后：翻译上下文 / 缓存 key / 清理 / Web jobs 定向回归 32 tests passed
- T-Q 完成后：F0 后 gender turn 重切段定向 15 tests passed；ASR job/cache 定向 7 tests passed；`py_compile src/main.py src/audio/f0_gender.py` passed
