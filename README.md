# JAVTrans

JAVTrans 是一个本地字幕生成工具，面向 Windows + NVIDIA 显卡，也可在 WSL2 / Linux 下源码运行。它把视频处理成日文字幕、中文字幕或中日双语字幕，并把音频准备、SpeechBoundary-JA scorer v5、Boundary Refiner v6、Pre-ASR CueQC v6、ASR、Boundary chunk 字幕时间轴、LLM 翻译和质量报告串成一条本地优先的流水线。

项目目标：本地完成视频、音频、边界切分、ASR 和时间轴重计算；LLM 只负责翻译、术语一致和口吻连贯，不负责脑补剧情或修正 ASR 误听。

致谢：[WhisperJAV](https://github.com/a63n/WhisperJAV) 为本项目早期路线提供了重要参考。实验历史、路线取舍、调试记录和参考来源见 [HISTORY.md](HISTORY.md)。

---

## 当前状态

- 默认 ASR：`jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`。
- 低配置可选 ASR：`jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`。
- 默认边界系统：SpeechBoundary-JA，backend key 为 `speech_boundary_ja`。
- 当前主链路：SpeechBoundary-JA scorer v5 -> Boundary Refiner v6 -> Pre-ASR CueQC v6 预留位 -> ASR -> 字幕时间轴 -> 翻译。
- 三个 Mamba checkpoint 按当前 ASR repo id 自动解析；metadata 的 repo id / schema / feature hash 必须匹配。
- Pre-ASR CueQC v6 checkpoint 仍在训练准备阶段，默认关闭；ASR-after CueQC v4 仅保留为显式 opt-in 的 shadow/mining，不参与默认 keep/drop。
- 当前是断兼容重构状态：旧 scorer、旧 ASR-after active CueQC checkpoint 和旧数据集不再作为 active contract。
- 默认显存目标：单阶段峰值适配 6GB 级 NVIDIA 显卡。8GB 本机可提高 `.env` 中的 ASR batch size；更小显存则手动降低。
- ASR chunk 切分只使用 scorer 的 primary acoustic cut candidates；weak cut candidates 只作为字幕布局、审计和后续训练的时间锚点。
- 7 秒是字幕显示 soft guard，不是 ASR chunk 上限；字幕 layout 可把过长 cue 按文本断句拆开，并优先吸附到 weak cut time，再退回比例估算。

---

## 快速开始

### Release 版

GitHub Releases 只用于发布 source code、release notes 和版本说明，不上传大型 Windows `.7z` 运行包。Windows 运行包由维护者按需本地打包成单个 `.7z`，通过网盘或其他外部分发渠道提供；拿到包后解压运行：

```text
JAVTrans.exe
```

Windows 打包版默认内置 1.7B ASR、0.6B 低配 ASR、`ffmpeg` / `ffprobe`，以及 repo-id registry 使用的 Boundary Refiner / CueQC / SpeechBoundary-JA scorer checkpoint。

Release 版不内置 Microsoft Edge WebView2 Runtime。大多数 Windows 10/11 已自带；如果无法打开窗口，请安装 [Microsoft Edge WebView2 Evergreen Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/)。

### 源码运行

推荐环境：

- Windows 10/11 或 WSL2 / Linux。
- NVIDIA 独立显卡和较新的驱动。
- Python 3.13+。
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
LLM_MODEL_NAME=deepseek-v4-flash
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

Web 提交是否使用 CUDA 取决于后端服务进程是否能看到 GPU，而不是浏览器本身。完整 SpeechBoundary-JA / ASR smoke 应确认日志中出现 `cuda_available=True`、`device=cuda:0` 或 `actual_device=cuda`。

---

## 使用流程

1. 打开网页控制台。
2. 选择视频文件。
3. 选择字幕模式、ASR 后端和翻译设置。
4. 提交任务。
5. 在输出目录查看 SRT、质量报告和日志。

勾选“不翻译（仅日文字幕）”时，流水线仍会执行边界规划、可选 Pre-ASR CueQC、ASR 和 Boundary chunk 字幕时间轴生成，但跳过 LLM 翻译，最终输出 `<视频名>.ja.srt`。这是验证本地边界 / ASR / 字幕时间轴链路的推荐 smoke 模式。

主流水线：

```text
视频 -> 音频准备 -> Boundary/Scorer primary cuts -> Refiner -> 可选 Pre-ASR CueQC -> ASR -> 字幕 layout -> 翻译 -> SRT / quality report
```

LLM 翻译前会先固定 cue plan。SRT writer 只处理字幕显示层：保留 2-frame gap 和 20-frame 最小显示时长；遇到 `>7s` cue 时先按 ASR 文本断句确定文本拆分点，再吸附到对应时间窗内的 weak cut，没有可用 weak cut 才按比例估算。这个步骤不反向修改 ASR chunk 语义。

---

## 默认模型

| 用途 | 默认来源 | registry 目标文件 |
| --- | --- | --- |
| 默认 ASR | `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame` | `models/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame` |
| 低配 ASR | `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame` | `models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame` |
| SpeechBoundary-JA frozen feature | 跟随当前 ASR repo id | `models/<repo-tag>` |
| SpeechBoundary-JA scorer v5 | Mamba2 frame boundary scorer；输出 speech/split 两头；decoder contract 为 `topographic_split_micro_resolver_v3`；primary cuts 用于 ASR chunk split，weak cuts 透传到字幕 layout | `src/boundary/ja/checkpoints/speech_boundary_ja_frame_boundary_scorer_v5.<repo-tag>.pt` |
| Boundary Refiner v6 | learned `transformers.Mamba2Model` edge-only delta checkpoint | `src/boundary/checkpoints/boundary_edge_refiner_v6.<repo-tag>.pt` |
| Pre-ASR CueQC v6 | learned binary checkpoint；pre-ASR numeric + micro chunk features -> keep/drop；训练完成前默认关闭 | `src/asr/checkpoints/cueqc_pre_asr_mamba_v6_binary.<repo-tag>.pt` |

常用配置见 [.env.example](.env.example)。通常只需要修改 API key、翻译模型、`HF_ENDPOINT`、`ASR_BACKEND` 和 batch size。

推荐 batch 档位：

| 显存档 | `ASR_BATCH_SIZE_BY_REPO` |
| --- | --- |
| 6GB 默认 / 分发 | `1.7B=32, 0.6B=64` |
| 8GB 本机实验 | `0.6B=128, 1.7B=64` |

8GB 档面向本机调参和快速审计；如果后台还有其他 CUDA 进程或出现 OOM，先退回 6GB 档。

推理需要 ASR / SpeechBoundary-JA frozen feature Hugging Face 模型，以及与当前 repo id 匹配的三个 Mamba checkpoint。Windows 打包版默认应内置 1.7B ASR、0.6B 低配 ASR和 repo-id registry 目标 checkpoint；源码运行时如果本地没有 Hugging Face 模型，仍会按需下载到 `models/`。Mamba checkpoint 的 repo id 归属只看 metadata，不靠文件名推断；文件名里的 repo tag 只是人工可读提示。registry 缺失、覆盖映射未命中当前 repo id、文件不存在、schema 不匹配或 metadata 不匹配都会 fail-fast。训练时生成的 CUDA feature cache、synthetic WAV、sequence JSONL、tensor cache 和 `datasets/train/...` 产物都不是运行依赖，不随源码或 Windows release 打包。

---

## 字幕与文本策略

- ASR 文本会做 Unicode NFKC、空白归一、换行折叠和展示安全处理。
- `ASR_CONTEXT` / `ASR_HEAD_CONTEXT` 只作为 Qwen ASR 提示词，不作为字幕后处理删除规则。
- Runtime 不使用具体词黑名单或时长启发式删除短促人声；是否进入 ASR 由 Pre-ASR CueQC 模型标签决定。
- 字幕时间轴来自 Boundary chunk；ASR 输出文本只负责显示，不驱动默认切分。

---

## 输出与缓存

- `video/<视频名>/`：正式字幕、质量报告和人工质检报告。
- `models/`：Hugging Face 模型缓存。
- `tmp/jobs/<job_id>/`：Web / pipeline 单次任务临时目录；`JOB_TEMP_DIR` 默认是 `./tmp/jobs`。
- `tmp/chunks/`：ASR wav chunk 和 crash-resume checkpoint 的一次性运行目录。
- `tmp/cache/boundary/`：SpeechBoundary-JA frame score 到 Boundary Refiner 输出的 boundary-cache v10。
- `tmp/cache/torch/`、`tmp/cache/hf/`：torch / Hugging Face 运行缓存。
- `tmp/log/`：启用运行日志后的任务日志。
- `datasets/`：本地训练、验证、测试数据归档，默认 ignored；不参与普通推理和 release 打包。
- `agents/temp/`：研究脚本、smoke、临时日志和中间产物。
- `agents/audits/`：可长期复查的本地审计页，统一从 `agents/audits/index.html` 进入；导航按更新时间倒序排列，最上面是最新需要审计的页面。该目录是本地研究产物，默认 ignored，不会随 `git push` 发布。Windows 使用 `.\tools\audits\serve_audits.ps1`，Linux / WSL2 使用 `tools/audits/serve_audits.sh`；脚本启动轻量审计静态服务，支持音频 Range seek，不注入自动刷新脚本。导航删除按钮需要通过对应脚本启动才会实际移动本地审计目录并重建导航，直接打开 HTML 只能显示手动删除命令。CueQC/duration 音频审计页支持按视频、簇/时长桶、alignment、alignment issue、时长、置信度筛选，并按 duration、confidence、char count、CPS、start/chunk 排序；CueQC 簇广播页不加载媒体，只导出簇级标签和 sample-level broadcast 标签。

成功运行后默认删除一次性 job 临时目录；保留可复用缓存，例如 `models/`、`tmp/cache/boundary/` 和 Web 状态。

---

## 常见问题

### 模型下载慢

设置：

```env
HF_ENDPOINT=https://hf-mirror.com
```

或提前把模型下载到 `models/` 对应目录。

### CUDA 没有被使用

确认日志中出现：

```text
actual_device=cuda
model_param_device=cuda:*
```

受限 sandbox、错误的 PyTorch wheel、驱动问题或从非 GPU 环境启动 Web 服务都可能导致 CPU fallback。

### 显存不足

默认配置按 6GB 级显存目标设置：

```env
ASR_BATCH_SIZE_BY_REPO=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=32,jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=64
```

8GB 本机可尝试：

```env
ASR_BATCH_SIZE_BY_REPO=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=64,jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=128
```

如果仍然 OOM，优先降低：

```env
ASR_BATCH_SIZE_BY_REPO=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame=8,jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame=24
```

### 长任务怎么排查

启用运行日志后，日志会写入 `tmp/log/` 或任务输出目录。反馈问题时请保留 `.run.log`、质量报告和对应 SRT。

---

## 开发

主要代码位置：

- `src/main.py`：主流程编排。
- `src/core/`：配置和任务上下文。
- `src/pipeline/`：音频、缓存、输出、质量报告和阶段日志。
- `src/asr/`：ASR、Boundary 字幕时间轴分配、Pre-ASR CueQC / ASR-after shadow CueQC 和转写流程。
- `src/boundary/`：Boundary Refiner v6 checkpoint loader、edge-sequence Mamba2 adapter、core planner 和 boundary-cache v10。
- `src/boundary/ja/`：SpeechBoundary-JA scorer v5、PTM/MFCC feature cache schema、训练数据 manifest 和 frame-score 训练工具。
- `src/llm/`：翻译 prompt、cache、glossary、API patch 和 translator。
- `src/subtitles/`：SRT writer、字幕选项和字幕 QC。
- `src/web/`：FastAPI 接口和静态前端。
- `tools/`：训练、字幕审计、workflow smoke 和发布辅助脚本。

常用测试：

```bash
PYTHONIOENCODING=utf-8 uv run pytest tests/test_config.py tests/web/test_jobs_api.py tests/test_asr_backend_dispatch.py
PYTHONIOENCODING=utf-8 uv run pytest tests/test_boundary_cache.py tests/test_boundary_planner.py tests/test_chunk_packer.py tests/test_pipeline_chunk_config_runtime.py
PYTHONIOENCODING=utf-8 uv run pytest tests/test_translation_cache.py tests/test_translator_prompt.py tests/test_quality_report_output.py
```

维护者本地打包 Windows 运行包：

```powershell
uv pip install pyinstaller torchcodec
.\packaging\build_windows.ps1 -Clean
```

构建产物位于 `dist\JAVTrans\JAVTrans.exe`。如需分发，可再压缩为 `.7z` 并上传到网盘；GitHub Releases 只发布源码和版本说明。打包细节见 `packaging/README.md`。

---

## 工具索引

所有 Python 工具都从项目根目录执行，并使用当前 `.venv`：

```powershell
$env:PYTHONIOENCODING='utf-8'
uv run python -m <module> --help
```

- `tools.workflows.run_full_workflow`：命令行完整工作流 smoke。
- `tools.web.smoke.start_server` / `submit_job` / `poll_job` / `summarize_job`：Web 服务 smoke 和任务汇总。
- `tools.audits.audit_nav`、`tools.audits.serve_static`、`tools.audits.serve_audits.ps1`、`tools.audits.serve_audits.sh`：维护和启动本地审计导航页；静态服务支持音频 Range seek 和导航页删除 API。
- `tools.audits.generate_cueqc_cluster_audit_html`：生成 CueQC/duration 音频审计页，支持 chunk/context 播放和字幕对照。
- `tools.audits.generate_cueqc_cluster_broadcast_html`：生成独立 CueQC 簇级 keep/drop 广播标注页；混簇/跳过只记录 abstain，不导出 sample-level 标签。
- `tools.audits.generate_cueqc_prediction_audit_html`：生成 CueQC 预测 false-drop 审计页。

```powershell
uv run python -m tools.workflows.run_full_workflow --video video/<your-video>.mp4 --task-name 20260617_191654_cli-smoke --label smoke
uv run python -m tools.web.smoke.start_server --run-dir agents/temp/20260617_191654_web-smoke
uv run python -m tools.web.smoke.submit_job --video-path video/<your-video>.mp4 --output-dir video --run-dir agents/temp/20260617_191654_web-smoke
uv run python -m tools.web.smoke.poll_job --job-id-file agents/temp/20260617_191654_web-smoke/job_id.txt --run-dir agents/temp/20260617_191654_web-smoke --interval-seconds 300
uv run python -m tools.web.smoke.summarize_job --job-id <job_id> --run-dir agents/temp/20260617_191654_web-smoke
```

<details>
<summary>训练、诊断和历史对照工具</summary>

当前 active 训练入口：

- `tools.boundary.ja.build_scorer_v5_native_dataset`
- `tools.boundary.ja.build_feature_cache`
- `tools.boundary.ja.train_feature_scorer`
- `tools.boundary.build_refiner_frame_sequence_dataset`
- `tools.boundary.train_refiner`
- `tools.asr.cueqc.compile_pre_asr_v6_features`
- `tools.asr.cueqc.train_pre_asr_v6_binary`

诊断和数据准备：

- `tools.boundary.ja.export_frame_scores`
- `tools.boundary.ja.diagnose_split_peaks`
- `tools.boundary.ja.summarize_scorer_checkpoint_by_dataset`
- `tools.boundary.build_weighted_source_manifest`
- `tools.boundary.export_cueqc_cluster_seed_hardcases`
- `tools.boundary.export_cueqc_seed_drop_background_spans`
- `tools.boundary.prepare_cueqc_drop_hard_negative_sources`

ASR-after CueQC v4 工具只用于 shadow/mining 和历史对照，不是默认 active keep/drop：

- `tools.asr.cueqc.extract_features_v4_binary`
- `tools.asr.cueqc.extract_feature_shards`
- `tools.asr.cueqc.merge_features_v4_binary`
- `tools.asr.cueqc.train_mamba_v4_binary`
- `tools.asr.cueqc.predict_v4_binary`
- `tools.asr.cueqc.compile_stage2a_features_v4_binary`

</details>
