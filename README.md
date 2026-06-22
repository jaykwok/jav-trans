# JAVTrans

JAVTrans 是一个本地字幕生成工具，面向 Windows + NVIDIA 显卡，也可在 WSL2 / Linux 下源码运行。它把视频处理成日文字幕、中文字幕或中日双语字幕，并把音频准备、SpeechBoundary-JA scorer v4、Boundary Refiner v6、ASR、CueQC v4 binary keep/drop 路由、Boundary chunk 字幕时间轴、LLM 翻译和质量报告串成一条本地优先的流水线。

项目目标：本地完成视频、音频、边界切分、ASR 和时间轴重计算；LLM 只负责翻译、术语一致和口吻连贯，不负责脑补剧情或修正 ASR 误听。

致谢：[WhisperJAV](https://github.com/a63n/WhisperJAV) 为本项目早期路线提供了重要参考。实验历史、路线取舍、调试记录和参考来源见 [HISTORY.md](HISTORY.md)。

---

## 当前状态

- 默认 ASR：`jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`。
- 低配置可选 ASR：`jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`。
- 默认边界系统：SpeechBoundary-JA，backend key 为 `speech_boundary_ja`。它不是传统 VAD，而是面向字幕 chunk 的 frame-level boundary pipeline。
- SpeechBoundary-JA scorer 当前 schema 是 `speech_boundary_ja_mamba2_frame_boundary_scorer_v4`，输出三头：`speech_prob`、`split_boundary_prob`、`drop_gap_prob`。运行时先用 speech hysteresis 找 coarse island，用 `drop_gap_prob` 删除明确非语音 gap，再用 `split_boundary_prob` 做 peak picking / NMS / snap-to-valley 切开 island；不再支持旧 `cut_prob` gate。
- Boundary Refiner 当前 schema 是 `boundary_edge_refiner_v6`，只接收 scorer v4 产生的 island edge 上下文，只输出 `start_delta_s` / `end_delta_s`。旧 v5 checkpoint、`frame_sequence_v1` adapter、merge/split/context/padding 语义全部 fail-fast。
- CueQC 当前 schema 是 `cueqc_mamba_v4_binary`，在 ASR 后只做 `keep/drop` 二分类，runtime 规则是 `p_drop >= CUEQC_DROP_THRESHOLD` 才 drop。输入包含 1.7B/0.6B ASR internals、token trace、decoder stats、boundary/asr/timing 等数值特征；不再有 `text_bucket`、`drop_threshold_profile`、`density_level` 或 `vocalization_repetition`。
- 三个 Mamba checkpoint 都按当前 ASR repo id 从后端 registry 自动解析。默认目标文件名为 `speech_boundary_ja_frame_boundary_scorer_v4.<repo-tag>.pt`、`boundary_edge_refiner_v6.<repo-tag>.pt`、`cueqc_mamba_v4_binary.<repo-tag>.pt`；checkpoint metadata 的 repo id / schema / feature hash 必须匹配，否则任务报错。
- 本次是断兼容重构：旧 v3 scorer、v5 refiner、v3-Fusion CueQC checkpoint 不会被兼容加载。1.7B scorer v4 / Boundary Refiner v6 已落盘并进入 registry；CueQC v4 和 0.6B 同 schema checkpoint 尚未完成时会按 repo id fail-fast。
- 默认显存目标：单阶段峰值适配 6GB 级 NVIDIA 显卡。8GB 本机可提高 `.env` 中的 ASR batch size；更小显存则手动降低。
- 当前字幕策略：切分发生在 ASR 前的 scorer/refiner 阶段。ASR 后的 SRT writer 默认只做排序、no-overlap、2-frame gap 和必要 timing polish，不再做 subtitle-level soft split 或 display policy。

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

勾选“不翻译（仅日文字幕）”时，流水线仍会执行边界规划、ASR、CueQC v4 binary keep/drop 路由和 Boundary chunk 字幕时间轴生成，但跳过 LLM 翻译，最终输出 `<视频名>.ja.srt`。这是验证本地边界 / ASR / CueQC / 字幕时间轴链路的推荐 smoke 模式。

主流水线：

```text
视频
-> 音频准备
-> SpeechBoundary-JA scorer v4 frame scores
-> speech island / drop gap / split peak decoding
-> Boundary Refiner v6 edge trim
-> ASR speech-core chunks
-> CueQC v4 binary keep/drop routing
-> Boundary chunk subtitle timing
-> cue plan 时间轴归一化
-> LLM 翻译
-> SRT / quality report
```

LLM 翻译前会先固定 cue plan。SRT writer 只写入已经归一化的时间轴，不再隐式改变时间轴。

Boundary Refiner v6 只修 island 两端：Mamba2 输出 `start_delta_s` 和 `end_delta_s`。它不新增 chunk、不做中间切分、不跨 island merge，也不学习 ASR padding/context budget。

翻译缓存分三层：

- `translation_cache.jsonl`：本地 batch cache，用于完全相同 cue / timing / prompt 的精确复跑与 crash resume。
- `translation_cache.memory.jsonl`：本地 translation memory，按日文文本、目标语言、词汇表、人物参考、prompt 语义版本和模型族复用译文。
- Provider prompt cache：通过稳定的 system prompt / 全片 JSON 前缀降低 API 成本；它不等同于本地翻译缓存。

---

## 默认模型

| 用途 | 默认来源 | registry 目标文件 |
| --- | --- | --- |
| 默认 ASR | `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame` | `models/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame` |
| 低配 ASR | `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame` | `models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame` |
| SpeechBoundary-JA frozen feature | 跟随当前 ASR repo id | `models/<repo-tag>` |
| SpeechBoundary-JA scorer v4 | Mamba2 frame boundary scorer；输出 speech/split/drop_gap 三头 | `src/boundary/ja/checkpoints/speech_boundary_ja_frame_boundary_scorer_v4.<repo-tag>.pt` |
| Boundary Refiner v6 | learned `transformers.Mamba2Model` edge-only delta checkpoint | `src/boundary/checkpoints/boundary_edge_refiner_v6.<repo-tag>.pt` |
| CueQC v4 binary | learned Mamba2 binary checkpoint；ASR internals + token trace + decoder stats + structured numeric metadata -> keep/drop | `src/asr/checkpoints/cueqc_mamba_v4_binary.<repo-tag>.pt` |

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

- ASR 文本只保留用于显示和 CueQC 的规范化文本：Unicode NFKC、空白归一、换行折叠和展示安全处理。
- `ASR_CONTEXT` / `ASR_HEAD_CONTEXT` 只作为 Qwen ASR 提示词，不作为字幕后处理删除规则。
- 不使用具体词黑名单，不直接删除 `ん`、`あ`、喘息、呻吟、拟声、低信息短句或常见台词。
- 旧规则 ASR QC 已退役；字幕保留/丢弃由 CueQC v4 binary 的二元 `keep/drop` 决策负责。CueQC 模型级映射、加载或 capture 失败会终止任务；只有模型内部的单样本失败会保守 keep 并记录 fallback 原因。
- CueQC v4 不支持 `threshold_profile`、`text_bucket`、`density_level` 或 `vocalization_repetition`。主动学习训练候选必须先通过 false-drop 人工 gate，未审 drop pseudo 不能直接进入训练。
- 字幕时间轴来自 Boundary chunk；ASR 输出文本只负责显示和 CueQC，不再驱动默认时长切分。

---

## 输出与缓存

- `video/<视频名>/`：正式字幕、质量报告和人工质检报告。
- `models/`：Hugging Face 模型缓存。
- `tmp/jobs/<job_id>/`：Web / pipeline 单次任务临时目录；`JOB_TEMP_DIR` 默认是 `./tmp/jobs`。
- `tmp/chunks/`：ASR wav chunk 和 crash-resume checkpoint 的一次性运行目录。
- `tmp/cache/boundary/`：SpeechBoundary-JA frame score 到 Boundary Planner 输出的 boundary-cache v6。
- `tmp/cache/torch/`、`tmp/cache/hf/`：torch / Hugging Face 运行缓存。
- `tmp/log/`：启用运行日志后的任务日志。
- `datasets/`：本地训练、验证、测试数据归档，默认 ignored；不参与普通推理和 release 打包。
- `agents/temp/`：研究脚本、smoke、临时日志和中间产物。
- `agents/audits/`：可长期复查的本地审计页，统一从 `agents/audits/index.html` 进入；导航按更新时间倒序排列，最上面是最新需要审计的页面。该目录是本地研究产物，默认 ignored，不会随 `git push` 发布。Windows 使用 `.\tools\audits\serve_audits.ps1`，Linux / WSL2 使用 `tools/audits/serve_audits.sh`；脚本会限制 live-server watch 范围，避免训练日志 / cache 写入触发持续刷新。删除按钮也需要通过对应脚本启动才会实际移动本地审计目录并重建导航，裸 `live-server` / 直接打开 HTML 只能显示手动删除命令。

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
- `src/asr/`：ASR、Boundary 字幕时间轴分配、CueQC 和转写流程。
- `src/boundary/`：Boundary Refiner v6 checkpoint loader、edge-sequence Mamba2 adapter、core planner 和 boundary-cache v6。
- `src/boundary/ja/`：SpeechBoundary-JA scorer v4、PTM/MFCC feature cache schema、训练数据 manifest 和 frame-score 训练工具。
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

### 全链路与 Web smoke

- `tools.workflows.run_full_workflow`：从命令行跑完整本地工作流；默认不翻译、保留 ASR chunks，并把运行归档写到 `agents/temp/speech-boundary-ja/YYYYMMDD_HHMMSS_*`。
- `tools.web.smoke.start_server`：启动 FastAPI Web 服务并把 pid / stdout / stderr 写入 `agents/temp/YYYYMMDD_HHMMSS_*`。
- `tools.web.smoke.submit_job`：通过 Web API 提交分发任务；默认是不翻译 smoke，并启用 CueQC v4 binary。
- `tools.web.smoke.poll_job`：按固定间隔轮询 `/api/jobs/{id}`，默认可用于 5 分钟长任务检查。
- `tools.web.smoke.summarize_job`：读取 `tmp/web/jobs/<job_id>` 的产物，汇总阶段耗时和完整 CueQC keep/drop/fallback 统计。

示例：

```powershell
uv run python -m tools.workflows.run_full_workflow --video video/NAMH-055.mp4 --task-name 20260617_191654_cli-smoke --label smoke
uv run python -m tools.web.smoke.start_server --run-dir agents/temp/20260617_191654_web-smoke
uv run python -m tools.web.smoke.submit_job --video-path video/NAMH-055.mp4 --output-dir video --run-dir agents/temp/20260617_191654_web-smoke
uv run python -m tools.web.smoke.poll_job --job-id-file agents/temp/20260617_191654_web-smoke/job_id.txt --run-dir agents/temp/20260617_191654_web-smoke --interval-seconds 300
uv run python -m tools.web.smoke.summarize_job --job-id <job_id> --run-dir agents/temp/20260617_191654_web-smoke
```

### CueQC 训练与预测

- `tools.asr.cueqc.export_candidates`：从全链路产物导出 CueQC candidate JSONL。
- `tools.asr.cueqc.cluster_candidates`：仅用于 cold-start 的一次性 Torque 聚类和审计素材生成，不进入 runtime；训练时只接收人工确认的高精 seed，不要求簇级标签覆盖全量。
- `tools.asr.cueqc.compile_training_set`：只把显式 `seed_action=use_seed` 且 `display_decision=keep/drop` 的簇级审计标签广播为初始训练 JSONL；`mixed_skip` / `skip` 只保留为审计元数据。
- `tools.asr.cueqc.extract_features_v4_binary`：从 ASR internals 提取 CueQC v4 binary 特征。
- `tools.asr.cueqc.extract_feature_shards`：参数化分片提取大规模 CueQC 特征，替代 `agents/temp` 中的一次性硬编码脚本。
- `tools.asr.cueqc.merge_features_v4_binary`：合并分片特征 bundle。
- `tools.asr.cueqc.train_mamba_v4_binary`：训练 CueQC v4 binary checkpoint；1.7B 大特征验证/测试可用 `--eval-batch-size` 控制显存。
- `tools.asr.cueqc.predict_v4_binary`：对特征 bundle 输出 keep/drop prediction 和 high-confidence pseudo labels；checkpoint 里出现旧 `drop_threshold_profile` 会报错。
- `tools.asr.cueqc.compile_stage2a_features_v4_binary`：合并 cold-start、人工 false-drop 审计和高置信 keep pseudo，生成 Stage 2 训练 bundle；未审 drop pseudo 会被跳过。

### 审计与 Boundary

- `tools.audits.audit_nav`、`tools.audits.serve_audits.ps1`、`tools.audits.serve_audits.sh`：维护和启动本地审计导航页。
- `tools.audits.generate_cueqc_cluster_audit_html`：生成 CueQC 簇级 keep/drop 审计页，必须显式传 `--archived-root` 和一个或多个 `--media-root`；页面使用单播放器懒加载播放 chunk 与上下文。
- `tools.audits.generate_cueqc_prediction_audit_html`：生成 CueQC 预测后的 false-drop 人工审计页，用于 Stage 2 / 主动学习闭环；必须显式传 `--archived-root` 和一个或多个 `--media-root`。
- `tools.boundary.build_weighted_source_manifest`：按权重合并 anime / galgame source manifest，供 scorer/refiner 训练数据构造使用。
- `tools.boundary.build_refiner_frame_sequence_dataset`：基于 scorer v4 predicted island edge 导出 `boundary_edge_refiner_dataset_v6` 训练 JSONL；必须显式传 `--scorer-checkpoint`，row metadata 会记录 scorer schema / repo id / sha。
- `tools.boundary.train_refiner`：训练 `boundary_edge_refiner_v6` edge-only checkpoint，输出文件名默认带 repo id tag。
- `tools.boundary.export_cueqc_drop_hardcases`：把 CueQC false-drop 审计中已确认可丢弃的 chunk 导出为 SpeechBoundary-JA hard-negative 候选池；不会生成 Boundary Refiner 训练标签。
- `tools.boundary.prepare_cueqc_drop_hard_negative_sources`：把 CueQC `drop_ok` hard-negative 候选补回审计音频，并切出 SpeechBoundary-JA negative labels；不会直接启动训练。
- `tools.boundary.prepare_speech_boundary_hard_negative_replay`：校验 CueQC `drop_ok` hard-negative replay source，并混合 positive/synthetic anchor replay；该工具不生成 first-scorer 训练脚本。
- `tools.boundary.ja.build_positive_anchor_replay`：从 anime / galgame 源 manifest 按权重抽样生成 SpeechBoundary-JA positive anchor replay labels。
- `tools.boundary.ja.build_galgame_synthetic_timeline`：按 v5-style 随机时间线生成带 `speech_frames`、`cut_point_segments` 和 `cut_drop_zones` 的 SpeechBoundary-JA synthetic labels。
- `tools.boundary.ja.build_feature_cache`：为 SpeechBoundary-JA scorer 训练缓存 Qwen PTM + MFCC feature。
- `tools.boundary.ja.train_feature_scorer`：训练 runtime-loadable SpeechBoundary-JA Mamba2 scorer v4，输出三头 `speech/split_boundary/drop_gap`。
- `tools.boundary.ja.export_frame_scores`：导出 scorer v4 frame scores，用于诊断 speech island、split peak 和 drop gap 行为。
