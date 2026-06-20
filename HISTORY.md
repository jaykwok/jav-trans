# JAVTrans History

本文件记录实验过程、idea 来源、调试坑、失败路线、指标和参考来源。README 只保留新用户使用说明、当前工作流和当前状态。

公开记录统一使用匿名样片名，不写真实视频 stem。

---

## 路线总览

当前主线是 **SpeechBoundary-JA + Boundary Refiner v5 delta-only**：

```text
SpeechBoundary-JA frame scores
-> boundary candidates
-> frame-sequence Mamba2 Boundary Refiner
-> candidate_sequence_core_planner_v5
-> Qwen ASR speech-core chunks
-> CueQC v3-Fusion keep/drop routing
-> Boundary chunk subtitle timing
-> SRT / translation
```

目标已经从“传统 VAD 高 recall”修正为“字幕边界可用”：优先保证 speech core start 贴近真实语音起点，end 也要学习但允许字幕层在相邻 cue 过近时压缩前一条 end 来保 2-frame gap；chunk 数只作为成本指标，不再作为主要否决条件。

当前运行时只规划 speech core。Boundary Refiner 由后端 registry 按当前 ASR repo id 自动选择 checkpoint，`BOUNDARY_REFINER_MODEL_PATH_BY_REPO` 仅作为实验覆盖项；默认 1.7B 文件为 `src/boundary/checkpoints/boundary_refiner.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt`，低配 0.6B 文件为 `src/boundary/checkpoints/boundary_refiner.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame.pt`。checkpoint schema `boundary_refiner_v5`，head 只输出 `start_delta/end_delta`，metadata `ptm_repo_id` 必须匹配当前 ASR repo id。运行时不跨 island merge，不学习或应用 ASR padding/context budget，不暴露 refiner disable/backbone/threshold 兼容开关。

## 路线修正

- `FusionVAD-JA / VAD` -> `SpeechBoundary-JA / Boundary`：项目已经不是严格 speech/non-speech VAD，而是面向字幕切分的边界系统。
- `recall 优先` -> `start/end boundary 优先`：recall 仍要够用，但边界粗、多个 island 拼一起、fallback 20-30s 这类问题优先级更高。
- `per-gap merge 二分类` -> `frame-sequence Mamba2 start/end delta`：不再预测“合并/切开”，而是对候选边界做序列级边界修正。
- `merge + padding/context budget` -> `core-only no-padding`：no-padding A/B 证明 padding 会引入上下文污染、空转写和字幕密度问题；v5 已删除 merge 语义和 learned padding。
- `DP/Viterbi planner` -> `candidate_sequence_core_planner_v5`：DP 曾改善粗 fallback 但增加复杂度和维护成本，当前 active runtime 已删除 DP/reward planner。
- `speaker sidecar 辅助 cue merge` -> `默认不接入`：ERes2NetV2 / CAM++ 等可作为离线研究参考，但不进入当前默认 pipeline；先把 speech core 切准。
- `forced-aligner runtime / silver teacher` -> `归档`：人工 A/B 证明 Boundary Refiner speech-core chunk 时间轴优于 forced aligner span；active runtime 不再加载 forced aligner，不再挖 forced silver，不再维护 ASR 文本与 forced 时间轴匹配妥协。
- `自动指标替换默认` -> `人工观感 gate`：silver-ft01 自动指标更好但人工确认观感和准确度变差，因此不采用为默认。
- `旧规则 ASR QC` -> `CueQC v3-Fusion 二元路由`：旧规则 QC 不再运行、不再观测、不再作为候选排序或质量报告信号；最终是否丢弃由 CueQC 模型的 `keep/drop` 决策负责。CueQC 模型级映射、加载或 capture 失败会终止任务；只有模型内部的单样本 capture/feature/inference 失败才保守 keep 并记录原因。
- `单 checkpoint 路径` -> `repo-id registry + env override`：默认/推荐 Qwen ASR 改为 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`，0.6B 保留为低配置可选 backend。三个 Mamba checkpoint 都按 repo id 绑定：Boundary Refiner 与 CueQC 默认由后端 registry 跟随 `ASR_BACKEND` 自动解析；`BOUNDARY_REFINER_MODEL_PATH_BY_REPO`、`CUEQC_MODEL_PATH_BY_REPO` 仅保留为实验覆盖项；SpeechBoundary-JA scorer 空值走 repo-id 默认策略：1.7B 与 0.6B 均使用注册 scorer；`SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO` 只作为实验覆盖。旧单路径 fallback 不再保留；registry 缺失、覆盖映射格式错误、未命中当前 repo id、文件不存在或 metadata 不匹配都必须 fail-fast 并显示到日志/Web 任务错误。

## 已否决/归档路线

- FSMN / Silero / TEN / WhisperSeg / Whisper / FusionLite：不作为当前主切分路线，旧代码已清理或归档。
- BiLSTM / imitation / endpoint 小头：已断兼容删除，后续不再启用。
- F0 / gender / speaker proxy：不再作为切分、翻译提示或 synthetic metadata。
- pyannote 默认依赖：官方预训练 diarization 通常需要 HF token / 条款接受，不进入默认分发。
- CAM++ / 3D-Speaker / ERes2NetV2 默认 sidecar：不替代 Boundary；当前不接入 runtime。
- runtime merge、subtitle merge、context leak 删除规则、低信息文本删除规则、具体词黑名单：目标域误伤成本高，已从默认产物链路中移除或改为 review-only。
- learned padding/context budget、ASR padded fallback 时间轴、DP/Viterbi planner、per-gap binary merge classifier：均已被当前 v5 core-only 路线取代。
- 旧规则 ASR QC、QC 覆盖、per-sample CueQC 旧标签 schema、`content_type/qc_decision/alignment_policy` 多头标签：均已断兼容退役，后续不恢复 shadow 观测或兼容入口。
- Qwen3-ForcedAligner 默认模型、runtime 对齐路径、forced/native/hybrid timestamp mode、forced silver mining、fallback/sentinel 诊断和为 forced 字幕匹配保留的 `align_text` / 日志解析 / 审计入口：均已断兼容归档；字幕时间轴以 Boundary chunk 为准。
- silver-ft01 默认替换：已被人工 A/B 否决。
- Unified Joint Model 直接改 Qwen ASR decoder：保留为长期 backlog，不进入当前重构。

## Backlog

- Boundary Refiner v5：当前默认仍是 delta-only 32768 hardmix checkpoint，但旧 120 条边界偏好盲测和后续微调计划已归档，不再作为 active 训练路线；相关工具已移到 `agents/rm/20260618_124129_*` 归档目录。
- real-domain forced-aligner silver：已归档，不再作为 teacher、候选排序或诊断信号。后续 Boundary 训练若需要真实域监督，优先来自 synthetic true-structure、人工审计、CueQC drop replay 或新的人工标注闭环。
- CueQC v3-Fusion 二元路由：CueQC 是旧规则 QC 的替代，而不是并行 shadow。当前目标是 `Boundary切分 -> ASR文本 -> CueQC v3-Fusion -> keep/drop -> Boundary subtitle timing -> 字幕`，只保留 `display_hint=keep/drop` 这个训练和运行目标；不再打 `content_type`、`qc_decision`、`alignment_policy`，也不再输出 compact/review 作为模型监督。CueQC 输入采用 ASR encoder features、ASR token trace、decoder aggregate stats 与 structured metadata；不引入 BGE、sentence-transformers、HuBERT、UMAP/HDBSCAN/FINCH 或额外音频 embedding 模型作为默认依赖。
- CueQC bootstrap 聚类：Torque Clustering 仅用于第一次约 300 条样本的粗标签启动，读取簇级 `cueqc_cluster_labels.jsonl` 后广播为 `keep/drop` 种子；混簇和粗标签噪声可接受，后续靠全量训练修正。聚类算法不进入 runtime、不进入自训练循环、不作为长期 QC 分类器；`--method`、旧 embedding 增强、旧 HDBSCAN/FINCH/UMAP/PCA 参数和 per-sample `cueqc_manual_labels.jsonl` 均不再保留兼容。
- CueQC 阶段 2 自训练：第一轮 Stage 2b 已完成并替换默认 checkpoint，当前采用 base `drop_threshold=0.85` + `short_text=0.87` 的保守自适应阈值。NAMH-055 Web 全链路 smoke 已完成；剩余验证是更多全片人工抽检、以及后续按相同 false-drop gate 继续扩充训练集。核心安全指标仍是 keep recall / false drop rate，宁可欠删，也不能误删有效字幕。
- CueQC 阶段 3 SpeechBoundary 反哺：三轮 CueQC false-drop 审计中的人工 `drop_ok` 都是直接丢弃监督，包括呻吟声、呼吸声、环境声、短碎片、过碎无效 island 等；分类标签只保留为审计元数据，不再影响 route。因这些 hard negative 来自旧启发式 chunk，已废弃为 first scorer bootstrap 输入，只保留为后续自训练/finetune replay material。当前 first scorer 断兼容路线是从 synthetic true-structure timeline 重新训练 `speech_boundary_ja_mamba2_frame_boundary_scorer_v3`：Mamba2 输出 `[speech_prob, cut_prob]`，`speech_frames` 监督 speech head，`cut_point_segments` + `cut_drop_zones` 监督 cut/drop gate。旧 speech-only v2、`neg20-s800` 和 hard-negative-first 训练路径不再作为 active 候选；任何 scorer checkpoint 只通过 `SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO` 按当前 PTM repo id 显式启用，完整 smoke + 人工审计通过前不替换默认 scorer。
- ASR dense feedback：旧规则 QC 不再提供 dense feedback。若后续需要 reference-free 质量信号，只从 CueQC v3-Fusion 的 token trace / decoder stats / ASR internals 继续扩展，不恢复 `evaluate_asr_chunk_qc`。
- 直接字幕边界 / timeline model：当前 active runtime 已采用 Boundary chunk 作为字幕时间轴；未来研究重点是让 Boundary first scorer / refiner 更稳定地产生一句台词一行的 speech-core chunk，而不是恢复外部对齐器。
- Unified Joint Model：等 Boundary Refiner 输出稳定 pseudo boundary label 后，再评估 Qwen ASR boundary-token SFT / DPO / RL 或长上下文 joint segmentation-transcription。
- 小模型蒸馏：在质量稳定后，再考虑把 SpeechBoundary-JA / Boundary Refiner 蒸馏成更小的分发模型。

## 当前结论

- 2026-06-18 repo-id checkpoint 绑定决策：保留 0.6B 作为低配置 ASR backend，但默认/推荐 ASR 改为 1.7B；所有 Mamba checkpoint 不再用模型维度或文件名推导归属，必须以 checkpoint metadata 的 repo id + repo-id map 显式绑定。`BOUNDARY_REFINER_MODEL_PATH_BY_REPO` 与 `CUEQC_MODEL_PATH_BY_REPO` 是默认链路必填项，`SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO` 是 learned scorer 的 opt-in 必填 map；`SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT` 旧单路径 env 会直接报错。训练入口也要求 repo id 元数据：SpeechBoundary scorer feature manifest 必须有唯一 `ptm`，Boundary Refiner v5 rows 必须有唯一 `metadata.ptm_repo_id`，CueQC feature bundle 必须有 `feature_config.asr_model_id`；运行时加载时再次校验 SpeechBoundary scorer / Boundary Refiner 的 `metadata.ptm_repo_id` 和 CueQC 的 `metadata.asr_repo_id`。文件名只作为默认可读 tag，可用 `--checkpoint-name` 自定义，不参与归属判断。当前已提交的 0.6B Boundary Refiner / CueQC checkpoint 已改名为 `boundary_refiner.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame.pt` 与 `cueqc_mamba_v3_fusion.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame.pt`，并已补写 0.6B repo-id metadata；1.7B 默认链路需要重训 scorer、Boundary Refiner 和 CueQC，并把三个 repo-id map 切到 1.7B checkpoint 后再运行。
- 2026-06-19 0.6B SpeechBoundary-JA scorer 候选固定：正式作为 0.6B scorer 候选的是 `agents/temp/20260618_221851_speech-boundary-frame-boundary-scorer-v3-negative-rich8192-dil06-prep/frame-boundary-scorer-v3/speech_boundary_ja_feature_scorer.pt`，因为它是当前 v3 schema `speech_boundary_ja_mamba2_frame_boundary_scorer_v3`、model type `mamba2_frame_boundary_scorer`、feature manifest `ptm=jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame` 的最新 speech+cut 双头训练产物；两个 v2 scorer 已是废弃 schema，不作为绑定候选。该文件已复制为 `src/boundary/ja/checkpoints/speech_boundary_ja_feature_scorer.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame.pt`，补写 `metadata.ptm_repo_id=jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`，sha256 `79cdab90a980e9e77fe68a5e0b58e0f0ec555f12cc247ad4dcf62f5f6afc2a1e`，大小 `2477377` bytes。经 NAMH-055 A/B 审计确认 0.6B scorer 观感优于 bootstrap 后，该 checkpoint 晋升为 0.6B 默认 scorer；1.7B scorer 继续按当前 1.7B feature cache 重训，未过 gate 前不进入 1.7B 默认。
- 2026-06-19 1.7B scorer 评估内存修正：`prepare_frame_boundary_scorer_v3` 过去把 eval device 复用为训练 device，导致 1.7B `input_dim=2088` 的 Mamba2 threshold eval 生成 `--device cuda --batch-size 8`，离线评估会占满独显并进入共享显存，速度大幅下降。评估不是 runtime，不需要抢 GPU；已将 prep 的 eval 默认改为 `eval_device=cpu`、`eval_batch_size=1`，并给当前 1.7B prep 的 `evaluate_frame_boundary_scorer_v3.ps1` 做同样修正。若本机 GPU 空闲，1.7B 离线评估可显式用 `--device cuda --batch-size 1`，但不要再用大 batch。训练和 feature cache 仍继续使用 CUDA。
- 2026-06-19 scorer boundary-aware gate 正式化：`tools.boundary.ja.evaluate_feature_scorer_thresholds` 的诊断输出已升级为 `recommendation.boundary_aware_runtime_profile`，候选 profile 必须带 `--diagnostic-runtime-profile` 才会按核心 island、predicted island precision、cut-drop zone clean、far-region recall/FPR 生成正式通过/失败原因。用 1.7B recall-first scorer 跑 128-row CUDA smoke（`agents/temp/20260619_054706_scorer17b-boundary-aware-eval-limit128/`）验证 CLI/JSON/Markdown 输出：`boundary_aware_runtime_profile_status=no_profile_passes_policy`，最佳边界感知候选 `on=0.6/off=0.4/cut=1.0` 的 `speech_island_recall=1.0`，但失败于 `cut_drop_zone_clean_rate`、`mean_cut_drop_zone_predicted_ratio` 和 `far_region_false_positive_rate`；这说明已有 1.7B scorer 的 core speech 保留不是主要瓶颈，gap/drop 不误连和数据/label 设计才是下一步重点。
- 2026-06-19 1.7B Mamba 重训路线补充：scorer 只是 opt-in first scorer，不能替代默认链路所需的 Boundary Refiner 与 CueQC。1.7B 默认 ASR 链路要正式可用，必须产出并提交三类 repo-id 绑定 checkpoint：`speech_boundary_ja_feature_scorer.*1.7B*.pt` 写 `metadata.ptm_repo_id` 且只通过 `SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT_BY_REPO` opt-in；`boundary_refiner.*1.7B*.pt` 写 `metadata.ptm_repo_id`，通过 `BOUNDARY_REFINER_MODEL_PATH_BY_REPO` 命中当前 `ASR_BACKEND` 后加载；`cueqc_mamba_v3_fusion.*1.7B*.pt` 写 `metadata.asr_repo_id` 且 `feature_config.asr_model_id` 一致，通过 `CUEQC_MODEL_PATH_BY_REPO` 命中当前 `ASR_BACKEND` 后加载。文件名只保留 repo tag 便于人工识别，不作为归属判断；映射缺失、未命中、文件不存在或 metadata 不匹配都必须 fail-fast 到日志/Web 任务错误。下一步执行顺序是：先把 scorer 的 boundary-aware/island-level gate 正式化并复评已有 1.7B candidate，再生成 1.7B Boundary Refiner v5 frame-sequence dataset 并训练 refiner，随后用 1.7B ASR internals 重新抽取 CueQC v3-Fusion features、复用既有人工 keep/drop gate 训练 1.7B CueQC；三者齐备后才切 1.7B repo-id maps 做 NAMH-055 smoke 与多片人工抽检。
- 2026-06-19 1.7B Boundary Refiner / CueQC 重训落地：Boundary Refiner 使用 `agents/temp/20260619_104428_boundary-refiner-v5-qwen17b-full/frame_sequence_dataset_v5.qwen17b.full.jsonl`（8192 sequences / 17465 supervised items）训练，输出 `src/boundary/checkpoints/boundary_refiner.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt`，metadata `ptm_repo_id=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`、`runtime_adapter=frame_sequence_v1`、feature hash `cb44d7804ad4eaec9e5e4db80123f817a99650fa`，loader smoke 通过，val start/end MAE 约 `0.0236/0.0249s`。CueQC 复用 Stage2b 的 2177 条 keep/drop 监督重建训练 JSONL，使用本地 1.7B ASR 抽取 internals，输出 `src/asr/checkpoints/cueqc_mamba_v3_fusion.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt`，metadata `asr_repo_id=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame` 且 `feature_config.asr_model_id` 一致，`asr_dim=2048`，loader expected 1.7B 通过、expected 0.6B fail-fast。后端 registry 已登记 Boundary Refiner / CueQC 的 1.7B 与 0.6B checkpoint，`.env` 不再持久化这两个 map；1.7B scorer 仍因 boundary-aware gate 未过而不进入默认启用；0.6B scorer 已进入默认 registry。CueQC 1.7B 固定测试片 `867HTTM-0045` 指标 `accuracy=0.8039, keep_recall=0.9767, false_drop_rate=0.0233`，NAMH-055 no-translate clean smoke 已完成，仍需多片人工抽检后再判断质量是否可接受。
- 2026-06-19 0.6B scorer A/B 审计与 `.env` 清理：已用 0.6B scorer v3 opt-in 产物跑 NAMH-055 no-translate，输出 `agents/temp/speech-boundary-ja/20260619_113807_namh055-qwen06b-scorer-v3-audit/`，结果 `transcript_chunks=1780, segments=1738, blocks=1730`；A/B 审计页生成到 `agents/audits/20260619_115017_namh055-qwen06b-scorer-v3-ab/index.html`，对比基线 `agents/temp/speech-boundary-ja/20260617_cueqc-v3-namh055-smoke2/`；人工审计结论为 0.6B scorer 观感更好，已晋升为 0.6B repo 默认 scorer。此前混配 1.7B smoke（ASR/PTM 为 1.7B 但 boundary model path 仍是 0.6B）已按临时产物规则移到 `agents/rm/20260619_113706_bad-qwen17b-mamba-smoke-mixed-boundary-model/`。`.env` 现在只用 `ASR_BACKEND` 作为模型族选择；SpeechBoundary-JA PTM/model path、Boundary Refiner checkpoint 和 CueQC checkpoint 都由后端按 repo id 自动解析，env map 仅作为实验覆盖。
- 2026-06-19 1.7B clean smoke：使用清理后的 1.7B 配置跑 NAMH-055 no-translate clean smoke，输出 `agents/temp/speech-boundary-ja/20260619_115218_namh055-qwen17b-mamba-smoke-clean/`，状态 done，elapsed `643.288s`，`transcript_chunks=2110, segments=1860, blocks=1828`。summary 确认 `asr_backend`、`boundary_signature.ptm` 和 `boundary_signature.model_path` 均为 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame` / `models/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame`，Boundary Refiner 实际加载 `src/boundary/checkpoints/boundary_refiner.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt`，scorer 为空继续 bootstrap。
- 2026-06-19 1.7B Boundary Refiner 升级为 delta-only 32768 hardmix：确认用户所指 32768 是 2026-06-11 的 true v5 delta-only hardmix checkpoint 路线，不是 2026-06-18 scorer negative-rich8192 数据。新训练目录为 `agents/temp/20260619_130400_boundary-v5-qwen17b-delta-only-32768-hardmix/`：weighted source manifest `196608` 条，组成为 `anime_nsfw=88474`、`anime_sfw=58982`、`galgame=49152`；synthetic hardmix `32768` 条，`cut_point_segment_count=85187`、`cut_drop_zone_count=78653`、`overlap_mix_count=6511`、gap modes `fade_noise=59009/hum=26115/silence=58893/white_noise=52399`，与 6 月 11 true v5 hardmix 分布基本对齐。1.7B CUDA/bf16/no-compress feature cache 输出 `feature-cache-qwen17b/`，`cached=32768/errors=0/skipped=0`，metadata `ptm=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`。frame sequence v5 dataset `frame_sequence_dataset_v5.qwen17b.32768.jsonl` 输出 `32768` sequences、`169156` supervised items、feature hash `cb44d7804ad4eaec9e5e4db80123f817a99650fa`、row metadata `ptm_repo_id=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`。训练从零开始，CUDA `600` steps、batch `512`、lr `5e-4`、weight decay `0.01`、hidden/layers/state `128/2/32`、start/end loss weights `1.0/0.6`；输出 `train-mamba2-v5-qwen17b-32768/boundary_refiner.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt`，已替换默认 `src/boundary/checkpoints/boundary_refiner.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt`，sha256 `3dca4e3404ccafdeeef18206c18cbf3f1acfd7c93c27840d0573476dc31c2589`，大小 `2278413` bytes，schema `boundary_refiner_v5`、runtime adapter `frame_sequence_v1`、output dim `2`、repo metadata 匹配 1.7B。训练指标：last loss `0.00000194`，train start/end MAE `0.000939/0.001071s`，val start/end MAE `0.000954/0.001093s`。聚焦回归 `tests/test_boundary_refiner.py tests/test_boundary_refiner_training.py tests/test_boundary_ja_current.py tests/test_asr_backend_dispatch.py` 为 `59 passed`。此前 1.7B 8192/100-step Boundary checkpoint 已被该 32768 版 supersede。NAMH-055 no-translate clean smoke 已完成，输出 `agents/temp/speech-boundary-ja/20260619_173000_namh055-qwen17b-boundary32768-smoke/`，状态 done，elapsed `704.48s`，`transcript_chunks=2087, segments=1842, blocks=1807`，ASR+Boundary subtitle timing `617.1s`，pipeline total `689.4s`；summary 确认实际加载默认 1.7B Boundary Refiner `src/boundary/checkpoints/boundary_refiner.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt`，checkpoint sha1 `734a50bca8eaa6f7ea11d39a5baead49ea50494a`，metadata `ptm_repo_id=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`，scorer checkpoint 为空继续 bootstrap。质量报告：subtitle duration p50/p90/p95/max `1.675/2.803/3.063/8.89s`，short/micro/long `101/15/6`，overlap `0`，ASR generation errors `0`，per-minute subtitle count `20.09` 仍偏高；是否优于 8192 版需要人工观感 gate。
- 2026-06-19 1.7B recall-first scorer 训练/评估结论：复用 `agents/temp/20260619_010008_speech-boundary-frame-boundary-scorer-v3-qwen17b-negative-rich8192-dil06-prep/feature-cache/feature_manifest.jsonl`，按 `positive=2.0/negative=1.0/cut_positive=6.0/cut_negative=1.0/cut_loss=1.0/focal_gamma=2.0/max_steps=4000` 训练 1.7B candidate。首次沿用长 prep 路径 + full repo-id 文件名时，4000 step 训练完成后 `torch.save` 因 Windows path too long 失败；已给 `train_feature_scorer` 增加 `--log-every`，底层训练增加进度输出，并在 Windows 下对过长 checkpoint path fail-fast。有效重训输出为 `agents/temp/20260619_024948_scorer17b-recallfirst/ckpt/speech_boundary_ja_feature_scorer.recallfirst.qwen17b.pt`，metadata `ptm_repo_id=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`，sha256 `16aad81dc406765f8abd9ed1520a3024c519515f854ac636b21e66d0e304e97b`。训练期指标：speech precision/recall/F1 `0.982413/0.991384/0.986878`，cut `0.936193/0.983557/0.959291`。全量 `8192` threshold eval 使用 `cuda --batch-size 1 --log-every 256`，显存约 `7.8GB/8GB`、未 OOM，耗时 `1128.9s`；输出 `agents/temp/20260619_024948_scorer17b-recallfirst/eval-gpu-bs1-full/summary.md`。结果仍未过旧 strict frame policy：selected speech `threshold=0.6 recall=0.989314/FPR=0.011696/F1=0.990541`，selected cut `threshold=0.6 recall=0.975428/FPR=0.007861/F1=0.974029`，selected runtime profile `on=0.65/off=0.6/cut=1.0 recall=0.987938/FPR=0.009682/F1=0.990546`；可达 `recall>=0.995` 的 runtime profile 仍有 `FPR=0.030-0.038`，因此不替换默认 runtime、不进入 repo-id map。下一步不要继续盲调单一 loss，应把 boundary-aware/island-level gate 正式化，或重新设计数据/label 使 recall/FPR tradeoff 在真实 workflow 中可审计。
- 2026-06-20 1.7B scorer 32768 hardmix balanced 候选结论：复用 `agents/temp/20260619_130400_boundary-v5-qwen17b-delta-only-32768-hardmix/synthetic/labels.jsonl` 与 `feature-cache-qwen17b/feature_manifest.jsonl`，按 `positive=2.0/negative=2.5/cut_positive=8.0/cut_negative=1.0/cut_loss=1.5/focal_gamma=2.0/max_steps=3000` 训练输出 `agents/temp/20260619_174743_scorer17b-boundary32768-balanced/ckpt/speech_boundary_ja_feature_scorer.boundary32768-balanced.qwen17b.pt`，训练期 speech precision/recall/F1 `0.997415/0.991064/0.994229`，cut precision/recall/F1 `0.926050/0.985802/0.954992`。1024-row boundary-aware eval 输出 `agents/temp/20260620_004000_scorer17b-boundary32768-balanced-eval-gpu-limit1024-emptycache/`：frame-level speech policy 通过，selected `threshold=0.4 recall=0.995270/FPR=0.019781`；cut policy 通过，selected `threshold=0.7 recall=0.965291/FPR=0.003928`；runtime profile 仍未通过，最佳 `on=0.65/off=0.5/cut=0.9 recall=0.985499/FPR=0.003617`；boundary-aware profile 仍未通过，失败项为 `speech_island_recall=0.993665 < 0.995` 与 `mean_label_island_coverage=0.965370 < 0.985`，但 cut-drop clean `0.939606`、far-region recall/FPR `0.995257/0.001187` 已达标。因此该 1.7B scorer 仍保持 opt-in，不进入默认 map；下一步应转向 label/island 设计或真实 workflow 聚类标注闭环，而不是继续单纯调 loss。评估工具新增 CUDA 进度日志（batch size、max frames、padded feature MB、allocated/reserved/free/total MB）与 `--cuda-empty-cache-every`；同一 1024 eval 在 `cuda --batch-size 1` 下若不清 cache 耗时约 `1216s` 且 `cuda_reserved_mb` 可超过 `8GB` 进入共享显存，启用 `--cuda-empty-cache-every 1` 后耗时约 `250s`，`cuda_reserved_mb` 稳定约 `24MB`。
- 2026-06-20 CueQC 1.7B 聚类标注原型：基于最新 1.7B Boundary32768 NAMH-055 smoke 的 `archived/NAMH-055/NAMH-055.aligned_segments.json` 导出 `2087` 条 ASR 后候选到 `agents/temp/20260620_005500_cueqc-qwen17b-boundary32768-namh055-cluster-prototype/cueqc_candidates.jsonl`。Torque structured 默认 gap-cut 只得到 `2` 个簇 + noise，粒度过粗；`--merge-layer 0` 得到 `575` 个簇，粒度过细；`--merge-layer 1` 得到 `53` 个簇，输出 `cluster_structured_layer1/cluster_audit.html`、`cueqc_clusters.jsonl`、`cueqc_cluster_representatives.jsonl` 与 `summary.json`，更适合作为簇级 keep/drop 审计原型。该原型只用于验证标注页面和簇粒度，不作为最终 CueQC 训练集；正式 1.7B CueQC 重训仍需要先跑多片 1.7B no-translate/keep-ASR-chunks 输出，再用同一 `merge-layer 1` 口径聚类给用户打簇标签。
- 2026-06-20 1.7B scorer 进入默认 workflow：虽然 32768 hardmix balanced scorer 的 formal boundary-aware gate 仍有 `speech_island_recall/mean_label_island_coverage` 两项略低，用户确认当前指标已足够进入 workflow。该 checkpoint 已复制为 `src/boundary/ja/checkpoints/speech_boundary_ja_feature_scorer.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt` 并登记到 `DEFAULT_SPEECH_BOUNDARY_SCORER_CHECKPOINT_BY_REPO`，metadata `ptm_repo_id=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame` 与当前 ASR repo id 匹配。1.7B 默认链路现在是 repo-id 自动解析的 scorer + Boundary Refiner 32768 + CueQC 1.7B；后续优化不再阻塞 workflow，重点转为用多片 1.7B ASR 后输出聚类打簇标签，并把 REAL-988 留作最终测试片。
- 2026-06-20 scorer eval batch-size 复核：使用已落地的 1.7B scorer checkpoint 做 8-row CUDA smoke，`--cuda-empty-cache-every 1` 下 batch size 不是越大越快。`bs=1` 耗时约 `3.1s`、`cuda_max_reserved_mb=3568`；`bs=2` 耗时约 `31.9s`、`cuda_max_reserved_mb=7118`，已贴近 8GB 独显上限；`bs=4` 耗时约 `38.8s`、`cuda_max_reserved_mb=14216`，会进入共享显存。结论：1.7B scorer 离线 eval 保持 `--batch-size 1 --cuda-empty-cache-every 1` 最稳，最多只在短样本且显存空闲时试 `bs=2`；不要用 `bs>=4`。
- 2026-06-20 CueQC 1.7B 10-film cold-start 主动学习闭环：读取用户簇级审计 `agents/audits/20260620_025639_cueqc-qwen17b-10film-cluster-label-audit/cueqc_cluster_labels.jsonl`，共 `104` 个簇，`seed_action` 为 `use_seed=85 / mixed_skip=19`，display 标注为 `drop=80 / keep=5 / empty=19`。训练集编译已收紧为只广播 `seed_action=use_seed` 且 `display_decision=keep/drop` 且 `training_label_included` 不是 `False` 的簇；`mixed_skip/skip` 即使误填 display_decision 也不进训练。编译输出 `agents/temp/20260620_211746_cueqc-qwen17b-cluster-seed-training/`，得到 `3399` 条 seed 监督（`drop=3321 / keep=78`），跳过 `758` 条未标/混簇候选。1.7B ASR internals 特征抽取输出 `cueqc_train_features_v3_fusion.pt`，`asr_dim=2048`、`skipped=0`；keep-heavy seed checkpoint `seed_model_keepw25/cueqc_mamba_v3_fusion.cluster-seed.keepw25.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt` 在默认 `drop_threshold=0.85` 下不可作为 runtime（几乎全 keep），但校准后 `drop_threshold=0.4` 可用于主动学习候选筛选。全量 10-film 候选池 `19552` 条已分片抽取并合并为 `full_candidate_features/cueqc_full_candidates_features_v3_fusion.pt`，预测输出 `seed_predictions_t040/`：`drop=11159 / keep=8393`，high-confidence pseudo `16065` 条。因该 seed 模型概率尺度压缩到 `p_drop=0.400147..0.434565`，`tools.asr.cueqc.predict_v3_fusion` 新增 `--allow-low-drop-threshold`，默认仍拒绝低于 `0.5` 的 drop 阈值；低阈值只允许离线主动学习审计，不进入 runtime/default。false-drop 抽审页已生成 `agents/audits/20260620_232325_cueqc-qwen17b-seed-t040-false-drop-audit/index.html`，200 条 mixed drop 样本，采样阈值 `min=0.4/high=0.43/near_margin=0.03`。下一步需等人工导出 `cueqc_false_drop_audit_labels.jsonl` 后再编译 Stage 2；REAL-988 继续作为最终测试片保留，不进入当前 seed/prediction 池。
- 2026-06-21 CueQC prediction 审计页播放修复：`20260620_232325_cueqc-qwen17b-seed-t040-false-drop-audit` 的 `audio_url` 解析为 `/agents/temp/...wav` 且本地 HTTP HEAD 返回 `200 audio/wav`，媒体源路径本身可达；实际问题是 prediction false-drop 审计页沿用旧播放器逻辑，切换大 wav 后立即 `currentTime/play()`，没有等待 `loadedmetadata`，表现为点击播放无响应。已把 `tools.audits.generate_cueqc_prediction_audit_html` 改为单播放器懒加载 `<source>`，补 `mediaLink`、`mediaError`、`loadedmetadata` 后 seek/play、播放失败提示，并覆盖重生成当前审计页；cluster 审计页原本已有类似保护逻辑。
- 2026-06-18 force aligner 断兼容结论：NAMH-055 forced vs Boundary A/B 人工审计确认 Boundary Refiner speech-core chunk 时间轴更自然；force aligner 成功样本也系统性向内裁剪 chunk，排除首条异常后的 `905` 个 forced-success chunk 中，`delta_start = forced_start - chunk_start` mean/p50/p90/p95 为 `+0.440/+0.160/+1.280/+1.600s`，`delta_end = forced_end - chunk_end` 为 `-0.871/-0.640/-0.068/-0.020s`，duration delta p50 `-1.140s`。因此 active runtime、默认配置、Web 模型需求、Windows packaging 和主动诊断工具全部移除 forced aligner；ASR 只做 text-only，字幕时间轴由 Boundary chunk 等比分配生成。旧 forced 代码、silver mining、fallback/sentinel 诊断、forced 对比审计生成器和相关测试已归档到 `agents/rm/20260618_191036_forced-aligner-retired-code/`；历史 ASR diagnostics 工具已归档到 `agents/rm/20260618_193224_retired-asr-diagnostics-tools/`；为 forced 匹配保留的 `align_text` 双文本策略、旧日志解析和 batch/env 参数也不再保留。
- 2026-06-10 断兼容审计结论：active runtime 已升级为 **Boundary Refiner v5 core-only**。当前链路是 `SpeechBoundary-JA frame scores -> boundary candidates -> frame-sequence Mamba2 refiner -> candidate_sequence_core_planner_v5 -> ASR speech-core chunks`。运行时不再使用 DP/Viterbi planner、不再使用 learned padding/context budget，也不再保留 gap-only/per-gap refiner loader、heuristic fallback 或任何 merge decision。
- Boundary Refiner v5 checkpoint 通过 `BOUNDARY_REFINER_MODEL_PATH_BY_REPO` 固定到当前 ASR repo id；已提交的 0.6B 文件为 `src/boundary/checkpoints/boundary_refiner.jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame.pt`。checkpoint schema 必须是 `boundary_refiner_v5`、`output_dim=2`、`metadata.runtime_adapter=frame_sequence_v1`、`metadata.ptm_repo_id` 匹配当前 ASR repo id。v5 head 只输出 `start_delta/end_delta`；`BOUNDARY_REFINER_THRESHOLD`、`BOUNDARY_REFINER_ENABLED`、`BOUNDARY_REFINER_BACKBONE`、`sequence_labels`、`merge_positive`、`split_negative`、`boundary_merge_prob`、`boundary_split_prob` 和 `boundary_decision_merge` 均已从 active code / config 移除。pipeline 必须加载 canonical `transformers.Mamba2Model` v5 checkpoint；`SPEECH_BOUNDARY_JA_EXPORT_SEQUENCE_FEATURES` 仅是 pipeline/exporter 内部特征导出通道，不进入 boundary-cache signature。
- 2026-06-11 默认 checkpoint 决策更新：人工 A/B 审计确认 `32768 hardmix` 的字幕观感和转写准确度优于 `4096`，因此默认继续使用 32768 hardmix 规模；但默认文件已从“v4 裁头迁移”改为 true v5 delta-only 重新训练 checkpoint。旧默认备份到 `agents/temp/speech-boundary-ja/default-checkpoint-backups/boundary_refiner.before-true-v5-delta-only-32768.pt`，旧 sha256 `39876ce32738f685112fbfa521c2210b0286ce7fafb59c1e9f1922250241059b`；当前 `src/boundary/checkpoints/boundary_refiner.pt` sha256 `503d7e2299460aff555e02cba2b840c59195e577719bd0637a5ae98657ef919f`，大小 `2275245` bytes。true v5 数据与 checkpoint 不再包含 merge head / merge label / merge weight / split label / split weight，sequence feature 名继续使用 `gap_reference_s`，feature schema hash 为 `cb44d7804ad4eaec9e5e4db80123f817a99650fa`。checkpoint payload 同时在顶层和 `metadata` 镜像 `feature_schema` / `feature_schema_hash` / `feature_signature`，loader 会拒绝两者不一致的文件，避免审计脚本误判。
- active tree 审计结果：`BOUNDARY_DP_*`、`BOUNDARY_PLANNER_START_WEIGHT`、`BOUNDARY_CONTEXT_*`、`BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S`、`speech_left_padding_s/right_padding_s`、`sequence_context_targets`、`build_refiner_gap_dataset.py`、`load_boundary_refiner()`、`RefinerInput`、`HeuristicBoundaryRefiner`、merge score/label/threshold 均已从当前代码/测试/配置入口移除。保留的 `start_weighted_*` 是评测指标，不是 Boundary Refiner 训练标签。
- 2026-06-10 二次清理：旧 `tools/boundary/ja/plan_reward_boundary_segments.py` / `tests/test_reward_boundary_planner.py` 已归档到 `agents/rm/dp-planner-removal/`，active `tools/` 不再保留 DP/reward planner 入口；ASR 侧旧 `merged_from` 消费分支和未使用的 chunk-packer 参数也已删除，避免后续误判仍存在 runtime merge。验证：`compileall` 通过，`pytest -q` 为 `459 passed`。
- 当前路线取舍：保留 `frame-sequence Mamba2 start/end delta -> core planner -> ASR` 主线；RL/DPO 和 preliminary-ASR semantic feedback 只作为下一阶段 backlog，等 supervised true v5 与人工审计稳定后再接入。已否掉或不再进入 active tree 的路线包括 per-gap merge 二分类、runtime merge、DP/Viterbi planner、learned padding/context budget、speaker sidecar 默认链路、silver-ft01 默认替换和 Unified Joint Model 直接改 ASR decoder。Unified Joint Model 仍可在未来用稳定 Boundary Refiner 输出做 pseudo boundary label 后再评估。

## 近期实验记录

- 2026-06-18 SpeechBoundary-JA first scorer 路线断兼容重置：`neg20-s800` NAMH-055 对比显示 speech-only scorer 会把长 speech island 交给约 3 秒机械切分，开头“中出し”被拆成多块并诱发 ASR 误识别，因此旧 `speech_boundary_ja_mamba2_frame_scorer_v2`、`mamba2_frame_scorer` runtime model type、单通道输出和 hard-negative-first 训练路线全部废弃。active code 升级为 `speech_boundary_ja_mamba2_frame_boundary_scorer_v3`：checkpoint 必须是 schema v3、model type `mamba2_frame_boundary_scorer`、`output_dim=2`，训练/评估张量为 `[speech, cut]` 双通道，loader 对旧 schema/model type/缺失或错误 `output_dim` fail-fast。`tools.boundary.ja.train_feature_scorer` 和 `evaluate_feature_scorer_thresholds` 改为 v3 speech/cut 双指标；`tools.boundary.prepare_speech_boundary_hard_negative_finetune` 重命名为 `prepare_speech_boundary_hard_negative_replay`，不再生成 scorer 训练脚本。当前下一步是用 synthetic true-structure timeline 重建 first-scorer bootstrap 数据，再缓存 Qwen PTM + MFCC features 并训练 v3；CueQC `drop_ok` hard negatives 只保留为后续 replay material。
- 2026-06-18 SpeechBoundary-JA frame boundary scorer v3 synthetic8192 训练完成：先提交断兼容重构 `f357a1e Refactor SpeechBoundary scorer v3`，随后按 `anime_nsfw=55/anime_sfw=20/galgame=25` 抽样 30000 条 source manifest 到 `agents/temp/20260618_144022_speech-boundary-v3-mixed-source-manifest/`，生成 synthetic true-structure timeline `agents/temp/20260618_144022_speech-boundary-frame-boundary-v3-synthetic8192/`。数据为 `8192` records、总时长 `133615.697s`、`speech_frame_ratio=0.819862`、`cut_point_segment_count=8114`、`cut_drop_zone_count=7842`、`overlap_mix_count=1230`、`skipped=0`。Qwen PTM + MFCC full-dim feature cache 输出到 `agents/temp/20260618_144022_speech-boundary-frame-boundary-scorer-v3-prep/feature-cache/`，`cached=8192/errors=0/skipped=0`，`ptm_dim=1024/mfcc_dim=40`。三版训练对比：旧保守权重 `positive=1/negative=15/max_steps=1000` 的 sweep speech `threshold=0.25 recall=0.988853/FPR=0.038906/F1=0.990139`，cut `threshold=0.5 recall=0.923106/FPR=0.023875/F1=0.869518`；探索版 `positive=2/negative=10/cut_positive=6/max_steps=3000` 的 speech `threshold=0.3 recall=0.990402/FPR=0.025671/F1=0.992366`，cut `threshold=0.7 recall=0.940312/FPR=0.010310/F1=0.927905`；按 Grok 搜索结论调整的 recall-first 版 `positive=4/negative=1/cut_positive=8/max_steps=3000` 训练期 speech recall 提升到 `0.996610`，sweep speech `threshold=0.6 recall=0.992281/FPR=0.038477/F1=0.991917`，cut `threshold=0.7 recall=0.947447/FPR=0.014012/F1=0.917663`。Grok 结论是：BCE/focal 的权重起点可参考 `negative/positive` class ratio，但业务代价会覆盖频率平衡；`pos_weight > 1` 偏 recall，`< 1` 偏 precision，最终 threshold 必须用验证集 sweep 校准而不是固定 `0.5`。当前三版均未过旧 speech policy `recall>=0.995/FPR<=0.02`，cut head 已能过；下一步不要替换默认 runtime，应重新校准双头 scorer 的 speech policy，或调整 synthetic negative/gap 分布与 loss，而不是把 0.5 当默认阈值。
- 2026-06-18 SpeechBoundary-JA v3 threshold 策略落地：Grok 复核 SpeechBrain / neural VAD 常见做法后，决定不把单点 `0.5` 当运行时默认，而是给 speech head 增加 activation/deactivation hysteresis：`SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD` 负责进入 speech，`SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD` 负责退出 speech，cut head 继续用 `SPEECH_BOUNDARY_JA_CUT_THRESHOLD` 做 gate。代码默认兼容旧运行效果：未显式设置 on/off 时两者都等于旧 `SPEECH_BOUNDARY_JA_THRESHOLD`；v3 Grok candidate 的下一次 opt-in smoke 建议使用 `speech_on=0.7 / speech_off=0.5 / cut=0.7`，但默认 runtime 仍不替换。boundary cache signature、`tools.workflows.run_full_workflow` CLI/env/summary 和测试已同步。CueQC 模型不可用路径也同步清理为纯 `fallback_keep`，不再用文本长度、重复度等启发式给 fallback confidence，避免误判旧规则 QC 仍在运行。
- 2026-06-18 SpeechBoundary-JA v3 runtime-profile 校准结论：force-aligner runtime 退役已提交为 `56f6890 Retire forced aligner workflow`。继续 Mamba 优化时，`evaluate_feature_scorer_thresholds` 增加 runtime profile 评估，并把 hysteresis 评估改为区间扫描、Mamba 推理改为 batch + 长度分桶；`prepare_frame_boundary_scorer_v3` 也参数化 speech/cut loss weights、focal gamma、eval batch size 和 runtime profiles，避免靠手改临时 ps1。已有全量 12-profile 评估 `agents/temp/20260618_200421_speech-boundary-v3-runtime-profile-eval/` 仍无 profile 通过旧 policy，最佳 `on=0.7/off=0.5/cut=0.8` 为 `recall=0.981691/FPR=0.027299/F1=0.987771`；补充 1024-row 宽扫 `agents/temp/20260618_204217_speech-boundary-v3-runtime-profile-wide-eval-limit1024-bucketed/` 中，即使近似禁用 cut gate 的 `on=0.65/off=0.6/cut=1.0` 也只有 `recall=0.991086/FPR=0.031271`。结论：当前 scorer 不能靠阈值/门控直接 promoted；根因更像 bootstrap 数据分布，`20260618_144022` synthetic8192 的 `speech_frame_ratio=0.819862` 且 `negative_rows/background_rows=0`，非语音帧太少且过于合成。下一步改为 negative-rich synthetic true-structure 重建：降低 speech ratio、加长 leading/trailing/internal gap、降低 speech label dilation，先不把 CueQC hard negatives 纳入 first scorer bootstrap，待新 scorer 产出后再用人工审计的 drop replay 做后续 finetune。
- 2026-06-18 SpeechBoundary-JA v3 negative-rich 训练复盘：`agents/temp/20260618_205057_speech-boundary-frame-boundary-v3-negative-rich-synthetic4096/` 将 speech frame ratio 降到 `0.550969`，但 `pos2/neg1/cut6/s3000`、`pos4/neg1/cut6/s3000` 和 `pos2/neg1/cut6/s6000` 仍没有 checkpoint 通过旧 frame policy；`pos2/s6000` 最接近低 FPR，`threshold=0.6` 为 `recall=0.986210/FPR=0.007259`。随后 `agents/temp/20260618_221545_speech-boundary-frame-boundary-v3-negative-rich-synthetic8192-dil06/` 使用 `8192` rows、`speech_frame_ratio=0.587592`、`cut_drop_zone_count=10708`、`cut_point_segment_count=5676`，输出 checkpoint `agents/temp/20260618_221851_speech-boundary-frame-boundary-scorer-v3-negative-rich8192-dil06-prep/frame-boundary-scorer-v3/speech_boundary_ja_feature_scorer.pt`；全量 threshold eval 仍无 speech pass：`threshold=0.4` 为 `recall=0.995932/FPR=0.037859`，`0.5` 为 `0.992537/0.020272`，`0.6` 为 `0.987190/0.009828`，runtime profile 最佳 `on=0.6/off=0.55/cut=1.0` 为 `recall=0.988960/FPR=0.012228`。结论：negative-rich 数据修复了 FPR 侧，但 strict frame recall/FPR tradeoff 仍卡住；默认 scorer 继续不替换。
- 2026-06-18 SpeechBoundary-JA v3 error diagnostics：`evaluate_feature_scorer_thresholds` 新增 `--diagnostic-threshold` / `--diagnostic-runtime-profile`，输出 `speech_error_diagnostics.jsonl` 和 Markdown 的 boundary-distance / region / island-level / top-error rows。限量 1024-row 诊断 `agents/temp/20260618_230723_speech-boundary-v3-error-diagnostic-limit1024/` 显示错误高度集中在边界邻近帧：`threshold=0.5` 总体 `recall=0.992411/FPR=0.020234`，但 far-from-boundary region 已达 `recall=0.996881/FPR=0.004334`，near-boundary region 只有 `recall=0.915944/FPR=0.206994`，FP/FN 的 near-boundary share 分别为 `0.802624/0.611695`；runtime `on=0.6/off=0.55/cut=1.0` 也类似，far `0.995505/0.002182`、near `0.874780/0.130147`。补充 128-row island smoke `agents/temp/20260618_232007_speech-boundary-v3-island-diagnostic-limit128/` 显示 `threshold=0.5` 的 `speech_island_recall=1.0`、`mean_label_island_coverage=0.986958`，但 `cut_drop_zone_clean_rate=0.574074`、`mean_cut_drop_zone_predicted_ratio=0.162603`，说明核心 speech island 保留已经可观，gap/drop 不误连仍是主要 gate。当前判断：旧 `recall>=0.995/FPR<=0.02` 的逐帧 policy 对边界 120ms 内过于敏感，已经不适合作为 first scorer 唯一 gate。下一步应把 boundary-aware / segment-island-level eval 变成正式推荐口径，用核心 speech 保留、gap/drop 不误连和真实 workflow 人工观感作为主 gate；在新 gate 通过前不继续盲调 loss，也不替换默认 runtime。
- 2026-06-18 SpeechBoundary-JA v3 Grok hysteresis NAMH-055 opt-in smoke 已完成：首次运行在进入模型前暴露 Windows run-log 文件名过长问题，`task + label + backend` 全量拼接导致 `logging.FileHandler` 不能打开路径；已将 run-log 文件名组件改为保留前缀 + sha1 短 hash，避免长 label/backend 触发 MAX_PATH。随后使用 `SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT=agents/temp/20260618_144022_speech-boundary-frame-boundary-scorer-v3-prep/frame-boundary-scorer-v3-grok-pos4-neg1-cut8-s3000/speech_boundary_ja_feature_scorer.pt`、`SPEECH_BOUNDARY_JA_SCORER_DEVICE=cuda`、`--speech-boundary-speech-on-threshold 0.7 --speech-boundary-speech-off-threshold 0.5 --speech-boundary-cut-threshold 0.7 --no-boundary-cache --keep-asr-chunks` 跑通 NAMH-055 no-translate full workflow，输出 `agents/temp/speech-boundary-ja/20260618_161906_speech-boundary-v3-grok-hys-namh055-smoke/`。runtime signature 确认 `score_model=mamba2_frame_boundary_scorer`、`speech_threshold_mode=hysteresis`、on/off/cut 为 `0.7/0.5/0.7`，scorer sha256 `03ea7e646c3ed857a84260c45e1be2ac254e14a1fa1ff84387ee9097b2375382`。结果 `transcript_chunks=1486/segments=1751/blocks=1746`，ASR+Alignment `430.1s`，pipeline total `502.8s`，CueQC `candidates=1842/drop=356/keep=1486/fallback=0`。质量报告仍提示候选风险：`per_min_subtitle_count=19.41`、`alignment_fallback_ratio=0.378`、`subtitle_overlap_count=3`、`long_segment_count=53`；这说明 workflow 和新阈值 plumbing 已通，但该 scorer/阈值组合仍需人工 A/B 审计，默认 runtime 继续不替换。
- 2026-06-18 SpeechBoundary-JA v3 Grok hysteresis 音频 A/B 审计页已生成：按“只看音频切分和字幕文本、不展示 NSFW 视频画面”的要求，使用同一抽取 WAV `agents/temp/speech-boundary-ja/20260618_161906_speech-boundary-v3-grok-hys-namh055-smoke/jobs/NAMH-055_sbv3-grok-hys/audio/NAMH-055.2bb8f0e7.wav`，base 为 `agents/temp/speech-boundary-ja/20260617_cueqc-v3-namh055-smoke2/archived/NAMH-055/NAMH-055.ja.srt`，candidate 为 `agents/temp/speech-boundary-ja/20260618_161906_speech-boundary-v3-grok-hys-namh055-smoke/archived/NAMH-055/NAMH-055.ja.srt`，输出 `agents/audits/20260618_163540_speech-boundary-v3-grok-hysteresis-namh055-audio-ab/index.html`。页面为 `<audio>` 播放器，上方显示 base 当前 cue，下方显示 candidate 当前 cue，并按 30s 窗口列出 `179` 个切分/文本差异窗口；base/new cue 数为 `2259 -> 1746`，用于人工判断是否存在漏切、过合并或 ASR 文本退化。
- 2026-06-18 音频 A/B 页面字幕同步修复：用户反馈音频页 base/candidate 都有延迟后，先用 ffprobe 确认 MP4 与抽取 WAV 均为 0 起点，并用原视频无滤镜 180s WAV 与 workflow 处理后 WAV 做包络互相关，最佳 offset 为 `0.0s`，排除整体音频抽取偏移。根因收口到页面只依赖 `<audio>` 的 `timeupdate` 事件，浏览器对音频元素触发频率较低，字幕显示会滞后。`tools.audits.generate_subtitle_ab_compare_audit_html` 已改为播放中用 `requestAnimationFrame` 连续刷新，seek 时按目标时间立即渲染；已重建 `agents/audits/20260618_163540_speech-boundary-v3-grok-hysteresis-namh055-audio-ab/index.html`，新增回归测试 `tests/test_subtitle_ab_compare_audit.py`。
- 2026-06-18 A/B 审计页改为双独立播放器：用户反馈同时叠加听/看会干扰字幕观感审计后，`tools.audits.generate_subtitle_ab_compare_audit_html` 的 audio/video 模式均改为上下两个独立播放器，播放任意一边会自动暂停另一边；差异窗口和上一处/下一处只把两边定位到同一时间，不自动播放。当前音频审计页已重建为 `oldPlayer/newPlayer` 双 `<audio>`，仍不包含 `<video>`。同时确认本轮“怪”的一部分来自最终 SRT 而非页面：candidate 首条 final SRT cue 为 `10.810-11.063`，仅 `0.253s`，而其 ASR transcript chunk 为 `9.2906875-11.13`；这说明 forced aligner / subtitle postprocess 会把 boundary chunk 再拆成很短显示 cue。用户反馈第一条确实奇怪但后面整体还可以，所以下一步先用双独立播放器继续人工审计，不急着改 workflow 默认。
- 2026-06-18 forced aligner 去留 A/B plumbing 修正：复核 `qwen_asr` 包源码后确认 `return_time_stamps=True` 并不是 ASR 原生词时间戳，而是由 `Qwen3ForcedAligner` 产生，且官方 wrapper 要求初始化时带 forced aligner；因此不能用它代表“native/no-aligner”。`ALIGNMENT_TIMESTAMP_MODE=native` 已改为真正不加载 forced aligner：ASR 只做 text-only，字幕时间轴使用 Boundary chunk / speech_core fallback window 等比分配；`hybrid` 定义为 native timestamp 可用时优先 native、不可用且 chunk/text 满足 `ALIGNMENT_HYBRID_FORCE_MAX_CHUNK` / `ALIGNMENT_HYBRID_FORCE_MIN_TEXT_LEN` 时再 forced，否则 fallback。`tools.workflows.run_full_workflow` 新增 `--alignment-timestamp-mode forced|native|hybrid` 并写入 env/context/summary；native/quality/CLI 回归测试已补。无效实验 `agents/temp/speech-boundary-ja/20260618_173908_speech-boundary-v3-grok-hys-namh055-native-real/` 是误用 `return_time_stamps=True` 产生的 0 字幕结果，后续不要引用。有效 no-aligner NAMH-055 opt-in 结果为 `agents/temp/speech-boundary-ja/20260618_174707_speech-boundary-v3-grok-hys-namh055-no-aligner/`：同一 v3 scorer + `speech_on/off/cut=0.7/0.5/0.7`，`transcript_chunks=1486/segments=1468/blocks=1466`，Alignment 对齐 `0.06s`，pipeline total `429.45s`，quality report `alignment_fallback_ratio=0.987` 符合 no-aligner fallback 预期。首条 cue 对照显示 forced 为 `10.810-11.063` / `0.253s`，no-aligner 为 `9.290-11.063` / `1.773s`，首条异常根因进一步收口为 forced aligner 晚起点。音频双播放器 A/B 页面已生成：`agents/audits/20260618_175534_speech-boundary-v3-forced-vs-no-aligner-namh055-audio-ab/index.html`，forced/no-aligner cue 数 `1746 -> 1466`，差异窗口 `159`。
- 2026-06-18 forced-success delta 复核：基于同一 forced run，排除首条异常后统计 `905` 个 forced 成功 chunk；按 aligned segment span 相对原始 Boundary chunk 计算，`delta_start = forced_start - chunk_start` 的 mean/p50/p90/p95 为 `+0.440/+0.160/+1.280/+1.600s`，`delta_end = forced_end - chunk_end` 为 `-0.871/-0.640/-0.068/-0.020s`，整体 duration delta p50 `-1.140s`。按最终 subtitle block 统计，postprocess 会回填一部分 end，但仍为 mean/p50 `-0.481/-0.260s`；说明除第一条外，forced aligner 成功样本也系统性把 Boundary chunk 向内收，尤其 end 侧提前明显，这会减少尾部停留但也可能截掉非词尾音或造成短 cue。
- 2026-06-18 forced delta 审计页已生成：新增 `tools.audits.generate_forced_delta_audit_html`，默认按 `max(abs(delta_start), abs(delta_end))` 从大到小排序，页面内可分别播放 Boundary chunk、forced span 和最终字幕 span。NAMH-055 页面为 `agents/audits/20260618_181144_forced-delta-namh055-audit/index.html`，数据 `items.json` 共 `905` 条，已排除首条异常 chunk。
- 2026-06-18 forced delta 视频审计页已生成：同一生成器扩展为 `--media-mode audio|video`，保持 `--audio` 旧命令可用；视频版使用原片 `video/NAMH-055.mp4`，页面为 `agents/audits/20260618_182509_forced-delta-namh055-video-audit/index.html`，同样 `905` 条 forced 成功样本并按最大绝对 delta 降序。
- 2026-06-18 forced delta 视频字幕 overlay 审计页已生成：视频模式新增画面内字幕 overlay，`播放 Boundary` 会按 Boundary chunk 时间轴显示当前字幕，`播放 Forced` 会按 forced aligner span 显示同一字幕，`播放最终字幕` 保留最终 block 时间轴用于三方对照。新版页面为 `agents/audits/20260618_182859_forced-delta-namh055-video-caption-audit/index.html`。
- 2026-06-18 forced delta 音频字幕 overlay 审计页已生成：音频模式同样新增播放器下方字幕窗口，`播放 Boundary` / `播放 Forced` / `播放最终字幕` 分别按对应时间轴显示同一字幕，便于只听音频时比较起止点观感。新版页面为 `agents/audits/20260618_183136_forced-delta-namh055-audio-caption-audit/index.html`。
- 2026-06-18 项目手工阈值 / 启发式盘点：当前 active runtime 仍有四类需要区分处理的阈值。第一类是 SpeechBoundary-JA first scorer 相关：bootstrap scorer 仍是 `_range_normalize` 的 20/95 percentile、`energy/ptm/mfcc_delta=0.70/0.20/0.10` 加权、`frame_dilation_s=0.2`、`min_segment_s=0.05`、`max_group_s=6.0`、`chunk_threshold_s=1.0` 和 cut gate；这些是优先被 v3 learned scorer + calibration profile 取代的对象。第二类是 Boundary candidate/planner 约束：`cut_score_threshold=0.94`、`valley_score_threshold=0.10`、`target_chunk_s=3.0`、`max_core_chunk_s=5.0`、`min_chunk_s=0.4`、`max_splits_per_segment=16`，其中 candidate 阈值属于可学习/可校准候选，planner 时长上限更像字幕产品约束，不应简单删除。第三类是 CueQC：runtime 是 learned v3-Fusion，但决策仍有 checkpoint `drop_threshold=0.85` 与 `short_text=0.87` 的保守 profile；后续应把它变成由审计集生成的 calibration artifact，而不是继续手写常量。第四类是 ASR/aligner/subtitle 后处理：alignment sentinel、retry/refine 分段、fallback 时间戳、word grouping、subtitle density/QC 阈值仍有启发式；其中大部分是诊断或展示安全约束，不等同于 drop/keep 或 boundary 模型替代目标。后续清理优先级：先做 SpeechBoundary v3 calibration artifact，再做 CueQC threshold calibration artifact，最后单独审计 aligner/subtitle fallback 规则。
- 2026-06-18 CueQC -> SpeechBoundary-JA hard-negative-only 路由重置：用户确认人工 `drop_ok` 都是 hard negative，直接 drop，不再按“对话/呻吟/呼吸/环境/短碎片”等分类标签分流。`tools.boundary.export_cueqc_drop_hardcases` 已断兼容清理 CueQC drop -> Boundary Refiner 训练标签路线，`drop_ok` 统一导出为 `speech_boundary_frame_negative_candidate`；`tools.boundary.prepare_cueqc_drop_hard_negative_sources` 只生成 SpeechBoundary-JA negative labels，并会 fail-fast 拒绝非 frame-negative manifest。已基于三轮 false-drop 审计导出 hardnegative-only 数据：原始标签 `600` 行、unique item `578`、duplicate extra rows `22`、confirmed drop candidates `551`、safety holdout `27`，routes 为 `speech_boundary_frame_negative_candidate=551`；source prep 已 materialize `551/551` 个 frame-negative，`missing_audit_item_samples=0`、`frame_negative_skipped=0`，生成 `speech_boundary_negative_labels.jsonl`、`speech_boundary_negative_manifest.json`、`speech_boundary_training_manifest.jsonl`，训练可读 examples `551`。旧 20260617 CueQC drop hard-case 临时产物只保留为历史产物，不再作为后续训练输入。
- 2026-06-18 SpeechBoundary-JA hard-negative mixed feature cache 与 tiny plumbing 训练已完成但被新标签策略 supersede：早先基于 `agents/temp/20260617_230948_speech-boundary-hard-negative-finetune-prep/` 的 2022 条 mixed labels（522 negative + 1500 anchor），执行 `build_mixed_feature_cache.ps1` 成功生成 Qwen/MFCC feature cache，summary 为 `feature-cache-mixed-hard-negative-anchor/feature_summary.json`，结果 `cached=2022`、`skipped=0`、`errors=0`，label qualities 为 `negative=522/supervised=1500`。随后执行 `tiny_mixed_plumbing_train.ps1`，输出 `tiny-mixed-hard-negative-anchor/speech_boundary_ja_tiny.pt` 与 `train_metrics.json`，`steps=200`、`loss=0.607988`、`frame_accuracy=0.73`、`positive_ratio=0.73`。该 checkpoint 是 research/plumbing artifact，不接入 runtime，也不替换 `qwen-feature-energy-bootstrap-v1`；由于 `drop_ok` 路由已改为 551 全部 hard negative，该旧 feature cache / tiny train 不再作为下一轮正式 scorer 输入。
- 2026-06-18 SpeechBoundary-JA MLP feature scorer v1 失败结论：旧 runtime-loadable MLP 候选虽然能加载并保留多数正样本，但在 `recall>=0.995` 且 `FPR<=0.02` policy 下无 threshold 通过；最佳 F1 threshold `0.5` 为 `recall=0.998066/FPR=0.086026/F1=0.997301`，threshold `0.8` 虽满足 `FPR=0.019377` 但 recall 仅 `0.989536`。该路线已断兼容移除，旧 checkpoint schema 不再被 active loader 接受，避免后续误把 MLP v1 当作可 promoted 候选。
- 2026-06-18 SpeechBoundary-JA frame scorer v2 路线确定并开始落地：只保留 `speech_boundary_ja_mamba2_frame_scorer_v2`，复用现有 `BoundarySequenceClassifier` / `transformers.Mamba2Model`，输入仍为缓存的 Qwen PTM + MFCC features；默认 runtime 仍是 `qwen-feature-energy-bootstrap-v1`，只有显式设置 `SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT` 才加载 v2 scorer。训练入口固定为 Mamba2，不提供 MLP/`--architecture`/dropout 兼容分支；loss 使用 frame weights + class weights + optional focal，默认 `negative_weight=15.0`、`focal_gamma=2.0`。该 scorer 只替代 SpeechBoundary-JA bootstrap frame scores，不替代 Boundary Refiner v5；`src/boundary/checkpoints/boundary_refiner.pt` 不变。
- 2026-06-18 SpeechBoundary-JA Mamba2 frame scorer v2 首个候选已训练并通过离线 threshold policy，但现已被新 hardnegative-only 数据策略 supersede：基于旧 `agents/temp/20260617_230948_speech-boundary-hard-negative-finetune-prep/` 的 2022 条 mixed labels 和 feature cache，CUDA 训练 200 steps 输出到 `agents/temp/20260618_120005_speech-boundary-mamba2-frame-scorer-v2/speech_boundary_ja_feature_scorer.pt`，sha256 `159de61095e06e8919e9d0d16d4295e45c52a32b3be5c46ed16718e34401c5f4`。训练配置为 hidden/layers/state/heads/groups/chunk `128/2/32/4/2/8`、`negative_weight=15.0`、`focal_gamma=2.0`，held-out metrics：`eval_loss=0.009267`、`precision=0.998873`、`recall=0.995676`、`f1=0.997272`。全量 threshold sweep 输出 `agents/temp/20260618_120005_speech-boundary-mamba2-frame-scorer-v2-threshold-eval/`，policy `recall>=0.995/FPR<=0.02` 下 `threshold=0.5` 通过：`recall=0.998328`、`FPR=0.018753`、`precision=0.999243`、`F1=0.998785`。默认 runtime 未替换；下一轮可用候选必须基于 `agents/temp/20260618_124129_speech-boundary-hard-negative-finetune-prep-hardnegative-only-mixed/` 重新 build feature cache 并重训。
- 2026-06-18 SpeechBoundary-JA Mamba2 frame scorer v2 NAMH-055 opt-in smoke 已完成：使用 `SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT=agents/temp/20260618_120005_speech-boundary-mamba2-frame-scorer-v2/speech_boundary_ja_feature_scorer.pt`、`SPEECH_BOUNDARY_JA_SCORER_DEVICE=cuda`、no-translate、`--keep-asr-chunks`、`--no-boundary-cache` 跑 `video/NAMH-055.mp4`，输出 `agents/temp/speech-boundary-ja/20260618_121253_speech-boundary-mamba2-v2-namh055-smoke/`。runtime signature 确认 `score_model=mamba2_frame_scorer`，scorer schema `speech_boundary_ja_mamba2_frame_scorer_v2`，sha256 `159de61095e06e8919e9d0d16d4295e45c52a32b3be5c46ed16718e34401c5f4`；结果 `transcript_chunks=1065/segments=1476/blocks=1476`，ASR+Alignment `361.5s`，pipeline total `427.0s`。对比默认 baseline `agents/temp/speech-boundary-ja/20260617_cueqc-v3-namh055-smoke2/`：baseline `chunks=3199/segments=2444/blocks=2259`；alignment diagnostics compare 输出 `agents/temp/20260618_122344_mamba2-v2-namh055-diagnostics-compare/`，显示 baseline fallback `863` -> candidate fallback `421`、nonlexical bucket `842` -> `12`，但 candidate fallback ratio 按 chunk 为 `39.5%`，并且 chunk 数大幅减少，存在过度合并/漏切风险。人工对比审计页已生成 `agents/audits/20260618_122344_speech-boundary-mamba2-v2-namh055-compare/index.html`，`review_item_count=863`，outcome breakdown 为 `resolved=463/still_fallback=299/mixed=100/changed=1`；默认 runtime 仍未替换，必须等人工审计后再决定是否继续扩大 held-out。
- 2026-06-18 SpeechBoundary-JA hard-negative finetune readiness gate 已按 hardnegative-only 规则重跑：复用 `tools.boundary.ja.build_positive_anchor_replay` 在 `agents/temp/20260617_230636_speech-boundary-positive-anchor-replay/` 生成的 1500 条 positive anchors（`anime_nsfw=825/anime_sfw=300/galgame=375`），与新 551 条 negative source 混合到 `agents/temp/20260618_124129_speech-boundary-hard-negative-finetune-prep-hardnegative-only-mixed/`。该包生成 `speech_boundary_mixed_hard_negative_anchor_labels.jsonl` / manifest，`records=2051`、`trainable_examples=2051`、`negative_share=0.268649`，五项 gate 全部通过，并写出 `build_mixed_feature_cache.ps1`、`train_mixed_feature_scorer.ps1`、`tiny_mixed_plumbing_train.ps1`。该条记录为训练前 gate；后续已用新断兼容目录重训 v2 scorer。SpeechBoundary-JA 默认 runtime 仍是 `qwen-feature-energy-bootstrap-v1`。
- 2026-06-18 SpeechBoundary-JA Mamba2 frame scorer v2 hardnegative-only 重训：按新断兼容命名重新生成候选、source 和 mixed prep 到 `agents/temp/20260618_130257_*`，仍为 551 条 CueQC `drop_ok` hard negatives + 1500 条 positive anchors，mixed feature cache `cached=2051/errors=0/skipped=0`。第一版沿用 `negative_weight=15.0/max_steps=200` 未过离线 policy：推荐 `threshold=0.25` 时 `recall=0.996908` 但 `FPR=0.040462`。随后重训 `negative_weight=20.0/max_steps=800`，输出 `agents/temp/20260618_130257_speech-boundary-mamba2-frame-scorer-v2-neg20-s800/speech_boundary_ja_feature_scorer.pt`，sha256 `9d9f78f9cfeb1c2b0c6dc6d5764f10601786a44b0666ba3ff181dd394b4a6f58`；离线 threshold eval `agents/temp/20260618_130257_speech-boundary-mamba2-frame-scorer-v2-neg20-s800-threshold-eval/` 通过 policy，选中 `threshold=0.8`：`recall=0.995668`、`FPR=0.019718`、`precision=0.999111`、`F1=0.997386`。NAMH-055 opt-in no-translate smoke 使用该 checkpoint + `--speech-boundary-threshold 0.8` + `--no-boundary-cache`，输出 `agents/temp/speech-boundary-ja/20260618_130257_speech-boundary-mamba2-v2-neg20-s800-namh055-smoke/`，runtime signature 确认 `score_model=mamba2_frame_scorer`、scorer sha256 同上；结果 `transcript_chunks=1801/segments=1952/blocks=1947`，ASR+Alignment `468.9s`，pipeline total `546.6s`，quality report `alignment_fallback=622/1801=0.345`、`per_min_subtitle_count=21.65`、`short_segment_ratio=0.100`、`long_segment_count=4`、`overlap_count=5`。相比旧 522-negative v2 smoke 的 `1065` chunks，新候选明显更保守；默认 runtime 仍未替换，下一步应生成与默认 baseline 的人工对比审计页，重点看漏切/过合并和非语言片段丢弃是否符合预期。
- 2026-06-18 旧边界偏好路线归档：120 条盲测和 77 条 compiled label 只作为历史实验结论保留，不再作为 active 微调入口。CueQC 已确认 `display=drop` / `drop_ok` 全部作为 SpeechBoundary-JA frame-level hard negative；过碎短切片和无效 speech island 的目标是直接 drop，而不是并入相邻字幕。旧准备、生成、编译、汇总工具和对应测试已移动到 `agents/rm/20260618_124129_*` 归档目录，active tree 不再提供这条训练工具链。
- 2026-06-17 Web 分发链路 NAMH-055 不翻译 smoke 已完成，并暴露/修复 CueQC runtime 诊断盲区：早先一次 Web smoke 曾出现 CueQC 推理异常后全量 fallback keep，但旧日志只显示保守 keep，容易误判模型已生效。根因是 runtime refiner 把候选放进单个大批次推理，批次内异常会导致整批 fallback，且 fallback detail 不进入 pipeline summary。修复后 CueQC v3-Fusion runtime 按 `CUEQC_INFERENCE_BATCH_SIZE` 分批推理，capture/feature/inference fallback 会写入 `fallback_stage/fallback_detail`，pipeline 日志和 timings 报告 `fallback_summary`。重启 Web 后 job `38a5d14ea1a54236a24b56e716f36175` 完成：CueQC 完整决策 `3199` 条、`drop=1052`、`keep=2147`、`fallback=0`，最终 `transcript_chunks=2147`、`segments=2157`、`blocks=2087`，输出 summary 在 `agents/temp/20260617_191654_web-smoke-namh055-cueqc-batched/job_summary.json`。同时把可复用 Web smoke 脚本整理到 `tools/web/smoke/`，把大规模 CueQC feature 分片提取从一次性 `agents/temp/.../run_extract_shards.ps1` 抽象为 `tools/asr/cueqc/extract_feature_shards.py`，README 底部新增工具索引，后续临时硬编码脚本只保留为历史运行记录，不作为复用入口。
- 2026-06-17 CueQC v3-Fusion Stage 2b 已通过 false-drop gate 并替换默认 checkpoint：当前默认 `src/asr/checkpoints/cueqc_mamba_v3_fusion.pt` 来自 Stage 2b 自训练，sha1 `98f9631a63dc19736b50619100fb4be4d08075e8`，旧默认备份在 `agents/temp/20260617_174154_cueqc-default-checkpoint-backup/cueqc_mamba_v3_fusion.before-stage2b-adaptive.pt`。三轮 false-drop 审计合计 `600` 条，结果 `drop_ok=573 / false_drop_keep=25 / uncertain=2`；第三轮 Stage 2b 审计为 `198 drop_ok / 1 false_drop_keep / 1 uncertain`，裸 false-drop rate `0.5%`。新增 `src/asr/cueqc_thresholds.py` 作为 runtime 与 offline prediction 共用的阈值校准层：基础 `drop_threshold=0.85`，`text_bucket=short_text` 提升到 `0.87`，profile 只允许抬高阈值，不允许低于 base threshold。该 profile 回放三轮审计时保留全部 `25` 个 `false_drop_keep`，同时仍 drop `540/573` 个人工确认 `drop_ok`；10-film adaptive prediction `agents/temp/20260617_174344_cueqc-v3-stage2b-adaptive-10film-predictions/` 为 records `45643`、`drop=19380/keep=26263`、高置信 pseudo `drop=19380/keep=1372`。
- 2026-06-17 CueQC Stage 2 自训练闭环已完成一轮并收敛到保守默认：10-film 全量特征在 `agents/temp/20260617_113159_cueqc-v3-10film-sharded-features/` 按 46 个 shard 提取并合并为约 `3.02GB` 的 `cueqc_full_features_v3_fusion.pt`，共 `45643` 条未标注候选。初始 prediction `keep=31055/drop=14588` 后，第一轮人工审计 `178 drop_ok / 21 false_drop_keep / 1 uncertain`，因此未接收未审 drop pseudo；Stage 2a 训练包 `538` 条仅纳入 cold-start、人工确认 drop、人工纠正 keep 和高置信 keep。t=0.88 第二轮审计降到 `197 drop_ok / 3 false_drop_keep`，但阈值-only 到 `0.92` 只剩 `337` drop，收益太低；Stage 2b 训练包 `2177` 条，labels `drop=532/keep=1645`，固定 holdout `867HTTM-0045` 达 `keep_recall=1.0/false_drop_rate=0.0/drop_precision=1.0/drop_recall=0.8475`。结论：Stage 2 当前以 keep recall / false-drop 安全为优先，未审高置信 drop 不直接进入训练。
- 2026-06-17 CueQC v3-Fusion runtime 已替代旧规则 ASR QC：用户确认“旧 QC 决策直接删除，不观测不运行”。active tree 已移除 `src/asr/qc.py`、`src/asr/qc_stage.py` 和旧 ASR QC 专属测试；pipeline 不再调用 `_run_TRANSCRIPTION_qc` / `collect_adaptive_precision_review`，quality report、alignment quality、旧边界偏好实验、silver mining 和审计页不再读取或展示 `asr_qc_*` / `asr_review_uncertain`。CueQC 输出只保留 `display_hint=keep/drop`；模型不可用或异常时 fallback keep。`tools/asr/cueqc/compile_training_set.py` 只接受簇级 `cueqc_cluster_labels.jsonl`，per-sample `cueqc_manual_labels.jsonl` 和 `content_type/qc_decision/alignment_policy` 旧标签头不再兼容；旧 `generate_asr_attribution_audit_html.py` 已移到 `agents/rm/20260617_101851_retired-asr-attribution-qc-audit/`。
- 2026-06-16/17 CueQC v3-Fusion 断兼容重构与 cold-start 完成：v3 从 v2 multimodal 切到 `ASR encoder features + ASR token trace + decoder aggregate stats + structured metadata -> display keep/drop`，删除 BGE-m3、sentence-transformers、HuBERT、UMAP/HDBSCAN/FINCH/PCA 和边界 frame 作为默认 CueQC 输入；runtime 复用已加载的 Qwen3-ASR backend，不加载第二份 ASR。关键实现包括 `src/asr/asr_internals.py`、`src/asr/cueqc_features.py`、`src/asr/cueqc_model.py`、`src/asr/cueqc_refiner.py`、`tools/asr/cueqc/extract_features_v3_fusion.py`、`train_mamba_v3_fusion.py`、`predict_v3_fusion.py` 和 `compile_stage2a_features_v3_fusion.py`。实施期修复两个关键坑：teacher-forced logits 必须走 `wrapper.model.thinker.forward(...)`，`get_audio_features` 在 batch=1 返回 `[T,D]` 时需补 batch 维。300 条 cold-start 最终 labels `keep=133/drop=167`，首轮过拟合模型已废弃；保守配置加 label smoothing、early-stop 和 keep 权重后，内部 holdout `867HTTM-0045` 达 `keep_recall=1.0/false_drop_rate=0.0`，作为 Stage 2 起点。
- 2026-06-15/16 CueQC bootstrap 聚类已完成并退役为一次性种子步骤：NAMH-055 曾误入 11-film 审计池，复核后固定为 smoke/holdout，实际训练池来自 CueQC 合入前 baseline commit `5afe535` 的 10 部全片、`skip_translation=True`、保留 ASR chunks，候选池 `agents/temp/20260615_152934_cueqc-10film-candidates/cueqc_candidates.full.jsonl` 共 `45643` 条，并分层抽样 `300` 条。旧 HDBSCAN/FINCH/UMAP/PCA、embedding 增强和 17 簇 taxonomy 路线均已被废弃；最终只保留 Torque Clustering 作为一次性 coarse seed 工具，`--merge-layer 1` 得到 7 簇并生成审计页 `agents/audits/20260616_clueqc-torque-layer1-audit/index.html`。用户完成 7 簇 keep/drop 标注后，簇级广播生成 cold-start 训练种子；structured 聚类不进入 runtime、不进入 Stage 2 自训练，也不再作为长期 QC 分类器。
- 2026-06-15 CueQC 短切片定位结论保留为 Boundary 反哺线索：`cluster_00` 中代表性短噪声/短切片样本如 `867HTTM-0045 chunk875` 是 boundary cache 中独立 `speech_island` 造成的原始 ASR core chunk，duration 约 `0.43s`，不是审计页或 forced aligner 二次切割。后续可把人工确认的 `display=drop`、短噪声、过碎短切片、无效 speech island 或应合并片段整理为 Boundary hard-case / preference 数据；这一步必须保持离线偏好训练，不把 CueQC 结果耦合进 Boundary runtime。
- 2026-06-14 ASR 非词声音可读性与 forced-aligner 去留判断：人工审计观察到 forced aligner 时间轴与 SpeechBoundary-JA / Boundary Refiner 产出的 speech core 窗口已经接近，说明当前 v5 core window 基本找准了，低信息/重复人声问题更像显示策略问题，而不是继续放大窗口或重新依赖 forced aligner 的问题。当前代码已有 `ALIGNMENT_TIMESTAMP_MODE=forced/native/hybrid`，并且 Web model requirements 在 `native` 模式下不会要求 `forced_aligner`；但默认配置仍是 `forced`。因此当前不直接删除 forced aligner：它继续保留为 word-timing polish、候选排序、诊断审计和未来 teacher/silver source；若要节约空间，应先做同片 `forced` vs `native`/`hybrid` A/B。
- 2026-06-15 临时产物命名约定更新：后续所有新生成的 `agents/temp/` 临时文件或目录统一加本地时间前缀 `YYYYMMDD_HHMMSS_`，例如 `20260615_081230_cueqc-11film-baseline`。清理旧临时产物仍移动到 `agents/rm/`，清理归档目录同样使用时间前缀，避免后续无法从路径判断生成或归档时间。
- 2026-06-13 旧边界偏好试点归档：120 条盲测和 77 条 compiled label 曾用于验证 v5 边界扰动偏好信号，但未训练、未替换默认 checkpoint。2026-06-18 起该路线退出 active tree；后续不再从这批标签继续微调。
- 2026-06-11 true v5 后续路线决策已归档：当时选择从当前 true v5 做边界扰动偏好试点，不恢复 v4、不引入 forced-aligner silver 作为首轮微调。该决策的有效结论只保留为“Boundary Refiner v5 delta-only 当前默认、旧 v4/merge 路线不恢复”；后续训练计划已被 SpeechBoundary-JA hard-negative frame scorer 路线取代。
- 2026-06-11 翻译吞吐、缓存预热与 Web 状态闭环：翻译默认 worker 从 `4` 提高到 `16`，允许范围扩大到 `1-64`；自动 batch 公式提高并封顶 `400` cues，provider prefix warmup 同时覆盖 full JSON prefix 与 summary fallback。使用 NAMH-055 完整跑通 SpeechBoundary-JA、Qwen ASR、forced alignment、DeepSeek 翻译与 SRT 写出：`3199` transcript chunks、`2444` cues/blocks，总 elapsed `803.7s`，pipeline `779.9s`，其中 ASR+Alignment `286.5s`、翻译 `362.4s`、输出 `44.1s`；实际翻译批次为 `6x400 + 44`，全部 `missing_count=0`，API retry events `0`。翻译请求统计累计 provider cache hit `478976` tokens、cache miss `189486` tokens，证明 summary fallback 前缀预热和 400-cue 大批次可稳定工作；两个 400-cue 批次各触发一次批内补全但最终完整返回。Web 同步修复 FIFO 卡片顺序和 SSE 最终状态可能早于持久化的问题：活动任务保留轻量 reconciliation polling，避免后台已完成而前端持续显示运行中。Windows 审计入口新增 `tools/audits/serve_audits.ps1`，与现有 live-server middleware 保持一致。
- 2026-06-11 true v5 delta-only 32768 训练落地：按断兼容要求清理旧 merge-era 生成训练产物，`agents/temp/speech-boundary-ja/v4-core-32768-hardmix` 和旧 full-video silver refiner train/data 已移到 `agents/rm/20260610_v4_merge_weight_training_artifacts/`；`tools/boundary/train_refiner.py` 新增 fail-fast guard，只接受 `boundary_refiner_frame_sequence_dataset_v5`，并拒绝 `sequence_labels`、`sequence_context_targets`、`merge_positive`、`split_negative`、`boundary_merge_prob`、`boundary_split_prob`、`boundary_decision_merge`、`merge_label`、`merge_weight`、`split_label`、`split_weight` 等旧字段。新数据生成到 `agents/temp/speech-boundary-ja/v5-delta-only-32768/`：mixed source `196608` 条，组成为 `nsfw=88474`、`sfw=58982`、`galgame=49152`；synthetic `32768` 条，skipped `4`，`cut_point_segment_count=85192`、`cut_drop_zone_count=77820`、`overlap_mix_count=6558`，gap modes 为 `fade_noise=58967`、`hum=26161`、`silence=58851`、`white_noise=52494`。CUDA/bf16/no-compress feature cache 使用 `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`，`cached=32768`、`errors=0`、`skipped=0`；frame sequence v5 输出 `32768` sequences、`167928` items，`start_supervised=167928`、`end_supervised=167928`，label reasons 为 `boundary_cut_point=46308`、`boundary_gap_zone=74201`、`boundary_long_gap=2585`、`boundary_overlap=44834`，feature dim `630`、hash `cb44d7804ad4eaec9e5e4db80123f817a99650fa`。训练从零开始，CUDA `600` steps、batch `512`、lr `5e-4`、weight decay `0.01`、hidden/layers/state `128/2/32`、start/end loss weights `1.0/0.6`、target/core/min `3.0/5.0/0.4s`；last loss `0.003754`，train start/end MAE `0.0576/0.0591s`，val start/end MAE `0.0616/0.0603s`。checkpoint schema `boundary_refiner_v5`、output dim `2`、backbone `transformers.Mamba2Model`、runtime adapter `frame_sequence_v1`，已替换默认 `src/boundary/checkpoints/boundary_refiner.pt`；补齐顶层 feature metadata 镜像后的 sha256 为 `503d7e2299460aff555e02cba2b840c59195e577719bd0637a5ae98657ef919f`。替换后聚焦回归 `tests/test_boundary_refiner.py tests/test_boundary_refiner_training.py tests/test_boundary_cache.py tests/test_boundary_planner.py tests/test_chunk_packer.py tests/test_pipeline_chunk_config_runtime.py tests/test_pipeline_chunk_packing.py` 为 `50 passed`；补 metadata 镜像后 `tests/test_boundary_refiner.py tests/test_boundary_refiner_training.py` 为 `18 passed`。不能只凭 synthetic MAE 判定优劣，仍需真实 downstream 与人工审计 gate。
- 2026-06-11 true v5 delta-only 匿名样片 A GPU 闭环完成：命令使用默认 `src/boundary/checkpoints/boundary_refiner.pt`、`jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`、ASR batch `64`、aligner batch `128`、日文-only、不翻译、`--no-boundary-cache` 强制重算，输出 `agents/temp/speech-boundary-ja/sample-a-true-v5-delta-only-qwen17b-gpu/`。结果：`transcript_chunks=3199`、最终 `segments/blocks=2260`，pipeline total `297.9s`，整体 elapsed `300.1s`，ASR model load `8.26s`、ASR transcription `162.51s`、alignment `52.00s`、ASR+Alignment `254.1s`。质量报告：subtitle duration p50/p90/p95/max `1.09/2.171/2.473/3.438s`，long segment `0`，short ratio `0.158`，micro `81`，overlap `3` / `0.124s` total / `0.049s` max，per-min subtitle count `25.13`，density>4cps `1841`。ASR QC：reject `5`、warn `257`、empty-for-speech `245`、review uncertain `5`。Alignment fallback `731/3199` (`22.9%`)。对比 v4 core 32768 同片：chunks 相同 `3199`，blocks `2255 -> 2260`，ASR+Alignment `263.6s -> 254.1s`，pipeline total `307.6s -> 297.9s`，warn `268 -> 257`，empty `256 -> 245`，fallback `780 (24.4%) -> 731 (22.9%)`，duration p90/p95 `2.210/2.564s -> 2.171/2.473s`；短字幕比例 `0.149 -> 0.158` 略升。结论：true v5 指标小幅优于 v4，尤其是 fallback、ASR empty、warning 和耗时；但差距不大，不能直接宣布替换成功，下一步应生成 v4 vs true v5 审计页做人工观感 gate，并补 held-out 全片。
- 2026-06-11 post-v5 loss 实验回滚决策：匿名样片 A / 匿名样片 B / 匿名样片 C 都已做 v5 vs v6e 字幕对比。v6e 在部分 forced/fallback 数字上更好，但 ASR empty、字幕密度或人工观感风险没有稳定胜出；用户决定“切回 v5”。确认默认 `src/boundary/checkpoints/boundary_refiner.pt` 仍是 true v5 sha256 `503d7e2299460aff555e02cba2b840c59195e577719bd0637a5ae98657ef919f`，未替换为 v6e sha256 `d96df85f26df96688ef2abd8e9ec27b211ea6bd5ab6beb7dfa33ca188e7c7aee`。断兼容清理后，active 训练入口只保留 v5 的 `smooth_l1`、`start_delta_loss_weight=1.0`、`end_delta_loss_weight=0.6`；post-v5 的 centered/asymmetric/exp-band loss、tolerance-band 指标、独立误差评估脚本和 v6/v6e 审计/临时产物均已移出 active tree，只在 HISTORY 保留结论。
- 2026-06-11 v5 审计准备与提交：active tree 扫描确认 v6/v6e 产物文件名不再残留，v6/v6e 只保留在本文件的实验结论中；默认 checkpoint 仍是 true v5 sha256 `503d7e2299460aff555e02cba2b840c59195e577719bd0637a5ae98657ef919f`。基于 v5 全片输出重建 alignment diagnostics：`chunks=3199`、`failure candidates=1785`，生成本地音频字幕审计页 `agents/audits/v5-subtitle-audio-audit/index.html`，抽样 `102` 条，完整日文 cue `2260` 条；审计页与 `agents/` 目录按 `.gitignore` 保留本地，不随 push 发布。聚焦回归 `tests/test_audit_nav.py tests/test_boundary_refiner.py tests/test_boundary_refiner_training.py tests/test_boundary_cache.py tests/test_boundary_planner.py tests/test_chunk_packer.py tests/test_config.py tests/test_asr_stage_env_scope.py tests/test_pipeline_chunk_config_runtime.py tests/test_pipeline_chunk_packing.py` 为 `66 passed`。已提交并推送 `bff811f Finalize v5 boundary refiner cleanup`。
- 2026-06-10 v4 core-only 32768 hardmix 真实域闭环完成：按用户要求旧 11 部 silver artifacts 直接作废，使用 `agents/temp/speech-boundary-ja/v4-core-32768-hardmix/train-mamba2-v4-core-32768-s07e055/boundary_refiner.pt`、`jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`、ASR batch `64`、aligner batch `128`，对 11 部匿名全片从视频重新跑 SpeechBoundary / ASR / forced aligner。产物 `agents/temp/speech-boundary-ja/full-video-qwen17b-v4-core-32768-s07e055/`：总 transcript chunks `48848`、最终 blocks `31339`、ASR+Alignment 合计 `3790.3s`、总 elapsed `4410.5s`。单片 chunks/blocks 最高为匿名长片 `5303/4067`，字幕密度整体偏高，说明该 32768 synthetic checkpoint 适合挖真实域 silver label，但还不能只凭合成验证或训练 loss 直接替换默认。
- 2026-06-10 full-video silver boundary mining 与真实域训练链路更新：`tools/boundary/mine_silver_boundary_labels.py` 的 label policy 从 `start_weight=1.0 / end_weight=0.35` 改为 `start_weight=1.0 / end_weight=0.6`：结论是 end 不是低价值，只是字幕显示层在相邻 cue 过近时可以压缩前一条 end 来保持 2-frame gap；Boundary Refiner 训练仍应学习 end。基于上述新边界完整重跑结果挖 silver，产物 `agents/temp/speech-boundary-ja/full-video-qwen17b-v4-core-32768-s07e055/silver-boundary-labels/`：chunks `48848`、silver labels `12574`、hard cases `36274`；alignment quality 为 forced `18682`、nonlexical `18643`、proportional `11442`、drop/review `64`、partial `17`；silver start error p50/p90/p95/max `-0.16/0/0.000187/0.0005s`，end error p50/p90/p95/max `0.222375/1.059869/1.405856/2.831875s`。这说明当前 speech core start 通常略早于 forced first word，end 通常偏晚，适合作为真实域 display-boundary correction 的监督信号。
- 2026-06-10 新增真实域 silver -> Boundary Refiner v4 数据与微调链路：`tools/boundary/export_silver_sequence_features.py` 从新 silver labels 去重 11 个源音频，提权 CUDA 重新导出 full-audio `Qwen3-ASR-0.6B-JA-Anime-Galgame` PTM + MFCC sequence frame `.npz`，产物 `agents/temp/speech-boundary-ja/full-video-qwen17b-v4-core-32768-s07e055/silver-sequence-features-ptm64/`，导出 `11/11`、errors `0`。`tools/boundary/build_silver_refiner_dataset.py` 再把 silver labels、aligned transcript chunks 和 sequence feature manifest 转成 `boundary_refiner_frame_sequence_dataset_v4` 训练 JSONL，产物 `.../silver-refiner-dataset-v1/`：rows `47`、sequence items `20734`、split_negative `20734`、start_supervised `12570`、end_supervised `12573`、overlapping_chunk_gap `616`、feature_dim `630`、feature_schema_hash `ab5ca6f85b2e4da013a7ffaeef0ba73ccbd598a6`。随后从 32768 hardmix checkpoint 初始化，冻结 Mamba2 backbone，只训练输出头，`merge_loss_weight=0`、`start/end_delta_loss_weight=0.80/0.60`、`preserve_init_normalization=1`、`lr=1e-4`、`300` steps，得到 `.../train-mamba2-v4-core-32768-s07e055-silver-ft01/boundary_refiner.pt`。该 silver-ft01 尚未设为默认，下一步必须用 匿名样片 A / 匿名样片 B 做 downstream A/B。
- 2026-06-10 silver-ft01 downstream A/B 完成：使用同一 Qwen3-ASR-1.7B、ASR batch `64`、aligner batch `128`、`--no-boundary-cache`，只把 Boundary Refiner checkpoint 从 32768 hardmix 换成 silver-ft01，重跑 匿名样片 A / 匿名样片 B。匿名样片 A：chunks `3199 -> 1710`、blocks `2251 -> 1288`、ASR+Alignment `254.6s -> 161.5s`、QC warn `265 -> 125`、ASR empty `255 -> 119`、reject `5 -> 4`、fallback ratio `0.229 -> 0.205`、per-min subtitle count `25.0 -> 14.3`、short ratio `0.161 -> 0.071`、overlap `1 -> 1`。匿名样片 B：chunks `5303 -> 3618`、blocks `4067 -> 2954`、ASR+Alignment `518.8s -> 348.7s`、QC warn `484 -> 294`、ASR empty `448 -> 265`、reject `11 -> 8`、fallback ratio `0.197 -> 0.197`、per-min subtitle count `26.9 -> 19.5`、short ratio `0.162 -> 0.094`、overlap `3 -> 1`。结论：真实域 forced-aligner silver label 用于 start/end display-boundary delta 微调的方向有效，显著降低过切、短字幕密度和 ASR 空输出；仍需生成审计页确认是否误合并真实短促台词，再决定是否替换默认 checkpoint。
- 2026-06-10 人工审计修正 silver-ft01 结论：生成 `subtitle-ab-v4-32768-vs-silver-ft01-{video,audio}` 后，用户人工确认 silver-ft01 “学坏了”，观感和准确度不如 32768 hardmix。说明单看 chunk/cue 减少、ASR empty 下降和 fallback ratio 下降会误导模型选择；后续真实域 silver 微调必须加入人工审计 gate，并控制 hard negative / forced-aligner silver label 对 start/end delta 的过度保守影响。
- 2026-06-10 4096 vs 32768 A/B 审计完成：生成 `agents/audits/subtitle-ab-v4-4096-vs-32768-audio/index.html`，同片 匿名样片 A 对比旧 4096 默认与 32768 hardmix。人工观感确认 32768 更好，因此默认 checkpoint 切到 32768 hardmix。该结果支持保留 32768 synthetic 规模和更高组合复杂度；下一轮训练应沿用 32768 hardmix 策略，而不是回到 4096 或采用 silver-ft01 的过度压缩行为。
- 2026-06-10 Boundary Refiner v4 core-only 匿名样片 A GPU 闭环完成：命令显式使用 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`、ASR batch `64`、aligner batch `128`、日文-only、不翻译、`--no-boundary-cache` 强制重算，输出 `agents/temp/speech-boundary-ja/sample-a-v4-core-qwen17b-gpu/`。CUDA 链路确认：SpeechBoundary-JA `actual_device=cuda`、PTM `cuda:0`；运行中 ASR 阶段显存采样约 `7116/8188 MiB`，适合本机 8GB 实验档，不应写成 6GB 安全档。结果：`transcript_chunks=3199`、最终 `segments/blocks=2255`，ASR+Alignment `263.6s`，pipeline total `307.6s`，整体 elapsed `309.7s`。质量报告：subtitle duration p50/p90/p95/max `1.087/2.210/2.564/3.436s`，long segment `0`，short ratio `0.149`，micro `81`，overlap `3` / `0.08s` total / `0.04s` max，per-min subtitle count `25.07`，density>4cps `1806`。ASR QC：reject `5`、warn `268`、empty-for-speech `256`、review uncertain `5`，reject 主要是 `repeat_ngram_loop`。Alignment diagnostics：chunks `3199`、forced `1384`、nonlexical `1032`、proportional `777`、partial `1`、drop/review `5`；fallback chunks `780` (`24.4%`)，fallback subtype 为 `proportional_after_sentinel=777`、`word_timing_high_cps=1`、`asr_review_uncertain=5`，nonlexical text `1032` 不计入 fallback type。结论：v4 core-only 没有回到长粗时间轴，真实瓶颈是 subtitle density / 短非词人声 / repeat loop / forced-aligner sentinel，而不是 padding 或 runtime merge。
- 2026-06-10 闭环后元数据修复：正常 ASR 完成路径此前没有把 `_LAST_BOUNDARY_SIGNATURE` 写入 `asr_details`，导致 `tools/boundary/ja/run_full_workflow.py` 的 `summary.json` 中 `boundary_signature` 为空；这不影响已生成字幕，但削弱后续复盘。已在 `src/asr/pipeline.py` 正常完成 details 中写入 `boundary_signature`，并在 `tests/test_pipeline_chunk_packing.py` 覆盖 version `4`。验证：`compileall` 通过，`tests/test_pipeline_chunk_packing.py` 8 passed，`pytest -q` 为 `459 passed`。
- 2026-06-09 Boundary Refiner v4 断兼容重构完成：no-padding GPU A/B 已证明 padded ASR context 会明显增加空转写、字幕密度和重叠风险，因此主线从 v3 `merge + left_context + right_context + start_delta + end_delta` 改为 v4 core-only `merge + start_delta + end_delta`。runtime 删除 learned padding/context budget，`PackedChunk`、ASR chunk metadata、boundary cache 和训练数据不再写 `speech_left_padding_s` / `speech_right_padding_s` / `sequence_context_targets` / `BOUNDARY_CONTEXT_MAX_PADDING_S`；ASR chunk、fallback window 和字幕初始时间轴直接使用 refined speech core。`boundary-cache` 升 v4，旧 cache 直接 miss；旧 v3 checkpoint 不靠 sha256 白名单拦截，而是会因 `schema=boundary_refiner_v3` / `output_dim=5` 与 v4 `schema=boundary_refiner_v4` / `output_dim=3` 不匹配而拒绝加载。
- 2026-06-09 v4 core-only 4096 checkpoint 已训练并替换默认分发文件：从本地 `japanese-anime-speech-v2` NSFW/SFW 与 Galgame 100k 源池按 `nsfw=45 / sfw=30 / galgame=25` 生成 `20480` 条 mixed source，再合成 `4096` 条 multi-island 样本，覆盖 touching speech、short gap、long gap、轻量 overlap、gain/filter/codec 增强；summary 记录 `overlap_mix_count=644`、`filter_aug_count=1232`、`codec_aug_count=229`、`cut_point_segment_count=6382`、`cut_drop_zone_count=10000`。CUDA/bf16/no-compress feature cache 使用 `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`，`batch_size=128`、`prepare_workers=8`，`cached=4096`、`errors=0`、`skipped=0`；frame-sequence v4 输出 `4096` sequences、`21022` boundary items、`merge_positive=4096`、`split_negative=16926`、`feature_dim=630`、`feature_schema_hash=ab5ca6f85b2e4da013a7ffaeef0ba73ccbd598a6`。Mamba2 CUDA 训练 `300` steps、batch `512`、`lr=5e-4`、hidden/layers/state `128/2/32`，last train loss `0.004981`，synthetic train/val accuracy 与 merge F1 均为 `1.0`。默认 `src/boundary/checkpoints/boundary_refiner.pt` 已替换为 v4 checkpoint，sha256 `61c7c948d27944b2399b94742263b787403d151c658b2c553b783242010703ff`，大小约 `2.2MB`；训练大中间产物已删除，只保留 `agents/temp/speech-boundary-ja/v4-core-only-4096/keep/` 下的 summary、metrics 和 checkpoint 备份。验证：`tests/test_boundary_refiner.py tests/test_boundary_refiner_training.py tests/test_chunk_packer.py tests/test_boundary_cache.py tests/test_pipeline_chunk_config_runtime.py tests/test_boundary_timing_formula.py tests/test_pipeline_chunk_packing.py tests/test_fallback_safe_boundary_metrics.py` 共 `67 passed`。下一步真实 gate 是匿名样片 A / held-out GPU downstream，观察 start attribution、fallback duration、ASR empty、repeat/nonlexical 风险和 forced/fallback 分布。
- 2026-06-09 审计页持续刷新根因确认并修复：页面本身没有 `meta refresh` / `location.reload()` / `window.location` 自动跳转；真正问题是从项目根目录裸跑 `live-server` 时，它会 watch 全项目并给 HTML 注入 websocket，训练和 GPU 闭环持续写 `agents/temp`、`tmp`、cache、日志时会触发浏览器 reload。新增 `tools/audits/serve_audits.sh`，仍从项目根目录提供视频/音频等相对资源，但只 watch `agents/audits`，并保留 `tools/audits/live_server_audit_middleware.js` 删除入口；`tools/audits/audit_nav.py` 和当前 `agents/audits/index.html` 已同步提示该启动方式。
- 2026-06-09 no-padding A/B GPU 闭环完成：使用当时的 v3 4096 默认 Mamba2 checkpoint，不覆盖模型，只在 runtime 显式 `--boundary-context-max-padding-s 0 --no-boundary-cache` 重跑匿名样片 A / Qwen3-ASR-1.7B / ASR bs64 / aligner bs128。输出 `agents/temp/speech-boundary-ja/sample-a-v3-4096-qwen17b-nopad-gpu/`，pipeline total `306.7s`，`transcript_chunks=3067`、最终 `blocks=2161`、ASR empty `258`、QC reject `4`、fallback ratio `0.245`、short ratio `0.146`、overlap `3`。对比 padded v3 4096 的 `blocks=3687`、ASR empty `530`、QC reject `16`、overlap `486`，no-padding 明显减少 ASR context 污染和字幕密度/重叠问题。当时默认 4096 checkpoint 已备份到 `agents/temp/speech-boundary-ja/v3-boundary-delta-4096/backup-current-4096-before-nopad/boundary_refiner.v3-4096-before-nopad.pt`，sha256 与当时 `src/boundary/checkpoints/boundary_refiner.pt` 同为 `763ef5cbb66e3d2e49bf1003adccefabf0ff4540aa227b0ff8a6fa866d6d46fe`；后续已被 v4 core-only checkpoint 替换。
- 2026-06-09 历史 v3 4096 重训版记录：先清理可重建生成产物并释放旧 `v3-boundary-delta` 大目录约 `175G`，只保留本地源数据集 `boundary-sources/*` 与 `datasets/raw/musan`；随后重新生成 `4096` synthetic records，manifest 权重 `anime_nsfw=1843`、`anime_sfw=1229`、`galgame=1024`。CUDA/bf16 feature cache 使用 `--no-compress --resume --batch-size 128 --prepare-workers 8` 跑完，summary 确认 `compressed=false`、`cached=4096`、`errors=0`、`skipped=0`。frame-sequence v3 输出 `4096` sequences、`21036` items、`merge_positive=4096`、`split_negative=16940`、`feature_dim=630`、`feature_schema_hash=ab5ca6f85b2e4da013a7ffaeef0ba73ccbd598a6`。Mamba2 CUDA 训练 `300` steps、batch `512`、`lr=5e-4`、hidden/layers/state `128/2/32`，synthetic train/val accuracy 与 merge F1 均为 `1.0`。`src/boundary/checkpoints/boundary_refiner.pt` 曾短暂替换为该 checkpoint，sha256 `763ef5cbb66e3d2e49bf1003adccefabf0ff4540aa227b0ff8a6fa866d6d46fe`；现已被 v4 core-only checkpoint 替换。
- 4096 重训后的分发边界：普通推理只需要源码内置 `src/boundary/checkpoints/boundary_refiner.pt`，不需要训练时的 CUDA feature cache、synthetic WAV、frame-sequence JSONL 或 tensor cache。4096 训练完成后已删除大体积 synthetic WAV、feature-cache、frame-sequence JSONL/tensor cache 和 mixed-source manifest，只保留 `keep/*.json`、训练 `metrics.json`、新 checkpoint 与旧 checkpoint 备份；`agents/temp/speech-boundary-ja/v3-boundary-delta-4096/` 当前约 `4.5M`。验证：`tests/test_boundary_refiner_training.py tests/test_boundary_refiner.py tests/test_boundary_cache.py tests/test_boundary_planner.py tests/test_chunk_packer.py tests/test_pipeline_chunk_config_runtime.py` 共 `54 passed`。下一步真实 gate 仍是用匿名样片 A / held-out 重跑 GPU downstream，观察 start attribution、fallback duration、ASR empty、repeat/nonlexical 风险和 forced/fallback 分布。
- 2026-06-09 Boundary Refiner v3 断兼容协议：主线升级为 `boundary_refiner_v3`，Mamba2 head 从 `merge + left_context + right_context` 扩展为 `merge + left_context + right_context + start_refine_delta + end_refine_delta`，`boundary-cache` 升 v3。runtime 取消固定 `BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S`、`BOUNDARY_CONTEXT_MAX_SPEECH_OVERLAP_S`、gap 比例裁剪和 split-overlap 0.25s cap；ASR padded window 只按模型输出 context budget 与 `BOUNDARY_CONTEXT_MAX_PADDING_S` 裁剪，左侧仅保证不小于 0。`PackedChunk` / ASR chunk metadata / cache 显式写入 `boundary_start_refine_delta_s` 和 `boundary_end_refine_delta_s`，这些 delta 现在会真实调整 chunk core start/end，而不是只写诊断字段。训练协议同步升级：`boundary_refiner_frame_sequence_dataset_v3` 必须包含 `sequence_context_targets` 与 `sequence_boundary_delta_targets`；训练 loss 为 merge BCE + context SmoothL1 + start/end delta SmoothL1，start delta 权重大于 end delta。32768 v3 是上一轮大样本实验，不再是当前默认 checkpoint；其 synthetic validation 同样不能替代真实 downstream。
- 2026-06-08 no-runtime-merge + Qwen3-ASR-1.7B bs64 匿名样片 A GPU 闭环完成：命令显式使用 `--asr-backend jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame --asr-batch-size 64 --no-keep-asr-chunks`，输出 `agents/temp/speech-boundary-ja/sample-a-no-runtime-merge-qwen17b-bs64-gpu/`，总耗时约 `317s`，其中 ASR+alignment `275.9s`、ASR transcribe `178.5s`、alignment `57.3s`。ASR batch64 运行中观测显存约 `6.9-7.0GB`，说明适合本机 8GB 实验档，但不应写成 6GB 安全档。结果 `transcript_chunks=3199`、`segments=2359`、`blocks=2359`，确认 runtime merge / subtitle merge 删除后没有再把相邻 cue 合并；QC 为 `reject=6`、`warn=426`、`empty_text_for_speech=414`。诊断输出 `agents/temp/speech-boundary-ja/diagnostics-sample-a-no-runtime-merge-qwen17b-bs64-gpu/`，审计页 `agents/audits/asr-attribution-no-runtime-merge-qwen17b-bs64-video/index.html`，归因计数 forced `1382`、nonlexical `1007`、proportional_after_sentinel `804`、drop/review `6`，density>4cps `724`。批量策略同步更新：源码 / `.env.example` 默认按 6GB 档 `0.6B=64 / 1.7B=32 / aligner=64`；本机 `.env` 使用 8GB 实验档 `0.6B=128 / 1.7B=64 / aligner=128`，OOM 时退回 6GB 档。
- 2026-06-08 runtime merge 策略断兼容删除收尾：按“先删除所有 merge 策略，重跑后再审计”的方向，主流程不再跨 speech island 打包成一个 ASR chunk，`src/boundary/ja/backend.py` 的 raw segment gap merge 改为纯 `filter_segments`，`SPEECH_BOUNDARY_JA_MERGE_GAP_S` 从 cache signature、workflow CLI 和 advanced env 中移除；subtitle writer 已移除 adjacent/dense/overlap cue merge，ASR fragment postprocess merge 也已移除，旧 cue-merge 分析工具/测试移入 `agents/rm/merge-strategy-removal/`。审计页同步改为区分 `ASR padded window` 与 `speech/fallback core`：最新页面中 chunk2 `15.332-17.160` 和 chunk3 `16.660-19.229` 的重叠仅是 ASR padding/context overlap，实际 core 为 `15.460-16.910` 与 `16.910-19.220`，没有 speech-core 重叠。验证：`tests/test_reward_boundary_planner.py tests/test_asr_attribution_audit.py tests/test_run_full_workflow_env.py tests/test_boundary_planner.py` 13 passed。
- 2026-06-08 real-domain silver boundary mining 第一轮正式结果改用 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame` 重跑；误用 0.6B 的旧产物已移到 `agents/rm/20260608_0p6b_silver_boundary_results/`，不作为训练数据。目标不是 runtime 新增一层模型，而是从真实片源挖 forced-aligner 成功、QC ok、词级时间轴可靠的实例数据，作为后续 Mamba2 Boundary Refiner v3 的 display-boundary correction / context budget 监督信号，并为 RL/DPO reward 准备 hard-case 池。实现：主 ASR/aligner 流程显式导出 per-chunk/per-segment `alignment_mode`、`alignment_quality`、`fallback_type`、`fallback_subtype`、`forced_success`、QC/review 标记和 word timing stats，避免后续 miner 继续从日志猜 forced/fallback；新增 `tools/boundary/mine_silver_boundary_labels.py`，输出 `silver_boundary_labels.jsonl`、`hard_cases.jsonl`、`summary.md/json` 和轻量 HTML 审计页。过滤策略：只收 `alignment_quality=forced`、`fallback_subtype=none`、QC ok、非 nonlexical/align-text-empty、非短/重复 vocalization、无 word timing failure 且有正时长 words；短假名台词若 forced/QC 正常允许进入 silver，避免误伤 JAV/Galgame 真实短句。1.7B 两片源日文-only GPU 闭环输出 `agents/temp/speech-boundary-ja/silver-boundary-mining-v1-two-videos-qwen17b/`，ASR+alignment 分别约 `435.5s` / `842.8s`，fallback ratio 约 `0.264` / `0.223`；silver mining 输出 `agents/temp/speech-boundary-ja/silver-boundary-labels-v1-two-videos-qwen17b/` 和 `agents/audits/silver-boundary-labels-v1-two-videos-qwen17b/index.html`，汇总 `chunks=8482`、`silver=3202`、`hard_cases=5280`，start error p50/p90/p95/max `-0.123/0.250/0.250/1.012s`，end error p50/p90/p95/max `0.172/1.018/1.351/3.560s`；alignment quality 为 `forced=4075`、`nonlexical=2366`、`proportional=2018`、`drop_or_review=20`、`partial=3`。同时修复 `tools/boundary/ja/run_full_workflow.py` 的显式 CLI 参数优先级：`--asr-batch-size` / `--aligner-batch-size` / 模型与 boundary 参数现在会覆盖本机 `.env`，避免 1.7B 实验命令记录和实际 batch 不一致；`.env` 仍可保留本机 8GB 更激进配置，`.env.example` 继续面向分发/6GB。
- 2026-06-08 cue start anchor 继续收紧：用户指出可靠 forced word start 应比 segment/cue start 更可信。实现落点在 `src/subtitles/writer.py` 的 subtitle block window：当 cue/block 带有正时长 timed words 时，`start` 锚到 earliest `word.start`，覆盖普通 cue、soft split 后 cue 和 merge 后 cue；没有有效 words 时仍用原 block/speech-core/fallback start。冲突处理继续沿用 start-locked 策略：不推迟当前 cue start，必要时压缩上一条 end 保 2-frame gap。新增回归覆盖单 cue `segment.start` 晚于 first word、以及相邻 cue merge 后仍保留 earliest word start。验证：`tests/test_subtitle_quality_pass.py tests/test_subtitle_options.py tests/test_asr_segmentation_boundaries.py tests/test_subtitle_qc.py` 66 passed。
- 2026-06-08 ASR “低信息”口径断兼容改为中性 `text_density`：Grok MCP 两次返回 403 后回退检索，外部 ASR/字幕经验仍支持“不要只凭文本短/重复就删除”，应结合 `no_speech_prob`、`avg_logprob`、`compression_ratio`、重复度、时长和人工审计；JAV / Galgame 目标域里的短假名、喘息、呻吟、语气词可能是真实可转写内容。实现：`src/asr/qc.py` 将旧 `low_information` profile 改为 `text_density`，枚举改为 `normal_dialogue`、`empty_or_punctuation`、`long_sparse_text`、`short_vocalization_candidate`、`repeated_vocalization_candidate`、`short_kana_dialogue_candidate`；保留行为不变，仍是 `preserve_with_review`，不新增删除规则。`tools/asr/diagnostics/diagnose_asr_alignment.py` 不再把短/重复低信息人声推成 `low_information_text` failure bucket，避免后续 failure manifest / hard-negative 训练把真实目标域人声误当错误；ASR attribution audit 仍保留 `low_info_vocal` 人工标签，但底层数据字段改为 `text_density_*`。回归：`tests/test_asr_qc_signals.py tests/test_asr_alignment_diagnostics.py tests/test_asr_attribution_audit.py tests/test_alignment_failure_manifest.py` 28 passed。
- 2026-06-08 start-locked subtitle timing 策略落地：根据用户修正，字幕观感目标从“start 准、end 可略长”改为 **start 必须准，end 可略短 / 可被压缩**。Grok 复核 Netflix Timed Text Timing Guidelines：in-time 要贴近第一帧音频或 1-2 帧内，字幕间保留 2-frame gap，小间隔主要通过 out-time / chaining 调整；这支持“不要推迟下一条 start，优先压前一条 end”。实现：`_repair_postprocessed_segment_windows()` 不再在 overlap 时把当前 segment start 推到上一条 end，而是压缩上一条 end 到当前 start；`_normalize_subtitle_timeline()` 极端冲突分支不再 shift next start，而是允许前一条 end 压到 start。新增回归覆盖 ASR postprocess 与 subtitle writer 的 start-lock。验证：`tests/test_asr_segmentation_boundaries.py tests/test_subtitle_quality_pass.py tests/test_subtitle_options.py tests/test_pipeline_chunk_packing.py tests/test_subtitle_qc.py tests/test_quality_report_output.py` 76 passed。匿名样片 A 既有产物 replay 显示 prepared cue 仍有少数 `cue_start - first_word_start > 0.2s`，但主要来自已有 aligned segment / cue merge 的词段不一致，不是新 no-overlap 修复继续后移 start；下一步如要彻底归零，需要在 cue merge/segment 构造中保留 earliest word start 作为 start anchor。
- 2026-06-08 v2 context-budget 默认 checkpoint 的匿名样片 A GPU 闭环已完成，日文-only、不翻译，命令入口 `tools/boundary/ja/run_full_workflow.py --video video/匿名样片 A.mp4 --task-name sample-a-full-v2-context-budget-gpu --label v2_context_budget_gpu --subtitle-mode ja --no-keep-asr-chunks`，输出 `agents/temp/speech-boundary-ja/sample-a-full-v2-context-budget-gpu/`。CUDA 链路确认：SpeechBoundary-JA actual device `cuda`，PTM `cuda:0`，ASR backend `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`。结果：`transcript_chunks=3190`、`segments=2600`、`blocks=1796`，ASR+Alignment `283.8s`，pipeline total `326.9s`；ASR batch64 / aligner batch64 未 OOM，运行中 ASR 显存约 `5GB`，符合本机 8GB 与 6GB 分发目标的安全区间。字幕质量：duration p50/p90/p95/max `1.64/3.162/3.516/5.313s`，overlap `0`，ASR empty-for-speech `361`，review uncertain `7`，per-minute `19.97`。chunk 侧 p50/p90/p95/max `1.274/2.668/2.964/5.169s`，fallback speech-core window p50/p90/p95/max `0.98/2.33/2.66/4.91s`，说明 v2 context-budget 已把 fallback 时间窗压到短 speech core，当前瓶颈主要是 cue density、低信息/重复人声和 forced-aligner sentinel，而不是 20-30s 粗 fallback。
- 2026-06-08 GPU 闭环质量报告口径修正：旧 `alignment_fallback_ratio=0.990` 是假高，根因是 QC 把 `Alignment 回退窗口: speech_core` 这种“fallback 插值时间窗来源”误计为 alignment fallback，并且 fallback count 分母使用最终 SRT blocks 而不是 ASR chunk count。已修正 `_alignment_fallback_count_from_log()`：`Alignment 回退窗口` 不再计数，真实失败只按 `even_fallback` / `aligner_vad_fallback` / sentinel / 降级等 chunk marker 去重；`write_quality_report()` 改用 `asr_details.chunk_count` 作为 fallback ratio 分母，并在报告中写出 `alignment_fallback_count` / `alignment_fallback_total`。用同一份 匿名样片 A run log 重算后：final alignment mode 为 forced `1412`、even_fallback `1008`、nonlexical `770`，修正后 `alignment_fallback_ratio=1008/3190=0.316`。这仍是需要观察的 forced-aligner 域不匹配/短非词文本问题，但不是 `0.99` 级别失败。回归：`tests/test_subtitle_qc.py tests/test_quality_report_output.py tests/test_quality_report_per_job.py tests/test_pipeline_chunk_packing.py::test_alignment_fallback_count_deduplicates_chunk_log_markers` 20 passed。
- 2026-06-08 learned context budget 正式 v2 checkpoint 已完成并替换默认分发文件。数据重新生成到 `agents/temp/speech-boundary-ja/v2-context-budget/`：mixed source 为 anime NSFW/SFW `60/40` 后再按 anime/galgame `65/35` 抽样，synthetic `32768` 条，hard negatives 包含 real negatives/background、touch/short/regular gaps、overlap、gain/filter/codec aug；summary：`cut_point_segment_count=64019`、`cut_drop_zone_count=57337`、`overlap_mix_count=6539`、`background_mix_count=21277`。CUDA/bf16/no-compress feature cache 使用 `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`，`batch_size=32`、`prepare_workers=8`，`cached=32768`、`errors=0`、`skipped=0`，训练中间 cache 约 `196G`。frame-sequence dataset：`32768` 条 sequence、`161815` 个 boundary item，`merge_positive=32768`、`split_negative=129047`，`feature_dim=630`，`feature_schema_hash=ab5ca6f85b2e4da013a7ffaeef0ba73ccbd598a6`。Mamba2 CUDA 训练 `600` steps、`batch_size=512`、`lr=3e-4`、`weight_decay=0.01`、hidden/layers/state `128/2/32`、`context_loss_weight=0.25`、`target_domain_speedup=1.5`、`target/core/padded=3.0/5.0/6.5s`，final loss `0.001233`；train accuracy/F1 `1.0/1.0`，validation accuracy `0.999938`、merge precision `1.0`、recall `0.999695`、F1 `0.999847`。新 checkpoint 已复制到 `src/boundary/checkpoints/boundary_refiner.pt`，sha256 `4e075f388426086cfb020c57f4f02c65cbd312a73b968ac9e17c5df7159f7841`，大小 `2.2MB`；加载 smoke 与 `tests/test_boundary_refiner.py tests/test_boundary_refiner_training.py tests/test_boundary_planner.py tests/test_boundary_cache.py` 共 `36 passed`。
- 2026-06-08 learned context budget 方案落地：按本地 `japanese-anime-speech-v2` 统计，NSFW split 实际 `20849` 条、p50/p80/p90/p95/max `8.65/12.17/14.38/16.31/35.32s`，SFW `40000` 条、p50/p80/p90/p95/max `4.44/6.73/8.16/9.54/28.18s`；结论是不把 NSFW p90/p95 当 runtime speech-core 上限，而是继续 `target/core=3.0/5.0s`，把长 NSFW 作为 hard negative/长句拆分训练素材。Boundary Refiner schema 断兼容升级为 `boundary_refiner_v2`：Mamba2 head 从单一 merge logit 扩为 `merge_logit + left_context_s + right_context_s`，`BOUNDARY_PLANNER_TARGET_PADDING_S` 删除，新增 `BOUNDARY_CONTEXT_MAX_PADDING_S=1.5`、`BOUNDARY_CONTEXT_MAX_SPEECH_OVERLAP_S=0.25`，默认 padded cap 收紧到 `BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S=6.5`；boundary-cache 升 v2。早期默认 checkpoint 曾先迁移为 v2 中性 context 结构以保持源码可运行，现已被正式 v2 context-budget 重训 checkpoint 替换。
- 2026-06-07 Boundary Refiner 分发边界确认：推理 / 新用户分发只需要源码内置的 `src/boundary/checkpoints/boundary_refiner.pt`（约 `2.2MB`）加 Hugging Face 下载的 0.6B full SFT ASR / frozen feature 模型和 forced aligner；32768 重训过程中生成的 CUDA feature cache、synthetic WAV、sequence JSONL 和 `datasets/train/...` 都是可重建训练产物，不参与普通推理。PyInstaller spec 已把 `src/boundary/checkpoints/` 改为 required data include，并显式收集 `transformers.models.mamba2`，避免 release 缺 checkpoint 或 Mamba2 lazy import。
- 2026-06-07 README 分发化精简：README 删除真实番号 / 匿名样片 A 指标、Boundary Refiner 训练路线、Galgame 100k 参数公式、失败实验、审计标签细节和历史路径记录，只保留新用户需要的安装、使用、默认模型、当前工作流、目录结构、常见问题和开发入口。实验依据、路线取舍、指标和调试过程继续只写 HISTORY，避免 README 变成实验日志。
- 2026-06-07 匿名样片 A full-run 回归：先跑 60s smoke，CUDA 链路正常，`30` ASR chunks、`23` aligned segments、`19` SRT blocks，total `21.9s`。随后全片第一次卡在 `_build_processing_spans()`，根因是 `FrameSequenceFeatureProvider.features_for_gap()` 每个 gap 都重复把整片 PTM/MFCC frame list 转成 numpy array，长片候选多时变成 CPU-bound。修复为 provider 初始化时缓存 PTM/MFCC numpy arrays、使用 `_gap_window_sequence_features_from_arrays()` 复用整片 frame buffer，并新增 `test_frame_sequence_feature_provider_caches_frame_arrays` 回归。修复后全片完整跑通：输出 `agents/temp/speech-boundary-ja/sample-a-full-default-boundary-refiner-fixed-seqfeat-cache/`，`transcript_chunks=3178`、`segments=4240`、`blocks=2105`、ASR+Alignment `422.0s`、total `457.8s`；ASR 显存约 `4.8-5.2GB`，符合 6GB 目标。质量报告：subtitle duration p50/p90/p95/max `1.65/3.392/3.533/5.323s`，`per_min_subtitle_count=23.4`，`alignment_fallback_ratio=0.811`，`asr_empty_text_for_speech_count=470`。本轮确认 Boundary Refiner 可完成全片，下一步质量瓶颈仍是 cue density、低信息/重复人声和 forced-aligner fallback 观察项。
- 2026-06-07 README 旧路线表述清理：active tree 已没有 speaker sidecar / runtime diarization / WhisperSeg / TEN / Silero / FSMN / 旧 FusionVAD 运行入口；README 不再写“默认不启用这些路线”，避免让新用户误以为仍存在可选开关。旧路线说明继续只保留在 HISTORY。
- 2026-06-07 `speaker_proxy` 数据与训练链路断兼容清理：不再把 speaker proxy / speaker turn 当作 Boundary Refiner synthetic metadata。旧 `datasets/train/fusionvad-ja/v1-23-boundary-refiner/` 已移到 `agents/rm/speaker-proxy-datasets-20260607/`，中断/错误权重产物移到 `agents/rm/interrupted-boundary-dataset-20260607/` 和 `agents/rm/wrong-weight-boundary-dataset-20260607/`，旧 anime 512/256 小源池移到 `agents/rm/obsolete-small-anime-boundary-sources-20260607/`。active `datasets/` 扫描 `speaker_proxy|speaker_turn|speaker_changed|split_speaker` 为 0。代码层同步删除 `speaker_proxy_id()`、`--speaker-proxy-*`、`speaker_proxy_ids`、`speaker_turn_boundaries`、`split_speaker_*` label，改为 `utterance_boundaries` + `cut_point`。新本地源池：`datasets/train/boundary-sources/japanese-anime-speech-v2-nsfw-60k/` 实际 `valid_rows=20849`（HF nsfw split 已耗尽，无法达到 60000 唯一条）、`datasets/train/boundary-sources/japanese-anime-speech-v2-sfw-40k/` 为 `40000` 条、Galgame 100k 继续用 `datasets/train/boundary-sources/galgame-asr-100k-ogg/`。按 anime 内部 `nsfw:sfw=60:40` 有放回抽样生成 `datasets/train/speech-boundary-ja/v1-boundary-refiner/anime-source-nsfw60-sfw40-100k/`，再按 `anime=65 / galgame=35` 随机生成 `mixed-source-anime65-galgame35/`，最终 4096 synthetic timeline 输出 `mixed-anime65-galgame35-boundary4096/`：`records=4096`、`utterance_boundary_count=16384`、`cut_point_segment_count=11524`、`cut_drop_zone_count=4813`、`real_negative_gap=12299`、`overlap_mix=320`。
- 2026-06-07 32768 hard-negative Boundary Refiner 重训：按用户要求从 anime65/galgame35 mixed source 生成更强 hard negatives，输出 `datasets/train/speech-boundary-ja/v1-boundary-refiner/mixed-anime65-galgame35-hardneg32768-v2/`，`records=32768`、WAV 约 `34G`、`utterance_boundary_count=163840`、`cut_point_segment_count=123964`、`cut_drop_zone_count=31760`、`background_mix_count=20244`、`overlap_mix_count=5808`，gap modes：`real_negative=150507`、`fade_noise=10160`、`silence=10132`、`white_noise=8479`、`hum=4356`。随后用 `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame` CUDA/bf16 生成 no-compress feature cache，路径 `.../mixed-anime65-galgame35-hardneg32768-v2-feature-cache-nocompress/`，`cached=32768`、`errors=0`、`skipped=0`，体积约 `214G`。frame-sequence dataset `boundary-refiner-frame-sequence-hardneg32768-speedup15-v2/`：`32768` 条 sequence、`225464` 个候选 item，`merge_positive=65536`、`split_negative=159928`，label reasons 为 `merge_synthetic_intra_island=65536`、`split_cut_point=69468`、`split_gap_zone=30101`、`split_long_gap=1334`、`split_overlap=59025`，`feature_dim=630`，`feature_schema_hash=ab5ca6f85b2e4da013a7ffaeef0ba73ccbd598a6`。Mamba2 CUDA 训练 `600` steps、`batch_size=512`、`lr=3e-4`、`weight_decay=0.01`、hidden/layers/state `128/2/32`，产物 `boundary-refiner-frame-sequence-mamba2-hardneg32768-speedup15-v2/boundary_refiner.pt`。validation：accuracy `0.999911`、merge precision/recall/F1 `0.999847`、FP `1`、FN `1`。该 checkpoint 已替换默认分发文件 `src/boundary/checkpoints/boundary_refiner.pt`，sha1 `0497912ff24ac63fec6c4c94bb9dea513490a3a5`，体积约 `2.2MB`。
- 2026-06-07 32768 训练内存坑修复：首次训练直接读取 `3.5G` `sequences.jsonl` 时，旧 `tools/boundary/train_refiner.py` 会把 JSON rows 全量解析成 Python `list[dict]`，再构造 padded tensor；WSL2 8GB RAM 被吃满并使用 swap，最终系统卡崩且没有有用 traceback。现已改为两遍流式扫描：第一遍只读取 schema / feature_names / max_len / class counts / metadata，第二遍直接填紧凑 `torch.Tensor`；训练集用 `Subset(TensorDataset(...))`，评估按 batch/index streaming，不再复制整份 train/val tensor。回归测试新增 `test_train_refiner_uses_streaming_tensor_loader`，防止未来把训练入口改回整量 JSON rows。
- 2026-06-07 Mamba2 Boundary Refiner 4096 版基线：先用 `mixed-anime65-galgame35-boundary4096/` 生成 Qwen3-ASR-0.6B full SFT CUDA feature cache，路径 `datasets/train/speech-boundary-ja/v1-boundary-refiner/qwen3-asr-0.6b-fullsft/mixed-anime65-galgame35-boundary4096-feature-cache-nocompress/`，结果 `cached=4096`、`errors=0`、`skipped=0`。随后生成 `boundary-refiner-frame-sequence-speedup15-v1/`：`4096` 条 sequence、`20660` 个 boundary item，`merge_positive=4096`、`split_negative=16564`，`feature_dim=630`，`feature_schema_hash=ab5ca6f85b2e4da013a7ffaeef0ba73ccbd598a6`，feature signature 显式包含 `target_chunk_s=3.0`。CUDA 训练 `300` steps，`batch_size=512`、`lr=5e-4`、hidden `128`、layers `2`、state `32`、backbone `transformers.Mamba2Model`，产物 `boundary-refiner-frame-sequence-mamba2-speedup15-v1/boundary_refiner.pt`。validation：accuracy `0.998545`、merge precision `0.997555`、merge recall `0.995122`、F1 `0.996337`、FP `1`、FN `2`。该版曾短暂作为默认分发 checkpoint，现已被 32768 hard-negative v2 checkpoint 替换。
- 2026-06-07 Boundary Refiner 训练指标口径修复：`tools/boundary/train_refiner.py` 的 `_evaluate()` 过去把 padded sequence slots 也计入 `accuracy/count` 分母，导致 accuracy 被低估；现改为只统计 `mask` 内有效 boundary items。`_hash_from_signature()` 同步断兼容要求 `feature_signature.feature_config.target_chunk_s` 必填，避免旧 sequence feature signature 缺字段时静默复用默认值。
- 2026-06-04 命名决策并执行断兼容迁移：当前系统不再适合继续叫 `FusionVAD-JA`。它已经不是严格 speech/non-speech VAD，而是 `Qwen PTM + MFCC/energy bootstrap frame scores -> boundary candidate extraction -> Boundary Refiner scoring -> constrained planner -> ASR chunks` 的 speech-island boundary system。新概念名定为 **SpeechBoundary-JA**，backend key 为 `speech_boundary_ja`；active package 已从 `src/vad/fusionvad_ja/` / `tools/vad/fusionvad_ja/` 迁到 `src/boundary/ja/` / `tools/boundary/ja/`，不保留 `fusionvad_ja` alias、旧 cache 或旧路径兼容。
- 2026-06-04 破坏式重构：旧 `ASR_PRE_ASR_*` / R15-R23 规则 packer、旧 `ASR_CHUNK_PACK_*` 配置和旧 `vad-cache` 语义已从 active code、测试和 Web 可见配置中移除。当前主线是 `SpeechBoundary-JA frame probabilities -> boundary candidate extraction -> Boundary Refiner scoring/context budget -> constrained boundary planner -> ASR chunks`；cache 当前为 `boundary-cache v2`，signature 包含 SpeechBoundary-JA、Qwen feature/PTM、Boundary Refiner、candidate extractor、planner/context config 和 `BOUNDARY_FEATURE_FRAME_HOP_S`。
- 2026-06-04 SpeechBoundary-JA schema 断兼容迁移：feature cache / manifest / checkpoint / model layer 统一改为 `ptm`、`ptm_dim`、`ptm_proj`，不再沿用早期 `whisper_*` 命名；仓库内 v1.17、v1.19b、v1.21 小 checkpoint 已迁移 state_dict key。旧 feature cache 和旧实验 checkpoint 不做兼容迁移，重新生成。
- 2026-06-04 配置面清理：旧 `VAD_MIN_OFF` / `VAD_PAD` / `SEGMENT_*` 已从 active defaults、`.env.example`、boundary-cache signature 和 Web advanced 透传中移除；当前只保留语义明确的 `ASR_CHUNK_MIN_DURATION_S`（导出 wav chunk 最小时长）和 `ASR_CONTEXT_RESET_GAP_S`（滑动 ASR 文本上下文重置 gap）。
- 2026-06-04 审计发现并修复 chunk metadata 错位：`_extract_wav_chunks` 过滤过短 span 后，`_annotate_packed_chunks` 过去按输出位置 zip `PackedChunk`，会把 `vad_seg_count` / boundary reason/source/score 错贴到后一个 chunk；现在 chunk info 记录 `source_span_index`，annotation 按原始 span index 对齐，并补回归测试。
- 2026-06-04 timing 语义修正：Boundary / ASR chunk planning 采用秒级上下文参数，`BOUNDARY_FEATURE_FRAME_HOP_S` 只表示 VAD/frame-score 网格 fallback，默认 `0.02s`；字幕显示 / timing polish 单独按真实视频 FPS 计算 `frame_duration_s` 和 Netflix-style `2-frame gap`。主流程不再把视频 FPS 注入 Boundary 配置，aligned-cache 签名改为 v3，并在 `subtitle` 签名中记录 `video_fps` / `frame_gap_s`。
- 2026-06-04 BiLSTM 路线断兼容删除：旧 `AdditionFusion*BiLSTM`、v1.17/v1.19b/v1.21 endpoint / imitation checkpoint、addition / endpoint / imitation 训练导出 CLI、drop-gap imitation offline packer 和旧大测试文件已移出 active tree 到 `agents/rm/bilstm-removal/`。当前 `src/boundary/ja/` 模块只做 Qwen frozen feature + MFCC / energy bootstrap frame scoring；边界决策主线只走 `src/boundary/` 的 Boundary Refiner / Mamba2。
- 新主线模块结构：`src/boundary/features.py` 组装帧级特征，`candidates.py` 提取 gap midpoint / cut peak / low-score valley 候选，`refiner.py` 提供 BoundaryRefiner interface 和 bootstrap refiner，`backbones.py` 放 Windows-friendly Mamba2 research wrapper，`planner.py` 做 constrained planning，`cache.py` 负责 boundary-cache v2。`src/boundary/ja/` 放 JA 目标域 bootstrap scorer。`src/audio/chunk_packer.py` 只保留把 planner 输出和 learned context budget materialize 成 `PackedChunk` 的 ASR-facing 职责。
- 2026-06-04 Boundary Refiner backbone 入口收束到唯一实现路径 `transformers.Mamba2Model`：learned checkpoint schema 当前为 `boundary_refiner_v2`，runtime / CLI / cache signature / checkpoint payload 都使用该值。它直接对应 Hugging Face Transformers 的纯 PyTorch Mamba2 wrapper；`mamba2`、`torch_mamba2`、BiGRU、TCN、Transformer fallback 不再作为可选入口，避免训练和分发形成多套模型协议。
- 2026-06-04 Sequence feature schema authority 落地：`src/boundary/sequence_features.py` 统一提供 `frame_sequence_features_v1` 的 default config、feature dim、feature names、`feature_schema_hash` 和 train/runtime validation。`runtime_adapter=frame_sequence_v1` checkpoint 必须带 `feature_schema`、`feature_schema_hash` 和 `feature_signature`；主 pipeline 会用 SpeechBoundary-JA 导出的 PTM/MFCC frame windows 重新计算 names/hash，不匹配直接 fail-fast，不做旧 checkpoint alias 或静默回退。`PackedChunk` / boundary-cache / ASR chunk metadata 已写入 `boundary_decision_merge`、`boundary_merge_prob`、`boundary_split_prob`、`boundary_refine_delta_s` 和 `boundary_decision_source`，供后续 QC、forced alignment 和审计页追踪决策依据。
- 2026-06-07 runtime speaker / sidecar 断兼容清理：运行时人声聚类、声纹 sidecar、Web speaker 显示、speaker-aware cue merge、cue planner speaker score、speaker sidecar 工具/测试和低能量 pre-ASR drop 已从 active tree 移除或移到 `agents/rm/speaker-runtime-removal/`。字幕层只保留 Netflix-style timing/readability/fallback 风险、2-frame gap、显示时长和短 cue 合并这些观感核心；`speaker_proxy` / `speaker_turn` 不再进入 runtime 决策、用户配置或 Boundary Refiner synthetic metadata。
- 2026-06-07 翻译缓存分层：Grok / 官方文档检索确认 provider prompt cache 依赖“稳定长前缀 + 变量放后”的 exact-prefix 机制，适合降低 API token/latency，但不解决本地复跑 cache miss。当前本地翻译改为 `translation_cache.jsonl` 精确 batch cache + `translation_cache.memory.jsonl` 文本级 translation memory：一级仍绑定 cue timing / batch / prompt signature 用于 crash resume；二级绑定 normalized JA text、target language、normalized glossary / auto glossary、character reference、prompt version 和 model family，允许同一任务 Boundary / cue timing 调整后复用译文。低信息或单字符循环文本不写入 memory，避免目标域呻吟/短促发声被过度复用。timings 中区分 `translation_cache_hit`、`translation_memory_hit`、`translation_memory_hit_count` 和 provider prompt-cache usage，避免把三种缓存混为一谈。参考来源：OpenAI Prompt Caching、Claude Prompt Caching、DelTA 多级翻译记忆、2024 BUCC 字幕自适应翻译 fuzzy-match 研究。
- 2026-06-07 翻译链路审计修复：短片 / 审计片段过去会走 `single_request_full_context`，但该 path 没有接 `cache_path`，因此不会命中本地 batch cache / translation memory，也不会写入二级 memory；现在 single-request 与 batched path 共用缓存语义。另修 prompt user message 硬编码“中文字幕”的问题，改为使用 `TARGET_LANG`；自动全片 glossary cache 从固定 `translation_global_glossary.json` 改为 `translation_global_glossary.<source_hash>.json`，避免共享 translation cache 路径时串用上一部片的自动术语表。
- 2026-06-07 项目级运行目录断兼容从 `temp/` 改为 `tmp/`：`JOB_TEMP_DIR=./tmp/jobs`，ASR chunk/checkpoint 为 `./tmp/chunks`，boundary cache 为 `./tmp/cache/boundary`，torch / HF 运行缓存为 `./tmp/cache/torch` 与 `./tmp/cache/hf`。`agents/temp/` 继续只用于研究脚本和审计中间产物，不纳入用户运行缓存结构。旧 `temp/` 不做 alias 或迁移逻辑，避免长期维护两套 runtime root。
- 2026-06-06 Galgame 100k 本地源池落地：先用 streaming duration scan 读取 `litagin/Galgame_Speech_ASR_16kHz` 前 100000 条，只解析 OGG header，不保存音频，确认分布为 mean/p50/p75/p90/p95/p99/max `5.136/4.504/6.937/9.560/11.353/15.187/28.981s`，`>=5s/9s/12s/15s/20s` 比例 `0.441/0.123/0.039/0.011/0.001`，与 dataset card 全量平均约 `5.145s` 一致。随后按用户要求把这 100000 条保留在本地，输出 `datasets/train/boundary-sources/galgame-asr-100k-ogg/`，保存原始压缩 OGG + TXT + `manifest.jsonl`，不转 WAV。校验：manifest `100000` 行、OGG `100000` 个、TXT `100000` 个、errors `0`，summary 记录实际 OGG 字节 `2.313 GiB`，文件系统占用约 `3.0G`（10 万小文件有块占用）。这批数据作为后续 Boundary Refiner / synthetic speech-island dataset 的本地源池，避免每轮构造都重新流式下载 HF。基于该分布新增可复算公式与工具 `tools/boundary/recommend_timing_params.py`：`target_core=round_0.1(clamp(p50/speedup,2.0,3.5))`、`max_core=floor_0.5(clamp(p80/speedup,target+1.0,5.5))`、`target_padding=round_0.1(clamp((p90-max_core)/2,1.0,2.0))`、`max_padded=floor_0.5(min(p90,max_core+2*padding,9.0))`、`min_chunk=round_0.05(clamp(p5/speedup*0.60,0.25,0.50))`。当前 `target_domain_speedup=1.5` 推导出 `target/core/padded/min/padding = 3.0/5.0/9.0/0.4/2.0s`，与现行默认一致；后续换成 anime 混合源或重新采样时直接换 summary 复算。
- 2026-06-05 fallback 时间轴窗口修正：soft-candidate DP v2 已把 20s+ 粗 chunk 消掉，但剩余长 fallback 多数是 `core_duration_s≈8-9s` 被 2s 左右 ASR padding 显示成 `12-13s`。主流程现在继续给 ASR 输入 padded chunk 保留识别上下文，但在 forced aligner 失败、走比例/VAD fallback 时间戳时，只在 speech core 窗口插值；`alignment_fallback_start_s/end_s/source` 会随 text_result 进入 `LocalAsrBackend.finalize_text_results()`，retry / sentinel fallback 分支也使用同一窗口。完整 GPU 闭环实测确认：padded fallback p50/p90/max 仍是 `8.73/12.60/13.10s`，但实际 fallback 时间轴窗口 p50/p90/max 已降到 `6.10/8.76/9.10s`，safe ratio `0.732`，`>10s` fallback 消失。
- 2026-06-05 JAV 短字幕 core 策略收紧：公开检索未找到可靠 JAV 单句字幕时长统计；可用参考是 Netflix timing / Japanese 读速、OOONA、ATA、Karamitroglou 等通用字幕规范，它们都更支持短 cue、贴近 in-time 和长音频拆分。本地 `agents/audits/fallback-window-risk-audit-video/manual_fallback_window_risk_labels.jsonl` 40 条 `8.75-9.10s` fallback-window 人工审计中，`needs_split=35/40`、`multiple_islands=11/40`、`timing_end_early=11/40`。因此 `9s` 不再作为 subtitle/fallback speech-core 默认上限，只保留为 ASR padded context 上限；新默认是 `BOUNDARY_PLANNER_TARGET_CHUNK_S=3.0`、`BOUNDARY_PLANNER_MAX_CORE_CHUNK_S=5.0`、`BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S=9.0`。旧 `BOUNDARY_PLANNER_MAX_CHUNK_S` 断兼容删除，不保留 alias。
- 2026-06-05 匿名样片 A 短 chunk 批量与显存监控：基于已生成的 2459 个 ASR chunks 做阶段 benchmark。0.6B ASR-only 256 chunk：batch `32` 用时 `32.25s`、峰值 `4272 MiB`；batch `48` 用时 `29.62s`、峰值 `5139 MiB`；batch `64` 曾未监控跑通 `29.50s`，但本次 0.2s 持续 `nvidia-smi` 采样在 `6200 MiB` 超过 6GB 目标并被中断；batch `128` 之前已因 `CUDA driver error: device not ready` 失败。1.7B ASR：batch `48` 峰值 `7178 MiB`，batch `24` 峰值 `6327 MiB`，batch `16` 峰值 `6119 MiB`（当时空闲底噪约 `1.1GB`）；用户当时决定 1.7B 默认保守用 `12`。aligner-only 全量 2459 chunk：batch `16` 用时 `83.72s`、峰值 `3259 MiB`；batch `32` 用时 `80.78s`、峰值 `3864 MiB`；batch `48` 用时 `75.71s`、峰值 `4349 MiB`，均无错误。该条保留为当时 benchmark 记录；当前默认档位以后续 2026-06-08 的 6GB / 8GB 配置记录为准。
- 2026-06-06 ASR context 后处理断兼容删除：匿名样片 A 审计发现带人名 `ASR_CONTEXT` 时，旧 fragment-level prompt/context leak 规则会把真实自我介绍从最终 cue 中删掉，只剩问候句；但词级 `words` 已有对应时间轴。结论是该规则在 JAV / Galgame 目标域误伤成本高、价值低。主流程已删除 `context leak` QC reason、相似度阈值、fragment 删除函数和 Web/env advanced 前缀；`ASR_CONTEXT` / `ASR_HEAD_CONTEXT` 只作为 ASR 提示词，不再驱动最终字幕删除或 QC reject。字幕层继续依赖空文本、纯标点、低信息/重复、信号质量、fallback 和人工审计等更直接的信号。
- 2026-06-06 F0/gender 与最终文本突变清理：旧 F0/gender route、`run_asr_alignment_f0` 历史命名、Web `show_gender` 可见项和相关测试已从 active tree 移除。ASR QC `reject_count` / `review_uncertain`、重复循环 `suggested_text` 和低信息 profile 均保持 review-only；审计页标签从“采用去重建议”改为“重复建议需复核”，点击不会再自动写 `manual_text`。字幕 merge 不再去重 overlap，保留目标域真实重复语气词、呻吟和短促发声。
- Windows 分发约束：默认路线不依赖 Linux-only `mamba-ssm`、Triton 或自定义 CUDA kernel。`src/boundary/backbones.py` 只提供 Hugging Face Transformers `Mamba2Model` 的纯 PyTorch wrapper 作为研究 backbone。
- Unified Joint Model 路线进入 backlog，不进入当前重构：未来可考虑在 Qwen3-ASR decoder 中加入 `<boundary>` / `<dramatic_pause>` / `<sentence_end>` 等 token，做 joint segmentation + transcription 或 Samba-ASR 类长上下文模型。但这需要重新准备 boundary supervision、ASR SFT/RL/DPO 和云端 GPU 训练，当前维护成本和训练成本高，不作为默认路线。
- active backend key 已改为 `speech_boundary_ja`，旧 `fusionvad_ja` key / package / path 不做兼容。
- 默认 ASR backend key 已切到 Hugging Face repo ID 本身：`jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`；可选 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`。短 key 不再进入主线，避免 Web/API/cache/download 出现两套命名。
- SpeechBoundary-JA frozen feature 默认使用 `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`，不再默认下载 base 0.6B。
- Boundary Refiner 默认 learned checkpoint 固定为 `src/boundary/checkpoints/boundary_refiner.pt` 并随源码分发；旧 `src/vad` checkpoint 只作为历史产物保存在本地回收区。
- 破坏式维护重构：`src/whisper/` 改为 `src/asr/`，`tools/fusionvad_ja/` 按职责拆到 `tools/asr/`、`tools/vad/`、`tools/subtitles/`、`tools/audits/`；Whisper/WhisperSeg/TEN/Silero/FusionLite 当前主线代码和测试移入 `agents/rm/obsolete-mainline-cleanup-20260603/`。
- 当前主线不再是 high-recall proposal VAD，也不再是固定 gap packer，而是 SpeechBoundary-JA / Boundary Refiner 驱动的 speech-island boundary pipeline：ASR 前 chunk 要尽量接近一句台词，避免长连续 chunk、内部 gap、多 speech island 和非语音多送诱发 ASR 空输出、非语音幻觉和 forced aligner sentinel。
- 边界优先级：`start` 略高于 `end`，但两者都要进 gate；允许为了切准 speech island 牺牲少量 frame recall，但不能漏掉完整台词 island。
- 下一步应把 v1.20-v1.23 的经验收敛到 learned Boundary Refiner：显式优化 start/end error、fallback chunk duration、gap crossing、单 chunk 台词数和 ASR/aligner QC reward；recall 继续作为 guardrail，而不是唯一主目标。
- 现行 `tools/` 已按职责重构：`tools/asr/qwen/` 放 Qwen SFT，`tools/asr/diagnostics/` 放 ASR/alignment 诊断，`tools/boundary/` 放 Boundary Refiner 数据和训练，`tools/boundary/ja/` 放 SpeechBoundary-JA 训练评测，`tools/subtitles/` 放字幕 postprocess / cue planner / 审计校准，`tools/audits/` 放审计页与人工审计工具。旧历史段落里的 `tools/vad/fusionvad_ja/...` 路径保留为当时记录。

### 历史计划快照：SpeechBoundary-JA v2/v3

以下计划是 v5 delta-only 之前的路线快照，已被当前 `frame-sequence Mamba2 start/end delta -> core planner -> ASR` 主线取代；保留是为了追踪断兼容重构的来源。

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

<details>
<summary>历史：FusionVAD-JA / R14-R18 早期 VAD 与 chunk packing 记录</summary>

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

- 2026-06-20 1.7B scorer workflow / CueQC 标注源 / REAL 性能口径：1.7B scorer 已作为默认进入 10 部 workflow，输出根目录 `agents/temp/speech-boundary-ja/20260620_005837_qwen17b-scorer-default-10film-cueqc-source/`，10 部全部 `done`，合计 segments `18210`、blocks `18098`，均使用 `speech_boundary_ja_feature_scorer.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt` 且 scorer repo id 匹配 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`。本轮实际参数：ASR batch size `32`，Boundary planner sequence batch size `256`，CueQC inference batch size 默认 `64`；Boundary / SpeechBoundary feature extraction另有内部窗口和 Qwen feature extractor `max_inference_batch_size=1`，不要和 ASR 转写 batch 混淆。
- CueQC 1.7B 标注源：从 10 部 aligned output 导出全量候选 `19552` 条；为避免 Torque 全量 O(n^2) 距离矩阵过大，第一轮只聚类高风险 density level（`empty_or_punctuation`、`short_vocalization_candidate`、`repeated_vocalization_candidate`），共 `4157` 条。按 NAMH-055 原型沿用 `structured + euclidean + merge-layer 1` 得到 `104` 个簇，产物在 `agents/temp/20260620_025639_cueqc-qwen17b-10film-cluster-source/cluster_highrisk_structured_layer1/`，其中 `cluster_label_audit.html` 用于按簇打 `drop/keep/mixed/skip` 标签，`cluster_label_template.jsonl` 是空标签模板。REAL-988 继续保留为测试集，不进入这批训练/标注源。
- CueQC 正向样本补充规则：如果簇级高风险标签之外还需要混合正向样本，正向样本从 anime 数据集抽取，NSFW:SFW 按 `1:1` 比例混合；REAL-988 仍只作为测试集，不进入训练或正向样本池。后续 CueQC 训练数据构造需在 metadata/summary 中记录 anime 来源、NSFW/SFW 计数和抽样 seed。
- REAL-988 后续性能测试口径：主 workflow 已新增 CUDA memory snapshot，写入 `timings.json -> asr_details.cuda_memory`（ASR 内部 split/load/transcribe/cueqc/unload/align/segment/total）和 `asr_details.pipeline_cuda_memory`（run/audio/asr/write 外层阶段），run log 同步打印 allocated/reserved/max_reserved/free/total MB；`tools.workflows.run_full_workflow` summary 会汇总 `cuda_memory_peak_reserved_mb`、`pipeline_cuda_memory_peak_reserved_mb`，并暴露 `--cueqc-inference-batch-size`，便于 REAL-988 用同一入口比较 ASR bs、Boundary sequence bs 和 CueQC bs。下一步测试 REAL 时先跑当前基线（ASR bs 32 / boundary seq 256 / CueQC bs 64），再只改一个 batch 参数做对比，重点看 `asr_text_transcribe_done`、`cueqc_done` 和 `max_reserved_mb` 是否接近共享显存边界。
- 2026-06-20 审计导航清理：正式审计页已放入 `agents/audits/20260620_025639_cueqc-qwen17b-10film-cluster-label-audit/index.html`，导航 `agents/audits/index.html` 和 `latest-audit.html` 已重建并指向该页；旧审计目录统一移动到 `agents/rm/audit-deletions/20260620-103946-*`，未硬删除。
- 验证：`uv run python -m py_compile src/asr/pipeline.py src/main.py tools/workflows/run_full_workflow.py` passed；`uv run python -m tools.workflows.run_full_workflow --help` 已确认 `--asr-batch-size`、`--boundary-planner-sequence-batch-size`、`--cueqc-inference-batch-size` 均可见。

</details>

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

<details>
<summary>历史：R19-v1.23 / DP / context-budget / cue-density 试验记录</summary>

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
- 匿名样片 A `.env` 正常加载短 core 完整闭环：配置为 `BOUNDARY_PLANNER_TARGET_CHUNK_S=3.0`、`BOUNDARY_PLANNER_MAX_CORE_CHUNK_S=5.0`、`BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S=9.0`，全片 chunks `2459`、aligned segments `3794`、原 ja-only SRT blocks `3233`、ASR+alignment `371.0s`、total `406.2s`。fallback core p50/p90/p95/max `1.20/3.07/3.86/5.00s`，unsafe fallback `0`、safe ratio `1.0`。结论：边界/fallback 粗时间轴在匿名样片 A 上已达安全口径，当前主要问题转为日文 cue density 和低信息人声/重复语气词，而不是继续把 ASR chunk 放粗。
- cue-density runtime 修复：发现 `prepare_srt_blocks()` 过去不是单次收敛。第一次 prepare 先按 raw alignment gap 合并，再 timing polish 把部分短 gap 压到 2-frame gap；这些新可合并的 micro cues 只有手动第二遍 prepare 才会合并。现在 `_prepare_subtitle_blocks()` 在 timing polish + final no-overlap normalize 后，再执行一次受 `SUBTITLE_MERGE_ADJACENT` 控制的 bounded short-cue merge，并立刻做最终 normalize。`prepare_srt_blocks()` 同时断开旧的 `mode == bilingual` 隐式开关，日文-only / 双语统一由 `SubtitleOptions.merge_adjacent` 控制；dense cue merge 也受同一个总开关约束。
- 匿名样片 A ja-only cue replay 结果：工具 `tools/subtitles/replay_subtitle_postprocess.py` 新增 `--source aligned` / `--mode srt|bilingual` 后，用最新 aligned segments 重放。final merge 前：blocks `2166`、per-minute `24.01`、short_segment_ratio `0.1316`、duration p50/p90/max `1.447/3.533/5.65s`。final merge 后单次 prepare 已直接得到 blocks `2002`、per-minute `22.19`、short_segment_ratio `0.1169`、duration p50/p90/max `1.733/3.533/5.65s`、overlap `0`、nonlexical repetition count `14`。
- cue planner analyzer 断兼容扩展：`tools/subtitles/analyze_subtitle_cue_merge_candidates.py` 新增 `--aligned`、`--source blocks|aligned`、`--mode srt|bilingual`，可直接分析 ja-only aligned segments，并按模式导出 `before/planner.{ja,bilingual}.srt/.vtt`。在匿名样片 A final-merge 后复算，额外 planner merges 为 `0`、blocks `2002 -> 2002`；这说明当前最大收益来自 runtime final merge，而不是 analyzer 的临时 planner score。后续 cue-density 路线应继续研究 low-info vocal 合并/标记、speaker-aware merge guard 和更强 cue planner，不要把这次收益误归因给 `min_score=0.62` 的 planner heuristic。
- 验证：`tests/test_subtitle_quality_pass.py tests/test_subtitle_options.py tests/test_replay_subtitle_postprocess.py tests/test_subtitle_cue_merge_candidates.py` 52 passed。
- ASR 低信息/重复/幻觉归因审计入口落地：新增 `tools/audits/generate_asr_attribution_audit_html.py`，从 alignment diagnostics + aligned segments + 完整日语 SRT 离线抽样，不重跑 ASR/aligner。采样桶固定为 `repeat_or_qc_reject`、`nonlexical_empty`、`sentinel_fallback`、`low_info_vocal`、`asr_qc_warn`、`forced_control`，用于区分真实低信息人声、ASR 幻觉/错听、非语音噪声/BGM、多人/重叠、轻声弱人声、边界上下文过短/过长、文本可用和时间轴准确等原因。匿名样片 A 当前 diagnostics 全量统计：chunks `2459`，alignment quality 为 forced `1200`、proportional `861`、nonlexical `378`、drop_or_review `20`；ASR QC ok/warn/reject 为 `2102/337/20`；fallback subtype 为 none `1200`、proportional_after_sentinel `861`、nonlexical_text `378`、asr_qc_reject `20`；low-information 分布为 not_low_information `1132`、short_kana `462`、short_nonlexical `385`、empty `378`、repeated_nonlexical `98`、long_sparse `4`。已生成音频审计页，共 `84` 条、每桶 `14` 条，完整日语 VTT 和人工导出文件 `manual_asr_attribution_labels.jsonl`；导航 latest 已指向该页。该步骤是评估闭环：先看人工分布，再决定是否改 ASR QC、cue planner、speaker sidecar、hard-negative 训练或 0.6B/1.7B 稳定性对比。
- 验证：`tests/test_asr_attribution_audit.py tests/test_audit_nav.py tests/test_long_fallback_audit_media_mode.py` 7 passed；`python -m py_compile tools/audits/generate_asr_attribution_audit_html.py` passed。

</details>

---

## 历史降级记录

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
- 10k+ sequence dataset 不能把 JSONL 先全量解析成 Python `list[dict]` 再转 tensor；32768 hard-negative 的 `3.5G` JSONL 曾直接吃满 8GB RAM + swap 并卡崩。训练入口必须使用流式扫描 / 紧凑 tensor / batched evaluation。v3 以后训练脚本支持 `--tensor-cache-path`：第一次只流式扫描 JSONL 并写 `.pt` tensor cache，后续直接读取 tensor cache；cache 内带源 JSONL path/size/mtime 指纹，源文件变动会自动拒绝复用。WSL2 8GB 下训练 10k+ sequence 时应始终显式设置该参数。
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

- 2026-06-20 CueQC 1.7B cluster audit blank-page 修复：`agents/audits/20260620_025639_cueqc-qwen17b-10film-cluster-label-audit/index.html` 数据本身完整，空白原因是导出按钮 JS 里 JSONL newline 被写成未转义换行，导致整段脚本加载中断。已修为 `join("\\n") + "\\n"`，静态执行确认脚本可 parse，最小 DOM render 可生成 `104` 个簇按钮、`4` 个决策按钮和首簇 `5` 个代表样本。审计导航仍指向该页；`tools/audits/serve_audits.ps1` / `.sh` 默认加 `--no-browser` 并打印导航 URL，避免 live-server 在 Windows/Codex 环境自动拉浏览器失败影响审计服务。
- 2026-06-20 CueQC 1.7B cluster audit playback/schema 修复：用户指出审计页必须能逐字幕播放对应片段和上下文，且 CueQC 是 keep/drop 二分类自动路由。已用 `tools/audits/generate_cueqc_cluster_audit_html.py` 重建 `agents/audits/20260620_025639_cueqc-qwen17b-10film-cluster-label-audit/index.html` 为完整音频审计页：`review_item_count=4157`、`cluster_review_group_count=104`、支持“播放 chunk / 播放上下文”、完整 VTT、chunk/fallback/context cue 列表，cluster 导出只保留 `display_decision in {keep, drop}`。同时修复生成器旧式 `jobs/{video_id}_b5/audio` 假设，改为优先从 candidate 的 `source_audio_path` / `audio.path` 解析真实音频；验证 10/10 视频 audio 与 subtitle 均存在，`missing_media_videos=[]`、`missing_subtitle_videos=[]`。
- 2026-06-20 CueQC cluster audit media 参数断兼容：`tools/audits/generate_cueqc_cluster_audit_html.py` 不再接受 `--baseline-root`，也不再按旧 `jobs/{video_id}_b5/audio` 命名猜媒体路径。生成完整 CueQC 簇级 keep/drop 审计页时必须显式传 `--archived-root`（包含 `<video_id>/<video_id>.ja.srt` 与 aligned segments）和一个或多个 `--media-root`（递归搜索 job/audio/media 根目录）；candidate 中的 `source_audio_path` / `audio.path` 仍作为显式行级路径优先级。README 工具索引已同步，测试覆盖自定义 job 名媒体目录。
- 2026-06-20 CueQC prediction audit media/active sampling 参数断兼容：`tools/audits/generate_cueqc_prediction_audit_html.py` 已同步为显式 `--archived-root` + 可重复 `--media-root`，不再接受旧 `--baseline-root`，summary 记录 `archived_root/media_roots`。该页继续用于 CueQC Stage 2 / 主动学习中的 false-drop 人工审计，默认只在模型判定为 drop 的池内混合抽样高置信、近阈值、风险桶和随机监控样本，并把 `audit_sample_reason/audit_risk_bucket/drop_margin` 写入审计项与导出标签；未审 drop 仍不进训练。测试覆盖自定义媒体根目录和 mixed sampling，避免后续 1.7B 闭环被旧 job 命名假设或单一高置信抽样卡住。
- 2026-06-20 CueQC 输入特征口径确认：CueQC v3-Fusion 不是只看 ASR 文本的分类器。训练集编译保留 `text_features`、`cue_features`、`boundary`、`adjacency`、`asr_signals`、`subtitle_timing`；特征抽取 / runtime 还会通过 ASR internals 构建 teacher-forced `token_trace`、`decoder_stats`，并与 ASR encoder features、structured metadata 一起进入 `CueQCMambaV3Fusion` 的 ASR arm、token arm、decoder arm、structured arm，最终输出 `keep/drop` 二分类。README 模型表已同步该口径。
- 2026-06-20 CueQC cluster audit 卡顿修复：10 部高风险簇审计页的源音频是 101-434MB 的整片 WAV，旧页面在“全部音频”里给当前簇每条样本都创建一个 `<audio preload="metadata">`，大簇会同时触发大量长音频 metadata/range 请求，表现为页面未响应或点击播放无反应。生成器已改为单主播放器懒加载：样本列表只渲染“播放 chunk / 播放上下文 / 打开详情”按钮，每簇默认分页显示 80 条，主播放器移出 hidden legacy 容器，字幕 cue 高亮刷新降到 0.5s 粒度；summary 标记 `cluster_review_audio_render_mode=single_player_lazy`。
- 2026-06-20 CueQC 1.7B cold-start 标注策略回收为 0.6B 式高精 seed：Grok/弱监督方向和 0.6B 实践一致，cluster 只负责第一次高精粗标签启动，不应覆盖全量或把混簇全量广播。当前 10 部 4157 条高风险 cluster 审计页视为探索/候选池；训练集构造只接收人工显式 `seed_action=use_seed` 且 `display_decision in {keep,drop}` 的簇，`mixed_skip` / `skip` 为 abstain，不产生训练标签。后续采用 “小 seed model -> 全量预测 -> 每轮约 200 条人工 false-drop/uncertainty/风险桶/随机混合审计 -> 再训练” 的主动学习闭环，未审高置信 drop 仍不直接进训练，keep recall / false-drop safety 优先。
