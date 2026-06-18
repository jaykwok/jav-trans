# CueQC Mamba v3-Fusion Plan

本文件是当前 CueQC 权威计划。`cueqc_mamba_v2_plan.md` 只保留为历史废案，不再作为实现依据。实验流水账和取舍背景记录在 `HISTORY.md`，本文件只保留当前边界、已执行状态和剩余计划。

## 当前边界

- CueQC v3-Fusion 替代旧规则 ASR QC。
- 旧规则 QC 不再运行、不再观测、不再参与候选排序、质量报告或 fallback subtype。
- Runtime 只接受二元路由：`keep` / `drop`。
- 模型不可用、加载失败或推理异常时一律 fallback `keep`。
- 不再打 `content_type`、`qc_decision`、`alignment_policy`、`compact`、`review` 训练标签。
- 不引入 BGE、sentence-transformers、HuBERT、UMAP/HDBSCAN/FINCH 或额外 audio embedding 作为默认依赖。
- Torque Clustering 只作为第一次 cold-start 粗标签工具，不进入 runtime、自训练或长期 QC 分类器。

## Runtime Flow

```text
SpeechBoundary-JA / Boundary Refiner
-> ASR speech-core chunks
-> Qwen ASR text
-> CueQC v3-Fusion keep/drop decision
-> drop: skip alignment and subtitle output
-> keep: forced alignment / subtitle timing polish
```

CueQC 在 ASR 文本之后、forced alignment 之前运行。只有 `mode=cueqc_mamba_v3_fusion` 且 `display_hint=drop` 的模型决策可以过滤 chunk；fallback / heuristic shadow 不允许丢弃。

## Model Inputs

v3-Fusion 使用现有 ASR backend 内部状态，不加载第二份 Qwen3-ASR：

- ASR encoder features
- ASR token trace
- decoder aggregate stats
- structured chunk metadata

这些特征由 `src/asr/asr_internals.py`、`src/asr/cueqc_features.py` 和提取脚本生成。旧规则 QC 不作为特征源。

## Model Output

Checkpoint schema: `cueqc_mamba_checkpoint_v3_fusion`

Runtime decision schema:

```json
{
  "decision_version": "cueqc_display_binary_v1",
  "model_version": "cueqc_mamba_v3_fusion",
  "mode": "cueqc_mamba_v3_fusion",
  "display_hint": "keep",
  "confidence": 0.0,
  "display_prob_keep": 0.0,
  "display_prob_drop": 0.0,
  "drop_threshold": 0.85,
  "threshold_profile": {},
  "reasons": []
}
```

Training labels are encoded as:

- `drop = 0`
- `keep = 1`

The safety objective is conservative subtraction: maximize keep recall and minimize false drop rate, accepting lower drop recall during cold start.

## Executed State

- Bootstrap pool: CueQC 合入前 baseline commit `5afe535` 的 10 部全片，不翻译、保留 ASR chunks；NAMH-055 固定为 smoke / holdout，不进入 10-film training pool。
- Candidate pool: `agents/temp/20260615_152934_cueqc-10film-candidates/cueqc_candidates.full.jsonl`，共 `45643` 条；分层抽样 `300` 条用于 cold-start。
- Bootstrap clustering: Torque Clustering `--merge-layer 1` 得到 7 簇；用户完成簇级 `keep/drop` 标注后广播成 cold-start seed。旧 HDBSCAN/FINCH/UMAP/PCA/embedding 路线已退役。
- Cold-start training: 300 条种子最终 labels `keep=133/drop=167`；首轮过拟合模型废弃，保守配置作为 Stage 2 起点。
- Stage 2 feature pool: `agents/temp/20260617_113159_cueqc-v3-10film-sharded-features/cueqc_full_features_v3_fusion.pt`，46 个 shard 合并，`45643` unlabeled records。
- Stage 2 audits: 三轮 false-drop 审计合计 `600` 条，`drop_ok=573 / false_drop_keep=25 / uncertain=2`。
- Stage 2b default checkpoint: `src/asr/checkpoints/cueqc_mamba_v3_fusion.pt`，sha1 `98f9631a63dc19736b50619100fb4be4d08075e8`。
- Adaptive threshold profile: base `drop_threshold=0.85`，`text_bucket=short_text` 提升到 `0.87`，由 `src/asr/cueqc_thresholds.py` 在 runtime 与 offline prediction 共用。
- Adaptive 10-film prediction: `agents/temp/20260617_174344_cueqc-v3-stage2b-adaptive-10film-predictions/`，records `45643`，`drop=19380/keep=26263`，high-confidence pseudo `drop=19380/keep=1372`。
- NAMH-055 Web full-workflow smoke after Stage 2b promotion: job `38a5d14ea1a54236a24b56e716f36175` completed with `translation_skipped=true`，CueQC `candidates=3199/drop=1052/keep=2147/fallback=0`，最终 `transcript_chunks=2147/segments=2157/blocks=2087`，输出 `video/NAMH-055.ja.srt`，summary 为 `agents/temp/20260617_191654_web-smoke-namh055-cueqc-batched/job_summary.json`。
- Stage 3 SpeechBoundary hard-negative candidate export: 2026-06-18 已按“人工 `drop_ok` 全部直接丢弃”的断兼容规则重新导出 hardnegative-only manifest。三轮 false-drop 审计原始标签 `600` 行，去重后 `578` 个 unique item，其中 `551` 个 confirmed drop candidate、`27` 个 safety holdout；`551/551` 全部 route 为 `speech_boundary_frame_negative_candidate`。分类标签（对话、呻吟、呼吸、环境、短碎片等）只保留为审计元数据，不再决定 route，也不再生成 Boundary Refiner 训练标签。
- Stage 3 SpeechBoundary hard-negative source preparation: 2026-06-18 已用 hardnegative-only manifest 切出短音频并生成 SpeechBoundary-JA negative labels/training manifest。`551/551` 个 frame-negative candidate 已 materialize，`missing_audit_item_samples=0`、`frame_negative_skipped=0`；该步骤不生成 Boundary Refiner 训练标签。
- Stage 3 SpeechBoundary-JA hard-negative replay: 三轮 CueQC false-drop 审计得到的 `551` 条人工 `drop_ok` 已确认都是直接丢弃监督，但这些样本来自旧启发式 chunk，因此不再作为 first scorer bootstrap 训练输入。当前只保留为后续自训练/finetune 阶段的 replay material；`tools.boundary.prepare_speech_boundary_hard_negative_replay` 不再生成 scorer 训练脚本。
- Stage 3 SpeechBoundary-JA v2/neg20-s800 结论：旧 Mamba2 speech-only scorer v2 和 `neg20-s800` 诊断候选已废弃。失败根因是模型只输出 speech score，runtime `cut_probs=0`，长 speech island 会回落到约 3 秒机械切分，导致 NAMH-055 开头“中出し”被拆坏并引发 ASR 误识别。相关 checkpoint 和审计页只保留为失败证据，不作为训练、推广或 runtime 候选。
- Stage 3 SpeechBoundary-JA first scorer v3: 当前唯一 active 路线是 `speech_boundary_ja_mamba2_frame_boundary_scorer_v3`，Mamba2 输出 `[speech_prob, cut_prob]` 两路 frame score。训练目标来自 synthetic true-structure timeline：`speech_frames` 监督 speech head，`cut_point_segments` 与 `cut_drop_zones` 并集监督 cut/drop gate。旧 v1/v2 checkpoint schema、旧 `mamba2_frame_scorer` model type、旧单通道输出和缺失 `output_dim=2` 的 payload 均 fail-fast 拒绝。
- Stage 3 SpeechBoundary-JA v3 runtime policy: 默认 runtime 仍是 `qwen-feature-energy-bootstrap-v1`；v3 scorer 只通过 `SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT` opt-in 加载，runtime signature 为 `score_model=mamba2_frame_boundary_scorer`。任何 v3 scorer 必须先经过离线 speech/cut threshold eval、NAMH-055 smoke、人工对比审计和至少一个额外 held-out full-film smoke，才可讨论默认替换。

## Remaining V3 Plan

- Continue false-drop / keep-recall audits before accepting any new high-confidence drop pseudo labels into training.
- If more self-training is needed, repeat the Stage 2 loop: predict on full pool, audit false drops, compile accepted labels, retrain, replay all historical false-drop audits, then promote only if keep recall remains safe.
- Build a larger synthetic true-structure first-scorer dataset using the existing v5-style random timeline generator (`tools.boundary.ja.build_galgame_synthetic_timeline`) and local anime/galgame source manifests. This dataset replaces old heuristic-derived hard negatives as the bootstrap source.
- Generate Qwen PTM + MFCC feature cache for that synthetic dataset, then train `speech_boundary_ja_mamba2_frame_boundary_scorer_v3` with two heads: speech and cut/drop gate.
- Run offline threshold eval for both heads. Speech threshold must prioritize recall / low false positive rate; cut threshold must be evaluated separately because cut failure causes mechanical island splitting.
- Run NAMH-055 no-translate smoke against the default baseline, then generate a manual compare page before any default replacement discussion.
- CueQC `drop_ok` decisions stay offline SpeechBoundary-JA replay material and must not be coupled into Boundary Refiner training or first-scorer bootstrap.
- Revisit forced `forced/native/hybrid` aligner A/B separately if model size or cost becomes a priority. CueQC v3-Fusion does not decide aligner removal.

## Active Files

- `src/asr/cueqc.py`
- `src/asr/cueqc_features.py`
- `src/asr/cueqc_model.py`
- `src/asr/cueqc_refiner.py`
- `src/asr/cueqc_thresholds.py`
- `src/asr/asr_internals.py`
- `src/asr/pipeline.py`
- `tools/asr/cueqc/cluster_candidates.py`
- `tools/asr/cueqc/torque.py`
- `tools/asr/cueqc/compile_training_set.py`
- `tools/asr/cueqc/extract_features_v3_fusion.py`
- `tools/asr/cueqc/merge_features_v3_fusion.py`
- `tools/asr/cueqc/predict_v3_fusion.py`
- `tools/asr/cueqc/compile_stage2a_features_v3_fusion.py`
- `tools/asr/cueqc/train_mamba_v3_fusion.py`
- `tools/boundary/export_cueqc_drop_hardcases.py`
- `tools/boundary/prepare_cueqc_drop_hard_negative_sources.py`
- `tools/boundary/prepare_speech_boundary_hard_negative_replay.py`
- `tools/boundary/ja/build_positive_anchor_replay.py`
- `tools/boundary/ja/build_galgame_synthetic_timeline.py`
- `tools/boundary/ja/prepare_frame_boundary_scorer_v3.py`
- `tools/boundary/ja/build_feature_cache.py`
- `tools/boundary/ja/train_feature_scorer.py`
- `tools/boundary/ja/evaluate_feature_scorer_thresholds.py`
- `tools/audits/generate_cueqc_cluster_audit_html.py`
- `tools/audits/generate_cueqc_prediction_audit_html.py`
