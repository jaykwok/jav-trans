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
- Stage 3 Boundary hard-case candidate export: `tools.boundary.export_cueqc_drop_hardcases` 已把三轮 false-drop 审计导出到 `agents/temp/20260617_214103_boundary-hardcase-candidates-from-cueqc/`。原始标签 `600` 行，去重后 `578` 个 unique item，其中 `551` 个 confirmed drop candidate，`27` 个 safety holdout；候选 route 为 `speech_boundary_frame_negative_candidate=522`、`boundary_preference_candidate=29`。该产物仍是候选 manifest，不是可直接训练 v5.1 的 delta dataset。
- Stage 3 v5.1 source preparation: `tools.boundary.prepare_cueqc_drop_v51_sources` 已使用三份审计页的 `cueqc_prediction_audit_items.jsonl` 补回音频路径，并生成 `agents/temp/20260617_220854_boundary-v5.1-sources-from-cueqc/`。`522/522` 个 frame-negative candidate 已切出短音频并生成 SpeechBoundary-JA negative labels/training manifest，`29` 个 Boundary preference seed 已保留上下文字幕和 aligned segments；`missing_audit_item_samples=0`、`frame_negative_skipped=0`。该步骤仍不生成 `boundary_refiner_frame_sequence_dataset_v5`。
- Stage 3 SpeechBoundary-JA hard-negative readiness: `tools.boundary.prepare_speech_boundary_hard_negative_finetune` 的 negative-only gate 先确认 522 条 negative examples 全部可解析，但缺少 anchor 时 `formal_training_ready=false`。随后 `tools.boundary.ja.build_positive_anchor_replay` 已按 `anime_nsfw=55 / anime_sfw=20 / galgame=25` 生成 `agents/temp/20260617_230636_speech-boundary-positive-anchor-replay/`，共 1500 条 positive anchors，实际 `anime_nsfw=825/anime_sfw=300/galgame=375`。混合 gate 包 `agents/temp/20260617_230948_speech-boundary-hard-negative-finetune-prep/` 已输出 `speech_boundary_mixed_hard_negative_anchor_labels.jsonl` / manifest，`records=2022`、`trainable_examples=2022`、`negative_share=0.25816`，`formal_training_ready=true`。当前停止在训练开启前，尚未跑 feature cache 或训练。
- Stage 3 SpeechBoundary-JA mixed feature cache / tiny plumbing train: 2026-06-18 已执行 `build_mixed_feature_cache.ps1`，feature summary 为 `cached=2022/skipped=0/errors=0`，label qualities `negative=522/supervised=1500`。随后执行 `tiny_mixed_plumbing_train.ps1`，输出 `tiny-mixed-hard-negative-anchor/speech_boundary_ja_tiny.pt`，metrics 为 `steps=200/loss=0.607988/frame_accuracy=0.73/positive_ratio=0.73`。这只是当前 `train_tiny` research/plumbing artifact，不替换 `qwen-feature-energy-bootstrap-v1` runtime。

## Remaining V3 Plan

- Continue false-drop / keep-recall audits before accepting any new high-confidence drop pseudo labels into training.
- If more self-training is needed, repeat the Stage 2 loop: predict on full pool, audit false drops, compile accepted labels, retrain, replay all historical false-drop audits, then promote only if keep recall remains safe.
- Next SpeechBoundary-JA step is deciding whether to implement a real runtime-loadable scorer / formal evaluation path; the existing `train_tiny` checkpoint must stay local and non-promoted. Do not promote any scorer without full downstream smoke and human audit.
- Convert the `boundary_preference_seed_candidates.jsonl` rows into neighbor-context A/B or explicit start/end delta targets before Boundary Refiner v5.1 training. CueQC decisions must stay offline and must not be coupled into Boundary runtime planning.
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
- `tools/boundary/prepare_cueqc_drop_v51_sources.py`
- `tools/boundary/prepare_speech_boundary_hard_negative_finetune.py`
- `tools/boundary/ja/build_positive_anchor_replay.py`
- `tools/audits/generate_cueqc_cluster_audit_html.py`
- `tools/audits/generate_cueqc_prediction_audit_html.py`
