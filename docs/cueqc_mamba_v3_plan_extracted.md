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

## Remaining V3 Plan

- Continue false-drop / keep-recall audits before accepting any new high-confidence drop pseudo labels into training.
- If more self-training is needed, repeat the Stage 2 loop: predict on full pool, audit false drops, compile accepted labels, retrain, replay all historical false-drop audits, then promote only if keep recall remains safe.
- Compile confirmed `display=drop` and short invalid chunks into a separate Boundary hard-case / preference dataset. This Stage 3 task has not been executed and must stay offline; CueQC decisions must not be coupled into Boundary runtime planning. `display=drop` is not a direct v5.1 `start_delta/end_delta` label. It must first be converted into one of two training sources: SpeechBoundary-JA frame-level negative examples for pure non-speech/short noise, or Boundary Refiner preference/hard-case samples for over-fragmented chunks that should be shortened, merged into neighbors, or suppressed by a better boundary choice.
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
- `tools/audits/generate_cueqc_cluster_audit_html.py`
- `tools/audits/generate_cueqc_prediction_audit_html.py`
