# CueQC Mamba v3-Fusion Plan

本文件是当前 CueQC 权威计划。`cueqc_mamba_v2_plan.md` 只保留为历史废案，不再作为实现依据。

## 当前边界

- CueQC v3-Fusion 替代旧规则 ASR QC。
- 旧规则 QC 不再运行、不再观测、不再参与候选排序、质量报告或 fallback subtype。
- Runtime 只接受二元路由：`keep` / `drop`。
- 模型不可用、加载失败或推理异常时一律 fallback `keep`。
- 不再打 `content_type`、`qc_decision`、`alignment_policy`、`compact`、`review` 训练标签。
- 不引入 BGE、sentence-transformers、HuBERT、UMAP/HDBSCAN/FINCH 或额外 audio embedding 作为默认依赖。

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
  "reasons": []
}
```

Training labels are encoded as:

- `drop = 0`
- `keep = 1`

The safety objective is conservative subtraction: maximize keep recall and minimize false drop rate, accepting lower drop recall during cold start.

## Bootstrap Clustering

Torque Clustering is only a one-time bootstrap tool for roughly 300 sampled chunks:

1. Cluster sampled candidates.
2. Human labels each cluster as `keep` or `drop`.
3. `compile_training_set.py --cluster-labels cueqc_cluster_labels.jsonl` broadcasts cluster labels to sample rows.
4. Train the first v3-Fusion checkpoint.

Important constraints:

- Clustering does not enter runtime.
- Clustering does not enter self-training.
- Mixed clusters are acceptable seed noise.
- No `--method` compatibility layer.
- No per-sample `cueqc_manual_labels.jsonl` compatibility.

## Stage 2: Self-Training

After cold start:

1. Run CueQC v3-Fusion on the 10-film candidate pool.
2. Collect high-confidence `keep` / `drop` pseudo labels.
3. Manually audit false drop risk.
4. Expand the training set.
5. Retrain CueQC v3-Fusion.

Primary acceptance metrics:

- keep recall
- false drop rate
- drop precision / recall as secondary metrics

Current Stage 2 tooling:

```powershell
$env:PYTHONIOENCODING='utf-8'
uv run python -B tools/asr/cueqc/extract_features_v3_fusion.py `
  --input agents/temp/20260615_152934_cueqc-10film-candidates/cueqc_candidates.full.jsonl `
  --audio-root agents/temp/20260615_094437_b5/agents/temp/speech-boundary-ja/20260615_094437_o10 `
  --output agents/temp/YYYYMMDD_HHMMSS_cueqc-v3-10film-full-features/cueqc_full_features_v3_fusion.pt `
  --device auto

uv run python -B tools/asr/cueqc/predict_v3_fusion.py `
  --features agents/temp/YYYYMMDD_HHMMSS_cueqc-v3-10film-full-features/cueqc_full_features_v3_fusion.pt `
  --checkpoint src/asr/checkpoints/cueqc_mamba_v3_fusion.pt `
  --output-dir agents/temp/YYYYMMDD_HHMMSS_cueqc-v3-10film-predictions `
  --device auto
```

`extract_features_v3_fusion.py --input` writes unlabeled samples with label `-1`.
`train_mamba_v3_fusion.py` rejects unlabeled bundles; run `predict_v3_fusion.py`
first, audit false-drop risk from `cueqc_pseudo_labels.high_conf.jsonl`, then
merge accepted pseudo labels with the cold-start seed before retraining.

## Stage 3: Boundary Feedback

Later, `display=drop` chunks can be mined as Boundary preference / hard-case data:

- very short noise chunks
- invalid speech islands
- chunks that should be merged into neighbors
- repeated no-subtitle-value fragments

This remains an offline preference-training loop. CueQC decisions must not be coupled into Boundary runtime planning.

## Active Files

- `src/asr/cueqc.py`
- `src/asr/cueqc_features.py`
- `src/asr/cueqc_model.py`
- `src/asr/cueqc_refiner.py`
- `src/asr/asr_internals.py`
- `src/asr/pipeline.py`
- `tools/asr/cueqc/cluster_candidates.py`
- `tools/asr/cueqc/compile_training_set.py`
- `tools/asr/cueqc/extract_features_v3_fusion.py`
- `tools/asr/cueqc/predict_v3_fusion.py`
- `tools/asr/cueqc/train_mamba_v3_fusion.py`
- `tools/audits/generate_cueqc_cluster_audit_html.py`
