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
  --output agents/temp/YYYYMMDD_HHMMSS_cueqc-v3-10film-full-features/shards/shard_00000.pt `
  --device auto `
  --start-index 0 `
  --max-samples 1000 `
  --audio-cache-size 1

uv run python -B tools/asr/cueqc/merge_features_v3_fusion.py `
  --input agents/temp/YYYYMMDD_HHMMSS_cueqc-v3-10film-full-features/shards/shard_00000.pt `
  --output agents/temp/YYYYMMDD_HHMMSS_cueqc-v3-10film-full-features/cueqc_full_features_v3_fusion.pt

uv run python -B tools/asr/cueqc/predict_v3_fusion.py `
  --features agents/temp/YYYYMMDD_HHMMSS_cueqc-v3-10film-full-features/cueqc_full_features_v3_fusion.pt `
  --checkpoint src/asr/checkpoints/cueqc_mamba_v3_fusion.pt `
  --output-dir agents/temp/YYYYMMDD_HHMMSS_cueqc-v3-10film-predictions `
  --device auto
```

For the 45643-row pool, run feature extraction in shards (`--start-index` +
`--max-samples`) and merge them with `merge_features_v3_fusion.py`; a single
all-in-one extraction has no intermediate save point and is too costly to retry.
The extractor caches decoded full-video wav arrays while processing a shard
(`--audio-cache-size`, default `1`), which avoids rereading the same 100-400MB
wav for every chunk while keeping memory bounded.
`extract_features_v3_fusion.py --input` writes unlabeled samples with label `-1`.
`train_mamba_v3_fusion.py` rejects unlabeled bundles; run `predict_v3_fusion.py`
first, audit false-drop risk from `cueqc_pseudo_labels.high_conf.jsonl`, then
merge accepted pseudo labels with the cold-start seed before retraining.

After the false-drop audit, compile Stage 2a features without rerunning ASR
internals:

```powershell
$env:PYTHONIOENCODING='utf-8'
uv run python -B tools/asr/cueqc/compile_stage2a_features_v3_fusion.py `
  --cold-start-features agents/temp/20260617_cueqc-mamba-v3-fusion/cueqc_train_features_v3_fusion.pt `
  --full-features agents/temp/20260617_113159_cueqc-v3-10film-sharded-features/cueqc_full_features_v3_fusion.pt `
  --pseudo-labels agents/temp/20260617_130642_cueqc-v3-10film-predictions/cueqc_pseudo_labels.high_conf.jsonl `
  --false-drop-audit-labels agents/audits/20260617_130642_cueqc-v3-false-drop-audit/cueqc_false_drop_audit_labels.jsonl `
  --output agents/temp/YYYYMMDD_HHMMSS_cueqc-v3-stage2a-training-features/cueqc_stage2a_train_features_v3_fusion.pt
```

Stage 2a policy:

- Manual false-drop audit labels override cold-start seed labels.
- `drop_ok` becomes `drop=0`; `false_drop_keep` becomes `keep=1`.
- `uncertain` is skipped.
- High-confidence `keep` pseudo labels are included.
- Unaudited high-confidence `drop` pseudo labels are skipped.

2026-06-17 run status:

- Full 10-film feature extraction completed in 46 shards under
  `agents/temp/20260617_113159_cueqc-v3-10film-sharded-features/`.
- Merged bundle:
  `agents/temp/20260617_113159_cueqc-v3-10film-sharded-features/cueqc_full_features_v3_fusion.pt`
  (`45643` unlabeled records).
- Prediction output:
  `agents/temp/20260617_130642_cueqc-v3-10film-predictions/`.
- Prediction counts: `keep=31055`, `drop=14588`; high-confidence pseudo labels:
  `14629` total (`drop=14588`, `keep=41`).
- Because high-confidence drop volume is large, do not retrain directly from
  these pseudo labels. First audit false-drop risk via
  `agents/audits/20260617_130642_cueqc-v3-false-drop-audit/index.html`.
- User audit export:
  `agents/audits/20260617_130642_cueqc-v3-false-drop-audit/cueqc_false_drop_audit_labels.jsonl`.
- Audit result: `200` reviewed, `178 drop_ok`, `21 false_drop_keep`,
  `1 uncertain`; raw false-drop rate `10.5%`. Stratified population-weighted
  estimate over `drop>=0.85` is about `4.2%`, still too high for accepting all
  `14588` high-confidence drop pseudo labels.
- Stage 2a compiled bundle:
  `agents/temp/20260617_143121_cueqc-v3-stage2a-training-features/cueqc_stage2a_train_features_v3_fusion.pt`.
  Records: `538`, labels `drop=344/keep=194`. Sources:
  `298` cold-start seed, `178` manual `drop_ok`, `21` manual false-drop keep,
  `41` high-confidence keep pseudo. The `14588` unaudited drop pseudo labels
  were skipped.
- Stage 2a training output:
  `agents/temp/20260617_143911_cueqc-v3-stage2a-train/cueqc_mamba_v3_fusion.pt`.
  At `drop_threshold=0.85`, fixed holdout `867HTTM-0045` still has
  `keep_recall=0.9375` and `false_drop_rate=0.0625`, so it is not safe to
  replace the default checkpoint.
- Threshold scan shows `drop_threshold=0.88` reaches holdout
  `keep_recall=1.0 / false_drop_rate=0.0` with lower `drop_recall=0.618`.
  The t=0.88 10-film prediction output is
  `agents/temp/20260617_144200_cueqc-v3-stage2a-10film-predictions-t088/`:
  `keep=35936/drop=9707`, high-confidence pseudo labels
  `drop=9707/keep=1507`.
- A new t=0.88 false-drop audit is required before replacing the runtime
  checkpoint:
  `agents/audits/20260617_144327_cueqc-v3-stage2a-t088-false-drop-audit/index.html`.
  In this state, keep `src/asr/checkpoints/cueqc_mamba_v3_fusion.pt` unchanged.
- The t=0.88 audit result is improved but not clean enough:
  `197 drop_ok / 3 false_drop_keep`, false-drop rate `1.5%`.
  Raising threshold to `0.92` avoids these audited false drops but leaves only
  `337` predicted drops, so threshold-only replacement is not useful.
- Stage 2b compiled bundle:
  `agents/temp/20260617_161704_cueqc-v3-stage2b-training-features/cueqc_stage2b_train_features_v3_fusion.pt`.
  Records: `2177`, labels `drop=532/keep=1645`. Sources:
  `375` manual `drop_ok`, `24` manual false-drop keep, `1504` high-confidence
  keep pseudo, plus remaining cold-start seed. The `9707` unaudited t=0.88 drop
  pseudo labels were skipped.
- Stage 2b training output:
  `agents/temp/20260617_161806_cueqc-v3-stage2b-train/cueqc_mamba_v3_fusion.pt`.
  Fixed holdout `867HTTM-0045` reaches `keep_recall=1.0`,
  `false_drop_rate=0.0`, `drop_precision=1.0`, `drop_recall=0.8475`.
  Replay over the first two audit rounds keeps all `24` manually corrected
  `false_drop_keep` samples.
- Stage 2b full prediction:
  `agents/temp/20260617_162231_cueqc-v3-stage2b-10film-predictions/`.
  Counts: `drop=19456/keep=26187`. Because the model is more aggressive on the
  full pool, a fresh false-drop audit is required before replacing the runtime
  checkpoint:
  `agents/audits/20260617_162350_cueqc-v3-stage2b-false-drop-audit/index.html`.
  Keep `src/asr/checkpoints/cueqc_mamba_v3_fusion.pt` unchanged until that gate
  is reviewed.
- The Stage 2b false-drop audit page includes reason tags:
  `dialogue`, `vocalization`, `breath`, `environment`, `overlap`,
  `short_fragment`. `breath / 呼吸声` was added after the initial Stage 2b page
  generation and the same audit directory was regenerated before manual labels
  were exported.

## Stage 3: Boundary Feedback

Later, `display=drop` chunks can be mined as Boundary preference / hard-case data:

- very short noise chunks
- invalid speech islands
- chunks that should be merged into neighbors
- repeated no-subtitle-value fragments

The current false-drop audit has `178` manually confirmed `drop_ok` chunks.
They are good candidates for a later Boundary hard-negative / preference
finetune set, but they are not compiled into Boundary training yet. Keep this
as a separate Stage 3 task after CueQC Stage 2a retraining is evaluated.

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
- `tools/asr/cueqc/merge_features_v3_fusion.py`
- `tools/asr/cueqc/predict_v3_fusion.py`
- `tools/asr/cueqc/compile_stage2a_features_v3_fusion.py`
- `tools/asr/cueqc/train_mamba_v3_fusion.py`
- `tools/audits/generate_cueqc_cluster_audit_html.py`
