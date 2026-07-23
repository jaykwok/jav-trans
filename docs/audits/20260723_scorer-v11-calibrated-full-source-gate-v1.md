# Scorer v11 calibrated full-source gate 与监督分布审计

## 最终结论

固定 25 条 Gemini Protect×Remove 双独立证据已成功编入 Scorer v11 canonical，但由此训练的 P2048/H256 基线不能晋升。失败不是显存、训练时长或 Mamba 容量问题，而是现实 full-source 监督在 source 内缺少同时出现的 inside/outside，模型学到了“真实 full-source 大多整体保留”的 provenance shortcut。

当前 checkpoint SHA256 为 `3a6f38678fd21eaa11b95b73fd05fbba51a261c277671e2410492fc32e476c12`，固定 `numeric_gate_pass=false / promotion_allowed=false`，production registry 未修改。0.6B 未读取、修改或训练。

## 固定合同

- 中央序列化合同：`boundary_acoustic_binary_v12`。
- 架构：raw PTM2048 → trainable full-width adapter → hidden256 bidirectional Mamba2 → Linear(2)。
- 训练/runtime：two-logit softmax argmax；plain CE；`unsure=-100`。
- 未使用 runtime threshold、hysteresis、duration hard rule、Focal、class weight、hard veto 或规则 fallback。
- 双证据映射：Protect-only→inside，Remove-only→outside，重叠或两路均无证据→unsure；禁止补集补标签。

## 正式训练

canonical SHA256：`8b9cc041b92b116514293a0e602de876b4e7a2c710bfce4a03f0c0e2d7a6d33c`。

训练在 epoch 6 按 patience=3 早停并恢复 epoch 3 最佳权重：

| partition | inside recall | outside recall | truth-inside→outside | truth-outside→inside |
|---|---:|---:|---:|---:|
| val | 99.2403% | 9.1040% | 112 frames | 18880 frames |
| test | 99.6352% | 13.8260% | 45 frames | 31681 frames |

执行预算符合合同：物理 VRAM 8188 MiB、95% cap 7779 MiB，峰值 allocated/reserved 约 `2303.766/3796 MiB`，shared VRAM spill 增量为 0；物理 RAM 未超过 95% cap，cleanup 后 CUDA allocated 约 16.25 MiB。

## 完整 source replay

24 条 held-out replay 的结构指标：

| partition | start coverage | end coverage | truth-run continuity | 完整 truth-inside run 删除 |
|---|---:|---:|---:|---:|
| val | 100% | 95.45% | 97.73% | 0 |
| test | 100% | 97.37% | 97.37% | 0 |

结构 recall 看似较强，但大量真实 background 仍被保留，因此不能据此推翻 outside recall 失败。

人工审计页为：

`http://127.0.0.1:8080/agents/audits/20260723_192300_scorer-v11-real-full-source-retrain-full-source-gate/`

页面共 80 项：24 个 held-out full source、40 个 `>8s` residual、16 个 prediction-drop/truth-keep。24 个 source WAV 均已物化；人工 gate 当前仍是 pending，本文不伪造试听结论。numeric gate 已失败，人工 verdict 只能补充 zero-clipping/zero-true-speech-deletion 证据，不能把 checkpoint 改判为可晋升。

## 可复现的监督拓扑证据

运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:UV_CACHE_DIR=(Resolve-Path '.uv-cache').Path
uv run python tools/boundary/ja/audit_candidate_island_scorer_v11_supervision_distribution.py `
  --canonical-sources agents/temp/20260723_192300_scorer-v11-real-full-source-retrain/02-canonical/canonical_sources.jsonl `
  --output-dir agents/temp/20260723_194753_scorer-v11-supervision-distribution
```

结果：

| source group | sources | inside frames | outside frames | unsure frames | same-source inside+outside |
|---|---:|---:|---:|---:|---:|
| semantic composite | 870 | 489370 | 34266 | 0 | 870/870 |
| isolated vocal | 256 | 12032 | 10108 | 0 | 256/256 |
| real outside masked | 20 | 0 | 14506 | 60432 | 0/20 |
| calibrated dual full-source | 25 | 65358 | 1476 | 27178 | 8/25 |
| held-out real full-source | 24 | 27077 | 57535 | 5138 | 22/24 |

真实 train full-source 合计只有 `8/45=17.78%` 同时含 inside/outside，held-out 则为 `22/24=91.67%`。25 条 calibrated dual source 中 `17/25` 没有任何 outside；逐 source outside definite fraction 的中位数为 0，p75 也只有约 1.14%。20 条旧 masked real source 则没有任何 inside truth。synthetic source 虽大多同时含两类，但其 provenance 和声学构造与 held-out 真实 full-source 不同，无法阻止模型学习 source-level shortcut。

## 决策与下一轮准入

本轮结论固定为 `rebuild_real_mixed_supervision_before_gpu_retrain`：

1. 不延长当前训练，不调 runtime threshold，不用 Focal/class weight 掩盖分布问题，不因本次失败放弃 Mamba。
2. 下一份训练监督必须在冻结 train source 内同时提供独立 Protect-only inside 与 Remove-only outside；neither/conflict 继续为 unsure，不得用补集补齐。
3. 优先候选是对现有 20 条 `real_train_outside_masked` source 做独立 Protect+Remove，或从其余冻结 train source 中按真实 Remove 潜力选择小批量；先比较 mixed-source 数量、零 outside source 数和 outside fraction，再决定是否启动 GPU。
4. 本轮用户授权仅覆盖已经完成的固定 25 条；没有向 Gemini 发送额外音频。对新增 source 的 API 请求及费用必须另行授权。
5. 6 条 `independent_background_needs_downstream_isolation` 仍保持 Scorer unsure；Proposal/Split/CueQC 同源 replay 应在 Scorer 数值分布恢复合理后继续，Inner 仍只裁 keep 岛首尾，不能挖内部空洞。

监督分布工具的 summary 明确写入 `training_manifest_allowed=false`；它只用于在 GPU 前做数据审计，不是训练标签 compiler，也不代替最终数值与人工 gate。
