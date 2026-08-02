# Scorer v11 CRF structured-decoder A/B

## 最终结论

线性链 CRF 证明了 Scorer v11 的碎片化确实能由学习型序列转移显著缓解，但当前 CRF 臂不能晋升。相对冻结的 frame-argmax baseline，它把 Val/Test truth-run continuity 从 `68.18% / 57.89%` 提高到 `84.09% / 89.47%`，同时把内部 drop gap 数从 `85 / 70` 降到 `10 / 10`；代价是 outside recall 从 `28.80% / 46.88%` 降到 `22.14% / 33.04%`，且 continuity、outside 与边缘 coverage 仍未同时达到最高 `95%` 数值门槛。

因此本臂固定为 `diagnostic_success_but_not_promotable`：学习型序列一致性是有效方向，但单一二状态全局转移会把“消除短抖动”和“跨越真实长背景”混在一起。生产 registry、现役 checkpoint 与 runtime 均未修改；下一结构臂继续使用同一 trunk，实现 Frame–Event Query-Mask，而不是给 CRF 增加 threshold、gap merge、时长规则或 fallback。

## 冻结 A/B 合同

- 中央序列化合同：`boundary_acoustic_binary_v12`。
- canonical、partition、raw feature cache、signed windows 与 seed=`117` 均与 baseline 相同。
- trunk：raw PTM2048 → trainable full-width Linear(2048→2048)+GELU → normalized MFCC40 → hidden256 bidirectional Mamba2 → two-class emissions。
- context=`1000 frames`、nominal overlap=`200 frames`、midpoint unique ownership；训练 batch budget=`1000 padded frames`。
- baseline loss 为 plain two-logit CE；CRF 只把 decoder/loss 替换为 learned `2×2` transition、连续 definite-owner run CRF NLL 与精确 Viterbi sequence argmax。
- `unsure=-100` 和 non-owner frame 不进入 gold score、normalization、loss 或 metrics；相邻监督 run 不跨 unsure 建立转移。
- runtime 概率为 forward-backward marginal，最终标签为精确 Viterbi sequence argmax；没有 threshold、hysteresis、NMS、duration filter 或规则补洞。
- A/B axis：`linear_chain_crf_decoder_and_sequence_nll_only`。

CRF checkpoint：

`36a724b1b56e91bf113ad54670c5d0eb30cec032f95efef982c7fa2bc737ac95`

baseline checkpoint：

`bcba961b0d1e11cc73c1b8a58f31a76060e5c954df2892a815d3069c2ddcc521`

## 训练与资源

CRF full training 在 epoch 5 触发 patience=`3`，恢复 epoch 2 最佳权重。最佳窗口级 Val inside/outside recall=`98.77% / 22.14%`，Test=`98.07% / 33.04%`。峰值 CUDA allocated/reserved=`2303.777 / 3206 MiB`，shared VRAM spill=`0`；物理 RAM 保持在 `0.95` 上限内。训练结束后独立进程退出，gate 阶段重新加载模型，并在每阶段记录 allocator cleanup。

学得的 transition matrix 为：

```text
[[ 0.137474, -0.224583],
 [-0.224103,  0.123211]]
```

它对两个状态都学习到明显的 same-state 偏好，解释了短孔洞减少以及长背景被更多桥接的共同现象。

## 同口径 full-source A/B

两臂均以更新后的同一 gate、同一 24 条 held-out source、同一 `max_padded_frames=1000` 重放。

| partition | arm | inside recall | outside recall | start coverage | end coverage | continuity | whole truth-run deletions |
|---|---|---:|---:|---:|---:|---:|---:|
| val | baseline | 96.50% | 28.80% | 93.18% | 100.00% | 68.18% | 0 |
| val | CRF | 98.77% | 22.14% | 100.00% | 95.45% | 84.09% | 0 |
| test | baseline | 96.85% | 46.88% | 92.11% | 97.37% | 57.89% | 0 |
| test | CRF | 98.07% | 33.04% | 100.00% | 92.11% | 89.47% | 0 |

| partition | arm | prediction islands | internal gaps | 1-frame | 2-frame | 3-frame | 4+-frame | gap frames | >8s residuals |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | baseline | 345 | 85 | 46 | 12 | 5 | 22 | 357 | 7 |
| val | CRF | 81 | 10 | 2 | 2 | 0 | 6 | 54 | 10 |
| test | baseline | 866 | 70 | 31 | 17 | 5 | 17 | 222 | 11 |
| test | CRF | 164 | 10 | 0 | 1 | 0 | 9 | 127 | 21 |

CRF 把 20–60ms gap 从 Val/Test `63 / 53` 降到 `4 / 1`，也把 prediction island 数降低约 `76.5% / 81.1%`。但 Test outside recall 下降 `13.85` 个百分点，`>8s` residual 从 `11` 增到 `21`；这说明它不只去除了抖动，也过度桥接了真实 background。它减少 frame-level truth-inside deletion，但不能单凭这一项覆盖 outside 与边缘 gate 的退化。

## 人工审计页

`http://127.0.0.1:8080/agents/audits/20260724_184244_scorer-v11-crf-ab-heldout/`

页面包含 `24` 个 held-out full source、`35` 个 prediction-drop/truth-keep residual 和 `31` 个 `>8s` residual，共 `90` 项；24 个 WAV 均已物化。实际浏览器验证为 `90 articles`，首条区间音频成功解码、播放和停止，控制台无 error/warn。

人工 verdict 仍为 pending，本文不伪造试听结果。由于数值 gate 已明确失败，人工结果只能补充错误类型与后续 Query-Mask 设计证据，不能把 CRF 改判为可晋升。

## 决策

1. CRF 源码、训练器、checkpoint audit 与 focused tests 保留为独立结构实验能力。
2. CRF checkpoint 不注册、不替换 baseline、不进入生产 cache signature。
3. 下一臂保持相同 P2048/H256 trunk、canonical、partition、seed 与窗口合同，使用 `K=8` Frame–Event Query-Mask 区分局部事件归属与全局 frame evidence。
4. Query-Mask runtime 仍回到 two-logit softmax argmax；query 仅通过可微 mask/existence 聚合形成 learned residual evidence，不使用 top-k、NMS、threshold 或 duration filter。
5. 0.6B checkpoint、runtime、data 与空 registry placeholder 均未读取、修改或训练。
