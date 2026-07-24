# Scorer v11 Frame–Event Query-Mask A/B

## 最终结论

Frame–Event Query-Mask 在冻结的 Scorer v11 数据、partition、seed 和 P2048/H256 trunk 上显著减少了 frame-argmax 碎片，但本臂仍不能晋升。相对 baseline，Val/Test truth-run continuity 从 `68.18% / 57.89%` 提高到 `88.64% / 78.95%`，内部 drop gap 从 `85 / 70` 降到 `7 / 12`；代价是 outside recall 从 `28.80% / 46.88%` 降到 `22.17% / 35.45%`，且 Val/Test start coverage 只有 `90.91% / 94.74%`，没有达到最高 `95%` 数值门槛。

本臂固定为 `diagnostic_success_but_not_promotable`。它比 linear-chain CRF 更少跨越长背景，但 Test continuity 不如 CRF，说明 query mask 的局部事件表示比单一全局转移更接近所需折中点，仍未同时解决边缘 coverage、短孔洞和独立长背景。生产 registry、现役 checkpoint 与 runtime 均未修改；下一冻结结构臂为 Dense Span + learned DP。

## 冻结 A/B 合同

- 中央序列化合同：`boundary_acoustic_binary_v12`。
- canonical、partition、raw feature cache、signed windows、seed=`117`、训练顺序、class weight=`1/1` 与 baseline/CRF 完全相同。
- trunk：raw PTM2048 → trainable full-width Linear(2048→2048)+GELU → normalized MFCC40 → hidden256 bidirectional Mamba2 → dense two-class frame logits。
- context=`1000 frames`、nominal overlap=`200 frames`、midpoint unique ownership；训练 batch budget=`1000 padded frames`。
- 结构新增 `K=8` learnable island queries、Query→Frame / Frame→Query attention、two-logit query existence 与 soft temporal masks。
- query masks 通过 existence-weighted probabilistic union 形成 log-odds residual；zero-init `tanh` scalar gate 将其加入 dense two-logit head，runtime 仍为 `softmax(2 logits)+argmax`。
- 训练只新增 Hungarian matching、query existence CE、matched mask BCE+Dice，固定总 auxiliary weight=`1.0`；matching 不参与 runtime candidate filtering。
- definite outside 才分隔 target event；owned `unsure=-100` 不进入 mask、loss、normalization 或 metrics，也不会单独把一个 event 拆成两个；non-owner 终止当前 event。单窗 target 超过 `K=8` 时 fail-closed。
- 无 threshold、hysteresis、top-k、NMS、duration rule、规则补洞或 fallback。

Query-Mask checkpoint：

`0ef791ff036b79d29a273a27a61614fd94a6a1d7dd7eaae0b048738e9bd18f16`

冻结 baseline checkpoint：

`bcba961b0d1e11cc73c1b8a58f31a76060e5c954df2892a815d3069c2ddcc521`

CRF 对照 checkpoint：

`36a724b1b56e91bf113ad54670c5d0eb30cec032f95efef982c7fa2bc737ac95`

## 训练与资源

60-step CUDA smoke 完成且 shared VRAM spill=`0`。正式训练在 epoch 7 触发 patience=`3`，恢复 epoch 4 最佳权重，共 `7440` steps。最佳窗口级 Val inside/outside recall=`98.28% / 22.17%`，Test=`98.84% / 35.45%`。

正式训练峰值 CUDA allocated/reserved=`2380.764 / 3184 MiB`，shared spill=`0`；物理 RAM 保持在 `0.95` 上限内。训练结束后进程释放模型与大 tensor，独立 gate 阶段重新加载 checkpoint；gate 清理前峰值 allocated/reserved=`1110.675 / 1194 MiB`，清理后 allocated/reserved=`8.125 / 20 MiB`。

最终 learned residual gate=`0.02415168`。这个数值只说明直接 query residual 的标量幅度较小；query auxiliary 同时训练共享 trunk，因此不能仅凭 gate 大小把全部改善归因于或排除 query 分支。

## 同口径 full-source 三臂比较

三臂均在同一 24 条 held-out full source、同一 canonical 与同一 `max_padded_frames=1000` 上重放。

| partition | arm | inside recall | outside recall | start coverage | end coverage | continuity | whole truth-run deletions |
|---|---|---:|---:|---:|---:|---:|---:|
| val | baseline | 96.50% | 28.80% | 93.18% | 100.00% | 68.18% | 0 |
| val | CRF | 98.77% | 22.14% | 100.00% | 95.45% | 84.09% | 0 |
| val | Query-Mask | 98.28% | 22.17% | 90.91% | 97.73% | 88.64% | 0 |
| test | baseline | 96.85% | 46.88% | 92.11% | 97.37% | 57.89% | 0 |
| test | CRF | 98.07% | 33.04% | 100.00% | 92.11% | 89.47% | 0 |
| test | Query-Mask | 98.84% | 35.45% | 94.74% | 100.00% | 78.95% | 0 |

| partition | arm | prediction islands | internal gaps | 1-frame | 2-frame | 3-frame | 4+-frame | gap frames | >8s residuals |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| val | baseline | 345 | 85 | 46 | 12 | 5 | 22 | 357 | 7 |
| val | CRF | 81 | 10 | 2 | 2 | 0 | 6 | 54 | 10 |
| val | Query-Mask | 327 | 7 | 5 | 0 | 0 | 2 | 26 | 6 |
| test | baseline | 866 | 70 | 31 | 17 | 5 | 17 | 222 | 11 |
| test | CRF | 164 | 10 | 0 | 1 | 0 | 9 | 127 | 21 |
| test | Query-Mask | 469 | 12 | 6 | 3 | 0 | 3 | 32 | 15 |

Query-Mask 把 20–60ms internal gaps 从 baseline 的 Val/Test `63 / 53` 降到 `5 / 9`，同时没有像 CRF 一样把 prediction island 数压到 `81 / 164`。它的 `>8s` residual=`6 / 15` 也低于 CRF 的 `10 / 21`。但 Val start coverage 退化到 `90.91%`，Test outside recall 仍比 baseline 低 `11.44` 个百分点，无法宣称达到最终职责 gate。

## 人工审计页

`http://127.0.0.1:8080/agents/audits/20260724_192540_scorer-v11-query-mask-heldout/`

页面包含 `24` 个 held-out full source、`44` 个 prediction-drop/truth-keep residual 和 `21` 个 `>8s` residual，共 `89` 项；24 个 WAV 均已物化。实际浏览器验证为 `89 articles / 89 audio`，首条精确色条成功解码、播放和停止，控制台无 error/warn。

人工 verdict 仍为 pending，本文不伪造试听结果。数值结果已足以阻止本臂晋升；人工页用于确认边缘误删、continuity harm 与 canonical 修正类型。

## 决策

1. Query-Mask 源码、trainer、strict checkpoint schema、batch/source 等价路径与 checkpoint audit 保留为独立结构实验能力。
2. Query-Mask checkpoint 不注册、不替换 baseline、不进入生产 cache signature。
3. 不继续调 query threshold、mask threshold、top-k、NMS、Focal、boundary band、class weight 或规则 merge；这些都会破坏冻结结构 A/B 或掩盖数据/结构问题。
4. 下一臂保持同一 P2048/H256 trunk、canonical、partition、seed 和训练合同，测试 Dense Span + learned DP，显式学习 span evidence 与全局非重叠路径，但最终仍输出二分类 frame argmax 兼容结果。
5. 0.6B checkpoint、runtime、data 与空 registry placeholder 均未读取、修改或训练。
