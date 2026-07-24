# Scorer v11 Dense Span + learned DP A/B

## 最终结论

Dense Span 在冻结的 Scorer v11 canonical、partition、raw feature、seed 和 P2048/H256 trunk 上，把 Val/Test truth-run continuity 都提高到 `100%`，并把内部 drop gap 从 baseline 的 `85 / 70` 降到 `0 / 0`。但它主要通过输出很长的 `inside_candidate` span 达成连续性：Val/Test outside recall 只有 `15.66% / 25.82%`，低于 baseline 的 `28.80% / 46.88%`，`>8s` residual 增至 `10 / 24`，Val/Test end coverage=`93.18% / 94.74%` 也未达到最高 `95%` 数值门槛。

本臂固定为 `diagnostic_success_but_not_promotable`。完整 span lattice 与 exact learned DP 证明了“直接优化连续非重叠 span”可以彻底消除 frame 抖动，但当前 structured-hinge-only 目标会把短孔洞和真实长背景一起桥接。checkpoint 不注册、不替换 baseline、不进入生产 cache signature；生产 registry、现役 runtime 和 0.6B 均未修改。

下一阶段不继续给 Dense Span 添加 threshold、最大时长、NMS、规则补洞、Focal 或 class weight。更有信息量的实验是固定新一批真实 multi-island mixed-source canonical，在同一数据上重训 plain frame baseline 与当前折中最好的 Query-Mask，先分离“真实 source 内 outside 多样性不足”和“结构 decoder 不足”两种因素。

## 冻结 A/B 合同

- 中央序列化合同：`boundary_acoustic_binary_v12`。
- canonical、partition、raw PTM2048/MFCC40 feature、signed windows、seed=`117`、训练顺序和 class weight=`1/1` 与 baseline、CRF、Query-Mask 相同。
- trunk：raw PTM2048 → trainable full-width Linear(2048→2048)+GELU → normalized MFCC40 → hidden256 bidirectional Mamba2 → dense two-class frame logits。
- context=`1000 frames`、nominal overlap=`200 frames`、midpoint unique ownership；Dense Span schema 的 padded-frame capacity 固定为 `1000`。
- 完整 `[B,2,T,T]` lattice 保留每个合法 start/end/label span，不使用最大 span 时长、top-k、NMS 或 score threshold。
- span score=`dense frame-logit sum + tanh(gate) × (low-rank endpoint + learned duration residual)`，并学习 start/end 与 cross-label transition。
- runtime 使用 exact binary full-lattice Viterbi argmax；相邻最大 run 自动交替标签。对外 probabilities 仅表示 dense frame softmax evidence，不伪称 DP marginal。
- 训练使用 exact loss-augmented Viterbi 与 frame-Hamming structured hinge。`unsure=-100` 和 non-owner 不进入 score、loss、normalization、metrics；连续 definite-owner run 独立计算，不跨 unsure 建立 span。
- 无 threshold、hysteresis、duration-only rule、hard veto、NMS、规则 fallback 或旧 schema alias。

Dense Span checkpoint：

`2b60e366a847e166ebe7c601fcedcdde7067a19a1ac713f65f82483afca77db6`

冻结 baseline checkpoint：

`bcba961b0d1e11cc73c1b8a58f31a76060e5c954df2892a815d3069c2ddcc521`

CRF 对照 checkpoint：

`36a724b1b56e91bf113ad54670c5d0eb30cec032f95efef982c7fa2bc737ac95`

Query-Mask 对照 checkpoint：

`0ef791ff036b79d29a273a27a61614fd94a6a1d7dd7eaae0b048738e9bd18f16`

## 可行性与资源

1000-frame、2-label 完整 lattice 仅占约 `7.63 MiB`。在实际 bidirectional trunk 输出宽度 512、rank 32 下：

- full log-partition NLL 前后向耗时约 `65.97s`，不具备训练可行性；
- exact loss-augmented Viterbi structured hinge 耗时约 `0.712s`；
- 两者的 lattice/decoder 峰值 allocated 约 `103.77 MiB`，说明瓶颈是 log-partition 计算时间而不是 lattice 显存。

因此正式臂冻结使用 exact Viterbi structured hinge，不以近似 top-k、最大时长过滤或候选裁剪换取速度。

60-step CUDA smoke 完成，峰值 allocated/reserved=`2313 / 3144 MiB`、shared VRAM spill=`0`。正式训练共 `5306` steps，在 epoch 5 触发 patience=`3`，恢复 epoch 2 最佳权重；峰值 allocated/reserved=`2313.76 / 3146 MiB`、shared spill=`0`。训练进程内 cleanup 后 allocated=`16.25 MiB`，进程退出后 allocator 不跨阶段；独立 full-source gate cleanup 后 allocated/reserved=`8.125 / 20 MiB`。

learned decoder 参数：

- residual gate=`0.05608919`；
- start scores=`[0.02899311, -0.02899311]`；
- end scores=`[0.02502525, -0.02502525]`；
- cross-label transitions 约为 `-0.01582 / -0.01604`。

这些值表明 span 分支确实参与训练，但不能把改善归因于某个手工 transition；共享 trunk 也被 structured hinge 一并更新。

## 同口径 full-source 四臂比较

四臂均在同一 24 条 held-out full source、同一 canonical 与同一 `max_padded_frames=1000` 上重放。

| partition | arm | inside recall | outside recall | start coverage | end coverage | continuity | whole truth-run deletions |
|---|---|---:|---:|---:|---:|---:|---:|
| val | baseline | 96.50% | 28.80% | 93.18% | 100.00% | 68.18% | 0 |
| val | CRF | 98.77% | 22.14% | 100.00% | 95.45% | 84.09% | 0 |
| val | Query-Mask | 98.28% | 22.17% | 90.91% | 97.73% | 88.64% | 0 |
| val | Dense Span | 99.17% | 15.66% | 97.73% | 93.18% | 100.00% | 0 |
| test | baseline | 96.85% | 46.88% | 92.11% | 97.37% | 57.89% | 0 |
| test | CRF | 98.07% | 33.04% | 100.00% | 92.11% | 89.47% | 0 |
| test | Query-Mask | 98.84% | 35.45% | 94.74% | 100.00% | 78.95% | 0 |
| test | Dense Span | 99.08% | 25.82% | 100.00% | 94.74% | 100.00% | 0 |

| partition | arm | prediction islands | internal gaps | gap frames | >8s residuals | local truth-inside frames predicted outside |
|---|---|---:|---:|---:|---:|---:|
| val | baseline | 345 | 85 | 357 | 7 | 516 |
| val | CRF | 81 | 10 | 54 | 10 | 181 |
| val | Query-Mask | 327 | 7 | 26 | 6 | 254 |
| val | Dense Span | 26 | 0 | 0 | 10 | 123 |
| test | baseline | 866 | 70 | 222 | 11 | 389 |
| test | CRF | 164 | 10 | 127 | 21 | 238 |
| test | Query-Mask | 469 | 12 | 32 | 15 | 143 |
| test | Dense Span | 36 | 0 | 0 | 24 | 113 |

`whole truth-run deletions=0` 只表示没有把某个完整 canonical inside run 全部删除，不等于人工 zero-clipping 已通过。Dense Span 仍有 Val/Test `123 / 113` 个局部 truth-inside frame 被判 outside，共形成 `17` 个精确 prediction-drop/truth-keep residual；人工 gate 保持 pending。

## 人工审计页

`http://127.0.0.1:8080/agents/audits/20260724_210323_scorer-v11-dense-span-heldout/`

页面包含 `24` 个 held-out full source、`17` 个 prediction-drop/truth-keep residual 和 `34` 个 `>8s` residual，共 `75` 项；24 个 WAV 均已物化。实际浏览器验证为 `75 articles / 75 audio / 24 unique audio sources`，首条精确绿色区间成功解码、播放和停止，console 无 error/warn，音频路径均为页面目录内的 `audio/source-XXX.wav`。

人工 verdict 尚未填写，本文不伪造试听结果。数值 gate 已足以阻止本臂晋升；人工页用于判断局部边缘裁剪、长蓝段中的独立背景和 canonical 修正类型。

## 决策

1. Dense Span 源码、strict checkpoint schema、完整 lattice、structured hinge、batch/source scoring 与 checkpoint audit 保留为离线结构实验能力。
2. Dense Span checkpoint 不注册、不替换 baseline、不进入生产 cache signature。
3. 不对本臂追加 runtime threshold、最大 span 时长、top-k、NMS、duration hard rule、规则 merge 或 fallback。
4. 下一数据实验选择 plain frame baseline 与 Query-Mask，而不是 Dense Span/CRF：前者保留当前最高 outside recall，后者是结构臂中 continuity 与长背景 residual 的较好折中。两臂必须共享新增的真实 multi-island mixed-source canonical、partition、raw feature、seed 和训练预算。
5. 0.6B checkpoint、runtime、data 与空 registry placeholder 均未读取、修改或训练。
