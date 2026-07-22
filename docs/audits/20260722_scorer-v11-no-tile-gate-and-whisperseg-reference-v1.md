# Scorer v11 no-tile full-source gate 与 WhisperSeg 参考审计

## 结论

Scorer v11 no-tile、definite-owner、full P2048/H256 baseline 被否决，不得晋升 registry。epoch 10 触发 held-out early stopping 后恢复 epoch 7，checkpoint SHA256=`4866d22335be8b3dd199aa7357c0a5fab42b66cd40245f40f9475e028b0a115d`。完整 source 重放与 window evaluator 完全一致，但暴露出更严重的 source-domain 泛化失败：

- val inside/outside recall=`61.87/49.44%`，test=`27.54/85.15%`；
- val start/end coverage=`72.73/65.91%`、truth-run continuity=`40.91%`；
- test start/end coverage=`39.47/23.68%`、truth-run continuity=`21.05%`；
- test `14` 条 source 中 `11` 条发生完整 truth-inside run 删除；
- prediction-drop/truth-keep frame val/test=`5621/8937`；
- shared VRAM spill=`0`，因此失败不是 soft OOM 或 batch 执行偏差。

当前最强证据指向训练 source 类型和标签 provenance 混杂，而不是先验认定 Mamba 容量不足：train 的 `501402` 个 inside frame 全来自 `256` 条 isolated vocal synthetic 与 `870` 条 semantic composite；`20` 条真实约 75 秒 source 只有 outside=`14506` 与 unsure=`60432`，真实 full-source inside truth 为 `0`。相对地，val/test 全部是人工标注的真实约 75 秒工作流 source。模型因此可以把 synthetic/short-composite 域当成 inside，把真实 full-source 域当成 outside；不同 held-out video 又出现相反偏置。

下一轮必须先引入固定 train source identity 的真实 full-source 正例与严格三态监督，再比较架构。不能用 runtime threshold、hysteresis、gap merge、Focal 或 class weight 掩盖这一 provenance shortcut。

## 完整 source gate 产物

- checkpoint audit：`agents/temp/20260722_193504_scorer-v11-no-tile-full-source-gate/`
- 人工页：`agents/audits/20260722_193900_scorer-v11-no-tile-full-source-failed-gate/`
- 页面包含 `24` 条 held-out 完整 source、全部 `562` 个 prediction-drop/truth-keep span，以及 `7` 个 `>8s` residual；人工 gate 保持 pending，不伪造听感结论。

由于数值 gate 已明确失败，本页当前用于定位错误形态，不要求通过人工审计挽救该 checkpoint。

## WhisperSeg 的实际结构

WhisperSeg 是 ICASSP 2024 工作《Positive Transfer of the Whisper Speech Transformer to Human and Animal Voice Activity Detection》的官方实现：

- 论文/项目：<https://github.com/nianlonggu/WhisperSeg>
- IEEE：<https://ieeexplore.ieee.org/document/10447620>
- DOI/preprint：<https://doi.org/10.1101/2023.09.30.560270>

它把 onset/offset 离散成特殊 timestamp token，并把完整标签序列编码为：

```text
<|species|><|onset_index|>cluster_id<|offset_index|>...
```

Whisper encoder-decoder 对完整 spectrogram clip 做 conditional generation，而不是对每帧概率再选 operating threshold。这个“学习型结构化 span 输出”与 Scorer v11 的连续 candidate-island 职责有直接参考价值。

但官方 runtime 不是本项目合同所要求的纯学习型 two-logit argmax：

- 长音频被切成固定 clip，可运行多个 offset trial；
- 多 trial 通过 DBSCAN 或 frame voting 合并，使用 `eps`；
- 小于 `min_segment_length` 的预测直接删除；
- 默认使用 beam generation，并允许 top-k/top-p sampling；
- 生成后另做 FFT blur onset/offset 修正与 duplicate 清理。

官方 `train_val_split` 还会把同一录音按时间切成 train/val 两段，不符合本项目 source/core identity 冻结和 partition 不重叠合同。官方数据要求目标事件全量标注，也没有本项目必须保留的 `unsure→-100` 路径。

## 可迁移与不可迁移部分

可迁移：

1. 用长上下文一次预测连续 span，而不是把 Scorer 退化成局部 frame detector。
2. 显式预测 start/end 或 span query，使 inside continuity 成为模型输出结构的一部分。
3. 让后续 Proposal/Split消费学习到的边界/查询表示，而不是 candidate threshold、local-max、NMS。

不可直接迁移：

1. `min_segment_length`、DBSCAN `eps`、多 trial voting 等规则后处理。
2. beam/sampling 导致的 batch/decoder不确定性，以及不能直接保证完整推理与分批推理概率/argmax等价。
3. 同录音 train/val 时间切分。
4. 忽略 unsure 或把未标注区域默认成 background。

因此当前不应直接替换为完整 WhisperSeq2Seq。更合理的后续 A/B 是在修复真实 full-source train supervision 后，固定同一数据比较：

- A：现有 bidirectional Mamba two-logit frame argmax；
- B：同一 backbone 加学习型 start/end span query，并把训练期结构损失与 runtime two-logit argmax解耦；
- C：若 B 仍无法解决连续性，再设计无阈值、无 NMS、无时长规则的受约束 span decoder新 schema。

任何 B/C 都必须随机初始化、保留 raw PTM2048 学习型 adapter、`unsure=-100`，并重新验证 batch-equivalence、shared spill=`0`和人工 zero-clipping。
