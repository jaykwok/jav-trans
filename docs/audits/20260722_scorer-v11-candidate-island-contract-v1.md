# Scorer v11 Candidate-Island Contract Audit

## 结论

Scorer 的职责从“逐帧判断目标语音内容”收窄为不可逆工作流入口的高召回候选岛 membership：

- `outside_candidate`：可以在进入 Proposal/Outer/Split/CueQC 前安全删除的清晰非人声背景。
- `inside_candidate`：应继续送往下游判断的连续候选波形。同一 ASR 单元内部的停顿、尾音和短背景声属于此类；孤立呻吟、喘息、含混人声也优先保留，由 CueQC 决定是否 drop。
- `unsure`：只保留在 canonical 和人工审计，训练映射为 `-100`，不进入 normalization、loss、metrics 或 gate。

runtime 只允许两 logit softmax 后的逐帧 argmax。禁止 threshold、hysteresis、dilation、时长合并、hard veto、规则 fallback 和旧 checkpoint alias。Proposal 只给 Split 附加非绑定候选，不能改变 Scorer 的 keep/drop。

## 审计证据

旧 r9 不能通过换标签直接成为 v11 数据：

- 2665 个 source 中，1036 个 speech source 有 1021 个是双 Galgame core composite，1022 个含人工 internal background；val/test 主要衡量合成拓扑，而非真实工作流 source window。
- r9 worst-frame `0.20` 与 `0.10` 均没有任何 held-out step 通过 gate；继续扫 loss 没有修复背景误留，并损害连续性。
- 61+1 条 fragmentation 人工审计中只有 2 条确认是同一 ASR 单元误切，多数分离本身合理，因此没有证据支持 runtime gap merge 或直接扩大 span decoder。
- `20260722_003817` 的 47 条 held-out background false-keep 中，15 条被发现 canonical 错误。用户随后完成完整 source 精确标注页 `agents/audits/20260722_073633_scorer-v10-r9-heldout-full-source-truth-repair15/`：15/15 reviewed，13 条包含目标语音、2 条全背景，frame=`speech 1258 / background 752 / unsure 0`。严格 evaluator gate 位于 `agents/temp/20260722_080641_scorer-v10-r9-heldout-full-source-truth-repair15-gate/`，`manual_gate_passed=true`，但保持 `training_manifest_allowed=false`。

这 15 条只纠正旧 semantic canonical 的事实，不自动映射为 v11 candidate membership。特别是同一句内部的旧 `background` span 在 v11 中可能必须标为 `inside_candidate`。

## v11 模型与 checkpoint 合同

独立 schema 为 `speech_boundary_ja_candidate_island_scorer_v11`，不兼容 v8/v9/v10：

- raw PTM2048 → checkpoint 内随机初始化、可训练 `Linear(2048→128)`；
- MFCC40 使用仅由 definite train owner frames 计算的 normalization；
- 不使用 row-relative position，避免完整 source 与 runtime window 的相对位置不一致；
- valid-prefix bidirectional Mamba2 stack → `Linear(2)`；
- 固定标签 `outside_candidate / inside_candidate`；
- 固定中央合同 `boundary_acoustic_binary_v12`；
- 只允许 random init，结构或数据合同不匹配时直接拒绝，不做 warm-start/alias。

生产 `segment()` 仍 fail-fast 为 `pending_binary_scorer_audit`。v11 只有在真实数据、人工 zero-clipping gate 和 Scorer→CueQC 工作流 gate 全部通过后才能接入。

## 固定上下文与批处理合同

训练和 runtime 共用 `20s / nominal 4s overlap` 的窗口规划。长 source 的末窗向 source 末端对齐，保证不产生上下文缩水的短尾窗；末窗允许比名义值更多的重叠。

物理重叠按中点划分唯一 owner interval：

- 每个 source frame 只由一个完整上下文窗口负责输出；
- 不平均概率、不投票、不用 batch size 改变窗口起点或可见上下文；
- loss/metrics 也只计算 owner frames，context-only overlap 不重复计权；
- batch 只能按 frame/token/state 预算重组既定窗口。

聚焦测试已覆盖 0 帧、短 source、窗口边界与长 source 的无缺口/无重复 ownership、末窗完整上下文、owned output 精确拼接、checkpoint roundtrip、旧 schema 交叉调用拒绝、batch/singleton 概率与 argmax 等价，以及 argmax-only decoder。

## 新数据与训练分布

下一阶段必须重新编译 candidate membership canonical：

1. 先冻结真实 source/core identity 与 train/val/test partition；同 core 最多一次，source/core 均不得跨 partition。
2. val/test 只使用真实工作流 source windows，覆盖连续对白、同句停顿/尾音、呻吟喘息、背景人声、音乐/impact/静音和全背景。
3. Galgame/NSFW 是贴合 JAV 的真实人声来源，可以作为 train speech material；CueQC definite-drop 呻吟、喘息、呼吸等可以作为 train overlay/hardmix。但合成 composite 只能作为 train augmentation，不能进入 val/test 或替代真实 source 分布。
4. 配对消融固定同一 core、同一前后文和同一 partition，只改变 clean、overlay 类型和 overlay SNR。
5. canonical 三态映射为 `outside_candidate=0 / inside_candidate=1 / unsure=-100`；unsure 不进入 normalization、loss、metrics、split 或 gate。
6. 普通 frame cross-entropy 作为首个基线。Focal、weighted、boundary/continuity auxiliary 只有固定数据 A/B 明确改善人工安全和 held-out gate 才采用。

数字 gate 最高为 start/end coverage 和 same-ASR-unit continuity `>=95%`。人工 `zero true-speech deletion / zero clipping` 优先；所有 prediction drop/truth keep、held-out hard case 和 `>8s` residual 必须人工审计。背景误留还必须跑真实 Scorer→CueQC v13 argmax 链，确认下游 ASR empty/重复/遗漏不劣化，不能用 CueQC 掩盖 Scorer 的真语音删除。
