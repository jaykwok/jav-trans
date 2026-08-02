# 1.7B Boundary 模型职责与容量迁移审计 v1

本审计在 Scorer v11 compact/full 真实数据训练期间完成，只覆盖 1.7B。0.6B registry、checkpoint、runtime 和 data 均不修改、不训练。所有新 schema 仍只使用中央合同 `boundary_acoustic_binary_v12`，不恢复整数 generation 或旧 alias。

## 总结

Scorer v11 的 `raw PTM2048 -> full-width nonlinear adapter -> hidden256 bidirectional Mamba2 -> two-logit argmax` 是适合完整 source membership 的主臂，但不能机械复制到所有小模型。各阶段首先要有互不重叠的职责，其次才比较容量。

| 阶段 | 当前独立职责 | 架构结论 | 当前动作 |
| --- | --- | --- | --- |
| Scorer v11 | 完整 source 上的连续 candidate-island membership | compact P128/H128 与 full P2048/H256 同数据 A/B 正在执行 | 训练与人工 gate 后决定主臂 |
| Proposal | 为 Split 提供高召回、非绑定的候选事件 | v1 单 logit及启发式 peak decoder不可作为新基线；新模型应使用 raw PTM、自有 adapter、时序二分类或 learned event query | 等最终 Scorer 输出后重编数据和新 schema |
| Outer | Scorer island 的外缘 refinement | 当前模型会在 CueQC 前整岛 `all-background` drop，与 Scorer/CueQC职责重叠；容量不是首要问题 | 先做 no-Outer / edge-only / current-duty 三臂消融，未证明独立收益前保持 registry 空 |
| Split v4 | 对 Proposal candidate 做 `cut/continue` | 显式 left/gap/right query、candidate Mamba、island Mamba可保留；需新 scalar schema与容量 A/B | 最终上游后重编真实 candidate-query 数据 |
| CueQC v13 | 对 Split provisional sub-island 做 `drop/keep` | chunk序列 Mamba职责正确；下一版应拥有自己的 raw-PTM adapter，不再借用 Split 投影作为语义瓶颈 | 当前权重不改，最终 Split 后再决定是否重构重训 |
| Inner v2 | post-CueQC keep sub-island 到 ASR acoustic semantic core | raw2048->learned128 + MFCC/position + bidirectional Mamba是有效基线 | 不随 Scorer 盲改；最终上游 gate 失败时才做同数据容量 A/B |

## Proposal

仓库只有 Proposal v1 checkpoint builder/loader与promotion入口，没有可复现的当前职责 trainer或 current compiler。旧模型是单 `boundary_prob` logit；实际候选还依赖 smoothing、local maximum、prominence、quantile floor、NMS、speech-valley snap和edge exclusion。即使 Proposal 不做最终 cut，这些步骤也会永久删除 Split 看不到的查询点，因此承担了 recall-bearing runtime decision。

新 Proposal 应断开 v1 schema。首选基线是：

1. 复用同一 source raw PTM2048/MFCC40 cache，但使用 Proposal 自己的 trainable adapter和checkpoint metadata；
2. 输出 `non_candidate/candidate` 两 logit softmax argmax，teacher unsure映射 `-100`；
3. 连续 candidate argmax run只做确定性事件聚合，不使用 probability threshold、quantile、prominence floor、NMS或duration veto；
4. 如果帧二分类在极稀疏正例下不能兼顾 recall，再在固定数据上比较 learned event-query decoder，不先用 Focal/class weight掩盖数据问题。

训练样本必须来自最终 Scorer candidate islands。正例是人工/teacher确认的可查询 boundary event；同一句内部停顿、尾音、呻吟插入但不应切开的点是 hard negative；不同语义单元、speech/noise转移和长 island 中真实可切点都要覆盖。source/core/partition预先冻结，同一 core最多一次。gate以候选事件 coverage最高95%和所有漏掉真边界的人工 zero-clipping审计为先，不以降低候选数量为目标。

容量 A/B只改变 adapter/hidden：compact P128/H128 对 full-width P2048/H256，保持相同 raw cache、partition、seed、steps、plain CE和 batch frame budget。先 smoke验证8GB物理显存和shared spill=0。

## Outer

当前 Outer v3 与 Inner 共用 `BinaryFrameEdgeNetwork`，默认 raw PTM2048 -> Linear128 + MFCC40 + position -> H128 bidirectional Mamba -> two logits。runtime在没有任何 semantic frame时返回整岛 drop。

这一行为位于 CueQC 前，会删除 Scorer有意保留的含混人声或短 vocal candidate，和“Scorer高召回保留、CueQC负责sub-island语义keep/drop”的分工重叠。直接把 Outer升级为P2048/H256只会扩大一个尚未证明必要的决策阶段。

最终 Scorer晋升后先固定同一真实 island集合比较：

- no-Outer：原 Scorer island直接进入 Proposal/Split；
- edge-only Outer：只评估外缘coverage，禁止把职责解释为通用语义过滤；
- current binary Outer：保留整岛background drop，作为风险对照而非默认路线。

比较 Split候选coverage、最终ASR empty/遗漏、边缘真语音误删和人工zero clipping。如果 no-Outer不劣，删除该阶段比训练冗余模型更符合职责；如果 edge-only明确改善，再在相同数据比较P128/H128和更宽adapter。不得通过boundary band、runtime threshold或fallback放行。

## Split

Split v4已有当前最需要的显式查询结构：每个 candidate读取left/gap/right和多尺度frame bins，candidate内双向Mamba保留局部顺序，结构化readout保留gap与left-right差异，随后island级Mamba读取候选序列。它不是仅依赖整个context mean/max，因此不应为追求统一而改成普通全岛分类器。

必须重编的是输入语义。旧 `prominence/strength/speech_valley` 来自 Proposal启发式decoder和旧 Scorer speech概率；相同shape不能通过metadata rebind。新 scalar只保留学习链可定义的量，例如 candidate event位置、island相对位置、左右/gap长度和上游两logit概率统计；任何字段都不得再次承载threshold或hard veto。

容量比较建议三臂预注册后由显存smoke裁剪：

- P128/H128现结构基线；
- raw2048 -> learned nonlinear P256，candidate/island hidden256；
- full-width 2048 nonlinear adapter -> hidden256，仅在8GB、2000-frame/token预算和shared spill=0时保留。

三臂必须固定真实 candidate group、source/core partition、seed、steps、plain CE、weights 1/1、Focal 0、role/pair auxiliary 0。先比较 candidate-event basin recall、continue recall和人工错误切分；Focal、role/pair auxiliary只允许作为后续单轴A/B。

## CueQC 与 Inner

当前 pipeline从 Split checkpoint抽取 `Linear(2048->128)` 权重作为 CueQC PTM pooling输入。这节省计算，但让 CueQC语义容量被另一个模型的 cut任务投影约束。若最终 Split重构，CueQC新 schema应改为 raw PTM2048 bins加自己的learned adapter；仍需绑定 Split SHA来证明输入分布，但不再绑定其投影矩阵作为特征定义。CueQC继续以完整 provisional-sub-island序列做 `drop/keep` argmax，不能缩短上下文或使用时长规则。

Inner当前有效职责和结构保持不变。它只接收CueQC keep，输出送ASR的acoustic semantic core；边缘裁剪不改变display时间。只有在最终Scorer/Proposal/Split/CueQC输出上重新 gate后发现coverage或zero-clipping失败，才比较P128/H128与更宽adapter。不能因为Scorer full-width胜出就无证据废弃现役Inner。

## 执行顺序

1. 完成Scorer v11 compact/full训练、prediction residual页和人工gate。
2. 从最终Scorer真实island编译Proposal新数据；先比较v1重放recall，再决定two-logit frame event或learned event query。
3. 同一真实island做Outer职责三臂消融，先决定是否需要Outer，再决定容量。
4. 用最终Proposal/Outer路线重编Split candidate-query schema和数据，跑neutral容量A/B。
5. 用最终Split真实sub-island重编CueQC；若改变PTM adapter则新schema随机初始化重训。
6. Inner真正移到CueQC keep后执行；按最终上游重跑gate，只有失败才扩容A/B。
7. 最后完成full/batched等价、阶段释放、shared spill=0、匿名样片 C/A和full workflow smoke。
