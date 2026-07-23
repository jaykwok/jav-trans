# Scorer v11 Candidate-Island Contract Audit

## 结论

Scorer 的职责从“逐帧判断目标语音内容”收窄为不可逆工作流入口的高召回候选岛 membership：

- `inside_candidate`：存在明确或很可能的日语词语/对白锚点，并覆盖同一轮连续对白所需的完整波形包络；同一包络内部的停顿、尾音、短呼吸、呻吟或动作声随对白一起保留。
- `outside_candidate`：明确不含词语、且能独立于对白安全删除的非语义声音，包括纯呻吟、喘息、呼吸、亲吻、impact、音乐、静音或环境声；JAV/Galgame 场景和 vocal activity 本身都不能自动把它升级为 inside。
- `unsure`：局部可能是词语、也可能只是呻吟或噪声时使用；只保留在 canonical 和人工审计，训练映射为 `-100`，不进入 normalization、loss、metrics 或 gate。

runtime 只允许两 logit softmax 后的逐帧 argmax。禁止 threshold、hysteresis、dilation、时长合并、hard veto、规则 fallback 和旧 checkpoint alias。Proposal 只给 Split 附加非绑定候选，不能改变 Scorer 的 keep/drop。

### 高召回不等于把职责下推

Scorer 的错误代价应当不对称：任何可能含词语、尾音或同一轮对白连续性的区间都不能成为 `outside_candidate`；但这不等于把所有呻吟、喘息或呼吸都标成 `inside_candidate`。黄色差集只有在“确认不含词语/对白”并且“能够独立于同一轮对白波形安全删除”两个条件同时成立时才是确定负例。独立纯非语义活动仍属于 Scorer 的可学习 outside；夹在同一对白包络内、紧贴边缘或删除会造成碎片化的非语义声随 inside 保留；无法确定时标 `unsure`。

下游不能替 Scorer 恢复已删除的波形。CueQC 只对 Split 已形成的完整 provisional sub-island 做整段 `keep/drop`，Inner 只在 CueQC keep 项上定位首尾 acoustic semantic core 或学习到 all-background drop；当前 runtime 取首个到最后一个 semantic frame，不移除中间内部背景。因此“独立纯呻吟交给 CueQC、边缘残留交给 Inner”只是一道后续安全网，不是把 Scorer 退化成广义 vocal/VAD keep-all 的理由。下一轮先按这一标签合同完成真实 source 人工审计和数据重编，架构保持 P2048/H256 baseline；只有固定数据下仍出现连续性失败，才进行 heatmap/span decoder A/B。

## 审计证据

旧 r9 不能通过换标签直接成为 v11 数据：

- 2665 个 source 中，1036 个 speech source 有 1021 个是双 Galgame core composite，1022 个含人工 internal background；val/test 主要衡量合成拓扑，而非真实工作流 source window。
- r9 worst-frame `0.20` 与 `0.10` 均没有任何 held-out step 通过 gate；继续扫 loss 没有修复背景误留，并损害连续性。
- 61+1 条 fragmentation 人工审计中只有 2 条确认是同一 ASR 单元误切，多数分离本身合理，因此没有证据支持 runtime gap merge 或直接扩大 span decoder。
- `20260722_003817` 的 47 条 held-out background false-keep 中，15 条被发现 canonical 错误。用户随后完成完整 source 精确标注页 `agents/audits/20260722_073633_scorer-v10-r9-heldout-full-source-truth-repair15/`：15/15 reviewed，13 条包含目标语音、2 条全背景，frame=`speech 1258 / background 752 / unsure 0`。严格 evaluator gate 位于 `agents/temp/20260722_080641_scorer-v10-r9-heldout-full-source-truth-repair15-gate/`，`manual_gate_passed=true`，但保持 `training_manifest_allowed=false`。

这 15 条只纠正旧 semantic canonical 的事实，不自动映射为 v11 candidate membership。特别是同一句内部的旧 `background` span 在 v11 中可能必须标为 `inside_candidate`。

### Split 级人工锚点的评估口径

当前既有人工区间一定程度上按 Split 后的语音单元标注，因此不能把每个锚点之间的人工 `background` 直接当作 Scorer 必须删除的逐帧负例。对于 `anchor1 - bg1 - anchor2 - bg2 - anchor3`，若这些锚点属于同一轮近连续对话，Protect 输出覆盖 `anchor1…anchor3` 的连续候选包络是允许的；中间短停顿、动作声或非语义发声不会降低 Protect precision，也不要求 Scorer 提前切开，后续切分仍由 Split 负责。

例外是锚点间背景在听感上构成声学独立、明显过长且可以在不破坏对话完整性的前提下安全删除的段落，此时连续覆盖属于过度合并。这里的“长”由完整 source 上下文和声学独立性共同决定，不设置固定时长阈值，也不按 duration 自动判错。评估因此固定为：漏保护人工语音锚点、或 Remove-only 命中人工语音锚点属于硬失败；Protect 额外覆盖的锚点间 background 单独进入人工 bridge-gap 审计，裁决为 `acceptable_continuous_envelope / overmerged_independent_background / unsure`，不直接编译成训练标签。

## v11 模型与 checkpoint 合同

主容量 schema 为 `speech_boundary_ja_candidate_island_scorer_v11_full_capacity_v1`，紧凑对照 schema 为 `speech_boundary_ja_candidate_island_scorer_v11_compact_control_v1`；二者均不兼容旧 v11 scaffold 或 v8/v9/v10：

- 主臂：raw PTM2048 → checkpoint 内随机初始化、可训练 `Linear(2048→2048)+GELU`，与 MFCC40 拼接后以 `Linear(2088→256)` 输入 bidirectional Mamba2(hidden=256)；
- 紧凑对照：raw PTM2048 → `Linear(2048→128)+GELU`，与 MFCC40 拼接后以 `Linear(168→128)` 输入 bidirectional Mamba2(hidden=128)；
- MFCC40 使用仅由 definite train owner frames 计算的 normalization；
- 不使用 row-relative position，避免完整 source 与 runtime window 的相对位置不一致；
- valid-prefix bidirectional Mamba2 stack → `Linear(2)`；
- 固定标签 `outside_candidate / inside_candidate`；
- 固定中央合同 `boundary_acoustic_binary_v12`；
- 只允许 random init，结构或数据合同不匹配时直接拒绝，不做 warm-start/alias。

容量 A/B 只允许使用相同 canonical、partition、seed、steps、plain CE、class weights=`1/1`、固定窗口和 `max_padded_frames=2000`。RTX 4060 Ti 8GB 的完整 forward+backward+AdamW smoke 表明主臂 batch=2/2000 padded frames 可行，peak allocated/reserved=`4452/4526 MiB`、shared spill=`0`；batch=3 发生 CUDA OOM，因此代码硬拒绝更大的 padded-frame budget。该 smoke 只证明可训练，不代表 P2048/H256 泛化一定优于 P128/H128。

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

## Boundary heatmap A/B

在纯 argmax baseline 独立提交后，第一轮结构实验固定为 B1 soft start/end heatmap，而不是 proposal-style offset distribution 或 Semi-CRF：

```text
PTM2048 + MFCC40 → projection + Bi-Mamba
                         ├─ inside/outside 主头 → 唯一 runtime 输出
                         ├─ start heatmap 辅助头 → train only
                         └─ end heatmap 辅助头   → train only
```

- 独立 schema=`speech_boundary_ja_candidate_island_scorer_v11_full_capacity_heatmap_aux_v2`；仅允许在 full-capacity baseline 完成后单独 A/B，不能与 compact/full 容量轴同时变化，baseline checkpoint 不能作为 alias 或 warm-start。
- heatmap target 由完整 source definite candidate run 生成，再切固定上下文；高斯 `sigma=2 frames` 只定义训练 target，不是 runtime boundary band。
- touching unsure 的 transition 不产生已知边界，unsure frame 不进入 auxiliary loss。
- `forward()`、checkpoint decoder 和 source scoring API 仍只返回两类 logits/probabilities；metadata 固定 `runtime_auxiliary_decoder=disabled_ab_only`。
- 第一轮总 loss 只允许 `L_inside + lambda_b * L_heatmap`；不加入 duration、IoU、span matching、top-k、NMS、DP 或规则合并。
- A/B 固定相同 canonical、partition、seed、batch/window、训练步数和主 loss。除 baseline `lambda_b=0` 外，B 首轮只比较预注册的小权重；同时记录共享 trunk 的 main/aux gradient norm 与 cosine，长期冲突时不直接引入 PCGrad，而先判定该辅助任务失败或降低权重。
- B 只有在不新增任何 left/right clipping、middle break、whole-island deletion 的前提下，才允许继续测试边界软融合；结构化 decoder 不属于本轮。
