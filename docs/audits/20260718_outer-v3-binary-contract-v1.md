# Outer v3 Binary Contract Audit

本页只覆盖 1.7B。0.6B 空 registry、checkpoint、runtime 和 data 未读取、未修改、未训练。中央兼容键仍只有 `boundary_acoustic_binary_v12`。

## Decision

Outer v3 的 schema、随机初始化 trainer 和 padding-invariant runtime plumbing 可以保留，但 registry 继续为空，状态继续为 `pending_outer_v3_audit`。2026-07-22 Scorer 职责重构后，上游合同已断兼容切到 v11 candidate-island：Outer 数据必须来自实际 `post_candidate_island_scorer_v11_islands`，旧 Scorer v8/v9/v10、旧 Outer v2 输出或任意 Galgame full clip 都不能冒充该分布。当前没有可晋升 checkpoint，也没有启动 GPU 或全量训练。

## Inner v2 lessons applied

Outer 的职责是把 Scorer 的完整 provisional island 收成供 Split 查询的 acoustic outer core；它不拥有 Display 时间。Inner v2 已证明以下合同必须前置到数据、模型和 gate，而不能靠 runtime 修补：

- canonical/teacher 可以保存 `background / semantic_core / unsure`，但 `unsure -> -100`，不进入 normalization、loss、metrics 或 gate；runtime 只输出两 logit softmax argmax。
- all-background 是模型学习出的 drop。没有 sigmoid operating threshold、duration keep/drop、hard veto、规则 fallback 或旧三分类 alias。
- 输入使用 raw PTM2048、checkpoint-owned trainable Linear(2048->128)、MFCC40 和 relative-position frame feature，再进入 bidirectional Mamba2 与 Linear(2)。不前 128 截断、不 PCA、不从旧 Outer/Inner checkpoint warm-start。
- source/core identity 在生成特征前冻结，train/val/test 的 source/core 均不交叉，每个 core 最多一次；每个 partition 同时需要 semantic 与 all-background rows。
- 数值 gate 上限为 start/end coverage、background drop recall 均至少 95%，且 val/test true-speech deletion 为 0；即使数值通过，`promotion_ready` 仍为 false，直到 prediction-drop/truth-keep、held-out hard case 和大于 8 秒 residual 的人工零截断/零真语音误删 gate 闭合。

## Rejected route

此前 Outer v3 draft 不能作为继续训练的基线：它从旧 Outer v2 warm-start，并把旧三类 head 裁成两行；数据仍把任意 Galgame sample extent 当精确 semantic target。相同 warm-start/partition/tail-band 条件下，weighted CE、Focal gamma=2、directional auxiliary 的 val end coverage 分别为 `92.08% / 91.86% / 91.40%`，均低于 95%。这些结果只证明 Focal、boundary band 和 auxiliary 没有修复分布问题，不证明当前随机初始化合同的质量，也不允许 warm-start 权重进入新 checkpoint。

## Implemented contract

| surface | current audited behavior |
| --- | --- |
| schema | `outer_edge_refiner_v3` |
| upstream | exact `speech_boundary_ja_candidate_island_scorer_v11` output distribution |
| labels | `background / semantic_core`, `unsure=-100` |
| decision | frame-level two-logit softmax argmax; first/last semantic frame forms the paired outer edges; all-background drops the island |
| model | raw PTM -> learned projection + MFCC + relative position -> bidirectional Mamba2 -> Linear(2) |
| batching | sorted frame-budget groups, restored source order, valid-prefix reverse scan; no context truncation |
| checkpoint | random initialization only; exact dataset contract and central serialization contract checked by builder and loader |
| release | numeric gate cannot promote; registry/status remain pending |

Runtime shape checks now reject empty groups, wrong feature width, non-positive spans and invalid frame hops. Singleton versus padded-batch tests require identical argmax/order and probabilities within `1e-5`. The Mamba reverse pass reverses only the valid prefix, so right padding cannot become artificial leading context.

## Dataset and gate safeguards

The trainer rejects empty/floating partitions, source or core partition leakage, duplicate core use, non-Scorer-v11 rows and partitions missing semantic or all-background examples. Normalization uses only definite frames with positive source weight. Evaluation also masks unsure frames before constructing predicted outer edges; an argmax semantic prediction on an unsure frame therefore cannot improve coverage or hide a true-speech deletion.

The CPU one-step plumbing smoke at `agents/temp/20260718_204608_outer-v3-binary-plumbing-smoke/` used six synthetic unique source/core rows, two per partition, with semantic and all-background presence in every partition. It recorded `unsure=7` as excluded and ended with `numeric_gate_pass=false`, `gate_pass=false`, `promotion_ready=false`. Its checkpoint is synthetic, local-only and must not be promoted or committed.

GPU stages use the physical VRAM and RAM `0.95` caps. Shared VRAM is not a budget; any positive post-baseline spill is a soft OOM. The trainer deletes the model, optimizer, retained best state and final GPU tensors before `gc`/CUDA cache cleanup, then records the post-release snapshot.

## Human audit status

No listening verdict is claimed here: the only new prediction artifact is a synthetic plumbing smoke with no human-auditable source audio. A playable/saveable verdict page must be generated with the existing `tools/audits` framework after a real post-Scorer-v11 smoke exists. It must include every prediction-drop/truth-keep case, held-out hard case and greater-than-8-second residual; no promotion is possible before those verdicts are saved and evaluated.
