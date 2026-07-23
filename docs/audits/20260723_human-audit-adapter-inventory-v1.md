# Human Audit Adapter Inventory v1

## 当前迁移结论

Human Audit Page Core 当前已有三个职责适配器：

| Adapter | 审计对象 | 选项合同 | 保存兼容性 | 状态 |
|---|---|---|---|---|
| Scorer bridge-gap | Protect 覆盖的人工锚点间 gap | `内容 × 语义覆盖 × 非语义包络`，15 个合法组合 | schema 升级为 `candidate_island_scorer_v11_bridge_gap_manual_verdict_v3` | 已迁移 |
| CueQC false-drop | 模型准备整块删除的完整 provisional sub-island | `safe_drop / true_speech / unsure` | 保持 `cueqc_v13_false_drop_manual_verdict_v1`，既有 evaluator 不变 | 已迁移 |
| Split missing-cut candidate | 已确认漏切 residual 中的真实 Proposal candidate | `cut / continue / unsure`，`unsure→ignore=-100` | 保持 `split_v4_missing_cut_candidate_manual_verdict_v1`，既有 override compiler 不变 | 已迁移 |

CueQC 与 Split candidate 都是单一、明确的二分类查询，因此单轴三态已经完备；不为了形式统一强行拆成多轴。页面顶部必须写清楚选项的职责语义、训练映射和 unsure 行为。

## 播放证据合同

- CueQC false-drop：每个媒体文件就是待删除的完整 sub-island，提供原生完整播放器和“直接播放完整 sub-island”精确停止按钮，不添加上下文。
- Split candidate：先保留完整 chunk 和完整 residual，再为每个 candidate 提供左侧、右侧和左右合并播放。左右边界来自相邻真实 candidate 或 residual 边界，不使用固定秒数窗口。
- 所有页面沿用 Core 的可取消精确播放、`localStorage`、完成度和保存 API；音频切换不会让旧的 metadata 回调启动错误播放。

## 暂不机械迁移的页面

### Split v4 binary gate

旧主 gate 的平面标签不完备，不能直接换壳：

1. `valid_cut / false_cut / unsure` 没有区分“确实需要切，但当前切点截断目标事件”和“同一目标事件内部的不必要切分”。
2. `acceptable / missing_cut / unsure` 没有区分：
   - Proposal 已给出正确 candidate，但 Split 判成 continue；
   - 必须切开的真实边界附近没有 Proposal candidate；
   - 同一 residual 内同时存在上述两类问题。
3. 实际历史产物已经证明该缺口：把没有 `residual_candidates` 的 residual 标为 `missing_cut` 后，candidate 补标页无法继续。当前补标 Adapter 会 fail-closed，并明确要求将其归因到 Proposal candidate coverage。

主 gate 下一版应至少拆成：

- predicted cut event：`是否需要 split × 当前 cut 是否安全`；
- long residual：`是否需要一个或多个 split × 必要边界的 Proposal candidate 覆盖（全部 / 部分 / 全无 / 不确定）`。

在该组合合同和旧 `false_cut→continue` override 的兼容映射确定前，不迁移主 gate，也不把新裁决编译成训练标签。

### Scorer v11 prediction / structure gate

旧整条 source 选项把以下可同时发生的问题做成互斥按钮：真语音损失、ASR 连续性破坏、canonical 错误。长 residual 和精确 drop 页也存在类似压缩。迁移前应按类别分别定义独立轴，例如：

- full source：`语音安全 × 工作流连续性 × canonical 有效性`；
- prediction drop：`区间内容 × 删除后的 ASR 连续性 × canonical 有效性`；
- long residual：`语义保护 × 独立背景过度合并 × canonical 有效性`。

这些页面必须先列出合法组合和保存兼容策略，再接入 Core。

## 迁移原则

1. 只迁移仍会继续生成或用于当前模型 gate 的页面；已完成且只作历史证据的旧页面不批量重写。
2. Adapter 必须先证明选项对当前审计查询完备，再讨论 UI；单轴足够时不滥用多轴，多问题可并存时禁止平面互斥标签。
3. 人工裁决默认只用于 gate、错误归因或精确 override；除非另有 compiler 合同，不直接成为训练标签。
4. 所有 summary 和新保存行记录中央 `boundary_serialization_contract_id=boundary_acoustic_binary_v12`；不新增整数 generation 或旧 alias。
5. `--prompt / --prompt-file` 只改变页面任务说明，不得改变固定选项 schema、合法组合或训练映射。
