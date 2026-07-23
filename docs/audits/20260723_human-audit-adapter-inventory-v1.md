# Human Audit Adapter Inventory v1

## 当前迁移结论

Human Audit Page Core 当前已有四个职责适配器：

| Adapter | 审计对象 | 选项合同 | 保存兼容性 | 状态 |
|---|---|---|---|---|
| Scorer bridge-gap | Protect 覆盖的人工锚点间 gap | `内容 × 语义覆盖 × 非语义包络`，15 个合法组合 | schema 升级为 `candidate_island_scorer_v11_bridge_gap_manual_verdict_v3` | 已迁移 |
| CueQC false-drop | 模型准备整块删除的完整 provisional sub-island | `safe_drop / true_speech / unsure` | 保持 `cueqc_v13_false_drop_manual_verdict_v1`，既有 evaluator 不变 | 已迁移 |
| Split missing-cut candidate | 已确认漏切 residual 中的真实 Proposal candidate | `cut / continue / unsure`，`unsure→ignore=-100` | 保持 `split_v4_missing_cut_candidate_manual_verdict_v1`，既有 override compiler 不变 | 已迁移 |
| Split canonical candidate | canonical teacher 已标注的固定 Proposal candidate query | `cut / continue / unsure`，`unsure→ignore=-100` | 保持 `acoustic_split_canonical_manual_verdict_v1`，既有 evaluator 不变 | 已迁移 |

CueQC 与 Split candidate 都是单一、明确的二分类查询，因此单轴三态已经完备；不为了形式统一强行拆成多轴。页面顶部必须写清楚选项的职责语义、训练映射和 unsure 行为。

## 播放证据合同

- CueQC false-drop：每个媒体文件就是待删除的完整 sub-island，提供原生完整播放器和“直接播放完整 sub-island”精确停止按钮，不添加上下文。
- Split candidate：先保留完整 chunk 和完整 residual，再为每个 candidate 提供左侧、右侧和左右合并播放。左右边界来自相邻真实 candidate 或 residual 边界，不使用固定秒数窗口。
- Split canonical candidate：优先物化 teacher 请求时实际使用的 `request_clip_start_s/end_s`；只有请求坐标缺失或非法时才回退 `center±context_s`。manifest、WAV、标尺、candidate 红线、左右播放全部使用同一组实际 `clip_start_s/clip_end_s/clip_duration_s/candidate_offset_s`，不再出现旧页“音频按 context 裁、标尺按 request 画”的证据错位，也不再提供固定 `±1s` 跨点按钮。
- 所有页面沿用 Core 的可取消精确播放、`localStorage`、完成度和保存 API；音频切换不会让旧的 metadata 回调启动错误播放。
- Core 保存接口允许 Adapter 通过 `shouldSerialize` 只写已完成裁决；部分审计保存不会再把未审行伪装为 `unreviewed` invalid row。

## 下一批应迁移的现役页面

### Scorer v11 full-source span editor

以下两个现役页面应共用一个新的 Scorer full-source span editor Adapter：

- `generate_candidate_island_v11_heldout_audit_html.py`
- `generate_candidate_island_v11_train_teacher_review_html.py`

当前实现不是独立页面合同，而是导入 Scorer v10 旧模板后连续执行字符串、正则和 CSS 属性修补；train 页又在 held-out 页上继续替换 HTML/JavaScript 片段。只要旧模板文案或 DOM 轻微变化就会 fail 或更危险地静默生成错误页，因此这是当前最高优先级的结构性迁移对象。

新 Adapter 不能退化成简单的 `inside/outside/unsure` 三按钮。它需要保留完整 source span editor 的真实合同：

1. 显式区间只允许 `inside_candidate / unsure`；未标记补集只有在人工确认从头听到尾后才成为 `outside_candidate`。
2. 区间可新增、删除、改边界并执行 `inside_candidate↔unsure` 转换；任何编辑都撤销 source 完整确认。
3. spans 必须按 frame 坐标非重叠、连续投影到合法 source 范围；`unsure→-100`，不进 normalization/loss/metrics/gate。
4. held-out 与 train teacher 保持各自现有 manual verdict schema；Gemini 仅能作为 train 页可编辑底稿，不能自动完成确认。

这属于“共享编辑器能力 + 两个职责 Adapter”，不是把旧 HTML 再包一层 Core。

### Multicore composite

`generate_multicore_composite_audit_html.py` 仍用于训练构成与 Split/Inner 证据复核，适合后续迁成 composite Adapter。它至少要保存彼此独立的轴：clean core 是否可懂、overlay 是否确为非语义且声学强度合理、语义上是否需要 Split、候选处是否声学可安全切。现有页面已经接近该合同，在保存 schema 和合法组合证明完成前不机械改写。

### Subtitle A/B

`generate_subtitle_ab_compare_audit_html.py` 是 sample-c/sample-a 最终 A/B 的现役可视化入口，后续需要补全人工 A/B Adapter，覆盖 empty、重复、遗漏、时间连续性和总体偏好；当前尚未保存人工 verdict，因此本轮只保留，不把播放器对照页误称为已迁移人工 gate。

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

## 不需要迁移为人工 Adapter 的页面

- `generate_candidate_island_teacher_comparison_html.py`、`generate_candidate_island_outside_prompt_matrix.py` 与 `compare_candidate_island_preaudits.py` 是冻结 source 上的只读 prompt/teacher 对照。它们可以复用精确播放器 helper，但没有人工裁决 schema，不应为了形式统一强行接入 Review Core。
- `generate_semantic_timeline_audit_html.py`、`generate_omni_timeline_audit_html.py` 和 `generate_semantic_source_text_alignment_audit_html.py` 当前是 legacy inventory、teacher provenance 或未来数据复核工具，不是 Scorer v11 当前训练入口。保留历史复现能力，但不投入迁移成本。
- Scorer v10 canonical/fragmentation/full-source repair 页面及 evaluator 仍被 v11 数据构造、修复 apply 工具或历史人工 provenance 引用。它们不是当前新 gate，不批量重写，也不能只因 CLI 没有 Python import 就判为死代码。

## 已退役的旧页面

以下两个外部 CueQC cluster 页面及其专属测试已移入 `agents/rm/20260723_154249_retired-cueqc-cluster-audit-pages/`，Git 只记录源码删除，不提交 `agents/rm` 本地归档：

- `generate_cueqc_cluster_audit_html.py`
- `generate_cueqc_cluster_broadcast_html.py`

退役依据：除专属测试和 HISTORY 历史记录外无当前调用；CueQC v13 主路线已经是 teacher/canonical/false-drop；cluster 数据 compiler、历史 label ingestion 和 `tools/asr/cueqc/cluster_candidates.py` 均继续保留。删除的是重复且过时的外部 HTML 壳，不是 cluster 数据合同。

## 迁移原则

1. 只迁移仍会继续生成或用于当前模型 gate 的页面；已完成且只作历史证据的旧页面不批量重写。
2. Adapter 必须先证明选项对当前审计查询完备，再讨论 UI；单轴足够时不滥用多轴，多问题可并存时禁止平面互斥标签。
3. 人工裁决默认只用于 gate、错误归因或精确 override；除非另有 compiler 合同，不直接成为训练标签。
4. 所有 summary 和新保存行记录中央 `boundary_serialization_contract_id=boundary_acoustic_binary_v12`；不新增整数 generation 或旧 alias。
5. `--prompt / --prompt-file` 只改变页面任务说明，不得改变固定选项 schema、合法组合或训练映射。
