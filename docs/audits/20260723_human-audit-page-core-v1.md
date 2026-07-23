# Human Audit Page Core v1

## 结论

审计页导出改为 `Core + Adapter`，不再由每个模型脚本分别复制播放器、浏览器状态和保存逻辑。

- Core：`tools/audits/review_page_core.py`，负责统一页面外壳、精确区间播放器、停止播放、`localStorage`、完成度、`manual_verdicts.jsonl` 保存 API 和状态显示。
- Prompt resolver：`tools/audits/audit_prompt.py`，允许 Adapter 使用内置说明，或通过 `--prompt / --prompt-file` 替换任务提示；提示不能改变固定裁决 schema。
- Adapter：负责模型职责相关的数据轨道、审计轴、选项语义、组合结果、完成条件、可选的 `shouldSerialize` 保存过滤和保存字段。Scorer bridge-gap 是首个迁移 Adapter；CueQC、Split、Scorer 结构验证等页面在后续实际修改时逐个迁移，不进行无证据的全量机械改写。

## 选项完备性合同

以后每个新 Adapter 必须先声明独立审计轴、组合合法性规则，再显式枚举所有可完成的组合结果。Core 的 `validate_audit_option_contract()` 会从各轴笛卡尔积和合法性规则生成应有的完整组合空间，并固定检查：

1. 审计轴与选项名称唯一；
2. 每个选项至少在一个合法组合中可达；
3. 结果表没有遗漏任何合法组合，也没有混入逻辑上无效的组合；
4. 每个合法组合都有非 `unreviewed` 的结果；
5. 合同必须提供 `unsure` 路径；
6. 页面保存原始各轴裁决和派生组合结果，不能只保存一个丢失信息的平面标签。

“选项完备”不等于穷举所有自然语言描述，而是确保同一条样本可以同时表达彼此独立的问题。例如“存在语义对白”和“背景过度合并”不能互斥；“语义存在”还必须继续区分是否被模型完整覆盖。

## 已接入 Adapter

### Scorer bridge-gap

当前 Adapter 使用三个审计轴：

- A `content_verdict`：`contains_semantic_dialogue / no_semantic_dialogue / content_unsure`；
- B `semantic_coverage_verdict`：`semantic_fully_protected / semantic_missed_or_clipped / semantic_coverage_unsure / not_applicable_no_semantic`；
- C `envelope_verdict`：`acceptable_continuous_envelope / overmerged_independent_background / envelope_unsure`。

条件组合共 `15` 个，映射为七类最终结果：可接受短背景桥接、人工 background 含语义、语义漏保护/截断、语义与过度合并并存、语义漏保护与过度合并并存、纯 Teacher overmerge、unsure。页面同时提供完整 source、完整 gap、Protect 覆盖子区间和未覆盖子区间的精确播放，避免只听整段却无法判断语言是否落在模型覆盖内。

该人工裁决只用于审计发现和后续精确修复选择，不直接成为训练标签；不使用固定时长、概率阈值或声音类别规则自动决定结果。

### CueQC false-drop

CueQC false-drop 审计保持既有 `safe_drop / true_speech / unsure` 单轴合同与 `cueqc_v13_false_drop_manual_verdict_v1` 保存 schema。这里单轴已经完整覆盖“确认无 semantic core / 确认存在 semantic core / 无法确认”，不需要为了形式一致拆成冗余多轴。页面明确 CueQC 只做完整 provisional sub-island keep/drop，不负责 Split 或 Inner 裁边。

### Split missing-cut candidate

Split candidate 补标保持 `cut / continue / unsure`，其中 `unsure→ignore=-100`，并保留既有 `split_v4_missing_cut_candidate_manual_verdict_v1` 供 override compiler 使用。每个 candidate 同时提供左侧、右侧和左右合并的精确播放，避免旧页只播放 candidate 之前的区间。没有 eligible candidate 的漏切 residual 会 fail-closed 并归因到 Proposal candidate coverage，不能伪造 Split candidate 标签。

### Split canonical candidate

Split canonical teacher 人工 gate 保持 `cut / continue / unsure` 单轴合同与 `acoustic_split_canonical_manual_verdict_v1` evaluator 兼容。Adapter 优先物化 teacher 请求的精确 clip；manifest 记录实际 `clip_start_s / clip_end_s / clip_duration_s / candidate_offset_s`，页面标尺、红线和左/右播放只读这些实际坐标。旧页按另一套 request 坐标绘制标尺、同时按 `context_s` 裁音频的错位已移除，固定 `±1s` 跨点播放也已删除。

该页使用 Core 的 `shouldSerialize`，只保存已选择 `cut/continue/unsure` 的条目。部分审计可以安全保存；evaluator 仍通过 missing ids 明确报告尚未完成的 gate，而不是把未审行当成非法 verdict。

其余 Adapter 的迁移审计与阻塞项见 `docs/audits/20260723_human-audit-adapter-inventory-v1.md`。
