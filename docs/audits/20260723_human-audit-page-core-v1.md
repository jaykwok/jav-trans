# Human Audit Page Core v1

## 结论

审计页导出改为 `Core + Adapter`，不再由每个模型脚本分别复制播放器、浏览器状态和保存逻辑。

- Core：`tools/audits/review_page_core.py`，负责统一页面外壳、精确区间播放器、停止播放、`localStorage`、完成度、`manual_verdicts.jsonl` 保存 API 和状态显示。
- Prompt resolver：`tools/audits/audit_prompt.py`，允许 Adapter 使用内置说明，或通过 `--prompt / --prompt-file` 替换任务提示；提示不能改变固定裁决 schema。
- Adapter：负责模型职责相关的数据轨道、审计轴、选项语义、组合结果、完成条件和保存字段。Scorer bridge-gap 是首个迁移 Adapter；CueQC、Split、Scorer 结构验证等页面在后续实际修改时逐个迁移，不进行无证据的全量机械改写。

## 选项完备性合同

以后每个新 Adapter 必须先声明独立审计轴、组合合法性规则，再显式枚举所有可完成的组合结果。Core 的 `validate_audit_option_contract()` 会从各轴笛卡尔积和合法性规则生成应有的完整组合空间，并固定检查：

1. 审计轴与选项名称唯一；
2. 每个选项至少在一个合法组合中可达；
3. 结果表没有遗漏任何合法组合，也没有混入逻辑上无效的组合；
4. 每个合法组合都有非 `unreviewed` 的结果；
5. 合同必须提供 `unsure` 路径；
6. 页面保存原始各轴裁决和派生组合结果，不能只保存一个丢失信息的平面标签。

“选项完备”不等于穷举所有自然语言描述，而是确保同一条样本可以同时表达彼此独立的问题。例如“存在语义对白”和“背景过度合并”不能互斥；“语义存在”还必须继续区分是否被模型完整覆盖。

## Scorer bridge-gap Adapter

当前 Adapter 使用三个审计轴：

- A `content_verdict`：`contains_semantic_dialogue / no_semantic_dialogue / content_unsure`；
- B `semantic_coverage_verdict`：`semantic_fully_protected / semantic_missed_or_clipped / semantic_coverage_unsure / not_applicable_no_semantic`；
- C `envelope_verdict`：`acceptable_continuous_envelope / overmerged_independent_background / envelope_unsure`。

条件组合共 `15` 个，映射为七类最终结果：可接受短背景桥接、人工 background 含语义、语义漏保护/截断、语义与过度合并并存、语义漏保护与过度合并并存、纯 Teacher overmerge、unsure。页面同时提供完整 source、完整 gap、Protect 覆盖子区间和未覆盖子区间的精确播放，避免只听整段却无法判断语言是否落在模型覆盖内。

该人工裁决只用于审计发现和后续精确修复选择，不直接成为训练标签；不使用固定时长、概率阈值或声音类别规则自动决定结果。
