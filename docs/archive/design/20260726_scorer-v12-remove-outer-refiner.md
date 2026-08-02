# Scorer v12 Pipeline 简化：删除 Outer Refiner

## 决策日期
2026-07-26

## 背景

Outer Refiner v3 设计为 Scorer 和 Split 之间的中间层，将 Scorer 的 candidate islands 精修为 acoustic outer core。但存在以下问题：

1. **职责重叠**：HISTORY.md L68 明确指出 Outer 与 Scorer 高召回/CueQC 路由职责重叠
2. **未完成消融**：HISTORY 要求先做 no-Outer/edge-only/current 三臂职责消融实验
3. **当前状态**：pending_outer_v3_audit，未训练未部署，registry 为空
4. **架构冗余**：如果 Scorer v12 训练好，不应需要额外的边界修复层

## 决策

**立即删除 Outer Refiner v3**，简化为 4 层 pipeline：

```
旧 Pipeline（5 层）:
Scorer v12 → Outer v3 → Split v4 → CueQC v13 → Inner v2 → ASR

新 Pipeline（4 层）:
Scorer v12 → Split v4 → CueQC v13 → Inner v2 → ASR
```

## 职责重新定义

### Scorer v12
- **职责**：高召回检测所有人类发声事件的物理包络
- **输出**：`candidate_islands`（二分类：vocal_candidate / non_vocal_candidate）
- **训练目标**：
  - Inside recall ≥ 95%
  - Outside recall ≥ 95%
  - Continuity ≥ 95%
  - Zero complete vocal-run deletions
- **不做**：不区分说话人、不转写、不切句、不做语义价值判断

### Split v4
- **职责**：将 Scorer 的连续 candidate islands 按独立发声事件切开
- **输入**：直接接收 Scorer 的 `candidate_islands`（不再需要 Outer 的 acoustic_outer_core）
- **输出**：`provisional_sub_islands`（binary cut/continue argmax）
- **架构**：双层 Mamba（candidate-level + island-level）
- **训练数据**：需要用最终 Scorer v12 输出重新编译

### CueQC v13
- **职责**：对 Split 后的 provisional sub-islands 执行 keep/drop 路由
- **输出**：`drop_before_asr` 或 `keep_for_asr`（binary argmax）
- **执行顺序**：在 Inner 之前
- **不变**：已训练已部署，保持现有 checkpoint

### Inner v2
- **职责**：只对 CueQC keep 的 sub-islands 精修边界
- **输出**：裁剪首尾到 acoustic core
- **限制**：不挖内部空洞
- **不变**：已训练已部署，保持现有 checkpoint

## 实施步骤

### 1. 代码层面（标记废弃，保留审计）

```python
# src/boundary/outer_refiner_v3.py
# 添加 @deprecated 装饰器，保留代码供历史审计

# src/boundary/runtime_pipeline.py
# 删除 Outer 执行步骤

# src/boundary/split_model.py
# 修改输入 schema：
# 旧：post_outer_v3_acoustic_core
# 新：post_candidate_island_scorer_v12
```

### 2. 数据层面

Split v4 需要重新编译训练数据：
- 输入从 Outer v3 输出改为 Scorer v12 输出
- 旧的 scalar features（prominence/strength/speech_valley）已失效
- 必须使用最终 Scorer v12 重新生成特征

### 3. 文档更新

- README.md：更新 pipeline 架构图
- HISTORY.md：记录 Outer 删除决策
- 各模型文档：更新职责说明

## 理由

### 为什么现在删除？

1. **成本低**：Outer v3 未训练未部署，删除无影响
2. **职责清晰**：4 层 pipeline 分工更明确
3. **简化维护**：减少一个模型的训练和维护成本
4. **Scorer 改进**：v12 的训练目标包括边界质量，不应依赖下游修补

### 如果 Scorer v12 边界仍不理想？

**短期**：调整 Scorer 的 loss（增加 boundary auxiliary）
**长期**：如果确实需要，Inner v2 已经在尾部精修边界

### Outer 的历史包袱

从 HISTORY 看，Outer 可能是为了修补 Scorer v10/v11 的边界问题：
- v10/v11 continuity 只有 73-87%
- 大量内部 gap 和 fragmentation
- 引入 Outer 作为"补丁"

但正确的方向是**改进 Scorer 本身**，而不是加中间层。

## 验证

删除 Outer 后，验证 end-to-end pipeline：

```bash
# 1. Scorer v12 训练完成后
# 2. 重新编译 Split v4 训练数据
# 3. 重新训练 Split v4
# 4. 运行完整 pipeline smoke test
# 5. 对比有 Outer vs 无 Outer 的 end-to-end 指标
```

如果 end-to-end 指标下降超过 2%，重新评估是否需要保留 Outer。

## 相关文件

- `src/boundary/outer_refiner_v3.py`（标记废弃）
- `src/boundary/runtime_pipeline.py`（删除 Outer 步骤）
- `src/boundary/split_model.py`（修改输入 schema）
- `tools/boundary/ja/compile_split_v4_features.py`（重新编译特征）

## 结论

删除 Outer Refiner v3，简化为 4 层 pipeline：Scorer → Split → CueQC → Inner。

职责更清晰，维护成本更低，符合"改进上游而非补丁下游"的原则。
