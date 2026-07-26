# Scorer v12 完整重构方案：Decoder、Loss 与训练策略

## 决策日期
2026-07-26

## 一、总体架构决策

### 1.1 Pipeline 简化（删除 Outer）

```
新 Pipeline（4 层）:
Scorer v12 (高召回宽泛包络)
    ↓ candidate_islands
Split v4 (切分独立事件)
    ↓ provisional_sub_islands
CueQC v13 (keep/drop 路由)
    ↓ keep_items_only
Inner v2 (精修边界)
    ↓ acoustic_cores
ASR (转写)
```

详见 `20260726_scorer-v12-remove-outer-refiner.md`。

### 1.2 Teacher 标注策略（已确定）

- **单次全量**：一次 API 调用标注完整 source（约 75 秒）
- **v6 prompt**：无固定时长阈值，基于声学证据的灵活边界判断
- **已验证**：v4 prompt 在 full144 上平均 2.4 vocal / 1.7 non-vocal runs，
  无过度碎片化；v5 的硬性阈值导致 0 分割（已回退）
- **滑动窗口 Teacher 明确放弃**：碎片化 5-8 倍（147 vs 27 runs）

**关键澄清**：Teacher 标注（单次全量）与 Scorer 推理（滑动窗口）是两个
独立阶段。Scorer 模型推理必然使用 1000-frame 滑动窗口 + midpoint
ownership，这与 Teacher 是否分窗标注无关。时间漂移风险由严格的
`MM:SS.mmm` 时间合同、整数 sample 域量化和 fail-closed 验证控制，
不需要缩短 Teacher 输入时长。

## 二、Decoder 选择

### 2.1 主臂：Dense Span + DP

**选择理由**：
1. 直接输出完整 vocal envelope spans，天然保证连续性
2. v11 实验证据：continuity 达到 100%（val/test），内部 gaps 从 85/70 降到 0/0
3. v11 的失败点（outside recall 15.66%/25.82%）源于 structured-hinge-only
   训练桥接过多背景，v12 通过 dense frame CE auxiliary 修复

**架构**：
```
Input: PTM2048 + MFCC40 + relative position
    ↓
Trainable Linear(2048 → 2048) + GELU     # PTM adapter
    ↓
Concat → Linear(2088 → 256)              # Fusion
    ↓
Valid-prefix Bi-Mamba2 (hidden=256)      # Trunk（与 v11 相同容量）
    ↓
Dense Span Decoder:
    ├─ Span lattice [B, 2, T, T]         # dense frame-sum evidence
    ├─ Low-rank endpoint residual
    └─ Low-rank duration residual
    ↓
Training: exact loss-augmented span Viterbi structured hinge
          + 0.5 dense frame CE auxiliary
Runtime:  exact learned DP argmax（无 threshold/NMS/时长规则/fallback）
```

**Loss 配置**：
```
total_loss = structured_hinge + 0.5 * dense_frame_ce
```

- `structured_hinge`：exact loss-augmented Viterbi（v11 已实现，约 0.712s/window，
  不用 full log-partition NLL 的 65.97s 版本）
- `dense_frame_ce`：帧级 CE auxiliary，unsure=-100 排除；
  防止 DP 为了 continuity 桥接真实背景（v11 的教训）

### 2.2 对照臂：CRF

**选择理由**：
1. v11 证据：显著减少碎片（gaps 85/70 → 10/10，20-60ms gaps 63/53 → 4/1）
2. 训练稳定、实现简单，作为 Dense Span 的对照基线
3. v11 失败点（outside recall 下降、>8s residual 增加）同样通过
   run-balanced emission CE auxiliary 缓解

**架构**：
```
Same trunk as Dense Span
    ↓
Learned 2×2 transition matrix
    ↓
Training: CRF sequence NLL + 0.5 run-balanced emission CE
Runtime:  exact Viterbi argmax
```

**Loss 配置**：
```
total_loss = crf_nll + 0.5 * run_balanced_emission_ce
```

- `crf_nll`：连续 definite-owner run 上的 sequence NLL，
  unsure/non-owner 不进入 score 或 normalization
- `run_balanced_emission_ce`：先对每个 truth run 内部平均，再按 class 平均；
  防止长 vocal runs 主导梯度、保护稀有 non-vocal 证据

### 2.3 保留为 baseline：argmax-CE

仅作为 fragmentation baseline 对照，不是 production 候选。

### 2.4 不训练的臂

- **argmax-structured**（CE + run-balanced CE + adjacency）：patch 式方案，
  结构化能力弱于 Dense Span/CRF，从 4 臂缩减为 2+1 臂以节省算力
- **Query-Mask**：v11 结果混合（continuity 改善但 outside recall/event
  coverage 退化），且 query 容量需按 train topology 冻结，复杂度高收益低

## 三、Loss 与激励设计细节

### 3.1 共同原则

1. **无 runtime 规则**：不使用 threshold、hysteresis、NMS、时长规则、
   gap merge、fallback；runtime 只有 exact argmax/Viterbi/DP
2. **unsure=-100**：不进入 loss、normalization、metrics、gate
3. **类平衡通过 loss 结构解决**，不用 class weight/Focal 预先掩盖
   分布问题（v10/v11 的既定结论）
4. **激励对齐**：任何 loss 项都不能单独奖励"预测更多 vocal 面积"——
   这是 v11 structured-hinge-only 桥接背景的根因

### 3.2 Run-balanced 的必要性

JAV 数据的天然分布：vocal runs 长（几十秒），non-vocal runs 短（1-5 秒）。
逐帧平均 CE 会让 non-vocal 的梯度贡献被稀释到 <10%，模型学成"全保留"。

Run-balanced：每个 truth run 先内部平均 → 每类 runs 再平均 → 两类等权。
一个 3 秒的 non-vocal run 与一个 60 秒的 vocal run 贡献相同的梯度。

### 3.3 Auxiliary weight 的选择

初始 0.5，基于 v12 smoke 已验证的配置。如果 val selection 发现：
- outside recall 不足 → 提高到 0.8
- continuity 不足 → 降低到 0.3
一次只调一个变量，做同 seed A/B。

## 四、Selection Metric（词典序，非单一分数）

### Gate 1: Safety Floor（必须全部通过）

1. vocal frame recall ≥ 95%
2. zero complete vocal-run deletions
3. all-vocal source keep recall ≥ 95%（该层存在时）

### Gate 2: Weakest-Link 最大化

在通过 Safety Floor 的 checkpoints 中，最大化以下各项的最小值：
- non-vocal frame recall
- non-vocal event recall
- vocal continuity
- all-nonvocal full-drop recall（该层存在时）

### Gate 3: Tie-breaker

- 更少 internal holes
- 更少 excess vocal-run count
- 更少 overmerged non-vocal duration

### Gate 4: 人工审计

1. Val 上选 epoch 和 decoder，test 只跑 val 前两名
2. 生成 side-by-side 人工审计页
3. 人工 gate 通过后才注册 production checkpoint
4. 注册前完整重放 Scorer→Split→CueQC→Inner→ASR 工作流

## 五、训练超参数

```
seed = 117（冻结）
trunk = P2048/H256 Bi-Mamba（与 smoke 相同）
context_frames = 1000
overlap_frames = 200（midpoint unique ownership）
batch_frames_budget = 1600
max_batch_rows = 8（防止 background-only batch 过大稀释梯度）
optimizer = AdamW(lr=1e-4, weight_decay=0.01)
max_epochs = 20
early_stopping_patience = 3（val weakest-link 无改善即停）
```

资源合同：0.95 物理 RAM/VRAM 上限、shared-VRAM spill soft-OOM、
臂间显式释放 CUDA、原子写 progress/checkpoint。

## 六、执行计划

| 阶段 | 任务 | 状态 |
|------|------|------|
| 1 | 修复 v5 → v6 prompt，更新 contract SHA | ✅ 完成（prompt 回退见 afa5398；SHA 须从实际导入常量计算，正确值 `43093fd0...` 随本文档提交）|
| 2 | 删除 Outer，简化 pipeline | 📝 设计文档完成，代码待改 |
| 3 | v6 test5 验证（确认恢复正常分割） | ✅ 完成：含实质背景区的样本正常分割（294-w01 为 6 vocal / 5 non_vocal），v5 的 5/5 单段回归消除；亚秒级停顿按仲裁规则并入包络属预期。审计页 `agents/temp/20260727_003000_scorer-v12-test5-v6-verification/audit_html/`，2026-07-26 用户人工审听 5/5 通过 |
| 4 | v6 full144 标注（OpenRouter 16 workers） | 🔄 进行中 |
| 5 | heldout 人工 gate + canonical 编译 | 等待标注 |
| 6 | 训练 Dense Span（主）+ CRF（对照）+ argmax-CE（baseline） | 等待 canonical |
| 7 | Val selection → test top2 → 人工审计 | 等待训练 |
| 8 | Split v4 输入 schema 改绑 Scorer v12，重编特征重训 | 等待 Scorer 晋升 |

## 七、风险与备选

### 风险 1：Dense Span outside recall 仍不足
- 备选 A：auxiliary weight 0.5 → 0.8
- 备选 B：切换 CRF 为主臂
- 备选 C：扩充 all-nonvocal 训练覆盖（数据问题优先于 loss 调参）

### 风险 2：Continuity 仍不足
- 备选：加入 continuity auxiliary（惩罚 truth vocal run 内部的预测切换），
  weight 0.2 起步做同 seed A/B

### 风险 3：删除 Outer 后 end-to-end 退化
- 验证方式：完整工作流重放对比
- 如 end-to-end 指标下降 >2%，重新评估（详见 Outer 删除文档）

## 八、明确不做的事

1. ❌ Teacher 滑动窗口（碎片化已证实）
2. ❌ v5 式固定时长阈值 prompt
3. ❌ runtime threshold/NMS/时长规则/gap merge/fallback
4. ❌ class weight/Focal 掩盖分布问题
5. ❌ 加载 v10/v11 权重 warm-start（全部 seed=117 随机初始化）
6. ❌ 在 test 上调参或选 epoch
