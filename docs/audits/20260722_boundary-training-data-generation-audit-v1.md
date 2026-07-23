# 1.7B Boundary 训练数据生成链职责审计 v1

本审计只覆盖 1.7B。0.6B 空 registry placeholder、checkpoint、runtime 和 data 均未修改、未训练。中央兼容键固定为 `boundary_acoustic_binary_v12`。

## 最终结论

Scorer v11 当前职责 canonical 已闭合，但仍需针对新音频重新提取 raw PTM2048、重编 window features、随机初始化训练并完成人工 gate。旧 synthetic 审计发现 outside control、bracket、overlay和内部组件大量依赖短音频tile；当前生成器已禁止所有重复，退役320条20s mosaic及432条需要重复的full-candidate overlay。125条train真实source不再要求高成本全量人工逐帧标注：Gemini只做高召回预审，其inside/unsure不作truth；outside补集再经1.7B ASR，ASR非空只转unsure，只有Gemini outside且ASR空、无错误才成为outside。最终canonical为train/val/test=`1146/10/14`，全量frame=`inside 528479 / outside 121553 / unsure 60432`，SHA=`957e8758...d9902b`。Proposal v1、Outer v3尚无可复现的当前职责数据闭环；Split v4、CueQC v13、Inner v2的现役权重来自旧provenance，不能用修改metadata或重新绑定SHA冒充新合同。

| 模型 | 当前职责 | 合法训练样本 | 审计结论 |
| --- | --- | --- | --- |
| Scorer v11 | 高召回连续 candidate-island membership | 完整 source window；明确/很可能的词语或对白锚点及同一轮连续包络为 `inside_candidate`；明确无词义且可独立安全删除的纯非语义声为 `outside_candidate`；词语与呻吟/噪声无法区分时为 `unsure=-100` | no-tile canonical与held-out 24/24已闭合；train definite比例`inside 89.49% / outside 10.51%`，不预先用重复或loss调权平衡。新raw PTM2048尚未提取，故训练待启动；旧v10数据不可改名复用 |
| Proposal v1 | 为 Split 枚举高召回非绑定候选 | 当前 Scorer v11/Outer v3 输出中的真实可查询候选；监督目标是候选覆盖，不是最终 cut | 现役 checkpoint 的 labels/feature manifest 已丢失，且 metadata 无 source/core/partition provenance；只能保留为历史候选源，不能复现训练 |
| Outer v3 | 将完整 candidate island 收成供 Split 查询的 acoustic outer core | 实际 `post_candidate_island_scorer_v11_islands`，真实边缘分布，随机初始化二分类 | trainer 合同已严格，但仓库没有生成这种 row 的 current compiler；registry 仍为空 `pending_outer_v3_audit` |
| Split v4 | 对 Proposal candidate 做 `cut/continue` 二分类 | 当前 Scorer v11→Proposal→Outer v3 的 candidate query、局部 bins/scalar 和完整 island context；source/core 固定且同 core 最多一次 | 旧 dataset 缺失；现役 checkpoint含临时 forced-train/repeat/Focal/aux 配置。新入口已禁止临时分区、重复 core 和旧 runtime 数据伪装，默认回到 neutral CE baseline |
| CueQC v13 | 对 provisional sub-island 做 `drop/keep` argmax | 实际 Split 输出；每个 sub-island 保留完整序列上下文；显式 source/core/partition，精确 Split/Inner SHA | 现役 bundle 1039 groups/4646 rows，但 group `source_id=0/1039`，无 Split/Inner SHA、无中央合同；checkpoint仍为 `role_holdout`，并绑定旧 Split SHA，必须重编数据后重训 |
| Inner v2 | CueQC keep sub-island → acoustic semantic core | post-CueQC keep、真实 acoustic edge；clean 与 noisy-edge/hardmix；随机初始化；background/semantic_core，unsure=-100 | 现役 manifest 3593 rows，但 source/core/input distribution/CueQC keep/SHA 均缺失；现役 checkpoint从 Inner v1 warm-start，不能视为当前合同闭合 |

## 证据

### 现役 checkpoint

- Scorer v8 SHA `9d72382b212d84f2180237dac35a644017a3d7aa455a95ec1e82b86e3747b117`：单 `speech_prob`、threshold/hysteresis 历史链，只作审计参考。
- Proposal v1 SHA `e8595cd7cdc2106562cf2ac3a37cf88e18fcc62ebe635ca12df3314c78df139b`：metadata 指向 `agents\temp\20260704_162852_real-omni-drop-hardmix-smoke4096\labels.jsonl` 和 `...\feature_manifest.jsonl`，两者均不存在。
- Split v4 SHA `d35844084d434cd7796b0af269a41044016bd2bae9bc901f469bfe273d0277c0`：dataset `agents\temp\20260716_103000_acoustic-split-v3-hardmix-runtime-dim2048.npz` 不存在；metadata记录 forced train groups=5、`manual_group_repeat=32`、cut weight=3、Focal=1.5、role/pair auxiliary=0.3。
- CueQC v13 SHA `49b6c0b8206cbdfdcc818a53a49ac036b6c1c30a67d72c223abbab389ef6d2ae`：`split_mode=role_holdout`；绑定 Split SHA `3aedc0e5131c154332d15c983a6219d1771783fb7c19c6c7be80d7c28a26e69d`，与现役 Split 文件 SHA `d358...` 不一致；Inner SHA仍为 `cddf...`。
- Inner v2 SHA `cddf86f863d397c9b51779d3bc64db6a768b05fb132bc7b61be9c186efde2310`：metadata明确从 Inner v1 SHA `fe2c21ba...` warm-start，违反当前随机初始化合同。

### 现存数据产物

- CueQC bundle `agents/temp/20260717_232705_cueqc-v13-runtime-v11-features/cueqc_v13_features.pt`：1039 groups、4646 rows、train/val/test=`883/105/51`；0 个 group 有 `source_id`，bundle 无 Split SHA、Inner SHA或中央合同 ID。
- Inner manifest `agents/temp/20260718_095500_inner-v2-exact-frame-dataset-full1024-bg/manifest.jsonl`：3593 rows、1022 unique sample IDs、train/val/test=`3070/354/169`；`source_id/core_id/input_distribution/cueqc_label=keep/cueqc_checkpoint_sha256` 全部缺失。

## 已修复的数据生成入口

### Scorer v11

- 旧synthetic component repetition审计确认：outside control=`1600/1600 repeated`、outside bracket=`2073/2252 repeated`、overlay=`430/432 repeated`，内部gap/negative也大量重复。修复后实际全量复核为left/right/negative各`870/870 natural`、outside bracket=`2252/2252 natural`，输出sample与可用source sample逐项相等；outside control与重复overlay均为0。
- 真实train outside teacher链固定为Gemini预审→outside complement→1.7B ASR。49段中ASR非空27段同时混有真实台词和喘息/亲吻拟声，因此ASR文本不作inside truth；空文本也不能单独证明outside。最终只接受Gemini outside与ASR empty/error-free交集，20条输出source为`outside 14506 / unsure 60432`，其余一律ignore。
- canonical compiler只接受中央合同、frozen source/video/partition和完整覆盖区间；标签限定`outside_candidate/inside_candidate/unsure`。held-out必须人工完整确认，train real outside则必须使用独立masked schema并证明Gemini inside未作truth、ASR text未作inside、ASR empty未脱离Gemini outside单独使用。当前24条held-out final verdict与20条real train masked source均通过。
- feature compiler定义独立raw cache、extractor、signed manifest和gate schema；只接受当前1.7B PTM的raw PTM2048+MFCC40，旧PTM128/PCA/前128截断直接拒绝。训练row引用完整source cache而不复制重叠特征，记录1000-frame context、200-frame nominal overlap和midpoint unique owner；loss/metrics只计算owner。
- `unsure`保留在source labels，训练映射`-100`；MFCC normalization、CE、metrics、numeric gate和heatmap auxiliary均排除unsure。heatmap target先从完整source definite run生成再切window，touching unsure不生成伪边界。
- trainer只允许random init、two-logit CE和argmax；baseline固定class weights=`1/1`且auxiliary=`0`，heatmap只能通过显式`heatmap_aux` A/B启用。CPU仅允许plumbing smoke；正式CUDA执行物理RAM/VRAM×0.95、Windows shared-VRAM spill soft-OOM、frame-budget batch及阶段结束显式释放。smoke checkpoint固定`promotion_allowed=false`。
- 新增聚焦测试=`26 passed`；使用项目内`agents/temp`作为pytest basetemp的全量结果=`917 passed / 6 skipped`。默认系统tmp位于C:时，一个既有job-temp测试会对D:项目根调用`Path.relative_to`并跨卷失败；这不是Scorer改动，按项目临时产物约束使用工作区内basetemp后全量通过。

### Split v4

- joint compiler 现在必须读取冻结的 `source_id` 和显式 train/val/test，从 source absolute coordinates 派生 core identity，并拒绝 source/core leakage、重叠 window 重复 core、重复 sequence。
- Omni prepare/resume 会验证 source identity、partition 和坐标；未闭合的 runtime 导出固定 `semantic_split_training_manifest_allowed=false`。
- dataset loader不再从 group 名猜 identity；merge不再支持 row-wise legacy、repeat 或分区重写；manual override不能把 held-out source搬入 train。
- refresh/rehydrate/reexport 和旧 threshold/hysteresis runtime builder明确为 audit-only，输出 `training_manifest_allowed=false`。
- trainer默认 `repeat=1`、class weights=`1/1`、Focal=`0`、role/pair auxiliary=`0`；只有固定 A/B 明确改善后才允许改变。

### CueQC v13

- canonical、runtime teacher、feature compiler和trainer均要求冻结 source identity、显式 partition、当前中央合同及精确 Split/Inner SHA；未批准 runtime row在 feature 编译阶段即拒绝。
- Runtime exporter把每个 provisional sub-island绑定到精确 `source_core_ids`；同一 core落入多个 sub-island会阻止训练，避免把工作流碎片化重复计作独立样本。
- feature bundle和 merge保留完整 group context，禁止 `max_chunks` 截短、role重分区、duplicate sub-island或duplicate core；trainer只允许 `fixed_partition`。
- 旧 Omni-drop negative exporter不再用 video hash重新划分 train/val/test，只接受上游冻结的 `source_id/source_partition`。
- unique-core composite修复了 quota 循环缩进错误，要求 negative source identity、同 partition合成、每 partition非空、core全局最多一次。

### Outer v3 / Inner v2

- Outer trainer继续要求 `post_candidate_island_scorer_v11_islands`、精确 Scorer schema/SHA、source/core隔离和 training gate；缺的是上游数据 compiler，不是放宽 trainer。
- Inner trainer现在要求 `post_cueqc_keep_provisional_subislands`、`cueqc_label=keep`、精确 CueQC SHA、source/core/sub-island identity、固定 partition和 max core use=1；只允许随机初始化。
- PTM/MFCC/labels/weights frame mismatch不再静默取最短长度，直接拒绝错误缓存。

### 旧生成器隔离

`build_galgame_synthetic_timeline.py` 的 CLI 仍保留用于历史分布实验和供新 builder复用纯音频 helper，但它会自行 hash partition且缺当前 core provenance，因此所有直接输出均标记 `training_manifest_allowed=false / legacy_hash_partition_synthetic_timeline`。它不能作为 Scorer v11、Split v4、CueQC v13、Outer v3或Inner v2的 current compiler。

## 合法重训顺序

1. 完成 Scorer v11 canonical compiler、raw PTM2048/MFCC feature compiler和随机初始化 trainer；固定真实 train/val/test source，完成 prediction-drop/truth-keep、held-out hard case、连续性和 `>8s` residual人工 gate。
2. 用晋升后的 Scorer v11重放真实 source，重新审计 Proposal candidate recall；若 Proposal继续使用，必须生成可复现的 source/core/partition manifest和当前 checkpoint metadata。
3. 生成真实 `post_candidate_island_scorer_v11_islands`，训练并人工 gate Outer v3。
4. 用最终 Scorer→Proposal→Outer 输出生成 Split v4 candidate-query dataset；先 neutral baseline，再按固定数据做 loss/architecture A/B。
5. 导出真实 provisional sub-islands，重新做 CueQC teacher/canonical/feature compile；fixed partition随机初始化重训 CueQC，不允许只 rebind旧 checkpoint。
6. 只从新 CueQC argmax keep输出编译 Inner v2数据，随机初始化重训并做 start/end coverage最高95%数值 gate及人工 zero-clipping/zero-true-speech-deletion gate。
7. 最后才运行 Scorer→Proposal→Outer→Split→CueQC→Inner 的 batch/full equivalence、VRAM lifecycle、sample-c/sample-a、OOM和完整 workflow smoke。

在步骤1开始前，本轮不启动任何 GPU训练，也不修改任何生产 checkpoint或registry。
