# Scorer v11 mixed-source full-source gate 与 teacher 分层指标

## 最终结论

这轮实验确认用户指出的例外必须进入选优：Gemini teacher 的 held-out source 中确实存在整条都是非语义的 source，不能只看 inside recall 或 continuity。

本轮冻结同一 canonical、partition、raw PTM2048/MFCC40、seed=`117`、P2048/H256 trunk、1000-frame ownership 和 `1/1` class weight，只比较 plain frame baseline 与 Frame–Event Query-Mask。两臂都保留为离线诊断，均未通过最高 `95%` 数值 gate，均不注册生产 registry。

- baseline 的 outside recall 较高，且 2 条全-outside test source 中有 `1/2` 被整段 drop；但 truth-run continuity 只有 Val/Test=`68.18%/68.42%`，仍有大量碎片。
- Query-Mask 把 continuity 提高到 `93.18%/89.47%`，internal gap 降到 `27/6`，但 outside recall 降至 `40.96%/54.35%`，2 条全-outside test source 均未整段 drop（`0/2`）。
- 因此不能用“连续性最高”单独选 Query-Mask，也不能用“全保留时 continuity 高”掩盖 teacher outside；当前没有可晋升的 Scorer v11 checkpoint。

## Teacher 分层与指标合同

新 canonical 共 `1176` 条，train/val/test=`1152/10/14`，frame=`inside 573995 / outside 124776 / unsure 34522`。逐 source 检查得到：

| partition | mixed inside+outside | all definite outside | unsure-only |
|---|---:|---:|---:|
| train | 1150 | 1 | 1 |
| val | 10 | 0 | 0 |
| test | 12 | 2 | 0 |

`all definite outside` 由 teacher canonical 的 definite outside 帧占满 source 的 definite 部分；若存在 unsure，仍不把 unsure 当 background。该层单独报告：

- `all_outside_source_drop_recall`：该类 source 中，模型在所有 definite outside 帧上没有输出 inside 的 source 比例；
- `outside_source_macro_recall`：每个含 outside truth 的 source 先求 outside recall，再做 source 宏平均；只作诊断，不替代 frame/event gate；
- `truth_run_continuity`：仅对存在 definite inside truth run 的 source/runs 适用；无 inside truth 时不把“不适用”解释成模型失败；
- `outside_candidate_recall` 与 `outside_run_mean_recall`：分别衡量 outside 帧和每个 teacher outside event；两者都不能被 inside reward 抵消；
- `true_inside_deletion_count` 与 inside recall：仍是语义安全硬门。unsure=`-100`，不进入 loss、normalization 或 metrics。

选优合同为 `full_source_teacher_outside_continuity_v3`：先满足 inside safety（`>=95%` 且无完整 truth-inside run 被删除），再在 outside frame/event、适用的 continuity 和存在时的 all-outside source drop 之间取最弱项。start/end coverage 只作诊断，不再作为独立硬选优项。

## 冻结训练条件

- 数据：`agents/temp/20260724_212617_scorer-v11-mixed-canonical/canonical_sources.jsonl`，每个 source/core 只属于一个 frozen partition；未使用父标签继承或 source 随机冒充 holdout。
- 特征：`agents/temp/20260724_212617_scorer-v11-mixed-raw-complete/raw_feature_manifest.jsonl`，完整 raw PTM2048；signed windows=`1286/50/70`（train/val/test）。
- 训练：随机初始化、raw PTM2048 full-width adapter、hidden256 双向 Mamba2、two-logit softmax argmax；GPU/RAM 95% 预算，shared VRAM spill=`0`。
- baseline checkpoint：`agents/temp/20260724_212617_scorer-v11-mixed-ab/09-baseline-full-selection-v2/scorer-v11-full_p2048_h256-baseline.pt`，最佳 epoch=`11`，epoch 15 早停。
- Query-Mask checkpoint：`agents/temp/20260724_212617_scorer-v11-mixed-ab/10-query-mask-full-selection-v3/scorer-v11-full_p2048_h256-query_mask.pt`，最佳 epoch=`7`，epoch 11 早停；`K=8`，learned residual gate=`0.0251566`。

没有加入 threshold、hysteresis、时长规则、NMS、规则 fallback、Focal 或 class-weight 调参；Query-Mask 的 set loss 仍只作为预注册结构 A/B。

## Full-source gate 结果

以下数字来自独立 checkpoint replay，而不是训练窗口平均；同一 source 只聚合一次。

| partition | arm | inside recall | outside frame | outside event | outside source macro | continuity | internal gaps | all-outside drop |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| val | baseline | 95.77% | 59.89% | 37.31% | 59.35% | 68.18% | 70 | n/a |
| val | Query-Mask | 98.63% | 40.96% | 19.04% | 41.12% | 93.18% | 27 | n/a |
| test | baseline | 98.60% | 60.32% | 39.53% | 53.11% | 68.42% | 33 | 1/2 (50%) |
| test | Query-Mask | 99.38% | 54.35% | 30.32% | 47.78% | 89.47% | 6 | 0/2 (0%) |

完整 replay 还记录了 baseline 的 prediction-drop/truth-keep=`173/623`（test/val）帧、Query-Mask=`77/202` 帧；`>8s` residual 分别为 baseline=`5/1`、Query-Mask=`12/5`（test/val）。这些仅用于人工审计，不被固定时长规则自动判错。

两个全-outside test source 的具体表现：

- `匿名样片 R-5bd7f3141b-w03`：baseline outside frame=`94.93%`、仍保留 14 个 prediction island；Query-Mask outside frame=`64.48%`、58 个 island；两者均未整段 drop。
- `匿名样片 O-de8bf4a629-w03`：baseline 整段 drop 成功；Query-Mask outside frame=`97.28%`，但仍保留 20 个 island，因此 source-level drop 失败。

这正是“teacher 全部判定非语义”不能被 frame 平均或 continuity 指标隐藏的证据。

## 审计页

- baseline：<http://127.0.0.1:8080/agents/audits/20260724_233904_scorer-v11-mixed-baseline-full-source/>
  - `161` 项：22 条普通 full source、2 条 teacher all-outside source、131 条 prediction-drop/truth-keep、6 条 `>8s` residual。
- Query-Mask：<http://127.0.0.1:8080/agents/audits/20260724_233904_scorer-v11-mixed-query-mask-full-source/>
  - `90` 项：22 条普通 full source、2 条 teacher all-outside source、49 条 prediction-drop/truth-keep、17 条 `>8s` residual。

页面顶部明确说明 all-outside source 的连续性不适用，并允许单独判断“确实全是非语义、模型是否整段 drop”。人工 verdict 仍由用户填写，本文不伪造听感结论。

## 决策与下一步

1. 保留 baseline 与 Query-Mask 源码、strict checkpoint schema、full-source scorer、teacher 分层指标和审计页 adapter；不注册任何 checkpoint。
2. 当前诊断优先级是 baseline 的 outside 覆盖与 Query-Mask 的连续性之间的冲突，而不是继续提高 inside reward。全-outside teacher source 必须继续作为独立 held-out control。
3. 不在没有人工/下游证据时把 Query-Mask 直接替换 baseline；下一次结构或数据实验必须同时报告 mixed source、all-outside source、outside event 和 continuity 四类结果。
4. Proposal/Split/CueQC/Inner 的职责不回收到 Scorer：Scorer 仍提供高召回候选包络；内部独立背景的隔离与删除仍由下游完成。
5. 0.6B、生产 registry、旧 alias 和无关用户文件均未读取或修改。
