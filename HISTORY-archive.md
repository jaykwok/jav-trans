# HISTORY 归档：FusionVAD-JA / Forced Alignment 历史迭代记录

本文件收纳从 HISTORY.md 迁出的已沉淀实验迭代记录：FusionVAD-JA 版本迭代（v0 → v1.18）与 Forced Alignment / Chunk Packing 的 R14–R19 阶段流水账。内容原样保留（含原 `<details>` 折叠结构）。

---

<details>
<summary>历史：FusionVAD-JA / R14-R18 早期 VAD 与 chunk packing 记录</summary>

## FusionVAD-JA 版本记录

### v0 / v1-mini

目的：验证数据、feature cache、addition-fusion BiLSTM 训练链路能跑通。

- 真实 ja whisper 1.5B feature cache 验证：`whisper_dim=1280`、`mfcc_dim=40`、`frame_hop_s=0.02`。
- 早期 addition-fusion BiLSTM 可训练参数约 `1.94M`。
- v1-mini 使用 VoxConverse supervised-positive + MUSAN / synthetic negative，只能证明训练闭环可行，不能代表目标域泛化。

### v1.5

目的：引入 Galgame synthetic timeline v2、MUSAN negative gap、背景混合和 positive loss weight。

结果：

- posw2 + threshold `0.001` + pad `0.2s` 在人工 Galgame 上 precision `0.7310`、recall `0.9501`、F1 `0.8263`。
- threshold `0.0001` + pad `0.2s` 达到 recall `0.9838`、extra audio ratio `1.3809`。
- 结论：低阈值 + padding 是当前 high-recall proposal 模式的必要选择。

### v1.6 real-heldout

目的：从本地视频抽真实 held-out，人工标注 VAD 片段，验证 Galgame synthetic 是否泛化。

数据：

- 10 个本地视频各抽 8 条、每条 8s。
- 候选 80 条，人工导出 79 条。
- 强标签：`supervised=55`、`negative=24`，总时长 `632.0s`，speech frame ratio `0.6138`。

基线：

- `fusion_lite`：F1 `0.7969`、precision `0.8534`、recall `0.7475`。
- `whisperseg-adaptive`：F1 `0.7697`、precision `0.7404`、recall `0.8015`。
- FusionVAD-JA v1.5 posw2 threshold `0.00015` + pad `0.2s`：recall `0.9551`、precision `0.6941`、F1 `0.8039`、extra audio ratio `1.3761`。

结论：FusionVAD-JA 达成高召回目标，但会把更多 negative / no-overlap 音频送给 ASR；后续必须用 ASR / alignment downstream 验证多送代价。

### v1.8 / v1.9 ASR 与 alignment 清理

问题暴露：

- 旧规则里存在具体词黑名单、假名/呻吟短句 direct drop、工具签名特例、AnimeWhisper 后置括号/重复清洗、翻译前重复压缩等非泛化策略。
- 这些规则会误伤目标域真实文本，例如 `はぁ`、`うん`、喘息、呻吟和短促发声。

处理：

- 删除词表驱动 direct drop。
- ASR QC 高风险文本改为 review-only，不再提供 `ASR_QC_DROP_UNCERTAIN` 删除开关，最终字幕不因该诊断被清空。
- 建立 `display_text` / `align_text` 双文本策略。
- alignment 诊断增加 `forced`、`partial`、`nonlexical`、`vad_coarse`、`proportional`、`drop_or_review`。
- 失败样本池统一用 `failure_candidate` 和 `failure_bucket`。

匿名样片 A 当前规则复测：

- base：`806` segments、`829` cues、`8085` chars、fallback `172/337`。
- 200k SFT：`794` segments、`843` cues、`13846` chars、fallback `166/337`。
- full checkpoint-15500：`802` segments、`870` cues、`15203` chars、fallback `170/337`。

结论：full SFT 方向成立，但主要瓶颈已经转向 alignment / fallback / QC，而不是 ASR 是否能输出文本。

### v1.10 / v1.11 synthetic timeline

v4 证明 crossfade、背景混合、overlap speech 和 `boundary_manifest.jsonl` bench 可用，但 gap 太短，speech frame ratio 约 `0.83-0.84`，不适合作为长期基线。

v5 long-gap 成为默认生成口径：

- train/val/test：`256/64/64` 条。
- speech frame ratio：`0.574/0.551/0.568`。
- 总时长 p50 约 `17s`，p90 约 `22s`。
- 默认启用长 gap、`speech_label_pad_s=0.08`、real negative gap 概率 `0.75`、背景混合概率 `0.5`。
- 支持 5-30ms equal-power crossfade、随机 gain、轻量 filter、低概率 codec、overlap speech。

v1.11 训练结果：

- 混合 v1-mini strong/negative `302` 条 + synthetic v5 `256` 条。
- val sweep 选 threshold `0.02`。
- test padded recall `0.9934`、missed speech `4.18s`、extra audio ratio `1.3240`。
- real-heldout recall 从 v1.5 的 `0.9556` 提升到 `0.9809`，missed speech 从 `17.24s` 降到 `7.42s`，extra audio ratio 升到 `1.5021`。

下游问题：

- 匿名样片 A 使用 v1.11 + Qwen3-ASR-1.7B full SFT checkpoint-21000，未做长段保护时只切出 `89` 个更长 VAD chunks。
- forced 仅 `14/89`，fallback `38/89`，failure candidates `76/89`。
- 主要 bucket 是 `empty_text_for_chunk` 和 `vad_coarse_alignment`。
- 结论：召回收益成立，但 chunk 边界/合并策略成为新瓶颈。

### v1.13

变化：切到 Qwen3-ASR-0.6B full SFT frozen feature，并把 synthetic v5 标签改为 exact speech-island。

synthetic exact-island test64：

- threshold `0.10` + pad `0.2s`：recall `0.9935`、missed `1.82s`、extra audio ratio `1.6012`。
- start/end p50 约 `0.628s/2.002s`。
- 主要收益是 start 边界更接近真实 speech island。

匿名样片 A downstream：

- 对比 v1.11 framepack baseline，chunks `240 -> 227`。
- fallback chunks `137 -> 114`。
- `vad_coarse_after_sentinel 122 -> 104`。
- forced `101 -> 106`。

结论：方向改善，但还不能替代后续 alignment repair。

### v1.14

变化：在 v1.13 上做 boundary-aware fine-tune，加入 `boundary_loss_weight=0.25` 和 `gap_loss_weight=0.10`。

结果：

- synthetic 上有信号。
- 真实 held-out 和匿名样片 A downstream 未过 gate。
- 匿名样片 A：chunks `222`、segments `986`、fallback chunks `115`、`vad_coarse_after_sentinel=103`、forced `101`。

结论：boundary-aware loss 方向保留，但 v1.14 不替换默认。

### v1.15

变化：明确改成 endpoint / boundary refiner，不再只是单头 speech VAD。

输出头：

- `speech`
- `start`
- `end`
- `cut`

训练目标：

- `speech` 继续服务 recall。
- `start/end` 学 speech island 边界。
- `cut` 学长 gap / 内部非语音可切点。
- 允许 end 偏长一点，但禁止 fallback chunk 长到 20-30s。

558-row checkpoint 未过 gate，但证明四头训练入口可行。

### v1.16

变化：扩大到 4096 条 Galgame multi-island synthetic。

结论：

- synthetic boundary gate 明显改善。
- 说明 synthetic exact-island 数据规模对 boundary refiner 有直接收益。

### v1.17

变化：使用 32768 条 synthetic 训练 endpoint refiner，并提交小型 checkpoint 到仓库。

当前默认：

```env
FUSIONVAD_JA_CHECKPOINT=src/vad/fusionvad_ja/checkpoints/fusionvad_ja_v1_17_endpoint_refiner.pt
FUSIONVAD_JA_MODEL_PATH=models/jaykwok-Qwen3-ASR-0.6B-JA-Anime-Galgame
FUSIONVAD_JA_THRESHOLD=0.020
FUSIONVAD_JA_CUT_THRESHOLD=0.960
FUSIONVAD_JA_PAD_S=0.2
```

结论：

- v1.17 是默认 head 升级，能改善 synthetic boundary gate。
- 但匿名样片 A downstream 的 forced-aligner sentinel / unsafe fallback 未被根治。
- 后续重点不是继续提高 frame recall，而是 pre-ASR speech-island / boundary packing。

---

## Forced Alignment 与 Chunk Packing 记录

### R14 Phase 0

目标：确认 fallback 是否真的影响时间轴。

发现：

- `vad_coarse` 比 `forced` 时间轴差约 `2.16s p90`，超过门槛。
- forced 自身 p90 也约 `2.3s`，但主要来自 synthetic 真值边界模糊、crossfade/pad/transition 和 VAD pad，不代表 aligner 必然差。
- 根因链收敛为：high-recall VAD 把多 island + 长 gap 合成超长 chunk，Qwen forced aligner 在长 chunk + 大段非语音上吐 sentinel。

### R14 Phase 1a

尝试：`ASR_CHUNK_PACK_MAX_CORE_FRAMES=419`，只在长 gap 处拆超长 chunk。

结果：

- chunks `137 -> 148`，增长 `+8%`。
- forced `77 -> 84`。
- fallback `60 -> 64`。
- `vad_coarse_after_sentinel 25 -> 28`。
- 未过 gate。

结论：只在长 gap 处拆超长 chunk只能改善一部分粗时间轴误差，不能解决 sentinel。

### R14 Phase 1b

处理：`nonlexical` / `align_text_empty` 显式分流。

结果：

- 纯省略号/符号保留 display_text 并走粗时间轴。
- 不再计入真正 `vad_coarse` fallback。
- 剩余瓶颈集中在 `vad_coarse_after_sentinel` 非空文本块。

### R14 Phase 1c

尝试：`ALIGNMENT_SENTINEL_ISLAND_SPLIT=1`，只对 `vad_coarse_after_sentinel` 的非空文本 chunk 做 aligner-local speech-island splitting。

synthetic64：

- fallback `28 -> 11`。
- `vad_coarse_after_sentinel=11`。
- gate `PASS_RECLASSIFICATION_CLEANUP`。

匿名样片 A：

- chunks `240 -> 267`。
- forced `101 -> 154`。
- fallback `137 -> 48`。
- `vad_coarse_after_sentinel 122 -> 42`。

问题：

- 初版每个失败 chunk 串行卸载/重载 ASR 与 aligner。
- 耗时约 `1053.5s -> 3150.4s`。

### R14 Phase 1d

优化：staged batch island retry。

流程：

```text
收集全部 sentinel chunk
-> 一次性物化 island clips
-> 批量 ASR
-> 卸载 ASR
-> 批量 forced align
-> merge 回原 chunk
```

结果：

- synthetic64 指标与 Phase 1c 对齐。
- 匿名样片 A：forced `154`、fallback `48`、`vad_coarse_after_sentinel=42`。
- ASR+Alignment `1613.1s`、总计 `1641.7s`。

结论：适合作为 opt-in repair / 质量上限参考，不宜默认开启。

### R15 / R16

路线改为 pre-ASR speech-island / boundary-aware chunking：

- 不回到 FSMN。
- 不默认引入 pyannote。
- CAM++ / 3D-Speaker / WeSpeaker 只作为 speaker sidecar，在 speech-island 足够细后辅助判断相邻 island 是否跨 speaker。
- rule-based valley split 覆盖风险高，chunk 增幅过大，不进默认。
- 验收指标不只看 forced 数，还看 ASR empty、unsafe fallback、fallback duration、SRT 观感。

### R17

尝试：使用 v1.17 endpoint refiner 的 `cut` score 做 opt-in pre-ASR cut split。

离线样片 A：

- threshold `0.96`：chunks `241 -> 267`，增长 `1.11x`。
- threshold `0.95`：chunks `241 -> 287`，增长 `1.19x`。
- threshold `0.94`：chunks `241 -> 313`，增长 `1.30x`。

GPU 闭环 threshold `0.95`：

- chunks `241 -> 258`，增长 `+7.1%`。
- forced `105 -> 123`。
- `vad_coarse_after_sentinel 114 -> 113`。
- unsafe fallback `114 -> 109`。
- fallback safe ratio `0.0 -> 0.035`。
- fallback duration p90 仍为 `28.47s`。

结论：

- cut score 能改善少量 forced alignment。
- 它没有解决长连续 chunk 的粗 fallback。
- 不默认启用，不继续只扫 cut threshold。
- 下一步应转向更强的 pre-ASR boundary packing / endpoint refiner。

### R18

动机：R17 证明局部 cut score 有信号，但不能解决 sentinel / unsafe fallback。根因更像全局 packing 问题：长 chunk、多个 speech island、内部 gap、连续长人声、cut/valley 信号和 fallback 风险需要一起决策。

参考思路：

- WhisperX 的 VAD Cut & Merge 证明长音频 ASR 前需要显式切分和合并策略。
- Semantic VAD / endpoint detection 说明 endpoint 应作为独立目标，而不是只做 speech/non-speech。
- streaming ASR endpoint detection 的辅助 SAD / endpoint loss 思路适合迁移为 start/end/cut/risk 多头。

已落地的最小版本：

- 新增 env-gated `ASR_PRE_ASR_RISK_SPLIT_*`，默认关闭，不改变正式默认。
- 新增 `r18_pre_asr_risk_v1` policy：先计算 chunk fallback 风险，再选择边界。
- 风险因子：long core、unsafe duration、multi island、internal gap。
- 切点优先级：明确 internal gap -> endpoint cut score -> low VAD valley。
- 输出 metadata：`risk_split_count`、`risk_score`、`risk_reasons`，并进入 transcript chunk annotation 和 VAD chunk cache。
- `run_full_workflow.py` 已透传 R18 env，避免 GPU 闭环时参数只存在于父进程。

当前状态：

- 单测覆盖默认关闭、多 island 长 chunk 切分、连续长 chunk 使用 cut score、cache key 变化、cache round-trip、ASR stage env 透传和 full workflow env 透传。
- 这仍是 rule / cost packer 的第一版，不是最终模型。GPU 小闭环已经证明 R18 gap-first 没有显著解决 `vad_coarse_after_sentinel`、unsafe fallback 和 fallback p90。

实施验收记录：

- 代码范围：`src/audio/chunk_packer.py`、`src/whisper/pipeline.py`、`src/whisper/vad_chunk_cache.py`、`src/core/config.py`、`.env.example`、`tools/vad/fusionvad_ja/run_full_workflow.py`。
- 测试范围：`tests/test_chunk_packer.py`、`tests/test_vad_chunk_cache.py`、`tests/test_asr_stage_env_scope.py`、`tests/test_pipeline_chunk_config_runtime.py`、`tests/test_run_full_workflow_env.py`。
- 验收命令：`.venv/bin/python -m pytest tests/test_chunk_packer.py tests/test_vad_chunk_cache.py tests/test_run_full_workflow_env.py tests/test_asr_stage_env_scope.py tests/test_pipeline_chunk_config_runtime.py -q`。
- 结果：`44 passed`，仅有 Codex sandbox 内 NVML 初始化 warning；不影响 packing / cache / env 透传结论。
- README 决策：不更新。R18 仍是 opt-in 实验路线，默认关闭，不改变新用户安装、默认工作流或分发说明。

离线复算：

- 工具：`tools/vad/fusionvad_ja/analyze_r18_risk_splits.py`。
- 输入：匿名样片 A v1.17 endpoint-refiner 的 VAD cache、diagnostics、R17 frame/cut score。
- 输出：`agents/temp/fusionvad-ja/r18-risk-split-offline-sample-a*/summary.json`、`summary.md`、`risk_split_plan.jsonl`、`simulated_chunks.jsonl`。
- 口径限制：离线复算只重打 cached VAD segments，并用 core overlap 把旧 diagnostics 映射到新 chunk；不跑 ASR / forced aligner，因此只能评估 chunk 分布和风险覆盖，不能替代 GPU 闭环。

| 参数 | chunks | 增幅 | sentinel 风险旧 chunk 被拆 | duration p50/p90 | 结论 |
|------|--------|------|-----------------------------|------------------|------|
| 默认 R18 `risk=1.0,gap=6` | `241 -> 372` | `1.544x` | `56/114` | `16.34/28.47s` | 覆盖较多，但 chunk +54%，过激，不适合作为默认 GPU 闭环起点。 |
| 保守 `risk=2.0,gap=6/12/18` | `241 -> 268-269` | `1.112-1.116x` | `11/114` | `27.00/28.47s` | 成本可控，但只覆盖少数 sentinel 风险。 |
| 更保守 `risk=2.5,gap=6/12/18` | `241 -> 253-258` | `1.050-1.071x` | `2-6/114` | `27.29/28.47s` | 太保守，对粗 fallback 基本无杠杆。 |

结论：

- 只靠 R18 的 rule/cost packing 不能十拿九稳解决 p90 粗 fallback；一旦控制 chunk 增幅，能处理的主要是少量多 island / internal gap chunk。
- 剩余瓶颈主要落在连续长 island / overlong chunk：旧 cache 中大量 chunk 本身已被 hard-cap overlong 切到接近 `28.47s`，即使重新 packing，fallback duration p90 仍不动。
- 下一步不建议直接用默认 R18 跑 GPU；更合理路线是继续 endpoint/boundary refiner，让模型在连续长 island 内提供更可靠 cut/valley 信号，或把 R18 改成更明确的 fallback-risk objective 后再做小闭环。
- 当前 R18 保持 opt-in、默认关闭。

Netflix / WhisperX 复核后的策略修正：

- Netflix timed text 规则强调 cue 需要贴近对白起点、保持可读时长、字幕间保留最小 gap；通用字幕 event 通常不应长期接近 `20-30s`。日语规则还有更严格的行长和读速限制。
- WhisperX 类长音频 ASR 路线允许 ASR chunk 接近 `30s`，但那依赖后续 word-level forced alignment。当前项目在 forced aligner sentinel 时只能退回 chunk 粗时间轴，所以 ASR chunk 过长会直接污染 fallback 字幕。
- 因此 R18 不应简单照搬“30s ASR chunk 合法”或“7s 字幕 cue 最大”任何一边，而应先把明显多句、多 island、内部 gap 的 chunk 拆开；连续长语音则保守处理。
- 已新增 `ASR_PRE_ASR_RISK_SPLIT_CONTINUOUS_THRESHOLD=2.0`：`risk=1.0,gap=6` 时仍会积极切明确内部 gap，但没有内部 gap 的连续长 island 需要更高风险分才允许使用 endpoint cut / VAD valley 切。
- 该改动保持 R18 默认关闭，但改变 R18 opt-in 行为和 VAD chunk cache signature。

gap-first 离线复算：

| 参数 | chunks | 增幅 | sentinel 风险旧 chunk 被拆 | duration p50/p90 | 结论 |
|------|--------|------|-----------------------------|------------------|------|
| `risk=1.0,continuous=2.0,gap=6` | `241 -> 269` | `1.116x` | `11/114` | `27.00/28.47s` | 新默认实验档，主要只切明确 gap；成本可控。 |
| `risk=1.0,continuous=1.5,gap=6` | `241 -> 372` | `1.544x` | `56/114` | `16.34/28.47s` | 连续长 island 也被 cut score 大量切，回到旧激进行为。 |
| `risk=1.0,continuous=2.5,gap=6` | `241 -> 258` | `1.071x` | `6/114` | `27.29/28.47s` | 太保守，收益更小。 |

GPU 闭环：

- 命令脚本：`agents/temp/run_r18_gapfirst_sample_a_gpu.sh`。
- 工作流输出：`agents/temp/fusionvad-ja/full-workflow-qwen29239-sample-a-v1-17-r18-gapfirst/`。
- 诊断输出：`agents/temp/fusionvad-ja/diagnostics-sample-a-v1-17-r18-gapfirst/`。
- fallback-safe 指标：`agents/temp/fusionvad-ja/fallback-safe-boundary-metrics-sample-a-v1-17-r18-gapfirst/`。
- 对比表：`agents/temp/fusionvad-ja/r18-gapfirst-gpu-compare/summary.md`。
- VAD 日志确认使用 CUDA：`requested_device=cuda actual_device=cuda`。
- 全片匿名样片 A 耗时 `1116.54s`，ASR chunks `250`，输出 segments `899`。

| 版本 | chunks | forced | `vad_coarse_after_sentinel` | unsafe fallback | fallback p50/p90/max | ASR empty warn | QC reject |
|------|--------|--------|-----------------------------|-----------------|----------------------|----------------|-----------|
| baseline v1.17 | `241` | `105` | `114` | `114` | `28.47 / 28.47 / 28.47s` | `6` | `16` |
| R17 cut th0.95 | `258` | `123` | `113` | `109` | `27.41 / 28.47 / 28.47s` | `5` | `17` |
| R18 gap-first | `250` | `109` | `117` | `114` | `28.47 / 28.47 / 28.47s` | `8` | `16` |

结论：

- `risk=1.0,continuous=2.0,gap=6` 的 chunk 增幅可控，但真实 GPU 闭环没有过 gate：forced 只从 baseline `105 -> 109`，sentinel `114 -> 117`，unsafe fallback 不变，fallback p90 不变。
- R18 规则没有触达核心瓶颈。多数粗 fallback 仍是接近 hard-cap 的长连续 chunk / overlong chunk；只切明确 gap 的收益太小，而无差别切连续长 island 又依赖当前还不够可靠的 cut/valley 信号。
- R18 保持 opt-in、默认关闭。不建议继续围绕 `risk/gap/continuous` 做大规模扫参；下一步应训练更强的 endpoint / boundary refiner，或设计 fallback-risk objective，让模型直接学习“哪里适合切成一句台词”。

R18 后续 cut-signal 离线审计：

- 工具：`tools/vad/fusionvad_ja/analyze_fallback_cut_signal.py`。
- 输入：全量 `chunk_metrics.jsonl`，不是 `unsafe_fallback_chunks.jsonl` top-N；后者只保留最长 20 条审计样本。
- 目的：确认现有 v1.17 的 `speech/cut` 概率在 unsafe fallback chunk 内是否已经包含足够切点。如果已有信号足够，说明 packer 阈值还有空间；如果信号不够，说明必须改训练目标。
- 输出：
  - `agents/temp/fusionvad-ja/fallback-cut-signal-sample-a-v1-17-baseline-full/`
  - `agents/temp/fusionvad-ja/fallback-cut-signal-sample-a-v1-17-r17-full/`
  - `agents/temp/fusionvad-ja/fallback-cut-signal-sample-a-v1-17-r18-full/`

| 版本 | unsafe rows | 有 cut/valley 候选 | 可贪心拆到 9s 子 chunk | greedy 后 max-child p90 |
|------|-------------|--------------------|-------------------------|-------------------------|
| baseline v1.17 | `114` | `69` | `7` | `24.47s` |
| R17 cut th0.95 | `109` | `55` | `9` | `24.47s` |
| R18 gap-first | `114` | `67` | `8` | `24.47s` |

结论：现有 v1.17 cut/valley 信号只能覆盖约 `6-8%` 的 unsafe fallback，且 p90 子 chunk 时长不动。继续扫 R17/R18 规则阈值不是主杠杆；下一步应训练新的 boundary/cut head，使目标直接服务 fallback-safe “一句台词边界”，尤其是连续长 island 内的可切点。

### R19 / v1.18 训练目标调整

动机：R18 后的全量 cut-signal 审计证明，现有 v1.17 cut head 在 unsafe fallback chunk 内缺少足够可用切点。继续调 packer 只是在不存在的信号上扫阈值。

最小实现：

- 新增 `EndpointRefinerTrainConfig.cut_boundary_radius_frames`，默认 `0`，不改变 v1.17 旧行为。
- `endpoint_targets_from_record()` 仍保留原逻辑：长 gap（`gap >= cut_min_gap_s`）整段标为 cut。
- 当 `gap < cut_min_gap_s` 且 `cut_boundary_radius_frames > 0` 时，把相邻 speech island 的 `previous.end` / `current.start` 附近若干帧也标为 cut 正样本。
- CLI 新增 `tools/vad/fusionvad_ja/train_endpoint_refiner.py --cut-boundary-radius-frames`。

目的：

- v1.18 训练不再只让 cut 学“大段静音 gap”，还要让 cut/head 学“相邻台词边界附近可以切”。
- 这直接服务 fallback-safe chunk packing：即使 VAD high-recall 把连续人声或短 gap 合成一坨，也希望 cut head 能提供更密集、可解释的句子边界候选。
- 默认仍关闭，直到新 checkpoint 通过 synthetic boundary gate 与匿名样片 A GPU downstream gate。

验收：

- `tests/test_fusionvad_ja_dataset.py` 覆盖短 gap boundary cut target 与 CLI 参数校验。
- smoke：`agents/temp/fusionvad-ja/v1-18-cut-boundary-radius-smoke/`，1 step 训练成功，checkpoint config 写入 `cut_boundary_radius_frames=1`。

训练与 test64 阈值扫描：

- 训练脚本：`agents/temp/run_v1_18_cutboundary2_train.sh`。
- checkpoint：`datasets/train/fusionvad-ja/v1-18/qwen3-asr-0.6b-full29239/endpoint-refiner-boundary32768-cutboundary2-batch16-lr2e-4-steps2048-posaux120-cut8-nogap/fusionvad_ja_endpoint_refiner.pt`。
- 训练集：32768 条 Galgame synthetic exact-island / long-gap 样本，使用 Qwen3-ASR-0.6B full SFT frozen feature。
- 训练参数：batch 16，lr `2e-4`，2048 steps，`cut_boundary_radius_frames=2`，`positive_aux_weight=120`，`internal_gap_loss_weight=0`。
- 训练结果：final loss `0.8426`，frame accuracy `0.9378`，trainable params `1889252`。
- 预测导出：
  - v1.18：`agents/temp/fusionvad-ja/v1-18-cutboundary2-test64-predictions-th002-cut05/`
  - v1.17 对照：`agents/temp/fusionvad-ja/v1-17-test64-predictions-th002-cut05/`
- 阈值扫描：
  - v1.18 no-cut：`agents/temp/fusionvad-ja/v1-18-cutboundary2-threshold-sweep/`
  - v1.18 cut-applied：`agents/temp/fusionvad-ja/v1-18-cutboundary2-threshold-sweep-cut-applied/`

关键对比：

| 版本 / operating point | recall | missed speech | extra audio ratio | segments | start p50 | end p50 | cut gap coverage |
|------------------------|--------|---------------|-------------------|----------|-----------|---------|------------------|
| v1.17 `speech=0.020,cut=0.960,cut-applied` | `0.999992` | `0.005s` | `1.2545` | `192` | `0.305s` | `0.298s` | `0.984` |
| v1.18 no-cut best recall-safe `speech=0.030` | `1.000000` | `0.000s` | `1.3674` | `190` | `0.552s` | `0.776s` | `1.000` |
| v1.18 cut-applied best recall-safe `speech=0.030,cut=0.960` | `1.000000` | `0.000s` | `1.3329` | `193` | `0.484s` | `0.632s` | `0.905` |
| v1.18 cut-applied `recall>=0.999` best | `0.999623` | `0.229s` | `1.3152` | `195` | `0.411s` | `0.572s` | `0.937` |

结论：

- v1.18 的 cut-boundary 目标确实让 cut 更积极，但在 synthetic boundary gate 上没有超过 v1.17。
- 即使放宽到 `recall>=0.999`，v1.18 仍比 v1.17 多送音频、边界更粗。
- 不替换默认 head，不进入匿名样片 A GPU downstream 闭环。
- 失败原因更像训练目标仍不够直接：短 gap boundary 正样本会增加 cut 密度，但没有明确约束“fallback-safe 子 chunk 时长 / 一句台词边界 / 避免长连续 island 粗 fallback”。
- 下一步应设计 v1.19：显式 fallback-risk / max-child-duration / sentence-island objective，或做一个 post-VAD boundary proposal 模型，而不是继续在 v1.18 上扫阈值。

补充 fallback-safe synthetic gate：

- `tools/vad/fusionvad_ja/benchmark_boundary_predictions.py` 新增预测段级指标：
  - `fallback_target_duration_s`，默认 `8.0s`。
  - `fallback_gap_overlap_s`，默认 `0.5s`。
  - `long_predicted_segment_count / ratio`。
  - `predicted_gap_crossing_segment_count / ratio`。
  - `predicted_segment_duration` p50/p90/p95/max。
  - `predicted_gap_overlap` p50/p90/p95/max。
- 目的：模拟 forced aligner sentinel 时的最坏情况。如果某个 VAD/packing operating point fallback 后会生成过长 cue 或跨大段 truth gap，即使 frame recall 高也不能算过 gate。
- 新 sweep 产物：
  - v1.17：`agents/temp/fusionvad-ja/v1-17-endpoint-refiner-threshold-sweep-cut-applied-fallbacksafe/`
  - v1.18 no-cut：`agents/temp/fusionvad-ja/v1-18-cutboundary2-threshold-sweep-fallbacksafe/`
  - v1.18 cut-applied：`agents/temp/fusionvad-ja/v1-18-cutboundary2-threshold-sweep-cut-applied-fallbacksafe/`

fallback-safe 对比（`recall>=0.9999`）：

| 版本 / operating point | recall | extra | segments | long segments | gap crossing | pred dur p90/max | start/end p50 |
|------------------------|--------|-------|----------|---------------|--------------|------------------|---------------|
| v1.17 `speech=0.040,cut=0.960,cut-applied` | `0.999992` | `1.2063` | `171` | `29` | `13` | `8.620/12.225s` | `0.300/0.281s` |
| v1.18 no-cut `speech=0.030` | `1.000000` | `1.3674` | `190` | `40` | `67` | `9.222/17.940s` | `0.552/0.776s` |
| v1.18 cut-applied `speech=0.030,cut=0.960` | `1.000000` | `1.3329` | `193` | `37` | `48` | `8.996/17.940s` | `0.484/0.632s` |

结论：

- v1.18 的问题不是单纯阈值；在 fallback-safe 指标下也明显劣于 v1.17。
- v1.17 的 `speech=0.04,cut=0.96` synthetic gate 甚至优于早期 `speech=0.02,cut=0.96`，但是否替换默认 operating point 需要真实样片 GPU 闭环验证，不能只凭 synthetic gate。
- v1.19 训练目标应直接减少 `long_predicted_segment_count` 和 `predicted_gap_crossing_segment_count`，同时守住 near-1 recall。

匿名样片 A GPU 验证 `v1.17 speech=0.04,cut=0.96`：

- 脚本：`agents/temp/run_v1_17_th04_sample_a_gpu.sh`。
- workflow：`agents/temp/fusionvad-ja/full-workflow-qwen29239-sample-a-v1-17-th04-cut096/`。
- diagnostics：`agents/temp/fusionvad-ja/diagnostics-sample-a-v1-17-th04-cut096/`。
- fallback-safe metrics：`agents/temp/fusionvad-ja/fallback-safe-boundary-metrics-sample-a-v1-17-th04-cut096/`。
- 日志确认 VAD 使用 CUDA：`requested_device=cuda actual_device=cuda`。
- 全片耗时 `1056.7s`，ASR chunks `241`，字幕 segments `884`。

| 版本 | chunks | forced | partial | nonlexical | `vad_coarse_after_sentinel` | unsafe fallback | fallback p50/p90/max | ASR empty warn | QC reject |
|------|--------|--------|---------|------------|-----------------------------|-----------------|----------------------|----------------|-----------|
| baseline v1.17 `0.02/0.96` | `241` | `105` | `0` | `6` | `114` | `114` | `28.47 / 28.47 / 28.47s` | `6` | `16` |
| v1.17 `0.04/0.96` | `241` | `108` | `1` | `6` | `113` | `113` | `28.47 / 28.47 / 28.47s` | `6` | `13` |
| R17 cut th0.95 | `258` | `123` | `0` | `5` | `113` | `109` | `27.41 / 28.47 / 28.47s` | `5` | `17` |
| R18 gap-first | `250` | `109` | `0` | `8` | `117` | `114` | `28.47 / 28.47 / 28.47s` | `8` | `16` |

结论：

- `0.04/0.96` 在 synthetic gate 上更漂亮，但真实样片只带来很小变化：forced `105 -> 108`、sentinel/unsafe `114 -> 113`、fallback p90 不变。
- 不替换默认 operating point；继续保持 v1.17 默认，并把 `0.04/0.96` 记录为 synthetic 优但真实闭环收益不足的负例。
- 这进一步确认主瓶颈不是全局 speech threshold，而是 long overlong chunk / continuous island 内缺少可靠句边界。

- 2026-06-20 1.7B scorer workflow / CueQC 标注源 / REAL 性能口径：1.7B scorer 已作为默认进入 10 部 workflow，输出根目录 `agents/temp/speech-boundary-ja/20260620_005837_qwen17b-scorer-default-10film-cueqc-source/`，10 部全部 `done`，合计 segments `18210`、blocks `18098`，均使用 `speech_boundary_ja_feature_scorer.jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame.pt` 且 scorer repo id 匹配 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`。本轮实际参数：ASR batch size `32`，Boundary planner sequence batch size `256`，CueQC inference batch size 默认 `64`；Boundary / SpeechBoundary feature extraction另有内部窗口和 Qwen feature extractor `max_inference_batch_size=1`，不要和 ASR 转写 batch 混淆。
- CueQC 1.7B 标注源：从 10 部 aligned output 导出全量候选 `19552` 条；为避免 Torque 全量 O(n^2) 距离矩阵过大，第一轮只聚类高风险 density level（`empty_or_punctuation`、`short_vocalization_candidate`、`repeated_vocalization_candidate`），共 `4157` 条。按 NAMH-055 原型沿用 `structured + euclidean + merge-layer 1` 得到 `104` 个簇，产物在 `agents/temp/20260620_025639_cueqc-qwen17b-10film-cluster-source/cluster_highrisk_structured_layer1/`，其中 `cluster_label_audit.html` 用于按簇打 `drop/keep/mixed/skip` 标签，`cluster_label_template.jsonl` 是空标签模板。REAL-988 继续保留为测试集，不进入这批训练/标注源。
- CueQC 正向样本补充规则：如果簇级高风险标签之外还需要混合正向样本，正向样本从 anime 数据集抽取，NSFW:SFW 按 `1:1` 比例混合；REAL-988 仍只作为测试集，不进入训练或正向样本池。后续 CueQC 训练数据构造需在 metadata/summary 中记录 anime 来源、NSFW/SFW 计数和抽样 seed。
- REAL-988 后续性能测试口径：主 workflow 已新增 CUDA memory snapshot，写入 `timings.json -> asr_details.cuda_memory`（ASR 内部 split/load/transcribe/cueqc/unload/align/segment/total）和 `asr_details.pipeline_cuda_memory`（run/audio/asr/write 外层阶段），run log 同步打印 allocated/reserved/max_reserved/free/total MB；`tools.workflows.run_full_workflow` summary 会汇总 `cuda_memory_peak_reserved_mb`、`pipeline_cuda_memory_peak_reserved_mb`，并暴露 `--cueqc-inference-batch-size`，便于 REAL-988 用同一入口比较 ASR bs、Boundary sequence bs 和 CueQC bs。下一步测试 REAL 时先跑当前基线（ASR bs 32 / boundary seq 256 / CueQC bs 64），再只改一个 batch 参数做对比，重点看 `asr_text_transcribe_done`、`cueqc_done` 和 `max_reserved_mb` 是否接近共享显存边界。
- 2026-06-20 审计导航清理：正式审计页已放入 `agents/audits/20260620_025639_cueqc-qwen17b-10film-cluster-label-audit/index.html`，导航 `agents/audits/index.html` 和 `latest-audit.html` 已重建并指向该页；旧审计目录统一移动到 `agents/rm/audit-deletions/20260620-103946-*`，未硬删除。
- 验证：`uv run python -m py_compile src/asr/pipeline.py src/main.py tools/workflows/run_full_workflow.py` passed；`uv run python -m tools.workflows.run_full_workflow --help` 已确认 `--asr-batch-size`、`--boundary-planner-sequence-batch-size`、`--cueqc-inference-batch-size` 均可见。

</details>
