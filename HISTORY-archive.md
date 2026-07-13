# HISTORY 归档

本文件收纳从 [HISTORY.md](HISTORY.md) 迁出的已沉淀或被取代的历史实验记录，保持主文件聚焦于活跃路线与近期决策。内容原样保留（含原 `<details>` 折叠结构），仅做位置迁移，未改写事实。

收录的归档段：

- **FusionVAD-JA 版本记录（v0 → v1.18）与 Forced Alignment / Chunk Packing R14–R19**：早期 VAD 与 chunk packing 的版本迭代流水账，已被当前的 SpeechIslandScorer + 语义切分管线取代。
- **旧实验记录（2026-06-18 SpeechBoundary-JA v3 训练流水账）**：v3 frame-boundary scorer 的阈值 / 数据 / 训练迭代细节，已被 v8 speech-island scorer 取代。
- **Qwen3-ASR SFT 路线（早期云端训练记录）**：1.7B / 0.6B full SFT 的云端训练配置与踩坑记录；线上 `jaykwok/Qwen3-ASR-*-JA-Anime-Galgame-hf` 即该自训链路的发布产物，自训工具 `tools/asr/qwen/` 仍活跃保留。

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


---

## 旧实验记录（2026-06-18 SpeechBoundary-JA v3 训练流水账，已被 v8 取代）

- 2026-06-18 SpeechBoundary-JA first scorer 路线断兼容重置：`neg20-s800` NAMH-055 对比显示 speech-only scorer 会把长 speech island 交给约 3 秒机械切分，开头“中出し”被拆成多块并诱发 ASR 误识别，因此旧 `speech_boundary_ja_mamba2_frame_scorer_v2`、`mamba2_frame_scorer` runtime model type、单通道输出和 hard-negative-first 训练路线全部废弃。active code 升级为 `speech_boundary_ja_mamba2_frame_boundary_scorer_v3`：checkpoint 必须是 schema v3、model type `mamba2_frame_boundary_scorer`、`output_dim=2`，训练/评估张量为 `[speech, cut]` 双通道，loader 对旧 schema/model type/缺失或错误 `output_dim` fail-fast。`tools.boundary.ja.train_feature_scorer` 和 `evaluate_feature_scorer_thresholds` 改为 v3 speech/cut 双指标；`tools.boundary.prepare_speech_boundary_hard_negative_finetune` 重命名为 `prepare_speech_boundary_hard_negative_replay`，不再生成 scorer 训练脚本。当前下一步是用 synthetic true-structure timeline 重建 first-scorer bootstrap 数据，再缓存 Qwen PTM + MFCC features 并训练 v3；CueQC `drop_ok` hard negatives 只保留为后续 replay material。
- 2026-06-18 SpeechBoundary-JA frame boundary scorer v3 synthetic8192 训练完成：先提交断兼容重构 `f357a1e Refactor SpeechBoundary scorer v3`，随后按 `anime_nsfw=55/anime_sfw=20/galgame=25` 抽样 30000 条 source manifest 到 `agents/temp/20260618_144022_speech-boundary-v3-mixed-source-manifest/`，生成 synthetic true-structure timeline `agents/temp/20260618_144022_speech-boundary-frame-boundary-v3-synthetic8192/`。数据为 `8192` records、总时长 `133615.697s`、`speech_frame_ratio=0.819862`、`cut_point_segment_count=8114`、`cut_drop_zone_count=7842`、`overlap_mix_count=1230`、`skipped=0`。Qwen PTM + MFCC full-dim feature cache 输出到 `agents/temp/20260618_144022_speech-boundary-frame-boundary-scorer-v3-prep/feature-cache/`，`cached=8192/errors=0/skipped=0`，`ptm_dim=1024/mfcc_dim=40`。三版训练对比：旧保守权重 `positive=1/negative=15/max_steps=1000` 的 sweep speech `threshold=0.25 recall=0.988853/FPR=0.038906/F1=0.990139`，cut `threshold=0.5 recall=0.923106/FPR=0.023875/F1=0.869518`；探索版 `positive=2/negative=10/cut_positive=6/max_steps=3000` 的 speech `threshold=0.3 recall=0.990402/FPR=0.025671/F1=0.992366`，cut `threshold=0.7 recall=0.940312/FPR=0.010310/F1=0.927905`；按 Grok 搜索结论调整的 recall-first 版 `positive=4/negative=1/cut_positive=8/max_steps=3000` 训练期 speech recall 提升到 `0.996610`，sweep speech `threshold=0.6 recall=0.992281/FPR=0.038477/F1=0.991917`，cut `threshold=0.7 recall=0.947447/FPR=0.014012/F1=0.917663`。Grok 结论是：BCE/focal 的权重起点可参考 `negative/positive` class ratio，但业务代价会覆盖频率平衡；`pos_weight > 1` 偏 recall，`< 1` 偏 precision，最终 threshold 必须用验证集 sweep 校准而不是固定 `0.5`。当前三版均未过旧 speech policy `recall>=0.995/FPR<=0.02`，cut head 已能过；下一步不要替换默认 runtime，应重新校准双头 scorer 的 speech policy，或调整 synthetic negative/gap 分布与 loss，而不是把 0.5 当默认阈值。
- 2026-06-18 SpeechBoundary-JA v3 threshold 策略落地：Grok 复核 SpeechBrain / neural VAD 常见做法后，决定不把单点 `0.5` 当运行时默认，而是给 speech head 增加 activation/deactivation hysteresis：`SPEECH_BOUNDARY_JA_SPEECH_ON_THRESHOLD` 负责进入 speech，`SPEECH_BOUNDARY_JA_SPEECH_OFF_THRESHOLD` 负责退出 speech，cut head 继续用 `SPEECH_BOUNDARY_JA_CUT_THRESHOLD` 做 gate。代码默认兼容旧运行效果：未显式设置 on/off 时两者都等于旧 `SPEECH_BOUNDARY_JA_THRESHOLD`；v3 Grok candidate 的下一次 opt-in smoke 建议使用 `speech_on=0.7 / speech_off=0.5 / cut=0.7`，但默认 runtime 仍不替换。boundary cache signature、`tools.workflows.run_full_workflow` CLI/env/summary 和测试已同步。CueQC 模型不可用路径也同步清理为纯 `fallback_keep`，不再用文本长度、重复度等启发式给 fallback confidence，避免误判旧规则 QC 仍在运行。
- 2026-06-18 SpeechBoundary-JA v3 runtime-profile 校准结论：force-aligner runtime 退役已提交为 `56f6890 Retire forced aligner workflow`。继续 Mamba 优化时，`evaluate_feature_scorer_thresholds` 增加 runtime profile 评估，并把 hysteresis 评估改为区间扫描、Mamba 推理改为 batch + 长度分桶；`prepare_frame_boundary_scorer_v3` 也参数化 speech/cut loss weights、focal gamma、eval batch size 和 runtime profiles，避免靠手改临时 ps1。已有全量 12-profile 评估 `agents/temp/20260618_200421_speech-boundary-v3-runtime-profile-eval/` 仍无 profile 通过旧 policy，最佳 `on=0.7/off=0.5/cut=0.8` 为 `recall=0.981691/FPR=0.027299/F1=0.987771`；补充 1024-row 宽扫 `agents/temp/20260618_204217_speech-boundary-v3-runtime-profile-wide-eval-limit1024-bucketed/` 中，即使近似禁用 cut gate 的 `on=0.65/off=0.6/cut=1.0` 也只有 `recall=0.991086/FPR=0.031271`。结论：当前 scorer 不能靠阈值/门控直接 promoted；根因更像 bootstrap 数据分布，`20260618_144022` synthetic8192 的 `speech_frame_ratio=0.819862` 且 `negative_rows/background_rows=0`，非语音帧太少且过于合成。下一步改为 negative-rich synthetic true-structure 重建：降低 speech ratio、加长 leading/trailing/internal gap、降低 speech label dilation，先不把 CueQC hard negatives 纳入 first scorer bootstrap，待新 scorer 产出后再用人工审计的 drop replay 做后续 finetune。
- 2026-06-18 SpeechBoundary-JA v3 negative-rich 训练复盘：`agents/temp/20260618_205057_speech-boundary-frame-boundary-v3-negative-rich-synthetic4096/` 将 speech frame ratio 降到 `0.550969`，但 `pos2/neg1/cut6/s3000`、`pos4/neg1/cut6/s3000` 和 `pos2/neg1/cut6/s6000` 仍没有 checkpoint 通过旧 frame policy；`pos2/s6000` 最接近低 FPR，`threshold=0.6` 为 `recall=0.986210/FPR=0.007259`。随后 `agents/temp/20260618_221545_speech-boundary-frame-boundary-v3-negative-rich-synthetic8192-dil06/` 使用 `8192` rows、`speech_frame_ratio=0.587592`、`cut_drop_zone_count=10708`、`cut_point_segment_count=5676`，输出 checkpoint `agents/temp/20260618_221851_speech-boundary-frame-boundary-scorer-v3-negative-rich8192-dil06-prep/frame-boundary-scorer-v3/speech_boundary_ja_feature_scorer.pt`；全量 threshold eval 仍无 speech pass：`threshold=0.4` 为 `recall=0.995932/FPR=0.037859`，`0.5` 为 `0.992537/0.020272`，`0.6` 为 `0.987190/0.009828`，runtime profile 最佳 `on=0.6/off=0.55/cut=1.0` 为 `recall=0.988960/FPR=0.012228`。结论：negative-rich 数据修复了 FPR 侧，但 strict frame recall/FPR tradeoff 仍卡住；默认 scorer 继续不替换。
- 2026-06-18 SpeechBoundary-JA v3 error diagnostics：`evaluate_feature_scorer_thresholds` 新增 `--diagnostic-threshold` / `--diagnostic-runtime-profile`，输出 `speech_error_diagnostics.jsonl` 和 Markdown 的 boundary-distance / region / island-level / top-error rows。限量 1024-row 诊断 `agents/temp/20260618_230723_speech-boundary-v3-error-diagnostic-limit1024/` 显示错误高度集中在边界邻近帧：`threshold=0.5` 总体 `recall=0.992411/FPR=0.020234`，但 far-from-boundary region 已达 `recall=0.996881/FPR=0.004334`，near-boundary region 只有 `recall=0.915944/FPR=0.206994`，FP/FN 的 near-boundary share 分别为 `0.802624/0.611695`；runtime `on=0.6/off=0.55/cut=1.0` 也类似，far `0.995505/0.002182`、near `0.874780/0.130147`。补充 128-row island smoke `agents/temp/20260618_232007_speech-boundary-v3-island-diagnostic-limit128/` 显示 `threshold=0.5` 的 `speech_island_recall=1.0`、`mean_label_island_coverage=0.986958`，但 `cut_drop_zone_clean_rate=0.574074`、`mean_cut_drop_zone_predicted_ratio=0.162603`，说明核心 speech island 保留已经可观，gap/drop 不误连仍是主要 gate。当前判断：旧 `recall>=0.995/FPR<=0.02` 的逐帧 policy 对边界 120ms 内过于敏感，已经不适合作为 first scorer 唯一 gate。下一步应把 boundary-aware / segment-island-level eval 变成正式推荐口径，用核心 speech 保留、gap/drop 不误连和真实 workflow 人工观感作为主 gate；在新 gate 通过前不继续盲调 loss，也不替换默认 runtime。
- 2026-06-18 SpeechBoundary-JA v3 Grok hysteresis NAMH-055 opt-in smoke 已完成：首次运行在进入模型前暴露 Windows run-log 文件名过长问题，`task + label + backend` 全量拼接导致 `logging.FileHandler` 不能打开路径；已将 run-log 文件名组件改为保留前缀 + sha1 短 hash，避免长 label/backend 触发 MAX_PATH。随后使用 `SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT=agents/temp/20260618_144022_speech-boundary-frame-boundary-scorer-v3-prep/frame-boundary-scorer-v3-grok-pos4-neg1-cut8-s3000/speech_boundary_ja_feature_scorer.pt`、`SPEECH_BOUNDARY_JA_SCORER_DEVICE=cuda`、`--speech-boundary-speech-on-threshold 0.7 --speech-boundary-speech-off-threshold 0.5 --speech-boundary-cut-threshold 0.7 --no-boundary-cache --keep-asr-chunks` 跑通 NAMH-055 no-translate full workflow，输出 `agents/temp/speech-boundary-ja/20260618_161906_speech-boundary-v3-grok-hys-namh055-smoke/`。runtime signature 确认 `score_model=mamba2_frame_boundary_scorer`、`speech_threshold_mode=hysteresis`、on/off/cut 为 `0.7/0.5/0.7`，scorer sha256 `03ea7e646c3ed857a84260c45e1be2ac254e14a1fa1ff84387ee9097b2375382`。结果 `transcript_chunks=1486/segments=1751/blocks=1746`，ASR+Alignment `430.1s`，pipeline total `502.8s`，CueQC `candidates=1842/drop=356/keep=1486/fallback=0`。质量报告仍提示候选风险：`per_min_subtitle_count=19.41`、`alignment_fallback_ratio=0.378`、`subtitle_overlap_count=3`、`long_segment_count=53`；这说明 workflow 和新阈值 plumbing 已通，但该 scorer/阈值组合仍需人工 A/B 审计，默认 runtime 继续不替换。
- 2026-06-18 SpeechBoundary-JA v3 Grok hysteresis 音频 A/B 审计页已生成：按“只看音频切分和字幕文本、不展示 NSFW 视频画面”的要求，使用同一抽取 WAV `agents/temp/speech-boundary-ja/20260618_161906_speech-boundary-v3-grok-hys-namh055-smoke/jobs/NAMH-055_sbv3-grok-hys/audio/NAMH-055.2bb8f0e7.wav`，base 为 `agents/temp/speech-boundary-ja/20260617_cueqc-v3-namh055-smoke2/archived/NAMH-055/NAMH-055.ja.srt`，candidate 为 `agents/temp/speech-boundary-ja/20260618_161906_speech-boundary-v3-grok-hys-namh055-smoke/archived/NAMH-055/NAMH-055.ja.srt`，输出 `agents/audits/20260618_163540_speech-boundary-v3-grok-hysteresis-namh055-audio-ab/index.html`。页面为 `<audio>` 播放器，上方显示 base 当前 cue，下方显示 candidate 当前 cue，并按 30s 窗口列出 `179` 个切分/文本差异窗口；base/new cue 数为 `2259 -> 1746`，用于人工判断是否存在漏切、过合并或 ASR 文本退化。
- 2026-06-18 音频 A/B 页面字幕同步修复：用户反馈音频页 base/candidate 都有延迟后，先用 ffprobe 确认 MP4 与抽取 WAV 均为 0 起点，并用原视频无滤镜 180s WAV 与 workflow 处理后 WAV 做包络互相关，最佳 offset 为 `0.0s`，排除整体音频抽取偏移。根因收口到页面只依赖 `<audio>` 的 `timeupdate` 事件，浏览器对音频元素触发频率较低，字幕显示会滞后。`tools.audits.generate_subtitle_ab_compare_audit_html` 已改为播放中用 `requestAnimationFrame` 连续刷新，seek 时按目标时间立即渲染；已重建 `agents/audits/20260618_163540_speech-boundary-v3-grok-hysteresis-namh055-audio-ab/index.html`，新增回归测试 `tests/test_subtitle_ab_compare_audit.py`。
- 2026-06-18 A/B 审计页改为双独立播放器：用户反馈同时叠加听/看会干扰字幕观感审计后，`tools.audits.generate_subtitle_ab_compare_audit_html` 的 audio/video 模式均改为上下两个独立播放器，播放任意一边会自动暂停另一边；差异窗口和上一处/下一处只把两边定位到同一时间，不自动播放。当前音频审计页已重建为 `oldPlayer/newPlayer` 双 `<audio>`，仍不包含 `<video>`。同时确认本轮“怪”的一部分来自最终 SRT 而非页面：candidate 首条 final SRT cue 为 `10.810-11.063`，仅 `0.253s`，而其 ASR transcript chunk 为 `9.2906875-11.13`；这说明 forced aligner / subtitle postprocess 会把 boundary chunk 再拆成很短显示 cue。用户反馈第一条确实奇怪但后面整体还可以，所以下一步先用双独立播放器继续人工审计，不急着改 workflow 默认。
- 2026-06-18 forced aligner 去留 A/B plumbing 修正：复核 `qwen_asr` 包源码后确认 `return_time_stamps=True` 并不是 ASR 原生词时间戳，而是由 `Qwen3ForcedAligner` 产生，且官方 wrapper 要求初始化时带 forced aligner；因此不能用它代表“native/no-aligner”。`ALIGNMENT_TIMESTAMP_MODE=native` 已改为真正不加载 forced aligner：ASR 只做 text-only，字幕时间轴使用 Boundary chunk / speech_core fallback window 等比分配；`hybrid` 定义为 native timestamp 可用时优先 native、不可用且 chunk/text 满足 `ALIGNMENT_HYBRID_FORCE_MAX_CHUNK` / `ALIGNMENT_HYBRID_FORCE_MIN_TEXT_LEN` 时再 forced，否则 fallback。`tools.workflows.run_full_workflow` 新增 `--alignment-timestamp-mode forced|native|hybrid` 并写入 env/context/summary；native/quality/CLI 回归测试已补。无效实验 `agents/temp/speech-boundary-ja/20260618_173908_speech-boundary-v3-grok-hys-namh055-native-real/` 是误用 `return_time_stamps=True` 产生的 0 字幕结果，后续不要引用。有效 no-aligner NAMH-055 opt-in 结果为 `agents/temp/speech-boundary-ja/20260618_174707_speech-boundary-v3-grok-hys-namh055-no-aligner/`：同一 v3 scorer + `speech_on/off/cut=0.7/0.5/0.7`，`transcript_chunks=1486/segments=1468/blocks=1466`，Alignment 对齐 `0.06s`，pipeline total `429.45s`，quality report `alignment_fallback_ratio=0.987` 符合 no-aligner fallback 预期。首条 cue 对照显示 forced 为 `10.810-11.063` / `0.253s`，no-aligner 为 `9.290-11.063` / `1.773s`，首条异常根因进一步收口为 forced aligner 晚起点。音频双播放器 A/B 页面已生成：`agents/audits/20260618_175534_speech-boundary-v3-forced-vs-no-aligner-namh055-audio-ab/index.html`，forced/no-aligner cue 数 `1746 -> 1466`，差异窗口 `159`。
- 2026-06-18 forced-success delta 复核：基于同一 forced run，排除首条异常后统计 `905` 个 forced 成功 chunk；按 aligned segment span 相对原始 Boundary chunk 计算，`delta_start = forced_start - chunk_start` 的 mean/p50/p90/p95 为 `+0.440/+0.160/+1.280/+1.600s`，`delta_end = forced_end - chunk_end` 为 `-0.871/-0.640/-0.068/-0.020s`，整体 duration delta p50 `-1.140s`。按最终 subtitle block 统计，postprocess 会回填一部分 end，但仍为 mean/p50 `-0.481/-0.260s`；说明除第一条外，forced aligner 成功样本也系统性把 Boundary chunk 向内收，尤其 end 侧提前明显，这会减少尾部停留但也可能截掉非词尾音或造成短 cue。
- 2026-06-18 forced delta 审计页已生成：新增 `tools.audits.generate_forced_delta_audit_html`，默认按 `max(abs(delta_start), abs(delta_end))` 从大到小排序，页面内可分别播放 Boundary chunk、forced span 和最终字幕 span。NAMH-055 页面为 `agents/audits/20260618_181144_forced-delta-namh055-audit/index.html`，数据 `items.json` 共 `905` 条，已排除首条异常 chunk。
- 2026-06-18 forced delta 视频审计页已生成：同一生成器扩展为 `--media-mode audio|video`，保持 `--audio` 旧命令可用；视频版使用原片 `video/NAMH-055.mp4`，页面为 `agents/audits/20260618_182509_forced-delta-namh055-video-audit/index.html`，同样 `905` 条 forced 成功样本并按最大绝对 delta 降序。
- 2026-06-18 forced delta 视频字幕 overlay 审计页已生成：视频模式新增画面内字幕 overlay，`播放 Boundary` 会按 Boundary chunk 时间轴显示当前字幕，`播放 Forced` 会按 forced aligner span 显示同一字幕，`播放最终字幕` 保留最终 block 时间轴用于三方对照。新版页面为 `agents/audits/20260618_182859_forced-delta-namh055-video-caption-audit/index.html`。
- 2026-06-18 forced delta 音频字幕 overlay 审计页已生成：音频模式同样新增播放器下方字幕窗口，`播放 Boundary` / `播放 Forced` / `播放最终字幕` 分别按对应时间轴显示同一字幕，便于只听音频时比较起止点观感。新版页面为 `agents/audits/20260618_183136_forced-delta-namh055-audio-caption-audit/index.html`。
- 历史记录 2026-06-18 项目手工阈值 / 启发式盘点：当时 active runtime 仍有四类需要区分处理的阈值。第一类是 SpeechBoundary-JA first scorer 相关：bootstrap scorer 仍是 `_range_normalize` 的 20/95 percentile、`energy/ptm/mfcc_delta=0.70/0.20/0.10` 加权、`frame_dilation_s=0.2`、`min_segment_s=0.05`、`max_group_s=6.0`、`chunk_threshold_s=1.0` 和 cut gate；这些是优先被 v3 learned scorer + calibration profile 取代的对象。第二类是 Boundary candidate/planner 约束：`cut_score_threshold=0.94`、`valley_score_threshold=0.10`、`target_chunk_s=3.0`、`max_core_chunk_s=5.0`、`min_chunk_s=0.4`、`max_splits_per_segment=16`，其中 candidate 阈值属于可学习/可校准候选，planner 时长上限更像字幕产品约束，不应简单删除。第三类是 CueQC：runtime 是 learned v3-Fusion，但决策仍有 checkpoint `drop_threshold=0.85` 与 `short_text=0.87` 的保守 profile；后续应把它变成由审计集生成的 calibration artifact，而不是继续手写常量。第四类是 ASR/aligner/subtitle 后处理：alignment sentinel、retry/refine 分段、fallback 时间戳、word grouping、subtitle density/QC 阈值仍有启发式；其中大部分是诊断或展示安全约束，不等同于 drop/keep 或 boundary 模型替代目标。后续清理优先级：先做 SpeechBoundary v3 calibration artifact，再做 CueQC threshold calibration artifact，最后单独审计 aligner/subtitle fallback 规则。
- 2026-06-18 CueQC -> SpeechBoundary-JA hard-negative-only 路由重置：用户确认人工 `drop_ok` 都是 hard negative，直接 drop，不再按“对话/呻吟/呼吸/环境/短碎片”等分类标签分流。`tools.boundary.export_cueqc_drop_hardcases` 已断兼容清理 CueQC drop -> Boundary Refiner 训练标签路线，`drop_ok` 统一导出为 `speech_boundary_frame_negative_candidate`；`tools.boundary.prepare_cueqc_drop_hard_negative_sources` 只生成 SpeechBoundary-JA negative labels，并会 fail-fast 拒绝非 frame-negative manifest。已基于三轮 false-drop 审计导出 hardnegative-only 数据：原始标签 `600` 行、unique item `578`、duplicate extra rows `22`、confirmed drop candidates `551`、safety holdout `27`，routes 为 `speech_boundary_frame_negative_candidate=551`；source prep 已 materialize `551/551` 个 frame-negative，`missing_audit_item_samples=0`、`frame_negative_skipped=0`，生成 `speech_boundary_negative_labels.jsonl`、`speech_boundary_negative_manifest.json`、`speech_boundary_training_manifest.jsonl`，训练可读 examples `551`。旧 20260617 CueQC drop hard-case 临时产物只保留为历史产物，不再作为后续训练输入。
- 2026-06-18 SpeechBoundary-JA hard-negative mixed feature cache 与 tiny plumbing 训练已完成但被新标签策略 supersede：早先基于 `agents/temp/20260617_230948_speech-boundary-hard-negative-finetune-prep/` 的 2022 条 mixed labels（522 negative + 1500 anchor），执行 `build_mixed_feature_cache.ps1` 成功生成 Qwen/MFCC feature cache，summary 为 `feature-cache-mixed-hard-negative-anchor/feature_summary.json`，结果 `cached=2022`、`skipped=0`、`errors=0`，label qualities 为 `negative=522/supervised=1500`。随后执行 `tiny_mixed_plumbing_train.ps1`，输出 `tiny-mixed-hard-negative-anchor/speech_boundary_ja_tiny.pt` 与 `train_metrics.json`，`steps=200`、`loss=0.607988`、`frame_accuracy=0.73`、`positive_ratio=0.73`。该 checkpoint 是 research/plumbing artifact，不接入 runtime，也不替换 `qwen-feature-energy-bootstrap-v1`；由于 `drop_ok` 路由已改为 551 全部 hard negative，该旧 feature cache / tiny train 不再作为下一轮正式 scorer 输入。
- 2026-06-18 SpeechBoundary-JA MLP feature scorer v1 失败结论：旧 runtime-loadable MLP 候选虽然能加载并保留多数正样本，但在 `recall>=0.995` 且 `FPR<=0.02` policy 下无 threshold 通过；最佳 F1 threshold `0.5` 为 `recall=0.998066/FPR=0.086026/F1=0.997301`，threshold `0.8` 虽满足 `FPR=0.019377` 但 recall 仅 `0.989536`。该路线已断兼容移除，旧 checkpoint schema 不再被 active loader 接受，避免后续误把 MLP v1 当作可 promoted 候选。
- 2026-06-18 SpeechBoundary-JA frame scorer v2 路线确定并开始落地：只保留 `speech_boundary_ja_mamba2_frame_scorer_v2`，复用现有 `BoundarySequenceClassifier` / `transformers.Mamba2Model`，输入仍为缓存的 Qwen PTM + MFCC features；默认 runtime 仍是 `qwen-feature-energy-bootstrap-v1`，只有显式设置 `SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT` 才加载 v2 scorer。训练入口固定为 Mamba2，不提供 MLP/`--architecture`/dropout 兼容分支；loss 使用 frame weights + class weights + optional focal，默认 `negative_weight=15.0`、`focal_gamma=2.0`。该 scorer 只替代 SpeechBoundary-JA bootstrap frame scores，不替代 Boundary Refiner v5；`src/boundary/checkpoints/boundary_refiner.pt` 不变。
- 2026-06-18 SpeechBoundary-JA Mamba2 frame scorer v2 首个候选已训练并通过离线 threshold policy，但现已被新 hardnegative-only 数据策略 supersede：基于旧 `agents/temp/20260617_230948_speech-boundary-hard-negative-finetune-prep/` 的 2022 条 mixed labels 和 feature cache，CUDA 训练 200 steps 输出到 `agents/temp/20260618_120005_speech-boundary-mamba2-frame-scorer-v2/speech_boundary_ja_feature_scorer.pt`，sha256 `159de61095e06e8919e9d0d16d4295e45c52a32b3be5c46ed16718e34401c5f4`。训练配置为 hidden/layers/state/heads/groups/chunk `128/2/32/4/2/8`、`negative_weight=15.0`、`focal_gamma=2.0`，held-out metrics：`eval_loss=0.009267`、`precision=0.998873`、`recall=0.995676`、`f1=0.997272`。全量 threshold sweep 输出 `agents/temp/20260618_120005_speech-boundary-mamba2-frame-scorer-v2-threshold-eval/`，policy `recall>=0.995/FPR<=0.02` 下 `threshold=0.5` 通过：`recall=0.998328`、`FPR=0.018753`、`precision=0.999243`、`F1=0.998785`。默认 runtime 未替换；下一轮可用候选必须基于 `agents/temp/20260618_124129_speech-boundary-hard-negative-finetune-prep-hardnegative-only-mixed/` 重新 build feature cache 并重训。
- 2026-06-18 SpeechBoundary-JA Mamba2 frame scorer v2 NAMH-055 opt-in smoke 已完成：使用 `SPEECH_BOUNDARY_JA_SCORER_CHECKPOINT=agents/temp/20260618_120005_speech-boundary-mamba2-frame-scorer-v2/speech_boundary_ja_feature_scorer.pt`、`SPEECH_BOUNDARY_JA_SCORER_DEVICE=cuda`、no-translate、`--keep-asr-chunks`、`--no-boundary-cache` 跑 `video/NAMH-055.mp4`，输出 `agents/temp/speech-boundary-ja/20260618_121253_speech-boundary-mamba2-v2-namh055-smoke/`。runtime signature 确认 `score_model=mamba2_frame_scorer`，scorer schema `speech_boundary_ja_mamba2_frame_scorer_v2`，sha256 `159de61095e06e8919e9d0d16d4295e45c52a32b3be5c46ed16718e34401c5f4`；结果 `transcript_chunks=1065/segments=1476/blocks=1476`，ASR+Alignment `361.5s`，pipeline total `427.0s`。对比默认 baseline `agents/temp/speech-boundary-ja/20260617_cueqc-v3-namh055-smoke2/`：baseline `chunks=3199/segments=2444/blocks=2259`；alignment diagnostics compare 输出 `agents/temp/20260618_122344_mamba2-v2-namh055-diagnostics-compare/`，显示 baseline fallback `863` -> candidate fallback `421`、nonlexical bucket `842` -> `12`，但 candidate fallback ratio 按 chunk 为 `39.5%`，并且 chunk 数大幅减少，存在过度合并/漏切风险。人工对比审计页已生成 `agents/audits/20260618_122344_speech-boundary-mamba2-v2-namh055-compare/index.html`，`review_item_count=863`，outcome breakdown 为 `resolved=463/still_fallback=299/mixed=100/changed=1`；默认 runtime 仍未替换，必须等人工审计后再决定是否继续扩大 held-out。
- 2026-06-18 SpeechBoundary-JA hard-negative finetune readiness gate 已按 hardnegative-only 规则重跑：复用 `tools.boundary.ja.build_positive_anchor_replay` 在 `agents/temp/20260617_230636_speech-boundary-positive-anchor-replay/` 生成的 1500 条 positive anchors（`anime_nsfw=825/anime_sfw=300/galgame=375`），与新 551 条 negative source 混合到 `agents/temp/20260618_124129_speech-boundary-hard-negative-finetune-prep-hardnegative-only-mixed/`。该包生成 `speech_boundary_mixed_hard_negative_anchor_labels.jsonl` / manifest，`records=2051`、`trainable_examples=2051`、`negative_share=0.268649`，五项 gate 全部通过，并写出 `build_mixed_feature_cache.ps1`、`train_mixed_feature_scorer.ps1`、`tiny_mixed_plumbing_train.ps1`。该条记录为训练前 gate；后续已用新断兼容目录重训 v2 scorer。SpeechBoundary-JA 默认 runtime 仍是 `qwen-feature-energy-bootstrap-v1`。
- 2026-06-18 SpeechBoundary-JA Mamba2 frame scorer v2 hardnegative-only 重训：按新断兼容命名重新生成候选、source 和 mixed prep 到 `agents/temp/20260618_130257_*`，仍为 551 条 CueQC `drop_ok` hard negatives + 1500 条 positive anchors，mixed feature cache `cached=2051/errors=0/skipped=0`。第一版沿用 `negative_weight=15.0/max_steps=200` 未过离线 policy：推荐 `threshold=0.25` 时 `recall=0.996908` 但 `FPR=0.040462`。随后重训 `negative_weight=20.0/max_steps=800`，输出 `agents/temp/20260618_130257_speech-boundary-mamba2-frame-scorer-v2-neg20-s800/speech_boundary_ja_feature_scorer.pt`，sha256 `9d9f78f9cfeb1c2b0c6dc6d5764f10601786a44b0666ba3ff181dd394b4a6f58`；离线 threshold eval `agents/temp/20260618_130257_speech-boundary-mamba2-frame-scorer-v2-neg20-s800-threshold-eval/` 通过 policy，选中 `threshold=0.8`：`recall=0.995668`、`FPR=0.019718`、`precision=0.999111`、`F1=0.997386`。NAMH-055 opt-in no-translate smoke 使用该 checkpoint + `--speech-boundary-threshold 0.8` + `--no-boundary-cache`，输出 `agents/temp/speech-boundary-ja/20260618_130257_speech-boundary-mamba2-v2-neg20-s800-namh055-smoke/`，runtime signature 确认 `score_model=mamba2_frame_scorer`、scorer sha256 同上；结果 `transcript_chunks=1801/segments=1952/blocks=1947`，ASR+Alignment `468.9s`，pipeline total `546.6s`，quality report `alignment_fallback=622/1801=0.345`、`per_min_subtitle_count=21.65`、`short_segment_ratio=0.100`、`long_segment_count=4`、`overlap_count=5`。相比旧 522-negative v2 smoke 的 `1065` chunks，新候选明显更保守；默认 runtime 仍未替换，下一步应生成与默认 baseline 的人工对比审计页，重点看漏切/过合并和非语言片段丢弃是否符合预期。
- 2026-06-18 旧边界偏好路线归档：120 条盲测和 77 条 compiled label 只作为历史实验结论保留，不再作为 active 微调入口。CueQC 已确认 `display=drop` / `drop_ok` 全部作为 SpeechBoundary-JA frame-level hard negative；过碎短切片和无效 speech island 的目标是直接 drop，而不是并入相邻字幕。旧准备、生成、编译、汇总工具和对应测试已移动到 `agents/rm/20260618_124129_*` 归档目录，active tree 不再提供这条训练工具链。
- 2026-06-17 Web 分发链路 NAMH-055 不翻译 smoke 已完成，并暴露/修复 CueQC runtime 诊断盲区：早先一次 Web smoke 曾出现 CueQC 推理异常后全量 fallback keep，但旧日志只显示保守 keep，容易误判模型已生效。根因是 runtime refiner 把候选放进单个大批次推理，批次内异常会导致整批 fallback，且 fallback detail 不进入 pipeline summary。修复后 CueQC v3-Fusion runtime 按 `CUEQC_INFERENCE_BATCH_SIZE` 分批推理，capture/feature/inference fallback 会写入 `fallback_stage/fallback_detail`，pipeline 日志和 timings 报告 `fallback_summary`。重启 Web 后 job `38a5d14ea1a54236a24b56e716f36175` 完成：CueQC 完整决策 `3199` 条、`drop=1052`、`keep=2147`、`fallback=0`，最终 `transcript_chunks=2147`、`segments=2157`、`blocks=2087`，输出 summary 在 `agents/temp/20260617_191654_web-smoke-namh055-cueqc-batched/job_summary.json`。同时把可复用 Web smoke 脚本整理到 `tools/web/smoke/`，把大规模 CueQC feature 分片提取从一次性 `agents/temp/.../run_extract_shards.ps1` 抽象为 `tools/asr/cueqc/extract_feature_shards.py`，README 底部新增工具索引，后续临时硬编码脚本只保留为历史运行记录，不作为复用入口。
- 2026-06-17 CueQC v3-Fusion Stage 2b 已通过 0.6B/旧单文件链路 false-drop gate 并替换当时默认 checkpoint：当时默认 `src/asr/checkpoints/cueqc_mamba_v3_fusion.pt` 来自 Stage 2b 自训练，sha1 `98f9631a63dc19736b50619100fb4be4d08075e8`，旧默认备份在 `agents/temp/20260617_174154_cueqc-default-checkpoint-backup/cueqc_mamba_v3_fusion.before-stage2b-adaptive.pt`；后续已迁移到 repo-id tagged 的 0.6B/1.7B checkpoint registry，此条保留 Stage2b provenance。三轮 false-drop 审计合计 `600` 条，结果 `drop_ok=573 / false_drop_keep=25 / uncertain=2`；第三轮 Stage 2b 审计为 `198 drop_ok / 1 false_drop_keep / 1 uncertain`，裸 false-drop rate `0.5%`。新增 `src/asr/cueqc_thresholds.py` 作为 runtime 与 offline prediction 共用的阈值校准层：基础 `drop_threshold=0.85`，`text_bucket=short_text` 提升到 `0.87`，profile 只允许抬高阈值，不允许低于 base threshold。该 profile 回放三轮审计时保留全部 `25` 个 `false_drop_keep`，同时仍 drop `540/573` 个人工确认 `drop_ok`；10-film adaptive prediction `agents/temp/20260617_174344_cueqc-v3-stage2b-adaptive-10film-predictions/` 为 records `45643`、`drop=19380/keep=26263`、高置信 pseudo `drop=19380/keep=1372`。
- 2026-06-17 CueQC Stage 2 自训练闭环已完成一轮并收敛到保守默认：10-film 全量特征在 `agents/temp/20260617_113159_cueqc-v3-10film-sharded-features/` 按 46 个 shard 提取并合并为约 `3.02GB` 的 `cueqc_full_features_v3_fusion.pt`，共 `45643` 条未标注候选。初始 prediction `keep=31055/drop=14588` 后，第一轮人工审计 `178 drop_ok / 21 false_drop_keep / 1 uncertain`，因此未接收未审 drop pseudo；Stage 2a 训练包 `538` 条仅纳入 cold-start、人工确认 drop、人工纠正 keep 和高置信 keep。t=0.88 第二轮审计降到 `197 drop_ok / 3 false_drop_keep`，但阈值-only 到 `0.92` 只剩 `337` drop，收益太低；Stage 2b 训练包 `2177` 条，labels `drop=532/keep=1645`，固定 holdout `867HTTM-0045` 达 `keep_recall=1.0/false_drop_rate=0.0/drop_precision=1.0/drop_recall=0.8475`。结论：Stage 2 当前以 keep recall / false-drop 安全为优先，未审高置信 drop 不直接进入训练。
- 2026-06-17 CueQC v3-Fusion runtime 已替代旧规则 ASR QC：用户确认“旧 QC 决策直接删除，不观测不运行”。active tree 已移除 `src/asr/qc.py`、`src/asr/qc_stage.py` 和旧 ASR QC 专属测试；pipeline 不再调用 `_run_TRANSCRIPTION_qc` / `collect_adaptive_precision_review`，quality report、alignment quality、旧边界偏好实验、silver mining 和审计页不再读取或展示 `asr_qc_*` / `asr_review_uncertain`。CueQC 输出只保留 `display_hint=keep/drop`；模型不可用或异常时 fallback keep。`tools/asr/cueqc/compile_training_set.py` 只接受簇级 `cueqc_cluster_labels.jsonl`，per-sample `cueqc_manual_labels.jsonl` 和 `content_type/qc_decision/alignment_policy` 旧标签头不再兼容；旧 `generate_asr_attribution_audit_html.py` 已移到 `agents/rm/20260617_101851_retired-asr-attribution-qc-audit/`。
- 2026-06-16/17 CueQC v3-Fusion 断兼容重构与 cold-start 完成：v3 从 v2 multimodal 切到 `ASR encoder features + ASR token trace + decoder aggregate stats + structured metadata -> display keep/drop`，删除 BGE-m3、sentence-transformers、HuBERT、UMAP/HDBSCAN/FINCH/PCA 和边界 frame 作为默认 CueQC 输入；runtime 复用已加载的 Qwen3-ASR backend，不加载第二份 ASR。关键实现包括 `src/asr/asr_internals.py`、`src/asr/cueqc_features.py`、`src/asr/cueqc_model.py`、`src/asr/cueqc_refiner.py`、`tools/asr/cueqc/extract_features_v3_fusion.py`、`train_mamba_v3_fusion.py`、`predict_v3_fusion.py` 和 `compile_stage2a_features_v3_fusion.py`。实施期修复两个关键坑：teacher-forced logits 必须走 `wrapper.model.thinker.forward(...)`，`get_audio_features` 在 batch=1 返回 `[T,D]` 时需补 batch 维。300 条 cold-start 最终 labels `keep=133/drop=167`，首轮过拟合模型已废弃；保守配置加 label smoothing、early-stop 和 keep 权重后，内部 holdout `867HTTM-0045` 达 `keep_recall=1.0/false_drop_rate=0.0`，作为 Stage 2 起点。
- 2026-06-15/16 CueQC bootstrap 聚类已完成并退役为一次性种子步骤：NAMH-055 曾误入 11-film 审计池，复核后固定为 smoke/holdout，实际训练池来自 CueQC 合入前 baseline commit `5afe535` 的 10 部全片、`skip_translation=True`、保留 ASR chunks，候选池 `agents/temp/20260615_152934_cueqc-10film-candidates/cueqc_candidates.full.jsonl` 共 `45643` 条，并分层抽样 `300` 条。旧 HDBSCAN/FINCH/UMAP/PCA、embedding 增强和 17 簇 taxonomy 路线均已被废弃；最终只保留 Torque Clustering 作为一次性 coarse seed 工具，`--merge-layer 1` 得到 7 簇并生成审计页 `agents/audits/20260616_clueqc-torque-layer1-audit/index.html`。用户完成 7 簇 keep/drop 标注后，簇级广播生成 cold-start 训练种子；structured 聚类不进入 runtime、不进入 Stage 2 自训练，也不再作为长期 QC 分类器。
- 2026-06-15 CueQC 短切片定位结论保留为 Boundary 反哺线索：`cluster_00` 中代表性短噪声/短切片样本如 `867HTTM-0045 chunk875` 是 boundary cache 中独立 `speech_island` 造成的原始 ASR core chunk，duration 约 `0.43s`，不是审计页或 forced aligner 二次切割。后续可把人工确认的 `display=drop`、短噪声、过碎短切片、无效 speech island 或应合并片段整理为 Boundary hard-case / preference 数据；这一步必须保持离线偏好训练，不把 CueQC 结果耦合进 Boundary runtime。
- 历史记录 2026-06-14 ASR 非词声音可读性与 forced-aligner 去留判断：当时人工审计观察到 forced aligner 时间轴与 SpeechBoundary-JA / Boundary Refiner 产出的 speech core 窗口已经接近，说明 v5 core window 基本找准了，低信息/重复人声问题更像显示策略问题，而不是继续放大窗口或重新依赖 forced aligner 的问题。该判断已被 2026-06-18 forced-aligner 退役和 2026-06-22 CueQC v4 binary 重构 supersede。
- 2026-06-15 临时产物命名约定更新：后续所有新生成的 `agents/temp/` 临时文件或目录统一加本地时间前缀 `YYYYMMDD_HHMMSS_`，例如 `20260615_081230_cueqc-11film-baseline`。清理旧临时产物仍移动到 `agents/rm/`，清理归档目录同样使用时间前缀，避免后续无法从路径判断生成或归档时间。
- 2026-06-13 旧边界偏好试点归档：120 条盲测和 77 条 compiled label 曾用于验证 v5 边界扰动偏好信号，但未训练、未替换默认 checkpoint。2026-06-18 起该路线退出 active tree；后续不再从这批标签继续微调。
- 2026-06-11 true v5 后续路线决策已归档：当时选择从当前 true v5 做边界扰动偏好试点，不恢复 v4、不引入 forced-aligner silver 作为首轮微调。该决策的有效结论只保留为“Boundary Refiner v5 delta-only 当前默认、旧 v4/merge 路线不恢复”；后续训练计划已被 SpeechBoundary-JA hard-negative frame scorer 路线取代。
- 2026-06-11 翻译吞吐、缓存预热与 Web 状态闭环：翻译默认 worker 从 `4` 提高到 `16`，允许范围扩大到 `1-64`；自动 batch 公式提高并封顶 `400` cues，provider prefix warmup 同时覆盖 full JSON prefix 与 summary fallback。使用 NAMH-055 完整跑通 SpeechBoundary-JA、Qwen ASR、forced alignment、DeepSeek 翻译与 SRT 写出：`3199` transcript chunks、`2444` cues/blocks，总 elapsed `803.7s`，pipeline `779.9s`，其中 ASR+Alignment `286.5s`、翻译 `362.4s`、输出 `44.1s`；实际翻译批次为 `6x400 + 44`，全部 `missing_count=0`，API retry events `0`。翻译请求统计累计 provider cache hit `478976` tokens、cache miss `189486` tokens，证明 summary fallback 前缀预热和 400-cue 大批次可稳定工作；两个 400-cue 批次各触发一次批内补全但最终完整返回。Web 同步修复 FIFO 卡片顺序和 SSE 最终状态可能早于持久化的问题：活动任务保留轻量 reconciliation polling，避免后台已完成而前端持续显示运行中。Windows 审计入口新增 `tools/audits/serve_audits.ps1`，与现有 live-server middleware 保持一致。
- 2026-06-11 true v5 delta-only 32768 训练落地：按断兼容要求清理旧 merge-era 生成训练产物，`agents/temp/speech-boundary-ja/v4-core-32768-hardmix` 和旧 full-video silver refiner train/data 已移到 `agents/rm/20260610_v4_merge_weight_training_artifacts/`；`tools/boundary/train_refiner.py` 新增 fail-fast guard，只接受 `boundary_refiner_frame_sequence_dataset_v5`，并拒绝 `sequence_labels`、`sequence_context_targets`、`merge_positive`、`split_negative`、`boundary_merge_prob`、`boundary_split_prob`、`boundary_decision_merge`、`merge_label`、`merge_weight`、`split_label`、`split_weight` 等旧字段。新数据生成到 `agents/temp/speech-boundary-ja/v5-delta-only-32768/`：mixed source `196608` 条，组成为 `nsfw=88474`、`sfw=58982`、`galgame=49152`；synthetic `32768` 条，skipped `4`，`cut_point_segment_count=85192`、`cut_drop_zone_count=77820`、`overlap_mix_count=6558`，gap modes 为 `fade_noise=58967`、`hum=26161`、`silence=58851`、`white_noise=52494`。CUDA/bf16/no-compress feature cache 使用 `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`，`cached=32768`、`errors=0`、`skipped=0`；frame sequence v5 输出 `32768` sequences、`167928` items，`start_supervised=167928`、`end_supervised=167928`，label reasons 为 `boundary_cut_point=46308`、`boundary_gap_zone=74201`、`boundary_long_gap=2585`、`boundary_overlap=44834`，feature dim `630`、hash `cb44d7804ad4eaec9e5e4db80123f817a99650fa`。训练从零开始，CUDA `600` steps、batch `512`、lr `5e-4`、weight decay `0.01`、hidden/layers/state `128/2/32`、start/end loss weights `1.0/0.6`、target/core/min `3.0/5.0/0.4s`；last loss `0.003754`，train start/end MAE `0.0576/0.0591s`，val start/end MAE `0.0616/0.0603s`。checkpoint schema `boundary_refiner_v5`、output dim `2`、backbone `transformers.Mamba2Model`、runtime adapter `frame_sequence_v1`，已替换默认 `src/boundary/checkpoints/boundary_refiner.pt`；补齐顶层 feature metadata 镜像后的 sha256 为 `503d7e2299460aff555e02cba2b840c59195e577719bd0637a5ae98657ef919f`。替换后聚焦回归 `tests/test_boundary_refiner.py tests/test_boundary_refiner_training.py tests/test_boundary_cache.py tests/test_boundary_planner.py tests/test_chunk_packer.py tests/test_pipeline_chunk_config_runtime.py tests/test_pipeline_chunk_packing.py` 为 `50 passed`；补 metadata 镜像后 `tests/test_boundary_refiner.py tests/test_boundary_refiner_training.py` 为 `18 passed`。不能只凭 synthetic MAE 判定优劣，仍需真实 downstream 与人工审计 gate。
- 2026-06-11 true v5 delta-only 匿名样片 A GPU 闭环完成：命令使用默认 `src/boundary/checkpoints/boundary_refiner.pt`、`jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`、ASR batch `64`、aligner batch `128`、日文-only、不翻译、`--no-boundary-cache` 强制重算，输出 `agents/temp/speech-boundary-ja/sample-a-true-v5-delta-only-qwen17b-gpu/`。结果：`transcript_chunks=3199`、最终 `segments/blocks=2260`，pipeline total `297.9s`，整体 elapsed `300.1s`，ASR model load `8.26s`、ASR transcription `162.51s`、alignment `52.00s`、ASR+Alignment `254.1s`。质量报告：subtitle duration p50/p90/p95/max `1.09/2.171/2.473/3.438s`，long segment `0`，short ratio `0.158`，micro `81`，overlap `3` / `0.124s` total / `0.049s` max，per-min subtitle count `25.13`，density>4cps `1841`。ASR QC：reject `5`、warn `257`、empty-for-speech `245`、review uncertain `5`。Alignment fallback `731/3199` (`22.9%`)。对比 v4 core 32768 同片：chunks 相同 `3199`，blocks `2255 -> 2260`，ASR+Alignment `263.6s -> 254.1s`，pipeline total `307.6s -> 297.9s`，warn `268 -> 257`，empty `256 -> 245`，fallback `780 (24.4%) -> 731 (22.9%)`，duration p90/p95 `2.210/2.564s -> 2.171/2.473s`；短字幕比例 `0.149 -> 0.158` 略升。结论：true v5 指标小幅优于 v4，尤其是 fallback、ASR empty、warning 和耗时；但差距不大，不能直接宣布替换成功，下一步应生成 v4 vs true v5 审计页做人工观感 gate，并补 held-out 全片。
- 2026-06-11 post-v5 loss 实验回滚决策：匿名样片 A / 匿名样片 B / 匿名样片 C 都已做 v5 vs v6e 字幕对比。v6e 在部分 forced/fallback 数字上更好，但 ASR empty、字幕密度或人工观感风险没有稳定胜出；用户决定“切回 v5”。确认默认 `src/boundary/checkpoints/boundary_refiner.pt` 仍是 true v5 sha256 `503d7e2299460aff555e02cba2b840c59195e577719bd0637a5ae98657ef919f`，未替换为 v6e sha256 `d96df85f26df96688ef2abd8e9ec27b211ea6bd5ab6beb7dfa33ca188e7c7aee`。断兼容清理后，active 训练入口只保留 v5 的 `smooth_l1`、`start_delta_loss_weight=1.0`、`end_delta_loss_weight=0.6`；post-v5 的 centered/asymmetric/exp-band loss、tolerance-band 指标、独立误差评估脚本和 v6/v6e 审计/临时产物均已移出 active tree，只在 HISTORY 保留结论。
- 2026-06-11 v5 审计准备与提交：active tree 扫描确认 v6/v6e 产物文件名不再残留，v6/v6e 只保留在本文件的实验结论中；默认 checkpoint 仍是 true v5 sha256 `503d7e2299460aff555e02cba2b840c59195e577719bd0637a5ae98657ef919f`。基于 v5 全片输出重建 alignment diagnostics：`chunks=3199`、`failure candidates=1785`，生成本地音频字幕审计页 `agents/audits/v5-subtitle-audio-audit/index.html`，抽样 `102` 条，完整日文 cue `2260` 条；审计页与 `agents/` 目录按 `.gitignore` 保留本地，不随 push 发布。聚焦回归 `tests/test_audit_nav.py tests/test_boundary_refiner.py tests/test_boundary_refiner_training.py tests/test_boundary_cache.py tests/test_boundary_planner.py tests/test_chunk_packer.py tests/test_config.py tests/test_asr_stage_env_scope.py tests/test_pipeline_chunk_config_runtime.py tests/test_pipeline_chunk_packing.py` 为 `66 passed`。已提交并推送 `bff811f Finalize v5 boundary refiner cleanup`。
- 2026-06-10 v4 core-only 32768 hardmix 真实域闭环完成：按用户要求旧 11 部 silver artifacts 直接作废，使用 `agents/temp/speech-boundary-ja/v4-core-32768-hardmix/train-mamba2-v4-core-32768-s07e055/boundary_refiner.pt`、`jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`、ASR batch `64`、aligner batch `128`，对 11 部匿名全片从视频重新跑 SpeechBoundary / ASR / forced aligner。产物 `agents/temp/speech-boundary-ja/full-video-qwen17b-v4-core-32768-s07e055/`：总 transcript chunks `48848`、最终 blocks `31339`、ASR+Alignment 合计 `3790.3s`、总 elapsed `4410.5s`。单片 chunks/blocks 最高为匿名长片 `5303/4067`，字幕密度整体偏高，说明该 32768 synthetic checkpoint 适合挖真实域 silver label，但还不能只凭合成验证或训练 loss 直接替换默认。
- 2026-06-10 full-video silver boundary mining 与真实域训练链路更新：`tools/boundary/mine_silver_boundary_labels.py` 的 label policy 从 `start_weight=1.0 / end_weight=0.35` 改为 `start_weight=1.0 / end_weight=0.6`：结论是 end 不是低价值，只是字幕显示层在相邻 cue 过近时可以压缩前一条 end 来保持 2-frame gap；Boundary Refiner 训练仍应学习 end。基于上述新边界完整重跑结果挖 silver，产物 `agents/temp/speech-boundary-ja/full-video-qwen17b-v4-core-32768-s07e055/silver-boundary-labels/`：chunks `48848`、silver labels `12574`、hard cases `36274`；alignment quality 为 forced `18682`、nonlexical `18643`、proportional `11442`、drop/review `64`、partial `17`；silver start error p50/p90/p95/max `-0.16/0/0.000187/0.0005s`，end error p50/p90/p95/max `0.222375/1.059869/1.405856/2.831875s`。这说明当前 speech core start 通常略早于 forced first word，end 通常偏晚，适合作为真实域 display-boundary correction 的监督信号。
- 2026-06-10 新增真实域 silver -> Boundary Refiner v4 数据与微调链路：`tools/boundary/export_silver_sequence_features.py` 从新 silver labels 去重 11 个源音频，提权 CUDA 重新导出 full-audio `Qwen3-ASR-0.6B-JA-Anime-Galgame` PTM + MFCC sequence frame `.npz`，产物 `agents/temp/speech-boundary-ja/full-video-qwen17b-v4-core-32768-s07e055/silver-sequence-features-ptm64/`，导出 `11/11`、errors `0`。`tools/boundary/build_silver_refiner_dataset.py` 再把 silver labels、aligned transcript chunks 和 sequence feature manifest 转成 `boundary_refiner_frame_sequence_dataset_v4` 训练 JSONL，产物 `.../silver-refiner-dataset-v1/`：rows `47`、sequence items `20734`、split_negative `20734`、start_supervised `12570`、end_supervised `12573`、overlapping_chunk_gap `616`、feature_dim `630`、feature_schema_hash `ab5ca6f85b2e4da013a7ffaeef0ba73ccbd598a6`。随后从 32768 hardmix checkpoint 初始化，冻结 Mamba2 backbone，只训练输出头，`merge_loss_weight=0`、`start/end_delta_loss_weight=0.80/0.60`、`preserve_init_normalization=1`、`lr=1e-4`、`300` steps，得到 `.../train-mamba2-v4-core-32768-s07e055-silver-ft01/boundary_refiner.pt`。该 silver-ft01 尚未设为默认，下一步必须用 匿名样片 A / 匿名样片 B 做 downstream A/B。
- 2026-06-10 silver-ft01 downstream A/B 完成：使用同一 Qwen3-ASR-1.7B、ASR batch `64`、aligner batch `128`、`--no-boundary-cache`，只把 Boundary Refiner checkpoint 从 32768 hardmix 换成 silver-ft01，重跑 匿名样片 A / 匿名样片 B。匿名样片 A：chunks `3199 -> 1710`、blocks `2251 -> 1288`、ASR+Alignment `254.6s -> 161.5s`、QC warn `265 -> 125`、ASR empty `255 -> 119`、reject `5 -> 4`、fallback ratio `0.229 -> 0.205`、per-min subtitle count `25.0 -> 14.3`、short ratio `0.161 -> 0.071`、overlap `1 -> 1`。匿名样片 B：chunks `5303 -> 3618`、blocks `4067 -> 2954`、ASR+Alignment `518.8s -> 348.7s`、QC warn `484 -> 294`、ASR empty `448 -> 265`、reject `11 -> 8`、fallback ratio `0.197 -> 0.197`、per-min subtitle count `26.9 -> 19.5`、short ratio `0.162 -> 0.094`、overlap `3 -> 1`。结论：真实域 forced-aligner silver label 用于 start/end display-boundary delta 微调的方向有效，显著降低过切、短字幕密度和 ASR 空输出；仍需生成审计页确认是否误合并真实短促台词，再决定是否替换默认 checkpoint。
- 2026-06-10 人工审计修正 silver-ft01 结论：生成 `subtitle-ab-v4-32768-vs-silver-ft01-{video,audio}` 后，用户人工确认 silver-ft01 “学坏了”，观感和准确度不如 32768 hardmix。说明单看 chunk/cue 减少、ASR empty 下降和 fallback ratio 下降会误导模型选择；后续真实域 silver 微调必须加入人工审计 gate，并控制 hard negative / forced-aligner silver label 对 start/end delta 的过度保守影响。
- 2026-06-10 4096 vs 32768 A/B 审计完成：生成 `agents/audits/subtitle-ab-v4-4096-vs-32768-audio/index.html`，同片 匿名样片 A 对比旧 4096 默认与 32768 hardmix。人工观感确认 32768 更好，因此默认 checkpoint 切到 32768 hardmix。该结果支持保留 32768 synthetic 规模和更高组合复杂度；下一轮训练应沿用 32768 hardmix 策略，而不是回到 4096 或采用 silver-ft01 的过度压缩行为。
- 2026-06-10 Boundary Refiner v4 core-only 匿名样片 A GPU 闭环完成：命令显式使用 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`、ASR batch `64`、aligner batch `128`、日文-only、不翻译、`--no-boundary-cache` 强制重算，输出 `agents/temp/speech-boundary-ja/sample-a-v4-core-qwen17b-gpu/`。CUDA 链路确认：SpeechBoundary-JA `actual_device=cuda`、PTM `cuda:0`；运行中 ASR 阶段显存采样约 `7116/8188 MiB`，适合本机 8GB 实验档，不应写成 6GB 安全档。结果：`transcript_chunks=3199`、最终 `segments/blocks=2255`，ASR+Alignment `263.6s`，pipeline total `307.6s`，整体 elapsed `309.7s`。质量报告：subtitle duration p50/p90/p95/max `1.087/2.210/2.564/3.436s`，long segment `0`，short ratio `0.149`，micro `81`，overlap `3` / `0.08s` total / `0.04s` max，per-min subtitle count `25.07`，density>4cps `1806`。ASR QC：reject `5`、warn `268`、empty-for-speech `256`、review uncertain `5`，reject 主要是 `repeat_ngram_loop`。Alignment diagnostics：chunks `3199`、forced `1384`、nonlexical `1032`、proportional `777`、partial `1`、drop/review `5`；fallback chunks `780` (`24.4%`)，fallback subtype 为 `proportional_after_sentinel=777`、`word_timing_high_cps=1`、`asr_review_uncertain=5`，nonlexical text `1032` 不计入 fallback type。结论：v4 core-only 没有回到长粗时间轴，真实瓶颈是 subtitle density / 短非词人声 / repeat loop / forced-aligner sentinel，而不是 padding 或 runtime merge。
- 2026-06-10 闭环后元数据修复：正常 ASR 完成路径此前没有把 `_LAST_BOUNDARY_SIGNATURE` 写入 `asr_details`，导致 `tools/boundary/ja/run_full_workflow.py` 的 `summary.json` 中 `boundary_signature` 为空；这不影响已生成字幕，但削弱后续复盘。已在 `src/asr/pipeline.py` 正常完成 details 中写入 `boundary_signature`，并在 `tests/test_pipeline_chunk_packing.py` 覆盖 version `4`。验证：`compileall` 通过，`tests/test_pipeline_chunk_packing.py` 8 passed，`pytest -q` 为 `459 passed`。
- 2026-06-09 Boundary Refiner v4 断兼容重构完成：no-padding GPU A/B 已证明 padded ASR context 会明显增加空转写、字幕密度和重叠风险，因此主线从 v3 `merge + left_context + right_context + start_delta + end_delta` 改为 v4 core-only `merge + start_delta + end_delta`。runtime 删除 learned padding/context budget，`PackedChunk`、ASR chunk metadata、boundary cache 和训练数据不再写 `speech_left_padding_s` / `speech_right_padding_s` / `sequence_context_targets` / `BOUNDARY_CONTEXT_MAX_PADDING_S`；ASR chunk、fallback window 和字幕初始时间轴直接使用 refined speech core。`boundary-cache` 升 v4，旧 cache 直接 miss；旧 v3 checkpoint 不靠 sha256 白名单拦截，而是会因 `schema=boundary_refiner_v3` / `output_dim=5` 与 v4 `schema=boundary_refiner_v4` / `output_dim=3` 不匹配而拒绝加载。
- 2026-06-09 v4 core-only 4096 checkpoint 已训练并替换默认分发文件：从本地 `japanese-anime-speech-v2` NSFW/SFW 与 Galgame 100k 源池按 `nsfw=45 / sfw=30 / galgame=25` 生成 `20480` 条 mixed source，再合成 `4096` 条 multi-island 样本，覆盖 touching speech、short gap、long gap、轻量 overlap、gain/filter/codec 增强；summary 记录 `overlap_mix_count=644`、`filter_aug_count=1232`、`codec_aug_count=229`、`cut_point_segment_count=6382`、`cut_drop_zone_count=10000`。CUDA/bf16/no-compress feature cache 使用 `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`，`batch_size=128`、`prepare_workers=8`，`cached=4096`、`errors=0`、`skipped=0`；frame-sequence v4 输出 `4096` sequences、`21022` boundary items、`merge_positive=4096`、`split_negative=16926`、`feature_dim=630`、`feature_schema_hash=ab5ca6f85b2e4da013a7ffaeef0ba73ccbd598a6`。Mamba2 CUDA 训练 `300` steps、batch `512`、`lr=5e-4`、hidden/layers/state `128/2/32`，last train loss `0.004981`，synthetic train/val accuracy 与 merge F1 均为 `1.0`。默认 `src/boundary/checkpoints/boundary_refiner.pt` 已替换为 v4 checkpoint，sha256 `61c7c948d27944b2399b94742263b787403d151c658b2c553b783242010703ff`，大小约 `2.2MB`；训练大中间产物已删除，只保留 `agents/temp/speech-boundary-ja/v4-core-only-4096/keep/` 下的 summary、metrics 和 checkpoint 备份。验证：`tests/test_boundary_refiner.py tests/test_boundary_refiner_training.py tests/test_chunk_packer.py tests/test_boundary_cache.py tests/test_pipeline_chunk_config_runtime.py tests/test_boundary_timing_formula.py tests/test_pipeline_chunk_packing.py tests/test_fallback_safe_boundary_metrics.py` 共 `67 passed`。下一步真实 gate 是匿名样片 A / held-out GPU downstream，观察 start attribution、fallback duration、ASR empty、repeat/nonlexical 风险和 forced/fallback 分布。
- 2026-06-09 审计页持续刷新根因确认并修复：页面本身没有 `meta refresh` / `location.reload()` / `window.location` 自动跳转；真正问题是从项目根目录裸跑 `live-server` 时，它会 watch 全项目并给 HTML 注入 websocket，训练和 GPU 闭环持续写 `agents/temp`、`tmp`、cache、日志时会触发浏览器 reload。新增 `tools/audits/serve_audits.sh`，仍从项目根目录提供视频/音频等相对资源，但只 watch `agents/audits`，并保留 `tools/audits/live_server_audit_middleware.js` 删除入口；`tools/audits/audit_nav.py` 和当前 `agents/audits/index.html` 已同步提示该启动方式。
- 2026-06-09 no-padding A/B GPU 闭环完成：使用当时的 v3 4096 默认 Mamba2 checkpoint，不覆盖模型，只在 runtime 显式 `--boundary-context-max-padding-s 0 --no-boundary-cache` 重跑匿名样片 A / Qwen3-ASR-1.7B / ASR bs64 / aligner bs128。输出 `agents/temp/speech-boundary-ja/sample-a-v3-4096-qwen17b-nopad-gpu/`，pipeline total `306.7s`，`transcript_chunks=3067`、最终 `blocks=2161`、ASR empty `258`、QC reject `4`、fallback ratio `0.245`、short ratio `0.146`、overlap `3`。对比 padded v3 4096 的 `blocks=3687`、ASR empty `530`、QC reject `16`、overlap `486`，no-padding 明显减少 ASR context 污染和字幕密度/重叠问题。当时默认 4096 checkpoint 已备份到 `agents/temp/speech-boundary-ja/v3-boundary-delta-4096/backup-current-4096-before-nopad/boundary_refiner.v3-4096-before-nopad.pt`，sha256 与当时 `src/boundary/checkpoints/boundary_refiner.pt` 同为 `763ef5cbb66e3d2e49bf1003adccefabf0ff4540aa227b0ff8a6fa866d6d46fe`；后续已被 v4 core-only checkpoint 替换。
- 2026-06-09 历史 v3 4096 重训版记录：先清理可重建生成产物并释放旧 `v3-boundary-delta` 大目录约 `175G`，只保留本地源数据集 `boundary-sources/*` 与 `datasets/raw/musan`；随后重新生成 `4096` synthetic records，manifest 权重 `anime_nsfw=1843`、`anime_sfw=1229`、`galgame=1024`。CUDA/bf16 feature cache 使用 `--no-compress --resume --batch-size 128 --prepare-workers 8` 跑完，summary 确认 `compressed=false`、`cached=4096`、`errors=0`、`skipped=0`。frame-sequence v3 输出 `4096` sequences、`21036` items、`merge_positive=4096`、`split_negative=16940`、`feature_dim=630`、`feature_schema_hash=ab5ca6f85b2e4da013a7ffaeef0ba73ccbd598a6`。Mamba2 CUDA 训练 `300` steps、batch `512`、`lr=5e-4`、hidden/layers/state `128/2/32`，synthetic train/val accuracy 与 merge F1 均为 `1.0`。`src/boundary/checkpoints/boundary_refiner.pt` 曾短暂替换为该 checkpoint，sha256 `763ef5cbb66e3d2e49bf1003adccefabf0ff4540aa227b0ff8a6fa866d6d46fe`；现已被 v4 core-only checkpoint 替换。
- 4096 重训后的分发边界：普通推理只需要源码内置 `src/boundary/checkpoints/boundary_refiner.pt`，不需要训练时的 CUDA feature cache、synthetic WAV、frame-sequence JSONL 或 tensor cache。4096 训练完成后已删除大体积 synthetic WAV、feature-cache、frame-sequence JSONL/tensor cache 和 mixed-source manifest，只保留 `keep/*.json`、训练 `metrics.json`、新 checkpoint 与旧 checkpoint 备份；`agents/temp/speech-boundary-ja/v3-boundary-delta-4096/` 当前约 `4.5M`。验证：`tests/test_boundary_refiner_training.py tests/test_boundary_refiner.py tests/test_boundary_cache.py tests/test_boundary_planner.py tests/test_chunk_packer.py tests/test_pipeline_chunk_config_runtime.py` 共 `54 passed`。下一步真实 gate 仍是用匿名样片 A / held-out 重跑 GPU downstream，观察 start attribution、fallback duration、ASR empty、repeat/nonlexical 风险和 forced/fallback 分布。
- 2026-06-09 Boundary Refiner v3 断兼容协议：主线升级为 `boundary_refiner_v3`，Mamba2 head 从 `merge + left_context + right_context` 扩展为 `merge + left_context + right_context + start_refine_delta + end_refine_delta`，`boundary-cache` 升 v3。runtime 取消固定 `BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S`、`BOUNDARY_CONTEXT_MAX_SPEECH_OVERLAP_S`、gap 比例裁剪和 split-overlap 0.25s cap；ASR padded window 只按模型输出 context budget 与 `BOUNDARY_CONTEXT_MAX_PADDING_S` 裁剪，左侧仅保证不小于 0。`PackedChunk` / ASR chunk metadata / cache 显式写入 `boundary_start_refine_delta_s` 和 `boundary_end_refine_delta_s`，这些 delta 现在会真实调整 chunk core start/end，而不是只写诊断字段。训练协议同步升级：`boundary_refiner_frame_sequence_dataset_v3` 必须包含 `sequence_context_targets` 与 `sequence_boundary_delta_targets`；训练 loss 为 merge BCE + context SmoothL1 + start/end delta SmoothL1，start delta 权重大于 end delta。32768 v3 是上一轮大样本实验，不再是当前默认 checkpoint；其 synthetic validation 同样不能替代真实 downstream。
- 2026-06-08 no-runtime-merge + Qwen3-ASR-1.7B bs64 匿名样片 A GPU 闭环完成：命令显式使用 `--asr-backend jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame --asr-batch-size 64 --no-keep-asr-chunks`，输出 `agents/temp/speech-boundary-ja/sample-a-no-runtime-merge-qwen17b-bs64-gpu/`，总耗时约 `317s`，其中 ASR+alignment `275.9s`、ASR transcribe `178.5s`、alignment `57.3s`。ASR batch64 运行中观测显存约 `6.9-7.0GB`，说明适合本机 8GB 实验档，但不应写成 6GB 安全档。结果 `transcript_chunks=3199`、`segments=2359`、`blocks=2359`，确认 runtime merge / subtitle merge 删除后没有再把相邻 cue 合并；QC 为 `reject=6`、`warn=426`、`empty_text_for_speech=414`。诊断输出 `agents/temp/speech-boundary-ja/diagnostics-sample-a-no-runtime-merge-qwen17b-bs64-gpu/`，审计页 `agents/audits/asr-attribution-no-runtime-merge-qwen17b-bs64-video/index.html`，归因计数 forced `1382`、nonlexical `1007`、proportional_after_sentinel `804`、drop/review `6`，density>4cps `724`。批量策略同步更新：源码 / `.env.example` 默认按 6GB 档 `0.6B=64 / 1.7B=32 / aligner=64`；本机 `.env` 使用 8GB 实验档 `0.6B=128 / 1.7B=64 / aligner=128`，OOM 时退回 6GB 档。
- 2026-06-08 runtime merge 策略断兼容删除收尾：按“先删除所有 merge 策略，重跑后再审计”的方向，主流程不再跨 speech island 打包成一个 ASR chunk，`src/boundary/ja/backend.py` 的 raw segment gap merge 改为纯 `filter_segments`，`SPEECH_BOUNDARY_JA_MERGE_GAP_S` 从 cache signature、workflow CLI 和 advanced env 中移除；subtitle writer 已移除 adjacent/dense/overlap cue merge，ASR fragment postprocess merge 也已移除，旧 cue-merge 分析工具/测试移入 `agents/rm/merge-strategy-removal/`。审计页同步改为区分 `ASR padded window` 与 `speech/fallback core`：最新页面中 chunk2 `15.332-17.160` 和 chunk3 `16.660-19.229` 的重叠仅是 ASR padding/context overlap，实际 core 为 `15.460-16.910` 与 `16.910-19.220`，没有 speech-core 重叠。验证：`tests/test_reward_boundary_planner.py tests/test_asr_attribution_audit.py tests/test_run_full_workflow_env.py tests/test_boundary_planner.py` 13 passed。
- 2026-06-08 real-domain silver boundary mining 第一轮正式结果改用 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame` 重跑；误用 0.6B 的旧产物已移到 `agents/rm/20260608_0p6b_silver_boundary_results/`，不作为训练数据。目标不是 runtime 新增一层模型，而是从真实片源挖 forced-aligner 成功、QC ok、词级时间轴可靠的实例数据，作为后续 Mamba2 Boundary Refiner v3 的 display-boundary correction / context budget 监督信号，并为 RL/DPO reward 准备 hard-case 池。实现：主 ASR/aligner 流程显式导出 per-chunk/per-segment `alignment_mode`、`alignment_quality`、`fallback_type`、`fallback_subtype`、`forced_success`、QC/review 标记和 word timing stats，避免后续 miner 继续从日志猜 forced/fallback；新增 `tools/boundary/mine_silver_boundary_labels.py`，输出 `silver_boundary_labels.jsonl`、`hard_cases.jsonl`、`summary.md/json` 和轻量 HTML 审计页。过滤策略：只收 `alignment_quality=forced`、`fallback_subtype=none`、QC ok、非 nonlexical/align-text-empty、非短/重复 vocalization、无 word timing failure 且有正时长 words；短假名台词若 forced/QC 正常允许进入 silver，避免误伤 JAV/Galgame 真实短句。1.7B 两片源日文-only GPU 闭环输出 `agents/temp/speech-boundary-ja/silver-boundary-mining-v1-two-videos-qwen17b/`，ASR+alignment 分别约 `435.5s` / `842.8s`，fallback ratio 约 `0.264` / `0.223`；silver mining 输出 `agents/temp/speech-boundary-ja/silver-boundary-labels-v1-two-videos-qwen17b/` 和 `agents/audits/silver-boundary-labels-v1-two-videos-qwen17b/index.html`，汇总 `chunks=8482`、`silver=3202`、`hard_cases=5280`，start error p50/p90/p95/max `-0.123/0.250/0.250/1.012s`，end error p50/p90/p95/max `0.172/1.018/1.351/3.560s`；alignment quality 为 `forced=4075`、`nonlexical=2366`、`proportional=2018`、`drop_or_review=20`、`partial=3`。同时修复 `tools/boundary/ja/run_full_workflow.py` 的显式 CLI 参数优先级：`--asr-batch-size` / `--aligner-batch-size` / 模型与 boundary 参数现在会覆盖本机 `.env`，避免 1.7B 实验命令记录和实际 batch 不一致；`.env` 仍可保留本机 8GB 更激进配置，`.env.example` 继续面向分发/6GB。
- 2026-06-08 cue start anchor 继续收紧：用户指出可靠 forced word start 应比 segment/cue start 更可信。实现落点在 `src/subtitles/writer.py` 的 subtitle block window：当 cue/block 带有正时长 timed words 时，`start` 锚到 earliest `word.start`，覆盖普通 cue、soft split 后 cue 和 merge 后 cue；没有有效 words 时仍用原 block/speech-core/fallback start。冲突处理继续沿用 start-locked 策略：不推迟当前 cue start，必要时压缩上一条 end 保 2-frame gap。新增回归覆盖单 cue `segment.start` 晚于 first word、以及相邻 cue merge 后仍保留 earliest word start。验证：`tests/test_subtitle_quality_pass.py tests/test_subtitle_options.py tests/test_asr_segmentation_boundaries.py tests/test_subtitle_qc.py` 66 passed。
- 2026-06-08 ASR “低信息”口径断兼容改为中性 `text_density`：外部 ASR/字幕经验支持“不要只凭文本短/重复就删除”，应结合 `no_speech_prob`、`avg_logprob`、`compression_ratio`、重复度、时长和人工审计；JAV / Galgame 目标域里的短假名、喘息、呻吟、语气词可能是真实可转写内容。实现：`src/asr/qc.py` 将旧 `low_information` profile 改为 `text_density`，枚举改为 `normal_dialogue`、`empty_or_punctuation`、`long_sparse_text`、`short_vocalization_candidate`、`repeated_vocalization_candidate`、`short_kana_dialogue_candidate`；保留行为不变，仍是 `preserve_with_review`，不新增删除规则。`tools/asr/diagnostics/diagnose_asr_alignment.py` 不再把短/重复低信息人声推成 `low_information_text` failure bucket，避免后续 failure manifest / hard-negative 训练把真实目标域人声误当错误；ASR attribution audit 仍保留 `low_info_vocal` 人工标签，但底层数据字段改为 `text_density_*`。回归：`tests/test_asr_qc_signals.py tests/test_asr_alignment_diagnostics.py tests/test_asr_attribution_audit.py tests/test_alignment_failure_manifest.py` 28 passed。
- 2026-06-08 start-locked subtitle timing 策略落地：根据用户修正，字幕观感目标从“start 准、end 可略长”改为 **start 必须准，end 可略短 / 可被压缩**。Grok 复核 Netflix Timed Text Timing Guidelines：in-time 要贴近第一帧音频或 1-2 帧内，字幕间保留 2-frame gap，小间隔主要通过 out-time / chaining 调整；这支持“不要推迟下一条 start，优先压前一条 end”。实现：`_repair_postprocessed_segment_windows()` 不再在 overlap 时把当前 segment start 推到上一条 end，而是压缩上一条 end 到当前 start；`_normalize_subtitle_timeline()` 极端冲突分支不再 shift next start，而是允许前一条 end 压到 start。新增回归覆盖 ASR postprocess 与 subtitle writer 的 start-lock。验证：`tests/test_asr_segmentation_boundaries.py tests/test_subtitle_quality_pass.py tests/test_subtitle_options.py tests/test_pipeline_chunk_packing.py tests/test_subtitle_qc.py tests/test_quality_report_output.py` 76 passed。匿名样片 A 既有产物 replay 显示 prepared cue 仍有少数 `cue_start - first_word_start > 0.2s`，但主要来自已有 aligned segment / cue merge 的词段不一致，不是新 no-overlap 修复继续后移 start；下一步如要彻底归零，需要在 cue merge/segment 构造中保留 earliest word start 作为 start anchor。
- 2026-06-08 v2 context-budget 默认 checkpoint 的匿名样片 A GPU 闭环已完成，日文-only、不翻译，命令入口 `tools/boundary/ja/run_full_workflow.py --video video/匿名样片 A.mp4 --task-name sample-a-full-v2-context-budget-gpu --label v2_context_budget_gpu --subtitle-mode ja --no-keep-asr-chunks`，输出 `agents/temp/speech-boundary-ja/sample-a-full-v2-context-budget-gpu/`。CUDA 链路确认：SpeechBoundary-JA actual device `cuda`，PTM `cuda:0`，ASR backend `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`。结果：`transcript_chunks=3190`、`segments=2600`、`blocks=1796`，ASR+Alignment `283.8s`，pipeline total `326.9s`；ASR batch64 / aligner batch64 未 OOM，运行中 ASR 显存约 `5GB`，符合本机 8GB 与 6GB 分发目标的安全区间。字幕质量：duration p50/p90/p95/max `1.64/3.162/3.516/5.313s`，overlap `0`，ASR empty-for-speech `361`，review uncertain `7`，per-minute `19.97`。chunk 侧 p50/p90/p95/max `1.274/2.668/2.964/5.169s`，fallback speech-core window p50/p90/p95/max `0.98/2.33/2.66/4.91s`，说明 v2 context-budget 已把 fallback 时间窗压到短 speech core，当前瓶颈主要是 cue density、低信息/重复人声和 forced-aligner sentinel，而不是 20-30s 粗 fallback。
- 2026-06-08 GPU 闭环质量报告口径修正：旧 `alignment_fallback_ratio=0.990` 是假高，根因是 QC 把 `Alignment 回退窗口: speech_core` 这种“fallback 插值时间窗来源”误计为 alignment fallback，并且 fallback count 分母使用最终 SRT blocks 而不是 ASR chunk count。已修正 `_alignment_fallback_count_from_log()`：`Alignment 回退窗口` 不再计数，真实失败只按 `even_fallback` / `aligner_vad_fallback` / sentinel / 降级等 chunk marker 去重；`write_quality_report()` 改用 `asr_details.chunk_count` 作为 fallback ratio 分母，并在报告中写出 `alignment_fallback_count` / `alignment_fallback_total`。用同一份 匿名样片 A run log 重算后：final alignment mode 为 forced `1412`、even_fallback `1008`、nonlexical `770`，修正后 `alignment_fallback_ratio=1008/3190=0.316`。这仍是需要观察的 forced-aligner 域不匹配/短非词文本问题，但不是 `0.99` 级别失败。回归：`tests/test_subtitle_qc.py tests/test_quality_report_output.py tests/test_quality_report_per_job.py tests/test_pipeline_chunk_packing.py::test_alignment_fallback_count_deduplicates_chunk_log_markers` 20 passed。
- 2026-06-08 learned context budget 正式 v2 checkpoint 已完成并替换默认分发文件。数据重新生成到 `agents/temp/speech-boundary-ja/v2-context-budget/`：mixed source 为 anime NSFW/SFW `60/40` 后再按 anime/galgame `65/35` 抽样，synthetic `32768` 条，hard negatives 包含 real negatives/background、touch/short/regular gaps、overlap、gain/filter/codec aug；summary：`cut_point_segment_count=64019`、`cut_drop_zone_count=57337`、`overlap_mix_count=6539`、`background_mix_count=21277`。CUDA/bf16/no-compress feature cache 使用 `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`，`batch_size=32`、`prepare_workers=8`，`cached=32768`、`errors=0`、`skipped=0`，训练中间 cache 约 `196G`。frame-sequence dataset：`32768` 条 sequence、`161815` 个 boundary item，`merge_positive=32768`、`split_negative=129047`，`feature_dim=630`，`feature_schema_hash=ab5ca6f85b2e4da013a7ffaeef0ba73ccbd598a6`。Mamba2 CUDA 训练 `600` steps、`batch_size=512`、`lr=3e-4`、`weight_decay=0.01`、hidden/layers/state `128/2/32`、`context_loss_weight=0.25`、`target_domain_speedup=1.5`、`target/core/padded=3.0/5.0/6.5s`，final loss `0.001233`；train accuracy/F1 `1.0/1.0`，validation accuracy `0.999938`、merge precision `1.0`、recall `0.999695`、F1 `0.999847`。新 checkpoint 已复制到 `src/boundary/checkpoints/boundary_refiner.pt`，sha256 `4e075f388426086cfb020c57f4f02c65cbd312a73b968ac9e17c5df7159f7841`，大小 `2.2MB`；加载 smoke 与 `tests/test_boundary_refiner.py tests/test_boundary_refiner_training.py tests/test_boundary_planner.py tests/test_boundary_cache.py` 共 `36 passed`。
- 2026-06-08 learned context budget 方案落地：按本地 `japanese-anime-speech-v2` 统计，NSFW split 实际 `20849` 条、p50/p80/p90/p95/max `8.65/12.17/14.38/16.31/35.32s`，SFW `40000` 条、p50/p80/p90/p95/max `4.44/6.73/8.16/9.54/28.18s`；结论是不把 NSFW p90/p95 当 runtime speech-core 上限，而是继续 `target/core=3.0/5.0s`，把长 NSFW 作为 hard negative/长句拆分训练素材。Boundary Refiner schema 断兼容升级为 `boundary_refiner_v2`：Mamba2 head 从单一 merge logit 扩为 `merge_logit + left_context_s + right_context_s`，`BOUNDARY_PLANNER_TARGET_PADDING_S` 删除，新增 `BOUNDARY_CONTEXT_MAX_PADDING_S=1.5`、`BOUNDARY_CONTEXT_MAX_SPEECH_OVERLAP_S=0.25`，默认 padded cap 收紧到 `BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S=6.5`；boundary-cache 升 v2。早期默认 checkpoint 曾先迁移为 v2 中性 context 结构以保持源码可运行，现已被正式 v2 context-budget 重训 checkpoint 替换。
- 2026-06-07 Boundary Refiner 分发边界确认：推理 / 新用户分发只需要源码内置的 `src/boundary/checkpoints/boundary_refiner.pt`（约 `2.2MB`）加 Hugging Face 下载的 0.6B full SFT ASR / frozen feature 模型和 forced aligner；32768 重训过程中生成的 CUDA feature cache、synthetic WAV、sequence JSONL 和 `datasets/train/...` 都是可重建训练产物，不参与普通推理。PyInstaller spec 已把 `src/boundary/checkpoints/` 改为 required data include，并显式收集 `transformers.models.mamba2`，避免 release 缺 checkpoint 或 Mamba2 lazy import。
- 2026-06-07 README 分发化精简：README 删除真实番号 / 匿名样片 A 指标、Boundary Refiner 训练路线、Galgame 100k 参数公式、失败实验、审计标签细节和历史路径记录，只保留新用户需要的安装、使用、默认模型、当前工作流、目录结构、常见问题和开发入口。实验依据、路线取舍、指标和调试过程继续只写 HISTORY，避免 README 变成实验日志。
- 2026-06-07 匿名样片 A full-run 回归：先跑 60s smoke，CUDA 链路正常，`30` ASR chunks、`23` aligned segments、`19` SRT blocks，total `21.9s`。随后全片第一次卡在 `_build_processing_spans()`，根因是 `FrameSequenceFeatureProvider.features_for_gap()` 每个 gap 都重复把整片 PTM/MFCC frame list 转成 numpy array，长片候选多时变成 CPU-bound。修复为 provider 初始化时缓存 PTM/MFCC numpy arrays、使用 `_gap_window_sequence_features_from_arrays()` 复用整片 frame buffer，并新增 `test_frame_sequence_feature_provider_caches_frame_arrays` 回归。修复后全片完整跑通：输出 `agents/temp/speech-boundary-ja/sample-a-full-default-boundary-refiner-fixed-seqfeat-cache/`，`transcript_chunks=3178`、`segments=4240`、`blocks=2105`、ASR+Alignment `422.0s`、total `457.8s`；ASR 显存约 `4.8-5.2GB`，符合 6GB 目标。质量报告：subtitle duration p50/p90/p95/max `1.65/3.392/3.533/5.323s`，`per_min_subtitle_count=23.4`，`alignment_fallback_ratio=0.811`，`asr_empty_text_for_speech_count=470`。本轮确认 Boundary Refiner 可完成全片，下一步质量瓶颈仍是 cue density、低信息/重复人声和 forced-aligner fallback 观察项。
- 2026-06-07 README 旧路线表述清理：active tree 已没有 speaker sidecar / runtime diarization / WhisperSeg / TEN / Silero / FSMN / 旧 FusionVAD 运行入口；README 不再写“默认不启用这些路线”，避免让新用户误以为仍存在可选开关。旧路线说明继续只保留在 HISTORY。
- 2026-06-07 `speaker_proxy` 数据与训练链路断兼容清理：不再把 speaker proxy / speaker turn 当作 Boundary Refiner synthetic metadata。旧 `datasets/train/fusionvad-ja/v1-23-boundary-refiner/` 已移到 `agents/rm/speaker-proxy-datasets-20260607/`，中断/错误权重产物移到 `agents/rm/interrupted-boundary-dataset-20260607/` 和 `agents/rm/wrong-weight-boundary-dataset-20260607/`，旧 anime 512/256 小源池移到 `agents/rm/obsolete-small-anime-boundary-sources-20260607/`。active `datasets/` 扫描 `speaker_proxy|speaker_turn|speaker_changed|split_speaker` 为 0。代码层同步删除 `speaker_proxy_id()`、`--speaker-proxy-*`、`speaker_proxy_ids`、`speaker_turn_boundaries`、`split_speaker_*` label，改为 `utterance_boundaries` + `cut_point`。新本地源池：`datasets/train/boundary-sources/japanese-anime-speech-v2-nsfw-60k/` 实际 `valid_rows=20849`（HF nsfw split 已耗尽，无法达到 60000 唯一条）、`datasets/train/boundary-sources/japanese-anime-speech-v2-sfw-40k/` 为 `40000` 条、Galgame 100k 继续用 `datasets/train/boundary-sources/galgame-asr-100k-ogg/`。按 anime 内部 `nsfw:sfw=60:40` 有放回抽样生成 `datasets/train/speech-boundary-ja/v1-boundary-refiner/anime-source-nsfw60-sfw40-100k/`，再按 `anime=65 / galgame=35` 随机生成 `mixed-source-anime65-galgame35/`，最终 4096 synthetic timeline 输出 `mixed-anime65-galgame35-boundary4096/`：`records=4096`、`utterance_boundary_count=16384`、`cut_point_segment_count=11524`、`cut_drop_zone_count=4813`、`real_negative_gap=12299`、`overlap_mix=320`。
- 2026-06-07 32768 hard-negative Boundary Refiner 重训：按用户要求从 anime65/galgame35 mixed source 生成更强 hard negatives，输出 `datasets/train/speech-boundary-ja/v1-boundary-refiner/mixed-anime65-galgame35-hardneg32768-v2/`，`records=32768`、WAV 约 `34G`、`utterance_boundary_count=163840`、`cut_point_segment_count=123964`、`cut_drop_zone_count=31760`、`background_mix_count=20244`、`overlap_mix_count=5808`，gap modes：`real_negative=150507`、`fade_noise=10160`、`silence=10132`、`white_noise=8479`、`hum=4356`。随后用 `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame` CUDA/bf16 生成 no-compress feature cache，路径 `.../mixed-anime65-galgame35-hardneg32768-v2-feature-cache-nocompress/`，`cached=32768`、`errors=0`、`skipped=0`，体积约 `214G`。frame-sequence dataset `boundary-refiner-frame-sequence-hardneg32768-speedup15-v2/`：`32768` 条 sequence、`225464` 个候选 item，`merge_positive=65536`、`split_negative=159928`，label reasons 为 `merge_synthetic_intra_island=65536`、`split_cut_point=69468`、`split_gap_zone=30101`、`split_long_gap=1334`、`split_overlap=59025`，`feature_dim=630`，`feature_schema_hash=ab5ca6f85b2e4da013a7ffaeef0ba73ccbd598a6`。Mamba2 CUDA 训练 `600` steps、`batch_size=512`、`lr=3e-4`、`weight_decay=0.01`、hidden/layers/state `128/2/32`，产物 `boundary-refiner-frame-sequence-mamba2-hardneg32768-speedup15-v2/boundary_refiner.pt`。validation：accuracy `0.999911`、merge precision/recall/F1 `0.999847`、FP `1`、FN `1`。该 checkpoint 已替换默认分发文件 `src/boundary/checkpoints/boundary_refiner.pt`，sha1 `0497912ff24ac63fec6c4c94bb9dea513490a3a5`，体积约 `2.2MB`。
- 2026-06-07 32768 训练内存坑修复：首次训练直接读取 `3.5G` `sequences.jsonl` 时，旧 `tools/boundary/train_refiner.py` 会把 JSON rows 全量解析成 Python `list[dict]`，再构造 padded tensor；WSL2 8GB RAM 被吃满并使用 swap，最终系统卡崩且没有有用 traceback。现已改为两遍流式扫描：第一遍只读取 schema / feature_names / max_len / class counts / metadata，第二遍直接填紧凑 `torch.Tensor`；训练集用 `Subset(TensorDataset(...))`，评估按 batch/index streaming，不再复制整份 train/val tensor。回归测试新增 `test_train_refiner_uses_streaming_tensor_loader`，防止未来把训练入口改回整量 JSON rows。
- 2026-06-07 Mamba2 Boundary Refiner 4096 版基线：先用 `mixed-anime65-galgame35-boundary4096/` 生成 Qwen3-ASR-0.6B full SFT CUDA feature cache，路径 `datasets/train/speech-boundary-ja/v1-boundary-refiner/qwen3-asr-0.6b-fullsft/mixed-anime65-galgame35-boundary4096-feature-cache-nocompress/`，结果 `cached=4096`、`errors=0`、`skipped=0`。随后生成 `boundary-refiner-frame-sequence-speedup15-v1/`：`4096` 条 sequence、`20660` 个 boundary item，`merge_positive=4096`、`split_negative=16564`，`feature_dim=630`，`feature_schema_hash=ab5ca6f85b2e4da013a7ffaeef0ba73ccbd598a6`，feature signature 显式包含 `target_chunk_s=3.0`。CUDA 训练 `300` steps，`batch_size=512`、`lr=5e-4`、hidden `128`、layers `2`、state `32`、backbone `transformers.Mamba2Model`，产物 `boundary-refiner-frame-sequence-mamba2-speedup15-v1/boundary_refiner.pt`。validation：accuracy `0.998545`、merge precision `0.997555`、merge recall `0.995122`、F1 `0.996337`、FP `1`、FN `2`。该版曾短暂作为默认分发 checkpoint，现已被 32768 hard-negative v2 checkpoint 替换。
- 2026-06-07 Boundary Refiner 训练指标口径修复：`tools/boundary/train_refiner.py` 的 `_evaluate()` 过去把 padded sequence slots 也计入 `accuracy/count` 分母，导致 accuracy 被低估；现改为只统计 `mask` 内有效 boundary items。`_hash_from_signature()` 同步断兼容要求 `feature_signature.feature_config.target_chunk_s` 必填，避免旧 sequence feature signature 缺字段时静默复用默认值。
- 2026-06-04 命名决策并执行断兼容迁移：当前系统不再适合继续叫 `FusionVAD-JA`。它已经不是严格 speech/non-speech VAD，而是 `Qwen PTM + MFCC/energy bootstrap frame scores -> boundary candidate extraction -> Boundary Refiner scoring -> constrained planner -> ASR chunks` 的 speech-island boundary system。新概念名定为 **SpeechBoundary-JA**，backend key 为 `speech_boundary_ja`；active package 已从 `src/vad/fusionvad_ja/` / `tools/vad/fusionvad_ja/` 迁到 `src/boundary/ja/` / `tools/boundary/ja/`，不保留 `fusionvad_ja` alias、旧 cache 或旧路径兼容。
- 2026-06-04 破坏式重构：旧 `ASR_PRE_ASR_*` / R15-R23 规则 packer、旧 `ASR_CHUNK_PACK_*` 配置和旧 `vad-cache` 语义已从 active code、测试和 Web 可见配置中移除。当前主线是 `SpeechBoundary-JA frame probabilities -> boundary candidate extraction -> Boundary Refiner scoring/context budget -> constrained boundary planner -> ASR chunks`；cache 当前为 `boundary-cache v2`，signature 包含 SpeechBoundary-JA、Qwen feature/PTM、Boundary Refiner、candidate extractor、planner/context config 和 `BOUNDARY_FEATURE_FRAME_HOP_S`。
- 2026-06-04 SpeechBoundary-JA schema 断兼容迁移：feature cache / manifest / checkpoint / model layer 统一改为 `ptm`、`ptm_dim`、`ptm_proj`，不再沿用早期 `whisper_*` 命名；仓库内 v1.17、v1.19b、v1.21 小 checkpoint 已迁移 state_dict key。旧 feature cache 和旧实验 checkpoint 不做兼容迁移，重新生成。
- 2026-06-04 配置面清理：旧 `VAD_MIN_OFF` / `VAD_PAD` / `SEGMENT_*` 已从 active defaults、`.env.example`、boundary-cache signature 和 Web advanced 透传中移除；当前只保留语义明确的 `ASR_CHUNK_MIN_DURATION_S`（导出 wav chunk 最小时长）和 `ASR_CONTEXT_RESET_GAP_S`（滑动 ASR 文本上下文重置 gap）。
- 2026-06-04 审计发现并修复 chunk metadata 错位：`_extract_wav_chunks` 过滤过短 span 后，`_annotate_packed_chunks` 过去按输出位置 zip `PackedChunk`，会把 `vad_seg_count` / boundary reason/source/score 错贴到后一个 chunk；现在 chunk info 记录 `source_span_index`，annotation 按原始 span index 对齐，并补回归测试。
- 2026-06-04 timing 语义修正：Boundary / ASR chunk planning 采用秒级上下文参数，`BOUNDARY_FEATURE_FRAME_HOP_S` 只表示 VAD/frame-score 网格 fallback，默认 `0.02s`；字幕显示 / timing polish 单独按真实视频 FPS 计算 `frame_duration_s` 和 Netflix-style `2-frame gap`。主流程不再把视频 FPS 注入 Boundary 配置，aligned-cache 签名改为 v3，并在 `subtitle` 签名中记录 `video_fps` / `frame_gap_s`。
- 2026-06-04 BiLSTM 路线断兼容删除：旧 `AdditionFusion*BiLSTM`、v1.17/v1.19b/v1.21 endpoint / imitation checkpoint、addition / endpoint / imitation 训练导出 CLI、drop-gap imitation offline packer 和旧大测试文件已移出 active tree 到 `agents/rm/bilstm-removal/`。当前 `src/boundary/ja/` 模块只做 Qwen frozen feature + MFCC / energy bootstrap frame scoring；边界决策主线只走 `src/boundary/` 的 Boundary Refiner / Mamba2。
- 新主线模块结构：`src/boundary/features.py` 组装帧级特征，`candidates.py` 提取 gap midpoint / cut peak / low-score valley 候选，`refiner.py` 提供 BoundaryRefiner interface 和 bootstrap refiner，`backbones.py` 放 Windows-friendly Mamba2 research wrapper，`planner.py` 做 constrained planning，`cache.py` 负责 boundary-cache v2。`src/boundary/ja/` 放 JA 目标域 bootstrap scorer。`src/audio/chunk_packer.py` 只保留把 planner 输出和 learned context budget materialize 成 `PackedChunk` 的 ASR-facing 职责。
- 2026-06-04 Boundary Refiner backbone 入口收束到唯一实现路径 `transformers.Mamba2Model`：learned checkpoint schema 当前为 `boundary_refiner_v2`，runtime / CLI / cache signature / checkpoint payload 都使用该值。它直接对应 Hugging Face Transformers 的纯 PyTorch Mamba2 wrapper；`mamba2`、`torch_mamba2`、BiGRU、TCN、Transformer fallback 不再作为可选入口，避免训练和分发形成多套模型协议。
- 2026-06-04 Sequence feature schema authority 落地：`src/boundary/sequence_features.py` 统一提供 `frame_sequence_features_v1` 的 default config、feature dim、feature names、`feature_schema_hash` 和 train/runtime validation。`runtime_adapter=frame_sequence_v1` checkpoint 必须带 `feature_schema`、`feature_schema_hash` 和 `feature_signature`；主 pipeline 会用 SpeechBoundary-JA 导出的 PTM/MFCC frame windows 重新计算 names/hash，不匹配直接 fail-fast，不做旧 checkpoint alias 或静默回退。`PackedChunk` / boundary-cache / ASR chunk metadata 已写入 `boundary_decision_merge`、`boundary_merge_prob`、`boundary_split_prob`、`boundary_refine_delta_s` 和 `boundary_decision_source`，供后续 QC、forced alignment 和审计页追踪决策依据。
- 2026-06-07 runtime speaker / sidecar 断兼容清理：运行时人声聚类、声纹 sidecar、Web speaker 显示、speaker-aware cue merge、cue planner speaker score、speaker sidecar 工具/测试和低能量 pre-ASR drop 已从 active tree 移除或移到 `agents/rm/speaker-runtime-removal/`。字幕层只保留 Netflix-style timing/readability/fallback 风险、2-frame gap、显示时长和短 cue 合并这些观感核心；`speaker_proxy` / `speaker_turn` 不再进入 runtime 决策、用户配置或 Boundary Refiner synthetic metadata。
- 2026-06-07 翻译缓存分层：Grok / 官方文档检索确认 provider prompt cache 依赖“稳定长前缀 + 变量放后”的 exact-prefix 机制，适合降低 API token/latency，但不解决本地复跑 cache miss。当前本地翻译改为 `translation_cache.jsonl` 精确 batch cache + `translation_cache.memory.jsonl` 文本级 translation memory：一级仍绑定 cue timing / batch / prompt signature 用于 crash resume；二级绑定 normalized JA text、target language、normalized glossary / auto glossary、character reference、prompt version 和 model family，允许同一任务 Boundary / cue timing 调整后复用译文。低信息或单字符循环文本不写入 memory，避免目标域呻吟/短促发声被过度复用。timings 中区分 `translation_cache_hit`、`translation_memory_hit`、`translation_memory_hit_count` 和 provider prompt-cache usage，避免把三种缓存混为一谈。参考来源：OpenAI Prompt Caching、Claude Prompt Caching、DelTA 多级翻译记忆、2024 BUCC 字幕自适应翻译 fuzzy-match 研究。
- 2026-06-07 翻译链路审计修复：短片 / 审计片段过去会走 `single_request_full_context`，但该 path 没有接 `cache_path`，因此不会命中本地 batch cache / translation memory，也不会写入二级 memory；现在 single-request 与 batched path 共用缓存语义。另修 prompt user message 硬编码“中文字幕”的问题，改为使用 `TARGET_LANG`；自动全片 glossary cache 从固定 `translation_global_glossary.json` 改为 `translation_global_glossary.<source_hash>.json`，避免共享 translation cache 路径时串用上一部片的自动术语表。
- 2026-06-07 项目级运行目录断兼容从 `temp/` 改为 `tmp/`：`JOB_TEMP_DIR=./tmp/jobs`，ASR chunk/checkpoint 为 `./tmp/chunks`，boundary cache 为 `./tmp/cache/boundary`，torch / HF 运行缓存为 `./tmp/cache/torch` 与 `./tmp/cache/hf`。`agents/temp/` 继续只用于研究脚本和审计中间产物，不纳入用户运行缓存结构。旧 `temp/` 不做 alias 或迁移逻辑，避免长期维护两套 runtime root。
- 2026-06-06 Galgame 100k 本地源池落地：先用 streaming duration scan 读取 `litagin/Galgame_Speech_ASR_16kHz` 前 100000 条，只解析 OGG header，不保存音频，确认分布为 mean/p50/p75/p90/p95/p99/max `5.136/4.504/6.937/9.560/11.353/15.187/28.981s`，`>=5s/9s/12s/15s/20s` 比例 `0.441/0.123/0.039/0.011/0.001`，与 dataset card 全量平均约 `5.145s` 一致。随后按用户要求把这 100000 条保留在本地，输出 `datasets/train/boundary-sources/galgame-asr-100k-ogg/`，保存原始压缩 OGG + TXT + `manifest.jsonl`，不转 WAV。校验：manifest `100000` 行、OGG `100000` 个、TXT `100000` 个、errors `0`，summary 记录实际 OGG 字节 `2.313 GiB`，文件系统占用约 `3.0G`（10 万小文件有块占用）。这批数据作为后续 Boundary Refiner / synthetic speech-island dataset 的本地源池，避免每轮构造都重新流式下载 HF。基于该分布新增可复算公式与工具 `tools/boundary/recommend_timing_params.py`：`target_core=round_0.1(clamp(p50/speedup,2.0,3.5))`、`max_core=floor_0.5(clamp(p80/speedup,target+1.0,5.5))`、`target_padding=round_0.1(clamp((p90-max_core)/2,1.0,2.0))`、`max_padded=floor_0.5(min(p90,max_core+2*padding,9.0))`、`min_chunk=round_0.05(clamp(p5/speedup*0.60,0.25,0.50))`。当前 `target_domain_speedup=1.5` 推导出 `target/core/padded/min/padding = 3.0/5.0/9.0/0.4/2.0s`，与现行默认一致；后续换成 anime 混合源或重新采样时直接换 summary 复算。
- 2026-06-05 fallback 时间轴窗口修正：soft-candidate DP v2 已把 20s+ 粗 chunk 消掉，但剩余长 fallback 多数是 `core_duration_s≈8-9s` 被 2s 左右 ASR padding 显示成 `12-13s`。主流程现在继续给 ASR 输入 padded chunk 保留识别上下文，但在 forced aligner 失败、走比例/VAD fallback 时间戳时，只在 speech core 窗口插值；`alignment_fallback_start_s/end_s/source` 会随 text_result 进入 `LocalAsrBackend.finalize_text_results()`，retry / sentinel fallback 分支也使用同一窗口。完整 GPU 闭环实测确认：padded fallback p50/p90/max 仍是 `8.73/12.60/13.10s`，但实际 fallback 时间轴窗口 p50/p90/max 已降到 `6.10/8.76/9.10s`，safe ratio `0.732`，`>10s` fallback 消失。
- 2026-06-05 JAV 短字幕 core 策略收紧：公开检索未找到可靠 JAV 单句字幕时长统计；可用参考是 Netflix timing / Japanese 读速、OOONA、ATA、Karamitroglou 等通用字幕规范，它们都更支持短 cue、贴近 in-time 和长音频拆分。本地 `agents/audits/fallback-window-risk-audit-video/manual_fallback_window_risk_labels.jsonl` 40 条 `8.75-9.10s` fallback-window 人工审计中，`needs_split=35/40`、`multiple_islands=11/40`、`timing_end_early=11/40`。因此 `9s` 不再作为 subtitle/fallback speech-core 默认上限，只保留为 ASR padded context 上限；新默认是 `BOUNDARY_PLANNER_TARGET_CHUNK_S=3.0`、`BOUNDARY_PLANNER_MAX_CORE_CHUNK_S=5.0`、`BOUNDARY_PLANNER_MAX_PADDED_CHUNK_S=9.0`。旧 `BOUNDARY_PLANNER_MAX_CHUNK_S` 断兼容删除，不保留 alias。
- 2026-06-05 匿名样片 A 短 chunk 批量与显存监控：基于已生成的 2459 个 ASR chunks 做阶段 benchmark。0.6B ASR-only 256 chunk：batch `32` 用时 `32.25s`、峰值 `4272 MiB`；batch `48` 用时 `29.62s`、峰值 `5139 MiB`；batch `64` 曾未监控跑通 `29.50s`，但本次 0.2s 持续 `nvidia-smi` 采样在 `6200 MiB` 超过 6GB 目标并被中断；batch `128` 之前已因 `CUDA driver error: device not ready` 失败。1.7B ASR：batch `48` 峰值 `7178 MiB`，batch `24` 峰值 `6327 MiB`，batch `16` 峰值 `6119 MiB`（当时空闲底噪约 `1.1GB`）；用户当时决定 1.7B 默认保守用 `12`。aligner-only 全量 2459 chunk：batch `16` 用时 `83.72s`、峰值 `3259 MiB`；batch `32` 用时 `80.78s`、峰值 `3864 MiB`；batch `48` 用时 `75.71s`、峰值 `4349 MiB`，均无错误。该条保留为当时 benchmark 记录；当前默认档位以后续 2026-06-08 的 6GB / 8GB 配置记录为准。
- 2026-06-06 ASR context 后处理断兼容删除：匿名样片 A 审计发现带人名 `ASR_CONTEXT` 时，旧 fragment-level prompt/context leak 规则会把真实自我介绍从最终 cue 中删掉，只剩问候句；但词级 `words` 已有对应时间轴。结论是该规则在 JAV / Galgame 目标域误伤成本高、价值低。主流程已删除 `context leak` QC reason、相似度阈值、fragment 删除函数和 Web/env advanced 前缀；ASR prompt context 只作为识别提示词，不再驱动最终字幕删除或 QC reject。字幕层继续依赖空文本、纯标点、低信息/重复、信号质量、fallback 和人工审计等更直接的信号。
- 2026-06-06 F0/gender 与最终文本突变清理：旧 F0/gender route、`run_asr_alignment_f0` 历史命名、Web `show_gender` 可见项和相关测试已从 active tree 移除。ASR QC `reject_count` / `review_uncertain`、重复循环 `suggested_text` 和低信息 profile 均保持 review-only；审计页标签从“采用去重建议”改为“重复建议需复核”，点击不会再自动写 `manual_text`。字幕 merge 不再去重 overlap，保留目标域真实重复语气词、呻吟和短促发声。
- Windows 分发约束：默认路线不依赖 Linux-only `mamba-ssm`、Triton 或自定义 CUDA kernel。`src/boundary/backbones.py` 只提供 Hugging Face Transformers `Mamba2Model` 的纯 PyTorch wrapper 作为研究 backbone。
- Unified Joint Model 路线进入 backlog，不进入当前重构：未来可考虑在 Qwen3-ASR decoder 中加入 `<boundary>` / `<dramatic_pause>` / `<sentence_end>` 等 token，做 joint segmentation + transcription 或 Samba-ASR 类长上下文模型。但这需要重新准备 boundary supervision、ASR SFT/RL/DPO 和云端 GPU 训练，当前维护成本和训练成本高，不作为默认路线。
- active backend key 已改为 `speech_boundary_ja`，旧 `fusionvad_ja` key / package / path 不做兼容。
- 默认 ASR backend key 已切到 Hugging Face repo ID 本身：`jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`；可选 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`。短 key 不再进入主线，避免 Web/API/cache/download 出现两套命名。
- SpeechBoundary-JA frozen feature 默认使用 `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`，不再默认下载 base 0.6B。
- Boundary Refiner 默认 learned checkpoint 固定为 `src/boundary/checkpoints/boundary_refiner.pt` 并随源码分发；旧 `src/vad` checkpoint 只作为历史产物保存在本地回收区。
- 破坏式维护重构：`src/whisper/` 改为 `src/asr/`，`tools/fusionvad_ja/` 按职责拆到 `tools/asr/`、`tools/vad/`、`tools/subtitles/`、`tools/audits/`；Whisper/WhisperSeg/TEN/Silero/FusionLite 当前主线代码和测试移入 `agents/rm/obsolete-mainline-cleanup-20260603/`。
- 当前主线不再是 high-recall proposal VAD，也不再是固定 gap packer，而是 SpeechBoundary-JA / Boundary Refiner 驱动的 speech-island boundary pipeline：ASR 前 chunk 要尽量接近一句台词，避免长连续 chunk、内部 gap、多 speech island 和非语音多送诱发 ASR 空输出、非语音幻觉和 forced aligner sentinel。
- 边界优先级：`start` 略高于 `end`，但两者都要进 gate；允许为了切准 speech island 牺牲少量 frame recall，但不能漏掉完整台词 island。
- 下一步应把 v1.20-v1.23 的经验收敛到 learned Boundary Refiner：显式优化 start/end error、fallback chunk duration、gap crossing、单 chunk 台词数和 ASR/aligner QC reward；recall 继续作为 guardrail，而不是唯一主目标。
- 现行 `tools/` 已按职责重构：`tools/asr/qwen/` 放 Qwen SFT，`tools/asr/diagnostics/` 放 ASR/alignment 诊断，`tools/boundary/` 放 Boundary Refiner 数据和训练，`tools/boundary/ja/` 放 SpeechBoundary-JA 训练评测，`tools/subtitles/` 放字幕 postprocess / cue planner / 审计校准，`tools/audits/` 放审计页与人工审计工具。旧历史段落里的 `tools/vad/fusionvad_ja/...` 路径保留为当时记录。

### 历史计划快照：SpeechBoundary-JA v2/v3

以下计划是 v5 delta-only 之前的路线快照，已被当前 `frame-sequence Mamba2 start/end delta -> core planner -> ASR` 主线取代；保留是为了追踪断兼容重构的来源。

1. 断兼容改名已完成：`src/vad/fusionvad_ja/` -> `src/boundary/ja/`，`tools/vad/fusionvad_ja/` -> `tools/boundary/ja/`，配置前缀 `FUSIONVAD_JA_` -> `SPEECH_BOUNDARY_JA_`，backend key `fusionvad_ja` -> `speech_boundary_ja`。旧 key / 旧 path / 旧 cache 不做 alias。
2. 数据格式升级：删除 gap-only / BiLSTM / endpoint-head 训练格式，把 Galgame 与 `joujiboi/japanese-anime-speech-v2` clean speech islands 重新生成 sequence dataset。每条样本包含多 island、touching speech、short/long gap、real negative gap、BGM/noise、轻量 overlap、source/utterance switch。
3. Learned refiner：只保留 `transformers.Mamba2Model` backbone。输入升级为连续窗口序列：Qwen PTM、MFCC、energy、speech_prob、cut_prob、candidate metadata。输出 split / merge / refine score 和可选 boundary offset。
4. Planner 接入：`pack_speech_segments()` 只 materialize planner 输出。planner 负责 start/end 权重、fallback-safe duration、gap-crossing penalty、最小/最大 chunk 约束和 ASR-facing span 输出。
5. 验收：synthetic exact truth 与匿名样片 A 双闭环。主 gate 是 start p90/p95、fallback chunk duration、long/gap-crossing chunk、ASR empty / hallucination、forced/partial 比例；chunk 数只作为成本指标。
6. 后续强化：supervised 稳定后再加入 preliminary ASR text、token confidence、local CER、aligner sentinel、fallback duration 和 QC reject 做 dense reward / DPO / RL。Unified Joint Model 继续放 backlog，等 SpeechBoundary-JA 能产出稳定 pseudo boundary labels 后再评估。

---

---

## Qwen3-ASR SFT 路线（早期云端训练记录）

### 1.7B full SFT

目标：让 ASR 能覆盖目标域中的对白、喘息、呻吟和短促拟声，降低“VAD 送进去了但 ASR 不认”的问题。

云端训练结论：

- 数据：`litagin/Galgame_Speech_ASR_16kHz` full ASR-only。
- 初始学习率：`2e-5`。
- effective batch：`128`。
- RTX 5090 32GB 曾用于 0.6B full SFT；RTX PRO 6000 96GB 用于 1.7B full SFT。
- 最终模型上传到 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`。

### 0.6B full SFT

目标：

- 替代 ja-whisper-anime 做更快的日语 ASR probe。
- 作为 FusionVAD-JA frozen feature extractor。
- 降低分发时的模型数量和空间成本。

结果：

- Galgame 16 clip direct probe 中，CER 从 base `0.2348` 降到 full `0.1288`。
- RTF 约 `0.232`。
- 最终模型上传到 `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`。
- 该模型成为 v1.13+ FusionVAD-JA 默认 frozen feature。

### 云端训练坑

- 数据集几十 GB，用户本地上传到云服务器太慢；更合理方式是在云端脚本直连 Hugging Face 下载并生成训练集。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 可缓解显存碎片，但不能弥补真实峰值不足。
- 5090 32GB 上 0.6B `batch_size=16`、`grad_acc=8` 曾在 step 36 OOM；稳定配置为 `batch_size=8`、`grad_acc=16`、effective batch `128`。
- 大 batch feature cache 在 WSL2 8GB RAM 下可能被系统 kill，没有 Python traceback；需要查内存和系统日志，不能只看显存。

---
