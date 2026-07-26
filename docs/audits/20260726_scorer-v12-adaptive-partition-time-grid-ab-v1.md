# Scorer v12 adaptive partition / time-grid A/B

## 结论

本轮只比较 Teacher 标注策略，不生成训练 canonical，也不改 production registry。完整 source 主线继续使用严格 `MM:SS.mmm`；adaptive A/B 将每次可听上下文限制在最多 20 秒，使用 10ms 局部坐标，再统一编译到 Scorer 的 20ms 帧。

Adaptive arm 在固定 held-out12 上明显比 broad-v3 碎：A 的 vocal/non-vocal run 为 `27/17`，B 为 `147/142`，两臂共有 `157` 个差异 run、`8001` 个差异帧。共享边界量化不会额外制造空隙或重叠，因此这些 run 主要来自 Teacher 的逐窗分区判断。人工 A/B gate 仍未完成，B 不能进入训练。

## 时间合同

- PCM 保持 16kHz，单 sample 为 `0.0625ms`；这只是波形采样间隔，不是多模态 Teacher 可承诺的边界精度。
- 完整 source Teacher wire 继续使用 `omni_audio_timestamp_mmss_mmm_v1`，只接受 `MM:SS.mmm` 字符串。
- Adaptive 局部窗口最长 20 秒，wire 只接受 10ms 对齐的 JSON 数字；任何更细坐标、负值、越界、倒序或坍缩到不足一个 20ms frame 的 span 都 fail-closed。
- Canonical Scorer 始终是 20ms frame。相邻 spans 的切点只量化一次并由两侧共享；vocal start 向前、vocal end 向后保护，保证完整覆盖、无缝、无重叠。
- 合同 ID 为 `scorer_v12_10ms_wire_20ms_frame_v1`。

## Adaptive rolling

- 每个请求窗口最多 `1000 frames / 20s`。
- 目标提交点为 `750 frames / 15s`。
- 优先在 `500–900 frames` 范围内选择至少 `20 frames / 0.4s` 的 definite non-vocal 内部切点；找不到时按目标点提交并保留 5 秒前瞻。
- 每次提交后从同一个 source 绝对 frame 重新建立局部零点，最终 source partition 仍由共享 20ms frame 边界组成。
- window checkpoint 原子落盘，可从已完成窗口恢复；聚合 preaudit 可由 checkpoint 确定性重建。

## Provider 与配额

原生 Gemini 尝试在本地东八区 16:00 观察边界后仍出现跨 Key daily-429，因此该时刻不能作为真实配额恢复证据。未完成 native 目录只保留诊断 checkpoint，不生成 source-level Teacher 标签。

当前保护策略：

- 原生 Gemini provider 全局最多 2 worker，不随 Key 数增长。
- Scorer adaptive rolling 默认 `--workers=1`。
- Key 数只增加配额轮换槽；RPD 使用每 Key 最近 24 小时请求事件的保守滚动账本。
- 明确 daily-429 为对应 Key 写入至少 24 小时冷却；东八区 16:00 仅作为 advisory 观察字段。
- 本轮完整 A/B 显式改用 OpenRouter，模型为 `google/gemini-3.6-flash`、reasoning=`medium`、`max_tokens=8192`，不发送 temperature/top-p/top-k。

## 固定证据

- source manifest SHA: `84a4bc420ad87fd1f4e4c8aa70ad403ca1e05b0b25004a2f0e2d223dcc63ed14`
- held-out selection manifest SHA: `9157d38d8cceecee3e3bbdee62e31aec6f24910f91a9f3959c82849f78931b2d`
- adaptive preaudit SHA: `26b63403780ea6352e5933dc5aa1d6ebe83e8ecb404cd1da2dacda530524b489`
- A/B audit manifest SHA: `121d1a537a8d22f206a552fbfb4d9571b1bddfa213c643c4674723c09bc49017`
- adaptive 请求数: `67`
- source 数: `12`
- `training_manifest_allowed=false`

审计页：`http://127.0.0.1:8080/agents/audits/20260726_172340_scorer-v12-broad-v3-vs-adaptive-partition-v3-heldout12/`

浏览器 QA 已确认页面完整渲染 12 条 source；点击 `00:00.000–00:02.100` 后只播放对应 WAV 区间，到终点自动停止并回到起点，控制台无 warning/error。
