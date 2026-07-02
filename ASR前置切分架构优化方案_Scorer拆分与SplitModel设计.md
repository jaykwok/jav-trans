# ASR 前置切分架构优化方案：Scorer 拆分与专用 Split Model 设计

## 1. 背景与问题

当前 ASR 前置流水线中，Scorer 同时承担两个任务：

1. `speech island detection`：判断哪里有人类语义语音；
2. `split boundary detection`：判断语音内部哪里应该切分成 ASR chunk。

现有问题是：音频 chunk 的整体边界切得较好，但在完整一句话内部的小停顿、喘息、短静音、语气停顿等位置，模型有时会误判为 split boundary，导致：

- chunk 数量过多；
- 一句完整字幕被拆成两句；
- ASR 上下文变短；
- 字幕翻译更容易断裂、缺主语、缺上下文；
- 后续 subtitle layout 和翻译阶段处理难度上升。

这个问题的本质不是单纯的阈值问题，而是 **speech detection 与 semantic split decision 的任务目标冲突**。

---

## 2. 核心结论

建议果断拆分当前 Scorer：

```text
原 Scorer:
  speech_prob + split_boundary_prob

拆分后:
  SpeechIslandScorer:
    只负责 speech / non-speech island 检测

  Dedicated Split Model:
    专门负责判断 speech core 内部候选点是否应该切分
```

一句话概括：

```text
Scorer 负责“哪里有人说话”；
Split Model 负责“这里该不该断句”；
Refiner 负责“边界落在哪里”；
CueQC 负责“这段值不值得送 ASR”。
```

---

## 3. 推荐的新流水线

建议将 ASR 前置流水线调整为：

```text
Audio
→ Shared Feature Extractor
   Qwen-ASR PTM + MFCC / log-mel / energy

→ ① SpeechIslandScorer
   只做 speech / non-speech
   目标：高召回找出语音岛，不负责句内切分

→ ② Outer Edge Refiner
   只修 speech island 的 start/end
   目标：得到干净 speech core
   不判断中间是否切

→ ③ Dedicated Split Model
   只负责 speech core 内部是否切分
   输出 cut / continue / unsure
   目标：判断“这里是不是适合变成两个 ASR chunk”

→ ④ Cut Edge Snapper / Cut Refiner
   只对已确认的 cut 找精确落点
   目标：切在 gap / valley / 低语音概率处
   不再决定切不切

→ ⑤ Pre-ASR CueQC
   只判断最终 chunk keep/drop
   目标：无语义片段丢弃，有语义片段送 ASR

→ ASR
→ Subtitle Layout / Translation
```

关键原则：

```text
split_boundary_prob 不再等于最终切分决策；
声学候选点只负责召回；
最终切不切由专门 Split Model 决定。
```

---

## 4. 为什么应该拆分 Scorer

### 4.1 speech 和 split 的目标冲突

SpeechIslandScorer 的目标应该是：

```text
宁可多包一点，也不要漏语音。
```

Dedicated Split Model 的目标应该是：

```text
宁可少切一点，也不要把完整句子切碎。
```

这两个目标天然不同。
speech detection 是高召回任务；split decision 是保守决策任务。

如果继续放在一个双头 Scorer 中，split head 容易过度依赖声学变化点，例如：

- 短静音；
- breath；
- moan；
- laughter；
- energy valley；
- speech probability valley；
- BGM 或噪声下的局部低能量点。

但这些声学变化点不一定是语义边界。

---

### 4.2 split 不是纯声学边界任务

真正应该问的问题不是：

```text
这里有没有声学边界？
```

而是：

```text
切开之后，左右两边是否都适合作为独立 ASR / 字幕 / 翻译单元？
```

一句话内部的小停顿可以有明显声学边界，但仍然不应该切开。

因此，split model 应该从“边界检测模型”升级为“语义完整单元切分模型”。

---

### 4.3 Refiner 与 CueQC 不适合解决过切

Refiner 的职责是修边界，不应该撤销中间切点。
CueQC 的职责是 keep/drop，不应该负责合并两个都包含语义的 chunk。

例如一句话被误切成 A、B 两段：

```text
A：有语义 → CueQC keep
B：有语义 → CueQC keep
```

CueQC 会保留两段，但翻译上下文已经被破坏。
所以合并/不切必须在 Split Model 阶段解决，不能指望 CueQC 后补。

---

## 5. Dedicated Split Model 设计

### 5.1 模型定位

Dedicated Split Model 不建议做纯 frame-level 模型，而建议做 **candidate-level model**。

先由轻量规则或 proposal head 生成候选点：

```text
- speech_prob valley
- energy valley
- split-like acoustic peak
- 长静音 gap
- speech island 内部低 active_ratio 区域
```

然后每个候选点交给 Split Model 判断：

```json
{
  "label": "cut | continue | unsure",
  "p_cut": 0.0,
  "p_continue": 0.0,
  "p_unsure": 0.0
}
```

### 5.2 输入窗口

每个候选点取局部上下文：

```text
left_context:  候选点左侧 1.2s ~ 2.0s
gap_context:   候选点附近 ±0.3s
right_context: 候选点右侧 1.2s ~ 2.0s
```

模型实际看到的是：

```text
[left semantic/acoustic context] + [candidate pause] + [right semantic/acoustic context]
```

这比单独看某一帧的 split probability 更适合判断“该不该切”。

---

## 6. Split Model 特征建议

建议输入以下特征：

### 6.1 PTM bins

使用 Qwen-ASR PTM 特征，按左右上下文分别做 bin pooling：

```text
left PTM bins:  8 bins × 128/256 dim
right PTM bins: 8 bins × 128/256 dim
gap PTM bins:   可选 2~4 bins
```

不建议一开始只截 64 维。
如果显存或样本量允许，建议先试 128 或 256 维。

### 6.2 声学特征

保留更直接的声学边界信息：

```text
- MFCC
- log-mel
- energy
- zero crossing rate，可选
- speech active ratio
```

### 6.3 Scorer 侧特征

复用 SpeechIslandScorer 的输出分布：

```text
- speech_prob mean / max / p90 / p10 / std
- candidate 附近 speech_prob valley depth
- valley width
- left/right active ratio
- island duration
```

### 6.4 候选点特征

```text
- candidate score
- candidate prominence
- candidate rank
- valley snap distance
- candidate 到 left/right edge 的距离
- gap duration
- left chunk duration
- right chunk duration
```

### 6.5 上下文完整性特征

```text
- left 是否疑似完整语义片段
- right 是否疑似完整语义片段
- merge 后是否更像完整一句话
- 切开后是否出现过短 fragment
```

这些可以先作为模型隐式学习目标，也可以通过辅助 head 监督。

---

## 7. Split Model 架构建议

可以从轻量结构开始：

```text
left PTM bins  → small Mamba / Transformer encoder
right PTM bins → small Mamba / Transformer encoder
gap features   → MLP
scalar features → MLP

concat(left_repr, right_repr, gap_repr, scalar_repr)
→ interaction block
→ classifier
→ cut / continue / unsure
```

可选输出：

```json
{
  "p_cut": 0.0,
  "p_continue": 0.0,
  "p_unsure": 0.0,
  "left_complete": 0.0,
  "right_complete": 0.0,
  "merged_better": 0.0
}
```

建议优先做三分类：

```text
cut / continue / unsure
```

而不是二分类。
因为真实音频中边界不确定的情况很多，强行二分类会引入噪声标签。

---

## 8. 数据构造方案

你当前的数据非常适合训练这个模型。

已有数据特点：

```text
Galgame / anime clip 已经是切得刚刚好的完整句音频；
虽然没有时间轴，但每个音频文件本身就代表一个完整字幕句。
```

因此可以构造 cut 和 continue 两类高质量监督。

---

### 8.1 cut 正样本

把两个不同完整句子拼接：

```text
clip_A + gap + clip_B
```

中间 gap 是正样本：

```json
{
  "label": "cut",
  "reason": "两个完整句子之间的边界"
}
```

gap 分布不要只用长静音，要覆盖：

```text
0ms
40ms
100ms
300ms
800ms
1.5s
```

这样可以避免模型学成：

```text
长 gap = cut
短 gap = continue
```

真实对白中，两个句子之间也可能几乎贴接。

---

### 8.2 continue 负样本

从同一个完整 clip 内部找低能量 valley，然后构造：

```text
clip_left + fake_pause + clip_right
```

中间 fake_pause 是负样本：

```json
{
  "label": "continue",
  "reason": "同一句话内部的小停顿"
}
```

fake_pause 可以包括：

```text
- 静音 200ms ~ 1200ms
- 呼吸声
- 喘息声
- 轻呻吟
- 笑声
- 环境噪声
- 低音量 BGM
- 拖音后的短停顿
```

这一步最关键。
它会明确告诉模型：

```text
同一句话内部即使有停顿，也不能切。
```

这正是当前 Scorer v7 缺失的监督信号。

---

### 8.3 hard negative 样本

应重点构造“像边界但不该切”的样本：

```text
- energy valley 很深，但左右仍是一句话
- speech_prob valley 明显，但语义连续
- 中间有 breath / moan，但不是句界
- 左右分开后其中一边太短
- 左右单独 ASR 可能不完整，合并后更完整
```

这些 hard negative 对降低过切最有价值。

---

### 8.4 真实 workflow 样本

从真实视频 workflow 中导出以下候选点，交给 Omni 或人工标注：

```text
1. split score 高，但左右 chunk 都很短
2. gap 在 0.3s ~ 1.5s
3. left/right 其中一边疑似 fragment
4. micro resolver 没合并但人工听起来像一句话
5. 翻译效果明显变差的 chunk pair
6. p_cut 与 p_continue 接近的边界
```

这些是真实域的高价值样本，适合作主动学习集。

---

## 9. Qwen Omni 标注任务设计

你已经验证 Omni 做 Pre-ASR keep/drop 标注很准。
下一步可以让 Omni 承接更关键的 split 标注。

### 9.1 标注输入

把候选点左右各 2 秒裁出来，候选点固定放在音频正中间：

```text
[left 2s] + [right 2s]
候选切点永远在 2.000s
```

### 9.2 推荐 Prompt

```text
你是 ASR chunk 切分质量标注器。音频中间位置 2.000 秒是候选切分点。
请判断这里是否适合把音频切成两个 ASR chunk。

标签定义：
- cut：候选点左右像两个独立语义片段，切开后各自适合 ASR/字幕。
- continue：候选点只是同一句话内部停顿、喘息、拖音、犹豫、短静音，不应切开。
- unsure：边界不确定，或左右语义都不完整，或音频太模糊。

注意：
- 不要因为有静音、喘息、呻吟、呼吸声就判断为 cut。
- 判断重点是：切开后左右两段是否都适合作为独立 ASR chunk。
- 如果合在一起更像完整一句话，应标 continue。

只输出 JSON：
{
  "label": "cut|continue|unsure",
  "confidence": 0.0-1.0,
  "left_complete": true|false,
  "right_complete": true|false,
  "merged_better": true|false,
  "flags": ["short_pause", "breath", "moan", "laughter", "same_sentence", "topic_shift", "speaker_change", "low_snr", "music"],
  "reason": "简短中文理由"
}
```

### 9.3 标签使用策略

建议映射为：

```text
cut + confidence >= 0.80      → definite_cut
continue + confidence >= 0.80 → definite_continue
unsure 或低置信              → ignore / unsure
```

在早期阶段，宁可少用低置信标签，也不要污染 split decision。

---

## 10. Cut Edge Snapper / Cut Refiner

Split Model 只决定“切不切”，不负责“切在哪里”。

建议新增或保留一个 edge-only 模块：

```text
if p_cut >= threshold:
    在候选点附近 ±250ms / ±400ms 搜索最优落点
    优先选择：
      - speech_prob 最低点
      - energy 最低点
      - gap 中心
      - split acoustic evidence 最强点
else:
    不切，最多记录为 weak_cut
```

也就是说：

```text
语义/完整性模型决定 yes/no；
声学 edge-only 模块决定 timestamp。
```

这样职责最清晰。

---

## 11. Runtime 策略建议

建议使用保守切分策略：

```text
1. SpeechIslandScorer 高召回输出 speech islands
2. OuterEdgeRefiner 修 island start/end，得到 speech core
3. 如果 speech core duration <= 6s：
      默认不切
      除非 SplitModel p_cut 极高
4. 如果 speech core duration > 6s：
      生成候选 cut
      SplitModel 逐个判断 cut / continue / unsure
5. 对 p_cut 高的候选做 NMS
6. Cut Edge Snapper 找精确落点
7. 生成最终 ASR chunks
8. Pre-ASR CueQC 做 keep/drop
9. 送 ASR
```

推荐初始阈值：

```text
short_core_cut_threshold = 0.90
normal_cut_threshold = 0.75
continue_threshold = 0.60
unsure_policy = prefer_continue
min_chunk_after_split = 1.0s ~ 1.2s
```

核心原则：

```text
短 chunk 不轻易切；
长 chunk 才积极找 cut；
不确定时偏向不切。
```

---

## 12. 评估指标建议

不要只看 split F1。
应该新增更贴近真实目标的指标：

```text
1. over_split_rate
   一条参考完整句被切成多个 chunk 的比例

2. avg_chunks_per_reference_sentence
   理想值接近 1.0

3. false_primary_cut_rate
   不该切却进入 primary cut 的比例

4. continue_recall
   句内停顿被正确判为 continue 的比例

5. cut_precision
   被切开的点中，真正应该切的比例

6. semantic_unit_quality
   每个 chunk 是否语义完整，是否适合作 ASR 和字幕翻译输入

7. translation_damage_rate
   因过切导致翻译断裂、缺主语、上下文丢失的比例

8. ASR cost
   chunk 数量、总 ASR 调用次数、平均 chunk duration
```

对当前项目来说：

```text
false cut 的代价 > missed cut 的代价
```

漏切最多导致 chunk 偏长；
误切会直接破坏字幕翻译上下文。

---

## 13. 迁移计划

### Phase 1：先止血

目标：减少明显过切。

改动：

```text
- Scorer 只保留 speech island 输出
- 暂时关闭 primary split 或降低 split 权重
- 提高 min_chunk_after_split
- 对 <= 6s speech core 默认不切
- 对长 speech core 使用保守规则切分
```

产出：

```text
SpeechIslandScorer v8-lite
runtime conservative split policy
```

---

### Phase 2：构造 Split Model 数据集

目标：建立 cut / continue / unsure 训练集。

数据来源：

```text
- Galgame / anime 完整句 clip
- 合成 cut 正样本
- 同句内部 fake pause continue 负样本
- 真实 workflow hard cases
- Omni 弱标注候选切点
```

产出：

```text
semantic_split_verifier_dataset_v1
omni_split_labels_v1
manual_override_split_labels.jsonl
```

---

### Phase 3：训练 Dedicated Split Model

目标：让模型学会区分“句界”和“句内停顿”。

模型：

```text
SemanticSplitVerifier v1
candidate-level cut / continue / unsure classifier
```

重点优化：

```text
- continue recall
- cut precision
- over_split_rate
- avg chunks per reference sentence
```

---

### Phase 4：接入 Cut Edge Snapper

目标：将 “切不切” 与 “切在哪里” 解耦。

流程：

```text
SplitModel 决定 cut
CutEdgeSnapper 在 ±250ms / ±400ms 内找精确落点
```

产出：

```text
final_cut_points
weak_cut_candidates
split_audit_metadata
```

---

### Phase 5：全链路 smoke + 人工审计

目标：验证真实字幕质量。

审计内容：

```text
- chunk 数量是否下降
- 单句被拆碎的比例是否下降
- ASR 是否更稳定
- 翻译是否更自然
- 是否出现明显漏切导致 chunk 过长
- CueQC 是否仍能稳定 keep/drop
```



---

## 14. 最终建议

建议将当前 Scorer v7 果断拆分为：

```text
SpeechIslandScorer v8
SemanticSplitVerifier v1
```

并将整体 ASR 前置架构调整为：

```text
Speech island 检测
→ edge-only 修外边界
→ 专门 Split Model 判断 cut / continue / unsure
→ cut edge snapper 精确落点
→ Pre-ASR QC keep/drop
→ ASR
```

最终目标不是“找到所有声学停顿”，而是：

```text
找到最适合作为 ASR、字幕和翻译输入的语义完整 chunk。
```

这会比继续在 Scorer v7 里调 split head、NMS、micro resolver 更可控，也更符合当前过切问题的本质。
