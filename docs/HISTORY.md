# jav-trans History

本文件记录实验过程、idea 来源、调试坑、失败路线、指标和参考来源。README 只保留新用户使用说明、当前工作流和当前状态。

公开记录统一使用“匿名样片 A/B…”和 `sample-a/sample-b` 等占位符，不写真实片名、番号、视频 stem 或本地片源路径。

---

## 当前有效状态

- **识别与时间轴**：正式流程使用 Qwen3-ASR 1.7B；默认 CTC 对齐头为 `ctc_aligner_jav_vocalisation_v2.pt`，字幕按实测字词时间生成，不使用比例时间伪造切点。
- **字幕布局**：当前引擎为 `measured_safe_boundary_dp_v3_1`。20 个日文源字符与 7 秒是软目标；优先在句末、强停顿、分句标点和可靠词间隙处拆分，没有安全点时保留较长 cue。
- **字幕过滤与质检**：仅过滤连续出现且整条可拆解为非语义人声的 cue；含可辨识词语的条目保留。质量报告在 Web 中直接展示时间轴、布局、翻译和格式风险。
- **API 翻译**：默认组合为 `https://openrouter.ai/api/v1` + `deepseek/deepseek-v4-flash`。推理强度只保留 `none / low / high`，默认 `low`；复译使用首轮档位且下限为 `low`。Base URL 决定供应商兼容行为，模型 id 必须与端点配套。
- **翻译成本与一致性**：`TRANSLATION_BATCH_SIZE` 默认 200，和并发数解耦；术语提取同时完成全片前缀预热。用户术语表优先于自动术语，源文回显、假名残留、术语漏用和长度异常会触发选择性复译。
- **本地翻译**：唯一默认模型为 Hy-MT2-7B Q4_K_M，由 llama.cpp 托管；适合隐私优先和零 API 成本的草稿，不使用 API 路径的术语表、角色参考或全片上下文。
- **已知边界**：ASR 的自动 batch 与缓存命中会改变批组成，严格对比时应固定 `ASR_BATCH_SIZE`。默认 CTC 头仍保留少量高 blank 风险样例，质量报告与人工复查不能省略。

## 2026-08-24

### 默认端点改为 OpenRouter，供应商判据从模型名换成主机名

改默认 Base URL 只是一行，跟着它走的四件事才是这次的内容。

**模型 id 属于端点，不属于模型。** OpenRouter 只认 `author/slug`（裸 id 返回 400 `model_not_found`），DeepSeek 官方 API 反过来只认裸 id。默认值原本是 `deepseek-v4-flash`，配上新的默认 Base URL 就是一组开箱即挂的组合——本机 `.env` 早就填了自己的模型，所以这条只会砸在全新安装上。默认模型同步改成 `deepseek/deepseek-v4-flash`。

**判据从「模型名里有没有 deepseek」换成 Base URL 的主机名。** 旧判据在 OpenRouter 上正好判反：同一份权重经 OpenRouter 是支持严格 `json_schema` 的，经 DeepSeek 自己的 API 则没有这个东西。主机名比较用 `urlsplit().hostname` 而不是字符串前缀，`https://api.deepseek.com.example/v1` 不算 DeepSeek。同时 `_normalize_openai_compat_base_url` 不再给 DeepSeek 主机补 `/v1`：官方兼容地址就是不带版本段的 `https://api.deepseek.com`，另一个是 `/beta`，补 `/v1` 会拼出一个哪里都不存在的路径。

**`none` 档在 OpenRouter 上是错的，而且错得静默。** 它不认识 `thinking` 约定，未知字段是被丢弃而不是被拒绝，所以 `none` 必须写成 `reasoning.enabled=false`，否则整片每个请求都照常思考、照常计费——和 `medium` 被静默忽略是同一类事故，只是这次在改默认端点时就堵上了。

**`LLM_STRUCTURED_OUTPUT` 是三态的，因为两个方向都有真实需求。** 空（默认）＝按主机要严格 schema，但不限制路由；`json_object` ＝退回宽松约束，给中转 relay 用（它们的域名无法探测，而旧判据下「模型名含 deepseek 的中转」本来自动走 `json_object`，这就是它们的迁移路径）；`json_schema` ＝反方向钉死，在 OpenRouter 上追加 `provider.require_parameters=true`，只路由到真能强制执行 schema 的供应商。

**线上三探针实测（2026-08-24，Responses 面，模型 `stealth/ox-alpha`，各 2 条 cue）**，因为 `require_parameters` 到底按什么过滤，文档说不清楚：

| 请求形状 | 结果 |
| --- | --- |
| 严格 `json_schema` + `provider.require_parameters` | **404 `No endpoints found that can handle the requested parameters`** |
| 严格 `json_schema`，不带 provider 约束 | 200，回复恰好合规（140 in / 23 out，缓存命中 64） |
| `json_object` | 200，回复合规（201 in / 27 out） |

所以过滤看的是**结构化输出这项能力**而不是参数名是否出现：`stealth/ox-alpha` 在 `/models` 里声明了 `response_format` 却没有 `structured_outputs`（全站 422 个模型中 335 个有），于是该约束把它整条路由掐掉。

**因此它没有成为默认。** 第一版实现是「OpenRouter 一律带 `require_parameters`」，而同一天 15:29 的 匿名样片 C 就是用 `stealth/ox-alpha` 跑完的：609 条 cue、`missing_count: 0`、`reasoning_tokens: 0`，即上表第二行那种「没有 schema 强制、模型自愿合规」的状态。默认开启等于用一个能跑的配置去换一份形式保证，而未强制不等于无人检查——批解析器本来就校验自己要过的 id 并补发缺失项。于是开关改成三态，`require_parameters` 只在显式 `LLM_STRUCTURED_OUTPUT=json_schema` 时发出。这条 404 仍单独成一条 `stage_errors` 规则：泛用的 404 文案会让用户去换一个其实可用的模型，新文案直接给出两条真实出路。

改完按最终形状再实测一次（走真实的 `_chat_responses`，不是手搓请求）：默认档 200 且合规、`extra_body` 为空；钉 `json_schema` 档 404 并被翻成上述文案。OpenRouter 的 `/api/v1/responses` 无状态（不发 `store` / `previous_response_id`）这条按文档实现，探针也走这条面。全量测试 1615 passed。

### `medium` 从来不是 DeepSeek 的合法值：默认档十天里一直跑在 `high`

从一次成本投诉倒推出来的。匿名样片 AB（1,396 cue，`deepseek-v4-flash`，UI 显示 `medium`）单片 3,114,302 token、约 ¥4–6。按官方价拆开，账单结构和直觉完全相反：

| | token | 占 token | 占成本 |
|---|---:|---:|---:|
| 输入·缓存命中 | 2,450,432 | 79% | ~4% |
| 输入·缓存未命中 | 93,839 | 3% | ~5% |
| **输出** | **570,031** | **18%** | **~91%** |

高峰价 ¥5.66、空闲价 ¥2.83，**输出占比 91% 与时段无关**（进出同比例翻倍）。570,031 输出 token 里译文 JSON 只占约 4 万（1,396 × ~28），**其余 ~93% 是思维链——思维链单独吃掉整张账单的 ~85%**。输入侧那个 143,170 字符的全片 JSON prefix 缓存命中率 96%，只值 ¥0.25，在那里做任何优化都是白费力气。

原因在 [官方思考模式文档](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode)：Chat Completions 的 `reasoning_effort` 只接受 `low` / `high` / `max`，**思考默认开启且 effort 默认 `high`**；映射表明确写着 v4-flash 上 `low → low`。而本项目 `REASONING_EFFORTS` 是 `("low","medium","max")`，`medium` 是默认值。DeepSeek 对无法识别的值**不报错、静默忽略**，于是每个任务都落在 `high`。UI 上写着「medium（默认）」，实际计费按第二贵的档。

连带塌掉的是 2026-08-14 那条「low 与 medium 的实测需求分不开」的结论——那组对比（low 花 7,860/14,034/9,383 字符 vs medium 花 2,058/18,393/20,231）比的其实是 **low vs high**，不是 low vs medium。08-23 记的「官方会把 low/medium 映射为 high」也只对了一半：`medium` 确实变成 high，但那是因为它非法，`low` 是真实的独立低档。

**修法：把 thinking 折进 effort 轴，而不是并排放第二个参数。** DeepSeek 的 Responses 面把整根轴写成 `reasoning.effort ∈ {none, low, high, max}`，`none` 就是关思考——所以开关本来就是档位。`REASONING_EFFORTS` 改为 `("none","low","high")`（`max` 一并去掉：flash 上映射到最贵的 `max`，这条流水线没有被证明需要它），默认 `low`。落地只在 `openai_compat._chat_reasoning_fields` 一个边界函数：`none` → `thinking.type=disabled` 且不发 `reasoning_effort`，其余 → `enabled` + 原值。存量值 `medium`/`max`/`xhigh` 一律读作 `high`，即它们**实际跑成的样子**——落回默认会让旧任务重跑时悄悄换档。

**因此删掉的东西**：贯穿 base/openai_compat/engine/translator 四层的 `thinking_mode` 参数、profile 的 `reasoning_enabled`（现由 `effort != "none"` 就地算出）、translator 里的 `_uses_deepseek_adaptive_thinking` 厂商嗅探、以及 `apply_adaptive_thinking_pass`（130 行，是 `apply_repair_pass` 的近似复制，各带一套递归/拆分/timing/progress phase）。两个检测器选择函数合并成一个。级联现在是 provider 无关的：首轮按任务档位跑全片，`apply_repair_pass` 把标记 id 按 `escalated_reasoning_effort`（升一档，`high` 到顶自停）重发。`TRANSLATION_REPAIR_MAX_IDS` 12 → 400——12 是给「已经不错的译文挑长度离群值」定的，而级联下检测器**本来就该大量触发**（10.1% × 1,700 = 171 条），12 会漏掉 159 条。

**并发默认 16 → 4。** 真正决定 reasoning 成本的是**请求数**而不是并发数，但两者通过批大小耦合：`batch = min(cap, ⌈cue ÷ (2×workers)⌉)`。按「每请求 ~18,000 reasoning token」建模（该模型对 1,396 cue/16 worker 预测 576,000，实测 completion 570,031）：

| workers | batch | 请求数 | reasoning token |
|---:|---:|---:|---:|
| 1 / 2 / 4 / 6 / 8 / 10 | 64 | 22 | 396,000 |
| 12 | 59 | 24 | 432,000 |
| 16 | 44 | 32 | 576,000 |

注意 1 到 10 完全持平——`TRANSLATION_BATCH_SIZE=64` 这个**上限**才是当时真正在决定批大小的东西，所以并发降到 1 一分钱不省却要 4 倍墙钟时间；4 是最便宜那一档里最快的。同时把 cap 64 → 200，让 08-13 那张 regret 表定下的「每 worker 2 批」平衡规则重新成为生效规则而不是被 cap 顶掉：1,396 cue / 4 worker 下 ⌈1396/8⌉=175，请求数 22 → 8，reasoning token 再降到约 144,000。08-23 那次把规则本身改成「每 worker 1 批」并把 cap 提到 400 的做法已回滚——它为了省 reasoning 牺牲了负载均衡（regret 表：1 批/worker 是 26–77%，2 批/worker 是 7–16%），而把 cap 让开可以同时拿到两者。

**顺带记两件文档事实**：思考模式下 `temperature` / `top_p` **接受但不生效**（Chat 与 Responses 两面都是），所以 `LLM_TEMPERATURE=0.6` 一直是装饰品，只有在 `none` 档才开始真正起作用；Responses 的 usage 会返回 `output_tokens_details.reasoning_tokens`，Chat 没有等价字段——`transport_util` 现在记录它，这是整件事里唯一一个「本来该有却一直没记」的数（这次的 93% 是从交错的 `reasoning_chars` 流式事件里重建出来的）。

全量测试 **1590 passed / 1 skipped**；另有 2 项在 HEAD 上同样失败（临时目录在 C 盘、项目在 D 盘的路径断言），与本次无关。

### 复译不再无条件升档：升档只对 `none` 成立

上一节把整片压到 ¥0.886 之后，账单里剩下一个刺眼的单点：**复译只发 1 次请求，却吃掉 22,585 思维链 token = ¥0.224 = 整片的 25%**，而首轮 11 次请求加起来才 ¥0.50。原因是它无条件比首轮升一档，`low` 首轮于是一律按 `high` 复译。

升档的存在理由只对 `none` 成立：不思考首轮实测有 10.1%（171/1,700）的 cue 原样回显日文，而这道复译末尾的闸门正是为此会让整个任务失败。`low` 以上没有任何证据说明必须升档——而且复译的提示词本来就和首轮不同（带局部上下文、写明标记原因、只问少量 id），所以同档位并不等于重复同一次失败。

改成 **复译档位 = 首轮档位，下限 `low`**（`_repair_reasoning_effort`）；`TRANSLATION_REPAIR_REASONING_EFFORT` 可钉死，钉 `none` 会被拒（那等于把可修复的首轮变成必死任务），无法识别的值回落到规则而不是被 `normalize_reasoning_effort` 悄悄夹成 `low`——为此给 `core.config` 加了 `recognized_reasoning_effort`（返回 `None` 表示「这个值什么都不是」）。

实测（匿名样片 V，1,595 cue，4 worker，其余配置相同）：

| 复译档位 | 复译输出 | 复译思维链 | 复译墙钟 | 整片 ¥高峰 | 整片 ¥空闲 | 整片墙钟 | 终态门 | echo / kana / 肉棒 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `high`（升档） | 24,835 | 22,585 | 171s | 0.886 | 0.443 | 315s | 6 | 0 / 0 / 100% |
| **`low`（同档）** | **9,814** | **7,673** | **67s** | **0.752** | **0.376** | **170s** | 12 | 0 / 0 / 100% |

两次都把标记出来的全部修好（146/146、158/158，`missing=0`），成品的三项硬指标完全一致。**代价只有长度比例离群项 6 → 12 条**（0.38% → 0.75%），而那是诊断项不是错译，且落在本轮已量到的运行间波动带里。整片省 15%、快 46%。

### 术语提取并进预热：一片只买一次全片前缀；批大小与并发解耦

两件事，都为同一个量服务：**一次请求 = 一份几乎固定的思维链**。

**一、全片前缀原来一片要买两次。** 翻译前有两个前置请求：术语预抽取（把全片源文当自己的私有格式发一遍）和 prefix warmup（发一个空批，把全片 JSON 前缀灌进供应商的 prompt cache）。两者内容几乎一样，却是两段不同的 token 前缀，所以缓存互不相认——143K 字符的全片源文按 miss 计费两次，warmup 一条就吃掉 ¥0.232（占整片 ¥1.127 的 20%）。

合并的障碍是 `<glossary>`（每片自动提取的术语块）原来挂在 **system prompt** 末尾。缓存匹配的是 token 前缀，而这个块只有在提取返回之后才存在，于是提取请求和后面每个批天然分属两段前缀。**把这个块从 system prompt 移到用户消息的任务尾部**（`_build_extra_glossary_block`），system + `【全片字幕 JSON】` + 全片 payload 就在提取请求和所有批之间逐字节相同，实测共享前缀 141,335 字符，分歧点正好落在 `【本次任务】`。提取请求于是同时就是预热请求，`prefix_warmup` 只在提取走了磁盘缓存（没发请求）时才补跑。`PROMPT_VERSION` v3.3 → v3.4。

**二、批大小不再由并发数算出来。** 旧规则 `每批 = ⌈总 cue ÷ (并发 × 2)⌉` 是为负载均衡设计的，但它让**并发变成了一个计费旋钮**：同一部 1,396 cue 的片，4 并发 8 个请求、16 并发 32 个，思考账单差 4 倍而活是同一份。现在 `每批 = TRANSLATION_BATCH_SIZE`（默认 200），并发只决定同时飞几个批、且不超过批总数（`_auto_translation_workers`）。

批大小该按「一次能不能答完」定，而这件事有实测：200 条一批时，四次整片运行的 32 个请求里 **7 个没答完**——总是丢掉末尾一段连续 id（缺 9 / 50 / 100 / 100 / 184 条），或返回不在本批范围的 id。当时输出预算 42,495 token、实际最多用到 31,486，**所以不是截断，是模型自己提前收尾**。这种失败在思维链上是纯损失：废掉那次的思考照付，补发再从头想一遍。

**中途踩了一个自己造的反馈环，值得单独记。** 提取请求共享翻译 system prompt 之后，就连带继承了里面「人名汉字不确定时按日语读音罗马音化」这条规则——那对一行字幕是对的，对术语表是灾难。第一次合并运行提取出 `ジェイ-Jay`、`シルス-Sirusu`、`おなみ-Onami`，这些又被当作「本片已确定译法」灌回批提示，等于告诉模型「保持源文形态」：**首轮 1,595 条里 239 条原样回显日文、272 条残留假名**，复译候选从 32 条暴涨到 298 条，复译单独跑了 806 秒。这正是 2026-08-14 判死「不思考档」的那个失效模式，只是这次由提示词自己诱发。两处修：提取任务文本显式推翻继承来的人名规则并要求 `zh` 必须是中文；`_filter_global_glossary_terms` 增加结构性闸门，**目标词含假名或不含汉字一律丢弃**——后者不依赖提示词措辞，因为模型在被要求用中文回答的情况下照样返回了罗马字。

实测（匿名样片 V，1,595 cue，4 worker，`deepseek-v4-flash`）：

| arm | 请求 | prompt | 缓存命中 | **未命中** | 输出 | 思维链 | ¥高峰 | ¥空闲 | 墙钟 | 终态门 | 肉棒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 08-24 基线 chat/16w | 28 | 1,922,183 | 1,796,224 | 125,959 | 68,094 | – | 1.170 | 0.585 | 327.9 | 5 | 97.3% |
| 上一节收尾（未合并） | 11 | 794,675 | 703,488 | 91,187 | 86,962 | 61,600 | 1.127 | 0.563 | 399.3 | 4 | 100% |
| 合并·反馈环未修 | 11 | 748,839 | 695,552 | 53,287 | 101,023 | 72,154 | 1.139 | 0.569 | 909.6 | 10 | 100% |
| **合并 + 解耦（当前）** | 12 | 886,311 | 861,056 | **25,255** | 80,419 | 52,146 | **0.886** | **0.443** | 314.8 | **6** | **100%** |

缓存未命中 91,187 → 25,255（−72%），首轮各批命中率 99.4%（859,904 / 865,061）。整片高峰价比 08-24 基线低 24%，首轮零回显、零假名残留，终态门只剩 6 条长度诊断项。

全量测试 **1600 passed / 1 skipped**（另 2 项在 HEAD 上同样失败的路径断言与本次无关）。

### 自动术语提取一直在和用户术语表吵架，`low` 把这场架打输了

上一节把默认档从「实际是 `none` 首轮」改成 `low` 之后，按用户要求在匿名样片 V（1,595 cue）上实跑对照。省钱确实成立，但对照顺手暴露了一个和成本无关的回归：**术语表合规从 97.3% 掉到 83.8%**——37 条术语 cue 里 6 条译成「鸡巴」而不是配置里写死的「肉棒」，而且**一条都没被复译碰到**。三个检测器（源文回显 / 假名残留 / 长度异常）看不见换词：换词不是回显、不含假名、长度还一样。

顺着「为什么 `low` 比 `none` 更不听话」查下去，原因不在推理档，在提示词自己打架。`resolve_extra_glossary` 每次运行都会用整片源文挖 15 个高频词塞进系统提示，而它挖出来的正是这部片自己的说法：

```
用户术语表:  ちんぽ-肉棒, チンポ-肉棒, おちんちん-肉棒, チンポコ-肉棒
自动提取:    ちんちん-鸡巴  おちんちん-鸡巴  ちんぽ-鸡巴  おちんぽ-鸡巴
             ち○ぽ-鸡巴    おち○ちん-鸡巴  おち○ぽ-鸡巴
```

原来的抑制规则是**精确 key 相等**，只挡掉 `ちんぽ` 和 `おちんちん` 两条，剩下五条照样进提示词。于是同一段提示里同时写着「这个词译作肉棒」和「这个词译作鸡巴」，模型还会**跨词形泛化**——连被挡掉的 `チンポ` 也跟着译成鸡巴。`none` 之所以扛住，是因为它不在两条规则之间权衡；**思考越多越容易输，因为在两条并列指令之间做取舍正是「思考」的样子**。

两处都修，因为它们各修一半：

- **根因**：抑制改成「词形变体 + 目标不一致」。`○`/`〇`/`●` 这类打码字符当通配符，且**两侧都要通配**——`おち○ぽ` 要能匹配上术语表里的 `ちんぽ`，只把其中一边正则化就永远漏。目标相同的变体保留（它是加强不是竞争）。三次运行的真实提取结果实测：7 条冲突项全部挡掉，`まんこ-小穴`、`中出し-中出`、人名等 8–12 条无冲突项一条没误伤。残留风险是「术语表 key 是某个无关复合词的短子串」时会过度抑制，代价是少一条自动提示，对面是实测 16% 的指定术语翻错。
- **兜底**：质量门加第四个检测器 `glossary_violation`——源文命中术语表条目、译文却没出现对应译词就标记。纯字面单向判断，不评价替代词；没配术语表时该检测器完全不参与。离线回放确认它能抓全 `low` 漏掉的 6 条。

实测（同一份 aligned_segments，4 worker，`deepseek-v4-flash`）：

| arm | 请求 | 输出 | 思维链 | ¥高峰 | ¥空闲 | 墙钟 | 终态门 | 肉棒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 08-24 基线 chat/16w（`none` 首轮 + `low` 复译） | 28 | 68,094 | – | 1.170 | 0.585 | 327.9 | 5 | 97.3% |
| responses/`none`/4w | 15 | 61,325 | 27,105 | 0.962 | 0.481 | 311.5 | 17 | 97.3% |
| responses/`low`/4w（修复前） | 10 | 62,419 | 39,181 | 0.898 | 0.449 | 174.1 | 10 | **83.8%** |
| responses/`low`/4w（只加检测器） | 11 | 120,034 | 94,587 | 1.436 | 0.718 | 295.8 | 13 | 100.0% |
| **responses/`low`/4w（两处都修）** | 11 | 86,962 | 61,600 | **1.127** | **0.563** | 399.3 | **4** | **100.0%** |

终态门候选 17 → 4（1.07% → 0.25%，全部是长度异常这类诊断项，零术语违规、零回显），是四条臂里最干净的一次。

**同时必须记一条方法论警告：固定档位下思维链用量不可重复。** 三次配置完全相同的 `responses/low/4w`，首轮思维链分别是 38,641 / 87,727 / 45,718 token，**跨度 2.3 倍**；单个批次从 15 token 到 28,555 token 都出现过。部分来自自动提取词表每次不同（它进 cache 签名，三次 key 分别是 `68fc11d03777` / `cb8a8f8c5b40` / `470245722155`），但主要是模型自己决定想多久。**因此本表里 ±30% 的成本差读不出结论**，能读的是数量级和质量列。上一节那张「每请求约 18,000 reasoning token」的估算表要按同样的误差带看待。

顺带修正一处此前的记录：`responses-none` 与 `responses-low` 并**没有**共用缓存签名（`reasoning_effort` 确实在签名里），三条臂的 key 各不相同。

全量测试 **1596 passed / 1 skipped**，同 2 项在 HEAD 上一样失败的路径断言与本次无关。测量产物在 `agents/temp/20260824_090000_sample-v-responses-ab/`。

### 匿名样片 V：7B Q4 与 DeepSeek 成本/质量实测

使用同一份 `sample-v.aligned_segments.json` 生成完全相同的 1,595 个 cue（25,595 日文字符），分别跑唯一的本地 Hy-MT2 路径和当前 DeepSeek 自动级联。对比产物位于 `agents/temp/20260824_001500_sample-v-7bq4-deepseek-ab/`；`report.json` 是排除了合成 timing 总行后的规范统计，`local/api/bilingual.json` 保留逐 cue 双语结果。

| 指标 | Hy-MT2-7B Q4_K_M | DeepSeek v4 Flash |
|---|---:|---:|
| 墙钟时间 | 249.1s | 327.9s |
| 实际计量请求 | 1,595 | 28，另有 1 次未计量的全局术语提取 |
| prompt token | 57,814 | 1,922,183（命中 1,796,224 / 未命中 125,959） |
| completion token | 16,062 | 68,094 |
| API 成本 | `$0` | 非高峰 `$0.085227` / 高峰 `$0.170453` |
| 最终本地质量门 | 38/1,595，含源文回显 11 | 5/1,595，源文回显 0 |
| 用户术语“肉棒”命中 | 0/37；32 条使用“阴茎” | 36/37 |

成本按 DeepSeek [官方价格](https://api-docs.deepseek.com/quick_start/pricing/)计算；日志没有把全局术语提取放进 timing，表中 API 请求数与金额都少算那 1 次。基础阶段本身只用 27.0s、28,948 completion，但 16 个 batch 发生 7 次结构补请求，实际发出 23 次基础请求；第一遍随后被本地门标出 316 条（源文回显 247、残留日文 303、长度异常 13，原因可重叠）。4 次集中思考复译又产生 39,140 completion、用时 296.6s，分别占全部 completion 的 57.5% 和整段墙钟的 90.4%。这证明级联已经显著压低旧全量思考的 509,730 completion，但当前最值钱的下一步不是压缩 JSON，而是在进入思考前对可疑项做一次更小跨度的非思考复译，只把二次仍失败者升级。

本地端由项目 `.env` 选择 `<llama.cpp CUDA 目录>\llama-server.exe`；该目录是二进制运行时，不保存项目参数。`tmp/log/llamacpp_server.log` 的本次启动行明确为 7B Q4、`-c 16384 -np 2 -ngl 999`，即两个并发槽、每槽 8,192 context、全层 GPU。日志前部的 `n_slots=8` 属于旧 1.8B 运行，不能套到这次成绩。7B Q4 在两槽下仍比 API 整段快 78.8s，但字符闸能直接看到 38 条残留日文/11 条原文照抄，且 Hy-MT2 单句合同按设计不消费术语表与全片上下文；它适合本地草稿，不足以凭这次数据替代 API 成品。

已生成 60 条匿名盲审页 `blind-audit/index.html`，甲乙位置严格 30/30。助手仅看文字的预审为 API 胜 30、本地胜 18、都可用 8、都不可用 1、不确定 3；分出胜负的 48 条中 API 占 62.5%，但双侧符号检验 `p=0.1114`，未达到统计显著，而且这不是听过音频的人审结论。最终质量判断应以打开页面、实际听音后的人工裁决为准。

本次测试还暴露两项应在下轮重构中处理的成本入口：自动全局术语提取既未计量，又把 `オナ美` 生成成 `小穴`，应优先改为用户术语表/确定性规则；batch 100 的结构补请说明“大批总是更省”也不成立，失败恢复应先缩小问题 span，而不是把整组候选直接送昂贵思考。

首次本地臂脚本误把后端环境键写成 `LLM_BACKEND`，实际启动了第二次 API；发现后中止、脚本改为 `TRANSLATION_BACKEND` 并加后端断言，误跑产物已可恢复地移到 `agents/rm/20260824_002200_mislabeled-api-arm/`。这次误跑在中止前至少产生 1,715,072 cache-hit prompt、86,932 cache-miss prompt 和 26,617 completion，按非高峰价下限约 `$0.0487`；另有未计量的术语调用和已发出的在途修复请求，因此整次测试实际额外费用大致 `$0.049–$0.083`。连同正式臂，已记录 token 的非高峰费用至少约 `$0.134`，上界约 `$0.17`，再加两次未计量术语提取。

## 2026-08-17

### v1.4 发布前审计

发布前审计闭合了四个仍影响当前行为的边界：旧 schema 中已退役的推理强度值会被拒绝并备份原任务记录；翻译 A/B 审计复用输出目录时强制重切音频；外部审计页入口统一使用可解析的相对链接并去重；未知人工 verdict 直接报错，不再静默计入已审核。定向 Web/Audit 测试 106 项通过，随后完成 v1.4 发布验证。

## 2026-08-13

### 把 v3 一直没花的阅读时间还回去

**问题**：v3 让 cue 严格在最后一个实测发音字结束，`layout_timeline_locked` 又让它跳过 `_polish_subtitle_timeline`，于是 TTSG 的「出点 +0.5s」在 v3 之后**一次都没跑过**。八片实测代价：7,016 条里 487 条短于 20 帧最小显示时长，日文源 CPS>7 占 40.6%，同一批词的在屏总秒数比原通用头少 24%。这是当时字幕侧最大的单项缺口。

**为什么这一条可以还、另外两条不能**：最短时长和 2 帧间隔都要求把边界移到没有语音证据的位置，出点不需要——cue 之后的静音本来就空着（每片自由静音中位数 0.57–2.13s，只有 727/7,016 条即 10.4% 的 cue 后面完全没空隙）。多停一会儿不发明任何时间戳。

**四条上限同时生效**，每条都有各自要挡的事故：`linger_s`(0.5) 是规范本身；下一条起点前 2 帧防止压住后一句；`acoustic_end + max_display_shift_from_acoustic_end_s`(0.5) 是绝对上限而不是相对增量，因此**重复运行不会叠加**（locked cue 的 `acoustic_end` 就是切分时的 `piece_end`，不随显示终点变）；不越 `max_display_duration_s`(7s) 是因为 `spec_duration_over_7s_count` 是零容忍计数器，买阅读时间不能顺手给它造条目。下一条在 2 帧内开始时**整条不动**——回缩去截断实测语音正是 v3 存在的理由，而八片里正有 556 对相邻 cue 处在这个 2 帧之内。

**pass 放在纯人声过滤之后**（按用户裁决）：被删掉的呻吟段留下的是真静音，前一句可以用；天花板因此按最终存活的邻居算。

八片跑生产 `prepare_srt_blocks`，对照臂把该函数打成 no-op，两臂只差这一步：

| | 关闭延伸 | 开启（v3） |
| --- | ---: | ---: |
| 短于最小显示 | 487 | **198** |
| 日文源 CPS>7 | 40.65% | **27.85%** |
| 在屏总秒数 | 21,114.7s | 23,732.5s |
| 时长 p50 / p90 | 2.692 / 5.615s | 3.077 / 6.070s |
| 超 7s / 重叠 / 间隔<2 帧 | 34 / 0 / 556 | 34 / 0 / 556 |
| 超 20 字 / 比例回退 | 41 / 0 | 41 / 0 |
| 起点被移动 / 越过 0.5s 上限 | 0 / 0 | 0 / 0 |

cue 总数 7,016 不变，八片文本逐字一致。剩下的 198 条是后面确实没有可用静音的，属于明知偏离。（探索阶段那份估算给的是 23,792s，比生产实现多 59.8s，差在生产还多了一条 7s 软上限。）

版本戳只升 `TIMING_MODEL`（`measured_lexical_extent_v2 → v3`），`LAYOUT_ENGINE` 不动：切点一个都没变。两个戳分开正是为了这种情况。

### postgate 的标记第一次有了消费者

`asr.postgate` 每次运行都在算 `repeated_unit`、`runaway_repetition` 等标记，`_annotate_segments_with_postgate` 也把它们并到 segment 上，但 `rg postgate src/main.py src/subtitles/` 零命中——`_build_japanese_srt_blocks` 按固定 key 列表重建 block，标记就是在那里丢的，到 `aligned_segments.json` 为止再没有任何东西读它。检测器的钱一直在付，什么也没换回来。

现在标记随 block 下传（DP 拆出的每片用 `dict(block)` 自动继承——音频不支持的 chunk 不会在中途变成被支持的），cue plan 统计条数与分类，质量报告**同时给两层**：chunk 级说检测器看到了什么，cue 级说有多少真的活过了布局和纯人声过滤、进了成品字幕。**只有 cue 级那一列构成行动理由。**

两层都不设阈值：`repeated_unit` 在本域本来就有约 10% 的 chunk 命中率，且多数是真实的重复语气词，阈值只能靠发明。`postgate_alignment_score_checked` 如实报 0（未标定的对齐分检查没有运行），免得被读成「每条 cue 都有音频支持」。**下一步先看数再定**：如果被标记的 cue 基本都已经被纯人声过滤删掉，就保持只观测；如果 `repeated_unit` 大量活进成品，最便宜且正确的动作是文本层折叠重复串（产物保留 raw 文本），不是删 cue、更不是重解码——08-02 已实测短块会把 `repeated_unit` 从 10.5% 推到 14.6%。

**补完最后一段（同日）**：标记此前止步于「质量报告有个数」，产物里定位不到具体哪几条——翻译路径的 `srt_blocks` 也是按固定 key 列表重建的，`bilingual.json` 里 1,700 条 block **一条都没有** `postgate_flags`（`aligned_segments.json` 里同一部片有 28 条）。当时误记成 `_copy_sorted_blocks` 丢的，实际是 `src/main.py` 里那段字面量；`_prepare_subtitle_blocks` 全程 `dict(block)`，从头到尾没丢过键。现在只在**确实带标记的 cue 上**写这个键，空列表不写，所以 `rg postgate_flags sample-a.bilingual.json` 直接落到那几条上，而未标记的 1,672 条不会给产物增重。跳过翻译那条路径本来就直接写 cue，不受影响。

全量测试 **1529 passed / 1 skipped**（补完这段后 1567 passed / 1 skipped）。

### 五片真实运行暴露的块边界越界（首个带翻译的真实口径）

前面所有字幕数字都是离线重放，止步于日文那一半；这次跑了五部真实影片（匿名样片 A / 匿名样片 V / 匿名样片 J / 匿名样片 C / 匿名样片 B，全在离线八片之内），双语模式、开翻译、`keep_temp_files`，因此第一次拿到成品中文与生产切点几何下的产物。质量报告里冒出一项离线从来没出现过的东西：**重叠 cue**（匿名样片 V 2 对、匿名样片 C 1 对、匿名样片 A 3 对、匿名样片 J 5 对），而八片离线对照里这一项一直是 0。

**先排除新改的那一步**：这些 cue 的 `display_end == acoustic_end`，说明出点延伸根本没动它们——「下一条在 2 帧内开始就整条不动」那条分支按设计生效了。真正的成因在更上游：匿名样片 V 的 chunk 108 边界在 2932.673，下一块正好从这里开始，而 chunk 108 的**末词 `?` 对齐终点是 2932.711**，越界 0.038s，正好是一个上采样编码帧（76.9/2 ms）。`speech_extent` 的外扩走在编码帧上，而帧铺的是补零后的信号，所以走到张量最后一帧就会报出一个本块没有音频的时刻。

**为什么这不是无害的余量**：`asr.chunking` 输出精确铺满且相邻块共边，所以越界的那一段不是空地，是下一块的首词。字幕层收到的是两条**实测**时间重叠的 cue，而它拒绝截断实测语音（这正是 v3 锁定时间轴的意义），于是重叠原样写进 SRT。

**修法**：`build_aligned_word_timestamps` 此前只在比例回退分支用 `window_start` / `window_end`，对齐分支完全不看它们。现在它们是两条分支共同的硬边界——本块自身的音频，任何对它的测量都不可能落在外面（生产里 `alignment_window_source` 恒为 `chunk`，窗口就是 `[0, duration]`，不存在被更窄的 speech-core 窗口误伤的情况）。

修复前产物的实测（`agents/temp/20260813_105202_chunk-seam-overrun/`，四片已完成任务）：

| 片 | segment | 词终点落在自己块外 | 最大越界 | 成品重叠对 |
| --- | ---: | ---: | ---: | ---: |
| 匿名样片 A | 212 | 16 | 0.0385s | 3 |
| 匿名样片 V | 302 | 16 | 0.0577s | 2 |
| 匿名样片 J | 258 | 26 | 0.0577s | 5 |
| 匿名样片 C | 301 | 21 | 0.0577s | 1 |

11 对重叠里 7 对是整一帧（0.0384–0.0385s），4 对是 1e-5 量级的浮点触碰（wav 实际时长与切点计划的舍入差），两类都由这条 clamp 收掉。**另有一类没有修**：匿名样片 C 里有极短的块内 segment（seg 154 只有 19ms），其 `end` 与自己末词的终点不一致，最大分歧 0.173s；它没有产生成品重叠，成因也与块边界无关，留待单独查。

这条同时给 08-12 那句「被压成零宽的语义字符全落在源块首尾两个字符内，是 chunk-edge 问题」补上了另一半：块边界确实是问题所在，而且在生产的按停顿切几何下依然存在。

**clamp 生效后的实测（匿名样片 B）**：`past_own_chunk` **24 → 0**、`seam_overruns` **3 → 0**。同时出现两处附带修复，都发生在同一条块边界上：chunk 200 的结尾 `...`（三个字符）原本**整个丢失**——它们的 span 被 coda 外扩推到了块外，下游据此丢弃；clamp 把它们钉在块边界上后保留了下来。chunk 201 原本被劈成 `ん`（5375.827–5376.019，0.192s，单字符）和 `ー...はい…` 两段，现在是完整的一段 `んー...はい…`。这正是 08-12 那条「零宽语义字符落在源块首尾」的生产几何版本，实测代价是**每片个位数的标点字符 + 一条虚假的 0.19s cue**，不涉及语义字符。

**改完第一次重跑毫无变化，原因是 finalize 缓存**：匿名样片 B 新建任务重跑，`stage_done asr_alignment elapsed=26.03s`，对齐阶段确实执行了，产出的 `aligned_segments.json` 却和修复前的归档副本逐项相同——`past_own_chunk=24`、`max=0.038495`、`seam_overruns=3`，连微秒都一样。`asr.result_cache` 的 finalize 缓存（`tmp/cache/boundary/`，本机 1.7GB）按「模型签名 + 对齐头 digest + 边缘 cap + `word_build_version`」寻址，而**改的是把 span 变成词的那段代码，这四项一个都不动**，于是每个 chunk 都被修复前的条目服务。`word_build_version` 2→3 后才真正生效。

这个坑在 `result_cache.py` 的注释里写着，而且是**同一个坑第二次**：version 2（不再丢弃标点的零宽 span）当初也是"修完重跑，输出逐字节相同"。所以补了一条测试把耦合钉住，而不是只改数字。

全量测试 **1534 passed / 1 skipped**。

### postgate Phase B 裁决：保持只观测，不做文本层折叠

五片真实运行给出了 cue 级数字，按 Phase A 定下的规则裁决。**结论是不动运行时**，理由不是"量不够"，而是**规则的触发条件成立、但它给出的处置对这批数据是错的**。

数字（`agents/temp/20260813_105202_chunk-seam-overrun/adjudicate_postgate_cues.py`，用生产 layout 重放归档 segment，重建出的每片条数与质量报告逐项吻合）：

| 片 | cue | 被标记 | 占比 | repeated_unit | runaway_repetition |
| --- | ---: | ---: | ---: | ---: | ---: |
| 匿名样片 A | 870 | 82 | 9.43% | 73 | 13 |
| 匿名样片 V | 1595 | 78 | 4.89% | 66 | 12 |
| 匿名样片 J | 893 | 91 | 10.19% | 70 | 36 |
| 匿名样片 C | 609 | 65 | 10.67% | 59 | 35 |
| 匿名样片 B | 1700 | 128 | 7.53% | 99 | 51 |
| 合计 | 5667 | 444 | 7.84% | 367 | 147 |

规则说「`repeated_unit` 大量活进成品 → 文本层折叠重复串」。367 条确实算"大量"，但规则的前提是**被标记＝有缺陷**，而证据说不是。最长重复次数的分布是 `{1:58, 2:60, 3:282, 4:27, 5:11, 6:4, 7:1, 8:1}`——**压倒性地集中在恰好重复 3 次**，也就是本域最常见的正常说法。全语料重复 ≥5 次的只有 17 条（0.3%）。

按重复次数排序的最严重那些，逐条看都是**转写正确**：

```
x8 好き    3.54s  っ、好き好き好き好き好き好き好き好き
x6 すごい  1.84s  すごいすごいすごいすごいすごいすごい!
x6 そう    2.76s  そうそうそうそうそうそう。そうそうそう
x5 あっ、  3.54s  あっ、あっ、あっ、あっ、あっ、あっ!
```

速率也说得通：`好き`×8 用 3.54s 是 2.3 次/秒，`すごい`×6 用 1.84s 是 3.3 次/秒，都在人能说出来的范围内——解码失控通常给出物理上不可能的速率，这批里一条都没有。真正可疑的只有 `添い寝添い寝添い寝添い寝添い寝`、`ぐもぐもぐもぐもぐも` 这类个位数。

所以**折叠会删掉说话人真的说了的内容**：把「好き好き好き好き」压成「好き」是改语义和情绪强度，属于翻译质量回退而不是修复。删 cue 和重解码本来就更靠后（08-02 已实测短块把 `repeated_unit` 从 10.5% 推到 14.6%）。维持只观测。

顺带暴露一个 Phase A 的缺口：`postgate_flags` 到 `aligned_segments.json` 为止，**没有进 `bilingual.json`**（`_copy_sorted_blocks` 丢掉了），所以从产物里没法定位被标记的是哪几条 cue，这次裁决是靠离线重放做的。

### 一条被截断的回复杀掉整部片，而提示指向一个改了没用的旋钮

同一批五片里，匿名样片 B 在翻译阶段失败，报 `LLM JSON response was cut off by max_tokens; increase TRANSLATION_MAX_TOKENS.`，此时 1,701 条里已经翻好并付过费的有 1,310 条，全部丢弃。ASR 侧完好（339 segment，产物仍在 job 目录里）。

**提示指错了旋钮，这一点从代码就能确定**：`_chat` 用的是

```python
effective_max_tokens = min(TRANSLATION_MAX_TOKENS, batch_budget)
```

而 `batch_budget` 来自 `JsonProfile.response_token_budget`：`源字符数 × TRANSLATION_OUTPUT_CHAR_RATIO(1.5) + 28×条数 + 32`。这次运行里一个 54 条的批算出来是 **12,794**，天花板是 **384,000**，`min()` 永远取前者。所以照着提示去调 `TRANSLATION_MAX_TOKENS` 不可能有任何效果。

**哪一个请求被截断没有定论，而这本身就是个缺陷**：日志只记成功的批（`translation_batch_done`），失败那次什么都没留下。可用的间接证据是已完成批的 `completion_tokens` 峰值只有 2,500，对着 12,794 的预算，说明正常批离预算还很远。**中途一度怀疑 `_MIN_TOKEN_BUDGET = 96` 这条地板**（修复请求只重发缺的几条，源字符少时预算走地板）——但读清楚公式后这条不成立：`max(96, body + structure)` 里单条的 structure 就有 60，地板只会把预算**抬高**，从来不会压低，所以它不是元凶，也就没有改它。

**改动**（三处，行为最小）：

1. 新增 `ResponseTruncatedError(TranslationError)`，带 `limit` 字段。**故意不继承 `RetryableTranslationError`**：通用重试路径原样重发同一个请求，对截断毫无意义，只会按 `TRANSLATION_API_RETRIES` 把一次失控重复付费。
2. 两个 transport（Chat 与 Responses）都改抛它，消息里写**真正生效的那个数**。
3. `_chat` 捕获后按 `TRANSLATION_TRUNCATION_RETRY_FACTOR`(2.0) **加大预算重试一次**；仍被截断才判死，终局消息点名 `TRANSLATION_OUTPUT_CHAR_RATIO` 而不是天花板。预算已经顶到天花板时不重试（那才是「同一个请求再发一遍」）。重试前发一个 `output_truncated` 诊断事件进运行日志——今天答不出「是预算太紧还是模型失控」，正是因为没有这个。

只重试一次是有意的：预算是对**合法**译文长度的算术上限，撞上它只有两种可能，而 transport 分不清。一次升级能救「预算太紧」，「模型失控」则多付一个请求后照样失败——比旧行为（当场终止并丢掉整片）严格更好。

全量测试 **1542 passed / 1 skipped**。

### 同一部片的第二种死法：id 整体偏移，而重试只会把同样的请求再发一遍

修掉截断后，匿名样片 B 重跑，换了个死法：

```
Batch translation returned invalid or incomplete JSON after 4 attempts:
batch=24, start_index=1296, size=54,
error=LLM JSON output returned invalid batch translation id: 1350.
```

批 24 要的是 id 1296–1349。模型返回了 **54 条**（数量检查通过），但里面出现了 1350——**整批 id 偏移了一位**，不是"多吐一条"。同一现象在进度条上也能看到：`translated` 涨到 1741 而 `expected` 是 1701，因为流式计数器数的是回复里 `"id":` 这个字面量出现了几次（`transport_util._count_translation_markers`），范围外的 id 照数不误。（顺带说明这个计数器本来就是估算：它也会**往回跳**，某批开始重发修复请求时计数从 0 重来。）

**拒绝是对的**：接受一个偏移的 id 集合，等于把每条译文挂到相邻 cue 上，而且没有任何迹象。这类静默错位不能为了"跑通"而放行。

**错的是拒绝之后做什么**：旧逻辑把 `pending_ids` 原样重发，于是 `TRANSLATION_API_RETRIES` 买到的是四次**形状完全相同**的请求——同样 54 条、同样的前缀，模型自然同样偏移，四次全废，整片丢弃。

**改法**（`llm/engine.py::run_batch`）：新增 `request_span_limit`，单个请求最多要这么多 id。每次格式失败或零进展就**减半**（54→27→13→6），批内只降不升——让模型丢失 id 序列的那个原因并没有消失。剩下的 id 走原本就有的 missing-ids 路径再要一次。这与 ASR 阶段遇 OOM 降 batch 是同一个动作，只是这里的信号是"抄不对 id"而不是显存。

两处配套修改是必需的，否则新逻辑自己会打架：

- **进展判据**从 `len(missing) < len(requested_ids)` 改成 `len(missing) < pending_before_request`。窄了以后两者不再相等：一个完好的半批返回后 missing 仍等于另外半批，按旧判据会被记成失败尝试并继续收窄，永远降不到底。无收窄时两者恒等，所以旧行为逐字保持。
- **`TRANSLATION_BATCH_MAX_REQUESTS` 12 → 24**。12 是按"一次请求＝一次尝试"定的；收窄后一个 54 条的批降到 13 需要 1 次失败 + 1 次失败 + `ceil(54/13)=5` 次覆盖请求，而重试预算允许四次这样的下降。12 会正好在它快要成功时中止。

顺手删掉了 `pending_segments`：改动后每个请求的 segment 列表由 `requested_ids` 现算，这个变量只剩赋值没有读取。

全量测试 **1548 passed / 1 skipped**。

### 质量报告第一次有了读者：Web 端质检面板

这个月加的指标——切分来源（`chunk_cut_*`）、布局断点类型（`layout_break_type_counts`）、词间隔切点（`layout_word_gap_*`）、出点延伸（`display_linger_*`）、两层复读标记（`postgate_*`）——全部写进了 `<stem>.quality_report.json`，而**只有 `.md` 被登记成任务产物**，JSON 连下载入口都没有。要看这些数就得自己去 `video/<stem>/` 打开文件，等于付了检测的钱不看结果。

新增 `GET /api/quality/{job_id}`：从任务产物里找 `.quality_report.md`，走既有的越权校验解析出真实路径后**换后缀取同名 JSON**——授权文件的同目录兄弟必然还在授权目录内，所以不需要第二套路径判断。报告是可选产物，因此「没有」不是错误：`available:false` 分 `not_generated`（没开开关）和 `markdown_only`（只剩 .md，仍可用系统程序打开）两种，页面据此说不同的话。

面板（`src/web/static/js/qcReport.js`）按流水线的产生顺序分七组：交付规格（TTSG）/ 时间轴与阅读时间 / 切分与布局 / 密度 / 文本与译文 / ASR 健康度 / 复读检测，另有断点类型条形分布、复读标记的**两层对照表**（音频块 vs 成品 cue）、以及规格 / 密度 / 重叠三张样例表（时间码按 `HH:MM:SS.mmm` 给出，可直接到播放器里核对）。`warnings` 里每条形如 `<metric>=<value> > <ENV>=<limit>`，因此第一个 `=` 前就是指标名——被触发的行自己高亮，阈值挂在 tooltip 上，不需要在前端复制一份阈值表。

**不认识的指标不会消失**：分组之外的标量键落进「其他指标」组照搬。质量报告的键还在长，页面漏写标签是常态，漏写不该等于看不见。用真实的匿名样片 B 报告在 node 里跑一遍生产渲染路径核对：81 个标量键全部出现在页面上、4 条警告对应 4 个高亮行、fallback 组为空、没有 `undefined`/`NaN`。

任务卡上的「📊 质检」按钮**只在这次运行真的写了报告时出现**——否则点开只能道歉。全量测试 **1556 passed / 1 skipped**。

### 80 条无人反驳的静音 cue：定点重问 Grok，确认 9 条，其余 71 条仍然洗不清

八片验收留下的 86 条「词义对白读成 100% blank」里，归档全片教师只证实了 6 条，另外 80 条一直按**无人反驳**记着。这次按 cue 定点重问：每条 cue 取 `acoustic_start/end` ±8s 的窗口（中位 span 2.77s，窗口约 19s），从验收当时那份 wav 上切片（同一个时钟，不需要再推 PTS），逐条送 Grok STT，86 条共 28.3 分钟音频，**实际花费 $0.0473**。

**已被归档证实的 6 条一起重问，当正对照**——这是整件事的关键，没有它这份结果会被读反：

| 判定 | 全部 86 | 80 条无人反驳 | 6 条正对照 |
| --- | ---: | ---: | ---: |
| span 内有词义字（汉字/拉丁/数字） | 11 | **9** | 2 |
| span 内只有假名 | 11 | 8 | 3 |
| 该窗口有转写但 span 内什么都没有 | 27 | 26 | 1 |
| 整个窗口一个词都没返回 | 37 | 37 | 0 |

**正对照的灵敏度只有 2/6 = 33.3%**：已知有真实语音的 span，这套定点方法也只在三分之一上给出词义证人。所以「span 内没有词义字」**不能**当成洗清——三条正对照恰好落在「只有假名」那格，而假名正是呻吟被转写出来的样子，对判决没有信息量。按 9 / 0.333 外推，80 条里真正的误伤点估计 **约 27 条**（n=6 的对照，区间很宽）。

结论按证据分三级如实记：**确认从 6 条升到 15 条**；**71 条仍然无人反驳**，且现在有了量化的理由说明为什么它们清不掉；37/86 的窗口整段返回空，正是 2026-08-11 记的那个失败模式（Grok 在呻吟密集处整条失败），只是现在被压缩到 19 秒的窗口而不是 300 秒的分块。

新确认的 9 条里，证据强度差别很大，一并记下免得被当成同一档：匿名样片 B#1457 字幕「そいついきまってるな。大丈夫?」对 Grok「最盛期になってるよ大丈夫い」、匿名样片 W#878 对「か俺が言うのっていいの」是明确吻合；而匿名样片 J#819 只落了一个「難」、匿名样片 V#1468 只落了一个「願」——这些确认的是**那里有词状发声**，不是 ASR 那句写对了。

**这不推翻 v2 头的晋升**：15 条落在八片约 7,000 条 cue 上，各候选头的证实数也在同一量级。它改的是风险的写法——原来的「6 条」是测量能力的下限而不是真实值，现在的写法是「确认 15、点估计约 27、71 条无法判定」。判据「词义对白不得读成 100% blank」在本域**仍然无法用 Grok 度量**。

**踩到的坑记一条**：`create_speech_to_text_transport` 不给 `model_override` 就回落到共享配置里的 `OMNI_MODEL`（当前是 `google/gemini-3.6-flash`），OpenRouter 直接 400「does not support response_format verbose_json」。全片 runner 有 `--model` 默认值所以从没暴露过，新写的脚本必须自己带上。

### 翻译配置终于可以被人眼裁决：逐 cue 盲化 A/B

翻译侧到现在为止的每一次改动，靠的都是 prompt 推理和零散抽查——没有任何一次是「同一句话、同一段音频、两个配置，人来选」得出的。新增 `tools.audits.generate_translation_ab_audit_html` + `evaluate_translation_ab_audit` 把这件事补上，复用既有的人工审计 Core（播放器、状态、完成度、保存 API）与 CTC 边界 A/B 的产物形状（`manifest.jsonl` 给页面、`answers.jsonl` 是答案、`summary.json` 记生成参数）。

**两臂就是同一部片的两次运行，不重新翻译任何东西**。这样做的前提是两臂的 cue 必须逐条相同，所以工具会校验 cue 数、每条起止点和日文原文，任一条不同就直接停——理由正是上一条记的解码不可复现：拿两次重新解码的运行当臂，比的是 ASR 不是翻译。造臂最省事的方法恰好也最正确：任务跑完后改「翻译设置」再点「重试翻译」，重试复用 ASR 产物，几何天然一致。

三条设计决定值得记下来：

- **只抽两臂中文确实不同的 cue**。相同的译文不含偏好信息，混进样本只会稀释效应量。匿名样片 B 上 1,700 条 cue 里合成臂差异 241 条，抽样池就是这 241 条。
- **盲化按结构校验，不按字符串搜索**。页面行只允许携带 `row_id / span / ja / clip_src / arm_1_text / arm_2_text` 六个键，多一个就报错；反过来「页面里不许出现臂名」是不可用的判据——臂叫 `none` 会命中 Core CSS 里的 `display:none`，叫 `flash` 可能命中字幕正文。甲/乙 的先后按半数平衡随机。
- **统计只在分出胜负的卡片上做**。「都可用」不折半计给两边——它是审计者在说这个差异不重要，把它折进胜率就是把平局报成结果；未审阅卡片单独计数并报出，避免「看了一半的胜率」被当成整体胜率。给符号检验 p 值与 Wilson 95% 区间（6 张卡的冒烟里 2:1 的区间是 0.21–0.94，正好说明这种样本量什么都证明不了）。

冒烟用合成臂（同一份 `bilingual.json` 改若干条中文）跑通了切片、盲化、答案与统计全链路，产物在 `agents/temp/20260813_140000_translation-ab-smoke/`；统计与对齐校验有 8 个单测。

**真实页面已生成，不需要再跑一次翻译**：用户问「剩余的只需要在 index 里做审计吗」，顺着查发现 08-11/08-12 那次本地 Hy-MT2 vs DeepSeek 的对照产物还在盘上，而且**两臂共用同一条 run3 时间轴**（1,504 条 cue，几何与日文逐条相同，中文 1,486 条不同），正好是工具要求的合法臂对。据此生成 `agents/audits/20260813_223000_translation-ab-local-vs-deepseek/`（60 张卡，甲/乙 各领先 30 张，60 段 mp3，页面结构化盲化校验通过）。**这也是当时那份对照第一次能被人耳裁决**——08-12 记的 18,074 vs 13,414 字、假名残留 1.9% vs 11.0% 都是文本统计，说不了「哪句更好」。

**顺带修掉两个让页面进不了导航的真 bug**：① `update_audit_entrypoints` 对**不在 `agents/audits/` 下的页面直接静默 return**——每个生成器都收 `--output-dir`，`agents/temp` 又是本文件写在复现步骤里的目的地，于是工具打印成功摘要、页面躺在盘上、`agents/audits/index.html` 里什么都没有，审计根本不会发生。`register_external_audit_page` 本来就是为这种页面准备的，但只有 CLI 子命令会调它。现在外部页面自动登记到 `external_pages.jsonl` 再刷新导航。② `write_latest_audit_entry` 用 `rel_url` 生成链接，而 `rel_url` 对不在审计根下的路径**回退成绝对文件系统路径**，HTTP 下点不开；改用 `_nav_href`（走 `os.path.relpath`，支持 `../`）。两条各加一个测试。

### 冷跑一次拿到未被缓存污染的阶段耗时：ASR 解码占 86.8%，其余全部加起来 121 秒

此前所有阶段耗时都是在 ASR 结果缓存命中的情况下测的（`asr_text_transcribe_s` = 0.0025s），于是「翻译是不是大头」这个问题一直没有真数据。匿名样片 B（151.2 分钟）带 `ASR_RESULT_CACHE_ENABLED=0` 整片重跑一次，与同一部片的缓存运行对照：

| 阶段 | 冷跑 | 缓存命中 |
| --- | ---: | ---: |
| 音频准备 | 5.62s | 5.93s |
| 静音分析与切块 | 23.24s | 22.20s |
| ASR 模型加载 | 4.22s | 4.10s |
| **ASR 文本转写** | **798.32s** | 0.0025s |
| 字幕时间轴（对齐） | 32.21s | 35.07s |
| 字幕 Cue Plan | 0.85s | 0.85s |
| 翻译 | 47.34s | 60.23s |
| 输出写入 | 0.82s | 0.85s |
| **总计** | **919.78s** | 135.77s |

**结论没有悬念**：解码占冷跑的 **86.8%**，其余七个阶段合计 121 秒；翻译在 `reasoning=none` 下只有 47.3s（5.1%）。整片 9.9× 实时，单看转写 11.4× 实时（RTX 4060 Ti，batch=5，bf16）。要提速只有解码这一处值得动，其它阶段就算清零也只省 13%。

**顺带撞出一件更要紧的事：同一份音频重解一遍，结果不一样。** 两次运行的**块边界逐条相同**（339 块，0 处不同），但 **262/339 块的文本不同**，总字符 29,097 vs 28,817（+1.0%），cue 数 1,729 vs 1,700（+1.7%）。分歧位置散布在块内（共同前缀占比中位 30%，24% 的块在前 10% 就分岔），长度差两边对称（冷跑更长 121 块、缓存更长 123 块）。机制当天没有定论，同日的对照实验把它定死在**批大小**上，见下一节。

产物：`agents/temp/full-workflow/20260813_121940_cold-speed-run/`（含 `summary.json` 与逐阶段 `timings.json`），对照脚本 `agents/temp/20260813_121940_cold-speed-run-check/compare_runs.py`。

### 重解不一致的机制：是批大小，不是温度、也不是 kernel 噪声

**先排除采样。** 我们的调用是 `self.model.generate(**moved, max_new_tokens=cap, do_sample=False, ...)`，`_normalize_deterministic_generation_config` 在 `do_sample=False` 时把 `temperature` 置空，本地 `generation_config.json` 里根本没有 `temperature`/`top_p`。

**官方口径逐处核过（用户指出后补查，三处不一致但结论一致）**：

- `QwenLM/Qwen3-ASR` 的高层封装 `qwen_asr/inference/qwen3_asr.py:272`，vLLM 后端**写死** `SamplingParams(temperature=0.0, max_tokens=max_new_tokens)`——不是默认值，调用方连改都改不了。
- 同一个封装的 transformers 后端（同文件 :510）是 `self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)`，**连 `do_sample=False` 都不传**，纯靠模型自带 `generation_config.json` 里的 `do_sample: false`。我们显式传，等价但不依赖权重目录里的那个文件。
- 官方 README 的 Evaluation 章节写「`dtype=torch.bfloat16`、`max_new_tokens=1024`、vLLM、**Greedy search was used for all decoding**」；但同一份 README 里那段裸 vLLM 示例给的是 `SamplingParams(temperature=0.01, max_tokens=256)`——**0.01 不是贪心**（vLLM 只把 0.0 当贪心），这是官方文档自己的不一致，示例那行不要照抄。
- Transformers 侧的 `model_doc/qwen3_asr` 文档页则通篇没有 temperature/top_p/num_beams/repetition_penalty，示例里唯一的生成参数是 `do_sample=False`。

四处说的是同一件事：**这个模型就该贪心解码**，我们已经是。所以温度不是本次分歧的入口。另外**官方自己也批量推理**（`apply_transcription_request` 收 audio 列表，封装还带 `max_inference_batch_size`），下面那条批组成的性质因此不是我们的用法造成的。

顺带记一条检索教训：grok 搜索给的「官方推荐 temperature=0.0、batch 8–16 更稳」这条，**结论的前半截碰巧对上了**（见上面第一点），但它引用的 `huggingface.co/Qwen/Qwen3-ASR` 取回是 **401**、页面根本不存在，后半截的 batch 建议在任何官方文本里都找不到。凑巧说对不等于有出处——这类条目要落到仓库源码或真实页面上才能引。

**对照实验**（`agents/temp/20260813_210000_decode-batch-determinism/`）：从冷跑保留的整片 wav 重切前 30 个块（span 取自冷跑 `transcript.json`，逐条相同），在同一进程里解三遍：

| 跑法 | 结果 |
| --- | --- |
| A: batch=5 vs B: batch=5（同批重复） | **0/30 不同** |
| A: batch=5 vs 几天前那次 batch=5 整片运行的存档 | **0/30 不同**（不同进程、不同日期） |
| A: batch=5 vs C: batch=11 | **20/30 不同**（窗口内字符 3,158 vs 3,060） |
| C: batch=11 vs 存档 batch=11 运行的同组前缀（块 0–21） | **1/22 不同** |

最后那 1 条正好把链条闭合：那次 batch=11 运行日志写着 `ASR 缓存命中 18/339`，**第一批只有 10 行**，而块 #0 的缓存条目日期是 08-12（早于该次运行）——它是那 18 个命中之一，根本没在那次运行里解码。**它没解的那一块就是唯一对不上的那一块**，其余 21 块逐字一致。

**结论**：贪心解码在批组成固定时逐位可复现；变的是一次 `generate` 里同批放了哪几个块——同批要补零对齐到最长的那条，批大小一改，bf16 累加顺序跟着改，个别位置 argmax 翻面，后续 token 顺着走偏。两次运行差的正是这个：缓存运行 `asr_batch_size=11`（`asr_batch_source=learned_profile`），冷跑 `=5`（`auto_scaled_from_vram`），因为冷跑传的 `ASR_MAX_NEW_TOKENS="0"` 与档案身份里的空串不匹配，学到的档案没命中。

**还有第二个改批组成的入口，比批大小更隐蔽**：`_transcribe_asr_chunks_text_only` 把命中缓存的块从 `pending_chunks` 里剔除后才发给后端，所以**部分命中会让剩下的块重新分组**——「缓存半满 + 没钉批大小」是最容易得到第三种结果的组合。

**这不是第一次撞上**：2026-08-03 拿 bar=72 与 bar=158 两次运行逐块配对时就见过同样的现象，当时归因到「bf16 + sdpa 归约顺序在近似平局上的翻转」并留下「两次 GPU 运行之间的逐块文本差异不能当门槛效果的度量」。方向是对的，缺的是**归约顺序为什么会变**——批组成。补上这一环之后，那条禁令有了解除条件：批大小钉住、且两次都是全新解码（不是部分命中），逐块文本就可以直接比。

**实践后果**：要让重新解码可比就在 `.env` 里钉 `ASR_BATCH_SIZE=<n>`；缓存全命中时根本不解码，仍是最省事的可比办法。任何仍需重新解码的离线 A/B，自带约 1% 字符、约 1.7% cue 数的底噪，小于这个量级的差异不能当信号读。

### 复现步骤

```powershell
$env:PYTHONIOENCODING = "utf-8"
# 空闲静音能还回多少阅读时间（含「顶到下一条」那档为什么否掉）
uv run python agents/temp/20260813_090000_next-directions/measure_recoverable_reading_time.py
# 生产实现开/关延伸的八片逐片对照
uv run python agents/temp/20260813_090000_next-directions/verify_display_linger.py
# 真实运行产物里的块边界越界（需要 keep_temp_files 保留的 aligned_segments.json）
uv run python agents/temp/20260813_105202_chunk-seam-overrun/measure_seam_overrun.py
# 质检面板：用真实报告跑一遍生产渲染路径，核对键覆盖与格式化（需要 node）
node agents/temp/20260813_180000_qc-panel/render_real_report.mjs `
    ../../../video/sample-b/sample-b.quality_report.json
# 静音 cue 定点重问 Grok（会真的花钱；--dry-run 只出工单和预算）
uv run python agents/temp/20260813_121940_silent-cue-grok/adjudicate_unrefuted_silent_cues.py --dry-run
# 四分判定与正对照灵敏度（只读已保存的响应，不再调用 provider）
uv run python agents/temp/20260813_121940_silent-cue-grok/recount_verdicts.py
# 冷跑阶段耗时（约 15 分钟 GPU；--asr-max-new-tokens 0 才等价于生产的「跟着音频走」）
$env:ASR_RESULT_CACHE_ENABLED = "0"; $env:QUALITY_REPORT_ENABLED = "1"
uv run python -m tools.workflows.run_full_workflow --video video\sample-b.mp4 `
    --asr-max-new-tokens 0 --subtitle-mode zh --translate --translation-max-workers 16 `
    --task-name 20260813_121940_cold-speed-run
# 冷跑与缓存运行的块边界 / 文本对照
uv run python agents/temp/20260813_121940_cold-speed-run-check/compare_runs.py
# 批大小决定论对照：先取实验窗口，再同进程解三遍（约 4 分钟 GPU），最后核对存档
uv run python agents/temp/20260813_210000_decode-batch-determinism/prep.py
uv run python agents/temp/20260813_210000_decode-batch-determinism/rerun.py
uv run python agents/temp/20260813_210000_decode-batch-determinism/find_cached_chunk.py
# 翻译盲化 A/B：生成页面（两臂需来自共用 ASR 产物的两次运行）
# 输出目录放 agents/audits/ 下，导航靠扫描发现；放别处会走外部页面登记，也能进导航
uv run python -m tools.audits.generate_translation_ab_audit_html `
    --arm none=<run_a>/<stem>.bilingual.json --arm medium=<run_b>/<stem>.bilingual.json `
    --audio <job>/audio/<stem>.<key>.wav `
    --output-dir agents/audits/<ts>_translation-ab --sample 60
# 盘上已有的合法臂对（同一时间轴、不同翻译后端）
uv run python agents/temp/20260813_220000_ab-arm-hunt/find_pairs.py
# 人工裁决保存后统计（answers.jsonl 与页面保存下来的 manual_verdicts.jsonl）
uv run python -m tools.audits.evaluate_translation_ab_audit `
    --answers agents/audits/<ts>_translation-ab/answers.jsonl `
    --verdicts agents/audits/<ts>_translation-ab/manual_verdicts.jsonl
```

## 2026-08-12

### 头因果隔离评估：先判 run3，再被精确布局复核翻回 run1

同一天两份评估给出相反建议，最终晋升的是后一份。两份都留在 `agents/temp/20260812_081331_fixed30-head-factorial/`（`evaluation_report.md` / `exact_char_cap_report.md` / `exact_joint_layout_report.md`），先读第一份再读第二份才能理解结论为什么反转。

**测量设计终于把头单独隔离出来了**：八片全部改用固定 30 秒、与头无关的相同音频块，每片三臂复用同一份 Qwen 转写，只替换对齐头；8/8 片的 spans hash 与 transcript hash 三臂一致，因此差异不能由分块或 ASR 文本解释。此前所有端到端对照都做不到这一点——换头会同时改分块、改文本、改版面。

**第一份评估判 run3、不判 run1**：语义声学字符保留率 shipped 99.8956% / run1 99.8531% / run3 99.8858%，被压成零宽的语义字符 64 / 90 / 70，呻吟字符 190 / 506 / 471；CPS>7 为 11.06% / 27.19% / 16.89%，假名-only 在屏 11.02% / 10.26% / 8.82%。**同时它自己否掉了 08-11 判给 run1 的那条决定性理由**：生产已把标点类经 `silent_classes` 传给 `blank_runs`，「带标点词表会把停顿劈碎」不再是纯声学头的独有优势。另外两个结论一并记下：所有被压零宽的语义字符都落在源块首尾两个字符内，是 chunk-edge 问题而不是头在片中把正常对白当 blank，因此「按同一配方完全重训」只会测随机种子，不做；Grok `speaker` 从未进入 run1/run3 的训练目标或切点标签，词内 speaker 抖动不是这版头学到的坏边界，因此不因 speaker 重训。

**切点策略同轮评估，两项都维持原样**：同一个已选停顿内移动落点（排除「换了停顿」的混杂），落入 Grok 发声岛的比例是起点 17.43% / 25% 8.53% / 中点 7.83% / 75% 7.88% / 终点 14.56%，0.1–0.5s 多种发声岛合并口径下方向不变，保留中点；「末 5 秒内选最宽停顿」在代理指标上只把落岛数从 169 降到 160（0.49pp、八片 4 胜 2 负 2 平），V/B/W 三片真实端到端配对解码后 V 获益、B 混合、W 打平且呻吟残留变差，收益不能跨片复现，不改。

**第二份评估把第一份的布局相关指标判为作废**：旧 `cues.json` 已经过当时的生产布局，八片里 run1 带 135 条 `display_clamped_to_max`、1,200 条 `proportional_fallback_used`，run3 是 408 / 1,058——拿它们比字数与时长，比的是旧布局的伪造时间。改为从原始 CTC `segments.json` 用 v3 精确布局重排后：shipped 7,090 cue、时长 P50/P90 `3.81/6.35s`、超 7s 0.97%；**run1 7,016、`2.73/5.65s`、0.48%**；run3 7,108、`3.19/6.04s`、1.29%，三臂的比例边界与 clamp 均为 0。时长上限固定 run1 / 20 字再扫：6s 出 7,144 cue、7s 出 7,016、8s 出 6,962，字数上限已承担主要切分作用，7s 保留作异常慢句的第二约束。**最终晋升 run1 + 20 字 + 7 秒。**

**审计复核（2026-08-12 晚，晋升提交之后）**：第一份评估里那条 CPS 反对意见**不随布局作废而消失**。用同一份 `segments.json`、同一份生产 v3 布局把三臂重算（`agents/temp/20260812_142856_audit-promote-commit/`）：在屏总秒数 shipped 27,892s / **run1 21,133s** / run3 24,522s，时长中位 3.81 / 2.73 / 3.19s，日文源 CPS 中位 4.29 / 6.07 / 5.08、P90 8.21 / **13.00** / 9.96，CPS>7 占比 17.3% / **40.6%** / 27.3%，短于 20 帧最小显示的 cue 111（1.6%）/ **493（7.0%）**/ 196（2.8%）。**即同一批词，run1 在屏时间比原头少 24%，而这不是旧布局造成的**。反方向也如实记：假名-only 在屏占比在 v3 下三臂几乎持平且 run1 最低（7.91% / 7.88% / 9.04%），与旧布局下的 11.02/10.26/8.82% 排序不同——这一项确实被旧布局扭曲过。CPS 按日文源字符计，中文成品通常更短，绝对值高估阅读负担；三臂共享同一份分块与转写，所以排序本身是干净的。**结论没有改**（run1 仍是当前默认），但代价从「未测」变成「已测且写进当前状态」。

### 晋升提交的审计：一个真 bug、一处死通道、四处过时文档

审计对象是晋升提交本身（`d47fea9`）。核对通过的部分：两份 checkpoint 的 SHA256 与 HISTORY 所记一致、远端回读确认原 `ctc_aligner.pt` 未被改写、HF 钉住的 commit 确实含两个文件、`.revision` 与 `config.py` 默认值一致；加载晋升头实测 `acoustic_only=True`、词表 2,603 类、标点 0 类，因此 `blank_runs(silent_classes=...)` 对默认头是构造性 no-op，只在显式切回原头（10 个标点类）时起作用。

**真 bug：丢弃纯人声 cue 后，两侧的续句标记还在声称自己相邻。** `drop_vocalisation_runs` 只删 cue，不动邻居的 `continues_from_previous` / `continues_into_next`，而这两个标记会以 `cont_prev` / `cont_next` 进入翻译 prompt。一段呻吟被整体删掉后，它两侧的对白就被告知「这两条是同一句的两半」，而中间少的往往是 20 秒以上的音频。八片实测 **513 个被删连续段里 442 个（86%）留下了这种断言**。修法是删完把紧邻缺口的那一侧标记清零：跨越被删音频的连续性是未知，而未知不能报成连续。修后同口径复测 **513 个缺口、0 条仍在声称**，共清理 574 条 cue 的标记，新增 `vocalisation_continuity_flags_cleared` 诊断与三项测试（含「没挨着缺口的 cue 不许被改」和「不得改调用方的 dict」）。

**死通道：v3 晋升后整套 `anchor_aware_dp_v2` 仍留在 `writer.py` 里。** `_split_long_display_block_legacy` 无任何调用者，`_long_display_dp_plan` / `_choose_anchor_for_target` / `_anchor_times` / `_word_gap_anchors` / `_measured_word_text_map` / `_candidate_text_positions_for_dp` 只能从它或彼此到达，仅有一个测试直接调进去。连同随之失效的 `_weak_cut_snap_window_s`、`_candidate_text_boundaries`、`_text_unit_prefix_ratios` 与两组标点常量，`writer.py` 从 1,719 行降到 1,066 行；原文归档在 `agents/rm/20260812_150500_dead-layout-v2-code/`。**同时清掉三个已经不接任何代码的环境变量**：`SUBTITLE_WEAK_CUT_SNAP_SHORT_S/NORMAL_S/LONG_S`（消费方随 DP 一起死了）、`SUBTITLE_MAX_DISPLAY_SHIFT_FROM_ACOUSTIC_START_S` 与 `SUBTITLE_MAX_TOTAL_DISPLAY_EXTENSION_S`（`_SUBTITLE_OPTION_KEYS` 里列着，但 `SubtitleOptions.from_env` 从来没读过）——**一个静默无效的旋钮比没有这个旋钮更坏**。生产侧另清掉 `postgate.ALL_FLAGS`、`llm.errors.TranslationContextLengthError`、`llm.settings._REASONING_EFFORTS`、`gpu_worker._PROFILE_STAGE_BY_SETTING`、`runtime_paths.resource_path` 与一处未使用 import。清理后重跑八片生产布局，**7,016 cue 与各项分布逐项不变**。

**过时文档四处**：README 说对齐头未配置时「退化为按字数比例摊开」（v3 已不做比例回退）、README 的 TTSG 条目仍把最短时长/2 帧间隔/+0.5s 出点列为遵循项（v3 明知放弃）、`web/routes/config.py` 生成的 `.env` 模板同样写着 proportional 回退、`config.py` 的 `MIN_SUBTITLE_DURATION` 注释仍自称「最小显示时长」。均已改写为当前事实并给出量化代价。另外 `tools/sft/*` 四个自训入口从未登记进 README 工具索引（线上默认 ASR 权重正是它们的产物），`compare_head_to_teacher` 只读 `sys.argv` 因而 `--help` 直接抛 IndexError——README 承诺所有工具都能 `--help`，已改为 argparse。

**顶部当前状态里那句 `7,016 cue … 7,257 个内部边界` 也是这次审计改的**：前者是过滤后、后者出自未过滤的 9,008 条，同一 population 下 7,016 − 1,751 = 5,265，两个数摆在一句话里必然误导。

**QC 阈值随之改口径**：v3 明知放弃的两条时间规则用条数阈值必然天天告警，而条数随片长走——八片实测同样 5–10% 的份额在短片是 21 条、长片是 97 条，能放过长片的条数会放过短片上四倍的回归。改为 `spec_duration_under_min_share` / `spec_gap_under_2frames_share` 两个新指标按份额判，默认 `0.15`（现默认头单片最高 10.6% / 9.7%，留了余量），其余 `spec_*` 保持零容忍条数。**换头要重标**：保留的带标点头在同样八片上 gap 份额是 12.6–23.8%，会直接顶到阈值——这正是「换头改变了 cue 形状」该被看见的地方。

**tools 侧的无引用件也一并清掉**：`openai_compat` 的 9 个 helper 加 `env_names`、`binary_clip_audit.copy_clips` 与 `CLIP_DURATION_TOLERANCE_S`、`pause_frame_audit.partition_label_at`、`apply_drop_span_relabels.TOLERANCE_S`，归档在 `agents/rm/20260812_161500_unreferenced-tool-helpers/`。**只留下一个无引用件**：`timestamp_contract.TIMESTAMP_PROMPT_CONTRACT_ZH` 没有任何注入点，但它是 `parse_mmss_timestamp` 所强制格式的唯一人类可读表述，删掉会让严格解析器失去自己的说明书——改为在原处写明「这是要粘进 `--prompt-file` 的正文」。

**`SUBTITLE_LAYOUT_ENGINE` / `SUBTITLE_TIMING_MODEL` 改为拒绝未知值**。它们从来不是开关——没有任何代码按它们分派，值只写进 cue 的版本戳字段。两套布局并存时这个区别不可见：把它设回 `anchor_aware_dp_v2` 得到的是**打着 v2 标签的 v3 输出**，即一份谎报自己出身的产物，比没有这个旋钮更坏。现在 v2 已删，唯一实现只有一个，于是 `SubtitleOptions.__post_init__` 对非法值直接报错并说明「它命名输出、不选择实现」。想回滚只能整体退版本，这一点现在会立刻说出来而不是事后从产物里发现。

全量测试 **1498 passed / 1 skipped**。审计脚本与产物在 `agents/temp/20260812_142856_audit-promote-commit/`。

### 字幕切点改按证据强度选（Layout v3_1），音频切块维持最靠后

提问是「切点既然有概率，能不能不切最末尾的，而是切概率最大的」。**先问对了是哪一层**：第一轮我按音频切块理解并做了一轮离线评估，用户指出指的是字幕层切分，音频送进 ASR 的块本来就该在 30s 内取最大长度。两层的结论正好相反，所以两层都记下来。

**音频层：维持「窗口内最靠后的合法停顿」，理由是块长本身就是目标。** 用归档停顿离线复算（`agents/temp/20260812_180000_cut-tracking/baseline_cut_provenance.py`）：把选点换成「窗口内最宽」，块长中位数 A 26.6→18.4s、V 27.8→19.5s、B 27.8→19.9s、J 26.5→20.5s、Z 27.7→22.3s，八片全部下压。**这正落进 2026-08-02 实测伤转写的区间**（20s vs 30s：`成人になるために` 读成 `政治になるために`、纯人声块幻觉 `カレロンか`、`repeated_unit` 10.5%→14.6%），因为 `max_s` 就是 encoder 的音频窗口，短块会被 padding 补回去。此外这一层也没有概率可用：`blank_runs` 只对 argmax 取游程，归档的 `pause_spans/*.json` 只有区间没有后验，要排序必须重跑 GPU 前向。

**字幕层：DP 原本确实丢掉了它已经算出来的概率。** `_exact_boundary_kind` 同时返回边界类型和实测静音长度，但代价函数只用类型——句末 0.0 / 强停顿 0.05 / 分句 0.10 / 词间隙 0.20——于是 0.12s 和 0.59s 的词间隙同价，谁胜出改由「谁更能填满 20 字」决定。八片 7,257 个内部边界实测：只有 45.8% 落在「最靠后的合法边界」上（所以 DP 从来不是纯贪心），但 18.6%（1,351 个）的切点后面还有更宽的静音仍在字数上限内没被选。

**改法只动 `word_gap` 内部**：`0.20 + 0.20 × (0.6 − gap)/0.6`，即 0.12s 罚 0.36、0.60s 罚 0.20，其余类别不动。词间隙是唯一「全部依据就是那段静音」的类别，而 0.12s 下限已经贴近连续语流里音节之间的间隔。**把同样的加权推广到所有类别的版本先做了、再被否掉**：它把边缘切点降到 353（对 330），代价是 134 个切点从写出来的逗号挪到声学停顿上——拿语法换静音，方向错了。

生产实现八片复算（`verify_shipped_layout_change.py`，跑真实 `prepare_srt_blocks` 并开启纯人声过滤；对照臂把罚函数拍平回旧值，两臂只差这一项）：

| | 旧（v3） | 新（v3_1） |
| --- | ---: | ---: |
| cue 总数 | 7,016 | 7,016 |
| 落在 0.12–0.2s 边缘静音的切点 | 399 | **301**（−24.6%） |
| 词间隙切点的静音中位数 | 0.192s | **0.269s** |
| 分句标点切点 | 776 | 794 |
| 字数 p50/p90，超 20 字 | 17/20，41 | 17/20，41 |
| 时长 p50/p90，超 7s | 2.731/5.654s，34 | 2.692/5.615s，34 |
| 重叠 / 低于 2 帧间隔 / 短于最小显示 | 0 / 553 / 493 | 0 / 556 / 487 |
| 续句声明 | 3,591 | 3,588 |

即：边缘切点少了四分之一，其余各项在 ±3 条以内，比例回退与文本保真仍是 0 / 100%。**版本戳因此从 `measured_safe_boundary_dp_v3` 升到 `measured_safe_boundary_dp_v3_1`**，旧名字和其它未知值一样被拒绝——两版有约 1.4% 的切点落点不同，让旧名字通过就是把新布局的产物标成旧布局的。

**追踪按用户要求加在质量报告里，两组其实是一件事。** `continues_into_next` 的定义就是「这条不是以句末标点结束的」，所以 break type 分布正是续句数量的成因：报告现在同时记 `layout_break_type_counts`、`layout_word_gap_cut_count` / `_under_0p2s` / `_median_s`，与 `cue_continues_*` 三项、`vocalisation_runs_dropped`、`vocalisation_continuity_flags_cleared`（后者是**撤回**的声明数，脱离声明总数没有意义）。音频层的切点来源同样进报告（`chunk_cut_*`、`chunk_duration_*`），硬切份额八片 0.7%–53%（V 0.7% / AA 53.4%），跨度来自片子本身，因此**只观测不设阈值**；它随 `asr_details` 透传而不写进 `boundary_signature`，因为那是缓存键，塞进去会让所有已缓存任务失效。全量测试 **1515 passed / 1 skipped**。

### 复现步骤

```powershell
$env:PYTHONIOENCODING = "utf-8"
# 三臂固定 30s 分块、同一份转写，只换头
uv run python agents/temp/20260812_081331_fixed30-head-factorial/run_fixed30_arm.py
uv run python agents/temp/20260812_081331_fixed30-head-factorial/aggregate_fixed30.py
# 切点落点与「最宽停顿」对照
uv run python agents/temp/20260812_081331_fixed30-head-factorial/evaluate_cut_policies.py
# 用生产 v3 布局从 segments.json 重排三臂（判旧布局指标作废的那一步）
uv run python agents/temp/20260812_081331_fixed30-head-factorial/evaluate_exact_joint_layout_grid.py
# 音频切点来源基线（含「最宽停顿」对块长的影响）与生产实现回读
uv run python agents/temp/20260812_180000_cut-tracking/baseline_cut_provenance.py
uv run python agents/temp/20260812_180000_cut-tracking/verify_provenance_matches_spans.py
# 字幕层三种边界罚函数对照，以及生产 prepare_srt_blocks 的新旧逐片对照
uv run python agents/temp/20260812_180000_cut-tracking/evaluate_layout_boundary_policy.py
uv run python agents/temp/20260812_180000_cut-tracking/verify_shipped_layout_change.py
```

**两个口径陷阱**：①比较字数/时长前先确认输入 cue 不带 clamp 与比例回退，否则测的是旧布局而不是头；②「一次编码多头同测」只保证音频相同，**还要逐片核对三臂的 spans hash 与 transcript hash 全等**（本轮 8/8 相等）——只要有一片不等，那片的差异就可能来自分块或 ASR 文本，整组结论都不能按「只换了头」来读。

## ASR / Alignment 文本策略

当前策略来自 v1.8 / v1.9 的清理。

原则：

- `display_text` 是最终字幕显示文本，只做展示安全处理。
- `align_text` 是 forced aligner 专用文本，可删除标点、emoji、装饰符、音乐符号和不可发音标记。
- 不使用具体字样黑名单。
- 不直接删除目标域常见短促发声、喘息、呻吟、拟声和低信息短句。
- 重复循环、低置信、文本/音频比例异常、align-text-empty、forced-aligner fallback、`asr_review_uncertain` 默认只作为 QC / 诊断 / 样本池信号，不再触发最终字幕文本删除。
- forced aligner 失败时不伪造精确时间轴，保留 fallback quality label。

失败样本池闭环：

```text
diagnose_asr_alignment.py
-> failure_candidates.jsonl
-> export_alignment_failure_manifest.py
-> materialize_alignment_failure_audio.py
-> 人工审计 / hard-negative / 下轮 VAD 或 ASR 数据
```

---

## 字幕时间轴

当前组合为 `measured_safe_boundary_dp_v3_1` + `measured_lexical_extent_v3`：

- cue plan 在 LLM 翻译前冻结，翻译始终逐条对应，不改变时间轴。
- cue 起点和声学终点来自首个、末个实测发音字，不按字符比例伪造时间点。
- 显示终点可以在末字后的实测静音中延长最多 0.5 秒，但不得越过下一条前 2 帧，也不得借此突破 7 秒软目标。
- 内部切点优先使用句末、强停顿、分句标点和可靠词间隙；没有安全点时保留较长 cue。
- 上游只有粗时间或对齐失败时保留 fallback quality label；后置 timeline polish 只处理这些未锁定块，不能冒充实测字词时间。
- 当前布局不以移动实测语音边界的方式强行满足最小时长或最小间隔；顺序性与不重叠仍是硬约束。

## 常见坑

- Windows 必须使用 FFmpeg Shared；只有 `ffmpeg.exe` 而缺少 `avcodec/avformat/avutil` 共享 DLL 时，TorchCodec 会在运行期失败。
- 是否真正使用 GPU 只看运行时证据，例如 `actual_device=cuda`、`model_param_device=cuda:*` 和 llama-server 的设备枚举，不看目录名或配置名。
- `ASR_BATCH_SIZE=auto` 会根据显存和既有 profile 调整批组成；需要严格对比两次转写时应固定 batch，并明确缓存是否命中。
- 改动对齐、词时间或字幕布局逻辑时必须同步更新对应缓存签名；否则新任务可能逐字节复用旧产物，看起来像代码没有生效。
- 长任务应由前台进程持有并持续写日志；不要启动静默后台进程后立即退出宿主 shell。
- 大型 JSONL/sequence 数据应流式读取并使用紧凑 tensor 或 tensor cache，避免一次性构造 Python 对象列表。
- 模型下载、依赖安装或外部 API 失败时先检查网络与代理；正式 GPU 工作流不得为了绕过错误而静默退回 CPU。

---

## 参考来源

- WhisperJAV: <https://github.com/a63n/WhisperJAV>
- FusionVAD: <https://arxiv.org/abs/2506.01365>
- Whisper hallucination on non-speech: <https://arxiv.org/abs/2501.11378>
- Dynamic Speech Endpoint Detection: <https://arxiv.org/abs/2210.14252>
- Semantic VAD: <https://arxiv.org/abs/2305.12450>
- WhisperX: <https://github.com/m-bain/whisperX>
- stable-ts: <https://github.com/jianfch/stable-ts>
- Qwen3-ASR: <https://github.com/QwenLM/Qwen3-ASR>
- Qwen3-ASR finetuning: <https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning>
- Qwen3-ASR-0.6B: <https://huggingface.co/Qwen/Qwen3-ASR-0.6B>
- Qwen3-ASR-1.7B: <https://huggingface.co/Qwen/Qwen3-ASR-1.7B>
- 本项目 Qwen3-ASR-0.6B SFT: <https://huggingface.co/jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame>
- 本项目 Qwen3-ASR-1.7B SFT: <https://huggingface.co/jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame>
- AVA-Speech VAD: <https://huggingface.co/datasets/nccratliri/vad-human-ava-speech>
- VoxConverse: <https://huggingface.co/datasets/diarizers-community/voxconverse>
- MUSAN: <https://www.openslr.org/17/>
- DNS Challenge: <https://github.com/microsoft/DNS-Challenge>
- pyannote speaker diarization: <https://huggingface.co/pyannote/speaker-diarization-3.1>
- 3D-Speaker: <https://github.com/modelscope/3D-Speaker>
- WeSpeaker / CAM++: <https://github.com/wenet-e2e/wespeaker>
- Reazon Japanese HuBERT: <https://huggingface.co/reazon-research/japanese-hubert-base-k2>
- rinna Japanese HuBERT: <https://huggingface.co/rinna/japanese-hubert-base>
- rinna Japanese wav2vec2: <https://huggingface.co/rinna/japanese-wav2vec2-base>
- NonverbalTTS: <https://arxiv.org/abs/2507.13155>
- Rochester non-word transcription notes: <https://www.cs.rochester.edu/research/speech/nonwords.html>
- Switchboard transcription guidelines: <https://isip.piconepress.com/projects/switchboard/doc/transcription_guidelines/transcription_guidelines.pdf>
