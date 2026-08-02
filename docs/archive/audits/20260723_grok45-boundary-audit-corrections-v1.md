# Grok 4.5 Boundary 审计比对与修正

本页只覆盖 1.7B。0.6B 空 registry placeholder、checkpoint、runtime 和 data 均未修改或训练；中央序列化合同仍只有 `boundary_acoustic_binary_v12`。

## 结论对照

| Grok 审计意见 | 本地证据复核 | 本轮处理 |
| --- | --- | --- |
| Scorer 不应把所有呻吟、喘息或含混 vocal 自动视为 inside | 接受。当前职责应按“词语/对白锚点 + 同一轮连续波形包络”定义 inside；明确无词义且可独立删除的纯非语义声可以 outside；词语与呻吟/噪声无法区分时必须 unsure | active teacher锁回v5；README、Scorer合同页和训练数据审计统一三态定义。`unsure→-100`，不进normalization/loss/metrics/gate |
| v6 日语/JAV领域提示会诱发过度合并，不应直接替换v5 | 接受。相同25条A/B中合法23条inside占比`26.74%→34.37%`、unsure`5.93%→0%`，并出现3条full-source inside；另2条返回原视频坐标而fail-closed | 默认`PROMPT_VERSION`恢复`dialogue_islands_v5`；v6请求与A/B页只保留为实验证据，不参与resume或新训练入口 |
| 一次Semantic Timeline teacher输出不能同时成为Scorer/Outer/Split/Inner当前训练真值 | 接受。各模型职责、输入分布和可验证监督不同；旧view compiler即使人工通过也可能误报`training_ready=true` | compiler固定为`legacy_inventory_only`，所有派生row和summary写`training_manifest_allowed=false`，只保留审计/provenance用途 |
| Proposal v1的smoothing/local-max/quantile/NMS/snap承载候选recall，不能当作无影响后处理 | 接受，但本轮不改checkpoint或注册。现有99.12% aggregate recall不对应最终Scorer v11输入，不能据此晋升或直接判废 | 继续保持audit-only；等Scorer通过gate后在真实island重放miss，再决定two-logit candidate或learned event-query替代。禁止新增threshold/NMS规则路线 |
| Outer当前all-background整岛drop与Scorer/CueQC职责重叠 | 接受。Outer registry仍为空，缺少真实post-Scorer-v11输入分布 | 本轮不训练、不注册、不改权重；后续只做no-Outer / edge-only / current三臂职责消融 |
| Inner实际在CueQC前预计算，违反post-CueQC keep-only合同 | 接受，源码调用顺序已确认 | 已改为Outer→Split→provisional→CueQC→Inner。Inner只接收CueQC argmax keep项，drop项不会进入Inner模型 |
| 移动Inner不能通过重复PTM、隐式全局状态或规则fallback绕过feature生命周期 | 接受 | Boundary cache继续保存provisional chunk JSON，并新增同一audio/model/config digest绑定的raw PTM/MFCC NPZ sidecar；cache hit显式重建feature provider，sidecar缺失/损坏时fail-closed重跑Boundary |
| pre-CueQC数据和审计工具不应继续假装已有Inner结果 | 接受 | provisional schema升级为`runtime_v12_provisional_subisland_v2`，删除`inner_edge_prediction`并写`inner_execution_status=deferred_until_cueqc_keep`；teacher拒绝旧v1。CueQC drop safety第二抽样轴改为上游Scorer membership，不再读取Inner概率 |

## 未改变的路线

- Scorer v11仍是two-logit softmax argmax；没有新增runtime threshold、hysteresis、duration merge、hard veto、NMS或fallback。
- Scorer full-capacity主臂仍是raw PTM2048全宽adapter + hidden256双向Mamba；本轮失败证据指向真实full-source train监督和partition泛化，不用结构改名或loss调参掩盖。
- Split v4仍保留显式candidate query与island sequence baseline；只有最终Scorer/Proposal/Outer输入闭合后才重编数据和比较架构。
- CueQC和Inner现役checkpoint未修改；执行顺序修正不等于现役provenance已通过新数据合同，仍须在最终上游上重新gate。

## 回归合同

- active Scorer teacher必须为v5，v6行不得被resume混入。
- cache sidecar必须精确round-trip raw PTM/MFCC；功能cache仍只使用中央合同ID与内容签名，不恢复整数generation。
- CueQC drop必须在Inner输入集合之外；stage顺序必须为Outer、Split、CueQC、Inner，且各模型结束后显式释放。
- 旧Semantic Timeline多模型view即使全部人工approve，也不能变成current training manifest。
- Split teacher label必须同时绑定`feature_index`和candidate `time_s`；只按索引匹配的旧candidate export不得进入训练。
- CueQC teacher resume与canonical编译必须绑定当前schema/prompt/model、source/audio、坐标、Split/Inner SHA和Boundary合同；旧通用Pre-ASR teacher不得混入v13。
- CueQC feature merge按每行`source_core_ids`核对group精确并集，不再错误假定“core数等于chunk数”。

## 最终验证

- 在项目根目录 `.venv` 中以 `PYTHONIOENCODING=utf-8`、项目内临时
  `UV_CACHE_DIR` 运行全量回归：`978 passed, 6 skipped`；仅有既有 SciPy
  sparse-efficiency warnings。
- `git diff --check` 通过。
- 另加了 raw feature shape gate：post-CueQC Inner/cache sidecar 必须是精确
  `PTM2048`；旧 `PTM128` sidecar 现在 fail-closed，不会静默复用。
- 本轮没有训练，也没有修改 0.6B checkpoint/runtime/data/registry；没有提交或推送。
