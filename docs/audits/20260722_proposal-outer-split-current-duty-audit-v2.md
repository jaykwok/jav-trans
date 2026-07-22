# 1.7B Proposal / Outer / Split 当前职责审计 v2

本页只覆盖 1.7B。0.6B registry placeholder、checkpoint、runtime 和 data 均未修改或训练。所有序列化与 cache 兼容仍只使用 `boundary_acoustic_binary_v12`。

## 当前结论

- Scorer v11 已完成真实音频几何修复、1470-source raw PTM2048/MFCC40 cache 和1667-window signed dataset compile，但尚未训练或晋升。因此 Proposal、Outer、Split 不能先用旧 Scorer 分布重训后再改 metadata。
- Proposal v1 暂不直接改权重。它的旧数值 gate 为 aggregate candidate recall `99.12%`、最弱 bucket `96.39%`，均高于最高95%的数值要求；但约98个 eligible truth boundary未被提出，尚无完整人工 zero-clipping 证据，而且上游职责已经从碎片 speech scorer 改为连续 candidate-island membership。旧结论不能外推到 Scorer v11。
- Outer v3 registry 继续为空并保持 `pending_outer_v3_audit`。现有随机初始化 binary scaffold职责正确，但必须从晋升后的 Scorer v11完整 island生成真实边缘数据后，才比较现有 Linear128/H128 与更高容量 adapter；不能因为 Scorer P2048/H256可行就机械复制。
- Split v4 的 candidate-query结构本身可保留为基线：它显式编码 left/gap/right及多尺度 bins、candidate scalar arm、candidate内Mamba和island candidate序列Mamba，不是只做整个context mean/max。问题在输入语义和数据 provenance，而不是缺少时序建模。

## Proposal v1

现役 checkpoint SHA=`e8595cd7cdc2106562cf2ac3a37cf88e18fcc62ebe635ca12df3314c78df139b`，是单 `boundary_prob` logit的 PTM128+MFCC40 Mamba scorer；训练使用 Focal gamma=2、positive weight=30，metadata指向已不存在的2026-07-04 smoke labels/features。

源码 `src/boundary/ja/backend.py::_attach_split_proposals()` 在模型分数之后还会执行：

1. 80ms smoothing；
2. local maximum与`prominence>0`；
3. island内score/prominence各10% quantile floor；
4. 120ms NMS；
5. 在Scorer概率低谷附近snap；
6. 距island边缘至少80ms。

这些操作不做最终cut，但会永久删除Split看不到的候选，因此是recall-bearing runtime decision，不能仅以“non-binding”免责。当前 full-workflow CLI 已移除并主动清除这些旧 quantile/smoothing/NMS/snap/min-distance控制项，防止它们继续伪装成可调生产操作点；legacy decoder仍只留给离线重放。

Scorer v11晋升后应先在真实完整candidate island重放旧Proposal，输出所有missed truth boundary、held-out hard case和长island可播放审计页。若人工零误删通过，才有证据保留v1；若不通过，优先替换为两logit frame candidate/non-candidate argmax或learned event-query模型，再把连续candidate argmax run聚成一个查询event。替代方案不得使用probability threshold、quantile、prominence floor、NMS或duration veto。训练必须固定source/core/partition，boundary邻域监督与同island hard negatives同时存在，unsure映射`-100`。

## Outer v3

Outer职责是完整 Scorer candidate island到供Split查询的acoustic outer core，不决定Display。当前模型为raw PTM2048→checkpoint Linear128 + MFCC40 + relative position→valid-prefix bidirectional Mamba2→Linear(2)，runtime二分类argmax；teacher unsure映射`-100`。

架构没有足够证据立即升级。Outer只查询两个边缘，容量需求可能低于Scorer；正确顺序是先生成真实 `post_candidate_island_scorer_v11_islands`，固定相同数据/seed/steps/plain CE后比较P128/H128与更宽adapter，且以start/end coverage最高95%和人工zero clipping为准。Focal、boundary band或auxiliary只有固定数据A/B明确改善才采用。

## Split v4

现役 checkpoint SHA=`d35844084d434cd7796b0af269a41044016bd2bae9bc901f469bfe273d0277c0`，binary `cut/continue` softmax argmax、unsure excluded。模型内含trainable Linear(2048→128)、candidate frame-bin encoder、结构化readout（global mean + gap mean + left-right contrast）、scalar arm、candidate内双向Mamba和island candidate序列双向Mamba。

必须重编而不能rebind的原因：

- 旧13维scalar前四项是`candidate_score/prominence/speech_valley/strength`。其中prominence/strength来自启发式Proposal后处理；`speech_valley`来自旧speech概率。Scorer v11输出的是candidate-island membership，不再具有“speech valley”语义。
- 其余`left/right/gap_speech_*`也需要显式改名并按v11 membership概率重新定义，避免相同shape掩盖语义断兼容。
- 旧训练provenance含 Focal=1.5、cut weight=3、pair/role auxiliary和`manual_group_repeat=32`；新数据应先跑plain CE、weights 1/1、repeat 1的neutral baseline。
- 当前连续argmax=cut run取最高`p_cut`代表event，不依赖threshold，作为同一学习事件的确定性聚合可以保留；但必须新增不同batch size、full/partition inference的顺序、argmax和probability等价测试。

## 执行与生命周期

- Boundary、Outer、Split、CueQC、Inner的`device=auto`过去会在CUDA不可用时静默落到CPU。现统一为auto必须解析到CUDA，否则fail-fast；显式`device=cpu`只保留给单元测试和审计smoke。
- pipeline已在Outer、Split、Inner和CueQC后显式释放model并清allocator，但当前Inner预测实际在CueQC之前预计算，只在CueQC keep过滤后应用。这与“Inner只接收post-CueQC keep sub-island”的职责不一致，也浪费被CueQC drop样本的推理。由于feature provider生命周期目前封装在Boundary阶段，本轮先记录为Outer/Scorer晋升后的必改项，不用规则或重复PTM提取临时绕过。
- README已更新Scorer v11主臂为P2048/H256并把Proposal v1写为审计参考；full-workflow不再公开旧Proposal启发式旋钮。

## 后续顺序

1. 完成Scorer v11 compact/full相同条件训练和人工gate。
2. 用最终Scorer重放Proposal v1并生成miss audit；据证据保留或换成全学习型candidate/event query。
3. 生成真实post-Scorer island数据，训练/gate Outer v3。
4. 用最终Proposal/Outer生成新schema Split candidate-query数据，移除旧heuristic scalar语义后先跑neutral baseline。
5. 将Inner推理真正移动到CueQC keep之后，再重跑CueQC/Inner fixed partition gate和batch/lifecycle回归。
