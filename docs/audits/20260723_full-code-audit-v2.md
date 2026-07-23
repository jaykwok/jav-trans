# 2026-07-23 全量代码审计 v2

本审计覆盖本轮 Grok 4.5 结论涉及的 Boundary/CueQC 模型链、训练数据生成与
合并、checkpoint/summary 导出、审计静态服务、旧代码入口和文档。范围限定为
1.7B 当前源码；0.6B 仍是空 registry placeholder，本轮没有训练、改权重或改
生产 registry。

## 当前合法入口

| 阶段 | 当前职责 | 训练/运行状态 |
| --- | --- | --- |
| Scorer v11 | candidate-island membership | v5 teacher；真实 source gate 未通过，registry 为空 |
| Proposal v1 | 高召回候选审计 | audit-only；不能作为已晋升的训练或 final-cut truth |
| Outer v3 | post-Scorer island outer edge | 合同和 trainer plumbing 已有，缺当前数据与人工 gate，registry 为空 |
| Split v4 | candidate-level cut/continue event | 只接受当前 sequence summary；`unsure` 映射 `-100` |
| CueQC v13 | provisional sub-island keep/drop | 二分类 argmax；teacher unsure 不进 loss/metrics |
| Inner v2 | post-CueQC keep-only acoustic core | 只在 CueQC keep 后运行；当前 promotion 仍关闭 |

Runtime 顺序固定为：

`Scorer → Proposal → Outer → Split → provisional sub-islands → CueQC → Inner`。

## 本轮修正

### Split 数据合同与训练器

- `acoustic_split_v4_dataset.py` 新增当前 summary schema、input distribution、
  PTM/MFCC/scalar geometry、finite/shape/label/partition/group/source/core/timing
  校验，以及 dataset/frame-sidecar SHA 校验。
- `compile_joint_boundary_preasr_dataset.py` 现在要求每个 approved window 和
  feature NPZ 同时声明当前 Boundary contract、PTM repo、输入分布、feature
  schema、音频 SHA 与 Scorer/Proposal/Outer 三个 SHA；缺失、混用或 stale
  provenance 直接拒绝。
- compiler 拒绝重复 `(window_id, feature_index)` 标签，拒绝未知标签和越界/非
  单调候选坐标；每条 label 的 `time_s` 还必须是有限数值，并与绑定 NPZ 中同一
  `feature_index` 的 candidate time 在 `1e-6s` 内精确一致，避免旧 candidate
  export 的标签按索引误投影。输出使用临时 dataset/sidecar、原子替换，并写
  companion summary 和历史 alias。summary 记录实际 dataset/sidecar SHA、source
  feature/audio binding、label prompt counts 和分区统计。
- Split trainer 只加载 `require_training_summary=True` 的当前数据；seed、CUDA
  fail-fast、正值参数、normalization/logit/probability/loss finite 检查已补齐。
  没有 definite label 的 group 不进入训练；空 role target 不计算辅助 loss，
  不再用 `nan_to_num` 掩盖坏 loss。checkpoint/metrics 原子写入，metadata 明确
  dataset summary/SHA、三个上游 SHA、`training_manifest_allowed=true` 和
  `promotion_ready=false`。
- merge 工具拒绝旧 row-wise、rehydrate、audit-only、缺 summary、geometry 或
  upstream SHA 不一致的数据；保持 whole-island/source/core isolation，限制
  role 唯一，pair id 重编号，输出同一 current summary schema 并绑定所有输入
  summary/path/SHA。

### 训练数据导出与 CueQC merge

- `prepare_joint_boundary_omni_dataset.py` 和
  `export_semantic_boundary_candidates.py` 不再把“设置环境变量”当成导出成功。
  feature 输出先写临时路径，必须同时出现 Split NPZ 与 metadata（以及请求的 raw
  PTM/MFCC sidecar）才原子替换；当前 Runtime 尚未实现 candidate feature exporter
  时明确报 pending/audit-only 错误，旧文件不会被静默复用。source-bound feature
  输出要求恰好一个 audio，避免多 audio 覆盖同一路径。
- `merge_pre_asr_features.py` 增加 tensor geometry、mask、label、group/core 和
  finite 校验；row id 必须唯一且恰好归属一个 group，group 的
  `source_core_ids` 必须等于所属 rows 的 core 精确并集，因此一个 chunk 对应多个
  core、或 core 数少于 chunk 数的合法数据不再被错误拒绝。权重、summary 与
  source bundle SHA 均原子写入。
- CueQC v13 teacher resume 现在逐行核对 teacher schema/prompt/model、source/audio、
  坐标/时长、Split/Inner checkpoint SHA 与 Boundary contract；旧 checkpoint 或
  Runtime 换源后不能续写。canonical compiler 同样只接受当前 CueQC v13 teacher，
  拒绝旧通用 `pre_asr_omni_label_v1`，并在 summary 固化 teacher schema、prompt 和
  Omni model identity。
- joint Pre-ASR compiler 的二次 payload 保存改为原子 checkpoint 写入，并在冻结
  partition 写回完成后重新计算最终 `output_sha256`，summary 不再指向旧字节。

### 审计与死代码

- 保留 HTML JSON 安全转义、静态 audit server 的路径/Range/Content-Length 防护
  和已有 SHA/身份校验。
- AST 审计覆盖 `src/` 与 `tools/` 的 270 个 Python 文件：无 parse error、无零
  引用的 private 顶层定义。确认无生产调用的旧 joint Omni v2 prompt、
  missed-boundary normalizer 及其专属测试已移入
  `agents/rm/20260723_105300_retired-dead-platform-and-joint-v2/`。
- Unix-only `tools/audits/serve_audits.sh` 已移入同一归档目录；Windows 入口为
  `tools/audits/serve_audits.ps1`。Proposal、rehydrate、旧 Runtime audit
  builder 等仍被审计复现引用的 legacy 工具没有误删，并继续标记 audit-only。
- README 的 Python 版本从过时的 3.13+ 修正为 3.14+，安装入口改为与
  `pyproject.toml`/lock 一致的 `uv sync`。

## 仍然 pending 的问题

1. 当前 Boundary backend 尚未真正实现 `SEMANTIC_SPLIT_FEATURE_EXPORT_PATH` 对
   candidate frame/scalar/metadata 的写出。因此不能生成或晋升新的 Split v4
   training manifest；fail-closed 是预期状态，不是训练成功。
2. Scorer v11 真实 full-source teacher/zero-clipping gate 未通过，Proposal 仍
   只作 audit replay；Outer v3 需要 no-Outer/edge-only/current 消融和人工 gate。
3. Split/CueQC/Inner 现有 checkpoint 不能通过 rebind SHA 冒充当前上游重训结果；
   没有任何 production registry 变更。

## 验证

以下命令均使用项目 `.venv`、PowerShell 和 `PYTHONIOENCODING=utf-8`：

```powershell
uv run python -m compileall -q src tools tests
uv run pytest -q
git diff --check
```

AST 静态审计脚本为
`agents/temp/20260723_105300_repo_static_audit.py`；其结果写入同名 `.json`。
最终结果为 `978 passed, 6 skipped`；仅有 4 条既有 SciPy
sparse-efficiency warnings，不影响测试结论。
