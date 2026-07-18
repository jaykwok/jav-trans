# Speech Island Scorer v10 Binary Contract Audit

本页只覆盖 1.7B。0.6B 空 registry、checkpoint、runtime 和 data 未读取、未修改、未训练。中央序列化兼容键仍只有 `boundary_acoustic_binary_v12`。

## Decision

Scorer v10 的断兼容 schema、两 logit模型、argmax decoder、严格 dataset validator 和 random-init trainer plumbing 可以保留；production registry 与 `segment()` 继续 fail-fast 为 `pending_binary_scorer_audit`。没有启动 GPU 或真实训练，也没有可晋升 checkpoint。

## Why v8/v9 cannot continue

Scorer v8 是单 sigmoid logit，历史高召回依赖 `0.15/0.20` hysteresis threshold 和 dilation；训练 threshold `0.5` 的 recall 只有 `89.78%`。其 `53,760` manifest 行只有 `33,096` unique audio/features，随机 row split 造成 `328` 个 exact train/eval overlap，前 512 eval 中 `144` 个已在 train。v9 又是从 v8 warm-start 的 content/membership 六 logit双头，runtime 把三类 membership 中所有非-outside argmax 都视为保留；它不是新的 binary contract。两者都不能 warm-start v10，也不能提供冻结 gate。

## Implemented candidate contract

| surface | v10 candidate |
| --- | --- |
| schema | `speech_boundary_ja_binary_island_scorer_v10` |
| labels | canonical `background / speech / unsure`; training `background / speech`; `unsure=-100` |
| model | raw PTM2048 -> checkpoint-owned trainable Linear128 + MFCC40 + relative forward/backward position -> valid-prefix bidirectional Mamba2 -> Linear(2) |
| decision | two-logit softmax frame argmax; contiguous speech frames form coarse islands; all-background drops the source |
| runtime rules | no sigmoid operating point, hysteresis, dilation, minimum-duration filter, hard veto or fallback |
| initialization/loss | random initialization; weighted cross entropy baseline only; no warm-start/Focal/boundary band surface |
| batching | valid-prefix padded batches preserve source order, probabilities within `1e-5`, and exact argmax |

The builder and loader reject non-2048 PTM, non-128 projection, non-40 MFCC, non-position2, missing valid-prefix masking, non-random initialization, wrong labels, wrong dataset contract and missing/retired central contract IDs. The pure decoder ignores threshold, dilation and minimum-duration config values.

## Dataset audit

The frozen `source_manifest_stratified1024.jsonl` is useful only as an identity/partition base: it has `1024 sources / 2048 unique cores / max core use=1` and source partitions `870 train / 103 val / 51 test`. It is not a Scorer training manifest:

- every selected source is a two-core semantic composite, so no partition has an all-background full-source control;
- its `core_spans` are exact composition/sample extents, not independently verified frame-level speech boundaries; labeling every frame inside them as speech would repeat the rejected arbitrary-full-clip assumption;
- the Inner v2 frame manifest contains post-old-chain sub-islands, not full Scorer inputs, so it has Scorer/Outer/Split selection bias and cannot repair the missing full-source negatives.

The new validator therefore requires one frozen full-source row per source, explicit `row_role=speech|all_background`, unique semantic `core_ids` only on speech rows, unique `background_id` on all-background rows, fixed source/core partitions and max core use 1. Each partition must contain both roles. Canonical labels must come from a new auditable frame teacher/annotation pass; unsure is retained in the label asset but excluded from normalization, loss, metrics and gate.

## Plumbing smoke

`agents/temp/20260718_220006_scorer-v10-binary-plumbing-smoke/` contains six synthetic rows: one speech and one all-background source in each partition, three unique semantic cores, three unique background identities, and 12 unsure frames. A CPU one-step smoke with production input widths completed and strict loader replay passed. Its summary is deliberately `numeric_gate_pass=false / gate_pass=false / promotion_ready=false`; CPU shared VRAM is marked not applicable and post-release RAM is recorded. The local checkpoint is synthetic and must not be committed, registered or used as a teacher.

## Human audit status

No listening verdict is claimed because no real v10 prediction set exists. After a real canonical smoke, the existing `tools/audits` HTML/navigation framework must generate playable/saveable pages for every prediction-drop/truth-keep case, all held-out hard cases and every greater-than-8-second residual. Numeric gates are capped at 95%; zero clipping and zero true-speech deletion require saved human verdicts before promotion.
