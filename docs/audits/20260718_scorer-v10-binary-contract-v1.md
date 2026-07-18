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

The first version of this audit was too strict when it treated the frozen `source_manifest_stratified1024.jsonl` as identity-only. The corrected responsibility is important: Galgame, especially NSFW material, is real target speech and closely matches the JAV acoustic domain. The selected cores came from the teacher-approved semantic-core inventory. For the coarse Scorer task, their exact composition extents are valid positive `speech` supervision; this does not reinstate the rejected Inner assumption that every arbitrary Galgame clip is already a precisely trimmed acoustic semantic core.

The fixed1024 sources already contain the required hard distribution:

- `1024 sources / 2048 unique cores / max core use=1`, partitioned `870 train / 103 val / 51 test`;
- two Galgame speech cores per source, with exact sample coordinates;
- partitioned high-confidence CueQC/Omni `definite_drop` assets in the central negative unit and both gaps, including moaning, breathing, kissing, crying, music, impact and other JAV-domain negatives;
- additive real-negative overlays on 504 selected sources. An overlay does not change a core frame from speech when clear target speech is still present.

`compile_speech_island_scorer_v10_canonical.py` now turns those exact spans into canonical frames and adds strict all-background controls from the same frozen negative pool. A frame whose exact 20 ms cell crosses two differently labelled sample spans becomes `unsure`; no wider boundary band is invented. The prepare result at `agents/temp/20260718_230743_scorer-v10-fixed1024-canonical-prepare/` is:

| item | result |
| --- | --- |
| sources | `2671` |
| unique cores / max use | `2048 / 1` |
| speech rows train/val/test | `870 / 103 / 51` |
| all-background rows train/val/test | `1234 / 210 / 203` |
| canonical frames | `speech 496546 / background 144326 / unsure 2022` |
| background identities / videos | `1647 / 148` |

The compiler rejects background video/asset or core identity crossing partitions, non-strict negative assets, missing audio, missing/old central contracts and non-1.7B/non-PTM2048 feature caches. The 1542 background assets reused as augmentation and isolated controls remain inside the same frozen partition; this is reported explicitly and creates no train/held-out crossing. `unsure` remains in canonical assets but is zero-weighted for feature-cache statistics and maps to `-100` for normalization, loss, metrics and gate. Finalization also requires a passing manual-gate summary bound to the exact canonical manifest SHA, so merely generating the audit page cannot unlock a trainer manifest.

This is a canonical source plan, not a trained dataset claim. Raw 1.7B PTM2048+MFCC40 feature cache and the final trainer manifest do not exist yet.

## Plumbing smoke

`agents/temp/20260718_220006_scorer-v10-binary-plumbing-smoke/` contains six synthetic rows: one speech and one all-background source in each partition, three unique semantic cores, three unique background identities, and 12 unsure frames. A CPU one-step smoke with production input widths completed and strict loader replay passed. Its summary is deliberately `numeric_gate_pass=false / gate_pass=false / promotion_ready=false`; CPU shared VRAM is marked not applicable and post-release RAM is recorded. The local checkpoint is synthetic and must not be committed, registered or used as a teacher.

## Human audit status

The canonical fixed24 page at `agents/audits/20260718_231220_scorer-v10-canonical-data-fixed24/` contains four speech and four all-background controls from every partition. It is playable and saves `speech_scorer_v10_canonical_manual_verdict_v1`; its status is deliberately `pending`, and no listening verdict is claimed.

After a real v10 model exists, the same audit framework must generate pages for every prediction-drop/truth-speech case, all held-out hard cases, edge clipping and every greater-than-8-second residual. Numeric gates are capped at 95%; zero clipping and zero true-speech deletion require saved human verdicts before promotion.
