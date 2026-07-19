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

This is now a manually gated canonical source plan, not a trained-dataset or model claim. Raw 1.7B PTM2048+MFCC40 feature cache and the final trainer manifest do not exist yet.

## Plumbing smoke

`agents/temp/20260718_220006_scorer-v10-binary-plumbing-smoke/` contains six synthetic rows: one speech and one all-background source in each partition, three unique semantic cores, three unique background identities, and 12 unsure frames. A CPU one-step smoke with production input widths completed and strict loader replay passed. Its summary is deliberately `numeric_gate_pass=false / gate_pass=false / promotion_ready=false`; CPU shared VRAM is marked not applicable and post-release RAM is recorded. The local checkpoint is synthetic and must not be committed, registered or used as a teacher.

## Human audit status

The canonical fixed24 page at `agents/audits/20260718_231220_scorer-v10-canonical-data-fixed24/` contains four speech and four all-background controls from every partition. Its manual gate is complete but failed: `correct=18 / contains_target_speech=3 / speech_in_background=1 / background_in_speech=1 / unsure=1`, so `manual_gate_pass=false` and no feature/training manifest is allowed. Three confirmed all-background assets are quarantined; one also occurs in another speech composite.

The follow-up page at `agents/audits/20260719_095522_scorer-v10-canonical-span-repair15/` contains the 15 exact spans from the three failed/unsure speech sources. Every span plays without context and requires an independent `speech/background/unsure` verdict. Its evaluator can only produce quarantine/recompile decisions; it cannot unlock training. Raw PTM2048 caching remains paused until corrected canonical sources pass a replacement audit.

The repair was applied without turning uncertain material into background: the corrected plan removes four affected source rows, records four ignored core identities, changes two spans, and produces `2667 sources / 2044 definite cores / max core use=1` with `speech 495995 / background 143163 / unsure 2066` frames. The targeted replacement page at `agents/audits/20260719_101106_scorer-v10-corrected-replacement7/` covers the three repaired sources and four replacements from the affected partitions/roles.

Replacement7 found one additional contaminated val all-background asset (`correct=6 / contains_target_speech=1`); that asset was also present in another speech composite. A source-level quarantine gate removed both rows while inheriting the previous quarantine and ignored-core ledger. Corrected r2 is `2665 sources / 2042 definite cores / max core use=1`, with `speech 495666 / background 142908 / unsure 2064` frames and canonical SHA256 `ee15d695b2c331dd03effb76036a8f943a91ca19e1c4df56b80d695831434569`.

The final page at `agents/audits/20260719_103542_scorer-v10-corrected-replacement2/` is complete at `2/2 correct`. A combined evidence-chain gate binds the original fixed24 failure, span-repair15, corrected r1, replacement7 quarantine, corrected r2 and replacement2 by schema, SHA and exact target IDs. It passes with `feature_cache_allowed=true / training_manifest_allowed=true`; this authorizes raw PTM2048 caching and later manifest finalization only. It does not claim a trained Scorer v10, a model gate or production promotion.

## Raw feature-cache smoke

The CUDA/bfloat16 raw PTM2048 smoke at `agents/temp/20260719_111207_scorer-v10-raw-ptm2048-cache-smoke/` cached one 18.9-second canonical source as `945×2048 PTM + 945×40 MFCC` with zero errors. Peak CUDA allocated/reserved was `4039.861/4100 MiB`; physical RAM stayed below its 0.95 budget. Windows PDH reports a fixed CUDA context/Qwen execution raw shared baseline (`76→78 MiB`), while model load, the real canonical forward and post-model-release growth against the warmed execution baseline were all exactly `0.0 MiB`.

Batch equivalence was checked on the same two sources. Sending both variable-length windows through one padded bfloat16 PTM forward preserved order and shapes but changed raw PTM values (`max abs delta=0.0390335`), so that optimization is rejected. CPU grouping with `ptm_window_batch_size=1` keeps every complete window in an independent PTM forward and is byte-identical to batch size 1 for PTM and MFCC. Full caching must use this singleton PTM path; it may not shorten visible context or use global-max padding.

## Real-manifest trainer smoke

The full cache was finalized into `agents/temp/20260719_112128_scorer-v10-corrected-r2-training-manifest/` with `2665` rows, `2042` unique cores, max core use `1`, and `2064` unsure frames excluded from normalization/loss/metrics. A one-step CUDA smoke at `agents/temp/20260719_114236_scorer-v10-real-canonical-train-smoke-s1/` used the actual manifest, random initialization (`seed=17`), weighted cross entropy and binary argmax. It had zero true-speech deletion in the short smoke (`val start/end=100/98.51%`, `test=100/95.10%`) but is explicitly `numeric_gate_pass=false / gate_pass=false / promotion_ready=false` and is not a production checkpoint.

The trainer now warms the exact forward/backward/loss/evaluator paths, then resets only the execution shared-memory baseline. It computes evaluator metrics with GPU argmax and scalar transfers instead of copying full prediction arrays to CPU. In the smoke, execution raw shared baseline was `142 MiB`; train/val/test/release increments were `0.0 MiB`, peak CUDA reserved was `6748 MiB`, and physical RAM remained below the 0.95 budget. Full training still requires a serial 3000-step run followed by held-out, >8-second residual and zero-clipping/zero-true-speech-deletion human audit; no registry promotion is implied.

After a real v10 model exists, the same audit framework must generate pages for every prediction-drop/truth-speech case, all held-out hard cases, edge clipping and every greater-than-8-second residual. Numeric gates are capped at 95%; zero clipping and zero true-speech deletion require saved human verdicts before promotion.
