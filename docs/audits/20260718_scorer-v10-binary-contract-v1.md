# Speech Island Scorer v10 Binary Contract Audit

本页只覆盖 1.7B。0.6B 空 registry、checkpoint、runtime 和 data 未读取、未修改、未训练。中央序列化兼容键仍只有 `boundary_acoustic_binary_v12`。

## Decision

Scorer v10 的断兼容 schema、两 logit 模型、argmax decoder、严格 dataset validator 和 random-init trainer plumbing 已完成真实 full training。原 numeric checkpoint与首个 internal-background A/B 在 corrected-r3 人工 gate 中出现真语音误删/截断，均被否决。提高 worst-frame 保护后的保守候选最终没有真语音误删，但仍把同一 ASR 单元切碎且依赖尚未完成的 canonical 修标/重训，因此也不能晋升。production registry 与 `segment()` 继续 fail-fast 为 `pending_binary_scorer_audit`。

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
| initialization/loss | random initialization; weighted cross entropy plus explicitly configured training-only run losses; no warm-start/Focal/boundary band |
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

This is now a manually gated canonical source plan with a completed raw cache and trainer manifest; it is not a model-promotion claim. The full checkpoint candidate remains gate-failed.

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

The trainer now warms the exact forward/backward/loss/evaluator paths, then resets only the execution shared-memory baseline. It computes evaluator metrics with GPU argmax and scalar transfers instead of copying full prediction arrays to CPU. The first `max_batch_frames=4096` full-run attempt was rejected by CUDA OOM on a `7×578` shuffled batch; no CPU fallback or enlarged config was used. With frame budget `1024` (rows above the budget remain complete singletons), both the ordinary smoke and a seed `975` smoke forcing the maximum `1643`-frame singleton passed: shared increment remained `0.0 MiB`, and the long-singleton smoke peak CUDA reserved was `4082 MiB`.

Full 3000-step training completed at `agents/temp/20260719_115127_scorer-v10-corrected-r2-train-full-frame1024/`, checkpoint SHA256=`509d35e4179456f5c9b0cd42957a4c23cb7767ee093909ce631dee70495da2c5`, best step=`2750`. Test start/end coverage=`100/99.02%`, background-drop recall=`95.54%`; val=`99.00/98.01%`, background-drop recall=`92.75%`; true-speech deletion was zero on both partitions. The val background-drop miss keeps `numeric_gate_pass=false / gate_pass=false / promotion_ready=false`.

Singleton checkpoint audit at `agents/temp/20260719_115933_scorer-v10-checkpoint-audit/` scored all `2665` rows with zero shared increment. It found one row-level true-speech deletion, `24` val/test all-background false-keeps, `100` val/test speech edge/partial rows and `15` predicted speech runs longer than 8 seconds. The complete exact-span page `agents/audits/20260719_121639_scorer-v10-prediction-residuals-all-drop-keep/` contains `720` selected cards, including all `681` speech prediction-drop/truth-keep rows, and saves `speech_scorer_v10_prediction_manual_verdict_v1`; all manual verdicts remain pending.

The same predictions expose a separate island-fragmentation failure. Scorer v10 does not merge argmax runs: one background frame splits an upstream speech island, while Proposal only attaches non-binding candidates and Outer/Split cannot reconnect separate Scorer islands. Train/val/test speech-run continuity is only `77.52/73.63/69.61%` against the `95%` gate; fragmented truth runs are `390/53/31`. Held-out predictions overlapping true speech contain `123` runs shorter than 100 ms, `150` shorter than 200 ms and `161` shorter than 500 ms. Those duration counts are diagnostic only. Runtime must not repair them with gap closing, dilation, thresholds or duration rules: the model/training objective must produce stable binary argmax islands. This is downstream-critical because every Scorer island independently passes through Outer/Split and becomes a CueQC candidate; CueQC `drop_before_asr` removes it before Inner and ASR can recover neighboring speech. The derived result is stored in `agents/temp/20260719_115933_scorer-v10-checkpoint-audit/continuity_summary.json`.

A strict full A/B then compared the same data, seed, 3000 steps and frame budget under the new continuity-aware checkpoint selection. Weighted-CE baseline (`agents/temp/20260719_230809_scorer-v10-continuity-ab-baseline-s3000/`) selected step 2250 and reached val/test continuity `87.06/93.14%`, background drop `92.27/94.55%`, with zero run deletion. A training-only adjacent speech-probability total-variation auxiliary at weight 1.0 (`agents/temp/20260719_231206_scorer-v10-continuity-ab-tv100-s3000/`) improved continuity to `90.05/94.12%` and reduced gap counts from `63/17` to `51/9`, but val internal-drop frames regressed from `180` to `239`, precision fell about `0.58/0.81pp`, and background drop did not improve. This mixed result rejects the auxiliary as the default; weight zero remains default, and the option is retained only for explicit experiments. Both checkpoints fail the numeric gate and are not promotion candidates. The comparison is stored in `agents/temp/20260719_231607_scorer-v10-continuity-full-ab/ab_summary.json`.

The new baseline was then scored row-by-row at `agents/temp/20260719_232035_scorer-v10-continuity-baseline2250-checkpoint-audit/`. The reproducible composition join at `agents/temp/20260719_235624_scorer-v10-fragmentation-distribution-audit/` shows that additive overlay, not clean core speech, dominates the remaining failure: overlay/clean fragmented-row rates are `29.86/6.62%` on train, `31.25/11.32%` on val and `21.74/3.57%` on test. Train overlays below 12 dB reach `42.53%`. Val has 43 of 63 internal gaps in the middle 80% of a core, so an Outer edge refiner cannot recover them. Breathing, non-speech vocalization and kissing-like overlay types are prominent hard cases. The tool writes 158 train-only diagnostic hardcases and 27 held-out audit cases to separate files; it does not duplicate cores, repartition sources or alter the training manifest. Since both the old and new candidates already fail numeric gates, completing the old 720-card manual page cannot promote either checkpoint and may be paused until a numerically viable model exists.

After a real v10 model exists, the same audit framework must generate pages for every prediction-drop/truth-speech case, all held-out hard cases, edge clipping and every greater-than-8-second residual. Numeric gates are capped at 95%; zero clipping and zero true-speech deletion require saved human verdicts before promotion.

## Corrected-r3 diagnostic rescore

The fragmentation audit was evaluated against the actual downstream workflow: each binary argmax speech run is an independent Scorer island, and neither Proposal, Outer, Split nor CueQC reconnects islands. The 61 internal gaps were reviewed `61/61`: one is the same ASR unit and therefore a real model fragmentation; 29 require at least one nonsemantic side to remain independently droppable; six contain two independent target-speech events; and 25 local clusters are not speech core. There were no unsure verdicts.

Those topology decisions yielded 155 atomic intervals. Constraint propagation resolved 116; the reviewer completed the remaining 39 at `agents/audits/20260720_143410_scorer-v10-fragment-atomic-repair39/`. The final atomic labels are `background=115 / speech=40 / unsure=0`, with no relation violations. Applying them produced corrected-r3 without deleting sources or cores and without changing audio bytes or partitions:

| item | corrected-r3 |
| --- | --- |
| sources / cores / max core use | `2665 / 2042 / 1` |
| affected sources / cores | `32 / 33` |
| changed atomic spans | `115 -> background` |
| canonical frames | `speech 495089 / background 143491 / unsure 2058` |
| canonical SHA256 | `f88d4cfc20fb46077f357ca15bc8c3368938ecfebfbc21728e50b4d957480f1a` |

The canonical frame projector now uses `Fraction` plus integer sample-domain boundaries. This fixes exact 20 ms endpoints that binary floating-point could previously classify as a mixed/unsure frame.

Because corrected-r2 and r3 have a byte-identical audio manifest, the existing PTM2048/MFCC40 values may be reused for checkpoint diagnosis only. Every r3 diagnostic row has `diagnostic_only=true / training_manifest_allowed=false`, and the trainer rejects it. Three repeated full rescoring runs produced the identical predictions SHA256 `d483d9ddd14307e8bbf6dd10701f7486ef94f7dc42d7fe0131bfceadf56a19f0`. The final 2665-row result is:

| metric | r3 result |
| --- | --- |
| train / val / test continuity | `99.54 / 97.58 / 100%` |
| internal gaps | `1 train / 0 val / 0 test` |
| prediction-drop/truth-keep rows | `36` |
| whole truth-run deletions | `12` |
| rows with a model speech run >8s | `277` |
| shared VRAM increment | `0 MiB` |
| peak CUDA allocated / reserved | `571.375 / 1062 MiB` |

The formal selection is complete at `321/321`: all `36/36` prediction-drop/truth-keep rows, all `277/277` >8-second rows, and every held-out hard case. The page `agents/audits/20260720_162315_scorer-v10-r3-residuals-final321/` saves `speech_scorer_v10_prediction_manual_verdict_v2`. It shows exact red `truth_speech ∩ model_background` spans, complete canonical and actual blue argmax timelines at absolute positions, and whole-source context without changing runtime segmentation. A separate page `agents/audits/20260720_162315_scorer-v10-r3-fragmentation-gap1/` contains the one remaining 60 ms internal gap.

Both pages are now complete. Residual verdicts are `acceptable_long_residual=78 / missed_background_or_gap=194 / canonical_should_be_background=35 / canonical_contains_target_speech=6 / model_false_keep=7 / true_speech_clipped=1`; unsure is zero. The gap page confirms that `000973 0.52–0.58s` is part of the same ASR unit. The old candidate therefore fails the manual zero-clipping gate. The 35 exact red spans may be changed to background, while the six all-background sources require exact per-island speech repair rather than inheriting speech across the full source.

## Internal background false-keep A/B

The 194 `missed_background_or_gap` rows are not 194 semantic-subject failures. Event-level inspection found 193 rows with a background-only model-speech island bracketed by canonical speech. Across those rows there are 441 complete background-only islands; 439 were already pure background in the corrected-r2 training labels. This exposed a loss-aggregation blind spot: `speech_run_and_background_source_worst_frame_ce_v1` focuses on speech runs whenever a source contains speech and only applies its background term to all-background sources, so internal negative runs received only frame-averaged CE.

The trainer now provides `speech_bracketed_background_run_worst_frame_ce_v1`. It is training-only and uses canonical topology, not duration: a background run is eligible only when canonical speech exists on both sides. Runtime remains two-logit frame argmax, without a short-island rule, threshold, merge or veto. Evaluation separately reports all background leakage and complete argmax speech islands lying wholly inside speech-bracketed background. Checkpoint selection first requires zero deletion and the existing start/end/background/continuity safety floor of 95%; only among passing validation steps does it minimize independent false-keep islands and frames.

The strict full A/B kept corrected-r2, seed 17, random initialization, frame budget 1024, max packed rows 8, worst-frame weight 0.10 and 5000 steps fixed. The candidate changed only internal-background weight to 0.01 and enabled the already-defined speech-run continuity loss at 1.0. Its SHA256 is `ec7b1c8895c40bb1fae19f85947b795325656735bebc9d0a5cbaedc8c1aa4f58`; it is experimental and not registered.

| r2 held-out metric | old numeric candidate | internal-background A/B |
| --- | ---: | ---: |
| val independent background islands | 89 | 72 |
| test independent background islands | 35 | 24 |
| val/test independent-island frames | 373 / 82 | 276 / 77 |
| val/test speech continuity | 96.02 / 98.04% | 95.52 / 97.06% |
| val/test speech precision | 95.81 / 95.76% | 97.03 / 96.51% |
| val/test background drop | 96.62 / 97.03% | 95.65 / 97.52% |
| val/test true speech deletion | 0 / 0 | 0 / 0 |
| shared VRAM spill | 0 MiB | 0 MiB |

On the same corrected-r3 diagnostic manifest, independent internal-background islands fall from `1651` to `1000`, frames from `8262` to `5144`, and total island duration from `165.24s` to `102.88s`. The known `000973` clipping is removed. However, candidate-vs-baseline frame differencing finds 29 sources with 191 newly dropped canonical-speech frames. The audit at `agents/audits/20260720_200040_scorer-v10-internal-bg-ab-extra-drop29/` displays only that exact red difference while retaining the candidate's complete blue output and whole-source playback.

That page is complete at `29/29`: 27 rows are canonical speech spans that should be background, but `001823` contains true speech deleted by the candidate and `001955` contains a clipped true-speech tail. The resulting manual gate has `zero_clipping_violation_count=2 / checkpoint_promotion_authorized=false`; the `ec7b1c88...c1aa4f58` checkpoint is rejected rather than delegating these errors to CueQC.

A conservative rerun retained internal-background weight `0.01` and continuity weight `1.0` but increased source worst-frame protection from `0.10` to `0.20`. Checkpoint SHA256=`6a851f556f5ff96b93d44a4c75de985306bc42ef428d358340e5b5aba61e21ab`, best step=`3250`. On corrected-r2 it passes the capped numeric gate: val/test independent false-keep islands=`73/12`, continuity=`95.02/98.04%`, true-speech deletion=`0/0`, shared spill=`0 MiB`. This is not sufficient for promotion: corrected-r3 has `46` truth-keep/model-drop rows, and relative to the old numeric candidate it introduces `304` drop frames across `38` sources; total internal-background false-keep duration also regresses from `165.24s` to `176.68s` even though island count falls from `1651` to `1257`.

Prior manual evidence is now inherited only at frame resolution. The selector cryptographically binds the earlier audit manifest and verdict file, accepts only `canonical_should_be_background`, verifies canonical frame identity, intersects the new candidate difference with the exact prior red spans, and never inherits a whole-source verdict. Of 304 new drop frames, only 31 are covered by previously listened background spans; `001763` and `002022` are fully covered. The remaining `36 sources / 273 frames` are at `agents/audits/20260720_231741_scorer-v10-worst020-remaining-extra-drop36/`.

The final verdict set is complete at `36/36`: `canonical_should_be_background=35 / same_asr_unit_fragmented=1 / true_speech_clipped=0 / unsure=0`. The initially disputed `000673` is a nonsemantic repeated-vocalization prefix and is safe background for this responsibility. `001747` contains two 20–40 ms dropped intervals inside the spoken hesitation `wa… watashi mo`; the red frames may themselves be silence, but the current no-merge workflow turns the original `0–2.32s` unit into three downstream islands. The resulting gate has `zero_clipping_pass=true / workflow_continuity_pass=false / canonical_repair_count=35`, so the checkpoint cannot be promoted before repair and retraining.

The six r3 all-background rows manually found to contain target speech now have a separate exact repair gate. Five of their negative assets are also used by seven speech composites, so whole-source relabeling or control-only quarantine would both be wrong. `agents/audits/20260721_002526_scorer-v10-background-speech-repair6/` contains all six full-source players, 30 exact actual-workflow blue islands and 24 adjacent gaps. Every island is independently classified as usable target speech, target speech with incomplete model boundaries, background/nonsemantic or unsure. A gap becomes required only when both neighboring islands are usable target speech; `same_asr_unit` includes it in one repair event, while `separate_target_events` keeps distinct core identities. Boundary-incomplete and unsure results block canonical writes and require a follow-up exact-boundary page. No model prediction, gap duration or source-level verdict is promoted directly to training truth.

The island page is complete: `8` islands are usable target speech, `22` are background/nonsemantic, and there are no incomplete boundaries or unsure labels. Only three links are dynamically required after island labels (`same_asr_unit=2 / separate_target_events=1`); nonrequired link selections are ignored by the evaluator. One source has no target island despite its earlier source-level `canonical_contains_target_speech` verdict, so the gate remains deliberately not repair-ready. `agents/audits/20260721_073703_scorer-v10-background-source-recheck1/` is a one-source recheck built from the same immutable r3 prediction row. If the previous verdict is withdrawn as `model_false_keep`, the remaining five contaminated assets can yield deterministic repair events; if target speech still exists outside the blue islands, an exact-boundary follow-up is required instead.

The recheck resolves the disputed source as `model_false_keep`: it is a continuous nonsemantic “ah” vocalization and remains canonical all-background. A new immutable override gate binds both the old residual manifest/verdict and the recheck manifest/verdict by path and SHA, permits only this exact `canonical_contains_target_speech -> model_false_keep` withdrawal, and leaves the historical verdict file untouched. Re-evaluating the island page then produces `5 repair assets / 6 repair events / canonical_repair_ready=true`.

Canonical r4 is compiled at `agents/temp/20260721_080836_scorer-v10-canonical-r4/`. The three background-verdict pages contribute `97 verdict rows / 156 exact spans / 80 sources / 725 unique 20 ms frames`. The five contaminated assets are mapped through the original `source_manifest_stratified1024.jsonl`, including exact crop offsets and short-source tiling rather than treating a reused negative as one undifferentiated span. All six active composite uses are covered. Of 19 rendered occurrences, 12 change canonical background to speech and receive deterministic placement-specific core identities; seven fall wholly inside existing speech and are recorded without creating duplicate cores. The resulting dataset retains all 2665 source identities, audio bytes and partitions, has `2054` cores with max use `1`, and no propagated-speech/direct-background conflict. Its SHA256 is `5e71075921d53878dcbbbcc8dbacf7fd67b237cb026b0eb6464227e58e1d6f3f`.

This remains a candidate, not a training authorization. The current page is `agents/audits/20260721_094456_scorer-v10-canonical-r4-replacement19-mapped-coordinates/`. Its three columns have distinct contracts: the complete source event determines whether the upstream evidence is valid; the second column presents the exact source-side occurrence and the equal-length rendered target occurrence as a pair; the third column contains only target ranges whose canonical label would change from background to speech, and is not another audio source. Source and target timestamps belong to different files and therefore need not overlap numerically. Each occurrence is resolved against the SHA-bound dependency mapping and its tile index, and source/target sample lengths must agree. All displayed ranges and canonical color spans are standalone WAV files. The page also distinguishes an invalid source event from an event that becomes invalid only after a particular crop/mix: `source_event_not_target` is applied to the complete control/crop/tile/overlay placement group and requires canonical recompilation, while `not_target_after_render` affects only one rendered occurrence. A boundary-incomplete, no-longer-target, source-event rejection or unsure verdict blocks the repair. Until the replacement gate passes, feature-label recompilation, a new training manifest, Scorer retraining and registry promotion remain disabled.
