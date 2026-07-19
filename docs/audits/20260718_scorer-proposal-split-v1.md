# Scorer / Proposal / Split v1 Audit

本页只覆盖 1.7B。0.6B registry placeholder、checkpoint、runtime 和 data 未读取、未修改、未训练。中央兼容键固定为 `boundary_acoustic_binary_v12`。

## Evidence

| stage | current artifact | evidence | disposition |
| --- | --- | --- | --- |
| Scorer v8 | `speech_island_scorer_v8` | checkpoint `ptm_dim=128`, `MFCC=40`, one sigmoid logit; training `positive_ratio=0.4845`, `recall=0.8978` at training threshold `0.5`; historical `recall=0.9968` used runtime `0.20`; row-random eval leaked exact training features | audit-only; runtime guard rejects v8 and v9 |
| Proposal v1 | `boundary_proposal_scorer_v1` | one `boundary_prob` logit; 12k steps, `positive_weight=30`, frame recall `0.9518`; runtime candidate coverage `0.9912`, single-noise bucket `0.9639` | temporarily retained as non-binding candidate source; miss audit open |
| Split v4 | `semantic_split_model_v4` | raw PTM2048 learned in-checkpoint Linear128, 13 scalar features, binary argmax; val/test event basin recall `0.9676/0.9746` | architecture retained pending source/core identity re-audit and remaining listening gate |

The Scorer dataset is still present and was inspected directly. The Proposal metadata and Split checkpoint point to source dataset paths that are no longer available in the current workspace. Their reported metrics are retained as evidence, but they are not treated as a reproducible training manifest.

## Scorer

Scorer v8 is not just a probability producer. `_hysteresis_frames` converts its sigmoid output through configurable `on/off` thresholds, then `frame_dilation_s` changes island membership. The model was trained with a negative weight four times the positive weight and is evaluated on cropped windows. That combination explains why the published high recall operating point is a runtime threshold rather than a learned binary decision. It creates the exact false-negative/edge-clipping risk highlighted by the Inner v2 review: changing the operating point can hide a data-distribution defect without changing the model.

The dataset has `53,760` manifest rows but only `33,096` unique audio/feature pairs; the 220 real windows were repeated 64 times. The trainer performs a random row split, so 328 exact audio/feature pairs occur in both train and eval. It reports only the first 512 eval rows, and 144 of those are exact training overlaps. There is no frozen source/core holdout, so the published evaluation cannot gate a replacement model.

The existing experimental v9 is not an acceptable drop-in: its membership head is `outside/inside/unsure` and runtime treats every non-`outside` argmax as retained. It is therefore still a three-state teacher/runtime alias, not the required binary argmax contract.

The replacement contract is a new repo-bound two-logit scorer (v10): `background/speech` argmax, unsure mapped to ignore in training only, raw PTM2048 with a checkpoint-owned Linear128, MFCC40 and relative position, bidirectional sequence context, no sigmoid operating threshold, no dilation, no duration rule and no fallback. The fixed1024 canonical audit compiled Galgame speech cores plus partitioned CueQC/Omni definite-drop gaps, overlays and all-background controls, then quarantined four contaminated background assets and their affected composites through two replacement rounds. Corrected r2 now passes the complete listening evidence chain at `2665 sources / 2042 definite cores / max core use=1`; raw PTM2048 cache and training have not started.

## Proposal

Proposal v1 is correctly scoped as a candidate source, not a cut decision. Local maxima, prominence, per-island quantile filtering, NMS and speech-valley snap are deterministic candidate enumeration operators. They are not allowed to decide a final cut, but they do control what Split can observe. The runtime gate passed aggregate coverage (`99.12%`), while the weakest `single_noise_101` bucket was only `96.39%`; approximately 98 eligible truth boundaries were not proposed. Those misses need a playable audio/verdict page before any candidate distribution is changed or a replacement is promoted.

The existing Split v4 hard-case and missing-cut pages remain the authoritative listening entry points:

- `agents/audits/20260717_153336_split-v4-missing-cut-candidates/index.html`
- `agents/audits/20260717_000820_split-v4-binary-event-gate/index.html`

No Scorer/Proposal miss verdict is fabricated in this report.

## Split

Split v4 does query the candidate neighborhood rather than reducing an entire island to one mean: left/gap/right bins, multi-scale bins, candidate scalar features and a masked candidate-sequence Mamba stack are all present. The current evidence does not justify blind Focal/boundary-band tuning. Binary argmax event runs and `unsure=-100` are contract-correct.

The previous trainer only guaranteed group-level partition isolation. It now rejects a source whose `|islandNNNN` groups cross train/val/test and moves every island of a manually corrected source into train. The current dataset itself is unavailable, and its metadata does not expose an independent core identity, so a future retrain must recompile a source/core manifest before using this checkpoint as a new teacher.

## Shared workflow blockers

- Outer v3 now has audited candidate plumbing, but its registry remains empty with status `pending_outer_v3_audit`; no end-to-end promotion is implied.
- The current audit-only Scorer v8/v9 path now releases a rejected scorer before returning; the downstream path loads and releases Outer, Split and Inner one at a time, with stage memory diagnostics kept outside the functional cache signature. CueQC also releases its model before postprocessing. A future successful Scorer v10 implementation must still split Scorer and Proposal inference into independent scopes; no old scorer can reach that path because runtime remains fail-fast.
- Inner v2 now preserves provisional display timestamps while changing acoustic/ASR timestamps. This is covered by a focused regression test.
- Scorer and Proposal batch scoring was checked on production checkpoints: singleton versus padded batch outputs preserve order and argmax, with max probability delta below `6e-8` on the smoke vectors.

### Decision

Do not train or alter 0.6B. Do not promote Scorer v8, experimental v9, or Proposal v1 as a new decision chain. Continue with a resource-checked 1.7B Scorer v10 raw PTM2048 cache smoke and full cache, then train/gate v10 before regenerating the actual Outer v3 distribution.
