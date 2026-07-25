# Scorer v12 structured-decoder training plan

## Decision

Scorer v12 detects continuous human-vocal event envelopes. All Scorer weights
are trained from random initialization with seed `117`; v10/v11 checkpoints are
never loaded. The ASR PTM is only a frozen raw feature extractor.

The approved Teacher envelope is continuous while frame predictions fragment,
so decoder/loss structure is now a first-class A/B axis. No arm may use a
runtime threshold, duration rule, hysteresis, NMS, rule merge, or fallback.

## Frozen evidence and data requirements

- Keep source/core/video partition, PTM2048/MFCC40, P2048/H256 Bi-Mamba trunk,
  1000-frame context, midpoint ownership, seed and batch budget identical.
- The 25-source pilot is only an implementation/data-contract gate. It has 13
  train sources and no all-nonvocal held-out source, so it cannot choose a
  production decoder.
- Before full training, expand real full-source train data and add real mixed,
  all-vocal and all-nonvocal val controls. Missing strata remain `n/a`; synthetic
  data cannot impersonate held-out evidence.
- Old semantic v11 labels and old human verdicts that removed moans are invalid
  for v12 and are never converted.

## Loss arms

1. `argmax-CE`: plain frame CE, retained only as the fragmentation baseline.
2. `argmax-structured`: `CE + 0.5 run-balanced CE + 0.25 balanced adjacency`.
   Run loss averages truth runs and then classes. Adjacency first averages real
   boundary edges and same-class edges separately, then gives both groups equal
   weight, so long vocal runs cannot reward an all-vocal collapse.
3. `CRF`: exact binary CRF sequence NLL plus `0.5 run-balanced emission CE`.
   Runtime is exact learned Viterbi; the auxiliary emission loss protects rare
   non-vocal evidence from the vocal-frame imbalance.
4. `Query-Mask`: dense frame CE plus Hungarian query existence, mask BCE and
   Dice. Queries represent complete vocal envelopes, not syllables. Query
   capacity is frozen from train topology with held-out headroom and never
   tuned on test.
5. `Dense Span+DP`: exact loss-augmented span Viterbi structured objective plus
   dense frame CE auxiliary. Runtime is exact learned DP. The CE auxiliary is
   required because v11 structured-hinge-only achieved continuity by bridging
   too much true background.

## Selection metric

Selection is source-level and lexicographic, not a single inside score:

1. Require vocal frame recall at least 95%, zero complete vocal-run deletions,
   and all-vocal keep recall at least 95% when the stratum exists.
2. Among safe arms, maximize the weakest of non-vocal frame recall, non-vocal
   event recall, vocal continuity and all-nonvocal full-drop recall when present.
3. Break ties by fewer internal holes, lower excess vocal-run count and lower
   overmerged non-vocal duration. No metric rewards predicted vocal area alone.
4. Select epoch and decoder on val only. Run test once for the best two val
   arms, then generate one side-by-side human audit page.

## Execution order

1. Finish BF16/resource/runtime smoke for argmax-structured and CRF.
2. Port Query-Mask and Dense Span+DP to breaking v12 schemas and smoke both.
3. Expand/calibrate v12 real full-source supervision; compile a final dataset
   with required held-out strata and immutable hashes.
4. Train all arms serially from random initialization with identical budgets and
   early stopping; write atomic progress and release CUDA between arms.
5. Run source-level val selection, then test only the top two.
6. Human-audit the two finalists and replay the real
   Scorer→Proposal→Split→CueQC→Inner→ASR workflow before any registry change.

The current production registry remains empty until numerical and human gates
both pass.
