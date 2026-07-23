# Scorer v11 downstream isolation responsibility audit

## Final duty decision

The frozen responsibility chain is:

1. Scorer keeps a high-recall continuous candidate envelope and must not delete uncertain speech edges.
2. Proposal supplies candidate cut locations inside that envelope.
3. Split makes the binary `cut/continue` decisions needed to isolate an independent background region.
4. CueQC deletes the resulting pure-background provisional sub-island.
5. Inner only trims the leading and trailing acoustic background of a kept island; it cannot carve an internal hole.

Accordingly, an independent background interval between protected dialogue anchors is not by itself a Scorer false keep. It is `independent_background_needs_downstream_isolation`. Until a bound workflow replay proves the downstream result, its Scorer canonical label is `unsure` and its training label is `-100`.

## Frozen selection

This stage intentionally freezes the six intervals that were confirmed at the selected hand-off point. The original human bridge verdict file is retained byte-for-byte; the selection and responsibility view are separate derived artifacts.

| Item | Duration | Current evidence status | Scorer label |
|---:|---:|---|---|
| 1 | 23.46s | `evidence_missing` | `unsure=-100` |
| 2 | 21.00s | `evidence_missing` | `unsure=-100` |
| 3 | 14.70s | `evidence_missing` | `unsure=-100` |
| 4 | 17.00s | `evidence_missing` | `unsure=-100` |
| 5 | 13.60s | `evidence_missing` | `unsure=-100` |
| 6 | 13.00s | `evidence_missing` | `unsure=-100` |

The six intervals span five sources and 5,138 frames. No aligned Proposal candidates, Split cut events, provisional sub-islands, or CueQC drop decisions exist for this exact source/audio/coordinate set. Old Scorer-only predictions and older-chain Proposal/Split artifacts are not accepted as substitutes.

## Fail-closed evidence contract

`tools/boundary/ja/compile_candidate_island_scorer_v11_downstream_isolation.py` validates:

- `boundary_serialization_contract_id=boundary_acoustic_binary_v12` on every input row;
- the full-source human verdict file and held-out audit manifest through the SHA values recorded by the dual-evidence summary;
- the explicit selection against the complete human bridge verdict and its exact source, partition and frame coordinates;
- the playable source WAV through its manifest SHA, including an explicitly supplied alternate audio root when the original audit page no longer carries audio;
- all Scorer, Proposal, Split and CueQC checkpoint SHAs when downstream evidence is supplied;
- a Scorer envelope covering the interval, Proposal candidates and Split argmax cuts at both edges, a matching provisional sub-island, and a CueQC binary argmax `drop` for that same island.

Missing any required stage produces `evidence_missing`. Complete but non-isolating evidence produces `isolation_not_demonstrated`. Only a fully bound replay can produce `downstream_isolation_demonstrated`; thresholds, duration rules, hysteresis and fallback decisions are not evidence.

The local result is stored under `agents/audits/20260723_172940_scorer-v11-downstream-isolation-required6/`. It reports `evidence_missing=6`, preserves the raw manual SHA, and emits a full 24-source responsibility verdict view with exactly the selected 5,138 frames changed from Split-level background to Scorer-level `unsure`.

The responsibility view was then compiled through the real Scorer canonical compiler without altering source/core/partition identity. The 1,170-source result has `inside_candidate=528479`, `outside_candidate=116415`, `unsure=65570`, and SHA256 `a720f7ac00f872cda4d4bbb9fd05ba010bb853d88c0734eec859e1d80e783716`. Relative to the prior view, inside frames are unchanged and exactly 5,138 outside frames become unsure.

## Validation

- Focused compiler/canonical/driver tests are `19 passed`; they cover missing evidence, exact four-checkpoint binding, binary argmax workflow evidence, exclusion of unselected bridge verdicts, alternate-audio SHA verification, and `unsure=-100` enforcement.
- The local audit page loads all six source WAVs with no media error. Its controls start at the exact interval boundary and stop immediately; no context is appended.
- Full regression with a project-local pytest base directory is `994 passed / 6 skipped`; the only four warnings are the existing SciPy sparse-efficiency warnings in retained cluster tests.
- Scorer production registry and all 0.6B artifacts remain unchanged.
