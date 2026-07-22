# Full code audit v1 — 2026-07-22

## Scope

This audit covers the current `src/`, `tools/`, `tests/`, Web ES modules, README,
and documentation links.  It is source-only: no checkpoint, dataset, cache, model
registry, or 0.6B path was modified.

The review checked:

- Python module reachability, local import targets, and imported symbol existence;
- unreferenced top-level private/public functions and classes;
- import-time loading of every `src/` and `tools/` Python module;
- Web ES-module reachability and named import/export consistency;
- Python/JavaScript syntax and local Markdown link targets;
- retired Scorer/Proposal/Split threshold, hysteresis, alias, and trainer surfaces.

## Fixed findings

1. `src/audio/audio_metrics.py` was an unreferenced RMS/dBFS helper from a retired
   rule-based short-span drop path.  Its error behavior returned `0 dBFS` to avoid
   deletion, but no current caller existed.  The file was retired instead of being
   reintroduced into the learned binary chain.
2. The old Stage F speech/proposal dual-head module, gate CLI, threshold/dilation
   recall CLI, and their dedicated test had no current caller and represented a
   superseded combined-decision route.  They were retired; Proposal v1 remains a
   separate high-recall candidate source and does not make final cut decisions.
3. Confirmed private dead helpers were removed: index-based feature pooling,
   random group-window batching superseded by anchor windows, unused TORC helper
   translations, an unused JSONL counter, an unreachable translation-summary
   fallback, an unused job-context value reader, and obsolete scalar distance
   wrappers.
4. Two Split v4 audit/evaluation CLIs imported the removed trainer-private symbol
   `_pad_batch`.  They now import the current shared
   `tools.boundary.ja.acoustic_split_v4_dataset.pad_batch`; an import-smoke test
   locks this dependency.
5. `build_runtime_semantic_split_dataset.py` still intentionally reconstructs the
   retired v8/v10 threshold/hysteresis data path for historical audits.  Its module
   and CLI descriptions now state that it is offline-only and forbidden for new
   Scorer v11 training data.

Retired files were moved to:

- `agents/rm/20260722_100801_dead-audio-rms-rule/`
- `agents/rm/20260722_101155_retired-legacy-boundary-tools/`

## Intentional retained surfaces

- v8/v9/v10 decoder functions remain internal to `boundary.ja.backend` for
  historical offline reproduction.  Production `segment()` still fails fast as
  `pending_binary_scorer_audit` and exposes only the v11 two-logit argmax contract.
- The old semantic-v9 scoring functions remain internal model-level audit helpers;
  they are not exported by `boundary.ja` and are not reachable from production.
- `build_boundary_proposal_checkpoint` remains the artifact construction API for a
  future evidence-driven Proposal audit/retrain.  Keeping it does not register or
  promote a checkpoint.
- Scorer v10 canonical/audit tools remain because they contain the evidence chain
  used to construct and review v11 data.  They are not production aliases.

## Verification

- Python modules: `417`, missing local imports: `0`, unreferenced non-CLI modules:
  `0`, missing imported symbols: `0`.
- Import smoke: `250/250` `src/` and `tools/` modules loaded successfully.
- Web: `13/13` ES modules reachable from `main.js`; missing named imports: `0`;
  JavaScript syntax passed.
- Markdown local links: missing targets `0`.
- Focused regression: `138 passed`.
- Full regression: `888 passed / 6 skipped`; four existing SciPy sparse-efficiency
  warnings only.
- `python -m compileall -q src tools tests` passed.
- `git diff --check` passed apart from Git's existing LF-to-CRLF notices.
