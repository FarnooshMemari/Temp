# Causal RAG — SPY/TLT engineering fixes (2026-10-30)

## Scope

This release changes ONLY the experimental `ap3_spytlt` fork and its provenance/validation
handoff. The original `native_backtest/ap3` is not modified.

1. The Testing Agent prompt now explicitly distinguishes Treasury yields from TLT prices,
   excludes forecasts/outcomes after the decision time, and treats theory-only text as
   inadequate evidence. The original vendored quote validator remains unchanged.
2. The assessor rejects missing, unexpected, and duplicate edge IDs rather than silently
   overwriting one response or treating it as a valid completed assessment.
3. The encoder no longer treats model failure as a genuine Insufficient-evidence decision.
   The original F_S, F_C and F_I signals remain intact for valid tests. Audit-only
   F_FAIL, F_NET, F_AGREE_NET and F_VALID_COUNT reveal invalid calls and cancellation.
4. The monthly record includes required decision_time and SPY/TLT 1M provenance fields.
5. The Causal RAG staging exporter strictly rejects any test failure, and exports only
   the original three calibrated status features, not exploratory diagnostics.
6. Added tests for invalid evidence, duplicate edges, failed calls, cancellation,
   incomplete test results, and the export contract. Added a signal diagnostic CLI:

       python -m native_backtest.ap3_spytlt.signal_diagnostics --records PATH_TO_SPYTLT_MONTH_RECORDS.jsonl --out report.json

## Validation

### Repository continuation (2026-10-30)

- Restored the implementation directories missing from the GitHub branch.
- Failed Testing Agent calls now have raw status `Failed`; legacy failed records
  remain readable but cannot contribute to `F_I` or the scored features.
- Encoding rejects duplicate hypotheses and unknown result IDs. Staging recomputes
  features from complete, matched judgments and rejects altered status features.
- The focused and selected legacy suites passed: **79 passed, 10 skipped**. Skips
  require external claims data or inference resources; these are engineering tests.
- Restored market and v2 mediator snapshots with their manifest SHA-256 checks.
- Completed `python -m unified_comparison.run --grid --mapping both`: six configurations,
  46 development months (2019-03 through 2022-12), Static/DET/Jev/Committee only.
  `site/data/unified.json` now contains freshly recomputed common-Ω results.
- Causal RAG still requires real two-asset inference and independent reliability gates.
  No new model-based accuracy or portfolio performance claim is made for Causal RAG.

### Original package validation

- The focused new and unified offline suites passed (17 tests).
- The legacy AP3 suites: 51 passed, 13 skipped, 2 failed solely because pinned historical
  market and mediator files are not in the supplied ZIPs. Those missing files must be
  restored via the project's hash-pinned manifests before full integration testing.
- **No new real-model inference has been performed.** The prior Testing Agent reliability
  result has not been improved or remeasured. These changes are not proof of predictive
  performance or valid causal inference. The Causal RAG dashboard slot stays PENDING.
- The 30 older 3-asset synthetic G3b fixtures were NOT silently relabelled as valid
  two-asset TLT fixtures. A new independently specified SPY/TLT evaluation is needed.

## Next validation steps

1. Restore the pinned market/mediator data on any new checkout. Keep the 2023+ test set sealed.
2. Run the SPY/TLT model on independently designed, time-stamped reliability fixtures.
3. Report confusion matrix and the 80% G3b gate; do not tune on that same held-out set.
4. Generate real 2015-02 through 2022-12 SPY/TLT monthly agent features and run the
   signal diagnostic. Compare raw status counts with the richer diagnostics only in
   predeclared, walk-forward development ablations.
5. Only after gate passage, stage the real month records and rerun the unified backtest.

**Do not copy the original 3-asset Causal RAG results to the SPY/TLT dashboard.**
