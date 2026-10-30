# Experimental Causal RAG SPY/TLT fork

This isolates the SPY/TLT hypothesis, graph-path and testing-agent schemas from the original three-asset AP3. Original `native_backtest/ap3` remains unchanged.

- 1-month horizon; SPY and TLT directional hypotheses only.
- Graph concepts for TLT use long-Treasury yield/term-premium/price pathways. **These are conceptual proxy nodes, not TLT-specific market observations.** Expert graph audit needed.
- `native_backtest.ap3_spytlt.runner.run_month` runs with the original LLMClient infrastructure when point-in-time claim files, evidence corpus and inference endpoint are available. Use its `status_features` to create the month records for `unified_comparison/data/causal_rag_spytlt_1m.jsonl` (one record per month).
- The legacy three-asset fixtures and failed reliability gates do **not** validate this fork. New SPY/TLT fixture validation is required. Do not replace original data or claim Causal RAG results before this passes.
- Allocation is deliberately disabled in the fork; all systems' weights are calculated by `unified_comparison.run` with common covariance and Omega.

## October 2026 engineering hardening (not yet revalidated)

- The Testing Agent adds time- and asset-direction safeguards to its assessor prompt and rejects duplicate, missing, or unexpected edge IDs instead of silently overwriting them. The original vendor quote validator remains intact.
- The signal encoder distinguishes **model failure** from an evidence-based **Insufficient** finding. `F_FAIL`, `F_NET`, `F_AGREE_NET`, and `F_VALID_COUNT` are diagnostics only; the published calibration still uses `F_S`, `F_C`, `F_I`. A failed month must be rejected by the staging gate.
- Raw monthly records now include `decision_time`, `horizon_months=1`, and explicit SPY/TLT universe, so the staging export receives its required provenance fields.
- **These changes do not prove a higher G3b accuracy or stronger return prediction.** Run genuine model inference on independent two-asset fixtures, report accuracy/confusion matrix, verify the 80% threshold, then run a development-only signal ablation before a full backtest. Do not reuse saved 3-asset experiment scores.
