# Protofin — unified five-approach comparison (Claude Artifact update)

## Status — IMPORTANT

**Repository update, 2026-10-30:** the pinned market snapshots were restored and
SHA-256 verified, and the full six-configuration development grid was rerun.
`out/` and `site/data/unified.json` now contain fresh common-Ω results for Static,
DET, Jev, and Committee over 2019-03 through 2022-12. Causal RAG remains pending.
The paragraphs below describe the original ZIP delivery limitations; new checkouts
still need to restore the git-ignored pinned inputs before reproducing the run.

**Implemented and tested offline:** a new five-slot dashboard, a shared SPY/TLT evaluator, the two saved one-month Jev and Committee signal datasets, 36-month sample covariance (`np.cov(..., ddof=1)`), a **common, expanding, past-only deterministic Ω**, Black–Litterman, SPY prior options 50/60/70%, ±20 percentage-point relative bounds, drift-aware 5-bp costs, and development-only CER/Sharpe/bootstrap scoring. The original three-asset pipelines remain untouched. A separate `native_backtest/ap3_spytlt/` fork changes hypothesis/test schemas, concept mappings and status encoding to SPY/TLT. The agent fork **has not been inferred or validated**.

**Not yet complete:** the updated numerical backtest cannot be run from these ZIPs alone. Required price files in `stage2/data/_pinned/` are **not included** in the provided repositories (this was already true of the source). They must be fetched from the project's pinned, hash-verified snapshots. The new Causal RAG forecast **must be rerun with a model and the two-asset causal agent fork**, and its reliability gates must pass. A dashboard that displays its old SPY/IEF/LQD results as SPY/TLT would be scientifically invalid. Therefore Causal RAG is marked **Pending**.

The original ZIP's `site/data/unified.json` was an **archived preview only** showing old *model-specific* Ω results for Static, Deterministic, Jev and Committee. The fresh repository run has replaced it with common-Ω results and changed the banner to `updated:`. The original preview remains separately under `archived_reference/`.

## What changed

1. Compare **STATIC, DET, JEV, COMMITTEE, CAUSAL_RAG**; no Placebo, no 12-month series.
2. Restrict the market calendar, covariances, forecasts and target weights to **SPY/TLT**, not the original four-asset shared calendar.
3. Use only real **one-month** Jev and Committee decisions copied with provenance from `protofin-level-playing-field.zip` (files `data/jev_1m.json` and `committee_1m.json`). These are genuine previously saved signals, not converted 12-month replies.
4. Run the original Stage 2 macro/Ridge forecasting procedure as DET; no return fitting on the evaluation month.
5. Recompute each view with the same Ridge or directional calibration. **DET and Causal RAG always use Ridge**, irrespective of the UI sign-map toggle. This is intentional: those raw features have no signed expected-return prediction.
6. Set a **single Ω vector per month**, derived entirely from DET's prior inner-validation MSE through month t−2, and use it for each active forecaster. Sample covariance uses 36 completed excess-return observations.
7. Relative allocation bounds ±20 percentage points around the prior, shared optimizer and 5 bp costs, plus consistent paired 6-month block-bootstrap testing.
8. Causal RAG fork targets SPY/TLT 1M, but **will not be scored** unless all required 2015-02–2022-12 agent features are genuinely generated and validated.
9. New dashboard at `site/unified.html`; legacy pages and backtests retained.

## Quick start / data restoration

From the repository root (Python 3.10+ suggested):

```bash
pip install numpy pandas scikit-learn scipy pytest networkx requests jsonschema

# Restore the original pinned financial snapshots. Requires internet and Git.
python stage2/scoring/fetch_pinned_inputs.py \
  --manifest stage2/reproducibility/stage2_data_snapshot_manifest_v3_2026-10-27.json
python stage2/scoring/fetch_pinned_inputs.py \
  --manifest native_backtest/config/native_data_manifest_v1.json

# Run the exact shared evaluation on development months only.
python -m unified_comparison.run --grid --mapping both

# Open new dashboard (do not double-click HTML as a file:// page).
python -m http.server 8000 --directory site
# Browse http://localhost:8000/unified.html

# Basic offline validation (works before fetching historical prices).
python -m pytest -q unified_comparison/test_level_playing_field.py unified_comparison/test_unified.py native_backtest/tests/test_ap3_spytlt_hardening.py
```

The pinned cash series and macro panel are already in `stage2/data/`. Input hashes are verified by the shared market loader. If upstream data retrieval fails, **do not** replace it with live Yahoo prices: that would silently change the experiment.

### Causal RAG: next step requires model inference

- Keep the original `native_backtest/ap3/` unchanged as the three-asset baseline.
- In the new fork `native_backtest/ap3_spytlt/`, `hypothesis_agent.py`, `testing_agent.py`, `encoder.py`, `calibration.py` and `concepts_v1.json` now target **SPY/TLT**. TLT uses graph concepts for long-duration Treasuries, so expert validation of path direction and rate sensitivity is still necessary. The original AP3 knowledge graph was not rebuilt or improved.
- Build separate graph libraries for these concepts and run the fork's `runner.run_month` using its point-in-time evidence and a real supported inference provider. The project does **not** include a turnkey approved inference launch for the new fork. Do not run the legacy `ap3_dev_replay.py` and claim its output is the fork: that script still uses the old three-asset architecture.
- Validate the new fork on appropriate two-asset versions of the G3a/G3b reliability gates, plus the original 80% Testing Agent target. The old failed Qwen3 4B run does not pass these gates; do not set `reliability_passed` by hand.
- Stage genuine native `run_month` JSONL records with a separate passed gate report:

```bash
python -m unified_comparison.causal_record_export \
 --month-records /path/to/new_spytlt_agent_months.jsonl \
 --gate-report /path/to/new_spytlt_gate_report.json
# Then run python -m unified_comparison.run --grid --mapping both
```

The gate report must have `{"gates":{"G3a":{"passed":true},"G3b":{"passed":true}}}` plus independent supporting gate evidence. The staging script rejects engineering mocks, wrong asset universes, missing model hashes, failed agent steps, and incomplete output. The scorer also requires complete development and calibration-period coverage; it never copies IEF/LQD views to TLT or interpolates a missing signal.

## Output and integrity

- **Dashboard**: `site/unified.html` + `site/data/unified.json`
- **Generated**: `unified_comparison/out/summary.json`, `returns_spy{50,60,70}_{sign,ridge}.csv`, `views_weights_*.csv`.
- **Archived preview**: `unified_comparison/archived_reference/*`, labeled as *old, model-specific Ω*, not recalculated.
- **Protected test period**: 2023-01 onward remains sealed. Do not run or retune against it without a preregistered release decision.

### Important limitations

This is a fast engineering integration, **not a completed causal-model revalidation**. The old Causal RAG Testing Agent did not pass the original reliability gates. No genuine new SPY/TLT Causal RAG responses were present in the source ZIPs and none have been fabricated. The sample covariance and common Ω are methodological choices, not proof they improve investment performance. The Static portfolio deliberately does **not** pass through Black–Litterman; it is the passive benchmark.
