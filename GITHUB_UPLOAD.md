# Upload this updated Claude Artifact project to GitHub

1. Extract **the whole ZIP**. The top-level folder is the updated repository; don't upload the ZIP as a single file if you want GitHub to host the code.
2. Make a backup branch of the original repository before replacing any files. This package is based on the provided **protofin-backtest** branch and merges the verified one-month signals / evaluation logic from **protofin-level-playing-field**; it is **not** an exact continuation of every historical branch.
3. Commit the new directories `unified_comparison/` and `native_backtest/ap3_spytlt/`, plus `site/unified.html`, `site/data/unified.json`, and this documentation. Existing research code and legacy dashboard remain present.
4. On a machine with access to the pinned market data, follow `unified_comparison/README.md` to fetch SHA-verified inputs and recalculate the shared evaluation. The packaged dashboard JSON is a clearly labeled **ARCHIVED PREVIEW** until the rerun completes.
5. For Cloudflare Pages (if your deployment currently serves `site/`), deploy the `site/` directory and open `/unified.html`. The archived warning is visible until a valid fresh evaluator run regenerates `site/data/unified.json`.
6. Causal RAG is **pending**: do not publish its former three-asset performance as a five-approach two-asset result. New SPY/TLT agent inference and reliability gates require your inference environment.

## What was verified without external data

- Static imports and bytecode compilation of the updated modules.
- 10 offline tests covering forecast time guard, allocation bounds, sign mapping, shared-source schema, the two-asset AP3 fork, provenance and archive labels.
- Live rerun is blocked by absent pinned market files; no performance metrics under the *new common Ω* are asserted by this package.
