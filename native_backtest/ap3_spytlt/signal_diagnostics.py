"""Development-only Causal RAG signal diagnostics; NO return labels are read.

A large number of 'INTERCEPT_ONLY' ridge fits can reflect weak information, low
feature variation, or cancellation of opposing hypotheses. This report separates
those engineering possibilities. It does not claim any investment prediction edge.

Run: python -m native_backtest.ap3_spytlt.signal_diagnostics --records MONTHS.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from native_backtest.ap3_spytlt.encoder import ASSETS, status_features


def summarize(records: list[dict]) -> dict:
    total = {a: Counter() for a in ASSETS}
    months = {a: [] for a in ASSETS}
    for record in records:
        hypotheses = record.get("hypothesis_agent", {}).get("hypotheses", [])
        tests = record.get("tests", [])
        encoded = status_features(hypotheses, tests)
        t_by_id = {t["hypothesis_id"]: t for t in tests}
        for asset in ASSETS:
            counts = Counter()
            for h in hypotheses:
                direction = h["predicted_assets"][asset]
                test = t_by_id[h["hypothesis_id"]]
                if direction == 0:
                    continue
                if test.get("failure"):
                    counts["failures"] += 1
                    continue
                counts["active_hypotheses"] += 1
                counts[f"{test['status']}_positive" if direction > 0 else f"{test['status']}_negative"] += 1
            if counts["active_hypotheses"] and all(abs(encoded[asset][f]) < 1e-12 for f in ("F_S", "F_C", "F_I")):
                total[asset]["fully_cancelled_months"] += 1
            if not counts["active_hypotheses"]:
                total[asset]["zero_valid_signal_months"] += 1
            total[asset].update(counts)
            months[asset].append({"month": record["decision_month"], "features": encoded[asset], "counts": dict(counts)})
    return {"methodology": "no outcomes read; exploratory feature audit only", "months": len(records),
            "by_asset": {a: {"summary": dict(total[a]), "per_month": months[a]} for a in ASSETS}}


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, required=True, help="JSONL of native SPY/TLT month records")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    rows = [json.loads(s) for s in args.records.read_text().splitlines() if s.strip()]
    result = summarize(rows)
    content = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content + "\n")
    else:
        print(content)


if __name__ == "__main__":
    main()
