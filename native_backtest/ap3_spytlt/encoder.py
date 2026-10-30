"""Status features: the three Testing Agent outcomes kept distinct.

For asset a in month t: F^S = sum of the predicted directions s_{h,a} of hypotheses whose majority
status is Supported; F^C the same for Contradicted; F^I for Insufficient. A month with no
hypotheses has all three at 0. No status is scored +1/-1 against another.
"""
from __future__ import annotations

from typing import Dict, List

ASSETS = ("SPY", "TLT")
STATUS_COL = {"Supported": "F_S", "Contradicted": "F_C", "Insufficient": "F_I"}


def status_features(hypotheses: List[dict], tests: List[dict]) -> Dict[str, Dict[str, float]]:
    """Evidence outcomes are not the same thing as inference failures.

    F_S/F_C/F_I preserve the original pre-registered signals for valid judgments.
    F_FAIL is an *audit-only* signed count; an unsuccessful call must never masquerade
    as observed insufficient evidence. F_NET and F_AGREE_NET are optional diagnostics
    showing signed, normalised causal-vote strength and seed agreement. Do not use
    them as predictors without a fresh, pre-declared development-only evaluation.
    """
    by_id = {t["hypothesis_id"]: t for t in tests}
    if len(by_id) != len(tests):
        raise ValueError("duplicate hypothesis_id in Testing Agent results")
    wanted = [h["hypothesis_id"] for h in hypotheses]
    if len(set(wanted)) != len(wanted):
        raise ValueError("duplicate hypothesis_id in hypotheses")
    if set(by_id) - set(wanted):
        raise ValueError("unexpected hypothesis_id in Testing Agent results")
    out = {a: {"F_S": 0.0, "F_C": 0.0, "F_I": 0.0, "F_FAIL": 0.0,
               "F_NET": 0.0, "F_AGREE_NET": 0.0, "F_VALID_COUNT": 0.0} for a in ASSETS}
    for h in hypotheses:
        tid = h["hypothesis_id"]
        if tid not in by_id:
            raise ValueError(f"hypothesis {tid} is missing from Testing Agent results")
        t = by_id[tid]
        st = t.get("status")
        failed = bool(t.get("failure"))
        if not failed and st not in STATUS_COL:
            raise ValueError(f"hypothesis {tid} has no valid status ({st})")
        agree = t.get("run_agreement")
        if not failed and (agree is None or not 0.0 <= float(agree) <= 1.0):
            raise ValueError(f"hypothesis {tid} lacks valid seed agreement")
        for a in ASSETS:
            direction = float(h["predicted_assets"][a])
            if direction not in (-1.0, 0.0, 1.0):
                raise ValueError(f"invalid predicted direction {direction} for {tid}/{a}")
            if failed:
                out[a]["F_FAIL"] += direction
                continue
            out[a][STATUS_COL[st]] += direction
            if direction:
                out[a]["F_VALID_COUNT"] += 1.0
            sign = 1 if st == "Supported" else (-1 if st == "Contradicted" else 0)
            out[a]["F_NET"] += sign * direction
            out[a]["F_AGREE_NET"] += sign * direction * float(agree)
    for a in ASSETS:
        n = out[a]["F_VALID_COUNT"]
        if n:
            out[a]["F_NET"] /= n
            out[a]["F_AGREE_NET"] /= n
    return out
