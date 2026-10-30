"""Unified, point-in-time SPY/TLT evaluation.

Five slots: STATIC, DET, JEV (1M), COMMITTEE (1M), CAUSAL_RAG (1M).
Causal RAG is not scored until genuine SPY/TLT inference passes the specified gates.
Each forecasting model uses the same view-calibration family, past-only DET Omega,
36-month sample covariance, Black-Litterman, constraints and accounting.
The original research pipelines are preserved separately.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage2" / "scoring"))
import stage2_backtest as s2  # noqa: E402
from native_backtest.framework import market as mk  # noqa: E402
from native_backtest.framework import records as rc  # noqa: E402
from native_backtest.framework.accounting import simulate_records  # noqa: E402
from native_backtest.framework.metrics import cer, holm, metrics, paired_cer_contrast  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402

HERE = Path(__file__).resolve().parent
CONF_NUM = {"Low": 0.0, "Medium": 1.0, "High": 2.0}


# ------------------------------------------------------------------ raw signals (one table per approach)
def month_of(as_of: str) -> pd.Period:
    return pd.Period(as_of[:7], "M")


def signals_det(spec: dict) -> pd.DataFrame:
    cfg = dict(spec["macro_panel"]); cfg["path"] = str(ROOT / cfg["path"])
    macro = s2.load_macro_panel(cfg)
    df = macro.set_index("month")[list(s2.MACRO_FEATURES)].copy()
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df.sort_index()


def signals_jev(path: Path) -> pd.DataFrame:
    rows = {}
    for as_of, j in json.loads(path.read_text(encoding="utf-8"))["months"].items():
        rows[month_of(as_of)] = {"p_stocks_beat_bonds": j["p"], "expected_tilt": j["offset"], "confidence": j["confidence"]}
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    # stated direction, sign-preserving mode: Jev's tilt is relative (stocks vs bonds), so +tilt for SPY, -tilt for TLT
    df.attrs["stated"] = {"SPY": df["expected_tilt"], "TLT": -df["expected_tilt"]}
    return df


def signals_committee(path: Path) -> pd.DataFrame:
    rows = {}
    for as_of, c in json.loads(path.read_text(encoding="utf-8"))["months"].items():
        f = c["cio"]
        ok = not c["problems"]
        rows[month_of(as_of)] = {"cio_spy": f["spy_q"] if ok else np.nan, "cio_tlt": f["tlt_q"] if ok else np.nan,
                                 "confidence": CONF_NUM.get(f["confidence"], np.nan) if ok else np.nan}
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.attrs["stated"] = {"SPY": df["cio_spy"], "TLT": df["cio_tlt"]}   # the CIO's own number per fund
    return df


def signals_causal(path: Path, macro: pd.DataFrame, *, required_months: Sequence[pd.Period]) -> pd.DataFrame | None:
    """Load native, independently generated 1-month SPY/TLT agent records; fail closed on incompatible runs.

    Input = JSONL with decision_month, horizon_months=1, universe=[SPY,TLT],
    model, engineering_mock=false, status_features={SPY:{F_S,F_C,F_I},TLT:{...}},
    and reliability_passed=true. No 3-asset substitutions or fabricated returns.
    The view calibration receives the same 8 macro surprises plus the 3 per-asset status counts.
    """
    if not path.exists():
        return None
    months = {}
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        m = pd.Period(r["decision_month"][:7], "M")
        if m in months:
            raise ValueError(f"duplicate Causal RAG month {m}")
        if r.get("horizon_months") != 1 or set(r.get("universe", [])) != {"SPY", "TLT"}:
            raise ValueError(f"Causal RAG month {m} is not SPY/TLT with 1-month horizon")
        if r.get("engineering_mock") or not r.get("reliability_passed") or not r.get("model"):
            raise ValueError(f"Causal RAG month {m} lacks independently validated real-model inference")
        feats = r.get("status_features", {})
        if set(feats) != {"SPY", "TLT"}:
            raise ValueError(f"Causal RAG month {m} has incomplete status features")
        for a in ("SPY", "TLT"):
            for k in ("F_S", "F_C", "F_I"):
                if k not in feats[a] or not np.isfinite(float(feats[a][k])):
                    raise ValueError(f"Causal RAG month {m} {a} missing finite {k}")
        months[m] = r
    need = set(required_months)
    if not need.issubset(months):
        raise ValueError(f"Causal RAG lacks {len(need-set(months))} required months; no imputation allowed")
    rows = {}
    for m,r in sorted(months.items()):
        if m not in macro.index:
            raise ValueError(f"Causal RAG month {m} missing macro features")
        rows[m] = {**{c: float(macro.at[m,c]) for c in macro.columns},
                   **{f"{a}_{k}": float(r['status_features'][a][k]) for a in ('SPY','TLT') for k in ('F_S','F_C','F_I')}}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


# ------------------------------------------------------------------ views and Omega (identical learner)
def _scale_fit(v: np.ndarray, y: np.ndarray) -> float:
    """Least-squares scale k >= 0 of y on centred v: the approach's direction is kept, only the size is learned."""
    vc = v - v.mean()
    den = float(vc @ vc)
    return max(0.0, float(vc @ (y - y.mean()) / den)) if den > 0 else 0.0


def view_and_omega(X: pd.DataFrame | None, y: pd.Series, dm: pd.Period, train_start: pd.Period, asset: str = "", mapping: str = "ridge"):
    """Q and inner-validation MSE for one fund at decision month dm. X None = historical average.

    mapping "ridge": Stage 2 Ridge on all of the approach's raw numbers (may learn any sign).
    mapping "sign": Q = average + k * (stated view - its mean), k >= 0 fitted on past months; the stated
    direction can never be inverted, and an anti-predictive signal collapses to the placebo (k = 0).
    """
    last = dm - 2
    months = [m for m in y.index if train_start <= m <= last and np.isfinite(y[m])]
    ytr = y.loc[months].to_numpy(dtype=float)
    if X is None:
        q = float(ytr.mean())
        errs = [(ytr[j] - ytr[tr].mean()) ** 2 for tr, j in s2.inner_splits(len(ytr))]
        return q, float(np.mean(errs)), "INTERCEPT_ONLY", len(ytr)
    if mapping == "sign" and "stated" in X.attrs:
        v_all = X.attrs["stated"][asset]
        v = v_all.reindex(months).to_numpy(dtype=float)
        v = np.where(np.isfinite(v), v, np.nanmean(v))          # an unusable month counts as "no view"
        x = float(v_all.get(dm, np.nan)); x = x if np.isfinite(x) else float(v.mean())
        errs = []
        for tr, j in s2.inner_splits(len(ytr)):
            k = _scale_fit(v[tr], ytr[tr]); errs.append((ytr[j] - (ytr[tr].mean() + k * (v[j] - v[tr].mean()))) ** 2)
        k = _scale_fit(v, ytr)
        return float(ytr.mean() + k * (x - v.mean())), float(np.mean(errs)), f"k={k:.4g}", len(ytr)
    if X is not None and X.attrs.get("causal_asset_features"):
        cols = [c for c in X.columns if not c.startswith(("SPY_", "TLT_"))] + [f"{asset}_{k}" for k in ("F_S", "F_C", "F_I")]
        sub = X[cols]
    else:
        sub = X
    Xtr = sub.reindex(months).to_numpy(dtype=float)
    x = sub.reindex([dm]).to_numpy(dtype=float)[0]
    alpha, _, losses = s2.choose_alpha(Xtr, ytr)
    q, _ = s2.ridge_fit_predict(Xtr, ytr, x, alpha)
    return float(q), float(losses[str(alpha)]), alpha, len(ytr)


# ------------------------------------------------------------------ allocation (Black-Litterman, bounded)
def allocate(sigma, q, omega, prior, lo, hi, delta, tau):
    pi = delta * sigma @ prior
    ts_inv = np.linalg.inv(tau * sigma)
    om_inv = np.diag(1.0 / omega)
    m = np.linalg.inv(ts_inv + om_inv)
    mu = m @ (ts_inv @ pi + om_inv @ q)
    S = sigma + m
    best, best_u = None, -math.inf
    n = len(mu)
    for states in itertools.product((0, 1, 2), repeat=n):
        w = np.zeros(n)
        free = [i for i, s in enumerate(states) if s == 0]
        fixed = [i for i, s in enumerate(states) if s != 0]
        for i in fixed:
            w[i] = lo[i] if states[i] == 1 else hi[i]
        rem = 1.0 - w[fixed].sum()
        if not free:
            if abs(rem) > 1e-12:
                continue
        else:
            b = mu[free] - (delta * S[np.ix_(free, fixed)] @ w[fixed] if fixed else 0.0)
            inv = np.linalg.inv(S[np.ix_(free, free)])
            one = np.ones(len(free))
            lam = (one @ inv @ b - delta * rem) / (one @ inv @ one)
            w[free] = inv @ (b - lam * one) / delta
        if (w < lo - 1e-12).any() or (w > hi + 1e-12).any():
            continue
        u = float(w @ mu - 0.5 * delta * w @ S @ w)
        if u > best_u + 1e-15:
            best, best_u = w.copy(), u
    return best, pi, mu


# ------------------------------------------------------------------ run
def run(cfg: dict, prior_spy: float, market, spec, months, raw, y_by_asset, mapping: str = "ridge") -> Dict:
    assets = cfg["assets"]
    prior = np.array([prior_spy, 1 - prior_spy])
    band = cfg["bounds_band"]
    lo, hi = np.clip(prior - band, 0, 1), np.clip(prior + band, 0, 1)
    train_start = pd.Period(cfg["train_start"], "M")
    excess = pd.DataFrame({a: y_by_asset[a] for a in assets})
    recs: Dict[str, List[rc.DecisionRecord]] = {k: [] for k in raw}
    detail_rows = []
    for dm in months:
        win = [m for m in excess.index if dm - 1 - cfg["cov_window_months"] <= m <= dm - 2]
        xw = excess.loc[win].to_numpy()
        sigma = LedoitWolf(assume_centered=False).fit(xw).covariance_ if cfg.get("covariance") == "ledoit_wolf" else np.cov(xw, rowvar=False, ddof=1)
        day = market.decision_dates[dm]
        t = mk.decision_time_iso(day)
        # A single, past-only Omega estimate from deterministic-model inner validation at this decision.
        # Do not use any model's contemporaneous evaluation errors (or the full 2019-22 period).
        common_omega = np.array([view_and_omega(raw["DET"], excess[a], dm, train_start, a, "ridge")[1] for a in assets])
        if not np.isfinite(common_omega).all() or (common_omega <= 0).any():
            raise ValueError(f"invalid common Omega for {dm}: {common_omega}")
        for name, X in raw.items():
            q, om, al = np.zeros(2), common_omega.copy(), []
            for i, a in enumerate(assets):
                q[i], _own_validation_mse, alpha, n = view_and_omega(X, excess[a], dm, train_start, a, "ridge" if name in ("DET", "CAUSAL_RAG") else mapping)
                al.append(str(alpha))
            w, pi, mu = allocate(sigma, q, om, prior, lo, hi, cfg["delta"], cfg["tau"])
            if w is None:
                raise ValueError(f"no feasible portfolio for {name} in {dm}")
            recs[name].append(rc.DecisionRecord(
                system_id=f"LPF_{name}", decision_month=str(dm), decision_time=t, info_cutoff=t, universe=list(assets),
                target_weights={a: float(w[i]) for i, a in enumerate(assets)},
                generation={"mode": "computed_native", "pipeline": "level_playing_field", "prior": dict(zip(assets, prior.tolist()))},
                forecast={"horizon_months": 1, "kind": "excess_return_view", "values": dict(zip(assets, q.tolist()))},
                confidence={"omega_diag": dict(zip(assets, om.tolist())), "ridge_alpha": dict(zip(assets, al))}))
            detail_rows.append({"system": name, "decision_month": str(dm), **{f"q_{a}": q[i] for i, a in enumerate(assets)},
                                **{f"omega_{a}": om[i] for i, a in enumerate(assets)}, **{f"pi_{a}": pi[i] for i, a in enumerate(assets)},
                                **{f"mu_{a}": mu[i] for i, a in enumerate(assets)}, **{f"w_{a}": w[i] for i, a in enumerate(assets)},
                                **{f"alpha_{a}": al[i] for i, a in enumerate(assets)}})
    static = [dataclasses.replace(r, system_id="LPF_STATIC", target_weights={a: float(prior[i]) for i, a in enumerate(assets)},
                                  generation={"mode": "computed_native", "benchmark": "static prior, monthly rebalance"},
                                  forecast=None, confidence=None) for r in recs[next(iter(recs))]]
    for lst in list(recs.values()) + [static]:
        for r in lst:
            rc.validate(r)
    sims = {k: simulate_records(v, market, months, cost_per_dollar=cfg["cost_per_dollar"]) for k, v in recs.items()}
    sims["STATIC"] = simulate_records(static, market, months, cost_per_dollar=cfg["cost_per_dollar"])
    b = cfg["bootstrap"]
    kw = dict(delta=cfg["delta"], block=b["block_months"], reps=b["reps"], seed=b["seed"])
    table = {k: {**metrics(s), "net_cer": cer(s["net_excess_return"], cfg["delta"]),
                 "mean_spy_weight": float(s[f"w_{assets[0]}"].mean()), "min_spy_weight": float(s[f"w_{assets[0]}"].min()),
                 "max_spy_weight": float(s[f"w_{assets[0]}"].max())} for k, s in sims.items()}
    approaches = list(raw)
    primary = [k for k in cfg["primary"] if k in raw]
    vs_static = {k: paired_cer_contrast(sims[k], sims["STATIC"], **kw) for k in approaches}
    vs_det = {k: paired_cer_contrast(sims[k], sims["DET"], **kw) for k in approaches if k != "DET"}
    return {"prior_spy": prior_spy, "mapping": mapping, "metrics": table,
            "vs_static": vs_static, "vs_static_holm": holm({k: vs_static[k]["p_two_sided_bootstrap"] for k in primary}),
            "vs_det": vs_det, "vs_det_holm": holm({k: vs_det[k]["p_two_sided_bootstrap"] for k in primary if k != "DET"}),
            "_detail": pd.DataFrame(detail_rows), "_sims": sims}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true", help="also run the other prior mixes in config.json")
    ap.add_argument("--mapping", default="both", choices=["ridge", "sign", "both"], help="how raw signals become views (see view_and_omega)")
    args = ap.parse_args(argv)
    cfg = json.loads((HERE / "config.json").read_text())
    fcfg = mk.load_config()
    # The original framework used a four-asset common calendar; restrict this new
    # experiment to SPY/TLT only, including hash checks and execution dates.
    fcfg["calendar_assets"] = list(cfg["assets"])
    fcfg["_prices"] = {a: fcfg["_prices"][a] for a in cfg["assets"]}
    try:
        market = mk.load_market(fcfg)
    except (FileNotFoundError, ValueError, mk.MarketDataError) as exc:
        raise SystemExit("Missing or invalid pinned market data; see unified_comparison/README.md. " + str(exc)) from exc
    spec = json.loads((ROOT / fcfg["inherits_stage2_run_spec"]["path"]).read_text())
    d0, d1 = cfg["development"]
    months = mk.months_between(d0, d1)
    if months[-1] >= pd.Period("2023-01", "M"):
        raise SystemExit("refusing: development window reaches the protected window")
    # completed monthly labels (excess return over the holding period) for every month we may train on
    label_months = mk.months_between("2012-01", d1)   # labels for training (from 2015-02) and the 36-month covariance
    y = {a: pd.Series({m: market.period_returns(m, [a])[a] - market.rf_return(m) for m in label_months}) for a in cfg["assets"]}
    src = cfg["sources"]
    raw = {"DET": signals_det(spec), "JEV": signals_jev(ROOT / src["jev_1m"]), "COMMITTEE": signals_committee(ROOT / src["committee_1m"])}
    causal_path = ROOT / src["causal_rag_1m"]
    causal_months = mk.months_between(cfg["train_start"], d1)
    causal = signals_causal(causal_path, raw["DET"], required_months=causal_months)
    if causal is not None:
        causal.attrs["causal_asset_features"] = True
        raw["CAUSAL_RAG"] = causal
    else:
        print("PENDING CAUSAL_RAG: no genuine validated 1-month SPY/TLT inference records; not scored")
    # a source with no answers would silently act as the placebo; skip it loudly instead
    need = mk.months_between(cfg["train_start"], d1)
    for name in [k for k, X in raw.items() if X is not None]:
        cov = raw[name].reindex(need).notna().all(axis=1).mean() if raw[name].shape[1] else 0.0
        if cov < 0.9:
            print(f"SKIPPED {name}: answers for only {cov:.0%} of {need[0]}..{need[-1]} (need 90%); run collect_1m.py first")
            del raw[name]
    cfg["primary"] = [k for k in cfg["primary"] if k in raw]
    out_dir = HERE / "out"; out_dir.mkdir(exist_ok=True)
    mixes = [cfg["prior_weights"]["SPY"]] + ([p for p in cfg["prior_grid"] if abs(p - cfg["prior_weights"]["SPY"]) > 1e-9] if args.grid else [])
    summary = {}
    mappings = ["ridge", "sign"] if args.mapping == "both" else [args.mapping]
    for p, mapping in itertools.product(mixes, mappings):
        res = run(cfg, p, market, spec, months, raw, y, mapping)
        tag = f"spy{int(round(p * 100))}_{mapping}"
        res.pop("_detail").to_csv(out_dir / f"views_weights_{tag}.csv", index=False)
        pd.concat(res.pop("_sims").values()).to_csv(out_dir / f"returns_{tag}.csv", index=False)
        summary[tag] = res
        m = res["metrics"]
        print(f"\nprior {p:.0%} SPY / {1 - p:.0%} TLT  ({len(months)} development months)")
        for k in [k for k in ["STATIC", "DET", "JEV", "COMMITTEE", "CAUSAL_RAG"] if k in m]:
            line = f"  {k:13s} net CER {m[k]['net_cer'] * 100:6.2f}%  SPY {m[k]['min_spy_weight'] * 100:4.1f}-{m[k]['max_spy_weight'] * 100:4.1f}%"
            if k in res["vs_static"]:
                c = res["vs_static"][k]
                line += f"  vs static {c['cer_difference'] * 100:+6.2f} pts  95% [{c['ci95'][0] * 100:+.2f}, {c['ci95'][1] * 100:+.2f}]  p={c['p_two_sided_bootstrap']:.2f}"
            print(line)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    from unified_comparison.dashboard_export import export_dashboard
    export_dashboard(out_dir, summary, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
