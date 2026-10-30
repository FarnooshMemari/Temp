"""Validate and stage genuine SPY/TLT Causal RAG agent records for unified evaluation.

Never converts old three-asset forecast returns into TLT forecasts. This script consumes
native month records from native_backtest.ap3_spytlt.runner and an independent quality
report. A failing/missing quality report means NO Causal RAG result is staged.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import pandas as pd
from native_backtest.ap3_spytlt.encoder import status_features

ASSETS={'SPY','TLT'}
FEATURES={'F_S','F_C','F_I'}

def stage_records(records: list[dict], report: dict) -> list[dict]:
    gates=report.get('gates', {})
    if not {'G3a','G3b'} <= gates.keys() or any(gates[k].get('passed') is not True for k in ('G3a','G3b')):
        raise ValueError('Causal RAG reliability gate G3a or G3b not independently passed; refusing results')
    out=[];seen=set()
    for r in records:
        m=r.get('decision_month')
        pd.Period(m,'M')
        if m in seen: raise ValueError(f'duplicate month: {m}')
        seen.add(m)
        if r.get('engineering_mock') is not False:
            raise ValueError(f'{m}: engineering mock/unverified inference cannot be scored')
        if not r.get('model') or not r.get('record_sha256'):
            raise ValueError(f'{m}: provenance missing')
        if str(r.get('decision_time',''))[:7] != m:
            raise ValueError(f'{m}: decision_time is not within decision month')
        if r.get('hypothesis_agent',{}).get('failure'):
            raise ValueError(f'{m}: hypothesis agent failed')
        x=r.get('status_features',{})
        if set(x)!=ASSETS: raise ValueError(f'{m}: expected exact SPY/TLT features; got {list(x)}')
        for a in ASSETS:
            if not FEATURES <= set(x[a]) or any(not math.isfinite(float(v)) for v in x[a].values()):
                raise ValueError(f'{m}: invalid status features for {a}')
        if any(t.get('failure') for t in r.get('tests',[])):
            raise ValueError(f'{m}: Testing Agent failure')
        hypotheses = r.get('hypothesis_agent', {}).get('hypotheses', [])
        recomputed = status_features(hypotheses, r.get('tests', []))
        if any(float(x[a][k]) != recomputed[a][k] for a in ASSETS for k in FEATURES):
            raise ValueError(f'{m}: status features do not match Testing Agent judgments')
        out.append({'decision_month':m,'decision_time':r['decision_time'], 'horizon_months':1,'universe':['SPY','TLT'],
                    'model':r['model'],'source_sha256':r['record_sha256'], 'engineering_mock':False,
                    'reliability_passed':True,'status_features':{a:{k:x[a][k] for k in sorted(FEATURES)} for a in sorted(ASSETS)}})
    return sorted(out,key=lambda r:r['decision_month'])

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--month-records',required=True,type=Path,help='JSONL generated with native_backtest.ap3_spytlt.runner')
    p.add_argument('--gate-report',required=True,type=Path,help='Independent gate assessment: {"gates":{"G3a":{"passed":true},"G3b":{"passed":true}}}')
    p.add_argument('--out',type=Path,default=Path('unified_comparison/data/causal_rag_spytlt_1m.jsonl'))
    a=p.parse_args(argv)
    records=[json.loads(s) for s in a.month_records.read_text().splitlines() if s.strip()]
    staged=stage_records(records,json.loads(a.gate_report.read_text()))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in staged))
    print(f'Staged {len(staged)} Causal RAG monthly records into {a.out}; coverage checked again by unified runner')
if __name__=='__main__':main()
