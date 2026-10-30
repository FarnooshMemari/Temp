"""Generate dashboard data only from identifiable saved runs; never invent unavailable results."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

NAMES={"STATIC":"Static 60/40", "DET":"Deterministic (macro / Ridge)", "JEV":"Jev (1-month)", "COMMITTEE":"Committee (1-month)", "CAUSAL_RAG":"Causal RAG (1-month)"}
ORDER=list(NAMES)

def _series(df: pd.DataFrame, key: str):
    d=df[df.system_id==f"LPF_{key}"].sort_values('decision_month')
    if d.empty: return None
    return {"months":d.decision_month.tolist(),"nav":(100*(1+d.net_total_return).cumprod()).round(8).tolist(),"w_spy":d.w_SPY.round(8).tolist()}

def build_data(out:Path, summary:dict, *, provenance:str, pending: list[str]):
    payload={"schema_version":1,"provenance":provenance,"evaluation":"development only (2019-03–2022-12)", "assets":["SPY","TLT"],"one_month_only":True,"systems":ORDER,"pending":pending,"runs":{}}
    for key,res in sorted(summary.items()):
        path=out/f"returns_{key}.csv"
        if not path.exists():continue
        sims=pd.read_csv(path)
        series={}
        for name in ORDER:
            ser=_series(sims,name)
            if ser is not None:
                m=res["metrics"][name]
                series[name]={"label":(f"Static {int(res.get('prior_spy', .6)*100)}/{100-int(res.get('prior_spy', .6)*100)}" if name=="STATIC" else NAMES[name]),"net_cer":float(m['net_cer']),"sharpe":float(m['net_sharpe']),"maxdd":float(m['max_drawdown']),"turnover":float(m['annualized_one_way_turnover']),**ser}
                if name in res['vs_static']:
                    c=res['vs_static'][name]
                    series[name]['vs_static']={"d":float(c['cer_difference']),"lo":float(c['ci95'][0]),"hi":float(c['ci95'][1]),"p":float(c['p_two_sided_bootstrap'])}
        payload['runs'][key]={"systems":series}
    return payload

def export_dashboard(out:Path,summary:dict,cfg:dict)->Path:
    # A fresh recalculation is the only way to change status from 'archived' to 'updated'.
    pending=['CAUSAL_RAG'] if not any('CAUSAL_RAG' in v['metrics'] for v in summary.values()) else []
    data=build_data(out,summary,provenance='updated: common expanding past-only DET Ω, sample covariance, 1-month SPY/TLT, new backtest',pending=pending)
    dest=Path(__file__).resolve().parents[1]/'site/data/unified.json'
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(data,indent=2)+'\n')
    return dest

def archived_preview(dest: Path | None = None):
    """Create unmistakably labeled archive from past model-specific-Omega output for display only."""
    here=Path(__file__).resolve().parent
    src=here/'archived_reference'
    summary=json.loads((src/'summary.json').read_text())
    a={'spy60_sign':summary['spy60_sign']}
    data=build_data(src,a,provenance='ARCHIVED PREVIEW ONLY — old model-specific Ω. NOT results of updated methodology. Recompute after fetching pinned market data.',pending=['CAUSAL_RAG'])
    dest=dest or here.parents[0]/'site/data/unified.json';dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(data,indent=2)+'\n')
    return dest

if __name__=='__main__':
    print(archived_preview())
