"""Offline regression tests for the unified comparison; no protected data access."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from unified_comparison import run
from unified_comparison.causal_record_export import stage_records
from unified_comparison.dashboard_export import archived_preview


def _series():
    t=pd.period_range('2015-02','2019-03',freq='M'); rng=np.random.default_rng(13)
    return t, pd.Series(rng.normal(.004,.03,len(t)),index=t),pd.DataFrame(rng.normal(size=(len(t),8)),index=t)

def test_omega_no_future_information():
    months,y,X=_series(); dm=pd.Period('2019-03','M')
    _,om,*_=run.view_and_omega(X,y,dm,months[0], 'SPY','ridge')
    yy=y.copy();yy[dm-1]=12345.0;yy[dm]=-23456.0
    _,om_changed,*_=run.view_and_omega(X,yy,dm,months[0], 'SPY','ridge')
    assert np.isclose(om,om_changed)

def test_causal_rejects_3_assets_and_bad_mock(tmp_path):
    macro=pd.DataFrame({'c':[0.0]},index=pd.PeriodIndex(['2019-03'],freq='M'))
    p=tmp_path/'ap3.jsonl';base={'decision_month':'2019-03','model':'qwen3','horizon_months':1,'universe':['SPY','IEF','LQD'],
      'engineering_mock':False,'reliability_passed':True,'status_features':{'SPY':{'F_S':1,'F_C':0,'F_I':0},'TLT':{'F_S':0,'F_C':0,'F_I':1}}}
    p.write_text(json.dumps(base)+'\n')
    with pytest.raises(ValueError,match='SPY/TLT'):
        run.signals_causal(p,macro,required_months=[pd.Period('2019-03','M')])
    base['universe']=['SPY','TLT'];base['engineering_mock']=True;p.write_text(json.dumps(base)+'\n')
    with pytest.raises(ValueError,match='validated real-model'):
        run.signals_causal(p,macro,required_months=[pd.Period('2019-03','M')])

def test_causal_requires_reliability_gates():
    with pytest.raises(ValueError,match='reliability gate'):
        stage_records([],{'gates':{'G3a':{'passed':True},'G3b':{'passed':False}}})

def test_fork_asset_schema():
    from native_backtest.ap3_spytlt import hypothesis_agent,testing_agent,encoder,graph_vintage
    assert encoder.ASSETS==('SPY','TLT')
    sch=hypothesis_agent.turn_schema(['CPI'],['UST10Y'],1)
    req=sch['oneOf'][1]['properties']['hypotheses']['items']['properties']['predicted_assets']['required']
    assert req==['SPY','TLT']
    assert testing_agent.PLAN_SCHEMA['properties']['rival']['properties']['predicted_assets']['required']==['SPY','TLT']
    assert set(graph_vintage.load_concepts()['assets'])=={'SPY','TLT'}

def test_dashboard_preview_is_truthfully_archived(tmp_path):
    dest=archived_preview(tmp_path/'unified.json');d=json.loads(dest.read_text())
    assert d['provenance'].startswith('ARCHIVED PREVIEW')
    assert 'CAUSAL_RAG' in d['pending']
    assert 'PLACEBO' not in d['systems']
    assert d['one_month_only']


def test_shared_omega_and_two_asset_full_accounting_synthetic():
    """End-to-end integration on synthetic development data; no claimed market results."""
    tr = pd.period_range('2015-02','2019-06',freq='M')
    rng=np.random.default_rng(19)
    y={a:pd.Series(rng.normal(mu,sd,len(tr)),index=tr) for a,mu,sd in [('SPY',.005,.04),('TLT',.003,.025)]}
    macro=pd.DataFrame(rng.normal(size=(len(tr),8)),index=tr)
    jev=pd.DataFrame(rng.normal(size=(len(tr),3)),index=tr)
    committee=pd.DataFrame(rng.normal(size=(len(tr),3)),index=tr)
    months=list(pd.period_range('2019-03','2019-06',freq='M'))
    class FakeMarket:
        decision_dates={m:m.to_timestamp(how='end').normalize() for m in months}
        def period_returns(self,m,assets):return {a:float(y[a][m])+.001 for a in assets}
        def rf_return(self,m):return .001
        def holding_period(self,m):return (m.to_timestamp()+pd.Timedelta(days=1), (m+1).to_timestamp()+pd.Timedelta(days=1))
    cfg={'assets':['SPY','TLT'],'bounds_band':.2,'delta':2.5,'tau':.05,'cov_window_months':36,'cost_per_dollar':.0005,
         'train_start':'2015-02','covariance':'sample','bootstrap':{'block_months':2,'reps':40,'seed':7},'primary':['DET','JEV','COMMITTEE']}
    out=run.run(cfg,.6,FakeMarket(),None,months,{'DET':macro,'JEV':jev,'COMMITTEE':committee},y,'ridge')
    d=out['_detail']
    for m in months:
        rows=d[d.decision_month==str(m)]
        assert len(rows)==3
        assert rows.omega_SPY.nunique()==1
        assert rows.omega_TLT.nunique()==1
    assert set(out['metrics'])=={'STATIC','DET','JEV','COMMITTEE'}
    for sim in out['_sims'].values():
        assert (sim.w_SPY.between(.4-.00001,.8+.00001)).all()
        assert np.allclose(sim.w_SPY+sim.w_TLT,1)
