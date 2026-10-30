"""No-model regression tests for the NEW SPY/TLT Causal RAG fork.

All fixtures here are synthetic engineering checks, NOT an 80%-accuracy evaluation.
"""
from __future__ import annotations

import pytest

from native_backtest.ap3_spytlt.encoder import status_features
from native_backtest.ap3_spytlt.signal_diagnostics import summarize
from native_backtest.ap3_spytlt.testing_agent import ASSESS_SYSTEM, validate_edge_coverage
from native_backtest.ap3_spytlt.llm_client import ModelOutputError
from unified_comparison.causal_record_export import stage_records


def hyp(i: int, spy=1, tlt=-1):
    return {"hypothesis_id": f"H{i}", "predicted_assets": {"SPY": spy, "TLT": tlt}}


def result(i: int, st="Insufficient", *, failed=False, agreement=1.0):
    return {"hypothesis_id": f"H{i}", "status": st, "run_agreement": agreement,
            **({"failure": "model output invalid"} if failed else {})}


def test_failed_model_call_is_not_ordinary_insufficient_evidence():
    feat = status_features([hyp(1)], [result(1, failed=True)])
    assert feat["SPY"]["F_I"] == 0
    assert feat["SPY"]["F_FAIL"] == 1
    assert feat["TLT"]["F_I"] == 0
    assert feat["TLT"]["F_FAIL"] == -1
    assert feat["SPY"]["F_VALID_COUNT"] == 0


def test_conflicting_directions_cancel_in_legacy_but_remain_visible_in_diagnostic():
    hs = [hyp(1, spy=1, tlt=0), hyp(2, spy=-1, tlt=0)]
    ts = [result(1, "Supported", agreement=.8), result(2, "Supported", agreement=1)]
    f = status_features(hs, ts)
    assert f["SPY"]["F_S"] == 0
    assert f["SPY"]["F_VALID_COUNT"] == 2
    assert f["SPY"]["F_AGREE_NET"] == pytest.approx(-.1)
    report = summarize([{"decision_month": "2019-03", "hypothesis_agent": {"hypotheses": hs}, "tests": ts}])
    assert report["by_asset"]["SPY"]["summary"]["fully_cancelled_months"] == 1


def test_missing_and_duplicate_hypothesis_results_fail_closed():
    with pytest.raises(ValueError, match="missing"):
        status_features([hyp(1)], [])
    with pytest.raises(ValueError, match="duplicate"):
        status_features([hyp(1)], [result(1), result(1)])
    with pytest.raises(ValueError, match="unexpected"):
        status_features([hyp(1)], [result(2)])
    with pytest.raises(ValueError, match="duplicate"):
        status_features([hyp(1), hyp(1)], [result(1)])


def test_explicit_failed_status_is_an_audit_signal_only():
    features = status_features([hyp(1)], [result(1, "Failed", failed=True)])
    assert features["SPY"]["F_FAIL"] == 1
    assert features["SPY"]["F_I"] == 0
    assert features["SPY"]["F_VALID_COUNT"] == 0


def test_runner_records_execution_failure_separately(monkeypatch):
    from types import SimpleNamespace
    from native_backtest.ap3_spytlt import runner
    h = {**hyp(1), "paths": {}, "trigger": "CPI", "channel": "rates",
         "confidence_applied": "low", "confidence_caps": []}
    monkeypatch.setattr(runner.hypothesis_agent, "run", lambda *args: {"hypotheses": [h]})
    def fail(*args):
        raise ModelOutputError("invalid assessor output")
    monkeypatch.setattr(runner.testing_agent, "test_hypothesis", fail)
    client = SimpleNamespace(provider=SimpleNamespace(name="engineering", model="fixture"), is_mock=True)
    record = runner.run_month(client, {"decision_month": "2019-03", "decision_time": "2019-03-29T17:00:00-04:00"},
                              SimpleNamespace(year=2018), {"entries": [], "library_sha256": "fixture"}, [], None, None)
    assert record["tests"][0]["status"] == "Failed"
    assert record["status_features"]["SPY"]["F_I"] == 0
    assert record["status_features"]["SPY"]["F_FAIL"] == 1


def test_directions_and_agreement_are_validated():
    with pytest.raises(ValueError, match="direction"):
        status_features([hyp(1, spy=10)], [result(1)])
    with pytest.raises(ValueError, match="seed agreement"):
        status_features([hyp(1)], [result(1, agreement=None)])


def test_assessor_duplicate_missing_unknown_edges_rejected():
    edges = [{"id": "H:SPY"}, {"id": "H:TLT"}]
    a = lambda edge: {"edge_id": edge, "label": "insufficient_evidence",
                      "supporting_evidence": [], "contradicting_evidence": [], "conflict": False, "explanation": "n/a"}
    for got in ([a("H:SPY")], [a("H:SPY"), a("H:SPY")], [a("H:SPY"), a("R:SPY")]):
        with pytest.raises(ModelOutputError, match="edge coverage"):
            validate_edge_coverage({"assessments": got}, edges, set(), {})


def test_valid_quote_required_and_directions_not_confused_in_prompt():
    edges = [{"id": "H:TLT"}]
    assess = {"edge_id": "H:TLT", "label": "supported", "supporting_evidence": [{"chunk_id": "id", "doc_id": "doc", "quote": "made up"}],
              "contradicting_evidence": [], "conflict": False, "explanation": "x"}
    items = validate_edge_coverage({"assessments": [assess]}, edges, {"id"}, {"id": {"doc_id": "doc", "text": "Treasury yields rose."}})
    assert items[0]["validated_label"] == "insufficient_evidence"
    assert "TLT PRICES" in ASSESS_SYSTEM


def test_staging_refuses_failed_tests_and_strips_diagnostic_columns():
    x = status_features([hyp(1)], [result(1, "Supported")])
    r = {"decision_month": "2019-03", "decision_time": "2019-03-29T17:00:00-04:00", "model": "real-model",
         "record_sha256": "deadbeef", "engineering_mock": False, "status_features": x,
         "hypothesis_agent": {"hypotheses": [hyp(1)]}, "tests": [result(1, "Supported")]}
    gates = {"gates": {"G3a": {"passed": True}, "G3b": {"passed": True}}}
    staged = stage_records([r], gates)
    assert set(staged[0]["status_features"]["SPY"]) == {"F_S", "F_C", "F_I"}
    r["tests"] = [result(2, "Supported")]
    with pytest.raises(ValueError, match="unexpected"):
        stage_records([r], gates)
    r["tests"] = [result(1, "Contradicted")]
    with pytest.raises(ValueError, match="do not match"):
        stage_records([r], gates)
    r["tests"] = [result(1, failed=True)]
    with pytest.raises(ValueError, match="Testing Agent failure"):
        stage_records([r], gates)
