"""Deprecated in the SPY/TLT fork: allocation is handled exclusively by unified_comparison.run.
The Causal RAG fork generates status features only. Legacy three-asset allocation is unchanged in native_backtest.ap3.
"""

def weights_for_months(*args, **kwargs):
    raise RuntimeError("Use unified_comparison.run for common SPY/TLT allocation and Omega")

def omega_from_development_errors(*args, **kwargs):
    raise RuntimeError("Use unified_comparison.run for common past-only Omega")
