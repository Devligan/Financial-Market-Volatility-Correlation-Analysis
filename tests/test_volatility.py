"""Offline tests for GARCH volatility models (``finrisk.volatility``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import finrisk as an


def _synthetic_garch(n: int = 1200, seed: int = 3) -> pd.Series:
    """Daily returns from a GARCH(1,1) process with persistence 0.95."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.000004, 0.10, 0.85
    returns = np.zeros(n)
    variance = np.zeros(n)
    variance[0] = omega / (1 - alpha - beta)
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]
        returns[t] = np.sqrt(variance[t]) * shocks[t]
    return pd.Series(returns, index=pd.bdate_range("2015-01-01", periods=n))


def test_garch_fit_recovers_persistence():
    params = an.garch_fit(_synthetic_garch())
    assert 0.80 < params["Persistence"] < 1.0
    assert params["Alpha"] > 0
    assert params["Beta"] > 0
    assert np.isfinite(params["LogLikelihood"])


def test_garch_forecast_is_positive_and_plausible():
    forecast = an.garch_forecast(_synthetic_garch(), horizon=21)
    assert 0.05 < forecast["Forecast_Vol"] < 0.60
    assert forecast["Conditional_Vol"] > 0
    assert forecast["Forecast_Vol"] == pytest.approx(forecast["Conditional_Vol"], rel=0.5)


def test_forecast_universe_table():
    calm = pd.Series(
        np.random.default_rng(4).normal(0, 0.004, size=1200),
        index=pd.bdate_range("2015-01-01", periods=1200),
    )
    returns = pd.concat({"AAA": _synthetic_garch(), "BBB": calm}, axis=1)
    frame = an.forecast_universe(returns, horizon=21)
    assert set(frame.index) == {"AAA", "BBB"}
    assert {
        "Conditional_Vol",
        "Forecast_Vol",
        "Persistence",
        "Realized_Vol_60d",
        "Forecast_vs_Realized",
        "Regime",
        "Name",
    } <= set(frame.columns)
    assert frame["Forecast_Vol"].notna().all()
    assert frame["Forecast_Vol"].is_monotonic_decreasing
    assert set(frame["Regime"]) <= {"Elevated", "Calm", "Normal", "n/a"}


def test_garch_fit_requires_history():
    with pytest.raises(ValueError):
        an.garch_fit(_synthetic_garch().iloc[:100])
