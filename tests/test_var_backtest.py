"""Offline tests for VaR backtesting (``finrisk.var_backtest``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import finrisk as an


def test_rolling_var_has_no_lookahead():
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.normal(0, 0.01, size=120), index=pd.bdate_range("2020-01-01", periods=120))
    returns.iloc[60] = -0.50  # a huge loss must not influence the VaR forecast for the same day
    var = an.rolling_var(returns, window=50, level=0.05)
    manual = -np.quantile(returns.iloc[10:60].to_numpy(), 0.05)
    assert var.iloc[60] == pytest.approx(manual)
    assert var.iloc[:50].isna().all()
    assert (var.dropna() > 0).all()


def test_breach_flags_mark_worse_than_var():
    returns = pd.Series([0.01, -0.02, -0.03, 0.02])
    var = pd.Series([0.015, 0.015, 0.015, 0.015])
    flags = an.breach_flags(returns, var)
    assert flags.tolist() == [False, True, True, False]


def test_kupiec_pof_exact_proportion_has_zero_statistic():
    returns = pd.Series([-0.02] * 5 + [0.01] * 95)
    var = pd.Series([0.01] * 100)
    result = an.kupiec_pof(returns, var, level=0.05)
    assert result["Observations"] == 100
    assert result["Breaches"] == 5
    assert result["Breach_Rate"] == pytest.approx(0.05)
    assert result["LR_Statistic"] == pytest.approx(0.0, abs=1e-9)
    assert result["P_Value"] == pytest.approx(1.0)


def test_var_backtest_summary_coverage():
    rng = np.random.default_rng(7)
    series = pd.Series(rng.normal(0, 0.01, size=3000), index=pd.bdate_range("2010-01-01", periods=3000))
    summary = an.var_backtest_summary({"Synthetic": series}, window=500, levels=(0.05,))
    assert list(summary.columns) == [
        "Portfolio",
        "Level",
        "Observations",
        "Breaches",
        "Breach_Rate",
        "Expected_Rate",
        "LR_Statistic",
        "P_Value",
        "Verdict",
        "Avg_Breach_Loss",
    ]
    row = summary.iloc[0]
    assert row["Verdict"] in {"Pass", "Reject"}
    assert 0.02 <= row["Breach_Rate"] <= 0.09
    assert row["Expected_Rate"] == pytest.approx(0.05)
