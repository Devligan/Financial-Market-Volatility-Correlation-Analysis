"""Offline tests for the extended analysis layer (``finrisk.compute_extended``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import finrisk as an

LOOKBACK = 252


@pytest.fixture(scope="module")
def bundle() -> an.AnalysisBundle:
    rng = np.random.default_rng(17)
    n_days = 1300
    dates = pd.bdate_range("2017-01-02", periods=n_days)
    daily = rng.normal(loc=[0.0004, 0.0002, 0.0003, 0.0001], scale=[0.010, 0.020, 0.014, 0.012], size=(n_days, 4))
    prices = pd.DataFrame(100 * np.cumprod(1 + daily, axis=0), index=dates, columns=["AAA", "BBB", "CCC", "SPY"])
    return an.compute_all(tickers=list(prices.columns), prices=prices, rf=0.04, benchmark="SPY", mc_portfolios=200)


@pytest.fixture(scope="module")
def extended(bundle) -> an.ExtendedAnalysis:
    return an.compute_extended(bundle, lookback=LOOKBACK, cost_bps=10.0, var_window=250)


def test_compute_extended_integrity(extended, bundle):
    assert not extended.backtest.stats.empty
    assert {"Scenario", "Portfolio", "Total_Return", "Max_Drawdown"} <= set(extended.scenario_table.columns)
    assert "Max Sharpe" in extended.scenario_returns.columns
    assert not extended.risk_summary.empty
    assert not extended.risk_contributions.empty
    assert {"Portfolio", "Level", "P_Value", "Verdict"} <= set(extended.var_backtest.columns)
    assert len(extended.vol_forecast) == bundle.returns.shape[1]


def test_var_backtest_uses_the_same_portfolio_series_everywhere(bundle, extended):
    # The extended suite's VaR table must be reproducible from the bundle's own
    # portfolio choices, so the dashboard, report and notebook never disagree.
    expected = an.var_backtest_summary(
        {name: an.portfolio_returns(bundle.returns, weights) for name, weights in bundle.portfolio_choices().items()},
        window=250,
    )
    pd.testing.assert_frame_equal(extended.var_backtest, expected)


def test_extended_summary_bullets(bundle, extended):
    bullets = an.executive_summary(bundle, extended=extended)
    assert any("backtest" in bullet.lower() for bullet in bullets)
    plain = an.executive_summary(bundle)
    assert len(bullets) > len(plain)


def test_save_outputs_with_extended_writes_all_artifacts(tmp_path, bundle, extended):
    written = an.save_outputs(bundle, extended, outdir=tmp_path)
    names = {path.name for path in written}
    assert names == {
        "financial_analysis_summary.csv",
        "financial_analysis_summary.xlsx",
        "asset_correlations.csv",
        "portfolio_optimization.csv",
        "financial_report.html",
        "backtest_performance.csv",
        "backtest_equity_curves.csv",
        "scenario_analysis.csv",
        "risk_contributions.csv",
        "var_backtest.csv",
        "volatility_forecasts.csv",
    }
    assert all(path.exists() for path in written)

    report = (tmp_path / "financial_report.html").read_text(encoding="utf-8")
    assert "Walk-Forward Backtest" in report
    assert "Stress Scenarios" in report
    assert "Risk Decomposition" in report
    assert "VaR Backtest" in report
    assert "GARCH(1,1)" in report
    # Forecast_vs_Realized is a ratio; it must not be scaled to percent display.
    assert "Forecast_vs_Realized (%)" not in report

    with pd.ExcelFile(tmp_path / "financial_analysis_summary.xlsx") as workbook:
        assert {
            "risk_return_metrics",
            "correlations",
            "portfolio_weights",
            "calendar_year_returns",
            "backtest_performance",
            "scenarios",
            "risk_contributions",
            "var_backtest",
            "volatility_forecasts",
        } <= set(workbook.sheet_names)
