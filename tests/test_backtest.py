"""Offline tests for the walk-forward backtest engine (``finrisk.backtest``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import finrisk as an

LOOKBACK = 120


@pytest.fixture(scope="module")
def backtest_returns() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n_days = 420
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    daily = rng.normal(loc=[0.0004, 0.0001, 0.0002, 0.00015], scale=[0.010, 0.020, 0.015, 0.012], size=(n_days, 4))
    return pd.DataFrame(daily, index=dates, columns=["AAA", "BBB", "CCC", "SPY"])


def test_rebalance_schedule_uses_month_ends():
    index = pd.bdate_range("2020-01-01", "2020-03-15")
    schedule = an.rebalance_schedule(index)
    assert list(schedule) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28"), pd.Timestamp("2020-03-13")]


def test_backtest_requires_enough_history(backtest_returns):
    with pytest.raises(ValueError):
        an.walk_forward_backtest(backtest_returns.iloc[:100], lookback=LOOKBACK)


def test_equal_weight_equals_daily_cross_sectional_mean(backtest_returns):
    result = an.walk_forward_backtest(backtest_returns, benchmark="SPY", lookback=LOOKBACK, cost_bps=0.0)
    series = result.daily_returns["Equal Weight"].dropna()
    expected = backtest_returns.mean(axis=1).loc[series.index]
    assert np.allclose(series.to_numpy(), expected.to_numpy())


def test_backtest_holds_weights_between_rebalances(backtest_returns):
    result = an.walk_forward_backtest(backtest_returns, lookback=LOOKBACK, cost_bps=0.0)
    for name, history in result.weights_history.items():
        assert not history.empty, name
        assert np.allclose(history.sum(axis=1), 1.0)
        assert list(history.columns) == list(backtest_returns.columns)


def test_transaction_costs_reduce_performance(backtest_returns):
    free = an.walk_forward_backtest(backtest_returns, lookback=LOOKBACK, cost_bps=0.0)
    costed = an.walk_forward_backtest(backtest_returns, lookback=LOOKBACK, cost_bps=50.0)
    for name in an.STRATEGIES:
        assert costed.stats.loc[name, "Total_Return"] <= free.stats.loc[name, "Total_Return"] + 1e-12
    assert costed.stats.loc["Max Sharpe", "Total_Return"] < free.stats.loc["Max Sharpe", "Total_Return"]
    assert costed.stats.loc["Max Sharpe", "Annual_Cost"] > 0


def test_backtest_stats_and_equity_curves(backtest_returns):
    result = an.walk_forward_backtest(backtest_returns, benchmark="SPY", lookback=LOOKBACK)
    assert {"Max Sharpe", "Min Variance", "Equal Weight", "SPY"} <= set(result.stats.index)
    curves = result.equity_curves()
    expected_first = 100 * (1 + result.daily_returns.iloc[0]).to_numpy()
    assert np.allclose(curves.iloc[0].to_numpy(), expected_first)
    assert (curves > 0).all().all()
    assert set(curves.columns) == set(result.daily_returns.columns)
    assert result.lookback == LOOKBACK
    assert result.stats.loc["Equal Weight", "Start"] < result.stats.loc["Equal Weight", "End"]
    # Metric columns must stay numeric next to the timestamp columns.
    assert result.stats["Sharpe_Ratio"].dtype.kind == "f"
    assert result.stats["Avg_Turnover"].dtype.kind == "f"
