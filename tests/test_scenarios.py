"""Offline tests for stress scenarios (``finrisk.scenarios``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import finrisk as an


@pytest.fixture(scope="module")
def scenario_returns() -> pd.DataFrame:
    rng = np.random.default_rng(23)
    dates = pd.bdate_range("2018-01-01", "2024-06-28")
    daily = rng.normal(loc=0.0003, scale=[0.010, 0.015, 0.020], size=(len(dates), 3))
    return pd.DataFrame(daily, index=dates, columns=["AAA", "BBB", "CCC"])


def test_window_total_compounds_within_bounds():
    series = pd.Series(
        [0.10, -0.10, 0.05],
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
    )
    # (1.10 * 0.90) - 1 = -0.01
    assert an.window_total(series, "2020-01-02", "2020-01-03") == pytest.approx(-0.01)


def test_window_outside_sample_is_nan(scenario_returns):
    series = scenario_returns["AAA"]
    assert np.isnan(an.window_total(series, "1999-01-01", "1999-12-31"))
    assert np.isnan(an.window_max_drawdown(series, "1999-01-01", "1999-12-31"))


def test_scenario_summary_is_tidy_and_bounded(scenario_returns):
    weights_map = {
        "Equal Weight": an.equal_weights(scenario_returns.shape[1]),
        "All AAA": np.array([1.0, 0.0, 0.0]),
    }
    summary = an.scenario_summary(scenario_returns, weights_map)
    assert list(summary.columns) == ["Scenario", "Portfolio", "Total_Return", "Max_Drawdown"]
    assert len(summary) == len(an.SCENARIOS) * len(weights_map)
    valid = summary.dropna()
    assert not valid.empty
    assert (valid["Max_Drawdown"] <= 0).all()


def test_scenario_return_table_shape(scenario_returns):
    weights = {"Equal Weight": an.equal_weights(scenario_returns.shape[1])}
    table = an.scenario_return_table(scenario_returns, weights)
    assert list(table.index) == list(an.SCENARIOS)
    assert list(table.columns) == ["Equal Weight"]


def test_scenario_asset_table_sorted_worst_first(scenario_returns):
    table = an.scenario_asset_table(scenario_returns, "COVID crash (2020)")
    assert list(table.columns) == ["Total_Return", "Max_Drawdown"]
    assert table["Total_Return"].is_monotonic_increasing
