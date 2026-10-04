"""Offline tests for risk decomposition (``finrisk.attribution``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import finrisk as an

RF = 0.04


@pytest.fixture(scope="module")
def attribution_returns() -> pd.DataFrame:
    rng = np.random.default_rng(31)
    dates = pd.bdate_range("2016-01-01", periods=600)
    daily = rng.normal(loc=0.0003, scale=[0.008, 0.016, 0.024], size=(600, 3))
    return pd.DataFrame(daily, index=dates, columns=["LOW", "MID", "HIGH"])


def test_risk_contributions_sum_to_portfolio_volatility(attribution_returns):
    weights = np.array([0.5, 0.3, 0.2])
    frame = an.risk_contributions(attribution_returns, weights)
    _, series = an.portfolio_stats(attribution_returns, weights, rf=RF)
    portfolio_vol = float(series.std() * np.sqrt(an.TRADING_DAYS))
    assert frame["Risk_Contribution"].sum() == pytest.approx(portfolio_vol, rel=1e-9)
    assert frame["Risk_Percent"].sum() == pytest.approx(1.0, rel=1e-9)
    assert (frame["Risk_Percent"] >= 0).all()


def test_single_asset_has_no_diversification(attribution_returns):
    frame = an.risk_contributions(attribution_returns, np.array([1.0, 0.0, 0.0]))
    assert frame.loc["LOW", "Risk_Percent"] == pytest.approx(1.0)
    assert an.diversification_ratio(attribution_returns, np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert an.concentration(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_diversification_reduces_risk(attribution_returns):
    equal = an.equal_weights(attribution_returns.shape[1])
    assert an.diversification_ratio(attribution_returns, equal) > 1.0
    assert an.concentration(equal) == pytest.approx(1 / 3)


def test_risk_summary_and_table_shapes(attribution_returns):
    weights_map = {
        "Equal Weight": an.equal_weights(attribution_returns.shape[1]),
        "Low only": np.array([1.0, 0.0, 0.0]),
    }
    summary = an.risk_summary(attribution_returns, weights_map)
    assert list(summary.index) == list(weights_map)
    assert summary.loc["Low only", "Top_Risk_Asset"] == "LOW"

    table = an.risk_contribution_table(attribution_returns, weights_map)
    assert set(table["Portfolio"]) == set(weights_map)
    assert len(table) == attribution_returns.shape[1] * len(weights_map)
