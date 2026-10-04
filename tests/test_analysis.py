"""Offline unit tests for the ``finrisk`` analytics engine.

Everything runs on deterministic synthetic data — no network access required.

Run from the project root:

    pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import finrisk as an

RF = 0.04


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    """Synthetic price panel with three assets of distinct risk profiles."""
    rng = np.random.default_rng(7)
    n_days = 750
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    daily = rng.normal(loc=[0.0004, 0.0001, 0.0002], scale=[0.010, 0.020, 0.015], size=(n_days, 3))
    levels = 100 * np.cumprod(1 + daily, axis=0)
    return pd.DataFrame(levels, index=dates, columns=["AAA", "BBB", "CCC"])


@pytest.fixture(scope="module")
def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return an.compute_returns(prices)


# ---------------------------------------------------------------------------
# Returns & annualization
# ---------------------------------------------------------------------------
def test_annualized_return_matches_compounded_daily_rate():
    daily = pd.DataFrame({"X": [0.001] * an.TRADING_DAYS})
    cagr = an.annualized_return(daily)["X"]
    assert cagr == pytest.approx(1.001**an.TRADING_DAYS - 1, rel=1e-9)


def test_annualized_volatility_scales_linearly(returns):
    assert np.allclose(an.annualized_volatility(2 * returns), 2 * an.annualized_volatility(returns))


# ---------------------------------------------------------------------------
# Drawdowns
# ---------------------------------------------------------------------------
def test_max_drawdown_known_path():
    rets = pd.DataFrame({"X": [0.10, -0.50, 0.20]})
    assert an.max_drawdown(rets)["X"] == pytest.approx(-0.50)


def test_drawdown_series_never_positive(returns):
    drawdowns = an.drawdown_series(returns)
    assert (drawdowns <= 1e-12).all().all()


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------
def test_compute_metrics_table_integrity(returns):
    metrics = an.compute_metrics(returns, benchmark="BBB", rf=RF)
    assert list(metrics.index) == list(returns.columns)
    expected = {
        "Asset_Class",
        "Annualized_Return",
        "Annualized_Volatility",
        "Sharpe_Ratio",
        "Sortino_Ratio",
        "Max_Drawdown",
        "Calmar_Ratio",
        "VaR_95",
        "CVaR_95",
        "VaR_99",
        "CVaR_99",
        "Skewness",
        "Kurtosis",
        "Beta",
        "Alpha",
        "Correlation_with_Market",
        "R_Squared",
        "Tracking_Error",
        "Information_Ratio",
        "Up_Capture",
        "Down_Capture",
        "Data_Points",
    }
    assert expected <= set(metrics.columns)
    assert (metrics["Data_Points"] == len(returns)).all()
    assert metrics.at["BBB", "Beta"] == pytest.approx(1.0)
    assert metrics.at["BBB", "Correlation_with_Market"] == pytest.approx(1.0)
    assert (metrics["Max_Drawdown"] <= 0).all()
    assert (metrics["VaR_95"] >= 0).all()


def test_benchmark_relative_metrics(returns):
    metrics = an.compute_metrics(returns, benchmark="BBB", rf=RF)
    active = returns.sub(returns["BBB"], axis=0)["AAA"]
    assert metrics.at["AAA", "Tracking_Error"] == pytest.approx(active.std() * np.sqrt(an.TRADING_DAYS))
    # The benchmark has zero active risk against itself
    assert metrics.at["BBB", "Tracking_Error"] == pytest.approx(0.0)
    assert np.isnan(metrics.at["BBB", "Information_Ratio"])
    assert metrics.at["BBB", "Up_Capture"] == pytest.approx(1.0)
    assert metrics.at["BBB", "Down_Capture"] == pytest.approx(1.0)
    assert metrics.at["AAA", "Up_Capture"] > 0


def test_calendar_year_returns():
    series = pd.Series(
        [0.01, 0.02, -0.01, 0.03],
        index=pd.DatetimeIndex(["2020-01-02", "2020-06-01", "2021-01-04", "2021-07-01"]),
    )
    yearly = an.calendar_year_returns(series)
    assert list(yearly.index) == [2020, 2021]
    assert yearly.loc[2020] == pytest.approx(1.01 * 1.02 - 1)
    assert yearly.loc[2021] == pytest.approx(0.99 * 1.03 - 1)


def test_calendar_year_returns_dataframe(returns):
    yearly = an.calendar_year_returns(returns)
    assert list(yearly.columns) == list(returns.columns)
    assert yearly.index.name == "Year"
    assert len(yearly) == returns.index.year.nunique()


def test_drawdown_episode_durations_are_trading_days():
    # Peak Friday Jan 3 -> trough Monday Jan 6: one trading day, three calendar days.
    series = pd.Series([0.05, 0.0, -0.50, 1.30], index=pd.bdate_range("2020-01-02", periods=4))
    episodes = an.drawdown_episodes(series, top=1)
    row = episodes.iloc[0]
    assert row["Trough"] == pd.Timestamp("2020-01-06")
    assert row["Trough_Days"] == 1
    assert row["Recovery_Days"] == 1


def test_sortino_uses_excess_downside_deviation():
    series = [-0.012, 0.004, -0.008, 0.010, -0.020]
    returns = pd.DataFrame({"X": series})
    metrics = an.compute_metrics(returns, benchmark=None, rf=RF)
    mean_ann = returns["X"].mean() * an.TRADING_DAYS
    excess = returns["X"] - RF / an.TRADING_DAYS
    downside_dev = np.sqrt((excess.clip(upper=0) ** 2).mean()) * np.sqrt(an.TRADING_DAYS)
    assert metrics.at["X", "Sortino_Ratio"] == pytest.approx((mean_ann - RF) / downside_dev)

    # The portfolio-level statistic must use the same definition.
    stats, _ = an.portfolio_stats(returns, [1.0], rf=RF)
    assert stats["Sortino_Ratio"] == pytest.approx(metrics.at["X", "Sortino_Ratio"])


def test_up_down_capture_uses_average_monthly_returns():
    # Two months: the benchmark rises in month 1 and falls in month 2.
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-02-03", "2020-02-04"])
    market = pd.Series([0.02, -0.01, -0.03, 0.01], index=dates)
    asset = market * 2.0
    metrics = an.compute_metrics(pd.DataFrame({"MKT": market, "X": asset}), benchmark="MKT", rf=RF)
    monthly = (1 + pd.DataFrame({"MKT": market, "X": asset})).groupby(dates.to_period("M")).prod() - 1
    up, down = monthly["MKT"] > 0, monthly["MKT"] < 0
    assert up.any() and down.any()
    expected_up = monthly.loc[up, "X"].mean() / monthly.loc[up, "MKT"].mean()
    expected_down = monthly.loc[down, "X"].mean() / monthly.loc[down, "MKT"].mean()
    assert metrics.at["X", "Up_Capture"] == pytest.approx(expected_up)
    assert metrics.at["X", "Down_Capture"] == pytest.approx(expected_down)
    assert metrics.at["MKT", "Up_Capture"] == pytest.approx(1.0)
    assert metrics.at["MKT", "Down_Capture"] == pytest.approx(1.0)


def test_up_down_capture_stays_realistic_over_long_samples():
    # Compounding every daily up day for a decade once produced ratios in the 1e5 range;
    # monthly compounding must keep capture ratios bounded and meaningful.
    rng = np.random.default_rng(9)
    market = pd.Series(rng.normal(0.0004, 0.01, size=2500), index=pd.bdate_range("2013-01-02", periods=2500))

    # A beta-1 asset should capture both sides close to 1.0.
    peer = market + rng.normal(0, 0.002, size=2500)
    peer_metrics = an.compute_metrics(pd.DataFrame({"MKT": market, "X": peer}), benchmark="MKT", rf=RF)
    assert 0.7 < peer_metrics.at["X", "Up_Capture"] < 1.4
    assert 0.7 < peer_metrics.at["X", "Down_Capture"] < 1.4

    # A 2x daily-reset asset compounds above 2.0, but must stay in a sane range
    # (the bug this guards against produced ~136,000).
    levered = 2.0 * market
    lev_metrics = an.compute_metrics(pd.DataFrame({"MKT": market, "X": levered}), benchmark="MKT", rf=RF)
    assert 2.0 < lev_metrics.at["X", "Up_Capture"] < 100.0
    assert 1.0 < lev_metrics.at["X", "Down_Capture"] < 100.0


def test_drawdown_episodes_with_recovery_and_ongoing():
    series = pd.Series(
        [0.10, -0.20, 0.25, -0.50, -0.10, 1.00],
        index=pd.bdate_range("2020-01-01", periods=6),
    )
    episodes = an.drawdown_episodes(series, top=5)
    assert len(episodes) == 2
    ongoing = episodes.iloc[0]  # deepest first
    assert ongoing["Depth"] == pytest.approx(-0.55)
    assert ongoing["Peak"] == series.index[2]
    assert ongoing["Trough"] == series.index[4]
    assert pd.isna(ongoing["Recovery"]) and pd.isna(ongoing["Recovery_Days"])
    recovered = episodes.iloc[1]
    assert recovered["Depth"] == pytest.approx(-0.20)
    assert recovered["Recovery"] == series.index[2]
    assert recovered["Trough_Days"] == 1


def test_zero_variance_and_zero_drawdown_metrics_are_nan_not_inf():
    # A constant-return series has zero volatility and never draws down; the
    # Sharpe and Calmar ratios must be undefined (NaN), never infinite.
    flat = pd.DataFrame({"FLAT": [0.0005] * 120})
    metrics = an.compute_metrics(flat, benchmark=None, rf=RF)
    assert np.isnan(metrics.at["FLAT", "Sharpe_Ratio"])
    assert np.isnan(metrics.at["FLAT", "Calmar_Ratio"])
    assert np.isnan(metrics.at["FLAT", "Sortino_Ratio"])
    assert not np.isinf(metrics.select_dtypes(include=[np.number]).to_numpy()).any()


def test_cvar_is_at_least_var(returns):
    metrics = an.compute_metrics(returns, benchmark="BBB", rf=RF)
    for tail in ("95", "99"):
        assert (metrics[f"CVaR_{tail}"] >= metrics[f"VaR_{tail}"] - 1e-12).all()


def test_class_summary_covers_all_assets(returns):
    metrics = an.compute_metrics(returns, benchmark=None, rf=RF)
    summary = an.class_summary(metrics)
    assert summary["Assets"].sum() == len(returns.columns)


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------
def test_correlation_helpers(returns):
    corr = returns.corr()
    assert -1.0 <= an.average_correlation(corr) <= 1.0
    (high_a, high_b, high), (low_a, low_b, low) = an.correlation_extremes(corr)
    assert high >= low
    assert {high_a, high_b} <= set(corr.columns) and {low_a, low_b} <= set(corr.columns)


def test_average_correlation_needs_two_assets(returns):
    single = returns[["AAA"]].corr()
    assert np.isnan(an.average_correlation(single))


# ---------------------------------------------------------------------------
# Weights & portfolios
# ---------------------------------------------------------------------------
def test_normalize_weights_clips_and_normalizes():
    weights = an.normalize_weights([-1.0, 1.0, 3.0])
    assert weights[0] == 0.0
    assert weights.sum() == pytest.approx(1.0)
    assert np.allclose(an.normalize_weights({"a": 1.0, "b": 3.0}.values()), [0.25, 0.75])


def test_normalize_weights_rejects_all_zero():
    with pytest.raises(ValueError):
        an.normalize_weights([0.0, 0.0, 0.0])


def test_equal_weights_sum_to_one():
    assert an.equal_weights(7).sum() == pytest.approx(1.0)


def test_portfolio_returns_matches_weighted_sum(returns):
    weights = np.array([0.2, 0.3, 0.5])
    expected = returns.to_numpy() @ weights
    assert np.allclose(an.portfolio_returns(returns, weights).to_numpy(), expected)
    assert np.allclose(an.portfolio_returns(returns, [2.0, 3.0, 5.0]).to_numpy(), expected)


def test_portfolio_stats_are_consistent(returns):
    stats, series = an.portfolio_stats(returns, an.equal_weights(returns.shape[1]), rf=RF)
    assert stats["Total_Return"] == pytest.approx((1 + series).prod() - 1)
    assert stats["Annualized_Volatility"] > 0
    assert stats["Max_Drawdown"] <= 0
    assert stats["VaR_95"] >= 0


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
def test_min_variance_beats_equal_weight_volatility(returns):
    weights = an.optimize_min_variance(returns)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= -1e-9).all() and (weights <= 1 + 1e-9).all()
    mv_stats, _ = an.portfolio_stats(returns, weights, rf=RF)
    eq_stats, _ = an.portfolio_stats(returns, an.equal_weights(returns.shape[1]), rf=RF)
    assert mv_stats["Annualized_Volatility"] <= eq_stats["Annualized_Volatility"] + 1e-9


def test_max_sharpe_beats_equal_weight_sharpe(returns):
    weights = an.optimize_max_sharpe(returns, rf=RF)
    ms_stats, _ = an.portfolio_stats(returns, weights, rf=RF)
    eq_stats, _ = an.portfolio_stats(returns, an.equal_weights(returns.shape[1]), rf=RF)
    assert ms_stats["Sharpe_Ratio"] >= eq_stats["Sharpe_Ratio"] - 1e-6


def test_max_sharpe_recovers_after_transient_solver_failure(monkeypatch):
    from types import SimpleNamespace

    import finrisk.portfolio as portfolio_module

    rng = np.random.default_rng(3)
    returns = pd.DataFrame(
        rng.normal(loc=0.0003, scale=0.01, size=(400, 3)),
        index=pd.bdate_range("2020-01-02", periods=400),
        columns=["A", "B", "C"],
    )
    real_minimize = portfolio_module.minimize
    calls = {"count": 0}

    def flaky_minimize(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return SimpleNamespace(success=False, message="forced first-attempt failure", x=args[1])
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(portfolio_module, "minimize", flaky_minimize)
    weights = an.optimize_max_sharpe(returns, rf=RF)
    assert calls["count"] >= 2  # the restart ladder was used
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()


def test_efficient_frontier_is_monotone(returns):
    frontier = an.efficient_frontier(returns, points=25)
    assert len(frontier) >= 10
    assert (np.diff(frontier["Annualized_Return"].to_numpy()) > 0).all()
    assert (np.diff(frontier["Annualized_Volatility"].to_numpy()) > -1e-9).all()


def test_monte_carlo_is_seeded_and_well_formed(returns):
    first = an.monte_carlo_portfolios(returns, n_portfolios=500, rf=RF, seed=42)
    second = an.monte_carlo_portfolios(returns, n_portfolios=500, rf=RF, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 500
    assert np.isfinite(first.to_numpy()).all()


def test_monte_carlo_volatility_matches_engine_sample_std(returns):
    # The MC cloud must use the same sample standard deviation (ddof=1) as
    # portfolio_stats and the metrics table, not numpy's population default.
    n = 3
    mc = an.monte_carlo_portfolios(returns, n_portfolios=n, rf=RF, seed=7)
    rng = np.random.default_rng(7)
    for i, weights in enumerate(rng.dirichlet(np.ones(returns.shape[1]), size=n)):
        series = an.portfolio_returns(returns, weights)
        assert mc.iloc[i]["Annualized_Volatility"] == pytest.approx(series.std() * np.sqrt(an.TRADING_DAYS), rel=1e-12)


# ---------------------------------------------------------------------------
# Orchestration & I/O
# ---------------------------------------------------------------------------
def test_compute_all_bundle_integrity(prices):
    bundle = an.compute_all(tickers=list(prices.columns), prices=prices, rf=RF, benchmark="BBB", mc_portfolios=300)
    assert bundle.metrics.shape[0] == prices.shape[1]
    assert bundle.correlation.shape == (prices.shape[1], prices.shape[1])
    assert np.allclose(bundle.correlation.to_numpy(), bundle.correlation.to_numpy().T)
    assert list(bundle.drawdowns.columns) == list(bundle.returns.columns)
    assert bundle.equal_weight.sum() == pytest.approx(1.0)
    assert {"Max Sharpe", "Min Variance", "Equal Weight"} <= set(bundle.portfolio_choices())
    assert an.executive_summary(bundle)


def test_executive_summary_without_benchmark_does_not_crash(prices):
    bundle = an.compute_all(tickers=list(prices.columns), prices=prices, rf=RF, benchmark=None, mc_portfolios=200)
    assert bundle.benchmark is None
    bullets = an.executive_summary(bundle)
    assert bullets and any("drawdown" in bullet.lower() for bullet in bullets)
    assert "n/a" in an.report_html(bundle)


def test_executive_summary_single_asset_without_benchmark(prices):
    bundle = an.compute_all(tickers=["AAA"], prices=prices[["AAA"]], rf=RF, benchmark=None, mc_portfolios=50)
    assert bundle.benchmark is None
    assert an.executive_summary(bundle)


def test_compute_all_aligns_partial_nan_rows(prices):
    gappy = prices.copy()
    gappy.iloc[5:9, 0] = np.nan
    bundle = an.compute_all(tickers=list(gappy.columns), prices=gappy, rf=RF, benchmark="BBB", mc_portfolios=100)
    assert bundle.returns.notna().all().all()
    assert not bundle.max_sharpe.isna().any()
    assert np.isfinite(bundle.monte_carlo.to_numpy()).all()


def test_fetch_prices_rejects_empty_universe():
    with pytest.raises(ValueError):
        an.fetch_prices([])


def test_save_outputs_writes_all_artifacts(tmp_path, prices):
    bundle = an.compute_all(tickers=list(prices.columns), prices=prices, rf=RF, benchmark="BBB", mc_portfolios=100)
    written = an.save_outputs(bundle, outdir=tmp_path)
    names = {path.name for path in written}
    assert names == {
        "financial_analysis_summary.csv",
        "financial_analysis_summary.xlsx",
        "asset_correlations.csv",
        "portfolio_optimization.csv",
        "financial_report.html",
    }
    assert all(path.exists() for path in written)

    report = (tmp_path / "financial_report.html").read_text(encoding="utf-8")
    assert "Calendar-Year Returns" in report
    assert "Information_Ratio" in report

    with pd.ExcelFile(tmp_path / "financial_analysis_summary.xlsx") as workbook:
        assert {"risk_return_metrics", "correlations", "portfolio_weights", "calendar_year_returns"} <= set(
            workbook.sheet_names
        )


def test_save_and_load_prices_roundtrip(tmp_path):
    frame = pd.DataFrame(
        {"AAA": [10.0, 11.5, 12.25], "BBB": [100.0, 99.5, 101.0]},
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
    )
    path = tmp_path / "snapshot.csv"
    an.save_prices(frame, path)
    loaded = an.load_prices(path, tickers=["AAA", "BBB"])
    pd.testing.assert_frame_equal(loaded, frame, check_exact=False, rtol=1e-12, check_names=False)
    assert list(an.load_prices(path, tickers=["AAA"]).columns) == ["AAA"]
    with pytest.raises(RuntimeError):
        an.load_prices(path, tickers=["ZZZ"])
