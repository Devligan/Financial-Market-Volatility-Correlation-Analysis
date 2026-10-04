"""Analysis pipeline: run the full engine once and package the results in AnalysisBundle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .attribution import risk_contribution_table, risk_summary
from .backtest import DEFAULT_COST_BPS, DEFAULT_LOOKBACK, BacktestResult, walk_forward_backtest
from .config import BENCHMARK, DEFAULT_RF
from .data import fetch_prices
from .metrics import compute_metrics, compute_returns, drawdown_series
from .portfolio import (
    efficient_frontier,
    equal_weights,
    monte_carlo_portfolios,
    optimize_max_sharpe,
    optimize_min_variance,
    portfolio_returns,
)
from .scenarios import scenario_drawdown_table, scenario_return_table, scenario_summary
from .universe import default_tickers
from .var_backtest import var_backtest_summary
from .volatility import forecast_universe


@dataclass
class AnalysisBundle:
    """Everything the dashboard and notebook need for one analysis run."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    metrics: pd.DataFrame
    correlation: pd.DataFrame
    drawdowns: pd.DataFrame
    monte_carlo: pd.DataFrame
    max_sharpe: pd.Series
    min_variance: pd.Series
    frontier: pd.DataFrame
    benchmark: str | None
    risk_free: float
    fetched_at: datetime
    source: str = "live"
    error: str | None = None

    @property
    def equal_weight(self) -> pd.Series:
        return pd.Series(equal_weights(self.returns.shape[1]), index=self.returns.columns, name="Equal_Weight")

    def portfolio_choices(self) -> dict[str, pd.Series]:
        choices = {
            "Max Sharpe": self.max_sharpe,
            "Min Variance": self.min_variance,
            "Equal Weight": self.equal_weight,
        }
        if self.benchmark and self.benchmark in self.returns.columns:
            weights = pd.Series(0.0, index=self.returns.columns, name=f"Benchmark ({self.benchmark})")
            weights[self.benchmark] = 1.0
            choices[f"Benchmark ({self.benchmark})"] = weights
        return choices


def compute_all(
    tickers: Sequence[str] | None = None,
    prices: pd.DataFrame | None = None,
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
    rf: float = DEFAULT_RF,
    benchmark: str | None = BENCHMARK,
    mc_portfolios: int = 10_000,
    seed: int = 42,
    source: str = "live",
    error: str | None = None,
) -> AnalysisBundle:
    """Run the full pipeline and return an :class:`AnalysisBundle`."""
    tickers = [t.upper() for t in (tickers or default_tickers())]
    if prices is None:
        prices = fetch_prices(tickers, period=period, start=start, end=end)
    # Align the panel: keep only dates where every analyzed asset has data, so all
    # downstream matrix math (covariance, portfolios, Monte Carlo) stays NaN-free.
    prices = prices.reindex(columns=[t for t in tickers if t in prices.columns]).dropna()
    if prices.shape[1] == 0:
        raise RuntimeError("No overlapping price history for the requested tickers")

    returns = compute_returns(prices)
    benchmark = benchmark if benchmark in returns.columns else None
    metrics = compute_metrics(returns, benchmark=benchmark, rf=rf)

    return AnalysisBundle(
        prices=prices,
        returns=returns,
        metrics=metrics,
        correlation=returns.corr(),
        drawdowns=drawdown_series(returns),
        monte_carlo=monte_carlo_portfolios(returns, n_portfolios=mc_portfolios, rf=rf, seed=seed),
        max_sharpe=optimize_max_sharpe(returns, rf=rf),
        min_variance=optimize_min_variance(returns),
        frontier=efficient_frontier(returns),
        benchmark=benchmark,
        risk_free=rf,
        fetched_at=datetime.now(),
        source=source,
        error=error,
    )


@dataclass
class ExtendedAnalysis:
    """Out-of-sample and risk-model analytics layered on top of :class:`AnalysisBundle`."""

    backtest: BacktestResult
    scenario_returns: pd.DataFrame
    scenario_drawdowns: pd.DataFrame
    scenario_table: pd.DataFrame
    risk_summary: pd.DataFrame
    risk_contributions: pd.DataFrame
    var_backtest: pd.DataFrame
    vol_forecast: pd.DataFrame


def compute_extended(
    bundle: AnalysisBundle,
    lookback: int = DEFAULT_LOOKBACK,
    cost_bps: float = DEFAULT_COST_BPS,
    var_window: int = 500,
    var_levels: tuple[float, ...] = (0.05, 0.01),
    garch_horizon: int = 21,
    include_volatility: bool = True,
) -> ExtendedAnalysis:
    """Run the extended suite: walk-forward backtest, stress scenarios and risk models."""
    returns = bundle.returns
    weights_map = bundle.portfolio_choices()

    backtest = walk_forward_backtest(
        returns,
        benchmark=bundle.benchmark,
        lookback=lookback,
        cost_bps=cost_bps,
        rf=bundle.risk_free,
    )
    # VaR backtests run on the same fixed-weight portfolio series the dashboard,
    # scenario table and risk decomposition use, so every view agrees. The
    # walk-forward result stays the out-of-sample validation in its own section.
    var_table = var_backtest_summary(
        {name: portfolio_returns(returns, weights) for name, weights in weights_map.items()},
        window=var_window,
        levels=var_levels,
    )
    vol_forecast = forecast_universe(returns, horizon=garch_horizon) if include_volatility else pd.DataFrame()
    return ExtendedAnalysis(
        backtest=backtest,
        scenario_returns=scenario_return_table(returns, weights_map),
        scenario_drawdowns=scenario_drawdown_table(returns, weights_map),
        scenario_table=scenario_summary(returns, weights_map),
        risk_summary=risk_summary(returns, weights_map, rf=bundle.risk_free),
        risk_contributions=risk_contribution_table(returns, weights_map),
        var_backtest=var_table,
        vol_forecast=vol_forecast,
    )
