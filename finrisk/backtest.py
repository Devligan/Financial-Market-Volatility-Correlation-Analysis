"""Walk-forward portfolio backtesting: rolling estimation, monthly rebalancing and costs.

Each rebalance date uses **only** the trailing ``lookback`` observations to estimate
weights; those weights are then applied out-of-sample until the next rebalance. One-way
trading costs are charged on turnover at every rebalance, so the resulting performance
is an implementable estimate rather than an in-sample fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_RF, TRADING_DAYS
from .portfolio import (
    equal_weights,
    optimize_max_sharpe,
    optimize_min_variance,
    series_stats,
)

DEFAULT_LOOKBACK = 756  # ~3 years of daily observations per estimate
DEFAULT_COST_BPS = 10.0  # one-way transaction cost in basis points
STRATEGIES = ("Max Sharpe", "Min Variance", "Equal Weight")


@dataclass
class BacktestResult:
    """Out-of-sample daily returns, the performance table and the weights actually held."""

    daily_returns: pd.DataFrame
    stats: pd.DataFrame
    weights_history: dict[str, pd.DataFrame]
    lookback: int
    cost_bps: float

    def equity_curves(self, start_value: float = 100.0) -> pd.DataFrame:
        """Growth of ``start_value`` for every backtested strategy and the benchmark."""
        return start_value * (1 + self.daily_returns).cumprod()


def rebalance_schedule(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month present in ``index``."""
    grouped = index.to_series().groupby(index.to_period("M")).max()
    return pd.DatetimeIndex(grouped.to_numpy())


def _estimate_weights(train: pd.DataFrame, strategy: str, rf: float) -> pd.Series:
    if strategy == "Max Sharpe":
        return optimize_max_sharpe(train, rf=rf)
    if strategy == "Min Variance":
        return optimize_min_variance(train)
    return pd.Series(equal_weights(train.shape[1]), index=train.columns, name="Equal_Weight")


def walk_forward_backtest(
    returns: pd.DataFrame,
    benchmark: str | None = None,
    lookback: int = DEFAULT_LOOKBACK,
    cost_bps: float = DEFAULT_COST_BPS,
    rf: float = DEFAULT_RF,
) -> BacktestResult:
    """Run the monthly walk-forward backtest for every supported strategy.

    Weights for each month are estimated on the trailing ``lookback`` observations and
    held until the next rebalance; ``cost_bps`` of one-way cost is charged on turnover
    (including the initial allocation).
    """
    if returns.empty:
        raise ValueError("walk_forward_backtest() received an empty returns frame")
    if len(returns) <= lookback:
        raise ValueError(
            f"Need more than {lookback} observations for a {lookback}-day estimation window (got {len(returns)})"
        )

    dates = returns.index
    schedule = rebalance_schedule(dates)
    valid = [
        (date, pos)
        for date, pos in zip(schedule, dates.get_indexer(schedule), strict=False)
        if pos + 1 < len(dates) and pos >= lookback - 1
    ]
    if not valid:
        raise ValueError("No valid rebalance dates: the sample is too short for the estimation window")

    strategy_series = {name: pd.Series(np.nan, index=dates, dtype=float) for name in STRATEGIES}
    weight_history: dict[str, pd.DataFrame] = {name: pd.DataFrame() for name in STRATEGIES}
    turnover_rows: dict[str, list[dict[str, object]]] = {name: [] for name in STRATEGIES}
    previous: dict[str, np.ndarray | None] = dict.fromkeys(STRATEGIES)

    for order, (date, pos) in enumerate(valid):
        train = returns.iloc[pos - lookback + 1 : pos + 1]
        next_pos = valid[order + 1][1] if order + 1 < len(valid) else len(dates) - 1
        hold = returns.iloc[pos + 1 : next_pos + 1]
        if hold.empty:
            continue
        for name in STRATEGIES:
            weights = _estimate_weights(train, name, rf)
            w = weights.to_numpy(dtype=float)
            series = pd.Series(hold.to_numpy() @ w, index=hold.index)
            if previous[name] is None:
                turnover = float(np.abs(w).sum())  # initial allocation, one-way
            else:
                turnover = float(0.5 * np.abs(w - previous[name]).sum())
            series.iloc[0] -= turnover * cost_bps / 10_000.0
            strategy_series[name].iloc[pos + 1 : next_pos + 1] = series.to_numpy()
            turnover_rows[name].append({"Date": date, "Turnover": turnover})
            weight_history[name] = pd.concat([weight_history[name], weights.to_frame(date).T])
            previous[name] = w

    daily = pd.DataFrame({name: strategy_series[name] for name in STRATEGIES})
    if benchmark and benchmark in returns.columns:
        daily[benchmark] = returns[benchmark]
    daily = daily.iloc[valid[0][1] + 1 :]

    stats: dict[str, dict[str, object]] = {}
    for name in daily.columns:
        series = daily[name].dropna()
        if series.empty:
            continue
        row: dict[str, object] = dict(series_stats(series, rf=rf))
        row["Observations"] = float(len(series))
        row["Start"] = series.index[0]
        row["End"] = series.index[-1]
        if name in turnover_rows:
            turnover = pd.DataFrame(turnover_rows[name])
            years = len(series) / TRADING_DAYS
            row["Avg_Turnover"] = float(turnover["Turnover"].mean())
            row["Annual_Cost"] = float(turnover["Turnover"].sum() * cost_bps / 10_000.0 / years)
        stats[name] = row

    stats_frame = pd.DataFrame(stats).T
    # Mixing timestamps with metrics leaves the frame object-dtyped; keep numeric
    # columns numeric so formatting, round() and downstream checks behave correctly.
    for column in stats_frame.columns:
        if column not in ("Start", "End"):
            stats_frame[column] = pd.to_numeric(stats_frame[column], errors="coerce")

    return BacktestResult(
        daily_returns=daily,
        stats=stats_frame,
        weights_history=weight_history,
        lookback=lookback,
        cost_bps=cost_bps,
    )
