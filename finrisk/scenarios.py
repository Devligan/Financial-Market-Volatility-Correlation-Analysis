"""Stress scenarios: fixed crisis windows with portfolio and asset outcomes.

Windows are historical peak-to-trough or event periods, so the tables answer the
question an investor actually asks: *what would this portfolio have done during
COVID, the 2022 rate shock, the Q4 2018 selloff, the oil crash or the 2023
banking stress?*
"""

from __future__ import annotations

import pandas as pd

from .portfolio import portfolio_returns

SCENARIOS: dict[str, tuple[str, str]] = {
    "COVID crash (2020)": ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
    "Q4 2018 selloff": ("2018-10-01", "2018-12-24"),
    "Oil crash (2014-16)": ("2014-06-20", "2016-02-11"),
    "2023 banking stress": ("2023-02-01", "2023-03-31"),
}


def _window(series: pd.Series, start: str, end: str) -> pd.Series:
    return series.loc[pd.Timestamp(start) : pd.Timestamp(end)]


def window_total(series: pd.Series, start: str, end: str) -> float:
    """Compounded total return inside a window (NaN when the window has no data)."""
    window = _window(series, start, end)
    if window.empty:
        return float("nan")
    return float((1 + window).prod() - 1)


def window_max_drawdown(series: pd.Series, start: str, end: str) -> float:
    """Maximum peak-to-trough drawdown inside a window (NaN when the window has no data)."""
    window = _window(series, start, end)
    if window.empty:
        return float("nan")
    wealth = (1 + window).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def _portfolio_series(returns: pd.DataFrame, weights_map: dict[str, pd.Series]) -> dict[str, pd.Series]:
    return {name: portfolio_returns(returns, weights) for name, weights in weights_map.items()}


def scenario_return_table(
    returns: pd.DataFrame,
    weights_map: dict[str, pd.Series],
    scenarios: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Total return per scenario (rows) and portfolio (columns)."""
    scenarios = SCENARIOS if scenarios is None else scenarios
    series_map = _portfolio_series(returns, weights_map)
    data = {
        name: {portfolio: window_total(series, start, end) for portfolio, series in series_map.items()}
        for name, (start, end) in scenarios.items()
    }
    return pd.DataFrame(data).T


def scenario_drawdown_table(
    returns: pd.DataFrame,
    weights_map: dict[str, pd.Series],
    scenarios: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Maximum drawdown per scenario (rows) and portfolio (columns)."""
    scenarios = SCENARIOS if scenarios is None else scenarios
    series_map = _portfolio_series(returns, weights_map)
    data = {
        name: {portfolio: window_max_drawdown(series, start, end) for portfolio, series in series_map.items()}
        for name, (start, end) in scenarios.items()
    }
    return pd.DataFrame(data).T


def scenario_summary(
    returns: pd.DataFrame,
    weights_map: dict[str, pd.Series],
    scenarios: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Tidy long table: Scenario, Portfolio, Total_Return, Max_Drawdown."""
    scenarios = SCENARIOS if scenarios is None else scenarios
    series_map = _portfolio_series(returns, weights_map)
    rows = [
        {
            "Scenario": scenario,
            "Portfolio": portfolio,
            "Total_Return": window_total(series, start, end),
            "Max_Drawdown": window_max_drawdown(series, start, end),
        }
        for scenario, (start, end) in scenarios.items()
        for portfolio, series in series_map.items()
    ]
    return pd.DataFrame(rows, columns=["Scenario", "Portfolio", "Total_Return", "Max_Drawdown"])


def scenario_asset_table(returns: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Per-asset return and drawdown for one scenario, sorted from worst to best."""
    start, end = SCENARIOS[scenario]
    rows = {
        ticker: {
            "Total_Return": window_total(returns[ticker], start, end),
            "Max_Drawdown": window_max_drawdown(returns[ticker], start, end),
        }
        for ticker in returns.columns
    }
    return pd.DataFrame(rows).T.sort_values("Total_Return")
