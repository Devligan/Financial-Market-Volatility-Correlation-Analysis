"""Risk & return statistics: annualization, drawdowns, calendar years and benchmark-relative metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import BENCHMARK, DEFAULT_RF, TRADING_DAYS
from .universe import asset_class


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns from a price frame."""
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return returns.dropna(how="all")


def annualized_return(returns: pd.DataFrame) -> pd.Series:
    """Geometric (CAGR) annualized return."""
    n = returns.count()
    return (1 + returns).prod() ** (TRADING_DAYS / n) - 1


def annualized_volatility(returns: pd.DataFrame) -> pd.Series:
    return returns.std() * math.sqrt(TRADING_DAYS)


def calendar_year_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Total (compounded) return per calendar year.

    The first and last years of the sample may be partial; label them as such
    when presenting the table.
    """
    yearly = (1 + returns).groupby(returns.index.year).prod() - 1
    yearly.index.name = "Year"
    return yearly


def drawdown_series(returns: pd.DataFrame) -> pd.DataFrame:
    """Drawdown time series (fraction below the running peak) for each column."""
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(returns: pd.DataFrame) -> pd.Series:
    return drawdown_series(returns).min()


def drawdown_episodes(returns: pd.Series, top: int = 5) -> pd.DataFrame:
    """Largest peak-to-trough drawdown episodes with depth and recovery timing.

    Returns one row per episode: ``Peak`` (date of the prior high), ``Trough``,
    ``Recovery`` (NaT while the episode is ongoing), ``Depth``, ``Trough_Days``
    (peak to trough) and ``Recovery_Days`` (trough to recovery), both measured in
    trading days.
    """
    series = returns.dropna()
    wealth = (1 + series).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    values = drawdown.to_numpy()

    episodes: list[tuple[int, int]] = []
    start: int | None = None
    for position, value in enumerate(values):
        if value < 0 and start is None:
            start = position
        elif value >= 0 and start is not None:
            episodes.append((start, position))
            start = None
    if start is not None:
        episodes.append((start, len(values) - 1))

    rows = []
    for start, end in episodes:
        trough = start + int(np.argmin(values[start : end + 1]))
        peak = start - 1 if start > 0 else start  # previous position was the running high
        recovered = drawdown.iloc[end] >= 0
        rows.append(
            {
                "Peak": drawdown.index[peak],
                "Trough": drawdown.index[trough],
                "Recovery": drawdown.index[end] if recovered else pd.NaT,
                "Depth": float(drawdown.iloc[trough]),
                "Trough_Days": int(trough - peak),
                "Recovery_Days": int(end - trough) if recovered else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("Depth").head(top).reset_index(drop=True)


def omega_ratio(returns: pd.Series | pd.DataFrame, rf: float = DEFAULT_RF, threshold: float = 0.0) -> pd.Series:
    """Omega ratio: probability-weighted gain vs loss relative to threshold."""
    r = returns - threshold
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return gains / losses.replace(0, np.nan)


def gain_to_pain_ratio(returns: pd.Series | pd.DataFrame, rf: float = DEFAULT_RF) -> pd.Series:
    """Gain-to-pain ratio: (total return - rf*years) / sum of |drawdowns| in dollars terms (approx)."""
    ann_ret = annualized_return(returns)
    years = len(returns) / TRADING_DAYS
    total_gain = (1 + returns).prod() - 1
    pain = (returns[returns < 0].abs()).sum()
    return (total_gain - rf * years) / pain.replace(0, np.nan)


def upside_downside_capture(returns: pd.DataFrame, benchmark: str) -> tuple[pd.Series, pd.Series]:
    """Up/Down capture ratios."""
    up = returns.gt(0) & returns[benchmark].gt(0)
    down = returns.lt(0) & returns[benchmark].lt(0)
    up_capture = returns[up].mean() / returns[benchmark][up].mean().replace(0, np.nan)
    down_capture = returns[down].mean() / returns[benchmark][down].mean().replace(0, np.nan)
    return up_capture, down_capture


def realized_volatility(returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Realized volatility (rolling/std) annualized."""
    return returns.rolling(window=window).std() * math.sqrt(TRADING_DAYS)


def compute_metrics(
    returns: pd.DataFrame,
    benchmark: str | None = BENCHMARK,
    rf: float = DEFAULT_RF,
) -> pd.DataFrame:
    """Full risk/return metric table, one row per asset.

    Conventions: ``Annualized_Return`` / ``Calmar_Ratio`` use the geometric CAGR;
    ``VaR_*`` / ``CVaR_*`` are expressed as positive daily losses. Benchmark-relative
    columns (beta, alpha, tracking error, information ratio, up/down capture) are only
    added when the benchmark is part of the analyzed universe.
    """
    metrics = pd.DataFrame(index=returns.columns)
    mean_ann = returns.mean() * TRADING_DAYS

    metrics["Mean_Daily_Return"] = returns.mean()
    metrics["Std_Daily_Return"] = returns.std()
    metrics["Annualized_Return"] = annualized_return(returns)
    metrics["Annualized_Volatility"] = annualized_volatility(returns)
    metrics["Realized_Vol_30d"] = returns.rolling(30).std().iloc[-1] * math.sqrt(TRADING_DAYS)
    metrics["Realized_Vol_90d"] = returns.rolling(90).std().iloc[-1] * math.sqrt(TRADING_DAYS)
    metrics["Realized_Vol_252d"] = returns.rolling(252).std().iloc[-1] * math.sqrt(TRADING_DAYS)
    # A constant series leaves a tiny floating-point residue in its standard deviation;
    # mask numerically-zero denominators explicitly so ratios are NaN, never astronomically large.
    vol = metrics["Annualized_Volatility"]
    metrics["Sharpe_Ratio"] = (mean_ann - rf) / vol.mask(vol < 1e-12)

    # Downside deviation of the *excess* return below the risk-free rate: only days
    # that underperform the target contribute; days above it contribute zero.
    excess = returns - rf / TRADING_DAYS
    downside_dev = np.sqrt((excess.clip(upper=0) ** 2).mean()) * math.sqrt(TRADING_DAYS)
    metrics["Sortino_Ratio"] = (mean_ann - rf) / downside_dev.mask(downside_dev < 1e-12)

    metrics["Max_Drawdown"] = max_drawdown(returns)
    drawdown_size = metrics["Max_Drawdown"].abs()
    metrics["Calmar_Ratio"] = metrics["Annualized_Return"] / drawdown_size.mask(drawdown_size < 1e-12)
    metrics["Omega_Ratio"] = omega_ratio(returns, rf=rf)
    metrics["Gain_to_Pain_Ratio"] = gain_to_pain_ratio(returns, rf=rf)

    q05, q01 = returns.quantile(0.05), returns.quantile(0.01)
    metrics["VaR_95"] = -q05
    metrics["CVaR_95"] = -returns[returns <= q05].mean()
    metrics["VaR_99"] = -q01
    metrics["CVaR_99"] = -returns[returns <= q01].mean()

    metrics["Skewness"] = returns.skew()
    metrics["Kurtosis"] = returns.kurtosis()  # excess kurtosis
    metrics["Min_Daily_Return"] = returns.min()
    metrics["Max_Daily_Return"] = returns.max()

    if benchmark and benchmark in returns.columns:
        market = returns[benchmark]
        cov_with_market = returns.apply(lambda col: col.cov(market))
        beta = cov_with_market / market.var()
        correlation = returns.apply(lambda col: col.corr(market))
        metrics["Beta"] = beta
        metrics["Alpha"] = mean_ann - (rf + beta * (mean_ann[benchmark] - rf))
        metrics["Correlation_with_Market"] = correlation
        metrics["R_Squared"] = correlation**2

        active = returns.sub(market, axis=0)
        tracking_error = active.std() * math.sqrt(TRADING_DAYS)
        metrics["Tracking_Error"] = tracking_error
        metrics["Information_Ratio"] = (mean_ann - mean_ann[benchmark]) / tracking_error.replace(0, np.nan)

        # Up/down capture (Morningstar-style): the asset's average monthly return in the
        # months the benchmark rose (fell), divided by the benchmark's own average over
        # those months. Averaging keeps both sides on the same scale — compounding raw
        # daily up days over a decade inflates the ratio into meaninglessness.
        monthly = (1 + returns).groupby(returns.index.to_period("M")).prod() - 1
        monthly_market = monthly[benchmark]
        up_months, down_months = monthly_market > 0, monthly_market < 0
        up_base = float(monthly_market[up_months].mean())
        down_base = float(monthly_market[down_months].mean())
        up_capture = monthly[up_months].mean()
        down_capture = monthly[down_months].mean()
        metrics["Up_Capture"] = up_capture / up_base if abs(up_base) > 1e-12 else np.nan
        metrics["Down_Capture"] = down_capture / down_base if abs(down_base) > 1e-12 else np.nan

    metrics["Data_Points"] = returns.count().astype(int)
    metrics.insert(0, "Asset_Class", [asset_class(t) for t in returns.columns])
    return metrics


def class_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average risk/return profile per asset class."""
    numeric = metrics.select_dtypes(include=[np.number])
    grouped = numeric.groupby(metrics["Asset_Class"]).mean()
    grouped.insert(0, "Assets", metrics.groupby("Asset_Class").size())
    return grouped


def average_correlation(correlation: pd.DataFrame) -> float:
    """Mean pairwise correlation across the upper triangle (NaN if < 2 assets)."""
    values = correlation.to_numpy()
    if values.shape[0] < 2:
        return float("nan")
    off_diagonal = values[np.triu_indices_from(values, k=1)]
    return float(np.nanmean(off_diagonal))


def correlation_extremes(correlation: pd.DataFrame) -> tuple[tuple[str, str, float], tuple[str, str, float]]:
    """Return ((a, b, value) for highest and lowest pairwise correlation)."""
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    stacked = upper.stack()
    (hi_a, hi_b), hi_val = stacked.idxmax(), stacked.max()
    (lo_a, lo_b), lo_val = stacked.idxmin(), stacked.min()
    return (hi_a, hi_b, float(hi_val)), (lo_a, lo_b, float(lo_val))


def rolling_volatility(returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    return returns.rolling(window=window).std() * math.sqrt(TRADING_DAYS)


def rolling_correlation(returns: pd.DataFrame, asset_a: str, asset_b: str, window: int = 90) -> pd.Series:
    return returns[asset_a].rolling(window=window).corr(returns[asset_b]).dropna()


def rolling_sharpe(returns: pd.Series | pd.DataFrame, window: int = 252, rf: float = DEFAULT_RF) -> pd.Series | pd.DataFrame:
    """Rolling annualized Sharpe ratio."""
    mean = returns.rolling(window).mean() * TRADING_DAYS
    vol = returns.rolling(window).std() * math.sqrt(TRADING_DAYS)
    return (mean - rf) / vol.replace(0, np.nan)


def rolling_beta(returns: pd.DataFrame, benchmark: str, window: int = 252) -> pd.Series:
    """Rolling beta vs benchmark."""
    market = returns[benchmark]
    def beta_row(x):
        cov = x.cov(market.loc[x.index])
        var_m = market.loc[x.index].var()
        return cov / var_m if var_m > 1e-12 else np.nan
    return returns.rolling(window).apply(beta_row, raw=False)


def correlation_eigenvalues(correlation: pd.DataFrame) -> pd.DataFrame:
    """Eigenvalue decomposition of correlation matrix (for PCA analysis)."""
    try:
        vals, vecs = np.linalg.eigh(correlation.to_numpy())
        vals = np.flip(vals)
        total = vals.sum()
        explained = vals / total if total > 0 else vals
        cum_explained = np.cumsum(explained)
        return pd.DataFrame({
            "Eigenvalue": vals,
            "Explained_Variance": explained,
            "Cumulative_Explained": cum_explained,
        })
    except Exception:
        return pd.DataFrame(columns=["Eigenvalue", "Explained_Variance", "Cumulative_Explained"])


def condition_number(correlation: pd.DataFrame) -> float:
    """Condition number of correlation matrix (ill-conditioning indicator)."""
    try:
        vals = np.linalg.eigvalsh(correlation.to_numpy())
        vals = vals[vals > 1e-12]
        if len(vals) < 2:
            return float("inf")
        return float(vals[-1] / vals[0]) if vals[0] > 0 else float("inf")
    except Exception:
        return float("nan")
