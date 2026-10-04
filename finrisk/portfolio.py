"""Portfolio construction: weights, portfolio stats, mean-variance optimization and Monte Carlo."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import DEFAULT_RF, TRADING_DAYS


def normalize_weights(weights: Iterable[float]) -> np.ndarray:
    w = np.clip(np.asarray(list(weights), dtype=float), 0, None)
    total = w.sum()
    if total <= 0:
        raise ValueError("Portfolio weights must sum to a positive value")
    return w / total


def equal_weights(n: int) -> np.ndarray:
    w = np.full(n, 1.0 / n)
    w[-1] = 1.0 - w[:-1].sum()  # exact sum to satisfy SLSQP equality constraints
    return w


def portfolio_returns(returns: pd.DataFrame, weights: Iterable[float]) -> pd.Series:
    w = normalize_weights(weights)
    series = returns.to_numpy() @ w
    return pd.Series(series, index=returns.index, name="Portfolio")


def series_stats(series: pd.Series, rf: float = DEFAULT_RF) -> dict[str, float]:
    """Risk/return statistics for a daily return series (portfolio, strategy or benchmark)."""
    series = series.dropna()
    if series.empty:
        raise ValueError("series_stats() received an empty series")
    mean_ann = series.mean() * TRADING_DAYS
    vol = series.std() * math.sqrt(TRADING_DAYS)
    wealth = (1 + series).cumprod()
    cagr = wealth.iloc[-1] ** (TRADING_DAYS / len(series)) - 1
    mdd = float((wealth / wealth.cummax() - 1).min())

    # Downside deviation of the excess return below the risk-free rate (see metrics.py).
    excess = series - rf / TRADING_DAYS
    downside_dev = float(np.sqrt((excess.clip(upper=0) ** 2).mean()) * math.sqrt(TRADING_DAYS))
    var95 = float(-series.quantile(0.05))
    cvar95 = float(-series[series <= series.quantile(0.05)].mean())

    # Numerically-zero denominators (constant series) yield undefined ratios, not huge ones.
    return {
        "Annualized_Return": float(cagr),
        "Annualized_Volatility": float(vol),
        "Sharpe_Ratio": float((mean_ann - rf) / vol) if vol > 1e-12 else float("nan"),
        "Sortino_Ratio": float((mean_ann - rf) / downside_dev) if downside_dev > 1e-12 else float("nan"),
        "Max_Drawdown": mdd,
        "Calmar_Ratio": float(cagr / abs(mdd)) if abs(mdd) > 1e-12 else float("nan"),
        "VaR_95": var95,
        "CVaR_95": cvar95,
        "Total_Return": float(wealth.iloc[-1] - 1),
    }


def portfolio_stats(
    returns: pd.DataFrame,
    weights: Iterable[float],
    rf: float = DEFAULT_RF,
) -> tuple[dict[str, float], pd.Series]:
    """Risk/return statistics and daily return series for a weighted portfolio."""
    series = portfolio_returns(returns, weights)
    return series_stats(series, rf=rf), series


def _annualized_inputs(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu = (returns.mean() * TRADING_DAYS).to_numpy()
    cov = (returns.cov() * TRADING_DAYS).to_numpy()
    return mu, cov


def _solve(objective, n_assets: int, extra_constraints: list | None = None, x0: np.ndarray | None = None):
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if extra_constraints:
        constraints.extend(extra_constraints)
    return minimize(
        objective,
        x0 if x0 is not None else equal_weights(n_assets),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )


def _jittered_start(n_assets: int, base: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """Deterministic feasible restart point: mostly the base weights plus a Dirichlet mix."""
    rng = np.random.default_rng(seed)
    reference = equal_weights(n_assets) if base is None else np.asarray(base, dtype=float)
    return normalize_weights(0.7 * reference + 0.3 * rng.dirichlet(np.ones(n_assets)))


def _clipped_covariance(covariance: np.ndarray, floor_ratio: float) -> np.ndarray:
    """Raise the covariance's near-zero eigenvalues to ``floor_ratio`` x the largest one.

    Universes with near-duplicate assets have numerically null eigen-directions, which is
    what makes SLSQP line searches fail intermittently; clipping only those directions
    leaves the matrix exact wherever it is well determined.
    """
    eigenvalues, vectors = np.linalg.eigh(covariance)
    floor = floor_ratio * float(eigenvalues[-1])
    if float(eigenvalues[0]) >= floor:
        return covariance
    clipped = vectors @ np.diag(np.clip(eigenvalues, floor, None)) @ vectors.T
    return 0.5 * (clipped + clipped.T)


def _solve_robust(
    objective_factory,
    covariance: np.ndarray,
    n_assets: int,
    extra_constraints: list | None = None,
    x0: np.ndarray | None = None,
    label: str = "Optimization",
):
    """SLSQP with a deterministic restart ladder so no rebalance can fail by chance.

    Tries (1) the exact problem from the default start, (2) the exact problem from a
    perturbed start, then (3) and (4) an eigenvalue-clipped covariance, which stabilizes
    universes containing near-duplicate assets.
    """
    attempts = [
        (covariance, x0),
        (covariance, _jittered_start(n_assets, base=x0, seed=0)),
        (_clipped_covariance(covariance, 1e-6), x0),
        (_clipped_covariance(covariance, 1e-6), _jittered_start(n_assets, base=x0, seed=1)),
    ]
    message = "unknown"
    for attempt_covariance, start in attempts:
        result = _solve(objective_factory(attempt_covariance), n_assets, extra_constraints=extra_constraints, x0=start)
        if result.success:
            return result
        message = result.message
    raise RuntimeError(f"{label} failed after {len(attempts)} attempts ({message})")


def optimize_max_sharpe(returns: pd.DataFrame, rf: float = DEFAULT_RF) -> pd.Series:
    """Long-only tangency portfolio: maximum Sharpe ratio."""
    mu, cov = _annualized_inputs(returns)
    n = len(mu)

    def objective_factory(covariance: np.ndarray):
        def neg_sharpe(w: np.ndarray) -> float:
            vol = math.sqrt(w @ covariance @ w)
            return -(w @ mu - rf) / vol if vol > 0 else 0.0

        return neg_sharpe

    result = _solve_robust(objective_factory, cov, n, label="Max-Sharpe optimization")
    return pd.Series(normalize_weights(result.x), index=returns.columns, name="Max_Sharpe")


def optimize_min_variance(returns: pd.DataFrame) -> pd.Series:
    """Global minimum-variance long-only portfolio."""
    _, cov = _annualized_inputs(returns)
    n = cov.shape[0]

    result = _solve_robust(
        lambda covariance: (lambda w: w @ covariance @ w), cov, n, label="Minimum-variance optimization"
    )
    return pd.Series(normalize_weights(result.x), index=returns.columns, name="Min_Variance")


def efficient_frontier(returns: pd.DataFrame, points: int = 40) -> pd.DataFrame:
    """Markowitz efficient frontier: minimum volatility for a grid of target returns."""
    mu, cov = _annualized_inputs(returns)
    n = len(mu)
    if n < 2:
        return pd.DataFrame(columns=["Annualized_Return", "Annualized_Volatility"])

    min_var_weights = optimize_min_variance(returns).to_numpy()
    targets = np.linspace(float(min_var_weights @ mu), float(mu.max()), points)
    rows: list[dict[str, float]] = []
    x0 = min_var_weights
    for target in targets:
        constraints = [{"type": "eq", "fun": lambda w, t=target: w @ mu - t}]
        try:
            result = _solve_robust(
                lambda covariance: (lambda w: w @ covariance @ w),
                cov,
                n,
                extra_constraints=constraints,
                x0=x0,
                label="Efficient frontier point",
            )
        except RuntimeError:  # pragma: no cover - skip a point rather than break the curve
            continue
        rows.append({"Annualized_Return": float(target), "Annualized_Volatility": float(math.sqrt(result.fun))})
        x0 = result.x
    return pd.DataFrame(rows)


def _random_weight_matrix(rng: np.random.Generator, n_assets: int, n_portfolios: int) -> np.ndarray:
    """Dirichlet samples over the long-only weight simplex; handles the single-asset edge case."""
    if n_assets == 1:
        return np.ones((n_portfolios, 1))
    return rng.dirichlet(np.ones(n_assets), size=n_portfolios)


def monte_carlo_portfolios(
    returns: pd.DataFrame,
    n_portfolios: int = 10_000,
    rf: float = DEFAULT_RF,
    seed: int = 42,
) -> pd.DataFrame:
    """Random long-only portfolios sampled from a Dirichlet distribution."""
    rng = np.random.default_rng(seed)
    weight_matrix = _random_weight_matrix(rng, returns.shape[1], n_portfolios)

    portfolio_daily = returns.to_numpy() @ weight_matrix.T
    ann_return = portfolio_daily.mean(axis=0) * TRADING_DAYS
    # ddof=1 (sample) to match every other volatility in the engine, which uses
    # pandas' .std(); numpy defaults to the population statistic (ddof=0).
    ann_vol = portfolio_daily.std(axis=0, ddof=1) * math.sqrt(TRADING_DAYS)
    return pd.DataFrame(
        {
            "Annualized_Return": ann_return,
            "Annualized_Volatility": ann_vol,
            "Sharpe_Ratio": (ann_return - rf) / ann_vol,
        }
    )
