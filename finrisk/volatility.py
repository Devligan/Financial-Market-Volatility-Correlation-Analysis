"""GARCH(1,1) volatility models: conditional volatility and short-horizon forecasts.

The models are fitted per asset by maximum likelihood (via the ``arch`` package) and
summarized as an annualized forecast compared with recent realized volatility, which
is what a risk desk uses to flag whether an asset's risk is currently running hot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS
from .universe import asset_name

ANNUALIZATION = float(np.sqrt(TRADING_DAYS))


def _fit(series: pd.Series):
    from arch import arch_model  # lazy import: only needed for volatility functions

    clean = series.dropna().astype(float)
    if len(clean) < 250:
        raise ValueError(f"Need at least 250 observations to fit GARCH(1,1) (got {len(clean)})")
    scaled = clean * 100.0  # percent returns are numerically better for the optimizer
    model = arch_model(scaled, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    return model.fit(disp="off", show_warning=False)


def garch_fit(series: pd.Series) -> dict[str, float]:
    """Maximum-likelihood GARCH(1,1) parameters and persistence (alpha + beta)."""
    result = _fit(series)
    params = result.params
    alpha = float(params.get("alpha[1]", 0.0))
    beta = float(params.get("beta[1]", 0.0))
    return {
        "Omega": float(params["omega"]),
        "Alpha": alpha,
        "Beta": beta,
        "Persistence": alpha + beta,
        "LogLikelihood": float(result.loglikelihood),
    }


def garch_forecast(series: pd.Series, horizon: int = 21) -> dict[str, float]:
    """Annualized conditional volatility and average forecast over ``horizon`` days."""
    result = _fit(series)
    params = result.params
    conditional_daily = float(result.conditional_volatility.iloc[-1]) / 100.0
    variance_path = result.forecast(horizon=horizon, reindex=False).variance
    expected_daily_variance = float(variance_path.iloc[-1].mean()) / (100.0**2)
    return {
        "Conditional_Vol": float(conditional_daily * ANNUALIZATION),
        "Forecast_Vol": float(np.sqrt(expected_daily_variance) * ANNUALIZATION),
        "Persistence": float(params.get("alpha[1]", 0.0)) + float(params.get("beta[1]", 0.0)),
    }


def forecast_universe(returns: pd.DataFrame, horizon: int = 21, realized_window: int = 60) -> pd.DataFrame:
    """GARCH(1,1) forecasts for every asset, compared with recent realized volatility."""
    realized = returns.rolling(realized_window).std().iloc[-1] * ANNUALIZATION
    rows: dict[str, dict[str, float]] = {}
    for ticker in returns.columns:
        try:
            rows[ticker] = garch_forecast(returns[ticker], horizon=horizon)
        except Exception:  # noqa: BLE001 - a single non-converging series must not break the sweep
            rows[ticker] = {"Conditional_Vol": np.nan, "Forecast_Vol": np.nan, "Persistence": np.nan}
    frame = pd.DataFrame(rows).T
    frame["Realized_Vol_60d"] = realized
    frame["Forecast_vs_Realized"] = frame["Forecast_Vol"] / frame["Realized_Vol_60d"]
    ratio = frame["Forecast_vs_Realized"]
    frame["Regime"] = np.select([ratio > 1.15, ratio < 0.85], ["Elevated", "Calm"], default="Normal")
    frame.loc[ratio.isna(), "Regime"] = "n/a"
    frame["Name"] = [asset_name(ticker) for ticker in frame.index]
    return frame.sort_values("Forecast_Vol", ascending=False)
