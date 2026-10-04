"""Risk decomposition: where a portfolio's risk actually comes from.

Volatility contributions follow the Euler decomposition: the weighted marginal
contributions (weight times the asset's covariance with the portfolio, divided by
portfolio volatility) sum exactly to the portfolio volatility, so each asset's
share of total risk can be reported directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_RF, TRADING_DAYS
from .portfolio import normalize_weights, portfolio_stats


def risk_contributions(returns: pd.DataFrame, weights) -> pd.DataFrame:
    """Per-asset weight, volatility contribution and share of total portfolio risk."""
    w = normalize_weights(weights)
    covariance = returns.cov().to_numpy() * TRADING_DAYS
    portfolio_vol = float(np.sqrt(w @ covariance @ w))
    if portfolio_vol > 0:
        marginal = covariance @ w / portfolio_vol
        contribution = w * marginal
        share = contribution / portfolio_vol
    else:  # pragma: no cover - a zero-volatility portfolio is degenerate
        contribution = np.zeros_like(w)
        share = np.zeros_like(w)
    frame = pd.DataFrame(
        {
            "Weight": w,
            "Volatility": np.sqrt(np.diag(covariance)),
            "Risk_Contribution": contribution,
            "Risk_Percent": share,
        },
        index=returns.columns,
    )
    return frame


def diversification_ratio(returns: pd.DataFrame, weights) -> float:
    """Weighted-average asset volatility divided by realized portfolio volatility (>= 1)."""
    w = normalize_weights(weights)
    asset_vols = returns.std().to_numpy() * np.sqrt(TRADING_DAYS)
    covariance = returns.cov().to_numpy() * TRADING_DAYS
    portfolio_vol = float(np.sqrt(w @ covariance @ w))
    if portfolio_vol <= 0:  # pragma: no cover - degenerate
        return float("nan")
    return float((w @ asset_vols) / portfolio_vol)


def concentration(weights) -> float:
    """Herfindahl-Hirschman concentration of the weights (1 = single asset)."""
    w = normalize_weights(weights)
    return float((w**2).sum())


def portfolio_risk_profile(returns: pd.DataFrame, weights, rf: float = DEFAULT_RF) -> dict[str, object]:
    """Headline risk-decomposition metrics for one portfolio."""
    stats, _ = portfolio_stats(returns, weights, rf=rf)
    frame = risk_contributions(returns, weights)
    return {
        "Annualized_Volatility": stats["Annualized_Volatility"],
        "Diversification_Ratio": diversification_ratio(returns, weights),
        "Concentration_HHI": concentration(weights),
        "Top_Risk_Asset": str(frame["Risk_Percent"].idxmax()),
        "Top_Risk_Percent": float(frame["Risk_Percent"].max()),
    }


def risk_summary(returns: pd.DataFrame, weights_map: dict[str, pd.Series], rf: float = DEFAULT_RF) -> pd.DataFrame:
    """Risk-decomposition profile per portfolio, one row per portfolio."""
    return pd.DataFrame(
        {name: portfolio_risk_profile(returns, weights, rf=rf) for name, weights in weights_map.items()}
    ).T


def risk_contribution_table(returns: pd.DataFrame, weights_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Tidy long table of risk shares: Portfolio, Asset, Weight, Risk_Percent, Risk_Contribution."""
    rows = []
    for name, weights in weights_map.items():
        frame = risk_contributions(returns, weights)
        for asset, row in frame.iterrows():
            rows.append(
                {
                    "Portfolio": name,
                    "Asset": asset,
                    "Weight": float(row["Weight"]),
                    "Risk_Contribution": float(row["Risk_Contribution"]),
                    "Risk_Percent": float(row["Risk_Percent"]),
                }
            )
    return pd.DataFrame(rows)
