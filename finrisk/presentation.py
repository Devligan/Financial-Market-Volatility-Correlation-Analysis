"""Shared display layer for the dashboard, notebook and console study.

Presentation only — the analytics live in the ``finrisk`` engine modules; keeping
the two separate stops display concerns from leaking into the engine.
"""

from __future__ import annotations

import pandas as pd

CLASS_COLORS = {
    "STOCK": "#4C78A8",
    "ETF": "#9C6ADE",
    "BOND": "#54A24B",
    "COMMODITY": "#F58518",
    "FX": "#72B7B2",
    "OTHER": "#8C8C8C",
}

# Series palette shared by every presenter (dashboard, notebook, console study).
ACCENT = "#4C78A8"  # primary series / equal-weight portfolio
NEGATIVE = "#d62728"  # drawdown and loss series
CUSTOM = "#9C6ADE"  # user-defined portfolio series

PCT_COLS = {
    "Mean_Daily_Return",
    "Std_Daily_Return",
    "Annualized_Return",
    "Annualized_Volatility",
    "Max_Drawdown",
    "VaR_95",
    "CVaR_95",
    "VaR_99",
    "CVaR_99",
    "Alpha",
    "Tracking_Error",
    "Min_Daily_Return",
    "Max_Daily_Return",
    "Total_Return",
    "Avg_Turnover",
    "Annual_Cost",
}

RATIO_COLS = {
    "Sharpe_Ratio",
    "Sortino_Ratio",
    "Calmar_Ratio",
    "Beta",
    "Correlation_with_Market",
    "R_Squared",
    "Information_Ratio",
    "Up_Capture",
    "Down_Capture",
}

PORTFOLIO_METRICS = [
    "Annualized_Return",
    "Annualized_Volatility",
    "Sharpe_Ratio",
    "Sortino_Ratio",
    "Max_Drawdown",
    "Calmar_Ratio",
    "VaR_95",
    "CVaR_95",
    "Total_Return",
]


def metric_style(df: pd.DataFrame):
    """Per-column number formatting for metric tables (percentages, ratios, counts)."""
    formats = {}
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        if column in PCT_COLS:
            formats[column] = "{:.2%}"
        elif column in RATIO_COLS:
            formats[column] = "{:.3f}"
        elif column in ("Assets", "Data_Points"):
            formats[column] = "{:,.0f}"
        else:
            formats[column] = "{:,.4f}"
    return df.style.format(formats, na_rep="—")
