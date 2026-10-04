"""Project-wide configuration: market constants, defaults and artifact paths."""

from __future__ import annotations

from pathlib import Path

TRADING_DAYS = 252
BENCHMARK = "SPY"
DEFAULT_START = "2010-01-01"  # long-horizon analysis window (multi-regime history)
DEFAULT_RF = 0.04  # annual risk-free rate used for Sharpe / Sortino / alpha

DATA_DIR = Path("data")  # committed price snapshot for offline runs
SNAPSHOT_PATH = DATA_DIR / "price_history.csv"

REPORTS_DIR = Path("reports")  # generated outputs are written here
SUMMARY_CSV = "financial_analysis_summary.csv"
SUMMARY_XLSX = "financial_analysis_summary.xlsx"
CORRELATION_CSV = "asset_correlations.csv"
PORTFOLIO_CSV = "portfolio_optimization.csv"
REPORT_HTML = "financial_report.html"
BACKTEST_CSV = "backtest_performance.csv"
BACKTEST_CURVES_CSV = "backtest_equity_curves.csv"
SCENARIO_CSV = "scenario_analysis.csv"
RISK_CONTRIBUTIONS_CSV = "risk_contributions.csv"
VAR_BACKTEST_CSV = "var_backtest.csv"
VOL_FORECAST_CSV = "volatility_forecasts.csv"
