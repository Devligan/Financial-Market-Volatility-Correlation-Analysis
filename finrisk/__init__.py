"""finrisk: financial volatility, correlation and portfolio-risk analytics.

The package mirrors the analysis pipeline:

    config  universe  data  metrics  portfolio  pipeline  reporting

Everything public is re-exported here, so the whole system is available under a
single import:

    import finrisk as an

    bundle = an.compute_all()
    print(an.executive_summary(bundle))
"""

from __future__ import annotations

from . import presentation
from .attribution import (
    concentration,
    diversification_ratio,
    portfolio_risk_profile,
    risk_contribution_table,
    risk_contributions,
    risk_summary,
)
from .backtest import (
    DEFAULT_COST_BPS,
    DEFAULT_LOOKBACK,
    STRATEGIES,
    BacktestResult,
    rebalance_schedule,
    walk_forward_backtest,
)
from .config import (
    BENCHMARK,
    CORRELATION_CSV,
    DATA_DIR,
    DEFAULT_RF,
    DEFAULT_START,
    PORTFOLIO_CSV,
    REPORT_HTML,
    REPORTS_DIR,
    SNAPSHOT_PATH,
    SUMMARY_CSV,
    SUMMARY_XLSX,
    TRADING_DAYS,
)
from .data import fetch_prices, load_prices, save_prices
from .metrics import (
    annualized_return,
    annualized_volatility,
    average_correlation,
    calendar_year_returns,
    class_summary,
    compute_metrics,
    compute_returns,
    correlation_extremes,
    drawdown_episodes,
    drawdown_series,
    gain_to_pain_ratio,
    max_drawdown,
    omega_ratio,
    realized_volatility,
    rolling_beta,
    rolling_correlation,
    rolling_sharpe,
    rolling_volatility,
    correlation_eigenvalues,
    condition_number,
)
from .pipeline import AnalysisBundle, ExtendedAnalysis, compute_all, compute_extended
from .portfolio import (
    efficient_frontier,
    equal_weights,
    monte_carlo_portfolios,
    normalize_weights,
    optimize_max_sharpe,
    optimize_min_variance,
    portfolio_returns,
    portfolio_stats,
)
from .reporting import executive_summary, report_html, save_outputs, top_weights
from .scenarios import (
    SCENARIOS,
    scenario_asset_table,
    scenario_drawdown_table,
    scenario_return_table,
    scenario_summary,
    window_max_drawdown,
    window_total,
)
from .universe import ASSET_UNIVERSE, asset_class, asset_name, default_tickers, tickers_of_class
from .var_backtest import breach_flags, kupiec_pof, rolling_var, var_backtest_summary
from .volatility import forecast_universe, garch_fit, garch_forecast

__all__ = [
    "ASSET_UNIVERSE",
    "BENCHMARK",
    "CORRELATION_CSV",
    "DATA_DIR",
    "DEFAULT_COST_BPS",
    "DEFAULT_LOOKBACK",
    "DEFAULT_RF",
    "DEFAULT_START",
    "PORTFOLIO_CSV",
    "REPORTS_DIR",
    "REPORT_HTML",
    "SCENARIOS",
    "SNAPSHOT_PATH",
    "STRATEGIES",
    "SUMMARY_CSV",
    "SUMMARY_XLSX",
    "TRADING_DAYS",
    "AnalysisBundle",
    "BacktestResult",
    "ExtendedAnalysis",
    "annualized_return",
    "annualized_volatility",
    "asset_class",
    "asset_name",
    "average_correlation",
    "breach_flags",
    "calendar_year_returns",
    "class_summary",
    "compute_all",
    "compute_extended",
    "compute_metrics",
    "compute_returns",
    "concentration",
    "correlation_extremes",
    "default_tickers",
    "diversification_ratio",
    "drawdown_episodes",
    "drawdown_series",
    "efficient_frontier",
    "equal_weights",
    "executive_summary",
    "fetch_prices",
    "forecast_universe",
    "gain_to_pain_ratio",
    "garch_fit",
    "garch_forecast",
    "kupiec_pof",
    "load_prices",
    "max_drawdown",
    "monte_carlo_portfolios",
    "normalize_weights",
    "omega_ratio",
    "optimize_max_sharpe",
    "optimize_min_variance",
    "portfolio_returns",
    "portfolio_risk_profile",
    "portfolio_stats",
    "presentation",
    "realized_volatility",
    "rebalance_schedule",
    "report_html",
    "risk_contribution_table",
    "risk_contributions",
    "risk_summary",
    "rolling_beta",
    "rolling_correlation",
    "rolling_sharpe",
    "rolling_var",
    "rolling_volatility",
    "correlation_eigenvalues",
    "condition_number",
    "save_outputs",
    "save_prices",
    "scenario_asset_table",
    "scenario_drawdown_table",
    "scenario_return_table",
    "scenario_summary",
    "tickers_of_class",
    "top_weights",
    "var_backtest_summary",
    "walk_forward_backtest",
    "window_max_drawdown",
    "window_total",
]
