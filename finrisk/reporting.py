"""Reporting: executive summary bullets, standalone HTML report and artifact exports."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .config import (
    BACKTEST_CSV,
    BACKTEST_CURVES_CSV,
    CORRELATION_CSV,
    PORTFOLIO_CSV,
    REPORT_HTML,
    REPORTS_DIR,
    RISK_CONTRIBUTIONS_CSV,
    SCENARIO_CSV,
    SUMMARY_CSV,
    SUMMARY_XLSX,
    VAR_BACKTEST_CSV,
    VOL_FORECAST_CSV,
)
from .metrics import average_correlation, calendar_year_returns, class_summary, correlation_extremes
from .pipeline import AnalysisBundle, ExtendedAnalysis
from .portfolio import portfolio_stats
from .universe import asset_class


def top_weights(weights: pd.Series, n: int = 5) -> str:
    notable = weights[weights > 0.01].sort_values(ascending=False).head(n)
    return ", ".join(f"{ticker} {value:.0%}" for ticker, value in notable.items())


def executive_summary(bundle: AnalysisBundle, extended: ExtendedAnalysis | None = None) -> list[str]:
    """Auto-generated, presentation-ready insights for the report and dashboard."""
    metrics = bundle.metrics
    bullets: list[str] = []

    class_labels = {
        "STOCK": "stocks",
        "ETF": "ETFs",
        "BOND": "bonds",
        "COMMODITY": "commodities",
        "FX": "currencies",
        "OTHER": "other assets",
    }
    class_counts = metrics["Asset_Class"].value_counts()
    class_text = ", ".join(f"{count} {class_labels.get(klass, klass.lower())}" for klass, count in class_counts.items())
    bullets.append(
        f"Analyzed **{len(metrics)} assets** ({class_text}) over {len(bundle.returns)} trading days "
        f"({bundle.prices.index[0].date()}  {bundle.prices.index[-1].date()}), "
        f"using a {bundle.risk_free:.1%} risk-free rate."
    )

    most_volatile = metrics["Annualized_Volatility"].idxmax()
    least_volatile = metrics["Annualized_Volatility"].idxmin()
    bullets.append(
        f"**{most_volatile}** is the most volatile asset ({metrics.loc[most_volatile, 'Annualized_Volatility']:.1%} "
        f"annualized) while **{least_volatile}** is the least volatile "
        f"({metrics.loc[least_volatile, 'Annualized_Volatility']:.1%})."
    )

    best_sharpe = metrics["Sharpe_Ratio"].idxmax()
    best_sortino = metrics["Sortino_Ratio"].idxmax()
    bullets.append(
        f"Best risk-adjusted performance: **{best_sharpe}** (Sharpe {metrics.loc[best_sharpe, 'Sharpe_Ratio']:.2f}), "
        f"best downside-adjusted performance: **{best_sortino}** "
        f"(Sortino {metrics.loc[best_sortino, 'Sortino_Ratio']:.2f})."
    )

    worst_dd = metrics["Max_Drawdown"].idxmin()
    drawdown_bullet = f"Deepest peak-to-trough drawdown: **{worst_dd}** ({metrics.loc[worst_dd, 'Max_Drawdown']:.1%})"
    if "Alpha" in metrics.columns:  # only computed when a benchmark is available
        drawdown_bullet += (
            f"; Jensen's alpha vs {bundle.benchmark}: **{metrics['Alpha'].idxmax()}** "
            f"leads at {metrics['Alpha'].max():.1%}."
        )
    else:
        drawdown_bullet += "."
    bullets.append(drawdown_bullet)

    if "Omega_Ratio" in metrics.columns and metrics["Omega_Ratio"].notna().any():
        best_omega = metrics["Omega_Ratio"].idxmax()
        bullets.append(
            f"**Omega ratio** (gain/loss capture): **{best_omega}** leads with {metrics.loc[best_omega, 'Omega_Ratio']:.2f}, "
            f"indicating superior asymmetry of returns."
        )

    if "Information_Ratio" in metrics.columns and metrics["Information_Ratio"].notna().any():
        best_ir = metrics["Information_Ratio"].idxmax()
        bullets.append(
            f"Benchmark-relative: **{best_ir}** posts the highest information ratio "
            f"({metrics.loc[best_ir, 'Information_Ratio']:.2f}) at "
            f"{metrics.loc[best_ir, 'Tracking_Error']:.1%} tracking error vs {bundle.benchmark}."
        )

    if len(metrics) > 1:
        (hi_a, hi_b, hi_val), (lo_a, lo_b, lo_val) = correlation_extremes(bundle.correlation)
        avg_corr = average_correlation(bundle.correlation)
        bullets.append(
            f"Correlation structure: strongest pair **{hi_a} / {hi_b}** ({hi_val:.2f}), weakest pair "
            f"**{lo_a} / {lo_b}** ({lo_val:.2f}), average pairwise correlation {avg_corr:.2f}."
        )

    classes = class_summary(metrics)
    if {"STOCK", "COMMODITY"} <= set(classes.index):
        bullets.append(
            f"Average volatility: stocks {classes.loc['STOCK', 'Annualized_Volatility']:.1%} vs "
            f"commodities {classes.loc['COMMODITY', 'Annualized_Volatility']:.1%}  commodities and equities "
            f"provide different risk exposures for diversification."
        )

    equal_stats, _ = portfolio_stats(bundle.returns, bundle.equal_weight, rf=bundle.risk_free)
    sharpe_stats, _ = portfolio_stats(bundle.returns, bundle.max_sharpe, rf=bundle.risk_free)
    minvar_stats, _ = portfolio_stats(bundle.returns, bundle.min_variance, rf=bundle.risk_free)
    bullets.append(
        f"**Max-Sharpe portfolio optimization** ({top_weights(bundle.max_sharpe, 4)}): achieves "
        f"{sharpe_stats['Annualized_Return']:.1%} return at {sharpe_stats['Annualized_Volatility']:.1%} volatility "
        f"(Sharpe {sharpe_stats['Sharpe_Ratio']:.2f})  outperforming equal-weight (Sharpe {equal_stats['Sharpe_Ratio']:.2f}) "
        f"and best single asset (Sharpe {metrics['Sharpe_Ratio'].max():.2f}) with improved risk-adjusted efficiency."
    )
    bullets.append(
        f"**Minimum-variance portfolio** ({top_weights(bundle.min_variance, 4)}): delivers "
        f"{minvar_stats['Annualized_Volatility']:.1%} volatility with {minvar_stats['Max_Drawdown']:.1%} max drawdown  "
        f"offering a defensive, low-risk allocation for risk-averse mandates vs {sharpe_stats['Annualized_Volatility']:.1%} "
        f"volatility for the tangency portfolio."
    )
    bullets.append(
        "**Diversification benefit:** Optimized allocations reduce concentration risk while maintaining exposure "
        "to high Sharpe-ratio assets, supporting portfolio construction and risk budgeting decisions."
    )
    if extended is not None:
        bullets.extend(_extended_summary_bullets(bundle, extended))
    return bullets


def _extended_summary_bullets(bundle: AnalysisBundle, extended: ExtendedAnalysis) -> list[str]:
    """Backtest, stress-test and risk-model bullets added when the extended suite is present."""
    bullets: list[str] = []
    stats = extended.backtest.stats
    if {"Max Sharpe", "Equal Weight"} <= set(stats.index):
        benchmark_text = ""
        if bundle.benchmark and bundle.benchmark in stats.index:
            benchmark_text = f" and {bundle.benchmark} {stats.loc[bundle.benchmark, 'Sharpe_Ratio']:.2f}"
        bullets.append(
            f"Walk-forward backtest (monthly rebalance, {extended.backtest.cost_bps:.0f} bp costs): **Max Sharpe** "
            f"Sharpe {stats.loc['Max Sharpe', 'Sharpe_Ratio']:.2f} vs **Equal Weight** "
            f"{stats.loc['Equal Weight', 'Sharpe_Ratio']:.2f}{benchmark_text} out of sample."
        )
    if "Max Sharpe" in extended.scenario_returns.columns:
        column = extended.scenario_returns["Max Sharpe"].dropna()
        if not column.empty:
            worst = column.idxmin()
            drawdown = float(extended.scenario_drawdowns.loc[worst, "Max Sharpe"])
            bullets.append(
                f"Stress tests: the max-Sharpe portfolio's worst historical window was **{worst}** "
                f"({column.loc[worst]:.1%} total return, {drawdown:.1%} max drawdown)."
            )
    var_table = extended.var_backtest
    tested = var_table[var_table["Verdict"] != "n/a"]
    if not tested.empty:
        passed = int((tested["Verdict"] == "Pass").sum())
        bullets.append(
            f"VaR coverage (Kupiec test): {passed} of {len(tested)} strategy/level combinations "
            f"pass at the 5% level (trailing 500-day historical VaR)."
        )
    frame = extended.vol_forecast
    if not frame.empty:
        elevated = frame.index[frame["Regime"] == "Elevated"].tolist()
        if elevated:
            top = elevated[0]
            bullets.append(
                f"GARCH(1,1) risk flags: {len(elevated)} of {len(frame)} assets show elevated volatility "
                f"vs the last 60 days (highest among them: **{top}** at {frame.loc[top, 'Forecast_Vol']:.1%})."
            )
    return bullets


def report_html(bundle: AnalysisBundle, extended: ExtendedAnalysis | None = None) -> str:
    """Render a standalone HTML executive report (with the extended suite when provided)."""
    metrics = bundle.metrics.round(4)
    extended_html = _extended_sections(extended) if extended is not None else ""
    weights_table = pd.DataFrame(
        {
            "Max_Sharpe_Weight": bundle.max_sharpe,
            "Min_Variance_Weight": bundle.min_variance,
            "Equal_Weight": bundle.equal_weight,
        }
    ).round(4)
    weights_table.insert(0, "Asset_Class", [asset_class(t) for t in weights_table.index])
    yearly = (calendar_year_returns(bundle.returns) * 100).round(1)

    insights = "\n".join(
        f"<li>{bullet.replace('**', '')}</li>" for bullet in executive_summary(bundle, extended=extended)
    )
    source_text = "live Yahoo Finance fetch" if bundle.source == "live" else "local price snapshot (offline fallback)"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Financial Volatility & Correlation Report</title>
<style>
  body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 32px auto; max-width: 1100px; color: #22303f; }}
  h1 {{ color: #1f3b57; }} h2 {{ color: #2e5a7d; margin-top: 36px; }}
  h3 {{ color: #2e5a7d; margin-top: 22px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
  th, td {{ border: 1px solid #d7dee5; padding: 6px 9px; text-align: right; }}
  th {{ background: #eef3f8; text-align: center; }}
  td:first-child, th:first-child {{ text-align: left; }}
  ul {{ line-height: 1.6; }}
  .meta {{ color: #65758b; font-size: 13px; }}
</style>
</head>
<body>
  <h1>Financial Volatility &amp; Correlation Analysis</h1>
  <p class="meta">Generated {bundle.fetched_at:%Y-%m-%d %H:%M:%S}  {len(bundle.metrics)} assets 
  {len(bundle.returns)} trading days  data source: {source_text}</p>

  <h2>Executive Summary</h2>
  <ul>
{insights}
  </ul>

  <h2>Risk &amp; Return Metrics</h2>
  {metrics.to_html(border=0)}

  <h2>Calendar-Year Returns (%)</h2>
  <p class="meta">Total return per calendar year (daily compounding); the first and last years may be partial.</p>
  {yearly.to_html(border=0)}

  <h2>Correlation Matrix</h2>
  {bundle.correlation.round(3).to_html(border=0)}

  <h2>Optimized Portfolio Weights</h2>
  {weights_table.to_html(border=0)}
{extended_html}

  <p class="meta">Annualized return = geometric CAGR. Sharpe/Sortino/alpha use a
  {bundle.risk_free:.1%} risk-free rate. Tracking error is the annualized volatility of active returns
  vs the benchmark; information ratio = annualized active return / tracking error; up/down capture
  ratios compare average monthly up-market and down-market returns with the benchmark.
  VaR/CVaR are positive daily loss magnitudes (historical simulation).
  Portfolio optimization is long-only mean-variance (SLSQP), benchmark: {bundle.benchmark or "n/a"}.</p>
</body>
</html>
"""


def save_outputs(
    bundle: AnalysisBundle,
    extended: ExtendedAnalysis | None = None,
    outdir: str | Path = REPORTS_DIR,
) -> list[Path]:
    """Write all project artifacts (CSV / XLSX / HTML) and return their paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    summary_path = outdir / SUMMARY_CSV
    bundle.metrics.round(4).to_csv(summary_path, index=True)
    written.append(summary_path)

    xlsx_path = outdir / SUMMARY_XLSX
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        bundle.metrics.round(4).to_excel(writer, sheet_name="risk_return_metrics")
        bundle.correlation.round(4).to_excel(writer, sheet_name="correlations")
        pd.DataFrame(
            {
                "Max_Sharpe_Weight": bundle.max_sharpe,
                "Min_Variance_Weight": bundle.min_variance,
                "Equal_Weight": bundle.equal_weight,
            }
        ).round(4).to_excel(writer, sheet_name="portfolio_weights")
        calendar_year_returns(bundle.returns).round(4).to_excel(writer, sheet_name="calendar_year_returns")
        if extended is not None:
            extended.backtest.stats.round(4).to_excel(writer, sheet_name="backtest_performance")
            extended.scenario_table.round(4).to_excel(writer, sheet_name="scenarios", index=False)
            extended.risk_contributions.round(4).to_excel(writer, sheet_name="risk_contributions", index=False)
            extended.var_backtest.round(4).to_excel(writer, sheet_name="var_backtest", index=False)
            extended.vol_forecast.round(4).to_excel(writer, sheet_name="volatility_forecasts")
    written.append(xlsx_path)

    correlation_path = outdir / CORRELATION_CSV
    bundle.correlation.round(6).to_csv(correlation_path, index=True)
    written.append(correlation_path)

    portfolio_path = outdir / PORTFOLIO_CSV
    portfolio_table = pd.DataFrame(
        {
            "Max_Sharpe_Weight": bundle.max_sharpe,
            "Min_Variance_Weight": bundle.min_variance,
            "Equal_Weight": bundle.equal_weight,
        }
    )
    portfolio_table.insert(0, "Asset_Class", [asset_class(t) for t in portfolio_table.index])
    portfolio_table.round(4).to_csv(portfolio_path, index=True)
    written.append(portfolio_path)

    report_path = outdir / REPORT_HTML
    report_path.write_text(report_html(bundle, extended), encoding="utf-8")
    written.append(report_path)

    if extended is not None:
        for filename, frame in _extended_frames(extended).items():
            path = outdir / filename
            frame.round(6).to_csv(path)
            written.append(path)

    return written


def _extended_frames(extended: ExtendedAnalysis) -> dict[str, pd.DataFrame]:
    """Extended artifacts written by :func:`save_outputs`."""
    return {
        BACKTEST_CSV: extended.backtest.stats,
        BACKTEST_CURVES_CSV: extended.backtest.equity_curves(),
        SCENARIO_CSV: extended.scenario_table,
        RISK_CONTRIBUTIONS_CSV: extended.risk_contributions,
        VAR_BACKTEST_CSV: extended.var_backtest,
        VOL_FORECAST_CSV: extended.vol_forecast,
    }


def _percent_display(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Scale the given columns to percent and suffix their names with ``(%)``."""
    display = frame.copy()
    for column in columns:
        if column in display.columns:
            display[column] = (pd.to_numeric(display[column], errors="coerce") * 100).round(2)
            display = display.rename(columns={column: f"{column} (%)"})
    return display


def _extended_sections(extended: ExtendedAnalysis) -> str:
    """HTML blocks for the walk-forward backtest, stress tests and risk models."""
    backtest = extended.backtest.stats
    keep = [
        "Annualized_Return",
        "Annualized_Volatility",
        "Sharpe_Ratio",
        "Sortino_Ratio",
        "Max_Drawdown",
        "Calmar_Ratio",
        "Avg_Turnover",
        "Annual_Cost",
        "Total_Return",
    ]
    backtest_display = _percent_display(
        backtest[[column for column in keep if column in backtest.columns]],
        ["Annualized_Return", "Annualized_Volatility", "Max_Drawdown", "Avg_Turnover", "Annual_Cost", "Total_Return"],
    )
    scenario = _percent_display(extended.scenario_returns, list(extended.scenario_returns.columns))
    scenario_drawdown = _percent_display(extended.scenario_drawdowns, list(extended.scenario_drawdowns.columns))
    risk = _percent_display(
        extended.risk_summary[
            [
                "Annualized_Volatility",
                "Diversification_Ratio",
                "Concentration_HHI",
                "Top_Risk_Asset",
                "Top_Risk_Percent",
            ]
        ],
        ["Annualized_Volatility", "Top_Risk_Percent"],
    )
    var_table = _percent_display(extended.var_backtest, ["Breach_Rate", "Expected_Rate", "Avg_Breach_Loss"])
    forecast_columns = ["Name", "Forecast_Vol", "Realized_Vol_60d", "Forecast_vs_Realized", "Regime"]
    forecast = _percent_display(
        extended.vol_forecast[[column for column in forecast_columns if column in extended.vol_forecast.columns]].head(
            12
        ),
        ["Forecast_Vol", "Realized_Vol_60d"],  # Forecast_vs_Realized is a ratio, not a percentage
    )
    chart = _equity_curve_data_uri(extended)
    chart_html = (
        f'<p><img alt="Walk-forward equity curves" src="{chart}" style="max-width:100%; border:1px solid #d7dee5;"></p>'
        if chart
        else ""
    )
    return f"""
  <h2>Walk-Forward Backtest (out-of-sample)</h2>
  <p class="meta">Weights are re-estimated each month on the trailing {extended.backtest.lookback} trading days
  and held until the next rebalance; {extended.backtest.cost_bps:.0f} bp one-way costs are charged on turnover.
  The benchmark is a buy-and-hold reference over the same period.</p>
  {chart_html}
  {backtest_display.to_html(border=0)}

  <h2>Stress Scenarios</h2>
  <p class="meta">Total return inside each historical stress window; rows outside the sample are blank.</p>
  {scenario.to_html(border=0)}
  <h3>Maximum drawdown inside each window</h3>
  {scenario_drawdown.to_html(border=0)}

  <h2>Risk Decomposition</h2>
  <p class="meta">Volatility contributions use the Euler decomposition; the diversification ratio is the
  weighted-average asset volatility divided by portfolio volatility.</p>
  {risk.to_html(border=0)}

  <h2>VaR Backtest</h2>
  <p class="meta">Trailing 500-day historical VaR with Kupiec proportion-of-failures coverage tests; a p-value
  below 0.05 rejects the model at the 5% level.</p>
  {var_table.to_html(border=0)}

  <h2>GARCH(1,1) Volatility Forecasts</h2>
  <p class="meta">Annualized conditional volatility and 21-day forecast vs the last 60 days of realized volatility.</p>
  {forecast.to_html(border=0)}
"""


def _equity_curve_data_uri(extended: ExtendedAnalysis) -> str:
    """Base64 PNG of the walk-forward equity curves (empty string if plotting is unavailable)."""
    try:
        import base64
        import io

        import matplotlib.pyplot as plt

        curves = extended.backtest.equity_curves()
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for column in curves.columns:
            ax.plot(curves.index, curves[column], label=column, linewidth=1.6)
        ax.set_yscale("log")
        ax.set_ylabel("Growth of $100 (log scale)")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=110)
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001 - the report must render even without a plotting backend
        return ""
