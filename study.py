"""Notebook-style study runner built on the shared analytics engine.

Usage
-----
    python study.py               # live data, prints tables/plots, writes artifacts
    python study.py --offline     # use the local price snapshot only

    from study import run_study   # use inside analysis_walkthrough.ipynb
    bundle = run_study()
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import finrisk as an
from finrisk import presentation as pres

DISPLAY_COLUMNS = [
    "Asset_Class",
    "Annualized_Return",
    "Annualized_Volatility",
    "Sharpe_Ratio",
    "Sortino_Ratio",
    "Max_Drawdown",
    "Calmar_Ratio",
    "VaR_95",
    "CVaR_95",
    "Beta",
    "Alpha",
    "Tracking_Error",
    "Information_Ratio",
]


def _show(obj) -> None:
    """Pretty-print in Jupyter, plain-print elsewhere."""
    try:
        from IPython.display import display  # type: ignore

        display(obj)
    except ImportError:
        print(obj)


def _load(tickers: list[str], prefer_live: bool) -> tuple[pd.DataFrame, str]:
    if prefer_live:
        try:
            prices = an.fetch_prices(tickers)
            an.save_prices(prices)
            return prices, "live"
        except Exception as exc:  # noqa: BLE001 - fall back to the local snapshot
            print(f"Live fetch failed ({exc}); using the local snapshot.")
    return an.load_prices(tickers=tickers), "snapshot"


def _plot_results(bundle: an.AnalysisBundle) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Monte Carlo cloud + efficient frontier + optimal portfolios
    ax = axes[0]
    mc = bundle.monte_carlo.sample(min(5000, len(bundle.monte_carlo)), random_state=0)
    cloud = ax.scatter(
        mc["Annualized_Volatility"],
        mc["Annualized_Return"],
        c=mc["Sharpe_Ratio"],
        cmap="viridis",
        s=6,
        alpha=0.4,
    )
    if not bundle.frontier.empty:
        ax.plot(
            bundle.frontier["Annualized_Volatility"],
            bundle.frontier["Annualized_Return"],
            color="crimson",
            lw=2.5,
            label="Efficient frontier",
        )
    markers = [("Max Sharpe", bundle.max_sharpe, "*", "red"), ("Min Variance", bundle.min_variance, "D", "black")]
    for name, weights, marker, color in markers:
        stats, _ = an.portfolio_stats(bundle.returns, weights, rf=bundle.risk_free)
        ax.scatter(
            stats["Annualized_Volatility"],
            stats["Annualized_Return"],
            marker=marker,
            s=220,
            c=color,
            edgecolors="white",
            zorder=5,
            label=name,
        )
    ax.set_title("Monte Carlo portfolios & efficient frontier")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    plt.colorbar(cloud, ax=ax, label="Sharpe ratio")
    ax.legend()

    # Growth of $100 for the optimized portfolios
    ax = axes[1]
    curves = [
        ("Max Sharpe", bundle.max_sharpe, "red"),
        ("Min Variance", bundle.min_variance, "black"),
        ("Equal Weight", bundle.equal_weight, pres.ACCENT),
    ]
    for name, weights, color in curves:
        growth = 100 * (1 + an.portfolio_returns(bundle.returns, weights)).cumprod()
        ax.plot(growth.index, growth.values, label=name, color=color, lw=1.6)
    ax.set_title("Growth of $100 (in-sample backtest)")
    ax.set_ylabel("Portfolio value")
    ax.legend()

    plt.tight_layout()


def _plot_weights(bundle: an.AnalysisBundle) -> None:
    top_weights = bundle.max_sharpe[bundle.max_sharpe > 0.01].sort_values()
    plt.figure(figsize=(10, max(4, 0.35 * len(top_weights))))
    plt.barh(top_weights.index, top_weights.values, color=pres.ACCENT)
    plt.title("Maximum Sharpe portfolio weights (>1%)")
    plt.gca().xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    plt.tight_layout()


def run_study(
    tickers: list[str] | None = None,
    rf: float = an.DEFAULT_RF,
    benchmark: str = an.BENCHMARK,
    mc_portfolios: int = 10_000,
    prefer_live: bool = True,
) -> an.AnalysisBundle:
    """Run the complete study: data, metrics, optimization, plots, exports."""
    tickers = [t.upper() for t in (tickers or an.default_tickers())]
    prices, source = _load(tickers, prefer_live)

    print("Financial Volatility & Correlation Study")
    print(
        f"   Assets: {prices.shape[1]} | trading days: {len(prices)} | "
        f"range: {prices.index[0].date()}  {prices.index[-1].date()} | source: {source}"
    )

    bundle = an.compute_all(
        tickers=tickers,
        prices=prices,
        rf=rf,
        benchmark=benchmark,
        mc_portfolios=mc_portfolios,
        source=source,
    )

    print("\nRunning extended analysis (backtest, scenarios, risk models)...")
    extended = an.compute_extended(bundle)

    print("\n1. Risk & return metrics (sorted by Sharpe ratio):")
    _show(bundle.metrics[DISPLAY_COLUMNS].sort_values("Sharpe_Ratio", ascending=False).round(4))

    print("\n2. Average risk profile by asset class:")
    _show(an.class_summary(bundle.metrics).round(4))

    portfolios = {
        "Max Sharpe": bundle.max_sharpe,
        "Min Variance": bundle.min_variance,
        "Equal Weight": bundle.equal_weight,
    }
    print("\n3. Optimized portfolio statistics (in-sample):")
    stats_table = pd.DataFrame(
        {
            name: an.portfolio_stats(bundle.returns, weights, rf=bundle.risk_free)[0]
            for name, weights in portfolios.items()
        }
    ).T
    _show(stats_table.round(4))

    print("\n4. Portfolio weights above 1%:")
    weights_table = pd.DataFrame({name: weights[weights > 0.01] for name, weights in portfolios.items()})
    _show(weights_table.round(4))

    _plot_results(bundle)
    _plot_weights(bundle)

    print("\n5. Walk-forward backtest (out-of-sample):")
    backtest_columns = [
        "Annualized_Return",
        "Annualized_Volatility",
        "Sharpe_Ratio",
        "Max_Drawdown",
        "Avg_Turnover",
        "Annual_Cost",
    ]
    _show(extended.backtest.stats[backtest_columns].round(4))

    print("\n6. Stress scenarios (total return):")
    _show(extended.scenario_returns.round(4))

    print("\n7. Exported artifacts:")
    for path in an.save_outputs(bundle, extended):
        print(f"   wrote {path}")

    print("\nExecutive summary:")
    for bullet in an.executive_summary(bundle, extended=extended):
        print("   ", bullet.replace("**", ""))

    # Show every figure in a single blocking call, after all text output is done.
    backend = plt.get_backend().lower()
    if backend != "agg":
        if "inline" not in backend:
            print("\nClose the plot windows to finish.")
        plt.show()

    return bundle


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run the notebook-style study and write artifacts.")
    parser.add_argument("--offline", action="store_true", help="use the local price snapshot only")
    parser.add_argument("--mc", type=int, default=10_000, help="Monte Carlo portfolio count")
    parser.add_argument("--rf", type=float, default=an.DEFAULT_RF, help="annual risk-free rate (decimal)")
    args = parser.parse_args(argv)

    run_study(rf=args.rf, mc_portfolios=args.mc, prefer_live=not args.offline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
