"""Command-line entry point: fetch data, run the full pipeline and write the report suite."""

from __future__ import annotations

import argparse
import sys

from .config import DEFAULT_RF
from .data import fetch_prices, load_prices, save_prices
from .pipeline import compute_all, compute_extended
from .reporting import executive_summary, save_outputs
from .universe import default_tickers


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run the full financial analysis pipeline.")
    parser.add_argument("--offline", action="store_true", help="use the local price snapshot only")
    parser.add_argument(
        "--fast", action="store_true", help="skip the extended suite (backtest, scenarios, risk models)"
    )
    parser.add_argument("--mc", type=int, default=10_000, help="Monte Carlo portfolio count")
    parser.add_argument("--rf", type=float, default=DEFAULT_RF, help="annual risk-free rate (decimal)")
    args = parser.parse_args(argv)

    tickers = default_tickers()
    if args.offline:
        prices = load_prices(tickers=tickers)
        source, error = "snapshot", None
    else:
        try:
            prices = fetch_prices(tickers)
            save_prices(prices)
            source, error = "live", None
        except Exception as exc:  # noqa: BLE001 - fall back to snapshot for resilience
            print(f"Live fetch failed ({exc}); falling back to snapshot.")
            try:
                prices = load_prices(tickers=tickers)
            except Exception as snapshot_exc:  # noqa: BLE001
                print(f"No live data and no usable snapshot: {snapshot_exc}")
                return 1
            source, error = "snapshot", str(exc)

    bundle = compute_all(tickers=tickers, prices=prices, rf=args.rf, mc_portfolios=args.mc, source=source, error=error)

    extended = None
    if not args.fast:
        print("Running extended analysis: backtest, scenarios, risk models...")
        extended = compute_extended(bundle)

    print(f"Assets: {len(bundle.metrics)} | observations: {len(bundle.returns)} | source: {source}")
    for path in save_outputs(bundle, extended):
        print(f"wrote {path}")

    print("\n".join(f" {bullet.replace('**', '')}" for bullet in executive_summary(bundle, extended=extended)))
    return 0
