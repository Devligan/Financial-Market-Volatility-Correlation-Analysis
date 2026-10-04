"""Price data access: live Yahoo Finance fetch with a local snapshot fallback."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .config import DEFAULT_START, SNAPSHOT_PATH
from .universe import default_tickers


def fetch_prices(
    tickers: Sequence[str] | None = None,
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
    min_observations: int = 60,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance via ``yfinance``.

    Defaults to a long-horizon window starting ``DEFAULT_START`` (2010). Pass
    ``period`` for a rolling window instead (e.g. ``"1y"``). Short gaps (up to
    three trading days, e.g. a holiday print missing on one venue) are forward
    filled; rows that still contain any missing asset are dropped, so the panel
    starts at the common window of the requested universe and stays NaN-free.
    """
    import yfinance as yf  # imported lazily so the module works offline for math/tests

    tickers = list(default_tickers()) if tickers is None else [t.upper() for t in tickers]
    if not tickers:
        raise ValueError("fetch_prices() received an empty ticker list")
    kwargs = {"auto_adjust": True, "progress": False, "group_by": "column", "threads": True}
    if period is not None:
        data = yf.download(tickers, period=period, **kwargs)
    elif start or end:
        data = yf.download(tickers, start=start, end=end, **kwargs)
    else:
        data = yf.download(tickers, start=DEFAULT_START, **kwargs)

    if data is None or len(data) == 0:
        raise RuntimeError("yfinance returned no data (offline or invalid tickers)")

    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        field = "Close" if "Close" in level0 else level0[0]
        prices = data[field]
    else:
        prices = data[["Close"]].copy()
        prices.columns = [tickers[0]]

    prices = prices.reindex(columns=[t for t in tickers if t in prices.columns]).astype(float)
    prices.index = pd.DatetimeIndex(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.sort_index().ffill(limit=3).dropna()
    prices = prices.drop(columns=[c for c in prices.columns if prices[c].count() < min_observations])
    if prices.empty or prices.shape[1] == 0:
        raise RuntimeError("No price history left after cleaning")
    return prices


def save_prices(prices: pd.DataFrame, path: str | Path = SNAPSHOT_PATH) -> Path:
    """Persist a price snapshot so the dashboard can run offline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index_label="Date")
    return path


def load_prices(path: str | Path = SNAPSHOT_PATH, tickers: Sequence[str] | None = None) -> pd.DataFrame:
    """Load the local price snapshot written by :func:`save_prices`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Price snapshot not found: {path}. Run `python analysis.py` while online first.")
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    if tickers is not None:
        wanted = [t.upper() for t in tickers]
        prices = prices.reindex(columns=[t for t in wanted if t in prices.columns])
    prices = prices.dropna(how="all")
    if prices.empty or prices.shape[1] == 0:
        raise RuntimeError(f"Price snapshot {path} does not contain the requested tickers")
    return prices
