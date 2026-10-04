"""VaR model validation: rolling historical-simulation VaR and Kupiec coverage tests.

A risk number is only credible once it has been backtested. These functions estimate
VaR from trailing windows (no look-ahead), count the breaches and run Kupiec's
proportion-of-failures test for unconditional coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rolling_var(returns: pd.Series, window: int = 500, level: float = 0.05) -> pd.Series:
    """Positive daily VaR from the trailing window, shifted so there is no look-ahead."""
    estimates = -returns.rolling(window=window).quantile(level).shift(1)
    return estimates.rename("VaR")


def breach_flags(returns: pd.Series, var: pd.Series) -> pd.Series:
    """True where the realized return was worse than the VaR forecast."""
    return (returns < -var).rename("Breach")


def kupiec_pof(returns: pd.Series, var: pd.Series, level: float = 0.05) -> dict[str, float]:
    """Kupiec proportion-of-failures test for unconditional VaR coverage."""
    valid = returns.notna() & var.notna()
    realized = returns[valid]
    n = len(realized)
    if n == 0:
        return {
            "Observations": 0,
            "Breaches": 0,
            "Breach_Rate": float("nan"),
            "Expected_Rate": level,
            "LR_Statistic": float("nan"),
            "P_Value": float("nan"),
            "Avg_Breach_Loss": float("nan"),
        }
    breaches = realized < -var[valid]
    failures = int(breaches.sum())
    rate = failures / n
    if failures == 0:
        lr = -2.0 * n * np.log(1.0 - level)
    elif failures == n:
        lr = -2.0 * n * np.log(level)
    else:
        lr = -2.0 * (
            np.log((1.0 - level) ** (n - failures) * level**failures)
            - np.log((1.0 - rate) ** (n - failures) * rate**failures)
        )
    return {
        "Observations": n,
        "Breaches": failures,
        "Breach_Rate": rate,
        "Expected_Rate": level,
        "LR_Statistic": float(lr),
        "P_Value": float(stats.chi2.sf(lr, 1)),
        "Avg_Breach_Loss": float(-realized[breaches].mean()) if failures else float("nan"),
    }


def var_backtest_summary(
    series_map: dict[str, pd.Series],
    window: int = 500,
    levels: tuple[float, ...] = (0.05, 0.01),
) -> pd.DataFrame:
    """Coverage-test table: one row per return series and VaR level."""
    rows = []
    for name, series in series_map.items():
        for level in levels:
            var = rolling_var(series, window=window, level=level)
            row = kupiec_pof(series, var, level=level)
            row["Portfolio"] = name
            row["Level"] = f"{1 - level:.0%}"
            p_value = row["P_Value"]
            if p_value != p_value:  # NaN: no testable window
                row["Verdict"] = "n/a"
            else:
                row["Verdict"] = "Pass" if p_value > 0.05 else "Reject"
            rows.append(row)
    columns = [
        "Portfolio",
        "Level",
        "Observations",
        "Breaches",
        "Breach_Rate",
        "Expected_Rate",
        "LR_Statistic",
        "P_Value",
        "Verdict",
        "Avg_Breach_Loss",
    ]
    return pd.DataFrame(rows)[columns]
