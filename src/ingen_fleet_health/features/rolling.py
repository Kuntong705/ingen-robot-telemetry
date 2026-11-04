"""Trailing rolling statistics on a complete per-unit minute grid.

As-of rule: the value at minute t uses only minutes in (t - w, t]. All
windows are trailing (pandas ``rolling`` without ``center``); missing
minutes are NaN and simply reduce the count of valid samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS_MIN = {"1h": 60, "6h": 360, "24h": 1440}
STATS = ("mean", "std", "min", "max", "slope")


def min_periods(window: int) -> int:
    return max(2, window // 4)


def rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    """OLS slope of value vs time (per hour) over the trailing window.

    Uses rolling sums of x, y, xy, x^2 over valid samples only, so it is
    exact with gaps and causal by construction.
    """
    y = pd.Series(values, dtype="float64")
    valid = y.notna().astype("float64")
    x = pd.Series(np.arange(len(y), dtype="float64") / 60.0)  # hours
    yv = y.fillna(0.0)
    xv = x * valid
    mp = min_periods(window)

    def rsum(s: pd.Series) -> pd.Series:
        return s.rolling(window, min_periods=1).sum()

    n = rsum(valid)
    sx, sy = rsum(xv), rsum(yv)
    sxy, sxx = rsum(xv * yv), rsum(xv * xv)
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom.where(denom > 1e-12)
    return slope.where(n >= mp).to_numpy()


def rolling_block(series: pd.Series, prefix: str, windows: dict[str, int] | None = None) -> dict[str, np.ndarray]:
    windows = windows or WINDOWS_MIN
    out: dict[str, np.ndarray] = {}
    for label, w in windows.items():
        roll = series.rolling(w, min_periods=min_periods(w))
        out[f"{prefix}_mean_{label}"] = roll.mean().to_numpy()
        out[f"{prefix}_std_{label}"] = roll.std().to_numpy()
        out[f"{prefix}_min_{label}"] = roll.min().to_numpy()
        out[f"{prefix}_max_{label}"] = roll.max().to_numpy()
        out[f"{prefix}_slope_{label}"] = rolling_slope(series.to_numpy(), w)
    return out
