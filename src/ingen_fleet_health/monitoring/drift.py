"""Population stability index (PSI) against the training reference.

Reference bins are the deciles of each feature on the model's training rows of
that platform, plus a separate bin for missing values (a telemetry outage
shifts the missing share, which is drift an operator should see).

PSI = sum_i (a_i - e_i) * ln(a_i / e_i) over bins, with shares floored at
1e-4. Conventional reading: < 0.10 stable, 0.10-0.25 watch, > 0.25 shift.
PSI on one day of hourly rows is noisy for small fleets; the dashboard shows
it next to the row count it was computed on.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FLOOR = 1e-4


def reference_bins(values: pd.Series, n_bins: int = 10) -> dict[str, Any]:
    v = values.to_numpy(dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        edges: list[float] = []
    else:
        edges = sorted(set(np.quantile(finite, np.linspace(0, 1, n_bins + 1)[1:-1]).round(12).tolist()))
    counts = bin_counts(values, edges)
    return {"edges": edges, "expected": (counts / max(counts.sum(), 1)).tolist(), "n": len(v)}


def bin_counts(values: pd.Series, edges: list[float]) -> np.ndarray:
    """Counts per bin: len(edges) + 1 value bins, then one bin for missing values."""
    v = values.to_numpy(dtype=float)
    finite = np.isfinite(v)
    idx = np.searchsorted(np.asarray(edges, dtype=float), v[finite], side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    return np.append(counts, float((~finite).sum()))


def psi(expected: list[float] | np.ndarray, actual_counts: np.ndarray) -> float:
    e = np.maximum(np.asarray(expected, dtype=float), FLOOR)
    total = actual_counts.sum()
    if total == 0:
        return float("nan")
    a = np.maximum(actual_counts / total, FLOOR)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_band(value: float, watch: float = 0.10, shift: float = 0.25) -> str:
    if not np.isfinite(value):
        return "no data"
    return "shift" if value > shift else ("watch" if value > watch else "stable")


def psi_table(frame: pd.DataFrame, reference: dict[str, dict[str, Any]], watch: float = 0.10,
              shift: float = 0.25) -> pd.DataFrame:
    rows = []
    for feature, ref in reference.items():
        if feature not in frame.columns:
            continue
        value = psi(ref["expected"], bin_counts(frame[feature], ref["edges"]))
        rows.append({"feature": feature, "psi": value, "band": psi_band(value, watch, shift), "rows": len(frame)})
    return pd.DataFrame(rows, columns=["feature", "psi", "band", "rows"])
