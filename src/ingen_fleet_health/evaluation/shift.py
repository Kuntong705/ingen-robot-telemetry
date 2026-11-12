"""Distribution-shift measures between two feature tables (e.g. Rover vs Fari)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def psi(reference: np.ndarray, other: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index of ``other`` against ``reference``.

    Bin edges are the reference deciles; missing values form their own bin, so
    a feature that becomes unavailable shows up as shift. Rule of thumb:
    < 0.1 stable, 0.1-0.25 moderate, > 0.25 major shift.
    """
    ref = np.asarray(reference, dtype=float)
    new = np.asarray(other, dtype=float)
    ref_ok, new_ok = ref[np.isfinite(ref)], new[np.isfinite(new)]
    if len(ref) == 0 or len(new) == 0:
        return float("nan")
    if len(ref_ok) == 0:
        edges = np.array([])
    else:
        edges = np.unique(np.quantile(ref_ok, np.linspace(0, 1, bins + 1)[1:-1]))

    def shares(values: np.ndarray, ok: np.ndarray) -> np.ndarray:
        counts = np.bincount(np.searchsorted(edges, ok, side="right"), minlength=len(edges) + 1).astype(float)
        counts = np.append(counts, len(values) - len(ok))            # the missing-value bin
        return np.maximum(counts / len(values), eps)

    p, q = shares(ref, ref_ok), shares(new, new_ok)
    return float(np.sum((q - p) * np.log(q / p)))


def availability(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Share of valid values and whether a feature is constant, per column."""
    rows = []
    for col in columns:
        values = frame[col].to_numpy(dtype=float)
        ok = values[np.isfinite(values)]
        rows.append({"feature": col, "valid_share": len(ok) / len(values) if len(values) else float("nan"),
                     "constant": bool(len(ok) > 0 and np.ptp(ok) < 1e-12)})
    return pd.DataFrame(rows)
