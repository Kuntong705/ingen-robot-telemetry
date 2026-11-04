"""Operator-facing metrics: alert budget, lead time, false alarms per unit-week."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HOURS_PER_WEEK = 168.0


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, score))


def raise_alerts(scored: pd.DataFrame, threshold: float | None, refractory_hours: float) -> pd.DataFrame:
    """One alert per crossing: after an alert the unit is muted for
    ``refractory_hours`` (the operator is already handling it).

    ``scored`` needs columns unit_id, t, score; rows must be eligible times.
    With ``threshold=None`` a per-row ``thr`` column is used (thresholds are
    chosen per fold on that fold's validation rows).
    """
    alerts = []
    refractory = pd.Timedelta(hours=refractory_hours)
    for unit, frame in scored.sort_values(["unit_id", "t"]).groupby("unit_id", sort=False):
        thr = frame["thr"] if threshold is None else threshold
        hot = frame.loc[frame["score"] >= thr, "t"]
        muted_until = None
        for t in hot:
            if muted_until is None or t >= muted_until:
                alerts.append((unit, t))
                muted_until = t + refractory
    return pd.DataFrame(alerts, columns=["unit_id", "t"])


def events_in_scored_period(scored: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Faults whose qualifying onset falls after a unit's first scored time
    and within one horizon of its last scored time."""
    span = scored.groupby("unit_id")["t"].agg(["min", "max"])
    ev = events[events["unit_id"].isin(span.index)].copy()
    lo = ev["unit_id"].map(span["min"])
    hi = ev["unit_id"].map(span["max"])
    return ev[(ev["onset_utc"] > lo) & (ev["onset_utc"] <= hi + pd.Timedelta(hours=1))]


def evaluate_alerts(
    scored: pd.DataFrame,
    events: pd.DataFrame,
    threshold: float | None,
    refractory_hours: float,
    horizon_hours: float,
) -> dict[str, Any]:
    """``scored``: eligible test rows (unit_id, t, score, y).
    ``events``: faults whose qualifying onset is inside the scored period of
    their unit (unit_id, fault_id, onset_utc)."""
    alerts = raise_alerts(scored, threshold, refractory_hours)
    lab = scored.set_index(["unit_id", "t"])["y"]
    if len(alerts):
        alert_y = lab.reindex(pd.MultiIndex.from_frame(alerts)).fillna(0).to_numpy()
    else:
        alert_y = np.array([])
    n_alerts = len(alerts)
    tp_alerts = int(alert_y.sum())
    unit_weeks = len(scored) / HOURS_PER_WEEK  # scored rows are hourly unit-times
    horizon = pd.Timedelta(hours=horizon_hours)

    lead_times = []
    detected = 0
    for ev in events.itertuples(index=False):
        a = alerts[(alerts["unit_id"] == ev.unit_id) & (alerts["t"] >= ev.onset_utc - horizon)
                   & (alerts["t"] < ev.onset_utc)]
        if len(a):
            detected += 1
            lead_times.append((ev.onset_utc - a["t"].min()) / pd.Timedelta(hours=1))
    n_events = len(events)
    return {
        "threshold": float(threshold) if threshold is not None else float("nan"),
        "alerts": float(n_alerts),
        "alerts_per_unit_week": n_alerts / unit_weeks if unit_weeks else float("nan"),
        "precision": tp_alerts / n_alerts if n_alerts else float("nan"),
        "false_alarms_per_unit_week": (n_alerts - tp_alerts) / unit_weeks if unit_weeks else float("nan"),
        "events": float(n_events),
        "events_detected": float(detected),
        "event_recall": detected / n_events if n_events else float("nan"),
        "median_lead_time_h": float(np.median(lead_times)) if lead_times else float("nan"),
        "unit_weeks": float(unit_weeks),
        "lead_times_h": lead_times,
    }


def threshold_for_budget(
    scored: pd.DataFrame, budget_per_unit_week: float, refractory_hours: float, n_grid: int = 60
) -> float:
    """Lowest score threshold whose alert rate on ``scored`` (validation
    rows) stays within the budget. Chosen on validation, never on test."""
    unit_weeks = len(scored) / HOURS_PER_WEEK
    candidates = np.unique(np.quantile(scored["score"], np.linspace(0.5, 0.99999, n_grid)))
    best = float(scored["score"].max()) + 1e-9

    def within(thr: float) -> bool:
        return len(raise_alerts(scored, thr, refractory_hours)) / unit_weeks <= budget_per_unit_week

    # alert count falls (almost) monotonically with the threshold: bisect the
    # lowest candidate that stays within budget
    lo, hi = 0, len(candidates) - 1
    if hi < 0 or not within(float(candidates[hi])):
        return best
    while lo < hi:
        mid = (lo + hi) // 2
        if within(float(candidates[mid])):
            hi = mid
        else:
            lo = mid + 1
    return float(candidates[lo])


def unit_cluster_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    n_boot: int = 300,
    seed: int = 0,
    unit_col: str = "unit_id",
) -> tuple[float, float]:
    """95 % CI by resampling whole units (rows of one robot are correlated)."""
    rng = np.random.default_rng(seed)
    units = frame[unit_col].unique()
    by_unit = {u: g for u, g in frame.groupby(unit_col)}
    values = []
    for _ in range(n_boot):
        pick = rng.choice(units, len(units), replace=True)
        sample = pd.concat([by_unit[u] for u in pick], ignore_index=True)
        v = statistic(sample)
        if np.isfinite(v):
            values.append(v)
    if not values:
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def summarise(values: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in values.items()}
