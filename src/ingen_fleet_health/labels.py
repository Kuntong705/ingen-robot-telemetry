"""Prediction-time labels (ADR-003 rev1).

Early warning, at each hourly prediction time t for unit u:

* ``y`` = 1 if a qualifying fault onset of u falls in (t, t + H], H = 72 h.
* ``in_active_fault`` -- an onset has already happened and the unit is not yet
  repaired (onset <= t < repair_end): the fault is known, a "new onset"
  prediction is meaningless -> excluded.
* ``post_repair`` -- t within 24 h after a repair completes -> excluded.
* ``censored`` -- t + H beyond the end of observation: the future window is
  incomplete, so the row is dropped rather than labelled negative.

Labels read the hidden ground truth; they are joined to features only after
the features have been computed, and never feed back into them.

SoH forecasting: target is the true end-of-day SoH h days ahead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_fault_events(path: str | Path) -> pd.DataFrame:
    events = pd.read_csv(path)
    for col in ("initiation_utc", "onset_utc", "failure_utc", "repair_start_utc", "repair_end_utc"):
        events[col] = pd.to_datetime(events[col], utc=True)
    return events


def prediction_grid(
    unit_ids: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval_hours: int = 1,
    warmup_hours: int = 24,
) -> pd.DataFrame:
    """Hourly prediction times t in [start + warmup, end] for every unit."""
    times = pd.date_range(
        start + pd.Timedelta(hours=warmup_hours), end, freq=f"{interval_hours}h", tz="UTC"
    )
    return pd.DataFrame({
        "unit_id": np.repeat(unit_ids, len(times)),
        "t": np.tile(times, len(unit_ids)),
    })


def early_warning_labels(
    grid: pd.DataFrame,
    events: pd.DataFrame,
    observation_end: pd.Timestamp,
    horizon_hours: float = 72,
    post_repair_hours: float = 24,
) -> pd.DataFrame:
    horizon = pd.Timedelta(hours=horizon_hours)
    post = pd.Timedelta(hours=post_repair_hours)
    out = grid[["unit_id", "t"]].copy()
    out["y"] = 0
    out["in_active_fault"] = False
    out["post_repair"] = False
    out["fault_id"] = pd.Series([None] * len(out), dtype="object")
    out["hours_to_onset"] = np.nan

    t = out["t"]
    for ev in events.itertuples(index=False):
        unit = out["unit_id"] == ev.unit_id
        upcoming = unit & (t < ev.onset_utc) & (t + horizon >= ev.onset_utc)
        out.loc[upcoming, "y"] = 1
        out.loc[upcoming, "fault_id"] = ev.fault_id
        out.loc[upcoming, "hours_to_onset"] = (ev.onset_utc - t[upcoming]).dt.total_seconds() / 3600
        out.loc[unit & (t >= ev.onset_utc) & (t < ev.repair_end_utc), "in_active_fault"] = True
        out.loc[unit & (t >= ev.repair_end_utc) & (t < ev.repair_end_utc + post), "post_repair"] = True

    out["censored"] = t + horizon > observation_end
    out["eligible"] = ~(out["in_active_fault"] | out["post_repair"] | out["censored"])
    return out


def soh_targets(battery_truth_daily: pd.DataFrame, horizons_days: list[int]) -> pd.DataFrame:
    """One row per unit-day d with ``soh_true_pct`` at d and ``target_h{h}`` at d + h."""
    truth = battery_truth_daily.copy()
    truth["date_utc"] = pd.to_datetime(truth["date_utc"])
    truth = truth.sort_values(["unit_id", "date_utc"])
    for h in horizons_days:
        truth[f"target_h{h}"] = truth.groupby("unit_id")["soh_true_pct"].shift(-h)
    return truth
