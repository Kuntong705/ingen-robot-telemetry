"""Weekly performance check against newly confirmed faults (ADR-0006).

Labels arrive late: a fault's onset time is known only once a technician
confirms it at repair start (up to 96 h after onset in the simulator), and a
prediction at t needs every onset in (t, t + 72 h]. So on check day D the
pipeline evaluates the predictions of the week [D - maturity - 7 d, D - maturity),
with maturity = 7 days, using only confirmations that have arrived by D.

The labels are rebuilt with the Week 2 rule (ADR-003 rev1): y = 1 if a
confirmed onset falls in (t, t + 72 h]; rows inside an active fault (onset up
to the repair completion seen in the maintenance feed) or within 24 h after a
repair are excluded. Results are split by platform and by cohort: held-out
units (never seen in training: the honest number) and training units (later
time only).

With a handful of faults per week the weekly numbers are very noisy; a week
with fewer than 3 confirmed onsets is reported as "insufficient labels"
rather than as good or bad performance.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ingen_fleet_health.evaluation.metrics import HOURS_PER_WEEK, mae, pr_auc
from ingen_fleet_health.labels import early_warning_labels
from ingen_fleet_health.pipeline.settings import Settings, atomic_partition, daterange

MIN_EVENTS = 3


def confirmed_faults(settings: Settings, platform: str, day: date) -> pd.DataFrame:
    frames = [pd.read_csv(settings.landing_dir("labels", platform, d) / "fault_confirmations.csv")
              for d in settings.landing_dates("labels", platform) if d <= day]
    if not frames:
        return pd.DataFrame(columns=["fault_id", "unit_id", "fault_type", "onset_utc", "confirmed_at_utc"])
    out = pd.concat(frames, ignore_index=True)
    for c in ("onset_utc", "confirmed_at_utc"):
        out[c] = pd.to_datetime(out[c], utc=True)
    return out


def repairs_known(settings: Settings, platform: str, day: date) -> pd.DataFrame:
    frames = [pd.read_csv(settings.landing_dir("maintenance", platform, d) / "maintenance.csv")
              for d in settings.landing_dates("maintenance", platform) if d <= day]
    if not frames:
        return pd.DataFrame(columns=["unit_id", "completed_at_utc"])
    out = pd.concat(frames, ignore_index=True)
    out["completed_at_utc"] = pd.to_datetime(out["completed_at_utc"], utc=True)
    return out


def label_events(faults: pd.DataFrame, repairs: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Confirmed faults with the first known repair completion after confirmation."""
    ev = faults.copy()
    ends = []
    for f in ev.itertuples(index=False):
        done = repairs.loc[(repairs["unit_id"] == f.unit_id) & (repairs["completed_at_utc"] >= f.confirmed_at_utc),
                           "completed_at_utc"]
        ends.append(done.min() if len(done) else as_of)
    ev["repair_end_utc"] = pd.to_datetime(pd.Series(ends, dtype="object"), utc=True) if ends else pd.Series(
        dtype="datetime64[ns, UTC]")
    return ev


def _cohort_metrics(scored: pd.DataFrame, alerts: pd.DataFrame, events: pd.DataFrame,
                    horizon: pd.Timedelta) -> dict[str, Any]:
    y = scored["y"].to_numpy()
    out: dict[str, Any] = {"rows": len(scored), "positive_rows": int(y.sum()),
                           "unit_weeks": len(scored) / HOURS_PER_WEEK}
    for col, name in (("risk", "champion"), ("iforest_score", "iforest"), ("physics_rule_score", "physics_rule")):
        out[f"pr_auc_{name}"] = pr_auc(y, scored[col].to_numpy()) if len(scored) else float("nan")
    out["base_rate"] = float(y.mean()) if len(y) else float("nan")
    lab = scored.set_index(["unit_id", "t"])["y"]
    if len(alerts):
        hit = lab.reindex(pd.MultiIndex.from_frame(alerts[["unit_id", "t"]])).fillna(0).to_numpy()
    else:
        hit = np.array([])
    out["alerts"] = len(alerts)
    out["precision"] = float(hit.mean()) if len(hit) else float("nan")
    out["false_alarms_per_unit_week"] = (len(hit) - hit.sum()) / out["unit_weeks"] if out["unit_weeks"] else float("nan")
    detected, leads = 0, []
    for e in events.itertuples(index=False):
        a = alerts[(alerts["unit_id"] == e.unit_id) & (alerts["t"] >= e.onset_utc - horizon) & (alerts["t"] < e.onset_utc)]
        if len(a):
            detected += 1
            leads.append((e.onset_utc - a["t"].min()) / pd.Timedelta(hours=1))
    out["events"] = len(events)
    out["events_detected"] = detected
    out["event_recall"] = detected / len(events) if len(events) else float("nan")
    out["median_lead_time_h"] = float(np.median(leads)) if leads else float("nan")
    out["status"] = "insufficient labels" if len(events) < MIN_EVENTS else "evaluated"
    return out


def weekly_check(settings: Settings, day: date, training_units: dict[str, list[str]]) -> dict[str, Any]:
    maturity = int(settings.monitoring["label_maturity_days"])
    w_end = day - timedelta(days=maturity)          # exclusive
    w_start = w_end - timedelta(days=7)
    as_of = pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(days=1)
    report: dict[str, Any] = {"check_date": day.isoformat(), "window_start": w_start.isoformat(),
                              "window_end_exclusive": w_end.isoformat(), "label_maturity_days": maturity,
                              "min_events_for_status": MIN_EVENTS, "platforms": {}}
    for platform, ps in settings.platforms.items():
        horizon = pd.Timedelta(hours=float(ps.config["labels"]["horizon_hours"]))
        days = list(daterange(w_start, w_end - timedelta(days=1)))
        preds = [pd.read_parquet(settings.table_dir("predictions", platform, d) / "predictions.parquet")
                 for d in days if (settings.table_dir("predictions", platform, d) / "predictions.parquet").exists()]
        alerts = [pd.read_parquet(settings.table_dir("alerts", platform, d) / "alerts.parquet")
                  for d in days if (settings.table_dir("alerts", platform, d) / "alerts.parquet").exists()]
        entry: dict[str, Any] = {"prediction_days_found": len(preds), "prediction_days_expected": len(days)}
        if not preds:
            entry["status"] = "no predictions in window"
            report["platforms"][platform] = entry
            continue
        p = pd.concat(preds, ignore_index=True)
        a = pd.concat(alerts, ignore_index=True) if alerts else pd.DataFrame(columns=["unit_id", "t"])
        faults = confirmed_faults(settings, platform, day)
        events = label_events(faults, repairs_known(settings, platform, day), as_of)
        lab = early_warning_labels(p[["unit_id", "t"]], events, as_of, horizon / pd.Timedelta(hours=1),
                                   float(ps.config["labels"]["post_repair_exclusion_hours"]))
        scored = p.merge(lab[["unit_id", "t", "y", "eligible"]], on=["unit_id", "t"])
        scored = scored[scored["eligible"] & ~scored["insufficient_data"]]
        lo, hi = pd.Timestamp(w_start).tz_localize("UTC"), pd.Timestamp(w_end).tz_localize("UTC")
        in_window = events[(events["onset_utc"] > lo) & (events["onset_utc"] <= hi + horizon)]
        train = set(training_units.get(platform, []))
        for cohort, is_train in (("held_out_units", False), ("training_units", True)):
            sc = scored[scored["unit_id"].isin(train) == is_train]
            al = a[a["unit_id"].isin(train) == is_train] if len(a) else a
            ev = in_window[in_window["unit_id"].isin(train) == is_train]
            entry[cohort] = _cohort_metrics(sc, al, ev, horizon)
        entry["soh"] = soh_check(settings, platform, p, day)
        report["platforms"][platform] = entry
    target = settings.performance_dir(day)
    with atomic_partition(target) as tmp:
        (tmp / "performance.json").write_text(json.dumps(report, indent=2, default=_json), encoding="utf-8")
    return report


def soh_check(settings: Settings, platform: str, preds: pd.DataFrame, day: date) -> dict[str, Any]:
    """7-day SoH forecasts in the window whose target day has a capacity test by ``day``."""
    f = preds.dropna(subset=["soh_forecast_pct"])
    if f.empty:
        return {"forecasts": 0}
    tests = [pd.read_csv(settings.landing_dir("capacity_tests", platform, d) / "capacity_tests.csv")
             for d in settings.landing_dates("capacity_tests", platform) if d <= day]
    if not tests:
        return {"forecasts": len(f), "scored": 0}
    truth = pd.concat(tests, ignore_index=True)
    truth["target"] = pd.to_datetime(truth["date_utc"]).dt.tz_localize("UTC")
    m = f.assign(target=pd.to_datetime(f["soh_target_date"], utc=True)).merge(
        truth[["unit_id", "target", "soh_true_pct"]], on=["unit_id", "target"])
    if m.empty:
        return {"forecasts": len(f), "scored": 0}
    return {"forecasts": len(f), "scored": len(m),
            "mae_pp_model": mae(m["soh_true_pct"].to_numpy(), m["soh_forecast_pct"].to_numpy()),
            "mae_pp_persistence": mae(m["soh_true_pct"].to_numpy(), m["soh_now_pct"].to_numpy())}


def _json(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    return str(obj)
