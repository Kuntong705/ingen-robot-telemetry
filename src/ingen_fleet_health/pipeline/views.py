"""Read models for the fleet health dashboard (pure pandas, testable).

The dashboard answers one operator question: *which robots should be pulled
before the next shift, and why?* The decision rule is the pipeline's own,
nothing new is invented here:

* **Pull before next shift** -- the unit has an alert raised in the 24 h
  before the as-of time (new since the previous shift decision).
* **Alerted in last 72 h** -- an older alert is still inside its 72 h
  refractory window: the inspection should already be under way.
* **No recent data** -- the latest scored hour has too little telemetry
  (connectivity problem; check the robot's link, not its hardware).
* **OK** -- none of these.

With the default budget of one alert per robot-week, about one robot in seven
gets a new alert on an average day; the budget, not the dashboard, sets that rate.

Risk is the champion model's calibrated chance of a qualifying fault onset
within 72 h, learned on synthetic data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd

from ingen_fleet_health.pipeline.settings import Settings, daterange

PLATFORM_LABEL = {"aido_rover": "Aido Rover", "fari": "Fari"}
STATUS_ORDER = {"Pull before next shift": 0, "Alerted in last 72 h": 1, "No recent data": 2, "OK": 3}


def _read(settings: Settings, table: str, platforms: list[str], start: date, end: date, filename: str,
          columns: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for p in platforms:
        for d in daterange(start, end):
            path = settings.table_dir(table, p, d) / filename
            if path.exists():
                frames.append(pd.read_parquet(path, columns=columns))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def predictions(settings: Settings, platforms: list[str], start: date, end: date) -> pd.DataFrame:
    return _read(settings, "predictions", platforms, start, end, "predictions.parquet")


def alerts(settings: Settings, platforms: list[str], start: date, end: date) -> pd.DataFrame:
    return _read(settings, "alerts", platforms, start, end, "alerts.parquet")


def units(settings: Settings, platforms: list[str]) -> pd.DataFrame:
    frames = []
    for p in platforms:
        path = settings.data_root / "reference" / f"units_{p}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["unit_id", "platform", "site"])


def scored_dates(settings: Settings, platforms: list[str]) -> list[date]:
    out: set[date] = set()
    for p in platforms:
        out |= set(settings.partition_dates("predictions", p))
    return sorted(out)


@dataclass
class Freshness:
    platform: str
    last_scored: date | None
    missing_dates: list[date]
    failures: list[dict[str, Any]]


def freshness(settings: Settings, platforms: list[str], start: date, end: date) -> list[Freshness]:
    """Scored dates, gaps and recorded failures per platform in the window, so a
    missing day is shown as missing rather than silently filled with old scores."""
    out = []
    fail_dir = settings.data_root / "runs" / "failures"
    failures = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(fail_dir.glob("*.json"))] if fail_dir.exists() else []
    for p in platforms:
        have = set(settings.partition_dates("predictions", p))
        last = max((d for d in have if d <= end), default=None)
        missing = [d for d in daterange(start, end) if d not in have and (last is None or d <= last)]
        fl = [f for f in failures if f.get("task", "").startswith(f"{p}.") and start.isoformat() <= f["date"] <= end.isoformat()]
        out.append(Freshness(p, last, missing, fl))
    return out


def top_reason(drivers_json: str) -> str:
    drivers = json.loads(drivers_json) if isinstance(drivers_json, str) and drivers_json else []
    return drivers[0]["phrase"] if drivers else ""


def pull_list(preds: pd.DataFrame, alert_frame: pd.DataFrame, unit_ref: pd.DataFrame, as_of: pd.Timestamp,
              open_hours: float = 72.0, pull_hours: float = 24.0) -> pd.DataFrame:
    """One row per unit at the as-of time, ranked for the shift decision."""
    if preds.empty:
        return pd.DataFrame()
    p = preds[preds["t"] <= as_of].sort_values("t")
    latest = p.groupby("unit_id").tail(1).set_index("unit_id")
    day_ago = p[p["t"] <= as_of - pd.Timedelta(hours=24)].groupby("unit_id").tail(1).set_index("unit_id")["risk"]
    soh = p.dropna(subset=["soh_forecast_pct"]).groupby("unit_id").tail(1).set_index("unit_id")
    a = alert_frame[(alert_frame["t"] <= as_of) & (alert_frame["t"] > as_of - pd.Timedelta(hours=open_hours))] \
        if not alert_frame.empty else alert_frame
    open_alert = a.sort_values("t").groupby("unit_id").tail(1).set_index("unit_id") if not a.empty else pd.DataFrame()
    rows = []
    for unit, r in latest.iterrows():
        has_alert = not open_alert.empty and unit in open_alert.index
        if has_alert and open_alert.loc[unit, "t"] > as_of - pd.Timedelta(hours=pull_hours):
            status = "Pull before next shift"
        elif has_alert:
            status = "Alerted in last 72 h"
        elif bool(r["insufficient_data"]):
            status = "No recent data"
        else:
            status = "OK"
        rows.append({
            "unit_id": unit, "platform": PLATFORM_LABEL.get(r["platform"], r["platform"]), "status": status,
            "risk_now_pct": 100 * float(r["risk"]), "threshold_pct": 100 * float(r["threshold"]),
            "risk_24h_ago_pct": 100 * float(day_ago[unit]) if unit in day_ago.index else float("nan"),
            "alert_time": open_alert.loc[unit, "t"] if has_alert else pd.NaT,
            "why": open_alert.loc[unit, "summary"] if has_alert else "",
            "main_driver": top_reason(open_alert.loc[unit, "drivers"]) if has_alert else "",
            "battery_soh_pct": float(soh.loc[unit, "soh_now_pct"]) if unit in soh.index else float("nan"),
            "battery_soh_7d_forecast_pct": float(soh.loc[unit, "soh_forecast_pct"]) if unit in soh.index else float("nan"),
            "last_scored": r["t"],
        })
    out = pd.DataFrame(rows)
    if not unit_ref.empty and "site" in unit_ref.columns:
        out = out.merge(unit_ref[["unit_id", "site"]], on="unit_id", how="left")
    out["_order"] = out["status"].map(STATUS_ORDER)
    out = out.sort_values(["_order", "risk_now_pct"], ascending=[True, False]).drop(columns="_order")
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)


def telemetry(settings: Settings, platform: str, unit_id: str, start: date, end: date,
              resample: str = "10min") -> pd.DataFrame:
    """Validated minute telemetry of one unit, averaged to ``resample`` for display."""
    frames = []
    for d in daterange(start, end):
        path = settings.table_dir("validated", platform, d) / "telemetry.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path, filters=[("unit_id", "==", unit_id)]))
    if not frames:
        return pd.DataFrame()
    t = pd.concat(frames, ignore_index=True).set_index("timestamp_utc").sort_index()
    numeric = t.select_dtypes("number")
    return numeric.resample(resample).mean().reset_index()


def drift(settings: Settings, platforms: list[str], start: date, end: date) -> pd.DataFrame:
    frames = []
    for p in platforms:
        for d in daterange(start, end):
            path = settings.drift_dir(p, d) / "psi.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def performance_reports(settings: Settings, start: date, end: date) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    root = settings.data_root / "monitoring" / "performance"
    if not root.exists():
        return out
    for path in sorted(root.glob("date=*/performance.json")):
        d = date.fromisoformat(path.parent.name.removeprefix("date="))
        if start <= d <= end + timedelta(days=0):
            out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def quality(settings: Settings, platforms: list[str], start: date, end: date) -> pd.DataFrame:
    rows = []
    for p in platforms:
        for d in daterange(start, end):
            path = settings.table_dir("validated", p, d) / "quality_report.json"
            if path.exists():
                r = json.loads(path.read_text(encoding="utf-8"))
                rows.append({"platform": PLATFORM_LABEL.get(p, p), "date": d, "accepted_rows": r["accepted_rows"],
                             "rejected_rows": r["rejected_rows"], "units_reporting": r["units_reporting"],
                             "units_missing": len(r["units_missing"]), "gap_events": r["timestamp_gap_events"]})
    return pd.DataFrame(rows)
