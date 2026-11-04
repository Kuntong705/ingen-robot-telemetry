"""Build the hourly as-of feature table from validated minute telemetry.

Input:  validated telemetry Parquet (Week 1 pipeline output) + observable
        maintenance events. Ground-truth tables are never read here.
Output: one row per (unit_id, t) at each hourly prediction time t; every
        feature uses telemetry with timestamp <= t only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingen_fleet_health.config import load_config
from ingen_fleet_health.features.physics import physics_features
from ingen_fleet_health.features.rolling import rolling_block

ROLLING_CHANNELS = {
    "motor_temp_c": "motor_temp",
    "vibration_rms_g": "vibration",
    "power_w": "power",
    "battery_voltage_v": "voltage",
    "battery_soc_pct": "soc",
    "ambient_temp_c": "ambient",
    "terrain_factor": "terrain",
    "internal_resistance_ohm": "r_bms",
}
KEYS = ["unit_id", "t"]


def minute_grid(unit_frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Reindex one unit to every minute in [start, end); gaps become NaN rows."""
    idx = pd.date_range(start, end, freq="min", inclusive="left", tz="UTC")
    return unit_frame.set_index("timestamp_utc").reindex(idx)


def unit_features(
    unit_frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    prediction_times: pd.DatetimeIndex,
    config: dict[str, Any],
) -> pd.DataFrame:
    grid = minute_grid(unit_frame, start, end)
    cols: dict[str, np.ndarray] = {}
    for channel, prefix in ROLLING_CHANNELS.items():
        cols.update(rolling_block(grid[channel].astype("float64"), prefix))
    cols.update(physics_features(grid.reset_index(drop=True), config))
    full = pd.DataFrame(cols, index=grid.index)
    # sample the minute grid at prediction times: row t summarises minutes <= t
    sampled = full.reindex(prediction_times)
    sampled.index.name = "t"
    return sampled.reset_index()


def hours_since_last_repair(
    features: pd.DataFrame, maintenance: pd.DataFrame, cap_hours: float = 24 * 60
) -> np.ndarray:
    """As-of: only repairs whose *completion* time is <= t are visible."""
    out = np.full(len(features), cap_hours)
    if maintenance.empty:
        return out
    done = maintenance.assign(completed=pd.to_datetime(maintenance["completed_at_utc"], utc=True))
    for unit, idx in features.groupby("unit_id").groups.items():
        times = done.loc[done["unit_id"] == unit, "completed"].sort_values().to_numpy()
        if len(times) == 0:
            continue
        t = features.loc[idx, "t"].to_numpy()
        pos = np.searchsorted(times, t, side="right") - 1
        has = pos >= 0
        elapsed = (t[has] - times[pos[has]]) / np.timedelta64(1, "h")
        vals = out[np.asarray(idx)]
        vals[has] = np.minimum(elapsed, cap_hours)
        out[np.asarray(idx)] = vals
    return out


def build_feature_table(
    telemetry: pd.DataFrame,
    maintenance: pd.DataFrame,
    config: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval_hours: int = 1,
    warmup_hours: int = 24,
) -> pd.DataFrame:
    telemetry = telemetry.sort_values(["unit_id", "timestamp_utc"])
    times = pd.date_range(start + pd.Timedelta(hours=warmup_hours), end, freq=f"{interval_hours}h", tz="UTC")
    # the minute grid ends at the last observed minute; t == end summarises up to end - 1 min
    times = times[times < end] if len(times) and times[-1] >= end else times
    parts = []
    for unit, frame in telemetry.groupby("unit_id", sort=True):
        part = unit_features(frame, start, end, times, config)
        part.insert(0, "unit_id", unit)
        parts.append(part)
    table = pd.concat(parts, ignore_index=True)
    table["hours_since_repair"] = hours_since_last_repair(table, maintenance)
    table["hour_of_day"] = table["t"].dt.hour.astype(float)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hourly as-of features")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    data_dir = args.data_dir
    telemetry = pd.read_parquet(data_dir / "data" / "validated" / "telemetry.parquet")
    maintenance = pd.read_csv(data_dir / "events" / "maintenance_events.csv")
    manifest = json.loads((data_dir / "simulation_manifest.json").read_text(encoding="utf-8"))
    start = pd.Timestamp(manifest["start_utc"])
    end = pd.Timestamp(manifest["end_utc"])
    table = build_feature_table(telemetry, maintenance, config, start, end)
    out = data_dir / "features" / "features_hourly.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    print(f"features: {table.shape[0]:,} rows x {table.shape[1]} columns -> {out}")


if __name__ == "__main__":
    main()
