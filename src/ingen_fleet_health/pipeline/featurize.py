"""Daily features: the hourly as-of rows of one date, from the validated history.

The Week 2 features use each unit's whole history (commissioning baseline over
the first 5 days, cumulative cycle counts, the thermal model state), so the
rows of date D are computed from every validated partition up to D and then
cut to D. Because every feature is trailing, this reproduces the Week 2
feature table exactly (tested), and re-running a date gives the same rows.

Cost grows with history (about 15 s per 10 units at day 60 on a laptop), so
units are split into contiguous chunks built in parallel processes (each
reads only its own units; units are independent, so the result is identical).
Carrying the feature state from day to day is a documented next step.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingen_fleet_health.features.build import build_feature_table
from ingen_fleet_health.pipeline.settings import (
    PartitionNotReady,
    Settings,
    atomic_partition,
    fingerprint,
    write_manifest,
)

TELEMETRY_COLUMNS = [
    "timestamp_utc", "unit_id", "mode", "battery_soc_pct", "battery_soh_pct", "battery_voltage_v",
    "battery_current_a", "internal_resistance_ohm", "power_w", "ambient_temp_c", "motor_temp_c",
    "terrain_factor", "vibration_rms_g", "odometer_m",
]


def fleet_start(settings: Settings, platform: str) -> pd.Timestamp:
    """Commissioning time of the fleet (the simulator start), from the reference data."""
    path = settings.data_root / "reference" / f"simulation_manifest_{platform}.json"
    return pd.Timestamp(json.loads(path.read_text(encoding="utf-8"))["start_utc"])


def history_dates(settings: Settings, platform: str, day: date) -> tuple[list[date], list[date]]:
    """(available, missing) validated dates from the fleet start up to ``day``."""
    first = fleet_start(settings, platform).date()
    have = set(settings.partition_dates("validated", platform))
    wanted = [first + timedelta(days=i) for i in range((day - first).days + 1)]
    return [d for d in wanted if d in have], [d for d in wanted if d not in have]


def load_maintenance(settings: Settings, platform: str, day: date) -> pd.DataFrame:
    """Repairs completed on or before ``day`` (partitioned by completion date)."""
    frames = [pd.read_csv(settings.landing_dir("maintenance", platform, d) / "maintenance.csv")
              for d in settings.landing_dates("maintenance", platform) if d <= day]
    cols = ["maintenance_id", "unit_id", "started_at_utc", "completed_at_utc", "component", "action"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def _build_chunk(args: tuple[list[Path], list[str], pd.DataFrame, dict[str, Any], pd.Timestamp, pd.Timestamp,
                                date]) -> pd.DataFrame:
    paths, units, maintenance, config, start, end, day = args
    telemetry = pd.concat([pd.read_parquet(p, columns=TELEMETRY_COLUMNS, filters=[("unit_id", "in", units)])
                           for p in paths], ignore_index=True)
    if telemetry.empty:
        return pd.DataFrame()
    table = build_feature_table(telemetry, maintenance, config, start, end,
                                int(config["labels"]["prediction_interval_hours"]),
                                int(config["labels"]["history_warmup_hours"]))
    return table[table["t"].dt.date == day]


def feature_workers(n_units: int) -> int:
    """``FLEET_FEATURE_WORKERS`` (default: CPU count, at most 8), never more than units / 2."""
    want = int(os.environ.get("FLEET_FEATURE_WORKERS", min(8, os.cpu_count() or 1)))
    return max(1, min(want, n_units // 2))


def features_for_day(settings: Settings, platform: str, day: date) -> dict[str, Any]:
    available, missing = history_dates(settings, platform, day)
    if day not in available:
        raise PartitionNotReady(f"{platform} {day}: no validated partition; run ingest first")
    paths = [settings.table_dir("validated", platform, d) / "telemetry.parquet" for d in available]
    units = sorted({str(u) for p in paths for u in pd.read_parquet(p, columns=["unit_id"])["unit_id"].unique()})
    maintenance = load_maintenance(settings, platform, day)
    config = settings.platforms[platform].config
    start = fleet_start(settings, platform)
    end = pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(days=1)
    workers = feature_workers(len(units))
    chunks = [list(c) for c in np.array_split(np.array(units, dtype=object), workers) if len(c)]
    jobs = [(paths, c, maintenance, config, start, end, day) for c in chunks]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_build_chunk, jobs))
    else:
        parts = [_build_chunk(j) for j in jobs]
    parts = [x for x in parts if len(x)]
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    table = table.sort_values(["unit_id", "t"], kind="stable").reset_index(drop=True)
    table.insert(1, "platform", platform)
    table["history_missing_days"] = len(missing)
    target = settings.table_dir("features", platform, day)
    with atomic_partition(target) as tmp:
        table.to_parquet(tmp / "features.parquet", index=False)
    maint_paths = [settings.landing_dir("maintenance", platform, d) / "maintenance.csv"
                   for d in settings.landing_dates("maintenance", platform) if d <= day]
    summary = {"platform": platform, "date": day.isoformat(), "rows": len(table),
               "units": int(table["unit_id"].nunique()) if len(table) else 0,
               "history_days_used": len(available), "history_missing_dates": [d.isoformat() for d in missing]}
    write_manifest(settings.run_manifest(platform, day, "features"), {
        "step": "features", "inputs": fingerprint(paths + maint_paths),
        "output": fingerprint([target / "features.parquet"]), "summary": summary})
    return summary


def read_features(settings: Settings, platform: str, days: list[date]) -> pd.DataFrame:
    frames = [pd.read_parquet(settings.table_dir("features", platform, d) / "features.parquet")
              for d in days if (settings.table_dir("features", platform, d) / "features.parquet").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
