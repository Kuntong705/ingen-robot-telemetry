"""Daily ingestion: one landing partition -> validated + quarantine + quality report.

Reuses the Week 1 row rules (``ingest_pipeline.check_row``) with the
platform profile from Week 3, so the daily pipeline and the Week 1 batch
pipeline accept and reject exactly the same records. The only addition is
continuity across days: the last accepted timestamp and odometer of each unit
are carried over from the most recent earlier validated partition, so a
duplicate or a backwards odometer at midnight is still caught.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ingen_fleet_health.config import REPO_ROOT
from ingen_fleet_health.contract import assert_no_forbidden_fields, platform_profile
from ingen_fleet_health.pipeline.settings import (
    PartitionNotReady,
    Settings,
    atomic_partition,
    fingerprint,
    write_manifest,
)

if str(REPO_ROOT) not in sys.path:      # the Week 1 scripts live at the repository root
    sys.path.insert(0, str(REPO_ROOT))
from fleet_simulator import FIELDS
from ingest_pipeline import NUMERIC_FIELDS, check_row


def landing_file(settings: Settings, platform: str, day: date) -> Path:
    path = settings.landing_dir("telemetry", platform, day) / "telemetry.csv"
    if not path.exists():
        raise PartitionNotReady(f"{platform} {day}: landing partition {path} not found")
    return path


def previous_state(settings: Settings, platform: str, day: date) -> tuple[dict[str, datetime], dict[str, float], date | None]:
    """Last accepted timestamp and odometer per unit from the latest validated
    partition before ``day`` (as-of: nothing from ``day`` or later)."""
    earlier = [d for d in settings.partition_dates("validated", platform) if d < day]
    if not earlier:
        return {}, {}, None
    last = earlier[-1]
    frame = pd.read_parquet(settings.table_dir("validated", platform, last) / "telemetry.parquet",
                            columns=["unit_id", "timestamp_utc", "odometer_m"])
    tail = frame.sort_values("timestamp_utc").groupby("unit_id").tail(1)
    ts = {str(u): pd.Timestamp(t).to_pydatetime() for u, t in zip(tail["unit_id"], tail["timestamp_utc"], strict=True)}
    odo = {str(u): float(o) for u, o in zip(tail["unit_id"], tail["odometer_m"], strict=True)}
    return ts, odo, last


def expected_units(settings: Settings, platform: str) -> list[str]:
    path = settings.data_root / "reference" / f"units_{platform}.csv"
    return sorted(pd.read_csv(path)["unit_id"].astype(str)) if path.exists() else []


def ingest_day(settings: Settings, platform: str, day: date) -> dict[str, Any]:
    source = landing_file(settings, platform, day)
    profile = platform_profile(settings.platforms[platform].config)
    last_ts, last_odo, prev_day = previous_state(settings, platform, day)
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    reasons: Counter[str] = Counter()
    gap_events = 0
    missing_minutes = 0
    with source.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        assert_no_forbidden_fields(header, profile)            # privacy gate before anything is stored
        missing_cols = sorted(set(FIELDS) - set(header))
        if missing_cols:
            raise ValueError(f"{platform} {day}: landing file is missing columns {missing_cols}")
        for row in reader:
            problems, timestamp, unit = check_row(row, last_ts, last_odo, profile)
            if None in row:
                problems.append("extra_columns")
            if timestamp is not None and timestamp.date() != day:
                problems.append("wrong_partition_date")
            if problems:
                reasons.update(problems)
                rejected.append({**{k: row.get(k) or "" for k in FIELDS}, "error_codes": ";".join(problems)})
                continue
            prev = last_ts.get(unit)
            if prev is not None and timestamp is not None and timestamp - prev > timedelta(minutes=1):
                gap_events += 1
                missing_minutes += int((timestamp - prev).total_seconds() // 60) - 1
            assert timestamp is not None
            last_ts[unit] = timestamp
            last_odo[unit] = float(row["odometer_m"])
            accepted.append({k: row[k] for k in FIELDS})

    valid = pd.DataFrame(accepted, columns=FIELDS)
    valid["timestamp_utc"] = pd.to_datetime(valid["timestamp_utc"], utc=True).astype("datetime64[us, UTC]")
    for col in NUMERIC_FIELDS:
        valid[col] = valid[col].astype("float64")
    units_seen = sorted(valid["unit_id"].unique().tolist())
    expected = expected_units(settings, platform)
    report = {
        "platform": platform,
        "date": day.isoformat(),
        "dataset": "synthetic_fleet_telemetry",
        "total_rows": len(accepted) + len(rejected),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(sorted(reasons.items())),
        "timestamp_gap_events": gap_events,
        "missing_minutes_vs_previous_record": missing_minutes,
        "units_reporting": len(units_seen),
        "units_missing": sorted(set(expected) - set(units_seen)),
        "carried_state_from": prev_day.isoformat() if prev_day else None,
        "quality_passed": not rejected and not set(expected) - set(units_seen),
    }
    with atomic_partition(settings.table_dir("validated", platform, day)) as tmp:
        valid.to_parquet(tmp / "telemetry.parquet", index=False)
        pd.Series(report).to_json(tmp / "quality_report.json", indent=2)
    with atomic_partition(settings.table_dir("quarantine", platform, day)) as tmp:
        pd.DataFrame(rejected, columns=[*FIELDS, "error_codes"]).astype(str).to_parquet(
            tmp / "rejected.parquet", index=False)
    out = settings.table_dir("validated", platform, day) / "telemetry.parquet"
    inputs = [source]
    if prev_day is not None:      # carried-over state is an input too
        inputs.append(settings.table_dir("validated", platform, prev_day) / "telemetry.parquet")
    write_manifest(settings.run_manifest(platform, day, "ingest"), {
        "step": "ingest", "inputs": fingerprint(inputs), "output": fingerprint([out]), "report": report})
    return report
