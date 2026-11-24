"""Simulated upstream: publish a simulated fleet into daily landing partitions.

In a deployment the robots' gateways would drop one telemetry file per
platform and day into the landing zone. Here the simulator's output plays
that role. ``publish`` splits it by UTC date:

* ``telemetry``      raw minute records, unchanged strings (CSV, validated later);
* ``maintenance``    observable repairs, by completion date;
* ``labels``         fault confirmations, by confirmation date: a fault's onset
                     time becomes known when a technician starts the repair
                     (``repair_start_utc``); before that the pipeline cannot know it;
* ``capacity_tests`` true end-of-day SoH (stand-in for periodic capacity tests).

``--withhold`` drops chosen dates (a late or missing upstream partition), so
the pipeline's documented behaviour can be exercised. The simulator ground
truth for faults is never published beyond the confirmation fields.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from ingen_fleet_health.pipeline.settings import Settings, load_settings


def _sql(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def _in_range(d: date, start: date | None, end: date | None) -> bool:
    return (start is None or d >= start) and (end is None or d <= end)


def publish_telemetry(source_csv: Path, settings: Settings, platform: str, start: date | None = None,
                      end: date | None = None, withhold: set[date] | None = None) -> list[date]:
    """Split the raw CSV into ``landing/telemetry/platform=<p>/date=<d>/telemetry.csv``."""
    withhold = withhold or set()
    published: list[date] = []
    with tempfile.TemporaryDirectory(dir=settings.data_root.parent if settings.data_root.parent.exists() else None) as tmp:
        stage = Path(tmp) / "split"
        con = duckdb.connect()
        try:
            con.execute(f"""
                COPY (
                    SELECT *, substr(timestamp_utc, 1, 10) AS day
                    FROM read_csv('{_sql(source_csv)}', header=true, all_varchar=true)
                    ORDER BY unit_id, timestamp_utc
                ) TO '{_sql(stage)}' (FORMAT CSV, HEADER true, PARTITION_BY (day))
            """)
        finally:
            con.close()
        for part in sorted(stage.glob("day=*")):
            d = date.fromisoformat(part.name.removeprefix("day="))
            if not _in_range(d, start, end) or d in withhold:
                continue
            target = settings.landing_dir("telemetry", platform, d)
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)
            files = sorted(part.glob("*.csv"))
            frames = [pd.read_csv(f, dtype=str, keep_default_na=False) for f in files]
            pd.concat(frames, ignore_index=True).sort_values(["unit_id", "timestamp_utc"]).to_csv(
                target / "telemetry.csv", index=False)
            published.append(d)
    return published


def _publish_by_date(frame: pd.DataFrame, when: pd.Series, settings: Settings, kind: str, platform: str,
                     filename: str, start: date | None, end: date | None, withhold: set[date]) -> list[date]:
    out = []
    days = pd.to_datetime(when, utc=True).dt.date
    for d, part in frame.groupby(days.to_numpy(), sort=True):
        if not _in_range(d, start, end) or d in withhold:
            continue
        target = settings.landing_dir(kind, platform, d)
        target.mkdir(parents=True, exist_ok=True)
        part.to_csv(target / filename, index=False)
        out.append(d)
    return out


def publish_platform(settings: Settings, platform: str, start: date | None = None, end: date | None = None,
                     withhold: set[date] | None = None, telemetry: bool = True) -> dict[str, object]:
    withhold = withhold or set()
    src = settings.platforms[platform].source_dir
    summary: dict[str, object] = {"platform": platform, "source_dir": str(src)}
    if telemetry:
        summary["telemetry_dates"] = [d.isoformat() for d in publish_telemetry(
            src / "generated" / "fleet_telemetry.csv", settings, platform, start, end, withhold)]
    maint = pd.read_csv(src / "events" / "maintenance_events.csv")
    summary["maintenance_dates"] = [d.isoformat() for d in _publish_by_date(
        maint, maint["completed_at_utc"], settings, "maintenance", platform, "maintenance.csv", start, end, withhold)]
    faults = pd.read_csv(src / "ground_truth" / "fault_events.csv")
    faults = faults[faults["onset_within_run"].astype(str).str.lower() == "true"]
    confirm = faults[["fault_id", "unit_id", "fault_type", "onset_utc"]].assign(
        confirmed_at_utc=faults["repair_start_utc"])
    summary["label_dates"] = [d.isoformat() for d in _publish_by_date(
        confirm, confirm["confirmed_at_utc"], settings, "labels", platform, "fault_confirmations.csv",
        start, end, withhold)]
    truth = pd.read_csv(src / "ground_truth" / "battery_truth_daily.csv")
    summary["capacity_test_dates"] = len(_publish_by_date(
        truth[["unit_id", "date_utc", "soh_true_pct"]], truth["date_utc"], settings, "capacity_tests", platform,
        "capacity_tests.csv", start, end, withhold))
    ref = settings.data_root / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "events" / "units.csv", ref / f"units_{platform}.csv")
    shutil.copy2(src / "simulation_manifest.json", ref / f"simulation_manifest_{platform}.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish simulated fleets into daily landing partitions")
    parser.add_argument("--config", type=Path, help="pipeline YAML (default configs/pipeline.yaml)")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--platform", action="append", help="default: every platform in the config")
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--withhold", type=date.fromisoformat, action="append", default=[],
                        help="date to leave out (simulates a late/missing upstream partition); repeatable")
    args = parser.parse_args()
    settings = load_settings(args.config, args.data_root)
    for platform in args.platform or list(settings.platforms):
        summary = publish_platform(settings, platform, args.start, args.end, set(args.withhold))
        print(json.dumps({k: (v if not isinstance(v, list) else f"{len(v)} dates") for k, v in summary.items()}))


if __name__ == "__main__":
    main()
