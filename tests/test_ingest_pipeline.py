
import csv

import duckdb
import pytest

from fleet_simulator import FIELDS, simulate_fleet
from ingest_pipeline import run_pipeline


def make_csv(path, rows, fields=None):
    fields = fields if fields is not None else FIELDS

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                field: row.get(field, "")
                for field in fields
            })

    return path


def valid_row(timestamp="2026-10-26T08:00:00+00:00"):
    return {
        "timestamp_utc": timestamp,
        "unit_id": "Rover_001",
        "mode": "patrol",
        "battery_soc_pct": "80",
        "battery_soh_pct": "99.9",
        "battery_voltage_v": "48",
        "battery_current_a": "10",
        "internal_resistance_ohm": "0.04",
        "power_w": "480",
        "ambient_temp_c": "25",
        "motor_temp_c": "35",
        "terrain_factor": "0.5",
        "vibration_rms_g": "0.15",
    }


def parquet_count(path):
    con = duckdb.connect()
    try:
        result = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(path)],
        ).fetchone()
        return result[0]
    finally:
        con.close()


def test_clean_fleet_pipeline(tmp_path):
    source_dir = tmp_path / "generated"

    simulate_fleet(
        units=1,
        days=1,
        seed=42,
        output_dir=source_dir,
    )

    output = tmp_path / "processed"

    report = run_pipeline(
        source_dir / "fleet_telemetry.csv",
        out_root=output,
    )

    assert report["quality_passed"] is True
    assert report["accepted_rows"] == 1440
    assert report["rejected_rows"] == 0

    validated = output / "data/validated/telemetry.parquet"
    curated = output / "data/curated/daily_summary.parquet"

    assert validated.exists()
    assert curated.exists()
    assert parquet_count(validated) == 1440
    assert parquet_count(curated) == 1


def test_corruption_quarantine_and_gaps(tmp_path):
    rows = [
        valid_row(),
        {
            **valid_row("2026-10-26T08:01:00+00:00"),
            "battery_soc_pct": "150",
        },
        valid_row("2026-10-26T08:03:00+00:00"),
        valid_row("2026-10-26T08:03:00+00:00"),
    ]

    source = make_csv(tmp_path / "corrupt.csv", rows)
    output = tmp_path / "processed"

    report = run_pipeline(source, out_root=output)

    assert report["quality_passed"] is False
    assert report["total_rows"] == 4
    assert report["accepted_rows"] == 2
    assert report["rejected_rows"] == 2
    assert report["timestamp_gap_events"] == 1
    assert report["missing_minutes"] == 2

    assert report["rejection_reasons"][
        "out_of_range_battery_soc_pct"
    ] == 1

    assert report["rejection_reasons"][
        "duplicate_or_unordered_timestamp"
    ] == 1

    validated = output / "data/validated/telemetry.parquet"
    assert parquet_count(validated) == 2

    rejected = output / "data/quarantine/rejected.csv"
    with rejected.open(newline="", encoding="utf-8") as f:
        quarantined = list(csv.DictReader(f))

    assert len(quarantined) == 2


def test_missing_columns_rejected(tmp_path):
    source = make_csv(
        tmp_path / "missing.csv",
        [valid_row()],
        fields=["unit_id", "timestamp_utc"],
    )

    with pytest.raises(ValueError, match="Missing columns"):
        run_pipeline(source, out_root=tmp_path / "processed")


def test_simulation_reproducible(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    simulate_fleet(
        units=2, days=1, seed=42, output_dir=first
    )
    simulate_fleet(
        units=2, days=1, seed=42, output_dir=second
    )

    assert (
        first / "fleet_telemetry.csv"
    ).read_bytes() == (
        second / "fleet_telemetry.csv"
    ).read_bytes()
