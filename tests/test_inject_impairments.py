
import csv
from datetime import UTC, datetime, timedelta

import pytest

from fleet_simulator import FIELDS
from ingest_pipeline import run_pipeline
from inject_impairments import inject_impairments


@pytest.fixture
def source_csv(tmp_path):
    path = tmp_path / "source.csv"
    start = datetime(2026, 10, 26, tzinfo=UTC)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for unit in ("Rover_001", "Rover_002"):
            for minute in range(100):
                writer.writerow({
                    "timestamp_utc": (
                        start + timedelta(minutes=minute)
                    ).isoformat(),
                    "unit_id": unit,
                    "mode": "patrol",
                    "battery_soc_pct": 80,
                    "battery_soh_pct": 99.9,
                    "battery_voltage_v": 48,
                    "battery_current_a": 10,
                    "internal_resistance_ohm": 0.04,
                    "power_w": 480,
                    "ambient_temp_c": 25,
                    "motor_temp_c": 35,
                    "terrain_factor": 0.5,
                    "vibration_rms_g": 0.15,
                })

    return path


def test_dropout_detection(source_csv, tmp_path):
    output = tmp_path / "impaired"

    injection = inject_impairments(
        source_csv,
        output,
        dropout_rate=0.1,
        seed=43,
    )

    quality = run_pipeline(
        output / "fleet_telemetry.csv",
        out_root=tmp_path / "processed",
    )

    assert injection["dropped_rows"] > 0
    assert injection["input_rows"] == 200

    assert quality["total_rows"] == injection["output_rows"]
    assert quality["missing_minutes"] == injection["dropped_rows"]
    assert quality["rejected_rows"] == 0
    assert quality["quality_passed"] is False


def test_sensor_drift_and_labels(source_csv, tmp_path):
    output = tmp_path / "drift"

    report = inject_impairments(
        source_csv,
        output,
        dropout_rate=0,
        drift_c=4,
    )

    assert report["dropped_rows"] == 0
    assert report["drift_unit"] == "Rover_001"
    assert report["drifted_rows"] == 60

    with (output / "fleet_telemetry.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))

    rover_1 = [
        r for r in rows if r["unit_id"] == "Rover_001"
    ]
    rover_2 = [
        r for r in rows if r["unit_id"] == "Rover_002"
    ]

    assert float(rover_1[39]["motor_temp_c"]) == 35
    assert float(rover_1[40]["motor_temp_c"]) == 35
    assert float(rover_1[-1]["motor_temp_c"]) == pytest.approx(39)
    assert float(rover_2[-1]["motor_temp_c"]) == 35

    with (output / "sensor_drift_events.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        events = list(csv.DictReader(f))

    assert len(events) == 1
    assert events[0]["sensor"] == "motor_temp_c"
    assert float(events[0]["peak_bias_c"]) == 4


def test_reproducible_impairments(source_csv, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    inject_impairments(
        source_csv, first, dropout_rate=0.2, seed=43
    )
    inject_impairments(
        source_csv, second, dropout_rate=0.2, seed=43
    )

    assert (
        first / "fleet_telemetry.csv"
    ).read_bytes() == (
        second / "fleet_telemetry.csv"
    ).read_bytes()


def test_invalid_dropout_rate(source_csv, tmp_path):
    with pytest.raises(ValueError):
        inject_impairments(
            source_csv,
            tmp_path / "invalid",
            dropout_rate=1.0,
        )
