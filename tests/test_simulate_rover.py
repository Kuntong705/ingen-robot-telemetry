
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "simulate_rover.py"


@pytest.fixture(scope="module")
def simulation(tmp_path_factory):
    # Run in a temporary folder, not the actual project data folder
    workdir = tmp_path_factory.mktemp("simulation")

    subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
    )

    output = workdir / "data" / "generated"

    with (output / "telemetry.csv").open(newline="") as f:
        telemetry = list(csv.DictReader(f))

    with (output / "fault_events.csv").open(newline="") as f:
        faults = list(csv.DictReader(f))

    with (output / "quality_report.json").open() as f:
        report = json.load(f)

    return telemetry, faults, report


def test_row_count(simulation):
    telemetry, _, _ = simulation
    assert len(telemetry) == 1440


def test_timestamps_are_unique_and_ordered(simulation):
    telemetry, _, _ = simulation
    timestamps = [
        datetime.fromisoformat(row["timestamp_utc"])
        for row in telemetry
    ]
    assert timestamps == sorted(timestamps)
    assert len(timestamps) == len(set(timestamps))


def test_battery_soc_range(simulation):
    telemetry, _, _ = simulation
    assert all(
        0 <= float(row["battery_soc_pct"]) <= 100
        for row in telemetry
    )


def test_vibration_nonnegative(simulation):
    telemetry, _, _ = simulation
    assert all(
        float(row["vibration_rms_g"]) >= 0
        for row in telemetry
    )


def test_fault_labels(simulation):
    _, faults, _ = simulation
    assert len(faults) == 1
    assert faults[0]["fault_type"] == "bearing_wear"
    assert (
        datetime.fromisoformat(faults[0]["onset_utc"])
        < datetime.fromisoformat(faults[0]["failure_utc"])
    )


def test_quality_report(simulation):
    _, _, report = simulation
    assert report["passed"] is True
    assert report["row_count"] == 1440
