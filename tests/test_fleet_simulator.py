
import csv
import math
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import pairwise

import pytest

from fleet_simulator import simulate_fleet


@pytest.fixture(scope="module")
def fleet_data(tmp_path_factory):
    output = tmp_path_factory.mktemp("fleet_test")

    simulate_fleet(
        units=3,
        days=2,
        seed=42,
        output_dir=output,
    )

    with (output / "fleet_telemetry.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))

    with (output / "fleet_fault_events.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        faults = list(csv.DictReader(f))

    return rows, faults


def test_fleet_record_count(fleet_data):
    rows, _ = fleet_data
    assert len(rows) == 3 * 2 * 1440


def test_three_unique_robots(fleet_data):
    rows, _ = fleet_data
    ids = {r["unit_id"] for r in rows}
    assert ids == {"Rover_001", "Rover_002", "Rover_003"}


def test_unique_minute_timestamps(fleet_data):
    rows, _ = fleet_data
    by_unit = defaultdict(list)

    for row in rows:
        by_unit[row["unit_id"]].append(
            datetime.fromisoformat(row["timestamp_utc"])
        )

    for timestamps in by_unit.values():
        assert len(timestamps) == 2880
        assert len(set(timestamps)) == len(timestamps)
        assert all(
            b - a == timedelta(minutes=1)
            for a, b in pairwise(timestamps)
        )


def test_physical_value_ranges(fleet_data):
    rows, _ = fleet_data

    for row in rows:
        soc = float(row["battery_soc_pct"])
        soh = float(row["battery_soh_pct"])
        voltage = float(row["battery_voltage_v"])
        resistance = float(row["internal_resistance_ohm"])
        terrain = float(row["terrain_factor"])
        current = float(row["battery_current_a"])
        power = float(row["power_w"])

        assert all(math.isfinite(v) for v in (
            soc, soh, voltage, resistance,
            terrain, current, power
        ))

        assert 0 <= soc <= 100
        assert 0 < soh <= 100
        assert 30 <= voltage <= 60
        assert resistance > 0
        assert 0 <= terrain <= 1

        # Simulator uses a nominal 48V for current estimation.
        assert abs(power - current * 48) < 0.05


def test_battery_health_never_increases(fleet_data):
    rows, _ = fleet_data
    last_soh = {}

    for row in rows:
        unit = row["unit_id"]
        soh = float(row["battery_soh_pct"])

        if unit in last_soh:
            assert soh <= last_soh[unit]

        last_soh[unit] = soh


def test_fault_event_labels(fleet_data):
    _, faults = fleet_data

    assert len(faults) == 3

    for fault in faults:
        onset = datetime.fromisoformat(fault["onset_utc"])
        failure = datetime.fromisoformat(fault["failure_utc"])

        assert onset < failure
        assert fault["fault_type"] in {
            "bearing_wear",
            "cooling_degradation",
        }


def test_invalid_simulation_configuration(tmp_path):
    with pytest.raises(ValueError):
        simulate_fleet(
            units=0,
            days=2,
            output_dir=tmp_path,
        )
