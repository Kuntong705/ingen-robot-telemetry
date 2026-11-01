
import argparse
import csv
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

# All parameters are illustrative, not real hardware specifications.
BASE_CAPACITY_WH = 6000.0
BASE_RESISTANCE_OHM = 0.04
START = datetime(2026, 10, 26, tzinfo=UTC)

FIELDS = [
    "timestamp_utc", "unit_id", "mode",
    "battery_soc_pct", "battery_soh_pct",
    "battery_voltage_v", "battery_current_a",
    "internal_resistance_ohm", "power_w",
    "ambient_temp_c", "motor_temp_c",
    "terrain_factor", "vibration_rms_g",
]


def simulate_fleet(units=3, days=2, seed=42, output_dir=None):
    if units < 1 or days < 1:
        raise ValueError("Units and days must be positive")

    total_minutes = days * 1440
    output = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parent / "data" / "generated"
    )
    output.mkdir(parents=True, exist_ok=True)

    telemetry_path = output / "fleet_telemetry.csv"
    faults_path = output / "fleet_fault_events.csv"

    with (
        telemetry_path.open("w", newline="", encoding="utf-8") as tf,
        faults_path.open("w", newline="", encoding="utf-8") as ff,
    ):
        writer = csv.DictWriter(tf, fieldnames=FIELDS)
        writer.writeheader()

        fault_writer = csv.writer(ff)
        fault_writer.writerow([
            "unit_id", "fault_type", "onset_utc", "failure_utc"
        ])

        for index in range(units):
            unit_id = f"Rover_{index + 1:03d}"
            rng = random.Random(seed + index)

            soc = 95.0
            motor_temp = 25.0
            equivalent_cycles = 0.0

            fault_type = (
                "bearing_wear" if index % 2 == 0
                else "cooling_degradation"
            )

            onset = int(total_minutes * 0.40)
            failure = int(total_minutes * 0.80)

            fault_writer.writerow([
                unit_id,
                fault_type,
                (START + timedelta(minutes=onset)).isoformat(),
                (START + timedelta(minutes=failure)).isoformat(),
            ])

            for minute in range(total_minutes):
                timestamp = START + timedelta(minutes=minute)
                hour = (minute // 60) % 24
                patrol = 8 <= hour < 16
                mode = "patrol" if patrol else "dock"

                ambient = 24 + 5 * math.sin(
                    2 * math.pi * (hour - 6) / 24
                )

                terrain = 0.5 + 0.5 * math.sin(
                    2 * math.pi * minute / 90
                )

                power = (
                    400 + 120 * terrain if patrol
                    else (-700 if soc < 95 else 0)
                )

                # Simplified cycle-dependent battery degradation
                soh = max(70.0, 100 - 0.04 * equivalent_cycles)
                effective_capacity = BASE_CAPACITY_WH * soh / 100

                soc -= power / 60 / effective_capacity * 100
                soc = max(0, min(100, soc))

                if power > 0:
                    equivalent_cycles += power / 60 / BASE_CAPACITY_WH

                resistance = BASE_RESISTANCE_OHM * (
                    1 + 0.001 * equivalent_cycles
                )

                current = power / 48.0
                voltage = (
                    48 + (soc - 50) * 0.05
                    - current * resistance
                )

                # Fault develops gradually, not instantaneously
                severity = max(
                    0.0,
                    min(1.0, (minute - onset) / (failure - onset))
                )

                extra_heat = (
                    14 * severity
                    if fault_type == "cooling_degradation" else 0
                )

                target_temp = (
                    ambient + 0.028 * max(power, 0) + extra_heat
                )
                motor_temp += (target_temp - motor_temp) / 40

                vibration = 0.04
                if patrol:
                    vibration = 0.15
                    if fault_type == "bearing_wear":
                        vibration += 0.5 * severity

                vibration = max(
                    0.0, vibration + rng.gauss(0, 0.01)
                )

                writer.writerow({
                    "timestamp_utc": timestamp.isoformat(),
                    "unit_id": unit_id,
                    "mode": mode,
                    "battery_soc_pct": round(soc, 3),
                    "battery_soh_pct": round(soh, 4),
                    "battery_voltage_v": round(voltage, 3),
                    "battery_current_a": round(current, 3),
                    "internal_resistance_ohm": round(resistance, 6),
                    "power_w": round(power, 2),
                    "ambient_temp_c": round(ambient, 2),
                    "motor_temp_c": round(motor_temp, 2),
                    "terrain_factor": round(terrain, 3),
                    "vibration_rms_g": round(vibration, 4),
                })

    print(f"Fleet simulation complete: {units} units x {days} days")
    print(f"Telemetry rows: {units * total_minutes:,}")
    print(f"Fault events: {units}")
    print(f"Output: {telemetry_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=int, default=3)
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    simulate_fleet(args.units, args.days, args.seed, args.output_dir)
