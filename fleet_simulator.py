import argparse
import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "aido_rover.yaml"

FIELDS = [
    "timestamp_utc",
    "unit_id",
    "mode",
    "battery_soc_pct",
    "battery_soh_pct",
    "battery_voltage_v",
    "battery_current_a",
    "internal_resistance_ohm",
    "power_w",
    "ambient_temp_c",
    "motor_temp_c",
    "terrain_factor",
    "vibration_rms_g",
    "odometer_m",
]


def load_config(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(config, dict):
        raise TypeError("Invalid YAML configuration")

    return config


def simulate_fleet(
    units=None,
    days=None,
    seed=None,
    output_dir=None,
    config_path=None,
):
    config = load_config(config_path)

    sim = config["simulation"]
    battery = config["battery"]
    duty = config["duty_cycle"]
    patrol_speed_mps = float(duty["patrol_speed_mps"])
    if patrol_speed_mps < 0:
        raise ValueError("Patrol speed must be nonnegative")
    environment = config["environment"]
    thermal = config["thermal"]
    vibration_cfg = config["vibration"]
    faults = config["fault_injection"]

    units = int(sim["units"]) if units is None else units
    days = int(sim["days"]) if days is None else days
    seed = int(sim["seed"]) if seed is None else seed

    if units < 1 or days < 1:
        raise ValueError("Units and days must be positive")

    interval = int(sim["sample_interval_seconds"])
    if interval != 60:
        raise ValueError("Only 60-second sampling is supported")

    total_minutes = days * 1440

    start = datetime.fromisoformat(sim["start_utc"])
    if start.tzinfo is None or start.utcoffset() != timedelta(0):
        raise ValueError("Simulation start must be UTC")

    capacity_wh = float(battery["nominal_capacity_wh"])
    voltage_nominal = float(battery["nominal_voltage_v"])
    initial_soc = float(battery["initial_soc_pct"])
    initial_soh = float(battery["initial_soh_pct"])
    min_soh = float(battery["min_soh_pct"])
    fade_rate = float(battery["capacity_fade_pct_per_efc"])
    base_resistance = float(
        battery["base_internal_resistance_ohm"]
    )
    resistance_growth = float(
        battery["resistance_growth_fraction_per_efc"]
    )
    ocv_slope = float(
        battery.get("ocv_soc_slope_v_per_pct", 0.05)
    )

    response_time = float(thermal["response_time_minutes"])
    terrain_period = float(
        environment["terrain_cycle_minutes"]
    )
    onset_fraction = float(faults["onset_fraction"])
    failure_fraction = float(faults["failure_fraction"])

    if (
        capacity_wh <= 0
        or voltage_nominal <= 0
        or base_resistance <= 0
        or response_time <= 0
        or terrain_period <= 0
    ):
        raise ValueError("Physical parameters must be positive")

    if not 0 <= onset_fraction < failure_fraction < 1:
        raise ValueError("Invalid fault timing fractions")

    onset = int(total_minutes * onset_fraction)
    failure = int(total_minutes * failure_fraction)

    output = (
        Path(output_dir)
        if output_dir is not None
        else ROOT / "data" / "generated"
    )
    output.mkdir(parents=True, exist_ok=True)

    telemetry_path = output / "fleet_telemetry.csv"
    faults_path = output / "fleet_fault_events.csv"

    with (
        telemetry_path.open(
            "w", newline="", encoding="utf-8"
        ) as tf,
        faults_path.open(
            "w", newline="", encoding="utf-8"
        ) as ff,
    ):
        writer = csv.DictWriter(tf, fieldnames=FIELDS)
        writer.writeheader()

        fault_writer = csv.writer(ff)
        fault_writer.writerow([
            "unit_id",
            "fault_type",
            "onset_utc",
            "failure_utc",
        ])

        for index in range(units):
            unit_id = f"Rover_{index + 1:03d}"
            rng = random.Random(seed + index)

            soc = initial_soc
            motor_temp = float(
                thermal["initial_motor_temp_c"]
            )
            equivalent_cycles = 0.0
            odometer_m = 0.0

            fault_type = (
                "bearing_wear"
                if index % 2 == 0
                else "cooling_degradation"
            )

            fault_writer.writerow([
                unit_id,
                fault_type,
                (
                    start + timedelta(minutes=onset)
                ).isoformat(),
                (
                    start + timedelta(minutes=failure)
                ).isoformat(),
            ])

            for minute in range(total_minutes):
                timestamp = start + timedelta(minutes=minute)
                hour = timestamp.hour

                patrol = (
                    int(duty["patrol_start_hour_utc"])
                    <= hour
                    < int(duty["patrol_end_hour_utc"])
                )
                mode = "patrol" if patrol else "dock"
                if patrol:
                    odometer_m += patrol_speed_mps * interval

                ambient = (
                    float(environment["ambient_mean_c"])
                    + float(
                        environment["diurnal_amplitude_c"]
                    )
                    * math.sin(
                        2 * math.pi * (hour - 6) / 24
                    )
                )

                terrain = (
                    0.5
                    + 0.5
                    * math.sin(
                        2 * math.pi * minute / terrain_period
                    )
                )

                power = (
                    float(duty["baseline_patrol_power_w"])
                    + float(duty["terrain_extra_power_w"])
                    * terrain
                    if patrol
                    else (
                        -float(duty["charging_power_w"])
                        if soc < initial_soc
                        else 0.0
                    )
                )

                soh = max(
                    min_soh,
                    initial_soh - fade_rate * equivalent_cycles,
                )
                effective_capacity = (
                    capacity_wh * soh / 100
                )

                soc -= (
                    power / 60 / effective_capacity * 100
                )
                soc = max(0.0, min(100.0, soc))

                if power > 0:
                    equivalent_cycles += (
                        power / 60 / capacity_wh
                    )

                resistance = base_resistance * (
                    1 + resistance_growth * equivalent_cycles
                )

                current = power / voltage_nominal
                voltage = (
                    voltage_nominal
                    + (soc - 50) * ocv_slope
                    - current * resistance
                )

                severity = max(
                    0.0,
                    min(
                        1.0,
                        (minute - onset) / (failure - onset),
                    ),
                )

                extra_heat = (
                    float(thermal["cooling_fault_extra_heat_c"])
                    * severity
                    if fault_type == "cooling_degradation"
                    else 0.0
                )

                target_temp = (
                    ambient
                    + float(
                        thermal["load_heating_coefficient_c_per_w"]
                    )
                    * max(power, 0)
                    + extra_heat
                )

                motor_temp += (
                    target_temp - motor_temp
                ) / response_time

                vibration = float(
                    vibration_cfg["dock_baseline_g"]
                )

                if patrol:
                    vibration = float(
                        vibration_cfg["patrol_baseline_g"]
                    )
                    if fault_type == "bearing_wear":
                        vibration += (
                            float(
                                vibration_cfg[
                                    "bearing_fault_max_extra_g"
                                ]
                            )
                            * severity
                        )

                vibration = max(
                    0.0,
                    vibration
                    + rng.gauss(
                        0,
                        float(vibration_cfg["noise_std_g"]),
                    ),
                )

                writer.writerow({
                    "timestamp_utc": timestamp.isoformat(),
                    "unit_id": unit_id,
                    "mode": mode,
                    "battery_soc_pct": round(soc, 3),
                    "battery_soh_pct": round(soh, 4),
                    "battery_voltage_v": round(voltage, 3),
                    "battery_current_a": round(current, 3),
                    "internal_resistance_ohm": round(
                        resistance, 6
                    ),
                    "power_w": round(power, 2),
                    "ambient_temp_c": round(ambient, 2),
                    "motor_temp_c": round(motor_temp, 2),
                    "terrain_factor": round(terrain, 3),
                    "vibration_rms_g": round(vibration, 4),
                    "odometer_m": round(odometer_m, 3),
                })

    print(
        f"Fleet simulation complete: {units} units x {days} days"
    )
    print(f"Telemetry rows: {units * total_minutes:,}")
    print(f"Fault events: {units}")
    print(f"Output: {telemetry_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=int)
    parser.add_argument("--days", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    args = parser.parse_args()

    simulate_fleet(
        units=args.units,
        days=args.days,
        seed=args.seed,
        output_dir=args.output_dir,
        config_path=args.config,
    )
