"""Simulator v0.2 entry point.

The time loop runs minute by minute and is vectorised across units, so a
50-unit x 60-day fleet takes seconds rather than minutes. Telemetry keeps
the 14-field Week 1 schema so ``ingest_pipeline.run_pipeline`` validates it
unchanged; everything a model must not see goes to ``ground_truth/``.

Output layout (``out_dir``)::

    generated/fleet_telemetry.csv     prediction-time telemetry (with dropouts)
    events/maintenance_events.csv     observable repairs (usable as-of only)
    events/units.csv                  observable unit registry
    ground_truth/fault_events.csv     hidden P-F timelines (labels only)
    ground_truth/battery_truth_daily.csv  true SoH / R per unit-day (SoH target)
    ground_truth/unit_params.csv      hidden per-unit physical parameters
    ground_truth/impairment_events.csv  injected sensor drift and outages
    simulation_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingen_fleet_health.config import REPO_ROOT, load_config
from ingen_fleet_health.simulate.battery import (
    arrhenius_factor,
    resistance_ohm,
    terminal_voltage,
)
from ingen_fleet_health.simulate.duty_terrain import (
    patrol_schedule,
    site_ambient,
    terrain_factor,
)
from ingen_fleet_health.simulate.faults import (
    downtime_mask,
    sample_fault_events,
    severity_matrix,
)
from ingen_fleet_health.simulate.params import FleetParams, sample_fleet_params

MINUTES_PER_DAY = 1440
TELEMETRY_FIELDS = [
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


def _iso(start: datetime, minute: int) -> str:
    return (start + timedelta(minutes=int(minute))).isoformat()


def _sensor_drift(
    config: dict[str, Any], n_units: int, total: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    imp = config["impairments"]
    drift = np.zeros((n_units, total))
    records: list[dict[str, Any]] = []
    n_drift = round(float(imp["drift_unit_fraction"]) * n_units)
    if n_drift == 0:
        return drift, records
    minutes = np.arange(total)
    for u in rng.choice(n_units, n_drift, replace=False):
        ramp_days = rng.uniform(*imp["drift_ramp_days_range"])
        peak = rng.uniform(*imp["drift_peak_c_range"])
        start = int(rng.uniform(0.15, 0.75) * total)
        ramp_min = ramp_days * MINUTES_PER_DAY
        drift[u] = peak * np.clip((minutes - start) / ramp_min, 0.0, 1.0)
        records.append({
            "unit_index": int(u),
            "event_type": "sensor_drift",
            "sensor": "motor_temp_c",
            "start_min": start,
            "end_min": int(min(start + ramp_min, total)),
            "magnitude": float(peak),
            "shape": "linear",
        })
    return drift, records


def _dropout_mask(
    config: dict[str, Any], n_units: int, days: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    imp = config["impairments"]
    total = days * MINUTES_PER_DAY
    keep = rng.random((n_units, total)) >= float(imp["dropout_probability"])
    records: list[dict[str, Any]] = []
    lo, hi = imp["burst_minutes_range"]
    for u in range(n_units):
        for d in range(days):
            if rng.random() < float(imp["burst_probability_per_unit_day"]):
                start = d * MINUTES_PER_DAY + int(rng.integers(0, MINUTES_PER_DAY))
                length = int(rng.integers(int(lo), int(hi) + 1))
                keep[u, start:start + length] = False
                records.append({
                    "unit_index": u,
                    "event_type": "network_outage",
                    "sensor": "all",
                    "start_min": start,
                    "end_min": min(start + length, total),
                    "magnitude": float(length),
                    "shape": "burst",
                })
    keep[:, 0] = True
    keep[:, -1] = True
    return keep, records


def _run_dynamics(
    config: dict[str, Any],
    params: FleetParams,
    ambient: np.ndarray,
    schedule: np.ndarray,
    terrain: np.ndarray,
    sev: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Minute-by-minute state update for all units at once."""
    b = config["battery"]
    duty = config["duty_cycle"]
    th = config["thermal"]
    vib = config["vibration"]
    fcfg = config["faults"]
    n_units, total = ambient.shape

    cap_nominal = params.capacity_wh
    soc = np.full(n_units, float(b["initial_soc_pct"]))
    q_loss = np.full(n_units, 100.0 - float(b["initial_soh_pct"]))
    motor = np.full(n_units, float(th["initial_motor_temp_c"]))
    efc = np.zeros(n_units)
    odo = np.zeros(n_units)
    forced_dock = np.zeros(n_units, dtype=bool)
    charging_prev = np.zeros(n_units, dtype=bool)
    charge_minutes = np.zeros(n_units)
    bms_est = 100.0 - q_loss + params.bms_soh_bias_pct

    out = {
        name: np.empty((n_units, total))
        for name in (
            "soc", "soh_true", "bms_soh", "power", "drive_power", "r_true",
            "motor_true", "vibration", "odometer", "efc",
        )
    }
    out["patrol"] = np.empty((n_units, total), dtype=bool)

    taper_start = float(duty["charge_taper_start_soc_pct"])
    target_soc = float(duty["charge_target_soc_pct"])
    speed = float(duty["patrol_speed_mps"])
    heater_k = float(duty["cold_heater_w_per_c"])
    heater_t = float(duty["cold_heater_threshold_c"])
    doubling = float(b["arrhenius_doubling_c"])
    cal_per_min = float(b["calendar_fade_pct_per_day"]) / MINUTES_PER_DAY
    cell_extra_per_min = float(fcfg["cell_extra_fade_pct_per_day"]) / MINUTES_PER_DAY
    vib_noise = rng.normal(0.0, float(vib["noise_std_g"]), (n_units, total))

    for m in range(total):
        amb = ambient[:, m]
        f = terrain[:, m]
        sb, sc, scell = sev["bearing_wear"][:, m], sev["cooling_degradation"][:, m], sev["battery_cell_fault"][:, m]

        forced_dock &= soc < 60.0
        patrol = schedule[:, m] & ~forced_dock
        drive = (
            params.baseline_power_w
            + float(duty["terrain_extra_power_w"]) * f
            + float(vib["bearing_friction_extra_w"]) * sb
        )
        heater = heater_k * np.maximum(0.0, heater_t - amb)
        charging = ~patrol & (soc < target_soc)
        charge_w = float(duty["charging_power_w"]) * np.minimum(
            1.0, (100.0 - soc) / (100.0 - taper_start)
        )
        power = np.where(patrol, drive + heater, np.where(charging, -charge_w, 0.0))

        soh = 100.0 - q_loss
        cap_eff = cap_nominal * soh / 100.0 * (
            1.0 - float(b["cold_capacity_loss_per_c"]) * np.maximum(0.0, 20.0 - amb)
        )
        energy_wh = np.where(power > 0, power, power * float(duty["charge_efficiency"])) / 60.0
        soc = np.clip(soc - energy_wh / cap_eff * 100.0, 0.0, 100.0)
        forced_dock |= patrol & (soc < float(duty["return_to_dock_soc_pct"]))

        d_efc = np.maximum(power, 0.0) / 60.0 / cap_nominal
        efc += d_efc
        pack_t = amb + np.where(
            power > 0, float(b["pack_temp_rise_patrol_c"]),
            np.where(power < 0, float(b["pack_temp_rise_charge_c"]), 0.0),
        )
        arr = arrhenius_factor(pack_t, doubling)
        stress = arr * np.where(soc < float(b["low_soc_stress_threshold_pct"]), float(b["low_soc_stress_factor"]), 1.0)
        q_loss += params.cycle_fade_pct_per_efc * d_efc * stress + cal_per_min * arr + cell_extra_per_min * scell
        q_loss = np.minimum(q_loss, 100.0 - float(b["min_soh_pct"]))

        # BMS re-estimates SoH at the end of each charge session (>= 30 min)
        charge_minutes = np.where(charging, charge_minutes + 1, charge_minutes)
        ended = charging_prev & ~charging & (charge_minutes >= 30)
        if ended.any():
            noise = rng.normal(0.0, float(b["bms_soh_noise_std_pct"]), n_units)
            bms_est = np.where(ended, 100.0 - q_loss + params.bms_soh_bias_pct + noise, bms_est)
        charge_minutes = np.where(~charging, 0.0, charge_minutes)
        charging_prev = charging

        heating = params.heating_coefficient * (1.0 + float(fcfg["cooling_max_heating_increase"]) * sc)
        tau = params.thermal_time_constant_min * (1.0 + float(fcfg["cooling_max_time_constant_increase"]) * sc)
        target = amb + heating * np.where(patrol, drive, 0.0)
        motor = motor + (target - motor) / tau

        vibration = np.where(
            patrol,
            params.vibration_baseline_g * (1.0 + float(vib["terrain_gain"]) * f)
            + float(fcfg["bearing_max_extra_g"]) * sb,
            float(vib["dock_baseline_g"]),
        )

        odo += np.where(patrol, speed * (1.0 - float(duty["terrain_speed_reduction"]) * f) * 60.0, 0.0)

        out["soc"][:, m] = soc
        out["soh_true"][:, m] = 100.0 - q_loss
        out["bms_soh"][:, m] = bms_est
        out["power"][:, m] = power
        out["drive_power"][:, m] = np.where(patrol, drive, 0.0)
        out["motor_true"][:, m] = motor
        out["vibration"][:, m] = np.maximum(0.0, vibration + vib_noise[:, m])
        out["odometer"][:, m] = odo
        out["efc"][:, m] = efc
        out["patrol"][:, m] = patrol

    out["r_true"] = resistance_ohm(
        params.base_resistance_ohm[:, None],
        100.0 - out["soh_true"],
        ambient,
        sev["battery_cell_fault"],
        float(b["resistance_growth_per_pct_fade"]),
        float(b["cold_resistance_per_c"]),
        float(fcfg["cell_max_resistance_increase"]),
    )
    return out


def simulate_fleet_v02(
    out_dir: str | Path,
    units: int | None = None,
    days: int | None = None,
    seed: int | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    sim = config["simulation"]
    units = int(sim["units"]) if units is None else int(units)
    days = int(sim["days"]) if days is None else int(days)
    seed = int(sim["seed"]) if seed is None else int(seed)
    if units < 1 or days < 1:
        raise ValueError("Units and days must be positive")
    if int(sim["sample_interval_seconds"]) != 60:
        raise ValueError("Only 60-second sampling is supported")
    start = datetime.fromisoformat(str(sim["start_utc"]))
    if start.utcoffset() != timedelta(0):
        raise ValueError("Simulation start must be UTC")

    total = days * MINUTES_PER_DAY
    # Independent child streams so changing one component does not reshuffle the others
    streams = np.random.SeedSequence(seed).spawn(8)
    rng_params, rng_weather, rng_sched, rng_terrain, rng_faults, rng_dyn, rng_meas, rng_imp = (
        np.random.default_rng(s) for s in streams
    )

    params = sample_fleet_params(config, units, rng_params)
    site_amb = site_ambient(config, days, rng_weather)
    ambient = site_amb[params.site_index] + rng_weather.normal(
        0.0, float(config["weather"]["minute_noise_std_c"]), (units, total)
    )
    schedule = patrol_schedule(config, params, days, rng_sched)
    terrain = terrain_factor(config, params, days, rng_terrain)
    events = sample_fault_events(config, units, total, rng_faults)
    schedule &= ~downtime_mask(events, units, total)
    sev = {t: severity_matrix(events, t, units, total) for t in config["faults"]["types"]}

    dyn = _run_dynamics(config, params, ambient, schedule, terrain, sev, rng_dyn)

    meas = config["measurement"]
    b = config["battery"]
    vn = float(b["nominal_voltage_v"])
    current_true = dyn["power"] / vn
    current_meas = current_true + rng_meas.normal(0.0, float(meas["current_noise_std_a"]), (units, total))
    voltage = terminal_voltage(vn, dyn["soc"], current_true, dyn["r_true"], float(b["ocv_soc_slope_v_per_pct"]))
    voltage_meas = voltage + rng_meas.normal(0.0, float(meas["voltage_noise_std_v"]), (units, total))
    r_reported = dyn["r_true"] * (1.0 + rng_meas.normal(0.0, 0.03, (units, total)))
    drift, drift_records = _sensor_drift(config, units, total, rng_imp)
    motor_meas = dyn["motor_true"] + rng_meas.normal(
        0.0, float(config["thermal"]["measurement_noise_std_c"]), (units, total)
    ) + drift
    keep, outage_records = _dropout_mask(config, units, days, rng_imp)

    out = Path(out_dir)
    for sub in ("generated", "events", "ground_truth"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    stamps = pd.date_range(start, periods=total, freq="min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S+00:00")
    stamps_arr = np.asarray(stamps)
    current_round = np.round(current_meas, 3)
    frame = pd.DataFrame({
        "timestamp_utc": np.tile(stamps_arr, units),
        "unit_id": np.repeat(params.unit_ids, total),
        "mode": np.where(dyn["patrol"], "patrol", "dock").ravel(),
        "battery_soc_pct": np.round(dyn["soc"], 3).ravel(),
        "battery_soh_pct": np.round(np.minimum(dyn["bms_soh"], 100.0), 3).ravel(),
        "battery_voltage_v": np.round(voltage_meas, 3).ravel(),
        "battery_current_a": current_round.ravel(),
        "internal_resistance_ohm": np.round(r_reported, 6).ravel(),
        # Week 1 contract: power is derived by the BMS at nominal voltage
        "power_w": np.round(vn * current_round, 2).ravel(),
        "ambient_temp_c": np.round(ambient, 2).ravel(),
        "motor_temp_c": np.round(motor_meas, 2).ravel(),
        "terrain_factor": np.round(terrain, 3).ravel(),
        "vibration_rms_g": np.round(dyn["vibration"], 4).ravel(),
        "odometer_m": np.round(dyn["odometer"], 3).ravel(),
    })
    frame = frame[keep.ravel()]
    telemetry_path = out / "generated" / "fleet_telemetry.csv"
    frame.to_csv(telemetry_path, index=False)

    unit_ids = params.unit_ids
    fault_rows = [{
        "fault_id": ev.fault_id,
        "unit_id": unit_ids[ev.unit_index],
        "fault_type": ev.fault_type,
        "ramp_shape": ev.ramp_shape,
        "initiation_utc": _iso(start, ev.initiation_min),
        "onset_utc": _iso(start, ev.onset_min),
        "failure_utc": _iso(start, ev.failure_min),
        "repair_start_utc": _iso(start, ev.repair_start_min),
        "repair_end_utc": _iso(start, ev.repair_end_min),
        "precursor_days": round(ev.precursor_days, 3),
        "onset_within_run": ev.onset_min < total,
    } for ev in events]
    pd.DataFrame(fault_rows, columns=[
        "fault_id", "unit_id", "fault_type", "ramp_shape", "initiation_utc", "onset_utc",
        "failure_utc", "repair_start_utc", "repair_end_utc", "precursor_days", "onset_within_run",
    ]).to_csv(out / "ground_truth" / "fault_events.csv", index=False)

    component = {
        "bearing_wear": "drivetrain_bearing",
        "cooling_degradation": "motor_cooling",
        "battery_cell_fault": "battery_pack",
    }
    maint_rows = [{
        "maintenance_id": f"M{i + 1:04d}",
        "unit_id": unit_ids[ev.unit_index],
        "started_at_utc": _iso(start, ev.repair_start_min),
        "completed_at_utc": _iso(start, ev.repair_end_min),
        "component": component[ev.fault_type],
        "action": "replace",
    } for i, ev in enumerate(events) if ev.repair_start_min < total]
    pd.DataFrame(maint_rows, columns=[
        "maintenance_id", "unit_id", "started_at_utc", "completed_at_utc", "component", "action",
    ]).to_csv(out / "events" / "maintenance_events.csv", index=False)

    site_names = [str(s["name"]) for s in config["sites"]]
    pd.DataFrame([{
        "unit_id": u,
        "platform": config["platform"],
        "firmware_version": config["firmware_version"],
        "site": site_names[int(params.site_index[i])],
        "shift_pattern": "double" if params.shift_double[i] else "single",
    } for i, u in enumerate(unit_ids)]).to_csv(out / "events" / "units.csv", index=False)
    pd.DataFrame(params.to_records(site_names)).to_csv(out / "ground_truth" / "unit_params.csv", index=False)

    day_end = np.arange(1, days + 1) * MINUTES_PER_DAY - 1
    dates = [(start + timedelta(days=d)).date().isoformat() for d in range(days)]
    pd.DataFrame({
        "unit_id": np.repeat(unit_ids, days),
        "date_utc": np.tile(dates, units),
        "soh_true_pct": np.round(dyn["soh_true"][:, day_end], 5).ravel(),
        "resistance_true_ohm": np.round(dyn["r_true"][:, day_end], 6).ravel(),
        "efc_cumulative": np.round(dyn["efc"][:, day_end], 4).ravel(),
    }).to_csv(out / "ground_truth" / "battery_truth_daily.csv", index=False)

    pd.DataFrame([{
        "unit_id": unit_ids[r["unit_index"]],
        "event_type": r["event_type"],
        "sensor": r["sensor"],
        "start_utc": _iso(start, r["start_min"]),
        "end_utc": _iso(start, r["end_min"]),
        "magnitude": r["magnitude"],
        "shape": r["shape"],
    } for r in drift_records + outage_records], columns=[
        "unit_id", "event_type", "sensor", "start_utc", "end_utc", "magnitude", "shape",
    ]).to_csv(out / "ground_truth" / "impairment_events.csv", index=False)

    config_text = Path(config_path).read_text(encoding="utf-8") if config_path else (
        REPO_ROOT / "configs" / "aido_rover_v0_2.yaml"
    ).read_text(encoding="utf-8")
    manifest = {
        "simulator_version": "0.2",
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "units": units,
        "days": days,
        "seed": seed,
        "start_utc": start.isoformat(),
        "end_utc": _iso(start, total),
        "telemetry_rows_generated": units * total,
        "telemetry_rows_written": len(frame),
        "fault_events": len(events),
        "fault_events_with_onset_in_run": int(sum(ev.onset_min < total for ev in events)),
        "units_without_faults": int(units - len({ev.unit_index for ev in events})),
        "sensor_drift_events": len(drift_records),
        "network_outages": len(outage_records),
    }
    (out / "simulation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulator v0.2 (Week 2)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "w02")
    parser.add_argument("--units", type=int)
    parser.add_argument("--days", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    manifest = simulate_fleet_v02(args.out_dir, args.units, args.days, args.seed, args.config)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
