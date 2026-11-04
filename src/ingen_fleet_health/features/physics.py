"""Physics-derived as-of features, computed per unit on a complete minute grid.

Model coefficients come from the simulator config as *priors* (they are the
Week 1 nominal engineering values, not fitted on any data), so these
features cannot leak test information through a fitted normaliser.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ingen_fleet_health.features.rolling import min_periods, rolling_slope
from ingen_fleet_health.simulate.thermal import first_order_response

DAY = 1440


def _roll(series: pd.Series, window: int, how: str = "mean") -> np.ndarray:
    roll = series.rolling(window, min_periods=min_periods(window))
    return np.asarray(getattr(roll, how)().to_numpy())


def _commissioning_baseline(values: pd.Series, days: float = 5.0) -> np.ndarray:
    """Unit's own early-life median, as-of: expanding median until ``days``
    of history exist, then frozen. Uses only the unit's past."""
    n = int(days * DAY)
    expanding = values.expanding(min_periods=60).median().to_numpy().copy()
    if len(values) > n:
        expanding[n:] = expanding[n - 1]
    return expanding


def resistance_step_estimate(
    voltage: np.ndarray,
    current: np.ndarray,
    window: int,
    min_step_a: float = 1.0,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    """Trailing least-squares R_hat = -sum(dV*dI) / sum(dI^2) over current steps.

    Over one minute the OCV barely moves, so a current step shows up almost
    entirely as the ohmic drop dV = -dI * R (Week 1 battery model). Pooling
    the steps by least squares keeps the estimate unbiased under voltage
    noise (a per-step ratio with positivity clipping is biased upward).
    ``scale`` divides each step's dV, e.g. a temperature correction factor.
    """
    dv = np.diff(voltage, prepend=np.nan)
    di = np.diff(current, prepend=np.nan)
    if scale is not None:
        dv = dv / scale
    step = np.abs(di) >= min_step_a          # False where either minute is missing
    num = pd.Series(np.where(step, -dv * di, 0.0)).rolling(window, min_periods=1).sum()
    den = pd.Series(np.where(step, di * di, 0.0)).rolling(window, min_periods=1).sum()
    n = pd.Series(step.astype(float)).rolling(window, min_periods=1).sum()
    return (num / den.where(n >= 5)).to_numpy()


def physics_features(minute: pd.DataFrame, config: dict[str, Any]) -> dict[str, np.ndarray]:
    """``minute`` is one unit on a complete minute grid (NaN rows for gaps)."""
    duty = config["duty_cycle"]
    th = config["thermal"]
    vib = config["vibration"]
    b = config["battery"]

    patrol = minute["mode"].eq("patrol").to_numpy()
    observed = minute["mode"].notna().to_numpy()
    amb = minute["ambient_temp_c"]
    power = minute["power_w"]
    terrain = minute["terrain_factor"]
    out: dict[str, np.ndarray] = {}

    # --- data availability ---
    out["missing_fraction_24h"] = 1.0 - pd.Series(observed.astype(float)).rolling(DAY, min_periods=1).mean().to_numpy()

    # --- effective internal resistance from current steps ---
    volt = minute["battery_voltage_v"].to_numpy()
    curr = minute["battery_current_a"].to_numpy()
    cold = (1.0 + float(b["cold_resistance_per_c"]) * np.maximum(0.0, 25.0 - amb.ffill())).to_numpy()
    r_raw = resistance_step_estimate(volt, curr, DAY)
    r_norm = pd.Series(resistance_step_estimate(volt, curr, DAY, scale=cold))
    out["r_hat_24h"] = r_raw
    out["r_hat_temp_norm_24h"] = r_norm.to_numpy()
    out["r_hat_ratio_to_baseline"] = r_norm.to_numpy() / _commissioning_baseline(r_norm)

    # --- energy per km, raw and terrain/ambient-normalised ---
    speed = float(duty["patrol_speed_mps"]) * (1.0 - float(duty["terrain_speed_reduction"]) * terrain)
    dist_km = pd.Series(np.where(patrol, speed * 60.0 / 1000.0, 0.0))
    energy_wh = pd.Series(np.where(patrol, np.maximum(power, 0.0) / 60.0, 0.0))
    heater = float(duty["cold_heater_w_per_c"]) * np.maximum(0.0, float(duty["cold_heater_threshold_c"]) - amb)
    expected_w = float(duty["baseline_patrol_power_w"]) + float(duty["terrain_extra_power_w"]) * terrain + heater
    expected_wh = pd.Series(np.where(patrol, expected_w / 60.0, 0.0))
    e24 = energy_wh.rolling(DAY, min_periods=1).sum()
    d24 = dist_km.rolling(DAY, min_periods=1).sum()
    x24 = expected_wh.rolling(DAY, min_periods=1).sum()
    out["energy_per_km_24h"] = (e24 / d24.where(d24 > 0.5)).to_numpy()
    ratio = e24 / x24.where(x24 > 50.0)
    out["energy_ratio_to_expected_24h"] = ratio.to_numpy()
    out["energy_ratio_to_baseline"] = ratio.to_numpy() / _commissioning_baseline(ratio)

    # --- motor temperature residual vs. the nominal first-order thermal model ---
    # forward-fill only (past values); a constant covers minutes before the first record
    amb_f = amb.ffill().fillna(20.0)
    drive = pd.Series(np.where(patrol, np.maximum(power.fillna(0.0) - heater.fillna(0.0), 0.0), 0.0))
    target = amb_f.to_numpy() + float(th["load_heating_coefficient_c_per_w"]) * drive.to_numpy()
    initial = float(minute["motor_temp_c"].dropna().iloc[0]) if minute["motor_temp_c"].notna().any() else 20.0
    expected_t = first_order_response(target, float(th["response_time_minutes"]), initial)
    resid = minute["motor_temp_c"] - expected_t
    resid_patrol = resid.where(patrol)
    resid_dock = resid.where(~patrol & observed)
    out["motor_temp_expected_c"] = expected_t
    out["motor_temp_resid_c"] = resid.to_numpy()
    out["motor_temp_resid_mean_6h"] = _roll(resid, 360)
    out["motor_temp_resid_patrol_mean_24h"] = resid_patrol.rolling(DAY, min_periods=30).mean().to_numpy()
    out["motor_temp_resid_dock_mean_24h"] = resid_dock.rolling(DAY, min_periods=30).mean().to_numpy()
    rise_obs = (minute["motor_temp_c"] - amb).where(patrol)
    rise_exp = pd.Series(expected_t - amb_f.to_numpy()).where(patrol)
    gain = rise_obs.rolling(DAY, min_periods=30).sum() / rise_exp.rolling(DAY, min_periods=30).sum()
    out["thermal_gain_ratio_24h"] = gain.where(rise_exp.rolling(DAY, min_periods=30).mean() > 3.0).to_numpy()
    out["motor_temp_resid_slope_24h"] = rolling_slope(resid_patrol.to_numpy(), DAY)

    # --- vibration normalised for terrain, trend and unit baseline ---
    vib_norm = (minute["vibration_rms_g"] / (1.0 + float(vib["terrain_gain"]) * terrain)).where(patrol)
    vn24 = vib_norm.rolling(DAY, min_periods=30).mean()
    out["vib_norm_mean_6h"] = vib_norm.rolling(360, min_periods=30).mean().to_numpy()
    out["vib_norm_mean_24h"] = vn24.to_numpy()
    out["vib_norm_slope_24h"] = rolling_slope(vib_norm.to_numpy(), DAY)
    out["vib_norm_change_72h"] = (vn24 - vn24.shift(3 * DAY)).to_numpy()
    out["vib_norm_ratio_to_baseline"] = vn24.to_numpy() / _commissioning_baseline(vn24)

    # --- charge cycles and depth of discharge ---
    charging = (power < 0).fillna(False).to_numpy()
    charge_start = charging & ~np.roll(charging, 1)
    charge_start[0] = False
    out["charge_sessions_cum"] = np.cumsum(charge_start).astype(float)
    out["charge_sessions_24h"] = pd.Series(charge_start.astype(float)).rolling(DAY, min_periods=1).sum().to_numpy()
    soc = minute["battery_soc_pct"]
    session = np.cumsum(charge_start)
    soc_disch = soc.where(~charging)
    cyc_max = soc_disch.groupby(session).cummax()
    cyc_min = soc_disch.groupby(session).cummin()
    dod_running = (cyc_max - cyc_min)
    # DoD of the last *completed* discharge cycle, available once charging starts
    completed = pd.Series(np.where(charge_start, dod_running.shift(1).to_numpy(), np.nan))
    out["last_cycle_dod_pct"] = completed.ffill().to_numpy()
    out["min_soc_24h"] = _roll(soc, DAY, "min")
    out["efc_cum_nominal"] = (energy_wh.cumsum() / float(b["nominal_capacity_wh"])).to_numpy()
    out["efc_24h"] = (e24 / float(b["nominal_capacity_wh"])).to_numpy()

    # --- BMS state-of-health estimate ---
    soh = minute["battery_soh_pct"]
    out["bms_soh_last"] = soh.ffill().to_numpy()
    out["bms_soh_mean_24h"] = _roll(soh, DAY)

    # --- environment covariates used for normalisation ---
    out["terrain_patrol_mean_24h"] = terrain.where(patrol).rolling(DAY, min_periods=30).mean().to_numpy()
    out["patrol_fraction_24h"] = pd.Series(patrol.astype(float)).rolling(DAY, min_periods=1).mean().to_numpy()
    out["odometer_km"] = (minute["odometer_m"].ffill() / 1000.0).to_numpy()
    return out
