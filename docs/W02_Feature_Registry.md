# W02 — Feature Registry

**Project:** Robot Fleet Telemetry Analytics & Predictive Health  
**Generated from:** `src/ingen_fleet_health/features/registry.py` (do not edit by hand; run `python -m ingen_fleet_health.features.registry`)  
**Data:** synthetic Aido Rover fleet, simulator v0.2. Coefficients quoted below are config priors, not fitted values.

## 1. As-of rules (apply to every feature)

1. Prediction times are hourly, `t = hh:00 UTC`. A feature at `t` may use telemetry with `timestamp_utc <= t` only.
2. All windows are trailing, `(t - w, t]`, on a complete per-unit minute grid; missing minutes are NaN, never interpolated from later values. Forward fill is allowed (it only uses the past).
3. Unit baselines are built from the unit's *own* early history (as-of expanding median, frozen after 5 days); no statistic is pooled across units or fitted on the full dataset.
4. Physics normalisation uses fixed config priors. Learned normalisation (`EnvironmentResidualizer`) is fitted inside each training fold only.
5. Ground-truth tables (`ground_truth/`) are never read by the feature code. Maintenance events are used only through `hours_since_repair`, which sees a repair once it has been *completed*.
6. Features flagged **time proxy** grow monotonically with elapsed time; the leakage audit trains a probe model on them alone to check they do not carry label information through the calendar.

Total features: **152** (120 rolling statistics + 32 physics / context features).

## 2. Physics-derived, normalised and context features

| Feature | Group | Definition | Unit | Window | As-of logic | Physical rationale |
|---|---|---|---|---|---|---|
| `missing_fraction_24h` | data_quality | share of the last 1440 minutes with no record | 1 | 24h | trailing window (t - w, t]; minutes <= t only | Dropouts bias every other window statistic; lets the model discount sparse windows |
| `r_hat_24h` | battery | R = -sum(dV*dI)/sum(dI^2) over minute steps with |dI| >= 1 A | ohm | 24h | trailing window (t - w, t]; minutes <= t only | Ohmic drop at current steps (V = OCV - I*R) recovers internal resistance; least squares avoids noise bias |
| `r_hat_temp_norm_24h` | battery | as r_hat_24h with each dV divided by 1 + 0.01*max(0, 25 - ambient) | ohm @25C | 24h | trailing window (t - w, t]; minutes <= t only | Removes the cold-temperature rise in R so ageing/cell faults are not confused with weather |
| `r_hat_ratio_to_baseline` | battery | r_hat_temp_norm_24h divided by the unit's own commissioning baseline (expanding median of its first 5 days, then frozen) | 1 | 24h + baseline | baseline uses only the unit's own past | Unit-relative growth; units differ by +/-10 % at commissioning |
| `energy_per_km_24h` | energy | sum(max(P,0)/60) / patrol distance over the window | Wh/km | 24h | trailing window (t - w, t]; minutes <= t only | Drivetrain efficiency; distance from nominal speed model, energy from measured power |
| `energy_ratio_to_expected_24h` | energy | measured patrol energy / energy expected from terrain and ambient (config priors: 400 W + 160 W*terrain + heater) | 1 | 24h | trailing window (t - w, t]; minutes <= t only | Terrain- and ambient-normalised energy: a hilly or cold day is not degradation |
| `energy_ratio_to_baseline` | energy | energy_ratio_to_expected_24h divided by the unit's own commissioning baseline (expanding median of its first 5 days, then frozen) | 1 | 24h + baseline | baseline uses only the unit's own past | Bearing friction raises energy at equal terrain |
| `motor_temp_expected_c` | thermal | first-order model T += (ambient + 0.028*P_drive - T)/40 run causally on observed ambient and drive power | deg C | recursive (all past) | causal IIR filter of past inputs | Week 1 nominal thermal model as the healthy-motor reference |
| `motor_temp_resid_c` | thermal | motor_temp_c - motor_temp_expected_c at t | deg C | instant | value at t | Abnormal heating not explained by load and ambient |
| `motor_temp_resid_mean_6h` | thermal | mean residual | deg C | 6h | trailing window (t - w, t]; minutes <= t only | Smoothed residual |
| `motor_temp_resid_patrol_mean_24h` | thermal | mean residual over patrol minutes | deg C | 24h | trailing window (t - w, t]; minutes <= t only | Cooling faults show under load |
| `motor_temp_resid_dock_mean_24h` | thermal | mean residual over docked minutes | deg C | 24h | trailing window (t - w, t]; minutes <= t only | Docked motor should sit at ambient: an offset here indicates sensor drift, not a cooling fault |
| `thermal_gain_ratio_24h` | thermal | sum(T - ambient) / sum(T_expected - ambient) over patrol minutes (NaN if mean expected rise < 3 C) | 1 | 24h | trailing window (t - w, t]; minutes <= t only | Multiplicative loss of cooling: >1 means hotter than the model per watt |
| `motor_temp_resid_slope_24h` | thermal | OLS slope of the patrol residual | deg C/h | 24h | trailing window (t - w, t]; minutes <= t only | Progressive heating trend |
| `vib_norm_mean_6h` | vibration | mean of vibration / (1 + 0.6*terrain) over patrol minutes | g | 6h | trailing window (t - w, t]; minutes <= t only | Terrain-normalised vibration (rough surfaces raise RMS) |
| `vib_norm_mean_24h` | vibration | as above | g | 24h | trailing window (t - w, t]; minutes <= t only | Terrain-normalised vibration |
| `vib_norm_slope_24h` | vibration | OLS slope of normalised patrol vibration | g/h | 24h | trailing window (t - w, t]; minutes <= t only | Bearing wear grows gradually |
| `vib_norm_change_72h` | vibration | vib_norm_mean_24h(t) - vib_norm_mean_24h(t - 72 h) | g | 24h, lag 72h | both terms use minutes <= t | Vibration trend over the precursor time scale |
| `vib_norm_ratio_to_baseline` | vibration | vib_norm_mean_24h divided by the unit's own commissioning baseline (expanding median of its first 5 days, then frozen) | 1 | 24h + baseline | baseline uses only the unit's own past | Units differ by +/-15 % in baseline vibration |
| `charge_sessions_cum` **(time proxy)** | battery_usage | count of charge-session starts since first record | count | expanding | cumulative over minutes <= t | Cycle count (ageing exposure) |
| `charge_sessions_24h` | battery_usage | charge-session starts in the window | count | 24h | trailing window (t - w, t]; minutes <= t only | Duty pattern (single vs double shift) |
| `last_cycle_dod_pct` | battery_usage | max - min SoC of the last completed discharge cycle | % | last cycle | value fixed at the start of the next charge | Depth of discharge drives cycle ageing |
| `min_soc_24h` | battery_usage | minimum SoC | % | 24h | trailing window (t - w, t]; minutes <= t only | Deep-discharge stress |
| `efc_cum_nominal` **(time proxy)** | battery_usage | cumulative discharge energy / 6000 Wh | EFC | expanding | cumulative over minutes <= t | Equivalent full cycles (ageing exposure) |
| `efc_24h` | battery_usage | discharge energy in window / 6000 Wh | EFC | 24h | trailing window (t - w, t]; minutes <= t only | Recent ageing exposure |
| `bms_soh_last` | battery | last reported BMS SoH estimate | % | last value | forward fill of past values | Current health estimate (noisy, biased per unit) |
| `bms_soh_mean_24h` | battery | mean BMS SoH estimate | % | 24h | trailing window (t - w, t]; minutes <= t only | Smoothed health estimate |
| `terrain_patrol_mean_24h` | environment | mean terrain factor over patrol minutes | 1 | 24h | trailing window (t - w, t]; minutes <= t only | Normalisation covariate |
| `patrol_fraction_24h` | environment | share of minutes in patrol mode | 1 | 24h | trailing window (t - w, t]; minutes <= t only | Duty covariate; repairs and days off reduce it |
| `odometer_km` **(time proxy)** | usage | last odometer reading / 1000 | km | expanding | forward fill of past values | Mechanical exposure |
| `hours_since_repair` | maintenance | hours since the last maintenance *completed* at or before t (cap 1440) | h | expanding | only repairs with completed_at_utc <= t | Post-repair behaviour; future repairs never visible |
| `hour_of_day` | calendar | UTC hour of t | h | instant | value at t | Diurnal duty and temperature context |

## 3. Rolling statistics

For each channel below, the five statistics `mean, std, min, max, slope` are computed over 1 h, 6 h and 24 h trailing windows (feature name `<prefix>_<stat>_<window>`). Slope is the OLS slope against time in hours, computed from rolling sums over valid minutes only, so it is exact with gaps.

| Channel | Prefix | Unit | Rationale |
|---|---|---|---|
| `motor_temp_c` | `motor_temp` | deg C | Motor heating responds to load and cooling health |
| `vibration_rms_g` | `vibration` | g | Drivetrain/bearing condition; also rises on rough terrain |
| `power_w` | `power` | W | Drive load; bearing friction raises it at equal terrain |
| `battery_voltage_v` | `voltage` | V | Ohmic sag under load grows with internal resistance |
| `battery_soc_pct` | `soc` | % | Duty intensity and depth of discharge (ageing stress) |
| `ambient_temp_c` | `ambient` | deg C | Environment covariate for normalisation |
| `terrain_factor` | `terrain` | 1 | Terrain load covariate for normalisation |
| `internal_resistance_ohm` | `r_bms` | ohm | BMS-reported resistance (noisy), battery health |

Minimum valid minutes per window: 1 h → 15, 6 h → 90, 24 h → 360; otherwise the value is NaN. LightGBM handles NaN natively; the sequence model uses forward-filled daily values.

## 4. Availability at prediction time

All features are available at `t` in a deployed system that receives telemetry with at most one minute of latency. If telemetry is late, the affected windows contain fewer valid minutes (`missing_fraction_24h` rises) rather than being filled from the future.

## 5. Known limitations

- Physics priors are the simulator's own nominal values; on real hardware they would need calibration, and a model fitted on residuals would inherit any prior error.
- `energy_per_km_24h` uses the nominal speed model for distance between odometer samples; real odometry slip is not modelled.
- `r_hat_*` needs current steps of at least 1 A; units that are docked all day have no estimate (NaN).
