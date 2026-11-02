# W01 — Simulator Physics and Assumptions

**Project:** Robot Fleet Telemetry Analytics & Predictive Health  
**Version:** 0.1  
**Scope:** Synthetic Aido Rover fleet  
**Status:** Engineering prototype; not calibrated against production hardware

## 1. Simulation Scope

The simulator generates one-minute telemetry observations for a configurable fleet of outdoor patrol robots.

The baseline full-scale experiment covers 20 robots over 60 days, producing 1,728,000 observations.

All parameters are illustrative engineering assumptions. They do not represent verified inGen hardware specifications.

## 2. Battery Model

The model tracks battery state of charge (SoC), state of health (SoH), and internal resistance.

Battery energy is updated using:

\[
SOC_{t+1} = \operatorname{clip}\left(SOC_t - \frac{P_t\Delta t}{C_{\mathrm{eff}}}\times100,\ 0,\ 100\right)
\]

where power is positive during discharge and negative during charging.

Effective capacity decreases with equivalent full cycles. The baseline model assumes a nominal capacity of 6,000 Wh and capacity degradation of 0.04 percentage points per equivalent full cycle.

Internal resistance begins at 0.04 ohms and increases with accumulated cycling.

Limitations: The current implementation does not fully model temperature-dependent battery capacity, load-dependent electrochemical efficiency, or detailed charging dynamics.

## 3. Motor Thermal Model

Motor temperature follows a simplified first-order thermal response:

\[
T_{t+1}=T_t+\frac{T_{\mathrm{target}}-T_t}{40}
\]

where the denominator represents a nominal 40-minute discrete response scale at one-minute sampling.

The target temperature depends on ambient conditions, operating power, and an additional temperature contribution for cooling degradation.

The model is intended to reproduce heating during patrol and relaxation toward ambient temperature while docked.

Limitations: No detailed heat-transfer geometry, motor efficiency curve, or measured thermal calibration is included.

## 4. Duty Cycle and Terrain

The baseline operating schedule assumes patrol from 08:00 to 16:00 UTC, with docking during other hours.

Power consumption varies with a synthetic terrain factor between zero and one:

\[
P_{\mathrm{patrol}}=400+120f_{\mathrm{terrain}}
\]

where power is measured in watts.

Terrain varies periodically in the current implementation; it is not yet separated into explicit slope and surface-type categories.

Ambient temperature follows a deterministic diurnal sinusoidal pattern. Stochastic weather disturbances are not currently modeled.

## 5. Vibration and Bearing Degradation

The vibration model uses a low dock baseline and a higher patrol baseline.

A synthetic bearing fault increases vibration progressively during operation, with additive Gaussian noise.

The model is designed to demonstrate gradual degradation rather than an instantaneous change.

Its output should not be interpreted as calibrated vibration measurements from physical hardware.

## 6. Fault Injection and Sensor Impairments

The simulator includes synthetic bearing-wear and cooling-degradation events with recorded onset and failure timestamps.

A separate seeded impairment module introduces:

- Randomly missing telemetry records representing communication dropouts.
- Gradually increasing motor-temperature sensor bias.
- Ground-truth records of the injected sensor drift.

An experiment removed 8,486 observations, and the ingestion pipeline accounted for 8,486 missing minutes.

Sensor drift within plausible measurement ranges is not yet automatically detected by the existing physical-range validation rules.

## 7. Reproducibility

The simulator uses a fixed random seed. Repeated runs with identical inputs produce identical synthetic CSV outputs, as verified by automated testing.

The current configuration file documents the parameter values but is not yet connected to the simulator runtime.

Generated datasets are excluded from Git version control. Source code, tests, model assumptions, and reproducibility instructions are retained.

## 8. Known Limitations

The simulation does not constitute a validated digital twin of an inGen robot.

Important limitations include simplified battery aging, lack of detailed weather or surface-type variation, limited mechanical dynamics, and absence of real sensor calibration.

Fault-event timestamps are synthetic ground truth; a simulated failure timestamp does not currently force the robot to stop operating.

These limitations must be considered when interpreting future forecasting and predictive-maintenance results.

## 9. Physics Sanity-Check Results

The 20-unit, 60-day baseline simulation was evaluated using `fleet_physics_sanity.png`.

- **Battery SoC:** Decreases during patrol and recovers during docking and charging, consistent with the modeled duty cycle.
- **Motor temperature:** Increases under patrol load, reaching approximately 42.32°C on the first day, and gradually returns toward ambient temperature after patrol ends.
- **Battery SoH:** Decreases monotonically, with daily average SoH declining from 99.9878% to 98.5400% over the 60-day simulation.

These checks confirm expected qualitative behavior under the chosen model assumptions. They do not establish calibration against actual robot hardware.

**Known limitation:** Ambient temperature is currently updated using integer-hour time, producing artificial hourly steps. A continuous-time ambient model is recommended for future simulation runs.

Additional validation should directly compare temperature against motor load and examine degradation behavior under different operating conditions.

## 10. Odometer Model

The synthetic Rover includes a cumulative odometer measured in meters.

The baseline patrol speed is assumed to be 1.0 m/s. During patrol, the odometer increases by 60 meters per minute. During docking, it remains unchanged.

The ingestion pipeline enforces nonnegative, non-decreasing odometer values and quarantines records that violate this rule.

This model assumes a constant travel speed and does not yet account for stops, turning, wheel slip, or terrain-dependent speed variation.
