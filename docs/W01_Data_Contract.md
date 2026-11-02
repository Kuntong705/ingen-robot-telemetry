# W01 — Fleet Telemetry Data Contract

**Project:** Robot Fleet Telemetry Analytics & Predictive Health  
**Version:** 0.2 — Implemented Prototype  
**Updated:** November 1, 2026  
**Primary Platform:** Synthetic Aido Rover  
**Data Classification:** Synthetic demonstration data; no confidential telemetry

## 1. Purpose and Scope

This contract describes the telemetry schema, data validation rules, supporting event tables, storage architecture, and proposed predictive-maintenance labels for the synthetic Rover fleet.

The baseline simulation contains 20 robots operating over 60 days, generating 1,728,000 minute-level observations.

The pipeline processes these records through Raw, Validated, and Curated layers using CSV, DuckDB, and Parquet.

All physical parameters are illustrative rather than verified production hardware specifications.

## 2. Telemetry Schema

Each telemetry record represents one robot at one UTC timestamp.

| Field | Data Type | Unit / Meaning |
|---|---|---|
| `timestamp_utc` | datetime | Timezone-aware UTC |
| `unit_id` | string | Unique robot identifier |
| `mode` | string | `patrol` or `dock` |
| `battery_soc_pct` | float | State of charge, % |
| `battery_soh_pct` | float | Estimated state of health, % |
| `battery_voltage_v` | float | Battery voltage, V |
| `battery_current_a` | float | Battery current, A |
| `internal_resistance_ohm` | float | Internal resistance, Ω |
| `power_w` | float | Electrical power, W |
| `ambient_temp_c` | float | Ambient temperature, °C |
| `motor_temp_c` | float | Motor temperature, °C |
| `terrain_factor` | float | Dimensionless load factor |
| `vibration_rms_g` | float | Vibration RMS, g |
| `odometer_m` | float | Cumulative distance, m |

Positive electrical power represents discharge, while negative power represents charging.

Telemetry is sampled once per minute. Each `(unit_id, timestamp_utc)` pair should be unique.

A cumulative odometer is maintained independently for each robot and must never decrease during normal operation.

The schema is currently specific to the synthetic Rover platform. Other inGen platforms will require additional, platform-specific fields.

## 3. Supporting Data Tables

### 3.1 Fault Events — Implemented

The simulator produces a separate `fleet_fault_events.csv` table:

| Field | Type |
|---|---|
| `unit_id` | string |
| `fault_type` | string |
| `onset_utc` | datetime |
| `failure_utc` | datetime |

Current synthetic fault types are `bearing_wear` and `cooling_degradation`.

Each simulated robot receives one fault event. The event records indicate the modeled onset and designated failure time, but the simulator does not force robot shutdown upon reaching that time.

### 3.2 Sensor Impairment Events — Implemented

The impairment module independently records injected temperature-sensor drift, including affected unit, sensor, onset time, ramp completion time, maximum bias, and ramp shape.

These ground-truth records are kept separate from prediction-time telemetry.

### 3.3 Operational Events — Proposed

Future operational event records should include:

- `event_id`
- `unit_id`
- `timestamp_utc`
- `event_type`
- `metadata_json`

Examples include charging starts, patrol transitions, connectivity interruptions, and system shutdowns.

### 3.4 Maintenance Records — Proposed

Future maintenance records should include:

- `maintenance_id`
- `unit_id`
- `started_at_utc`
- `completed_at_utc`
- `component`
- `action`

These records will support maintenance exclusion windows and future predictive-maintenance labels.

## 4. Data Validation Rules

The implemented ingestion pipeline checks:

| Rule | Expected condition |
|---|---|
| Required columns | All 14 telemetry fields present |
| Timestamp | Valid timezone-aware UTC |
| Ordering | Increasing timestamps per robot |
| Duplicate records | No repeated unit/timestamp combination among accepted records |
| Battery SoC | 0–100% |
| Battery SoH | 0–100% |
| Battery voltage | 30–60 V |
| Ambient temperature | −60 to 70°C |
| Motor temperature | −40 to 120°C |
| Terrain factor | 0–1 |
| Vibration RMS | Nonnegative |
| Internal resistance | Positive |
| Odometer | Nonnegative and non-decreasing |
| Power/current | Consistent with the baseline nominal 48 V assumption |
| Sampling gaps | Counted per robot |

All numeric values must be finite.

Invalid records are written to a quarantine CSV with error codes instead of being silently corrected.

The ranges above are initial engineering assumptions, not production acceptance limits.

The pipeline currently uses custom validation logic. Full pandera or Great Expectations schema enforcement and broader unit consistency checks remain outstanding.

## 5. Data Processing Architecture

The pipeline contains three layers.

**Raw:** Preserve a copy of the incoming synthetic CSV without applying corrections.

**Validated:** Check schema, timestamps, numeric values, physical ranges, and odometer consistency. Write accepted records to typed Parquet; write rejected records separately to quarantine.

**Curated:** Aggregate accepted observations by robot and UTC date. Produce summary metrics for battery charge, motor temperature, vibration, and record counts.

DuckDB performs Parquet conversion and daily aggregation.

The ingestion process generates `reports/W01_Data_Quality_Report.json`, including accepted and rejected counts, rejection reasons, timestamp gap events, missing minutes, and overall quality status.

The current implementation does not yet provide a complete production-grade data lineage or schema registry.

## 6. Data Quality Experiments

### Baseline Fleet Simulation

The complete clean simulation contains 1,728,000 telemetry observations across 20 robots and 60 days.

The baseline pipeline is intended to accept all valid generated records with no sampling gaps.

### Impaired Fleet Simulation

A separate seeded impairment experiment introduced communication dropouts and progressive motor-temperature sensor bias.

Observed results:

| Metric | Result |
|---|---:|
| Original observations | 1,728,000 |
| Removed observations | 8,486 |
| Remaining valid observations | 1,719,514 |
| Timestamp gap events | 8,437 |
| Missing minutes | 8,486 |
| Rejected remaining records | 0 |

The quality report correctly indicated failure of the completeness requirement because missing minutes were present.

The present validator does not automatically identify temperature-sensor drift when the affected measurements remain within allowed physical ranges. This requires additional sensor-health analysis.

## 7. ADR-001 — Analytical Storage

**Decision:** Use DuckDB with local Parquet files.

**Rationale:** Parquet supports compressed columnar storage, while DuckDB enables local SQL analysis without requiring a cloud database.

**Alternative considered:** BigQuery or another managed cloud analytical platform.

**Trade-off:** Local analysis is inexpensive and reproducible, but production-scale orchestration, multiuser access, and centralized governance would require additional infrastructure.

## 8. ADR-002 — Telemetry Table Layout

**Decision:** Adopt a wide telemetry table with one row per unit per timestamp.

**Rationale:** Related sensor measurements are available in the same record, simplifying cross-channel comparisons, temporal aggregation, and future feature engineering.

**Alternative considered:** Long-format event/value tables.

**Trade-off:** The wide layout is convenient for the relatively fixed Rover schema but less flexible for heterogeneous high-dimensional robotic platforms.

Platform-specific extension schemas may be needed for future fleet integration.

## 9. ADR-003 — Failure Label Definition

**Proposed target:** A new qualifying fault onset within the next 72 hours.

At prediction time *t*, the binary target is positive if a qualifying fault begins after *t* and on or before *t + 72 hours*.

This choice aims to provide actionable lead time for maintenance teams.

Alternatives considered:

- **Current anomaly detection:** Useful for immediate diagnosis but does not necessarily provide advance warning.
- **Time-to-failure prediction:** Potentially informative but requires a reliable definition of actual failure and sufficient failure observations.
- **Onset-based early warning:** Chosen provisionally because it aligns with advance maintenance decisions.

A proposed 24-hour post-maintenance exclusion window will prevent immediate post-repair observations from distorting training labels.

Records without a complete future observation horizon should be censored rather than automatically labeled negative.

Ground-truth events must never be incorporated into features available at prediction time.

**Implementation status:** The 72-hour predictive target and maintenance exclusions are design decisions, not yet implemented as a production-ready labeling pipeline.

## 10. Reproducibility and Governance

The simulation uses a fixed random seed and a version-controlled YAML configuration profile.

Project dependencies are declared in `pyproject.toml` and resolved in `uv.lock`.

Automated tests validate simulator behavior, data ingestion, record quarantine, configuration changes, and selected physical consistency rules.

Thirty local pytest tests passed after adding odometer validation. GitHub Actions runs Ruff, mypy, and pytest, with the latest changes pending cloud verification.

Generated datasets and confidential information must not be committed to the repository. Only synthetic data, code, test results, design documents, and approved artifacts should be shared.

## 11. Remaining Work

The current Week 1 foundation supports configurable synthetic fleet generation, fault injection, record validation, Parquet storage, daily analytical summaries, and automated testing.

Remaining engineering improvements include:

- Strict external schema validation with pandera or Great Expectations
- Unit consistency driven by the YAML configuration rather than a hard-coded nominal voltage
- Full operational and maintenance event schemas
- Sensor drift detection beyond simple physical-range validation
- Expanded physical modeling and empirical calibration
- Final repository structure and cross-platform configuration profiles

The prototype is suitable for demonstrating a reproducible synthetic data engineering workflow, but is not a validated digital twin of production hardware.