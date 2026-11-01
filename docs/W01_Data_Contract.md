# W01 — Fleet Telemetry Data Contract

**Project:** Robot Fleet Telemetry Analytics & Predictive Health  
**Version:** 0.1 — Initial Design  
**Status:** Proposed; not yet fully implemented  
**Primary platform:** Synthetic outdoor patrol robot (Rover)  

## 1. Purpose and Scope

This document defines the data structures, validation rules, storage decisions, and failure-labeling strategy for a synthetic robot fleet telemetry pipeline.

The initial prototype generates one-minute observations for one robot over 24 hours. The target architecture supports multiple units over extended operating periods.

All data and physical parameters are synthetic or illustrative. No proprietary telemetry or confidential hardware specifications are used.

## 2. Telemetry Schema

The existing prototype generates the following fields.

| Field | Type | Description |
|---|---|---|
| `timestamp_utc` | datetime | Timezone-aware UTC timestamp |
| `unit_id` | string | Unique robot identifier |
| `mode` | string | Operating mode: patrol or dock |
| `battery_soc_pct` | float | Battery state of charge, 0–100% |
| `power_w` | float | Electrical power in watts; positive = discharge, negative = charging |
| `ambient_temp_c` | float | Ambient temperature in °C |
| `motor_temp_c` | float | Simulated motor temperature in °C |
| `vibration_rms_g` | float | Simulated vibration RMS in units of standard gravity |

Each robot produces one observation per minute. The combination of `unit_id` and `timestamp_utc` must be unique.

Planned extensions include `platform`, `firmware_version`, battery voltage/current, cumulative odometer readings, and sensor status. These fields are not yet implemented.

## 3. Supporting Data Tables

**Operational Events**

Proposed fields: `event_id`, `unit_id`, `timestamp_utc`, `event_type`, `metadata_json`.

Examples: patrol start, charging start, system shutdown, connectivity interruption.

**Maintenance Records**

Proposed fields: `maintenance_id`, `unit_id`, `started_at_utc`, `completed_at_utc`, `component`, `action`.

These will support the exclusion of post-repair periods during model training.

**Ground-Truth Fault Events**

Current prototype fields: `unit_id`, `fault_type`, `onset_utc`, `failure_utc`.

The prototype injects one synthetic bearing-wear event. Fault onset and failure time are generated labels, not observed failures from physical hardware.

## 4. Data Quality Rules

- Timestamps must be valid UTC times and ordered within each robot's data.
- No duplicate `(unit_id, timestamp_utc)` records.
- Battery SoC must remain within 0–100%.
- Motor temperature must remain within configurable physically plausible limits.
- Vibration magnitude must be nonnegative.
- Missing records and timestamp gaps must be reported.
- Invalid records must be rejected or quarantined instead of silently corrected.

Currently, only basic output-level checks are implemented. Independent record validation, gap reporting, and corrupted-record rejection remain planned.

## 5. ADR-001 — Analytical Storage

**Decision:** Use local Parquet files with DuckDB as the initial analytical storage solution.

This approach provides efficient columnar storage and SQL querying without requiring cloud infrastructure. It also enables reviewers to regenerate synthetic datasets locally without cloud credentials or usage costs.

BigQuery remains a possible future option for larger operational fleets. For the present synthetic prototype, cloud deployment would increase complexity without providing an essential benefit. This decision will be revisited if workload scale or deployment requirements change.

## 6. ADR-002 — Table Layout

**Decision:** Use a wide telemetry table with one row per robot per timestamp.

A wide layout simplifies multi-channel analysis because battery, motor, environmental, and vibration values are available in the same observation. This supports rolling-window feature engineering and downstream machine learning.

A long format would make adding arbitrary sensor channels easier, but would require additional pivoting for many model features. For extended datasets, Parquet files will be partitioned by date and optionally by platform. The partition design will be benchmarked before finalizing.

## 7. ADR-003 — Failure Label Definition

**Proposed prediction target:** Whether a new fault begins within the next 72 hours.

For a prediction at time *t*, the target is positive if a qualifying fault onset occurs after *t* and no later than *t + 72 hours*. Features must be computed using information available at or before *t*.

A proposed 24-hour exclusion period following completed repairs will prevent transient post-maintenance conditions from contaminating training labels. This duration is an initial assumption requiring evaluation. Observations without a full future prediction window will be excluded or marked censored, not automatically assigned negative labels.

The current 24-hour prototype is insufficient for valid 72-hour labeling or model evaluation. This labeling strategy will be implemented after extending the simulation period.

## 8. Reproducibility and Data Governance

- Synthetic data is generated using a fixed random seed.
- Simulator parameters will be documented in version-controlled configuration files.
- Generated datasets will not be committed to GitHub.
- No confidential company documents, customer data, credentials, or API keys will be stored in the repository.
- Ground-truth labels must remain separate from prediction-time telemetry to prevent target leakage.

## 9. Implementation Status and Next Steps

**Implemented:** Basic telemetry generation, synthetic fault-event output, range checks, six automated tests, and visualization.

**Remaining:** Independent data validation, multi-robot simulation, longer prediction horizons, Parquet ingestion, DuckDB integration, corrupted-record tests, and automated CI.

This document describes an evolving engineering contract. Unimplemented requirements are explicitly identified and must not be presented as completed capabilities.