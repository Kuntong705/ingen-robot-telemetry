# Robot Fleet Telemetry Analytics & Predictive Health

**Week 1 — Synthetic Fleet Data Engineering Prototype**

This project develops a reproducible synthetic telemetry pipeline for robot fleet monitoring and predictive-health research.

The initial implementation focuses on a simulated Aido Rover fleet. All telemetry and physical parameters are synthetic and illustrative, not verified inGen hardware specifications.

## 1. Project Overview

The prototype simulates robot operating conditions, generates sensor telemetry, validates data quality, and prepares structured datasets for downstream analytics and predictive maintenance.

**Current implementation:**

- 20 simulated robots over 60 days
- 1-minute sampling frequency
- 1,728,000 baseline telemetry records
- 14 telemetry fields, including battery, motor, vibration, terrain, and odometer measurements
- Synthetic bearing-wear and cooling-degradation scenarios
- Seeded network dropouts and temperature-sensor drift
- Raw → Validated → Curated ingestion pipeline
- DuckDB and Parquet analytical storage
- Data quality reporting and invalid-record quarantine
- Physics sanity-check visualizations
- 30 automated pytest tests and GitHub Actions CI

## 2. Repository Structure

| File / Directory | Purpose |
|---|---|
| `fleet_simulator.py` | Generate configurable multi-robot telemetry |
| `configs/aido_rover.yaml` | Synthetic Rover configuration and model parameters |
| `inject_impairments.py` | Inject network dropouts and sensor drift |
| `ingest_pipeline.py` | Validate records and produce Parquet datasets |
| `validate_telemetry.py` | Additional CSV validation utilities |
| `plot_fleet_sanity.py` | Generate physics sanity-check figures |
| `tests/` | Automated simulator and ingestion tests |
| `docs/` | Data contract, platform brief, and model assumptions |
| `reports/` | Generated data quality report |
| `weekly/` | Weekly engineering log |
| `.github/workflows/` | Automated linting, type checking, and testing |

## 3. Setup

**Requirements:** Python 3.11 and uv.

From the repository root, install the locked dependencies:

```powershell
py -3.11 -m uv sync --locked
```

## 4. Quickstart

### Step 1 — Generate Fleet Telemetry

```powershell
py -3.11 -m uv run --locked python fleet_simulator.py --units 20 --days 60 --seed 42
```

Outputs:

- `data/generated/fleet_telemetry.csv`
- `data/generated/fleet_fault_events.csv`

The baseline run produces 1,728,000 telemetry rows and 20 synthetic fault events.

### Step 2 — Run the Ingestion Pipeline

```powershell
py -3.11 -m uv run --locked python ingest_pipeline.py
```

The pipeline validates incoming records and generates:

- `data/raw/fleet_telemetry.csv`
- `data/validated/telemetry.parquet`
- `data/curated/daily_summary.parquet`
- `data/quarantine/rejected.csv`
- `reports/W01_Data_Quality_Report.json`

The validated layer contains accepted telemetry records. The curated layer provides daily robot-level summaries.

### Step 3 — Generate Physics Sanity Checks

```powershell
py -3.11 -m uv run --locked python plot_fleet_sanity.py
```

Figure output:

`docs/figures/fleet_physics_sanity.png`

The visualization examines battery state of charge, motor thermal response, and battery state of health.

### Step 4 — Run Automated Tests

```powershell
py -3.11 -m uv run --locked python -m pytest -q
```

Latest verified result: **30 passed**.

Additional engineering checks:

```powershell
py -3.11 -m uv run --locked ruff check .
py -3.11 -m uv run --locked mypy --ignore-missing-imports fleet_simulator.py
```

GitHub Actions runs automated checks on repository updates.

## 5. Data Quality Results

### Clean Baseline Dataset

| Metric | Result |
|---|---:|
| Total observations | 1,728,000 |
| Accepted observations | 1,728,000 |
| Rejected observations | 0 |
| Missing minutes | 0 |
| Quality passed | True |

### Network Impairment Experiment

A seeded experiment introduced missing telemetry records and gradual temperature-sensor drift.

| Metric | Result |
|---|---:|
| Records removed | 8,486 |
| Remaining accepted records | 1,719,514 |
| Missing minutes detected | 8,486 |
| Timestamp gap events | 8,437 |
| Quality passed | False |

The completeness failure is expected because network dropouts were intentionally introduced.

## 6. Design Decisions

The prototype uses:

1. **DuckDB + Parquet** for reproducible local analytical storage.
2. **Wide-format telemetry** to simplify multichannel time-series analysis.
3. **A proposed 72-hour fault-onset prediction target** to support future preventive-maintenance decisions.

The prediction target is currently a design proposal. A production-ready predictive model has not yet been implemented.

See `docs/W01_Data_Contract.md` for the full schema and architecture decisions.

## 7. Limitations and Future Work

- Physical parameters are illustrative and have not been calibrated against actual robot hardware.
- The simulator uses simplified battery degradation, thermal dynamics, terrain effects, and fault progression.
- The current telemetry implementation focuses on Aido Rover rather than all five inGen platforms.
- Sensor drift detection requires additional methods beyond physical-range validation.
- Further work includes stricter schema enforcement, operational and maintenance records, predictive-health modeling, and broader platform support.

## 8. Documentation

- `docs/W01_Fleet_Data_Brief.docx` — Five-platform data and analytics assessment
- `docs/W01_Data_Contract.md` — Telemetry schema, quality rules, and architecture decisions
- `docs/W01_Simulator_Assumptions.md` — Simulation parameters and physical assumptions
- `reports/W01_Data_Quality_Report.json` — Baseline data quality results
- `weekly/Wk-01-Log.md` — Week 1 engineering decisions and progress

## 9. Data Governance

This repository is intended to contain synthetic code, documentation, tests, and approved demonstration artifacts only.

Generated large datasets are excluded from version control and can be regenerated using the documented configuration and random seed.

No proprietary inGen telemetry, confidential internal documents, customer information, credentials, or API keys should be committed.

---

**Project status:** Week 1 prototype implemented and tested. Further validation and model development remain in progress.