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

## 6a. Week 2 — Features, Leakage-Safe Evaluation and Health Models

Week 2 code lives in the installable package `src/ingen_fleet_health/`. Week 1 scripts above are unchanged.

| Module | Purpose |
|---|---|
| `simulate/` | Simulator v0.2: per-unit parameters, site climates, route terrain, P-F fault timelines, repairs, BMS noise, drift, dropouts (`configs/aido_rover_v0_2.yaml`) |
| `labels.py` | 72 h qualifying-onset labels with active-fault, post-repair and censoring exclusions (ADR-003 rev1) |
| `features/` | As-of rolling statistics and physics features (R from current steps, energy/km, thermal residual, vibration trend, DoD); registry → `docs/W02_Feature_Registry.md` |
| `evaluation/` | Unit-grouped walk-forward splits with purge/embargo, fold audit, operator metrics (alert budget, lead time, false alarms per unit-week) |
| `models/` | SoH: persistence, linear, LightGBM, LSTM. Early warning: no-skill, physics rule, LightGBM (+ calibration), Isolation Forest |
| `experiments/` | Runners that write `reports/w02/*.json`, figures in `docs/figures/w02/`, and MLflow runs |
| `benchmarks/cmapss.py` | C-MAPSS FD001 external check (RUL in cycles; baselines, LightGBM, LSTM, engine-grouped leakage check) |

Week 2 commands (about 25 minutes end to end on a laptop):

```powershell
py -3.11 -m uv run python -m ingen_fleet_health.simulate.fleet --out-dir data/w02 --units 50 --days 60
py -3.11 -m uv run python -c "from ingest_pipeline import run_pipeline; run_pipeline('data/w02/generated/fleet_telemetry.csv', out_root='data/w02')"
py -3.11 -m uv run python -m ingen_fleet_health.features.build --data-dir data/w02
py -3.11 -m uv run python -m ingen_fleet_health.experiments.leakage_demo --data-dir data/w02
py -3.11 -m uv run python -m ingen_fleet_health.experiments.soh_run --data-dir data/w02
py -3.11 -m uv run python -m ingen_fleet_health.experiments.early_warning_run --data-dir data/w02
py -3.11 -m uv run python scripts/build_w02_docs.py
py -3.11 -m uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Week 2 headline results (unseen robots, unit-grouped walk-forward, 72 h embargo; 95 % CIs in `docs/W02_Model_Results.docx`):

| Track | Baselines | Models | Result |
|---|---|---|---|
| SoH forecast, MAE (pp) at 1 / 7 / 14 d | persistence 0.34 / 0.44 / 0.67; linear 0.31 / 0.39 / 0.50 | LightGBM 0.26 / 0.31 / 0.41; LSTM 0.27 / 0.40 / 0.52 | LightGBM beats persistence at every horizon and linear at 7 and 14 d; the LSTM does not beat linear |
| 72 h early warning, PR-AUC (base rate 0.055) | physics rule 0.215 | LightGBM 0.210; Isolation Forest 0.330 | Isolation Forest is best (significant vs both); LightGBM does not beat the physics rule |
| Leakage demo, PR-AUC | — | same LightGBM | random shuffle 0.999 vs proper protocol 0.261 |
| External: NASA C-MAPSS FD001, test RMSE in **cycles** (capped truth) | constant 41.94; age-linear 37.00 | LightGBM 13.09 (3 seeds 13.07 ± 0.01); LSTM 14.38 (14.29 ± 0.20) | inside the published range (CNN 18.45, Deep LSTM 16.14, DCNN ≈ 12.6); a different task from SoH, not comparable numerically |

C-MAPSS: download `6. Turbofan Engine Degradation Simulation Data Set.zip` from the [NASA PCoE data-set repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), extract `train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt` into `data/external/cmapss/` (git-ignored), then run `py -3.11 -m uv run python -m ingen_fleet_health.benchmarks.cmapss --data-dir data/external/cmapss`. Data: A. Saxena and K. Goebel (2008), NASA Prognostics Data Repository.

Deliverables: `docs/W02_Feature_Registry.md`, `docs/W02_Evaluation_Protocol.docx`, `docs/W02_Model_Results.docx`, `docs/adr/0004-evaluation-protocol.md`, `weekly/Wk-02-Log.md`. Progress and hand-over notes: `weekly/W02_Progress.md`.

## 6b. Week 3 — Maintenance Text, Fusion, Faithful Explanations and Fari

**All Week 3 data is synthetic**: the maintenance-log corpus is template-generated from the simulator's ground truth, and every parameter of the Fari profile (`configs/fari.yaml`) is an assumption, not an InGen specification. No hosted LLM API is called; the local open-weights models (`all-MiniLM-L6-v2`, `Qwen2.5-1.5B-Instruct`) are downloaded to the Hugging Face cache and run on CPU.

| Module | Purpose |
|---|---|
| `text/` | Synthetic log corpus generator (templates, rule-based paraphrase, seeded noise, label timing) and as-of text features |
| `models/text_classifier.py` | Majority, TF-IDF + LR, sentence-embedding + LR, both; fold-internal C choice and cross-fitting |
| `models/fusion.py` | Text-only, early fusion (one LightGBM) and late fusion (noisy-OR, logistic stacker) arms; fleet health score |
| `explain/` | Tree SHAP, evidence packs, template and local-LLM alert summaries, the faithfulness checker and its negative controls |
| `contract.py` | Platform profiles for ingestion and the enforced Fari no-imagery / no-audio rule |
| `experiments/` | `text_classifier_run`, `fusion_run`, `explain_run`, `paraphrase_run`, `fari_transfer_run` → `reports/w03/*.json`, `docs/figures/w03/`, MLflow |

Week 3 commands (about 1.5 h end to end on a 16-core laptop, CPU only):

```powershell
py -3.11 -m uv run python -m ingen_fleet_health.text.corpus --data-dir data/w02
py -3.11 -m uv run python -m ingen_fleet_health.experiments.text_classifier_run --data-dir data/w02
py -3.11 -m uv run python -m ingen_fleet_health.experiments.paraphrase_run --data-dir data/w02
py -3.11 -m uv run python -m ingen_fleet_health.experiments.fusion_run --data-dir data/w02 --sensitivity
py -3.11 -m uv run python -m ingen_fleet_health.experiments.explain_run --data-dir data/w02 --llm local
py -3.11 -m uv run python -m ingen_fleet_health.simulate.fleet --out-dir data/w03_fari --config configs/fari.yaml
py -3.11 -m uv run python ingest_pipeline.py --input data/w03_fari/generated/fleet_telemetry.csv --out-root data/w03_fari --platform-config configs/fari.yaml
py -3.11 -m uv run python -m ingen_fleet_health.features.build --data-dir data/w03_fari --config configs/fari.yaml
py -3.11 -m uv run python -m ingen_fleet_health.features.build --data-dir data/w03_fari --out-subdir features_asis
py -3.11 -m uv run python -m ingen_fleet_health.experiments.fari_transfer_run --rover-dir data/w02 --fari-dir data/w03_fari
py -3.11 -m uv run python scripts/run_w03_gates.py
py -3.11 -m uv run python scripts/build_w03_docs.py
```

Week 3 headline results (synthetic data; unseen robots / held-out period; 95 % CIs in the documents):

| Track | Baseline | Models | Result |
|---|---|---|---|
| Ticket classifier, macro-F1 (917 held-out tickets, 7 classes) | majority 0.108 | TF-IDF + LR 0.846; embeddings + LR 0.863; both 0.868 | both − TF-IDF +0.022 [0.007, 0.039]; unseen wording 0.481; early symptom notes are mostly read as 'no fault found' |
| Fusion, 72 h early warning PR-AUC (base rate 0.055) | LightGBM 0.210; Isolation Forest 0.330 | text only 0.059; early fusion 0.226; late stacked 0.226 | early − LightGBM +0.016 [-0.040, 0.078] (not detectable); text stacked on Isolation Forest -0.153 (worse); at 4x operator notice rate early − LightGBM +0.130 [0.044, 0.230] |
| Alert summaries, faithful on 20 sampled alerts | — | template; local Qwen2.5-1.5B | template 20/20 automatic, 20/20 on reading; LLM 8/20 automatic, 0/20 in the AI-assisted evidence review; checker flags 79/79 corrupted summaries |
| Fari transfer (synthetic profile), Rover LightGBM PR-AUC on Fari | as-is 0.046; contract acceptance 0 % | platform profile + feature guards 0.157; Fari-trained 0.151 | minimal change +0.111 [0.034, 0.189]; acceptance 100 %; per-platform threshold needed |

Deliverables: `docs/W03_Report.docx` (overview), `docs/W03_Log_Corpus.md`, `docs/W03_Fusion_Results.docx`, `docs/W03_Alert_Explanations.md`, `docs/W03_Fari_Gap_Analysis.docx`, data-contract §3.5–3.6 in `docs/W01_Data_Contract.md`, `weekly/Wk-03-Log.md`, demo script `docs/W03_Demo_Script.md`. LLM-summary faithfulness verdicts: `reports/w03/llm_faithfulness_review.json` (AI-assisted evidence review; review pack built by `scripts/build_w03_review_pack.py`). Progress and hand-over notes: `weekly/W03_Progress.md`.

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