# Week 4 Progress Tracker (living file — read this first in a new session)

**Branch:** `week4`, created on 2026-11-24 from `week3` at 64fdd3d. Week 3 is pushed and open as PR #1 (`week3` -> `week2`, not merged). Local commits only: no push, merge, PR or GitHub Release without Kuntong's approval.  
**Plan:** Internship Plan v3.0 (`Kuntong_Li_Internship_Plan_.pdf`, 4-week version), Week 4: Orchestration, Fleet Health Dashboard, Model Card & Capstone.  
**Source-of-truth check (2026-11-24):** Week 4 requirements were read from Plan v3.0 §3 Week 4, §4 Deliverables Index, §5.3 Quality Standards and §7 Repository Structure. v2.0 (the 8-week plan) is not used.

## AI-use policy

No written AI-use policy was found in the local programme materials (Plan v3.0, first-task details, meeting briefs and onboarding slides were searched). Plan v3.0 lists Claude Code and Codex in the intern's stack and asks for LLM-written alert summaries, so AI-assisted engineering is treated as allowed. **Kuntong should confirm the current InGen policy.**

These items must be done by the named people themselves, and none was drafted by the assistant:
- the personal reflection in `W04_Retrospective.docx`;
- the signed final evaluation rubric and the career note (Iqbal);
- the readout delivery and the Q&A.

## Week 4 execution checklist (dependency order)

| # | Item | Status |
|---|---|---|
| 1 | Production training recipe; models in the MLflow registry (`release/train.py`, ADR-0005) | done: `fleet_early_warning` v1, `fleet_soh_forecast` v1 in `mlflow.db` |
| 2 | Daily pipeline core: landing → validate → features → score → predictions → alert queue (`pipeline/`) | done |
| 3 | Idempotent re-runs, backfill, missing and late partitions (ADR-0006) | done: tested in pytest and checked through Airflow |
| 4 | Monitoring: PSI and the weekly performance check (`monitoring/`) | done |
| 5 | Airflow DAG and Docker Compose stack | done: built and run in Docker (see `reports/w04/docker_airflow_checks.json`) |
| 6 | Streamlit + Plotly dashboard (ADR-0007) | done: smoke-tested (AppTest) and viewed in the browser (local and Docker) |
| 7 | Model card and cost-benefit (`W04_Model_Card.docx`, `release/cost_benefit.py`, `configs/cost_assumptions.yaml`) | done: generated from `reports/w04/*.json` by `scripts/build_w04_docs.py` |
| 8 | Release files: README, ADRs 0001–0007, Dockerfile, compose, LICENSE, NOTICE, CONTRIBUTING, `.env.example`, CI | done; CI not run on GitHub (no push) |
| 9 | Capstone deck (PDF), retrospective template, demo script | done: 10 slides with finding titles; the retrospective has facts only, and its reflection sections are blank for Kuntong |
| 10 | Quality gates | done locally: ruff, mypy (71 files), pytest 142 passed (19 W04; coverage 58 %), uv lock, Docker build and smoke, Airflow DAG import, gitleaks (24 commits, no leaks); Airflow run, idempotency, backfill and missing partition in `docker_airflow_checks.json`; fresh-clone quickstart passed (README steps 2-7 in an empty directory; the first attempt found a warm-up-day bug, fixed in 362ae76) |

## Design decisions (made autonomously)

| # | Decision | Why |
|---|---|---|
| D1 | Champion: pooled telemetry-only LightGBM with a platform flag; the Isolation Forest and the physics rule run as shadow models that never alert | Explanations (tree SHAP + faithful template) and the best measured Fari transfer; the Rover ranking gap is reported, not hidden (ADR-0005) |
| D2 | Production training: 70 % of units per platform, rows up to day 30 − 72 h; dates before the cut-off are never scored (`InSampleDate` guard) | Held-out robots give an honest operational number; no in-sample scores reach the dashboard |
| D3 | Daily features are recomputed from the full validated history, parallel over unit chunks | Exactly equal to the Week 2 batch table (tested); carrying state day to day is a documented next step |
| D4 | Missing landing partition: wait 6 h, then fail; no outputs for that platform and date; the other platform runs; the run fails (`all_platforms_ok`) | No stale or partial risk can look current (ADR-0006) |
| D5 | Dashboard "Pull" means a new alert in the last 24 h; 24–72 h alerts are shown as "Alerted in last 72 h" | With 1 alert per robot-week, a 72 h window flagged about 45 % of the quickstart fleet; that is not a usable shift list |
| D6 | v1 does not use text | Week 3: no detectable gain at the assumed note rate; the summary says notes are not an input |
| D7 | Costs are placeholders in `configs/cost_assumptions.yaml`; the headline is the break-even ratio (false alarms per caught failure) | No InGen cost figures are available; the ratio uses measured counts only |
| D8 | Quickstart seeds 79 / 1098 (12 units × 40 days) were chosen by fault-onset counts only, before any model was fitted | With 12 units a random seed often has no onset in the training window; the quickstart is a plumbing demo, not an evaluation |

## Session log (2026-11-24)

- Read Plan v3.0 Week 4, the README, ADRs, the Week 1–3 reports and progress logs, and the code. Baseline: `week3` at 64fdd3d (PR #1 open). Created the `week4` branch.
- Built the `pipeline/`, `monitoring/` and `release/` packages, the Airflow DAG, Docker files and the dashboard.
- Dependencies added with uv: streamlit 1.67.0, plotly 7.1.0, pytest-cov 7.1.0.
- Defects found by tests and runs, all fixed:
  - SoH target-date dtype;
  - `status()` counted training-window dates as gaps;
  - Docker volume ownership;
  - the DAG hid platform failures behind an `all_done` leaf;
  - `HOME` broke Airflow;
  - a 72 h pull window made 11 of 24 robots "pull".
- Full-data run on `data/w04`:
  - history load 2026-10-26 → 2026-11-24 (ingest and features);
  - scoring 2026-11-25 → 2026-12-24 (both platforms, 50 + 50 robots).

## Measured results (synthetic; `reports/w04/operational_results.json`, `cost_benefit.json`)

- **Production training (MLflow `w04_release`).**
  - The selection rule picked `lgbm_all`: validation PR-AUC 0.312 vs 0.186 for `lgbm_physics`.
  - Calibration is isotonic.
  - Both platform thresholds came out at 2.43 %, the same isotonic step.
- **Operational period** (2026-11-25 → 2026-12-24; labels settled until 2026-12-18), held-out robots (15 per platform):
  - **Aido Rover:** champion PR-AUC 0.370 [0.088, 0.646]; Isolation Forest shadow 0.534, physics rule 0.303.
    - Champion − Isolation Forest: −0.164 [−0.409, 0.055] (not detectable).
    - Alert queue: 5 of 6 onsets caught, precision 0.10, 0.94 false alarms per robot-week.
  - **Fari:** champion 0.160 [0.029, 0.533]; Isolation Forest 0.075, rule 0.061.
    - Alert queue: 5 of 6 onsets caught, precision 0.09, 1.05 false alarms per robot-week.
- **SoH, 7 days, held-out robots:** model MAE 0.275 vs persistence 0.436 pp (Rover) and 0.314 vs 0.454 (Fari); both skill CIs are above 0.
- **Template summaries:** 209/209 (Rover) and 223/223 (Fari) pass checker v2. The top driver is a battery feature in 80 % and 93 % of alerts.
- **Drift:** 126/240 and 124/240 feature-days are in the "shift" band. The median PSI of `ambient_max_24h` is 4.0 and 3.4 (simulated seasonal cooling).
- **Weekly checks:** 4 checks written (2026-11-29 to 2026-12-20). Several cohort-weeks are reported as "insufficient labels" (fewer than 3 onsets).
- **Cost-benefit (placeholder costs).**
  - Break-even ratio, held-out champion: Rover 8.8, Fari 9.6. W2 walk-forward: LightGBM 9.0, Isolation Forest 10.6, rule 9.9.
  - The placeholder ratio is 5.8, so every strategy has a negative net at placeholder costs.
- **Docker / Airflow** (`reports/w04/docker_airflow_checks.json`):
  - images built; the quickstart ran inside the container in 6.5 min;
  - DAG import is clean;
  - `dags test` passed 14/14 tasks with byte-identical outputs;
  - the missing Fari partition gave a sensor timeout, untouched outputs, a failure record and a failed run;
  - Airflow backfill: 2 runs succeeded;
  - the dashboard container is healthy;
  - gitleaks found no leaks in 23 commits (before the Week 4 commit).

## Quality gates (2026-11-24, local; `reports/w04/quality_gates.json`)

| Gate | Result |
|---|---|
| ruff | pass |
| mypy (`src/`, `fleet_simulator.py`) | pass, 71 files |
| pytest + coverage | 142 passed at 7f031eb (coverage of `ingen_fleet_health` 58 %); after the warm-up fix (362ae76): **143 passed** (123 Weeks 1–3 + 20 Week 4) |
| `uv lock --check` | pass |
| Docker build (app + airflow) | pass |
| App image smoke | pass |
| Airflow DAG import (3.3.2) | pass, no import errors |
| gitleaks v8.30.0 (full history) | pass, 24 commits, no leaks |
| Airflow run: idempotency, missing partition, backfill | pass (`docker_airflow_checks.json`) |
| Dashboard smoke | pass: AppTest in pytest, container health, viewed in the browser |
| Fresh-directory quickstart | pass on the second attempt (`docker_airflow_checks.json` → `fresh_clone_readme_quickstart`) |
| GitHub Actions CI | **not run**: nothing pushed |

## Not done / needs Kuntong

- **Push, PR, GitHub Release v1.0.0:** not done (needs approval). CI with the badge and gitleaks therefore has not run on GitHub; local equivalents are recorded in `reports/w04/quality_gates.json`.
- **InGen AI-use policy:** confirm it (see above).
- **LICENSE holder:** `LICENSE` names "Kuntong Li" (MIT, as Plan v3.0 asks). Confirm whether InGen wants a different holder or licence.
- **Retrospective reflection, signed rubric, career note:** for the named people to complete.
- **Real cost figures:** `configs/cost_assumptions.yaml` holds placeholders. Replace them with InGen figures before quoting any money value.
