# Week 4 Progress Tracker (living file — read this first in a new session)

**Branch:** `week4`, created on 2026-11-24 from `week3` at 64fdd3d. Week 3 is pushed and open as PR #1 (`week3` -> `week2`, not merged). Local commits only: no push, merge, PR or GitHub Release without Kuntong's approval.  
**Plan:** Internship Plan v3.0 (`Kuntong_Li_Internship_Plan_.pdf`, 4-week version), Week 4: Orchestration, Fleet Health Dashboard, Model Card & Capstone.  
**Source-of-truth check (2026-11-24):** Week 4 requirements were read from Plan v3.0 §3 Week 4, §4 Deliverables Index, §5.3 Quality Standards and §7 Repository Structure. v2.0 (the 8-week plan) is not used.

## AI-use policy and AI assistance

No written AI-use policy was found in the local programme materials (Plan v3.0, first-task details, meeting briefs and onboarding slides were searched). Plan v3.0 lists Claude Code and Codex in the intern's stack and asks for LLM-written alert summaries, so AI-assisted engineering is treated as allowed. **Kuntong should confirm the current InGen policy with the supervisor.**

What AI assistance did (recorded, not hidden): the code, tests, configuration, ADR texts, README, model card, deck, demo script and these progress notes were written with Claude Code under Kuntong's direction; commits carry a `Co-Authored-By` trailer (kept). The Week 3 LLM-summary faithfulness verdicts are an AI-assisted evidence review. The README and the model card (§11) say so.

These items must be done by the named people themselves, and none was drafted by the assistant:
- the personal reflection in `W04_Retrospective.docx` (the generator no longer overwrites that file);
- the signed final evaluation rubric and the career note (supervisor);
- the readout delivery and the Q&A.

## Week 4 execution checklist (dependency order)

| # | Item | Status |
|---|---|---|
| 1 | Production training recipe; models in the MLflow registry (`release/train.py`, ADR-0005) | done: `fleet_early_warning` v1, `fleet_soh_forecast` v1 in `mlflow.db`; reproducible from committed code (`reports/w04/retrain_reproducibility.json`) |
| 2 | Daily pipeline core: landing → validate → features → score → predictions → alert queue (`pipeline/`) | done |
| 3 | Idempotent re-runs, backfill, missing and late partitions (ADR-0006) | done: tested in pytest and checked through Airflow |
| 4 | Monitoring: PSI and the weekly performance check (`monitoring/`) | done |
| 5 | Airflow DAG and Docker Compose stack | done (see `reports/w04/docker_airflow_checks.json`) |
| 6 | Streamlit + Plotly dashboard (ADR-0007) | done: AppTest, browser, container |
| 7 | Model card and cost-benefit (`W04_Model_Card.docx` and `.pdf`, `release/cost_benefit.py`, `configs/cost_assumptions.yaml`) | done: generated from `reports/w04/*.json` by `scripts/build_w04_docs.py` |
| 8 | Release files: README, ADRs 0001–0007, Dockerfile, compose, LICENSE, NOTICE, CONTRIBUTING, `.env.example`, CI, `data/sample/` | done; CI not run on GitHub (no push); licence holder pending |
| 9 | Capstone deck (PDF), retrospective template, demo script | done |
| 10 | Quality gates on the final commit | see "Quality gates" below |
| 11 | Final audit fixes (2026-11-24) | done, see below |

## Final audit fixes (2026-11-24)

No reported result, model, threshold or evaluation protocol was changed. Every existing number in `reports/w04/operational_results.json` is unchanged; only fields were added and the protocol text was corrected.

- **Factual error fixed.** The model card said "about 20 fault onsets per platform" in the training window. The measured count is 8 per platform on training robots before the cut-off, 3 of them in the 6-day validation slice (`training_window` in the evidence JSON).
- **Protocol paragraph.** The model card §4b showed the evidence script's input description instead of the evaluation protocol (wrong docstring paragraph); fixed.
- **Significance stated.** Model card §4a now shows the Week 2 LightGBM − Isolation Forest result (significant, IF better). Every comparison says whether its 95 % interval excludes 0; counts and break-even ratios are labelled as untested point comparisons.
- **Isolation Forest cost advantage on Rover** stated in the model card §8, deck slide 9 and README (break-even 6.5 vs 8.8, 6/6 vs 5/6 caught; not tested). On Fari the shadow is worse (15.5 vs 9.6).
- **v1 calibration measured** (model card §4e): reliability bins with unit-bootstrap intervals. On Fari the top-tier risk (about 1.00) was followed by an onset 36 % of the time, interval [0.00, 0.93]; on Rover 99 %. The middle bins are not monotone.
- **Tie statistics, one definition each.** The isotonic map has 21 (Rover) / 71 (Fari) distinct values on the evaluated held-out rows. Rows at exactly the most common value: 85.7 % / 72.3 %. Rows below the alert threshold: 88.0 % / 72.5 %. Rows exactly at the threshold value (they alert): 6.7 % / 9.8 %. On all scored rows of all robots: 79.6 % / 71.5 % at the most common value. The 94 % quoted during the audit came from an ad-hoc bin whose upper edge was the rounded threshold, so it also counted the rows sitting exactly at the threshold; the evidence script now uses the exact threshold as the edge.
- **Post-hoc diagnostic** (labelled, not a result): ranking by the uncalibrated margin gives PR-AUC 0.423 (Rover, +0.053 [0.021, 0.096]) and 0.272 (Fari, +0.113 [−0.006, 0.247]) vs 0.370 / 0.160. A v2 direction only.
- **Provenance.** v1 was registered from an uncommitted tree (`git_dirty=True`); `scripts/check_w04_retrain.py` shows an identical retrain from committed code.
- **Product names** not confirmed as public were generalised (README next steps, deck slides 2 and 10, NOTICE).
- **Deck layout:** larger architecture diagram (slide 1), a live dashboard alert screenshot (slide 7), a new readable Fari chart (slide 8), a cleaner cost chart (slide 9), a finding title on slide 10.
- **`W04_Model_Card.pdf`** added (Plan v3.0 §7); same content as the .docx, printed by headless Edge.
- **`data/sample/`** added (Plan v3.0 §7): 2 synthetic robots × 1 day per platform, `scripts/make_data_sample.py`, tests for reproducibility and the data contract.
- **Docker image:** `UV_NO_CACHE=1` keeps uv's download cache (about 2 GB) out of the image.
- **Gate records:** `scripts/run_w04_gates.py` records portable commands (no local paths or user name), the tested commit and whether the tree was clean, and refuses to mix results from two commits.

## Design decisions (made autonomously)

| # | Decision | Why |
|---|---|---|
| D1 | Champion: pooled telemetry-only LightGBM with a platform flag; the Isolation Forest and the physics rule run as shadow models that never alert | Tree-SHAP explanations with a faithful template and a higher (not significant) Fari point estimate; the Rover ranking and cost gap is reported, not hidden (ADR-0005 and its 2026-11-24 revision) |
| D2 | Production training: 70 % of units per platform, rows up to day 30 − 72 h; dates before the cut-off are never scored (`InSampleDate` guard) | Held-out robots give an honest operational number; no in-sample scores reach the dashboard |
| D3 | Daily features are recomputed from the full validated history, parallel over unit chunks | Exactly equal to the Week 2 batch table (tested); carrying state day to day is a documented next step |
| D4 | Missing landing partition: wait 6 h, then fail; no outputs for that platform and date; the other platform runs; the run fails (`all_platforms_ok`) | No stale or partial risk can look current (ADR-0006) |
| D5 | Dashboard "Pull" means a new alert in the last 24 h; 24–72 h alerts are shown as "Alerted in last 72 h" | With 1 alert per robot-week, a 72 h window flagged about 45 % of the quickstart fleet |
| D6 | v1 does not use text | Week 3: no detectable gain at the assumed note rate |
| D7 | Costs are placeholders in `configs/cost_assumptions.yaml`; the headline is the break-even ratio | No InGen cost figures are available; the ratio uses measured counts only |
| D8 | Quickstart seeds 79 / 1098 (12 units × 40 days) were chosen by fault-onset counts only, before any model was fitted | With 12 units a random seed often has no onset in the training window |

## Measured results (synthetic; `reports/w04/operational_results.json`, `cost_benefit.json`)

- **Production training (MLflow `w04_release`).** The selection rule picked `lgbm_all` (validation PR-AUC 0.312 vs 0.186 for `lgbm_physics`). Calibration is isotonic. Both platform thresholds came out at 2.43 %, the same isotonic step.
- **Operational period** (2026-11-25 → 2026-12-24, simulated dates; labels settled until 2026-12-18), held-out robots (15 per platform):
  - **Aido Rover:** champion PR-AUC 0.370 [0.088, 0.646]; Isolation Forest shadow 0.534, physics rule 0.303. Champion − IF −0.164 [−0.409, 0.055] (not significant). Alert queue: 5 of 6 onsets caught, precision 0.10, 0.94 false alarms per robot-week.
  - **Fari:** champion 0.160 [0.029, 0.533]; Isolation Forest 0.075, rule 0.061. Champion − IF +0.085 [−0.016, 0.382] (not significant). Alert queue: 5 of 6 caught, precision 0.09, 1.05 false alarms per robot-week.
- **SoH, 7 days, held-out robots:** model MAE 0.275 vs persistence 0.436 pp (Rover) and 0.314 vs 0.454 (Fari); both skill CIs are above 0 (significant).
- **Template summaries:** 209/209 (Rover) and 223/223 (Fari) pass checker v2.
- **Drift:** 126/240 and 124/240 feature-days in the "shift" band (simulated seasonal cooling).
- **Weekly checks:** 4 written; 3 of 6 held-out and 1 of 6 training cohort-weeks had insufficient labels.
- **Cost-benefit (placeholder costs):** break-even ratio, held-out champion Rover 8.8, Fari 9.6; shadow Isolation Forest Rover 6.5, Fari 15.5; W2 walk-forward LightGBM 9.0, IF 10.6, rule 9.9. The placeholder ratio is 5.8, so every strategy has a negative net at placeholder costs.

## Quality gates

All gates ran on commit **f21f2ee** with a clean working tree (`reports/w04/quality_gates.json` records the commit and `git_worktree_clean: true`). The commit that adds these records changes only `reports/w04/quality_gates.json`, `reports/w04/docker_airflow_checks.json` and this file.

| Gate | Result |
|---|---|
| ruff | pass |
| mypy (`src/`, `fleet_simulator.py`) | pass, 71 files |
| pytest + coverage | **148 passed** (123 Weeks 1–3 + 25 Week 4, including 5 new `data/sample` tests), coverage of `ingen_fleet_health` 58 % |
| `uv lock --check` | pass |
| Docker build (app + airflow) | pass; app image 3.55 GB (6.19 GB before `UV_NO_CACHE`) |
| App image smoke | pass |
| Airflow DAG import (3.3.2) | pass, no import errors |
| gitleaks v8.30.0 (full history) | pass, 28 commits, no leaks |
| Fresh-clone README quickstart at f21f2ee | pass: build, MLflow, trainer (144 s), dashboard, DAG import, idempotent re-run (12/12 partitions identical), missing partition (sensor timeout, outputs untouched, failure record, run failed), backfill (`docker_airflow_checks.json`) |
| v1 retrain reproducibility | identical at 5bea8fc, clean tree (`retrain_reproducibility.json`) |
| GitHub Actions CI | **not run**: nothing pushed |

**Week 4 demo video: not finished.** `media/W04_demo_draft.mp4` (54 s, git-ignored) is a draft of a real recorded run: Airflow is shown in the host's Chinese locale and local time zone, captions overlap at two scene boundaries, and static pages are held without a "static" label. Later takes that fixed this stalled on static pages; that recorder is kept in `git stash` (not committed). Kuntong decides how to finish it.

## Not done / needs Kuntong or the supervisor

- **Push, PR, GitHub Release v1.0.0:** not done (needs approval). CI and its badge have not run on GitHub.
- **InGen AI-use policy:** confirm it.
- **LICENSE holder and IP:** `LICENSE` names "Kuntong Li" (MIT). Not changed; awaiting InGen's answer.
- **Product names:** confirm whether further InGen product names may be published before adding them back.
- **Retrospective reflection, signed rubric, career note:** for the named people to complete.
- **`weekly/Wk-04-Log.md`:** drafted by the AI assistant in the first person; Kuntong to review or rewrite it in their own words if the policy requires.
- **Real cost figures:** `configs/cost_assumptions.yaml` holds placeholders.
