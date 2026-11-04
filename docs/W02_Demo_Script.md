# W02 Mid-Point Demo — 5-minute script (Monday 2026-11-09)

Every number below comes from `reports/w02/*.json`. Re-check them after any rerun.

**Before the call (open in this order):**
1. `docs/figures/w02/sim_v02_sanity.png`
2. `docs/figures/w02/leakage_demo.png`
3. `docs/figures/w02/soh_mae_by_horizon.png`
4. `docs/figures/w02/ew_budget_curve.png`, then `ew_unit_timeline.png`
5. Browser tab: GitHub Actions run for the `week2` branch (after push)
6. Terminal: `py -3.11 -m uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`, then a browser tab at http://127.0.0.1:5000 (experiments `w02_early_warning`, `w02_soh_forecast`, `w02_leakage_demo`)

---

**0:00–0:40 — What changed since Week 1 (figure 1)**
"Auditing the Week 1 simulator, I found that all 20 robots had identical battery trajectories and every fault started at the same minute. A model that only knew the clock scored PR-AUC 1.0. So I built simulator v0.2: per-robot parameters, three site climates, route terrain, and P-F fault timelines. A fault degrades silently for 2–10 days before its qualifying onset, which is the label event. I use 50 robots over 60 days, 4.3 million records, and the Week 1 ingest accepts all of them. The panels show discharge, heating with load, capacity fade that is faster in hot climates, and bearing vibration rising before onset."

**0:40–1:40 — Leakage demonstration (figure 2)**
"Same LightGBM, same features, three protocols. With a random shuffle, PR-AUC is 0.999. With time ordering on the same robots, it is 0.36. With robots held out and a 72-hour embargo, it is 0.26. The shuffle overstates performance by 0.74. The orange bars are a probe that sees only time-proxy features like the odometer. Under the proper protocol it drops to 0.055, exactly the base rate, so the calendar carries no exploitable signal. Every fold is checked automatically: zero shared robots, zero duplicate rows, embargo at least 72 hours."

**1:40–2:40 — Battery SoH (figure 3)**
"I forecast true SoH 1, 7 and 14 days ahead on unseen robots. LightGBM, trained on the residual over the BMS estimate, has the lowest error at every horizon: MAE 0.26, 0.31 and 0.41 percentage points. It is significantly better than persistence at all horizons and better than linear extrapolation at 7 and 14 days; at 1 day the difference is not significant. The LSTM did not beat linear extrapolation at any horizon, and I report that as it is."

**2:40–3:50 — Fault early warning (figures 4 and 5)**
"The early-warning target is a qualifying onset within 72 hours, with a budget of one alert per robot per week. The honest result: the best model is the unsupervised Isolation Forest, with PR-AUC 0.33 against 0.21 for LightGBM, and that gap is significant. At the budget, the physics rule warns 27 of 31 onsets with a 30-hour median lead and the Isolation Forest 25 of 31 with 32 hours. LightGBM warns only 13. Its precision is slightly better, but all models sit around 0.1. Why does the classifier lose? There are few training onsets per fold, and it misses fault types it never saw; with bearing faults held out it caught 1 of 11. One thing it does well: only 3 % of its false alarms come from sensor-drift robots, against 20 % for the anomaly detector. So the next step I propose is a two-stage alert: the anomaly detector raises it and the classifier filters drift."

**3:50–4:30 — CI and MLflow (browser)**
"CI runs ruff, mypy and pytest on every push: 56 tests, including an as-of test that corrupts all future data and checks the features do not change. In MLflow, every number in the results document links to a run."

**4:30–5:00 — Difficulties and remaining work**
"Difficulties: the label had to be redefined so it is learnable, the first resistance estimator was biased by 15 %, and the first classifier memorised robots. On the public NASA C-MAPSS FD001 benchmark, the same pipeline gets a test RMSE of 13.1 cycles with LightGBM and 14.4 with the LSTM, against 37–42 for the naive baselines. That is inside the published range, between the Deep LSTM's 16.1 and the DCNN's roughly 12.6. It is a different task, remaining life in cycles, so it validates the method, not the SoH numbers. Remaining work: a two-stage alert; and calibration, which did little here (Brier 0.0532 to 0.0530). Then Week 3: maintenance-log text and Fari."

---

## Likely questions — short answers

- **Why is LightGBM worse than an unsupervised model?** It has few labelled onsets per fold (3–30 across three fault types), conservative validation thresholds, and a label that only marks the last 72 hours. Detecting a deviation from each robot's own baseline needs no fault examples.
- **Is 0.26 PR-AUC good?** The base rate is 0.055, so it is about 5× no-skill. In operator terms the anomaly detector warns about 80 % of onsets, roughly 30 hours ahead, at about 1 false alarm per robot-week.
- **Why trust the proper number?** Robots are disjoint, a 72-hour embargo is asserted on every fold, and the time-proxy probe falls to the base rate.
- **Did you tune on test?** No. Hyper-parameters are fixed, and the variant choice, calibration and thresholds all use inner validation. The round-1 run is kept in MLflow, and the round-2 variant is disclosed.
- **Why C-MAPSS instead of the NASA battery data?** It has an official held-out test set and widely cited baselines, so it tests whether the protocol produces realistic numbers. Its metric is RMSE in cycles of remaining life, not SoH percentage points, so it is never put in the same table as the SoH results.
- **What depends on the simulator?** Everything quantitative. Fade rates, fault signatures and noise levels are assumptions in `configs/aido_rover_v0_2.yaml`, so all results need re-validation on real fleet data.
