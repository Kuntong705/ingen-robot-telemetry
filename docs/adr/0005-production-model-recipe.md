# ADR-0005 — Production (v1) Model Recipe: Champion, Shadow Models, Registry

**Status:** Accepted (Week 4, 2026-11-24)  
**Code:** `src/ingen_fleet_health/release/train.py`, `release/bundle.py`, `configs/pipeline.yaml` (`release_training`, `models`)

## Context

The daily pipeline has to score every robot with a model that is loaded from a versioned MLflow artifact. Weeks 2–3 left an awkward set of evidence. All PR-AUC figures below are from leakage-safe evaluation on unseen robots, with a base rate of 0.055.

**Aido Rover:**
- The Isolation Forest ranks best (0.330). It is significantly better than both the physics rule and the selected LightGBM.
- The physics rule (0.215) and the calibrated LightGBM (0.210) are not distinguishable.
- At the operator budget of one alert per unit-week, the rule and the Isolation Forest have the same precision (0.092).

**Fari:**
- The Isolation Forest transfers poorly (0.077 as-is, 0.142 fixed).
- The Rover LightGBM with feature guards reaches 0.157. A pooled Rover+Fari LightGBM reaches 0.216 on Fari test units; that pooled gain is not significant.

**Text and explanations:**
- Text adds no detectable gain: early fusion minus LightGBM is +0.016, 95 % CI −0.040 to 0.078.
- Only the template summaries were faithful (20/20). The local LLM was faithful in 0/20.

## Decision

1. **Champion:** one pooled, telemetry-only LightGBM with a platform flag.
   - It is trained on both platforms' training units.
   - The feature set (`lgbm_all` or `lgbm_physics`) is chosen on validation PR-AUC, the Week 2 rule.
   - It is calibrated on the validation slice.
   - It alone raises alerts. Its reasons are calibrated probabilities, exact tree SHAP explanations that the faithful template can verbalise, and the best measured Fari transfer.
2. **Shadow models (challengers):** the Isolation Forest and the physics rule.
   - Both are scored on every row and stored in the predictions table, but never alert.
   - The weekly performance check reports their PR-AUC next to the champion's on newly confirmed faults, so a switch can be made on evidence.
   - Deploying the Isolation Forest now would trade the explanations and the Fari transfer for a Rover-only ranking gain.
3. **Per-platform alert thresholds** are chosen on validation rows for the budget of 1 alert per unit-week, with a 72 h refractory period (Week 3: Fari needs its own threshold).
4. **SoH forecast:** a pooled LightGBM on the residual SoH(d+7) − persistence, as in Weeks 2–3.
5. **Training window and cohorts.**
   - 70 % of each platform's units are training units, fault-stratified and seeded.
   - Training rows satisfy t ≤ start + 30 d − 72 h. Validation is the last 6 days of that, and fit rows end one horizon earlier still.
   - The other 30 % of units are never seen. They are the headline cohort of monitoring.
   - Dates before the cut-off are never scored (`InSampleDate` guard). The history load only ingests and builds features for them.
6. **Registry.**
   - Each model is a joblib bundle (estimator, feature list, thresholds, PSI references, training-unit list) wrapped as an MLflow pyfunc and registered as `fleet_early_warning` and `fleet_soh_forecast`.
   - `configs/pipeline.yaml` pins the version. A retrain creates a new version, and serving changes only when the pin is changed.

## Consequences

- On Rover, the v1 alert ranking is expected to be weaker than the Isolation Forest shadow. This is stated in the model card, not hidden.
- Production numbers on the operational period are a different protocol from the Week 2 walk-forward folds: a single time cut and a single unit split. They are reported separately and do not replace the Week 2–3 numbers.
- The validation slice is small (6 days). Thresholds and calibration are therefore noisy. A retrain with more history is the first maintenance action.
