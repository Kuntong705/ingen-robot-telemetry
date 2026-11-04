# ADR-0004 — Leakage-Safe Evaluation Protocol

**Status:** Accepted (Week 2, 2026-11-04)  
**Context documents:** `W02_Evaluation_Protocol.docx`, `docs/W01_Data_Contract.md` (ADR-003 rev1)

## Context

Week 2 trains two model families on a synthetic fleet: 72-hour fault early warning (hourly prediction times) and battery state-of-health forecasting (daily, 1/7/14-day horizons). Consecutive samples of one robot are strongly autocorrelated, the 72-hour labels of neighbouring hours overlap, and each robot has its own baseline (capacity, resistance, vibration, thermal coefficient). A random row split would put near-duplicate rows of the same robot and the same fault window on both sides of the split.

## Decision

Every reported metric comes from **unit-grouped walk-forward evaluation with a purge/embargo of at least one prediction horizon**:

1. **Unit groups.** The 50 robots are split into 5 groups of 10, after a seeded shuffle, round-robin within strata of fault count, so each group mixes faulty and healthy robots. A group is either training or test in a fold, never both.
2. **Walk-forward cut-offs.** Cut-offs at simulation days 20, 30, 40 and 50; each test window is the 10 days after its cut-off. Training uses an expanding window of everything before the cut-off.
3. **Purge and embargo.** Training rows must satisfy `t <= cutoff − H`, so every training label window ends before the test period starts. Test rows start at `t = cutoff`. Gap ≥ H, with H = 72 h for early warning and H = h days for SoH.
4. **Inner validation.** The last 6 days of each fold's training period, purged again by H, hold the validation rows. They are used for model selection between pre-declared variants, probability calibration and the alert threshold. Test rows are never used for any fitted choice.
5. **Fitted preprocessing inside the fold.** Physics normalisation uses fixed config priors. Anything learned, such as the environment residualiser, scalers or calibration maps, is fitted on the fold's training or validation rows only.
6. **Pooling.** 4 cut-offs × 5 groups = 20 folds. Each test (unit, hour) is scored exactly once. Metrics are computed on the pooled out-of-fold predictions, with 95 % confidence intervals from a unit-level cluster bootstrap.
7. **Random split only once.** A random 80/20 row split is run in exactly one place, the leakage demonstration, to measure how far it overstates performance.

## Alternatives considered

- **Single time cut, same robots.** This is simpler, but the test robots are seen in training, so the model can memorise each robot's baseline. It is measured as the middle protocol in the leakage demonstration.
- **GroupKFold without time ordering.** This removes unit memorisation, but it still trains on the future relative to the test period: fleet-wide weather and the calendar distribution of faults leak across units.
- **Leave-one-unit-out.** This gives 50 folds per cut-off. It costs more and produces noisier per-fold metrics; with only 41 in-run fault onsets, groups of 10 keep events in most test folds.

## Consequences

- Early cut-offs train on few events (about 58 positive hours at day 20), which is pessimistic for supervised models. We report this rather than dropping the folds.
- The protocol ignores the first 20 days for testing. Results describe a fleet that already has some failure history.
- The per-fold alert threshold makes the alert budget a validation-time decision, which is how an operator would deploy it.
- `src/ingen_fleet_health/evaluation/splits.py::check_fold` asserts unit disjointness, no duplicate keys and the embargo on every fold of every run. A violation raises an error instead of producing a number.
