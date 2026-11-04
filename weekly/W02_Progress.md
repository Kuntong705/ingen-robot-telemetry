# Week 2 Progress Tracker (living file — read this first in a new session)

**Branch:** `week2` (local, nothing committed yet; commit/push only with Kuntong's approval)  
**Plan:** Internship Plan v3.0, Week 2; demo Monday 2026-11-09  
**Last updated:** 2026-11-04 (all P0–P7 work done; P8 run pending data)

## Decisions confirmed by Kuntong (2026-11-04)

1. Simulator v0.2 is a new module (`src/ingen_fleet_health/simulate/`); v0.1 and its 30 tests are unchanged.
2. Label: P-F timeline, "qualifying onset within 72 h" (ADR-003 rev1 in `docs/W01_Data_Contract.md`).
3. Dataset: 50 units × 60 days (`data/w02/`, git-ignored).
4. Dependencies installed. **No external download yet** (C-MAPSS).
5. Working mode: decide routine engineering autonomously; ask only for major stats disagreements, destructive changes, confidential data, cost, or push.

## Reproduce (from repo root)

```bash
py -3.11 -m uv run python -m ingen_fleet_health.simulate.fleet --out-dir data/w02 --units 50 --days 60      # ~2 min
py -3.11 -m uv run python -c "from ingest_pipeline import run_pipeline; run_pipeline('data/w02/generated/fleet_telemetry.csv', out_root='data/w02')"   # ~3 min
py -3.11 -m uv run python -m ingen_fleet_health.features.build --data-dir data/w02                          # ~2 min
py -3.11 -m uv run python -m ingen_fleet_health.experiments.leakage_demo --data-dir data/w02                # ~2 min
py -3.11 -m uv run python -m ingen_fleet_health.experiments.soh_run --data-dir data/w02                     # ~12 min
py -3.11 -m uv run python -m ingen_fleet_health.experiments.early_warning_run --data-dir data/w02           # ~7 min
py -3.11 -m uv run python -m ingen_fleet_health.simulate.sanity_plots --data-dir data/w02
py -3.11 -m uv run python -m ingen_fleet_health.features.registry
py -3.11 -m uv run python scripts/build_w02_docs.py
py -3.11 -m uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Status

| Phase | Status | Key output |
|---|---|---|
| P0 infra | done | package, uv deps (torch CPU index), CI mypy on `src/` |
| P1 simulator v0.2 + labels | done | 4,296,455 rows, 45 faults (41 in run), 100 % ingest acceptance; `docs/figures/w02/sim_v02_sanity.png` |
| P2 MLflow | done | `mlflow.db` + `mlruns/` (git-ignored) |
| P3 features + registry | done | 152 features; `docs/W02_Feature_Registry.md` |
| P4 protocol + leakage | done | ADR-0004; `W02_Evaluation_Protocol.docx`; leakage run `08531598aeeb44be8a5b03acb611588d` |
| P5 SoH | done | run `26f206a5214540fd86e3491465d08398` |
| P6 early warning | done | final run `c9d96429ec674e4d92f8320e11b99012`; round 1 `ff0eb6f5e2004c22824afb22e089e637` (superseded) |
| P7 docs + demo | done | `W02_Model_Results.docx`, `Wk-02-Log.md` (432 words), `docs/W02_Demo_Script.md`, README section 6a |
| P8 C-MAPSS | code + synthetic test only | needs data-source approval, then `python -m ingen_fleet_health.benchmarks.cmapss --data-dir data/external/cmapss` and rebuild docs |

Quality gates (2026-11-04): pytest 56 passed; ruff clean; mypy clean (32 files); `uv lock --check` ok. CI on GitHub is not yet verified (nothing pushed).

## Results (see reports/w02/*.json)

- Leakage: random shuffle PR-AUC 0.999 vs proper 0.261 (+0.738); the time-proxy probe falls to 0.055 = base rate under the proper protocol; the v0.1 clock-only model scores 1.000.
- SoH MAE (pp) at 1/7/14 d: persistence 0.335/0.445/0.675, linear 0.307/0.385/0.499, LightGBM 0.262/0.313/0.414, LSTM 0.271/0.399/0.515. LightGBM is significantly better than linear at 7 and 14 d; the LSTM never is.
- Early warning PR-AUC: physics rule 0.215, LightGBM selected 0.210, Isolation Forest 0.330 (significantly better than both). At 1 alert/unit-week: rule 27/31 onsets, lead 30 h; IF 25/31, lead 32 h; LightGBM 13/31, lead 27 h.
- About 90 % of false alarms fall on robots with no developing fault. Drift robots get 20 % of rule/IF false alarms but only 3 % of LightGBM's.

## Open items / next steps

- [ ] Kuntong: review the docx files; approve commit + push of `week2` (then check CI) and the PR
- [ ] Kuntong: confirm the C-MAPSS download from the NASA Prognostics Data Repository → run P8 and rebuild the docs
- [ ] Kuntong: record a 30–60 s demo clip (Plan working norms); rehearse `docs/W02_Demo_Script.md`
- [ ] Candidate follow-up: two-stage alert (IF raises, LightGBM filters drift); nested threshold tuning
- [ ] Deferred: pandera schema, Week 1 scripts into `src/`, autoencoder, Fari profile (Week 3)
