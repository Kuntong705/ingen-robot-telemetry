# Week 3 Progress Tracker (living file — read this first in a new session)

**Branch:** `week3` (created from `week2` at 382bdee on 2026-11-11; local only — do not push, merge to `main` or open a PR without Kuntong's approval)  
**Plan:** Internship Plan v3.0, Week 3 — Multimodal Fault Classification, Interpretability & Fari Generalisation  
**Last updated:** 2026-11-12 ~04:00 local (session 2, Opus 5.5). **Week 3 deliverables complete and committed locally** (not pushed). Open items are listed under "Open after session 2" at the bottom.

## Standing constraints (from Kuntong, 2026-11-11)

1. No leakage: text, maintenance events and model features use only information that exists at prediction time; test robots never enter training.
2. The GitHub repository is public. No company-confidential material, Dropbox internals, real customer data or keys. Dropbox internal files are not read.
3. Fari hardware parameters cannot be verified: everything in `configs/fari.yaml` is a **synthetic assumption**, never a real specification.
4. No paid external LLM API calls and no company material sent to external services without approval.
5. Commit locally only. No push, no merge to `main`, no PR.
6. Decide routine engineering autonomously; ask only for major statistical disagreements, destructive operations, confidential material, extra cost, or push.

## Design decisions (made autonomously)

| # | Decision | Why |
|---|---|---|
| D1 | Reuse `data/w02` (50 Rover units × 60 days) and the same 20 walk-forward folds | Fusion deltas must be comparable with Week 2 on identical test rows |
| D2 | A ticket's label may be used for training only if `label_available_utc <= cutoff − 72 h`; a ticket is visible to features only if `created_at_utc <= t` | The category is confirmed at triage/closure, not at creation |
| D3 | Operator notice probability (`notice_rate`, `notice_exponent` in `configs/maintenance_logs.yaml`) was fixed before any fusion result, with a sensitivity sweep (0.5×, 2×, 4×) | On synthetic text the value of the text modality is assumption-driven |
| D4 | Text signal = as-of aggregates of per-ticket class probabilities; the ticket classifier is fitted inside each fold and cross-fitted over training unit groups | No in-sample text scores on training rows; no test-unit ticket in training |
| D5 | No hosted LLM. Paraphrase is rule-based. Summaries: deterministic template generator + local open-weights `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, downloaded to the Hugging Face cache, ~3 GB, runs on CPU) | Constraint 4 |
| D6 | Fari minimal change = platform profile for ingestion + config-driven active mode / feature guards + per-platform alert threshold. Rover output verified byte-identical before/after (simulator, ingest, 154-column feature table) | "Smallest change that closes the most important gap" |

## Status

| Phase | Status | Notes |
|---|---|---|
| P0 branch + deps | done | `sentence-transformers`, `shap` added with uv (`pyproject.toml`, `uv.lock`) |
| P1 log corpus | code done, generated | `text/{templates,noise,corpus}.py`, `configs/maintenance_logs.yaml`; 1,371 synthetic tickets in `data/w02/text/` (git-ignored) |
| P2 text classifiers | code done; **full run not confirmed** | `models/text_classifier.py`, `experiments/text_classifier_run.py`. First full run crashed in the final bootstrap (fixed). A rerun was started at ~19:36 (log `runs/w03_text_classifier.log`); check whether `reports/w03/text_classifier_results.json` exists |
| P3 fusion + ablation | code done; **only smoke-tested (5 folds, reduced trees)** | `text/features.py`, `models/fusion.py`, `experiments/fusion_run.py`. Full run not started |
| P4 SHAP + summaries + faithfulness | code written; **checker has known false flags, not final** | `explain/{shap_utils,alert_summaries,faithfulness}.py`, `experiments/explain_run.py`. Local LLM not run yet |
| P5 Fari | simulator/contract/feature changes done; Fari fleet simulated, ingested (100 % accepted with the Fari profile), features built; **transfer experiment never run** | `configs/fari.yaml`, `contract.py`, `evaluation/shift.py`, `experiments/fari_transfer_run.py`; `data/w03_fari/features_asis` build was started (log `runs/w03_fari_features_asis.log`) |
| P6 docs, tests, gates | **not done** | `tests/w03/test_log_corpus.py`, `test_text_models.py` pass (18 tests). No tests yet for fusion, explain, Fari contract. No docx/md deliverables, no `Wk-03-Log.md`, no README section |

## Known defects to fix before any result is reported

1. `explain/faithfulness.py`: polarity check uses the whole sentence; the word "offset" inside a feature phrase ("docked motor-temperature offset") makes every feature in that sentence read as "lowering the risk". Evaluate polarity with feature phrases removed.
2. `explain/faithfulness.py`: the phrase "recent operator notes read as '<class>'" (a *feature* phrase) is parsed as a classifier-reading claim. Match feature phrases before readings, or rename the phrase.
3. `explain/alert_summaries.py` (real faithfulness problems found by reading smoke output, not checker artefacts): (a) the third sentence quotes the newest note as "among these drivers" even when a different note carries the text attribution; link the note whose class matches the attributed `txt_decay_<class>` feature, or do not quote; (b) values such as "1.00 x baseline" are presented as pushing risk up without saying the deviation is negligible.
4. `reversed_direction` negative control was flagged only 17/20 and `invented_cause` 19/20 in the smoke run; investigate after 1–2.
5. `fari_transfer_run.py`, `fusion_run.py --sensitivity` and `explain_run.py --llm local` have never been run end to end on the full data.

## Reproduce (from repo root; times are estimates)

```bash
py -3.11 -m uv sync
py -3.11 -m uv run python -m ingen_fleet_health.text.corpus --data-dir data/w02                      # 10 s
py -3.11 -m uv run python -m ingen_fleet_health.experiments.text_classifier_run --data-dir data/w02  # ~30 min (--reuse-oof to redo metrics only)
py -3.11 -m uv run python -m ingen_fleet_health.experiments.fusion_run --data-dir data/w02 --sensitivity   # not yet run in full
py -3.11 -m uv run python -m ingen_fleet_health.experiments.explain_run --data-dir data/w02 --llm local    # after fusion_run; not yet run in full
py -3.11 -m uv run python -m ingen_fleet_health.simulate.fleet --out-dir data/w03_fari --config configs/fari.yaml
py -3.11 -m uv run python ingest_pipeline.py --input data/w03_fari/generated/fleet_telemetry.csv --out-root data/w03_fari --platform-config configs/fari.yaml
py -3.11 -m uv run python -m ingen_fleet_health.features.build --data-dir data/w03_fari --config configs/fari.yaml
py -3.11 -m uv run python -m ingen_fleet_health.features.build --data-dir data/w03_fari --out-subdir features_asis   # Week 2 config as-is
py -3.11 -m uv run python -m ingen_fleet_health.experiments.fari_transfer_run --rover-dir data/w02 --fari-dir data/w03_fari   # not yet run
```

## Remaining work, in order

1. Fix defects 1–4; add tests `tests/w03/test_fusion.py`, `test_explain.py`, `test_fari_contract.py`.
2. Confirm/redo the full text classifier run; run fusion (with sensitivity), explain (with local LLM), Fari transfer.
3. Read the 20 summaries against their evidence packs by hand; report automatic and manual faithfulness separately.
4. Deliverables: `docs/W03_Log_Corpus.md`, `docs/W03_Fusion_Results.docx`, `docs/W03_Alert_Explanations.md`, `docs/W03_Fari_Gap_Analysis.docx`, `weekly/Wk-03-Log.md` (300–500 words), README Week 3 section, data-contract update (log schema, platform profiles, Fari privacy rule), `scripts/build_w03_docs.py` generating every number from `reports/w03/*.json`.
5. Gates: pytest, ruff, mypy, `uv lock --check`; code review; final local commit.

## Not done / needs Kuntong

- Model/effort: the session ran on Fable 5.2 (a session cannot switch its own model); pick Opus 5.5 in the model menu if that is required.
- LLM paraphrase of the corpus and a hosted-LLM summary run (Claude API / Vertex AI) are not done: they need a key, cost approval and a data-handling decision.


## Session 2 log (2026-11-12, Opus 5.5)

- Text classifier full run **finished** on 2026-11-11 19:48 (`reports/w03/text_classifier_results.json`, MLflow `3da22b63425f49d1b18ca1725acb1813`, log `runs/w03_text_classifier.log`). No rerun needed.
- Fari features `features_asis` **finished** (70,800 x 154).
- Defects 1–4 fixed: checker matches feature phrases first and replaces them with markers (no "offset"/"lowest" cues from inside a phrase; no reading parsed from a feature phrase); direction read per contrast segment; "notes are a driver" judged by the attribution; template quotes only the note that feeds the attributed text feature (`note_behind`); ratio features within 2 % of 1 are qualified "essentially at the reference level"; `invented_cause` corruption no longer picks a concept supported by a `txt_decay_<class>` attribution (that was the 19/20).
- Privacy gate bug fixed: substring matching refused `surface_temp_c` (face) and `dynamic_load` (mic); now word-prefix matching (still conservative: `framework_version` is refused, documented in a test).
- New tests: `tests/w03/test_explain.py`, `test_fusion.py`, `test_fari_contract.py` — 62 W03 tests pass.
- Running (started ~session start): `fusion_run --sensitivity` (log `runs/w03_fusion.log`), `fari_transfer_run` (log `runs/w03_fari_transfer.log`). Next: `explain_run --llm local` after fusion finishes.
- Fari transfer run 1 finished (MLflow `22a1519de3c443adad685b756d60b92f`, superseded): contract acceptance 0 % as-is -> 100 % with profile; 12 dead features revived; LightGBM PR-AUC 0.046 -> 0.157; physics rule got *worse* (0.183 -> 0.092). Added paired PR-AUC deltas, per-indicator rule diagnostics and fault-type mix; **rerun 2 started** (log `runs/w03_fari_transfer.log`).
- New experiment `experiments/paraphrase_run.py` (local Qwen rewrites ~200 test tickets; same fold classifiers). Smoke-tested (reproduces OOF 100 %); full run pending (after fusion, CPU).
- Data contract updated: §3.5 log schema, §3.6 platform profiles + Fari privacy rule (`docs/W01_Data_Contract.md`).
- `scripts/build_w03_docs.py` started: `build_log_corpus()` done -> `docs/W03_Log_Corpus.md`.
- Full pytest 119 passed; ruff + mypy clean (as of ~01:10).
- `scripts/build_w03_docs.py` complete (corpus, fusion, explanations, Fari, overall report `docs/W03_Report.docx`); prose verdicts computed from CIs. `scripts/run_w03_gates.py` records ruff/mypy/pytest/uv-lock into `reports/w03/quality_gates.json`.
- Full paraphrase run started (`runs/w03_paraphrase.log`, OMP 4 threads). Still running: fusion sensitivity (2x, 4x), Fari rerun.
- NEXT: when `reports/w03/fusion_results.json` exists -> `explain_run --data-dir data/w02 --llm local` -> read all 20+20 summaries vs evidence, write `reports/w03/manual_faithfulness.json` ({generator: {alert_id: {faithful, note}}}) and paraphrase `manual_check` -> gates -> build docs -> README, Wk-03-Log -> commit.
- Rover feature regression re-verified in session 2: rebuilt with Week 3 code == Week 2 table (70,800 x 154, DataFrame.equals True; log runs/w03_rover_feature_regression.log).
- **Fusion full run done** (78 min, MLflow `d51fff3518a140309ff8bf90176bd378`, log `runs/w03_fusion.log`). PR-AUC: no-skill 0.055, LightGBM 0.210, IF 0.330, text-only 0.059, early 0.226, late-stack 0.226, late-stack-IF 0.176. Text vs no-skill +0.004 [-0.014,0.034]; early−LGBM +0.016 [-0.040,0.078]; late-stack-IF − IF −0.153 [-0.243,-0.057] (stacker over-weights text in 18/20 folds); early − IF −0.103 [-0.214,-0.007]. Sensitivity early−LGBM: 0.5x −0.011, 1x +0.016, 2x +0.073 [0.002,0.136], 4x +0.130 [0.044,0.230]. Only 9/31 onsets had a note ≤72 h before.
- `experiments/fusion_diagnostics.py` (no refit): per-fold PR-AUC, stacker weights, reliability (scores >0.99 are positive ~73 %: overconfident calibration inherited from Week 2).
- Paraphrase run paused at 47/204 cached rewrites (CPU contention); resume after explain with full threads (cache resumes).
- explain_run --llm local running (log `runs/w03_explain.log`).
- **Explain run done** (51 min, MLflow `b01f56c899314e2c94a1b12a3766eb2b`): SHAP additivity 4e-14, reproduces fusion scores exactly; text = 36 % of mean |SHAP| (use, not value). Checker v1: template 20/20, LLM 3/20; controls 79/79.
- Manual read of all 40 summaries -> `reports/w03/manual_faithfulness.json`: template 20/20, **LLM 0/20** (every summary has a real error). Checker v2 (5 generic false-flag fixes + tests; `explain_run --recheck`, MLflow `f89918c62fd14cd29166177b3f313efd`, v1 numbers kept in `checker_history`): LLM 8/20 automatic, only 6 of 12 flags point at real errors. Conclusion: template for operators; lexical checker insufficient for LLM prose. Checker NOT tuned further (avoid fitting eval to output).
- Docs built: W03_Log_Corpus.md, W03_Fusion_Results.docx, W03_Alert_Explanations.md, W03_Fari_Gap_Analysis.docx (W03_Report.docx needs gates + paraphrase). README §6b with results table. `weekly/Wk-03-Log.md` written (469 words).
- Paraphrase run resumed (all threads), ~60/164 new rewrites at 03:03.
- REMAINING: paraphrase finish -> manual check of 30 pairs (`paraphrase_manual_check` in manual_faithfulness.json) -> `scripts/run_w03_gates.py` -> `scripts/build_w03_docs.py` -> final proofread -> local commit (no push).

- Paraphrase run done (MLflow `794f0ed917eb490ab661e1dbd5fd3246`, 210 tickets): selected macro-F1 0.848 -> 0.851, +0.002 [-0.026, 0.032]; manual check of 30 pairs: 21 kept, 6 detail changed, 3 meaning changed.
- Gates (`reports/w03/quality_gates.json`): ruff pass, mypy pass (53 files), pytest 123 passed (66 in tests/w03), `uv lock --check` pass.
- All five documents rebuilt from JSON by `scripts/build_w03_docs.py`; README §6b and Wk-03-Log (469 words) written. Local commit made on `week3`.

## Open after session 2 (need Kuntong or Week 4)

1. **Review the manual faithfulness read** (`reports/w03/manual_faithfulness.json`): written by Claude, not yet checked by a person. The LLM 0/20 conclusion rests on it.
2. **Hosted LLM** (Claude API / Vertex AI) for summaries and corpus paraphrase: not run; needs key, cost approval, data-handling decision. The local 1.5 B model was not faithful (0/20 manual).
3. **Ask InGen** (no data needed in the repo): what share of real faults had an operator note before onset? The fusion conclusion flips between 1x and 2x of the assumed notice rate.
4. Week 4: telemetry-first scoring (Isolation Forest best), recalibration (scores > 0.99 are positive ~73 %), regularised stacker if text is used, per-platform physics-rule composition chosen on validation data.
5. No 30–60 s demo recording for Week 3 was made in this session (plan asks for one per weekly deliverable).
6. Superseded MLflow runs kept for transparency: Fari run 1 `22a1519de3c443adad685b756d60b92f` (before paired deltas); explain v1 numbers are inside `alert_explanations.json -> checker_history`.
