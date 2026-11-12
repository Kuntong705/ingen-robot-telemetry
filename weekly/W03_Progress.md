# Week 3 Progress Tracker (living file — read this first in a new session)

**Branch:** `week3` (created from `week2` at 382bdee on 2026-11-11; local only — do not push, merge to `main` or open a PR without Kuntong's approval)  
**Plan:** Internship Plan v3.0, Week 3 — Multimodal Fault Classification, Interpretability & Fari Generalisation  
**Last updated:** 2026-11-11 ~19:45 local. **Session stopped early (usage limit). Week 3 is NOT finished.** No final results exist yet; every number seen so far came from reduced smoke runs and must not be reported.

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
