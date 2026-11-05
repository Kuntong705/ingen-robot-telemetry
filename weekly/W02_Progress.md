# Week 2 Progress Tracker (living file — read this first in a new session)

**Branch:** `week2` (pushed to origin at 2304f70; do not merge to `main` or open a PR without Kuntong's approval)  
**Plan:** Internship Plan v3.0, Week 2; demo Monday 2026-11-09  
**Last updated:** 2026-11-04 (P0–P8 done; Week 2 committed locally as e90d892 + C-MAPSS follow-up; not pushed)

## Decisions confirmed by Kuntong (2026-11-04)

1. Simulator v0.2 is a new module (`src/ingen_fleet_health/simulate/`); v0.1 and its 30 tests are unchanged.
2. Label: P-F timeline, "qualifying onset within 72 h" (ADR-003 rev1 in `docs/W01_Data_Contract.md`).
3. Dataset: 50 units × 60 days (`data/w02/`, git-ignored).
4. Dependencies installed. C-MAPSS download approved and done on 2026-11-04 (official NASA PCoE link only; no company-internal sources).
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
| P8 C-MAPSS | done (2026-11-04) | FD001 from the NASA PCoE official link (git-ignored `data/external/`); MLflow run `7399caffc1384d319be348f606fdf6c6`; superseded runs `a5246ae4b7584328924504a3c2740f48` (operational settings wrongly counted as sensors) and `27cffa0998684fdb8977dc7fc6bc9fe7` (single seed) |

Quality gates (2026-11-04): pytest 57 passed; ruff clean; mypy clean (32 files); `uv lock --check` ok. **GitHub Actions: passed.** Run #11 on `week2` at commit 2304f70 was green (confirmed by Kuntong on 2026-11-04). `week2` is pushed to origin; it is not merged to `main`, and no PR has been opened.

## Results (see reports/w02/*.json)

- Leakage: random shuffle PR-AUC 0.999 vs proper 0.261 (+0.738); the time-proxy probe falls to 0.055 = base rate under the proper protocol; the v0.1 clock-only model scores 1.000.
- SoH MAE (pp) at 1/7/14 d: persistence 0.335/0.445/0.675, linear 0.307/0.385/0.499, LightGBM 0.262/0.313/0.414, LSTM 0.271/0.399/0.515. LightGBM is significantly better than linear at 7 and 14 d; the LSTM never is.
- Early warning PR-AUC: physics rule 0.215, LightGBM selected 0.210, Isolation Forest 0.330 (significantly better than both). At 1 alert/unit-week: rule 27/31 onsets, lead 30 h; IF 25/31, lead 32 h; LightGBM 13/31, lead 27 h.
- C-MAPSS FD001 (RUL in cycles, capped truth): constant 41.94, age-linear 37.00, LightGBM 13.09 (3 seeds 13.07 ± 0.01), LSTM 14.38 (14.29 ± 0.20); engine-grouped CV 13.03 vs random-row CV 9.13. The published CNN 18.45 / Deep LSTM 16.14 come from secondary sources; DCNN ≈ 12.6 is unverified.
- About 90 % of false alarms fall on robots with no developing fault. Drift robots get 20 % of rule/IF false alarms but only 3 % of LightGBM's.

## Open items / next steps

- [x] Commit + push of `week2`; CI run #11 green
- [ ] Kuntong: review the docx files; decide on merge/PR later
- [x] 50 s demo video `media/W02_demo.mp4` (git-ignored, local only): generated from the live output of `scripts/w02_demo.py` (run 2026-11-04T19:42:14Z) by `scripts/render_demo_video.py`. It is **not a screen recording**, and this is stated on screen. Regenerate with `python scripts/w02_demo.py` and then `python scripts/render_demo_video.py`.
- [x] 55 s **real screen recording** `media/W02_screen_recording.mp4` (git-ignored, local only): FFmpeg gdigrab captures only the dedicated "W02 live demo" console window during a real run of `scripts/w02_demo.py` (run 2026-11-04T20:26:53Z, log `media/w02_screen_run_log.json`). Recorder: `python scripts/record_screen_demo.py`. Checked: 1360x772 at 15 fps, full output visible, no overlay frames.
- [x] Video decision (Kuntong, 2026-11-04): keep both. `W02_screen_recording.mp4` is the real screen recording the Plan asks for; `W02_demo.mp4` is the rendered visual walkthrough. Both stay local only.
- [x] 47 s **live-dashboard screen recording** `media/W02_live_dashboard_recording.mp4` (local only): FFmpeg gdigrab captures only the "W02 live dashboard" window while `scripts/w02_live_dashboard.py` computes one fold in a worker process (run finished 2026-11-04T21:06:34Z; computed in 16.8 s). Panels mark live one-fold values vs saved 20-fold values. Recorder: `python scripts/record_live_dashboard.py`.
- [ ] Kuntong: rehearse `docs/W02_Demo_Script.md`
- [x] **Final desktop recording (2026-11-04)**: `media/W02_desktop_recording_final.mp4` (59.45 s, 1920×1080, 20 fps; local only) from `media/W02_desktop_raw_final.mkv`; edit log in `media/W02_desktop_recording_final.edit.json`. Changes vs the first version: (1) typing at an ordinary, uneven pace (log-normal key gaps, mean 0.12 s, slightly longer after `_`/`.`; 0.5–1.3 s pauses between actions; eased mouse moves; no fake typos); (2) GitHub segment: shorter switch, the static page-load wait after the click shortened (0.6 s kept), loaded Success page held 4.0 s; the worker start-up wait is played at 10× (labelled) so the total stays ≤ 60 s; (3) dashboard now shows the report's LightGBM (`lgbm_selected`, 20-fold PR-AUC 0.210) instead of the physics-only variant (`lgbm_physics`, 0.230) — same 20-fold unit-grouped walk-forward protocol, different variant; the demo fold c1205_g2 selects the physics variant in both the formal run and the live run, so the live fold PR-AUC (0.438) is unchanged. Desktop pacing profile slightly faster (presentation only). Privacy fix found while checking: Edge's hover status bar shows the link URL with the username, and the Forward button shifts the URL bar — both now blurred. **The earlier `W02_desktop_recording.mp4` shows the username in the status bar for ~1.5 s (54.25–55.75 s) — do not share it.** Decision (Kuntong, 2026-11-05): keep that file unchanged; no corrected re-render. Use `W02_desktop_recording_final.mp4` for sharing.
- [x] **Desktop demo recorded (2026-11-04)**: `media/W02_desktop_recording.mp4` (59.0 s, 1920×1080, 20 fps; local only, git-ignored). `scripts/desktop_demo.py record` drives PyCharm and Edge with real SendInput events (Ctrl+Shift+N opens `models/early_warning.py` and `scripts/w02_live_dashboard.py`, Ctrl+Shift+F10 runs the dashboard — a real one-fold evaluation, no full retrain — then Edge is brought forward and run #11 is clicked) while FFmpeg ddagrab records the desktop. A watchdog stopped and deleted the raw file on every attempt where another window came to the front (Claude app, Alt+Tab task switcher); Alt+Tab was replaced by a direct window switch because the switcher shows thumbnails of all windows. `scripts/render_desktop_recording.py` makes the only two edits, both disclosed: privacy blur (PyCharm run-console path, GitHub username in URL/breadcrumb/run list/run page, GitHub and Edge avatars) and the ~13 s worker start-up wait played at 3× with an on-screen "3x speed" label. Checks: duration 59.05 s; frames inspected at all stages; freezedetect shows two ≥4 s static stretches (worker start-up inside the 3× segment; final dashboard takeaway hold). PyCharm's git-blame inlay shows the author name "Kuntong Li" (left visible). Fixes during recording: keyboard INPUT type (was 0 = mouse), click-to-focus before shortcuts, Edge maximised on switch.
- [ ] Candidate follow-up: two-stage alert (IF raises, LightGBM filters drift); nested threshold tuning
- [ ] Deferred: pandera schema, Week 1 scripts into `src/`, autoencoder, Fari profile (Week 3)
