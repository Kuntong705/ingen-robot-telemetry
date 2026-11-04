# Week 02 — Engineering Log

**Project:** Robot Fleet Telemetry Analytics & Predictive Health  
**Period:** November 2 – 8, 2026 (Week 2 scope, demo November 9)  
**Topic:** The leakage risk most likely to have slipped through, and how it was ruled out

## The risk: the calendar, not the robot

The obvious leaks were handled by construction. All windows are trailing. Splits are grouped by robot. A 72-hour purge separates the label windows. Scalers and calibration maps are fitted inside each fold. Each of these is enforced by a test or by `check_fold`. The leak I was most worried about was quieter. Features that only measure *how long the fleet has been running* could predict the label without describing a machine at all.

The audit of simulator v0.1 showed how bad this can get. All 20 robots had their fault onset at the same instant, 2026-11-19 00:00 UTC. A LightGBM given only "hours since start" reached PR-AUC 0.9999 under a random split. I rebuilt the simulator (v0.2) so that every robot draws its own fault timeline. But the artefact did not fully disappear. Initiation times are uniform, onset follows after a 2–10 day precursor, and windows at the end of the run are censored. Together these push positive hours late in the run. On eligible rows, the odometer alone still had a univariate AUC of 0.69. Odometer, cumulative equivalent full cycles and charge-session counts are legitimate physical exposures, so they could easily have stayed in the model unnoticed.

## Alternatives considered

1. **Keep the time proxies.** Wear really does accumulate with use. But in this simulator fault hazard does not depend on usage, so any skill they add would be skill at reading the calendar.
2. **Drop them silently.** That is safe, but nothing would be measured: we could not say whether the protocol itself would have stopped the leak.
3. **Exclude them from the production model and measure them with a probe.** I chose this option.

## What was done

I trained a probe model on the four time proxies plus elapsed hours, under three protocols:

| Protocol | Probe PR-AUC | Main model PR-AUC |
|---|---|---|
| Random 80/20 shuffle | 0.81 | 1.00 |
| Time-ordered + purge, same robots | 0.07 | 0.36 |
| Unit-grouped walk-forward + 72 h embargo | 0.055 | 0.26 |

The positive base rate of the test rows is 0.055. Under the proper protocol the probe falls exactly to no-skill, so the calendar carries no exploitable label information once training data ends 72 hours before each test window. The gap between the middle row and the last row (0.36 vs 0.26) is unit memorisation, which grouping by robot removes. The random shuffle overstates PR-AUC by 0.74. That is the measured gap the protocol exists to prevent.

## What I would check next

The probe rules out a calendar leak on *this* simulator. On real fleet data the hazard of wear faults does grow with usage, so time proxies may become legitimate. The same probe should be rerun on any new data before deciding whether to include them.
