# ADR-003 — Failure Label Definition

**Proposed target:** A new qualifying fault onset within the next 72 hours.

At prediction time *t*, the binary target is positive if a qualifying fault begins after *t* and on or before *t + 72 hours*.

This choice aims to provide actionable lead time for maintenance teams.

Alternatives considered:

- **Current anomaly detection:** Useful for immediate diagnosis but does not necessarily provide advance warning.
- **Time-to-failure prediction:** Potentially informative but requires a reliable definition of actual failure and sufficient failure observations.
- **Onset-based early warning:** Chosen provisionally because it aligns with advance maintenance decisions.

A proposed 24-hour post-maintenance exclusion window will prevent immediate post-repair observations from distorting training labels.

Records without a complete future observation horizon should be censored rather than automatically labeled negative.

Ground-truth events must never be incorporated into features available at prediction time.

**Implementation status:** The 72-hour predictive target and maintenance exclusions are design decisions, not yet implemented as a production-ready labeling pipeline.

### ADR-003 rev1 (Week 2, 2026-11-04) — qualifying onset in a P-F timeline

*The original decision text above is preserved. This revision makes it implementable and learnable.*

**Problem found in Week 2 audit.** In simulator v0.1 a fault "onset" was the moment the severity ramp *started*, so no telemetry signal existed before it, and "onset within 72 h" could not be predicted from telemetry. All 20 v0.1 onsets were also at the same instant (2026-11-19 00:00 UTC), so the label was a function of the clock.

**Revision.** Simulator v0.2 records a P-F timeline per fault: `initiation_utc` (hidden start of degradation) → `onset_utc` (**qualifying onset**: severity first reaches 0.3, the level at which a field-service manager would act) → `failure_utc` (severity 1.0) → `repair_start_utc` / `repair_end_utc`.

- Target at hourly prediction time *t*: `y = 1` if a qualifying onset of that unit lies in `(t, t + 72 h]`.
- Excluded rows: the unit is in an active fault (`onset <= t < repair_end`); the 24 hours after `repair_end`; censored rows where `t + 72 h` is beyond the end of observation.
- The precursor period (initiation → qualifying onset) is drawn uniformly from 2–10 days, so a 72-hour warning is physically possible but not guaranteed.
- Implemented in `src/ingen_fleet_health/labels.py`, with tests in `tests/w02/test_labels.py`.
- Maintenance events are now an observable table (`events/maintenance_events.csv`). Features see a repair only once it is completed (`hours_since_repair`).

## Revision history

- Original: Week 1, `docs/W01_Data_Contract.md` §9 (text above copied verbatim; the data contract keeps its copy).
- rev1 (Week 2, 2026-11-04) is reproduced above as written. Week 4 (2026-11-24): no change to the label; the weekly performance check rebuilds the same labels from fault confirmations that arrive late (ADR-0006).
