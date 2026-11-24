# ADR-0006 — Daily Pipeline: Partitions, Idempotency, Backfill, Late and Missing Data

**Status:** Accepted (Week 4, 2026-11-24)  
**Code:** `src/ingen_fleet_health/pipeline/`, `pipelines/airflow/dags/fleet_health_daily.py`, `configs/pipeline.yaml`  
**Tests:** `tests/w04/test_pipeline.py`

## Context

Plan v3.0 asks for a daily cycle (ingest → validate → features → score → predictions → alert queue) with idempotent re-runs, a documented backfill, stated behaviour for late or missing upstream partitions, PSI monitoring and a weekly performance check. The Week 2 features use each robot's whole history: the commissioning baseline over its first 5 days, cumulative cycle counts and the thermal-model state.

## Decisions

### Layout

The layout is one Parquet partition per table, platform and UTC date under `data_root`:
- the tables are `landing`, `validated`, `quarantine`, `features`, `predictions`, `alerts` and `monitoring`;
- each step writes a run manifest under `runs/`, with the input and output fingerprints and the model versions.

### Idempotency

- A step reads only partitions up to its date and writes its output to a temporary folder. The folder then replaces the old partition whole.
- Outputs contain no wall-clock time.
- Re-running a date gives byte-identical files; the test compares SHA-256 hashes.
- A failed step leaves the previous partition untouched.

### Features from history

- A date's features are computed from every validated partition since the fleet start, then cut to that date.
- This reproduces the Week 2 batch feature table exactly; the test compares frames.
- The cost grows with history. Carrying feature state from day to day is deferred: it is an optimisation that would need its own equality test.

### Late or missing landing partition (decision: wait, then fail loudly; never fill)

- **Wait.** The Airflow sensor re-checks every 30 minutes for up to 6 hours (`landing` in the config); the run starts at 02:00 UTC, which gives another 2 hours of grace.
- **Then fail.** The platform's tasks for that date fail and nothing downstream is written for that platform and date. The other platform still runs.
- **Record it.** The failure is written under `runs/failures/`, and the dashboard shows the gap ("no scores for …") instead of yesterday's risk presented as today's.

Why not the alternatives:
- **Skipping silently** would let an operator read stale risk as current.
- **Retrying without limit** hides an upstream outage.
- **Scoring without the day** gives meaningless features, because about 100 % of the last 24 h would be missing.

### An earlier date is missing when a later date runs

- The later date still runs.
- Its features are computed with the gap, exactly as a network outage would be, and every row carries `history_missing_days`.
- `status` lists the unscored dates.

### A late partition arrives

1. Publish it.
2. Backfill from that date to the latest scored date. Later dates must be redone because their features and alert refractory state depend on it.
3. After the backfill, the features equal those of a complete history (tested).

### Robot with too little data

- Rows with more than 50 % of the last 24 h missing are marked `insufficient_data` and never alert.
- The dashboard lists such a robot as "No recent data", meaning a connectivity check, not a hardware pull.

### Rejected records

- Rejected records are quarantined with their reason codes, using the Week 1 rules and the platform profile.
- The last timestamp and odometer reading are carried across midnight, so a duplicate or a backwards odometer at a day boundary is still rejected.

### Privacy gate

A Fari file with an imagery or audio column is refused before anything is stored (Week 3 contract, enforced again here).

### Backfill command

- **History load before the scoring start:** `python -m ingen_fleet_health.pipeline backfill --start <fleet start> --end <cut-off − 1> --steps ingest,features`.
- **Scoring range:** `python -m ingen_fleet_health.pipeline backfill --start <D> --end <E>`.
- **In Airflow:** `airflow backfill create --dag-id fleet_health_daily --from-date <D+1> --to-date <E+1>`. The DAG runs one date at a time (`max_active_runs=1`) and skips the model steps for dates in the training window.

### Monitoring

PSI:
- PSI is computed daily against decile bins of the training rows, per platform, for the champion's top 8 features by gain and for the risk score itself.
- Missing values get their own bin.
- Bands: < 0.10 stable, 0.10–0.25 watch, > 0.25 shift.

Weekly performance check (Sundays):
- **Window:** predictions from the week that ended 7 days earlier. That is the 72 h horizon plus up to 96 h until a fault is confirmed at repair start in the simulator.
- **Labels:** only the confirmations that have arrived, with the ADR-003 rev1 label rule.
- **Cohorts:** held-out and training units.
- **Models:** champion and shadow models.
- **Thin weeks:** a week with fewer than 3 confirmed onsets is reported as "insufficient labels".

## Consequences

- The pipeline is safe to re-run at any time. Correcting history is always "backfill from the first changed date".
- Daily cost is O(history). This is acceptable for 50 units × 60 days on a laptop (≈1–2 min per date for both platforms), not for years of history.
- Label maturity of 7 days means the performance check always lags by about two weeks.
