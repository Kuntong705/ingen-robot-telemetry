# Week 4 Log — What the pipeline does when a day of data does not arrive

*Week 4 (Plan v3.0), November 2026. All fleets are synthetic.*

The decision I spent the most care on this week was not a model choice but what the daily pipeline does when one platform's telemetry for a day is late or missing. Plan v3.0 lists three options: skip, retry, or alert.

**What makes it hard.** The Week 2 features are not local to one day. Each robot's vibration, energy and resistance ratios are relative to its own commissioning baseline over the first five days. Cycle counts are cumulative, and the thermal model carries state from the first minute. So the features of day D depend on every earlier day. The alert queue does too, through the 72 h refractory period. If a missing day is silently skipped, later days are computed from a history with a hole, and nothing tells the operator.

**The alternatives.**
- **Skip silently.** The dashboard would keep showing yesterday's risk as if it were current, which is the worst failure for an operator deciding whom to pull.
- **Retry indefinitely.** This hides an upstream outage behind a task that never ends.
- **Score anyway.** The model would score a day with almost no telemetry, and the score would be meaningless.

**The decision (ADR-0006).**
1. The sensor waits up to 6 hours, checking every 30 minutes; the run itself starts 2 hours after midnight UTC.
2. After that it fails loudly. Nothing downstream is written for that platform and date; the other platform still runs; a failure record drives a dashboard warning.
3. Later days still run. Their rows carry `history_missing_days`, so a reader knows the history has a gap.
4. When the data arrives, a backfill from that date onwards recomputes everything. A test checks that this equals the features of a complete history.

**What the test runs taught me.** I ran the scenario through Airflow, not just pytest: I removed Fari's landing partition for one day and ran the DAG. The behaviour was right: the sensor timed out, Fari's outputs were untouched, and the Rover group ran. But the run was marked *successful*. The only leaf task was the weekly performance check, which must run even when one platform fails (`all_done`), and Airflow judges a run by its leaves. A failed platform was invisible at the run level. I added a final `all_platforms_ok` task with `all_success`, and the same scenario now ends as a failed run.

The same Docker run found a second defect, a volume written as root that Airflow's user could not overwrite. Neither would have appeared in unit tests on my laptop.

**What it means for InGen.** The policy is a contract with the operator: a missing score is shown as missing, never filled. It is also why idempotency matters. Re-running a date gives byte-identical files (checked by hash, also through Airflow), so "backfill from the first changed date" is always a safe repair.
