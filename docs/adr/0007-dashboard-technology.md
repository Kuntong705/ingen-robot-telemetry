# ADR-0007 — Dashboard Technology: Streamlit + Plotly

**Status:** Accepted (Week 4, 2026-11-24)  
**Code:** `dashboard/app.py`, `dashboard/components/`, read model `src/ingen_fleet_health/pipeline/views.py`

## Context

The dashboard is for a fleet operations manager, not a data scientist. It must answer "which robots should I pull before the next shift, and why?" in under a minute. It must also give a per-unit drill-down, the alert queue with grounded summaries, and platform and date filters. Plan v3.0 allows Streamlit, Power BI or Tableau. The repository is public and must run from a clean clone without licences or cloud accounts.

## Decision

**Streamlit + Plotly**, reading the pipeline's Parquet partitions directly. All decision logic lives in the testable `pipeline/views.py`:

- **Pull** means an open alert, raised within the last 72 h.
- **No recent data** means the latest scored hour has insufficient data.
- **OK** is everything else.

## Alternatives

- **Power BI or Tableau.** Rejected:
  - they need licences or desktop installs, which are not reproducible from `docker compose up`;
  - they cannot be tested in CI;
  - the decision logic (open alert, refractory, insufficient data) would be re-implemented in DAX or calculated fields and drift from the pipeline.
- **A custom web app (FastAPI + JS).** Rejected: more code than the four-week scope allows, for no gain in the operator's answer.
- **The MLflow or Airflow UIs.** Rejected: built for engineers, not operators.

## Consequences

- Run with one command: `streamlit run dashboard/app.py`, or the `dashboard` service in Docker Compose.
- The smoke test runs the app headless (`streamlit.testing`) on a pipeline output.
- Streamlit is single-process and has no user management. It suits a local or internal tool. Authentication would be a deployment concern (next steps).
