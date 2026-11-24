"""Week 4 daily fleet-health pipeline.

landing -> validated -> features -> predictions -> alert queue -> monitoring,
one partition per (table, platform, date). Every step is a plain function
(testable without Airflow); ``pipelines/airflow/dags/fleet_health_daily.py``
only orchestrates them. ADR-0005 (production recipe) and ADR-0006 (late and
missing data, idempotency, backfill) document the behaviour.
"""

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
