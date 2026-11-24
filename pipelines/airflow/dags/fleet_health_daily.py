"""fleet_health_daily -- the Week 4 daily scoring DAG (Airflow 3).

A run triggered at 02:00 UTC on D+1 processes the UTC day D (the 2 h are the
grace period for late uploads). Per platform:

    wait_landing -> ingest -> features -> score -> alerts -> drift
then, once per run: weekly_performance (does work only on the check weekday).

Each task calls the package CLI in the project's own virtual environment
(``/opt/fleet/.venv``), so Airflow's dependencies and the model's dependencies
never mix. All logic lives in ``ingen_fleet_health.pipeline``; this file only
orchestrates it.

Behaviour (ADR-0006):
* idempotent: re-running a date replaces its partitions with identical files;
* backfill: ``airflow backfill create --dag-id fleet_health_daily --from-date
  <D+1> --to-date <E+1>`` (runs one date at a time, oldest first: max_active_runs=1),
  or without Airflow ``python -m ingen_fleet_health.pipeline backfill --start D --end E``
  (Airflow refuses future logical dates; the simulated fleets run past today);
* missing or late landing partition: the sensor re-checks every 30 min for up to
  6 h, then fails. Nothing downstream is written for that platform and date, the
  other platform still runs, and the failure is recorded under
  ``<data_root>/runs/failures/``. When the partition arrives later, re-run that
  date and every later date (backfill), because later features use the history.
"""

from __future__ import annotations

import itertools
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.sensors.python import PythonSensor
from airflow.sdk import DAG, TaskGroup
from airflow.timetables.trigger import CronTriggerTimetable

FLEET_HOME = Path(os.environ.get("FLEET_HOME", "/opt/fleet"))
FLEET_PYTHON = os.environ.get("FLEET_PYTHON", str(FLEET_HOME / ".venv" / "bin" / "python"))
CONFIG_PATH = Path(os.environ.get("FLEET_PIPELINE_CONFIG", str(FLEET_HOME / "configs" / "pipeline.yaml")))
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
DATA_ROOT = Path(os.environ.get("FLEET_DATA_ROOT", str(FLEET_HOME / CONFIG["data_root"])))   # absolute paths stay
# first logical date: one day after the first day to process (default: the full fleet's scoring start)
START = datetime.fromisoformat(os.environ.get("FLEET_DAG_START_DATE", "2026-11-26")).replace(tzinfo=UTC)
PLATFORMS = list(CONFIG["platforms"])
POKE_S = int(os.environ.get("FLEET_SENSOR_POKE_S", CONFIG["landing"]["sensor_poke_minutes"] * 60))
TIMEOUT_S = int(os.environ.get("FLEET_SENSOR_TIMEOUT_S", CONFIG["landing"]["sensor_timeout_hours"] * 3600))
# the UTC day this run processes: one day before the run's logical date
DAY = "{{ (logical_date - macros.timedelta(days=1)).strftime('%Y-%m-%d') }}"


def landing_ready(platform: str, day: str) -> bool:
    return (DATA_ROOT / "landing" / "telemetry" / f"platform={platform}" / f"date={day}" / "telemetry.csv").exists()


def record_failure(context: dict[str, Any]) -> None:
    """Failure record the dashboard shows as 'no fresh score' (no silent staleness)."""
    ti = context["task_instance"]
    logical = context["logical_date"]
    day = (logical - timedelta(days=1)).strftime("%Y-%m-%d")
    out = DATA_ROOT / "runs" / "failures" / f"{day}__{ti.task_id.replace('.', '__')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"date": day, "task": ti.task_id, "dag_run": context["run_id"],
                               "recorded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                               "reason": str(context.get("exception"))}, indent=2), encoding="utf-8")


def cli(step: str, platform: str | None = None) -> str:
    target = f"--platform {platform} " if platform else ""
    return (f"cd {FLEET_HOME} && {FLEET_PYTHON} -m ingen_fleet_health.pipeline run --date {DAY} "
            f"{target}--step {step}")


default_args = {
    "owner": "fleet-analytics",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": record_failure,
}

with DAG(
    dag_id="fleet_health_daily",
    description="Synthetic fleet: ingest -> validate -> features -> score -> alerts -> monitoring",
    schedule=CronTriggerTimetable("0 2 * * *", timezone="UTC"),
    start_date=START,                                 # default processes 2026-11-25, the first day after the cut-off
    catchup=False,                                    # history is filled with an explicit, documented backfill
    max_active_runs=1,                                # dates run in order: features and alerts read earlier dates
    default_args=default_args,
    tags=["ingen", "fleet-health", "synthetic-data"],
    doc_md=__doc__,
) as dag:
    performance = BashOperator(task_id="weekly_performance", bash_command=cli("performance-if-due"),
                               trigger_rule="all_done")   # runs even if one platform's data is missing
    # The run's outcome: fails when any platform failed, so a missing partition is never a "successful" run.
    run_status = EmptyOperator(task_id="all_platforms_ok", trigger_rule="all_success")
    for platform in PLATFORMS:
        with TaskGroup(group_id=platform):
            wait = PythonSensor(
                task_id="wait_landing", python_callable=landing_ready,
                op_kwargs={"platform": platform, "day": DAY},
                mode="reschedule", poke_interval=POKE_S, timeout=TIMEOUT_S, retries=0,
            )
            steps = [BashOperator(task_id=s, bash_command=cli(s, platform))
                     for s in ("ingest", "features", "score", "alerts", "drift")]
            wait >> steps[0]
            for a, b in itertools.pairwise(steps):
                a >> b
            steps[-1] >> [performance, run_status]
