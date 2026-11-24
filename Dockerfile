# ingen_fleet_health -- reproducible runtime (SYNTHETIC-data analytics package).
#
# target "app"     : the package in a locked uv environment (/opt/fleet/.venv):
#                    pipeline CLI, training, MLflow server, Streamlit dashboard
# target "airflow" : the official Airflow image plus a copy of that environment;
#                    DAG tasks call /opt/fleet/.venv/bin/python, so Airflow's own
#                    dependencies and the model's dependencies never mix.
#
#   docker compose build            (see docker-compose.yml and the README quickstart)

FROM python:3.11.17-slim-bookworm AS app
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/fleet/.venv \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PYTHONUNBUFFERED=1
WORKDIR /opt/fleet
# dependencies first (cached layer), then the project itself
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY . .
RUN uv sync --locked --no-dev
ENV PATH=/opt/fleet/.venv/bin:$PATH \
    FLEET_HOME=/opt/fleet
EXPOSE 8501 5000
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

FROM apache/airflow:3.3.2-python3.11 AS airflow
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=app --chown=airflow:0 /opt/fleet /opt/fleet
ENV FLEET_HOME=/opt/fleet \
    FLEET_PYTHON=/opt/fleet/.venv/bin/python \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    AIRFLOW__CORE__DAGS_FOLDER=/opt/fleet/pipelines/airflow/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=False
USER airflow
