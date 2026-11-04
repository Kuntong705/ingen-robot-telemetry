"""MLflow helpers: every reported number links to a run.

Local store: ``sqlite:///mlflow.db`` with artifacts under ``mlruns/`` (both
git-ignored). View with::

    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ingen_fleet_health.config import REPO_ROOT


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}")


def git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {"git_commit": run("rev-parse", "HEAD"), "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": str(bool(run("status", "--porcelain")))}


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def tracked_run(experiment: str, run_name: str, tags: dict[str, str] | None = None) -> Iterator[Any]:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(REPO_ROOT / "mlruns" / experiment).as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({**git_state(), **(tags or {})})
        yield run


def log_dict_json(payload: dict[str, Any], artifact_file: str) -> None:
    import mlflow

    mlflow.log_text(json.dumps(payload, indent=2, default=str), artifact_file)


def log_flat_metrics(metrics: dict[str, Any], prefix: str = "") -> None:
    import math

    import mlflow

    clean = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            clean[f"{prefix}{key}"] = float(value)
    if clean:
        mlflow.log_metrics(clean)
