"""Session fixture: a small synthetic two-platform fleet, trained and registered
in a temporary MLflow store, published to a temporary landing zone.

Built once (about 2-3 minutes on a laptop): simulator -> Week 1 ingestion ->
features -> production training -> registry -> daily landing partitions.
Then the history before the scoring start is loaded (ingest for every day,
features for the last 3 days) and the first scoring days are run.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pytest

UNITS, DAYS, CUTOFF, VAL_DAYS = 12, 30, 24, 4
START = date(2026, 10, 26)


@dataclass
class Fleet:
    root: Path
    config: Path
    settings: object
    scoring_start: date


@pytest.fixture(scope="session")
def fleet(tmp_path_factory: pytest.TempPathFactory) -> Fleet:
    root = tmp_path_factory.mktemp("w04_fleet")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    os.environ["MLFLOW_ARTIFACT_ROOT"] = str(root / "mlruns")
    os.environ.pop("FLEET_PIPELINE_CONFIG", None)
    os.environ.pop("FLEET_DATA_ROOT", None)

    from ingen_fleet_health.pipeline.publish import publish_platform
    from ingen_fleet_health.pipeline.runner import backfill
    from ingen_fleet_health.pipeline.settings import load_settings
    from ingen_fleet_health.release import train
    from ingen_fleet_health.release.quickstart import build

    config, _ = build(root, UNITS, DAYS, CUTOFF, VAL_DAYS)
    argv = sys.argv
    sys.argv = ["train", "--config", str(config)]
    try:
        train.main()
    finally:
        sys.argv = argv
    settings = load_settings(config)
    for p in settings.platforms:
        publish_platform(settings, p)
    s = START + timedelta(days=CUTOFF)
    quiet = lambda _msg: None
    backfill(settings, START, s - timedelta(days=4), steps=("ingest",), echo=quiet)
    backfill(settings, s - timedelta(days=3), s - timedelta(days=1), steps=("ingest", "features"), echo=quiet)
    backfill(settings, s, s + timedelta(days=2), echo=quiet)
    return Fleet(root, config, settings, s)
