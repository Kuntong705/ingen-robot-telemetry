"""Shared loading, fold construction and result I/O for Week 2 experiments."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from ingen_fleet_health.config import REPO_ROOT, load_config
from ingen_fleet_health.evaluation.splits import unit_groups
from ingen_fleet_health.labels import (
    early_warning_labels,
    load_fault_events,
    prediction_grid,
)

RESULTS_DIR = REPO_ROOT / "reports" / "w02"
FIGURES_DIR = REPO_ROOT / "docs" / "figures" / "w02"


@dataclass
class W02Data:
    data_dir: Path
    config: dict[str, Any]
    manifest: dict[str, Any]
    features: pd.DataFrame
    labels: pd.DataFrame
    events: pd.DataFrame
    battery_truth: pd.DataFrame
    groups: dict[str, int]

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.manifest["start_utc"])

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.manifest["end_utc"])

    def cutoffs(self, days: Sequence[float]) -> list[pd.Timestamp]:
        return [self.start + pd.Timedelta(days=d) for d in days]

    def eligible(self) -> pd.DataFrame:
        """Features joined with labels; only rows that may be scored."""
        joined = self.features.merge(self.labels, on=["unit_id", "t"], how="inner")
        joined = joined[joined["eligible"]].reset_index(drop=True)
        ft = self.events.set_index("fault_id")["fault_type"]
        joined["fault_type"] = joined["fault_id"].map(ft)
        return joined


def load_w02(data_dir: str | Path, n_groups: int = 5, seed: int = 0) -> W02Data:
    data_dir = Path(data_dir)
    config = load_config()
    manifest = json.loads((data_dir / "simulation_manifest.json").read_text(encoding="utf-8"))
    features = pd.read_parquet(data_dir / "features" / "features_hourly.parquet")
    events = load_fault_events(data_dir / "ground_truth" / "fault_events.csv")
    start, end = pd.Timestamp(manifest["start_utc"]), pd.Timestamp(manifest["end_utc"])
    lab_cfg = config["labels"]
    grid = prediction_grid(sorted(features["unit_id"].unique()), start, end,
                           int(lab_cfg["prediction_interval_hours"]), int(lab_cfg["history_warmup_hours"]))
    labels = early_warning_labels(grid, events, end, float(lab_cfg["horizon_hours"]),
                                  float(lab_cfg["post_repair_exclusion_hours"]))
    truth = pd.read_csv(data_dir / "ground_truth" / "battery_truth_daily.csv")
    units = sorted(features["unit_id"].unique())
    # stratify groups by number of faults so each group mixes faulty and healthy robots
    strata = events.groupby("unit_id").size().reindex(units, fill_value=0)
    groups = unit_groups(units, n_groups, strata=strata, seed=seed)
    return W02Data(data_dir, config, manifest, features, labels, events, truth, groups)


def write_json(payload: dict[str, Any], name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def figure_path(name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / name
