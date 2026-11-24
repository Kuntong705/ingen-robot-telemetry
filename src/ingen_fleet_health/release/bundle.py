"""Model bundles and their MLflow pyfunc wrapper.

A bundle is everything the pipeline needs to score one model family: the
fitted estimator(s), the feature list, per-platform alert thresholds and the
PSI reference distributions. It is pickled (joblib) as an MLflow artifact
and registered under a name; the pipeline loads ``models:/<name>/<version>``
(the version is pinned in ``configs/pipeline.yaml``), never a file path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from ingen_fleet_health.models.early_warning import (
    ANOMALY_FEATURES,
    CalibratedModel,
    physics_rule_score,
)

PLATFORM_FLAG = "platform_fari"


def add_platform_flag(frame: pd.DataFrame, platform: str) -> pd.DataFrame:
    return frame.assign(**{PLATFORM_FLAG: 1.0 if platform == "fari" else 0.0})


@dataclass
class EarlyWarningBundle:
    """Calibrated LightGBM (champion) plus the two shadow scores (challengers)."""

    variant: str                                  # lgbm_all / lgbm_physics, chosen on validation PR-AUC
    columns: list[str]
    model: CalibratedModel
    thresholds: dict[str, float]                  # platform -> risk threshold for the alert budget
    iforest: Any
    iforest_median: pd.Series
    shadow_thresholds: dict[str, dict[str, float]]  # model -> platform -> threshold (monitoring only)
    top_features: list[str]
    psi_reference: dict[str, dict[str, dict[str, Any]]]  # platform -> feature -> {edges, expected}
    training_units: dict[str, list[str]]
    info: dict[str, Any] = field(default_factory=dict)

    def features(self, frame: pd.DataFrame, platform: str) -> pd.DataFrame:
        return add_platform_flag(frame, platform)[self.columns]

    def risk(self, frame: pd.DataFrame, platform: str) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self.features(frame, platform)))[:, 1]

    def raw_margin(self, frame: pd.DataFrame, platform: str) -> np.ndarray:
        return np.asarray(self.model.model.predict_proba(self.features(frame, platform), raw_score=True))

    def shadow_scores(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            "iforest": -self.iforest.score_samples(frame[ANOMALY_FEATURES].fillna(self.iforest_median)),
            "physics_rule": physics_rule_score(frame),
        }


@dataclass
class SohBundle:
    """LightGBM on the residual SoH(d+h) - persistence, as in Week 2/3."""

    model: Any
    features: list[str]
    horizon_days: int
    training_units: dict[str, list[str]]
    info: dict[str, Any] = field(default_factory=dict)

    def predict(self, daily: pd.DataFrame) -> np.ndarray:
        return np.asarray(daily["bms_soh_mean_24h"].to_numpy() + self.model.predict(daily[self.features]))


class BundleModel(mlflow.pyfunc.PythonModel):  # type: ignore[misc,name-defined]
    """MLflow pyfunc: ``predict`` returns early-warning risk or SoH forecasts.

    The pipeline uses ``unwrap_python_model().bundle`` for SHAP and shadow
    scores; ``predict`` keeps the model usable from any MLflow client."""

    def load_context(self, context: Any) -> None:
        import joblib

        self.bundle = joblib.load(context.artifacts["bundle"])

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None) -> np.ndarray:
        b = self.bundle
        if isinstance(b, EarlyWarningBundle):
            platform = (params or {}).get("platform", "aido_rover")
            return b.risk(model_input, str(platform))
        return np.asarray(b.predict(model_input))


def load_bundle(name: str, version: int | str) -> Any:
    """Load a registered bundle by name and pinned version."""
    import os

    import mlflow

    from ingen_fleet_health.tracking import tracking_uri

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(tracking_uri())
    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    return model.unwrap_python_model().bundle
