"""Fold-fitted environmental normalisation.

Physics-prior normalisation lives in ``physics.py`` (no fitting). This
transformer is the *learned* variant: it regresses selected features on
environment covariates (ambient, terrain) and replaces them by residuals.
It must be fitted inside each training fold -- fitting it on the full
dataset is one of the leakage-audit items.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_COVARIATES = ("ambient_mean_24h", "terrain_patrol_mean_24h", "patrol_fraction_24h")
DEFAULT_TARGETS = ("energy_per_km_24h", "motor_temp_mean_24h", "vibration_mean_24h", "r_hat_24h")


class EnvironmentResidualizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        targets: tuple[str, ...] = DEFAULT_TARGETS,
        covariates: tuple[str, ...] = DEFAULT_COVARIATES,
    ) -> None:
        self.targets = targets
        self.covariates = covariates

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        z = X[list(self.covariates)].to_numpy(dtype=float)
        z = np.where(np.isfinite(z), z, self.cov_means_)
        return np.column_stack([np.ones(len(z)), z])

    def fit(self, X: pd.DataFrame, y: object = None) -> EnvironmentResidualizer:
        self.cov_means_ = np.nanmean(X[list(self.covariates)].to_numpy(dtype=float), axis=0)
        design = self._design(X)
        self.coef_: dict[str, np.ndarray] = {}
        for col in self.targets:
            v = X[col].to_numpy(dtype=float)
            ok = np.isfinite(v)
            self.coef_[col], *_ = np.linalg.lstsq(design[ok], v[ok], rcond=None)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        design = self._design(X)
        for col, coef in self.coef_.items():
            out[f"{col}_env_resid"] = X[col].to_numpy(dtype=float) - design @ coef
        return out
