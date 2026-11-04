"""Model track 2 -- fault early warning (qualifying onset within 72 h).

Models, all evaluated on the same unit-grouped walk-forward folds:

* ``no_skill``      -- constant score = training prevalence (PR-AUC = base rate)
* ``physics_rule``  -- naive operator rule: largest relative deviation of the
                       unit-normalised physics indicators from 1 (no learning
                       except the alert threshold)
* ``lgbm_all``      -- LightGBM on all non-time-proxy features
* ``lgbm_physics``  -- LightGBM on unit-normalised physics features only
* ``lgbm_selected`` -- per fold, whichever of the two has the higher PR-AUC on
                       the inner validation slice (selection never sees test)
  Each LightGBM is isotonic/Platt-calibrated on the inner validation slice.
* ``iforest``       -- Isolation Forest on physics features, fitted without labels

The alert threshold of every model is chosen on validation rows so that it
raises at most ``budget`` alerts per unit-week; it is then frozen for test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ingen_fleet_health.evaluation.metrics import pr_auc
from ingen_fleet_health.evaluation.splits import Fold
from ingen_fleet_health.features.registry import all_specs, time_proxy_features

RULE_FEATURES = [
    "vib_norm_ratio_to_baseline",
    "energy_ratio_to_baseline",
    "r_hat_ratio_to_baseline",
    "thermal_gain_ratio_24h",
]
ANOMALY_FEATURES = RULE_FEATURES + [
    "motor_temp_resid_patrol_mean_24h",
    "motor_temp_resid_dock_mean_24h",
    "vib_norm_change_72h",
    "vib_norm_slope_24h",
    "energy_ratio_to_expected_24h",
    "r_hat_temp_norm_24h",
]
# Unit-normalised physics indicators plus context; added after round 1 showed
# the all-feature model leaning on absolute per-unit levels (see W02 results).
PHYSICS_MODEL_FEATURES = ANOMALY_FEATURES + [
    "vib_norm_mean_6h",
    "motor_temp_resid_mean_6h",
    "motor_temp_resid_slope_24h",
    "missing_fraction_24h",
    "hours_since_repair",
    "ambient_mean_24h",
    "terrain_patrol_mean_24h",
    "patrol_fraction_24h",
]
DEFAULT_MODELS = ("no_skill", "physics_rule", "lgbm_all", "lgbm_physics", "lgbm_selected", "iforest")
LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": 8,
}


def model_feature_columns(include_time_proxies: bool = False) -> list[str]:
    proxies = set(time_proxy_features())
    return [s.name for s in all_specs() if include_time_proxies or s.name not in proxies]


def physics_rule_score(frame: pd.DataFrame) -> np.ndarray:
    dev = np.column_stack([np.abs(frame[c].to_numpy(dtype=float) - 1.0) for c in RULE_FEATURES])
    worst = np.max(np.where(np.isfinite(dev), dev, -np.inf), axis=1)
    return np.where(np.isfinite(worst), worst, 0.0)


def positive_proba(model: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X))[:, 1]


def fit_lgbm(X: pd.DataFrame, y: np.ndarray, seed: int = 0) -> lgb.LGBMClassifier:
    pos = max(int(y.sum()), 1)
    model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed, scale_pos_weight=(len(y) - pos) / pos)
    model.fit(X, y)
    return model


class CalibratedModel:
    """A fitted classifier plus a calibration map learned on held-out rows.

    Equivalent to ``CalibratedClassifierCV(..., cv="prefit")``: the base model
    is frozen and only the 1-D map score -> probability is fitted, on the
    fold's inner validation slice (never on test)."""

    def __init__(self, model: lgb.LGBMClassifier, method: str) -> None:
        self.model = model
        self.method = method
        self.mapper: Any = None

    def fit(self, X_val: pd.DataFrame, y_val: np.ndarray) -> CalibratedModel:
        raw = positive_proba(self.model, X_val)
        if self.method == "isotonic":
            self.mapper = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y_val)
        elif self.method == "sigmoid":
            self.mapper = LogisticRegression(C=1e6).fit(_logit(raw)[:, None], y_val)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = positive_proba(self.model, X)
        if self.method == "isotonic":
            p = self.mapper.predict(raw)
        elif self.method == "sigmoid":
            p = self.mapper.predict_proba(_logit(raw)[:, None])[:, 1]
        else:
            p = raw
        return np.column_stack([1 - p, p])


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibrate(model: lgb.LGBMClassifier, X_val: pd.DataFrame, y_val: np.ndarray) -> CalibratedModel:
    """Isotonic when the validation slice has enough positives, else Platt;
    with almost no positives the raw (uncalibrated) scores are kept."""
    pos = int(np.asarray(y_val).sum())
    method = "isotonic" if pos >= 100 else ("sigmoid" if pos >= 5 else "none")
    return CalibratedModel(model, method).fit(X_val, y_val) if method != "none" else CalibratedModel(model, "none")


@dataclass
class FoldScores:
    fold: Fold
    scores: dict[str, np.ndarray]           # model -> test scores
    val_scores: dict[str, np.ndarray]       # model -> validation scores
    feature_importance: pd.Series | None = None
    info: dict[str, Any] = field(default_factory=dict)


def score_fold(data: pd.DataFrame, fold: Fold, features: list[str], seed: int = 0,
               models: tuple[str, ...] = DEFAULT_MODELS) -> FoldScores:
    fit = data.iloc[fold.fit_idx]
    val = data.iloc[fold.val_idx]
    test = data.iloc[fold.test_idx]
    y_fit = fit["y"].to_numpy()
    scores: dict[str, np.ndarray] = {}
    val_scores: dict[str, np.ndarray] = {}
    importance = None
    info_cal: dict[str, str] = {}

    if "no_skill" in models:
        prevalence = float(y_fit.mean())
        scores["no_skill"] = np.full(len(test), prevalence)
        val_scores["no_skill"] = np.full(len(val), prevalence)
    if "physics_rule" in models:
        scores["physics_rule"] = physics_rule_score(test)
        val_scores["physics_rule"] = physics_rule_score(val)
    variants = {"lgbm_all": features, "lgbm_physics": [c for c in PHYSICS_MODEL_FEATURES if c in data.columns]}
    val_pr: dict[str, float] = {}
    for name, cols in variants.items():
        if name not in models and "lgbm_selected" not in models:
            continue
        model = fit_lgbm(fit[cols], y_fit, seed)
        cal = calibrate(model, val[cols], val["y"].to_numpy())
        scores[name] = cal.predict_proba(test[cols])[:, 1]
        val_scores[name] = cal.predict_proba(val[cols])[:, 1]
        scores[f"{name}_raw"] = positive_proba(model, test[cols])
        val_pr[name] = pr_auc(val["y"].to_numpy(), val_scores[name])
        info_cal[name] = cal.method
        if name == "lgbm_all":
            importance = pd.Series(model.booster_.feature_importance("gain"), index=cols)
    if "lgbm_selected" in models and val_pr:
        # model selection on the inner validation slice only (never on test)
        best = max(val_pr, key=lambda k: -1.0 if not np.isfinite(val_pr[k]) else val_pr[k])
        scores["lgbm_selected"] = scores[best]
        val_scores["lgbm_selected"] = val_scores[best]
        scores["lgbm_selected_raw"] = scores[f"{best}_raw"]
        info_cal["selected"] = best
    if "iforest" in models:
        Xa = fit[ANOMALY_FEATURES].fillna(fit[ANOMALY_FEATURES].median())
        med = Xa.median()
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed, n_jobs=8).fit(Xa)
        scores["iforest"] = -iso.score_samples(test[ANOMALY_FEATURES].fillna(med))
        val_scores["iforest"] = -iso.score_samples(val[ANOMALY_FEATURES].fillna(med))
    return FoldScores(fold, scores, val_scores, importance,
                      {"fit_rows": len(fit), "val_rows": len(val), "test_rows": len(test),
                       "fit_positives": int(y_fit.sum()), "val_positives": int(val["y"].sum()),
                       "calibration": info_cal, "val_pr_auc": val_pr})
