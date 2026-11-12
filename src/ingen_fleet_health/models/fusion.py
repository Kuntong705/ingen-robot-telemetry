"""Telemetry + text fusion for the unit-level fleet health score (Week 3).

All arms are scored on the same fold (same fit / validation / test rows):

Telemetry only (Week 2 models, unchanged, via ``early_warning.score_fold``)
    ``telemetry_lgbm``      calibrated LightGBM (``lgbm_selected``)
    ``telemetry_iforest``   Isolation Forest anomaly score
    ``physics_rule``, ``no_skill``
Text only
    ``text_only``           calibrated LightGBM on the as-of text features
Early fusion (one model, shared feature set)
    ``early_fusion``        LightGBM on telemetry + text features; the same
                            all-features / physics-features choice as Week 2,
                            made on the validation slice
Late fusion (one model per modality, outputs combined)
    ``late_noisy_or``       1 - (1 - p_tel)(1 - p_txt), nothing fitted
    ``late_stack``          logistic regression on [logit p_tel, logit p_txt]
    ``late_stack_iforest``  logistic regression on [iforest score, logit p_txt]
  The stackers are fitted on the fold's validation rows, like the Week 2
  calibration maps; test rows are never used for any fitted choice.

Health score: ``100 x (1 - fused risk)`` per unit and hour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ingen_fleet_health.evaluation.metrics import pr_auc
from ingen_fleet_health.evaluation.splits import Fold
from ingen_fleet_health.models.early_warning import (
    PHYSICS_MODEL_FEATURES,
    CalibratedModel,
    _logit,
    calibrate,
    fit_lgbm,
    score_fold,
)
from ingen_fleet_health.text.features import TEXT_FEATURES

TELEMETRY_ARMS = ("no_skill", "physics_rule", "telemetry_lgbm", "telemetry_iforest")
FUSION_ARMS = ("text_only", "early_fusion", "late_noisy_or", "late_stack", "late_stack_iforest")
ALL_ARMS = (*TELEMETRY_ARMS, *FUSION_ARMS)
MIN_STACK_POSITIVES = 5


@dataclass
class FusionFoldScores:
    fold: Fold
    scores: dict[str, np.ndarray]
    val_scores: dict[str, np.ndarray]
    info: dict[str, Any] = field(default_factory=dict)
    text_gain_share: float = float("nan")
    importance: pd.Series | None = None


def noisy_or(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 - (1.0 - np.clip(p_a, 0, 1)) * (1.0 - np.clip(p_b, 0, 1)))


def health_score(risk: np.ndarray) -> np.ndarray:
    """Unit-level fleet health score in [0, 100]; 100 = no elevated risk."""
    return np.asarray(100.0 * (1.0 - np.clip(risk, 0.0, 1.0)))


class StackedFusion:
    """Late fusion by logistic regression on two per-modality scores."""

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.mean_: np.ndarray = np.zeros(2)
        self.std_: np.ndarray = np.ones(2)

    def fit(self, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> StackedFusion:
        X = np.column_stack([a, b])
        self.mean_ = X.mean(axis=0)
        self.std_ = np.where(X.std(axis=0) > 1e-12, X.std(axis=0), 1.0)
        if int(np.sum(y)) >= MIN_STACK_POSITIVES and int(np.sum(y)) < len(y):
            self.model = LogisticRegression(C=1.0).fit((X - self.mean_) / self.std_, y)
        return self

    @property
    def fitted(self) -> bool:
        return self.model is not None

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("stacker was not fitted (too few validation positives)")
        X = (np.column_stack([a, b]) - self.mean_) / self.std_
        return np.asarray(self.model.predict_proba(X)[:, 1])

    def weights(self) -> list[float]:
        return [float(w) for w in self.model.coef_[0]] if self.model is not None else [float("nan")] * 2


@dataclass
class SelectedModel:
    """The LightGBM variant chosen on the validation slice, with its calibration map."""

    name: str
    cols: list[str]
    cal: CalibratedModel
    val_pr: dict[str, float]
    gain: pd.Series


def early_variants(data: pd.DataFrame, telemetry_features: list[str], text_features: list[str]) -> dict[str, list[str]]:
    return {
        "all+text": telemetry_features + text_features,
        "physics+text": [c for c in PHYSICS_MODEL_FEATURES if c in data.columns] + text_features,
    }


def fit_selected(fit: pd.DataFrame, val: pd.DataFrame, variants: dict[str, list[str]], seed: int = 0) -> SelectedModel:
    """Fit one calibrated LightGBM per pre-declared feature set and keep the
    one with the higher validation PR-AUC (the Week 2 selection rule)."""
    y_fit, y_val = fit["y"].to_numpy(), val["y"].to_numpy()
    fitted: dict[str, tuple[CalibratedModel, pd.Series]] = {}
    val_pr: dict[str, float] = {}
    for name, cols in variants.items():
        model = fit_lgbm(fit[cols], y_fit, seed)
        cal = calibrate(model, val[cols], y_val)
        fitted[name] = (cal, pd.Series(model.booster_.feature_importance("gain"), index=cols))
        val_pr[name] = pr_auc(y_val, cal.predict_proba(val[cols])[:, 1])
    best = max(val_pr, key=lambda k: -1.0 if not np.isfinite(val_pr[k]) else val_pr[k])
    return SelectedModel(best, variants[best], fitted[best][0], val_pr, fitted[best][1])


def score_fusion_fold(data: pd.DataFrame, fold: Fold, telemetry_features: list[str], seed: int = 0,
                      text_features: list[str] | None = None) -> FusionFoldScores:
    """``data`` holds telemetry features, text features, ``y``; indices as in ``fold``."""
    text_features = text_features or TEXT_FEATURES
    fit, val, test = data.iloc[fold.fit_idx], data.iloc[fold.val_idx], data.iloc[fold.test_idx]
    y_fit, y_val = fit["y"].to_numpy(), val["y"].to_numpy()

    base = score_fold(data, fold, telemetry_features, seed,
                      models=("no_skill", "physics_rule", "lgbm_all", "lgbm_physics", "lgbm_selected", "iforest"))
    rename = {"lgbm_selected": "telemetry_lgbm", "iforest": "telemetry_iforest",
              "no_skill": "no_skill", "physics_rule": "physics_rule"}
    scores = {new: base.scores[old] for old, new in rename.items()}
    val_scores = {new: base.val_scores[old] for old, new in rename.items()}
    info: dict[str, Any] = {"telemetry": base.info, "val_pr_auc": {}}

    # --- text only ---
    text_model = fit_lgbm(fit[text_features], y_fit, seed)
    text_cal = calibrate(text_model, val[text_features], y_val)
    scores["text_only"] = text_cal.predict_proba(test[text_features])[:, 1]
    val_scores["text_only"] = text_cal.predict_proba(val[text_features])[:, 1]

    # --- early fusion: shared feature set, same variant selection rule as Week 2 ---
    chosen = fit_selected(fit, val, early_variants(data, telemetry_features, text_features), seed)
    scores["early_fusion"] = chosen.cal.predict_proba(test[chosen.cols])[:, 1]
    val_scores["early_fusion"] = chosen.cal.predict_proba(val[chosen.cols])[:, 1]
    gain = chosen.gain
    info["early_variant"] = chosen.name
    info["val_pr_auc"].update({f"early[{k}]": v for k, v in chosen.val_pr.items()})
    total_gain = float(gain.sum())
    text_share = float(gain[text_features].sum() / total_gain) if total_gain > 0 else float("nan")

    # --- late fusion ---
    scores["late_noisy_or"] = noisy_or(scores["telemetry_lgbm"], scores["text_only"])
    val_scores["late_noisy_or"] = noisy_or(val_scores["telemetry_lgbm"], val_scores["text_only"])

    stack = StackedFusion().fit(_logit(val_scores["telemetry_lgbm"]), _logit(val_scores["text_only"]), y_val)
    if stack.fitted:
        scores["late_stack"] = stack.predict(_logit(scores["telemetry_lgbm"]), _logit(scores["text_only"]))
        val_scores["late_stack"] = stack.predict(_logit(val_scores["telemetry_lgbm"]), _logit(val_scores["text_only"]))
    else:   # fall back to the unfitted combination
        scores["late_stack"], val_scores["late_stack"] = scores["late_noisy_or"], val_scores["late_noisy_or"]
    info["late_stack_weights"] = stack.weights()

    stack_if = StackedFusion().fit(val_scores["telemetry_iforest"], _logit(val_scores["text_only"]), y_val)
    if stack_if.fitted:
        scores["late_stack_iforest"] = stack_if.predict(scores["telemetry_iforest"], _logit(scores["text_only"]))
        val_scores["late_stack_iforest"] = stack_if.predict(val_scores["telemetry_iforest"],
                                                            _logit(val_scores["text_only"]))
    else:
        scores["late_stack_iforest"] = scores["telemetry_iforest"]
        val_scores["late_stack_iforest"] = val_scores["telemetry_iforest"]
    info["late_stack_iforest_weights"] = stack_if.weights()
    for arm in ALL_ARMS:
        info["val_pr_auc"][arm] = pr_auc(y_val, val_scores[arm])
    return FusionFoldScores(fold, scores, val_scores, info, text_share, gain)
