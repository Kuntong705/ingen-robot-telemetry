"""Train the production (v1) models and register them in the MLflow model registry.

Recipe (ADR-0005), with the same rules as the Week 2-3 evaluations:

* **Data.** Both simulated fleets (Aido Rover v0.2 and the synthetic Fari
  profile) and the Week 2 hourly as-of features. 70 % of each platform's units
  are training units (fault-count-stratified, seeded); the other 30 % are never
  seen in training and are the headline cohort of the weekly performance check.
* **Time.** Training rows have prediction time t <= start + ``cutoff_day`` - 72 h,
  so every training label is settled before the cut-off (embargo = horizon).
  The last ``validation_days`` before that form the validation slice (fit rows
  end one horizon earlier still). The pipeline scores the period after the cut-off.
* **Early warning (champion).** One pooled LightGBM per pre-declared feature
  set (``lgbm_all``, ``lgbm_physics``) with a platform flag; the one with the
  higher validation PR-AUC is kept and calibrated on the validation slice
  (Week 2 selection rule; the pooled model is the Week 3 Fari finding).
  Telemetry only: the Week 3 text modality gave no detectable gain (+0.016
  PR-AUC, 95 % CI -0.040 to 0.078) and depends on how often operators write notes.
* **Per-platform alert thresholds** chosen on validation rows for the budget of
  one alert per unit-week with a 72 h refractory period (Week 3: Fari needs its own).
* **Shadow scores (challengers).** Isolation Forest and the physics rule are
  scored every day but never raise alerts: on Rover they ranked better in Week 2
  (PR-AUC 0.330 / 0.215 vs 0.210 for LightGBM) but transfer worse to Fari
  (Week 3). The weekly check compares them on newly confirmed faults.
* **SoH forecast.** Pooled LightGBM on the residual SoH(d+7) - persistence.

    python -m ingen_fleet_health.release.train            # registers the next version of both models
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import IsolationForest

from ingen_fleet_health.evaluation.metrics import pr_auc, threshold_for_budget
from ingen_fleet_health.evaluation.splits import unit_groups
from ingen_fleet_health.experiments.common import load_w02
from ingen_fleet_health.models.early_warning import (
    ANOMALY_FEATURES,
    PHYSICS_MODEL_FEATURES,
    model_feature_columns,
    physics_rule_score,
)
from ingen_fleet_health.models.fusion import fit_selected
from ingen_fleet_health.models.soh_forecast import (
    DAILY_FEATURES,
    SOH_LGBM_PARAMS,
    daily_table,
)
from ingen_fleet_health.monitoring.drift import reference_bins
from ingen_fleet_health.pipeline.settings import Settings, load_settings
from ingen_fleet_health.release.bundle import (
    PLATFORM_FLAG,
    BundleModel,
    EarlyWarningBundle,
    SohBundle,
    add_platform_flag,
)
from ingen_fleet_health.tracking import file_sha256, log_flat_metrics, tracked_run

EXPERIMENT = "w04_release"


def training_units(units: list[str], strata: pd.Series, fraction: float, seed: int) -> list[str]:
    groups = unit_groups(units, 10, strata=strata, seed=seed)
    n_train = round(fraction * 10)
    return sorted(u for u, g in groups.items() if g < n_train)


def platform_frames(settings: Settings, platform: str) -> dict[str, Any]:
    ps = settings.platforms[platform]
    data = load_w02(ps.source_dir, config_path=ps.config_path)
    rel = settings.release
    horizon = pd.Timedelta(hours=float(ps.config["labels"]["horizon_hours"]))
    units = sorted(data.features["unit_id"].unique())
    strata = data.events.groupby("unit_id").size().reindex(units, fill_value=0)
    train_units = training_units(units, strata, float(rel["training_unit_fraction"]), int(rel["seed"]))
    cutoff = data.start + pd.Timedelta(days=float(rel["cutoff_day"]))
    train_end = cutoff - horizon
    val_start = train_end - pd.Timedelta(days=float(rel["validation_days"]))
    el = add_platform_flag(data.eligible(), platform)
    el = el[el["unit_id"].isin(train_units)]
    fit = el[el["t"] <= val_start - horizon]
    val = el[(el["t"] > val_start) & (el["t"] <= train_end)]
    h = int(rel["soh_horizon_days"])
    daily = daily_table(data.features, data.battery_truth, [h])
    soh_train = daily[daily["unit_id"].isin(train_units)
                      & (daily["day"] + pd.Timedelta(days=h) < cutoff.normalize())].dropna(subset=[f"target_h{h}"])
    return {"data": data, "units": units, "train_units": train_units, "cutoff": cutoff, "fit": fit, "val": val,
            "soh_train": soh_train}


def train_early_warning(settings: Settings, frames: dict[str, dict[str, Any]], seed: int) -> EarlyWarningBundle:
    fit = pd.concat([f["fit"] for f in frames.values()], ignore_index=True)
    val = pd.concat([f["val"] for f in frames.values()], ignore_index=True)
    if int(fit["y"].sum()) == 0 or int(val["y"].sum()) == 0:
        raise ValueError(f"training window has {int(fit['y'].sum())} positive fit rows and {int(val['y'].sum())} "
                         "positive validation rows; a fault early-warning model cannot be trained (use a larger "
                         "fleet, a later cut-off or a fleet with faults before the cut-off)")
    features = model_feature_columns()
    variants = {"lgbm_all": [*features, PLATFORM_FLAG],
                "lgbm_physics": [*[c for c in PHYSICS_MODEL_FEATURES if c in fit.columns], PLATFORM_FLAG]}
    selected = fit_selected(fit, val, variants, seed)
    median = fit[ANOMALY_FEATURES].median()
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed, n_jobs=8).fit(
        fit[ANOMALY_FEATURES].fillna(median))
    budget = float(settings.alerting["budget_alerts_per_unit_week"])
    refractory = float(settings.alerting["refractory_hours"])
    thresholds: dict[str, float] = {}
    shadow_thr: dict[str, dict[str, float]] = {"iforest": {}, "physics_rule": {}}
    psi_ref: dict[str, dict[str, dict[str, Any]]] = {}
    gain = selected.gain.drop(PLATFORM_FLAG, errors="ignore").sort_values(ascending=False)
    top = gain.index[: int(settings.monitoring["psi_top_features"])].tolist()
    val_metrics: dict[str, Any] = {}
    for platform, f in frames.items():
        v = f["val"]
        risk = selected.cal.predict_proba(v[selected.cols])[:, 1]
        keys = v[["unit_id", "t"]]
        thresholds[platform] = threshold_for_budget(keys.assign(score=risk), budget, refractory)
        shadow = {"iforest": -iso.score_samples(v[ANOMALY_FEATURES].fillna(median)),
                  "physics_rule": physics_rule_score(v)}
        for m, s in shadow.items():
            shadow_thr[m][platform] = threshold_for_budget(keys.assign(score=s), budget, refractory)
        fit_p = f["fit"]
        fit_risk = selected.cal.predict_proba(fit_p[selected.cols])[:, 1]
        psi_ref[platform] = {c: reference_bins(fit_p[c], int(settings.monitoring["psi_bins"])) for c in top}
        psi_ref[platform]["risk"] = reference_bins(pd.Series(fit_risk), int(settings.monitoring["psi_bins"]))
        val_metrics[platform] = {"val_rows": len(v), "val_positives": int(v["y"].sum()),
                                 "val_pr_auc_champion": pr_auc(v["y"].to_numpy(), risk),
                                 **{f"val_pr_auc_{m}": pr_auc(v["y"].to_numpy(), s) for m, s in shadow.items()},
                                 "fit_rows": len(fit_p), "fit_positives": int(fit_p["y"].sum())}
    info = {"variant_val_pr_auc": selected.val_pr, "calibration": selected.cal.method, "validation": val_metrics,
            "budget_alerts_per_unit_week": budget, "refractory_hours": refractory,
            "gain_top": gain.head(15).round(1).to_dict()}
    return EarlyWarningBundle(selected.name, selected.cols, selected.cal, thresholds, iso, median, shadow_thr, top,
                              psi_ref, {p: f["train_units"] for p, f in frames.items()}, info)


def train_soh(settings: Settings, frames: dict[str, dict[str, Any]], seed: int) -> SohBundle:
    h = int(settings.release["soh_horizon_days"])
    train = pd.concat([f["soh_train"] for f in frames.values()], ignore_index=True)
    y = train[f"target_h{h}"] - train["bms_soh_mean_24h"]
    model = lgb.LGBMRegressor(**SOH_LGBM_PARAMS, random_state=seed).fit(train[DAILY_FEATURES], y)
    info = {"train_rows": len(train), "train_rows_by_platform": {p: len(f["soh_train"]) for p, f in frames.items()}}
    return SohBundle(model, list(DAILY_FEATURES), h, {p: f["train_units"] for p, f in frames.items()}, info)


def register(bundle: Any, name: str, params: dict[str, Any], metrics: dict[str, Any], tags: dict[str, str]) -> str:
    import mlflow

    with tracked_run(EXPERIMENT, f"release_{name}", tags) as run, tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bundle.joblib"
        joblib.dump(bundle, path)
        mlflow.log_params(params)
        log_flat_metrics(metrics)
        mlflow.log_text(json.dumps(bundle.info, indent=2, default=str), "bundle_info.json")
        info = mlflow.pyfunc.log_model(
            name="model", python_model=BundleModel(), artifacts={"bundle": str(path)},
            registered_model_name=name, pip_requirements=["ingen-robot-telemetry"])
        mlflow.set_tag("bundle_sha256", file_sha256(path))
        version = getattr(info, "registered_model_version", None)
        return f"{run.info.run_id} -> {name} v{version}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register the production v1 models")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="train and print, do not register")
    args = parser.parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    settings = load_settings(args.config)
    seed = int(settings.release["seed"])
    frames = {p: platform_frames(settings, p) for p in settings.platforms}
    ew = train_early_warning(settings, frames, seed)
    soh = train_soh(settings, frames, seed)
    summary = {"early_warning": {"variant": ew.variant, "thresholds": ew.thresholds, "top_features": ew.top_features,
                                 **ew.info},
               "soh": soh.info,
               "cutoff_utc": {p: f["cutoff"].isoformat() for p, f in frames.items()},
               "training_units": {p: len(f["train_units"]) for p, f in frames.items()},
               "held_out_units": {p: len(f["units"]) - len(f["train_units"]) for p, f in frames.items()}}
    print(json.dumps(summary, indent=2, default=str))
    if args.dry_run:
        return
    params = {"cutoff_day": settings.release["cutoff_day"], "validation_days": settings.release["validation_days"],
              "training_unit_fraction": settings.release["training_unit_fraction"], "seed": seed,
              "platforms": ",".join(settings.platforms)}
    tags = {"synthetic_data": "true", "pipeline_config_sha256": file_sha256(settings.source)}
    ew_metrics = {f"{p}_{k}": v for p, m in ew.info["validation"].items() for k, v in m.items()}
    ew_metrics.update({f"threshold_{p}": v for p, v in ew.thresholds.items()})
    print(register(ew, settings.models["early_warning"]["name"], {**params, "variant": ew.variant}, ew_metrics, tags))
    print(register(soh, settings.models["soh_forecast"]["name"], {**params, "horizon_days": soh.horizon_days},
                   {"train_rows": float(soh.info["train_rows"])}, tags))


if __name__ == "__main__":
    main()
