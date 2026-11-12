"""Fari generalisation test: what transfers from the Rover pipeline, what breaks.

    python -m ingen_fleet_health.experiments.fari_transfer_run --rover-dir data/w02 --fari-dir data/w03_fari

The Fari fleet is SYNTHETIC (``configs/fari.yaml``: every parameter is an
assumption, none is an InGen specification). Three layers are tested:

1. Data contract  -- does Week 1 ingestion accept Fari records?
2. Features       -- which Week 2 features exist, are constant, or shift?
3. Models         -- Rover-trained early-warning and SoH models on Fari units.

Arms for the early-warning models (all scored on the same Fari test rows,
which no arm ever trains on):

``asis``             Rover models, Week 2 feature code + Rover config, Rover threshold
``fixed_rover_thr``  Rover models, platform-conditional features, Rover threshold
``fixed_fari_thr``   same scores, alert threshold re-chosen on Fari validation rows
``fari_native``      trained on Fari training units (the Week 2 procedure on Fari)
``pooled``           trained on Rover + Fari training units with a platform flag

Split discipline: for cut-off c, Rover training rows end at c - 72 h; Fari
test rows are units of the held-out Fari group in [c, c + 10 d).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ingen_fleet_health import plotstyle
from ingen_fleet_health.config import REPO_ROOT, load_config
from ingen_fleet_health.contract import DEFAULT_PROFILE, platform_profile
from ingen_fleet_health.evaluation.metrics import mae, pr_auc, threshold_for_budget
from ingen_fleet_health.evaluation.shift import availability, psi
from ingen_fleet_health.experiments.common import W02Data, load_w02
from ingen_fleet_health.experiments.early_warning_run import (
    VAL_SPAN,
    alert_metrics,
    bootstrap_ci,
    build_folds,
    paired_prauc_diff,
    per_unit_table,
    prauc_bootstrap,
)
from ingen_fleet_health.experiments.w03_common import figure_path, write_json
from ingen_fleet_health.explain.shap_utils import feature_info
from ingen_fleet_health.features.registry import PHYSICS_SPECS, all_specs
from ingen_fleet_health.models.early_warning import (
    ANOMALY_FEATURES,
    PHYSICS_MODEL_FEATURES,
    RULE_FEATURES,
    model_feature_columns,
    physics_rule_score,
    score_fold,
)
from ingen_fleet_health.models.fusion import SelectedModel, fit_selected
from ingen_fleet_health.models.soh_forecast import (
    daily_table,
    fit_predict_lgbm,
    linear_extrapolation,
    persistence,
)
from ingen_fleet_health.tracking import log_dict_json, log_flat_metrics, tracked_run

FARI_CONFIG = REPO_ROOT / "configs" / "fari.yaml"
MODELS = ("lgbm", "iforest", "physics_rule")
ARMS = ("asis", "fixed_rover_thr", "fixed_fari_thr", "fari_native", "pooled")
ARM_LABEL = {
    "asis": "Rover models, Week 2 features as-is",
    "fixed_rover_thr": "Rover models, platform-conditional features",
    "fixed_fari_thr": "... plus per-platform alert threshold",
    "fari_native": "Trained on Fari units (reference)",
    "pooled": "Trained on Rover + Fari units",
}
PAIRED = [
    ("fixed_rover_thr:lgbm", "asis:lgbm"),
    ("fixed_rover_thr:iforest", "asis:iforest"),
    ("fixed_rover_thr:physics_rule", "asis:physics_rule"),
    ("fari_native:lgbm", "fixed_rover_thr:lgbm"),
    ("pooled:lgbm", "fari_native:lgbm"),
    ("pooled:lgbm", "fixed_rover_thr:lgbm"),
]
SOH_HORIZON = 7
SOH_CUTOFF_DAYS = [30, 40, 50]


def contract_acceptance(csv_path: Path, profile: dict[str, Any], n_rows: int = 200_000) -> dict[str, Any]:
    """Run the Week 1 row validator over the first ``n_rows`` raw records."""
    from collections import Counter

    from ingest_pipeline import check_row

    last_ts: dict[str, Any] = {}
    last_odo: dict[str, float] = {}
    reasons: Counter[str] = Counter()
    total = accepted = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if total >= n_rows:
                break
            total += 1
            problems, timestamp, unit = check_row(row, last_ts, last_odo, profile)
            if problems:
                reasons.update(problems)
                continue
            accepted += 1
            last_ts[unit] = timestamp
            last_odo[unit] = float(row["odometer_m"])
    return {"rows_checked": total, "accepted": accepted, "acceptance_rate_pct": 100.0 * accepted / max(total, 1),
            "rejection_reasons": dict(reasons.most_common())}


def feature_layer(rover: pd.DataFrame, asis: pd.DataFrame, fixed: pd.DataFrame) -> dict[str, Any]:
    cols = [s.name for s in all_specs()]
    tables = {"rover": availability(rover, cols), "fari_asis": availability(asis, cols),
              "fari_fixed": availability(fixed, cols)}
    merged = tables["rover"].rename(columns={"valid_share": "valid_rover", "constant": "constant_rover"})
    for name in ("fari_asis", "fari_fixed"):
        merged = merged.merge(tables[name].rename(columns={"valid_share": f"valid_{name}", "constant": f"constant_{name}"}),
                              on="feature")
    merged["psi_fari_fixed_vs_rover"] = [psi(rover[c].to_numpy(), fixed[c].to_numpy()) for c in cols]
    merged["rover_median"] = [float(np.nanmedian(rover[c])) for c in cols]
    merged["fari_median"] = [float(np.nanmedian(fixed[c])) if fixed[c].notna().any() else float("nan") for c in cols]

    def dead(col: str) -> list[str]:
        return merged.loc[merged[col] < 0.01, "feature"].tolist()

    def constant(col: str) -> list[str]:
        return merged.loc[merged[col], "feature"].tolist()

    repaired = merged[(merged["valid_fari_asis"] < 0.01) & (merged["valid_fari_fixed"] > 0.5)]["feature"].tolist()
    unit_norm = [c for c in RULE_FEATURES + ["motor_temp_resid_patrol_mean_24h", "vib_norm_change_72h"]]
    physics = [s.name for s in PHYSICS_SPECS]
    major = merged[merged["psi_fari_fixed_vs_rover"] > 0.25]
    return {
        "features_total": len(cols),
        "dead_features": {"rover": dead("valid_rover"), "fari_asis": dead("valid_fari_asis"),
                          "fari_fixed": dead("valid_fari_fixed")},
        "constant_features": {"rover": constant("constant_rover"), "fari_asis": constant("constant_fari_asis"),
                              "fari_fixed": constant("constant_fari_fixed")},
        "repaired_by_platform_conditional_features": repaired,
        "psi_major_shift_count": len(major),
        "psi_major_shift_share": float(len(major) / len(cols)),
        "psi_unit_normalised_indicators": {c: float(merged.loc[merged["feature"] == c, "psi_fari_fixed_vs_rover"].iloc[0])
                                           for c in unit_norm},
        "psi_top": merged.sort_values("psi_fari_fixed_vs_rover", ascending=False).head(12)[
            ["feature", "psi_fari_fixed_vs_rover", "rover_median", "fari_median"]].to_dict(orient="records"),
        "physics_feature_table": merged[merged["feature"].isin(physics)].to_dict(orient="records"),
        "rule_features_valid_share": {c: {k: float(merged.loc[merged["feature"] == c, f"valid_{k}"].iloc[0])
                                          for k in ("rover", "fari_asis", "fari_fixed")} for c in RULE_FEATURES},
    }


@dataclass
class RoverModels:
    selected: SelectedModel
    iso: IsolationForest
    iso_median: pd.Series
    rover_threshold: dict[str, float]
    fit_frame: pd.DataFrame


def _time_split(frame: pd.DataFrame, cutoff: pd.Timestamp, horizon: pd.Timedelta) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end = cutoff - horizon
    val_start = train_end - VAL_SPAN
    return frame[frame["t"] <= val_start - horizon], frame[(frame["t"] > val_start) & (frame["t"] <= train_end)]


def _score(models: RoverModels, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "lgbm": models.selected.cal.predict_proba(frame[models.selected.cols])[:, 1],
        "iforest": -models.iso.score_samples(frame[ANOMALY_FEATURES].fillna(models.iso_median)),
        "physics_rule": physics_rule_score(frame),
    }


def train_rover(rover: pd.DataFrame, cutoff: pd.Timestamp, horizon: pd.Timedelta, features: list[str],
                budget: float, refractory: float, seed: int) -> RoverModels:
    fit, val = _time_split(rover, cutoff, horizon)
    variants = {"lgbm_all": features, "lgbm_physics": [c for c in PHYSICS_MODEL_FEATURES if c in rover.columns]}
    selected = fit_selected(fit, val, variants, seed)
    median = fit[ANOMALY_FEATURES].median()
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed, n_jobs=8).fit(
        fit[ANOMALY_FEATURES].fillna(median))
    models = RoverModels(selected, iso, median, {}, fit)
    val_scores = _score(models, val)
    models.rover_threshold = {m: threshold_for_budget(val[["unit_id", "t"]].assign(score=val_scores[m]), budget, refractory)
                              for m in MODELS}
    return models


def model_layer(rover: W02Data, fari: W02Data, fari_asis: pd.DataFrame, seed: int) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    cfg = fari.config
    horizon_h = float(cfg["labels"]["horizon_hours"])
    horizon = pd.Timedelta(hours=horizon_h)
    refractory = float(cfg["alerting"]["refractory_hours"])
    budget = float(cfg["alerting"]["budget_alerts_per_unit_week"])
    features = model_feature_columns()
    rover_el = rover.eligible()
    fari_el = fari.eligible()
    keys = fari_el[["unit_id", "t"]]
    asis_el = keys.merge(fari_asis, on=["unit_id", "t"], how="left")
    folds = build_folds(fari, fari_el, horizon)
    rover_by_cutoff: dict[pd.Timestamp, RoverModels] = {}
    parts: dict[tuple[str, str], list[pd.DataFrame]] = {(a, m): [] for a in ARMS for m in MODELS}
    info: list[dict[str, Any]] = []
    for fold in folds:
        if fold.cutoff not in rover_by_cutoff:
            rover_by_cutoff[fold.cutoff] = train_rover(rover_el, fold.cutoff, horizon, features, budget, refractory, seed)
        rm = rover_by_cutoff[fold.cutoff]
        test, val, fit = fari_el.iloc[fold.test_idx], fari_el.iloc[fold.val_idx], fari_el.iloc[fold.fit_idx]
        base = test[["unit_id", "t", "y", "fault_id", "fault_type"]]

        asis_scores = _score(rm, asis_el.iloc[fold.test_idx])
        fixed_scores, fixed_val = _score(rm, test), _score(rm, val)
        native = score_fold(fari_el, fold, features, seed, models=("physics_rule", "lgbm_all", "lgbm_physics",
                                                                   "lgbm_selected", "iforest"))
        native_map = {"lgbm": "lgbm_selected", "iforest": "iforest", "physics_rule": "physics_rule"}

        pooled_fit = pd.concat([rm.fit_frame.assign(platform_fari=0.0), fit.assign(platform_fari=1.0)], ignore_index=True)
        variants = {"lgbm_all": [*features, "platform_fari"],
                    "lgbm_physics": [*[c for c in PHYSICS_MODEL_FEATURES if c in fari_el.columns], "platform_fari"]}
        pooled_sel = fit_selected(pooled_fit, val.assign(platform_fari=1.0), variants, seed)
        pooled_median = pooled_fit[ANOMALY_FEATURES].median()
        pooled_iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed, n_jobs=8).fit(
            pooled_fit[ANOMALY_FEATURES].fillna(pooled_median))
        pooled_models = RoverModels(pooled_sel, pooled_iso, pooled_median, {}, pooled_fit)
        pooled_test = _score(pooled_models, test.assign(platform_fari=1.0))
        pooled_val = _score(pooled_models, val.assign(platform_fari=1.0))

        for m in MODELS:
            fari_thr = threshold_for_budget(val[["unit_id", "t"]].assign(score=fixed_val[m]), budget, refractory)
            native_thr = threshold_for_budget(val[["unit_id", "t"]].assign(score=native.val_scores[native_map[m]]),
                                              budget, refractory)
            pooled_thr = threshold_for_budget(val[["unit_id", "t"]].assign(score=pooled_val[m]), budget, refractory)
            rows = {
                "asis": (asis_scores[m], rm.rover_threshold[m]),
                "fixed_rover_thr": (fixed_scores[m], rm.rover_threshold[m]),
                "fixed_fari_thr": (fixed_scores[m], fari_thr),
                "fari_native": (native.scores[native_map[m]], native_thr),
                "pooled": (pooled_test[m], pooled_thr),
            }
            for arm, (score, thr) in rows.items():
                parts[(arm, m)].append(base.assign(score=score, thr=thr, fold_id=fold.fold_id))
        info.append({"fold_id": fold.fold_id, "rover_variant": rm.selected.name, "pooled_variant": pooled_sel.name,
                     "rover_fit_rows": len(rm.fit_frame), "fari_fit_rows": len(fit), "fari_val_positives": int(val["y"].sum()),
                     "fari_test_rows": len(test)})

    table: dict[str, Any] = {}
    scored_frames: dict[str, pd.DataFrame] = {}
    for (arm, m), frames in parts.items():
        scored = pd.concat(frames, ignore_index=True)
        metrics = alert_metrics(scored, fari.events, refractory, horizon_h)
        metrics.pop("lead_times_h", None)
        metrics["ci95"] = bootstrap_ci(per_unit_table(scored, fari.events, refractory, horizon_h), seed=seed)
        metrics["ci95"]["pr_auc"] = prauc_bootstrap(scored, seed=seed)
        metrics["score_is_constant"] = bool(np.ptp(scored["score"].to_numpy()) < 1e-12)
        table.setdefault(arm, {})[m] = metrics
        scored_frames[f"{arm}:{m}"] = scored
    return {"budget_alerts_per_unit_week": budget, "arms": table, "fold_info": info,
            "fari_test_rows": int(sum(i["fari_test_rows"] for i in info)),
            "fari_base_rate": float(scored_frames["fari_native:lgbm"]["y"].mean())}, scored_frames


def rule_indicators(scored: pd.DataFrame, fixed: pd.DataFrame, asis: pd.DataFrame) -> dict[str, Any]:
    """PR-AUC of each physics-rule indicator |x - 1| on its own, on the Fari test rows.

    The rule takes the worst of its indicators; an indicator that is dead (NaN)
    drops out. This shows which indicator carries the rule on Fari."""
    key = ["unit_id", "t"]
    out: dict[str, Any] = {}
    for name, table in (("fari_fixed", fixed), ("fari_asis", asis)):
        rows = scored[[*key, "y"]].merge(table[[*key, *RULE_FEATURES]], on=key, how="left")
        out[name] = {}
        for col in RULE_FEATURES:
            dev = np.abs(rows[col].to_numpy(dtype=float) - 1.0)
            valid = np.isfinite(dev)
            out[name][col] = {"valid_share": float(valid.mean()),
                              "pr_auc": pr_auc(rows["y"].to_numpy(), np.where(valid, dev, 0.0)) if valid.any()
                              else float("nan")}
    return out


def soh_layer(rover: W02Data, fari: W02Data, seed: int) -> dict[str, Any]:
    """Battery SoH forecast at 7 days: Rover-trained LightGBM applied to Fari."""
    h = SOH_HORIZON
    tables = {}
    for name, data in (("rover", rover), ("fari", fari)):
        t = daily_table(data.features, data.battery_truth, [h])
        tables[name] = t[t[f"target_h{h}"].notna()].reset_index(drop=True)
    start = fari.start
    rows = []
    for day in SOH_CUTOFF_DAYS:
        cutoff = start + pd.Timedelta(days=day)
        rover_train = tables["rover"][tables["rover"]["day"] <= cutoff - pd.Timedelta(days=h)]
        for g in sorted(set(fari.groups.values())):
            in_group = tables["fari"]["unit_id"].map(fari.groups) == g
            test = tables["fari"][in_group & (tables["fari"]["day"] >= cutoff)
                                  & (tables["fari"]["day"] < cutoff + pd.Timedelta(days=10))]
            fari_train = tables["fari"][~in_group & (tables["fari"]["day"] <= cutoff - pd.Timedelta(days=h))]
            if test.empty:
                continue
            rows.append(test[["unit_id", "day"]].assign(
                y=test[f"target_h{h}"].to_numpy(),
                persistence=persistence(test, h),
                linear=linear_extrapolation(test, h),
                rover_lgbm=fit_predict_lgbm(rover_train, test, h, seed),
                fari_lgbm=fit_predict_lgbm(fari_train, test, h, seed),
                pooled_lgbm=fit_predict_lgbm(pd.concat([rover_train, fari_train]), test, h, seed)))
    pooled = pd.concat(rows, ignore_index=True)
    out: dict[str, Any] = {"horizon_days": h, "cutoff_days": SOH_CUTOFF_DAYS, "test_rows": len(pooled),
                           "mae_pp": {m: mae(pooled["y"].to_numpy(), pooled[m].to_numpy())
                                      for m in ("persistence", "linear", "rover_lgbm", "fari_lgbm", "pooled_lgbm")}}

    def updates_per_unit_day(data: W02Data) -> float:
        f = data.features.sort_values(["unit_id", "t"])
        changed = f.groupby("unit_id")["bms_soh_last"].diff().abs() > 1e-9
        days = (data.end - data.start) / pd.Timedelta(days=1)
        return float(changed.sum() / (f["unit_id"].nunique() * days))

    out["bms_soh_updates_per_unit_day"] = {"rover": updates_per_unit_day(rover), "fari": updates_per_unit_day(fari)}
    return out


def availability_figure(feature_res: dict[str, Any]) -> Path:
    plotstyle.apply_style()
    table = pd.DataFrame(feature_res["physics_feature_table"])
    cols = ["valid_rover", "valid_fari_asis", "valid_fari_fixed"]
    fig, ax = plt.subplots(figsize=(7.6, 8.2))
    grid = table[cols].to_numpy(dtype=float)
    ax.imshow(grid, cmap=plotstyle.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            constant = bool(table.iloc[i][cols[j].replace("valid", "constant")])
            label = f"{100 * grid[i, j]:.0f} %" + (" const" if constant else "")
            ax.text(j, i, label, ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if grid[i, j] > 0.55 else plotstyle.TEXT)
    ax.set_xticks(range(3), ["Rover", "Fari, Week 2 code as-is", "Fari, platform-conditional"], fontsize=8.5)
    ax.set_yticks(range(len(table)), table["feature"], fontsize=7.5)
    ax.xaxis.tick_top()
    ax.grid(False)
    ax.set_title("Share of prediction times with a valid value (physics / context features)", pad=28)
    path = figure_path("fari_feature_availability.png")
    plotstyle.save(fig, path)
    return path


def transfer_figure(model_res: dict[str, Any]) -> Path:
    plotstyle.apply_style()
    arms = list(ARMS)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.6, 4.0), sharey=True)
    offsets = {"lgbm": 0.22, "iforest": 0.0, "physics_rule": -0.22}
    names = {"lgbm": "LightGBM", "iforest": "Isolation Forest", "physics_rule": "Physics rule"}
    for k, m in enumerate(MODELS):
        for i, arm in enumerate(reversed(arms)):
            r = model_res["arms"][arm][m]
            lo, hi = r["ci95"]["pr_auc"]
            y = i + offsets[m]
            ax.plot([lo, hi], [y, y], color=plotstyle.SERIES[k], lw=2, solid_capstyle="round")
            ax.plot(r["pr_auc"], y, "o", color=plotstyle.SERIES[k], markersize=7, markeredgecolor=plotstyle.SURFACE,
                    markeredgewidth=1.5, label=names[m] if i == 0 else None)
            bx.plot(r["alerts_per_unit_week"], y, "o", color=plotstyle.SERIES[k], markersize=7,
                    markeredgecolor=plotstyle.SURFACE, markeredgewidth=1.5)
    ax.axvline(model_res["fari_base_rate"], color=plotstyle.NEUTRAL, lw=1, ls="--")
    bx.axvline(model_res["budget_alerts_per_unit_week"], color=plotstyle.NEUTRAL, lw=1, ls="--")
    bx.text(model_res["budget_alerts_per_unit_week"], len(arms) - 0.45, " budget", fontsize=8, color=plotstyle.TEXT_SECONDARY)
    ax.text(model_res["fari_base_rate"], len(arms) - 0.45, " base rate", fontsize=8, color=plotstyle.TEXT_SECONDARY)
    ax.set_yticks(range(len(arms)), [ARM_LABEL[a] for a in reversed(arms)], fontsize=8.5)
    ax.set_ylim(-0.6, len(arms) - 0.2)
    ax.set_xlabel("PR-AUC on Fari test units (95 % unit-bootstrap interval)")
    bx.set_xlabel("Alerts per unit-week actually raised on Fari")
    ax.set_title("Ranking quality after transfer")
    bx.set_title("Alert volume against the budget")
    for a in (ax, bx):
        a.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = figure_path("fari_transfer_models.png")
    plotstyle.save(fig, path)
    return path


def shift_figure(rover: pd.DataFrame, fari: pd.DataFrame) -> Path:
    plotstyle.apply_style()
    cols = ["last_cycle_dod_pct", "charge_sessions_24h", "energy_per_km_24h", "vib_norm_mean_24h"]
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 2.9))
    for ax, col in zip(axes, cols, strict=True):
        a, b = rover[col].dropna().to_numpy(), fari[col].dropna().to_numpy()
        both = np.concatenate([a, b])
        lo, hi = (np.percentile(both, [0.5, 99.5]) if len(both) else (0.0, 1.0))
        bins = np.linspace(lo, hi, 40)
        ax.hist(a, bins=bins, density=True, histtype="stepfilled", alpha=0.55, color=plotstyle.SERIES[0], label="Rover")
        ax.hist(b, bins=bins, density=True, histtype="stepfilled", alpha=0.55, color=plotstyle.SERIES[1], label="Fari")
        info = feature_info(col)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel(info.unit or "count", fontsize=8)
        ax.set_yticks([])
        ax.grid(False)
    axes[0].legend(fontsize=8)
    fig.suptitle("Same feature code, different platform: distributions on eligible prediction times", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    path = figure_path("fari_shift_examples.png")
    plotstyle.save(fig, path)
    return path


def run(rover_dir: Path, fari_dir: Path, seed: int = 0) -> dict[str, Any]:
    fari_cfg = load_config(FARI_CONFIG)
    rover = load_w02(rover_dir)
    fari = load_w02(fari_dir, config_path=FARI_CONFIG)
    asis = pd.read_parquet(fari_dir / "features_asis" / "features_hourly.parquet")
    raw = fari_dir / "generated" / "fleet_telemetry.csv"

    results: dict[str, Any] = {
        "synthetic_notice": "The Fari fleet is simulated from configs/fari.yaml. Every parameter in that file is a "
                            "synthetic assumption; none is an InGen hardware specification.",
        "fari_manifest": fari.manifest,
        "contract": {
            "week1_rules_as_is": contract_acceptance(raw, DEFAULT_PROFILE),
            "fari_platform_profile": contract_acceptance(raw, platform_profile(fari_cfg)),
            "profile": {k: list(v) if isinstance(v, tuple) else v for k, v in platform_profile(fari_cfg).items()},
            "privacy_gate": "files with a forbidden (imagery/audio) column are refused before any copy is stored; "
                            "see tests/w03/test_fari_contract.py",
        },
    }
    rover_el, fari_el = rover.eligible(), fari.eligible()
    asis_el = fari_el[["unit_id", "t"]].merge(asis, on=["unit_id", "t"], how="left")
    results["features"] = feature_layer(rover_el, asis_el, fari_el)
    results["models"], scored = model_layer(rover, fari, asis, seed)
    results["models"]["paired_pr_auc_diff"] = {
        f"{a} minus {b}": {"delta": pr_auc(scored[a]["y"].to_numpy(), scored[a]["score"].to_numpy())
                           - pr_auc(scored[b]["y"].to_numpy(), scored[b]["score"].to_numpy()),
                           "ci95": paired_prauc_diff(scored[a], scored[b], seed=seed)} for a, b in PAIRED}
    results["models"]["rule_indicators"] = rule_indicators(scored["fixed_rover_thr:physics_rule"], fari_el, asis_el)
    results["fault_types"] = {"rover": {str(k): int(v) for k, v in rover.events["fault_type"].value_counts().items()},
                              "fari": {str(k): int(v) for k, v in fari.events["fault_type"].value_counts().items()}}
    results["soh"] = soh_layer(rover, fari, seed)
    results["fari_events_in_test"] = results["models"]["arms"]["fari_native"]["lgbm"]["events"]

    figs = [availability_figure(results["features"]), transfer_figure(results["models"]), shift_figure(rover_el, fari_el)]
    with tracked_run("w03_fari_transfer", "fari_generalisation", {"track": "fari", "data_dir": str(fari_dir)}) as run_:
        import mlflow

        mlflow.log_params({"seed": seed, "fari_config_sha256": fari.manifest["config_sha256"],
                           "rover_config_sha256": rover.manifest["config_sha256"],
                           "fari_units": fari.manifest["units"], "fari_days": fari.manifest["days"]})
        log_flat_metrics({"asis": results["contract"]["week1_rules_as_is"]["acceptance_rate_pct"],
                          "profile": results["contract"]["fari_platform_profile"]["acceptance_rate_pct"]},
                         prefix="contract_acceptance_pct.")
        for arm, by_model in results["models"]["arms"].items():
            for m, metrics in by_model.items():
                log_flat_metrics({k: v for k, v in metrics.items() if not isinstance(v, (list, dict, bool))},
                                 prefix=f"{arm}.{m}.")
        log_flat_metrics(results["soh"]["mae_pp"], prefix="soh_mae_pp.")
        for fig in figs:
            mlflow.log_artifact(str(fig))
        log_dict_json(results, "fari_transfer_results.json")
        results["mlflow_run_id"] = run_.info.run_id
    write_json(results, "fari_transfer_results.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rover-dir", type=Path, required=True)
    parser.add_argument("--fari-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    res = run(args.rover_dir, args.fari_dir, args.seed)
    c = res["contract"]
    print("contract acceptance: as-is", c["week1_rules_as_is"]["acceptance_rate_pct"], c["week1_rules_as_is"]["rejection_reasons"],
          "| with profile", c["fari_platform_profile"]["acceptance_rate_pct"])
    f = res["features"]
    print("dead features:", {k: len(v) for k, v in f["dead_features"].items()},
          "constant:", {k: len(v) for k, v in f["constant_features"].items()}, "PSI>0.25:", f["psi_major_shift_count"])
    for arm in ARMS:
        for m in MODELS:
            v = res["models"]["arms"][arm][m]
            print(f"{arm:16s} {m:13s} PR-AUC {v['pr_auc']:.3f} {np.round(v['ci95']['pr_auc'], 3)}  "
                  f"alerts/unit-wk {v['alerts_per_unit_week']:.2f}  recall {v['event_recall']:.2f}  "
                  f"precision {v['precision']:.2f}")
    print("SoH:", res["soh"])
    print("Fari base rate:", res["models"]["fari_base_rate"])
    print("MLflow run:", res["mlflow_run_id"])


if __name__ == "__main__":
    main()
