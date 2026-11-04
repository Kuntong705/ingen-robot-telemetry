"""Run model track 1 (battery SoH forecasting) and log to MLflow.

    python -m ingen_fleet_health.experiments.soh_run --data-dir data/w02
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ingen_fleet_health.evaluation.metrics import mae, rmse
from ingen_fleet_health.evaluation.splits import check_fold, walk_forward_folds
from ingen_fleet_health.experiments.common import figure_path, load_w02, write_json
from ingen_fleet_health.models.soh_forecast import (
    DAILY_FEATURES,
    SOH_LGBM_PARAMS,
    daily_table,
    fit_predict_lgbm,
    fit_predict_lstm,
    linear_extrapolation,
    persistence,
)
from ingen_fleet_health.tracking import log_dict_json, log_flat_metrics, tracked_run

HORIZONS = [1, 7, 14]
CUTOFF_DAYS = [20, 30, 40, 50]
TEST_SPAN = pd.Timedelta(days=10)
MIN_TRAIN_ROWS = 200
MODELS = ["persistence", "linear", "lgbm", "lstm"]


def unit_bootstrap_mae(frame: pd.DataFrame, col: str, n_boot: int = 1000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    err = (frame[col] - frame["y"]).abs()
    per_unit = err.groupby(frame["unit_id"]).agg(["sum", "count"]).to_numpy()
    vals = []
    for _ in range(n_boot):
        s = per_unit[rng.integers(0, len(per_unit), len(per_unit))].sum(axis=0)
        vals.append(s[0] / s[1])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def paired_bootstrap_skill(frame: pd.DataFrame, model: str, ref: str, n_boot: int = 1000, seed: int = 0) -> list[float]:
    """CI of MAE(model) - MAE(ref), resampling units (negative = model better)."""
    rng = np.random.default_rng(seed)
    d = ((frame[model] - frame["y"]).abs() - (frame[ref] - frame["y"]).abs())
    per_unit = d.groupby(frame["unit_id"]).agg(["sum", "count"]).to_numpy()
    vals = []
    for _ in range(n_boot):
        s = per_unit[rng.integers(0, len(per_unit), len(per_unit))].sum(axis=0)
        vals.append(s[0] / s[1])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def run(data_dir: Path, seed: int = 0, lstm_epochs: int = 60) -> dict[str, Any]:
    data = load_w02(data_dir)
    table = daily_table(data.features, data.battery_truth, HORIZONS)
    # fold on the forecast-origin day d (prediction time is d + 1 day, 00:00 UTC)
    table = table.drop(columns="t").rename(columns={"day": "t"}).sort_values(["unit_id", "t"]).reset_index(drop=True)
    results: dict[str, Any] = {"horizons_days": HORIZONS, "cutoff_days": CUTOFF_DAYS, "features": DAILY_FEATURES,
                               "by_horizon": {}, "skipped_folds": []}
    pooled_all = []
    for h in HORIZONS:
        valid = table[table[f"target_h{h}"].notna()].reset_index(drop=True)
        horizon = pd.Timedelta(days=h)
        folds = walk_forward_folds(valid, data.groups, data.cutoffs(CUTOFF_DAYS), TEST_SPAN, horizon,
                                   pd.Timedelta(0))
        parts = []
        for fold in folds:
            if len(fold.test_idx) == 0:
                continue
            if len(fold.fit_idx) < MIN_TRAIN_ROWS:
                results["skipped_folds"].append({"h": h, "fold_id": fold.fold_id, "train_rows": len(fold.fit_idx)})
                continue
            check_fold(valid, fold, horizon)
            train, test = valid.iloc[fold.fit_idx], valid.iloc[fold.test_idx]
            out = test[["unit_id", "t"]].copy()
            out["y"] = test[f"target_h{h}"].to_numpy()
            out["persistence"] = persistence(test, h)
            out["linear"] = linear_extrapolation(test, h)
            out["lgbm"] = fit_predict_lgbm(train, test, h, seed)
            out["lstm"] = fit_predict_lstm(valid, fold.fit_idx, fold.test_idx, h, seed, lstm_epochs)
            out["fold_id"] = fold.fold_id
            parts.append(out)
        pooled = pd.concat(parts, ignore_index=True)
        pooled["h"] = h
        pooled_all.append(pooled)
        res_h: dict[str, Any] = {"test_rows": len(pooled), "test_units": int(pooled["unit_id"].nunique()),
                                 "folds_used": int(pooled["fold_id"].nunique())}
        for m in MODELS:
            res_h[m] = {"mae": mae(pooled["y"], pooled[m]), "rmse": rmse(pooled["y"], pooled[m]),
                        "mae_ci95": unit_bootstrap_mae(pooled, m, seed=seed)}
            if m != "persistence":
                res_h[m]["mae_minus_persistence_ci95"] = paired_bootstrap_skill(pooled, m, "persistence", seed=seed)
                res_h[m]["mae_minus_linear_ci95"] = paired_bootstrap_skill(pooled, m, "linear", seed=seed)
        results["by_horizon"][str(h)] = res_h

    fig_path = soh_plot(results)
    example_path = soh_example_plot(table, pd.concat(pooled_all, ignore_index=True))

    with tracked_run("w02_soh_forecast", "soh_main", {"track": "soh_forecast", "data_dir": str(data_dir)}) as run_:
        import mlflow

        mlflow.log_params({"horizons_days": str(HORIZONS), "cutoff_days": str(CUTOFF_DAYS), "test_span_days": 10,
                           "lstm_epochs": lstm_epochs, "lstm_hidden": 32, "seq_len_days": 14,
                           "simulator_config_sha256": data.manifest["config_sha256"],
                           **{f"lgbm_{k}": v for k, v in SOH_LGBM_PARAMS.items()}})
        for h, res_h in results["by_horizon"].items():
            for m in MODELS:
                log_flat_metrics({"mae": res_h[m]["mae"], "rmse": res_h[m]["rmse"]}, prefix=f"h{h}.{m}.")
        mlflow.log_artifact(str(fig_path))
        mlflow.log_artifact(str(example_path))
        log_dict_json(results, "soh_results.json")
        results["mlflow_run_id"] = run_.info.run_id
    write_json(results, "soh_results.json")
    return results


def soh_plot(results: dict[str, Any]) -> Path:
    colors = {"persistence": "#6b7280", "linear": "#a16207", "lgbm": "#1d4ed8", "lstm": "#7c3aed"}
    fig, ax = plt.subplots(figsize=(6, 3.8))
    hs = [int(h) for h in results["by_horizon"]]
    width = 0.2
    for i, m in enumerate(MODELS):
        vals = [results["by_horizon"][str(h)][m]["mae"] for h in hs]
        ci = np.array([results["by_horizon"][str(h)][m]["mae_ci95"] for h in hs])
        x = np.arange(len(hs)) + (i - 1.5) * width
        ax.bar(x, vals, width, color=colors[m], label=m)
        ax.errorbar(x, vals, yerr=[np.array(vals) - ci[:, 0], ci[:, 1] - np.array(vals)], fmt="none",
                    ecolor="#111111", lw=0.8, capsize=2)
    ax.set_xticks(np.arange(len(hs)), [f"{h} d" for h in hs])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("MAE of SoH (percentage points)")
    ax.set_title("SoH forecast error on unseen robots (95% unit-bootstrap CI)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = figure_path("soh_mae_by_horizon.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def soh_example_plot(table: pd.DataFrame, pooled: pd.DataFrame) -> Path:
    sub = pooled[pooled["h"] == 7]
    unit = sub.groupby("unit_id").size().idxmax()
    u = table[table["unit_id"] == unit]
    p = sub[sub["unit_id"] == unit].sort_values("t")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(u["t"], u["soh_true_pct"], color="#111111", lw=1.5, label="true SoH (ground truth)")
    ax.plot(u["t"], u["bms_soh_mean_24h"], color="#9ca3af", lw=1, label="BMS estimate (24 h mean)")
    shift = pd.Timedelta(days=7)
    for m, c in (("linear", "#a16207"), ("lgbm", "#1d4ed8"), ("lstm", "#7c3aed")):
        ax.plot(p["t"] + shift, p[m], color=c, lw=1, ls="--", label=f"{m} forecast, 7 d ahead")
    ax.set_ylabel("SoH (%)")
    ax.set_title(f"{unit}: 7-day-ahead SoH forecasts on an unseen robot")
    ax.legend(fontsize=7)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = figure_path("soh_example_unit.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lstm-epochs", type=int, default=60)
    args = parser.parse_args()
    res = run(args.data_dir, args.seed, args.lstm_epochs)
    for h, r in res["by_horizon"].items():
        print(f"h={h:>2} d  " + "  ".join(f"{m} MAE {r[m]['mae']:.3f}" for m in MODELS))
    print("MLflow run:", res["mlflow_run_id"])


if __name__ == "__main__":
    main()
