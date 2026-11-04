"""Deliberate leakage demonstration and leakage audit (W02 evaluation protocol).

Same LightGBM, same features, three protocols:

1. random 80/20 row shuffle                         (wrong)
2. time-ordered with purge, but the same robots    (partly wrong: unit memorisation)
3. unit-grouped walk-forward with 72 h embargo     (proper)

plus a time-proxy probe, the SoH analogue, and the v0.1 simulator's clock
artefact.

    python -m ingen_fleet_health.experiments.leakage_demo --data-dir data/w02
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ingen_fleet_health.config import REPO_ROOT
from ingen_fleet_health.evaluation.metrics import mae, pr_auc
from ingen_fleet_health.evaluation.splits import (
    check_fold,
    random_row_split,
    walk_forward_folds,
)
from ingen_fleet_health.experiments.common import (
    W02Data,
    figure_path,
    load_w02,
    write_json,
)
from ingen_fleet_health.experiments.early_warning_run import (
    CUTOFF_DAYS,
    TEST_SPAN,
    VAL_SPAN,
)
from ingen_fleet_health.features.registry import time_proxy_features
from ingen_fleet_health.labels import early_warning_labels, prediction_grid
from ingen_fleet_health.models.early_warning import (
    LGBM_PARAMS,
    fit_lgbm,
    model_feature_columns,
    positive_proba,
)
from ingen_fleet_health.models.soh_forecast import (
    DAILY_FEATURES,
    SOH_LGBM_PARAMS,
    daily_table,
)
from ingen_fleet_health.tracking import log_dict_json, log_flat_metrics, tracked_run

FORBIDDEN = {"y", "fault_id", "fault_type", "hours_to_onset", "in_active_fault", "post_repair", "censored",
             "eligible", "soh_true_pct", "target_h1", "target_h7", "target_h14"}


def _fit_score(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], seed: int) -> np.ndarray:
    model = fit_lgbm(train[cols], train["y"].to_numpy(), seed)
    return positive_proba(model, test[cols])


def ew_protocols(data: W02Data, eligible: pd.DataFrame, cols: list[str], seed: int) -> dict[str, Any]:
    horizon = pd.Timedelta(hours=float(data.config["labels"]["horizon_hours"]))
    eval_start = data.cutoffs([CUTOFF_DAYS[0]])[0]
    out: dict[str, Any] = {}

    tr, te = random_row_split(len(eligible), 0.2, seed)
    s = _fit_score(eligible.iloc[tr], eligible.iloc[te], cols, seed)
    test = eligible.iloc[te]
    later = (test["t"] >= eval_start).to_numpy()
    out["random_shuffle"] = {"pr_auc": pr_auc(test["y"].to_numpy(), s),
                             "pr_auc_same_period": pr_auc(test["y"].to_numpy()[later], s[later]),
                             "test_rows": len(te), "base_rate": float(test["y"].mean())}

    # time-ordered with purge, same robots in train and test
    ys, ss = [], []
    for c in data.cutoffs(CUTOFF_DAYS):
        train = eligible[eligible["t"] <= c - horizon]
        tst = eligible[(eligible["t"] >= c) & (eligible["t"] < c + TEST_SPAN)]
        ys.append(tst["y"].to_numpy())
        ss.append(_fit_score(train, tst, cols, seed))
    y_all, s_all = np.concatenate(ys), np.concatenate(ss)
    out["time_only_same_units"] = {"pr_auc": pr_auc(y_all, s_all), "test_rows": len(y_all),
                                   "base_rate": float(y_all.mean())}

    folds = walk_forward_folds(eligible, data.groups, data.cutoffs(CUTOFF_DAYS), TEST_SPAN, horizon, VAL_SPAN)
    ys, ss = [], []
    audits = []
    for fold in folds:
        audits.append(check_fold(eligible, fold, horizon))
        train, tst = eligible.iloc[fold.train_idx], eligible.iloc[fold.test_idx]  # fit + val, as in the shuffle
        ys.append(tst["y"].to_numpy())
        ss.append(_fit_score(train, tst, cols, seed))
    y_all, s_all = np.concatenate(ys), np.concatenate(ss)
    out["proper_unit_grouped_walk_forward"] = {"pr_auc": pr_auc(y_all, s_all), "test_rows": len(y_all),
                                               "base_rate": float(y_all.mean())}
    out["audit"] = {
        "folds_checked": len(audits),
        "max_shared_units": max(a["shared_units"] for a in audits),
        "max_duplicate_keys": max(a["duplicate_keys"] for a in audits),
        "min_embargo_hours": min(a["embargo_hours"] for a in audits),
    }
    return out


def time_proxy_probe(data: W02Data, eligible: pd.DataFrame, seed: int) -> dict[str, Any]:
    probe = eligible.copy()
    probe["hours_elapsed"] = (probe["t"] - data.start) / pd.Timedelta(hours=1)
    cols = time_proxy_features() + ["hours_elapsed"]
    res = ew_protocols(data, probe, cols, seed)
    return {"features": cols, **{k: v for k, v in res.items() if k != "audit"}}


def soh_protocols(data: W02Data, seed: int, h: int = 7) -> dict[str, Any]:
    table = daily_table(data.features, data.battery_truth, [h]).drop(columns="t").rename(columns={"day": "t"})
    table = table[table[f"target_h{h}"].notna()].reset_index(drop=True)

    def fit_pred(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        model = lgb.LGBMRegressor(**SOH_LGBM_PARAMS, random_state=seed)
        model.fit(train[DAILY_FEATURES], train[f"target_h{h}"] - train["bms_soh_mean_24h"])
        return test["bms_soh_mean_24h"].to_numpy() + model.predict(test[DAILY_FEATURES])

    tr, te = random_row_split(len(table), 0.2, seed)
    rand = mae(table.iloc[te][f"target_h{h}"], fit_pred(table.iloc[tr], table.iloc[te]))
    folds = walk_forward_folds(table, data.groups, data.cutoffs(CUTOFF_DAYS), TEST_SPAN, pd.Timedelta(days=h),
                               pd.Timedelta(0))
    ys, ps = [], []
    for fold in folds:
        if len(fold.test_idx) == 0 or len(fold.fit_idx) < 200:
            continue
        test = table.iloc[fold.test_idx]
        ys.append(test[f"target_h{h}"].to_numpy())
        ps.append(fit_pred(table.iloc[fold.fit_idx], test))
    return {"horizon_days": h, "random_shuffle_mae": rand, "proper_mae": mae(np.concatenate(ys), np.concatenate(ps))}


def v01_clock_artefact(seed: int) -> dict[str, Any] | None:
    """v0.1 put every fault onset at the same instant: the clock alone predicts the label."""
    path = REPO_ROOT / "data" / "generated" / "fleet_fault_events.csv"
    if not path.exists():
        return None
    ev = pd.read_csv(path)
    start = pd.Timestamp("2026-10-26T00:00:00Z")
    end = start + pd.Timedelta(days=60)
    ev = ev.assign(fault_id=[f"V{i}" for i in range(len(ev))],
                   onset_utc=pd.to_datetime(ev["onset_utc"], utc=True), repair_end_utc=end)
    grid = prediction_grid(sorted(ev["unit_id"].unique()), start, end)
    lab = early_warning_labels(grid, ev, end)
    lab = lab[lab["eligible"]].reset_index(drop=True)
    lab["hours_elapsed"] = (lab["t"] - start) / pd.Timedelta(hours=1)
    tr, te = random_row_split(len(lab), 0.2, seed)
    s = _fit_score(lab.iloc[tr], lab.iloc[te], ["hours_elapsed"], seed)
    return {"units": int(ev["unit_id"].nunique()), "distinct_onset_times": int(ev["onset_utc"].nunique()),
            "feature": "hours_elapsed only", "random_shuffle_pr_auc": pr_auc(lab.iloc[te]["y"].to_numpy(), s),
            "base_rate": float(lab["y"].mean())}


def leakage_plot(ew: dict[str, Any], probe: dict[str, Any]) -> Path:
    labels = ["random\nshuffle", "time-ordered,\nsame robots", "unit-grouped\nwalk-forward"]
    keys = ["random_shuffle", "time_only_same_units", "proper_unit_grouped_walk_forward"]
    main = [ew[k]["pr_auc"] for k in keys]
    prox = [probe[k]["pr_auc"] for k in keys]
    base = ew["proper_unit_grouped_walk_forward"]["base_rate"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    b1 = ax.bar(x - 0.18, main, 0.36, color="#1d4ed8", label="LightGBM, physics + rolling features")
    b2 = ax.bar(x + 0.18, prox, 0.36, color="#d97706", label="probe: time-proxy features only")
    ax.axhline(base, color="#6b7280", ls="--", lw=1, label=f"no-skill base rate ({base:.3f})")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.2f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("PR-AUC (72 h onset warning)")
    ax.set_title("Same model, three split protocols")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(main + prox) * 1.25)
    fig.tight_layout()
    path = figure_path("leakage_demo.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(data_dir: Path, seed: int = 0) -> dict[str, Any]:
    data = load_w02(data_dir)
    eligible = data.eligible()
    cols = model_feature_columns()
    assert not (set(cols) & FORBIDDEN), "label / ground-truth columns among features"
    ew = ew_protocols(data, eligible, cols, seed)
    probe = time_proxy_probe(data, eligible, seed)
    results = {
        "early_warning": ew,
        "time_proxy_probe": probe,
        "soh": soh_protocols(data, seed),
        "v01_clock_artefact": v01_clock_artefact(seed),
        "overstatement_pr_auc": ew["random_shuffle"]["pr_auc"] - ew["proper_unit_grouped_walk_forward"]["pr_auc"],
        "forbidden_columns_in_features": sorted(set(cols) & FORBIDDEN),
    }
    fig = leakage_plot(ew, probe)
    with tracked_run("w02_leakage_demo", "leakage_demo", {"track": "evaluation"}) as run_:
        import mlflow

        mlflow.log_params({"lgbm_" + k: v for k, v in LGBM_PARAMS.items()})
        for k in ("random_shuffle", "time_only_same_units", "proper_unit_grouped_walk_forward"):
            log_flat_metrics({"pr_auc": ew[k]["pr_auc"]}, prefix=f"ew.{k}.")
            log_flat_metrics({"pr_auc": probe[k]["pr_auc"]}, prefix=f"probe.{k}.")
        log_flat_metrics({"random_mae": results["soh"]["random_shuffle_mae"],
                          "proper_mae": results["soh"]["proper_mae"]}, prefix="soh_h7.")
        mlflow.log_artifact(str(fig))
        log_dict_json(results, "leakage_results.json")
        results["mlflow_run_id"] = run_.info.run_id
    write_json(results, "leakage_results.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    res = run(args.data_dir, args.seed)
    ew = res["early_warning"]
    for k in ("random_shuffle", "time_only_same_units", "proper_unit_grouped_walk_forward"):
        print(f"{k:34s} PR-AUC {ew[k]['pr_auc']:.3f}   time-proxy probe {res['time_proxy_probe'][k]['pr_auc']:.3f}")
    print("SoH h=7:", res["soh"])
    print("v0.1 clock artefact:", res["v01_clock_artefact"])
    print("MLflow run:", res["mlflow_run_id"])


if __name__ == "__main__":
    main()
