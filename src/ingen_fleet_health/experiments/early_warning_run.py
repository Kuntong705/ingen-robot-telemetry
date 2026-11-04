"""Run model track 2 (fault early warning) end to end and log to MLflow.

    python -m ingen_fleet_health.experiments.early_warning_run --data-dir data/w02
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from ingen_fleet_health.evaluation.metrics import (
    HOURS_PER_WEEK,
    evaluate_alerts,
    events_in_scored_period,
    pr_auc,
    raise_alerts,
    threshold_for_budget,
)
from ingen_fleet_health.evaluation.splits import Fold, check_fold, walk_forward_folds
from ingen_fleet_health.experiments.common import (
    W02Data,
    figure_path,
    load_w02,
    write_json,
)
from ingen_fleet_health.models.early_warning import (
    LGBM_PARAMS,
    model_feature_columns,
    score_fold,
)
from ingen_fleet_health.tracking import log_dict_json, log_flat_metrics, tracked_run

CUTOFF_DAYS = [20, 30, 40, 50]
TEST_SPAN = pd.Timedelta(days=10)
VAL_SPAN = pd.Timedelta(days=6)
MODELS = ["no_skill", "physics_rule", "lgbm_all", "lgbm_physics", "lgbm_selected", "iforest"]
BUDGETS = [0.25, 0.5, 1.0, 2.0]


def build_folds(data: W02Data, eligible: pd.DataFrame, horizon: pd.Timedelta) -> list[Fold]:
    folds = walk_forward_folds(eligible, data.groups, data.cutoffs(CUTOFF_DAYS), TEST_SPAN, horizon, VAL_SPAN)
    for fold in folds:
        check_fold(eligible, fold, horizon)
    return folds


def pooled_scores(eligible: pd.DataFrame, fold_scores: list[Any], model: str,
                  budget: float, refractory: float) -> pd.DataFrame:
    parts = []
    for fs in fold_scores:
        val = eligible.iloc[fs.fold.val_idx][["unit_id", "t"]].assign(score=fs.val_scores[model])
        thr = threshold_for_budget(val, budget, refractory)
        test = eligible.iloc[fs.fold.test_idx][["unit_id", "t", "y", "fault_id", "fault_type"]]
        parts.append(test.assign(score=fs.scores[model], thr=thr, fold_id=fs.fold.fold_id))
    return pd.concat(parts, ignore_index=True)


def alert_metrics(scored: pd.DataFrame, events: pd.DataFrame, refractory: float, horizon_h: float) -> dict[str, Any]:
    ev = events_in_scored_period(scored, events)
    m = evaluate_alerts(scored, ev, None, refractory, horizon_h)
    m["pr_auc"] = pr_auc(scored["y"].to_numpy(), scored["score"].to_numpy())
    m["base_rate"] = float(scored["y"].mean())
    return m


def per_unit_table(scored: pd.DataFrame, events: pd.DataFrame, refractory: float, horizon_h: float) -> pd.DataFrame:
    """Per-unit counts so CIs can be bootstrapped over robots."""
    alerts = raise_alerts(scored, None, refractory)
    lab = scored.set_index(["unit_id", "t"])["y"]
    alerts["tp"] = lab.reindex(pd.MultiIndex.from_frame(alerts[["unit_id", "t"]])).fillna(0).to_numpy()
    ev = events_in_scored_period(scored, events)
    h = pd.Timedelta(hours=horizon_h)
    rows = []
    for unit, frame in scored.groupby("unit_id"):
        ua = alerts[alerts["unit_id"] == unit]
        ue = ev[ev["unit_id"] == unit]
        detected = sum(
            bool(((ua["t"] >= e.onset_utc - h) & (ua["t"] < e.onset_utc)).any()) for e in ue.itertuples()
        )
        rows.append({"unit_id": unit, "rows": len(frame), "alerts": len(ua), "tp": float(ua["tp"].sum()),
                     "events": len(ue), "detected": detected})
    return pd.DataFrame(rows)


def bootstrap_ci(units: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    stats: dict[str, list[float]] = {"event_recall": [], "precision": [], "false_alarms_per_unit_week": []}
    arr = units[["rows", "alerts", "tp", "events", "detected"]].to_numpy(dtype=float)
    for _ in range(n_boot):
        s = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
        rows, alerts, tp, events, detected = s
        if events > 0:
            stats["event_recall"].append(detected / events)
        if alerts > 0:
            stats["precision"].append(tp / alerts)
        stats["false_alarms_per_unit_week"].append((alerts - tp) / (rows / HOURS_PER_WEEK))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2
            for k, v in stats.items()}


def prauc_bootstrap(scored: pd.DataFrame, n_boot: int = 300, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    groups = [g for _, g in scored.groupby("unit_id")]
    vals = []
    for _ in range(n_boot):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        v = pr_auc(sample["y"].to_numpy(), sample["score"].to_numpy())
        if np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def reliability_plot(scored_cal: pd.DataFrame, scored_raw: pd.DataFrame) -> tuple[Path, dict[str, float]]:
    fig, (ax, hx) = plt.subplots(2, 1, figsize=(5.2, 6.2), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="perfect calibration")
    out = {}
    for frame, label, color in ((scored_raw, "LightGBM uncalibrated", "#c2410c"), (scored_cal, "LightGBM calibrated", "#1d4ed8")):
        y, p = frame["y"].to_numpy(), frame["score"].to_numpy()
        frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y, p)
        out[label] = float(brier)
        ax.plot(mean_p, frac, marker="o", color=color, label=f"{label} (Brier {brier:.4f})")
        hx.hist(p, bins=20, range=(0, 1), color=color, alpha=0.5, log=True)
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability on unseen robots (pooled test folds, quantile bins)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    hx.set_xlabel("Predicted probability of onset within 72 h")
    hx.set_ylabel("rows (log)")
    fig.tight_layout()
    path = figure_path("ew_reliability.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, out


def budget_plot(curve: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    colors = {"physics_rule": "#6b7280", "lgbm_selected": "#1d4ed8", "iforest": "#059669"}
    for model, g in curve.groupby("model"):
        if model not in colors:
            continue
        axes[0].plot(g["budget"], g["event_recall"], marker="o", color=colors[model], label=model)
        axes[1].plot(g["budget"], g["precision"], marker="o", color=colors[model], label=model)
    axes[0].set_ylabel("Event recall (onsets warned 0-72 h ahead)")
    axes[1].set_ylabel("Alert precision")
    for ax in axes:
        ax.set_xlabel("Alert budget (alerts per unit-week, set on validation)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path = figure_path("ew_budget_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def paired_prauc_diff(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 300, seed: int = 0) -> list[float]:
    """95 % CI of PR-AUC(a) - PR-AUC(b) on the same test rows, resampling robots."""
    rng = np.random.default_rng(seed)
    key = ["unit_id", "t"]
    joined = a[key + ["y", "score"]].merge(b[key + ["score"]], on=key, suffixes=("_a", "_b"))
    groups = [g for _, g in joined.groupby("unit_id")]
    vals = []
    for _ in range(n_boot):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        da = pr_auc(s["y"].to_numpy(), s["score_a"].to_numpy())
        db = pr_auc(s["y"].to_numpy(), s["score_b"].to_numpy())
        if np.isfinite(da) and np.isfinite(db):
            vals.append(da - db)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def timeline_plot(pooled: dict[str, pd.DataFrame], events: pd.DataFrame, refractory: float) -> Path:
    """One unseen robot with an in-window onset: calibrated risk, threshold, alerts, onset."""
    sc = pooled["lgbm_selected"]
    ev = events_in_scored_period(sc, events).sort_values("onset_utc")
    ev = ev[ev["fault_type"] == "bearing_wear"] if (ev["fault_type"] == "bearing_wear").any() else ev
    e = ev.iloc[len(ev) // 2]
    u = sc[sc["unit_id"] == e.unit_id].sort_values("t")
    alerts = raise_alerts(u, None, refractory)
    # break the line where rows are excluded (active fault / post-repair) instead of interpolating
    u = u.set_index("t").reindex(pd.date_range(u["t"].min(), u["t"].max(), freq="h")).rename_axis("t").reset_index()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(u["t"], u["score"], color="#1d4ed8", lw=1, label="calibrated P(onset within 72 h)")
    ax.step(u["t"], u["thr"], color="#6b7280", lw=1, ls="--", where="post", label="alert threshold (set on validation)")
    ax.axvspan(e.onset_utc - pd.Timedelta(hours=72), e.onset_utc, color="#fecaca", alpha=0.5, label="72 h before onset")
    ax.axvline(e.onset_utc, color="#dc2626", lw=1.2, label=f"qualifying onset ({e.fault_type})")
    for i, t in enumerate(alerts["t"]):
        ax.axvline(t, color="#f59e0b", lw=1.5, label="alert" if i == 0 else None)
    ax.set_ylabel("risk")
    ax.set_title(f"{e.unit_id} (never seen in training): risk score, alerts and fault onset")
    ax.legend(fontsize=7, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = figure_path("ew_unit_timeline.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def false_alarm_breakdown(scored: pd.DataFrame, events: pd.DataFrame, refractory: float) -> dict[str, float]:
    """Where do 'false' alarms fall? Uses the hidden initiation time for analysis only.

    precursor_early: degradation already under way (initiation <= t < onset - 72 h) -- a real
    developing fault that the 72 h label calls negative; healthy: no fault developing."""
    alerts = raise_alerts(scored, None, refractory)
    lab = scored.set_index(["unit_id", "t"])["y"]
    alerts["y"] = lab.reindex(pd.MultiIndex.from_frame(alerts[["unit_id", "t"]])).fillna(0).to_numpy()
    fa = alerts[alerts["y"] == 0]
    early = np.zeros(len(fa), dtype=bool)
    for ev in events.itertuples(index=False):
        early |= ((fa["unit_id"] == ev.unit_id) & (fa["t"] >= ev.initiation_utc)
                  & (fa["t"] < ev.onset_utc - pd.Timedelta(hours=72))).to_numpy()
    n = len(fa)
    return {"false_alarms": float(n), "share_in_early_precursor": float(early.mean()) if n else float("nan"),
            "share_healthy": float(1 - early.mean()) if n else float("nan")}


def run(data_dir: Path, seed: int = 0) -> dict[str, Any]:
    data = load_w02(data_dir)
    cfg = data.config
    horizon_h = float(cfg["labels"]["horizon_hours"])
    horizon = pd.Timedelta(hours=horizon_h)
    refractory = float(cfg["alerting"]["refractory_hours"])
    budget = float(cfg["alerting"]["budget_alerts_per_unit_week"])
    eligible = data.eligible()
    features = model_feature_columns()
    folds = build_folds(data, eligible, horizon)

    fold_scores = [score_fold(eligible, f, features, seed) for f in folds]

    results: dict[str, Any] = {"budget_alerts_per_unit_week": budget, "horizon_hours": horizon_h,
                               "folds": len(folds), "feature_count": len(features), "models": {}}
    pooled = {m: pooled_scores(eligible, fold_scores, m, budget, refractory) for m in MODELS}
    for m in MODELS:
        metrics = alert_metrics(pooled[m], data.events, refractory, horizon_h)
        if m == "no_skill":
            # a constant score has PR-AUC = base rate by definition; pooling folds with different
            # training prevalences would otherwise inject spurious ranking between folds
            metrics["pr_auc"] = metrics["base_rate"]
        units = per_unit_table(pooled[m], data.events, refractory, horizon_h)
        metrics["ci95"] = bootstrap_ci(units, seed=seed)
        metrics["ci95"]["pr_auc"] = prauc_bootstrap(pooled[m], seed=seed) if m != "no_skill" else [
            metrics["base_rate"], metrics["base_rate"]]
        metrics["false_alarm_breakdown"] = false_alarm_breakdown(pooled[m], data.events, refractory)
        metrics["fold_pr_auc"] = [pr_auc(g["y"].to_numpy(), g["score"].to_numpy()) for _, g in pooled[m].groupby("fold_id")]
        results["models"][m] = metrics

    for m in ("physics_rule", "iforest"):
        results["models"]["lgbm_selected"][f"pr_auc_minus_{m}_ci95"] = paired_prauc_diff(
            pooled["lgbm_selected"], pooled[m], seed=seed)
    results["models"]["iforest"]["pr_auc_minus_physics_rule_ci95"] = paired_prauc_diff(
        pooled["iforest"], pooled["physics_rule"], seed=seed)
    pred_dir = data_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([pooled[m].assign(model=m) for m in MODELS], ignore_index=True).to_parquet(
        pred_dir / "early_warning_oof.parquet", index=False)
    timeline_fig = timeline_plot(pooled, data.events, refractory)

    raw_parts = [eligible.iloc[fs.fold.test_idx][["y"]].assign(score=fs.scores["lgbm_selected_raw"])
                 for fs in fold_scores]
    raw = pd.concat(raw_parts, ignore_index=True)
    rel_path, briers = reliability_plot(pooled["lgbm_selected"], raw)
    results["calibration"] = {"brier": briers, "figure": str(rel_path.name)}

    curve_rows = []
    for b in BUDGETS:
        for m in ("physics_rule", "lgbm_selected", "iforest"):
            sc = pooled_scores(eligible, fold_scores, m, b, refractory)
            mm = alert_metrics(sc, data.events, refractory, horizon_h)
            curve_rows.append({"model": m, "budget": b, **{k: v for k, v in mm.items() if k != "lead_times_h"}})
    curve = pd.DataFrame(curve_rows)
    results["budget_curve"] = curve.to_dict(orient="records")
    budget_fig = budget_plot(curve)

    importance = pd.concat([fs.feature_importance for fs in fold_scores if fs.feature_importance is not None], axis=1)
    top = importance.mean(axis=1).sort_values(ascending=False).head(15)
    results["lgbm_top_features_gain"] = {k: float(v) for k, v in top.items()}

    results["unseen_fault_type"] = leave_one_fault_type_out(eligible, folds, fold_scores, data, features,
                                                            budget, refractory, horizon_h, seed)
    results["fold_info"] = [{"fold_id": fs.fold.fold_id, **fs.info} for fs in fold_scores]

    with tracked_run("w02_early_warning", "ew_main", {"track": "early_warning", "data_dir": str(data_dir)}) as run_:
        import mlflow

        mlflow.log_params({"horizon_hours": horizon_h, "budget": budget, "refractory_hours": refractory,
                           "cutoff_days": str(CUTOFF_DAYS), "test_span_days": 10, "val_span_days": 6,
                           "n_unit_groups": len(set(data.groups.values())), "n_features": len(features),
                           "simulator_config_sha256": data.manifest["config_sha256"],
                           **{f"lgbm_{k}": v for k, v in LGBM_PARAMS.items()}})
        for m, metrics in results["models"].items():
            log_flat_metrics({k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}, prefix=f"{m}.")
        log_flat_metrics(results["calibration"]["brier"], prefix="brier.")
        mlflow.log_artifact(str(rel_path))
        mlflow.log_artifact(str(budget_fig))
        mlflow.log_artifact(str(timeline_fig))
        log_dict_json(results, "early_warning_results.json")
        results["mlflow_run_id"] = run_.info.run_id
    write_json(results, "early_warning_results.json")
    return results


def leave_one_fault_type_out(eligible: pd.DataFrame, folds: list[Fold], fold_scores: list[Any], data: W02Data,
                             features: list[str], budget: float, refractory: float, horizon_h: float,
                             seed: int) -> dict[str, Any]:
    """Faults not seen in training: drop one fault type's positive rows from
    training/validation, then measure recall on that type's onsets only.
    Isolation Forest never uses labels, so its main scores are reused."""
    out: dict[str, Any] = {}
    for ftype in sorted(data.events["fault_type"].unique()):  # ~3 x 20 refits
        hidden = (eligible["fault_type"] == ftype).to_numpy()
        parts: dict[str, list[pd.DataFrame]] = {"lgbm_without_type": [], "iforest": [], "physics_rule": []}
        for fold, fs in zip(folds, fold_scores, strict=True):
            keep_fit = fold.fit_idx[~hidden[fold.fit_idx]]
            keep_val = fold.val_idx[~hidden[fold.val_idx]]
            reduced = Fold(fold.fold_id, fold.cutoff, fold.group, fold.test_units, keep_fit, keep_val, fold.test_idx)
            s = score_fold(eligible, reduced, features, seed, models=("lgbm_all", "lgbm_physics", "lgbm_selected"))
            test = eligible.iloc[fold.test_idx][["unit_id", "t", "y"]]
            for name, score, val_score, val_idx in (
                ("lgbm_without_type", s.scores["lgbm_selected"], s.val_scores["lgbm_selected"], keep_val),
                ("iforest", fs.scores["iforest"], fs.val_scores["iforest"], fold.val_idx),
                ("physics_rule", fs.scores["physics_rule"], fs.val_scores["physics_rule"], fold.val_idx),
            ):
                val = eligible.iloc[val_idx][["unit_id", "t"]].assign(score=val_score)
                thr = threshold_for_budget(val, budget, refractory)
                parts[name].append(test.assign(score=score, thr=thr))
        type_events = data.events[data.events["fault_type"] == ftype]
        out[ftype] = {}
        for name, frames in parts.items():
            scored = pd.concat(frames, ignore_index=True)
            m = alert_metrics(scored, type_events, refractory, horizon_h)
            out[ftype][name] = {k: m[k] for k in ("events", "events_detected", "event_recall", "median_lead_time_h")}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    res = run(args.data_dir, args.seed)
    for m, v in res["models"].items():
        print(f"{m:14s} PR-AUC {v['pr_auc']:.3f}  recall {v['event_recall']:.2f}  precision {v['precision']:.2f}  "
              f"FA/unit-wk {v['false_alarms_per_unit_week']:.2f}  lead {v['median_lead_time_h']:.0f} h")
    print("MLflow run:", res["mlflow_run_id"])


if __name__ == "__main__":
    main()
