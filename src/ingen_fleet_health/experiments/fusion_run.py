"""Telemetry + text fusion and modality ablation, logged to MLflow.

    python -m ingen_fleet_health.experiments.fusion_run --data-dir data/w02

Same 20 unit-grouped walk-forward folds as Week 2. Inside each fold the
ticket classifier is fitted on training-unit tickets whose label was confirmed
by the training cut-off, its probabilities are aggregated as-of into text
features, and every arm (telemetry only, text only, early and late fusion) is
scored on the identical test rows.

``--sensitivity`` additionally regenerates the corpus with other operator
notice rates and repeats the fusion, because on synthetic text the value of
the text modality depends on that assumption.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ingen_fleet_health import plotstyle
from ingen_fleet_health.evaluation.metrics import (
    events_in_scored_period,
    pr_auc,
    raise_alerts,
)
from ingen_fleet_health.experiments.common import W02Data, load_w02
from ingen_fleet_health.experiments.early_warning_run import (
    alert_metrics,
    bootstrap_ci,
    build_folds,
    paired_prauc_diff,
    per_unit_table,
    pooled_scores,
    prauc_bootstrap,
)
from ingen_fleet_health.experiments.w03_common import (
    TextData,
    figure_path,
    load_text,
    text_train_mask,
    write_json,
)
from ingen_fleet_health.models.early_warning import model_feature_columns
from ingen_fleet_health.models.fusion import (
    ALL_ARMS,
    FusionFoldScores,
    health_score,
    score_fusion_fold,
)
from ingen_fleet_health.models.text_classifier import C_GRID, KINDS, crossfit_fold
from ingen_fleet_health.text.corpus import (
    generate_corpus,
    load_corpus_truth,
    write_corpus,
)
from ingen_fleet_health.text.features import TEXT_FEATURES, asof_text_features
from ingen_fleet_health.tracking import log_dict_json, log_flat_metrics, tracked_run

BUDGETS = [0.25, 0.5, 1.0, 2.0]
SENSITIVITY_RATES = [0.5, 2.0, 4.0]
PAIRS = [
    ("text_only", "no_skill"),
    ("text_only", "telemetry_lgbm"),
    ("early_fusion", "telemetry_lgbm"),
    ("late_noisy_or", "telemetry_lgbm"),
    ("late_stack", "telemetry_lgbm"),
    ("late_stack_iforest", "telemetry_iforest"),
    ("early_fusion", "late_stack"),
    ("early_fusion", "telemetry_iforest"),
    ("late_stack_iforest", "early_fusion"),
]
LABELS = {
    "no_skill": "No skill (base rate)",
    "physics_rule": "Physics rule (telemetry)",
    "telemetry_lgbm": "LightGBM (telemetry only)",
    "telemetry_iforest": "Isolation Forest (telemetry only)",
    "text_only": "Text only",
    "early_fusion": "Early fusion (LightGBM, telemetry + text)",
    "late_noisy_or": "Late fusion: noisy-OR (LightGBM + text)",
    "late_stack": "Late fusion: stacked (LightGBM + text)",
    "late_stack_iforest": "Late fusion: stacked (Isolation Forest + text)",
}


def fold_text_features(eligible: pd.DataFrame, fold: Any, text: TextData, horizon: pd.Timedelta, seed: int,
                       kinds: tuple[str, ...] = KINDS, c_grid: tuple[float, ...] = C_GRID,
                       ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """As-of text features for the fold's fit, validation and test rows."""
    train_end = fold.cutoff - horizon
    train_mask = text_train_mask(text, fold.group, train_end)
    fs = crossfit_fold(text.inputs, text.labels, text.group, train_mask, fold.group, kinds, c_grid, seed)
    rows = np.concatenate([fold.fit_idx, fold.val_idx, fold.test_idx])
    grid = eligible.iloc[rows][["unit_id", "t"]]
    feats = asof_text_features(text.corpus, fs.proba[fs.selected_kind], grid)
    info = {"selected_kind": fs.selected_kind, "chosen_c": fs.chosen_c[fs.selected_kind],
            "cv_macro_f1": fs.cv_macro_f1[fs.selected_kind], "train_tickets": fs.n_train,
            "ticket_proba": fs.proba[fs.selected_kind]}
    return feats, info


def run_folds(data: W02Data, text: TextData, seed: int, kinds: tuple[str, ...] = KINDS,
              c_grid: tuple[float, ...] = C_GRID) -> tuple[pd.DataFrame, list[FusionFoldScores]]:
    horizon = pd.Timedelta(hours=float(data.config["labels"]["horizon_hours"]))
    eligible = data.eligible()
    folds = build_folds(data, eligible, horizon)
    features = model_feature_columns()
    out: list[FusionFoldScores] = []
    for fold in folds:
        feats, info = fold_text_features(eligible, fold, text, horizon, seed, kinds, c_grid)
        frame = eligible.copy()
        for col in TEXT_FEATURES:
            frame[col] = np.nan
            frame.iloc[np.concatenate([fold.fit_idx, fold.val_idx, fold.test_idx]),
                       frame.columns.get_loc(col)] = feats[col].to_numpy()
        fs = score_fusion_fold(frame, fold, features, seed)
        fs.info["ticket_proba"] = info.pop("ticket_proba")
        fs.info["text"] = info
        out.append(fs)
    return eligible, out


def arm_metrics(eligible: pd.DataFrame, fold_scores: list[FusionFoldScores], data: W02Data, arm: str,
                budget: float, refractory: float, horizon_h: float, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    scored = pooled_scores(eligible, fold_scores, arm, budget, refractory)
    m = alert_metrics(scored, data.events, refractory, horizon_h)
    if arm == "no_skill":
        m["pr_auc"] = m["base_rate"]
    m["ci95"] = bootstrap_ci(per_unit_table(scored, data.events, refractory, horizon_h), seed=seed)
    m["ci95"]["pr_auc"] = prauc_bootstrap(scored, seed=seed) if arm != "no_skill" else [m["base_rate"]] * 2
    m["fold_pr_auc"] = [pr_auc(g["y"].to_numpy(), g["score"].to_numpy()) for _, g in scored.groupby("fold_id")]
    m.pop("lead_times_h", None)
    return m, scored


def recall_by_fault_type(scored: pd.DataFrame, events: pd.DataFrame, refractory: float, horizon_h: float) -> dict[str, Any]:
    alerts = raise_alerts(scored, None, refractory)
    ev = events_in_scored_period(scored, events)
    h = pd.Timedelta(hours=horizon_h)
    out: dict[str, Any] = {}
    for ftype, group in ev.groupby("fault_type"):
        hit = sum(bool(((alerts["unit_id"] == e.unit_id) & (alerts["t"] >= e.onset_utc - h)
                        & (alerts["t"] < e.onset_utc)).any()) for e in group.itertuples())
        out[str(ftype)] = {"events": len(group), "detected": int(hit)}
    return out


def text_coverage(data_dir: Path, subdir: str, eligible: pd.DataFrame, scored: pd.DataFrame,
                  data: W02Data, text: TextData) -> dict[str, Any]:
    """How much text exists where it matters (uses the hidden truth table; analysis only)."""
    truth = load_corpus_truth(data_dir, subdir).set_index("ticket_id").reindex(text.corpus["ticket_id"])
    notes = text.corpus.assign(source=truth["source"].to_numpy(), fault_id=truth["fault_id"].to_numpy())
    symptom = notes[notes["source"] == "fault_symptom"]
    ev = events_in_scored_period(scored, data.events)
    h72 = pd.Timedelta(hours=72)
    with_note = 0
    for e in ev.itertuples():
        s = symptom[(symptom["fault_id"] == e.fault_id) & (symptom["created_at_utc"] < e.onset_utc)
                    & (symptom["created_at_utc"] >= e.onset_utc - h72)]
        with_note += int(len(s) > 0)
    return {
        "onsets_in_test": len(ev),
        "onsets_with_symptom_note_in_72h_before_onset": with_note,
        "share": with_note / len(ev) if len(ev) else float("nan"),
        "symptom_notes_total": len(symptom),
    }


def drift_false_alarm_share(scored: pd.DataFrame, data_dir: Path, refractory: float) -> dict[str, float]:
    imp = pd.read_csv(data_dir / "ground_truth" / "impairment_events.csv")
    drift_units = set(imp.loc[imp["event_type"] == "sensor_drift", "unit_id"])
    alerts = raise_alerts(scored, None, refractory)
    lab = scored.set_index(["unit_id", "t"])["y"]
    alerts["y"] = lab.reindex(pd.MultiIndex.from_frame(alerts[["unit_id", "t"]])).fillna(0).to_numpy()
    fa = alerts[alerts["y"] == 0]
    return {"false_alarms": float(len(fa)),
            "share_on_drift_units": float(fa["unit_id"].isin(drift_units).mean()) if len(fa) else float("nan"),
            "drift_units": float(len(drift_units))}


def summarise(data: W02Data, data_dir: Path, subdir: str, text: TextData, eligible: pd.DataFrame,
              fold_scores: list[FusionFoldScores], seed: int, full: bool = True) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    cfg = data.config
    horizon_h = float(cfg["labels"]["horizon_hours"])
    refractory = float(cfg["alerting"]["refractory_hours"])
    budget = float(cfg["alerting"]["budget_alerts_per_unit_week"])
    res: dict[str, Any] = {"budget_alerts_per_unit_week": budget, "horizon_hours": horizon_h,
                           "folds": len(fold_scores), "arms": {}, "paired_pr_auc_diff": {}}
    pooled: dict[str, pd.DataFrame] = {}
    for arm in ALL_ARMS:
        res["arms"][arm], pooled[arm] = arm_metrics(eligible, fold_scores, data, arm, budget, refractory,
                                                    horizon_h, seed)
    for a, b in PAIRS:
        pa = pr_auc(pooled[a]["y"].to_numpy(), pooled[a]["score"].to_numpy())
        pb = res["arms"][b]["pr_auc"]
        ci = paired_prauc_diff(pooled[a], pooled[b], seed=seed) if b != "no_skill" else [
            v - pb for v in res["arms"][a]["ci95"]["pr_auc"]]
        res["paired_pr_auc_diff"][f"{a}_minus_{b}"] = {"delta": pa - pb, "ci95": ci}
    res["text_coverage"] = text_coverage(data_dir, subdir, eligible, pooled["telemetry_lgbm"], data, text)
    if not full:
        return res, pooled

    res["recall_by_fault_type"] = {arm: recall_by_fault_type(pooled[arm], data.events, refractory, horizon_h)
                                   for arm in ALL_ARMS if arm != "no_skill"}
    res["drift_false_alarms"] = {arm: drift_false_alarm_share(pooled[arm], data_dir, refractory)
                                 for arm in ALL_ARMS if arm != "no_skill"}
    curve = []
    for level in BUDGETS:
        for arm in ALL_ARMS:
            if arm == "no_skill":
                continue
            sc = pooled_scores(eligible, fold_scores, arm, level, refractory)
            mm = alert_metrics(sc, data.events, refractory, horizon_h)
            curve.append({"arm": arm, "budget": level, **{k: v for k, v in mm.items() if k != "lead_times_h"}})
    res["budget_curve"] = curve
    res["fold_info"] = [{"fold_id": fs.fold.fold_id, "early_variant": fs.info["early_variant"],
                         "late_stack_weights_tel_text": fs.info["late_stack_weights"],
                         "late_stack_iforest_weights_tel_text": fs.info["late_stack_iforest_weights"],
                         "text_gain_share_in_early_fusion": fs.text_gain_share,
                         "val_pr_auc": fs.info["val_pr_auc"], **fs.info["text"]} for fs in fold_scores]
    res["text_gain_share_in_early_fusion_mean"] = float(np.nanmean([fs.text_gain_share for fs in fold_scores]))
    gains = pd.concat([fs.importance / fs.importance.sum() for fs in fold_scores if fs.importance is not None], axis=1)
    top = gains.mean(axis=1).sort_values(ascending=False)
    res["early_fusion_top_features_gain_share"] = {k: float(v) for k, v in top.head(15).items()}
    res["early_fusion_text_feature_gain_share"] = {k: float(top.get(k, 0.0)) for k in TEXT_FEATURES}
    test_rows = pd.concat([eligible.iloc[fs.fold.test_idx][["unit_id", "t"]] for fs in fold_scores])
    res["test_rows"] = len(test_rows)
    return res, pooled


def ablation_figure(res: dict[str, Any]) -> Path:
    plotstyle.apply_style()
    order = ["physics_rule", "telemetry_lgbm", "telemetry_iforest", "text_only", "early_fusion",
             "late_noisy_or", "late_stack", "late_stack_iforest"]
    colour = {"physics_rule": plotstyle.SERIES[0], "telemetry_lgbm": plotstyle.SERIES[0],
              "telemetry_iforest": plotstyle.SERIES[0], "text_only": plotstyle.SERIES[1]}
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    for i, arm in enumerate(reversed(order)):
        m = res["arms"][arm]
        lo, hi = m["ci95"]["pr_auc"]
        c = colour.get(arm, plotstyle.SERIES[2])
        ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(m["pr_auc"], i, "o", color=c, markersize=8, markeredgecolor=plotstyle.SURFACE, markeredgewidth=2)
        ax.text(hi + 0.012, i, f"{m['pr_auc']:.3f}", va="center", fontsize=8.5, color=plotstyle.TEXT)
    base = res["arms"]["no_skill"]["pr_auc"]
    ax.axvline(base, color=plotstyle.NEUTRAL, lw=1, ls="--")
    ax.text(base + 0.005, len(order) - 0.45, f"base rate {base:.3f}", fontsize=8, color=plotstyle.TEXT_SECONDARY)
    ax.set_yticks(range(len(order)), [LABELS[a] for a in reversed(order)])
    ax.set_xlabel("PR-AUC on unseen robots (pooled 20 folds; bar = 95 % unit-bootstrap interval)")
    ax.set_xlim(0, max(res["arms"][a]["ci95"]["pr_auc"][1] for a in order) + 0.1)
    ax.set_ylim(-0.6, len(order) - 0.2)
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=plotstyle.SERIES[k], label=lab)
               for k, lab in ((0, "telemetry only"), (1, "text only"), (2, "combined"))]
    ax.legend(handles=handles, loc="lower right")
    ax.set_title("Modality ablation: early-warning PR-AUC by arm")
    path = figure_path("fusion_ablation_pr_auc.png")
    plotstyle.save(fig, path)
    return path


def budget_figure(res: dict[str, Any]) -> Path:
    plotstyle.apply_style()
    curve = pd.DataFrame(res["budget_curve"])
    arms = ["telemetry_iforest", "text_only", "late_stack_iforest", "early_fusion"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    for k, arm in enumerate(arms):
        g = curve[curve["arm"] == arm].sort_values("budget")
        axes[0].plot(g["budget"], g["event_recall"], marker="o", color=plotstyle.SERIES[k], label=LABELS[arm])
        axes[1].plot(g["budget"], g["precision"], marker="o", color=plotstyle.SERIES[k], label=LABELS[arm])
    axes[0].set_ylabel("Event recall (onsets warned 0-72 h ahead)")
    axes[1].set_ylabel("Alert precision")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
        ax.set_xlabel("Alert budget (alerts per unit-week, set on validation)")
        ax.set_ylim(0, 1)
    axes[0].set_title("Recall and precision against the alert budget")
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    path = figure_path("fusion_budget_curve.png")
    plotstyle.save(fig, path)
    return path


def sensitivity_figure(rows: list[dict[str, Any]]) -> Path:
    plotstyle.apply_style()
    frame = pd.DataFrame(rows).sort_values("notice_rate")
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    series = [("telemetry_iforest", "Isolation Forest (telemetry only)"), ("text_only", "Text only"),
              ("late_stack_iforest", "Late fusion (Isolation Forest + text)"), ("early_fusion", "Early fusion")]
    for k, (arm, label) in enumerate(series):
        ax.plot(frame["notice_rate"], frame[arm], marker="o", color=plotstyle.SERIES[k], label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(frame["notice_rate"], [f"{r:g}x" for r in frame["notice_rate"]])
    ax.set_xlabel("Operator notice rate (corpus assumption; 1x is the documented default)")
    ax.set_ylabel("PR-AUC on unseen robots")
    ax.set_ylim(0, max(0.6, float(frame[[a for a, _ in series]].to_numpy().max()) + 0.05))
    ax.set_title("How much text helps depends on how often operators write")
    ax.legend(loc="upper left", fontsize=7.5)
    path = figure_path("fusion_sensitivity_notice_rate.png")
    plotstyle.save(fig, path)
    return path


def fleet_health_table(pooled: dict[str, pd.DataFrame], arm: str) -> pd.DataFrame:
    """Latest out-of-fold health score per unit (unit-level fleet health score)."""
    latest = pooled[arm].sort_values("t").groupby("unit_id").tail(1)
    table = latest[["unit_id", "t", "score", "thr"]].rename(columns={"score": "risk", "thr": "alert_threshold"})
    table["health_score"] = health_score(table["risk"].to_numpy())
    table["above_threshold"] = table["risk"] >= table["alert_threshold"]
    return table.sort_values("health_score").reset_index(drop=True)


def run_sensitivity(data: W02Data, data_dir: Path, seed: int, kind: str, c_value: float) -> list[dict[str, Any]]:
    rows = []
    for rate in SENSITIVITY_RATES:
        subdir = f"text_nr{str(rate).replace('.', 'p')}"
        corpus, truth, manifest = generate_corpus(data_dir, notice_rate=rate)
        write_corpus(data_dir, corpus, truth, manifest, subdir)
        text = load_text(data_dir, data.groups, subdir)
        eligible, fold_scores = run_folds(data, text, seed, kinds=(kind,), c_grid=(c_value,))
        res, _ = summarise(data, data_dir, subdir, text, eligible, fold_scores, seed, full=False)
        rows.append({
            "notice_rate": rate, "tickets": manifest["tickets"],
            "fault_notes_pre_onset": manifest["fault_notes_pre_onset"],
            "onset_note_share": res["text_coverage"]["share"],
            **{arm: res["arms"][arm]["pr_auc"] for arm in ALL_ARMS},
            **{f"delta[{k}]": v["delta"] for k, v in res["paired_pr_auc_diff"].items()},
            **{f"ci95[{k}]": v["ci95"] for k, v in res["paired_pr_auc_diff"].items()},
            "text_classifier": {"kind": kind, "C": c_value},
        })
    return rows


def run(data_dir: Path, seed: int = 0, sensitivity: bool = False) -> dict[str, Any]:
    data = load_w02(data_dir)
    text = load_text(data_dir, data.groups)
    eligible, fold_scores = run_folds(data, text, seed)
    results, pooled = summarise(data, data_dir, "text", text, eligible, fold_scores, seed)
    results["text_features"] = TEXT_FEATURES
    results["telemetry_feature_count"] = len(model_feature_columns())

    combined = [a for a in ("early_fusion", "late_noisy_or", "late_stack", "late_stack_iforest")]
    best_combined = max(combined, key=lambda a: results["arms"][a]["pr_auc"])
    best_telemetry = max(("telemetry_lgbm", "telemetry_iforest", "physics_rule"),
                         key=lambda a: results["arms"][a]["pr_auc"])
    results["best_combined_arm_on_test"] = best_combined
    results["best_telemetry_arm_on_test"] = best_telemetry
    health = fleet_health_table(pooled, "late_stack")
    results["fleet_health_score"] = {
        "definition": "100 x (1 - fused risk), fused risk = late_stack calibrated probability of onset within 72 h",
        "arm": "late_stack",
        "units": len(health),
        "lowest_five": health.head(5).assign(t=lambda f: f["t"].astype(str)).to_dict(orient="records"),
        "median": float(health["health_score"].median()),
    }

    kinds = pd.Series([f["selected_kind"] for f in results["fold_info"]])
    main_kind = str(kinds.value_counts().idxmax())
    main_c = float(pd.Series([f["chosen_c"] for f in results["fold_info"] if f["selected_kind"] == main_kind]).median())
    results["text_classifier_in_fusion"] = {"selected_kind_counts": {k: int(v) for k, v in kinds.value_counts().items()},
                                            "sensitivity_kind": main_kind, "sensitivity_C": main_c}
    base_row = {"notice_rate": 1.0, "tickets": len(text.corpus),
                "onset_note_share": results["text_coverage"]["share"],
                **{arm: results["arms"][arm]["pr_auc"] for arm in ALL_ARMS},
                **{f"delta[{k}]": v["delta"] for k, v in results["paired_pr_auc_diff"].items()},
                **{f"ci95[{k}]": v["ci95"] for k, v in results["paired_pr_auc_diff"].items()},
                "text_classifier": "per-fold selection (main run)"}
    figs = [ablation_figure(results), budget_figure(results)]
    if sensitivity:
        rows = [base_row, *run_sensitivity(data, data_dir, seed, main_kind, main_c)]
        results["sensitivity_notice_rate"] = sorted(rows, key=lambda r: r["notice_rate"])
        figs.append(sensitivity_figure(results["sensitivity_notice_rate"]))

    pred_dir = data_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([pooled[a].assign(model=a) for a in ALL_ARMS], ignore_index=True).to_parquet(
        pred_dir / "fusion_oof.parquet", index=False)
    health.to_csv(pred_dir / "fleet_health_latest.csv", index=False)
    # per-fold ticket class probabilities (fold-fitted classifier): reused by the explanation run
    np.savez_compressed(pred_dir / "fold_ticket_proba.npz", ticket_id=text.corpus["ticket_id"].to_numpy(dtype=str),
                        **{fs.fold.fold_id: fs.info["ticket_proba"] for fs in fold_scores})

    with tracked_run("w03_fusion", "fusion_ablation", {"track": "fusion", "data_dir": str(data_dir)}) as run_:
        import mlflow

        mlflow.log_params({"folds": len(fold_scores), "budget": results["budget_alerts_per_unit_week"],
                           "horizon_hours": results["horizon_hours"], "n_text_features": len(TEXT_FEATURES),
                           "n_telemetry_features": results["telemetry_feature_count"], "seed": seed,
                           "corpus_notice_rate": 1.0, "sensitivity": sensitivity,
                           "simulator_config_sha256": data.manifest["config_sha256"]})
        for arm, m in results["arms"].items():
            log_flat_metrics({k: v for k, v in m.items() if not isinstance(v, (list, dict))}, prefix=f"{arm}.")
        for name, d in results["paired_pr_auc_diff"].items():
            log_flat_metrics({"delta": d["delta"], "ci_lo": d["ci95"][0], "ci_hi": d["ci95"][1]}, prefix=f"diff.{name}.")
        for fig in figs:
            mlflow.log_artifact(str(fig))
        log_dict_json(results, "fusion_results.json")
        results["mlflow_run_id"] = run_.info.run_id
    write_json(results, "fusion_results.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sensitivity", action="store_true", help="also sweep the operator notice rate")
    args = parser.parse_args()
    res = run(args.data_dir, args.seed, args.sensitivity)
    for arm in ALL_ARMS:
        v = res["arms"][arm]
        print(f"{arm:20s} PR-AUC {v['pr_auc']:.3f} {np.round(v['ci95']['pr_auc'], 3)}  recall {v['event_recall']:.2f}  "
              f"precision {v['precision']:.2f}  alerts/unit-wk {v['alerts_per_unit_week']:.2f}  "
              f"lead {v['median_lead_time_h']:.0f} h")
    for k, v in res["paired_pr_auc_diff"].items():
        print(f"{k:45s} {v['delta']:+.3f} {np.round(v['ci95'], 3)}")
    print("MLflow run:", res["mlflow_run_id"])


if __name__ == "__main__":
    main()
