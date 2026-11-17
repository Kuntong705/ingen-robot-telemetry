"""Global and per-alert SHAP, grounded alert summaries and the faithfulness rate.

    python -m ingen_fleet_health.experiments.explain_run --data-dir data/w02 [--llm local]

Run after ``fusion_run`` (it reuses that run's per-fold ticket probabilities
and checks that the refitted models reproduce its out-of-fold scores).

The explained model is the early-fusion LightGBM: the one model whose inputs
include both telemetry and text features, so an attribution can point at a
linked ticket. Every explanation is for an *unseen* robot: each fold's model
explains only its own test rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ingen_fleet_health import plotstyle
from ingen_fleet_health.evaluation.metrics import raise_alerts, threshold_for_budget
from ingen_fleet_health.experiments.common import load_w02
from ingen_fleet_health.experiments.early_warning_run import build_folds
from ingen_fleet_health.experiments.w03_common import (
    figure_path,
    load_text,
    read_json,
    write_json,
)
from ingen_fleet_health.explain.alert_summaries import (
    LOOKBACK_HOURS,
    Attribution,
    Evidence,
    LinkedTicket,
    TemplateSummariser,
    build_evidence,
    build_prompt,
    concept_label,
)
from ingen_fleet_health.explain.faithfulness import (
    CHECKER_VERSION,
    CORRUPTIONS,
    check_summary,
    corrupt,
)
from ingen_fleet_health.explain.shap_utils import (
    concept_importance,
    feature_info,
    global_importance,
    tree_shap,
)
from ingen_fleet_health.models.early_warning import (
    model_feature_columns,
    positive_proba,
)
from ingen_fleet_health.models.fusion import early_variants, fit_selected
from ingen_fleet_health.text.features import TEXT_FEATURES, asof_text_features
from ingen_fleet_health.tracking import log_dict_json, log_flat_metrics, tracked_run

N_ALERTS = 20
MAX_TRUE_POSITIVES = 10


def explain_folds(data_dir: Path, seed: int) -> dict[str, Any]:
    data = load_w02(data_dir)
    text = load_text(data_dir, data.groups)
    cfg = data.config
    horizon = pd.Timedelta(hours=float(cfg["labels"]["horizon_hours"]))
    refractory = float(cfg["alerting"]["refractory_hours"])
    budget = float(cfg["alerting"]["budget_alerts_per_unit_week"])
    eligible = data.eligible()
    folds = build_folds(data, eligible, horizon)
    telemetry = model_feature_columns()
    cache = np.load(data_dir / "predictions" / "fold_ticket_proba.npz", allow_pickle=False)
    if list(cache["ticket_id"]) != text.corpus["ticket_id"].astype(str).tolist():
        raise RuntimeError("fold_ticket_proba.npz does not match the current corpus; rerun fusion_run")
    all_cols = telemetry + TEXT_FEATURES

    scored_parts, shap_parts, value_parts = [], [], []
    models: dict[str, Any] = {}
    additivity = 0.0
    for fold in folds:
        proba = cache[fold.fold_id]
        rows = np.concatenate([fold.fit_idx, fold.val_idx, fold.test_idx])
        feats = asof_text_features(text.corpus, proba, eligible.iloc[rows][["unit_id", "t"]])
        frame = eligible.copy()
        for col in TEXT_FEATURES:
            frame[col] = np.nan
            frame.iloc[rows, frame.columns.get_loc(col)] = feats[col].to_numpy()
        fit, val, test = frame.iloc[fold.fit_idx], frame.iloc[fold.val_idx], frame.iloc[fold.test_idx]
        sel = fit_selected(fit, val, early_variants(frame, telemetry, TEXT_FEATURES), seed)
        risk = sel.cal.predict_proba(test[sel.cols])[:, 1]
        val_frame = val[["unit_id", "t"]].assign(score=sel.cal.predict_proba(val[sel.cols])[:, 1])
        thr = threshold_for_budget(val_frame, budget, refractory)
        values, base = tree_shap(sel.cal.model, test[sel.cols])
        raw = np.asarray(sel.cal.model.predict(test[sel.cols], raw_score=True))
        additivity = max(additivity, float(np.max(np.abs(values.sum(axis=1) + base - raw))))
        full = pd.DataFrame(0.0, index=test.index, columns=all_cols)
        full[sel.cols] = values
        scored_parts.append(test[["unit_id", "t", "y", "fault_id", "fault_type"]].assign(
            score=risk, thr=thr, fold_id=fold.fold_id, base=base, raw_uncalibrated=positive_proba(sel.cal.model, test[sel.cols])))
        shap_parts.append(full)
        value_parts.append(test[all_cols])
        models[fold.fold_id] = {"variant": sel.name, "cols": sel.cols, "proba": proba, "calibration": sel.cal.method}
    return {
        "data": data, "text": text, "scored": pd.concat(scored_parts), "shap": pd.concat(shap_parts),
        "values": pd.concat(value_parts), "models": models, "additivity_max_abs_error": additivity,
        "refractory": refractory, "budget": budget, "all_cols": all_cols,
    }


def check_all(per_alert: list[dict[str, Any]], packs: list[Evidence]) -> dict[str, Any]:
    """Run the faithfulness checker on every stored summary; writes each check into ``per_alert``."""
    by_gen: dict[str, list[Any]] = {}
    for record, pack in zip(per_alert, packs, strict=True):
        for name, entry in record["summaries"].items():
            check = check_summary(entry["text"], pack, name)
            entry["check"] = check.to_dict()
            by_gen.setdefault(name, []).append(check)
    rates: dict[str, Any] = {}
    for name, checks in by_gen.items():
        claims = [c for ch in checks for c in ch.claims]
        kinds = pd.Series([c.kind for c in claims if not c.supported], dtype=object)
        rates[name] = {
            "alerts": len(checks),
            "faithful": int(sum(ch.faithful for ch in checks)),
            "faithfulness_rate": float(np.mean([ch.faithful for ch in checks])),
            "claims_checked": len(claims),
            "claims_supported": int(sum(c.supported for c in claims)),
            "unsupported_by_kind": {str(k): int(v) for k, v in kinds.value_counts().items()},
        }
    return rates


def run_controls(packs: list[Evidence], seed: int) -> dict[str, Any]:
    """Negative controls: the checker must flag deliberately unfaithful summaries."""
    rng = np.random.default_rng(seed)
    template = TemplateSummariser()
    controls: dict[str, dict[str, int]] = {k: {"built": 0, "flagged": 0} for k in CORRUPTIONS}
    examples: dict[str, Any] = {}
    for pack in packs:
        base = template.summarise(pack)
        for kind in CORRUPTIONS:
            bad = corrupt(base, pack, kind, rng)
            if bad is None:
                continue
            result = check_summary(bad, pack, f"corrupted[{kind}]")
            controls[kind]["built"] += 1
            controls[kind]["flagged"] += int(not result.faithful)
            examples.setdefault(kind, {"alert_id": pack.alert_id, "text": bad, "check": result.to_dict()})
    built = sum(v["built"] for v in controls.values())
    flagged = sum(v["flagged"] for v in controls.values())
    return {"by_kind": controls, "built": built, "flagged": flagged,
            "detection_rate": flagged / built if built else float("nan"), "examples": examples}


def evidence_from_dict(d: dict[str, Any]) -> Evidence:
    return Evidence(**{**d, "positive": [Attribution(**a) for a in d["positive"]],
                       "negative": [Attribution(**a) for a in d["negative"]],
                       "tickets": [LinkedTicket(**t) for t in d["tickets"]]})


def recheck(seed: int = 0) -> dict[str, Any]:
    """Re-score the stored summaries and controls with the current checker (no refit, no regeneration).

    The previous checker version's numbers are kept under ``checker_history``."""
    res = read_json("alert_explanations.json")
    packs = [evidence_from_dict(a["evidence"]) for a in res["alerts"]]
    history = res.get("checker_history", [])
    history.append({
        "checker_version": res.get("checker_version", 1),
        "generators": res["generators"],
        "negative_controls": {k: v for k, v in res["negative_controls"].items() if k != "examples"},
        "faithful_per_alert": {a["alert_id"]: {n: e["check"]["faithful"] for n, e in a["summaries"].items()}
                               for a in res["alerts"]},
    })
    res["checker_history"] = history
    res["generators"] = check_all(res["alerts"], packs)
    res["negative_controls"] = run_controls(packs, seed)
    res["checker_version"] = CHECKER_VERSION
    with tracked_run("w03_explanations", "faithfulness_recheck",
                     {"track": "explain", "recheck_of": res["mlflow_run_id"]}) as run_:
        for name, r in res["generators"].items():
            key = "template" if name == "template" else "local_llm"
            log_flat_metrics({"faithfulness_rate": r["faithfulness_rate"], "claims_checked": r["claims_checked"],
                              "claims_supported": r["claims_supported"]}, prefix=f"{key}.")
        log_flat_metrics({"negative_control_detection_rate": res["negative_controls"]["detection_rate"],
                          "checker_version": float(CHECKER_VERSION)})
        log_dict_json(res, "alert_explanations.json")
        res["recheck_mlflow_run_id"] = run_.info.run_id
    write_json(res, "alert_explanations.json")
    return res


def sample_alerts(scored: pd.DataFrame, refractory: float, seed: int) -> pd.DataFrame:
    """20 alerts, seeded: up to 10 true positives, the rest false alarms."""
    alerts = raise_alerts(scored, None, refractory)
    keyed = scored.reset_index().set_index(["unit_id", "t"])
    alerts = alerts.join(keyed[["index", "y", "score", "thr", "fold_id"]], on=["unit_id", "t"])
    rng = np.random.default_rng(seed)
    tp, fp = alerts[alerts["y"] == 1], alerts[alerts["y"] == 0]
    n_tp = min(len(tp), MAX_TRUE_POSITIVES)
    n_fp = min(len(fp), N_ALERTS - n_tp)
    pick = pd.concat([tp.iloc[rng.choice(len(tp), n_tp, replace=False)] if n_tp else tp,
                      fp.iloc[rng.choice(len(fp), n_fp, replace=False)] if n_fp else fp])
    pick = pick.sort_values(["t", "unit_id"]).reset_index(drop=True)
    pick.attrs["alerts_total"] = len(alerts)
    pick.attrs["true_positive_alerts_total"] = len(tp)
    return pick


def build_packs(state: dict[str, Any], alerts: pd.DataFrame) -> list[Evidence]:
    data, text = state["data"], state["text"]
    features = data.features.set_index(["unit_id", "t"])
    packs = []
    for i, a in enumerate(alerts.itertuples(index=False)):
        info = state["models"][a.fold_id]
        cols = info["cols"]
        row = state["values"].loc[a.index].copy()
        row["unit_id"], row["t"] = a.unit_id, a.t
        before_t = a.t - pd.Timedelta(hours=LOOKBACK_HOURS)
        before: pd.Series | None = None
        if (a.unit_id, before_t) in features.index:
            tel = features.loc[(a.unit_id, before_t)]
            txt = asof_text_features(text.corpus, info["proba"],
                                     pd.DataFrame({"unit_id": [a.unit_id], "t": [before_t]})).iloc[0]
            before = pd.concat([tel, txt])
        unit_mask = (text.corpus["unit_id"] == a.unit_id).to_numpy()
        shap_row = state["shap"].loc[a.index, cols].to_numpy(dtype=float)
        base = float(state["scored"].loc[a.index, "base"])
        packs.append(build_evidence(f"A{i + 1:02d}", row, before, shap_row, cols, base, float(a.score), float(a.thr),
                                    a.fold_id, text.corpus[unit_mask].reset_index(drop=True), info["proba"][unit_mask]))
    return packs


def global_figure(importance: pd.DataFrame, concepts: pd.Series) -> Path:
    plotstyle.apply_style()
    top = importance.head(18).iloc[::-1]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 5.4), gridspec_kw={"width_ratios": [1.5, 1]})
    is_text = top["feature"].str.startswith("txt_")
    ax.barh(range(len(top)), top["mean_abs_shap"], height=0.62,
            color=[plotstyle.SERIES[1] if t else plotstyle.SERIES[0] for t in is_text])
    ax.set_yticks(range(len(top)), [feature_info(f).phrase[:62] for f in top["feature"]], fontsize=7.8)
    ax.set_xlabel("Mean |SHAP| (log-odds), pooled test rows of all folds")
    ax.set_title("Global SHAP: what the early-fusion model relies on")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=plotstyle.SERIES[0], label="telemetry feature"),
                       plt.Rectangle((0, 0), 1, 1, color=plotstyle.SERIES[1], label="text feature")], loc="lower right")
    share = concepts.iloc[::-1]
    bx.barh(range(len(share)), share.to_numpy(), height=0.62, color=plotstyle.SERIES[0])
    for i, v in enumerate(share.to_numpy()):
        bx.text(v + 0.005, i, f"{100 * v:.0f} %", va="center", fontsize=8)
    bx.set_yticks(range(len(share)), [concept_label(c) for c in share.index], fontsize=8)
    bx.set_xlabel("Share of total mean |SHAP|")
    bx.set_title("By concept")
    bx.grid(axis="y", visible=False)
    bx.set_xlim(0, float(share.max()) + 0.08)
    fig.tight_layout()
    path = figure_path("shap_global.png")
    plotstyle.save(fig, path)
    return path


def alert_figure(packs: list[Evidence], outcome: dict[str, int]) -> Path:
    """Per-alert SHAP for two sampled alerts: one true positive, one false alarm."""
    plotstyle.apply_style()
    chosen = [next((p for p in packs if outcome[p.alert_id] == 1), packs[0]),
              next((p for p in packs if outcome[p.alert_id] == 0), packs[-1])]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.9))
    for ax, ev in zip(axes, chosen, strict=True):
        items = ev.positive + ev.negative[::-1]
        vals = [a.shap for a in items][::-1]
        ax.barh(range(len(items)), vals, height=0.6,
                color=[plotstyle.SERIES[7] if v > 0 else plotstyle.SERIES[0] for v in vals])
        ax.set_yticks(range(len(items)), [a.phrase[:52] for a in items][::-1], fontsize=7.5)
        ax.axvline(0, color=plotstyle.NEUTRAL, lw=1)
        kind = "true positive" if outcome[ev.alert_id] == 1 else "false alarm"
        ax.set_title(f"{ev.alert_id} · {ev.unit_id} · {ev.t} ({kind})", fontsize=9.5)
        ax.set_xlabel("SHAP (log-odds): right raises the risk, left lowers it")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    path = figure_path("shap_alert_examples.png")
    plotstyle.save(fig, path)
    return path


def run(data_dir: Path, seed: int = 0, llm: str | None = None) -> dict[str, Any]:
    state = explain_folds(data_dir, seed)
    scored = state["scored"]
    importance = global_importance(state["shap"].to_numpy(), state["all_cols"])
    concepts = concept_importance(importance)

    fusion = pd.read_parquet(data_dir / "predictions" / "fusion_oof.parquet")
    fusion = fusion[fusion["model"] == "early_fusion"].set_index(["unit_id", "t"])["score"]
    joined = scored.set_index(["unit_id", "t"])["score"].to_frame("explained").join(fusion.rename("fusion_run"))
    reproduction = float((joined["explained"] - joined["fusion_run"]).abs().max())

    alerts = sample_alerts(scored, state["refractory"], seed)
    packs = build_packs(state, alerts)
    outcome = {p.alert_id: int(y) for p, y in zip(packs, alerts["y"], strict=True)}

    generators: list[Any] = [TemplateSummariser()]
    llm_error = None
    if llm == "local":
        try:
            from ingen_fleet_health.explain.alert_summaries import LocalLLMSummariser

            generators.append(LocalLLMSummariser())
        except (ImportError, OSError, RuntimeError, ValueError) as exc:  # report it instead of failing the run
            llm_error = f"{type(exc).__name__}: {exc}"

    per_alert: list[dict[str, Any]] = [{"alert_id": p.alert_id, "outcome": "true_positive" if outcome[p.alert_id] else "false_alarm",
                                        "evidence": p.to_dict(), "prompt": build_prompt(p)[1], "summaries": {}} for p in packs]
    for gen in generators:
        for record, pack in zip(per_alert, packs, strict=True):
            summary = gen.summarise(pack)
            record["summaries"][gen.name] = {"text": summary, "sentences": len([s for s in summary.split(". ") if s.strip()])}
    rates = check_all(per_alert, packs)
    negative_controls = run_controls(packs, seed)

    figs = [global_figure(importance, concepts), alert_figure(packs, outcome)]
    results: dict[str, Any] = {
        "explained_model": "early_fusion (LightGBM on telemetry + text features, variant chosen on validation)",
        "shap": {"method": "TreeExplainer, exact tree SHAP, raw margin (log-odds) of the uncalibrated model",
                 "additivity_max_abs_error": state["additivity_max_abs_error"],
                 "test_rows_explained": len(scored)},
        "reproduces_fusion_run_scores_max_abs_diff": reproduction,
        "global_importance_top": importance.head(25).to_dict(orient="records"),
        "concept_share_of_mean_abs_shap": {k: float(v) for k, v in concepts.items()},
        "text_feature_share_of_mean_abs_shap": float(
            importance.loc[importance["feature"].str.startswith("txt_"), "mean_abs_shap"].sum()
            / importance["mean_abs_shap"].sum()),
        "alert_sampling": {"budget_alerts_per_unit_week": state["budget"], "alerts_raised_on_test": alerts.attrs["alerts_total"],
                           "true_positive_alerts": alerts.attrs["true_positive_alerts_total"], "sampled": len(alerts),
                           "sampled_true_positive": int(alerts["y"].sum()), "seed": seed,
                           "rule": "up to 10 true-positive alerts, the rest false alarms, drawn with a seeded generator"},
        "fold_variants": {k: v["variant"] for k, v in state["models"].items()},
        "generators": rates,
        "llm": {"requested": llm, "error": llm_error,
                "hosted_llm_api_called": False,
                "note": "No hosted LLM API was called. Local generation uses open weights on this machine only."},
        "checker_version": CHECKER_VERSION,
        "negative_controls": negative_controls,
        "alerts": per_alert,
    }
    with tracked_run("w03_explanations", "shap_and_faithfulness", {"track": "explain", "data_dir": str(data_dir)}) as run_:
        import mlflow

        mlflow.log_params({"n_alerts": len(alerts), "seed": seed, "llm": str(llm), "budget": state["budget"]})
        for name, r in rates.items():
            key = "template" if name == "template" else "local_llm"
            log_flat_metrics({"faithfulness_rate": r["faithfulness_rate"], "claims_checked": r["claims_checked"],
                              "claims_supported": r["claims_supported"]}, prefix=f"{key}.")
        log_flat_metrics({"negative_control_detection_rate": results["negative_controls"]["detection_rate"],
                          "shap_additivity_max_abs_error": state["additivity_max_abs_error"],
                          "text_feature_shap_share": results["text_feature_share_of_mean_abs_shap"]})
        for fig in figs:
            mlflow.log_artifact(str(fig))
        log_dict_json(results, "alert_explanations.json")
        results["mlflow_run_id"] = run_.info.run_id
    write_json(results, "alert_explanations.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llm", choices=["local"], help="also write summaries with the local open-weights model")
    parser.add_argument("--recheck", action="store_true",
                        help="re-score the stored summaries with the current checker (no refit, no generation)")
    args = parser.parse_args()
    res = recheck(args.seed) if args.recheck else run(args.data_dir, args.seed, args.llm)
    for name, r in res["generators"].items():
        print(f"{name}: faithful {r['faithful']}/{r['alerts']} = {r['faithfulness_rate']:.2f}; "
              f"claims {r['claims_supported']}/{r['claims_checked']}; unsupported {r['unsupported_by_kind']}")
    print("negative controls:", res["negative_controls"]["by_kind"], res["negative_controls"]["detection_rate"])
    print("SHAP additivity error:", res["shap"]["additivity_max_abs_error"],
          "| score reproduction diff:", res["reproduces_fusion_run_scores_max_abs_diff"])
    print("MLflow run:", res.get("recheck_mlflow_run_id", res["mlflow_run_id"]))


if __name__ == "__main__":
    main()
