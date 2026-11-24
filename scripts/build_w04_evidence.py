"""Model-card evidence from the Week 4 pipeline run (operational period, production v1).

Reads only pipeline outputs (predictions, alert queue, drift, weekly checks),
the published fault confirmations and maintenance records, and the registered
bundles. Writes ``reports/w04/operational_results.json``.

Protocol (ADR-0005): one time cut (training rows end 72 h before day 30), one
unit split (30 % of each platform's robots never seen in training). The
operational period is every scored date; predictions whose labels are not yet
settled at the end of the data (last 7 days) are excluded. 95 % CIs resample
whole robots (unit bootstrap). This is a different, smaller protocol than the
Week 2 walk-forward folds, so its numbers are reported next to them, not instead.

    python scripts/build_w04_evidence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingen_fleet_health.evaluation.metrics import (
    HOURS_PER_WEEK,
    evaluate_alerts,
    mae,
    pr_auc,
    unit_cluster_bootstrap,
)
from ingen_fleet_health.labels import early_warning_labels
from ingen_fleet_health.monitoring.performance import (
    confirmed_faults,
    label_events,
    repairs_known,
)
from ingen_fleet_health.pipeline import views
from ingen_fleet_health.pipeline.scoring import models
from ingen_fleet_health.pipeline.settings import load_settings

OUT = ROOT / "reports" / "w04" / "operational_results.json"
N_BOOT = 500


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else round(float(obj), 6)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def paired_ci(frame: pd.DataFrame, a: str, b: str) -> tuple[float, float, float]:
    def stat(s: pd.DataFrame) -> float:
        return pr_auc(s["y"].to_numpy(), s[a].to_numpy()) - pr_auc(s["y"].to_numpy(), s[b].to_numpy())
    lo, hi = unit_cluster_bootstrap(frame, stat, n_boot=N_BOOT, seed=1)
    return stat(frame), lo, hi


def main() -> None:
    settings = load_settings()
    ew, _soh, versions = models(settings)
    result: dict[str, Any] = {"protocol": __doc__.split("\n\n")[1].replace("\n", " "), "models": versions,
                              "champion_variant": ew.variant, "thresholds": ew.thresholds,
                              "shadow_thresholds": ew.shadow_thresholds, "platforms": {}, "cost_counts": {}}
    refractory = float(settings.alerting["refractory_hours"])
    for platform, ps in settings.platforms.items():
        dates = settings.partition_dates("predictions", platform)
        if not dates:
            continue
        last = dates[-1]
        preds = views.predictions(settings, [platform], dates[0], last)
        alerts = views.alerts(settings, [platform], dates[0], last)
        as_of = pd.Timestamp(last).tz_localize("UTC") + pd.Timedelta(days=1)
        horizon = pd.Timedelta(hours=float(ps.config["labels"]["horizon_hours"]))
        settled_until = as_of - pd.Timedelta(days=int(settings.monitoring["label_maturity_days"]))
        events = label_events(confirmed_faults(settings, platform, last), repairs_known(settings, platform, last), as_of)
        lab = early_warning_labels(preds[["unit_id", "t"]], events, as_of, horizon / pd.Timedelta(hours=1),
                                   float(ps.config["labels"]["post_repair_exclusion_hours"]))
        scored = preds.merge(lab[["unit_id", "t", "y", "eligible", "fault_id"]], on=["unit_id", "t"])
        scored = scored[scored["eligible"] & ~scored["insufficient_data"] & (scored["t"] < settled_until)]
        scored = scored.rename(columns={"iforest_score": "iforest", "physics_rule_score": "physics_rule"})
        start = scored["t"].min()
        ev_window = events[(events["onset_utc"] > start) & (events["onset_utc"] <= settled_until + horizon)]
        train_units = set(ew.training_units[platform])
        unit_ref = pd.read_csv(settings.data_root / "reference" / f"units_{platform}.csv")
        site = unit_ref.set_index("unit_id")["site"]
        entry: dict[str, Any] = {"scored_dates": [dates[0].isoformat(), last.isoformat()],
                                 "evaluated_until_utc": settled_until.isoformat(), "cohorts": {}}
        for cohort, is_train in (("held_out_units", False), ("training_units", True)):
            sc = scored[scored["unit_id"].isin(train_units) == is_train].copy()
            al = alerts[alerts["unit_id"].isin(train_units) == is_train] if len(alerts) else alerts
            al = al[al["t"] < settled_until] if len(al) else al
            ev = ev_window[ev_window["unit_id"].isin(set(sc["unit_id"]))]
            c: dict[str, Any] = {"units": int(sc["unit_id"].nunique()), "rows": len(sc),
                                 "positive_rows": int(sc["y"].sum()), "base_rate": float(sc["y"].mean()),
                                 "fault_onsets": len(ev)}
            for m in ("risk", "iforest", "physics_rule"):
                name = "champion" if m == "risk" else m
                c[f"pr_auc_{name}"] = pr_auc(sc["y"].to_numpy(), sc[m].to_numpy())
                c[f"pr_auc_{name}_ci95"] = unit_cluster_bootstrap(
                    sc, lambda s, m=m: pr_auc(s["y"].to_numpy(), s[m].to_numpy()), n_boot=N_BOOT, seed=0)
            for other in ("iforest", "physics_rule"):
                d, lo, hi = paired_ci(sc, "risk", other)
                c[f"champion_minus_{other}"] = {"delta": d, "ci95": [lo, hi]}
            # champion alerts = the real alert queue; shadow alerts at their own budget thresholds
            hit = sc.set_index(["unit_id", "t"])["y"]
            q = al[["unit_id", "t"]] if len(al) else pd.DataFrame(columns=["unit_id", "t"])
            q_hit = hit.reindex(pd.MultiIndex.from_frame(q)).fillna(0).to_numpy() if len(q) else np.array([])
            detected, leads, missed = 0, [], []
            for e in ev.itertuples(index=False):
                a = q[(q["unit_id"] == e.unit_id) & (q["t"] >= e.onset_utc - horizon) & (q["t"] < e.onset_utc)]
                if len(a):
                    detected += 1
                    leads.append((e.onset_utc - a["t"].min()) / pd.Timedelta(hours=1))
                else:
                    missed.append({"fault_id": e.fault_id, "unit_id": e.unit_id, "fault_type": e.fault_type,
                                   "onset_utc": e.onset_utc.isoformat()})
            uw = len(sc) / HOURS_PER_WEEK
            c["alert_queue"] = {"alerts": len(q), "true_alerts": int(q_hit.sum()),
                                "precision": float(q_hit.mean()) if len(q_hit) else float("nan"),
                                "alerts_per_unit_week": len(q) / uw, "false_alarms_per_unit_week": (len(q) - q_hit.sum()) / uw,
                                "events": len(ev), "events_detected": detected,
                                "event_recall": detected / len(ev) if len(ev) else float("nan"),
                                "median_lead_time_h": float(np.median(leads)) if leads else float("nan"),
                                "unit_weeks": uw}
            c["missed_fault_onsets"] = missed
            c["recall_by_fault_type"] = {}
            for ft, g in ev.groupby("fault_type"):
                caught = sum(1 for e in g.itertuples(index=False)
                             if len(q[(q["unit_id"] == e.unit_id) & (q["t"] >= e.onset_utc - horizon)
                                      & (q["t"] < e.onset_utc)]))
                c["recall_by_fault_type"][ft] = {"events": len(g), "detected": caught}
            c["pr_auc_by_site"] = {s: {"rows": len(g), "positives": int(g["y"].sum()),
                                       "pr_auc_champion": pr_auc(g["y"].to_numpy(), g["risk"].to_numpy())}
                                   for s, g in sc.assign(site=sc["unit_id"].map(site)).groupby("site")}
            shadow = {}
            for m in ("iforest", "physics_rule"):
                thr = ew.shadow_thresholds[m][platform]
                res = evaluate_alerts(sc[["unit_id", "t", "y"]].assign(score=sc[m]), ev, thr, refractory,
                                      horizon / pd.Timedelta(hours=1))
                res.pop("lead_times_h", None)
                shadow[m] = res
                if cohort == "held_out_units":
                    result["cost_counts"][f"{platform}:{m}_shadow"] = {
                        "alerts": res["alerts"], "true_alerts": round(res["alerts"] * (res["precision"] or 0)),
                        "events": res["events"], "events_detected": res["events_detected"],
                        "unit_weeks": res["unit_weeks"]}
            c["shadow_alerts_at_budget"] = shadow
            if cohort == "held_out_units":
                result["cost_counts"][f"{platform}:champion"] = {
                    "alerts": len(q), "true_alerts": int(q_hit.sum()), "events": len(ev),
                    "events_detected": detected, "unit_weeks": uw}
            entry["cohorts"][cohort] = c
        # SoH: forecasts whose target day is in the data
        f = preds.dropna(subset=["soh_forecast_pct"]).copy()
        truth = pd.read_csv(ps.source_dir / "ground_truth" / "battery_truth_daily.csv")
        truth["target"] = pd.to_datetime(truth["date_utc"]).dt.tz_localize("UTC")
        f["target"] = pd.to_datetime(f["soh_target_date"], utc=True)
        m = f.merge(truth[["unit_id", "target", "soh_true_pct"]], on=["unit_id", "target"])
        soh_out = {}
        for cohort, is_train in (("held_out_units", False), ("training_units", True)):
            g = m[m["unit_id"].isin(train_units) == is_train]
            if g.empty:
                continue
            soh_out[cohort] = {
                "forecasts": len(g), "mae_pp_model": mae(g["soh_true_pct"].to_numpy(), g["soh_forecast_pct"].to_numpy()),
                "mae_pp_persistence": mae(g["soh_true_pct"].to_numpy(), g["soh_now_pct"].to_numpy()),
                "skill_ci95": unit_cluster_bootstrap(
                    g, lambda s: mae(s["soh_true_pct"].to_numpy(), s["soh_now_pct"].to_numpy())
                    - mae(s["soh_true_pct"].to_numpy(), s["soh_forecast_pct"].to_numpy()), n_boot=N_BOOT, seed=2)}
        entry["soh_7d"] = soh_out
        # explanations, drift, data quality
        if len(alerts):
            top = [json.loads(e)["positive"][0]["concept"] if json.loads(e)["positive"] else "none"
                   for e in alerts["evidence"]]
            entry["alert_summaries"] = {"alerts": len(alerts),
                                        "all_claims_supported_auto": int(alerts["summary_faithful_auto"].sum()),
                                        "claims_checked_total": int(alerts["claims_checked"].sum()),
                                        "top_driver_concept": pd.Series(top).value_counts().to_dict()}
        d = views.drift(settings, [platform], dates[0], last)
        if len(d):
            feat = d[d["feature"] != "risk"]
            entry["drift"] = {"feature_days": len(feat), "bands": feat["band"].value_counts().to_dict(),
                              "median_psi_by_feature": feat.groupby("feature")["psi"].median().sort_values(
                                  ascending=False).to_dict(),
                              "risk_psi_median": float(d.loc[d["feature"] == "risk", "psi"].median())}
        qd = views.quality(settings, [platform], dates[0], last)
        entry["data_quality"] = {"days": len(qd), "rejected_rows": int(qd["rejected_rows"].sum()),
                                 "accepted_rows": int(qd["accepted_rows"].sum()),
                                 "unit_days_missing": int(qd["units_missing"].sum())}
        entry["insufficient_data_rows"] = int(preds["insufficient_data"].sum())
        entry["prediction_rows"] = len(preds)
        result["platforms"][platform] = entry
    result["weekly_checks"] = views.performance_reports(settings, pd.Timestamp("2000-01-01").date(),
                                                         pd.Timestamp("2100-01-01").date())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_clean(result), indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
