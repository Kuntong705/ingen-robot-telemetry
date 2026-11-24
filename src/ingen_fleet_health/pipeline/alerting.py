"""Daily alert queue: threshold crossings -> alerts with grounded explanations.

* An alert is raised at the first hour a unit's risk reaches its platform
  threshold; the unit is then muted for ``refractory_hours`` (72 h), counting
  alerts of earlier days (read from earlier alert partitions, as-of).
* Rows flagged ``insufficient_data`` never alert: they are a connectivity
  problem, shown as such on the dashboard.
* Every alert carries an evidence pack (exact tree SHAP of the champion model,
  values now and 72 h earlier) and the deterministic Week 3 template summary
  (evaluated 20/20 faithful in Week 3). The checker v2 result is stored with it,
  so the faithfulness of production summaries is measured, not assumed.
* The local LLM is not used in production: it was faithful in 0 of 20 alerts
  in the Week 3 AI-assisted evidence review.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ingen_fleet_health.explain.alert_summaries import (
    TemplateSummariser,
    build_evidence,
)
from ingen_fleet_health.explain.faithfulness import check_summary
from ingen_fleet_health.explain.shap_utils import tree_shap
from ingen_fleet_health.pipeline.scoring import models, read_table
from ingen_fleet_health.pipeline.settings import (
    PartitionNotReady,
    Settings,
    atomic_partition,
    fingerprint,
    write_manifest,
)

ALERT_COLUMNS = [
    "alert_id", "platform", "unit_id", "t", "date", "risk", "threshold", "soh_now_pct", "drivers",
    "summary", "summary_faithful_auto", "claims_checked", "unsupported_claims", "evidence", "model", "status",
]


def previous_alert_times(settings: Settings, platform: str, day: date, refractory: pd.Timedelta) -> dict[str, pd.Timestamp]:
    days = [day - timedelta(days=k) for k in range(int(np.ceil(refractory / pd.Timedelta(days=1))), 0, -1)]
    prev = read_table(settings, "alerts", platform, days, "alerts.parquet")
    if prev.empty:
        return {}
    return prev.groupby("unit_id")["t"].max().to_dict()


def select_alerts(preds: pd.DataFrame, last_alert: dict[str, pd.Timestamp], refractory: pd.Timedelta) -> pd.DataFrame:
    hot = preds[preds["above_threshold"]].sort_values(["unit_id", "t"])
    keep = []
    for unit, frame in hot.groupby("unit_id", sort=True):
        muted_until = last_alert[unit] + refractory if unit in last_alert else None
        for idx, t in zip(frame.index, frame["t"], strict=True):
            if muted_until is None or t >= muted_until:
                keep.append(idx)
                muted_until = t + refractory
    return hot.loc[keep]


def alerts_for_day(settings: Settings, platform: str, day: date) -> dict[str, Any]:
    pred_path = settings.table_dir("predictions", platform, day) / "predictions.parquet"
    feat_path = settings.table_dir("features", platform, day) / "features.parquet"
    if not pred_path.exists():
        raise PartitionNotReady(f"{platform} {day}: no predictions; run score first")
    preds = pd.read_parquet(pred_path)
    refractory = pd.Timedelta(hours=float(settings.alerting["refractory_hours"]))
    last = previous_alert_times(settings, platform, day, refractory)
    chosen = select_alerts(preds, last, refractory)
    ew, _, versions = models(settings)
    rows: list[dict[str, Any]] = []
    if len(chosen):
        feats = pd.read_parquet(feat_path).set_index(["unit_id", "t"])
        before_days = sorted({(t - pd.Timedelta(hours=72)).date() for t in chosen["t"]})
        earlier = read_table(settings, "features", platform, before_days, "features.parquet")
        earlier = earlier.set_index(["unit_id", "t"]) if len(earlier) else earlier
        keys = list(zip(chosen["unit_id"], chosen["t"], strict=True))
        frame = feats.loc[keys].reset_index()
        X = ew.features(frame, platform)
        shap_values, base = tree_shap(ew.model.model, X)
        soh_latest = preds.dropna(subset=["soh_now_pct"]).groupby("unit_id")["soh_now_pct"].last()
        summariser = TemplateSummariser(notes_ingested=False)
        for i, (unit, t) in enumerate(keys):
            row = frame.iloc[i]
            k72 = (unit, t - pd.Timedelta(hours=72))
            row72 = earlier.loc[k72] if len(earlier) and k72 in earlier.index else None
            if row72 is not None:
                row72 = add_flag(row72, platform)
            p = chosen.iloc[i]
            alert_id = f"{platform}:{unit}:{t:%Y%m%dT%H%M}"
            ev = build_evidence(alert_id, add_flag(row, platform), row72, shap_values[i], ew.columns, base,
                                float(p["risk"]), float(p["threshold"]), versions["early_warning"],
                                pd.DataFrame(), np.empty((0, 0)))
            text = summariser.summarise(ev)
            check = check_summary(text, ev, "template")
            unsupported = [c for c in check.claims if not c.supported]
            rows.append({
                "alert_id": alert_id, "platform": platform, "unit_id": unit, "t": t, "date": day.isoformat(),
                "risk": float(p["risk"]), "threshold": float(p["threshold"]),
                "soh_now_pct": float(soh_latest.get(unit, np.nan)),
                "drivers": json.dumps([{"feature": a.feature, "phrase": a.phrase, "unit": a.unit,
                                        "value": a.value, "value_72h_ago": a.value_72h_ago, "shap": round(a.shap, 4)}
                                       for a in ev.positive[:3]]),
                "summary": text, "summary_faithful_auto": not unsupported, "claims_checked": len(check.claims),
                "unsupported_claims": json.dumps([{"kind": c.kind, "detail": c.detail} for c in unsupported]),
                "evidence": json.dumps(ev.to_dict(), default=str), "model": versions["early_warning"],
                "status": "open",
            })
    alerts = pd.DataFrame(rows, columns=ALERT_COLUMNS)
    if alerts.empty:
        alerts = alerts.astype({"risk": float, "threshold": float, "soh_now_pct": float,
                                "summary_faithful_auto": bool, "claims_checked": int})
        alerts["t"] = pd.Series(dtype="datetime64[us, UTC]")
    target = settings.table_dir("alerts", platform, day)
    with atomic_partition(target) as tmp:
        alerts.to_parquet(tmp / "alerts.parquet", index=False)
    summary = {"platform": platform, "date": day.isoformat(), "alerts": len(alerts),
               "summaries_faithful_auto": int(alerts["summary_faithful_auto"].sum()) if len(alerts) else 0,
               "units_muted_from_earlier_alerts": len(last)}
    prev_paths = [settings.table_dir("alerts", platform, day - timedelta(days=k)) / "alerts.parquet" for k in (1, 2, 3)]
    write_manifest(settings.run_manifest(platform, day, "alerts"), {
        "step": "alerts", "inputs": fingerprint([pred_path, feat_path] + [p for p in prev_paths if p.exists()]),
        "output": fingerprint([target / "alerts.parquet"]), "summary": summary})
    return summary


def add_flag(row: pd.Series, platform: str) -> pd.Series:
    out = row.copy()
    out["platform_fari"] = 1.0 if platform == "fari" else 0.0
    return out
