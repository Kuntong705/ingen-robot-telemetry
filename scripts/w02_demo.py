"""Live Week 2 demo run: one walk-forward fold, end to end, on unseen robots.

Everything printed and saved here is computed in this run by the project code
(no cached results): the fold is built and audited, the early-warning models
are trained and calibrated, alerts are raised for the test robots, and a SoH
forecaster is trained and scored. ``scripts/render_demo_video.py`` turns the
saved log into an MP4.

    python scripts/w02_demo.py --data-dir data/w02 --out media/w02_demo_log.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingen_fleet_health.evaluation.metrics import (
    evaluate_alerts,
    events_in_scored_period,
    mae,
    pr_auc,
    raise_alerts,
    threshold_for_budget,
)
from ingen_fleet_health.evaluation.splits import (
    check_fold,
    random_row_split,
    walk_forward_folds,
)
from ingen_fleet_health.experiments.common import load_w02
from ingen_fleet_health.experiments.early_warning_run import TEST_SPAN, VAL_SPAN
from ingen_fleet_health.models.early_warning import (
    fit_lgbm,
    model_feature_columns,
    positive_proba,
    score_fold,
)
from ingen_fleet_health.models.soh_forecast import (
    daily_table,
    fit_predict_lgbm,
    linear_extrapolation,
    persistence,
)

DEMO_UNIT = "Rover_023"
CUTOFF_DAY = 40


class Log:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.lines: list[dict[str, Any]] = []

    def __call__(self, text: str, section: str) -> None:
        elapsed = time.perf_counter() - self.t0
        self.lines.append({"elapsed_s": round(elapsed, 2), "section": section, "text": text})
        print(f"[{elapsed:6.1f}s] {text}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/w02"))
    parser.add_argument("--out", type=Path, default=Path("media/w02_demo_log.json"))
    args = parser.parse_args()
    log = Log()
    started = datetime.now(UTC).isoformat(timespec="seconds")

    log("$ python scripts/w02_demo.py --data-dir data/w02", "load")
    data = load_w02(args.data_dir)
    eligible = data.eligible()
    log(f"loaded {len(data.features):,} hourly feature rows, {data.features['unit_id'].nunique()} robots, "
        f"{len(model_feature_columns())} model features", "load")

    horizon = pd.Timedelta(hours=72)
    cutoff = data.cutoffs([CUTOFF_DAY])[0]
    group = data.groups[DEMO_UNIT]
    fold = next(f for f in walk_forward_folds(eligible, data.groups, [cutoff], TEST_SPAN, horizon, VAL_SPAN)
                if f.group == group)
    audit = check_fold(eligible, fold, horizon)
    log(f"fold {fold.fold_id}: cut-off {cutoff:%Y-%m-%d}, test robots {', '.join(fold.test_units[:4])}, ...", "split")
    log(f"audit: shared robots {audit['shared_units']}, duplicate rows {audit['duplicate_keys']}, "
        f"embargo {audit['embargo_hours']:.0f} h  -> PASS", "split")

    scores = score_fold(eligible, fold, model_feature_columns(), seed=0,
                        models=("physics_rule", "lgbm_physics", "iforest"))
    log(f"trained LightGBM on {scores.info['fit_rows']:,} rows ({scores.info['fit_positives']} positive hours), "
        f"calibration: {scores.info['calibration'].get('lgbm_physics')}", "train")
    log("trained Isolation Forest on the same rows, without labels", "train")

    test = eligible.iloc[fold.test_idx][["unit_id", "t", "y"]].reset_index(drop=True)
    results: dict[str, Any] = {}
    for model in ("physics_rule", "lgbm_physics", "iforest"):
        val = eligible.iloc[fold.val_idx][["unit_id", "t"]].assign(score=scores.val_scores[model])
        thr = threshold_for_budget(val, 1.0, 72)
        scored = test.assign(score=scores.scores[model], thr=thr)
        ev = events_in_scored_period(scored, data.events)
        m = evaluate_alerts(scored, ev, None, 72, 72)
        m["pr_auc"] = pr_auc(scored["y"].to_numpy(), scored["score"].to_numpy())
        results[model] = {"scored": scored, "metrics": m}
        log(f"{model:13s} PR-AUC {m['pr_auc']:.3f} | warned {int(m['events_detected'])}/{int(m['events'])} onsets | "
            f"{m['false_alarms_per_unit_week']:.2f} false alarms per robot-week", "score")

    unit_rows = results["iforest"]["scored"][results["iforest"]["scored"]["unit_id"] == DEMO_UNIT]
    alerts = raise_alerts(unit_rows, None, 72)
    onset = data.events.query("unit_id == @DEMO_UNIT and onset_utc >= @cutoff").iloc[0]
    warned = alerts[(alerts["t"] >= onset.onset_utc - horizon) & (alerts["t"] < onset.onset_utc)]
    lead = (onset.onset_utc - warned["t"].min()) / pd.Timedelta(hours=1) if len(warned) else float("nan")
    log(f"{DEMO_UNIT} (never in training): first alert {warned['t'].min():%m-%d %H:%M}, onset "
        f"{onset.onset_utc:%m-%d %H:%M} ({onset.fault_type}) -> {lead:.0f} h lead time", "alert")

    tr, te = random_row_split(len(eligible), 0.2, seed=0)
    cols = model_feature_columns()
    rand_model = fit_lgbm(eligible.iloc[tr][cols], eligible.iloc[tr]["y"].to_numpy())
    rand_auc = pr_auc(eligible.iloc[te]["y"].to_numpy(), positive_proba(rand_model, eligible.iloc[te][cols]))
    proper_model = fit_lgbm(eligible.iloc[fold.train_idx][cols], eligible.iloc[fold.train_idx]["y"].to_numpy())
    proper_auc = pr_auc(test["y"].to_numpy(), positive_proba(proper_model, eligible.iloc[fold.test_idx][cols]))
    log(f"leakage check, same LightGBM: random row split PR-AUC {rand_auc:.3f} vs this fold {proper_auc:.3f}", "leak")

    h = 7
    table = daily_table(data.features, data.battery_truth, [h]).drop(columns="t").rename(columns={"day": "t"})
    table = table[table[f"target_h{h}"].notna()].reset_index(drop=True)
    sfold = next(f for f in walk_forward_folds(table, data.groups, [cutoff], TEST_SPAN, pd.Timedelta(days=h),
                                               pd.Timedelta(0)) if f.group == group)
    check_fold(table, sfold, pd.Timedelta(days=h))
    s_train, s_test = table.iloc[sfold.fit_idx], table.iloc[sfold.test_idx]
    preds = {"persistence": persistence(s_test, h), "linear": linear_extrapolation(s_test, h),
             "lgbm": fit_predict_lgbm(s_train, s_test, h)}
    soh_mae = {k: mae(s_test[f"target_h{h}"], v) for k, v in preds.items()}
    log(f"SoH 7-day forecast, unseen robots: MAE persistence {soh_mae['persistence']:.3f} | linear "
        f"{soh_mae['linear']:.3f} | LightGBM {soh_mae['lgbm']:.3f} pp", "soh")
    log("done: all numbers above were computed in this run", "done")

    u = s_test["unit_id"] == DEMO_UNIT
    payload = {
        "started_utc": started,
        "lines": log.lines,
        "demo_unit": DEMO_UNIT,
        "fold_id": fold.fold_id,
        "onset_utc": onset.onset_utc.isoformat(),
        "fault_type": onset.fault_type,
        "lead_time_h": lead,
        "risk": {"t": [t.isoformat() for t in unit_rows["t"]], "score": unit_rows["score"].round(4).tolist(),
                 "thr": float(unit_rows["thr"].iloc[0])},
        "alerts": [t.isoformat() for t in alerts["t"]],
        "model_metrics": {k: {m: v["metrics"][m] for m in ("pr_auc", "events", "events_detected",
                                                           "false_alarms_per_unit_week")} for k, v in results.items()},
        "leakage": {"random_split_pr_auc": rand_auc, "fold_pr_auc": proper_auc},
        "soh": {"mae": soh_mae, "t": [t.isoformat() for t in s_test.loc[u, "t"]],
                "truth": s_test.loc[u, f"target_h{h}"].round(4).tolist(),
                "lgbm": np.round(preds["lgbm"][u.to_numpy()], 4).tolist(),
                "persistence": np.round(preds["persistence"][u.to_numpy()], 4).tolist()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    print(f"log written to {args.out}")


if __name__ == "__main__":
    main()
