"""Post-hoc diagnostics of the fusion run (reads its saved out-of-fold scores; no refitting).

    python -m ingen_fleet_health.experiments.fusion_diagnostics --data-dir data/w02

Pooled PR-AUC compares scores across folds, so a per-fold view is added: for
each pair of arms, PR-AUC inside every fold that has positives, and how often
the second arm wins. Stacker weights are summarised from ``fusion_results.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingen_fleet_health.evaluation.metrics import pr_auc
from ingen_fleet_health.experiments.w03_common import read_json, write_json

PAIRS = [("telemetry_iforest", "late_stack_iforest"), ("telemetry_lgbm", "early_fusion"),
         ("telemetry_lgbm", "late_stack"), ("telemetry_lgbm", "late_noisy_or")]


def per_fold(oof: pd.DataFrame, a: str, b: str) -> dict[str, Any]:
    key = ["unit_id", "t"]
    joined = oof[oof["model"] == a][[*key, "y", "score", "fold_id"]].merge(
        oof[oof["model"] == b][[*key, "score"]], on=key, suffixes=("_a", "_b"))
    rows = [(fid, pr_auc(g["y"].to_numpy(), g["score_a"].to_numpy()), pr_auc(g["y"].to_numpy(), g["score_b"].to_numpy()))
            for fid, g in joined.groupby("fold_id") if g["y"].sum() > 0]
    arr = np.array([[r[1], r[2]] for r in rows])
    diff = arr[:, 1] - arr[:, 0]
    worst = sorted(rows, key=lambda r: r[2] - r[1])[:3]
    return {"folds_with_positives": len(rows), "mean_per_fold": [float(arr[:, 0].mean()), float(arr[:, 1].mean())],
            "median_diff": float(np.median(diff)), "second_wins": int((diff > 0).sum()),
            "worst_folds": [{"fold_id": f, "first": float(x), "second": float(y)} for f, x, y in worst]}


RELIABILITY_BINS = [0.0, 0.05, 0.1, 0.2, 0.5, 0.9, 0.99, 1.0]


def reliability(oof: pd.DataFrame, arm: str) -> list[dict[str, Any]]:
    """Mean predicted vs observed positive rate per score bin (pooled unseen-robot test rows)."""
    g = oof[oof["model"] == arm]
    bins = pd.cut(g["score"], RELIABILITY_BINS, include_lowest=True)
    table = g.groupby(bins, observed=True).agg(rows=("y", "size"), observed=("y", "mean"), predicted=("score", "mean"))
    return [{"bin": str(k), "rows": int(r.rows), "observed": float(r.observed), "predicted": float(r.predicted)}
            for k, r in table.iterrows()]


def run(data_dir: Path) -> dict[str, Any]:
    oof = pd.read_parquet(data_dir / "predictions" / "fusion_oof.parquet")
    res = read_json("fusion_results.json")
    w = np.array([f["late_stack_iforest_weights_tel_text"] for f in res["fold_info"]], dtype=float)
    wl = np.array([f["late_stack_weights_tel_text"] for f in res["fold_info"]], dtype=float)
    out: dict[str, Any] = {
        "source_mlflow_run": res["mlflow_run_id"],
        "per_fold_pr_auc": {f"{b}_vs_{a}": per_fold(oof, a, b) for a, b in PAIRS},
        "stacker_weights_median": {
            "late_stack_iforest": {"telemetry": float(np.nanmedian(w[:, 0])), "text": float(np.nanmedian(w[:, 1]))},
            "late_stack": {"telemetry": float(np.nanmedian(wl[:, 0])), "text": float(np.nanmedian(wl[:, 1]))},
        },
        "folds_text_weight_exceeds_telemetry": {"late_stack_iforest": int(np.sum(w[:, 1] > w[:, 0])),
                                                "late_stack": int(np.sum(wl[:, 1] > wl[:, 0])), "folds": len(w)},
    }
    out["reliability"] = {arm: reliability(oof, arm) for arm in ("telemetry_lgbm", "early_fusion", "late_stack")}
    write_json(out, "fusion_diagnostics.json")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    out = run(parser.parse_args().data_dir)
    for k, v in out["per_fold_pr_auc"].items():
        print(k, v["mean_per_fold"], v["median_diff"], f"{v['second_wins']}/{v['folds_with_positives']}")
    print(out["stacker_weights_median"], out["folds_text_weight_exceeds_telemetry"])


if __name__ == "__main__":
    main()
