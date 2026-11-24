"""Check that the registered production v1 models are reproducible from the committed code.

The v1 models were registered from a working tree that was not yet committed
(MLflow tag ``git_dirty=True``). This script retrains both models in memory with
the same configuration (nothing is registered) and compares them with the
registered bundles: the selection metrics, the calibration, the validation
metrics, the thresholds, and the calibrated risk on every validation row.
Writes ``reports/w04/retrain_reproducibility.json`` with the commit it ran on.

    python scripts/check_w04_retrain.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingen_fleet_health.pipeline.scoring import models
from ingen_fleet_health.pipeline.settings import load_settings
from ingen_fleet_health.release.train import (
    platform_frames,
    train_early_warning,
    train_soh,
)

OUT = ROOT / "reports" / "w04" / "retrain_reproducibility.json"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def main() -> None:
    settings = load_settings()
    registered_ew, registered_soh, versions = models(settings)
    seed = int(settings.release["seed"])
    frames = {p: platform_frames(settings, p) for p in settings.platforms}
    ew = train_early_warning(settings, frames, seed)
    soh = train_soh(settings, frames, seed)
    val = pd.concat([f["val"] for f in frames.values()], ignore_index=True)
    risk_new = ew.model.predict_proba(val[ew.columns])[:, 1]
    risk_reg = registered_ew.model.predict_proba(val[registered_ew.columns])[:, 1]
    daily = pd.concat([f["soh_train"] for f in frames.values()], ignore_index=True)
    soh_new = soh.model.predict(daily[soh.features])
    soh_reg = registered_soh.model.predict(daily[registered_soh.features])
    checks: dict[str, Any] = {
        "variant": ew.variant == registered_ew.variant,
        "columns": ew.columns == registered_ew.columns,
        "training_units": ew.training_units == registered_ew.training_units,
        "thresholds": ew.thresholds == registered_ew.thresholds,
        "shadow_thresholds": ew.shadow_thresholds == registered_ew.shadow_thresholds,
        "bundle_info": json.dumps(ew.info, sort_keys=True, default=str)
        == json.dumps(registered_ew.info, sort_keys=True, default=str),
        "validation_risk_max_abs_diff": float(np.max(np.abs(risk_new - risk_reg))),
        "soh_info": soh.info == registered_soh.info,
        "soh_train_prediction_max_abs_diff": float(np.max(np.abs(soh_new - soh_reg))),
    }
    identical = all(v is True for k, v in checks.items() if not k.endswith("diff")) and all(
        v == 0.0 for k, v in checks.items() if k.endswith("diff"))
    out = {"run_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
           "git_commit": git("rev-parse", "HEAD"), "git_worktree_clean": git("status", "--porcelain") == "",
           "registered_models": versions, "validation_rows": len(val), "checks": checks,
           "identical": identical}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
