"""Daily scoring: features of one date -> predictions table partition.

Per hourly row: calibrated early-warning risk (champion), its raw margin,
the platform's alert threshold, the shadow scores (Isolation Forest,
physics rule) and a data-sufficiency flag. On the 00:00 rows (end of the
previous day) also the 7-day SoH forecast. Models come from the MLflow
registry at the versions pinned in ``configs/pipeline.yaml``.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from ingen_fleet_health.models.soh_forecast import daily_features
from ingen_fleet_health.pipeline.featurize import read_features
from ingen_fleet_health.pipeline.settings import (
    PartitionNotReady,
    Settings,
    atomic_partition,
    fingerprint,
    write_manifest,
)
from ingen_fleet_health.release.bundle import EarlyWarningBundle, SohBundle, load_bundle

SOH_HISTORY_DAYS = 21


@lru_cache(maxsize=8)
def _cached(name: str, version: str, uri: str) -> Any:
    return load_bundle(name, version)


def models(settings: Settings) -> tuple[EarlyWarningBundle, SohBundle, dict[str, str]]:
    from ingen_fleet_health.tracking import tracking_uri

    ew_spec, soh_spec = settings.models["early_warning"], settings.models["soh_forecast"]
    ew = _cached(str(ew_spec["name"]), str(ew_spec["version"]), tracking_uri())
    soh = _cached(str(soh_spec["name"]), str(soh_spec["version"]), tracking_uri())
    versions = {"early_warning": f"{ew_spec['name']}/{ew_spec['version']}",
                "soh_forecast": f"{soh_spec['name']}/{soh_spec['version']}"}
    return ew, soh, versions


class InSampleDate(ValueError):
    """The date lies inside the production model's training window."""


def scoring_start(settings: Settings, platform: str) -> date:
    """First date the production models may score: the training cut-off.
    Earlier dates are in-sample for the training units and are never scored."""
    from ingen_fleet_health.pipeline.featurize import fleet_start

    return (fleet_start(settings, platform) + pd.Timedelta(days=float(settings.release["cutoff_day"]))).date()


def score_day(settings: Settings, platform: str, day: date) -> dict[str, Any]:
    if day < scoring_start(settings, platform):
        raise InSampleDate(f"{platform} {day}: before the scoring start {scoring_start(settings, platform)} "
                           "(training window); only ingest and features run for these dates")
    path = settings.table_dir("features", platform, day) / "features.parquet"
    if not path.exists():
        raise PartitionNotReady(f"{platform} {day}: no feature partition; run features first")
    feats = pd.read_parquet(path)
    ew, soh, versions = models(settings)
    min_data = float(settings.alerting["min_data_fraction_24h"])
    out = feats[["unit_id", "platform", "t"]].copy()
    if len(feats):
        out["risk"] = ew.risk(feats, platform)
        out["raw_margin"] = ew.raw_margin(feats, platform)
        shadow = ew.shadow_scores(feats)
        out["iforest_score"] = shadow["iforest"]
        out["physics_rule_score"] = shadow["physics_rule"]
    else:
        for c in ("risk", "raw_margin", "iforest_score", "physics_rule_score"):
            out[c] = pd.Series(dtype=float)
    out["threshold"] = float(ew.thresholds[platform])
    missing = feats["missing_fraction_24h"].to_numpy(dtype=float) if len(feats) else np.array([])
    out["insufficient_data"] = ~(np.nan_to_num(1.0 - missing, nan=0.0) >= min_data)
    out["above_threshold"] = (out["risk"] >= out["threshold"]) & ~out["insufficient_data"]
    out["history_missing_days"] = feats["history_missing_days"].to_numpy() if len(feats) else []
    out["early_warning_model"] = versions["early_warning"]

    # SoH: forecast issued at 00:00 of ``day`` for the end of day (day - 1) + horizon
    hist_days = [day - timedelta(days=k) for k in range(SOH_HISTORY_DAYS, -1, -1)]
    hourly = read_features(settings, platform, hist_days)
    out["soh_now_pct"] = np.nan
    out["soh_forecast_pct"] = np.nan
    out["soh_target_date"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[us, UTC]")
    if len(hourly):
        daily = daily_features(hourly)
        today = daily[daily["day"].dt.date == day - timedelta(days=1)].copy()
        if len(today):
            today["forecast"] = soh.predict(today)
            idx = out["t"].dt.hour == 0
            key = out.loc[idx, "unit_id"]
            out.loc[idx, "soh_now_pct"] = key.map(today.set_index("unit_id")["bms_soh_mean_24h"]).to_numpy()
            out.loc[idx, "soh_forecast_pct"] = key.map(today.set_index("unit_id")["forecast"]).to_numpy()
            out.loc[idx, "soh_target_date"] = (pd.Timestamp(day - timedelta(days=1))
                                               + pd.Timedelta(days=soh.horizon_days)).tz_localize("UTC")
    out["soh_model"] = versions["soh_forecast"]
    out = out.sort_values(["unit_id", "t"]).reset_index(drop=True)
    target = settings.table_dir("predictions", platform, day)
    with atomic_partition(target) as tmp:
        out.to_parquet(tmp / "predictions.parquet", index=False)
    inputs = [path] + [settings.table_dir("features", platform, d) / "features.parquet" for d in hist_days
                       if (settings.table_dir("features", platform, d) / "features.parquet").exists() and d != day]
    summary = {"platform": platform, "date": day.isoformat(), "rows": len(out),
               "rows_above_threshold": int(out["above_threshold"].sum()),
               "rows_insufficient_data": int(out["insufficient_data"].sum()),
               "soh_forecasts": int(out["soh_forecast_pct"].notna().sum()), "models": versions}
    write_manifest(settings.run_manifest(platform, day, "score"), {
        "step": "score", "inputs": fingerprint(inputs), "models": versions,
        "output": fingerprint([target / "predictions.parquet"]), "summary": summary})
    return summary


def read_table(settings: Settings, table: str, platform: str, days: list[date], filename: str) -> pd.DataFrame:
    frames = [pd.read_parquet(settings.table_dir(table, platform, d) / filename) for d in days
              if (settings.table_dir(table, platform, d) / filename).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
