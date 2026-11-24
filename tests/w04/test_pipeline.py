"""Week 4 daily pipeline: as-of features, idempotency, missing partitions,
backfill recovery, privacy gate, registry-loaded models, monitoring."""

from __future__ import annotations

import shutil
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from ingen_fleet_health.contract import ContractViolation
from ingen_fleet_health.pipeline.ingest import ingest_day
from ingen_fleet_health.pipeline.runner import (
    backfill,
    performance_for_day,
    run_day,
    run_step,
    status,
)
from ingen_fleet_health.pipeline.settings import (
    PartitionNotReady,
    read_manifest,
    sha256_file,
)

TABLE_FILES = {
    "validated": "telemetry.parquet", "quarantine": "rejected.parquet", "features": "features.parquet",
    "predictions": "predictions.parquet", "alerts": "alerts.parquet",
}


def output_hashes(settings, platform, day) -> dict[str, str]:  # type: ignore[no-untyped-def]
    out = {}
    for table, name in TABLE_FILES.items():
        path = settings.table_dir(table, platform, day) / name
        if path.exists():
            out[table] = sha256_file(path)
    drift = settings.drift_dir(platform, day) / "psi.parquet"
    if drift.exists():
        out["drift"] = sha256_file(drift)
    return out


def batch_rows(fleet, platform, day) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    table = pd.read_parquet(fleet.root / "sources" / platform / "features" / "features_hourly.parquet")
    return table[table["t"].dt.date == day].reset_index(drop=True)


@pytest.mark.parametrize("platform", ["aido_rover", "fari"])
def test_daily_features_equal_the_batch_feature_table(fleet, platform) -> None:  # type: ignore[no-untyped-def]
    day = fleet.scoring_start
    daily = pd.read_parquet(fleet.settings.table_dir("features", platform, day) / "features.parquet")
    batch = batch_rows(fleet, platform, day)
    assert len(daily) == len(batch) > 0
    pd.testing.assert_frame_equal(daily[batch.columns].reset_index(drop=True), batch)


def test_first_fleet_day_writes_an_empty_feature_partition(fleet) -> None:  # type: ignore[no-untyped-def]
    """Day 0 lies inside the 24 h warm-up: no prediction time, but the step must succeed (regression:
    the parallel build once failed on it, found by the fresh-clone quickstart)."""
    from datetime import date

    from ingen_fleet_health.pipeline.featurize import features_for_day

    summary = features_for_day(fleet.settings, "aido_rover", date(2026, 10, 26))
    assert summary["rows"] == 0
    part = pd.read_parquet(fleet.settings.table_dir("features", "aido_rover", date(2026, 10, 26)) / "features.parquet")
    assert len(part) == 0 and {"unit_id", "t"} <= set(part.columns)


def test_rerunning_a_date_reproduces_identical_files(fleet) -> None:  # type: ignore[no-untyped-def]
    day = fleet.scoring_start + timedelta(days=2)
    before = {p: output_hashes(fleet.settings, p, day) for p in fleet.settings.platforms}
    assert all({"validated", "features", "predictions", "alerts", "drift"} <= set(h) for h in before.values())
    run_day(fleet.settings, day)
    after = {p: output_hashes(fleet.settings, p, day) for p in fleet.settings.platforms}
    assert before == after


def test_models_come_from_the_registry_at_the_pinned_version(fleet) -> None:  # type: ignore[no-untyped-def]
    day = fleet.scoring_start
    preds = pd.read_parquet(fleet.settings.table_dir("predictions", "aido_rover", day) / "predictions.parquet")
    assert set(preds["early_warning_model"]) == {"fleet_early_warning/1"}
    assert set(preds["soh_model"]) == {"fleet_soh_forecast/1"}
    manifest = read_manifest(fleet.settings.run_manifest("aido_rover", day, "score"))
    assert manifest is not None and manifest["models"]["early_warning"] == "fleet_early_warning/1"
    assert preds["risk"].between(0, 1).all()
    assert (preds["threshold"] > 0).all()


def test_dates_in_the_training_window_are_never_scored(fleet) -> None:  # type: ignore[no-untyped-def]
    day = fleet.scoring_start - timedelta(days=1)
    res = run_step(fleet.settings, "score", "fari", day)
    assert "skipped" in res
    assert not (fleet.settings.table_dir("predictions", "fari", day)).exists()


def test_missing_partition_writes_nothing_then_backfill_recovers(fleet) -> None:  # type: ignore[no-untyped-def]
    s = fleet.settings
    late = fleet.scoring_start + timedelta(days=3)
    after = late + timedelta(days=1)
    landing = s.landing_dir("telemetry", "fari", late)
    parked = landing.parent / f"_parked_{late}"
    shutil.move(str(landing), str(parked))
    try:
        res = run_day(s, late)
        assert res["platforms"]["fari"]["status"] == "missing"
        assert res["platforms"]["aido_rover"]["status"] == "ok"           # the other platform still runs
        for table in ("validated", "features", "predictions", "alerts"):
            assert not s.table_dir(table, "fari", late).exists()
        with pytest.raises(PartitionNotReady):
            ingest_day(s, "fari", late)
        # the next day still runs; its history gap is recorded on every row
        res2 = run_day(s, after, ["fari"])
        assert res2["platforms"]["fari"]["status"] == "ok"
        gap = pd.read_parquet(s.table_dir("features", "fari", after) / "features.parquet")
        assert (gap["history_missing_days"] == 1).all()
        assert late.isoformat() in status(s)["fari"]["unscored_dates_before_last_scored"]
    finally:
        shutil.move(str(parked), str(landing))
    # the late partition arrives: backfill from it onwards; results equal a complete history
    backfill(s, late, after, ["fari"], echo=lambda _m: None)
    fixed = pd.read_parquet(s.table_dir("features", "fari", after) / "features.parquet")
    assert (fixed["history_missing_days"] == 0).all()
    batch = batch_rows(fleet, "fari", after)
    pd.testing.assert_frame_equal(fixed[batch.columns].reset_index(drop=True), batch)
    assert status(s)["fari"]["unscored_dates_before_last_scored"] == []


def test_forbidden_imagery_field_refuses_the_file(fleet, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    s = fleet.settings
    day = fleet.scoring_start + timedelta(days=5)
    landing = s.landing_dir("telemetry", "fari", day) / "telemetry.csv"
    original = landing.read_bytes()
    try:
        frame = pd.read_csv(landing, dtype=str)
        frame["camera_frame_id"] = "0"
        frame.to_csv(landing, index=False)
        with pytest.raises(ContractViolation):
            ingest_day(s, "fari", day)
        assert not s.table_dir("validated", "fari", day).exists()
    finally:
        landing.write_bytes(original)


def test_alert_records_carry_grounded_summaries(fleet) -> None:  # type: ignore[no-untyped-def]
    s = fleet.settings
    frames = [pd.read_parquet(s.table_dir("alerts", p, d) / "alerts.parquet")
              for p in s.platforms for d in s.partition_dates("alerts", p)]
    alerts = pd.concat(frames, ignore_index=True)
    for _, a in alerts.iterrows():
        assert a["unit_id"] in a["summary"]
        assert "Operator notes are not an input" in a["summary"]
        assert a["claims_checked"] > 0
        assert a["risk"] >= a["threshold"]


def test_weekly_performance_report_has_both_cohorts(fleet) -> None:  # type: ignore[no-untyped-def]
    day = fleet.scoring_start + timedelta(days=14)                    # window = [scoring_start, +7 d)
    report = performance_for_day(fleet.settings, day, force=True)
    assert report is not None
    for platform in fleet.settings.platforms:
        entry = report["platforms"][platform]
        assert entry["prediction_days_found"] >= 3
        for cohort in ("held_out_units", "training_units"):
            m = entry[cohort]
            assert m["rows"] > 0
            assert m["status"] in ("evaluated", "insufficient labels")
