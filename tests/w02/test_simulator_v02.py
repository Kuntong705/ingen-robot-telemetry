import json

import numpy as np
import pandas as pd
import pytest

from ingen_fleet_health.simulate.faults import ramp, ramp_inverse
from ingen_fleet_health.simulate.fleet import TELEMETRY_FIELDS, simulate_fleet_v02
from ingest_pipeline import run_pipeline


@pytest.fixture(scope="module")
def sim_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("sim_v02")
    simulate_fleet_v02(out, units=6, days=8, seed=7)
    return out


def test_reproducible(sim_dir, tmp_path):
    simulate_fleet_v02(tmp_path, units=6, days=8, seed=7)
    for rel in ("generated/fleet_telemetry.csv", "ground_truth/fault_events.csv"):
        assert (sim_dir / rel).read_bytes() == (tmp_path / rel).read_bytes()


def test_units_are_heterogeneous(sim_dir):
    truth = pd.read_csv(sim_dir / "ground_truth" / "battery_truth_daily.csv")
    final_soh = truth.groupby("unit_id")["soh_true_pct"].last()
    assert final_soh.nunique() == len(final_soh)
    params = pd.read_csv(sim_dir / "ground_truth" / "unit_params.csv")
    assert params["cycle_fade_pct_per_efc"].nunique() == len(params)


def test_true_soh_monotone_without_repair(sim_dir):
    truth = pd.read_csv(sim_dir / "ground_truth" / "battery_truth_daily.csv")
    diffs = truth.groupby("unit_id")["soh_true_pct"].diff().dropna()
    assert (diffs <= 1e-9).all()


def test_telemetry_schema_and_no_ground_truth_columns(sim_dir):
    header = (sim_dir / "generated" / "fleet_telemetry.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == TELEMETRY_FIELDS


def test_week1_ingest_accepts_v02(sim_dir, tmp_path):
    report = run_pipeline(sim_dir / "generated" / "fleet_telemetry.csv", out_root=tmp_path)
    manifest = json.loads((sim_dir / "simulation_manifest.json").read_text(encoding="utf-8"))
    assert report["rejected_rows"] == 0
    assert report["accepted_rows"] == manifest["telemetry_rows_written"]
    assert report["missing_minutes"] == manifest["telemetry_rows_generated"] - manifest["telemetry_rows_written"]


def test_fault_timelines_are_ordered_and_varied(tmp_path):
    simulate_fleet_v02(tmp_path, units=30, days=10, seed=3)
    events = pd.read_csv(tmp_path / "ground_truth" / "fault_events.csv")
    assert len(events) > 5
    cols = ["initiation_utc", "onset_utc", "failure_utc"]
    times = events[cols].apply(pd.to_datetime)
    assert (times["initiation_utc"] < times["onset_utc"]).all()
    assert (times["onset_utc"] < times["failure_utc"]).all()
    assert events["onset_utc"].nunique() == len(events)
    assert events["fault_type"].nunique() >= 2


@pytest.mark.parametrize("shape", ["linear", "exponential", "sigmoid"])
def test_ramp_monotone_and_inverse(shape):
    x = np.linspace(0, 1, 1001)
    y = ramp(shape, x)
    assert y[0] == pytest.approx(0, abs=1e-9)
    assert y[-1] == pytest.approx(1, abs=1e-9)
    assert (np.diff(y) >= -1e-12).all()
    assert ramp(shape, np.array([ramp_inverse(shape, 0.3)]))[0] == pytest.approx(0.3, abs=1e-3)
