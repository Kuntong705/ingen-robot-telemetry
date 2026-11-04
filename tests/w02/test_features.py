import numpy as np
import pandas as pd
import pytest

from ingen_fleet_health.config import load_config
from ingen_fleet_health.features.build import KEYS, build_feature_table
from ingen_fleet_health.features.normalise import EnvironmentResidualizer
from ingen_fleet_health.features.physics import resistance_step_estimate
from ingen_fleet_health.features.registry import all_specs
from ingen_fleet_health.features.rolling import rolling_slope
from ingen_fleet_health.simulate.fleet import simulate_fleet_v02
from ingest_pipeline import run_pipeline


@pytest.fixture(scope="module")
def small_fleet(tmp_path_factory):
    out = tmp_path_factory.mktemp("feat")
    manifest = simulate_fleet_v02(out, units=2, days=4, seed=11)
    run_pipeline(out / "generated" / "fleet_telemetry.csv", out_root=out)
    telemetry = pd.read_parquet(out / "data" / "validated" / "telemetry.parquet")
    maintenance = pd.read_csv(out / "events" / "maintenance_events.csv")
    start, end = pd.Timestamp(manifest["start_utc"]), pd.Timestamp(manifest["end_utc"])
    return out, telemetry, maintenance, start, end


def test_features_ignore_future_data(small_fleet):
    """Core as-of test: corrupting everything after a cut-off must not change
    any feature at prediction times <= the cut-off."""
    _, telemetry, maintenance, start, end = small_fleet
    config = load_config()
    base = build_feature_table(telemetry, maintenance, config, start, end)
    cut = start + pd.Timedelta(hours=60)
    corrupted = telemetry.copy()
    future = corrupted["timestamp_utc"] > cut
    for col in ("motor_temp_c", "vibration_rms_g", "battery_voltage_v", "power_w", "ambient_temp_c"):
        corrupted.loc[future, col] = corrupted.loc[future, col] * 3.0 + 50.0
    corrupted = corrupted[~(future & (corrupted.index % 7 == 0))]  # also drop future rows
    pert = build_feature_table(corrupted, maintenance, config, start, end)
    a = base[base["t"] <= cut].set_index(KEYS).sort_index()
    b = pert[pert["t"] <= cut].set_index(KEYS).sort_index()
    pd.testing.assert_frame_equal(a, b)
    later = base[base["t"] > cut + pd.Timedelta(hours=2)].set_index(KEYS)["motor_temp_mean_1h"]
    assert not np.allclose(later, pert.set_index(KEYS).loc[later.index, "motor_temp_mean_1h"])


def test_registry_covers_every_feature(small_fleet):
    _, telemetry, maintenance, start, end = small_fleet
    table = build_feature_table(telemetry, maintenance, load_config(), start, end)
    feature_cols = set(table.columns) - set(KEYS)
    names = [s.name for s in all_specs()]
    assert len(names) == len(set(names))
    names = set(names)
    assert feature_cols == names


def test_rolling_slope_exact_with_gaps():
    y = 2.0 + 0.5 * np.arange(300) / 60.0  # 0.5 per hour
    y[50:80] = np.nan
    slope = rolling_slope(y, 120)
    assert np.nanmax(np.abs(slope[120:] - 0.5)) < 1e-9
    assert np.isnan(slope[0])


def test_resistance_estimate_recovers_known_r():
    rng = np.random.default_rng(0)
    current = np.repeat(rng.choice([0.0, 5.0, 10.0, -14.0], 200), 15)
    voltage = 48.0 - current * 0.05 + rng.normal(0, 0.05, current.size)
    r_hat = resistance_step_estimate(voltage, current, window=1440)
    assert r_hat[-1] == pytest.approx(0.05, rel=0.05)


def test_environment_residualizer_fits_only_on_given_rows():
    rng = np.random.default_rng(1)
    n = 500
    X = pd.DataFrame({
        "ambient_mean_24h": rng.uniform(0, 30, n),
        "terrain_patrol_mean_24h": rng.uniform(0, 1, n),
        "patrol_fraction_24h": rng.uniform(0.2, 0.6, n),
    })
    X["energy_per_km_24h"] = 150 + 60 * X["terrain_patrol_mean_24h"] - 0.8 * X["ambient_mean_24h"]
    model = EnvironmentResidualizer(targets=("energy_per_km_24h",)).fit(X.iloc[:250])
    out = model.transform(X.iloc[250:])
    assert np.abs(out["energy_per_km_24h_env_resid"]).max() < 1e-6
