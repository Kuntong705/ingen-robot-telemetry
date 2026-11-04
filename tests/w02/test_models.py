import numpy as np
import pandas as pd

from ingen_fleet_health.benchmarks.cmapss import run_benchmark, synthetic_cmapss
from ingen_fleet_health.models.early_warning import (
    calibrate,
    fit_lgbm,
    physics_rule_score,
)
from ingen_fleet_health.models.soh_forecast import _rolling_ols, linear_extrapolation


def test_cmapss_pipeline_on_synthetic_engines(monkeypatch):
    import ingen_fleet_health.benchmarks.cmapss as cm

    monkeypatch.setitem(cm.LGBM_PARAMS, "n_estimators", 60)  # keep the test fast
    train, test, rul = synthetic_cmapss(n_units=16, seed=1)
    res = run_benchmark(train, test, rul, with_lstm=False)
    assert res["test_engines"] == 8
    assert all(c.startswith("s") and c[1:].isdigit() for c in res["sensors_used"])
    # the learned model should beat the constant baseline on clearly degrading engines
    assert res["rmse_vs_capped_truth"]["lgbm"] < res["rmse_vs_capped_truth"]["constant_mean"]
    # holding whole engines out can only be harder than shuffling rows of the same engines
    internal = res["internal_split_check"]
    assert internal["engine_grouped_cv_rmse"] >= internal["random_row_cv_rmse"]


def test_cmapss_lstm_trains_and_predicts_in_range():
    from ingen_fleet_health.benchmarks.cmapss import (
        add_rul,
        fit_predict_lstm,
        informative_sensors,
    )

    train, test, _ = synthetic_cmapss(n_units=8, seed=2)
    train = add_rul(train)
    pred, info = fit_predict_lstm(train, test.assign(rul=float("nan")), informative_sensors(train), max_epochs=1)
    assert pred.shape == (4,)
    assert (pred >= 0).all() and (pred <= 125).all()
    assert info["epochs_run"] == 1


def test_linear_extrapolation_exact_on_linear_series():
    y = 100 - 0.05 * np.arange(30, dtype=float)
    level, slope = _rolling_ols(y, 14)
    frame = pd.DataFrame({"lin_level_14d": level, "lin_slope_14d": slope, "bms_soh_mean_24h": y})
    pred = linear_extrapolation(frame, 7)
    assert np.allclose(pred[13:], y[13:] - 0.35)


def test_calibration_maps_into_probability_range():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000)})
    y = (X["a"] + rng.normal(0, 1, 2000) > 1.5).astype(int).to_numpy()
    model = fit_lgbm(X.iloc[:1000], y[:1000])
    cal = calibrate(model, X.iloc[1000:1500], y[1000:1500])
    p = cal.predict_proba(X.iloc[1500:])[:, 1]
    assert cal.method in ("isotonic", "sigmoid")
    assert (p >= 0).all() and (p <= 1).all()
    # calibrated mean probability should be close to the observed rate
    assert abs(p.mean() - y[1500:].mean()) < 0.05


def test_physics_rule_handles_missing_values():
    frame = pd.DataFrame({
        "vib_norm_ratio_to_baseline": [1.0, 1.5, np.nan],
        "energy_ratio_to_baseline": [1.0, 1.0, np.nan],
        "r_hat_ratio_to_baseline": [1.1, np.nan, np.nan],
        "thermal_gain_ratio_24h": [1.0, 1.0, np.nan],
    })
    assert np.allclose(physics_rule_score(frame), [0.1, 0.5, 0.0])
