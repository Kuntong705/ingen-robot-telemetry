import pandas as pd

from ingen_fleet_health.labels import early_warning_labels, prediction_grid, soh_targets

T0 = pd.Timestamp("2026-01-01T00:00:00Z")


def _events():
    return pd.DataFrame([{
        "fault_id": "F1",
        "unit_id": "A",
        "onset_utc": T0 + pd.Timedelta(hours=100),
        "repair_end_utc": T0 + pd.Timedelta(hours=130),
    }])


def _labels():
    grid = prediction_grid(["A", "B"], T0, T0 + pd.Timedelta(hours=240), warmup_hours=0)
    return early_warning_labels(grid, _events(), T0 + pd.Timedelta(hours=240)).set_index(["unit_id", "t"])


def at(hours):
    return T0 + pd.Timedelta(hours=hours)


def test_positive_window_is_open_closed():
    lab = _labels()
    assert lab.loc[("A", at(27)), "y"] == 0      # onset 73 h ahead
    assert lab.loc[("A", at(28)), "y"] == 1      # exactly 72 h ahead -> included
    assert lab.loc[("A", at(99)), "y"] == 1
    assert lab.loc[("A", at(99)), "hours_to_onset"] == 1
    assert lab.loc[("A", at(100)), "y"] == 0     # onset itself is no longer "future"


def test_active_fault_post_repair_and_censoring():
    lab = _labels()
    assert lab.loc[("A", at(100)), "in_active_fault"]
    assert not lab.loc[("A", at(129)), "eligible"]
    assert lab.loc[("A", at(130)), "post_repair"]
    assert lab.loc[("A", at(154)), "eligible"]
    assert lab.loc[("A", at(168)), "eligible"]   # 168 + 72 = 240 = end -> complete window
    assert lab.loc[("A", at(169)), "censored"]
    assert lab.xs("B")["y"].sum() == 0


def test_soh_targets_shift_within_unit():
    truth = pd.DataFrame({
        "unit_id": ["A"] * 3 + ["B"] * 3,
        "date_utc": ["2026-01-01", "2026-01-02", "2026-01-03"] * 2,
        "soh_true_pct": [100, 99, 98, 90, 89, 88],
    })
    out = soh_targets(truth, [1])
    assert out["target_h1"].iloc[:2].tolist() == [99, 98]
    assert pd.isna(out["target_h1"].iloc[2])     # no leakage across units
    assert out["target_h1"].iloc[3] == 89
