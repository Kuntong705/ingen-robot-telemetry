import numpy as np
import pandas as pd
import pytest

from ingen_fleet_health.evaluation.metrics import (
    evaluate_alerts,
    raise_alerts,
    threshold_for_budget,
)
from ingen_fleet_health.evaluation.splits import (
    check_fold,
    random_row_split,
    unit_groups,
    walk_forward_folds,
)

T0 = pd.Timestamp("2026-01-01T00:00:00Z")
H = pd.Timedelta(hours=72)


def _frame(n_units=10, hours=24 * 40):
    times = pd.date_range(T0, periods=hours, freq="h", tz="UTC")
    units = [f"U{i}" for i in range(n_units)]
    return pd.DataFrame({"unit_id": np.repeat(units, hours), "t": np.tile(times, n_units)})


def test_folds_are_unit_disjoint_and_embargoed():
    frame = _frame()
    groups = unit_groups(frame["unit_id"].unique().tolist(), n_groups=5, seed=1)
    folds = walk_forward_folds(frame, groups, [T0 + pd.Timedelta(days=d) for d in (15, 25)],
                               pd.Timedelta(days=10), H, pd.Timedelta(days=4))
    assert len(folds) == 10
    for fold in folds:
        report = check_fold(frame, fold, H)
        assert report["embargo_hours"] >= 72
        test = frame.iloc[fold.test_idx]
        assert set(test["unit_id"]) == set(fold.test_units)
    # each test (unit, t) is scored exactly once across folds
    all_test = np.concatenate([f.test_idx for f in folds])
    assert len(all_test) == len(np.unique(all_test))


def test_check_fold_catches_leaky_split():
    frame = _frame(n_units=4)
    groups = unit_groups(frame["unit_id"].unique().tolist(), n_groups=2, seed=0)
    fold = walk_forward_folds(frame, groups, [T0 + pd.Timedelta(days=20)], pd.Timedelta(days=5), H,
                              pd.Timedelta(days=3))[0]
    leaky = type(fold)(fold.fold_id, fold.cutoff, fold.group, fold.test_units,
                       np.concatenate([fold.fit_idx, fold.test_idx[:5]]), fold.val_idx, fold.test_idx)
    with pytest.raises(AssertionError):
        check_fold(frame, leaky, H)


def test_random_split_is_partition():
    train, test = random_row_split(100, 0.2, seed=3)
    assert len(test) == 20 and len(set(train) | set(test)) == 100


def _scored():
    times = pd.date_range(T0, periods=200, freq="h", tz="UTC")
    score = np.zeros(200)
    score[100:110] = 0.9      # true alert before onset at hour 120
    score[20] = 0.95          # false alarm
    y = np.zeros(200, dtype=int)
    y[48:120] = 1             # onset at hour 120, 72 h window
    return pd.DataFrame({"unit_id": "A", "t": times, "score": score, "y": y})


def test_alerts_refractory_lead_time_and_false_alarms():
    scored = _scored()
    alerts = raise_alerts(scored, 0.5, refractory_hours=72)
    assert len(alerts) == 2                      # hours 20 and 100; 101-109 muted
    events = pd.DataFrame({"unit_id": ["A"], "fault_id": ["F"], "onset_utc": [T0 + pd.Timedelta(hours=120)]})
    m = evaluate_alerts(scored, events, 0.5, 72, 72)
    assert m["precision"] == 0.5
    assert m["event_recall"] == 1.0
    assert m["median_lead_time_h"] == 20.0
    assert m["false_alarms_per_unit_week"] == pytest.approx(1 / (200 / 168))


def test_threshold_respects_budget():
    scored = _scored()
    thr = threshold_for_budget(scored, budget_per_unit_week=0.9, refractory_hours=72)
    assert len(raise_alerts(scored, thr, 72)) / (200 / 168) <= 0.9
