"""Fast unit tests for Week 4 logic (no fleet needed)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ingen_fleet_health.monitoring.drift import (
    bin_counts,
    psi,
    psi_band,
    reference_bins,
)
from ingen_fleet_health.pipeline.alerting import select_alerts
from ingen_fleet_health.pipeline.settings import atomic_partition, daterange
from ingen_fleet_health.pipeline.views import pull_list
from ingen_fleet_health.release.cost_benefit import Assumptions, evaluate


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_psi_is_zero_for_the_reference_and_large_for_a_shift() -> None:
    rng = np.random.default_rng(0)
    ref = reference_bins(pd.Series(rng.normal(0, 1, 20000)))
    same = psi(ref["expected"], bin_counts(pd.Series(rng.normal(0, 1, 20000)), ref["edges"]))
    shifted = psi(ref["expected"], bin_counts(pd.Series(rng.normal(1.0, 1, 20000)), ref["edges"]))
    assert same < 0.01
    assert shifted > 0.25
    assert psi_band(same) == "stable" and psi_band(shifted) == "shift"


def test_psi_counts_missing_values_in_their_own_bin() -> None:
    ref = reference_bins(pd.Series([1.0, 2.0, 3.0, 4.0] * 100))
    outage = pd.Series([np.nan] * 50 + [2.0] * 50)
    assert psi(ref["expected"], bin_counts(outage, ref["edges"])) > 0.25


def test_refractory_period_spans_days() -> None:
    preds = pd.DataFrame({
        "unit_id": ["U1"] * 4 + ["U2"] * 2,
        "t": [ts("2026-12-02 01:00"), ts("2026-12-02 02:00"), ts("2026-12-02 03:00"), ts("2026-12-02 23:00"),
              ts("2026-12-02 05:00"), ts("2026-12-02 06:00")],
        "above_threshold": [True, True, False, True, True, True],
    })
    # U1 alerted 2026-11-30 20:00 -> muted until 2026-12-03 20:00; U2 has no earlier alert
    chosen = select_alerts(preds, {"U1": ts("2026-11-30 20:00")}, pd.Timedelta(hours=72))
    assert list(chosen["unit_id"]) == ["U2"]
    assert list(chosen["t"]) == [ts("2026-12-02 05:00")]


def test_atomic_partition_leaves_old_data_on_failure(tmp_path) -> None:  # type: ignore[no-untyped-def]
    target = tmp_path / "date=2026-12-01"
    with atomic_partition(target) as tmp:
        (tmp / "a.txt").write_text("v1")
    with pytest.raises(RuntimeError), atomic_partition(target) as tmp:
        (tmp / "a.txt").write_text("partial")
        raise RuntimeError("boom")
    assert (target / "a.txt").read_text() == "v1"
    assert [p.name for p in tmp_path.iterdir()] == ["date=2026-12-01"]


def test_daterange_is_inclusive() -> None:
    assert list(daterange(date(2026, 12, 30), date(2027, 1, 1))) == [
        date(2026, 12, 30), date(2026, 12, 31), date(2027, 1, 1)]


def test_pull_list_ranks_open_alerts_first_and_flags_missing_data() -> None:
    now = ts("2026-12-05 23:00")
    preds = pd.DataFrame({
        "unit_id": ["A", "B", "C"], "platform": ["aido_rover", "fari", "aido_rover"], "t": [now] * 3,
        "risk": [0.02, 0.30, 0.50], "threshold": [0.05, 0.05, 0.05], "insufficient_data": [False, False, True],
        "soh_forecast_pct": [np.nan] * 3, "soh_now_pct": [np.nan] * 3,
    })
    alerts = pd.DataFrame({"unit_id": ["A"], "t": [now - pd.Timedelta(hours=10)], "summary": ["A was flagged ..."],
                           "drivers": ['[{"phrase": "vibration"}]']})
    table = pull_list(preds, alerts, pd.DataFrame(), now)
    assert list(table["unit_id"]) == ["A", "C", "B"]
    assert list(table["status"]) == ["Pull before next shift", "No recent data", "OK"]
    assert table.loc[0, "main_driver"] == "vibration"


def test_alerts_older_than_a_shift_are_shown_as_already_alerted() -> None:
    now = ts("2026-12-05 23:00")
    preds = pd.DataFrame({"unit_id": ["A"], "platform": ["fari"], "t": [now], "risk": [0.01], "threshold": [0.05],
                          "insufficient_data": [False], "soh_forecast_pct": [np.nan], "soh_now_pct": [np.nan]})
    alerts = pd.DataFrame({"unit_id": ["A"], "t": [now - pd.Timedelta(hours=30)], "summary": ["older"],
                           "drivers": ["[]"]})
    assert pull_list(preds, alerts, pd.DataFrame(), now).loc[0, "status"] == "Alerted in last 72 h"


def test_old_alerts_are_closed_after_the_refractory_window() -> None:
    now = ts("2026-12-05 23:00")
    preds = pd.DataFrame({"unit_id": ["A"], "platform": ["fari"], "t": [now], "risk": [0.01], "threshold": [0.05],
                          "insufficient_data": [False], "soh_forecast_pct": [np.nan], "soh_now_pct": [np.nan]})
    alerts = pd.DataFrame({"unit_id": ["A"], "t": [now - pd.Timedelta(hours=80)], "summary": ["old"],
                           "drivers": ["[]"]})
    assert pull_list(preds, alerts, pd.DataFrame(), now).loc[0, "status"] == "OK"


def test_cost_benefit_is_traceable_to_its_assumptions() -> None:
    a = Assumptions(cost_unplanned_failure=1000.0, cost_planned_repair=400.0, cost_false_alarm_inspection=100.0,
                    cost_unnecessary_swap=200.0, swap_probability_on_false_alarm=0.5, repair_success_if_caught=1.0)
    r = evaluate(alerts=10, true_alerts=4, events=8, events_detected=4, unit_weeks=20.0, assumptions=a)
    # 4 caught failures save (1000 - 400) each; 6 false alarms cost 100 + 0.5 * 200 each
    assert r["savings_from_caught_failures"] == pytest.approx(4 * 600)
    assert r["false_alarm_cost"] == pytest.approx(6 * 200)
    assert r["net_benefit"] == pytest.approx(2400 - 1200)
    assert r["net_benefit_per_unit_week"] == pytest.approx(60.0)
    # break-even: (failure - planned) / false-alarm cost must exceed false alarms / caught failures = 6 / 4
    assert r["break_even_ratio"] == pytest.approx(1.5)
