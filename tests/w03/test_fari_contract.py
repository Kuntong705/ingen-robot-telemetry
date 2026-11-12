"""Fari platform profile: data contract and privacy gate (Week 3).

All Fari values come from configs/fari.yaml, which is a SYNTHETIC assumption.
"""

import csv

import numpy as np
import pytest

from fleet_simulator import FIELDS
from ingen_fleet_health.config import REPO_ROOT, load_config
from ingen_fleet_health.contract import (
    DEFAULT_PROFILE,
    ContractViolation,
    assert_no_forbidden_fields,
    forbidden_fields,
    platform_profile,
)
from ingen_fleet_health.evaluation.shift import availability, psi
from ingest_pipeline import check_row, run_pipeline

FARI = platform_profile(load_config(REPO_ROOT / "configs" / "fari.yaml"))


def fari_row(**overrides):
    row = {
        "timestamp_utc": "2026-10-26T08:00:00+00:00", "unit_id": "Fari_001", "mode": "interact",
        "battery_soc_pct": "80", "battery_soh_pct": "99.9", "battery_voltage_v": "14.8",
        "battery_current_a": "2", "internal_resistance_ohm": "0.08", "power_w": f"{2 * FARI['nominal_voltage_v']:.2f}",
        "ambient_temp_c": "22", "motor_temp_c": "28", "terrain_factor": "0", "vibration_rms_g": "0.03",
        "odometer_m": "10",
    }
    row.update(overrides)
    return row


def test_profile_comes_from_the_platform_yaml():
    assert FARI["platform"] == "fari"
    assert FARI["allowed_modes"] == ("interact", "dock")
    assert FARI["voltage_range_v"][1] < DEFAULT_PROFILE["voltage_range_v"][0]   # a 48 V rule would reject every row
    assert "terrain_factor" in FARI["not_applicable_fields"]


def test_week1_rules_reject_fari_and_the_profile_accepts_it():
    row = fari_row()
    problems_default, _, _ = check_row(row, {}, {})
    problems_fari, _, _ = check_row(row, {}, {}, FARI)
    assert {"invalid_mode"} <= set(problems_default)
    assert problems_fari == []


def test_profile_still_rejects_bad_fari_rows():
    assert "invalid_mode" in check_row(fari_row(mode="patrol"), {}, {}, FARI)[0]
    assert check_row(fari_row(battery_voltage_v="48"), {}, {}, FARI)[0]          # outside the Fari window


def test_rover_rules_unchanged_without_a_profile():
    rover = fari_row(unit_id="Rover_001", mode="patrol", battery_voltage_v="48", power_w="96")
    assert check_row(rover, {}, {})[0] == []
    assert check_row(rover, {}, {}, DEFAULT_PROFILE)[0] == []


@pytest.mark.parametrize("column", ["camera_frame_id", "audio_rms", "mic_level", "microphone_gain", "voice_activity",
                                    "speech_transcript", "face_count", "ImageHash", "videoUrl"])
def test_imagery_and_audio_fields_are_forbidden(column):
    assert forbidden_fields([*FIELDS, column], FARI) == [column]


@pytest.mark.parametrize("column", ["surface_temp_c", "dynamic_load", "interface_errors", "academic_mode"])
def test_harmless_fields_are_not_refused(column):
    """Word-level matching: 'face' must not refuse 'surface', 'mic' not 'dynamic'."""
    assert forbidden_fields([*FIELDS, column], FARI) == []


def test_word_prefix_matching_errs_on_the_safe_side():
    """A word that merely *starts* like a forbidden one is still refused
    (documented: a false refusal costs a rename, a false accept leaks data)."""
    assert forbidden_fields(["framework_version"], FARI) == ["framework_version"]


def test_rover_profile_forbids_nothing():
    assert forbidden_fields([*FIELDS, "camera_frame_id"], DEFAULT_PROFILE) == []
    assert_no_forbidden_fields(list(FIELDS), FARI)


def test_privacy_gate_refuses_the_file_before_anything_is_stored(tmp_path):
    src = tmp_path / "in.csv"
    fields = [*FIELDS, "audio_rms"]
    with src.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({**fari_row(), "audio_rms": "0.3"})
    out = tmp_path / "out"
    with pytest.raises(ContractViolation):
        run_pipeline(src, out, FARI)
    assert not out.exists() or not any(out.rglob("*.*"))


def test_psi_and_availability():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    assert psi(ref, rng.normal(size=5000)) < 0.02
    assert psi(ref, rng.normal(loc=1.5, size=5000)) > 0.25
    gone = np.full(5000, np.nan)
    assert psi(ref, gone) > 1.0                                  # an unavailable feature is a major shift
    import pandas as pd

    table = availability(pd.DataFrame({"a": ref, "b": np.zeros(5000), "c": gone}), ["a", "b", "c"]).set_index("feature")
    assert table.loc["a", "valid_share"] == 1.0 and not table.loc["a", "constant"]
    assert bool(table.loc["b", "constant"])
    assert table.loc["c", "valid_share"] == 0.0
