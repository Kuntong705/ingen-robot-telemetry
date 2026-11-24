"""SHAP attributions for the LightGBM early-warning model.

Tree SHAP is exact for tree ensembles. Values are in the model's raw margin
(log-odds of "qualifying onset within 72 h") *before* the calibration map;
calibration is monotone, so the sign and ranking of contributions carry over
to the calibrated risk shown to the operator, but the values are not
probability points.

Every feature belongs to one *concept* (what a technician would call it);
summaries and the faithfulness check work on concepts, not raw column names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap

from ingen_fleet_health.features.build import ROLLING_CHANNELS
from ingen_fleet_health.text import TAXONOMY

CONCEPT_LABEL = {
    "vibration": "drivetrain vibration",
    "thermal": "motor temperature / cooling",
    "battery": "battery",
    "energy": "energy use",
    "sensor_drift": "temperature-sensor offset",
    "environment": "environment (ambient, terrain)",
    "data_quality": "telemetry gaps",
    "usage": "duty pattern",
    "maintenance_history": "time since the last repair",
    "operator_notes": "operator notes",
    "user_error": "user or operational error",
    "platform": "robot platform",
}
CLASS_CONCEPT = {
    "battery": "battery",
    "drivetrain": "vibration",
    "thermal": "thermal",
    "sensor": "sensor_drift",
    "connectivity": "data_quality",
    "user_or_operational_error": "user_error",
    "no_fault_found": None,
}
_CHANNEL_PHRASE = {
    "motor_temp": ("motor temperature", "deg C", "thermal"),
    "vibration": ("vibration", "g", "vibration"),
    "power": ("electrical power", "W", "energy"),
    "voltage": ("battery voltage", "V", "battery"),
    "soc": ("battery state of charge", "%", "battery"),
    "ambient": ("ambient temperature", "deg C", "environment"),
    "terrain": ("terrain load factor", "", "environment"),
    "r_bms": ("BMS-reported internal resistance", "ohm", "battery"),
}
_STAT_PHRASE = {"mean": "mean", "std": "variability (std)", "min": "minimum", "max": "maximum", "slope": "trend"}
# (plain-language phrase, unit, concept)
_PHYSICS_PHRASE: dict[str, tuple[str, str, str]] = {
    "missing_fraction_24h": ("share of telemetry missing in the last 24 h", "", "data_quality"),
    "r_hat_24h": ("estimated battery internal resistance (24 h)", "ohm", "battery"),
    "r_hat_temp_norm_24h": ("temperature-corrected battery internal resistance (24 h)", "ohm", "battery"),
    "r_hat_ratio_to_baseline": ("battery internal resistance relative to the unit's own baseline", "x baseline", "battery"),
    "energy_per_km_24h": ("energy per kilometre (24 h)", "Wh/km", "energy"),
    "energy_ratio_to_expected_24h": ("energy use relative to what terrain and ambient predict (24 h)", "x expected", "energy"),
    "energy_ratio_to_baseline": ("terrain-normalised energy use relative to the unit's own baseline", "x baseline", "energy"),
    "motor_temp_expected_c": ("motor temperature expected from load and ambient", "deg C", "thermal"),
    "motor_temp_resid_c": ("motor temperature above the load/ambient model", "deg C", "thermal"),
    "motor_temp_resid_mean_6h": ("motor temperature above the load/ambient model (6 h mean)", "deg C", "thermal"),
    "motor_temp_resid_patrol_mean_24h": ("motor temperature above the load/ambient model while active (24 h mean)", "deg C", "thermal"),
    "motor_temp_resid_dock_mean_24h": ("docked motor-temperature offset from ambient (24 h mean)", "deg C", "sensor_drift"),
    "thermal_gain_ratio_24h": ("motor heating per watt relative to the nominal model (24 h)", "x nominal", "thermal"),
    "motor_temp_resid_slope_24h": ("trend of the motor-temperature excess (24 h)", "deg C/h", "thermal"),
    "vib_norm_mean_6h": ("terrain-normalised vibration (6 h mean)", "g", "vibration"),
    "vib_norm_mean_24h": ("terrain-normalised vibration (24 h mean)", "g", "vibration"),
    "vib_norm_slope_24h": ("trend of terrain-normalised vibration (24 h)", "g/h", "vibration"),
    "vib_norm_change_72h": ("change in terrain-normalised vibration over 72 h", "g", "vibration"),
    "vib_norm_ratio_to_baseline": ("terrain-normalised vibration relative to the unit's own baseline", "x baseline", "vibration"),
    "charge_sessions_cum": ("cumulative charge sessions", "", "usage"),
    "charge_sessions_24h": ("charge sessions in the last 24 h", "", "usage"),
    "last_cycle_dod_pct": ("depth of the last discharge cycle", "%", "battery"),
    "min_soc_24h": ("lowest state of charge in the last 24 h", "%", "battery"),
    "efc_cum_nominal": ("cumulative equivalent full cycles", "", "usage"),
    "efc_24h": ("equivalent full cycles in the last 24 h", "", "usage"),
    "bms_soh_last": ("latest BMS state-of-health estimate", "%", "battery"),
    "bms_soh_mean_24h": ("BMS state-of-health estimate (24 h mean)", "%", "battery"),
    "terrain_patrol_mean_24h": ("mean terrain load while active (24 h)", "", "environment"),
    "patrol_fraction_24h": ("share of the last 24 h spent active", "", "usage"),
    "odometer_km": ("odometer", "km", "usage"),
    "hours_since_repair": ("hours since the last completed repair", "h", "maintenance_history"),
    "hour_of_day": ("hour of day", "h UTC", "usage"),
    # Week 4 pooled production model: platform indicator (Aido Rover = 0, Fari = 1)
    "platform_fari": ("platform indicator (Fari = 1, Aido Rover = 0)", "", "platform"),
    "txt_fault_max_72h": ("strongest fault wording in operator notes of the last 72 h (classifier probability)", "",
                          "operator_notes"),
    "txt_note_count_72h": ("operator notes in the last 72 h", "", "operator_notes"),
    "txt_note_count_7d": ("operator notes in the last 7 days", "", "operator_notes"),
    "txt_service_count_7d": ("service tickets in the last 7 days", "", "maintenance_history"),
    "txt_hours_since_note": ("hours since the latest operator note", "h", "operator_notes"),
}


@dataclass(frozen=True)
class FeatureInfo:
    name: str
    phrase: str
    unit: str
    concept: str
    note_class: str | None = None   # for txt_decay_<class>: which ticket category


def feature_info(name: str) -> FeatureInfo:
    """Plain-language phrase, unit and concept of a model feature."""
    if name in _PHYSICS_PHRASE:
        phrase, unit, concept = _PHYSICS_PHRASE[name]
        return FeatureInfo(name, phrase, unit, concept)
    if name.startswith("txt_decay_"):
        cls = name.removeprefix("txt_decay_")
        if cls in TAXONOMY:
            return FeatureInfo(name, f"recent operator notes read as '{cls.replace('_', ' ')}' (time-decayed weight)",
                               "", "operator_notes", cls)
    for prefix in sorted(ROLLING_CHANNELS.values(), key=len, reverse=True):
        if name.startswith(prefix + "_"):
            rest = name[len(prefix) + 1:]
            stat, _, window = rest.partition("_")
            if stat in _STAT_PHRASE and window:
                channel, unit, concept = _CHANNEL_PHRASE[prefix]
                unit = f"{unit}/h" if stat == "slope" and unit else unit
                return FeatureInfo(name, f"{window.replace('h', ' h')} {_STAT_PHRASE[stat]} of {channel}", unit, concept)
    raise KeyError(f"No description for feature {name!r}; add it to explain.shap_utils")


def tree_shap(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact tree SHAP values (n, f) and base value, in raw-margin (log-odds) units."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):                 # older API: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:                         # (n, f, classes)
        values = values[:, :, 1]
    base = np.ravel(np.asarray(explainer.expected_value))
    return values, float(base[-1])


def global_importance(values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per feature and per concept."""
    frame = pd.DataFrame({"feature": columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
    frame["concept"] = [feature_info(c).concept for c in columns]
    return frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def concept_importance(importance: pd.DataFrame) -> pd.Series:
    out = importance.groupby("concept")["mean_abs_shap"].sum().sort_values(ascending=False)
    return out / out.sum()
