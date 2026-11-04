"""Per-unit parameter sampling: the source of unit-to-unit heterogeneity.

v0.1 gave every robot identical parameters, so all units had the same
SoH trajectory. Here every physical coefficient gets its own seeded draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FleetParams:
    unit_ids: list[str]
    site_index: np.ndarray          # (U,) int
    shift_double: np.ndarray        # (U,) bool
    route_initial: list[str]
    route_after: list[str]
    route_change_day: np.ndarray    # (U,) float, inf if never
    capacity_wh: np.ndarray
    cycle_fade_pct_per_efc: np.ndarray
    base_resistance_ohm: np.ndarray
    heating_coefficient: np.ndarray
    thermal_time_constant_min: np.ndarray
    vibration_baseline_g: np.ndarray
    baseline_power_w: np.ndarray
    bms_soh_bias_pct: np.ndarray

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    def to_records(self, site_names: list[str]) -> list[dict[str, Any]]:
        rows = []
        for i, unit in enumerate(self.unit_ids):
            change = self.route_change_day[i]
            rows.append({
                "unit_id": unit,
                "site": site_names[int(self.site_index[i])],
                "shift_pattern": "double" if self.shift_double[i] else "single",
                "route_initial": self.route_initial[i],
                "route_after_change": self.route_after[i],
                "route_change_day": float(change) if np.isfinite(change) else None,
                "capacity_wh": float(self.capacity_wh[i]),
                "cycle_fade_pct_per_efc": float(self.cycle_fade_pct_per_efc[i]),
                "base_resistance_ohm": float(self.base_resistance_ohm[i]),
                "heating_coefficient_c_per_w": float(self.heating_coefficient[i]),
                "thermal_time_constant_min": float(self.thermal_time_constant_min[i]),
                "vibration_baseline_g": float(self.vibration_baseline_g[i]),
                "baseline_power_w": float(self.baseline_power_w[i]),
                "bms_soh_bias_pct": float(self.bms_soh_bias_pct[i]),
            })
        return rows


def sample_fleet_params(
    config: dict[str, Any], n_units: int, rng: np.random.Generator
) -> FleetParams:
    var = config["unit_variation"]
    battery = config["battery"]
    thermal = config["thermal"]
    vib = config["vibration"]
    duty = config["duty_cycle"]
    terrain = config["terrain"]

    def factor(std: float) -> np.ndarray:
        return np.clip(rng.normal(1.0, std, n_units), 1 - 3 * std, 1 + 3 * std)

    route_names = list(terrain["routes"])
    route_initial = [str(r) for r in rng.choice(route_names, n_units)]
    route_after = list(route_initial)
    change_day = np.full(n_units, np.inf)
    lo, hi = terrain["route_change_day_range"]
    for i in range(n_units):
        if rng.random() < float(terrain["route_change_fraction"]):
            others = [r for r in route_names if r != route_initial[i]]
            route_after[i] = str(rng.choice(others))
            change_day[i] = rng.uniform(lo, hi)

    fade = float(battery["cycle_fade_pct_per_efc"]) * np.exp(
        rng.normal(0.0, float(var["fade_rate_log_std"]), n_units)
    )

    return FleetParams(
        unit_ids=[f"Rover_{i + 1:03d}" for i in range(n_units)],
        site_index=rng.integers(0, len(config["sites"]), n_units),
        shift_double=rng.random(n_units) < float(duty["double_shift_fraction"]),
        route_initial=route_initial,
        route_after=route_after,
        route_change_day=change_day,
        capacity_wh=float(battery["nominal_capacity_wh"]) * factor(float(var["capacity"])),
        cycle_fade_pct_per_efc=fade,
        base_resistance_ohm=float(battery["base_internal_resistance_ohm"])
        * factor(float(var["base_resistance"])),
        heating_coefficient=float(thermal["load_heating_coefficient_c_per_w"])
        * factor(float(var["heating_coefficient"])),
        thermal_time_constant_min=float(thermal["response_time_minutes"])
        * factor(float(var["thermal_time_constant"])),
        vibration_baseline_g=float(vib["patrol_baseline_g"])
        * factor(float(var["vibration_baseline"])),
        baseline_power_w=float(duty["baseline_patrol_power_w"])
        * factor(float(var["baseline_power"])),
        bms_soh_bias_pct=rng.normal(0.0, float(battery["bms_soh_bias_std_pct"]), n_units),
    )
