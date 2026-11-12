"""Weather, patrol schedule and route terrain, all as (U, T) minute arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from ingen_fleet_health.simulate.params import FleetParams

MINUTES_PER_DAY = 1440


def site_ambient(
    config: dict[str, Any], days: int, rng: np.random.Generator
) -> np.ndarray:
    """(S, T) continuous ambient temperature per site, deg C.

    Daily means follow an AR(1) around the site climate and are linearly
    interpolated between day centres (v0.1 had hourly steps); a sinusoidal
    diurnal cycle peaks at ``peak_hour_utc``.
    """
    weather = config["weather"]
    phi = float(weather["daily_ar1_phi"])
    sigma = float(weather["daily_noise_std_c"])
    peak = float(weather["peak_hour_utc"])
    total = days * MINUTES_PER_DAY
    hours = np.arange(total) / 60.0
    out = np.empty((len(config["sites"]), total))
    for s, site in enumerate(config["sites"]):
        mean = float(site["ambient_mean_c"])
        dev = np.empty(days + 1)
        dev[0] = rng.normal(0.0, sigma)
        for d in range(1, days + 1):
            dev[d] = phi * dev[d - 1] + rng.normal(0.0, sigma)
        centres = (np.arange(days + 1) + 0.5) * 24.0
        daily = mean + np.interp(hours, centres, dev)
        diurnal = float(site["diurnal_amplitude_c"]) * np.sin(
            2 * np.pi * (hours - peak + 6.0) / 24.0
        )
        out[s] = daily + diurnal
    return out


def patrol_schedule(
    config: dict[str, Any],
    params: FleetParams,
    days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(U, T) bool: the unit is scheduled to be active (before SoC/faults).

    ``duty_cycle.pattern`` selects the duty model: ``shifts`` (default, Aido
    Rover patrol shifts) or ``sessions`` (short interaction sessions with dock
    charging in between, used by the synthetic Fari profile)."""
    duty = config["duty_cycle"]
    if str(duty.get("pattern", "shifts")) == "sessions":
        return session_schedule(config, params, days, rng)
    hour_of_day = (np.arange(MINUTES_PER_DAY) / 60.0)
    single = np.zeros(MINUTES_PER_DAY, dtype=bool)
    for start, end in duty["single_shift_hours_utc"]:
        single |= (hour_of_day >= start) & (hour_of_day < end)
    double = np.zeros(MINUTES_PER_DAY, dtype=bool)
    for start, end in duty["double_shift_hours_utc"]:
        double |= (hour_of_day >= start) & (hour_of_day < end)

    sched = np.empty((params.n_units, days * MINUTES_PER_DAY), dtype=bool)
    for u in range(params.n_units):
        day_pattern = double if params.shift_double[u] else single
        working = rng.random(days) >= float(duty["day_off_probability"])
        sched[u] = (np.tile(day_pattern, (days, 1)) & working[:, None]).ravel()
    return sched


def session_schedule(
    config: dict[str, Any],
    params: FleetParams,
    days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(U, T) bool: many short sessions per day inside a waking-hours window.

    Sessions per day are Poisson; each lasts a uniform number of minutes.
    Overlapping sessions simply merge. ``params.shift_double`` marks heavy
    users, whose session rate is multiplied by ``heavy_user_rate_factor``."""
    duty = config["duty_cycle"]
    w0, w1 = (float(v) * 60.0 for v in duty["session_window_hours_utc"])
    lo, hi = (int(v) for v in duty["session_minutes_range"])
    mean = float(duty["sessions_per_day_mean"])
    factor = float(duty.get("heavy_user_rate_factor", 1.0))
    sched = np.zeros((params.n_units, days * MINUTES_PER_DAY), dtype=bool)
    for u in range(params.n_units):
        rate = mean * (factor if params.shift_double[u] else 1.0)
        working = rng.random(days) >= float(duty["day_off_probability"])
        counts = rng.poisson(rate, days)
        for d in range(days):
            starts = rng.uniform(w0, w1, int(counts[d])).astype(int)
            lengths = rng.integers(lo, hi + 1, int(counts[d]))
            if not working[d]:
                continue
            for s, n in zip(starts, lengths, strict=True):
                begin = d * MINUTES_PER_DAY + int(s)
                sched[u, begin:begin + int(n)] = True
    return sched


def _route_sequence(
    route: dict[str, Any],
    terrain_cfg: dict[str, Any],
    total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    seg_lo, seg_hi = terrain_cfg["segment_minutes"]
    surface_index = np.array(list(terrain_cfg["surfaces"].values()), dtype=float)
    n_seg = total // int(seg_lo) + 2
    lengths = rng.integers(int(seg_lo), int(seg_hi) + 1, n_seg)
    surfaces = rng.choice(surface_index, n_seg, p=route["surface_probs"])
    grades = rng.normal(route["grade_mean"], route["grade_std"], n_seg)
    grade_part = np.clip(
        (grades + float(terrain_cfg["grade_to_factor_offset"]))
        / float(terrain_cfg["grade_to_factor_span"]),
        0.0,
        1.0,
    )
    factor = np.clip(0.5 * surfaces + 0.5 * grade_part, 0.0, 1.0)
    return np.repeat(factor, lengths)[:total]


def terrain_factor(
    config: dict[str, Any],
    params: FleetParams,
    days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(U, T) dimensionless terrain load factor in [0, 1].

    Built from random route segments (surface type x grade); units that are
    reassigned switch route profile at ``route_change_day``.
    """
    terrain_cfg = config["terrain"]
    total = days * MINUTES_PER_DAY
    if not bool(terrain_cfg.get("enabled", True)):
        # platforms without terrain (indoor): the field is written as a constant 0
        return np.zeros((params.n_units, total))
    out = np.empty((params.n_units, total))
    for u in range(params.n_units):
        first = _route_sequence(terrain_cfg["routes"][params.route_initial[u]], terrain_cfg, total, rng)
        change = params.route_change_day[u]
        if np.isfinite(change):
            second = _route_sequence(terrain_cfg["routes"][params.route_after[u]], terrain_cfg, total, rng)
            cut = int(change * MINUTES_PER_DAY)
            first[cut:] = second[cut:]
        out[u] = first
    return out
