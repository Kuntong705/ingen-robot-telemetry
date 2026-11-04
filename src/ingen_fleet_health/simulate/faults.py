"""P-F fault timelines: initiation -> qualifying onset -> failure -> repair.

A fault starts degrading silently at *initiation*. Its severity follows a
ramp g(x), x = (t - initiation) / (failure - initiation), g(0)=0, g(1)=1.
The *qualifying onset* (the label event, ADR-003 rev1) is the first time
severity reaches ``qualifying_severity``. The precursor period between
initiation and onset is what makes a 72 h early warning learnable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

MINUTES_PER_DAY = 1440


def ramp(shape: str, x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    if shape == "linear":
        return x
    if shape == "exponential":
        return (np.exp(3.0 * x) - 1.0) / (math.exp(3.0) - 1.0)
    if shape == "sigmoid":
        lo = 1.0 / (1.0 + math.exp(5.0))
        hi = 1.0 / (1.0 + math.exp(-5.0))
        return (1.0 / (1.0 + np.exp(-10.0 * (x - 0.5))) - lo) / (hi - lo)
    raise ValueError(f"Unknown ramp shape: {shape}")


def ramp_inverse(shape: str, level: float) -> float:
    """x at which ramp(shape, x) == level (numerical, monotone ramps)."""
    grid = np.linspace(0.0, 1.0, 100_001)
    return float(grid[np.searchsorted(ramp(shape, grid), level)])


@dataclass(frozen=True)
class FaultEvent:
    fault_id: str
    unit_index: int
    fault_type: str
    ramp_shape: str
    initiation_min: int
    onset_min: int
    failure_min: int
    repair_start_min: int
    repair_end_min: int

    @property
    def precursor_days(self) -> float:
        return (self.onset_min - self.initiation_min) / MINUTES_PER_DAY


def sample_fault_events(
    config: dict[str, Any],
    n_units: int,
    total_minutes: int,
    rng: np.random.Generator,
) -> list[FaultEvent]:
    cfg = config["faults"]
    types = list(cfg["types"])
    shapes = list(cfg["ramp_shapes"])
    q = float(cfg["qualifying_severity"])
    init_lo, init_hi = (float(v) * total_minutes for v in cfg["initiation_fraction_range"])
    fault_mean = float(cfg["faults_per_unit_per_60_days"]) * total_minutes / (60 * MINUTES_PER_DAY)
    pre_lo, pre_hi = (float(v) for v in cfg["precursor_days_range"])
    det_lo, det_hi = (float(v) for v in cfg["detection_delay_days_range"])
    rep_lo, rep_hi = (float(v) for v in cfg["repair_hours_range"])
    gap = float(cfg["min_gap_after_repair_days"]) * MINUTES_PER_DAY
    fail_to_repair = float(cfg["failure_to_repair_days"]) * MINUTES_PER_DAY

    events: list[FaultEvent] = []
    for unit in range(n_units):
        n_faults = int(rng.poisson(fault_mean))
        earliest = init_lo
        for _ in range(n_faults):
            if earliest >= init_hi:
                break
            initiation = rng.uniform(earliest, init_hi)
            shape = str(rng.choice(shapes))
            fault_type = str(rng.choice(types))
            precursor = rng.uniform(pre_lo, pre_hi) * MINUTES_PER_DAY
            x_q = ramp_inverse(shape, q)
            duration = precursor / x_q
            onset = initiation + precursor
            failure = initiation + duration
            repair_start = min(
                onset + rng.uniform(det_lo, det_hi) * MINUTES_PER_DAY,
                failure + fail_to_repair,
            )
            repair_end = repair_start + rng.uniform(rep_lo, rep_hi) * 60
            events.append(FaultEvent(
                fault_id=f"F{len(events) + 1:04d}",
                unit_index=unit,
                fault_type=fault_type,
                ramp_shape=shape,
                initiation_min=int(initiation),
                onset_min=int(onset),
                failure_min=int(failure),
                repair_start_min=int(repair_start),
                repair_end_min=int(repair_end),
            ))
            earliest = repair_end + gap
    return events


def severity_matrix(
    events: list[FaultEvent],
    fault_type: str,
    n_units: int,
    total_minutes: int,
) -> np.ndarray:
    """(U, T) severity in [0, 1] for one fault type; zero after repair start."""
    sev = np.zeros((n_units, total_minutes), dtype=np.float64)
    minutes = np.arange(total_minutes)
    for ev in events:
        if ev.fault_type != fault_type:
            continue
        active = (minutes >= ev.initiation_min) & (minutes < ev.repair_start_min)
        x = (minutes[active] - ev.initiation_min) / (ev.failure_min - ev.initiation_min)
        sev[ev.unit_index, active] = np.maximum(sev[ev.unit_index, active], ramp(ev.ramp_shape, x))
    return sev


def downtime_mask(events: list[FaultEvent], n_units: int, total_minutes: int) -> np.ndarray:
    """(U, T) True while a unit is failed-but-unrepaired or in repair (no patrol)."""
    down = np.zeros((n_units, total_minutes), dtype=bool)
    for ev in events:
        start = min(ev.failure_min, ev.repair_start_min)
        down[ev.unit_index, max(start, 0):min(ev.repair_end_min, total_minutes)] = True
    return down
