"""Battery ageing relations (capacity fade, resistance growth, OCV)."""

from __future__ import annotations

import numpy as np


def arrhenius_factor(temp_c: np.ndarray, doubling_c: float) -> np.ndarray:
    """Ageing-rate multiplier relative to 25 C (doubles every ``doubling_c``)."""
    return np.power(2.0, (temp_c - 25.0) / doubling_c)


def resistance_ohm(
    base_ohm: np.ndarray,
    capacity_loss_pct: np.ndarray,
    ambient_c: np.ndarray,
    cell_severity: np.ndarray,
    growth_per_pct: float,
    cold_per_c: float,
    cell_max_increase: float,
) -> np.ndarray:
    return (
        base_ohm
        * (1.0 + growth_per_pct * capacity_loss_pct)
        * (1.0 + cold_per_c * np.maximum(0.0, 25.0 - ambient_c))
        * (1.0 + cell_max_increase * cell_severity)
    )


def terminal_voltage(
    nominal_v: float, soc_pct: np.ndarray, current_a: np.ndarray, r_ohm: np.ndarray, ocv_slope: float
) -> np.ndarray:
    """Linearised OCV(SoC) minus the ohmic drop I*R (I > 0 = discharge)."""
    return nominal_v + (soc_pct - 50.0) * ocv_slope - current_a * r_ohm
