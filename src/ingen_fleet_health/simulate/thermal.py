"""First-order motor thermal model (Newtonian cooling towards a load target).

The same recursion is used twice: by the simulator to generate the true
motor temperature, and by the feature pipeline to compute the *expected*
temperature of a healthy motor from observed ambient and power, so that
the residual isolates abnormal heating (Week 2 physics feature).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def thermal_target(
    ambient_c: np.ndarray, drive_power_w: np.ndarray, heating_c_per_w: float | np.ndarray
) -> np.ndarray:
    return ambient_c + heating_c_per_w * np.maximum(drive_power_w, 0.0)


def first_order_response(
    target: np.ndarray, time_constant_min: float, initial: float
) -> np.ndarray:
    """Causal response T_m = T_{m-1} + (target_m - T_{m-1}) / tau.

    Each output depends only on targets at or before m, so it is safe as an
    as-of feature when ``target`` itself is built from past observations.
    """
    a = 1.0 / time_constant_min
    y, _ = lfilter([a], [1.0, -(1.0 - a)], target, zi=np.array([(1.0 - a) * initial]))
    return np.asarray(y)
