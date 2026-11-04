"""Leakage-safe splits: unit-grouped x walk-forward with purge/embargo (ADR-0004).

For cut-off c, test unit group g and horizon H:

* train   = rows of units NOT in g with t <= c - H   (their label windows end by c)
* test    = rows of units in g with c <= t < c + span
* val     = the last ``val_days`` of the training period, itself purged by H
            (used for early stopping, calibration and the alert threshold)

The gap between the last training prediction time and the first test
prediction time is therefore >= H, and test robots are never seen in training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: str
    cutoff: pd.Timestamp
    group: int
    test_units: tuple[str, ...]
    fit_idx: np.ndarray     # training rows used to fit the model
    val_idx: np.ndarray     # inner validation rows (still training units, earlier time)
    test_idx: np.ndarray

    @property
    def train_idx(self) -> np.ndarray:
        return np.concatenate([self.fit_idx, self.val_idx])


def unit_groups(
    unit_ids: list[str], n_groups: int, strata: pd.Series | None = None, seed: int = 0
) -> dict[str, int]:
    """Assign units to ``n_groups`` groups, round-robin within strata
    (e.g. number of faults) after a seeded shuffle, so every group gets a
    similar mix of faulty and healthy robots."""
    rng = np.random.default_rng(seed)
    units = list(unit_ids)
    rng.shuffle(units)
    if strata is not None:
        units.sort(key=lambda u: -float(strata.get(u, 0)))
    return {u: i % n_groups for i, u in enumerate(units)}


def walk_forward_folds(
    frame: pd.DataFrame,
    groups: dict[str, int],
    cutoffs: list[pd.Timestamp],
    test_span: pd.Timedelta,
    horizon: pd.Timedelta,
    val_span: pd.Timedelta,
    time_col: str = "t",
    unit_col: str = "unit_id",
) -> list[Fold]:
    t = frame[time_col]
    g = frame[unit_col].map(groups)
    if g.isna().any():
        raise ValueError("Every unit must belong to a group")
    folds = []
    for c in cutoffs:
        train_end = c - horizon
        val_start = train_end - val_span
        for grp in sorted(set(groups.values())):
            in_test_group = (g == grp).to_numpy()
            train_mask = ~in_test_group & (t <= train_end).to_numpy()
            if val_span > pd.Timedelta(0):
                fit_mask = train_mask & (t <= val_start - horizon).to_numpy()
                val_mask = train_mask & (t > val_start).to_numpy()
            else:  # no inner validation: fit on the whole purged training period
                fit_mask = train_mask
                val_mask = np.zeros(len(frame), dtype=bool)
            test_mask = in_test_group & (t >= c).to_numpy() & (t < c + test_span).to_numpy()
            folds.append(Fold(
                fold_id=f"c{c:%m%d}_g{grp}",
                cutoff=c,
                group=grp,
                test_units=tuple(sorted(u for u, k in groups.items() if k == grp)),
                fit_idx=np.flatnonzero(fit_mask),
                val_idx=np.flatnonzero(val_mask),
                test_idx=np.flatnonzero(test_mask),
            ))
    return folds


def random_row_split(n_rows: int, test_fraction: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The WRONG protocol, used only in the leakage demonstration."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    n_test = round(test_fraction * n_rows)
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def check_fold(frame: pd.DataFrame, fold: Fold, horizon: pd.Timedelta, time_col: str = "t",
               unit_col: str = "unit_id") -> dict[str, Any]:
    """Leakage-audit assertions for one fold; raises on violation."""
    train = frame.iloc[fold.train_idx]
    test = frame.iloc[fold.test_idx]
    fit = frame.iloc[fold.fit_idx]
    val = frame.iloc[fold.val_idx]
    shared_units = set(train[unit_col]) & set(test[unit_col])
    if shared_units:
        raise AssertionError(f"{fold.fold_id}: units in both train and test: {sorted(shared_units)}")
    keys_train = set(zip(train[unit_col], train[time_col], strict=True))
    keys_test = set(zip(test[unit_col], test[time_col], strict=True))
    if keys_train & keys_test:
        raise AssertionError(f"{fold.fold_id}: duplicate (unit, t) across train/test")
    gap = test[time_col].min() - train[time_col].max() if len(test) and len(train) else horizon
    if gap < horizon:
        raise AssertionError(f"{fold.fold_id}: embargo {gap} shorter than horizon {horizon}")
    inner_gap = val[time_col].min() - fit[time_col].max() if len(val) and len(fit) else horizon
    if inner_gap < horizon:
        raise AssertionError(f"{fold.fold_id}: inner embargo {inner_gap} shorter than horizon {horizon}")
    return {"fold_id": fold.fold_id, "train_rows": len(train), "test_rows": len(test),
            "embargo_hours": gap / pd.Timedelta(hours=1), "shared_units": 0, "duplicate_keys": 0}
