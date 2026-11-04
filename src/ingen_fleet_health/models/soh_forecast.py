"""Model track 1 -- battery state-of-health forecasting at fixed horizons.

Prediction is made at the end of day d (t = d + 1 day, 00:00 UTC) using the
hourly as-of features at t. Target: true end-of-day SoH at d + h (simulator
ground truth -- in a real fleet this would be a periodic capacity test).

* ``persistence``  -- SoH(d+h) = BMS SoH estimate, mean of the last 24 h
* ``linear``       -- per-unit OLS on the last 14 daily BMS estimates, extrapolated h days
* ``lgbm``         -- LightGBM on the residual target SoH(d+h) - persistence
* ``lstm``         -- LSTM over the last 14 daily feature vectors, same residual target
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

LINEAR_WINDOW_DAYS = 14
SEQ_LEN = 14
DAILY_FEATURES = [
    "bms_soh_mean_24h",
    "lin_level_14d",
    "lin_slope_14d",
    "bms_soh_change_7d",
    "ambient_mean_24h",
    "ambient_mean_7d",
    "efc_24h",
    "efc_7d",
    "efc_cum_nominal",
    "min_soc_24h",
    "last_cycle_dod_pct",
    "r_hat_temp_norm_24h",
    "r_hat_ratio_to_baseline",
    "patrol_fraction_24h",
    "charge_sessions_24h",
]
SOH_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": 8,
}


def _rolling_ols(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing OLS level (fitted value at the last day) and slope per day."""
    n = len(values)
    level = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    for i in range(n):
        y = values[max(0, i - window + 1): i + 1]
        x = np.arange(len(y), dtype=float)
        ok = np.isfinite(y)
        if ok.sum() >= 5:
            b, a = np.polyfit(x[ok], y[ok], 1)
            slope[i] = b
            level[i] = a + b * x[-1]
    return level, slope


def daily_table(features_hourly: pd.DataFrame, battery_truth: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """One row per unit-day with as-of daily features and SoH targets."""
    day_rows = features_hourly[features_hourly["t"].dt.hour == 0].copy()
    day_rows["day"] = (day_rows["t"] - pd.Timedelta(days=1)).dt.normalize()
    keep = ["unit_id", "t", "day", "bms_soh_mean_24h", "ambient_mean_24h", "efc_24h", "efc_cum_nominal",
            "min_soc_24h", "last_cycle_dod_pct", "r_hat_temp_norm_24h", "r_hat_ratio_to_baseline",
            "patrol_fraction_24h", "charge_sessions_24h"]
    daily = day_rows[keep].sort_values(["unit_id", "day"]).reset_index(drop=True)
    parts = []
    for _, g in daily.groupby("unit_id", sort=False):
        g = g.copy()
        level, slope = _rolling_ols(g["bms_soh_mean_24h"].to_numpy(), LINEAR_WINDOW_DAYS)
        g["lin_level_14d"] = level
        g["lin_slope_14d"] = slope
        g["bms_soh_change_7d"] = g["bms_soh_mean_24h"] - g["bms_soh_mean_24h"].shift(7)
        g["ambient_mean_7d"] = g["ambient_mean_24h"].rolling(7, min_periods=1).mean()
        g["efc_7d"] = g["efc_24h"].rolling(7, min_periods=1).sum()
        parts.append(g)
    daily = pd.concat(parts, ignore_index=True)

    truth = battery_truth.copy()
    truth["day"] = pd.to_datetime(truth["date_utc"]).dt.tz_localize("UTC")
    truth = truth.sort_values(["unit_id", "day"])
    for h in horizons:
        truth[f"target_h{h}"] = truth.groupby("unit_id")["soh_true_pct"].shift(-h)
    cols = ["unit_id", "day", "soh_true_pct"] + [f"target_h{h}" for h in horizons]
    return daily.merge(truth[cols], on=["unit_id", "day"], how="inner")


def persistence(frame: pd.DataFrame, h: int) -> np.ndarray:
    return frame["bms_soh_mean_24h"].to_numpy()


def linear_extrapolation(frame: pd.DataFrame, h: int) -> np.ndarray:
    pred = frame["lin_level_14d"] + frame["lin_slope_14d"] * h
    return pred.fillna(frame["bms_soh_mean_24h"]).to_numpy()


def fit_predict_lgbm(train: pd.DataFrame, test: pd.DataFrame, h: int, seed: int = 0) -> np.ndarray:
    y = train[f"target_h{h}"] - train["bms_soh_mean_24h"]
    model = lgb.LGBMRegressor(**SOH_LGBM_PARAMS, random_state=seed)
    model.fit(train[DAILY_FEATURES], y)
    return test["bms_soh_mean_24h"].to_numpy() + model.predict(test[DAILY_FEATURES])


def _sequences(frame: pd.DataFrame, rows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(N, SEQ_LEN, F) windows ending at each requested row, within its unit,
    using only that unit's days <= the row's day (forward-filled, zero-padded)."""
    X = ((frame[DAILY_FEATURES].to_numpy(dtype=float) - mean) / std)
    unit = frame["unit_id"].to_numpy()
    out = np.zeros((len(rows), SEQ_LEN, len(DAILY_FEATURES)), dtype=np.float32)
    for k, r in enumerate(rows):
        lo = r
        while lo > 0 and r - lo + 1 < SEQ_LEN and unit[lo - 1] == unit[r]:
            lo -= 1
        window = pd.DataFrame(X[lo:r + 1]).ffill().fillna(0.0).to_numpy()
        out[k, SEQ_LEN - len(window):] = window
    return out


def fit_predict_lstm(frame: pd.DataFrame, train_rows: np.ndarray, test_rows: np.ndarray, h: int,
                     seed: int = 0, epochs: int = 60) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(8)
    train = frame.iloc[train_rows]
    mean = np.nanmean(train[DAILY_FEATURES].to_numpy(dtype=float), axis=0)   # fitted on training rows only
    std = np.nanstd(train[DAILY_FEATURES].to_numpy(dtype=float), axis=0)
    mean = np.nan_to_num(mean)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    y = (train[f"target_h{h}"] - train["bms_soh_mean_24h"]).to_numpy(dtype=np.float32)
    y_mu, y_sd = float(y.mean()), float(y.std() + 1e-6)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(len(DAILY_FEATURES), 32, batch_first=True)
            self.head = nn.Linear(32, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.head(out[:, -1]).squeeze(-1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    Xtr = torch.from_numpy(_sequences(frame, train_rows, mean, std))
    ytr = torch.from_numpy((y - y_mu) / y_sd)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(_sequences(frame, test_rows, mean, std))).numpy() * y_sd + y_mu
    return frame.iloc[test_rows]["bms_soh_mean_24h"].to_numpy() + pred
