"""External method check on NASA C-MAPSS FD001 (turbofan run-to-failure).

The data is NOT downloaded by this code. Place the public files
``train_FD001.txt``, ``test_FD001.txt`` and ``RUL_FD001.txt`` (NASA Prognostics
Data Repository, used under its stated terms) in ``data/external/cmapss/`` and run::

    python -m ingen_fleet_health.benchmarks.cmapss --data-dir data/external/cmapss

The same discipline as the fleet models applies: features are trailing
windows within each engine, scalers are fitted on training engines only,
and the official test split (engines never seen in training, truncated
before failure) is the only reported number.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from ingen_fleet_health.evaluation.metrics import rmse

COLUMNS = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s{i}" for i in range(1, 22)]
RUL_CAP = 125
WINDOW = 30
# Reference values reported in the literature for FD001 test RMSE. They must be
# checked against the original papers before being quoted in a deliverable.
PUBLISHED_REFERENCE_RMSE = {
    "CNN (Babu et al., 2016)": 18.45,
    "LSTM (Zheng et al., 2017)": 16.14,
    "DCNN (Li et al., 2018)": 12.61,
}


def read_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    frame = frame.iloc[:, : len(COLUMNS)]
    frame.columns = COLUMNS
    return frame


def add_rul(train: pd.DataFrame) -> pd.DataFrame:
    life = train.groupby("unit")["cycle"].transform("max")
    return train.assign(rul=np.minimum(life - train["cycle"], RUL_CAP))


def informative_sensors(train: pd.DataFrame) -> list[str]:
    sensors = [c for c in COLUMNS if c.startswith("s")]
    return [c for c in sensors if train[c].std() > 1e-4]


def window_features(frame: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Trailing per-engine mean / std / slope over the last WINDOW cycles."""
    parts = []
    for _, g in frame.groupby("unit", sort=False):
        g = g.sort_values("cycle")
        feats = {"unit": g["unit"].to_numpy(), "cycle": g["cycle"].to_numpy()}
        x = g["cycle"].astype(float)
        for s in sensors:
            roll = g[s].rolling(WINDOW, min_periods=3)
            feats[f"{s}_mean"] = roll.mean().to_numpy()
            feats[f"{s}_std"] = roll.std().to_numpy()
            cov = (g[s] * x).rolling(WINDOW, min_periods=3).mean() - roll.mean() * x.rolling(WINDOW, min_periods=3).mean()
            var = (x * x).rolling(WINDOW, min_periods=3).mean() - x.rolling(WINDOW, min_periods=3).mean() ** 2
            feats[f"{s}_slope"] = (cov / var.where(var > 0)).to_numpy()
            feats[f"{s}_last"] = g[s].to_numpy()
        parts.append(pd.DataFrame(feats))
    return pd.concat(parts, ignore_index=True)


def nasa_score(y: np.ndarray, p: np.ndarray) -> float:
    d = np.asarray(p) - np.asarray(y)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def run_benchmark(train: pd.DataFrame, test: pd.DataFrame, test_rul: np.ndarray, seed: int = 0) -> dict[str, Any]:
    train = add_rul(train)
    sensors = informative_sensors(train)
    ftr = window_features(train, sensors).merge(train[["unit", "cycle", "rul"]], on=["unit", "cycle"])
    fte_all = window_features(test, sensors)
    fte = fte_all.loc[fte_all.groupby("unit")["cycle"].idxmax()].sort_values("unit")
    y_test = np.minimum(np.asarray(test_rul, dtype=float), RUL_CAP)
    feat_cols = [c for c in ftr.columns if c not in ("unit", "cycle", "rul")]

    # baseline 1: constant -- mean capped RUL of training rows
    constant = np.full(len(fte), ftr["rul"].mean())
    # baseline 2: age-based linear degradation -- median training life minus current cycle
    median_life = train.groupby("unit")["cycle"].max().median()
    linear = np.clip(median_life - fte["cycle"].to_numpy(), 0, RUL_CAP)

    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=seed)
    model.fit(ftr[feat_cols], ftr["rul"])
    gbm = np.clip(model.predict(fte[feat_cols]), 0, RUL_CAP)

    preds = {"constant_mean": constant, "age_linear": linear, "lgbm": gbm}
    return {
        "test_engines": len(fte),
        "sensors_used": sensors,
        "rul_cap": RUL_CAP,
        "rmse": {k: rmse(y_test, v) for k, v in preds.items()},
        "nasa_score": {k: nasa_score(y_test, v) for k, v in preds.items()},
        "published_reference_rmse_unverified": PUBLISHED_REFERENCE_RMSE,
    }


def synthetic_cmapss(n_units: int = 20, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Small C-MAPSS-shaped data set for tests (exponential degradation + noise)."""
    rng = np.random.default_rng(seed)

    def engine(unit: int, truncate: bool) -> tuple[pd.DataFrame, int]:
        life = int(rng.integers(130, 260))
        cycles = np.arange(1, life + 1)
        wear = np.exp((cycles - life) / 40.0)
        rows = {"unit": unit, "cycle": cycles, "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0}
        for i in range(1, 22):
            gain = 0.0 if i in (1, 5, 10, 16, 18, 19) else rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
            rows[f"s{i}"] = 100 + i + gain * wear + rng.normal(0, 0.05, life)
        frame = pd.DataFrame(rows)
        if truncate:
            cut = int(rng.integers(30, life - 5))
            return frame[frame["cycle"] <= cut], life - cut
        return frame, 0

    train = pd.concat([engine(u, False)[0] for u in range(1, n_units + 1)], ignore_index=True)
    tests = [engine(u, True) for u in range(1, n_units // 2 + 1)]
    test = pd.concat([t[0] for t in tests], ignore_index=True)
    return train, test, np.array([t[1] for t in tests])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    train = read_split(args.data_dir / "train_FD001.txt")
    test = read_split(args.data_dir / "test_FD001.txt")
    rul = pd.read_csv(args.data_dir / "RUL_FD001.txt", header=None).iloc[:, 0].to_numpy()
    res = run_benchmark(train, test, rul)
    from ingen_fleet_health.experiments.common import write_json
    from ingen_fleet_health.tracking import log_dict_json, log_flat_metrics, tracked_run

    with tracked_run("w02_external_cmapss", "fd001", {"track": "external_benchmark"}) as run_:
        log_flat_metrics(res["rmse"], prefix="rmse.")
        log_dict_json(res, "cmapss_results.json")
        res["mlflow_run_id"] = run_.info.run_id
    write_json(res, "cmapss_fd001_results.json")
    print(res["rmse"])


if __name__ == "__main__":
    main()
