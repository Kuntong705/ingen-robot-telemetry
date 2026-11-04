"""External method check on NASA C-MAPSS FD001 (turbofan run-to-failure).

Data: "Turbofan Engine Degradation Simulation Data Set" (A. Saxena and K. Goebel,
2008, NASA Prognostics Data Repository, NASA Ames Research Center), downloaded
from the link on the NASA PCoE data-set repository page and kept git-ignored in
``data/external/cmapss/``. Run::

    python -m ingen_fleet_health.benchmarks.cmapss --data-dir data/external/cmapss

What this benchmark is, and is not, comparable to
-------------------------------------------------
* Task: remaining useful life (RUL, in operating *cycles*) of each official
  test engine at its last observed cycle. That is one point estimate per engine,
  and the engine has run to failure with no repairs.
* The fleet SoH track instead forecasts a health *level* (SoH, in percentage
  points) at fixed 1/7/14-day horizons. RMSE in cycles and MAE in percentage
  points measure different things, so the two are never put in one table.
* What transfers is the method: engine-grouped evaluation (test engines never
  seen in training), baselines first, the same model families (LightGBM on
  trailing-window features, an LSTM on raw windows), and fixed
  hyper-parameters (no tuning on the official test set).

The literature trains on a piecewise-linear target capped at 125-130 cycles.
Results here are reported against the capped truth (the usual convention in
the papers below) and against the raw official truth.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from ingen_fleet_health.evaluation.metrics import rmse

COLUMNS = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s{i}" for i in range(1, 22)]
RUL_CAP = 125
WINDOW = 30
LGBM_PARAMS: dict[str, Any] = {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 30,
                               "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "verbose": -1}
# Published FD001 test RMSE (cycles), cross-checked in two secondary sources
# (arXiv:2010.03207 Table 7; arXiv:2603.00745 Table V). The original papers
# were not accessed, so cite these with the original references, not as
# verified here.
PUBLISHED_REFERENCE_RMSE = {
    "MLP (Babu et al., 2016)": {"rmse": 37.56, "source_check": "secondary: 37.6"},
    "SVR (Babu et al., 2016)": {"rmse": 20.96, "source_check": "secondary: 21.0"},
    "CNN (Babu et al., 2016)": {"rmse": 18.45, "source_check": "secondary: 18.4 / 18.44"},
    "Deep LSTM (Zheng et al., 2017)": {"rmse": 16.14, "source_check": "secondary: 16.1 / 16.14"},
    "DCNN (Li et al., 2018)": {"rmse": 12.61, "source_check": "secondary: about 12.6 (table layout ambiguous); unverified"},
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
    """Sensors that vary in the *training* engines (fitted on train only)."""
    sensors = [f"s{i}" for i in range(1, 22)]  # sensor channels only, not operational settings
    return [c for c in sensors if train[c].std() > 1e-4]


def window_features(frame: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Trailing per-engine mean / std / slope / last value over the last WINDOW cycles."""
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
    """PHM08 asymmetric score (sum over engines); late predictions cost more."""
    d = np.asarray(p) - np.asarray(y)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def fit_lgbm(X: pd.DataFrame, y: pd.Series, seed: int) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=seed)
    model.fit(X, y)
    return model


def _windows(frame: pd.DataFrame, sensors: list[str], mean: np.ndarray, std: np.ndarray,
             last_only: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N, WINDOW, F) standardised sensor windows ending at each cycle (zero-padded at the start)."""
    xs, ys, units = [], [], []
    for unit, g in frame.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        arr = ((g[sensors].to_numpy(dtype=float) - mean) / std).astype(np.float32)
        ends = [len(g) - 1] if last_only else range(len(g))
        for e in ends:
            w = arr[max(0, e - WINDOW + 1): e + 1]
            pad = np.zeros((WINDOW - len(w), arr.shape[1]), dtype=np.float32)
            xs.append(np.vstack([pad, w]))
            ys.append(float(g["rul"].iloc[e]) if "rul" in g else np.nan)
            units.append(unit)
    return np.stack(xs), np.asarray(ys, dtype=np.float32), np.asarray(units)


def fit_predict_lstm(train: pd.DataFrame, test: pd.DataFrame, sensors: list[str], seed: int = 0,
                     max_epochs: int = 40, patience: int = 5) -> tuple[np.ndarray, dict[str, Any]]:
    """LSTM on raw sensor windows; scaler fitted on training engines only, and early
    stopping on 10 % of *training engines* held out (never on the official test set)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(8)
    rng = np.random.default_rng(seed)
    units = train["unit"].unique()
    val_units = set(rng.choice(units, max(1, len(units) // 10), replace=False))
    fit = train[~train["unit"].isin(val_units)]
    val = train[train["unit"].isin(val_units)]
    mean = fit[sensors].mean().to_numpy()
    std = fit[sensors].std().replace(0, 1).to_numpy()
    Xf, yf, _ = _windows(fit, sensors, mean, std, last_only=False)
    Xv, yv, _ = _windows(val, sensors, mean, std, last_only=False)
    Xt, _, _ = _windows(test, sensors, mean, std, last_only=True)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(len(sensors), 64, batch_first=True)
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.head(out[:, -1]).squeeze(-1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    Xf_t, yf_t = torch.from_numpy(Xf), torch.from_numpy(yf / RUL_CAP)
    Xv_t, yv_t = torch.from_numpy(Xv), torch.from_numpy(yv / RUL_CAP)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad, epochs = float("inf"), None, 0, 0
    for _ in range(max_epochs):
        epochs += 1
        net.train()
        perm = torch.randperm(len(Xf_t), generator=gen)
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            nn.functional.mse_loss(net(Xf_t[idx]), yf_t[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(net(Xv_t), yv_t))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(Xt)).numpy() * RUL_CAP
    info = {"epochs_run": epochs, "val_engines": len(val_units), "val_rmse_capped": float(np.sqrt(best) * RUL_CAP)}
    return np.clip(pred, 0, RUL_CAP), info


def internal_leakage_check(ftr: pd.DataFrame, feat_cols: list[str], seed: int) -> dict[str, float]:
    """Same LightGBM, 5-fold CV on the training engines: random rows vs whole engines held out.
    Scored on all cycles of the held-out rows, so it is a *different* quantity from the
    official last-cycle test RMSE and is only used to compare the two split schemes."""
    out = {}
    for name, splitter, groups in (("random_row_cv_rmse", KFold(5, shuffle=True, random_state=seed), None),
                                   ("engine_grouped_cv_rmse", GroupKFold(5), ftr["unit"])):
        err = np.zeros(len(ftr))
        for tr, te in splitter.split(ftr, groups=groups):
            model = fit_lgbm(ftr.iloc[tr][feat_cols], ftr.iloc[tr]["rul"], seed)
            err[te] = model.predict(ftr.iloc[te][feat_cols]) - ftr.iloc[te]["rul"].to_numpy()
        out[name] = float(np.sqrt(np.mean(err ** 2)))
    return out


def run_benchmark(train: pd.DataFrame, test: pd.DataFrame, test_rul: np.ndarray, seed: int = 0,
                  with_lstm: bool = True) -> dict[str, Any]:
    train = add_rul(train)
    sensors = informative_sensors(train)
    ftr = window_features(train, sensors).merge(train[["unit", "cycle", "rul"]], on=["unit", "cycle"])
    fte_all = window_features(test, sensors)
    fte = fte_all.loc[fte_all.groupby("unit")["cycle"].idxmax()].sort_values("unit")
    y_raw = np.asarray(test_rul, dtype=float)
    y_cap = np.minimum(y_raw, RUL_CAP)
    feat_cols = [c for c in ftr.columns if c not in ("unit", "cycle", "rul")]

    # baseline 1: constant -- mean capped RUL of training rows
    constant = np.full(len(fte), ftr["rul"].mean())
    # baseline 2: age-based linear degradation -- median training life minus current cycle
    median_life = train.groupby("unit")["cycle"].max().median()
    linear = np.clip(median_life - fte["cycle"].to_numpy(), 0, RUL_CAP)
    gbm = np.clip(fit_lgbm(ftr[feat_cols], ftr["rul"], seed).predict(fte[feat_cols]), 0, RUL_CAP)

    preds: dict[str, np.ndarray] = {"constant_mean": constant, "age_linear": linear, "lgbm": gbm}
    lstm_info: dict[str, Any] = {}
    if with_lstm:
        preds["lstm"], lstm_info = fit_predict_lstm(train, test.assign(rul=np.nan), sensors, seed)

    return {
        "task": "RUL in cycles at the last observed cycle of each official test engine (run-to-failure, no repair)",
        "not_comparable_with": "fleet SoH track (percentage points at 1/7/14-day horizons)",
        "train_engines": int(train["unit"].nunique()),
        "test_engines": len(fte),
        "sensors_used": sensors,
        "rul_cap": RUL_CAP,
        "window_cycles": WINDOW,
        "rmse_vs_capped_truth": {k: rmse(y_cap, v) for k, v in preds.items()},
        "rmse_vs_raw_truth": {k: rmse(y_raw, v) for k, v in preds.items()},
        "nasa_score_vs_capped_truth": {k: nasa_score(y_cap, v) for k, v in preds.items()},
        "test_engines_with_true_rul_above_cap": int((y_raw > RUL_CAP).sum()),
        "lstm_training": lstm_info,
        "internal_split_check": internal_leakage_check(ftr, feat_cols, seed),
        "published_reference_rmse": PUBLISHED_REFERENCE_RMSE,
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    files = {name: args.data_dir / name for name in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")}
    train = read_split(files["train_FD001.txt"])
    test = read_split(files["test_FD001.txt"])
    rul = pd.read_csv(files["RUL_FD001.txt"], header=None).iloc[:, 0].to_numpy()
    res = run_benchmark(train, test, rul, args.seeds[0])
    # seed robustness for the learned models (the primary table above uses the first seed)
    spread: dict[str, list[float]] = {"lgbm": [res["rmse_vs_capped_truth"]["lgbm"]],
                                      "lstm": [res["rmse_vs_capped_truth"]["lstm"]]}
    for seed in args.seeds[1:]:
        train_r = add_rul(train)
        sensors = informative_sensors(train_r)
        ftr = window_features(train_r, sensors).merge(train_r[["unit", "cycle", "rul"]], on=["unit", "cycle"])
        fte_all = window_features(test, sensors)
        fte = fte_all.loc[fte_all.groupby("unit")["cycle"].idxmax()].sort_values("unit")
        cols = [c for c in ftr.columns if c not in ("unit", "cycle", "rul")]
        y_cap = np.minimum(rul.astype(float), RUL_CAP)
        spread["lgbm"].append(rmse(y_cap, np.clip(fit_lgbm(ftr[cols], ftr["rul"], seed).predict(fte[cols]), 0, RUL_CAP)))
        spread["lstm"].append(rmse(y_cap, fit_predict_lstm(train_r, test.assign(rul=np.nan), sensors, seed)[0]))
    res["seeds"] = args.seeds
    res["rmse_capped_by_seed"] = spread
    res["rmse_capped_seed_mean_sd"] = {k: [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]
                                       for k, v in spread.items()}
    res["data_sha256"] = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()}
    res["data_source"] = ("A. Saxena and K. Goebel (2008), Turbofan Engine Degradation Simulation Data Set, "
                          "NASA Prognostics Data Repository, NASA Ames Research Center")

    import mlflow

    from ingen_fleet_health.experiments.common import write_json
    from ingen_fleet_health.tracking import log_dict_json, log_flat_metrics, tracked_run

    with tracked_run("w02_external_cmapss", "fd001", {"track": "external_benchmark", "dataset": "C-MAPSS FD001"}) as run_:
        mlflow.log_params({"rul_cap": RUL_CAP, "window_cycles": WINDOW, "seeds": str(args.seeds),
                           **{f"lgbm_{k}": v for k, v in LGBM_PARAMS.items()},
                           **{f"sha256_{k}": v[:16] for k, v in res["data_sha256"].items()}})
        log_flat_metrics(res["rmse_vs_capped_truth"], prefix="rmse_capped.")
        log_flat_metrics(res["rmse_vs_raw_truth"], prefix="rmse_raw.")
        log_flat_metrics(res["nasa_score_vs_capped_truth"], prefix="nasa_score.")
        log_flat_metrics(res["internal_split_check"], prefix="internal.")
        log_flat_metrics({f"{k}_mean": v[0] for k, v in res["rmse_capped_seed_mean_sd"].items()}, prefix="rmse_capped_seeds.")
        log_flat_metrics({f"{k}_sd": v[1] for k, v in res["rmse_capped_seed_mean_sd"].items()}, prefix="rmse_capped_seeds.")
        log_dict_json(res, "cmapss_fd001_results.json")
        res["mlflow_run_id"] = run_.info.run_id
    write_json(res, "cmapss_fd001_results.json")
    for k in res["rmse_vs_capped_truth"]:
        print(f"{k:14s} RMSE capped {res['rmse_vs_capped_truth'][k]:.2f}  raw {res['rmse_vs_raw_truth'][k]:.2f}  "
              f"score {res['nasa_score_vs_capped_truth'][k]:.0f}")
    print("internal:", res["internal_split_check"], "lstm:", res["lstm_training"])
    print("seed mean/sd (capped RMSE):", res["rmse_capped_seed_mean_sd"])
    print("MLflow run:", res["mlflow_run_id"])


if __name__ == "__main__":
    main()
