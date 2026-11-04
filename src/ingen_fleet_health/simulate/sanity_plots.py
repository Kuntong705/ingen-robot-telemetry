"""Physics sanity-check figure for simulator v0.2.

    python -m ingen_fleet_health.simulate.sanity_plots --data-dir data/w02
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ingen_fleet_health.config import REPO_ROOT

SITE_COLORS = {"site_cool": "#2563eb", "site_mild": "#16a34a", "site_hot": "#dc2626"}


def make_figure(data_dir: Path, out: Path) -> Path:
    tel = pd.read_parquet(data_dir / "data" / "validated" / "telemetry.parquet")
    units = pd.read_csv(data_dir / "events" / "units.csv")
    truth = pd.read_csv(data_dir / "ground_truth" / "battery_truth_daily.csv", parse_dates=["date_utc"])
    faults = pd.read_csv(data_dir / "ground_truth" / "fault_events.csv",
                         parse_dates=["initiation_utc", "onset_utc", "failure_utc", "repair_start_utc"])
    t0 = tel["timestamp_utc"].min()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    for pattern, color in (("single", "#1d4ed8"), ("double", "#c2410c")):
        unit = units.loc[units["shift_pattern"] == pattern, "unit_id"].iloc[0]
        u = tel[(tel["unit_id"] == unit) & (tel["timestamp_utc"] < t0 + pd.Timedelta(days=3))]
        ax.plot((u["timestamp_utc"] - t0) / pd.Timedelta(hours=1), u["battery_soc_pct"], color=color, lw=1,
                label=f"{unit} ({pattern} shift)")
    ax.set_xlabel("hours since start")
    ax.set_ylabel("SoC (%)")
    ax.set_title("(a) Discharge on patrol, CC-CV recharge in dock")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    healthy = set(units["unit_id"]) - set(faults["unit_id"])
    single = set(units.loc[units["shift_pattern"] == "single", "unit_id"])
    steady = tel[tel["unit_id"].isin(healthy & single) & (tel["mode"] == "patrol")
                 & tel["timestamp_utc"].dt.hour.between(11, 15)]
    hourly = steady.groupby(["unit_id", steady["timestamp_utc"].dt.floor("h")]).agg(
        power=("power_w", "mean"), rise=("motor_temp_c", "mean"), amb=("ambient_temp_c", "mean"))
    hourly["rise"] -= hourly["amb"]
    ax.scatter(hourly["power"], hourly["rise"], s=4, alpha=0.3, color="#1d4ed8", label="hourly means, hours 3-8 of shift")
    xs = pd.Series([hourly["power"].min(), hourly["power"].max()])
    ax.plot(xs, 0.028 * xs, color="#dc2626", lw=1.2, label="nominal steady state 0.028 C/W x P")
    ax.set_xlabel("battery power on patrol (W)")
    ax.set_ylabel("motor temp - ambient (deg C)")
    ax.set_title("(b) Motor heating rises with load (healthy units)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    site = units.set_index("unit_id")["site"]
    for unit, g in truth.groupby("unit_id"):
        ax.plot(g["date_utc"], g["soh_true_pct"], color=SITE_COLORS[site[unit]], lw=0.8, alpha=0.7)
    for name, color in SITE_COLORS.items():
        ax.plot([], [], color=color, label=name)
    ax.set_ylabel("true SoH (%)")
    ax.set_title("(c) Capacity fade: monotone, faster in hot climate and with cell faults")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 1]
    end = tel["timestamp_utc"].max()
    shown = faults[(faults["fault_type"] == "bearing_wear") & (faults["repair_start_utc"] < end)]
    ev = shown.sort_values("precursor_days").iloc[-1]
    u = tel[(tel["unit_id"] == ev["unit_id"]) & (tel["mode"] == "patrol")]
    window = (u["timestamp_utc"] > ev["initiation_utc"] - pd.Timedelta(days=5)) & (
        u["timestamp_utc"] < ev["repair_start_utc"] + pd.Timedelta(days=2))
    u = u[window]
    norm = u["vibration_rms_g"] / (1 + 0.6 * u["terrain_factor"])
    hourly = norm.groupby(u["timestamp_utc"].dt.floor("h")).mean()
    ax.plot(hourly.index, hourly.to_numpy(), color="#111111", lw=0.8, label="patrol vibration / (1 + 0.6 terrain)")
    for col, color, label in (("initiation_utc", "#6b7280", "initiation (hidden)"),
                              ("onset_utc", "#dc2626", "qualifying onset (label)"),
                              ("repair_start_utc", "#16a34a", "repair")):
        ax.axvline(ev[col], color=color, ls="--", lw=1, label=label)
    ax.axvspan(ev["onset_utc"] - pd.Timedelta(hours=72), ev["onset_utc"], color="#fecaca", alpha=0.5,
               label="72 h positive window")
    ax.set_ylabel("normalised vibration RMS (g)")
    ax.set_title(f"(d) Bearing wear grows gradually ({ev['unit_id']}, {ev['ramp_shape']} ramp)")
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Simulator v0.2 physics sanity checks (synthetic Aido Rover fleet)", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "w02")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "figures" / "w02" / "sim_v02_sanity.png")
    args = parser.parse_args()
    print(make_figure(args.data_dir, args.out))


if __name__ == "__main__":
    main()
