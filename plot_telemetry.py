
import csv
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
data_path = ROOT / "data" / "generated" / "telemetry.csv"
output = ROOT / "docs" / "figures"
output.mkdir(parents=True, exist_ok=True)

with data_path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

hours = [
    (datetime.fromisoformat(r["timestamp_utc"]) -
     datetime.fromisoformat(rows[0]["timestamp_utc"])).total_seconds() / 3600
    for r in rows
]

def values(column):
    return [float(r[column]) for r in rows]

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

axes[0].plot(hours, values("battery_soc_pct"))
axes[0].set_ylabel("SoC (%)")
axes[0].set_title("Battery State of Charge")

axes[1].plot(hours, values("motor_temp_c"), label="Motor")
axes[1].plot(hours, values("ambient_temp_c"), label="Ambient")
axes[1].set_ylabel("Temperature (C)")
axes[1].set_title("Motor Thermal Response")
axes[1].legend()

axes[2].plot(hours, values("vibration_rms_g"))
axes[2].axvline(13, color="red", linestyle="--", label="Fault onset")
axes[2].set_ylabel("Vibration RMS (g)")
axes[2].set_xlabel("Hours since simulation start (UTC)")
axes[2].set_title("Synthetic Bearing Wear")
axes[2].legend()

for ax in axes:
    ax.grid(alpha=0.3)

fig.suptitle("Aido Rover — Synthetic Telemetry Prototype")
fig.tight_layout()
fig.savefig(output / "rover_telemetry.png", dpi=160)
plt.show()

print("Visualization generated successfully!")
