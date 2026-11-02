
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/validated/telemetry.parquet"
OUTPUT = ROOT / "docs/figures/fleet_physics_sanity.png"

if not SOURCE.exists():
    raise FileNotFoundError(
        "Run ingest_pipeline.py first"
    )

con = duckdb.connect()

try:
    daily = con.execute(
        """
        SELECT
            timestamp_utc,
            battery_soc_pct,
            power_w,
            ambient_temp_c,
            motor_temp_c
        FROM read_parquet(?)
        WHERE unit_id = 'Rover_001'
        ORDER BY timestamp_utc
        LIMIT 1440
        """,
        [str(SOURCE)],
    ).fetchall()

    health = con.execute(
        """
        SELECT
            CAST(
                timestamp_utc AT TIME ZONE 'UTC'
                AS DATE
            ) AS date_utc,
            AVG(battery_soh_pct) AS soh
        FROM read_parquet(?)
        WHERE unit_id = 'Rover_001'
        GROUP BY date_utc
        ORDER BY date_utc
        """,
        [str(SOURCE)],
    ).fetchall()
finally:
    con.close()

if not daily or not health:
    raise ValueError("No data available for Rover_001")

hours = list(range(len(daily)))
hours = [minute / 60 for minute in hours]

soc = [row[1] for row in daily]
power = [row[2] for row in daily]
ambient = [row[3] for row in daily]
motor = [row[4] for row in daily]

days = list(range(1, len(health) + 1))
soh = [row[1] for row in health]

fig, axes = plt.subplots(
    3, 1, figsize=(11, 11)
)

axes[0].plot(hours, soc)
axes[0].set_title("Battery State of Charge")
axes[0].set_xlabel("Hours")
axes[0].set_ylabel("SoC (%)")
axes[0].grid(alpha=0.3)

axes[1].plot(hours, motor, label="Motor")
axes[1].plot(hours, ambient, label="Ambient")
axes[1].set_title("Motor Thermal Response")
axes[1].set_xlabel("Hours")
axes[1].set_ylabel("Temperature (C)")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(days, soh)
axes[2].set_title("Battery State of Health")
axes[2].set_xlabel("Simulation day")
axes[2].set_ylabel("SoH (%)")
axes[2].grid(alpha=0.3)

fig.suptitle(
    "Synthetic Rover Fleet — Physics Sanity Checks"
)
fig.tight_layout()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=160)
plt.close(fig)

print(f"Generated: {OUTPUT}")
print(f"Initial daily SoH: {soh[0]:.4f}%")
print(f"Final daily SoH: {soh[-1]:.4f}%")
print(f"Battery SoH change: {soh[-1] - soh[0]:.4f} pp")
print(f"Peak motor temperature: {max(motor):.2f} C")
print(f"Peak patrol power: {max(power):.2f} W")
