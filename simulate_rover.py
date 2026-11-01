
import csv
import json
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Synthetic demonstration only — NOT real inGen specifications
SEED = 42
CAPACITY_WH = 6000.0
MINUTES = 24 * 60

rng = random.Random(SEED)
start = datetime(2026, 10, 26, tzinfo=UTC)

soc = 95.0
motor_temp = 25.0
records = []

for minute in range(MINUTES):
    timestamp = start + timedelta(minutes=minute)
    hour = minute / 60

    # Daily outdoor temperature variation
    ambient = 24 + 5 * math.sin(2 * math.pi * (hour - 6) / 24)

    # Patrol between 08:00 and 16:00 UTC
    patrol = 8 <= hour < 16
    mode = "patrol" if patrol else "dock"

    # Terrain variation affects energy consumption
    terrain = 0.5 + 0.5 * math.sin(2 * math.pi * minute / 90)

    if patrol:
        power_w = 400 + 120 * terrain
    else:
        power_w = -700 if soc < 95 else 0

    # Energy balance: positive power discharges battery
    soc -= (power_w / 60 / CAPACITY_WH) * 100
    soc = max(0, min(100, soc))

    # First-order motor thermal model
    heat_w = 0.10 * max(power_w, 0)
    target_temp = ambient + 0.28 * heat_w
    motor_temp += (target_temp - motor_temp) / 60

    # Inject gradual bearing wear from 13:00
    fault_progress = max(0, min(1, (minute - 780) / 180))
    vibration = (
        0.15 + 0.5 * fault_progress if patrol else 0.04
    )
    vibration += rng.gauss(0, 0.01)
    vibration = max(0, vibration)

    records.append({
        "timestamp_utc": timestamp.isoformat(),
        "unit_id": "Rover_001",
        "mode": mode,
        "battery_soc_pct": round(soc, 3),
        "power_w": round(power_w, 2),
        "ambient_temp_c": round(ambient, 2),
        "motor_temp_c": round(motor_temp, 2),
        "vibration_rms_g": round(vibration, 4),
    })

# Save generated telemetry
output = Path("data/generated")
output.mkdir(parents=True, exist_ok=True)

with (output / "telemetry.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

# Separate ground-truth fault labels to avoid target leakage
with (output / "fault_events.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["unit_id", "fault_type", "onset_utc", "failure_utc"])
    writer.writerow([
        "Rover_001",
        "bearing_wear",
        (start + timedelta(hours=13)).isoformat(),
        (start + timedelta(hours=16)).isoformat(),
    ])

# Basic quality validation
timestamps = [r["timestamp_utc"] for r in records]
checks = {
    "row_count": len(records),
    "timestamps_unique": len(timestamps) == len(set(timestamps)),
    "timestamps_ordered": timestamps == sorted(timestamps),
    "soc_in_range": all(0 <= r["battery_soc_pct"] <= 100 for r in records),
    "temperature_in_range": all(-40 <= r["motor_temp_c"] <= 120 for r in records),
    "vibration_nonnegative": all(r["vibration_rms_g"] >= 0 for r in records),
}

checks["passed"] = all(
    value for key, value in checks.items() if key != "row_count"
)

with (output / "quality_report.json").open("w", encoding="utf-8") as f:
    json.dump(checks, f, indent=2)

print("Simulation complete")
print(f"Generated {len(records)} telemetry records")
print(f"Data directory: {output.resolve()}")
print(f"Quality checks passed: {checks['passed']}")
