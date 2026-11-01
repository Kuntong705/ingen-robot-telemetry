
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def inject_impairments(
    source,
    output_dir,
    dropout_rate=0.005,
    drift_c=4.0,
    seed=43,
):
    if not 0 <= dropout_rate < 1:
        raise ValueError("Invalid dropout rate")
    if drift_c < 0:
        raise ValueError("Drift must be nonnegative")

    source = Path(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "fleet_telemetry.csv"

    if source.resolve() == output_csv.resolve():
        raise ValueError("Source and output must differ")

    # First pass: count records for each robot
    with source.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        required = {
            "timestamp_utc", "unit_id", "motor_temp_c"
        }
        if not required.issubset(fields):
            raise ValueError("Missing required fields")

        counts = Counter(r["unit_id"] for r in reader)

    if not counts:
        raise ValueError("Empty input dataset")

    drift_unit = min(counts)
    n = counts[drift_unit]

    if n < 5:
        raise ValueError("Not enough records for drift")

    # Gradual sensor bias from 40% to 80% of the run
    drift_start = int(n * 0.40)
    drift_end = int(n * 0.80)

    rng = random.Random(seed)
    positions = defaultdict(int)

    dropped = 0
    drifted = 0
    written = 0
    onset_time = None
    ramp_end_time = None

    with (
        source.open(newline="", encoding="utf-8") as src,
        output_csv.open(
            "w", newline="", encoding="utf-8"
        ) as dst,
    ):
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()

        for row in reader:
            unit = row["unit_id"]
            index = positions[unit]
            positions[unit] += 1

            if unit == drift_unit:
                if index == drift_start:
                    onset_time = row["timestamp_utc"]
                if index == drift_end:
                    ramp_end_time = row["timestamp_utc"]

            # Keep boundary records so internal gaps are measurable
            if (
                0 < index < counts[unit] - 1
                and rng.random() < dropout_rate
            ):
                dropped += 1
                continue

            # Drift changes the measured value, not the physical
            # state of the simulator's motor.
            if unit == drift_unit and index >= drift_start:
                progress = min(
                    1.0,
                    (index - drift_start)
                    / (drift_end - drift_start),
                )
                row["motor_temp_c"] = (
                    f"{float(row['motor_temp_c']) + drift_c * progress:.3f}"
                )
                drifted += 1

            writer.writerow(row)
            written += 1

    # Ground truth stays separate from prediction telemetry
    event_path = output_dir / "sensor_drift_events.csv"

    with event_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "unit_id", "sensor", "onset_utc",
            "ramp_end_utc", "peak_bias_c", "ramp_shape",
        ])
        writer.writerow([
            drift_unit, "motor_temp_c",
            onset_time, ramp_end_time, drift_c, "linear",
        ])

    report = {
        "input_rows": sum(counts.values()),
        "output_rows": written,
        "dropped_rows": dropped,
        "drifted_rows": drifted,
        "drift_unit": drift_unit,
        "peak_drift_c": drift_c,
        "random_seed": seed,
    }

    (output_dir / "injection_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="data/generated/fleet_telemetry.csv",
    )
    parser.add_argument(
        "--output",
        default="data/generated/impairments",
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.005
    )
    parser.add_argument(
        "--seed", type=int, default=43
    )
    args = parser.parse_args()

    report = inject_impairments(
        args.source,
        args.output,
        dropout_rate=args.dropout_rate,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
