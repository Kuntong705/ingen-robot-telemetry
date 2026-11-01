
import csv
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

REQUIRED_COLUMNS = {
    "timestamp_utc",
    "unit_id",
    "mode",
    "battery_soc_pct",
    "power_w",
    "ambient_temp_c",
    "motor_temp_c",
    "vibration_rms_g",
}


def validate_csv(path):
    total = 0
    invalid = 0
    gaps = 0
    errors = []
    last_seen = {}

    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])

        if missing:
            return {
                "passed": False,
                "total_rows": 0,
                "invalid_rows": 0,
                "missing_columns": sorted(missing),
                "errors": ["Missing required columns"],
            }

        for line_number, row in enumerate(reader, start=2):
            total += 1
            problems = []

            unit = (row["unit_id"] or "").strip()
            if not unit:
                problems.append("Missing unit_id")

            if row["mode"] not in ("patrol", "dock"):
                problems.append("Invalid operating mode")

            timestamp = None
            try:
                timestamp = datetime.fromisoformat(row["timestamp_utc"])
                if (
                    timestamp.tzinfo is None
                    or timestamp.utcoffset() != timedelta(0)
                ):
                    problems.append("Timestamp must be UTC")
                    timestamp = None
            except (ValueError, TypeError):
                problems.append("Invalid timestamp")

            numeric_fields = [
                "battery_soc_pct",
                "power_w",
                "ambient_temp_c",
                "motor_temp_c",
                "vibration_rms_g",
            ]

            numbers = {}
            for field in numeric_fields:
                try:
                    value = float(row[field])
                    if not math.isfinite(value):
                        raise ValueError
                    numbers[field] = value
                except (ValueError, TypeError):
                    problems.append(f"Invalid numeric value: {field}")

            soc = numbers.get("battery_soc_pct")
            if soc is not None and not 0 <= soc <= 100:
                problems.append("Battery SoC out of range")

            temp = numbers.get("motor_temp_c")
            if temp is not None and not -40 <= temp <= 120:
                problems.append("Motor temperature out of range")

            vibration = numbers.get("vibration_rms_g")
            if vibration is not None and vibration < 0:
                problems.append("Negative vibration")

            if unit and timestamp is not None:
                previous = last_seen.get(unit)

                if previous is not None:
                    if timestamp <= previous:
                        problems.append("Duplicate or unordered timestamp")
                    elif timestamp - previous > timedelta(minutes=1):
                        gaps += 1

                if previous is None or timestamp > previous:
                    last_seen[unit] = timestamp

            if problems:
                invalid += 1
                if len(errors) < 20:
                    errors.append({
                        "line": line_number,
                        "problems": problems,
                    })

    return {
        "passed": invalid == 0 and total > 0,
        "total_rows": total,
        "invalid_rows": invalid,
        "timestamp_gaps": gaps,
        "errors": errors,
    }


if __name__ == "__main__":
    input_path = Path("data/generated/telemetry.csv")
    report = validate_csv(input_path)

    output_path = Path("data/generated/validation_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
