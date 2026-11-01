
import argparse
import csv
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

from fleet_simulator import FIELDS

ROOT = Path(__file__).resolve().parent
NUMERIC_FIELDS = [
    col for col in FIELDS
    if col not in ("timestamp_utc", "unit_id", "mode")
]


def sql_path(path):
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def check_row(row, last_timestamp):
    problems = []
    unit = (row.get("unit_id") or "").strip()

    if not unit:
        problems.append("missing_unit_id")

    if row.get("mode") not in ("patrol", "dock"):
        problems.append("invalid_mode")

    timestamp = None
    try:
        timestamp = datetime.fromisoformat(
            row.get("timestamp_utc") or ""
        )
        if (
            timestamp.tzinfo is None
            or timestamp.utcoffset() != timedelta(0)
        ):
            problems.append("not_utc")
            timestamp = None
    except ValueError:
        problems.append("invalid_timestamp")

    values = {}

    for field in NUMERIC_FIELDS:
        try:
            value = float(row.get(field) or "")
            if not math.isfinite(value):
                raise ValueError
            values[field] = value
        except ValueError:
            problems.append(f"invalid_{field}")

    ranges = {
        "battery_soc_pct": (0, 100),
        "battery_soh_pct": (0, 100),
        "battery_voltage_v": (30, 60),
        "ambient_temp_c": (-60, 70),
        "motor_temp_c": (-40, 120),
        "terrain_factor": (0, 1),
        "vibration_rms_g": (0, float("inf")),
        "internal_resistance_ohm": (0, float("inf")),
    }

    for field, (low, high) in ranges.items():
        if field in values:
            value = values[field]
            if not low <= value <= high:
                problems.append(f"out_of_range_{field}")

    if (
        "internal_resistance_ohm" in values
        and values["internal_resistance_ohm"] == 0
    ):
        problems.append("zero_internal_resistance")

    if (
            "power_w" in values
            and "battery_current_a" in values
            and abs(values["power_w"] - 48 * values["battery_current_a"]) > 0.1
    ):
        problems.append("power_current_mismatch")

    previous = last_timestamp.get(unit)

    if (
            timestamp is not None
            and previous is not None
            and timestamp <= previous
    ):
        problems.append("duplicate_or_unordered_timestamp")

    return problems, timestamp, unit


def run_pipeline(source, out_root=None):
    source = Path(source)
    base = Path(out_root) if out_root else ROOT

    raw_dir = base / "data" / "raw"
    validated_dir = base / "data" / "validated"
    curated_dir = base / "data" / "curated"
    quarantine_dir = base / "data" / "quarantine"
    report_dir = base / "reports"

    for directory in (
        raw_dir, validated_dir, curated_dir,
        quarantine_dir, report_dir
    ):
        directory.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "fleet_telemetry.csv"
    accepted_path = validated_dir / "accepted.csv"
    rejected_path = quarantine_dir / "rejected.csv"

    if source.resolve() != raw_path.resolve():
        shutil.copy2(source, raw_path)

    total = 0
    accepted = 0
    rejected = 0
    gap_events = 0
    missing_minutes = 0
    reasons = Counter()
    last_timestamp = {}

    with (
        raw_path.open(newline="", encoding="utf-8") as src,
        accepted_path.open(
            "w", newline="", encoding="utf-8"
        ) as good,
        rejected_path.open(
            "w", newline="", encoding="utf-8"
        ) as bad,
    ):
        reader = csv.DictReader(src)

        missing_columns = set(FIELDS) - set(
            reader.fieldnames or []
        )
        if missing_columns:
            raise ValueError(
                f"Missing columns: {sorted(missing_columns)}"
            )

        good_writer = csv.DictWriter(
            good, fieldnames=FIELDS
        )
        bad_writer = csv.DictWriter(
            bad, fieldnames=FIELDS + ["error_codes"]
        )

        good_writer.writeheader()
        bad_writer.writeheader()

        for row in reader:
            total += 1
            problems, timestamp, unit = check_row(
                row, last_timestamp
            )

            if None in row:
                problems.append("extra_columns")

            if problems:
                rejected += 1
                reasons.update(problems)

                bad_writer.writerow({
                    **{key: row.get(key) for key in FIELDS},
                    "error_codes": ";".join(problems),
                })
                continue

            previous = last_timestamp.get(unit)

            if previous is not None:
                delta = timestamp - previous
                if delta > timedelta(minutes=1):
                    gap_events += 1
                    missing_minutes += int(
                        delta.total_seconds() // 60
                    ) - 1

            last_timestamp[unit] = timestamp
            accepted += 1

            good_writer.writerow({
                key: row[key] for key in FIELDS
            })

    if accepted == 0:
        raise ValueError("No valid rows available")

    validated_parquet = validated_dir / "telemetry.parquet"
    curated_parquet = curated_dir / "daily_summary.parquet"

    numeric_sql = ",\n".join(
        f"CAST({col} AS DOUBLE) AS {col}"
        for col in NUMERIC_FIELDS
    )

    con = duckdb.connect()

    try:
        con.execute(f"""
            COPY (
                SELECT
                    CAST(timestamp_utc AS TIMESTAMPTZ)
                        AS timestamp_utc,
                    unit_id,
                    mode,
                    {numeric_sql}
                FROM read_csv(
                    '{sql_path(accepted_path)}',
                    header=true,
                    all_varchar=true
                )
            )
            TO '{sql_path(validated_parquet)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        con.execute(f"""
            COPY (
                SELECT
                    unit_id,
                    CAST(timestamp_utc AT TIME ZONE 'UTC' AS DATE) AS date_utc,
                    COUNT(*) AS record_count,
                    AVG(battery_soc_pct) AS avg_soc_pct,
                    MIN(battery_soc_pct) AS min_soc_pct,
                    AVG(motor_temp_c) AS avg_motor_temp_c,
                    MAX(motor_temp_c) AS max_motor_temp_c,
                    AVG(vibration_rms_g) AS avg_vibration_g
                FROM read_parquet(
                    '{sql_path(validated_parquet)}'
                )
                GROUP BY unit_id, date_utc
                ORDER BY unit_id, date_utc
            )
            TO '{sql_path(curated_parquet)}'
            (FORMAT PARQUET)
        """)
    finally:
        con.close()

    report = {
        "dataset": "synthetic_fleet_telemetry",
        "total_rows": total,
        "accepted_rows": accepted,
        "rejected_rows": rejected,
        "acceptance_rate_pct": round(
            100 * accepted / total, 3
        ),
        "timestamp_gap_events": gap_events,
        "missing_minutes": missing_minutes,
        "rejection_reasons": dict(reasons),
        "quality_passed": (
            rejected == 0 and missing_minutes == 0
        ),
        "notes": [
            "Synthetic data only",
            "Physical ranges are illustrative",
            "Validation does not establish hardware fidelity",
        ],
    }

    report_path = report_dir / "W01_Data_Quality_Report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=ROOT / "data/generated/fleet_telemetry.csv",
        type=Path,
    )
    args = parser.parse_args()

    result = run_pipeline(args.input)
    print(json.dumps(result, indent=2))
