
import csv

from validate_telemetry import validate_csv

FIELDS = [
    "timestamp_utc",
    "unit_id",
    "mode",
    "battery_soc_pct",
    "power_w",
    "ambient_temp_c",
    "motor_temp_c",
    "vibration_rms_g",
]

VALID_ROW = {
    "timestamp_utc": "2026-10-26T08:00:00+00:00",
    "unit_id": "Rover_001",
    "mode": "patrol",
    "battery_soc_pct": "80",
    "power_w": "450",
    "ambient_temp_c": "25",
    "motor_temp_c": "35",
    "vibration_rms_g": "0.15",
}


def make_csv(tmp_path, rows, fields=None):
    if fields is None:
        fields = FIELDS

    path = tmp_path / "test_telemetry.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: row.get(key, "") for key in fields
            })

    return path


def test_valid_record(tmp_path):
    path = make_csv(tmp_path, [VALID_ROW])
    report = validate_csv(path)

    assert report["passed"] is True
    assert report["invalid_rows"] == 0


def test_reject_invalid_soc(tmp_path):
    row = {**VALID_ROW, "battery_soc_pct": "150"}
    path = make_csv(tmp_path, [row])
    report = validate_csv(path)

    assert report["passed"] is False
    assert report["invalid_rows"] == 1
    assert "Battery SoC out of range" in report["errors"][0]["problems"]


def test_reject_duplicate_timestamp(tmp_path):
    path = make_csv(tmp_path, [VALID_ROW, VALID_ROW])
    report = validate_csv(path)

    assert report["passed"] is False
    assert report["invalid_rows"] == 1


def test_reject_missing_column(tmp_path):
    fields = [f for f in FIELDS if f != "unit_id"]
    path = make_csv(tmp_path, [VALID_ROW], fields)
    report = validate_csv(path)

    assert report["passed"] is False
    assert "unit_id" in report["missing_columns"]


def test_reject_invalid_numeric(tmp_path):
    row = {**VALID_ROW, "motor_temp_c": "broken"}
    path = make_csv(tmp_path, [row])
    report = validate_csv(path)

    assert report["passed"] is False
    assert report["invalid_rows"] == 1
