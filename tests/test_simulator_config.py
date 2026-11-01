import copy
import csv

import pytest
import yaml

from fleet_simulator import load_config, simulate_fleet


def test_config_changes_battery_behavior(tmp_path):
    base = load_config()
    modified = copy.deepcopy(base)

    modified["battery"]["nominal_capacity_wh"] = 3000.0

    custom_config = tmp_path / "custom.yaml"
    custom_config.write_text(
        yaml.safe_dump(modified),
        encoding="utf-8",
    )

    original_dir = tmp_path / "original"
    modified_dir = tmp_path / "modified"

    simulate_fleet(
        units=1,
        days=1,
        output_dir=original_dir,
    )

    simulate_fleet(
        units=1,
        days=1,
        output_dir=modified_dir,
        config_path=custom_config,
    )

    def noon_soc(path):
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return float(rows[720]["battery_soc_pct"])

    original_soc = noon_soc(
        original_dir / "fleet_telemetry.csv"
    )
    modified_soc = noon_soc(
        modified_dir / "fleet_telemetry.csv"
    )

    assert modified_soc < original_soc


def test_reject_unsupported_sampling(tmp_path):
    config = copy.deepcopy(load_config())
    config["simulation"]["sample_interval_seconds"] = 30

    path = tmp_path / "invalid.yaml"
    path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="60-second"):
        simulate_fleet(
            units=1,
            days=1,
            config_path=path,
            output_dir=tmp_path / "output",
        )
