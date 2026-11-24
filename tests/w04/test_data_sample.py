"""The committed data/sample is synthetic, reproducible and accepted by the Week 1 data contract."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "sample"
PLATFORMS = ("aido_rover", "fari")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))


@pytest.mark.parametrize("platform", PLATFORMS)
def test_sample_is_small_synthetic_and_documented(platform: str) -> None:
    manifest = json.loads((SAMPLE / platform / "simulation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["platform"] == platform
    assert manifest["simulator_version"] == "0.2"
    assert (manifest["units"], manifest["days"], manifest["seed"]) == (2, 1, 7)
    size = sum(p.stat().st_size for p in (SAMPLE / platform).rglob("*") if p.is_file())
    assert size < 1_000_000
    assert "SYNTHETIC" in (SAMPLE / "README.md").read_text(encoding="utf-8")


def test_sample_regenerates_identically(tmp_path: Path) -> None:
    import make_data_sample

    fresh = make_data_sample.build(tmp_path)
    for platform in PLATFORMS:
        committed = json.loads((SAMPLE / platform / "simulation_manifest.json").read_text(encoding="utf-8"))
        ignore = {"config_sha256"}  # depends on the config file's line endings on checkout
        assert {k: v for k, v in fresh[platform].items() if k not in ignore} == \
               {k: v for k, v in committed.items() if k not in ignore}
        for csv in sorted((SAMPLE / platform).rglob("*.csv")):
            rel = csv.relative_to(SAMPLE / platform)
            pd.testing.assert_frame_equal(pd.read_csv(csv), pd.read_csv(tmp_path / platform / rel))


@pytest.mark.parametrize("platform", PLATFORMS)
def test_sample_passes_the_data_contract(platform: str, tmp_path: Path) -> None:
    from ingen_fleet_health.config import load_config
    from ingen_fleet_health.contract import platform_profile
    from ingest_pipeline import run_pipeline

    cfg = {"aido_rover": "aido_rover_v0_2.yaml", "fari": "fari.yaml"}[platform]
    src = tmp_path / "fleet_telemetry.csv"
    shutil.copy(SAMPLE / platform / "generated" / "fleet_telemetry.csv", src)
    report = run_pipeline(src, out_root=tmp_path, profile=platform_profile(load_config(ROOT / "configs" / cfg)))
    assert report["accepted_rows"] == 2874
    assert report["rejected_rows"] == 0
