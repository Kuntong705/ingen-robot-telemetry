"""Regenerate the small committed data sample in ``data/sample/`` (Plan v3.0 §7). SYNTHETIC.

Two simulated robots x one day per platform (Aido Rover v0.2 and the synthetic
Fari profile), written by the same simulator code and configs as the full
study data. It is a schema and format example: it is too small to train or
evaluate anything, and it contains no fault onset. Every other dataset stays
git-ignored and is regenerated from the configs (see README).

    python scripts/make_data_sample.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingen_fleet_health.simulate.fleet import simulate_fleet_v02

SAMPLE = ROOT / "data" / "sample"
SPEC = {"aido_rover": ROOT / "configs" / "aido_rover_v0_2.yaml", "fari": ROOT / "configs" / "fari.yaml"}
UNITS, DAYS, SEED = 2, 1, 7


def build(root: Path = SAMPLE) -> dict[str, Any]:
    out = {}
    for platform, cfg in SPEC.items():
        target = root / platform
        if target.exists():
            shutil.rmtree(target)
        out[platform] = simulate_fleet_v02(target, UNITS, DAYS, SEED, cfg)
    return out


def main() -> None:
    for platform, manifest in build().items():
        print(f"{platform}: {manifest['telemetry_rows_written']} telemetry rows -> {SAMPLE / platform}")


if __name__ == "__main__":
    main()
