"""Shared paths and YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V02_CONFIG = REPO_ROOT / "configs" / "aido_rover_v0_2.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_V02_CONFIG
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Invalid YAML configuration: {config_path}")
    return config
