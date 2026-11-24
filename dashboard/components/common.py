"""Shared helpers: settings, colours, number formats."""

from __future__ import annotations

import math
import os
import sys

import streamlit as st

from ingen_fleet_health.pipeline.settings import Settings, load_settings
from ingen_fleet_health.pipeline.views import PLATFORM_LABEL

__all__ = ["PLATFORM_LABEL", "STATUS_COLOUR", "get_settings", "pct"]

# colour-blind-safe pairs; status is also always written out in text
STATUS_COLOUR = {"Pull before next shift": "#C0392B", "Alerted in last 72 h": "#E59866", "No recent data": "#7F8C8D",
                 "OK": "#2E86C1"}


def config_path() -> str | None:
    """``streamlit run dashboard/app.py -- --config <pipeline.yaml>``, else env FLEET_PIPELINE_CONFIG."""
    args = sys.argv[1:]
    if "--config" in args and args.index("--config") + 1 < len(args):
        return args[args.index("--config") + 1]
    return os.environ.get("FLEET_PIPELINE_CONFIG")


@st.cache_resource
def get_settings() -> Settings:
    return load_settings(config_path())


def pct(value: float, digits: int = 1) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}f} %"
