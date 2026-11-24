"""Dashboard smoke test: the Streamlit app renders on real pipeline output, headless."""

from __future__ import annotations

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"


def test_dashboard_renders_every_tab_without_errors(fleet) -> None:  # type: ignore[no-untyped-def]
    os.environ["FLEET_PIPELINE_CONFIG"] = str(fleet.config)
    try:
        at = AppTest.from_file(str(APP), default_timeout=120)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert [t.label for t in at.tabs] == ["Pull list (next shift)", "Unit drill-down", "Alert queue",
                                              "Model & data health"]
        labels = [m.label for m in at.metric]
        assert {"Pull before next shift", "Alerted in last 72 h", "No recent data", "Robots monitored"} <= set(labels)
        monitored = next(m for m in at.metric if m.label == "Robots monitored")
        assert int(monitored.value) == 24                       # 12 Rover + 12 Fari units
        # platform filter: Fari only
        at.sidebar.multiselect[0].set_value(["fari"]).run()
        assert not at.exception
        assert int(next(m for m in at.metric if m.label == "Robots monitored").value) == 12
    finally:
        os.environ.pop("FLEET_PIPELINE_CONFIG", None)
