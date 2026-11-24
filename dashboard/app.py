"""InGen fleet health dashboard (Streamlit + Plotly). SYNTHETIC DATA.

    streamlit run dashboard/app.py                          # reads configs/pipeline.yaml
    FLEET_PIPELINE_CONFIG=data/quickstart/pipeline.yaml streamlit run dashboard/app.py

Built for a fleet operations manager: the first tab answers "which robots
should I pull before the next shift, and why?"; the others give the evidence
(unit drill-down, alert queue) and, for analysts, the model's health.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from components import alert_queue, model_health, overview, unit_view
from components.common import PLATFORM_LABEL, get_settings

from ingen_fleet_health.pipeline import views

st.set_page_config(page_title="Fleet health - InGen (synthetic)", page_icon=":robot_face:", layout="wide")

settings = get_settings()
all_platforms = list(settings.platforms)

with st.sidebar:
    st.markdown("### Filters")
    chosen = st.multiselect("Platform", all_platforms, default=all_platforms,
                            format_func=lambda p: PLATFORM_LABEL.get(p, p))
    dates = views.scored_dates(settings, chosen or all_platforms)
    if not dates:
        st.warning("No scored dates yet. Run the pipeline first (see README, quickstart).")
        st.stop()
    default_start = max(dates[0], dates[-1] - timedelta(days=13))
    picked = st.date_input("Date range (UTC)", value=(default_start, dates[-1]), min_value=dates[0],
                           max_value=dates[-1])
    start, end = (picked if isinstance(picked, tuple) and len(picked) == 2 else (dates[0], dates[-1]))
    st.caption("The shift decision uses the **end** of the date range as 'now'.")
    st.divider()
    st.caption("All robots, faults and numbers here are **simulated** (synthetic fleets, simulator v0.2). "
               "Nothing is from a real InGen deployment.")

platforms = chosen or all_platforms
as_of = pd.Timestamp(end).tz_localize("UTC") + pd.Timedelta(hours=23)

st.title("Fleet health")
st.caption(f"{', '.join(PLATFORM_LABEL.get(p, p) for p in platforms)} | {start} to {end} (UTC) | "
           "synthetic fleets for demonstration")

for f in views.freshness(settings, platforms, start, end):
    name = PLATFORM_LABEL.get(f.platform, f.platform)
    if f.last_scored is None:
        st.error(f"{name}: no scores in this range.")
    elif f.last_scored < end:
        st.warning(f"{name}: latest scores are from {f.last_scored}, not {end}. "
                   "Upstream data for the later dates has not been processed; these robots' risk is not current.")
    if f.missing_dates:
        st.warning(f"{name}: no scores for {', '.join(str(d) for d in f.missing_dates)} "
                   "(upstream data missing or late; see Model & data health).")

preds = views.predictions(settings, platforms, start, end)
alerts = views.alerts(settings, platforms, start, end)
unit_ref = views.units(settings, platforms)

tab_pull, tab_unit, tab_alerts, tab_health = st.tabs(
    ["Pull list (next shift)", "Unit drill-down", "Alert queue", "Model & data health"])
with tab_pull:
    overview.render(preds, alerts, unit_ref, as_of)
with tab_unit:
    unit_view.render(settings, preds, alerts, start, end)
with tab_alerts:
    alert_queue.render(alerts)
with tab_health:
    model_health.render(settings, platforms, start, end)
