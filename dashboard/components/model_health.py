"""For analysts: data quality, drift (PSI) and weekly performance on confirmed faults."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from components.common import PLATFORM_LABEL
from ingen_fleet_health.pipeline import views
from ingen_fleet_health.pipeline.settings import Settings


def render(settings: Settings, platforms: list[str], start: date, end: date) -> None:
    st.markdown("#### Data quality per day")
    q = views.quality(settings, platforms, start, end)
    if q.empty:
        st.write("No validated partitions in range.")
    else:
        st.dataframe(q, hide_index=True, use_container_width=True)

    st.markdown("#### Drift: population stability index (PSI) against the training data")
    st.caption("PSI < 0.10 stable, 0.10-0.25 watch, > 0.25 shift. Computed on one day of hourly rows per "
               "platform, so small fleets give noisy values. Seasonal change (the simulated weather cools over "
               "the period) also shows up as drift.")
    d = views.drift(settings, platforms, start, end)
    if d.empty:
        st.write("No drift partitions in range.")
    else:
        for p in platforms:
            sub = d[d["platform"] == p]
            if sub.empty:
                continue
            pivot = sub.pivot_table(index="feature", columns="date", values="psi")
            fig = px.imshow(pivot, color_continuous_scale="OrRd", zmin=0, zmax=0.5, aspect="auto",
                            labels={"color": "PSI"}, title=PLATFORM_LABEL.get(p, p))
            fig.update_layout(height=60 + 28 * len(pivot), margin={"l": 10, "r": 10, "t": 40, "b": 10})
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Weekly performance on newly confirmed faults")
    st.caption("A week is evaluated once its labels are settled (7 days later). Held-out robots were never seen "
               "in training; that is the honest number. Weeks with fewer than 3 confirmed fault onsets are "
               "reported as 'insufficient labels'. Shadow models are scored but never raise alerts.")
    rows = []
    for rep in views.performance_reports(settings, start, end):
        for p, e in rep["platforms"].items():
            if p not in platforms:
                continue
            for cohort in ("held_out_units", "training_units"):
                m = e.get(cohort)
                if not m:
                    continue
                rows.append({"check date": rep["check_date"],
                             "week scored": f"{rep['window_start']} to {rep['window_end_exclusive']}",
                             "platform": PLATFORM_LABEL.get(p, p), "cohort": cohort.replace("_", " "),
                             "status": m["status"], "fault onsets": m["events"], "detected": m["events_detected"],
                             "alerts": m["alerts"], "precision": m["precision"],
                             "false alarms / unit-week": m["false_alarms_per_unit_week"],
                             "PR-AUC champion": m["pr_auc_champion"], "PR-AUC iforest (shadow)": m["pr_auc_iforest"],
                             "PR-AUC physics rule (shadow)": m["pr_auc_physics_rule"], "base rate": m["base_rate"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.write("No weekly check in range yet (the first one needs a week of predictions plus 7 days of label delay).")
