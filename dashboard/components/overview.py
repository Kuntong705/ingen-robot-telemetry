"""Pull list: which robots to pull before the next shift, and why."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.common import STATUS_COLOUR, pct
from ingen_fleet_health.pipeline import views


def render(preds: pd.DataFrame, alerts: pd.DataFrame, unit_ref: pd.DataFrame, as_of: pd.Timestamp) -> None:
    table = views.pull_list(preds, alerts, unit_ref, as_of)
    if table.empty:
        st.info("No scores in the selected range.")
        return
    pull = table[table["status"] == "Pull before next shift"]
    recent = table[table["status"] == "Alerted in last 72 h"]
    nodata = table[table["status"] == "No recent data"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pull before next shift", len(pull), help="New alert in the last 24 hours.")
    c2.metric("Alerted in last 72 h", len(recent), help="Older alert, still open: inspection should be under way.")
    c3.metric("No recent data", len(nodata), help="Telemetry gap: check the robot's connection, not its hardware.")
    c4.metric("Robots monitored", len(table))
    c5.metric("As of (UTC)", f"{as_of:%d %b %H:%M}", help=f"{as_of:%Y-%m-%d %H:%M} UTC")

    st.subheader("Robots to pull, and why")
    if pull.empty:
        st.success("No new alert in the last 24 hours: the model recommends no pull for the next shift.")
    for _, r in pull.iterrows():
        with st.container(border=True):
            left, right = st.columns([1, 3])
            left.markdown(f"**{r['unit_id']}**  \n{r['platform']}" + (f" | {r['site']}" if "site" in r and
                                                                     isinstance(r["site"], str) else ""))
            left.markdown(f"Risk now **{pct(r['risk_now_pct'])}**  \n(alert level {pct(r['threshold_pct'])})")
            if pd.notna(r["battery_soh_pct"]):
                left.caption(f"Battery health {r['battery_soh_pct']:.1f} %, "
                             f"7-day forecast {r['battery_soh_7d_forecast_pct']:.1f} %")
            right.markdown(f"**Alert raised {r['alert_time']:%Y-%m-%d %H:%M} UTC.** {r['why']}")
            right.caption("Explanation generated only from the model's own evidence (template; checked "
                          "automatically against the attributions). Main driver: " + (r["main_driver"] or "n/a"))
    if not recent.empty:
        st.info("Alerted in the last 72 h (inspection should be under way): " + ", ".join(recent["unit_id"]) + ".")
    if not nodata.empty:
        st.warning("No recent data from: " + ", ".join(nodata["unit_id"]) +
                   ". The model cannot judge these robots; check their connectivity before the shift.")

    st.subheader("Fleet risk ranking")
    st.caption("Risk = the model's estimated chance of a qualifying fault onset in the next 72 hours. "
               "The alert level is set per platform so that the fleet gets at most about one alert per robot per week.")
    top = table.head(25).copy()
    top["label"] = top["unit_id"]
    fig = px.bar(top.sort_values("risk_now_pct"), x="risk_now_pct", y="label", color="status", orientation="h",
                 color_discrete_map=STATUS_COLOUR, hover_data=["platform", "threshold_pct", "risk_24h_ago_pct"],
                 labels={"risk_now_pct": "Risk now (%)", "label": "", "status": "Status"}, height=620)
    fig.update_layout(margin={"l": 10, "r": 10, "t": 10, "b": 10}, legend={"orientation": "h", "y": -0.08})
    st.plotly_chart(fig, use_container_width=True)
    show = table[["rank", "unit_id", "platform", "status", "risk_now_pct", "threshold_pct", "risk_24h_ago_pct",
                  "main_driver", "battery_soh_pct", "battery_soh_7d_forecast_pct"]]
    st.dataframe(show, hide_index=True, use_container_width=True, column_config={
        "risk_now_pct": st.column_config.NumberColumn("Risk now %", format="%.1f"),
        "threshold_pct": st.column_config.NumberColumn("Alert level %", format="%.1f"),
        "risk_24h_ago_pct": st.column_config.NumberColumn("Risk 24 h ago %", format="%.1f"),
        "battery_soh_pct": st.column_config.NumberColumn("Battery health %", format="%.1f"),
        "battery_soh_7d_forecast_pct": st.column_config.NumberColumn("Battery in 7 days %", format="%.1f"),
        "main_driver": "Main driver of the alert",
    })
