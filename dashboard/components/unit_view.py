"""Per-unit drill-down: risk trend, recent telemetry, battery health, alert history."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.common import PLATFORM_LABEL
from ingen_fleet_health.pipeline import views
from ingen_fleet_health.pipeline.settings import Settings

CHANNELS = [("motor_temp_c", "Motor temperature (deg C)"), ("vibration_rms_g", "Vibration RMS (g)"),
            ("battery_soc_pct", "Battery charge (%)"), ("power_w", "Power (W)")]


def render(settings: Settings, preds: pd.DataFrame, alerts: pd.DataFrame, start: date, end: date) -> None:
    if preds.empty:
        st.info("No scores in the selected range.")
        return
    latest = preds.sort_values("t").groupby("unit_id").tail(1).sort_values("risk", ascending=False)
    unit = st.selectbox("Robot (highest current risk first)", latest["unit_id"].tolist())
    p = preds[preds["unit_id"] == unit].sort_values("t")
    platform = str(p["platform"].iloc[0])
    st.caption(f"{PLATFORM_LABEL.get(platform, platform)} | risk model {p['early_warning_model'].iloc[0]}")

    a = alerts[alerts["unit_id"] == unit] if not alerts.empty else alerts
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p["t"], y=100 * p["risk"], name="Risk (%)", mode="lines", line={"color": "#1F4E79"}))
    fig.add_trace(go.Scatter(x=p["t"], y=100 * p["threshold"], name="Alert level", mode="lines",
                             line={"color": "#C0392B", "dash": "dash"}))
    gaps = p[p["insufficient_data"]]
    if len(gaps):
        fig.add_trace(go.Scatter(x=gaps["t"], y=[0] * len(gaps), name="Insufficient data", mode="markers",
                                 marker={"color": "#7F8C8D", "symbol": "x"}))
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["t"], y=100 * a["risk"], name="Alert raised", mode="markers",
                                 marker={"color": "#C0392B", "size": 12, "symbol": "triangle-up"}))
    fig.update_layout(title="Risk of a qualifying fault onset within 72 h", yaxis_title="%", height=330,
                      margin={"l": 10, "r": 10, "t": 40, "b": 10}, legend={"orientation": "h", "y": -0.2})
    st.plotly_chart(fig, use_container_width=True)

    days = st.slider("Telemetry window (days before the end date)", 1, 14, 3)
    tel = views.telemetry(settings, platform, unit, max(start, end - timedelta(days=days - 1)), end)
    if tel.empty:
        st.info("No validated telemetry for this robot in the window.")
    else:
        sub = make_subplots(rows=len(CHANNELS), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            subplot_titles=[c[1] for c in CHANNELS])
        for i, (col, _) in enumerate(CHANNELS, start=1):
            if col in tel.columns:
                sub.add_trace(go.Scatter(x=tel["timestamp_utc"], y=tel[col], mode="lines", showlegend=False,
                                         line={"width": 1.2, "color": "#34495E"}), row=i, col=1)
        sub.update_layout(height=170 * len(CHANNELS), margin={"l": 10, "r": 10, "t": 30, "b": 10})
        st.plotly_chart(sub, use_container_width=True)
        st.caption("10-minute means of the validated telemetry. Gaps are missing data (network outages), "
                   "not zero readings.")

    soh = p.dropna(subset=["soh_forecast_pct"])
    if len(soh):
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=soh["t"], y=soh["soh_now_pct"], name="Battery health now (BMS, 24 h mean)",
                                mode="lines+markers"))
        f2.add_trace(go.Scatter(x=pd.to_datetime(soh["soh_target_date"]), y=soh["soh_forecast_pct"],
                                name="7-day forecast (for the target day)", mode="markers",
                                marker={"symbol": "diamond-open"}))
        f2.update_layout(title="Battery state of health (%)", height=300, margin={"l": 10, "r": 10, "t": 40, "b": 10},
                         legend={"orientation": "h", "y": -0.25})
        st.plotly_chart(f2, use_container_width=True)

    st.markdown("**Alerts for this robot**")
    if a.empty:
        st.write("No alert in the selected range.")
    for _, r in a.sort_values("t", ascending=False).iterrows():
        st.markdown(f"- {r['t']:%Y-%m-%d %H:%M} UTC: {r['summary']}")
