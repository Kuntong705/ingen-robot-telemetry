"""Alert queue with grounded summaries and their automatic evidence check."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from components.common import PLATFORM_LABEL, pct


def render(alerts: pd.DataFrame) -> None:
    if alerts.empty:
        st.info("No alerts in the selected range.")
        return
    a = alerts.sort_values("t", ascending=False)
    checked = int(a["summary_faithful_auto"].sum())
    st.caption(f"{len(a)} alerts. Each explanation is written by a fixed template from the alert's own evidence "
               f"(the model's top attributions and the readings now and 72 h earlier). The automatic check found "
               f"every stated fact supported in {checked} of {len(a)}. The automatic check is lexical: in Week 3 it "
               "missed errors in free-text LLM summaries, which is why free text is not used here.")
    for _, r in a.iterrows():
        title = (f"{r['t']:%Y-%m-%d %H:%M} UTC | {r['unit_id']} ({PLATFORM_LABEL.get(r['platform'], r['platform'])}) "
                 f"| risk {pct(100 * r['risk'])} (alert level {pct(100 * r['threshold'])})")
        with st.expander(title):
            st.write(r["summary"])
            drivers = pd.DataFrame(json.loads(r["drivers"]))
            if not drivers.empty:
                drivers = drivers.rename(columns={"phrase": "What the model looked at", "value": "Now",
                                                  "value_72h_ago": "72 h earlier", "unit": "Unit",
                                                  "shap": "Push on risk (log-odds)"})
                st.dataframe(drivers.drop(columns=["feature"]), hide_index=True, use_container_width=True)
            unsupported = json.loads(r["unsupported_claims"])
            st.caption(f"Automatic evidence check: {r['claims_checked']} statements checked, "
                       + ("all supported." if not unsupported else f"{len(unsupported)} unsupported: {unsupported}"))
            st.caption(f"Model {r['model']} | status: {r['status']}")
