"""As-of text features: what the tickets of a unit say at prediction time t.

A ticket contributes to the row (unit, t) only if ``created_at_utc <= t`` and
it is younger than the window. Inputs are per-ticket class probabilities from
a classifier fitted inside the evaluation fold (``models.text_classifier``);
ticket *labels* are never read here.

Operator notes (symptom reports written before any diagnosis) and service
tickets (work orders and closures, which document work already decided) are
aggregated separately: ``doc_type`` is known when the ticket is created.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ingen_fleet_health.text import FAULT_CLASSES, TAXONOMY

WINDOW_HOURS = 168.0
HALF_LIFE_HOURS = 48.0
RECENT_HOURS = 72.0


@dataclass(frozen=True)
class TextFeatureSpec:
    name: str
    definition: str
    as_of: str


AS_OF = "tickets with created_at_utc <= t and age < window only"
TEXT_FEATURE_SPECS = [
    *[TextFeatureSpec(f"txt_decay_{c}",
                      f"sum over operator notes of P({c} | text) x 0.5^(age / 48 h), age < 7 d", AS_OF)
      for c in TAXONOMY],
    TextFeatureSpec("txt_fault_max_72h",
                    "max over operator notes younger than 72 h of P(battery) + P(drivetrain) + P(thermal)", AS_OF),
    TextFeatureSpec("txt_note_count_72h", "number of operator notes younger than 72 h", AS_OF),
    TextFeatureSpec("txt_note_count_7d", "number of operator notes younger than 7 d", AS_OF),
    TextFeatureSpec("txt_service_count_7d", "number of service tickets younger than 7 d", AS_OF),
    TextFeatureSpec("txt_hours_since_note", "hours since the latest operator note (capped at 168)", AS_OF),
]
TEXT_FEATURES = [s.name for s in TEXT_FEATURE_SPECS]


def _utc_ns(series: pd.Series) -> np.ndarray:
    """Timezone-aware timestamps as naive UTC ``datetime64[ns]`` (searchable)."""
    values = pd.to_datetime(series, utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    return np.asarray(values.to_numpy(dtype="datetime64[ns]"))


def asof_text_features(tickets: pd.DataFrame, proba: np.ndarray, grid: pd.DataFrame) -> pd.DataFrame:
    """One row per row of ``grid`` (unit_id, t), in the same order.

    ``tickets`` needs unit_id, created_at_utc, doc_type; ``proba`` is the
    aligned (n_tickets, 7) class-probability matrix in ``TAXONOMY`` order.
    """
    if len(tickets) != len(proba):
        raise ValueError("tickets and proba must be aligned")
    out = {name: np.zeros(len(grid)) for name in TEXT_FEATURES}
    out["txt_hours_since_note"] = np.full(len(grid), WINDOW_HOURS)
    fault_cols = [TAXONOMY.index(c) for c in FAULT_CLASSES]
    created_all = _utc_ns(tickets["created_at_utc"])
    grid_t = _utc_ns(grid["t"])
    is_note_all = tickets["doc_type"].eq("operator_note").to_numpy()
    ticket_idx = {u: idx for u, idx in tickets.groupby("unit_id").indices.items()}
    hour = np.timedelta64(1, "h")

    for unit, rows in grid.groupby("unit_id", sort=False).indices.items():
        idx = ticket_idx.get(unit)
        if idx is None:
            continue
        t = grid_t[rows]
        order = np.argsort(t, kind="stable")
        t_sorted = t[order]
        dest = rows[order]
        for k in idx:
            created = created_all[k]
            lo = np.searchsorted(t_sorted, created, side="left")          # first t >= created
            hi = np.searchsorted(t_sorted, created + np.timedelta64(int(WINDOW_HOURS), "h"), side="left")
            if lo >= hi:
                continue
            target = dest[lo:hi]
            age = (t_sorted[lo:hi] - created) / hour
            if not is_note_all[k]:
                out["txt_service_count_7d"][target] += 1.0
                continue
            weight = 0.5 ** (age / HALF_LIFE_HOURS)
            for j, cls in enumerate(TAXONOMY):
                out[f"txt_decay_{cls}"][target] += proba[k, j] * weight
            recent = age < RECENT_HOURS
            fault_p = float(proba[k, fault_cols].sum())
            out["txt_fault_max_72h"][target[recent]] = np.maximum(out["txt_fault_max_72h"][target[recent]], fault_p)
            out["txt_note_count_72h"][target[recent]] += 1.0
            out["txt_note_count_7d"][target] += 1.0
            out["txt_hours_since_note"][target] = np.minimum(out["txt_hours_since_note"][target], age)
    return pd.DataFrame(out, index=grid.index)
