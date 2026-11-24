"""Grounded alert summaries.

An alert is summarised from an **evidence pack** and nothing else:

* the alert itself (unit, time, calibrated risk, threshold);
* the top SHAP attributions of that prediction (features pushing the risk up
  and the strongest ones pushing it down), with the feature's value now and
  72 h earlier;
* operator notes / service tickets of that unit created in the 7 days before
  the alert, with the fold classifier's reading of each (never the label).

Two generators write from the same pack:

``TemplateSummariser``
    Deterministic and fully offline. It can only verbalise fields of the pack.
``LocalLLMSummariser``
    A small open-weights instruct model run locally through ``transformers``
    (no API, nothing leaves the machine). It writes free prose from the same
    pack, so it *can* invent a cause or a number; the faithfulness check in
    ``explain.faithfulness`` measures how often it does.

A hosted LLM (Plan v3.0 names Claude via API or Vertex AI) is intentionally
not called here: it needs a key, a cost approval and a data-handling decision.
``build_prompt`` returns exactly the text such a provider would receive.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ingen_fleet_health.explain.shap_utils import (
    CLASS_CONCEPT,
    CONCEPT_LABEL,
    feature_info,
)
from ingen_fleet_health.text import FAULT_CLASSES, TAXONOMY

RATIO_UNITS = ("x baseline", "x expected", "x nominal")
NEAR_REFERENCE_TOL = 0.02
TOP_POSITIVE = 5
TOP_NEGATIVE = 2
DRIVERS_IN_TEXT = 3
LINK_WINDOW_HOURS = 168.0
LOOKBACK_HOURS = 72
LOCAL_LLM = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class Attribution:
    feature: str
    phrase: str
    unit: str
    concept: str
    value: float | None
    value_72h_ago: float | None
    shap: float
    note_class: str | None = None


@dataclass
class LinkedTicket:
    ticket_id: str
    doc_type: str
    age_hours: float
    text: str
    predicted_class: str
    predicted_probability: float


@dataclass
class Evidence:
    alert_id: str
    unit_id: str
    t: str
    fold_id: str
    risk: float
    threshold: float
    base_value: float
    raw_margin: float
    positive: list[Attribution]
    negative: list[Attribution]
    tickets: list[LinkedTicket] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def notes(self) -> list[LinkedTicket]:
        return [t for t in self.tickets if t.doc_type == "operator_note"]


def _num(v: Any) -> float | None:
    return None if v is None or not np.isfinite(float(v)) else float(v)


def build_evidence(alert_id: str, row: pd.Series, row_72h_ago: pd.Series | None, shap_row: np.ndarray,
                   columns: list[str], base_value: float, risk: float, threshold: float, fold_id: str,
                   tickets: pd.DataFrame, ticket_proba: np.ndarray) -> Evidence:
    """``tickets`` / ``ticket_proba``: the unit's tickets (aligned), any time;
    only those created in the 7 days up to the alert time are linked."""
    order = np.argsort(-shap_row)

    def attribution(j: int) -> Attribution:
        info = feature_info(columns[j])
        before = _num(row_72h_ago[columns[j]]) if row_72h_ago is not None else None
        return Attribution(columns[j], info.phrase, info.unit, info.concept, _num(row[columns[j]]), before,
                           float(shap_row[j]), info.note_class)

    positive = [attribution(j) for j in order[:TOP_POSITIVE] if shap_row[j] > 0]
    negative = [attribution(j) for j in order[::-1][:TOP_NEGATIVE] if shap_row[j] < 0]
    t = row["t"]
    linked = []
    if len(tickets):
        age = (t - tickets["created_at_utc"]) / pd.Timedelta(hours=1)
        keep = np.flatnonzero(((age >= 0) & (age < LINK_WINDOW_HOURS)).to_numpy())
        for k in keep[np.argsort(age.to_numpy()[keep])]:
            p = ticket_proba[k]
            linked.append(LinkedTicket(str(tickets["ticket_id"].iloc[k]), str(tickets["doc_type"].iloc[k]),
                                       float(age.iloc[k]), str(tickets["text"].iloc[k]),
                                       TAXONOMY[int(np.argmax(p))], float(np.max(p))))
    return Evidence(alert_id, str(row["unit_id"]), pd.Timestamp(t).strftime("%Y-%m-%d %H:%M"), fold_id, float(risk),
                    float(threshold), float(base_value), float(base_value + shap_row.sum()), positive, negative, linked)


def fmt(value: float | None) -> str:
    """Three significant digits, plain notation."""
    if value is None:
        return "n/a"
    if value == 0:
        return "0"
    mag = int(np.floor(np.log10(abs(value))))
    decimals = int(min(6, max(0, 2 - mag)))
    return f"{value:.{decimals}f}"


def pct(value: float) -> str:
    """Probability as a percentage: one decimal below 10 %, none above."""
    return f"{100 * value:.1f}" if value < 0.0995 else f"{100 * value:.0f}"


def near_reference(a: Attribution) -> bool:
    """A ratio feature (x baseline / expected / nominal) within 2 % of 1."""
    return a.unit in RATIO_UNITS and a.value is not None and abs(a.value - 1.0) < NEAR_REFERENCE_TOL


def describe_attribution(a: Attribution) -> str:
    unit = f" {a.unit}" if a.unit else ""
    text = f"{a.phrase} at {fmt(a.value)}{unit}" if a.value is not None else f"{a.phrase} (no valid reading)"
    if near_reference(a):
        # the model still uses the value, but a reader must not infer a large deviation
        text += ", i.e. essentially at the reference level"
    if a.value is not None and a.value_72h_ago is not None and fmt(a.value_72h_ago) != fmt(a.value):
        text += f" ({fmt(a.value_72h_ago)}{unit} 72 h earlier)"
    return text


def note_behind(a: Attribution, ev: Evidence) -> LinkedTicket | None:
    """The linked operator note that feeds text feature ``a``, or None.

    ``txt_decay_<class>``: the newest note whose classifier reading is that
    class. ``txt_fault_max_72h``: the note younger than 72 h with the strongest
    fault reading. Count / recency features: the newest note. A note is never
    quoted for a feature it does not feed.
    """
    notes = ev.notes
    if a.feature.startswith("txt_decay_"):
        return next((n for n in notes if n.predicted_class == a.note_class), None)
    if a.feature == "txt_fault_max_72h":
        fault = [n for n in notes if n.age_hours < 72.0 and n.predicted_class in FAULT_CLASSES]
        return max(fault, key=lambda n: n.predicted_probability) if fault else None
    if a.feature == "txt_note_count_72h":
        return next((n for n in notes if n.age_hours < 72.0), None)
    if a.feature in ("txt_note_count_7d", "txt_hours_since_note"):
        return notes[0] if notes else None        # notes are ordered newest first
    return None


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else "; ".join(items[:-1]) + "; and " + items[-1]


class TemplateSummariser:
    """Deterministic two-to-three sentence summary; verbalises the pack only."""

    name = "template"

    def __init__(self, notes_ingested: bool = True) -> None:
        # Week 4 production v1 is telemetry-only: no operator notes reach the
        # model, so the summary must not say "no note was linked" (that would
        # read as "the unit has no notes").
        self.notes_ingested = notes_ingested

    def summarise(self, ev: Evidence) -> str:
        s1 = (f"{ev.unit_id} was flagged at {ev.t} UTC: the model puts the chance of a qualifying fault onset "
              f"within 72 h at {pct(ev.risk)} %, against an alert threshold of {pct(ev.threshold)} %.")
        drivers = ev.positive[:DRIVERS_IN_TEXT]
        if drivers:
            s2 = "The score is pushed up mainly by " + _join([describe_attribution(a) for a in drivers]) + "."
        else:
            s2 = "No single feature pushes this score up; it sits at the model's baseline."
        if not self.notes_ingested:
            return (f"{s1} {s2} Operator notes are not an input of this model version, "
                    "so check the unit's recent tickets before acting.")
        notes = ev.notes
        text_driver = next((a for a in drivers if a.concept == "operator_notes"), None)
        source = note_behind(text_driver, ev) if text_driver is not None else None
        if source is not None:
            s3 = (f"An operator note written {fmt(source.age_hours)} h before the alert feeds one of these drivers: "
                  f"\"{source.text}\" (ticket classifier reading: {source.predicted_class.replace('_', ' ')}).")
        elif text_driver is not None and not notes:
            s3 = ("One of these drivers is an operator-note feature, and no operator note was linked to this unit "
                  "in the previous 7 days.")
        elif text_driver is not None:
            s3 = ("One of these drivers is derived from the unit's operator notes as a whole rather than from "
                  "a single quotable note.")
        elif notes:
            s3 = (f"The unit has {len(notes)} operator note(s) from the previous 7 days, "
                  "but they are not among the main drivers of this score.")
        else:
            s3 = "No operator note was linked to this unit in the previous 7 days."
        return f"{s1} {s2} {s3}"


def build_prompt(ev: Evidence) -> tuple[str, str]:
    """(system, user) messages for an LLM. Only pack fields are included."""
    system = (
        "You write short alert explanations for robot fleet operators. Use ONLY the facts listed under EVIDENCE. "
        "Do not add causes, numbers, advice or component names that are not in EVIDENCE. Copy numbers exactly as "
        "written. Write 2 to 3 plain English sentences in one paragraph, and nothing else."
    )
    lines = [
        "EVIDENCE",
        f"Unit: {ev.unit_id}. Alert time: {ev.t} UTC.",
        f"Model risk of a qualifying fault onset within 72 h: {pct(ev.risk)} %. Alert threshold: {pct(ev.threshold)} %.",
        "Features pushing the risk UP (strongest first):",
    ]
    lines += [f"- {describe_attribution(a)}" for a in ev.positive[:DRIVERS_IN_TEXT]] or ["- none"]
    lines.append("Features pushing the risk DOWN:")
    lines += [f"- {describe_attribution(a)}" for a in ev.negative[:1]] or ["- none"]
    notes = ev.notes
    if notes:
        lines.append("Operator notes for this unit in the previous 7 days (newest first):")
        lines += [f"- {fmt(n.age_hours)} h before the alert: \"{n.text}\" (ticket classifier reading: "
                  f"{n.predicted_class.replace('_', ' ')})" for n in notes[:2]]
    else:
        lines.append("Operator notes for this unit in the previous 7 days: none.")
    lines.append("Write the explanation now.")
    return system, "\n".join(lines)


class LocalLLMSummariser:
    """Free-text summary from a small open-weights model, run locally on CPU."""

    def __init__(self, model_name: str = LOCAL_LLM, max_new_tokens: int = 160) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"local_llm[{model_name}]"
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.torch = torch
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.eval()

    def summarise(self, ev: Evidence) -> str:
        system, user = build_prompt(ev)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      repetition_penalty=1.05, pad_token_id=self.tokenizer.eos_token_id)
        text = str(self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        return " ".join(text.split())


def concept_label(concept: str) -> str:
    return CONCEPT_LABEL.get(concept, concept)


def ticket_concepts(ev: Evidence) -> set[str]:
    """Concepts a summary may attribute to linked tickets."""
    out: set[str] = set()
    for t in ev.tickets:
        concept = CLASS_CONCEPT.get(t.predicted_class)
        if concept:
            out.add(concept)
    return out
