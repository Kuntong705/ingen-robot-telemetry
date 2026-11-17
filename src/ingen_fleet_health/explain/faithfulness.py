"""Faithfulness check: is every stated cause, number and quote in the evidence pack?

A summary is **faithful** only if every claim extracted from its text is
supported by the alert's evidence pack (``alert_summaries.Evidence``):

``feature``   a model feature named verbatim -> must be in the pack, and on the
              side (raising / lowering the risk) the sentence puts it
``cause``     a cause concept named in free wording (vibration, cooling,
              battery, ...) -> the concept must belong to a feature that pushes
              the risk up in the pack, or to a linked ticket when the clause
              attributes it to a note
``number``    every number -> must equal a pack number at the precision written
``quote``     quoted text -> must be a substring of a linked ticket
``reading``   "classifier reading: X" -> a linked ticket must have that reading
``notes``     statements that notes exist / do not exist / are a driver

The check is lexical and deliberately strict. It reads the *text*, not the
generator's internals, so it applies equally to template and LLM summaries.
What it cannot judge (tone, causal strength, an unlisted synonym) is listed in
``docs/W03_Alert_Explanations.md`` and covered by the manual read.

``corrupt`` builds deliberately unfaithful variants; the share of them the
checker flags is its measured sensitivity.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ingen_fleet_health.explain.alert_summaries import (
    DRIVERS_IN_TEXT,
    RATIO_UNITS,
    Attribution,
    Evidence,
    describe_attribution,
    fmt,
    ticket_concepts,
)
from ingen_fleet_health.explain.shap_utils import CLASS_CONCEPT
from ingen_fleet_health.text import TAXONOMY

CONCEPT_LEXICON: dict[str, str] = {
    "vibration": r"vibrat|bearing|drivetrain|drive train|wheel|mechanical wear|gearbox",
    "thermal": r"overheat|cooling|thermal|motor[- ]temperature|motor heating|running hot|\bfan\b|\bhot\b",
    "battery": (r"battery|internal resistance|\bcells?\b|state[- ]of[- ]health|\bsoh\b|voltage|state of charge"
                r"|depth of (?:the last )?discharge"),
    "energy": r"energy|power draw|power consumption|electrical power",
    "sensor_drift": (r"sensor drift|sensor offset|temperature[- ]sensor|sensor fault|faulty sensor|miscalibrat"
                     r"|thermistor"),
    "environment": r"ambient|weather|terrain(?![- ]?normali)|\broute\b",   # 'terrain-normalised' = adjusted for terrain
    "data_quality": r"telemetry gaps?|data gaps?|dropout|connectivity|telemetry missing|missing telemetry|network",
    "usage": r"\bduty\b|charge sessions|equivalent full cycles|odometer|hour of day",
    "maintenance_history": r"repair|service tickets?|maintenance",
    "operator_notes": r"operator(?:'s)? notes?|\bnotes?\b|\btickets?\b",
    "user_error": r"user error|operational error|operator error|misuse",
}
MITIGATING = re.compile(r"lower|reduc|downward|offset|counter|mitigat|push(?:es|ed|ing)? (?:\w+ ){0,3}down"
                        r"|pull(?:s|ed|ing)? (?:\w+ ){0,3}down|decreas\w* the (?:risk|score)")
UPWARD = re.compile(r"push(?:es|ed|ing)? (?:\w+ ){0,3}up|rais|increas|elevat|driv|contribut")
NEGATION = re.compile(r"\bno\b|\bnot\b|\bnone\b|\bwithout\b|n't\b|\babsence of\b|\black of\b")
DRIVER_CUE = re.compile(r"driver|contribut|push|rais|because|due to|caus|explain")
NOT_A_DRIVER = re.compile(r"not (?:\w+ ){0,3}(?:driver|contribut)|n't (?:\w+ ){0,3}contribut")
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w]*\d)")
QUOTE = re.compile(r"[\"“]([^\"”]{6,}?)[\"”]")
NEUTRAL = [
    r"\d{4}-\d{2}-\d{2}",
    r"\b\d{1,2}:\d{2}\b",
    r"\b(?:1|6|24|72)\s*-?\s*(?:h|hr|hrs|hours?)\b",
    r"\b7\s*-?\s*(?:d|days?)\b",
]


@dataclass
class Claim:
    kind: str
    text: str
    supported: bool
    detail: str


@dataclass
class FaithfulnessResult:
    alert_id: str
    generator: str
    faithful: bool
    claims: list[Claim]

    @property
    def unsupported(self) -> list[Claim]:
        return [c for c in self.claims if not c.supported]

    def to_dict(self) -> dict[str, Any]:
        return {"alert_id": self.alert_id, "generator": self.generator, "faithful": self.faithful,
                "claims": len(self.claims), "unsupported": [asdict(c) for c in self.unsupported]}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def allowed_numbers(ev: Evidence) -> list[float]:
    out = [ev.risk, ev.threshold, 100 * ev.risk, 100 * ev.threshold]
    for a in ev.positive + ev.negative:
        out.append(a.shap)
        for v in (a.value, a.value_72h_ago):
            if v is None:
                continue
            out += [v, 100 * v]
            if a.unit in RATIO_UNITS:
                out.append(100 * (v - 1.0))
        if a.value is not None and a.value_72h_ago is not None:
            out.append(a.value - a.value_72h_ago)
    for t in ev.tickets:
        out += [t.age_hours, t.age_hours / 24.0, t.predicted_probability, 100 * t.predicted_probability]
    out += [float(len(ev.notes)), float(len(ev.tickets)), float(len(ev.tickets) - len(ev.notes)),
            float(len(ev.positive[:DRIVERS_IN_TEXT])), float(len(ev.positive)), float(len(ev.negative))]
    return out


def _number_supported(token: str, allowed: list[float]) -> bool:
    value = float(token)
    decimals = len(token.split(".")[1]) if "." in token else 0
    tol = 0.5 * 10.0 ** (-decimals) + 1e-9
    return any(abs(value - a) <= tol or abs(value - abs(a)) <= tol for a in allowed)


CONTRAST = re.compile(r";\s*(?:while|whereas|but|however)\b|,?\s*\b(?:but|while|whereas|although|however)\b")


def _feature_directions(work: str, marker: str) -> list[str]:
    """Direction ("up" / "down") the text states for each occurrence of a feature marker.

    The direction cue is looked for in the contrast segment that holds the
    marker (a sentence split at but / while / whereas / although / however);
    if that segment has no cue, the sentence decides. Feature phrases are
    already replaced by markers, so words inside a phrase are not cues.
    """
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", work):
        if marker not in sentence:
            continue
        sentence_down = bool(MITIGATING.search(sentence.lower()))
        for segment in CONTRAST.split(sentence):
            if marker not in segment:
                continue
            low = segment.lower()
            if MITIGATING.search(low):
                out.append("down")
            elif UPWARD.search(low):
                out.append("up")
            else:
                out.append("down" if sentence_down else "up")
    return out


def _clauses(text: str) -> list[str]:
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        out += [c.strip() for c in re.split(r";|, but |\bbut\b|\balthough\b|\bhowever\b|\bwhile\b|\bwhereas\b", sentence)
                if c.strip()]
    return out


CHECKER_VERSION = 2
# v2 (2026-11-12, after the manual read of the first full run): US spellings of
# feature-phrase words, quoted category names, "no fault was found", "terrain-
# normalised" and the unit named in other words ("rover 46") no longer raise
# false flags. See tests/w03/test_explain.py.
US_SPELLING = re.compile(r"\b(normali|optimi|minimi|maximi|characteri)z(e|ed|es|ing|ation)\b", re.IGNORECASE)
NFF_WORDING = re.compile(r"no faults? (?:was |were |has been |is )?found", re.IGNORECASE)
CATEGORY_NAMES = {_c.replace("_", " ") for _c in TAXONOMY}


def check_summary(summary: str, ev: Evidence, generator: str = "") -> FaithfulnessResult:
    claims: list[Claim] = []
    summary = US_SPELLING.sub(lambda m: f"{m.group(1)}s{m.group(2)}", summary)
    work = summary

    # 1. quotes must come from a linked ticket (a quoted category name is a category mention, not a quote)
    ticket_texts = [_norm(t.text) for t in ev.tickets]
    for m in QUOTE.finditer(summary):
        quoted = _norm(m.group(1))
        if quoted in CATEGORY_NAMES:
            continue
        ok = bool(quoted) and any(quoted in t for t in ticket_texts)
        claims.append(Claim("quote", m.group(1), ok, "matches a linked ticket" if ok else "no linked ticket has this text"))
    work = QUOTE.sub(lambda m: f" {m.group(1)} " if _norm(m.group(1)) in CATEGORY_NAMES else " QUOTED ", work)

    # 2. model features named verbatim. Matched *before* classifier readings,
    #    because a feature phrase ("... notes read as 'thermal' ...") may itself
    #    contain reading-like wording. Every occurrence is replaced by a numbered
    #    marker so that the words inside a phrase ("offset", "lowest") never count
    #    as a direction cue.
    pack = [(a, "positive") for a in ev.positive] + [(a, "negative") for a in ev.negative]
    mentions: list[tuple[str, Attribution, str]] = []
    for k, (a, side) in enumerate(sorted(pack, key=lambda x: -len(x[0].phrase))):
        marker = f" FEATURE{k}X "
        pattern = re.compile(re.escape(a.phrase), re.IGNORECASE)
        if pattern.search(work):
            mentions.append((marker.strip(), a, side))
            work = pattern.sub(marker, work)
    for marker, a, side in mentions:
        for direction in _feature_directions(work, marker):
            says_down = direction == "down"
            ok = (side == "negative") if says_down else (side == "positive")
            claims.append(Claim("feature", a.phrase, ok,
                                f"in the pack as a {side} attribution (SHAP {a.shap:+.3f})" if ok
                                else f"stated as {'lowering' if says_down else 'raising'} the risk "
                                     f"but SHAP is {a.shap:+.3f}"))
    work = re.sub(r"FEATURE\d+X", "FEATURE", work)

    # 3. classifier readings
    classes = "|".join(re.escape(c.replace("_", " ")) for c in TAXONOMY)
    reading = re.compile(rf"(?:classifier reading:?|read(?:s|ing)? as|classified as)\s*['\"]?({classes})", re.IGNORECASE)
    for m in reading.finditer(work):
        cls = m.group(1).lower().replace(" ", "_")
        ok = any(t.predicted_class == cls for t in ev.tickets)
        claims.append(Claim("reading", m.group(0), ok, "a linked ticket has this reading" if ok
                            else "no linked ticket has this classifier reading"))
    work = reading.sub(" READING ", work)
    work = NFF_WORDING.sub(" NFFCLASS ", work)

    # 4. numbers
    numeric_text = work.replace(ev.unit_id, " ")
    unit_no = re.sub(r"\D", "", ev.unit_id).lstrip("0")
    if unit_no:   # the unit named in other words ("rover 46", "Unit 46") is not a numeric claim
        numeric_text = re.sub(rf"\b(?:rover|unit|robot|r)[ _-]?0*{unit_no}\b", " ", numeric_text, flags=re.IGNORECASE)
    for neutral in NEUTRAL:
        numeric_text = re.sub(neutral, " ", numeric_text, flags=re.IGNORECASE)
    allowed = allowed_numbers(ev)
    for m in NUMBER.finditer(numeric_text):
        ok = _number_supported(m.group(0), allowed)
        claims.append(Claim("number", m.group(0), ok, "equals a pack value at the written precision" if ok
                            else "no pack value rounds to this number"))

    # 5. cause concepts in free wording
    positive_concepts = {a.concept for a in ev.positive}
    for a in ev.positive:              # a note-class text feature also supports that class's concept
        mapped = CLASS_CONCEPT.get(a.note_class) if a.note_class else None
        if mapped:
            positive_concepts.add(mapped)
    negative_concepts = {a.concept for a in ev.negative}
    from_tickets = ticket_concepts(ev)
    top_driver_concepts = {a.concept for a in ev.positive[:DRIVERS_IN_TEXT]}
    for clause in _clauses(work):
        low = clause.lower()
        found = [c for c, pattern in CONCEPT_LEXICON.items() if re.search(pattern, low)]
        if not found:
            continue
        negated = bool(NEGATION.search(low))
        mitigating = bool(MITIGATING.search(low))
        has_note_wording = "operator_notes" in found
        for concept in found:
            if concept == "operator_notes":
                if negated and NOT_A_DRIVER.search(low):
                    ok = "operator_notes" not in top_driver_concepts
                    claims.append(Claim("notes", clause, ok, "notes are not a top driver" if ok
                                        else "a text feature is among the top drivers"))
                elif negated:
                    ok = len(ev.notes) == 0
                    claims.append(Claim("notes", clause, ok, "no linked operator note" if ok
                                        else f"{len(ev.notes)} linked operator note(s) exist"))
                elif DRIVER_CUE.search(low):     # "notes are a driver": the attribution decides
                    ok = "operator_notes" in positive_concepts
                    claims.append(Claim("notes", clause, ok, "a note feature is a positive attribution" if ok
                                        else "no note feature raises the risk"))
                else:                            # "the unit has notes": a linked ticket must exist
                    ok = len(ev.tickets) > 0
                    claims.append(Claim("notes", clause, ok, "linked ticket(s) exist" if ok
                                        else "no linked operator note or service ticket"))
                continue
            if negated:
                continue                      # "no sign of X" is not a stated cause
            if mitigating:
                ok = concept in (negative_concepts | positive_concepts | from_tickets)
            else:
                ok = concept in positive_concepts or (has_note_wording and concept in from_tickets)
            claims.append(Claim("cause", f"{concept}: {clause}", ok,
                                "concept is in the attributions / linked tickets" if ok
                                else "concept is not among the features raising the risk nor in a linked ticket"))
    return FaithfulnessResult(ev.alert_id, generator, all(c.supported for c in claims), claims)


# --- negative controls -------------------------------------------------------
INVENTED_CAUSE = {
    "battery": "The underlying cause is a degrading battery cell.",
    "vibration": "The underlying cause is wheel bearing wear.",
    "thermal": "The underlying cause is a failing cooling fan.",
    "data_quality": "The underlying cause is a run of telemetry gaps.",
    "sensor_drift": "The underlying cause is a faulty sensor.",
}
CORRUPTIONS = ("invented_cause", "wrong_number", "invented_quote", "reversed_direction")


def corrupt(summary: str, ev: Evidence, kind: str, rng: np.random.Generator) -> str | None:
    """A deliberately unfaithful variant of a faithful summary, or None when
    this corruption cannot be built for this alert."""
    if kind == "invented_cause":
        # every concept the checker may legitimately accept is excluded, including the
        # category concept of a txt_decay_<class> attribution
        supported = {a.concept for a in ev.positive + ev.negative} | ticket_concepts(ev)
        supported |= {str(CLASS_CONCEPT[a.note_class]) for a in ev.positive + ev.negative
                      if a.note_class and CLASS_CONCEPT.get(a.note_class)}
        options = [c for c in INVENTED_CAUSE if c not in supported]
        return f"{summary} {INVENTED_CAUSE[str(rng.choice(options))]}" if options else None
    if kind == "wrong_number":
        for a in ev.positive[:DRIVERS_IN_TEXT]:
            if a.value is None or a.value == 0:
                continue
            old, new = fmt(a.value), fmt(a.value * 1.6 + (0.37 if abs(a.value) < 1 else 3.7))
            marker = f"{a.phrase} at {old}"
            if marker in summary and old != new:
                return summary.replace(marker, f"{a.phrase} at {new}", 1)
        return None
    if kind == "invented_quote":
        return f"{summary} An operator wrote \"loud knocking from the gearbox since Monday\"."
    if kind == "reversed_direction":
        if not ev.negative:
            return None
        return f"{summary} The score is also pushed up by {describe_attribution(ev.negative[0])}."
    raise ValueError(f"Unknown corruption: {kind}")
