"""Grounded alert summaries and the faithfulness checker (Week 3)."""

import numpy as np
import pandas as pd
import pytest

from ingen_fleet_health.explain.alert_summaries import (
    Attribution,
    Evidence,
    LinkedTicket,
    TemplateSummariser,
    build_evidence,
    build_prompt,
    describe_attribution,
    note_behind,
)
from ingen_fleet_health.explain.faithfulness import CORRUPTIONS, check_summary, corrupt
from ingen_fleet_health.explain.shap_utils import (
    feature_info,
    global_importance,
    tree_shap,
)
from ingen_fleet_health.text import TAXONOMY


def attr(feature: str, value: float | None, shap: float, before: float | None = None) -> Attribution:
    info = feature_info(feature)
    return Attribution(feature, info.phrase, info.unit, info.concept, value, before, shap, info.note_class)


def note(tid: str, age: float, text: str, cls: str, p: float = 0.8, doc_type: str = "operator_note") -> LinkedTicket:
    return LinkedTicket(tid, doc_type, age, text, cls, p)


def pack(positive: list[Attribution], negative: list[Attribution] | None = None,
         tickets: list[LinkedTicket] | None = None) -> Evidence:
    return Evidence("A01", "Rover_007", "2026-11-20 06:00", "c30_g1", 0.31, 0.12, -3.0, -0.8,
                    positive, negative or [], tickets or [])


VIB = attr("vib_norm_ratio_to_baseline", 1.42, 0.61, 1.18)
THERM = attr("motor_temp_resid_patrol_mean_24h", 3.4, 0.35)
DOCK = attr("motor_temp_resid_dock_mean_24h", 2.1, 0.20)          # phrase contains "offset"
SOC_NEG = attr("min_soc_24h", 41.0, -0.15)                         # phrase contains "lowest"
TXT_THERMAL = attr("txt_decay_thermal", 0.74, 0.28)


def test_template_summary_is_faithful_and_two_to_three_sentences():
    ev = pack([VIB, THERM, DOCK], [SOC_NEG])
    text = TemplateSummariser().summarise(ev)
    result = check_summary(text, ev)
    assert result.faithful, result.unsupported
    assert 2 <= text.count(". ") + 1 <= 3


def test_words_inside_a_feature_phrase_are_not_direction_cues():
    """Defect 1: 'offset' inside 'docked motor-temperature offset' made every
    feature of that sentence read as lowering the risk."""
    ev = pack([DOCK, VIB, THERM], [SOC_NEG])
    text = "The score is pushed up mainly by " + "; ".join(describe_attribution(a) for a in (DOCK, VIB, THERM)) + "."
    result = check_summary(text, ev)
    assert result.faithful, result.unsupported
    # the lowest-SoC phrase ("lowest") also must not flip a negative attribution to positive
    text2 = f"{text} It is pulled down by {describe_attribution(SOC_NEG)}."
    assert check_summary(text2, ev).faithful


def test_direction_is_read_per_contrast_segment():
    ev = pack([VIB], [SOC_NEG])
    ok = f"{VIB.phrase} raises the risk, while {SOC_NEG.phrase} lowers it."
    assert check_summary(ok, ev).faithful
    swapped = f"{SOC_NEG.phrase} raises the risk, while {VIB.phrase} lowers it."
    bad = check_summary(swapped, ev)
    assert not bad.faithful
    assert {c.text for c in bad.unsupported} == {VIB.phrase, SOC_NEG.phrase}


def test_text_feature_phrase_is_not_parsed_as_a_classifier_reading():
    """Defect 2: the phrase "recent operator notes read as 'thermal'" is a
    feature, not a claim that a linked ticket was classified as thermal."""
    ev = pack([TXT_THERMAL, VIB])          # no linked ticket at all
    text = f"The score is pushed up mainly by {describe_attribution(TXT_THERMAL)}; and {describe_attribution(VIB)}."
    result = check_summary(text, ev)
    assert not any(c.kind == "reading" for c in result.claims)
    assert result.faithful, result.unsupported


def test_quoted_note_is_the_one_behind_the_text_driver():
    """Defect 3a: the newest note was quoted even when another note fed the
    attributed txt_decay_<class> feature."""
    newest = note("T2", 5.0, "app shows offline again", "connectivity")
    source = note("T1", 30.0, "motor housing very hot after loop", "thermal")
    ev = pack([TXT_THERMAL, VIB], tickets=[newest, source])
    assert note_behind(TXT_THERMAL, ev) is source
    text = TemplateSummariser().summarise(ev)
    assert source.text in text and newest.text not in text
    assert check_summary(text, ev).faithful


def test_no_quote_when_no_note_feeds_the_driver():
    ev = pack([TXT_THERMAL, VIB], tickets=[note("T2", 5.0, "app shows offline again", "connectivity")])
    assert note_behind(TXT_THERMAL, ev) is None
    text = TemplateSummariser().summarise(ev)
    assert '"' not in text
    assert check_summary(text, ev).faithful


def test_ratio_at_reference_level_is_qualified():
    """Defect 3b: '1.00 x baseline' must not read as a deviation."""
    flat = attr("energy_ratio_to_baseline", 1.004, 0.12)
    assert "essentially at the reference level" in describe_attribution(flat)
    assert "reference level" not in describe_attribution(VIB)
    ev = pack([flat, VIB])
    assert check_summary(TemplateSummariser().summarise(ev), ev).faithful


@pytest.mark.parametrize("kind", CORRUPTIONS)
def test_every_corruption_is_flagged(kind):
    ev = pack([VIB, DOCK, TXT_THERMAL], [SOC_NEG],
              tickets=[note("T1", 30.0, "motor housing very hot after loop", "thermal")])
    base = TemplateSummariser().summarise(ev)
    assert check_summary(base, ev).faithful
    bad = corrupt(base, ev, kind, np.random.default_rng(0))
    assert bad is not None and bad != base
    assert not check_summary(bad, ev).faithful, (kind, bad)


def test_invented_cause_never_uses_a_concept_the_pack_supports():
    """Corruption bug behind the 19/20 smoke result: a txt_decay_<class>
    attribution supports that class's concept, so it is not an invented cause."""
    ev = pack([VIB, attr("txt_decay_connectivity", 0.5, 0.2)])
    rng = np.random.default_rng(0)
    for _ in range(30):
        bad = corrupt("x", ev, "invented_cause", rng)
        assert bad is not None and "telemetry gaps" not in bad and "bearing" not in bad


def test_numbers_must_match_the_pack():
    ev = pack([VIB])
    assert check_summary(f"{VIB.phrase} is at 1.42 x baseline and pushes the risk up.", ev).faithful
    assert not check_summary(f"{VIB.phrase} is at 1.87 x baseline and pushes the risk up.", ev).faithful


def test_invented_quote_and_note_claims():
    ev = pack([VIB])
    assert not check_summary('An operator wrote "grinding from the left wheel".', ev).faithful
    assert check_summary("No operator note was linked to this unit in the previous 7 days.", ev).faithful
    with_note = pack([VIB], tickets=[note("T1", 3.0, "noisy wheel", "drivetrain")])
    assert not check_summary("No operator note was linked to this unit in the previous 7 days.", with_note).faithful


def test_build_evidence_links_only_past_week_tickets_and_never_labels():
    cols = ["vib_norm_ratio_to_baseline", "min_soc_24h", "txt_decay_thermal"]
    t = pd.Timestamp("2026-11-20T06:00:00Z")
    row = pd.Series({"unit_id": "Rover_007", "t": t, cols[0]: 1.4, cols[1]: 40.0, cols[2]: 0.7})
    tickets = pd.DataFrame({
        "ticket_id": ["old", "recent", "future"],
        "created_at_utc": [t - pd.Timedelta(days=9), t - pd.Timedelta(hours=10), t + pd.Timedelta(hours=1)],
        "doc_type": ["operator_note"] * 3, "text": ["a", "b", "c"],
        "label": ["battery", "battery", "battery"],          # must never be read
    })
    proba = np.zeros((3, len(TAXONOMY)))
    proba[:, TAXONOMY.index("thermal")] = 1.0
    ev = build_evidence("A01", row, None, np.array([0.5, -0.2, 0.1]), cols, -3.0, 0.3, 0.1, "f", tickets, proba)
    assert [x.ticket_id for x in ev.tickets] == ["recent"]
    assert ev.tickets[0].predicted_class == "thermal"
    assert [a.feature for a in ev.positive] == [cols[0], cols[2]]
    assert [a.feature for a in ev.negative] == [cols[1]]
    assert "battery" not in build_prompt(ev)[1].split("EVIDENCE")[1].split("DOWN")[0]


def test_tree_shap_is_additive():
    lightgbm = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 3)), columns=["vib_norm_mean_24h", "min_soc_24h", "odometer_km"])
    y = (X["vib_norm_mean_24h"] + 0.3 * rng.normal(size=400) > 0.8).astype(int)
    model = lightgbm.LGBMClassifier(n_estimators=40, verbose=-1).fit(X, y)
    values, base = tree_shap(model, X)
    raw = model.predict(X, raw_score=True)
    assert np.max(np.abs(values.sum(axis=1) + base - raw)) < 1e-6
    imp = global_importance(values, list(X.columns))
    assert imp.iloc[0]["feature"] == "vib_norm_mean_24h"


def test_checker_v2_us_spelling_matches_feature_phrase():
    """False flag found in the manual read: 'terrain-normalized' broke verbatim
    feature matching and 'terrain' was then read as an environment cause."""
    ev = pack([VIB])
    text = f"{VIB.phrase.replace('normalised', 'normalized')} at 1.42 x baseline pushes the risk up."
    result = check_summary(text, ev)
    assert result.faithful, result.unsupported
    assert any(c.kind == "feature" for c in result.claims)


def test_checker_v2_quoted_category_name_is_not_a_quote():
    ev = pack([TXT_THERMAL], tickets=[note("T1", 30.0, "motor housing very hot after loop", "thermal")])
    assert check_summary('Recent operator notes mentioning "thermal" push the risk up.', ev).faithful
    assert not check_summary('An operator wrote "thermal runaway smell".', ev).faithful


def test_checker_v2_no_fault_was_found_is_a_category_not_a_negation():
    ev = pack([VIB], [attr("txt_decay_no_fault_found", 0.4, -0.3)],
              tickets=[note("T1", 20.0, "checked, all fine", "no_fault_found")])
    text = "Recent operator notes stating no fault was found slightly lower the risk."
    assert check_summary(text, ev).faithful


def test_checker_v2_terrain_normalised_is_not_a_terrain_cause_and_unit_number_is_neutral():
    ev = pack([VIB])          # unit Rover_007
    text = "Rover 7 shows terrain-normalized vibration well above its own baseline, which pushes the risk up."
    assert check_summary(text, ev).faithful, check_summary(text, ev).unsupported
    assert not check_summary("Rough terrain on the route pushes the risk up.", ev).faithful
    assert not check_summary("Rover 8 shows high vibration, which pushes the risk up.", ev).faithful
