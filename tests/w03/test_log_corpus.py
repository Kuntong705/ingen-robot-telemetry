import numpy as np
import pandas as pd
import pytest

from ingen_fleet_health.labels import load_fault_events
from ingen_fleet_health.simulate.fleet import simulate_fleet_v02
from ingen_fleet_health.text import TAXONOMY
from ingen_fleet_health.text.corpus import (
    FAULT_CATEGORY,
    OBSERVABLE_COLUMNS,
    generate_corpus,
    load_corpus,
    unseen_phrase_ids,
    write_corpus,
)
from ingen_fleet_health.text.noise import abbreviate, apply_noise, paraphrase
from ingen_fleet_health.text.templates import phrase_ids


@pytest.fixture(scope="module")
def sim_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus_sim")
    simulate_fleet_v02(out, units=12, days=20, seed=5)
    return out


@pytest.fixture(scope="module")
def corpus(sim_dir):
    return generate_corpus(sim_dir)


def test_corpus_is_reproducible(sim_dir, corpus):
    again, truth_again, _ = generate_corpus(sim_dir)
    pd.testing.assert_frame_equal(corpus[0], again)
    pd.testing.assert_frame_equal(corpus[1], truth_again)


def test_schema_taxonomy_and_synthetic_flag(corpus):
    frame, truth, manifest = corpus
    assert list(frame.columns) == OBSERVABLE_COLUMNS
    assert set(frame["label"]) <= set(TAXONOMY)
    assert frame["is_synthetic"].all() and manifest["is_synthetic"] is True
    assert frame["ticket_id"].is_unique and len(frame) == len(truth)
    assert set(frame["doc_type"]) <= {"operator_note", "service_ticket"}
    assert (frame["text"].str.len() > 3).all()
    # nothing from the hidden truth table is in the observable file
    assert not {"source", "phase", "fault_id", "severity_at_creation", "phrase_id"} & set(frame.columns)


def test_times_are_inside_the_run_and_labels_come_later(sim_dir, corpus):
    frame, _, _ = corpus
    created = pd.to_datetime(frame["created_at_utc"], utc=True)
    available = pd.to_datetime(frame["label_available_utc"], utc=True)
    events = load_fault_events(sim_dir / "ground_truth" / "fault_events.csv")
    assert created.min() >= events["initiation_utc"].min() - pd.Timedelta(days=30)
    assert (available >= created).all()
    assert created.is_monotonic_increasing


def test_fault_notes_follow_the_fault_timeline(sim_dir, corpus):
    frame, truth, _ = corpus
    events = load_fault_events(sim_dir / "ground_truth" / "fault_events.csv").set_index("fault_id")
    merged = frame.merge(truth, on="ticket_id")
    notes = merged[merged["source"] == "fault_symptom"]
    assert len(notes) > 0
    for row in notes.itertuples():
        ev = events.loc[row.fault_id]
        created = pd.Timestamp(row.created_at_utc)
        assert row.label == FAULT_CATEGORY[ev.fault_type]
        assert ev.initiation_utc <= created < ev.repair_start_utc
        assert (row.phase == "pre_onset") == (created < ev.onset_utc)
        # the category is confirmed only when the repair closes
        assert pd.Timestamp(row.label_available_utc) == ev.repair_end_utc.floor("min")
        assert 0.0 <= row.severity_at_creation <= 1.0


def test_no_fault_found_complaints_only_on_healthy_units(sim_dir, corpus):
    frame, truth, _ = corpus
    events = load_fault_events(sim_dir / "ground_truth" / "fault_events.csv")
    merged = frame.merge(truth, on="ticket_id")
    for row in merged[merged["source"] == "nff_complaint"].itertuples():
        created = pd.Timestamp(row.created_at_utc)
        busy = events[(events["unit_id"] == row.unit_id) & (events["initiation_utc"] <= created)
                      & (events["repair_end_utc"] > created)]
        assert busy.empty and row.label == "no_fault_found"


def test_notice_rate_only_changes_fault_notes_and_is_nested(sim_dir):
    low, low_truth, _ = generate_corpus(sim_dir, notice_rate=0.5)
    high, high_truth, _ = generate_corpus(sim_dir, notice_rate=3.0)

    def keyed(frame, truth, fault):
        m = frame.merge(truth, on="ticket_id")
        m = m[(m["source"] == "fault_symptom") == fault]
        return set(zip(m["unit_id"], m["created_at_utc"], m["text"], strict=True))

    assert keyed(low, low_truth, False) == keyed(high, high_truth, False)      # other processes untouched
    assert keyed(low, low_truth, True) <= keyed(high, high_truth, True)        # common random numbers
    assert len(keyed(high, high_truth, True)) > len(keyed(low, low_truth, True))


def test_write_and_load_round_trip(sim_dir, corpus, tmp_path):
    frame, truth, manifest = corpus
    (tmp_path / "ground_truth").mkdir()
    write_corpus(tmp_path, frame, truth, manifest)
    loaded = load_corpus(tmp_path)
    assert len(loaded) == len(frame)
    assert str(loaded["created_at_utc"].dtype).startswith("datetime64") and loaded["created_at_utc"].dt.tz is not None


def test_unseen_phrases_are_a_stable_subset():
    held = unseen_phrase_ids(0.2, seed=303)
    assert held == unseen_phrase_ids(0.2, seed=303)
    assert set(held) < set(phrase_ids())
    assert {h.split(":")[0] for h in held} >= {"battery", "drivetrain", "thermal", "connectivity"}


def test_paraphrase_respects_word_boundaries():
    rng = np.random.default_rng(0)
    for _ in range(50):
        out = paraphrase("reconnects every few minutes", rng)
        assert "every" in out          # "very" inside "every" must never be replaced


def test_noise_is_seeded_and_counts_changes():
    cfg = {"abbreviation_probability": 1.0, "typo_probability": 0.0, "lowercase_fraction": 0.0,
           "uppercase_fraction": 0.0, "drop_punctuation_fraction": 0.0}
    a, stats = apply_noise("battery temperature warning, no fault found", cfg, np.random.default_rng(1))
    b, _ = apply_noise("battery temperature warning, no fault found", cfg, np.random.default_rng(1))
    assert a == b
    assert "batt" in a.lower() and "temp" in a.lower() and "NFF" in a
    assert stats["n_abbrev"] >= 3 and stats["n_typos"] == 0
    text, n = abbreviate("please check the sensor", 0.0, np.random.default_rng(0))
    assert text == "please check the sensor" and n == 0
