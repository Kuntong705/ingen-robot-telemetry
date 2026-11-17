import numpy as np
import pandas as pd
import pytest

from ingen_fleet_health.experiments.w03_common import (
    TextData,
    check_text_fold,
    text_test_mask,
    text_train_mask,
)
from ingen_fleet_health.models.text_classifier import (
    TicketClassifier,
    TicketInputs,
    crossfit_fold,
    macro_f1,
)
from ingen_fleet_health.text import TAXONOMY
from ingen_fleet_health.text.features import (
    HALF_LIFE_HOURS,
    TEXT_FEATURES,
    asof_text_features,
)

T0 = pd.Timestamp("2026-11-01T00:00:00Z")
WORDS = {
    "battery": "battery drains fast low charge",
    "drivetrain": "grinding wheel vibration noise",
    "thermal": "motor overheating hot fan",
    "connectivity": "offline lost connection wifi",
    "no_fault_found": "all good nothing to report",
}


def toy_text(n_per_class: int = 24, n_groups: int = 4) -> TextData:
    rows = []
    for label, words in WORDS.items():
        for i in range(n_per_class):
            unit = i % (n_groups * 2)
            rows.append({"ticket_id": f"T{len(rows):04d}", "unit_id": f"Rover_{unit:03d}",
                         "created_at_utc": T0 + pd.Timedelta(hours=7 * i + len(rows) % 5),
                         "doc_type": "operator_note", "text": f"{words} case {i}", "label": label})
    corpus = pd.DataFrame(rows)
    corpus["label_available_utc"] = corpus["created_at_utc"] + pd.Timedelta(hours=12)
    group = (corpus["unit_id"].str[-3:].astype(int) % n_groups).to_numpy()
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(corpus), 8)).astype(np.float32)
    return TextData(corpus, TicketInputs(corpus["text"].to_numpy(dtype=object), emb), group)


def test_classifier_learns_and_keeps_taxonomy_order():
    text = toy_text()
    idx = np.arange(len(text.corpus))
    train, test = idx[idx % 3 != 0], idx[idx % 3 == 0]
    model = TicketClassifier("tfidf_lr", C=10.0).fit(text.inputs.take(train), text.labels[train])
    proba = model.predict_proba(text.inputs.take(test))
    assert proba.shape == (len(test), len(TAXONOMY))
    assert np.allclose(proba.sum(axis=1), 1.0)
    # classes never seen in training get probability 0, in the fixed column order
    assert proba[:, TAXONOMY.index("sensor")].max() == 0.0
    assert macro_f1(text.labels[test], model.predict(text.inputs.take(test))) > 0.6   # 5 of 7 classes exist


def test_majority_baseline_predicts_one_class():
    text = toy_text()
    labels = text.labels.copy()
    labels[:10] = "no_fault_found"
    model = TicketClassifier("majority").fit(text.inputs, labels)
    assert set(model.predict(text.inputs)) == {"no_fault_found"}


def test_crossfit_never_uses_test_group_labels():
    text = toy_text()
    train = text_train_mask(text, test_group=0, train_end=T0 + pd.Timedelta(days=30))
    assert not (text.group[train] == 0).any()
    a = crossfit_fold(text.inputs, text.labels, text.group, train, 0, kinds=("tfidf_lr",), c_grid=(10.0,))
    scrambled = text.labels.copy()
    scrambled[text.group == 0] = "sensor"                # corrupt every label of the test units
    b = crossfit_fold(text.inputs, scrambled, text.group, train, 0, kinds=("tfidf_lr",), c_grid=(10.0,))
    np.testing.assert_allclose(a.proba["tfidf_lr"], b.proba["tfidf_lr"])
    assert a.selected_kind == "tfidf_lr" and a.n_train == int(train.sum())
    with pytest.raises(AssertionError):
        crossfit_fold(text.inputs, text.labels, text.group, np.ones(len(text.corpus), dtype=bool), 0,
                      kinds=("tfidf_lr",), c_grid=(10.0,))


def test_crossfit_scores_training_units_out_of_group():
    """A training-unit ticket must be scored by a model that did not see its group:
    relabelling one training group changes the *other* groups' scores, not its own fit."""
    text = toy_text()
    train = text_train_mask(text, 0, T0 + pd.Timedelta(days=30))
    base = crossfit_fold(text.inputs, text.labels, text.group, train, 0, kinds=("tfidf_lr",), c_grid=(10.0,))
    changed = text.labels.copy()
    changed[text.group == 1] = "sensor"
    alt = crossfit_fold(text.inputs, changed, text.group, train, 0, kinds=("tfidf_lr",), c_grid=(10.0,))
    own = text.group == 1
    np.testing.assert_allclose(base.proba["tfidf_lr"][own], alt.proba["tfidf_lr"][own])
    assert not np.allclose(base.proba["tfidf_lr"][text.group == 2], alt.proba["tfidf_lr"][text.group == 2])


def test_text_split_respects_label_availability_and_embargo():
    text = toy_text()
    cutoff = T0 + pd.Timedelta(days=4)
    embargo = pd.Timedelta(hours=72)
    train = text_train_mask(text, 1, cutoff - embargo)
    test = text_test_mask(text, 1, cutoff, pd.Timedelta(days=10))
    assert (text.corpus.loc[train, "label_available_utc"] <= cutoff - embargo).all()
    assert (text.corpus.loc[test, "created_at_utc"] >= cutoff).all()
    check_text_fold(text, train, test, embargo)
    leaky = train | ((text.group != 1) & (text.corpus["created_at_utc"] > cutoff - pd.Timedelta(hours=10)).to_numpy())
    with pytest.raises(AssertionError):
        check_text_fold(text, leaky, test, embargo)
    with pytest.raises(AssertionError):
        check_text_fold(text, train | test, test, embargo)


def _tickets(rows):
    frame = pd.DataFrame(rows, columns=["unit_id", "created_at_utc", "doc_type"])
    proba = np.zeros((len(frame), len(TAXONOMY)))
    proba[:, TAXONOMY.index("drivetrain")] = 0.8
    proba[:, TAXONOMY.index("no_fault_found")] = 0.2
    return frame, proba


def test_text_features_are_as_of():
    grid = pd.DataFrame({"unit_id": "Rover_001", "t": pd.date_range(T0, periods=240, freq="h")})
    note_time = T0 + pd.Timedelta(hours=50, minutes=30)
    tickets, proba = _tickets([("Rover_001", note_time, "operator_note")])
    feats = asof_text_features(tickets, proba, grid)
    before = (grid["t"] < note_time).to_numpy()
    assert (feats.loc[before, "txt_decay_drivetrain"] == 0).all()
    assert (feats.loc[before, "txt_hours_since_note"] == 168.0).all()
    first = int(np.flatnonzero(~before)[0])
    age = (grid["t"].iloc[first] - note_time) / pd.Timedelta(hours=1)
    assert feats["txt_decay_drivetrain"].iloc[first] == pytest.approx(0.8 * 0.5 ** (age / HALF_LIFE_HOURS))
    assert feats["txt_fault_max_72h"].iloc[first] == pytest.approx(0.8)
    assert feats["txt_note_count_72h"].iloc[first] == 1
    old = (grid["t"] >= note_time + pd.Timedelta(hours=168)).to_numpy()
    assert (feats.loc[old, "txt_note_count_7d"] == 0).all()          # outside the 7-day window
    recent_only = (grid["t"] >= note_time + pd.Timedelta(hours=72)).to_numpy() & ~old
    assert (feats.loc[recent_only, "txt_note_count_72h"] == 0).all()
    assert (feats.loc[recent_only, "txt_note_count_7d"] == 1).all()


def test_future_tickets_and_other_units_do_not_change_earlier_rows():
    grid = pd.DataFrame({"unit_id": ["Rover_001"] * 100 + ["Rover_002"] * 100,
                         "t": list(pd.date_range(T0, periods=100, freq="h")) * 2})
    one, p1 = _tickets([("Rover_001", T0 + pd.Timedelta(hours=10), "operator_note")])
    two, p2 = _tickets([("Rover_001", T0 + pd.Timedelta(hours=10), "operator_note"),
                        ("Rover_001", T0 + pd.Timedelta(hours=60), "operator_note"),
                        ("Rover_001", T0 + pd.Timedelta(hours=61), "service_ticket")])
    a, b = asof_text_features(one, p1, grid), asof_text_features(two, p2, grid)
    early = ((grid["unit_id"] == "Rover_001") & (grid["t"] < T0 + pd.Timedelta(hours=60))).to_numpy()
    pd.testing.assert_frame_equal(a[early], b[early])
    other = (grid["unit_id"] == "Rover_002").to_numpy()
    assert (b.loc[other, [c for c in TEXT_FEATURES if c != "txt_hours_since_note"]].to_numpy() == 0).all()
    late = ((grid["unit_id"] == "Rover_001") & (grid["t"] >= T0 + pd.Timedelta(hours=61))).to_numpy()
    assert (b.loc[late, "txt_service_count_7d"] == 1).all()
    assert (b.loc[late, "txt_note_count_7d"] == 2).all()             # the service ticket is not a note


def test_text_features_do_not_need_labels():
    grid = pd.DataFrame({"unit_id": "Rover_001", "t": pd.date_range(T0, periods=5, freq="h")})
    tickets, proba = _tickets([("Rover_001", T0, "operator_note")])
    assert "label" not in tickets.columns
    assert list(asof_text_features(tickets, proba, grid).columns) == TEXT_FEATURES
