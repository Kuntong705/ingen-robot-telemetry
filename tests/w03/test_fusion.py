"""Late / early fusion building blocks (Week 3)."""

import numpy as np
import pandas as pd
import pytest

from ingen_fleet_health.models.fusion import (
    ALL_ARMS,
    FUSION_ARMS,
    TELEMETRY_ARMS,
    StackedFusion,
    early_variants,
    fit_selected,
    health_score,
    noisy_or,
)
from ingen_fleet_health.text.features import TEXT_FEATURES


def test_arms_cover_the_ablation():
    """Telemetry only, text only and combined arms are all present, once."""
    assert set(TELEMETRY_ARMS) & set(FUSION_ARMS) == set()
    assert {"telemetry_lgbm", "text_only", "early_fusion", "late_noisy_or", "late_stack"} <= set(ALL_ARMS)


def test_noisy_or_is_a_monotone_union():
    a = np.array([0.0, 0.2, 0.5, 1.0])
    b = np.array([0.0, 0.5, 0.5, 0.3])
    out = noisy_or(a, b)
    assert np.allclose(out, [0.0, 0.6, 0.75, 1.0])
    assert np.all(out >= np.maximum(a, b) - 1e-12)
    assert np.allclose(noisy_or(a, np.zeros_like(a)), a)          # no text signal -> telemetry unchanged


def test_health_score_is_bounded_and_decreasing():
    risk = np.array([-0.1, 0.0, 0.25, 1.0, 1.3])
    hs = health_score(risk)
    assert np.allclose(hs, [100.0, 100.0, 75.0, 0.0, 0.0])
    assert np.all(np.diff(hs) <= 0)


def test_stacker_needs_enough_positives():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(size=200)
    y = np.zeros(200, dtype=int)
    y[:3] = 1
    stack = StackedFusion().fit(a, b, y)
    assert not stack.fitted
    assert all(np.isnan(stack.weights()))
    with pytest.raises(RuntimeError):
        stack.predict(a, b)


def test_stacker_weights_the_informative_modality():
    rng = np.random.default_rng(1)
    signal = rng.normal(size=600)
    y = (signal + 0.5 * rng.normal(size=600) > 1.0).astype(int)
    stack = StackedFusion().fit(signal, rng.normal(size=600), y)
    w_signal, w_noise = stack.weights()
    assert w_signal > 1.0 and abs(w_noise) < 0.3
    p = stack.predict(signal, np.zeros(600))
    assert np.all((p >= 0) & (p <= 1))


def test_early_variants_always_include_text_features():
    frame = pd.DataFrame(columns=["vib_norm_mean_24h", "odometer_km", *TEXT_FEATURES])
    variants = early_variants(frame, ["vib_norm_mean_24h", "odometer_km"], TEXT_FEATURES)
    assert set(variants) == {"all+text", "physics+text"}
    for cols in variants.values():
        assert cols[-len(TEXT_FEATURES):] == TEXT_FEATURES
        assert len(cols) == len(set(cols))


def test_fit_selected_picks_the_better_validation_variant():
    rng = np.random.default_rng(2)
    n = 1500
    frame = pd.DataFrame({"useful": rng.normal(size=n), "noise": rng.normal(size=n)})
    frame["y"] = (frame["useful"] + 0.4 * rng.normal(size=n) > 1.2).astype(int)
    fit, val = frame.iloc[:1000], frame.iloc[1000:]
    sel = fit_selected(fit, val, {"noise_only": ["noise"], "useful": ["useful"]}, seed=0)
    assert sel.name == "useful"
    assert sel.val_pr["useful"] > sel.val_pr["noise_only"]
    assert list(sel.gain.index) == ["useful"]
