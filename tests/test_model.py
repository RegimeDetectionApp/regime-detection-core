"""Tests for regime_detection_core.model."""

import numpy as np
import pandas as pd
import pytest
from regime_detection_core import (
    fit_hmm, decode_states, state_probabilities,
    label_regimes, build_regime_df, compute_bic,
)


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    n = 300
    features = pd.DataFrame({
        "log_return": np.random.normal(0, 0.01, n),
        "vol_20d": np.abs(np.random.normal(0.15, 0.05, n)),
        "drawdown": -np.abs(np.random.normal(0.05, 0.03, n)),
    }, index=pd.bdate_range("2020-01-01", periods=n))
    return features


@pytest.fixture
def fitted_model(synthetic_data):
    return fit_hmm(synthetic_data, n_states=3)


def test_fit_hmm_returns_correct_components(fitted_model):
    assert fitted_model.n_components == 3


def test_decode_states_length(fitted_model, synthetic_data):
    states = decode_states(fitted_model, synthetic_data)
    assert len(states) == len(synthetic_data)
    assert set(states).issubset({0, 1, 2})


def test_state_probabilities_shape(fitted_model, synthetic_data):
    probs = state_probabilities(fitted_model, synthetic_data)
    assert probs.shape == (len(synthetic_data), 3)


def test_state_probabilities_sum_to_one(fitted_model, synthetic_data):
    probs = state_probabilities(fitted_model, synthetic_data)
    sums = probs.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


def test_label_regimes_produces_valid_labels(fitted_model, synthetic_data):
    states = decode_states(fitted_model, synthetic_data)
    label_map = label_regimes(states, synthetic_data)
    assert set(label_map.values()) == {"Crash", "High Volatility", "Low Volatility"}


def test_compute_bic_returns_float(fitted_model, synthetic_data):
    bic = compute_bic(fitted_model, synthetic_data.values)
    assert isinstance(bic, float)
    assert np.isfinite(bic)


def test_build_regime_df_columns(fitted_model, synthetic_data):
    states = decode_states(fitted_model, synthetic_data)
    probs = state_probabilities(fitted_model, synthetic_data)
    label_map = label_regimes(states, synthetic_data)
    rdf = build_regime_df(synthetic_data.index, states, probs, label_map)
    assert "regime" in rdf.columns
    assert "prob_crash" in rdf.columns
    assert "prob_high_vol" in rdf.columns
    assert "prob_low_vol" in rdf.columns
