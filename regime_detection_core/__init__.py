"""Core domain: HMM-based market regime detection."""

from regime_detection_core.model import (
    fit_hmm,
    decode_states,
    state_probabilities,
    label_regimes,
    build_regime_df,
    walk_forward_hmm,
    select_n_states,
    compute_bic,
    _group_states_by_regime,
)

__all__ = [
    "fit_hmm",
    "decode_states",
    "state_probabilities",
    "label_regimes",
    "build_regime_df",
    "walk_forward_hmm",
    "select_n_states",
    "compute_bic",
]
