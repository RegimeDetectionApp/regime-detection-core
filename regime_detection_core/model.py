"""
Hidden Markov Model for market regime detection.

Fits a Gaussian HMM to the engineered feature set and provides utilities
for extracting state sequences, posterior probabilities, and mapping the
latent states to interpretable regime labels (Low Volatility, High
Volatility, Crash).
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def fit_hmm(
    X: pd.DataFrame,
    n_states: int = 3,
    n_iter: int = 200,
    random_state: int = 42,
) -> GaussianHMM:
    """
    Fit a Gaussian HMM with full covariance to the feature matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Standardised feature matrix (T x K).
    n_states : int
        Number of hidden states.
    n_iter : int
        Maximum EM iterations.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    GaussianHMM
        Fitted model.
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-4,
    )
    model.fit(X.values)
    return model


def decode_states(model: GaussianHMM, X: pd.DataFrame) -> np.ndarray:
    """Viterbi-decoded most-likely state sequence."""
    return model.predict(X.values)


def state_probabilities(model: GaussianHMM, X: pd.DataFrame) -> np.ndarray:
    """
    Posterior state probabilities at each timestep (T x n_states).

    Uses the forward-backward algorithm internally.
    """
    return model.predict_proba(X.values)


# ── Regime labelling ──────────────────────────────────────────────────────

def _state_statistics(
    states: np.ndarray,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-state summary statistics used for label assignment.

    Returns a DataFrame indexed by state id with columns:
        mean_return, mean_vol, mean_drawdown, std_return
    """
    cols = ["log_return", "vol_20d", "drawdown"]
    df = features[cols].copy()
    df["state"] = states

    stats = df.groupby("state").agg(
        mean_return=("log_return", "mean"),
        std_return=("log_return", "std"),
        mean_vol=("vol_20d", "mean"),
        mean_drawdown=("drawdown", "mean"),
    )
    return stats


def label_regimes(
    states: np.ndarray,
    features: pd.DataFrame,
) -> dict[int, str]:
    """
    Map each hidden state integer to an interpretable regime label.

    Uses an asset-agnostic rank-based heuristic that works across equities,
    commodities, and other asset classes.  Supports any number of states
    (>= 2): three archetypal regimes are identified first, then any
    remaining states are assigned to their nearest archetype by volatility.

    1. Rank each state by mean return (ascending) and by mean volatility
       (descending).  The state with the highest composite rank is
       **Crash**.
    2. Among the rest, the highest-volatility state is **High Volatility**
       and the lowest is **Low Volatility**.
    3. Extra states are labeled by proximity to these three archetypes.

    Returns
    -------
    dict[int, str]
        Mapping from state id to one of
        {"Crash", "High Volatility", "Low Volatility"}.
    """
    stats = _state_statistics(states, features)
    n = len(stats)

    # Rank-based scoring: avoids hard-coded assumptions about return/drawdown
    # scales that break for non-equity assets (Gold, FX, etc.)
    stats["rank_return"] = stats["mean_return"].rank(ascending=True)
    stats["rank_vol"] = stats["mean_vol"].rank(ascending=False)
    stats["crash_rank"] = stats["rank_return"] + stats["rank_vol"]

    crash_state = stats["crash_rank"].idxmax()

    remaining = stats.drop(crash_state)

    if n == 2:
        # Only two states: crash + everything else
        other = remaining.index[0]
        return {crash_state: "Crash", other: "Low Volatility"}

    high_vol_state = remaining["mean_vol"].idxmax()
    low_vol_state = remaining["mean_vol"].idxmin()

    label_map = {
        crash_state: "Crash",
        high_vol_state: "High Volatility",
        low_vol_state: "Low Volatility",
    }

    # Assign any extra states (n > 3) to nearest archetype by volatility
    archetype_vols = {
        "Crash": stats.loc[crash_state, "mean_vol"],
        "High Volatility": stats.loc[high_vol_state, "mean_vol"],
        "Low Volatility": stats.loc[low_vol_state, "mean_vol"],
    }
    for state_id in stats.index:
        if state_id in label_map:
            continue
        vol = stats.loc[state_id, "mean_vol"]
        closest = min(archetype_vols, key=lambda k: abs(archetype_vols[k] - vol))
        label_map[state_id] = closest

    return label_map


def build_regime_df(
    dates: pd.DatetimeIndex,
    states: np.ndarray,
    probs: np.ndarray,
    label_map: dict[int, str],
) -> pd.DataFrame:
    """
    Assemble a tidy DataFrame of regime information.

    When n_states > 3, multiple HMM states may map to the same regime
    label.  Probabilities are summed across all states sharing a label.

    Columns
    -------
    state           : int — raw HMM state
    regime          : str — human label
    prob_crash      : float — posterior probability of Crash regime
    prob_high_vol   : float — posterior probability of High Volatility regime
    prob_low_vol    : float — posterior probability of Low Volatility regime
    """
    # Group state indices by regime label
    regime_states: dict[str, list[int]] = {"Crash": [], "High Volatility": [], "Low Volatility": []}
    for state_id, label in label_map.items():
        regime_states[label].append(state_id)

    # Sum probabilities for each regime (handles n_states > 3)
    prob_crash = probs[:, regime_states["Crash"]].sum(axis=1) if regime_states["Crash"] else np.zeros(len(dates))
    prob_high_vol = probs[:, regime_states["High Volatility"]].sum(axis=1) if regime_states["High Volatility"] else np.zeros(len(dates))
    prob_low_vol = probs[:, regime_states["Low Volatility"]].sum(axis=1) if regime_states["Low Volatility"] else np.zeros(len(dates))

    regime_df = pd.DataFrame(
        {
            "state": states,
            "regime": [label_map[s] for s in states],
            "prob_crash": prob_crash,
            "prob_high_vol": prob_high_vol,
            "prob_low_vol": prob_low_vol,
        },
        index=dates,
    )
    return regime_df


# ── Walk-forward validation ──────────────────────────────────────────────

def walk_forward_hmm(
    features: pd.DataFrame,
    n_states: int = 3,
    initial_train_pct: float = 0.50,
    step_days: int = 63,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expanding-window walk-forward regime detection.

    Avoids look-ahead bias by only fitting the HMM on data available up to
    each evaluation point.  Standardisation is performed per window using
    only training data.

    Parameters
    ----------
    features : pd.DataFrame
        Raw (un-standardised) feature matrix.
    n_states : int
        Number of hidden states.
    initial_train_pct : float
        Fraction of data used for the first training window.
    step_days : int
        Number of trading days to step forward each iteration.
    random_state : int
        Seed for HMM reproducibility.

    Returns
    -------
    regime_df : pd.DataFrame
        Regime labels and probabilities for out-of-sample dates only.
    oos_features : pd.DataFrame
        The raw feature rows corresponding to OOS dates.
    """
    n = len(features)
    initial_end = int(n * initial_train_pct)

    oos_regimes = []
    oos_probs_list = []
    oos_states_list = []
    oos_indices = []

    cursor = initial_end
    while cursor < n:
        step_end = min(cursor + step_days, n)

        # Train on everything up to cursor
        train = features.iloc[:cursor]
        test = features.iloc[cursor:step_end]

        # Standardise using only training statistics
        scaler = StandardScaler()
        train_scaled = pd.DataFrame(
            scaler.fit_transform(train.values),
            index=train.index, columns=train.columns,
        )
        test_scaled = pd.DataFrame(
            scaler.transform(test.values),
            index=test.index, columns=test.columns,
        )

        # Fit and predict
        model = fit_hmm(train_scaled, n_states=n_states, random_state=random_state)
        states = decode_states(model, test_scaled)
        probs = state_probabilities(model, test_scaled)

        # Label regimes using training data statistics
        train_states = decode_states(model, train_scaled)
        label_map = label_regimes(train_states, train)

        # Group state indices by regime label (handles n_states > 3)
        regime_states: dict[str, list[int]] = {"Crash": [], "High Volatility": [], "Low Volatility": []}
        for state_id, label in label_map.items():
            regime_states[label].append(state_id)

        for i, idx in enumerate(test.index):
            oos_indices.append(idx)
            oos_states_list.append(label_map[states[i]])
            oos_probs_list.append({
                "prob_crash": probs[i, regime_states["Crash"]].sum(),
                "prob_high_vol": probs[i, regime_states["High Volatility"]].sum(),
                "prob_low_vol": probs[i, regime_states["Low Volatility"]].sum(),
            })

        cursor = step_end

    regime_df = pd.DataFrame({
        "regime": oos_states_list,
        "prob_crash": [p["prob_crash"] for p in oos_probs_list],
        "prob_high_vol": [p["prob_high_vol"] for p in oos_probs_list],
        "prob_low_vol": [p["prob_low_vol"] for p in oos_probs_list],
    }, index=oos_indices)

    oos_features = features.loc[oos_indices]
    return regime_df, oos_features


# ── BIC model selection ──────────────────────────────────────────────────

def compute_bic(model: GaussianHMM, X: np.ndarray) -> float:
    """
    Bayesian Information Criterion for a fitted Gaussian HMM.

    BIC = -2 * log_likelihood + k * ln(n)

    where k is the number of free parameters:
        - Transition matrix: n_states * (n_states - 1)
        - Initial state probs: n_states - 1
        - Means: n_states * n_features
        - Full covariance: n_states * n_features * (n_features + 1) / 2
    """
    n_samples = X.shape[0]
    n_states = model.n_components
    n_features = X.shape[1]

    # Free parameters
    k_trans = n_states * (n_states - 1)
    k_init = n_states - 1
    k_means = n_states * n_features
    k_cov = n_states * n_features * (n_features + 1) // 2
    k = k_trans + k_init + k_means + k_cov

    ll = model.score(X)  # total log-likelihood
    return -2 * ll + k * np.log(n_samples)


def select_n_states(
    X: pd.DataFrame,
    min_states: int = 2,
    max_states: int = 6,
    random_state: int = 42,
) -> tuple[GaussianHMM, int, pd.DataFrame]:
    """
    Select optimal number of hidden states via BIC.

    Returns
    -------
    best_model : GaussianHMM
    best_n : int
    results : pd.DataFrame
        BIC and log-likelihood for each candidate state count.
    """
    records = []
    models = {}

    for n in range(min_states, max_states + 1):
        model = fit_hmm(X, n_states=n, random_state=random_state)
        ll = model.score(X.values)
        bic = compute_bic(model, X.values)
        records.append({"n_states": n, "log_likelihood": ll, "BIC": bic})
        models[n] = model

    results = pd.DataFrame(records).set_index("n_states")
    best_n = int(results["BIC"].idxmin())
    return models[best_n], best_n, results
