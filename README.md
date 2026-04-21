# regime-detection-core

Core domain package for market regime detection using Gaussian Hidden Markov Models.

## Bounded Context

Owns HMM fitting, Viterbi decoding, posterior probability extraction, regime labelling, walk-forward validation, and BIC model selection.

## Installation

```bash
pip install git+https://github.com/govid13427742/regime-detection-core.git@main
```

## API

| Function | Description |
|----------|-------------|
| `fit_hmm(X, n_states, n_iter, random_state)` | Fit Gaussian HMM with full covariance |
| `decode_states(model, X)` | Viterbi-decoded state sequence |
| `state_probabilities(model, X)` | Posterior probabilities (forward-backward) |
| `label_regimes(states, features)` | Asset-agnostic rank-based regime labelling |
| `build_regime_df(dates, states, probs, label_map)` | Assemble regime DataFrame |
| `walk_forward_hmm(features, ...)` | Expanding-window out-of-sample validation |
| `select_n_states(X, min_states, max_states)` | BIC-based model selection |
| `compute_bic(model, X)` | Bayesian Information Criterion |

## Data Contracts

### Input
Features DataFrame must contain columns `log_return`, `vol_20d`, `drawdown` (used by `label_regimes`).

### Output: regime_df
- **Index**: `pd.DatetimeIndex`
- **Columns**: `state` (int), `regime` (str), `prob_crash` (float), `prob_high_vol` (float), `prob_low_vol` (float)
- **Invariant**: `prob_crash + prob_high_vol + prob_low_vol = 1.0`

## Dependencies

- numpy, pandas, hmmlearn, scikit-learn
