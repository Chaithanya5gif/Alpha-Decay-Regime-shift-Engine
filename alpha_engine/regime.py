import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# We'll map HMM states (0, 1, 2) to human-readable labels after fitting
REGIME_LABELS = {0: "Bull", 1: "Bear", 2: "High Volatility"}


def build_features(returns: pd.DataFrame, vol_lookback: int = 21) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Build the 2-feature input matrix for HMM:
      - Feature 1: Mean daily return across all tickers (market return proxy)
      - Feature 2: Mean realized volatility across all tickers (market vol proxy)

    We average across tickers to get a single market-level signal.
    HMM needs a 2D array of shape (n_days, n_features).
    """
    # Average return across all tickers each day
    market_return = returns.mean(axis=1)

    # Rolling realized volatility (21-day), annualized
    market_vol = returns.std(axis=1).rolling(window=vol_lookback).mean() * np.sqrt(252)

    # Combine into a DataFrame and drop NaN rows (from rolling window warmup)
    features_df = pd.DataFrame({
        "return": market_return,
        "volatility": market_vol
    }).dropna()

    # Scale features — HMM is sensitive to feature scale
    # StandardScaler makes each feature zero mean, unit variance
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df.values)

    return features_scaled, features_df.index


def fit_hmm(
    features: np.ndarray,
    n_states: int = 3,
    n_iter: int = 1000,
    random_state: int = 42
) -> GaussianHMM:
    """
    Fit a Gaussian HMM to the feature matrix.

    GaussianHMM assumes each hidden state emits observations drawn from
    a Gaussian distribution — reasonable for financial returns.

    n_states=3 gives us Bull / Bear / High-Vol regimes.
    Increasing n_iter helps the Baum-Welch algorithm converge.
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",   # Each state has its own covariance matrix
        n_iter=n_iter,
        random_state=random_state
    )
    model.fit(features)
    return model


def decode_regimes(model: GaussianHMM, features: np.ndarray) -> np.ndarray:
    """
    Use the Viterbi algorithm to find the most likely sequence of hidden states.
    Returns an array of state indices (0, 1, or 2) for each day.
    
    Viterbi is a dynamic programming algorithm — it finds the globally optimal
    state sequence, not just the locally best state at each step.
    """
    state_sequence = model.predict(features)
    return state_sequence


def label_regimes(
    state_sequence: np.ndarray,
    returns_index: pd.DatetimeIndex,
    model: GaussianHMM
) -> pd.Series:
    """
    Map raw HMM state indices (0, 1, 2) to meaningful regime labels.

    HMM doesn't know which state is "Bull" or "Bear" — we infer it by
    looking at the mean return of each state:
      - Highest mean return  → Bull
      - Lowest mean return   → Bear
      - Middle (highest vol) → High Volatility
    """
    # Get mean return for each HMM state from the model's learned means
    # means_ shape: (n_states, n_features) — first feature is return
    state_means = model.means_[:, 0]  # Return feature means per state

    # Rank states by mean return: highest = Bull, lowest = Bear, middle = High Vol
    sorted_states = np.argsort(state_means)  # Ascending order
    
    state_to_label = {
        sorted_states[0]: "Bear",
        sorted_states[1]: "High Volatility",
        sorted_states[2]: "Bull"
    }

    # Map integer states to string labels
    labeled = pd.Series(
        [state_to_label[s] for s in state_sequence],
        index=returns_index,
        name="regime"
    )

    return labeled


def detect_regimes(
    returns: pd.DataFrame,
    n_states: int = 3,
    vol_lookback: int = 21,
    sensitivity: float = 1.0
) -> pd.Series:
    """
    Main entry point: takes returns DataFrame, returns a Series of regime labels.

    sensitivity > 1.0 → model switches regimes more freely (more sensitive)
    sensitivity < 1.0 → model stays in regimes longer (more stable)
    
    We implement sensitivity by adjusting the transition matrix prior after fitting.
    Higher sensitivity = lower self-transition probability = more regime changes.
    """
    features, index = build_features(returns, vol_lookback=vol_lookback)

    model = fit_hmm(features, n_states=n_states)

    # Apply sensitivity by nudging the transition matrix
    # transmat_ is an (n_states x n_states) matrix where transmat_[i][j]
    # is the probability of going from state i to state j
    if sensitivity != 1.0:
        transmat = model.transmat_.copy()
        # Reduce self-transition probability to make regime changes more likely
        np.fill_diagonal(transmat, transmat.diagonal() / sensitivity)
        # Re-normalize rows to sum to 1 (valid probability distribution)
        transmat = transmat / transmat.sum(axis=1, keepdims=True)
        model.transmat_ = transmat

    state_sequence = decode_regimes(model, features)
    regime_labels = label_regimes(state_sequence, index, model)

    return regime_labels


def regime_persistence(regimes: pd.Series) -> dict:
    """
    Calculate how long the market tends to stay in each regime (in days).
    Useful for the metrics panel in the dashboard.
    
    We do this by finding consecutive runs of the same regime label
    and averaging their lengths.
    """
    persistence = {}

    for label in regimes.unique():
        # Find where this regime is active
        is_regime = (regimes == label).astype(int)
        # Identify regime change points
        changes = is_regime.diff().fillna(0)
        # Count length of each consecutive run
        run_lengths = []
        count = 0
        for val in is_regime:
            if val == 1:
                count += 1
            elif count > 0:
                run_lengths.append(count)
                count = 0
        if count > 0:
            run_lengths.append(count)

        persistence[label] = np.mean(run_lengths) if run_lengths else 0

    return persistence