"""Alpha decay analysis and measurement."""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


def calculate_alpha_decay(alpha_signal: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate forward returns aligned with each day's alpha signal.
    
    For each day, we compute the forward returns (next 1, 5, 10, 20, 60 days)
    that follow the alpha signal. This allows us to measure how quickly
    alpha effectiveness decays as we move into the future.
    
    Returns a DataFrame where columns are forward return windows:
      - "ret_1d": 1-day forward return
      - "ret_5d": 5-day forward return
      - "ret_10d": 10-day forward return
      - "ret_20d": 20-day forward return
      - "ret_60d": 60-day forward return
    """
    decay_data = pd.DataFrame(index=alpha_signal.index)
    
    for window in [1, 5, 10, 20, 60]:
        # Shift returns backward to align with today's alpha
        # returns.shift(-1) gives tomorrow's return on today's index
        forward_rets = returns.shift(-window)
        
        # Average across assets
        decay_data[f"ret_{window}d"] = forward_rets.mean(axis=1)
    
    # Align with alpha signal
    decay_data['signal'] = alpha_signal.mean(axis=1)
    
    return decay_data.dropna()


def decay_curve_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Exponential decay model: a * exp(-b * x) + c
    
    a: Initial alpha strength (intercept)
    b: Decay rate (how fast alpha decays)
    c: Asymptotic alpha (persistent component)
    
    A high b value means alpha decays quickly (not very useful).
    A low b value means alpha persists longer (very useful).
    """
    return a * np.exp(-b * x) + c


def fit_decay_curve(
    alpha_signal: pd.DataFrame,
    returns: pd.DataFrame,
    signal_percentile: float = 90
) -> dict:
    """
    Measure alpha decay by ranking securities by the alpha signal
    and measuring forward returns for the top percentile.
    
    This is similar to how quant funds measure signal quality:
      1. Rank each security by signal strength (top 10%)
      2. Measure their forward returns at different horizons
      3. Fit an exponential decay curve
      4. Extract the decay rate (b parameter)
    
    Args:
        alpha_signal: Alpha signal matrix (rows=time, cols=assets)
        returns: Forward returns matrix
        signal_percentile: Focus on top X% of signal strength
    
    Returns:
        Dict with decay parameters and metrics
    """
    decay_data = calculate_alpha_decay(alpha_signal, returns)
    
    # Separate signal strength and forward returns
    signals = decay_data['signal'].values
    
    # Filter to top percentile
    threshold = np.percentile(np.abs(signals), signal_percentile)
    mask = np.abs(signals) >= threshold
    
    # Organize forward returns at different horizons
    windows = [1, 5, 10, 20, 60]
    forward_returns = np.array([
        decay_data.loc[mask, f'ret_{w}d'].mean() for w in windows
    ])
    
    # Fit exponential decay model
    x_data = np.array(windows, dtype=float)
    
    try:
        popt, _ = curve_fit(
            decay_curve_model,
            x_data,
            forward_returns,
            p0=[forward_returns[0], 0.05, forward_returns[-1]],
            maxfev=5000
        )
        a, b, c = popt
        
        # Decay half-life: time to decay to 50% of initial value
        if b > 0:
            half_life = np.log(2) / b
        else:
            half_life = np.inf
    except Exception as e:
        print(f"Warning: Curve fit failed ({e}). Using zero decay.")
        a = forward_returns[0]
        b = 0
        c = forward_returns[-1]
        half_life = np.inf
    
    return {
        'initial_alpha': a,
        'decay_rate': b,
        'persistent_alpha': c,
        'half_life': half_life,
        'windows': windows,
        'forward_returns': forward_returns.tolist(),
        'num_observations': mask.sum()
    }


def decay_by_regime(
    alpha_signal: pd.DataFrame,
    returns: pd.DataFrame,
    regimes: pd.Series,
    signal_percentile: float = 90
) -> dict:
    """
    Measure alpha decay separately for each market regime.
    
    This reveals regime-dependent alpha quality:
      - Alpha may work better in trending markets (Bull)
      - But decay faster in choppy markets (High Volatility)
    
    Returns dict mapping regime_name -> decay_metrics
    """
    # Ensure index alignment
    common_idx = alpha_signal.index.intersection(returns.index).intersection(regimes.index)
    
    alpha_signal = alpha_signal.loc[common_idx]
    returns = returns.loc[common_idx]
    regimes = regimes.loc[common_idx]
    
    regime_decay = {}
    
    for regime_name in regimes.unique():
        regime_mask = regimes == regime_name
        
        # Filter data to this regime
        regime_alpha = alpha_signal[regime_mask]
        regime_returns = returns[regime_mask]
        
        if len(regime_alpha) > 60:  # Need enough data to fit
            try:
                decay = fit_decay_curve(regime_alpha, regime_returns, signal_percentile)
                regime_decay[regime_name] = decay
            except Exception as e:
                print(f"Warning: Could not fit decay for regime {regime_name}: {e}")
                regime_decay[regime_name] = None
    
    return regime_decay


def alpha_quality_score(decay_metrics: dict) -> float:
    """
    Composite score for alpha quality (0 to 100).
    
    Considers:
      - High initial_alpha → good signal strength at short horizons
      - Low decay_rate → alpha persists longer
      - High persistent_alpha → signal doesn't completely fade
    
    Score interpretation:
      - 80+: Excellent (long-lasting, strong signal)
      - 60-80: Good (moderate persistence, reasonable signal)
      - 40-60: Fair (decays quickly, weak signal)
      - <40: Poor (signal fades rapidly)
    """
    initial = np.clip(decay_metrics['initial_alpha'], 0, 0.05) / 0.05 * 50  # Max 50 points
    persistence = (1 - np.clip(decay_metrics['decay_rate'], 0, 0.2) / 0.2) * 30  # Max 30 points
    baseline = np.clip(decay_metrics['persistent_alpha'], 0, 0.02) / 0.02 * 20  # Max 20 points
    
    return initial + persistence + baseline
