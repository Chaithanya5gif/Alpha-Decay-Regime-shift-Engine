"""Alpha generation and calculation module."""

# Step 2: Alphas module implementation
import pandas as pd
import numpy as np


def momentum_alpha(prices: pd.DataFrame, lookback: int = 252, skip_days: int = 21) -> pd.DataFrame:
    """
    Momentum signal: return over [lookback] days, skipping the most recent [skip_days].
    
    Formula: P_{t - skip_days} / P_{t - lookback} - 1
    
    We skip the last month because short-term momentum actually reverses
    (bid-ask bounce, microstructure noise). The signal is then cross-sectionally
    ranked and normalized to [-1, 1] so different alphas are comparable.
    """
    # Shift by skip_days to avoid short-term reversal
    momentum = prices.shift(skip_days) / prices.shift(lookback) - 1

    # Cross-sectional rank normalization (rank each row, scale to [-1, 1])
    # This makes the signal market-neutral — longs offset shorts
    signal = momentum.apply(lambda row: _rank_normalize(row), axis=1)

    return signal


def mean_reversion_alpha(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Mean reversion signal based on z-score of price vs its rolling mean.
    
    Formula: -(P_t - mean(P, lookback)) / std(P, lookback)
    
    Negative sign because we SELL when price is high vs mean (overbought)
    and BUY when price is low (oversold).
    """
    rolling_mean = prices.rolling(window=lookback).mean()
    rolling_std = prices.rolling(window=lookback).std()

    # Z-score: how many standard deviations above/below the mean
    zscore = (prices - rolling_mean) / rolling_std

    # Flip sign: high z-score = overbought = short signal
    signal = -zscore

    # Clip extreme values (beyond ±3 std are likely outliers/data errors)
    signal = signal.clip(lower=-3, upper=3)

    # Normalize to [-1, 1]
    signal = signal.apply(lambda row: _rank_normalize(row), axis=1)

    return signal


def volatility_carry_alpha(returns: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """
    Volatility carry signal: go long low-vol assets, short high-vol assets.
    
    Formula: -realized_vol over [lookback] days
    
    Realized vol = rolling std of returns * sqrt(252) to annualize.
    Negative sign because we want to be long LOW volatility.
    
    The intuition: high vol assets have negative expected return after
    accounting for risk (investors overpay for lottery-like payoffs).
    """
    # Annualized realized volatility
    realized_vol = returns.rolling(window=lookback).std() * np.sqrt(252)

    # Negative: low vol → high score → long position
    signal = -realized_vol

    # Normalize cross-sectionally
    signal = signal.apply(lambda row: _rank_normalize(row), axis=1)

    return signal


def combine_alphas(
    momentum: pd.DataFrame,
    mean_rev: pd.DataFrame,
    vol_carry: pd.DataFrame,
    weights: tuple = (0.4, 0.3, 0.3)
) -> pd.DataFrame:
    """
    Combine the 3 alphas into a single composite signal using weighted average.
    
    Default weights favor momentum slightly (0.4) since it tends to have
    the highest IC in most equity markets historically.
    
    You can tune weights via the Streamlit sliders later.
    """
    w1, w2, w3 = weights
    composite = w1 * momentum + w2 * mean_rev + w3 * vol_carry

    # Final normalization of composite
    composite = composite.apply(lambda row: _rank_normalize(row), axis=1)

    return composite


def _rank_normalize(row: pd.Series) -> pd.Series:
    """
    Helper: Cross-sectionally rank a row, then scale to [-1, 1].
    NaNs are ignored and kept as NaN.
    
    This ensures the alpha is dollar-neutral (longs = shorts in terms of weight)
    and comparable across different alphas and time periods.
    """
    valid = row.dropna()
    if len(valid) < 2:
        return row  # Not enough data to rank

    # Rank from 0 to n-1, then scale to [-1, 1]
    ranks = valid.rank() - 1  # 0-indexed ranks
    normalized = 2 * ranks / (len(valid) - 1) - 1  # Scale to [-1, 1]

    result = row.copy()
    result[valid.index] = normalized
    return result
