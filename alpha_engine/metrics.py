"""Performance metrics and analytics."""

import pandas as pd
import numpy as np
from scipy import stats


def information_coefficient(alpha_signal: pd.DataFrame, forward_returns: pd.DataFrame) -> float:
    """
    Information Coefficient (IC): correlation between alpha signal and forward returns.
    
    IC measures how predictive the alpha signal is of next-day returns.
    
    IC interpretation:
      - IC > 0.05: Statistically significant signal (p < 0.05 typically)
      - IC > 0.10: Strong signal
      - IC > 0.20: Exceptional signal (rare in practice)
      - IC ≈ 0: Signal has no predictive power
      - IC < 0: Signal is anti-correlated (use negative of signal for long positions)
    
    Returns the rank correlation (Spearman) which is more robust to outliers than Pearson.
    """
    # Align indices
    common_idx = alpha_signal.index.intersection(forward_returns.index)
    signal = alpha_signal.loc[common_idx].values.flatten()
    rets = forward_returns.loc[common_idx].values.flatten()
    
    # Remove NaN pairs
    mask = ~(np.isnan(signal) | np.isnan(rets))
    signal_clean = signal[mask]
    rets_clean = rets[mask]
    
    if len(signal_clean) < 2:
        return np.nan
    
    # Rank correlation (more robust than Pearson for financial data)
    ic, _ = stats.spearmanr(signal_clean, rets_clean)
    return ic


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Sharpe Ratio: risk-adjusted return.
    
    Formula: (mean_return - risk_free_rate) / std_return
    
    Higher Sharpe is better. A Sharpe > 1 is considered good, > 2 is excellent.
    Assumes returns are daily; we annualize by multiplying by sqrt(252).
    """
    excess_returns = returns - risk_free_rate / 252  # Convert annual to daily
    
    if excess_returns.std() == 0:
        return np.nan
    
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    return sharpe


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Sortino Ratio: like Sharpe but only penalizes downside volatility.
    
    Formula: (mean_return - risk_free_rate) / downside_std
    
    downside_std only includes losses (negative returns).
    This is more relevant for investors who care about losses but not about upside.
    """
    excess_returns = returns - risk_free_rate / 252
    downside = excess_returns[excess_returns < 0]
    
    if len(downside) == 0 or downside.std() == 0:
        return np.nan
    
    sortino = excess_returns.mean() / downside.std() * np.sqrt(252)
    return sortino


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum Drawdown: largest peak-to-trough decline.
    
    Formula: (trough - peak) / peak, expressed as negative percentage.
    
    MDD = -30% means the strategy lost 30% from its peak before recovering.
    More negative is worse.
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()


def calmar_ratio(returns: pd.Series) -> float:
    """
    Calmar Ratio: annual return / absolute max drawdown.
    
    Formula: (annual_return) / abs(max_drawdown)
    
    Measures return relative to downside risk. Higher is better.
    A Calmar > 1 is decent, > 2 is good.
    """
    annual_return = returns.mean() * 252
    mdd = max_drawdown(returns)
    
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    
    calmar = annual_return / abs(mdd)
    return calmar


def portfolio_metrics(
    alpha_signal: pd.DataFrame,
    returns: pd.DataFrame,
    position_size: float = 1.0
) -> dict:
    """
    Comprehensive portfolio metrics assuming a simple long-short strategy:
      - Long the top quartile (highest alpha signal)
      - Short the bottom quartile (lowest alpha signal)
      - Hold equal weight within each quartile
    
    Returns dict with:
      - ic: Information coefficient with 1-day forward returns
      - sharpe: Sharpe ratio of strategy returns
      - sortino: Sortino ratio
      - max_dd: Maximum drawdown
      - calmar: Calmar ratio
      - annual_return: Annualized return
      - annual_vol: Annualized volatility
    """
    # Align indices
    common_idx = alpha_signal.index.intersection(returns.index)
    signal = alpha_signal.loc[common_idx]
    rets = returns.loc[common_idx]
    
    # Compute 1-day forward returns
    forward_rets = rets.shift(-1)
    
    # Build portfolio: long top quartile, short bottom quartile
    strategy_returns = []
    
    for i in range(len(signal)):
        if pd.isna(signal.iloc[i]).any() or pd.isna(forward_rets.iloc[i]).any():
            continue
        
        # Get signal ranking (per-row)
        row_signal = signal.iloc[i]
        valid = row_signal.dropna()
        
        if len(valid) < 2:
            continue
        
        # Top and bottom quartiles
        q75 = valid.quantile(0.75)
        q25 = valid.quantile(0.25)
        
        long_mask = valid >= q75
        short_mask = valid <= q25
        
        # Portfolio return: equal weight long minus equal weight short
        long_ret = rets.iloc[i][long_mask].mean() if long_mask.sum() > 0 else 0
        short_ret = rets.iloc[i][short_mask].mean() if short_mask.sum() > 0 else 0
        
        portfolio_ret = long_ret - short_ret
        strategy_returns.append(portfolio_ret)
    
    strategy_returns = pd.Series(strategy_returns)
    
    if len(strategy_returns) < 30:
        return {
            'ic': np.nan,
            'sharpe': np.nan,
            'sortino': np.nan,
            'max_dd': np.nan,
            'calmar': np.nan,
            'annual_return': np.nan,
            'annual_vol': np.nan,
            'observations': len(strategy_returns)
        }
    
    # Calculate all metrics
    ic = information_coefficient(signal, rets)
    sharpe = sharpe_ratio(strategy_returns)
    sortino = sortino_ratio(strategy_returns)
    mdd = max_drawdown(strategy_returns)
    calmar = calmar_ratio(strategy_returns)
    annual_return = strategy_returns.mean() * 252
    annual_vol = strategy_returns.std() * np.sqrt(252)
    
    return {
        'ic': ic,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': mdd,
        'calmar': calmar,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'observations': len(strategy_returns)
    }


def rolling_metrics(
    alpha_signal: pd.DataFrame,
    returns: pd.DataFrame,
    window: int = 63  # ~3 months
) -> dict:
    """
    Calculate metrics in rolling windows to track performance over time.
    
    Returns dict mapping window_start_date -> metrics
    """
    common_idx = alpha_signal.index.intersection(returns.index)
    signal = alpha_signal.loc[common_idx]
    rets = returns.loc[common_idx]
    
    rolling_results = {}
    
    for i in range(window, len(signal)):
        window_start = signal.index[i - window]
        window_end = signal.index[i]
        
        signal_window = signal.iloc[i-window:i]
        rets_window = rets.iloc[i-window:i]
        
        metrics = portfolio_metrics(signal_window, rets_window)
        rolling_results[window_end] = metrics
    
    return rolling_results
