import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from alpha_engine.data import fetch_data, get_price_data
from alpha_engine.alphas import momentum_alpha, mean_reversion_alpha, volatility_carry_alpha, combine_alphas
from alpha_engine.metrics import (
    information_coefficient, sharpe_ratio, sortino_ratio, max_drawdown,
    calmar_ratio, portfolio_metrics, rolling_metrics
)

# Fetch data
prices = get_price_data(["AAPL", "MSFT", "GOOGL"], start="2020-01-01", end="2024-01-01")
returns = fetch_data(["AAPL", "MSFT", "GOOGL"], start="2020-01-01", end="2024-01-01")

print("=" * 60)
print("PERFORMANCE METRICS ANALYSIS")
print("=" * 60)

# Generate alpha signals
momentum = momentum_alpha(prices)
mean_rev = mean_reversion_alpha(prices)
vol_carry = volatility_carry_alpha(returns)
composite = combine_alphas(momentum, mean_rev, vol_carry)

# Portfolio metrics
metrics = portfolio_metrics(composite, returns)
print(f"\nPortfolio Performance Metrics:")
print(f"  Information Coefficient: {metrics['ic']:.4f}")
print(f"  Annual Return: {metrics['annual_return']:.2%}")
print(f"  Annual Volatility: {metrics['annual_vol']:.2%}")
print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
print(f"  Sortino Ratio: {metrics['sortino']:.2f}")
print(f"  Max Drawdown: {metrics['max_dd']:.2%}")
print(f"  Calmar Ratio: {metrics['calmar']:.2f}")
print(f"  Observations: {metrics['observations']}")

# Rolling metrics
print(f"\n" + "=" * 60)
print("ROLLING METRICS (63-day windows)")
print("=" * 60)

rolling = rolling_metrics(composite, returns, window=63)
rolling_df = pd.DataFrame(rolling).T

print(f"\nRolling metrics shape: {rolling_df.shape}")
print(f"\nSharpe Ratio statistics:")
print(rolling_df['sharpe'].describe())

print(f"\nMax Drawdown statistics:")
print(rolling_df['max_dd'].describe())

print(f"\nInformation Coefficient statistics:")
print(rolling_df['ic'].describe())

# Recent performance
print(f"\nMost recent window (last 63 days):")
recent = rolling_df.iloc[-1]
print(f"  Sharpe: {recent['sharpe']:.2f}")
print(f"  IC: {recent['ic']:.4f}")
print(f"  Max DD: {recent['max_dd']:.2%}")
print(f"  Annual Return: {recent['annual_return']:.2%}")

print("\n" + "=" * 60)
print("Metrics analysis complete!")
print("=" * 60)
