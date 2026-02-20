import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from alpha_engine.data import fetch_data, get_price_data
from alpha_engine.alphas import momentum_alpha, mean_reversion_alpha, volatility_carry_alpha, combine_alphas
from alpha_engine.regime import detect_regimes
from alpha_engine.decay import calculate_alpha_decay, fit_decay_curve, decay_by_regime, alpha_quality_score

# Fetch data
prices = get_price_data(["AAPL", "MSFT", "GOOGL"], start="2020-01-01", end="2024-01-01")
returns = fetch_data(["AAPL", "MSFT", "GOOGL"], start="2020-01-01", end="2024-01-01")

print("=" * 60)
print("ALPHA DECAY ANALYSIS")
print("=" * 60)

# Generate alpha signals
momentum = momentum_alpha(prices)
mean_rev = mean_reversion_alpha(prices)
vol_carry = volatility_carry_alpha(returns)
composite = combine_alphas(momentum, mean_rev, vol_carry)

# Calculate alpha decay
decay_data = calculate_alpha_decay(composite, returns)
print(f"\nAlpha decay data shape: {decay_data.shape}")
print(f"Columns: {decay_data.columns.tolist()}")
print(f"\nForward returns by horizon:")
print(decay_data[['ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d']].describe())

# Fit decay curve
decay_metrics = fit_decay_curve(composite, returns)
print(f"\n" + "=" * 60)
print("DECAY CURVE FIT")
print("=" * 60)
print(f"Initial alpha: {decay_metrics['initial_alpha']:.4f}")
print(f"Decay rate: {decay_metrics['decay_rate']:.4f}")
print(f"Persistent alpha: {decay_metrics['persistent_alpha']:.4f}")
print(f"Half-life (days): {decay_metrics['half_life']:.1f}")
print(f"Observations: {decay_metrics['num_observations']}")

# Quality score
quality = alpha_quality_score(decay_metrics)
print(f"\nAlpha quality score: {quality:.1f}/100")

# Decay by regime
print(f"\n" + "=" * 60)
print("DECAY BY REGIME")
print("=" * 60)

regimes = detect_regimes(returns, n_states=3)
regime_decay = decay_by_regime(composite, returns, regimes)

for regime_name, metrics in regime_decay.items():
    if metrics:
        print(f"\n{regime_name}:")
        print(f"  Initial alpha: {metrics['initial_alpha']:.4f}")
        print(f"  Decay rate: {metrics['decay_rate']:.4f}")
        print(f"  Half-life: {metrics['half_life']:.1f} days")
        print(f"  Quality score: {alpha_quality_score(metrics):.1f}/100")

print("\n" + "=" * 60)
print("Alpha decay analysis complete!")
print("=" * 60)
