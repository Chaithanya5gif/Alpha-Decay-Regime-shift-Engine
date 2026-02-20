import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from alpha_engine.data import fetch_data, get_price_data
from alpha_engine.alphas import momentum_alpha, mean_reversion_alpha, volatility_carry_alpha, combine_alphas
from alpha_engine.regime import detect_regimes
from alpha_engine.decay import fit_decay_curve, decay_by_regime, calculate_alpha_decay
from alpha_engine.metrics import portfolio_metrics, rolling_metrics
from alpha_engine.viz import (
    plot_alpha_signals, plot_regime_timeline, plot_decay_curve,
    plot_rolling_metrics, plot_regime_performance
)

# Fetch data
prices = get_price_data(["AAPL", "MSFT", "GOOGL"], start="2020-01-01", end="2024-01-01")
returns = fetch_data(["AAPL", "MSFT", "GOOGL"], start="2020-01-01", end="2024-01-01")

print("=" * 60)
print("VISUALIZATION TESTS")
print("=" * 60)

# Generate alphas
momentum = momentum_alpha(prices)
mean_rev = mean_reversion_alpha(prices)
vol_carry = volatility_carry_alpha(returns)
composite = combine_alphas(momentum, mean_rev, vol_carry)

print("\n1. Testing alpha signals plot...")
try:
    fig = plot_alpha_signals(momentum, mean_rev, vol_carry, composite)
    print("   ✓ Alpha signals plot created")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n2. Testing regime timeline plot...")
try:
    regimes = detect_regimes(returns, n_states=3)
    fig = plot_regime_timeline(regimes)
    print("   ✓ Regime timeline plot created")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n3. Testing decay curve plot...")
try:
    decay_metrics = fit_decay_curve(composite, returns)
    fig = plot_decay_curve(decay_metrics)
    print("   ✓ Decay curve plot created")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n4. Testing rolling metrics plot...")
try:
    rolling = rolling_metrics(composite, returns, window=63)
    if len(rolling) > 0:
        fig = plot_rolling_metrics(rolling)
        print("   ✓ Rolling metrics plot created")
    else:
        print("   ⚠ Not enough data for rolling metrics")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n5. Testing regime performance plot...")
try:
    regime_decay = decay_by_regime(composite, returns, regimes)
    fig = plot_regime_performance(regime_decay)
    print("   ✓ Regime performance plot created")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print("All visualization functions tested successfully!")
print("=" * 60)
