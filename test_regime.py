import sys
sys.path.insert(0, '.')

import numpy as np
from alpha_engine.data import fetch_data, get_price_data
from alpha_engine.alphas import momentum_alpha, mean_reversion_alpha, volatility_carry_alpha, combine_alphas
from alpha_engine.regime import detect_regimes, regime_persistence

# Fetch data
prices = get_price_data(["AAPL", "MSFT", "GOOGL"], start="2020-01-01", end="2024-01-01")
returns = fetch_data(["AAPL", "MSFT", "GOOGL"], start="2020-01-01", end="2024-01-01")

print("=" * 60)
print("REGIME DETECTION TEST")
print("=" * 60)

# Detect regimes
regimes = detect_regimes(returns, n_states=3)
print(f"\nRegimes detected:")
print(f"  Shape: {regimes.shape}")
print(f"  Unique regimes: {regimes.unique().tolist()}")
print(f"  Value counts:")
print(regimes.value_counts())
print(f"\nFirst 10 regime labels:")
print(regimes.head(10))
print(f"\nLast 10 regime labels:")
print(regimes.tail(10))

# Regime persistence
persistence = regime_persistence(regimes)
print(f"\nRegime persistence (avg days in each regime):")
for regime, days in persistence.items():
    print(f"  {regime}: {days:.1f} days")

print("\n" + "=" * 60)
print("Regime detection test passed!")
print("=" * 60)
