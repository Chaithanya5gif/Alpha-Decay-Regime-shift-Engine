import sys
sys.path.insert(0, '.')

from alpha_engine.data import fetch_data, get_price_data
from alpha_engine.alphas import momentum_alpha, mean_reversion_alpha, volatility_carry_alpha, combine_alphas

# Fetch data
prices = get_price_data(["AAPL", "MSFT", "GOOGL"], start="2020-01-01", end="2024-01-01")
returns = fetch_data(["AAPL", "MSFT", "GOOGL"], start="2020-01-01", end="2024-01-01")

print("Prices shape:", prices.shape)
print("Returns shape:", returns.shape)
print()

# Test each alpha
momentum = momentum_alpha(prices)
print("Momentum alpha shape:", momentum.shape)
print("Momentum (first 5 rows):")
print(momentum.head())
print()

mean_rev = mean_reversion_alpha(prices)
print("Mean reversion alpha shape:", mean_rev.shape)
print("Mean reversion (first 5 rows):")
print(mean_rev.head())
print()

vol_carry = volatility_carry_alpha(returns)
print("Volatility carry alpha shape:", vol_carry.shape)
print("Volatility carry (first 5 rows):")
print(vol_carry.head())
print()

# Combine alphas
composite = combine_alphas(momentum, mean_rev, vol_carry)
print("Composite alpha shape:", composite.shape)
print("Composite (first 5 rows):")
print(composite.head())
print("Composite (last 5 rows):")
print(composite.tail())
