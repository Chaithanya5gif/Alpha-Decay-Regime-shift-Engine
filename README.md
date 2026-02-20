# Alpha Decay & Regime Shift Analysis Engine

An advanced quantitative research framework for analyzing how trading signals (alphas) decay over time and perform across different market regimes using Hidden Markov Models.

## 📊 Project Structure

```
alpha-engine/
├── alpha_engine/
│   ├── __init__.py          # Package initialization
│   ├── data.py              # Data loading from Yahoo Finance
│   ├── alphas.py            # Alpha signal generation (momentum, mean reversion, vol carry)
│   ├── regime.py            # Market regime detection via HMM
│   ├── decay.py             # Alpha decay analysis and curve fitting
│   ├── metrics.py           # Performance metrics (Sharpe, Sortino, IC, etc.)
│   └── viz.py               # Interactive visualizations with Plotly
├── streamlit_app.py         # Web dashboard
├── requirements.txt         # Python dependencies
└── test_*.py               # Unit tests for each module
```

## 🚀 Quick Start

### Installation

```bash
cd alpha-engine
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### Run the Web Dashboard

```bash
streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

### Run Tests

```bash
source venv/bin/activate

python test_data.py      # Test data loading
python test_alphas.py    # Test alpha generation
python test_regime.py    # Test regime detection
python test_decay.py     # Test decay analysis
python test_metrics.py   # Test performance metrics
python test_viz.py       # Test visualizations
```

## 📚 Core Components

### 1. Data Module (`data.py`)
- Fetches daily OHLCV data from Yahoo Finance via `yfinance`
- Computes log returns for stationarity
- Returns both log returns and raw prices for different analyses

**Key Functions:**
- `fetch_data()`: Get log returns
- `get_price_data()`: Get raw adjusted closing prices

---

### 2. Alpha Generation (`alphas.py`)
Implements three complementary trading signals:

**Momentum Alpha**
- Formula: `(P_{t-21} / P_{t-252}) - 1`
- Buy securities with high historical returns, skip last month to avoid reversal
- Cross-sectionally ranked to market-neutral

**Mean Reversion Alpha**
- Formula: `-(P_t - mean(P_20)) / std(P_20)`
- Buy oversold securities (low relative to 20-day average)
- Exploits temporary price deviations

**Volatility Carry Alpha**
- Formula: `-realized_vol`
- Buy low-volatility assets, sell high-volatility
- Based on volatility risk premium

**Composite Alpha**
- Weighted combination: 40% momentum + 30% mean reversion + 30% vol carry
- Cross-sectionally normalized to [-1, 1]

**Key Functions:**
- `momentum_alpha()`, `mean_reversion_alpha()`, `volatility_carry_alpha()`
- `combine_alphas()`: Blend multiple signals with custom weights
- `_rank_normalize()`: Market-neutral normalization

---

### 3. Regime Detection (`regime.py`)
Uses Hidden Markov Models to identify unobserved market states.

**Model:**
- Gaussian HMM with 3 states: Bull, Bear, High Volatility
- Features: daily market return + daily market volatility
- Viterbi decoding for optimal state sequence

**Key Insights:**
- Bull regime: High returns, low volatility
- Bear regime: Low returns, high drawdown risk
- High Vol regime: Transition periods, elevated uncertainty

**Key Functions:**
- `detect_regimes()`: Main entry point returning Series of regime labels
- `regime_persistence()`: Average days spent in each regime
- `build_features()`, `fit_hmm()`, `decode_regimes()`, `label_regimes()`

---

### 4. Alpha Decay Analysis (`decay.py`)
Measures how quickly alpha loses predictive power over time.

**Approach:**
1. Rank securities by alpha signal strength (top 10%)
2. Measure forward returns at different horizons: 1, 5, 10, 20, 60 days
3. Fit exponential decay curve: `a * exp(-b*x) + c`
4. Extract decay parameters

**Outputs:**
- `initial_alpha`: Signal strength at 1-day horizon
- `decay_rate`: How fast the signal decays (higher = faster decay)
- `persistent_alpha`: Baseline signal that doesn't fade
- `half_life`: Days to decay to 50% of initial value

**By Regime:**
- Reveal regime-dependent alpha quality
- E.g., momentum may work better in Bull markets but decay faster in choppy conditions

**Key Functions:**
- `calculate_alpha_decay()`: Compute forward returns for all horizons
- `fit_decay_curve()`: Exponential curve fitting
- `decay_by_regime()`: Separate analysis per market regime
- `alpha_quality_score()`: Composite 0-100 quality score

---

### 5. Performance Metrics (`metrics.py`)
Standard quant finance performance metrics.

**Metrics Included:**
- **Information Coefficient (IC)**: Spearman rank correlation between signal and forward returns
  - IC > 0.05: Significant
  - IC > 0.10: Strong
  - IC > 0.20: Exceptional (rare)

- **Sharpe Ratio**: `(return - rfr) / volatility`
  - Annualized, risk-free rate = 2%
  - Sharpe > 1: Good
  - Sharpe > 2: Excellent

- **Sortino Ratio**: Like Sharpe but only penalizes downside volatility

- **Maximum Drawdown**: Largest peak-to-trough decline

- **Calmar Ratio**: `annual_return / abs(max_drawdown)`
  - Calmar > 1: Decent
  - Calmar > 2: Good

**Portfolio Construction:**
- Long top quartile of securities (by alpha signal)
- Short bottom quartile
- Equal weight within each quartile
- Daily rebalancing

**Key Functions:**
- `information_coefficient()`, `sharpe_ratio()`, `sortino_ratio()`, `max_drawdown()`, `calmar_ratio()`
- `portfolio_metrics()`: Comprehensive metrics for a strategy
- `rolling_metrics()`: Compute metrics in rolling windows to track performance evolution

---

### 6. Visualizations (`viz.py`)
Interactive Plotly charts for analysis and exploration.

**Charts:**
- `plot_alpha_signals()`: Time series of all alpha components
- `plot_regime_timeline()`: Colored regime changes over time
- `plot_decay_curve()`: Fitted exponential decay with actual observations
- `plot_rolling_metrics()`: Time series of Sharpe, IC, and max DD
- `plot_regime_performance()`: Alpha decay compared across regimes
- `plot_cumulative_returns()`: Strategy growth (log scale)
- `plot_drawdown()`: Underwater plot showing peak-to-trough declines

---

### 7. Streamlit Dashboard (`streamlit_app.py`)
Interactive web application for exploration and parameter tuning.

**Tabs:**
1. **📊 Alphas**: Visualize individual and composite alpha signals
2. **🎭 Regimes**: Market regime timeline and persistence statistics
3. **📉 Decay**: Alpha decay curves with regime breakdown
4. **📈 Metrics**: Performance metrics and rolling performance tracking
5. **💡 Insights**: Automated interpretation of results

**Interactive Controls:**
- Select tickers, date range
- Tune momentum/mean reversion lookbacks
- Adjust alpha weights
- Regime sensitivity slider
- Real-time updates

**Run:**
```bash
streamlit run streamlit_app.py
```

---

## 🔬 Theoretical Background

### Why Alpha Decay?
Trading signals typically lose predictive power over time due to:
- **Crowding**: As more traders exploit the signal, returns diminish
- **Regime shifts**: Market conditions change, breaking statistical relationships
- **Microstructure**: Bid-ask bounce and noise become dominant at short horizons
- **Non-stationarity**: Financial data distributes change over time

### Why Regime Switching?
Different market regimes have different characteristic returns and volatilities:
- **Bull**: High returns, mean-reverting, alphas decay slowly
- **Bear**: Negative returns, momentum-driven, alphas may persist or reverse
- **High Vol**: Elevated uncertainty, correlations rise, diversification breaks down

A single alpha may work in Bull regimes but fail in Bear regimes.

### Why Hidden Markov Models?
- Regimes are **latent** (unobserved) — we only see returns
- HMM infers most likely regime sequence using Baum-Welch algorithm
- Viterbi decoding finds globally optimal state path
- Captures regime persistence and transition probabilities

---

## 📊 Example Analysis

### Input Parameters
- Tickers: AAPL, MSFT, GOOGL
- Period: 2020-01-01 to 2024-01-01
- Momentum lookback: 252 days (1 year)
- Mean reversion lookback: 20 days (1 month)

### Expected Output
```
Alpha Quality Score: 50-70/100
Information Coefficient: -0.05 to +0.10 (varies by signal)
Sharpe Ratio: -3 to +2 (depends on regime environment)
Alpha Half-life: 10-100 days (signal persists 1-3 months)
```

---

## 🛠️ Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| yfinance | Latest | Market data fetching |
| pandas | ≥1.3 | Data manipulation |
| numpy | ≥1.21 | Numerical computing |
| scikit-learn | ≥0.24 | Machine learning (preprocessing) |
| hmmlearn | Latest | Hidden Markov Models |
| scipy | Latest | Scientific computing (curve fitting) |
| plotly | ≥6.0 | Interactive visualizations |
| streamlit | ≥1.0 | Web dashboard framework |

---

## 🚦 Future Enhancements

- [ ] Support for intraday data (hourly, minute-level)
- [ ] Bayesian regime detection (alternative to HMM)
- [ ] Multi-factor alpha blending (PCA, independent components)
- [ ] Walk-forward backtesting with transaction costs
- [ ] Real-time data updates and live dashboard
- [ ] Performance attribution (which alpha is working when?)
- [ ] Risk factor exposure analysis
- [ ] Portfolio optimization with alpha and regime signals
- [ ] Export strategies to interactive trading platforms

---

## 📈 Use Cases

1. **Signal Research**: Evaluate new trading signals for persistence and regime dependence
2. **Portfolio Construction**: Weight alphas based on regime forecasts
3. **Risk Management**: Adjust position sizing when alpha quality decays
4. **Regime Trading**: Build regime-switching strategies
5. **Academic Research**: Test factor models across different market states
6. **Performance Monitoring**: Track how strategy performance evolves over time

---

## ⚠️ Disclaimers

- **Past Performance**: Historical backtests do not guarantee future results
- **Look-Ahead Bias**: Ensure walk-forward testing for real-world strategies
- **Transaction Costs**: Not included in analysis — real costs will reduce returns
- **Data Quality**: Yahoo Finance data is free but may have gaps or errors
- **Regime Detection**: HMM is unsupervised; regime labeling is post-hoc interpretation

---

## 📝 Citation

If you use this framework in research, please cite:

```bibtex
@software{alpha_decay_engine_2026,
  title={Alpha Decay \& Regime Shift Analysis Engine},
  author={Chaithanya5gif},
  year={2026},
  url={https://github.com/Chaithanya5gif/Alpha-Decay-Regime-shift-Engine}
}
```

---

## 📧 Support

For questions or feature requests, open an issue on GitHub.

Happy alpha hunting! 🚀📊
