"""Streamlit web application for alpha decay analysis."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our modules
import sys
sys.path.insert(0, '.')
from alpha_engine.data import fetch_data, get_price_data
from alpha_engine.alphas import momentum_alpha, mean_reversion_alpha, volatility_carry_alpha, combine_alphas
from alpha_engine.regime import detect_regimes, regime_persistence
from alpha_engine.decay import fit_decay_curve, decay_by_regime, alpha_quality_score
from alpha_engine.metrics import portfolio_metrics, rolling_metrics
from alpha_engine.viz import (
    plot_alpha_signals, plot_regime_timeline, plot_decay_curve,
    plot_rolling_metrics, plot_regime_performance, plot_cumulative_returns, plot_drawdown
)


# Configure page
st.set_page_config(
    page_title="Alpha Decay Analysis Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Alpha Decay & Regime Shift Analysis Engine")
st.markdown("""
An advanced framework for analyzing how trading signals decay over time and perform 
across different market regimes using Hidden Markov Models.
""")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")

with st.sidebar:
    # Data selection
    st.subheader("📈 Data Selection")
    
    tickers_input = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL,MSFT,GOOGL",
        help="Enter ticker symbols separated by commas"
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2020, 1, 1),
            help="Beginning of analysis period"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime(2024, 1, 1),
            help="End of analysis period"
        )
    
    # Alpha parameters
    st.subheader("🔧 Alpha Parameters")
    
    momentum_lookback = st.slider(
        "Momentum lookback (days)",
        min_value=21, max_value=252, value=252, step=21
    )
    
    mean_rev_lookback = st.slider(
        "Mean reversion lookback (days)",
        min_value=5, max_value=60, value=20, step=5
    )
    
    alpha_weights = st.columns(3)
    with alpha_weights[0]:
        w_momentum = st.slider("Momentum weight", 0.0, 1.0, 0.4, 0.1)
    with alpha_weights[1]:
        w_mean_rev = st.slider("Mean reversion weight", 0.0, 1.0, 0.3, 0.1)
    with alpha_weights[2]:
        w_vol_carry = st.slider("Vol carry weight", 0.0, 1.0, 0.3, 0.1)
    
    # Normalize weights
    total_weight = w_momentum + w_mean_rev + w_vol_carry
    weights = (w_momentum / total_weight, w_mean_rev / total_weight, w_vol_carry / total_weight)
    
    # Regime parameters
    st.subheader("🎭 Regime Parameters")
    n_regimes = st.slider("Number of market regimes", min_value=2, max_value=4, value=3)
    
    sensitivity = st.slider(
        "Regime sensitivity",
        min_value=0.5, max_value=2.0, value=1.0, step=0.1,
        help="Higher = more frequent regime changes"
    )

# Load and process data
try:
    st.sidebar.info("Loading data...")
    
    with st.spinner("Fetching price data..."):
        prices = get_price_data(tickers, start=start_date.isoformat(), end=end_date.isoformat())
        returns = fetch_data(tickers, start=start_date.isoformat(), end=end_date.isoformat())
    
    with st.spinner("Calculating alphas..."):
        momentum = momentum_alpha(prices, lookback=momentum_lookback)
        mean_rev = mean_reversion_alpha(prices, lookback=mean_rev_lookback)
        vol_carry = volatility_carry_alpha(returns)
        composite = combine_alphas(momentum, mean_rev, vol_carry, weights=weights)
    
    with st.spinner("Detecting regimes..."):
        regimes = detect_regimes(returns, n_states=n_regimes, sensitivity=sensitivity)
    
    st.sidebar.success("✅ Data loaded successfully!")
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Tabs for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Alphas",
    "🎭 Regimes",
    "📉 Decay",
    "📈 Metrics",
    "💡 Insights"
])

# Tab 1: Alpha Signals
with tab1:
    st.header("Alpha Signals")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = plot_alpha_signals(momentum, mean_rev, vol_carry, composite)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Signal Mean", f"{composite.mean().mean():.4f}")
        st.metric("Signal Std", f"{composite.std().mean():.4f}")
        st.metric("Signal Skew", f"{composite.mean(axis=1).skew():.2f}")

# Tab 2: Market Regimes
with tab2:
    st.header("Market Regime Analysis")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = plot_regime_timeline(regimes)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Regime Statistics")
        regime_counts = regimes.value_counts()
        for regime, count in regime_counts.items():
            pct = count / len(regimes) * 100
            st.metric(regime, f"{count} days ({pct:.1f}%)")
        
        st.subheader("Regime Persistence")
        persistence = regime_persistence(regimes)
        for regime, days in persistence.items():
            st.metric(f"{regime} avg length", f"{days:.0f} days")

# Tab 3: Alpha Decay
with tab3:
    st.header("Alpha Decay Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with st.spinner("Calculating decay metrics..."):
        decay_metrics = fit_decay_curve(composite, returns)
        quality = alpha_quality_score(decay_metrics)
    
    with col1:
        st.metric("Quality Score", f"{quality:.1f}/100")
    with col2:
        st.metric("Initial Alpha", f"{decay_metrics['initial_alpha']:.4f}")
    with col3:
        st.metric("Decay Rate", f"{decay_metrics['decay_rate']:.4f}")
    with col4:
        st.metric("Half-life", f"{decay_metrics['half_life']:.0f} days")
    
    # Decay curve
    fig = plot_decay_curve(decay_metrics)
    st.plotly_chart(fig, use_container_width=True)
    
    # Decay by regime
    st.subheader("Decay by Market Regime")
    regime_decay = decay_by_regime(composite, returns, regimes)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_regime_performance(regime_decay)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Regime Decay Metrics**")
        for regime_name, metrics in regime_decay.items():
            if metrics:
                quality = alpha_quality_score(metrics)
                st.write(f"**{regime_name}**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.write(f"Quality: {quality:.1f}")
                with col_b:
                    st.write(f"Half-life: {metrics['half_life']:.0f}d")
                with col_c:
                    st.write(f"Decay: {metrics['decay_rate']:.4f}")

# Tab 4: Performance Metrics
with tab4:
    st.header("Performance Metrics")
    
    with st.spinner("Computing portfolio metrics..."):
        metrics = portfolio_metrics(composite, returns)
        rolling = rolling_metrics(composite, returns, window=63)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Information Coefficient", f"{metrics['ic']:.4f}")
    with col2:
        st.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
    with col3:
        st.metric("Sortino Ratio", f"{metrics['sortino']:.2f}")
    with col4:
        st.metric("Calmar Ratio", f"{metrics['calmar']:.2f}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Annual Return", f"{metrics['annual_return']:.2%}")
    with col2:
        st.metric("Annual Volatility", f"{metrics['annual_vol']:.2%}")
    with col3:
        st.metric("Max Drawdown", f"{metrics['max_dd']:.2%}")
    with col4:
        st.metric("Observations", metrics['observations'])
    
    # Rolling metrics
    if len(rolling) > 0:
        fig = plot_rolling_metrics(rolling)
        st.plotly_chart(fig, use_container_width=True)

# Tab 5: Insights
with tab5:
    st.header("🧠 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Signal Quality")
        if abs(metrics['ic']) > 0.05:
            st.success("✅ Signal shows statistically significant predictive power")
        else:
            st.warning("⚠️ Signal IC is weak - consider parameter tuning")
        
        st.subheader("Risk-Adjusted Performance")
        if metrics['sharpe'] > 1:
            st.success("✅ Positive Sharpe ratio - risk-adjusted returns are positive")
        elif metrics['sharpe'] > 0:
            st.info("ℹ️ Modest risk-adjusted returns")
        else:
            st.error("❌ Negative risk-adjusted returns - strategy underperforming")
    
    with col2:
        st.subheader("Alpha Persistence")
        if decay_metrics['half_life'] > 20:
            st.success("✅ Alpha persists for multiple weeks - good signal quality")
        elif decay_metrics['half_life'] > 5:
            st.info("ℹ️ Alpha decays within ~1 week")
        else:
            st.warning("⚠️ Alpha decays very quickly - signal may be driven by noise")
        
        st.subheader("Regime Adaptation")
        best_regime = max(regime_decay.items(), 
                         key=lambda x: alpha_quality_score(x[1]) if x[1] else 0)
        if best_regime[1]:
            best_quality = alpha_quality_score(best_regime[1])
            st.info(f"Strategy works best in **{best_regime[0]}** markets (quality: {best_quality:.1f})")

st.divider()
st.markdown("""
---
**About this application**: This engine analyzes how trading signals (alphas) decay over time
and perform across market regimes. It uses Hidden Markov Models to detect unobserved market states
and exponential curve fitting to quantify alpha persistence.

**Key concepts**:
- **Alpha**: Trading signal that predicts forward returns
- **Regime**: Hidden market state (Bull, Bear, High Volatility)
- **Decay**: How quickly an alpha loses predictive power
- **Information Coefficient (IC)**: Correlation between signal and returns
""")
