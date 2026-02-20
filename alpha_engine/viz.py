"""Data visualization utilities."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_alpha_signals(
    momentum: pd.DataFrame,
    mean_rev: pd.DataFrame,
    vol_carry: pd.DataFrame,
    composite: pd.DataFrame,
    start_date: str = None,
    end_date: str = None
) -> go.Figure:
    """
    Visualize all alpha signals over time (rolling average across assets).
    
    Shows how each alpha component evolves and contributes to the composite.
    """
    # Rolling average across assets
    momentum_avg = momentum.mean(axis=1)
    mean_rev_avg = mean_rev.mean(axis=1)
    vol_carry_avg = vol_carry.mean(axis=1)
    composite_avg = composite.mean(axis=1)
    
    # Trim date range if specified
    if start_date:
        momentum_avg = momentum_avg[start_date:]
        mean_rev_avg = mean_rev_avg[start_date:]
        vol_carry_avg = vol_carry_avg[start_date:]
        composite_avg = composite_avg[start_date:]
    if end_date:
        momentum_avg = momentum_avg[:end_date]
        mean_rev_avg = mean_rev_avg[:end_date]
        vol_carry_avg = vol_carry_avg[:end_date]
        composite_avg = composite_avg[:end_date]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=momentum_avg.index, y=momentum_avg, name='Momentum', mode='lines'))
    fig.add_trace(go.Scatter(x=mean_rev_avg.index, y=mean_rev_avg, name='Mean Reversion', mode='lines'))
    fig.add_trace(go.Scatter(x=vol_carry_avg.index, y=vol_carry_avg, name='Vol Carry', mode='lines'))
    fig.add_trace(go.Scatter(
        x=composite_avg.index, y=composite_avg, name='Composite',
        mode='lines', line=dict(width=3, dash='solid')
    ))
    
    fig.update_layout(
        title='Alpha Signals Over Time',
        xaxis_title='Date',
        yaxis_title='Signal (normalized)',
        hovermode='x unified',
        height=500
    )
    
    return fig


def plot_regime_timeline(regimes: pd.Series) -> go.Figure:
    """
    Visualize market regimes as a colored timeline.
    
    Each regime gets a distinct color; this helps see how regimes
    cluster and transition over time.
    """
    # Map regime names to colors
    regime_colors = {
        'Bull': '#2ecc71',
        'Bear': '#e74c3c',
        'High Volatility': '#f39c12'
    }
    
    colors = [regime_colors.get(r, '#95a5a6') for r in regimes]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=regimes.index,
        y=np.ones_like(regimes),
        mode='markers',
        marker=dict(size=8, color=colors),
        name='Regime',
        hovertext=regimes.values
    ))
    
    fig.update_layout(
        title='Market Regime Timeline',
        xaxis_title='Date',
        yaxis_title='Regime',
        showlegend=False,
        height=300,
        yaxis=dict(showticklabels=False)
    )
    
    return fig


def plot_decay_curve(decay_metrics: dict) -> go.Figure:
    """
    Plot the fitted exponential decay curve of alpha.
    
    Shows how alpha effectiveness decays at different forecast horizons.
    """
    windows = decay_metrics['windows']
    forward_returns = decay_metrics['forward_returns']
    
    # Generate smooth curve
    x_smooth = np.linspace(windows[0], windows[-1], 100)
    a = decay_metrics['initial_alpha']
    b = decay_metrics['decay_rate']
    c = decay_metrics['persistent_alpha']
    y_smooth = a * np.exp(-b * x_smooth) + c
    
    fig = go.Figure()
    
    # Actual data points
    fig.add_trace(go.Scatter(
        x=windows, y=forward_returns,
        mode='markers+lines',
        name='Actual returns',
        marker=dict(size=10)
    ))
    
    # Fitted curve
    fig.add_trace(go.Scatter(
        x=x_smooth, y=y_smooth,
        mode='lines',
        name='Fitted curve',
        line=dict(dash='dash', width=2)
    ))
    
    fig.update_layout(
        title=f"Alpha Decay (Half-life: {decay_metrics['half_life']:.0f} days)",
        xaxis_title='Days ahead',
        yaxis_title='Forward return',
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_rolling_metrics(rolling_metrics_dict: dict) -> go.Figure:
    """
    Plot rolling Sharpe, IC, and max drawdown over time.
    
    Helps identify periods when the alpha is working well vs poorly.
    """
    rolling_df = pd.DataFrame(rolling_metrics_dict).T
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=('Sharpe Ratio', 'Information Coefficient', 'Max Drawdown')
    )
    
    # Sharpe Ratio
    fig.add_trace(
        go.Scatter(x=rolling_df.index, y=rolling_df['sharpe'], name='Sharpe', fill='tozeroy'),
        row=1, col=1
    )
    
    # IC
    fig.add_trace(
        go.Scatter(x=rolling_df.index, y=rolling_df['ic'], name='IC', fill='tozeroy'),
        row=2, col=1
    )
    
    # Max Drawdown
    fig.add_trace(
        go.Scatter(x=rolling_df.index, y=rolling_df['max_dd'], name='Max DD', fill='tozeroy'),
        row=3, col=1
    )
    
    fig.update_yaxes(title_text='Sharpe', row=1, col=1)
    fig.update_yaxes(title_text='IC', row=2, col=1)
    fig.update_yaxes(title_text='Max DD', row=3, col=1)
    fig.update_xaxes(title_text='Date', row=3, col=1)
    
    fig.update_layout(height=800, title_text='Rolling Performance Metrics', hovermode='x unified')
    
    return fig


def plot_regime_performance(regime_decay: dict) -> go.Figure:
    """
    Compare alpha decay across different market regimes.
    
    Shows which regimes are most favorable for the alpha strategy.
    """
    fig = go.Figure()
    
    for regime_name, metrics in regime_decay.items():
        if metrics is None:
            continue
        
        windows = metrics['windows']
        forward_returns = metrics['forward_returns']
        
        fig.add_trace(go.Scatter(
            x=windows, y=forward_returns,
            mode='lines+markers',
            name=regime_name,
            line=dict(width=3)
        ))
    
    fig.update_layout(
        title='Alpha Performance by Market Regime',
        xaxis_title='Days ahead',
        yaxis_title='Forward return',
        hovermode='x unified',
        height=500
    )
    
    return fig


def plot_cumulative_returns(returns: pd.Series, name: str = 'Strategy') -> go.Figure:
    """
    Plot cumulative return curve (growth of $1).
    """
    cumulative = (1 + returns).cumprod()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative.index, y=cumulative,
        mode='lines',
        name=name,
        line=dict(width=3)
    ))
    
    fig.update_layout(
        title=f'{name} Cumulative Return',
        xaxis_title='Date',
        yaxis_title='Cumulative Return ($)',
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_drawdown(returns: pd.Series) -> go.Figure:
    """
    Plot drawdown over time (underwater plot).
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='#e74c3c')
    ))
    
    fig.update_layout(
        title='Drawdown Over Time',
        xaxis_title='Date',
        yaxis_title='Drawdown',
        hovermode='x unified',
        height=400
    )
    
    return fig
