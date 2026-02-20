

# Step 1: Data module implementation

import yfinance as yf
import pandas as pd
import numpy as np


def fetch_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices from Yahoo Finance.
    Returns a DataFrame of daily log returns, one column per ticker.
    
    Log returns (ln(P_t / P_{t-1})) are preferred over simple returns in quant
    finance because they're time-additive and more normally distributed.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # If multiple tickers, yfinance returns MultiIndex columns — extract 'Close'
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    # Drop any ticker columns that are entirely NaN
    prices = prices.dropna(axis=1, how="all")

    # Compute daily log returns
    log_returns = np.log(prices / prices.shift(1))

    # Drop the first row (NaN from shift)
    log_returns = log_returns.dropna(how="all")

    return log_returns


def get_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Same as fetch_data but returns raw adjusted close prices (needed for
    some alpha calculations like momentum that work on price levels).
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    prices = prices.dropna(axis=1, how="all")
    return prices
