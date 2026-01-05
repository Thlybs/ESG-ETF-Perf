"""
Main module for fetching ETF data from Alpha Vantage API.
"""

import time
from config import api_key, ETF_tickers
from utils import get_data, plot_etf_data

etf_data = {}

for ticker in ETF_tickers:
    data = get_data(ticker, api_key)
    print(f"Ticker: {ticker}", data)
    etf = {"ticker": ticker, "data": data}
    etf_data[ticker] = etf
    time.sleep(1)

for ticker in ETF_tickers:
    plot_etf_data(ticker, etf_data)
