"""
Main module for fetching ETF data from Alpha Vantage API.
"""

from datetime import datetime
import time
from config import api_key, ETF_tickers, covid_regimes, SNP500, ROLLING_WINDOW
from utils import (
    get_data,
    plot_etf_data,
    build_etf_dataframe,
    plot_all_etfs,
    align_etf_dataframe,
)
from metrics import (
    compute_summary_table,
    format_summary_table,
    compute_rolling_metrics,
    plot_rolling_metric,
)

etf_data = {}

for ticker in ETF_tickers + [SNP500]:
    data = get_data(ticker, api_key)
    print(f"Ticker: {ticker}")
    etf_data[ticker] = data
    time.sleep(1)

for ticker in ETF_tickers:
    plot_etf_data(ticker, etf_data)

plot_etf_data(SNP500, etf_data)

df = build_etf_dataframe(etf_data, ETF_tickers + [SNP500])

plot_all_etfs(df)

# date of first alignment to be defined
df_aligned = align_etf_dataframe(df, datetime(2019, 8, 1))

plot_all_etfs(df_aligned, normalize=True)

summary = compute_summary_table(df_aligned, benchmark_ticker=SNP500)
format_summary_table(summary)

rolling_metrics = {
    ticker: compute_rolling_metrics(
        prices=df_aligned[ticker].dropna(),
        benchmark_prices=df_aligned[SNP500].dropna(),
        window=ROLLING_WINDOW,
        cvar_alpha=0.95,
    )
    for ticker in df_aligned.columns
}

plot_rolling_metric(
    rolling_dict=rolling_metrics,
    metric="Sharpe Ratio",
    benchmark=SNP500,
    etfs=["PHO", "VEGI", "CHGX"],
    regimes=covid_regimes,
)
