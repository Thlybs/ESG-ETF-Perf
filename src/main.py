"""
Main entry point of the project.

This module:
- Fetches ETF and benchmark price data
- Builds and aligns price DataFrames
- Computes descriptive performance metrics
- Computes rolling, regime-aware, and crisis-specific metrics
- Produces tables and visualizations for exploratory analysis

This script corresponds to the descriptive statistics phase
of the project. Forecasting models and ETF overlap analysis
will be added in later stages.
"""

from datetime import datetime
import time
from IPython.display import display
from src.config import api_key, ETF_tickers, covid_regimes, SNP500, ROLLING_WINDOW
from src.utils import (
    get_data,
    validate_alpha_vantage_response,
    plot_etf_data,
    build_etf_dataframe,
    plot_all_etfs,
    align_etf_dataframe,
)
from src.metrics import (
    compute_summary_table,
    format_summary_table,
    compute_rolling_metrics,
    plot_rolling_metric,
    compute_regime_metrics,
    crisis_performance_decomposition,
)

etf_data = {}

for ticker in ETF_tickers + [SNP500]:
    print(f"Ticker: {ticker}")
    data = get_data(ticker, api_key)
    validate_alpha_vantage_response(data)
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
display(format_summary_table(summary))

rolling_metrics = {
    ticker: compute_rolling_metrics(
        prices=df_aligned[ticker].dropna(),
        benchmark_prices=df_aligned[SNP500],
        window=ROLLING_WINDOW,
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

regime_metrics = compute_regime_metrics(
    df_aligned,
    benchmark_ticker=SNP500,
    regimes=covid_regimes,
)

display(format_summary_table(regime_metrics["COVID crash"]))
display(format_summary_table(regime_metrics["Recovery"]))

decomp = crisis_performance_decomposition(
    df_aligned,
    benchmark_ticker=SNP500,
    crisis_period=covid_regimes["COVID crash"],
    recovery_period=covid_regimes["Recovery"],
)

for phase, table in decomp.items():
    print(f"\n=== {phase} ===")
    display(format_summary_table(table))
