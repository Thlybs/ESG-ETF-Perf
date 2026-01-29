"""
Pytest configuration and shared fixtures.

This module provides common fixtures for pytest test functions,
including sample data for financial metrics testing.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(name="asset_price_series")
def asset_price_series_data():
    """
    Monotonically increasing asset price series
    with constant log-returns.
    """
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    prices = 100 * np.exp(0.01 * np.arange(24))
    return pd.Series(prices, index=dates, name="ASSET")


@pytest.fixture(name="benchmark_price_series")
def benchmark_price_series_data():
    """
    Benchmark price series with lower growth rate.
    """
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    prices = 100 * np.exp(0.008 * np.arange(24))
    return pd.Series(prices, index=dates, name="BENCHMARK")


@pytest.fixture
def aligned_price_dataframe(
    asset_price_series,
    benchmark_price_series,
):
    """
    DataFrame containing aligned ETF and benchmark prices.
    """
    return pd.concat(
        [
            asset_price_series.rename("ETF"),
            benchmark_price_series.rename("SP500"),
        ],
        axis=1,
    )
