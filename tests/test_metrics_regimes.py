"""
Test module for regime-based financial metrics calculations.

This module contains unit tests for functions that compute metrics
over specific time periods or regimes, verifying their correctness.
"""

from datetime import datetime
from src.metrics import compute_regime_metrics


def test_regime_metrics_keys(aligned_price_dataframe):
    """
    Test that regime metrics returns keys matching input regimes.

    Verifies that compute_regime_metrics produces results for each
    defined regime period, with keys matching the input regime names.
    """
    regimes = {
        "crisis": (datetime(2020, 2, 1), datetime(2020, 4, 1)),
        "recovery": (datetime(2020, 5, 1), datetime(2020, 12, 1)),
    }

    results = compute_regime_metrics(
        aligned_price_dataframe,
        benchmark_ticker="SP500",
        regimes=regimes,
    )

    assert set(results.keys()) == set(regimes.keys())
