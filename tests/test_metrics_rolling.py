"""
Test module for rolling metrics computation.

This module contains unit tests for rolling financial metrics functions,
verifying their correctness with sample data.
"""

from src.metrics import compute_rolling_metrics


def test_rolling_metrics_shape(asset_price_series, benchmark_price_series):
    """
    Test that the rolling metrics have the correct shape.
    """
    rolling = compute_rolling_metrics(
        prices=asset_price_series,
        benchmark_prices=benchmark_price_series,
        window=12,
    )

    assert not rolling.empty
    assert rolling.shape[0] == len(asset_price_series)


def test_rolling_metrics_nan_before_window(asset_price_series):
    """
    Test that rolling metrics are NaN before the window size.
    """
    rolling = compute_rolling_metrics(
        prices=asset_price_series,
        window=12,
    )

    assert rolling.iloc[:11].isna().all().all()
