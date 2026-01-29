"""
Test module for basic financial metrics calculations.

This module contains unit tests for core financial metrics functions,
verifying their correctness with sample data.
"""

import numpy as np
from src.metrics import (
    compute_log_returns,
    compute_cagr,
    compute_volatility,
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_cvar,
)


def test_log_returns(asset_price_series):
    """
    Test that log returns are constant for constant growth prices.

    Verifies that compute_log_returns produces identical values
    for a price series with constant log-returns.
    """
    returns = compute_log_returns(asset_price_series)
    assert np.allclose(returns, returns.iloc[0])


def test_cagr_positive(asset_price_series):
    """
    Test that CAGR is positive for increasing asset prices.

    Verifies that compute_cagr returns a positive value for
    monotonically increasing price series.
    """
    assert compute_cagr(asset_price_series) > 0


def test_volatility_zero_for_constant_returns(asset_price_series):
    """
    Test that volatility is zero for constant returns.

    Verifies that compute_volatility returns zero for a series
    of constant log-returns, indicating no variability.
    """
    returns = compute_log_returns(asset_price_series)
    assert np.isclose(compute_volatility(returns), 0.0, atol=1e-12)


def test_sharpe_finite(asset_price_series):
    """
    Test that Sharpe ratio is finite for valid returns.

    Verifies that compute_sharpe returns a finite value
    for a series of log-returns, indicating successful calculation.
    """
    returns = compute_log_returns(asset_price_series)
    sharpe = compute_sharpe(returns)
    assert np.isfinite(sharpe)


def test_sortino_nan_if_no_downside(asset_price_series):
    """
    Test that Sortino ratio is NaN when no downside returns exist.

    Verifies that compute_sortino returns NaN for a returns series
    with no negative values, as Sortino requires downside deviation.
    """
    returns = compute_log_returns(asset_price_series)
    assert np.isnan(compute_sortino(returns))


def test_max_drawdown_zero_for_monotonic_prices(asset_price_series):
    """
    Test that max drawdown is zero for monotonically increasing prices.

    Verifies that compute_max_drawdown returns zero for a price series
    that never decreases, indicating no drawdown occurred.
    """
    assert compute_max_drawdown(asset_price_series) == 0.0


def test_cvar_constant_returns(asset_price_series):
    """
    Test CVaR for constant returns series.

    Verifies that compute_cvar returns a finite value equal to the
    constant return value for a series with no variability.
    """
    returns = compute_log_returns(asset_price_series)
    cvar = compute_cvar(returns)

    assert np.isfinite(cvar)
    assert np.isclose(cvar, returns.iloc[0])
