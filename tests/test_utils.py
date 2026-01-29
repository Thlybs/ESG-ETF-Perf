"""
Test module for utility functions.

This module contains unit tests for general utility functions,
verifying their correctness and behavior.
"""

from datetime import datetime
from src.utils import align_etf_dataframe


def test_align_dataframe(aligned_price_dataframe):
    """
    Test dataframe alignment to a reference date.

    Verifies that align_etf_dataframe correctly aligns the dataframe
    to the specified reference date, ensuring no NaN values and
    monotonic increasing index.
    """
    aligned = align_etf_dataframe(
        aligned_price_dataframe,
        reference_date=datetime(2020, 6, 1),
    )

    assert aligned.notna().all().all()

    effective_date = aligned.index.min()
    assert aligned.loc[effective_date].notna().all()

    assert aligned.index.is_monotonic_increasing

    assert effective_date <= datetime(2020, 6, 1)
