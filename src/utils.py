"""
Utility functions for fetching and plotting ETF data.
"""

from datetime import datetime
import pandas as pd
import requests
import matplotlib.pyplot as plt


def get_data(ticker=str, api_key=str):
    """
    Fetch monthly adjusted time series data for a given ticker from Alpha Vantage.

    Args:
        ticker (str): The stock symbol to fetch data for.
        api_key (str): The Alpha Vantage API key.

    Returns:
        dict: The JSON response from the API containing the time series data.
    """
    url = (
        "https://www.alphavantage.co/query?"
        "function=TIME_SERIES_MONTHLY_ADJUSTED&"
        f"symbol={ticker}&"
        f"apikey={api_key}"
    )
    r = requests.get(url, timeout=10)
    return r.json()


def validate_alpha_vantage_response(data: dict) -> None:
    """
    Validate that Alpha Vantage response contains monthly adjusted data.

    Raises:
        ValueError if expected data is missing.
    """
    if "Monthly Adjusted Time Series" not in data:
        raise ValueError(
            "Alpha Vantage response does not contain 'Monthly Adjusted Time Series'."
        )

    if not data["Monthly Adjusted Time Series"]:
        raise ValueError("Monthly Adjusted Time Series is empty.")


def plot_etf_data(ticker, etf_data):
    """
    Plot the adjusted close price for a given ETF ticker.

    Args:
        ticker (str): The stock symbol.
        etf_data (dict): Dictionary containing ETF data.
    """
    time_series = etf_data[ticker]["Monthly Adjusted Time Series"]

    dates = [datetime.strptime(date, "%Y-%m-%d") for date in time_series.keys()]
    dates.sort()
    adjusted_closes = [
        float(time_series[date.strftime("%Y-%m-%d")]["5. adjusted close"])
        for date in dates
    ]

    years = set(date.year for date in dates)
    january_dates = []
    for year in years:
        for date in dates:
            if date.year == year and date.month == 1:
                january_dates.append(date)
                break

    plt.figure(figsize=(12, 6))
    plt.plot(dates, adjusted_closes, marker="o", linestyle="-", color="b")
    plt.xticks(january_dates, [date.year for date in january_dates], rotation=45)

    plt.ylim(bottom=0)
    plt.title("Évolution du prix ajusté de clôture - " + ticker)
    plt.xlabel("Années")
    plt.ylabel("Prix (USD)")
    plt.grid(True)
    plt.tight_layout()

    return plt.show()


def build_etf_dataframe(etf_data, tickers):
    """
    Build a standardized DataFrame from Alpha Vantage ETF data.

    Args:
        ticker (str): The stock symbol.
        etf_data (dict): Dictionary containing ETF data.

    Returns:
    pd.DataFrame: A wide DataFrame with:
        index  : datetime (monthly)
        columns: tickers
        values : adjusted close prices
    """
    series = []

    for ticker in tickers:
        time_series = etf_data[ticker]["Monthly Adjusted Time Series"]

        s = (
            pd.Series(
                {
                    datetime.strptime(date, "%Y-%m-%d"): float(
                        values["5. adjusted close"]
                    )
                    for date, values in time_series.items()
                }
            )
            .sort_index()
            .rename(ticker)
        )

        series.append(s)

    df = pd.concat(series, axis=1).sort_index()

    df = df[df.count().sort_values(ascending=False).index]

    return df


def plot_all_etfs(
    df,
    normalize=False,
    normalization_date=None,
    log_scale=False,
    exclude_tickers=None,
):
    """
    Plot ETF price evolution.

    Args:
        df (pd.DataFrame): ETF price DataFrame (date index, tickers as columns)
        normalize (bool): Normalize prices to base 100
        normalization_date (datetime or None): Date used for normalization.
            If None and normalize=True, uses the last common available date.
        log_scale (bool): Use logarithmic y-axis
        exclude_tickers (list or None): Tickers to exclude from the plot
    """

    if normalize and log_scale:
        raise ValueError("normalize and log_scale cannot both be True.")

    plt.figure(figsize=(14, 7))

    if exclude_tickers:
        df = df.drop(columns=exclude_tickers, errors="ignore")

        df = df.dropna(axis=1, how="all")

        if df.empty:
            raise ValueError("No ETF data left to plot after exclusions.")

    start_dates = df.apply(lambda s: s.first_valid_index())
    df = df[start_dates.sort_values().index]

    subtitle = ""

    if normalize:
        if normalization_date is None:
            normalization_date = df.dropna().index.min()

        if normalization_date not in df.index:
            raise ValueError("Normalization date not in DataFrame index.")

        base_prices = df.loc[normalization_date]

        if base_prices.isna().any():
            raise ValueError(
                "Some ETFs have no data at the normalization date. "
                "Align data before normalizing."
            )

        df = df / base_prices * 100
        subtitle = f"Base 100 at {normalization_date.strftime('%m/%Y')}"

    if log_scale:
        subtitle = "Log scale"

    for ticker in df.columns:
        series = df[ticker].dropna()
        first_date = series.index[0]

        label = f"{ticker} ({first_date.strftime('%m/%Y')})"

        (line,) = plt.plot(series.index, series.values, label=label)
        color = line.get_color()

        plt.axvline(first_date, linestyle="--", alpha=0.4, color=color)

    plt.xlim(df.index.min(), df.index.max())

    if log_scale:
        plt.yscale("log")

    plt.xlabel("Year")
    ylabel = "Adjusted Close Price"
    if normalize:
        ylabel += " (Base 100)"
    plt.ylabel(ylabel)

    title = "ETF Price Evolution"
    if subtitle:
        title += f"\n{subtitle}"

    plt.title(title)
    plt.grid(True)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.legend()
    plt.tight_layout()
    plt.show()


def align_etf_dataframe(df, reference_date):
    """
    Align ETF DataFrame to a common date range based on a reference date.

    Args:
        df (pd.DataFrame): price DataFrame (date index, tickers as columns)
        reference_date (datetime): reference date for alignment

    Returns:
        pd.DataFrame: aligned DataFrame (all ETFs have non-NA data)
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    eligible_dates = df.index[df.index <= reference_date]
    if eligible_dates.empty:
        raise ValueError("No data available before the reference date.")

    effective_date = eligible_dates.max()

    valid_etfs = df.loc[effective_date].dropna().index.tolist()
    if not valid_etfs:
        raise ValueError("No ETFs have data at the reference date.")

    df_valid = df[valid_etfs]

    mask = df_valid.notna().all(axis=1)
    if not mask.any():
        raise ValueError("No common date range with all ETFs having data.")

    common_start = mask.idxmax()
    common_end = mask[::-1].idxmax()

    return df_valid.loc[common_start:common_end]
