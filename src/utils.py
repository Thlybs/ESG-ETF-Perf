"""
Utility functions for fetching and plotting ETF data.
"""

from datetime import datetime
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


def plot_etf_data(ticker, etf_data):
    """
    Plot the adjusted close price for a given ETF ticker.

    Args:
        ticker (str): The stock symbol.
        etf_data (dict): Dictionary containing ETF data.
    """
    time_series = etf_data[ticker]["data"]["Monthly Adjusted Time Series"]

    dates = [datetime.strptime(date, "%Y-%m-%d") for date in time_series.keys()]
    dates.sort()
    adjusted_closes = [
        float(time_series[date.strftime("%Y-%m-%d")]["5. adjusted close"])
        for date in dates
    ]

    years = sorted(set(date.year for date in dates))
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
