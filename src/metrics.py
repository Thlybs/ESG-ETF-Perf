"""
This module computes a set of standard and tail-sensitive financial performance
metrics used in academic research and professional asset management to evaluate
ETF risk-return profiles and compare them to a benchmark (e.g. S&P 500).

All metrics are computed using monthly log returns unless stated otherwise,
which ensures time additivity, robustness to compounding effects, and consistent
treatment of crisis periods (e.g. COVID-19).

Interpretation of the main metrics:

• CAGR (Compound Annual Growth Rate)
  Measures the annualized growth rate of an investment over the full period.
  It reflects long-term performance but does not capture risk or drawdowns.
  Higher CAGR indicates stronger average growth.

• Volatility (Annualized)
  Standard deviation of returns, annualized.
  Measures total risk (upside + downside). Higher volatility implies greater
  uncertainty in returns.

• Sharpe Ratio
  Risk-adjusted performance using total volatility.
  Interpreted as excess return per unit of total risk.
  Higher values indicate better risk-adjusted performance.

• Sortino Ratio
  Downside-risk-adjusted performance.
  Similar to Sharpe, but penalizes only negative returns.
  Particularly relevant for investors concerned with losses rather than upside
  volatility. Higher is better.

• Max Drawdown
  Maximum peak-to-trough loss over the period.
  Captures worst historical loss and tail exposure.
  More negative values indicate deeper drawdowns.

• CVaR (Conditional Value at Risk, 95%)
  Expected loss conditional on being in the worst 5% of return outcomes.
  A tail-risk measure more informative than volatility during crisis periods.
  More negative values indicate higher downside tail risk.

• Beta (vs Benchmark)
  Measures sensitivity to benchmark market movements.
  Beta > 1 implies higher systematic risk than the benchmark,
  Beta < 1 implies lower market exposure.

• Tracking Error
  Standard deviation of excess returns relative to the benchmark.
  Measures how closely an ETF follows the benchmark.
  Lower values indicate closer tracking.

• Information Ratio
  Risk-adjusted excess return relative to the benchmark.
  Interpreted as active return per unit of tracking error.
  Higher values indicate better benchmark-relative performance.

Rolling versions of these metrics (when enabled) allow analysis of time-varying
risk, performance, and regime dependence, highlighting how ETFs behave during
crises, recoveries, and normal market conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute logarithmic returns from a price series.

    Formula
    r_t = log(P_t / P_{t-1})

    Notes
    Log returns are time-additive and preferred for risk and tail analysis.

    Args:
        prices (pd.Series): Time series of asset prices.

    Returns:
        pd.Series: Logarithmic returns, dropping the first NaN value.
    """
    return np.log(prices / prices.shift(1)).dropna()


def compute_cagr(prices: pd.Series, periods_per_year=12) -> float:
    """
    Compute the Compound Annual Growth Rate (CAGR).

    Formula
    CAGR = (P_T / P_0)^(k / N) - 1

    Where
    P_0 : initial price
    P_T : final price
    N   : number of observations
    k   : periods per year

    Args:
        prices (pd.Series): Time series of asset prices.
        periods_per_year (int, optional): Number of periods per year. Defaults to 12.

    Returns:
        float: CAGR as a decimal.
    """
    n = len(prices)
    return (prices.iloc[-1] / prices.iloc[0]) ** (periods_per_year / n) - 1


def compute_volatility(returns: pd.Series, periods_per_year=12) -> float:
    """
    Compute the annualized volatility of returns.

    Formula
    σ = sqrt(k) * std(r_t)

    Where
    r_t : log returns
    k   : periods per year

    Args:
        returns (pd.Series): Time series of asset returns.
        periods_per_year (int, optional): Number of periods per year. Defaults to 12.

    Returns:
        float: Volatility as a decimal.
    """
    return returns.std() * np.sqrt(periods_per_year)


def compute_sharpe(
    returns: pd.Series, risk_free_rate=0.0, periods_per_year=12
) -> float:
    """
    Compute the Sharpe ratio.

    Formula
    Sharpe = E[r_t - r_f] / std(r_t - r_f) * sqrt(k)

    Where
    r_t : returns
    r_f : risk-free rate
    k   : periods per year

    Args:
        returns (pd.Series): Time series of asset returns.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.0.
        periods_per_year (int, optional): Number of periods per year. Defaults to 12.

    Returns:
        float: Sharpe ratio.
    """
    excess = returns - risk_free_rate / periods_per_year
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)


def compute_sortino(
    returns: pd.Series, risk_free_rate=0.0, periods_per_year=12
) -> float:
    """
    Compute the Sortino ratio.

    Formula
    Sortino = E[r_t - r_f] / std(r_t | r_t < 0) * sqrt(k)

    Notes
    Penalizes only downside volatility.

    Args:
        returns (pd.Series): Time series of asset returns.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.0.
        periods_per_year (int, optional): Number of periods per year. Defaults to 12.

    Returns:
        float: Sortino ratio, or NaN if no downside returns.
    """
    downside = returns[returns < 0]
    if len(downside) == 0:
        return np.nan
    return (
        (returns.mean() - risk_free_rate / periods_per_year)
        / downside.std()
        * np.sqrt(periods_per_year)
    )


def compute_max_drawdown(prices: pd.Series) -> float:
    """
    Compute the maximum drawdown.

    Formula
    DD_t = P_t / max(P_s for s <= t) - 1
    Max DD = min(DD_t)

    Args:
        prices (pd.Series): Time series of asset prices.

    Returns:
        float: Maximum drawdown as a decimal.
    """
    cumulative = prices / prices.iloc[0]
    drawdown = cumulative / cumulative.cummax() - 1
    return drawdown.min()


def compute_cvar(returns: pd.Series, alpha=0.95) -> float:
    """
    Compute the Conditional Value at Risk (CVaR).

    Formula
    CVaR_α = E[r_t | r_t ≤ VaR_α]

    Notes
    Measures expected loss in the worst (1 - α)% of cases.

    Args:
        returns (pd.Series): Time series of asset returns.
        alpha (float, optional): Confidence level. Defaults to 0.95.

    Returns:
        float: CVaR as a decimal.
    """
    var = returns.quantile(1 - alpha)
    return returns[returns <= var].mean()


def compute_beta(asset_returns, benchmark_returns) -> float:
    """
    Compute the beta of an asset relative to a benchmark.

    Formula
    β = Cov(r_i, r_m) / Var(r_m)

    Args:
        asset_returns (pd.Series): Returns of the asset.
        benchmark_returns (pd.Series): Returns of the benchmark.

    Returns:
        float: Beta coefficient.
    """
    return np.cov(asset_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)


def compute_tracking_error(
    asset_returns, benchmark_returns, periods_per_year=12
) -> float:
    """
    Compute the tracking error.

    Formula
    TE = sqrt(k) * std(r_i - r_m)

    Args:
        asset_returns (pd.Series): Returns of the asset.
        benchmark_returns (pd.Series): Returns of the benchmark.
        periods_per_year (int, optional): Number of periods per year. Defaults to 12.

    Returns:
        float: Annualized tracking error as a decimal.
    """
    return (asset_returns - benchmark_returns).std() * np.sqrt(periods_per_year)


def compute_information_ratio(
    asset_returns, benchmark_returns, periods_per_year=12
) -> float:
    """
    Compute the information ratio.

    Formula
    IR = E[r_i - r_m] / std(r_i - r_m) * sqrt(k)

    Args:
        asset_returns (pd.Series): Returns of the asset.
        benchmark_returns (pd.Series): Returns of the benchmark.
        periods_per_year (int, optional): Number of periods per year. Defaults to 12.

    Returns:
        float: Information ratio.
    """
    excess = asset_returns - benchmark_returns
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)


def compute_summary_table(
    price_df: pd.DataFrame,
    benchmark_ticker: str,
    cvar_alpha=0.95,
):
    """
    Compute a summary table of performance metrics for multiple assets.

    Args:
        price_df (pd.DataFrame): DataFrame with price series for each asset.
        benchmark_ticker (str): Ticker of the benchmark asset.
        cvar_alpha (float, optional): Confidence level for CVaR. Defaults to 0.95.

    Returns:
        pd.DataFrame: Summary table with metrics for each asset.
    """
    summary = {}

    benchmark_prices = price_df[benchmark_ticker].dropna()
    benchmark_returns = compute_log_returns(benchmark_prices)

    for ticker in price_df.columns:
        prices = price_df[ticker].dropna()
        returns = compute_log_returns(prices)

        metrics = {
            "CAGR": compute_cagr(prices),
            "Volatility": compute_volatility(returns),
            "Sharpe Ratio": compute_sharpe(returns),
            "Sortino Ratio": compute_sortino(returns),
            "Max Drawdown": compute_max_drawdown(prices),
            f"CVaR ({int(cvar_alpha*100)}%)": compute_cvar(returns, cvar_alpha),
        }

        if ticker != benchmark_ticker:
            aligned = returns.align(benchmark_returns, join="inner")
            metrics.update(
                {
                    "Beta": compute_beta(*aligned),
                    "Tracking Error": compute_tracking_error(*aligned),
                    "Information Ratio": compute_information_ratio(*aligned),
                }
            )

        summary[ticker] = metrics

    return pd.DataFrame(summary).T


def format_summary_table(summary_df, benchmark="IVV"):
    """
    Format and style the summary table for display.

    Args:
        summary_df (pd.DataFrame): Summary table from compute_summary_table.
        benchmark (str, optional): Benchmark ticker for coloring. Defaults to "IVV".

    Returns:
        pd.io.formats.style.Styler: Styled DataFrame.
    """

    def color_cagr(val, ref):
        if pd.isna(val) | ((val > 0.75 * ref) & (val <= ref)):
            return ""
        return "color: green" if val > ref else "color: red"

    def color_high_good(val, green_thr, red_thr):
        if pd.isna(val):
            return ""
        if val >= green_thr:
            return "color: green"
        if val <= red_thr:
            return "color: red"
        return ""

    def color_low_good(val, green_thr, red_thr):
        if pd.isna(val):
            return ""
        if val >= green_thr:
            return "color: green"
        if val <= red_thr:
            return "color: red"
        return ""

    bench = summary_df.loc[benchmark]

    styled = (
        summary_df.style.format(
            {
                "CAGR": "{:.1%}",
                "Volatility": "{:.1%}",
                "Max Drawdown": "{:.1%}",
                "CVaR (95%)": "{:.1%}",
                "Sharpe Ratio": "{:.2f}",
                "Sortino Ratio": "{:.2f}",
                "Beta": "{:.2f}",
                "Tracking Error": "{:.1%}",
                "Information Ratio": "{:.2f}",
            }
        )
        .apply(
            lambda col: [
                color_cagr(v, bench["CAGR"]) if col.name == "CAGR" else "" for v in col
            ],
            axis=0,
        )
        .apply(
            lambda col: [
                color_high_good(v, 0.5, 0.3) if col.name == "Sharpe Ratio" else ""
                for v in col
            ],
            axis=0,
        )
        .apply(
            lambda col: [
                color_high_good(v, 0.7, 0.4) if col.name == "Sortino Ratio" else ""
                for v in col
            ],
            axis=0,
        )
        .apply(
            lambda col: [
                color_low_good(v, -0.25, -0.35) if col.name == "Max Drawdown" else ""
                for v in col
            ],
            axis=0,
        )
        .apply(
            lambda col: [
                color_cagr(v, bench["CVaR (95%)"]) if col.name == "CVaR (95%)" else ""
                for v in col
            ],
            axis=0,
        )
        .apply(
            lambda col: [
                color_high_good(v, 0.0, -0.2) if col.name == "Information Ratio" else ""
                for v in col
            ],
            axis=0,
        )
    )

    return styled


def _rolling_apply(series: pd.Series, func, window: int, **kwargs) -> pd.Series:
    """
    Apply a scalar-valued function on rolling windows of a Series.
    """
    return series.rolling(window).apply(lambda x: func(x, **kwargs), raw=False)


def _align_returns(asset_returns: pd.Series, benchmark_returns: pd.Series):
    """
    Align asset and benchmark returns on common dates.
    """
    return asset_returns.align(benchmark_returns, join="inner")


def compute_rolling_metrics(
    prices: pd.Series,
    window: int,
    benchmark_prices: pd.Series | None = None,
    cvar_alpha: float = 0.95,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """
    Compute rolling performance metrics for a single asset.

    Args:
        prices (pd.Series): Asset price series.
        window (int): Rolling window size (in periods).
        benchmark_prices (pd.Series, optional): Benchmark price series.
        cvar_alpha (float, optional): Confidence level for CVaR.
        periods_per_year (int, optional): Annualization factor.

    Returns:
        pd.DataFrame: Rolling metrics indexed by date.
    """
    returns = compute_log_returns(prices)

    metrics = {}

    metrics["CAGR"] = _rolling_apply(
        prices, compute_cagr, window, periods_per_year=periods_per_year
    )

    metrics["Max Drawdown"] = _rolling_apply(prices, compute_max_drawdown, window)

    metrics["Volatility"] = returns.rolling(window).std() * np.sqrt(periods_per_year)

    metrics["Sharpe Ratio"] = _rolling_apply(
        returns, compute_sharpe, window, periods_per_year=periods_per_year
    )

    metrics["Sortino Ratio"] = _rolling_apply(
        returns, compute_sortino, window, periods_per_year=periods_per_year
    )

    metrics[f"CVaR ({int(cvar_alpha*100)}%)"] = _rolling_apply(
        returns, compute_cvar, window, alpha=cvar_alpha
    )

    if benchmark_prices is not None:
        benchmark_returns = compute_log_returns(benchmark_prices)
        r, b = _align_returns(returns, benchmark_returns)

        metrics["Beta"] = r.rolling(window).cov(b) / b.rolling(window).var()

        excess = r - b

        metrics["Tracking Error"] = excess.rolling(window).std() * np.sqrt(
            periods_per_year
        )

        metrics["Information Ratio"] = (
            excess.rolling(window).mean()
            / excess.rolling(window).std()
            * np.sqrt(periods_per_year)
        )

    return pd.DataFrame(metrics)


def plot_rolling_metric(
    rolling_dict,
    metric,
    benchmark,
    etfs=None,
    regimes=None,
):
    """
    Plot rolling metrics for ETFs and benchmark.

    Args:
        rolling_dict (dict): Dictionary of rolling metrics DataFrames.
        metric (str): Name of the metric to plot.
        benchmark (str): Benchmark ticker.
        etfs (list, optional): List of ETF tickers to plot. If None, plots ESG median.
        regimes (dict, optional): Dictionary of regime periods to highlight.
    """
    plt.figure(figsize=(12, 6))

    if etfs is None:
        esg = pd.concat(
            [df[metric] for k, df in rolling_dict.items() if k != benchmark], axis=1
        )
        plt.plot(esg.index, esg.median(axis=1), label="ESG median", linewidth=2)
    else:
        for etf in etfs:
            plt.plot(rolling_dict[etf][metric], label=etf)

    plt.plot(
        rolling_dict[benchmark][metric],
        label=benchmark,
        linestyle="--",
        linewidth=2,
    )

    if regimes:
        colors = ["#f28e2b", "#4e79a7"]
        for (name, (start, end)), color in zip(regimes.items(), colors):
            plt.axvspan(start, end, alpha=0.25, color=color, label=name)

    plt.title(f"Rolling {metric}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
