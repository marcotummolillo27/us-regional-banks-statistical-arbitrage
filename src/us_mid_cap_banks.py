
"""
Statistical Arbitrage in US Regional Banks
============================================

Data acquisition and initial validation.

This module is part of an independent quantitative research project.
It does not make investment recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

@dataclass(frozen=True)
class ResearchConfig:
    start_date: str = "2005-01-01"
    end_date: str = "2026-01-01"
    min_observations: int = 252


CONFIG = ResearchConfig()

BANK_TICKERS = (
    "WAL", "ZION", "KEY", "WFC", "RF", "HBAN", "CFG",
    "FITB", "FNB", "EWBC", "ONB", "PB", "CBSH", "UMBF",
    "CFR", "BOKF", "WTFC", "WBS", "FHN", "FULT", "HOMB",
    "UBSI", "CVBF", "FFIN", "FHB", "INDB", "COLB", "BANF",
    "SSB", "BPOP", "CATY", "UCB", "OZK", "FCNCA", "FBK",
    "CAL", "NIC", "HOPE", "NBTB", "TFSL",
)


# --------------------------------------------------
# 2. LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 3. DATA ACQUISITION
# --------------------------------------------------

def download_price_data(
    tickers: tuple[str, ...] = BANK_TICKERS,
    start: str = CONFIG.start_date,
    end: str = CONFIG.end_date,
) -> pd.DataFrame:
    """
    Download adjusted daily closing prices using Yahoo Finance.

    Missing observations are preserved. We avoid dropping every
    date containing a missing value across the entire universe.
    """

    logger.info("Downloading data for %d tickers", len(tickers))

    raw_data = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw_data.empty:
        raise ValueError("Yahoo Finance returned an empty dataset.")

    # Handle both single-index and multi-index column formats.
    if isinstance(raw_data.columns, pd.MultiIndex):
        if "Close" in raw_data.columns.get_level_values(0):
            prices = raw_data["Close"]
        elif "Close" in raw_data.columns.get_level_values(1):
            prices = raw_data.xs(
                "Close", axis=1, level=1
            )
        else:
            raise ValueError("Could not locate closing prices.")
    else:
        if "Close" not in raw_data.columns:
            raise ValueError("Could not locate closing prices.")
        prices = raw_data[["Close"]]
        prices.columns = [tickers[0]]

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan)

    # Remove columns with no usable observations.
    prices = prices.dropna(axis=1, how="all")

    # Prices must be strictly positive.
    prices = prices.where(prices > 0)

    if prices.empty:
        raise ValueError("No valid price observations were found.")

    logger.info(
        "Downloaded %d tickers and %d trading dates.",
        prices.shape[1],
        prices.shape[0],
    )

    return prices


# --------------------------------------------------
# 4. DATA VALIDATION
# --------------------------------------------------

def validate_price_data(prices: pd.DataFrame) -> None:
    """Run basic structural and numerical checks."""

    if prices.empty:
        raise ValueError("Price dataframe is empty.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Dates are not sorted.")

    if prices.index.has_duplicates:
        raise ValueError("Duplicate dates detected.")

    if (prices.stack() <= 0).any():
        raise ValueError("Non-positive prices detected.")

    logger.info("Price data validation completed.")

    print("\nDATASET SUMMARY")
    print("------------------------------")
    print(f"Start date: {prices.index.min().date()}")
    print(f"End date:   {prices.index.max().date()}")
    print(f"Tickers:    {prices.shape[1]}")
    print(f"Rows:       {prices.shape[0]}")

    print("\nMissing observations by ticker:")
    print(prices.isna().sum().sort_values(ascending=False))


# --------------------------------------------------
# 5. EXECUTION
# --------------------------------------------------


# --------------------------------------------------
# 6. PAIR DISCOVERY AND COINTEGRATION
# --------------------------------------------------

from itertools import combinations

from statsmodels.tsa.stattools import coint
from statsmodels.stats.multitest import multipletests


def screen_cointegrated_pairs(
    prices: pd.DataFrame,
    correlation_threshold: float = 0.50,
    min_observations: int = CONFIG.min_observations,
) -> pd.DataFrame:
    """
    Screen pairs using return correlation, then test
    log-price cointegration with the Engle-Granger test.

    IMPORTANT:
    Call this function using training data only.
    Do not use future observations to select pairs.
    """

    log_prices = np.log(prices.where(prices > 0))
    returns = log_prices.diff()

    results = []

    for ticker_1, ticker_2 in combinations(prices.columns, 2):

        pair_returns = returns[
            [ticker_1, ticker_2]
        ].dropna()

        if len(pair_returns) < min_observations:
            continue

        correlation = pair_returns.corr().iloc[0, 1]

        if not np.isfinite(correlation):
            continue

        if abs(correlation) < correlation_threshold:
            continue

        pair_log_prices = log_prices[
            [ticker_1, ticker_2]
        ].dropna()

        if len(pair_log_prices) < min_observations:
            continue

        statistic, p_value, critical_values = coint(
            pair_log_prices[ticker_1],
            pair_log_prices[ticker_2],
            trend="c",
            autolag="aic",
        )

        results.append({
            "ticker_1": ticker_1,
            "ticker_2": ticker_2,
            "return_correlation": correlation,
            "cointegration_statistic": statistic,
            "p_value": p_value,
            "observations": len(pair_log_prices),
        })

    if not results:
        logger.warning("No candidate pairs passed the screening.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Correct for multiple hypothesis testing.
    reject, adjusted_p_values, _, _ = multipletests(
        results_df["p_value"],
        alpha=0.05,
        method="fdr_bh",
    )

    results_df["adjusted_p_value"] = adjusted_p_values
    results_df["significant"] = reject

    results_df = results_df.sort_values(
        "adjusted_p_value"
    ).reset_index(drop=True)

    logger.info(
        "Tested %d pairs; %d passed FDR correction.",
        len(results_df),
        int(reject.sum()),
    )

    return results_df

# --------------------------------------------------
# 7. TRAIN / TEST SPLIT
# --------------------------------------------------

def run_research_pipeline() -> None:
    """Run initial pair discovery using training data only."""

    price_data = download_price_data()
    validate_price_data(price_data)

    split_index = int(len(price_data) * 0.70)

    training_prices = price_data.iloc[:split_index].copy()
    testing_prices = price_data.iloc[split_index:].copy()

    logger.info(
        "Training period: %s to %s",
        training_prices.index.min().date(),
        training_prices.index.max().date(),
    )

    logger.info(
        "Testing period: %s to %s",
        testing_prices.index.min().date(),
        testing_prices.index.max().date(),
    )

    # Pair selection uses training observations only.
    candidate_pairs = screen_cointegrated_pairs(
        training_prices
    )

    print("\nTRAINING PERIOD — SELECTED PAIRS")
    print("--------------------------------")

    if candidate_pairs.empty:
        print("No statistically significant pairs found.")
    else:
        print(
            candidate_pairs[
                [
                    "ticker_1",
                    "ticker_2",
                    "return_correlation",
                    "p_value",
                    "adjusted_p_value",
                    "significant",
                ]
            ].head(20).to_string(index=False)
        )

    print("\nOut-of-sample observations reserved:")
    print(len(testing_prices))


if __name__ == "__main__":
    run_research_pipeline()


