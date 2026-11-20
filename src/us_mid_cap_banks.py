
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

if __name__ == "__main__":
    price_data = download_price_data()
    validate_price_data(price_data)
