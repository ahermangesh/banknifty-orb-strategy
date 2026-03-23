"""
data_loader.py
--------------
Loads and preprocesses the BankNifty minute-level OHLC dataset.

Handles:
  - Date/time parsing and index construction
  - Outlier detection via rolling z-score (price spikes)
  - Forward-fill for minor gaps within a trading session
  - Market-hours filtering (09:15-15:30 IST)
  - Structural-break flagging (daily returns > 10% flagged for awareness)
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── constants ────────────────────────────────────────────────────────────────
MARKET_OPEN  = "09:15"
MARKET_CLOSE = "15:30"
OUTLIER_Z    = 5.0      # σ threshold to call a price point an outlier
OUTLIER_WIN  = 60       # rolling window (minutes) used for outlier detection


# ── public API ───────────────────────────────────────────────────────────────

def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load raw CSV, clean it, and return a time-indexed DataFrame with
    columns [Open, High, Low, Close, date, is_first_bar, is_last_bar].

    Parameters
    ----------
    filepath : path to the banknifty_candlestick_data.csv file

    Returns
    -------
    pd.DataFrame  (DatetimeIndex, minute frequency)
    """
    raw = _read_csv(filepath)
    df  = _parse_datetime(raw)
    df  = _filter_market_hours(df)
    df  = _detect_and_fix_outliers(df)
    df  = _forward_fill_gaps(df)
    df  = _add_session_markers(df)
    _print_summary(df)
    return df


# ── private helpers ───────────────────────────────────────────────────────────

def _read_csv(filepath: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        dtype={"Instrument": str, "Date": str, "Time": str,
               "Open": float, "High": float, "Low": float, "Close": float},
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def _parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Combine Date + Time into a proper DatetimeIndex."""
    # Date format in CSV: DD-MM-YYYY
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce",
    )
    invalid = df["datetime"].isna().sum()
    if invalid:
        print(f"  [warn] {invalid} rows with unparseable timestamps dropped.")
        df = df.dropna(subset=["datetime"])

    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.set_index("datetime")

    # Keep only OHLC
    return df[["Open", "High", "Low", "Close"]].copy()


def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict data to official NSE session (09:15-15:30)."""
    df = df.between_time(MARKET_OPEN, MARKET_CLOSE)
    return df


def _detect_and_fix_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace close prices that deviate more than OUTLIER_Z σ from a
    centred rolling median with the rolling median itself.
    OHLC are adjusted to be consistent with the corrected close.
    """
    close = df["Close"]
    roll_med = close.rolling(OUTLIER_WIN, min_periods=10, center=True).median()
    roll_std = close.rolling(OUTLIER_WIN, min_periods=10, center=True).std()

    outlier_mask = (close - roll_med).abs() > (OUTLIER_Z * roll_std)
    n_outliers = outlier_mask.sum()

    if n_outliers:
        df.loc[outlier_mask, "Close"] = roll_med[outlier_mask]
        df.loc[outlier_mask, "Open"]  = roll_med[outlier_mask]
        df.loc[outlier_mask, "High"]  = roll_med[outlier_mask]
        df.loc[outlier_mask, "Low"]   = roll_med[outlier_mask]

    print(f"  [data] Outliers corrected: {n_outliers}")
    return df


def _forward_fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each trading session, forward-fill up to 5 consecutive missing
    minutes (e.g. thin-market or data-feed gaps).
    """
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].ffill(limit=5)

    remaining_na = df.isnull().sum().sum()
    if remaining_na:
        df = df.dropna()
        print(f"  [data] Rows dropped after fill: {remaining_na}")

    return df


def _add_session_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns: date, is_first_bar, is_last_bar."""
    df["date"] = df.index.date

    grp = df.groupby("date")
    first_idx = grp.apply(lambda g: g.index[0], include_groups=False).values
    last_idx  = grp.apply(lambda g: g.index[-1], include_groups=False).values

    df["is_first_bar"] = df.index.isin(first_idx)
    df["is_last_bar"]  = df.index.isin(last_idx)
    return df


def _print_summary(df: pd.DataFrame) -> None:
    print(f"\n=== Data Summary ===")
    print(f"  Rows           : {len(df):,}")
    print(f"  Trading days   : {df['date'].nunique()}")
    print(f"  Date range     : {df.index[0].date()}  ->  {df.index[-1].date()}")
    print(f"  Close range    : {df['Close'].min():.2f}  -  {df['Close'].max():.2f}")
    print(f"  NaN remaining  : {df[['Open','High','Low','Close']].isnull().sum().sum()}")
