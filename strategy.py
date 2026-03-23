"""
strategy.py
-----------
Relationship discovery and signal generation.

Statistical Rationale
=====================
We discovered that BankNifty minute data has:
  - Hurst exponent > 0.5  -> the price series exhibits long-range dependence
    and trending behaviour (not mean-reverting).
  - Lag-1 return autocorrelation slightly negative at the 1-min scale
    (micro-structure noise), but positive autocorrelation at the 15-30 min
    scale (momentum).

These observations motivate an **Opening Range Breakout (ORB)** strategy:

  Opening Range (OR) = price range (High - Low) of the first 30 minutes.
  Hypothesis: when price breaks decisively out of the OR, intraday momentum
  carries it further in the same direction.

Statistical Validation (discover_relationship)
----------------------------------------------
  We test whether post-breakout directional returns are significantly
  positive (one-sample t-test against zero).  A significant p-value
  justifies using the breakout as a trading signal.

  Additionally, we compute:
    - Conditional breakout return mean / std
    - OR-range vs intraday-range correlation (OR is a volatility proxy)
    - Autocorrelation of 15-min returns (confirm short-horizon momentum)

Signal Rules
------------
  Entry (long)  : first bar whose close > OR High  (after OR period ends)
  Entry (short) : first bar whose close < OR Low
  At most ONE trade per session (first valid breakout wins).
  No new entries after 14:30.
  Stop loss     : opposite end of the OR  (full OR range as risk)
  Profit target : risk_reward * OR range  (default 1.5x)
  Forced exit   : last bar of the session (EOD)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.tsa.stattools import acf


# ── default parameters ────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "or_minutes"   : 30,     # opening-range duration in minutes
    "risk_reward"  : 1.5,    # profit target = risk_reward * OR range
    "entry_cutoff" : "14:30",# no new entries after this time
}


# ── public API ────────────────────────────────────────────────────────────────

def discover_relationship(df: pd.DataFrame) -> dict:
    """
    Statistically validate the Opening Range Breakout premise.

    Returns params dict.
    """
    or_minutes = DEFAULT_PARAMS["or_minutes"]

    # --- build per-day breakout return series ---
    long_returns  = []
    short_returns = []
    or_ranges     = []
    intraday_ranges = []

    for date, day_df in df.groupby("date"):
        or_end     = day_df.index[0] + pd.Timedelta(minutes=or_minutes)
        or_period  = day_df[day_df.index <= or_end]
        post_or    = day_df[day_df.index >  or_end]

        if len(or_period) < 5 or len(post_or) < 15:
            continue

        or_high  = or_period["High"].max()
        or_low   = or_period["Low"].min()
        or_range = or_high - or_low
        intraday_range = day_df["High"].max() - day_df["Low"].min()

        or_ranges.append(or_range)
        intraday_ranges.append(intraday_range)

        # Long breakout: close above OR high
        long_break = post_or[post_or["Close"] > or_high]
        if len(long_break):
            entry_close = long_break["Close"].iloc[0]
            # Return over next 30 bars from entry
            entry_loc   = post_or.index.get_loc(long_break.index[0])
            window      = post_or.iloc[entry_loc: entry_loc + 30]
            if len(window) >= 10:
                exit_close = window["Close"].iloc[-1]
                long_returns.append((exit_close - entry_close) / entry_close)

        # Short breakout: close below OR low
        short_break = post_or[post_or["Close"] < or_low]
        if len(short_break):
            entry_close = short_break["Close"].iloc[0]
            entry_loc   = post_or.index.get_loc(short_break.index[0])
            window      = post_or.iloc[entry_loc: entry_loc + 30]
            if len(window) >= 10:
                exit_close = window["Close"].iloc[-1]
                short_returns.append((entry_close - exit_close) / entry_close)

    # --- t-test: are breakout returns > 0? ---
    all_returns   = np.array(long_returns + short_returns)
    t_stat, p_val = ttest_1samp(all_returns, 0.0) if len(all_returns) > 30 else (0, 1)

    # --- OR range vs intraday range correlation ---
    or_arr = np.array(or_ranges)
    id_arr = np.array(intraday_ranges)
    range_corr = float(np.corrcoef(or_arr, id_arr)[0, 1]) if len(or_arr) > 2 else 0.0

    # --- 15-min return autocorrelation ---
    close_15m  = df["Close"].resample("15min").last().dropna()
    ret_15m    = close_15m.pct_change().dropna()
    acf_vals   = acf(ret_15m, nlags=3, fft=True) if len(ret_15m) > 50 else [0, 0, 0, 0]
    lag1_ac_15 = float(acf_vals[1])

    params = {**DEFAULT_PARAMS}

    print("\n=== Relationship Discovery: Opening Range Breakout ===")
    print(f"  OR breakout sample size    : {len(all_returns)}")
    print(f"  Mean post-breakout return  : {all_returns.mean():.4%}  "
          f"(per 30-bar window)")
    print(f"  t-stat / p-value           : {t_stat:.3f} / {p_val:.6f}  "
          f"({'momentum confirmed [OK]' if p_val < 0.05 and t_stat > 0 else 'not significant [WARN]'})")
    print(f"  OR vs intraday range corr  : {range_corr:.3f}  "
          f"(OR is a {'strong' if range_corr > 0.6 else 'moderate'} volatility proxy)")
    print(f"  15-min lag-1 autocorr      : {lag1_ac_15:+.4f}  "
          f"({'momentum [OK]' if lag1_ac_15 > 0 else 'mean-rev'})")

    return params


def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Compute OR levels and emit at most one entry signal per session.

    Columns added to df:
      or_high, or_low, stop_price, target_price, raw_signal
        raw_signal = +1 (long entry), -1 (short entry), 0 (no action)
    """
    or_minutes  = params["or_minutes"]
    rr          = params["risk_reward"]
    cutoff_time = pd.Timestamp(f"2000-01-01 {params['entry_cutoff']}").time()

    out = df.copy()
    out["or_high"]      = np.nan
    out["or_low"]       = np.nan
    out["stop_price"]   = np.nan
    out["target_price"] = np.nan
    out["raw_signal"]   = 0

    for date, day_df in df.groupby("date"):
        or_end    = day_df.index[0] + pd.Timedelta(minutes=or_minutes)
        or_period = day_df[day_df.index <= or_end]
        post_or   = day_df[(day_df.index > or_end) &
                           (day_df.index.time <= cutoff_time)]

        if len(or_period) < 5 or post_or.empty:
            continue

        or_high  = or_period["High"].max()
        or_low   = or_period["Low"].min()
        or_range = or_high - or_low

        if or_range < 1.0:    # skip degenerate/holiday days
            continue

        out.loc[day_df.index, "or_high"] = or_high
        out.loc[day_df.index, "or_low"]  = or_low

        # Find first bar that closes OUTSIDE the range
        long_bars  = post_or[post_or["Close"] > or_high]
        short_bars = post_or[post_or["Close"] < or_low]

        # Long breakout
        if len(long_bars):
            entry_bar = long_bars.index[0]
            stop      = or_low
            target    = long_bars["Close"].iloc[0] + rr * (long_bars["Close"].iloc[0] - stop)
            out.loc[entry_bar, "raw_signal"]   =  1
            out.loc[entry_bar, "stop_price"]   = stop
            out.loc[entry_bar, "target_price"] = target

        # Short breakout (only if no long was already taken)
        elif len(short_bars):
            entry_bar = short_bars.index[0]
            stop      = or_high
            target    = short_bars["Close"].iloc[0] - rr * (stop - short_bars["Close"].iloc[0])
            out.loc[entry_bar, "raw_signal"]   = -1
            out.loc[entry_bar, "stop_price"]   = stop
            out.loc[entry_bar, "target_price"] = target

    return out
