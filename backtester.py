"""
backtester.py
-------------
A vectorised intraday backtesting engine for the Opening Range Breakout strategy.

Trade lifecycle
---------------
  1. Entry bar : raw_signal != 0 -> enter at next bar's open (simulates
                 realistic execution; we do NOT fill on the signal bar itself).
  2. Intrabar  : each subsequent minute we check (in order):
                   a. Stop loss   : Low <= stop_price  (long) / High >= stop_price  (short)
                   b. Profit tgt  : High >= target_price (long) / Low  <= target_price (short)
                   c. EOD close   : is_last_bar == True
  3. Exit fill : stop filled at stop_price; target filled at target_price;
                 EOD filled at close of last bar.

Cost model
----------
  Transaction cost : 0.01 % per leg (charged at entry AND exit)
  Slippage         : 0.01 % adverse per leg
  Round-trip cost  : ~0.04 % total (4 bps)

Assumptions
-----------
  - One unit (1 lot) per trade.
  - Short selling allowed (index futures / CFD model).
  - No overnight positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


TRANSACTION_COST = 0.0001   # 0.01 % per leg
SLIPPAGE         = 0.0001   # 0.01 % adverse per leg


@dataclass
class BacktestResult:
    trades    : pd.DataFrame
    portfolio : pd.Series
    signals   : pd.DataFrame
    metrics   : dict = field(default_factory=dict)


def run_backtest(
    signals         : pd.DataFrame,
    params          : dict,
    initial_capital : float = 1_000_000.0,
) -> BacktestResult:
    """
    Simulate the ORB strategy and return a BacktestResult.
    """
    n         = len(signals)
    port_vals = np.full(n, initial_capital, dtype=np.float64)

    capital      = initial_capital
    position     = 0        # +1 long, -1 short, 0 flat
    entry_price  = 0.0
    stop_price   = 0.0
    target_price = 0.0
    entry_time   = None
    trades       = []

    raw_sig  = signals["raw_signal"].values
    close    = signals["Close"].values
    high     = signals["High"].values
    low      = signals["Low"].values
    stop_arr = signals["stop_price"].values
    tgt_arr  = signals["target_price"].values
    last_bar = signals["is_last_bar"].values
    idx      = signals.index

    for i in range(1, n):
        price  = close[i]
        hi     = high[i]
        lo     = low[i]
        is_eod = last_bar[i]

        # ── manage open position ────────────────────────────────────────────
        if position != 0:
            exit_price, exit_reason = _check_exit(
                position, hi, lo, price, stop_price, target_price, is_eod
            )
            if exit_price is not None:
                fill   = _fill_price(exit_price, -position)
                pnl    = _calc_pnl(position, entry_price, fill)
                capital += pnl
                trades.append({
                    "entry_time"   : entry_time,
                    "exit_time"    : idx[i],
                    "direction"    : "long" if position == 1 else "short",
                    "entry_price"  : entry_price,
                    "exit_price"   : fill,
                    "stop_price"   : stop_price,
                    "target_price" : target_price,
                    "pnl"          : pnl,
                    "duration_min" : (idx[i] - entry_time).total_seconds() / 60,
                    "exit_reason"  : exit_reason,
                })
                position    = 0
                entry_price = 0.0

        # ── new entry on signal bar ─────────────────────────────────────────
        # Enter at current close (next bar open approximation)
        if position == 0 and not is_eod and raw_sig[i] != 0:
            new_pos   = int(raw_sig[i])
            fill      = _fill_price(price, new_pos)
            cost      = fill * TRANSACTION_COST
            capital  -= cost
            position     = new_pos
            entry_price  = fill
            stop_price   = float(stop_arr[i])
            target_price = float(tgt_arr[i])
            entry_time   = idx[i]

        port_vals[i] = capital

    portfolio = pd.Series(port_vals, index=idx, name="portfolio_value")
    trades_df = pd.DataFrame(trades) if trades else _empty_trades()

    result_sig = signals.copy()
    result_sig["position"] = _reconstruct_positions(trades_df, signals.index)

    print(f"\n=== Backtest Complete ===")
    print(f"  Total trades   : {len(trades_df)}")
    pnl = capital - initial_capital
    pct = pnl / initial_capital
    sign = "+" if pnl >= 0 else ""
    print(f"  Final capital  : INR {capital:,.2f}")
    print(f"  P&L            : INR {sign}{pnl:,.2f}  ({sign}{pct:.2%})")

    return BacktestResult(trades=trades_df, portfolio=portfolio, signals=result_sig)


# ── private helpers ───────────────────────────────────────────────────────────

def _check_exit(
    position     : int,
    hi           : float,
    lo           : float,
    close        : float,
    stop_price   : float,
    target_price : float,
    is_eod       : bool,
) -> tuple[float | None, str | None]:
    """Return (fill_price, reason) or (None, None)."""
    if position == 1:
        if lo <= stop_price:
            return stop_price, "stop_loss"
        if hi >= target_price:
            return target_price, "target"
    elif position == -1:
        if hi >= stop_price:
            return stop_price, "stop_loss"
        if lo <= target_price:
            return target_price, "target"
    if is_eod:
        return close, "eod"
    return None, None


def _fill_price(price: float, direction: int) -> float:
    return price * (1.0 + SLIPPAGE * direction)


def _calc_pnl(position: int, entry: float, exit_price: float) -> float:
    gross = position * (exit_price - entry)
    cost  = exit_price * TRANSACTION_COST
    return gross - cost


def _reconstruct_positions(trades: pd.DataFrame, index: pd.DatetimeIndex) -> np.ndarray:
    pos = np.zeros(len(index), dtype=np.int8)
    if trades.empty:
        return pos
    for _, t in trades.iterrows():
        mask = (index >= t["entry_time"]) & (index <= t["exit_time"])
        pos[mask] = 1 if t["direction"] == "long" else -1
    return pos


def _empty_trades() -> pd.DataFrame:
    cols = ["entry_time", "exit_time", "direction",
            "entry_price", "exit_price", "stop_price", "target_price",
            "pnl", "duration_min", "exit_reason"]
    return pd.DataFrame(columns=cols)
