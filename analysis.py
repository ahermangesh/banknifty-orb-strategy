"""
analysis.py
-----------
Performance metric computation and visualisation for the backtest results.

Metrics computed
----------------
  Total Return         - overall P&L as % of initial capital
  Annualized Return    - CAGR over the backtest period
  Sharpe Ratio         - annualised (252 trading-day basis)
  Maximum Drawdown     - worst peak-to-trough decline in portfolio value
  Win Rate             - fraction of trades with positive P&L
  Avg Trade Duration   - mean minutes per trade
  Profit Factor        - gross wins / gross losses

Plots generated (saved to results/)
------------------------------------
  1. equity_curve.png        - cumulative portfolio value
  2. drawdown.png            - rolling drawdown from peak
  3. price_signals.png       - close price + long/short entry markers (sample)
  4. zscore_signals.png      - z-score with threshold lines (sample)
  5. trade_pnl_dist.png      - histogram of per-trade P&L
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


RESULTS_DIR = "results"


# ── public API ───────────────────────────────────────────────────────────────

def compute_metrics(
    trades         : pd.DataFrame,
    portfolio      : pd.Series,
    initial_capital: float = 1_000_000.0,
) -> dict:
    """
    Compute and print all required performance metrics.

    Returns a dict of {metric_name: formatted_string}.
    """
    if len(trades) == 0:
        print("No trades executed - check signal thresholds.")
        return {}

    total_return     = (portfolio.iloc[-1] - initial_capital) / initial_capital

    years            = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
    ann_return       = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    # Daily portfolio values -> daily returns for Sharpe
    daily            = portfolio.resample("B").last().dropna()
    daily_ret        = daily.pct_change().dropna()
    sharpe           = (
        daily_ret.mean() / daily_ret.std() * np.sqrt(252)
        if daily_ret.std() > 0 else 0.0
    )

    rolling_peak     = portfolio.cummax()
    drawdown_series  = (portfolio - rolling_peak) / rolling_peak
    max_drawdown     = drawdown_series.min()

    wins             = trades["pnl"] > 0
    win_rate         = wins.mean()

    avg_duration     = trades["duration_min"].mean()

    gross_win  = trades.loc[wins, "pnl"].sum()
    gross_loss = trades.loc[~wins, "pnl"].sum()
    profit_factor = (
        gross_win / abs(gross_loss)
        if gross_loss != 0 else float("inf")
    )

    metrics = {
        "Total Return"           : f"{total_return:.2%}",
        "Annualized Return"      : f"{ann_return:.2%}",
        "Sharpe Ratio"           : f"{sharpe:.3f}",
        "Max Drawdown"           : f"{max_drawdown:.2%}",
        "Win Rate"               : f"{win_rate:.2%}",
        "Total Trades"           : str(len(trades)),
        "Avg Trade Duration (min)": f"{avg_duration:.1f}",
        "Profit Factor"          : f"{profit_factor:.3f}",
    }

    print("\n=== Performance Metrics ===")
    width = max(len(k) for k in metrics)
    for k, v in metrics.items():
        print(f"  {k:<{width}} : {v}")

    return metrics


def save_metrics(metrics: dict, path: str = "results/metrics.txt") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"\n  Metrics saved -> {path}")


def plot_results(
    signals   : pd.DataFrame,
    trades    : pd.DataFrame,
    portfolio : pd.Series,
    params    : dict,
    output_dir: str = RESULTS_DIR,
) -> None:
    """
    Generate and save all visualisation plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    _plot_equity_curve(portfolio, output_dir)
    _plot_drawdown(portfolio, output_dir)
    _plot_price_signals(signals, trades, output_dir)
    _plot_or_bands(signals, params, output_dir)
    if len(trades):
        _plot_pnl_distribution(trades, output_dir)

    print(f"\n  Plots saved -> {output_dir}/")


# ── private plot helpers ──────────────────────────────────────────────────────

def _plot_equity_curve(portfolio: pd.Series, out: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(portfolio.index, portfolio.values, color="#2196F3", lw=1.2)
    ax.set_title("Portfolio Equity Curve", fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value (INR)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"INR {x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/equity_curve.png", dpi=150)
    plt.close(fig)


def _plot_drawdown(portfolio: pd.Series, out: str) -> None:
    rolling_peak = portfolio.cummax()
    drawdown     = (portfolio - rolling_peak) / rolling_peak * 100

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0,
                    color="#F44336", alpha=0.6, label="Drawdown")
    ax.set_title("Portfolio Drawdown", fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/drawdown.png", dpi=150)
    plt.close(fig)


def _plot_price_signals(
    signals: pd.DataFrame,
    trades : pd.DataFrame,
    out    : str,
    sample_days: int = 20,
) -> None:
    """Plot close price with long/short trade entry markers for a sample window."""
    cutoff = signals.index[-1] - pd.Timedelta(days=sample_days)
    sample = signals[signals.index >= cutoff]
    if sample.empty:
        sample = signals.iloc[-3_000:]

    if len(trades) == 0:
        return

    t_sample = trades[
        (trades["entry_time"] >= sample.index[0]) &
        (trades["entry_time"] <= sample.index[-1])
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(sample.index, sample["Close"], color="#424242", lw=0.8, label="Close")

    longs  = t_sample[t_sample["direction"] == "long"]
    shorts = t_sample[t_sample["direction"] == "short"]

    ax.scatter(longs["entry_time"],
               signals.loc[signals.index.isin(longs["entry_time"]), "Close"],
               marker="^", color="#4CAF50", s=60, zorder=5, label="Long entry")
    ax.scatter(shorts["entry_time"],
               signals.loc[signals.index.isin(shorts["entry_time"]), "Close"],
               marker="v", color="#F44336", s=60, zorder=5, label="Short entry")

    ax.set_title(f"Price & Trade Signals (last {sample_days} trading days)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("BankNifty Close")
    ax.set_xlabel("Datetime")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.xticks(rotation=30)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/price_signals.png", dpi=150)
    plt.close(fig)


def _plot_or_bands(
    signals: pd.DataFrame,
    params : dict,
    out    : str,
    sample_days: int = 10,
) -> None:
    """Plot close price with OR high/low bands for a sample window."""
    if "or_high" not in signals.columns:
        return
    cutoff = signals.index[-1] - pd.Timedelta(days=sample_days)
    sample = signals[signals.index >= cutoff]
    if sample.empty:
        sample = signals.iloc[-2_000:]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sample.index, sample["Close"],   color="#424242", lw=0.8, label="Close")
    ax.plot(sample.index, sample["or_high"], color="#F44336", lw=0.8,
            ls="--", alpha=0.7, label="OR High")
    ax.plot(sample.index, sample["or_low"],  color="#4CAF50", lw=0.8,
            ls="--", alpha=0.7, label="OR Low")

    ax.set_title(f"Opening Range Bands (last {sample_days} days)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("BankNifty Close")
    ax.set_xlabel("Datetime")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.xticks(rotation=30)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/or_bands.png", dpi=150)
    plt.close(fig)


def _plot_pnl_distribution(trades: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(trades["pnl"], bins=60, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="#F44336", lw=1.5, ls="--", label="Breakeven")
    ax.axvline(trades["pnl"].mean(), color="#4CAF50", lw=1.5, ls="-",
               label=f"Mean P&L: INR {trades['pnl'].mean():,.1f}")
    ax.set_title("Per-Trade P&L Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trade P&L (INR)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/trade_pnl_dist.png", dpi=150)
    plt.close(fig)
