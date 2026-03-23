"""
bonus.py
--------
Three bonus extensions to the base ORB strategy.

1. Out-of-Sample Testing
   ---------------------
   Split the full dataset into an in-sample (IS) training period and an
   out-of-sample (OOS) test period that the strategy never "saw".
   Compare IS vs OOS metrics to check for overfitting.

   Split: 2015-2021 (IS)  |  2022-2024 (OOS)

2. Walk-Forward Parameter Optimisation
   ------------------------------------
   Instead of picking a fixed set of parameters and backtesting on the
   full history (which would be look-ahead biased), we roll a training
   window forward year by year:

     Train on year Y-2 to Y-1  -->  pick best Sharpe params
     Apply those params to year Y  (never seen before)
     Advance by 1 year, repeat

   This gives an honest, realistic estimate of live performance.

3. Volatility-Scaled Position Sizing
   ------------------------------------
   Replace fixed "1 unit per trade" with a position size that targets a
   constant INR risk per trade:

     ATR_20  =  20-bar Average True Range  (bar-level volatility)
     position_size = (capital * risk_pct) / (stop_distance * ATR_scale)

   Larger size when the stop is tight / volatility is low;
   smaller size when the stop is wide / volatility is high.
   Naturally reduces risk in choppy markets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import strategy
import backtester
import analysis


# ── 1. Out-of-Sample Test ─────────────────────────────────────────────────────

IS_END  = "2021-12-31"
OOS_START = "2022-01-01"


def run_oos_test(
    df             : pd.DataFrame,
    params         : dict,
    initial_capital: float = 1_000_000.0,
    output_dir     : str   = "results",
) -> dict:
    """
    Run backtest separately on IS and OOS periods and compare.
    Returns dict with both sets of metrics.
    """
    df_is  = df[df.index <= IS_END]
    df_oos = df[df.index >= OOS_START]

    print("\n=== Bonus 1: Out-of-Sample Test ===")
    print(f"  In-sample  : {df_is.index[0].date()} to {df_is.index[-1].date()}  "
          f"({df_is['date'].nunique()} days)")
    print(f"  Out-of-sample: {df_oos.index[0].date()} to {df_oos.index[-1].date()}  "
          f"({df_oos['date'].nunique()} days)")

    results = {}
    for label, subset in [("in_sample", df_is), ("out_of_sample", df_oos)]:
        sigs    = strategy.generate_signals(subset, params)
        result  = backtester.run_backtest(sigs, params, initial_capital=initial_capital)
        metrics = analysis.compute_metrics(
            result.trades, result.portfolio, initial_capital=initial_capital
        )
        results[label] = {"metrics": metrics, "result": result}
        print(f"\n  [{label.upper().replace('_', '-')}]")
        for k, v in metrics.items():
            print(f"    {k:<30} : {v}")

    _plot_oos_comparison(
        results["in_sample"]["result"].portfolio,
        results["out_of_sample"]["result"].portfolio,
        initial_capital,
        output_dir,
    )
    return results


# ── 2. Walk-Forward Optimisation ──────────────────────────────────────────────

PARAM_GRID = {
    "or_minutes"  : [20, 25, 30, 45],
    "risk_reward" : [1.0, 1.5, 2.0, 2.5],
}

TRAIN_YEARS = 2    # years of history used to optimise params
TEST_YEARS  = 1    # years of forward test per fold


def walk_forward_optimise(
    df             : pd.DataFrame,
    base_params    : dict,
    initial_capital: float = 1_000_000.0,
    output_dir     : str   = "results",
) -> pd.DataFrame:
    """
    Walk-forward optimisation loop.

    Returns a DataFrame with per-fold best params and OOS performance.
    """
    print("\n=== Bonus 2: Walk-Forward Optimisation ===")

    years     = sorted(df["date"].astype("datetime64[ns]").dt.year.unique())
    folds     = []

    for test_year_idx in range(TRAIN_YEARS, len(years) - TEST_YEARS + 1):
        train_years_range = years[test_year_idx - TRAIN_YEARS : test_year_idx]
        test_years_range  = years[test_year_idx : test_year_idx + TEST_YEARS]

        train_mask = df.index.year.isin(train_years_range)
        test_mask  = df.index.year.isin(test_years_range)

        df_train = df[train_mask]
        df_test  = df[test_mask]

        if len(df_train) < 1000 or len(df_test) < 100:
            continue

        # --- grid search on train set ---
        best_sharpe = -np.inf
        best_params = base_params.copy()

        keys   = list(PARAM_GRID.keys())
        values = list(PARAM_GRID.values())

        for combo in itertools.product(*values):
            trial_params = {**base_params, **dict(zip(keys, combo))}
            try:
                sigs   = strategy.generate_signals(df_train, trial_params)
                res    = backtester.run_backtest(sigs, trial_params,
                                                 initial_capital=initial_capital)
                if len(res.trades) < 10:
                    continue
                m = analysis.compute_metrics(
                    res.trades, res.portfolio,
                    initial_capital=initial_capital
                )
                sharpe = float(m.get("Sharpe Ratio", -99))
            except Exception:
                continue

            if sharpe > best_sharpe:
                best_sharpe  = sharpe
                best_params  = trial_params.copy()

        # --- apply best params to test set ---
        sigs_test = strategy.generate_signals(df_test, best_params)
        res_test  = backtester.run_backtest(sigs_test, best_params,
                                             initial_capital=initial_capital)
        m_test = analysis.compute_metrics(
            res_test.trades, res_test.portfolio,
            initial_capital=initial_capital
        )

        fold = {
            "train_years"   : f"{train_years_range[0]}-{train_years_range[-1]}",
            "test_year"     : str(test_years_range[0]),
            "best_or_min"   : best_params["or_minutes"],
            "best_rr"       : best_params["risk_reward"],
            "train_sharpe"  : round(best_sharpe, 3),
            "test_sharpe"   : float(m_test.get("Sharpe Ratio", 0)),
            "test_return"   : m_test.get("Total Return", "N/A"),
            "test_trades"   : int(m_test.get("Total Trades", 0)),
        }
        folds.append(fold)

        print(f"  Fold {test_years_range[0]} | "
              f"best params: OR={best_params['or_minutes']}min "
              f"RR={best_params['risk_reward']} | "
              f"train Sharpe={best_sharpe:.2f} | "
              f"OOS Sharpe={fold['test_sharpe']:.2f} | "
              f"OOS return={fold['test_return']}")

    folds_df = pd.DataFrame(folds)
    if not folds_df.empty:
        avg_oos = folds_df["test_sharpe"].mean()
        print(f"\n  Average OOS Sharpe across {len(folds_df)} folds: {avg_oos:.3f}")
        folds_df.to_csv(f"{output_dir}/wfo_results.csv", index=False)
        print(f"  Saved -> {output_dir}/wfo_results.csv")

    return folds_df


# ── 3. Volatility-Scaled Position Sizing ─────────────────────────────────────

RISK_PCT  = 0.01    # risk 1% of capital per trade
ATR_PERIOD = 20     # bars for ATR calculation


def add_volatility_sizing(
    df             : pd.DataFrame,
    params         : dict,
    initial_capital: float = 1_000_000.0,
    output_dir     : str   = "results",
) -> backtester.BacktestResult:
    """
    Run the ORB strategy with ATR-based dynamic position sizing and
    compare equity curve against the fixed-size version.
    """
    print("\n=== Bonus 3: Volatility-Scaled Position Sizing ===")

    sigs = strategy.generate_signals(df, params)

    # ATR = rolling mean of True Range
    tr     = pd.concat([
        sigs["High"] - sigs["Low"],
        (sigs["High"] - sigs["Close"].shift()).abs(),
        (sigs["Low"]  - sigs["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    sigs["atr"] = tr.rolling(ATR_PERIOD, min_periods=5).mean()

    result = _backtest_with_vol_sizing(sigs, params, initial_capital)

    metrics = analysis.compute_metrics(
        result.trades, result.portfolio, initial_capital=initial_capital
    )
    print("  Vol-sized strategy metrics:")
    for k, v in metrics.items():
        print(f"    {k:<30} : {v}")

    # Save comparison plot
    fixed_sigs   = strategy.generate_signals(df, params)
    fixed_result = backtester.run_backtest(fixed_sigs, params,
                                           initial_capital=initial_capital)
    _plot_sizing_comparison(
        fixed_result.portfolio, result.portfolio, initial_capital, output_dir
    )
    return result


def _backtest_with_vol_sizing(
    signals        : pd.DataFrame,
    params         : dict,
    initial_capital: float,
) -> backtester.BacktestResult:
    """Same ORB logic but position size scales with ATR."""
    n         = len(signals)
    port_vals = np.full(n, initial_capital, dtype=np.float64)

    capital      = initial_capital
    position     = 0
    entry_price  = 0.0
    stop_price   = 0.0
    target_price = 0.0
    lot_size     = 1.0
    entry_time   = None
    trades       = []

    raw_sig  = signals["raw_signal"].values
    close    = signals["Close"].values
    high     = signals["High"].values
    low      = signals["Low"].values
    stop_arr = signals["stop_price"].values
    tgt_arr  = signals["target_price"].values
    atr_arr  = signals["atr"].values
    last_bar = signals["is_last_bar"].values
    idx      = signals.index

    for i in range(1, n):
        price  = close[i]
        hi     = high[i]
        lo     = low[i]
        is_eod = last_bar[i]

        if position != 0:
            exit_price, reason = backtester._check_exit(
                position, hi, lo, price, stop_price, target_price, is_eod
            )
            if exit_price is not None:
                fill  = backtester._fill_price(exit_price, -position)
                pnl   = position * lot_size * (fill - entry_price)
                pnl  -= fill * lot_size * backtester.TRANSACTION_COST
                capital += pnl
                trades.append({
                    "entry_time"  : entry_time,
                    "exit_time"   : idx[i],
                    "direction"   : "long" if position == 1 else "short",
                    "entry_price" : entry_price,
                    "exit_price"  : fill,
                    "lot_size"    : lot_size,
                    "pnl"         : pnl,
                    "duration_min": (idx[i] - entry_time).total_seconds() / 60,
                    "exit_reason" : reason,
                })
                position    = 0
                entry_price = 0.0

        if position == 0 and not is_eod and raw_sig[i] != 0:
            atr = atr_arr[i]
            s   = stop_arr[i]
            if not np.isnan(atr) and not np.isnan(s) and atr > 0:
                stop_dist = abs(price - s)
                # scale: risk 1% of capital / stop_distance
                lot_size  = max(0.1, min(10.0,
                    (capital * RISK_PCT) / (stop_dist if stop_dist > 0 else atr)
                ))
            else:
                lot_size = 1.0

            new_pos      = int(raw_sig[i])
            fill         = backtester._fill_price(price, new_pos)
            cost         = fill * lot_size * backtester.TRANSACTION_COST
            capital     -= cost
            position     = new_pos
            entry_price  = fill
            stop_price   = float(stop_arr[i])
            target_price = float(tgt_arr[i])
            entry_time   = idx[i]

        port_vals[i] = capital

    portfolio = pd.Series(port_vals, index=idx, name="portfolio_value")
    trades_df = pd.DataFrame(trades) if trades else backtester._empty_trades()
    result_sig = signals.copy()

    print(f"  Trades: {len(trades_df)}  |  Final capital: INR {capital:,.2f}")
    return backtester.BacktestResult(
        trades=trades_df, portfolio=portfolio, signals=result_sig
    )


# ── plot helpers ──────────────────────────────────────────────────────────────

def _plot_oos_comparison(
    is_portfolio : pd.Series,
    oos_portfolio: pd.Series,
    initial_capital: float,
    out: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    for ax, port, label, color in [
        (axes[0], is_portfolio,  "In-Sample  (2015-2021)", "#2196F3"),
        (axes[1], oos_portfolio, "Out-of-Sample (2022-2024)", "#4CAF50"),
    ]:
        pct = (port / initial_capital - 1) * 100
        ax.plot(port.index, pct, color=color, lw=1.2)
        ax.set_title(f"Equity Curve - {label}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Return (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="black", lw=0.5, alpha=0.5)

    fig.tight_layout(pad=3)
    fig.savefig(f"{out}/oos_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved -> {out}/oos_comparison.png")


def _plot_sizing_comparison(
    fixed_port  : pd.Series,
    vol_port    : pd.Series,
    initial_capital: float,
    out: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    for port, label, color in [
        (fixed_port, "Fixed 1-lot sizing",            "#9E9E9E"),
        (vol_port,   "Volatility-scaled sizing (1% risk)", "#FF5722"),
    ]:
        pct = (port / initial_capital - 1) * 100
        ax.plot(port.index, pct, label=label, lw=1.2, color=color)

    ax.set_title("Position Sizing Comparison: Fixed vs Volatility-Scaled",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30)
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/sizing_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved -> {out}/sizing_comparison.png")
