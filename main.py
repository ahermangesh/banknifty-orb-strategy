"""
main.py
-------
Entry point for the BankNifty intraday mean-reversion strategy.

Pipeline
--------
  1. Load & clean data          (data_loader.py)
  2. Discover statistical relationship & select parameters  (strategy.py)
  3. Generate trading signals   (strategy.py)
  4. Run backtest               (backtester.py)
  5. Compute metrics & plots    (analysis.py)

Usage
-----
    python main.py
    python main.py --data path/to/data.csv --capital 500000
"""

import argparse
import time

import data_loader
import strategy
import backtester
import analysis


DATA_PATH       = "data/banknifty_candlestick_data.csv"
INITIAL_CAPITAL = 1_000_000.0


def main(data_path: str = DATA_PATH, capital: float = INITIAL_CAPITAL) -> None:
    t0 = time.perf_counter()

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("=" * 55)
    print("  BankNifty Opening Range Breakout Strategy")
    print("=" * 55)
    df = data_loader.load_data(data_path)

    # ── 2. Discover relationship & pick best EMA pair ─────────────────────────
    params = strategy.discover_relationship(df)

    # ── 3. Generate signals ───────────────────────────────────────────────────
    print("\n=== Generating Signals ===")
    signals = strategy.generate_signals(df, params)
    n_long  = (signals["raw_signal"] ==  1).sum()
    n_short = (signals["raw_signal"] == -1).sum()
    print(f"  Long  (OR breakout up)  : {n_long:,}")
    print(f"  Short (OR breakout dn)  : {n_short:,}")

    # ── 4. Backtest ───────────────────────────────────────────────────────────
    result = backtester.run_backtest(signals, params, initial_capital=capital)

    # ── 5. Metrics & visualisations ───────────────────────────────────────────
    metrics = analysis.compute_metrics(
        result.trades, result.portfolio, initial_capital=capital
    )
    analysis.save_metrics(metrics)
    analysis.plot_results(
        result.signals, result.trades, result.portfolio, params
    )

    elapsed = time.perf_counter() - t0
    print(f"\n  Total runtime  : {elapsed:.1f}s")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BankNifty Opening Range Breakout backtest"
    )
    parser.add_argument(
        "--data",
        default=DATA_PATH,
        help=f"Path to CSV dataset (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=INITIAL_CAPITAL,
        help=f"Initial capital in INR (default: {INITIAL_CAPITAL:,.0f})",
    )
    args = parser.parse_args()
    main(data_path=args.data, capital=args.capital)
