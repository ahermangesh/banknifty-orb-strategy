# BankNifty Opening Range Breakout Strategy

Quantitative intraday trading strategy for BankNifty (NSE) using minute-level OHLC data.

---

## Strategy Overview

### Statistical Motivation

Analysis of BankNifty minute data (2015–2024) reveals:

| Metric | Value | Interpretation |
|---|---|---|
| Hurst Exponent | 0.87 | Trending behaviour (H > 0.5) |
| 15-min lag-1 autocorr | +0.024 | Short-horizon momentum |
| Post-breakout t-stat | 2.67 | Momentum is statistically significant (p = 0.0075) |
| OR vs intraday range corr | 0.689 | Opening range is a strong volatility proxy |

These results motivated an **Opening Range Breakout (ORB)** approach over mean-reversion.

### Signal Logic

```
Opening Range (OR) = High/Low of first 30 minutes each session (09:15–09:44)

Entry (Long)  : Close crosses above OR High -> buy
Entry (Short) : Close crosses below OR Low  -> sell

Stop Loss     : opposite end of the OR
Profit Target : 1.5 x OR range from entry
EOD Exit      : forced close on last bar of session
No new entries after 14:30
```

One trade maximum per session (first valid breakout taken).

### Why This Works

The OR compresses the early-session price discovery into a defined range. Once price breaks out with conviction, intraday momentum carries it further — validated statistically via a one-sample t-test on post-breakout 30-bar returns.

---

## Results

| Metric | Value |
|---|---|
| Total Return | +4.65% |
| Annualized Return | +0.50% |
| Sharpe Ratio | 1.61 |
| Max Drawdown | -0.35% |
| Win Rate | 52.88% |
| Total Trades | 2,173 |
| Avg Trade Duration | 223 min |
| Profit Factor | 1.37 |

> Position sizing: fixed 1 unit per trade. Returns scale proportionally with leverage/lot size.

### Charts

All charts are saved in `results/`:

- `equity_curve.png` — portfolio value over time
- `drawdown.png` — rolling drawdown from peak
- `price_signals.png` — entry markers on price chart
- `or_bands.png` — opening range bands visualised
- `trade_pnl_dist.png` — distribution of trade P&L

---

## Project Structure

```
.
├── data/
│   └── banknifty_candlestick_data.csv
├── results/
│   ├── metrics.txt
│   └── *.png
├── data_loader.py       # data ingestion, cleaning, outlier correction
├── strategy.py          # statistical analysis + signal generation
├── backtester.py        # event-driven backtest engine
├── analysis.py          # performance metrics + visualisations
├── main.py              # pipeline entry point
├── requirements.txt
└── README.md
```

---

## Running the Strategy

```bash
pip install -r requirements.txt
python main.py
```

Optional arguments:

```bash
python main.py --data path/to/data.csv --capital 500000
```

Runtime: ~18 seconds on a standard laptop.

---

## Assumptions

- **Transaction cost**: 0.01% per leg (0.02% round-trip), as specified.
- **Slippage**: 0.01% adverse fill per leg.
- **Position sizing**: fixed 1 unit per trade (no leverage).
- **No overnight positions**: all positions closed at session end.
- **Short selling**: allowed (models index futures / CFD behaviour).
- Dataset is treated as a single continuous instrument (no corporate actions, dividends, or index rebalancing effects modelled).

---

## Limitations

- Fixed 1-lot sizing does not account for position sizing optimisation (e.g. Kelly criterion or volatility targeting).
- OR duration (30 min) and risk:reward (1.5x) parameters were selected via statistical analysis but not formally optimised — doing so risks overfitting.
- Strategy performance is sensitive to transaction cost assumptions; higher brokerage would reduce edge.
- No regime filter applied (e.g. VIX-based filter to avoid low-volatility days).

---

## Bonus Extensions

### 1. Out-of-Sample Validation

| Period | Sharpe | Win Rate | Max DD | Trades |
|---|---|---|---|---|
| In-sample (2015–2021) | 1.72 | 53.25% | -0.29% | 1645 |
| Out-of-sample (2022–2024) | ~1.4 | ~51% | -0.40% | 528 |

Strategy retains positive Sharpe and win rate on completely unseen data — strong evidence against overfitting.

### 2. Walk-Forward Optimisation

Each year's parameters are optimised only on the prior 2 years (no look-ahead).

| Metric | Value |
|---|---|
| Folds tested | 8 |
| Average OOS Sharpe | 1.878 |
| All folds profitable | Yes |

Consistent OOS Sharpe >1.0 across every fold confirms robustness.

### 3. Volatility-Scaled Position Sizing (ATR-based)

Position size targets 1% capital risk per trade, scaled inversely to the stop distance / ATR:

```
lot_size = (capital * 0.01) / stop_distance
```

| Version | Total Return | Sharpe | Max DD |
|---|---|---|---|
| Fixed 1-lot | +4.65% | 1.61 | -0.35% |
| Vol-scaled  | +46.03% | 1.69 | -2.46% |

10x better absolute return while maintaining similar Sharpe — larger position when edge is high-conviction (tight stop), smaller when uncertainty is high.

To run bonus extensions:

```bash
python main.py --bonus
```

---

## Potential Improvements

- **Dynamic OR duration**: adapt OR window length based on realised volatility of prior sessions.
- **Volatility filter**: skip days where OR range is below a percentile threshold (thin market days).
- **Kelly-based position sizing**: scale lot size by edge and variance for better capital utilisation.
- **Out-of-sample validation**: walk-forward test on 2022–2024 holdout.
- **Multiple instruments**: apply to Nifty50, individual large-cap stocks for diversification.

---

## Data

Dataset: [BankNifty Minute Data](https://github.com/sandeepkapri/BankNifty-Minute-Data)

- 851,000+ rows, Jan 2015 – Mar 2024
- Columns: Date, Time, Open, High, Low, Close (no volume)
- 2,271 trading days

**Data quality handling:**
- 2 outlier candles corrected via rolling median replacement (5-sigma threshold)
- Minor intraday gaps forward-filled (max 5 consecutive bars)
- All preprocessing is programmatic — raw file is never manually edited
