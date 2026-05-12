# MES/MNQ Futures Backtester — ICT Strategy

A Python backtesting framework for an ICT-based intraday futures strategy on Micro E-Mini S&P 500 (MES) and Micro Nasdaq (MNQ). Built to evaluate edge consistency across rolling time windows under Topstep evaluation constraints.

## Strategy Logic

**Entry conditions (all three must pass):**
1. **Liquidity sweep** — price wicks below the 20-bar rolling low (stop hunt), then closes back above it within 3 bars
2. **VWAP filter** — close is above the daily VWAP at entry
3. **Fair value gap confluence** — at least one active bullish FVG overlaps the sweep reversal zone `[sweep_low, swing_ref]`
   - FVG defined as a 3-candle bullish imbalance: `bar[i−2].high < bar[i].low`
   - Gap zone: `[bar[i−2].high, bar[i].low]` — unfilled demand imbalance
   - FVGs expire after 50 bars (~4 hours at 5m) or when any bar's low breaches `fvg_low`
   - FVGs carry across sessions — overnight gaps remain valid until filled

**Exit:**
- Stop loss: sweep-candle low − 2 ticks
- Target: 2R (fixed risk-reward)
- Force close: 3:45pm ET

**Session filter:** NY open kill zone only (9:30am–12:00pm ET)

## Why the FVG Filter Matters

The FVG confluence filter rejected ~75–90% of raw liquidity sweep setups. This is intentional — in trending or high-volatility regimes (e.g. the April 2026 tariff selloff), bullish FVGs form but get filled quickly as price continues lower, leaving few active demand zones when sweeps occur. The filter correctly sits out most of these conditions. In ranging or recovering markets, FVG overlap increases and trade frequency rises naturally.

## Topstep Eval Constraints

- Daily loss limit: −$500 (halts new entries for the day, logged as `DLL`)
- Max 3 trades per day
- Trailing drawdown monitor: warns when equity is within $300 of 2% below peak

## Results (Feb–Apr 2026, 5m bars, 60 trading days)

| Window | MES Win% | MES PF | NQ Win% | NQ PF | FVG- Filtered |
|---|---|---|---|---|---|
| Feb 4–27 | 57.1% | 3.02 | 63.6% | 1.88 | — |
| Mar 1–23 | 50.0% | 2.35 | 60.0% | 2.19 | — |
| Mar 24–Apr 16 | 66.7% | 10.07 | 0.0% | 0.00 | — |

> Update FVG- column after re-running with confluence filter active.

**MES total P&L: +$948.75 | NQ total P&L: +$950.50**

MES showed consistent edge across all three windows with no daily loss limit hits. NQ broke down during the April 2026 tariff-driven volatility — sweeps followed through instead of reversing, which is expected behavior in strongly trending conditions. **MES is the recommended instrument for evaluation.**

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run windowed consistency test on both instruments (main mode)
python backtester.py --compare

# Single instrument
python backtester.py --ticker MES=F --days 60

# Load from your own CSV (bypasses yfinance 60-day limit)
python backtester.py --csv mydata.csv

# Flags
--no-plot                      # Skip chart output
--no-daily-log                 # Skip per-day console output
--export-trades trades.csv     # Export full trade log
--output chart.png             # Save chart to file
```

## Data

Uses `yfinance` for live data pulls:
- **5m bars:** ~60 calendar days (yfinance hard limit — use `period="60d"`)
- **1h bars:** up to 730 days

For longer backtests, export your own CSV from Tradovate, Norgate, or CQG and use the `--csv` flag.

## Output

Running `--compare` generates two chart files:

- `backtest_results.png` — equity curve, monthly P&L, R-distribution, drawdown
- `window_comparison.png` — per-window equity curves, win rate bars, profit factor bars, monthly P&L

## Tech Stack

- Python 3.9+
- pandas, numpy, yfinance, matplotlib

## Notes

Edge is strongest in mean-reverting, range-bound sessions. Performance degrades in high-volatility trending regimes (major macro events, news-driven moves). The windowed consistency test is designed to surface regime sensitivity before committing real capital. To loosen FVG selectivity, reduce `FVG_LOOKBACK` or widen the overlap condition — both are single-line changes at the top of the file.
