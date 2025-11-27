# Algorithmic Trading Backtester (SPY & NVDA)

This project is a simple Python backtesting tool for a moving-average crossover strategy.  
It downloads historical price data using `yfinance`, computes indicators, generates trading signals,  
and evaluates performance using common trading metrics.

---

## Features

- Downloads historical data for multiple tickers (currently SPY and NVDA)
- Computes short- and long-window moving averages
- Generates long/flat trading signals based on MA crossovers
- Calculates daily strategy returns and an equity curve
- Reports:
  - Total return
  - Sharpe ratio (annualized)
  - Maximum drawdown
  - Win rate (percentage of positive days)

---

## File Structure
