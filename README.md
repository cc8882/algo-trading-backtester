# Algorithmic Trading Backtester (SPY & NVDA)

This project is a simple Python backtesting script for a moving-average crossover strategy applied to SPY and NVDA. It is designed to test basic systematic trading rules and compare them against a buy-and-hold benchmark.

## Features
- Downloads historical data using `yfinance`
- Computes short- and long-window moving averages
- Generates long/flat trading signals
- Calculates strategy returns vs. buy-and-hold
- Prints summary performance metrics

## File Structure
```
backtester.py       # Main backtesting script
requirements.txt    # Python dependencies
README.md           # Project overview
```

## How to Run

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the backtester:
```
python backtester.py
```

The script will print:
- Strategy total return  
- Buy-and-hold return  

## Future Improvements
- Add visualization of equity curves  
- Add more strategy types (e.g., breakout, volatility filters)  
- Add Sharpe ratio, max drawdown, and other metrics  
- Expand to additional tickers  
