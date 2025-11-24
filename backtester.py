
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

plt.switch_backend("Agg")

TICKERS = ["SPY", "NVDA"]
START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

SHORT_WINDOW = 20
LONG_WINDOW = 50
INITIAL_CAPITAL = 10000

def download_data(ticker):
    data = yf.download(ticker, start=START_DATE, end=END_DATE)
    data = data[["Adj Close"]].rename(columns={"Adj Close": "price"})
    data.dropna(inplace=True)
    return data

def add_signals(df):
    df["ma_short"] = df["price"].rolling(window=SHORT_WINDOW).mean()
    df["ma_long"] = df["price"].rolling(window=LONG_WINDOW).mean()
    df["signal"] = (df["ma_short"] > df["ma_long"]).astype(int)
    df["position"] = df["signal"].shift(1).fillna(0)
    return df

def backtest(df):
    df["returns"] = df["price"].pct_change().fillna(0)
    df["strategy_returns"] = df["position"] * df["returns"]
    df["equity_curve"] = (1 + df["strategy_returns"]).cumprod() * INITIAL_CAPITAL
    return df

def performance_stats(df):
    total_return = df["equity_curve"].iloc[-1] / INITIAL_CAPITAL - 1
    daily_ret = df["strategy_returns"]
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std()!=0 else 0
    cum_max = df["equity_curve"].cummax()
    max_dd = (df["equity_curve"]/cum_max -1).min()
    wins = (daily_ret>0).sum()
    losses = (daily_ret<0).sum()
    win_rate = wins/(wins+losses) if (wins+losses)>0 else 0
    return {
        "total_return_pct": total_return*100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd*100,
        "win_rate_pct": win_rate*100
    }

def main():
    for ticker in TICKERS:
        df = download_data(ticker)
        df = add_signals(df)
        df = backtest(df)
        stats = performance_stats(df)
        print(f"Stats for {ticker}:")
        for k,v in stats.items():
            print(k, v)

if __name__=="__main__":
    main()
