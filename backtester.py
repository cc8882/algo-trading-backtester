import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# --- Parameters ---
TICKERS = ["SPY", "NVDA"]          # Symbols to backtest
START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

SHORT_WINDOW = 20                  # Short moving average window
LONG_WINDOW = 50                   # Long moving average window
INITIAL_CAPITAL = 10_000           # Starting portfolio value


def download_data(ticker: str) -> pd.DataFrame:
    """
    Download historical adjusted close prices for a single ticker.
    """
    data = yf.download(ticker, start=START_DATE, end=END_DATE)
    data = data[["Adj Close"]].rename(columns={"Adj Close": "price"})
    data.dropna(inplace=True)
    return data


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add moving-average crossover signals and positions.
    signal = 1 when short MA > long MA, else 0.
    position is yesterday's signal (what we actually trade).
    """
    df["ma_short"] = df["price"].rolling(window=SHORT_WINDOW).mean()
    df["ma_long"] = df["price"].rolling(window=LONG_WINDOW).mean()
    df["signal"] = (df["ma_short"] > df["ma_long"]).astype(int)
    df["position"] = df["signal"].shift(1).fillna(0)
    return df


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily strategy returns and equity curve.
    """
    df["returns"] = df["price"].pct_change().fillna(0)
    df["strategy_returns"] = df["position"] * df["returns"]
    df["equity_curve"] = (1 + df["strategy_returns"]).cumprod() * INITIAL_CAPITAL
    return df


def performance_stats(df: pd.DataFrame) -> dict:
    """
    Calculate total return, Sharpe ratio, max drawdown, and win rate.
    """
    total_return = df["equity_curve"].iloc[-1] / INITIAL_CAPITAL - 1

    daily_ret = df["strategy_returns"]
    sharpe = 0.0
    if daily_ret.std() != 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)

    cum_max = df["equity_curve"].cummax()
    max_dd = (df["equity_curve"] / cum_max - 1).min()

    wins = (daily_ret > 0).sum()
    losses = (daily_ret < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    return {
        "total_return_pct": total_return * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "win_rate_pct": win_rate * 100,
    }


def main():
    for ticker in TICKERS:
        df = download_data(ticker)
        df = add_signals(df)
        df = backtest(df)
        stats = performance_stats(df)

        print(f"\n===== Stats for {ticker} =====")
        for k, v in stats.items():
            if "pct" in k:
                print(f"{k}: {v:.2f}")
            else:
                print(f"{k}: {v:.3f}")
        print("==============================")


if __name__ == "__main__":
    main()
