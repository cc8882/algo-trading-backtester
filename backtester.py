#!/usr/bin/env python3
"""
MES / MNQ Liquidity Grab Backtester — Topstep Eval Edition
===========================================================
Entry rules:
  1. Price sweeps a previous swing low (bar low < rolling swing-low level)
  2. Within SWEEP_WINDOW bars, price closes back above that level
  3. Close is above daily VWAP
  4. Trade time falls inside the NY open kill zone (default 9:30–12:00 ET)

Exit rules:
  - Stop loss  : below sweep candle low − SL_BUFFER_TICKS
  - Take profit: entry + 2R
  - Force close: 3:45 pm ET

Topstep constraints:
  - Daily loss limit  : −$500 → halt trading for the day
  - Max trades/day    : 3
  - Trailing DD warn  : flag when equity is within $300 of 2 % below peak

Interval / data-availability note:
  - 5m bars  : yfinance stores only the last ~60–70 calendar days
  - 1h bars  : up to 730 days available
  For a 3-window consistency test on 5m, the script splits the available
  ~60d block into three equal sub-windows and labels them accordingly.
  Older windows are probed first and their status is reported.

Usage:
  /usr/bin/python3 backtester.py --compare
  /usr/bin/python3 backtester.py --compare --no-plot
  /usr/bin/python3 backtester.py --ticker MES=F --days 60
  /usr/bin/python3 backtester.py --csv data.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import pytz
    _ET = pytz.timezone("America/New_York")
except ImportError:
    _ET = None


# ─── Strategy parameters ──────────────────────────────────────────────────────

SWING_LOW_LOOKBACK  = 20
SWEEP_WINDOW        = 3
SL_BUFFER_TICKS     = 2
TICK_SIZE           = 0.25
RISK_REWARD         = 2.0

# ─── Fair Value Gap parameters ────────────────────────────────────────────────
# Bullish FVG: 3-candle imbalance where bar[i-2].high < bar[i].low
# Gap zone = [bar[i-2].high, bar[i].low]  — unfilled demand imbalance
# An FVG is "filled" when any subsequent bar's low breaches its bottom (fvg_low).
# Entry only fires when at least one active FVG overlaps the sweep reversal zone.

FVG_LOOKBACK        = 50       # bars; FVGs older than this are discarded

# ─── Topstep eval constraints ─────────────────────────────────────────────────

SESSION_START_ET    = dtime(9, 30)
SESSION_END_ET      = dtime(12, 0)
FORCE_CLOSE_ET      = dtime(15, 45)
DAILY_LOSS_LIMIT    = -500.0
MAX_TRADES_PER_DAY  = 3
TRAILING_DD_PCT     = 0.02
TRAILING_DD_WARN    = 300.0

INITIAL_CAPITAL     = 100_000

# ─── Instrument presets ───────────────────────────────────────────────────────

INSTRUMENTS = {
    "MES": dict(ticker="MES=F", label="MES (ES)", multiplier=5,  contracts=1),
    "NQ":  dict(ticker="NQ=F",  label="NQ (MNQ)", multiplier=2,  contracts=1),
}

DEFAULT_TICKER  = "MES=F"
DEFAULT_DAYS    = 60       # 5m mode default
N_WINDOWS       = 3


# ─── Data loading ─────────────────────────────────────────────────────────────

def _fix_columns(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index)
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    raw.index = raw.index.tz_convert("America/New_York")
    return raw[["open", "high", "low", "close", "volume"]].dropna()


def load_period(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download using yfinance period= string (reliable for 5m)."""
    if not HAS_YFINANCE:
        sys.exit("yfinance not installed.  Run:  pip install yfinance")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(ticker, period=period, interval=interval,
                          auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    return _fix_columns(raw)


def attempt_window(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """
    Try to download a specific date-range window.
    Returns empty DataFrame if yfinance rejects the range (e.g. 5m > 60d).
    """
    if not HAS_YFINANCE:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(ticker, start=start, end=end, interval=interval,
                          auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    return _fix_columns(raw)


def load_from_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"CSV not found: {path}")
    df = pd.read_csv(p, parse_dates=True, index_col=0)
    df.columns = [c.lower().strip() for c in df.columns]
    missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        sys.exit(f"CSV missing columns: {missing}")
    df.index = pd.to_datetime(df.index)
    return df[["open", "high", "low", "close", "volume"]].dropna()


def split_into_windows(df: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    """
    Split df into n equal sub-windows by trading day count.
    Each window gets roughly total_trading_days / n days.
    """
    trading_dates = sorted(df.index.normalize().unique())
    total = len(trading_dates)
    windows = []
    for i in range(n):
        start_idx = int(round(i * total / n))
        end_idx   = int(round((i + 1) * total / n))
        lo = trading_dates[start_idx]
        if i < n - 1:
            hi   = trading_dates[min(end_idx, total - 1)]
            mask = (df.index.normalize() >= lo) & (df.index.normalize() < hi)
        else:
            mask = df.index.normalize() >= lo
        w = df[mask].copy()
        if not w.empty:
            windows.append(w)
    return windows


# ─── Indicators ───────────────────────────────────────────────────────────────

def daily_vwap(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    d["_date"] = d.index.normalize()
    d["_tp"]   = (d["high"] + d["low"] + d["close"]) / 3
    d["_tpv"]  = d["_tp"] * d["volume"]
    d["_ctpv"] = d.groupby("_date")["_tpv"].cumsum()
    d["_cvol"] = d.groupby("_date")["volume"].cumsum()
    return (d["_ctpv"] / d["_cvol"].replace(0, np.nan)).rename("vwap")


def swing_low_series(low: pd.Series, lookback: int) -> pd.Series:
    return low.shift(1).rolling(lookback, min_periods=lookback).min()


# ─── Fair Value Gap helpers ───────────────────────────────────────────────────

def _has_fvg_confluence(
    active_fvgs: list[dict],
    sweep_low: float,
    swing_ref: float,
) -> tuple[bool, dict | None]:
    """
    Return (True, fvg) if any active bullish FVG overlaps the sweep reversal zone.

    Reversal zone  = [sweep_low, swing_ref]   (where the wick dipped and is recovering)
    FVG zone       = [fvg_low,   fvg_high]    (unfilled bullish imbalance)
    Overlap test   : fvg_low <= swing_ref  AND  fvg_high >= sweep_low

    This captures the setup where the wick swept INTO a bullish demand imbalance
    and the close-back confirms demand absorbed the liquidity.
    """
    for f in active_fvgs:
        if f["fvg_low"] <= swing_ref and f["fvg_high"] >= sweep_low:
            return True, f
    return False, None


# ─── Backtest engine ──────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    contract_multiplier: float = 5,
    contracts: int = 1,
) -> dict:
    df = df.copy()
    df["vwap"]      = daily_vwap(df)
    df["swing_low"] = swing_low_series(df["low"], SWING_LOW_LOOKBACK)
    df.dropna(subset=["vwap", "swing_low"], inplace=True)

    equity: list[float]      = [float(INITIAL_CAPITAL)]
    peak_series: list[float] = [float(INITIAL_CAPITAL)]
    peak_equity              = float(INITIAL_CAPITAL)

    trades: list[dict] = []
    in_trade    = False
    entry_price = sl = tp = entry_time = None
    pending: dict | None = None

    active_fvgs: list[dict] = []   # active bullish FVGs; each: {fvg_low, fvg_high, formed_idx}
    fvg_filtered = 0               # close-backs that passed sweep/VWAP but failed FVG check

    active_day       = None
    day_start_equity = float(INITIAL_CAPITAL)
    day_trades       = 0
    stop_today       = False
    day_dll_hit      = False
    day_dd_warnings: list = []
    day_log: dict    = {}

    def _close(exit_px: float, reason: str, bar: pd.Series) -> None:
        nonlocal in_trade, pending, peak_equity, stop_today, day_dll_hit
        pnl_pts = exit_px - entry_price
        pnl_usd = pnl_pts * contract_multiplier * contracts
        risk    = entry_price - sl
        r_val   = round(pnl_pts / risk, 3) if risk > 0 else 0.0
        trades.append({
            "entry_time":  entry_time,
            "exit_time":   bar.name,
            "entry_price": round(entry_price, 2),
            "exit_price":  round(float(exit_px), 2),
            "sl":          round(sl, 2),
            "tp":          round(tp, 2),
            "pnl_pts":     round(pnl_pts, 2),
            "pnl_usd":     round(pnl_usd, 2),
            "r":           r_val,
            "win":         bool(pnl_pts > 0),
            "exit_reason": reason,
        })
        new_eq = equity[-1] + pnl_usd
        equity.append(new_eq)
        if new_eq > peak_equity:
            peak_equity = new_eq
        peak_series.append(peak_equity)
        if new_eq <= peak_equity * (1.0 - TRAILING_DD_PCT) + TRAILING_DD_WARN:
            day_dd_warnings.append(bar.name)
        if new_eq - day_start_equity <= DAILY_LOSS_LIMIT:
            stop_today  = True
            day_dll_hit = True
        in_trade = False
        pending  = None

    for i in range(len(df)):
        bar      = df.iloc[i]
        bar_date = bar.name.date()
        bar_time = bar.name.time()

        # ── Bullish FVG detection (runs every bar regardless of session/trade state) ──
        # Pattern: bar[i-2].high < bar[i].low  →  unfilled demand gap in between
        if i >= 2:
            b0 = df.iloc[i - 2]
            if b0["high"] < bar["low"]:
                active_fvgs.append({
                    "fvg_low":    float(b0["high"]),
                    "fvg_high":   float(bar["low"]),
                    "formed_idx": i,
                })
        # Prune: filled (any wick below fvg_low) or too old
        active_fvgs = [
            f for f in active_fvgs
            if bar["low"] > f["fvg_low"] and (i - f["formed_idx"]) <= FVG_LOOKBACK
        ]

        if bar_date != active_day:
            if active_day is not None:
                day_log[active_day] = {
                    "trades":      day_trades,
                    "pnl":         round(equity[-1] - day_start_equity, 2),
                    "end_equity":  round(equity[-1], 2),
                    "dll_hit":     day_dll_hit,
                    "dd_warnings": list(day_dd_warnings),
                }
            active_day       = bar_date
            day_start_equity = equity[-1]
            day_trades       = 0
            stop_today       = False
            day_dll_hit      = False
            day_dd_warnings  = []
            pending          = None

        if in_trade:
            hit_sl = bool(bar["low"]  <= sl)
            hit_tp = bool(bar["high"] >= tp)
            force  = bar_time >= FORCE_CLOSE_ET
            if hit_sl or hit_tp or force:
                if hit_sl:
                    _close(sl,           "sl",           bar)
                elif hit_tp:
                    _close(tp,           "tp",           bar)
                else:
                    _close(bar["close"], "force_3:45pm", bar)
            continue

        if stop_today or day_trades >= MAX_TRADES_PER_DAY:
            pending = None
            continue

        if not (SESSION_START_ET <= bar_time < SESSION_END_ET):
            if bar_time >= SESSION_END_ET:
                pending = None
            continue

        swing_ref = float(bar["swing_low"])
        if pending is None:
            if bar["low"] < swing_ref:
                pending = {"swing_ref": swing_ref, "sweep_low": float(bar["low"]),
                           "bars_since": 0}
        else:
            pending["bars_since"] += 1
            if bar["close"] > pending["swing_ref"] and bar["close"] > bar["vwap"]:
                # Gate 1 (existing): sweep close-back above swing_ref while above VWAP
                # Gate 2 (new):      at least one active bullish FVG overlaps the sweep zone
                fvg_ok, fvg_ref = _has_fvg_confluence(
                    active_fvgs, pending["sweep_low"], pending["swing_ref"]
                )
                if fvg_ok:
                    ep   = float(bar["close"])
                    sl_r = pending["sweep_low"] - SL_BUFFER_TICKS * TICK_SIZE
                    risk = ep - sl_r
                    if risk > 0:
                        entry_price = ep
                        sl          = sl_r
                        tp          = ep + RISK_REWARD * risk
                        entry_time  = bar.name
                        in_trade    = True
                        day_trades += 1
                else:
                    fvg_filtered += 1   # sweep + VWAP aligned but no FVG in zone
                # Close-back happened — pending is consumed whether or not FVG fired
                pending = None
            elif pending["bars_since"] >= SWEEP_WINDOW:
                pending = None

    if active_day is not None:
        day_log[active_day] = {
            "trades":      day_trades,
            "pnl":         round(equity[-1] - day_start_equity, 2),
            "end_equity":  round(equity[-1], 2),
            "dll_hit":     day_dll_hit,
            "dd_warnings": list(day_dd_warnings),
        }
    if in_trade:
        _close(float(df.iloc[-1]["close"]), "end_of_data", df.iloc[-1])

    return {
        "trades":        pd.DataFrame(trades),
        "equity":        np.array(equity,      dtype=float),
        "peak_series":   np.array(peak_series, dtype=float),
        "day_log":       day_log,
        "fvg_filtered":  fvg_filtered,
    }


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    trades: pd.DataFrame,
    equity: np.ndarray,
    peak_series: np.ndarray,
    day_log: dict,
) -> dict:
    if trades.empty:
        return {"empty": True, "total_trades": 0}
    n      = len(trades)
    n_wins = int(trades["win"].sum())
    gp     = float(trades.loc[trades["win"],  "pnl_usd"].sum())
    gl     = float(trades.loc[~trades["win"], "pnl_usd"].sum())
    running_max = np.maximum.accumulate(equity)
    drawdowns   = (equity - running_max) / running_max * 100
    dll_dates   = [str(d) for d, v in day_log.items() if v["dll_hit"]]
    warn_dates  = [str(d) for d, v in day_log.items() if v["dd_warnings"]]
    return {
        "total_trades":   n,
        "wins":           n_wins,
        "losses":         n - n_wins,
        "win_rate":       n_wins / n * 100,
        "avg_r":          float(trades["r"].mean()),
        "avg_win_usd":    float(trades.loc[trades["win"],  "pnl_usd"].mean()) if n_wins else 0.0,
        "avg_loss_usd":   float(trades.loc[~trades["win"], "pnl_usd"].mean()) if (n-n_wins) else 0.0,
        "gross_profit":   gp,
        "gross_loss":     gl,
        "profit_factor":  gp / abs(gl) if gl != 0 else float("inf"),
        "total_pnl":      float(trades["pnl_usd"].sum()),
        "max_dd_pct":     float(drawdowns.min()),
        "net_return_pct": (float(equity[-1]) - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "final_equity":   float(equity[-1]),
        "trading_days":   sum(1 for v in day_log.values() if v["trades"] > 0),
        "all_days":       len(day_log),
        "dll_count":      len(dll_dates),
        "dll_dates":      dll_dates,
        "dd_warn_count":  len(warn_dates),
        "dd_warn_dates":  warn_dates,
        "drawdowns":      drawdowns,
        "trail_limit":    peak_series * (1.0 - TRAILING_DD_PCT),
    }


def monthly_pnl(day_log: dict) -> dict[str, dict]:
    monthly: dict[str, dict] = {}
    for date, v in day_log.items():
        key = date.strftime("%Y-%m")
        if key not in monthly:
            monthly[key] = {"pnl": 0.0, "trades": 0, "dll_days": 0}
        monthly[key]["pnl"]      += v["pnl"]
        monthly[key]["trades"]   += v["trades"]
        monthly[key]["dll_days"] += int(v["dll_hit"])
    return dict(sorted(monthly.items()))


# ─── Windowed analysis ────────────────────────────────────────────────────────

def _window_date_label(df: pd.DataFrame) -> str:
    d0 = df.index[0].strftime("%Y-%m-%d")
    d1 = df.index[-1].strftime("%Y-%m-%d")
    trading_days = len(df.index.normalize().unique())
    return f"{d0} → {d1}  ({trading_days} trading days,  {len(df):,} bars)"


def probe_windows(ticker: str, interval: str) -> tuple[pd.DataFrame, list[dict]]:
    """
    1. Probe the three requested 60-day windows (oldest → newest).
    2. Return the newest available block (loaded via period=) and a status
       report list so the caller can print what was found vs missing.
    """
    today = datetime.now(_ET) if _ET else datetime.utcnow()
    probes = [
        {"label": "W1 (120–180 d ago)",
         "start": (today - timedelta(days=180)).strftime("%Y-%m-%d"),
         "end":   (today - timedelta(days=120)).strftime("%Y-%m-%d")},
        {"label": "W2  (60–120 d ago)",
         "start": (today - timedelta(days=120)).strftime("%Y-%m-%d"),
         "end":   (today - timedelta(days=60)).strftime("%Y-%m-%d")},
        {"label": "W3    (0–60 d ago)",
         "start": (today - timedelta(days=60)).strftime("%Y-%m-%d"),
         "end":   today.strftime("%Y-%m-%d")},
    ]
    statuses = []
    for p in probes:
        df_w = attempt_window(ticker, p["start"], p["end"], interval)
        if df_w.empty:
            statuses.append({**p, "available": False, "bars": 0, "note": "not in yfinance cache"})
        else:
            statuses.append({**p, "available": True,  "bars": len(df_w),
                             "actual_start": df_w.index[0].strftime("%Y-%m-%d"),
                             "actual_end":   df_w.index[-1].strftime("%Y-%m-%d")})

    # Always reload the full available block via period= (most reliable)
    full_df = load_period(ticker, "60d", interval)
    return full_df, statuses


def run_windowed(
    ticker: str,
    multiplier: float,
    contracts: int,
    interval: str,
    n: int = N_WINDOWS,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """
    Download the full available block, split into n windows, run each backtest.
    Returns: (full_df, probe_statuses, window_results).
    """
    full_df, statuses = probe_windows(ticker, interval)
    if full_df.empty:
        return full_df, statuses, []

    sub_windows = split_into_windows(full_df, n)
    results = []
    for i, w in enumerate(sub_windows):
        res     = run_backtest(w, multiplier, contracts)
        metrics = compute_metrics(res["trades"], res["equity"],
                                  res["peak_series"], res["day_log"])
        label   = f"Sub-W{i+1}"
        results.append({
            "label":      label,
            "df":         w,
            "date_label": _window_date_label(w),
            "result":     res,
            "metrics":    metrics,
        })
    return full_df, statuses, results


# ─── Console output ───────────────────────────────────────────────────────────

def _pnl_str(v: float) -> str:
    return f"{'+' if v >= 0 else ''}${v:,.2f}"


def _pf_str(pf: float) -> str:
    return "∞" if pf == float("inf") else f"{pf:.2f}"


def print_probe_report(statuses: list[dict], ticker: str, interval: str) -> None:
    print(f"\n  Data availability for {ticker} ({interval} bars):")
    for s in statuses:
        if s["available"]:
            print(f"    {s['label']}  {s['start']} → {s['end']}:  "
                  f"✓  {s['bars']:,} bars  "
                  f"(actual: {s['actual_start']} → {s['actual_end']})")
        else:
            print(f"    {s['label']}  {s['start']} → {s['end']}:  "
                  f"✗  {s['note']}")


def print_window_table(inst_windows: list[dict], label: str) -> None:
    """Print per-window stats for a single instrument."""
    W = 90
    print(f"\n  ── {label} ──")
    print(f"  {'Window':<9}  {'Date Range':>43}  {'Tr':>3}  {'FVG-':>4}  "
          f"{'Win%':>6}  {'PF':>5}  {'MaxDD':>7}  {'DLL':>3}")
    print("  " + "─" * (W - 2))
    for wr in inst_windows:
        m    = wr["metrics"]
        filt = wr["result"]["fvg_filtered"]
        if m.get("empty"):
            print(f"  {wr['label']:<9}  {wr['date_label']:<43}  "
                  f"  —    {filt:>4}  no trades")
            continue
        print(
            f"  {wr['label']:<9}  {wr['date_label']:<43}  "
            f"{m['total_trades']:>3}  "
            f"{filt:>4}  "
            f"{m['win_rate']:>5.1f}%  "
            f"{_pf_str(m['profit_factor']):>5}  "
            f"{m['max_dd_pct']:>6.2f}%  "
            f"{m['dll_count']:>3}"
        )


def print_window_comparison(
    mes_wins: list[dict], nq_wins: list[dict],
    interval: str,
) -> None:
    W = 92
    sep = "═" * W
    print(f"\n{sep}")
    print(f"  Window Consistency Analysis — 5m bars, {N_WINDOWS} sub-windows  "
          f"|  FVG confluence filter active  (lookback={FVG_LOOKBACK} bars)")
    print(f"  Columns: Tr=entries taken  FVG-=setups rejected by FVG filter  "
          f"PF=profit factor  MaxDD=max drawdown  DLL=daily loss limit days")
    print(sep)
    print_window_table(mes_wins, "MES (ES)  — $5/point  ×1 MES contract")
    print_window_table(nq_wins,  "NQ (MNQ)  — $2/point  ×1 MNQ contract")
    print(f"\n{sep}\n")


def print_monthly_comparison(
    day_log_a: dict, label_a: str,
    day_log_b: dict, label_b: str,
) -> None:
    mon_a = monthly_pnl(day_log_a)
    mon_b = monthly_pnl(day_log_b)
    all_months = sorted(set(mon_a) | set(mon_b))

    W = 76
    sep = "═" * W
    sub = "─" * W
    print(sep)
    print(f"  Monthly P&L Breakdown — full available period")
    print(f"  {'Month':<9}  {label_a[:12]:<12} {'Tr':>3} {'DLL':>3}   "
          f"{label_b[:12]:<12} {'Tr':>3} {'DLL':>3}   {'Diff':>10}")
    print(sub)
    total_a = total_b = 0.0
    prof_a = prof_b = 0
    active = 0
    for m in all_months:
        va = mon_a.get(m, {"pnl": 0.0, "trades": 0, "dll_days": 0})
        vb = mon_b.get(m, {"pnl": 0.0, "trades": 0, "dll_days": 0})
        if va["trades"] == 0 and vb["trades"] == 0:
            continue
        active += 1
        diff   = va["pnl"] - vb["pnl"]
        pa     = _pnl_str(va["pnl"]).rjust(12)
        pb     = _pnl_str(vb["pnl"]).rjust(12)
        dstr   = _pnl_str(diff).rjust(10)
        da     = str(va["dll_days"]) if va["dll_days"] else "-"
        db     = str(vb["dll_days"]) if vb["dll_days"] else "-"
        print(f"  {m:<9}  {pa} {va['trades']:>3} {da:>3}   "
              f"{pb} {vb['trades']:>3} {db:>3}   {dstr}")
        total_a += va["pnl"]
        total_b += vb["pnl"]
        if va["pnl"] > 0: prof_a += 1
        if vb["pnl"] > 0: prof_b += 1
    print(sub)
    pa_t = _pnl_str(total_a).rjust(12)
    pb_t = _pnl_str(total_b).rjust(12)
    dt   = _pnl_str(total_a - total_b).rjust(10)
    print(f"  {'TOTAL':<9}  {pa_t} {'':>3} {'':>3}   {pb_t} {'':>3} {'':>3}   {dt}")
    print(f"  Profitable months:  {label_a[:10]} {prof_a}/{active}   "
          f"{label_b[:10]} {prof_b}/{active}")
    print(f"{sep}\n")


def print_daily_log(day_log: dict, label: str = "") -> None:
    print("─" * 72)
    print(f"  Daily P&L Log{' — ' + label if label else ''}")
    print(f"  {'Date':<12} {'Trades':>6}  {'Day P&L':>10}  {'Cum. Equity':>13}  "
          f"{'DLL':>4}  {'DD':>4}")
    print("─" * 72)
    for date, v in sorted(day_log.items()):
        if v["trades"] == 0:
            continue
        dll_flag = "DLL!" if v["dll_hit"]     else "-"
        dd_flag  = "WARN" if v["dd_warnings"] else "-"
        print(
            f"  {str(date):<12} {v['trades']:>6}  "
            f"{_pnl_str(v['pnl']):>10}  ${v['end_equity']:>12,.2f}  "
            f"{dll_flag:>4}  {dd_flag:>4}"
        )
    print("─" * 72 + "\n")


# ─── Plots ────────────────────────────────────────────────────────────────────

_WINDOW_COLORS = ["#2980b9", "#27ae60", "#e67e22"]   # Sub-W1, Sub-W2, Sub-W3
_INST_COLORS   = {"MES (ES)": "#2980b9", "NQ (MNQ)": "#e67e22"}


def _eq_pct(eq: np.ndarray) -> np.ndarray:
    return (eq - eq[0]) / eq[0] * 100.0


def plot_window_comparison(
    mes_wins: list[dict],
    nq_wins: list[dict],
    full_day_log_mes: dict,
    full_day_log_nq: dict,
    output_path: str,
) -> None:
    mon_mes = monthly_pnl(full_day_log_mes)
    mon_nq  = monthly_pnl(full_day_log_nq)
    all_months = sorted(set(mon_mes) | set(mon_nq))
    active_months = [m for m in all_months
                     if mon_mes.get(m, {"trades": 0})["trades"] +
                        mon_nq.get(m,  {"trades": 0})["trades"] > 0]

    fig = plt.figure(figsize=(17, 15))
    fig.suptitle(
        f"MES vs NQ — Liquidity Grab  |  5m bars  |  "
        f"{N_WINDOWS}-window consistency test\n"
        f"kill zone {SESSION_START_ET.strftime('%H:%M')}–{SESSION_END_ET.strftime('%H:%M')} ET  "
        f"|  target {RISK_REWARD}R  |  DLL ${abs(DAILY_LOSS_LIMIT):.0f}  "
        f"|  max {MAX_TRADES_PER_DAY} trades/day",
        fontsize=11, fontweight="bold",
    )
    gs = fig.add_gridspec(3, 2, hspace=0.46, wspace=0.30)

    # ── Row 0: per-window equity curves ──────────────────────────────────────
    for col, (inst_label, wins) in enumerate([
        ("MES (ES)", mes_wins), ("NQ (MNQ)", nq_wins)
    ]):
        ax = fig.add_subplot(gs[0, col])
        for wi, wr in enumerate(wins):
            if wr["metrics"].get("empty"):
                continue
            eq  = wr["result"]["equity"]
            idx = np.arange(len(eq))
            ax.plot(idx, _eq_pct(eq),
                    color=_WINDOW_COLORS[wi % len(_WINDOW_COLORS)],
                    lw=1.6, label=f"{wr['label']} ({wr['df'].index[0].strftime('%b %d')}–"
                                  f"{wr['df'].index[-1].strftime('%b %d')})")
        ax.axhline(0, color="#7f8c8d", lw=0.7, ls="--")
        ax.axhline(-TRAILING_DD_PCT * 100, color="#c0392b", lw=0.8, ls="--",
                   label=f"{TRAILING_DD_PCT*100:.0f}% trailing limit", alpha=0.7)
        ax.set_title(f"{inst_label} — Equity Curve per Window (% return)", fontsize=9)
        ax.set_ylabel("% return")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    # ── Row 1: win rate and profit factor per window ──────────────────────────
    window_labels = [wr["label"] for wr in mes_wins]
    x = np.arange(len(window_labels))
    w = 0.35

    for col, metric_key, title, fmt in [
        (0, "win_rate",      "Win Rate by Window (%)",    lambda v: v),
        (1, "profit_factor", "Profit Factor by Window",   lambda v: min(v, 8.0)),
    ]:
        ax = fig.add_subplot(gs[1, col])
        mes_vals = [wr["metrics"].get(metric_key, 0) for wr in mes_wins]
        nq_vals  = [wr["metrics"].get(metric_key, 0) for wr in nq_wins]
        mes_cols = [("#2980b9" if v >= 50 else "#5d9fcf") if metric_key == "win_rate"
                    else ("#2980b9" if v >= 1.0 else "#5d9fcf") for v in mes_vals]
        nq_cols  = [("#e67e22" if v >= 50 else "#f0a76c") if metric_key == "win_rate"
                    else ("#e67e22" if v >= 1.0 else "#f0a76c") for v in nq_vals]
        bars_mes = ax.bar(x - w/2, [fmt(v) for v in mes_vals], width=w,
                          color=mes_cols, alpha=0.85, label="MES (ES)", zorder=2)
        bars_nq  = ax.bar(x + w/2, [fmt(v) for v in nq_vals],  width=w,
                          color=nq_cols,  alpha=0.85, label="NQ (MNQ)", zorder=2)
        # label bars
        for bar in list(bars_mes) + list(bars_nq):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.02 * h,
                        f"{h:.1f}" if metric_key == "win_rate" else f"{h:.2f}",
                        ha="center", va="bottom", fontsize=7)
        ref = 50 if metric_key == "win_rate" else 1.0
        ax.axhline(ref, color="#c0392b", lw=0.9, ls="--",
                   label=f"Break-even ({ref}{'%' if metric_key=='win_rate' else ''})", zorder=3)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("%" if metric_key == "win_rate" else "")
        ax.set_xticks(x)
        ax.set_xticklabels(window_labels, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, axis="y")

    # ── Row 2: monthly P&L (full period, both instruments) ────────────────────
    ax_mon = fig.add_subplot(gs[2, :])
    bw = 0.38
    xm = np.arange(len(active_months))
    vals_mes = [mon_mes.get(m, {"pnl": 0.0})["pnl"] for m in active_months]
    vals_nq  = [mon_nq.get(m,  {"pnl": 0.0})["pnl"] for m in active_months]
    ax_mon.bar(xm - bw/2, vals_mes, width=bw, alpha=0.85, zorder=2,
               color=["#2980b9" if v >= 0 else "#5d9fcf" for v in vals_mes],
               label="MES (ES)")
    ax_mon.bar(xm + bw/2, vals_nq,  width=bw, alpha=0.85, zorder=2,
               color=["#e67e22" if v >= 0 else "#f0a76c" for v in vals_nq],
               label="NQ (MNQ)")
    ax_mon.axhline(0, color="#2c3e50", lw=0.7)
    ax_mon.axhline(DAILY_LOSS_LIMIT, color="#c0392b", lw=0.9, ls="--",
                   label=f"DLL ${DAILY_LOSS_LIMIT:.0f}/day", zorder=3)
    ax_mon.set_title("Monthly P&L — full available period", fontsize=9)
    ax_mon.set_ylabel("USD")
    ax_mon.set_xticks(xm)
    ax_mon.set_xticklabels([m[5:] for m in active_months], rotation=45, fontsize=8)
    ax_mon.legend(fontsize=7, ncol=3)
    ax_mon.grid(True, alpha=0.2, axis="y")
    ax_mon.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved → {output_path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try: plt.show()
        except Exception: pass
    plt.close()


def plot_single(
    label: str, trades: pd.DataFrame, equity: np.ndarray,
    peak_series: np.ndarray, day_log: dict, metrics: dict,
    output_path: str,
) -> None:
    mon      = monthly_pnl(day_log)
    mon_keys = [k for k, v in mon.items() if v["trades"] > 0]
    mon_vals = [mon[k]["pnl"] for k in mon_keys]
    mon_cols = ["#e74c3c" if v < 0 else "#2ecc71" for v in mon_vals]
    eq_idx   = np.arange(len(equity))
    trail    = peak_series * (1.0 - TRAILING_DD_PCT)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    fig.suptitle(
        f"{label} — Liquidity Grab  |  kill zone "
        f"{SESSION_START_ET.strftime('%H:%M')}–{SESSION_END_ET.strftime('%H:%M')} ET  "
        f"|  target {RISK_REWARD}R  |  DLL ${abs(DAILY_LOSS_LIMIT):.0f}",
        fontsize=10, fontweight="bold",
    )
    ax = axes[0]
    ax.plot(eq_idx, equity, color="#2980b9", lw=1.8, label="Equity")
    ax.plot(eq_idx, trail,  color="#c0392b", lw=1.0, ls="--",
            label=f"{TRAILING_DD_PCT*100:.0f}% trailing limit")
    ax.axhline(INITIAL_CAPITAL, color="#7f8c8d", lw=0.7, ls="--")
    ax.fill_between(eq_idx, INITIAL_CAPITAL, equity,
                    where=(equity >= INITIAL_CAPITAL), alpha=0.12, color="#2ecc71")
    ax.fill_between(eq_idx, INITIAL_CAPITAL, equity,
                    where=(equity <  INITIAL_CAPITAL), alpha=0.12, color="#e74c3c")
    ax.set_title("Equity Curve", fontsize=9)
    ax.set_ylabel("USD")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))

    ax = axes[1]
    ax.bar(range(len(mon_keys)), mon_vals, color=mon_cols, alpha=0.85, width=0.7)
    ax.axhline(DAILY_LOSS_LIMIT, color="#c0392b", lw=1.0, ls="--",
               label=f"DLL ${DAILY_LOSS_LIMIT:.0f}")
    ax.axhline(0, color="#2c3e50", lw=0.6)
    ax.set_title("Monthly P&L", fontsize=9)
    ax.set_ylabel("USD")
    ax.set_xticks(range(len(mon_keys)))
    ax.set_xticklabels([k[5:] for k in mon_keys], rotation=45, fontsize=7)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, axis="y")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))

    ax = axes[2]
    cols = ["#2ecc71" if w else "#e74c3c" for w in trades["win"]]
    ax.bar(range(len(trades)), trades["r"].values, color=cols, alpha=0.8, width=0.7)
    ax.axhline(0,           color="#2c3e50", lw=0.7)
    ax.axhline(RISK_REWARD, color="#27ae60", lw=0.9, ls="--", label=f"+{RISK_REWARD}R")
    ax.axhline(-1,          color="#c0392b", lw=0.9, ls="--", label="−1R")
    ax.set_title("R-Multiple per Trade", fontsize=9)
    ax.set_ylabel("R")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, axis="y")

    ax = axes[3]
    dd = metrics["drawdowns"]
    ax.fill_between(eq_idx, dd, 0, color="#e74c3c", alpha=0.4)
    ax.plot(eq_idx, dd, color="#c0392b", lw=0.8)
    ax.axhline(0, color="#2c3e50", lw=0.6)
    mi = int(np.argmin(dd))
    ax.annotate(f"{dd[mi]:.2f}%", xy=(mi, dd[mi]),
                xytext=(min(mi + max(len(dd)//15, 2), len(dd)-1), dd[mi]*0.5),
                fontsize=8, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.7))
    ax.set_title("Drawdown (%)", fontsize=9)
    ax.set_ylabel("%")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {output_path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try: plt.show()
        except Exception: pass
    plt.close()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MES/MNQ Liquidity Grab — Topstep Eval Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv",    metavar="FILE")
    src.add_argument("--ticker", metavar="SYM", default=DEFAULT_TICKER)
    p.add_argument("--compare",       action="store_true",
                   help="Run MES + NQ with 3-window consistency analysis")
    p.add_argument("--days",          type=int, default=DEFAULT_DAYS,
                   help=f"Days for single-ticker mode (default: {DEFAULT_DAYS})")
    p.add_argument("--interval",      default="5m")
    p.add_argument("--output",        default="backtest_results.png")
    p.add_argument("--no-plot",       action="store_true")
    p.add_argument("--no-daily-log",  action="store_true")
    p.add_argument("--export-trades", metavar="CSV_OUT")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # ── Compare mode: 3-window consistency test ───────────────────────────────
    if args.compare:
        print(f"\nRunning {N_WINDOWS}-window consistency analysis — {args.interval} bars")
        print(f"  Session: {SESSION_START_ET.strftime('%H:%M')}–"
              f"{SESSION_END_ET.strftime('%H:%M')} ET  |  "
              f"DLL ${abs(DAILY_LOSS_LIMIT):.0f}  |  max {MAX_TRADES_PER_DAY} trades/day\n")

        all_wins: dict[str, list[dict]] = {}
        full_day_logs: dict[str, dict]  = {}

        for key, inst in INSTRUMENTS.items():
            print(f"[ {inst['label']} ]  ${inst['multiplier']}/point")
            full_df, statuses, wins = run_windowed(
                inst["ticker"], inst["multiplier"], inst["contracts"], args.interval
            )
            print_probe_report(statuses, inst["ticker"], args.interval)

            if full_df.empty or not wins:
                print(f"  No data available.\n")
                continue

            print(f"\n  Full block: {_window_date_label(full_df)}")
            print(f"  Split into {N_WINDOWS} sub-windows:\n")
            for i, wr in enumerate(wins):
                m    = wr["metrics"]
                filt = wr["result"]["fvg_filtered"]
                tot  = m["total_trades"] + filt
                if m.get("empty"):
                    flag = f"(no trades  —  {filt} filtered by FVG)" if filt else "(no trades)"
                else:
                    pass_rate = m["total_trades"] / tot * 100 if tot else 0
                    flag = (
                        f"Win% {m['win_rate']:.0f}%  PF {_pf_str(m['profit_factor'])}"
                        f"  |  {m['total_trades']} entries"
                        f"  ({filt} filtered by FVG, {pass_rate:.0f}% pass rate)"
                    )
                print(f"    {wr['label']}: {wr['date_label']}  →  {flag}")

            all_wins[inst["label"]]      = wins
            full_day_logs[inst["label"]] = {
                k: v
                for wr in wins
                for k, v in wr["result"]["day_log"].items()
            }

            if args.export_trades:
                all_trades = pd.concat(
                    [wr["result"]["trades"] for wr in wins
                     if not wr["result"]["trades"].empty],
                    ignore_index=True,
                )
                path = args.export_trades.replace(".csv", f"_{key}.csv")
                all_trades.to_csv(path, index=False)
                print(f"\n  Trades exported → {path}")
            print()

        keys = list(all_wins)
        if len(keys) < 2:
            print("Not enough data for comparison.")
            return

        la, lb = keys[0], keys[1]
        print_window_comparison(all_wins[la], all_wins[lb], args.interval)
        print_monthly_comparison(
            full_day_logs[la], la,
            full_day_logs[lb], lb,
        )
        if not args.no_daily_log:
            print_daily_log(full_day_logs[la], la)
            print_daily_log(full_day_logs[lb], lb)

        if not args.no_plot:
            plot_window_comparison(
                all_wins[la], all_wins[lb],
                full_day_logs[la], full_day_logs[lb],
                args.output,
            )
        return

    # ── Single-instrument mode ────────────────────────────────────────────────
    print(f"\nSingle-instrument mode  ticker={args.ticker}  "
          f"interval={args.interval}  days={args.days}\n")
    if args.csv:
        df = load_from_csv(args.csv)
        print(f"  Loaded {len(df):,} bars from CSV")
    else:
        t = args.ticker.upper()
        mult = 2 if ("MNQ" in t or t == "NQ=F") else 5
        df = load_period(args.ticker, f"{args.days}d", args.interval)
        if df.empty:
            sys.exit(f"No data returned for {args.ticker}.")
    print(f"  Loaded {len(df):,} bars  "
          f"[{df.index[0].strftime('%Y-%m-%d %H:%M')} → "
          f"{df.index[-1].strftime('%Y-%m-%d %H:%M')} ET]\n")

    t      = args.ticker.upper() if not args.csv else "CSV"
    mult   = 2 if ("MNQ" in t or t == "NQ=F") else 5
    res    = run_backtest(df, mult)
    metrics = compute_metrics(res["trades"], res["equity"],
                              res["peak_series"], res["day_log"])
    if metrics.get("empty"):
        print("No trades generated.")
        return

    # quick summary for single mode
    W = 52
    print("═" * W)
    print(f"  {args.ticker} — Topstep Eval Backtest")
    print("═" * W)
    print(f"  Total Trades   : {metrics['total_trades']}")
    print(f"  Win Rate       : {metrics['win_rate']:.1f}%")
    print(f"  Profit Factor  : {_pf_str(metrics['profit_factor'])}")
    print(f"  Avg R          : {metrics['avg_r']:+.3f}R")
    print(f"  Max Drawdown   : {metrics['max_dd_pct']:.2f}%")
    print(f"  Total P&L      : ${metrics['total_pnl']:,.2f}")
    print(f"  DLL Days       : {metrics['dll_count']}")
    print("═" * W + "\n")

    if not args.no_daily_log:
        print_daily_log(res["day_log"], args.ticker)

    if args.export_trades:
        res["trades"].to_csv(args.export_trades, index=False)
        print(f"  Trades exported → {args.export_trades}")

    if not args.no_plot:
        plot_single(args.ticker, res["trades"], res["equity"],
                    res["peak_series"], res["day_log"], metrics, args.output)


if __name__ == "__main__":
    main()
