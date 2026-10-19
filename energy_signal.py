#!/usr/bin/env python3
"""Energy spread signal tool

Downloads WTI/Brent prices, computes spread and z-score, generates signals,
and optionally saves CSV and plots.
"""
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def download_prices(period="2y"):
    wti = yf.download("CL=F", period=period, progress=False)
    brent = yf.download("BZ=F", period=period, progress=False)

    def pick_column(obj, name):
        if isinstance(obj, pd.Series):
            return obj.rename(name)
        if name in obj.columns:
            s = obj[name]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return s.rename(name)
        if hasattr(obj.columns, "nlevels") and obj.columns.nlevels > 1:
            matching = [col for col in obj.columns if col[-1] == name or col[0] == name]
            if matching:
                s = obj[matching[0]]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                return s.rename(name)
        return None

    wti_series = {}
    brent_series = {}
    for name in ["Open", "High", "Low", "Close", "Volume"]:
        wti_col = pick_column(wti, name)
        brent_col = pick_column(brent, name)
        if wti_col is None:
            raise ValueError(f"WTI data is missing required OHLCV column: {name}")
        if brent_col is None:
            raise ValueError(f"Brent data is missing required OHLCV column: {name}")
        wti_series[f"WTI_{name}"] = wti_col.rename(f"WTI_{name}")
        brent_series[f"Brent_{name}"] = brent_col.rename(f"Brent_{name}")

    df = pd.concat([wti_series["WTI_Open"], wti_series["WTI_High"], wti_series["WTI_Low"],
                    wti_series["WTI_Close"], wti_series["WTI_Volume"],
                    brent_series["Brent_Open"], brent_series["Brent_High"],
                    brent_series["Brent_Low"], brent_series["Brent_Close"],
                    brent_series["Brent_Volume"]], axis=1).dropna()
    df["WTI"] = df["WTI_Close"]
    df["Brent"] = df["Brent_Close"]
    return df


def compute_signals(df, window=20):
    df = df.copy()
    df["spread"] = df["WTI"] - df["Brent"]
    # Use only prior observations for rolling statistics to avoid look-ahead
    df["spread_ma"] = df["spread"].shift(1).rolling(window).mean()
    df["spread_std"] = df["spread"].shift(1).rolling(window).std()
    df["spread_zscore"] = (df["spread"] - df["spread_ma"]) / df["spread_std"]

    def generate_signal(z):
        if pd.isna(z):
            return "No data"
        elif z <= -2:
            return "Long WTI / Short Brent"
        elif z >= 2:
            return "Short WTI / Long Brent"
        else:
            return "Neutral"

    df["signal"] = df["spread_zscore"].apply(generate_signal)

    def explain_signal(row):
        z = row.get("spread_zscore")
        spread = row.get("spread")
        ma = row.get("spread_ma")
        std = row.get("spread_std")
        sig = row.get("signal")
        if pd.isna(z):
            return "Not enough data to evaluate."
        base = f"Current spread is ${spread:.2f}."
        ref = f"20-day average spread is ${ma:.2f} with std ${std:.2f}."
        zinfo = f"Z-score is {z:.2f}."
        if sig == "Long WTI / Short Brent":
            action = "Spread is unusually low → WTI may be cheap relative to Brent."
        elif sig == "Short WTI / Long Brent":
            action = "Spread is unusually high → WTI may be expensive relative to Brent."
        else:
            action = "Spread is within normal range → no clear relative-value signal."
        return f"{base} {ref} {zinfo} {action}"

    df["explanation"] = df.apply(explain_signal, axis=1)
    return df


def compute_technical_indicators(df):
    df = df.copy()
    required_price_cols = ["WTI_Close", "WTI_High", "WTI_Low"]
    missing = [col for col in required_price_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required price columns for technical indicators: {missing}")

    close = df["WTI_Close"]
    high = df["WTI_High"]
    low = df["WTI_Low"]

    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    rsi = pd.Series(np.nan, index=df.index, dtype=float)
    period = 14
    if len(df) > period:
        avg_gain = gains.iloc[:period].mean()
        avg_loss = losses.iloc[:period].mean()
        rsi.iloc[period] = 100 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))
        for i in range(period + 1, len(df)):
            avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period
            rsi.iloc[i] = 100 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))
    df["rsi_14"] = rsi

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = pd.Series(np.nan, index=df.index, dtype=float)
    atr_period = 14
    if len(df) > atr_period:
        atr.iloc[atr_period] = tr.iloc[1:atr_period + 1].mean()
        for i in range(atr_period + 1, len(df)):
            atr.iloc[i] = ((atr.iloc[i - 1] * (atr_period - 1)) + tr.iloc[i]) / atr_period
    df["atr_14"] = atr

    middle = close.shift(1).rolling(20).mean()
    std = close.shift(1).rolling(20).std()
    df["bb_middle_20"] = middle
    df["bb_upper_20"] = middle + (2 * std)
    df["bb_lower_20"] = middle - (2 * std)
    return df


def generate_combined_signal(df):
    df = df.copy()
    required_cols = ["spread_zscore", "ema_20", "ema_50", "rsi_14", "macd", "macd_signal", "bb_middle_20", "bb_upper_20", "bb_lower_20", "WTI_Close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for combined signal: {missing}")

    long_scores = []
    short_scores = []
    for _, row in df.iterrows():
        long_score = 0
        short_score = 0

        z = row.get("spread_zscore")
        if pd.notna(z):
            if z <= -1.5:
                long_score += 1
            elif z >= 1.5:
                short_score += 1

        ema_20 = row.get("ema_20")
        ema_50 = row.get("ema_50")
        if pd.notna(ema_20) and pd.notna(ema_50):
            if ema_20 > ema_50:
                long_score += 1
            else:
                short_score += 1

        rsi = row.get("rsi_14")
        if pd.notna(rsi):
            if rsi < 40:
                long_score += 1
            elif rsi > 60:
                short_score += 1

        macd = row.get("macd")
        signal = row.get("macd_signal")
        if pd.notna(macd) and pd.notna(signal):
            if macd > signal:
                long_score += 1
            else:
                short_score += 1

        close = row.get("WTI_Close")
        mid = row.get("bb_middle_20")
        if pd.notna(close) and pd.notna(mid):
            if close > mid:
                long_score += 1
            else:
                short_score += 1

        long_scores.append(long_score)
        short_scores.append(short_score)

    df["technical_long_score"] = long_scores
    df["technical_short_score"] = short_scores

    def determine_signal(long_score, short_score):
        if long_score >= 4 and long_score > short_score:
            return "Long"

        if short_score >= 4 and short_score > long_score:
            return "Short"

        return "Neutral"

    df["combined_signal"] = [determine_signal(ls, ss) for ls, ss in zip(long_scores, short_scores)]
    return df


def validate_data(df):
    df = df.copy()
    required_cols = ["spread", "spread_ma", "spread_std", "spread_zscore", "signal",
                     "WTI_Close", "Brent_Close", "ema_20", "ema_50", "rsi_14", "macd",
                     "macd_signal", "macd_histogram", "atr_14", "bb_middle_20",
                     "bb_upper_20", "bb_lower_20", "technical_long_score",
                     "technical_short_score", "combined_signal"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.index.has_duplicates:
        raise ValueError("Data contains duplicate timestamps.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Data is not chronologically sorted.")

    warmup = 14
    rsi_values = df["rsi_14"].iloc[warmup:].dropna()
    if not rsi_values.empty and ((rsi_values < 0) | (rsi_values > 100)).any():
        raise ValueError("RSI values must be between 0 and 100 after the warm-up period.")

    atr_values = df["atr_14"].iloc[warmup:].dropna()
    if not atr_values.empty and (atr_values <= 0).any():
        raise ValueError("ATR values must be positive after the warm-up period.")

    latest_score_cols = ["technical_long_score", "technical_short_score"]
    latest_missing = [col for col in latest_score_cols if pd.isna(df[col]).iloc[-1]]
    if latest_missing:
        raise ValueError(f"Latest technical scores are not available: {latest_missing}")

    return df


def save_csv(df, out_dir, name="energy_signals"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(path)
    return path


def save_plots(df, out_dir, name_prefix="energy"):
    os.makedirs(out_dir, exist_ok=True)
    p1 = os.path.join(out_dir, f"{name_prefix}_prices.png")
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["WTI"], label="WTI")
    plt.plot(df.index, df["Brent"], label="Brent")
    plt.title("WTI vs Brent Prices")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p1)
    plt.close()

    p2 = os.path.join(out_dir, f"{name_prefix}_spread.png")
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["spread"], label="Spread (WTI - Brent)")
    plt.plot(df.index, df["spread_ma"], label=f"{df['spread_ma'].rolling(1).mean().name} 20-day MA", color="orange")
    plt.title("WTI-Brent Spread")
    plt.xlabel("Date")
    plt.ylabel("Spread ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p2)
    plt.close()

    # Z-score plot with horizontal lines at +2, 0, -2
    p3 = os.path.join(out_dir, f"{name_prefix}_zscore.png")
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["spread_zscore"], label="Spread Z-score")
    plt.axhline(2, color="red", linestyle="--", label="+2")
    plt.axhline(0, color="gray", linestyle="-", label="0")
    plt.axhline(-2, color="green", linestyle="--", label="-2")
    plt.title("Spread Z-score")
    plt.xlabel("Date")
    plt.ylabel("Z-score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p3)
    plt.close()

    # Z-score with long/short markers
    p4 = os.path.join(out_dir, f"{name_prefix}_zscore_signals.png")
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["spread_zscore"], label="Spread Z-score")
    longs = df[df["signal"] == "Long WTI / Short Brent"]
    shorts = df[df["signal"] == "Short WTI / Long Brent"]
    if not longs.empty:
        plt.scatter(longs.index, longs["spread_zscore"], marker="^", color="green", label="Long signal")
    if not shorts.empty:
        plt.scatter(shorts.index, shorts["spread_zscore"], marker="v", color="red", label="Short signal")
    plt.axhline(2, color="red", linestyle="--", label="+2")
    plt.axhline(0, color="gray", linestyle="-", label="0")
    plt.axhline(-2, color="green", linestyle="--", label="-2")
    plt.title("Z-score with Signal Markers")
    plt.xlabel("Date")
    plt.ylabel("Z-score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p4)
    plt.close()

    return p1, p2, p3, p4


def save_last_5_days_summary(df: pd.DataFrame, out_dir: str = ".") -> pd.DataFrame:
    """Create a small zoomed-in last-5-days analysis without altering the main 2-year view."""
    last_5 = df.copy()
    if not isinstance(last_5.index, pd.DatetimeIndex):
        if "Date" in last_5.columns:
            last_5 = last_5.set_index(pd.to_datetime(last_5["Date"]))
        elif "date" in last_5.columns:
            last_5 = last_5.set_index(pd.to_datetime(last_5["date"]))
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a Date/date column to build last_5 days.")
    last_5 = last_5.last("5D").copy()

    if not last_5.empty:
        last_5_display = last_5[["WTI", "Brent", "spread", "spread_zscore", "signal"]].copy()
    else:
        last_5_display = pd.DataFrame(columns=["WTI", "Brent", "spread", "spread_zscore", "signal"])

    print("\nLAST 5 DAYS SUMMARY")
    print(last_5_display.to_string())

    avg_wti = float(last_5["WTI"].mean()) if "WTI" in last_5.columns and not last_5.empty else float("nan")
    avg_spread = float(last_5["spread"].mean()) if "spread" in last_5.columns and not last_5.empty else float("nan")
    high_wti = float(last_5["WTI"].max()) if "WTI" in last_5.columns and not last_5.empty else float("nan")
    low_wti = float(last_5["WTI"].min()) if "WTI" in last_5.columns and not last_5.empty else float("nan")
    high_z = float(last_5["spread_zscore"].max()) if "spread_zscore" in last_5.columns and not last_5.empty else float("nan")
    low_z = float(last_5["spread_zscore"].min()) if "spread_zscore" in last_5.columns and not last_5.empty else float("nan")

    long_count = int((last_5["signal"] == "Long WTI / Short Brent").sum()) if "signal" in last_5.columns and not last_5.empty else 0
    short_count = int((last_5["signal"] == "Short WTI / Long Brent").sum()) if "signal" in last_5.columns and not last_5.empty else 0
    neutral_count = int((last_5["signal"] == "Neutral").sum()) if "signal" in last_5.columns and not last_5.empty else 0

    print(f"Average WTI over last 5 days: {avg_wti}")
    print(f"Average spread over last 5 days: {avg_spread}")
    print(f"Highest WTI over last 5 days: {high_wti}")
    print(f"Lowest WTI over last 5 days: {low_wti}")
    print(f"Highest z-score over last 5 days: {high_z}")
    print(f"Lowest z-score over last 5 days: {low_z}")
    print(f"Long WTI / Short Brent days: {long_count}")
    print(f"Short WTI / Long Brent days: {short_count}")
    print(f"Neutral days: {neutral_count}")

    os.makedirs(out_dir, exist_ok=True)

    wti_path = os.path.join(out_dir, "wti_last5.png")
    plt.figure(figsize=(8, 4))
    plt.plot(last_5.index, last_5["WTI"], marker="o", linewidth=1.8)
    plt.title("WTI Last 5 Days")
    plt.xlabel("Date")
    plt.ylabel("WTI Price ($)")
    plt.tight_layout()
    plt.savefig(wti_path)
    plt.close()

    zscore_path = os.path.join(out_dir, "zscore_last5.png")
    plt.figure(figsize=(8, 4))
    plt.plot(last_5.index, last_5["spread_zscore"], marker="o", linewidth=1.8, color="tab:purple")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    plt.axhline(2, color="red", linestyle="--", linewidth=1.0)
    plt.axhline(-2, color="green", linestyle="--", linewidth=1.0)
    plt.title("Spread Z-score Last 5 Days")
    plt.xlabel("Date")
    plt.ylabel("Z-score")
    plt.tight_layout()
    plt.savefig(zscore_path)
    plt.close()

    return last_5


def run_backtest(df):
    # Backtest with explicit entry/exit rules and no look-ahead.
    # Rules:
    # - Enter Long when z <= -2
    # - Enter Short when z >= +2
    # - Exit when z returns to 0 (z >= 0 for Long, z <= 0 for Short)
    # - Exit immediately if opposite signal appears
    bt = df.copy()
    bt["spread_change"] = bt["spread"].diff()

    positions = []  # position applied for each day (used to compute pnl)
    current_pos = 0
    open_trade = None
    trades = []

    # Iterate rows in chronological order
    for idx, row in bt.iterrows():
        # position applied during this day is the current_pos (from previous signals)
        positions.append(current_pos)

        # Determine next position based on today's signal and z-score (no future info)
        sig = row.get("signal")
        z = row.get("spread_zscore")
        next_pos = current_pos

        if current_pos == 0:
            # no open position: check for entry signals
            if sig == "Long WTI / Short Brent":
                next_pos = 1
                open_trade = {"entry_date": idx, "entry_pos": 1}
            elif sig == "Short WTI / Long Brent":
                next_pos = -1
                open_trade = {"entry_date": idx, "entry_pos": -1}
        elif current_pos == 1:
            # long open: check for exit or flip to short
            if sig == "Short WTI / Long Brent":
                # close long and open short
                if open_trade is not None:
                    open_trade["exit_date"] = idx
                    trades.append(open_trade)
                next_pos = -1
                open_trade = {"entry_date": idx, "entry_pos": -1}
            else:
                # exit when z returns to 0 or above
                if pd.notna(z) and z >= 0:
                    if open_trade is not None:
                        open_trade["exit_date"] = idx
                        trades.append(open_trade)
                        open_trade = None
                    next_pos = 0
        elif current_pos == -1:
            # short open: check for exit or flip to long
            if sig == "Long WTI / Short Brent":
                if open_trade is not None:
                    open_trade["exit_date"] = idx
                    trades.append(open_trade)
                next_pos = 1
                open_trade = {"entry_date": idx, "entry_pos": 1}
            else:
                # exit when z returns to 0 or below
                if pd.notna(z) and z <= 0:
                    if open_trade is not None:
                        open_trade["exit_date"] = idx
                        trades.append(open_trade)
                        open_trade = None
                    next_pos = 0

        current_pos = next_pos

    bt["position"] = positions
    # PnL uses prior-day position on today's spread change (position taken at close, applied next day)
    bt["pnl"] = bt["position"].shift(1) * bt["spread_change"]
    bt["pnl"] = bt["pnl"].fillna(0)
    bt["cum_pnl"] = bt["pnl"].cumsum()

    # Close any open trade at the last date
    if open_trade is not None:
        open_trade["exit_date"] = bt.index[-1]
        trades.append(open_trade)

    perf = {
        "total_pnl": float(bt["pnl"].sum()),
        "sharpe": float(bt["pnl"].mean() / (bt["pnl"].std() if bt["pnl"].std() != 0 else 1) * np.sqrt(252)),
        "trades": trades,
    }
    return bt, perf


def period_performance(bt: pd.DataFrame) -> Dict[str, Any]:
    """Compute performance metrics from backtest dataframe for the given period."""
    res: Dict[str, Any] = {}
    cum = bt["cum_pnl"] if "cum_pnl" in bt.columns else bt["pnl"].cumsum()
    res["cumulative_pnl"] = float(cum.iloc[-1])
    # trades: detect entry rows where position != prev and position != 0
    pos = bt["position"]
    trade_entries = ((pos != pos.shift(1)) & (pos != 0))
    res["num_trades"] = int(trade_entries.sum())
    # average pnl per trade: sum realized pnl divided by num trades (avoid div0)
    realized = bt["pnl"][bt["pnl"].notna()]
    res["avg_pnl_per_trade"] = float(realized.sum() / res["num_trades"]) if res["num_trades"] > 0 else 0.0
    # win rate: fraction of realized trades with pnl>0
    wins = (realized > 0).sum()
    res["win_rate"] = float(wins / len(realized)) if len(realized) > 0 else 0.0
    # max drawdown
    running_max = cum.cummax()
    drawdowns = (cum - running_max)
    res["max_drawdown"] = float(drawdowns.min())
    # simplified Sharpe: mean(daily pnl) / std(daily pnl) * sqrt(252)
    daily = bt["pnl"].fillna(0)
    mean = daily.mean()
    std = daily.std()
    sharpe = float((mean / std) * (252 ** 0.5)) if std > 0 else 0.0
    res["sharpe"] = sharpe
    return res


def run_oos(df: pd.DataFrame, out_dir: Optional[Path] = None, name_prefix: str = "energy") -> Dict[str, Any]:
    """Run development / out-of-sample split and report metrics. Returns dict with dev/test results and paths."""
    n = len(df)
    split = int(n * 0.8)
    dev_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    # Run backtest on development and test separately, but do NOT re-fit thresholds.
    dev_bt, dev_perf = run_backtest(dev_df)
    test_bt, test_perf = run_backtest(test_df)

    dev_metrics = period_performance(dev_bt)
    test_metrics = period_performance(test_bt)

    o = {}
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{name_prefix}_oos_cum_pnl.png"
        plt.figure(figsize=(10, 4))
        cum = test_bt["cum_pnl"] if "cum_pnl" in test_bt.columns else test_bt["pnl"].cumsum()
        plt.plot(cum.index, cum.values)
        plt.title("Out-of-sample cumulative PnL")
        plt.xlabel("Date")
        plt.ylabel("Cumulative PnL")
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        o["oos_plot"] = str(p)

        # also save CSVs
        dev_bt.to_csv(out_dir / f"{name_prefix}_dev_backtest.csv")
        test_bt.to_csv(out_dir / f"{name_prefix}_oos_backtest.csv")

    return {"dev": dev_metrics, "test": test_metrics, "files": o}


def _is_long_signal(signal_value: Any) -> bool:
    if pd.isna(signal_value):
        return False
    signal_text = str(signal_value).strip()
    return signal_text in {"Long", "Long WTI / Short Brent"}


def _is_short_signal(signal_value: Any) -> bool:
    if pd.isna(signal_value):
        return False
    signal_text = str(signal_value).strip()
    return signal_text in {"Short", "Short WTI / Long Brent"}


def run_combined_backtest(df: pd.DataFrame, transaction_cost: float = 0.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run a separate backtest using combined_signal with zero-cost default assumption."""
    if "combined_signal" not in df.columns:
        raise ValueError("combined_signal column is required for combined backtest")

    bt = df.copy()
    bt["spread_change"] = bt["spread"].diff()

    positions = []
    current_pos = 0
    open_trade = None
    trades = []

    for idx, row in bt.iterrows():
        positions.append(current_pos)

        signal_value = row.get("combined_signal")
        spread_change = row.get("spread_change")
        daily_pnl = float(current_pos * spread_change) if pd.notna(spread_change) else 0.0

        if open_trade is not None:
            open_trade["pnl"] += daily_pnl
            open_trade["holding_period_bars"] += 1

        next_pos = current_pos

        if current_pos == 0:
            if _is_long_signal(signal_value):
                next_pos = 1
                open_trade = {
                    "entry_date": idx,
                    "entry_pos": 1,
                    "entry_bar": len(positions) - 1,
                    "pnl": -transaction_cost if transaction_cost != 0 else 0.0,
                    "holding_period_bars": 0,
                    "forced_exit": False,
                }
            elif _is_short_signal(signal_value):
                next_pos = -1
                open_trade = {
                    "entry_date": idx,
                    "entry_pos": -1,
                    "entry_bar": len(positions) - 1,
                    "pnl": -transaction_cost if transaction_cost != 0 else 0.0,
                    "holding_period_bars": 0,
                    "forced_exit": False,
                }
        elif current_pos == 1:
            if _is_short_signal(signal_value):
                if open_trade is not None:
                    open_trade["exit_date"] = idx
                    open_trade["exit_bar"] = len(positions) - 1
                    open_trade["exit_pos"] = 0
                    open_trade["pnl"] -= transaction_cost if transaction_cost != 0 else 0.0
                    open_trade["forced_exit"] = False
                    trades.append(open_trade)
                next_pos = -1
                open_trade = {
                    "entry_date": idx,
                    "entry_pos": -1,
                    "entry_bar": len(positions) - 1,
                    "pnl": -transaction_cost if transaction_cost != 0 else 0.0,
                    "holding_period_bars": 0,
                    "forced_exit": False,
                }
            elif _is_long_signal(signal_value):
                pass
            else:
                if open_trade is not None:
                    open_trade["exit_date"] = idx
                    open_trade["exit_bar"] = len(positions) - 1
                    open_trade["exit_pos"] = 0
                    open_trade["pnl"] -= transaction_cost if transaction_cost != 0 else 0.0
                    open_trade["forced_exit"] = False
                    trades.append(open_trade)
                    open_trade = None
                next_pos = 0
        elif current_pos == -1:
            if _is_long_signal(signal_value):
                if open_trade is not None:
                    open_trade["exit_date"] = idx
                    open_trade["exit_bar"] = len(positions) - 1
                    open_trade["exit_pos"] = 0
                    open_trade["pnl"] -= transaction_cost if transaction_cost != 0 else 0.0
                    open_trade["forced_exit"] = False
                    trades.append(open_trade)
                next_pos = 1
                open_trade = {
                    "entry_date": idx,
                    "entry_pos": 1,
                    "entry_bar": len(positions) - 1,
                    "pnl": -transaction_cost if transaction_cost != 0 else 0.0,
                    "holding_period_bars": 0,
                    "forced_exit": False,
                }
            elif _is_short_signal(signal_value):
                pass
            else:
                if open_trade is not None:
                    open_trade["exit_date"] = idx
                    open_trade["exit_bar"] = len(positions) - 1
                    open_trade["exit_pos"] = 0
                    open_trade["pnl"] -= transaction_cost if transaction_cost != 0 else 0.0
                    open_trade["forced_exit"] = False
                    trades.append(open_trade)
                    open_trade = None
                next_pos = 0

        current_pos = next_pos

    bt["position"] = positions
    bt["pnl"] = bt["position"].shift(1) * bt["spread_change"]
    bt["pnl"] = bt["pnl"].fillna(0)
    bt["cum_pnl"] = bt["pnl"].cumsum()

    if open_trade is not None:
        open_trade["exit_date"] = bt.index[-1]
        open_trade["exit_bar"] = len(bt) - 1
        open_trade["exit_pos"] = 0
        open_trade["pnl"] -= transaction_cost if transaction_cost != 0 else 0.0
        open_trade["forced_exit"] = True
        trades.append(open_trade)

    trade_metrics = calculate_trade_level_metrics(bt, trades)

    perf = {
        "total_pnl": float(bt["pnl"].sum()),
        "sharpe": float(bt["pnl"].mean() / (bt["pnl"].std() if bt["pnl"].std() != 0 else 1) * np.sqrt(252)),
        "trades": trades,
        "trade_metrics": trade_metrics,
    }
    return bt, perf


def calculate_trade_level_metrics(bt: pd.DataFrame, trades: list) -> Dict[str, Any]:
    """Compute trade-level metrics from completed trades (excluding forced exits from win-rate stats)."""
    completed_trades = [trade for trade in trades if not trade.get("forced_exit", False)]
    if not completed_trades:
        completed_trade_pnls = []
    else:
        completed_trade_pnls = [trade.get("pnl", 0.0) for trade in completed_trades]

    total_pnl = float(sum(trade.get("pnl", 0.0) for trade in completed_trades))
    winning_trades = [trade for trade in completed_trades if trade.get("pnl", 0.0) > 0]
    losing_trades = [trade for trade in completed_trades if trade.get("pnl", 0.0) < 0]
    num_trades = len(completed_trades)

    win_rate = float(len(winning_trades) / num_trades) if num_trades > 0 else 0.0
    avg_pnl = float(total_pnl / num_trades) if num_trades > 0 else 0.0
    median_pnl = float(np.median(completed_trade_pnls)) if num_trades > 0 else 0.0
    largest_winning_trade = float(max([trade.get("pnl", 0.0) for trade in winning_trades])) if winning_trades else 0.0
    largest_losing_trade = float(min([trade.get("pnl", 0.0) for trade in losing_trades])) if losing_trades else 0.0

    positive_pnl = sum(max(trade.get("pnl", 0.0), 0.0) for trade in completed_trades)
    negative_pnl = abs(sum(min(trade.get("pnl", 0.0), 0.0) for trade in completed_trades))
    profit_factor = float(positive_pnl / negative_pnl) if negative_pnl > 0 else (float("inf") if positive_pnl > 0 else 0.0)

    cum_pnl = bt["cum_pnl"] if "cum_pnl" in bt.columns else bt["pnl"].cumsum()
    running_max = cum_pnl.cummax()
    drawdowns = cum_pnl - running_max
    max_drawdown = float(drawdowns.min()) if not drawdowns.empty else 0.0

    daily = bt["pnl"].fillna(0)
    mean = daily.mean()
    std = daily.std()
    sharpe = float((mean / std) * np.sqrt(252)) if std > 0 else 0.0

    holding_periods = [trade.get("holding_period_bars", 0) for trade in completed_trades]
    avg_holding_period = float(np.mean(holding_periods)) if holding_periods else 0.0
    long_trades = sum(1 for trade in completed_trades if trade.get("entry_pos", 0) > 0)
    short_trades = sum(1 for trade in completed_trades if trade.get("entry_pos", 0) < 0)

    return {
        "num_completed_trades": num_trades,
        "num_winning_trades": len(winning_trades),
        "num_losing_trades": len(losing_trades),
        "trade_level_win_rate": win_rate,
        "total_pnl": total_pnl,
        "average_pnl_per_trade": avg_pnl,
        "median_pnl_per_trade": median_pnl,
        "largest_winning_trade": largest_winning_trade,
        "largest_losing_trade": largest_losing_trade,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "average_holding_period_bars": avg_holding_period,
        "long_trade_count": long_trades,
        "short_trade_count": short_trades,
        "forced_exit_count": sum(1 for trade in trades if trade.get("forced_exit", False)),
    }


def run_combined_oos(df: pd.DataFrame, out_dir: Optional[Path] = None, name_prefix: str = "energy", transaction_cost: float = 0.0) -> Dict[str, Any]:
    """Run development / out-of-sample split for the combined strategy using identical rules in both periods."""
    n = len(df)
    split = int(n * 0.8)
    dev_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    dev_bt, dev_perf = run_combined_backtest(dev_df, transaction_cost=transaction_cost)
    test_bt, test_perf = run_combined_backtest(test_df, transaction_cost=transaction_cost)

    dev_metrics = calculate_trade_level_metrics(dev_bt, dev_perf.get("trades", []))
    test_metrics = calculate_trade_level_metrics(test_bt, test_perf.get("trades", []))

    o = {}
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{name_prefix}_combined_oos_cum_pnl.png"
        plt.figure(figsize=(10, 4))
        cum = test_bt["cum_pnl"] if "cum_pnl" in test_bt.columns else test_bt["pnl"].cumsum()
        plt.plot(cum.index, cum.values)
        plt.title("Combined strategy out-of-sample cumulative PnL")
        plt.xlabel("Date")
        plt.ylabel("Cumulative PnL")
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        o["oos_plot"] = str(p)

        dev_bt.to_csv(out_dir / f"{name_prefix}_combined_dev_backtest.csv")
        test_bt.to_csv(out_dir / f"{name_prefix}_oos_combined_backtest.csv")
        pd.DataFrame([dev_metrics]).to_csv(out_dir / f"{name_prefix}_combined_dev_metrics.csv", index=False)
        pd.DataFrame([test_metrics]).to_csv(out_dir / f"{name_prefix}_combined_test_metrics.csv", index=False)

    return {"dev": dev_metrics, "test": test_metrics, "files": o}


def parse_args():
    p = argparse.ArgumentParser(description="Compute energy spread signals (WTI vs Brent)")
    p.add_argument("--period", default="2y", help="yfinance period (e.g. 1y, 2y, 5y)")
    p.add_argument("--window", type=int, default=20, help="rolling window for MA/std")
    p.add_argument("--out", default="./energy_output", help="output directory for CSV and plots")
    p.add_argument("--save-csv", action="store_true", help="save CSV of results")
    p.add_argument("--save-plots", action="store_true", help="save PNG plots")
    p.add_argument("--backtest", action="store_true", help="run a simple backtest on the generated signals")
    p.add_argument("--combined-backtest", action="store_true", help="run a separate backtest using the combined_signal column")
    p.add_argument("--oos", action="store_true", help="run out-of-sample split (80/20) and report dev/test metrics")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Downloading prices for period={args.period}...")
    df = download_prices(period=args.period)
    print(f"Downloaded {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")
    df = compute_signals(df, window=args.window)
    df = compute_technical_indicators(df)
    df = generate_combined_signal(df)
    df = validate_data(df)
    print(df[["spread", "spread_zscore", "signal"]].tail(5).to_string())

    # Print latest statistics
    latest = df.iloc[-1]
    latest_wti = float(latest["WTI_Close"]) if not pd.isna(latest["WTI_Close"]) else None
    latest_brent = float(latest["Brent_Close"]) if not pd.isna(latest["Brent_Close"]) else None
    latest_spread = float(latest["spread"]) if not pd.isna(latest["spread"]) else None
    latest_ma = float(latest["spread_ma"]) if not pd.isna(latest["spread_ma"]) else None
    latest_std = float(latest["spread_std"]) if not pd.isna(latest["spread_std"]) else None
    latest_z = float(latest["spread_zscore"]) if not pd.isna(latest["spread_zscore"]) else None
    latest_sig = latest["signal"]
    latest_ema20 = float(latest["ema_20"]) if not pd.isna(latest["ema_20"]) else None
    latest_ema50 = float(latest["ema_50"]) if not pd.isna(latest["ema_50"]) else None
    latest_rsi = float(latest["rsi_14"]) if not pd.isna(latest["rsi_14"]) else None
    latest_macd_hist = float(latest["macd_histogram"]) if not pd.isna(latest["macd_histogram"]) else None
    latest_atr = float(latest["atr_14"]) if not pd.isna(latest["atr_14"]) else None
    latest_long_score = float(latest["technical_long_score"]) if not pd.isna(latest["technical_long_score"]) else None
    latest_short_score = float(latest["technical_short_score"]) if not pd.isna(latest["technical_short_score"]) else None
    latest_combined_signal = latest["combined_signal"]

    longs = int((df["signal"] == "Long WTI / Short Brent").sum())
    shorts = int((df["signal"] == "Short WTI / Long Brent").sum())
    neutrals = int((df["signal"] == "Neutral").sum())

    print(f"Latest WTI: {latest_wti}")
    print(f"Latest Brent: {latest_brent}")
    print(f"Latest spread: {latest_spread}")
    print(f"Latest spread 20-day MA: {latest_ma}")
    print(f"Latest spread std: {latest_std}")
    print(f"Latest z-score: {latest_z}")
    print(f"Latest signal: {latest_sig}")
    print(f"Latest EMA 20: {latest_ema20}")
    print(f"Latest EMA 50: {latest_ema50}")
    print(f"Latest RSI: {latest_rsi}")
    print(f"Latest MACD histogram: {latest_macd_hist}")
    print(f"Latest ATR: {latest_atr}")
    print(f"Latest technical long score: {latest_long_score}")
    print(f"Latest technical short score: {latest_short_score}")
    print(f"Latest combined signal: {latest_combined_signal}")
    print(f"Count long signals: {longs}")
    print(f"Count short signals: {shorts}")
    print(f"Count neutral observations: {neutrals}")

    save_last_5_days_summary(df, out_dir=args.out)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.save_csv:
        csv_path = save_csv(df, args.out, name=f"energy_signals_{timestamp}")
        print(f"Saved CSV -> {csv_path}")
    if args.save_plots:
        p1, p2, p3, p4 = save_plots(df, args.out, name_prefix=f"energy_{timestamp}")
        print(f"Saved plots -> {p1}, {p2}, {p3}, {p4}")
    if args.backtest:
        bt, perf = run_backtest(df)
        bt_path = save_csv(bt, args.out, name=f"energy_backtest_{timestamp}")
        print(f"Saved backtest CSV -> {bt_path}")
        print(f"Backtest perf: total_pnl={perf['total_pnl']:.4f}, sharpe={perf['sharpe']:.3f}")
        # simple backtest plot
        pbt = os.path.join(args.out, f"backtest_cum_pnl_{timestamp}.png")
        plt.figure(figsize=(10, 4))
        plt.plot(bt.index, bt["cum_pnl"], label="Cumulative PnL")
        plt.title("Simple Spread Strategy Cumulative PnL")
        plt.xlabel("Date")
        plt.ylabel("Cumulative PnL")
        plt.legend()
        plt.tight_layout()
        plt.savefig(pbt)
        plt.close()
        print(f"Saved backtest plot -> {pbt}")
    if args.combined_backtest:
        bt_combined, perf_combined = run_combined_backtest(df)
        bt_path = save_csv(bt_combined, args.out, name=f"energy_combined_backtest_{timestamp}")
        print(f"Saved combined backtest CSV -> {bt_path}")
        trade_metrics = perf_combined.get("trade_metrics", {})
        print("Combined strategy trade-level summary:")
        print(f"- Completed trades: {trade_metrics.get('num_completed_trades')}")
        print(f"- Win rate: {trade_metrics.get('trade_level_win_rate'):.3%}")
        print(f"- Total PnL: {trade_metrics.get('total_pnl'):.4f}")
        print(f"- Profit factor: {trade_metrics.get('profit_factor')}")
        print(f"- Max drawdown: {trade_metrics.get('max_drawdown'):.4f}")
        print(f"- Sharpe ratio: {trade_metrics.get('sharpe_ratio'):.4f}")
        print(f"- Average holding period (bars): {trade_metrics.get('average_holding_period_bars'):.2f}")
        pbt = os.path.join(args.out, f"combined_backtest_cum_pnl_{timestamp}.png")
        plt.figure(figsize=(10, 4))
        plt.plot(bt_combined.index, bt_combined["cum_pnl"], label="Cumulative PnL")
        plt.title("Combined Strategy Cumulative PnL")
        plt.xlabel("Date")
        plt.ylabel("Cumulative PnL")
        plt.legend()
        plt.tight_layout()
        plt.savefig(pbt)
        plt.close()
        print(f"Saved combined backtest plot -> {pbt}")
        metrics_path = save_csv(pd.DataFrame([trade_metrics]), args.out, name=f"energy_combined_trade_metrics_{timestamp}")
        print(f"Saved combined trade metrics -> {metrics_path}")
    if args.oos:
        outp = Path(args.out)
        oos = run_oos(df, out_dir=outp, name_prefix=f"energy_{timestamp}")
        dev = oos.get("dev", {})
        test = oos.get("test", {})
        print("\nSPREAD-ONLY STRATEGY")
        print("Development metrics:")
        print(f"- Number of trades: {dev.get('num_trades')}")
        print(f"- Cumulative PnL: {dev.get('cumulative_pnl'):.6f}")
        print(f"- Average PnL per trade: {dev.get('avg_pnl_per_trade'):.6f}")
        print(f"- Win rate: {dev.get('win_rate'):.3%}")
        print(f"- Maximum drawdown: {dev.get('max_drawdown'):.6f}")
        print(f"- Simplified Sharpe ratio: {dev.get('sharpe'):.4f}")

        print("\nTest metrics:")
        print(f"- Number of trades: {test.get('num_trades')}")
        print(f"- Cumulative PnL: {test.get('cumulative_pnl'):.6f}")
        print(f"- Average PnL per trade: {test.get('avg_pnl_per_trade'):.6f}")
        print(f"- Win rate: {test.get('win_rate'):.3%}")
        print(f"- Maximum drawdown: {test.get('max_drawdown'):.6f}")
        print(f"- Simplified Sharpe ratio: {test.get('sharpe'):.4f}")
        if oos.get('files') and oos['files'].get('oos_plot'):
            print(f"Saved out-of-sample cumulative PnL plot -> {oos['files']['oos_plot']}")

        combined_oos = run_combined_oos(df, out_dir=outp, name_prefix=f"energy_{timestamp}")
        dev_combined = combined_oos.get("dev", {})
        test_combined = combined_oos.get("test", {})
        print("\nCOMBINED TECHNICAL STRATEGY")
        print("Development metrics:")
        print(f"- Number of completed trades: {dev_combined.get('num_completed_trades')}")
        print(f"- Total PnL: {dev_combined.get('total_pnl'):.6f}")
        print(f"- Trade-level win rate: {dev_combined.get('trade_level_win_rate'):.3%}")
        print(f"- Profit factor: {dev_combined.get('profit_factor')}")
        print(f"- Maximum drawdown: {dev_combined.get('max_drawdown'):.6f}")
        print(f"- Sharpe ratio: {dev_combined.get('sharpe_ratio'):.4f}")
        print(f"- Average holding period (bars): {dev_combined.get('average_holding_period_bars'):.2f}")

        print("\nTest metrics:")
        print(f"- Number of completed trades: {test_combined.get('num_completed_trades')}")
        print(f"- Total PnL: {test_combined.get('total_pnl'):.6f}")
        print(f"- Trade-level win rate: {test_combined.get('trade_level_win_rate'):.3%}")
        print(f"- Profit factor: {test_combined.get('profit_factor')}")
        print(f"- Maximum drawdown: {test_combined.get('max_drawdown'):.6f}")
        print(f"- Sharpe ratio: {test_combined.get('sharpe_ratio'):.4f}")
        print(f"- Average holding period (bars): {test_combined.get('average_holding_period_bars'):.2f}")
        if combined_oos.get('files') and combined_oos['files'].get('oos_plot'):
            print(f"Saved combined out-of-sample cumulative PnL plot -> {combined_oos['files']['oos_plot']}")


if __name__ == "__main__":
    main()
