Energy Signal Project

A beginner-friendly experiment that compares two major crude-oil benchmarks (WTI and Brent), computes their price difference (the spread), and looks for unusually large differences using z-scores. The project produces CSVs and charts, and includes a simple backtest to illustrate how a spread-based strategy might have behaved historically.

What are WTI and Brent?
- WTI (West Texas Intermediate) and Brent are two commonly quoted global crude-oil benchmark prices. They represent prices for oil from different regions and are used as reference prices in markets and contracts.

What does the spread mean?
- The spread = WTI − Brent (price difference per barrel). If the spread is negative, WTI is cheaper than Brent; if positive, WTI is more expensive. Changes in the spread can reflect regional supply/demand differences, transportation costs, or market events.

What is a z-score (here)?
- A z-score measures how many standard deviations the current spread is from its recent average. We compute a 20-day rolling mean and standard deviation (using only past data) and then:

  z = (current spread − recent mean) / recent std

- A z-score of −2 means the spread is two standard deviations below its recent average (unusually low). A z-score of +2 means the spread is two standard deviations above its recent average (unusually high).

Signal rules used by the script
- If z ≤ −2 → signal = "Long WTI / Short Brent" (expect WTI to rise relative to Brent)
- If z ≥ +2 → signal = "Short WTI / Long Brent" (expect WTI to fall relative to Brent)
- Otherwise → "Neutral"

Backtest entry/exit rules (implemented)
- Enter Long when z ≤ −2. Enter Short when z ≥ +2.
- Exit when z returns to 0 (Long exits when z ≥ 0; Short exits when z ≤ 0).
- If the opposite-entry signal occurs before hitting 0, exit and flip immediately.
- All decisions use only information available up to that date (no look-ahead).

Data source
- Price data are downloaded from Yahoo Finance (`yfinance`), using tickers `CL=F` (WTI) and `BZ=F` (Brent). The script extracts daily Close prices and aligns them.

Assumptions in the backtest
- Unit notional exposure (one unit of spread), no transaction costs, no slippage, trades executed at close, and positions applied from the next day after a signal.

Limitations and cautions
- This is a simplified educational example: real trading requires careful execution, transaction-cost modeling, risk management, and regulatory/operational considerations.
- The backtest ignores liquidity, fees, margin/leverage, and market impact.
- The z-score thresholds (±2) and window (20 days) are illustrative and not optimized.

Warning
- This project is for educational research only and is NOT financial advice. Do not trade with real money based solely on this code.

Quick start

```bash
python3 -m pip install --user -r requirements.txt
python3 energy_signal.py --save-csv --save-plots --backtest --out /tmp/energy_out
```

Files created by the script
- `energy_signal.py`: main script
- `wti_brent_data.csv`: downloaded and processed time series
- `energy_prices.png`, `energy_spread.png`, `energy_zscore.png`, `energy_zscore_signals.png`: charts

## Project contents (copied to this folder)

- `energy_signal.py` — main CLI script to download data, compute signals, run backtests, and save CSV/plots.
- `wti_brent_data.csv` — the downloaded WTI/Brent price series used by the script.
- `energy_signals_*.csv` — timestamped signal CSVs (one per run).
- `energy_backtest_*.csv` — timestamped backtest CSVs (one per run).
- `*_prices.png`, `*_spread.png`, `*_zscore.png`, `*_zscore_signals.png` — timestamped plot images produced by runs.
- `*_oos_cum_pnl.png` — out-of-sample cumulative PnL plot (if `--oos` was used).

To create a downloadable archive of this folder, run:

```bash
zip -r ~/energy_signal_project.zip ~/energy_signal_project
```

The zip file will be created at `~/energy_signal_project.zip`.
