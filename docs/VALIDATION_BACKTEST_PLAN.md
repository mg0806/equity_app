# Validation And Backtesting Plan

## Why This Matters

Analyst firms will not trust a signal engine only because it looks good. They need evidence that signals are stable, explainable and historically useful.

## Required Backtests

### 1. Point-In-Time Signal Backtest

For each ticker and historical date:

1. Reconstruct data available at that date.
2. Generate BUY/HOLD/SELL signal.
3. Measure forward returns over 1M, 3M, 6M and 12M.
4. Compare returns against Nifty and sector index.

Metrics:

- Hit rate.
- Median forward return.
- Excess return vs benchmark.
- Max drawdown after signal.
- False positive rate for BUY.
- False negative rate for SELL.

### 2. Factor Validation

Validate each model component:

- Quality score vs future ROE/ROCE.
- Red flags vs future drawdown.
- DCF upside vs future return.
- Monte Carlo undervaluation probability vs future return.
- Peer rank vs future relative performance.

### 3. Sector Validation

Backtest separately for:

- Banks/NBFCs.
- IT services.
- FMCG.
- Metals/commodities.
- Exchanges/capital-market infrastructure.
- Industrials.

### 4. Stability Tests

Check whether small assumption changes produce unstable signals.

- Growth +/- 2%.
- WACC +/- 1%.
- Terminal growth +/- 0.5%.
- Margin +/- 1%.

## Minimum Dataset For Credible Pilot

- 100-200 NSE companies.
- 5-10 years of annual snapshots.
- At least 4 forward-return windows.
- Survivorship-bias controls.

## Output To Show Buyers

| Metric | Target For Sales Demo |
|---|---|
| BUY hit rate | Above 55-60% over 12M |
| Median BUY excess return | Positive vs benchmark |
| SELL avoided drawdown | Meaningful downside avoidance |
| Signal stability | Low churn quarter-to-quarter |
| Explainability | Every signal has top drivers |

## Current Status

The app now includes a historical validation engine that replays rolling signal states and measures 1M, 3M, 6M and 12M forward returns, hit rates, excess returns, drawdown, sector breakdowns, market-cap buckets and market-regime performance.

The app also saves every generated report as a model-versioned SQLite snapshot. These snapshots preserve the data, ratios, assumptions, DCF result and market-ready layer used at generation time.

The generated headline is designed for buyer diligence, for example:

> Over 5 years, BUY signals delivered 18% median 12M return with 62% hit rate.

Remaining institutional upgrade:

- Continue populating the archived point-in-time fundamentals database.
- Replay full DCF/fundamental signals using only data available on each historical date.
- Add an in-app validation dashboard over `scripts/replay_snapshots.py` outputs.
- Add survivorship-bias controls for delisted or merged companies.
- Validate over a broad 100-200 stock universe before making audited performance claims.

Implemented snapshot tooling:

- `snapshot_store.py`
- `scripts/capture_snapshots.py`
- `scripts/replay_snapshots.py`
- `docs/POINT_IN_TIME_SNAPSHOT_DATABASE.md`
