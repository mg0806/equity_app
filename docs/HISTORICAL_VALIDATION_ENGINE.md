# Historical Validation Engine

## Purpose

The validation engine proves whether EquityLens signals have historically been useful. It answers the question analyst firms will ask first:

> When the model said BUY, what happened over the next 3M, 6M and 12M?

## What It Measures

For each rolling historical signal window, the engine records:

- Signal: BUY, HOLD or SELL.
- Signal score and rationale.
- 1M, 3M, 6M and 12M forward return.
- Excess return versus benchmark when benchmark history is available.
- Forward max drawdown after the signal.
- Market regime: Bull, Bear, Sideways or Unknown.
- Sector and market-cap bucket.

## Output Metrics

The core report headline is:

> Over N years, BUY signals delivered X% median 12M return with Y% hit rate.

The engine also produces:

- Median forward returns by signal.
- Hit rate by signal.
- Excess return by signal.
- Drawdown after BUY/HOLD/SELL signals.
- Accuracy by market regime.
- Accuracy by sector.
- Accuracy by market-cap bucket.
- Full trade/window observations for audit.

## Current Validation Mode

The current implementation uses a rolling historical market-signal replay based on:

- Price versus 50 DMA.
- Price versus 200 DMA.
- 6M and 12M momentum.
- Relative strength versus Nifty when available.
- Drawdown from 52-week high.
- Historical price percentile.

This is materially stronger than showing only today’s DCF output, because it tests how the signal policy behaved historically.

## Snapshot Replay Mode

EquityLens now also stores model-versioned report snapshots in SQLite through `snapshot_store.py`. Each generated report preserves the company data, ratios, peer table, assumptions, DCF result, market-ready layer, source version and model version.

The replay workflow in `scripts/replay_snapshots.py` can replay those saved snapshots without fetching live fundamentals again. This is the path toward a full point-in-time fundamental backtest.

## Point-In-Time Limitation

This is not yet a broad audited archived-fundamental replay. A fully institutional point-in-time backtest requires a large history of stored snapshots plus:

- Screener financial tables.
- Quarterly results.
- DCF assumptions.
- Broker consensus estimates.
- Sector classifications.
- Corporate actions and delisting/survivorship controls.

The engine is designed for this upgrade through saved snapshots and replay tooling. As the SQLite database grows over real reporting dates, the replay workflow can evaluate the full DCF/fundamental signal without look-ahead bias.

## Batch Validation

Run validation across many stocks:

```bash
python scripts/run_validation.py --tickers BSE RELIANCE INFY HDFCBANK
```

Or from a file:

```bash
python scripts/run_validation.py --tickers-file tickers.txt --output validation_results.xlsx
```

The Excel output includes:

- `summary`
- `by_regime`
- `by_sector`
- `by_market_cap`
- `trades`
- `errors`

## Commercial Readiness Impact

This upgrade moves the project from an explainable research model toward a validation-backed signal platform. The remaining step for true institutional-grade claims is populating the archived snapshot database across a survivorship-controlled stock universe and exposing replay results in the app.
