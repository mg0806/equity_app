# Point-In-Time Snapshot Database

## Purpose

The snapshot database is the foundation for institutional-grade backtesting. It stores the exact data available on a historical date, then replays the DCF and signal engine using only that saved snapshot.

This prevents look-ahead bias.

## What Gets Stored

Each snapshot stores:

- Screener company data.
- Annual financial tables.
- Quarterly results table.
- Cash-flow and balance-sheet tables.
- Yahoo price history available at capture time.
- Calculated ratios.
- Peer comparison table.
- DCF assumptions.
- DCF result.
- Market-ready decision layer.
- Snapshot date and fiscal-period label.
- Model version and source version.
- NSE/BSE corporate actions and announcements when public endpoints are available.

The Streamlit sidebar also exposes a Report History panel. Loading a saved report reconstructs the PDF and Excel outputs from the stored snapshot instead of fetching live fundamentals again.

Storage backend:

```text
data/equitylens_snapshots.sqlite
```

Current versions stored with app-generated reports:

```text
model_version = equitylens_v1.1_sector_nse_bse_events
source_version = screener_yahoo_nse_bse_v1
```

## Capture Snapshots

Capture one or more stocks:

```bash
python scripts/capture_snapshots.py --tickers BSE RELIANCE INFY
```

Capture with a quarter-end label:

```bash
python scripts/capture_snapshots.py --tickers-file tickers.txt --snapshot-date 2026-03-31 --fiscal-period FY26Q4
```

Recommended production schedule:

- Capture all tracked stocks after each quarterly result season.
- Capture price/market history weekly or monthly.
- Keep every snapshot immutable except for explicit recapture corrections.

## Replay Snapshots

Replay all saved snapshots:

```bash
python scripts/replay_snapshots.py
```

Replay a subset:

```bash
python scripts/replay_snapshots.py --tickers BSE RELIANCE --start 2024-01-01 --end 2026-03-31
```

Replay output:

- `summary`
- `replay_results`
- `snapshot_inventory`

## Backtest Claim Example

After enough snapshots exist, the summary can support claims like:

> Over 5 years, BUY snapshots delivered 18% median 12M return with 62% hit rate.

The key difference from the previous validation engine is that this replay uses stored point-in-time fundamentals, not today’s financial statement state.

## Remaining Production Controls

For a 9.5/10 institutional system, also add:

- Survivorship-bias-controlled ticker universe.
- Corporate action adjustment checks.
- Delisted/merged company handling.
- Vendor-licensed historical consensus estimates.
- Formal validation reports signed by run date, universe and data version.
