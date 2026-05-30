# Data Sources And Compliance Notes

## Current Sources

| Source | Used For | Adapter |
|---|---|---|
| Screener.in | Financial statements, top ratios, CAGR tables, peer links | `data_sources/screener_source.py`, `scraper.py` |
| Yahoo Finance | Price history, beta, volatility, returns | `data_sources/yahoo_source.py` |
| NSE/BSE public pages | Corporate actions and corporate announcements | `data_sources/exchange_source.py` |
| Internal Models | Ratios, forecasts, DCF, Monte Carlo, signal, risk scoring | `ratios.py`, `dcf.py`, `market_ready.py` |
| Fallback Peer Map | Peer list fallback when benchmark rows pollute source data | `scraper.py` |

## Current Default Stack

The project now uses Screener.in, Yahoo Finance and public NSE/BSE pages as the default data stack. This avoids dependency on unavailable third-party enrichment APIs and keeps setup simple with no required API keys.

Current integration:

- Parses Screener financial statements, quarterly tables, CAGR tables, top ratios and peer links.
- Enriches market history, returns, volatility and beta through Yahoo Finance.
- Pulls NSE/BSE corporate actions and announcements as event context when public endpoints are available.
- Caches exchange-event responses locally under `.cache/` to reduce repeated website calls.
- Recalculates ratios internally instead of relying only on displayed source values.
- Adds source audit fields for Screener, Yahoo, NSE and BSE availability.
- Continues with Screener fundamentals if Yahoo market data is unavailable.
- Continues without exchange-event context if NSE/BSE blocks or changes an endpoint.
- MSN can be evaluated later as a fallback market-data source if Yahoo coverage becomes insufficient.

## Source Audit

Generated reports include a source audit showing:

- Model version and source version.
- Field source.
- Yahoo Finance availability.
- NSE/BSE event availability.
- Price conflicts between Screener and Yahoo.
- Data quality warnings.
- SQLite snapshot persistence status.

## Compliance Considerations Before Selling

Analyst firms will ask whether data usage is licensed. Before commercial sale:

- Review Screener.in terms for commercial redistribution and scraping.
- Review Yahoo/yfinance terms and whether it is acceptable for commercial workflows.
- Review NSE/BSE terms before using public exchange pages in commercial workflows.
- Review MSN terms before adding it as a fallback source.
- Consider replacing web scraping with licensed market/fundamental APIs.
- Add a customer-specific data-source configuration layer.
- Keep cached data provenance and timestamps.

## Recommended Production Data Providers

For paid institutional versions, consider:

- Refinitiv / LSEG.
- Bloomberg Enterprise Data.
- FactSet.
- S&P Capital IQ.
- Morningstar.
- NSE/BSE licensed feeds.
- FinancialModelingPrep or EODHD for lower-cost prototypes.

## Data Governance Requirements

Before enterprise deployment:

- Store source name and fetch timestamp for every field.
- Track model version used for every signal.
- Preserve input snapshots used to generate reports.
- Add data validation alerts when sources disagree.
- Add manual override logs for analyst assumptions.

## API Keys And Licensed Data

No API keys are required for the current no-broker setup. For a 9.5/10 institutional version, the only credentials worth adding later are licensed exchange or vendor feeds:

- NSE/BSE licensed data feed if the product must redistribute exchange data commercially.
- Licensed fundamentals provider if clients require production-grade financial statements instead of public-page parsing.
- Optional MSN or another market-data fallback only if Yahoo coverage becomes unreliable.
